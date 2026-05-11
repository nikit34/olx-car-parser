"""CLI for OLX.pt Car Parser."""

import fcntl
import json
import logging
import multiprocessing
import signal
import sys
import threading
from hashlib import md5
from pathlib import Path
from queue import Queue, Empty

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

_LOCK_PATH = Path(__file__).resolve().parent.parent / "data" / "scrape.lock"

from src.parser.scraper import OlxScraper, StandVirtualScraper, ScraperConfig, SV_BASE_URL
from src.storage.database import get_session, init_db
from src.models.generations import get_generation, infer_model_from_title
from src.storage.repository import (
    add_price_snapshot, compute_market_stats, deduplicate_cross_platform,
    deduplicate_same_platform, revalidate_recent_sold,
    get_duplicate_ids, get_listings_df, heal_mass_sweeps, mark_inactive,
    upsert_listing, upsert_unmatched,
)

app = typer.Typer(help="OLX.pt Car Parser — scrape, store, analyze")
console = Console()
log = logging.getLogger("cli")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def _load_scraper_config() -> dict:
    """Load scraper defaults from settings.yaml."""
    from pathlib import Path
    import yaml
    cfg_path = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"
    if cfg_path.exists():
        with open(cfg_path) as f:
            return (yaml.safe_load(f) or {}).get("scraper", {})
    return {}


def _llm_worker(in_q: multiprocessing.Queue, out_q: multiprocessing.Queue,
                shutdown: multiprocessing.Event):
    """Separate process: reads (olx_id, description) from in_q, sends (olx_id, result) to out_q.

    Exits when it receives a poison pill (None) or when the shutdown event is set
    and the queue is empty.
    """
    from src.parser.llm_enrichment import enrich_from_description, _llm_available, _get_config
    from queue import Empty

    log = logging.getLogger("llm_worker")
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])

    cfg = _get_config()
    if not _llm_available():
        log.warning("Ollama unreachable at %s, worker exiting.", cfg["ollama_url"])
        return

    log.info("LLM worker started (Ollama %s)", cfg["ollama_model"])

    enriched = 0
    failures = 0
    while True:
        try:
            item = in_q.get(timeout=60)
        except Empty:
            if shutdown.is_set():
                break
            continue
        if item is None:  # poison pill
            break
        olx_id, title, description = item
        if not description or len(description.strip()) < 20:
            out_q.put((olx_id, None))
            continue
        result = enrich_from_description(description, title)
        out_q.put((olx_id, result))
        if result:
            enriched += 1
            failures = 0
            if enriched % 25 == 0:
                log.info("LLM progress: %d enriched", enriched)
        else:
            failures += 1
            if failures >= 5:
                log.warning("5 consecutive LLM failures, worker stopping.")
                break

    log.info("LLM worker done: %d enriched", enriched)


def _desc_hash(text: str) -> str:
    return md5(text.strip().encode()).hexdigest()


def _llm_to_db(llm_out: multiprocessing.Queue, raw_by_id: dict, db_queue: Queue):
    """Thread: pairs LLM results with raw listings, forwards to db_queue."""
    merged = 0
    while True:
        item = llm_out.get()
        if item is None:
            break
        olx_id, result = item
        raw = raw_by_id.get(olx_id)
        if raw:
            db_queue.put((raw, result))
            merged += 1
    log.info("LLM merger done: %d forwarded to DB", merged)


def _db_worker(db_queue: Queue, result: dict):
    """Thread: reads (raw_listing, llm_data|None) from db_queue, saves to DB."""
    from src.parser.llm_enrichment import correct_listing_data

    session = get_session()
    saved = 0
    enriched = 0
    unmatched = 0
    active_ids: set[str] = set()
    processed = 0

    while True:
        item = db_queue.get()
        if item is None:
            break

        raw, llm_data = item

        if not raw.brand and not raw.title:
            processed += 1
            continue

        if llm_data:
            raw._llm_extras = llm_data
            corrections = correct_listing_data(raw)
            enriched += 1
        else:
            corrections = {}

        # StandVirtual detail pages frequently leave model="" (the
        # data-testid="model" container isn't always populated). The title
        # ("Peugeot 308 SW 1.6 HDi …") still carries the model — recover it
        # via the brand→known-models lexicon before generation lookup.
        if not raw.model and raw.brand and raw.title:
            inferred = infer_model_from_title(raw.brand, raw.title)
            if inferred:
                raw.model = inferred

        data = {
            "olx_id": raw.olx_id, "url": raw.url, "title": raw.title,
            "brand": raw.brand, "model": raw.model or "", "year": raw.year,
            "mileage_km": raw.mileage_km, "engine_cc": raw.engine_cc,
            "fuel_type": raw.fuel_type, "horsepower": raw.horsepower,
            "transmission": raw.transmission, "segment": raw.segment,
            "doors": raw.doors, "seats": raw.seats, "color": raw.color,
            "condition": raw.condition, "drive_type": raw.drive_type,
            "photo_count": raw.photo_count,
            "description_length": len(raw.description) if raw.description else None,
            "registration_month": raw.registration_month,
            "city": raw.city, "district": raw.district,
            "seller_type": raw.seller_type, "description": raw.description,
            "llm_extras": json.dumps(llm_data, ensure_ascii=False) if llm_data else None,
            "llm_description_hash": _desc_hash(raw.description) if llm_data and raw.description else None,
            **{k: v for k, v in corrections.items() if not k.startswith("_")},
            "source": getattr(raw, "source", "olx"),
            "posted_at": getattr(raw, "_posted_at", None),
            # Seller-profile pointer (resolved to seller_uuid asynchronously
            # by scripts/backfill_sellers.py). seller_uuid is intentionally
            # omitted here — it requires the profile-page fetch.
            "seller_profile_url": getattr(raw, "seller_profile_url", None),
            "seller_displayed_as": getattr(raw, "seller_displayed_as", None),
        }

        generation = get_generation(raw.brand, raw.model or "", raw.year)
        if generation:
            data["generation"] = generation
            listing = upsert_listing(session, data)
            if raw.price_eur is not None:
                add_price_snapshot(session, listing.id, raw.price_eur, raw.negotiable)
            active_ids.add(raw.olx_id)
            saved += 1
        else:
            reason = "no_year" if not raw.year else "no_generation_match"
            data["price_eur"] = raw.price_eur
            upsert_unmatched(session, data, reason)
            unmatched += 1

        processed += 1
        if processed % 100 == 0:
            session.commit()
            log.info("DB save progress: %d saved, %d enriched, %d unmatched",
                     saved, enriched, unmatched)

    # Dedup + market stats used to run incrementally inside this loop every
    # 500 saves, but the final passes after the whole pipeline drains already
    # recompute everything — the incremental runs were duplicated work that
    # added seconds per batch.  Keep only the final passes (see main scrape()).
    session.commit()
    session.close()
    log.info("DB worker done: %d saved, %d enriched, %d unmatched",
             saved, enriched, unmatched)
    result["saved"] = saved
    result["enriched"] = enriched
    result["unmatched"] = unmatched
    result["active_ids"] = active_ids


@app.command()
def scrape(
    pages: int = typer.Option(None, help="Max pages to scrape (default from config)"),
    delay_min: float = typer.Option(None, help="Min delay between requests (sec)"),
    delay_max: float = typer.Option(None, help="Max delay between requests (sec)"),
    private_only: bool = typer.Option(None, help="Only private sellers (Particular)"),
    concurrency: int = typer.Option(None, help="Parallel detail page workers (default 8)"),
    llm_workers: int = typer.Option(None, help="Parallel LLM enrichment workers (default from config, or 1)"),
    deep: bool = typer.Option(False, "--deep", help="Disable early-stop: walk all pages so stale tail listings get last_seen_at + price refreshed. Already-known olx_ids skip detail-fetch (just SERP-card refresh) so the LLM cost stays flat."),
):
    """Scrape OLX.pt car listings, enrich with LLM, save to database.

    Streaming pipeline: Scraper -> LLM -> DB all run in parallel.
    Listings are saved to DB as soon as LLM finishes (or immediately if
    no enrichment needed), without waiting for the full scrape to complete.

    LLM enrichment runs inline whenever Ollama is reachable; if it isn't,
    listings are saved raw and can be filled in later via `enrich`.
    """
    _LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    lock_file = open(_LOCK_PATH, "w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log.error("Another scrape is already running. Exiting.")
        raise typer.Exit(1)

    init_db()
    session = get_session()

    # Self-heal historical mark_inactive sweeps before doing anything else.
    # Cheap when there's nothing to heal (one indexed GROUP BY) and the
    # only way to recover from the source-blind bug that wiped 4500 OLX
    # rows in a single cycle.
    healed = heal_mass_sweeps(session)
    if healed:
        log.warning("Auto-healed %d falsely-deactivated rows before scrape.", healed)

    cfg = _load_scraper_config()
    config = ScraperConfig(
        max_pages=pages if pages is not None else cfg.get("max_pages", 25),
        delay_min=delay_min if delay_min is not None else cfg.get("request_delay_min", 2.0),
        delay_max=delay_max if delay_max is not None else cfg.get("request_delay_max", 5.0),
        private_only=private_only if private_only is not None else cfg.get("private_only", True),
        concurrency=concurrency if concurrency is not None else cfg.get("concurrency", 5),
    )

    log.info("Starting scrape of OLX.pt: up to %d pages...", config.max_pages)

    # Load existing enrichment hashes to skip unchanged descriptions
    from src.storage.repository import get_enriched_hashes
    enriched_hashes = get_enriched_hashes(session)
    duplicate_ids = get_duplicate_ids(session)
    # Already-known olx_ids drive scrape early-stop: when a search page is
    # ≥95% revisits across 3 pages in a row, OLX has nothing new this cycle
    # and we exit the pagination loop. Cuts a 999-page worst-case to ~30
    # pages on a steady-state DB.
    from src.models.listing import Listing as _Listing
    known_ids: set[str] = {
        olx_id for (olx_id,) in session.query(_Listing.olx_id).all()
    }
    session.close()
    log.info(
        "Already enriched: %d listings, %d duplicates, %d known olx_ids in DB",
        len(enriched_hashes), len(duplicate_ids), len(known_ids),
    )

    # Deep sweep: walk every page (no early-stop), but skip detail-fetch for
    # already-known olx_ids. SERP card alone refreshes last_seen_at and adds
    # a price snapshot, so listings that sank below page ~30 — invisible to
    # the shallow 4h cycle — still get their prices kept current. Detail
    # fetch + LLM still fire for genuinely new listings.
    if deep:
        log.info(
            "DEEP SWEEP: early-stop disabled, %d known olx_ids will skip detail-fetch",
            len(known_ids),
        )
        scrape_skip_ids = duplicate_ids | known_ids
        scrape_early_stop_ratio = 2.0  # impossible threshold ⇒ never trips
    else:
        scrape_skip_ids = duplicate_ids
        scrape_early_stop_ratio = 0.95

    # --- Streaming pipeline: Scraper -> [LLM] -> DB ---

    db_queue: Queue = Queue()
    raw_by_id: dict = {}

    # LLM pipeline — runs inline if Ollama is reachable, otherwise we save
    # raw and let the separate `enrich` command catch up.
    llm_in: multiprocessing.Queue | None = None
    llm_out: multiprocessing.Queue | None = None
    llm_shutdown = None
    llm_procs = []
    merger = None

    from src.parser.llm_enrichment import _get_config as _get_llm_config, _llm_available
    llm_cfg = _get_llm_config()
    llm_enabled = _llm_available()
    if llm_enabled:
        # Bounded queue provides back-pressure: the scraper blocks on put()
        # once Ollama can't keep up, rather than buffering tens of thousands
        # of descriptions in RAM. 2000 ≈ four hours of LLM headroom — large
        # enough that scrape almost always finishes before the queue fills,
        # so the scrape and LLM phases stop competing for the same wall
        # budget. Each queued item is (id, title, desc) ≈ 3 KB → ~6 MB peak.
        llm_in = multiprocessing.Queue(maxsize=2000)
        llm_out = multiprocessing.Queue()
        llm_shutdown = multiprocessing.Event()
        num_workers = llm_workers if llm_workers is not None else llm_cfg.get("max_workers", 6)
        for _ in range(num_workers):
            p = multiprocessing.Process(target=_llm_worker, args=(llm_in, llm_out, llm_shutdown), daemon=True)
            p.start()
            llm_procs.append(p)
        merger = threading.Thread(target=_llm_to_db, args=(llm_out, raw_by_id, db_queue), daemon=True)
        merger.start()
        log.info("Inline LLM enrichment enabled (%d Ollama workers)", num_workers)
    else:
        log.warning("Ollama unreachable at %s — saving listings raw, run `enrich` later",
                    llm_cfg["ollama_url"])

    # DB worker thread: saves listings as they arrive
    db_result: dict = {}
    db_thread = threading.Thread(target=_db_worker, args=(db_queue, db_result), daemon=True)
    db_thread.start()

    # Scraper callback: sends each listing to LLM or directly to DB
    sent_to_llm = 0
    skipped_llm = 0
    skipped_no_photos = 0

    def _on_batch(batch):
        nonlocal sent_to_llm, skipped_llm, skipped_no_photos
        for listing in batch:
            # Drop listings the detail page confirmed have zero photos —
            # they're typically reposts of expired ads or low-effort scams,
            # and they pollute the dashboard with cards that have nothing
            # to show. We only skip on a definite 0; None means the gallery
            # selector didn't fire (detail-fetch failed / page restructure)
            # so we can't tell.
            if listing.photo_count == 0:
                skipped_no_photos += 1
                continue
            raw_by_id[listing.olx_id] = listing
            if not llm_enabled:
                db_queue.put((listing, None))
                continue
            if not listing.description or len(listing.description.strip()) < 20:
                db_queue.put((listing, None))
                continue
            if listing.olx_id in duplicate_ids:
                skipped_llm += 1
                db_queue.put((listing, None))
                continue
            # Listings that won't match a known generation flow into the
            # unmatched table, where llm_extras is unused — sending them to
            # Ollama is pure waste. ~18% of inline traffic on a steady-state
            # DB. Cheap to recompute (lookup table), worth the early skip.
            if not get_generation(listing.brand, listing.model or "", listing.year):
                skipped_llm += 1
                db_queue.put((listing, None))
                continue
            h = _desc_hash(listing.description)
            if enriched_hashes.get(listing.olx_id) == h:
                skipped_llm += 1
                db_queue.put((listing, None))
                continue
            llm_in.put((listing.olx_id, listing.title, listing.description))
            sent_to_llm += 1
        log.info("Page done: %d listings -> %d sent to LLM, %d skipped, %d dropped (no photos)",
                 len(batch), sent_to_llm, skipped_llm, skipped_no_photos)

    with OlxScraper(config) as scraper:
        raw_listings = scraper.scrape_all(
            on_batch_ready=_on_batch,
            skip_enrichment_ids=scrape_skip_ids,
            known_ids=known_ids,
            early_stop_known_ratio=scrape_early_stop_ratio,
        )

    # --- Scrape StandVirtual (same pipeline) ---
    sv_config = ScraperConfig(
        base_url=SV_BASE_URL,
        max_pages=config.max_pages,
        delay_min=config.delay_min,
        delay_max=config.delay_max,
        private_only=True,
        concurrency=config.concurrency,
    )
    log.info("Starting scrape of StandVirtual: up to %d pages...", sv_config.max_pages)
    with StandVirtualScraper(sv_config) as sv_scraper:
        sv_listings = sv_scraper.scrape_all(
            on_batch_ready=_on_batch,
            skip_enrichment_ids=scrape_skip_ids,
            known_ids=known_ids,
            early_stop_known_ratio=scrape_early_stop_ratio,
        )
    raw_listings.extend(sv_listings)

    # Collect scraped IDs now; mark_inactive runs after DB worker finishes
    # to avoid "database is locked" from concurrent SQLite writers.
    # Group by source so an OLX outage can't sweep SV rows (and vice-versa) —
    # mark_inactive is source-scoped.
    scraped_by_source: dict[str, set[str]] = {}
    for olx_id, raw in raw_by_id.items():
        src = getattr(raw, "source", None) or "olx"
        scraped_by_source.setdefault(src, set()).add(olx_id)

    def _mark_inactive_safely(session) -> None:
        """Per-source mark_inactive with anomaly gate + revalidate.

        Two phases per source:
          1. ``revalidate_recent_sold`` — probe URLs of listings marked
             sold in the last 14 days; restore the ones that turn out
             to still be live (false-positive heal, see heal_false_sold
             one-shot for the historical equivalent).
          2. ``mark_inactive`` — the usual "not in scrape results" sweep.
             Skipped for any source whose scrape returned <10% of its
             currently-active rows — almost always a partial scrape,
             not real churn (real per-cycle churn is ≤1–2%).
        """
        from src.models.listing import Listing as _Listing
        from sqlalchemy import or_ as _or_
        for src, ids in scraped_by_source.items():
            if not ids:
                log.warning("No %s listings scraped — skipping mark_inactive(%s)", src, src)
                continue
            try:
                stats = revalidate_recent_sold(session, src)
                if stats["restored"]:
                    session.commit()
                    log.info(
                        "Revalidate(%s): restored %d false-positive sold rows",
                        src, stats["restored"],
                    )
            except Exception as e:
                log.warning("revalidate_recent_sold(%s) failed: %s", src, e)
                session.rollback()

            if src == "olx":
                src_filter = _or_(_Listing.source == "olx", _Listing.source.is_(None))
            else:
                src_filter = _Listing.source == src
            active_count = session.query(_Listing).filter(
                _Listing.is_active == True, src_filter,
            ).count()
            if active_count > 100 and len(ids) < active_count * 0.10:
                log.warning(
                    "Refusing mark_inactive(%s): scraped %d, active %d "
                    "(%.1f%%, gate 10%%). Likely partial scrape — skip.",
                    src, len(ids), active_count,
                    len(ids) / max(active_count, 1) * 100,
                )
                continue
            mark_inactive(session, src, ids)

    # --- SIGTERM handler: graceful shutdown on timeout ---
    def _sigterm_handler(signum, frame):
        log.warning("SIGTERM received — initiating graceful shutdown...")
        if llm_shutdown:
            llm_shutdown.set()
        if llm_in:
            drained = 0
            while True:
                try:
                    item = llm_in.get_nowait()
                except Empty:
                    break
                if item is not None:
                    # llm_in carries (olx_id, title, description) — unpack
                    # tolerantly so the drain works regardless of payload arity.
                    olx_id, *_ = item
                    raw = raw_by_id.get(olx_id)
                    if raw:
                        db_queue.put((raw, None))
                        drained += 1
            if drained:
                log.warning("SIGTERM drain: %d listings saved without enrichment", drained)
        if llm_out:
            llm_out.put(None)
        if merger:
            merger.join(timeout=10)
        db_queue.put(None)
        db_thread.join(timeout=30)
        if any(scraped_by_source.values()):
            try:
                s = get_session()
                _mark_inactive_safely(s)
                s.commit()
                s.close()
            except Exception as e:
                log.warning("mark_inactive failed on SIGTERM: %s", e)
        log.info("Graceful shutdown complete: %d saved, %d enriched, %d unmatched",
                 db_result.get("saved", 0), db_result.get("enriched", 0),
                 db_result.get("unmatched", 0))
        sys.exit(0)

    signal.signal(signal.SIGTERM, _sigterm_handler)

    # --- Shutdown: drain pipeline in order ---
    log.info("Scraping done (%d OLX + %d SV = %d total). Waiting for pipeline to drain...",
             len(raw_listings) - len(sv_listings), len(sv_listings), len(raw_listings))

    if llm_enabled:
        # 1. Stop LLM workers
        for _ in llm_procs:
            llm_in.put(None)
        llm_shutdown.set()
        for p in llm_procs:
            p.join()

        # 2. Drain leftover llm_in
        drained = 0
        while True:
            try:
                item = llm_in.get_nowait()
            except Empty:
                break
            if item is not None:
                olx_id, *_ = item
                raw = raw_by_id.get(olx_id)
                if raw:
                    db_queue.put((raw, None))
                    drained += 1
        if drained:
            log.warning("LLM workers exited early: %d listings saved without enrichment", drained)

        # 3. Stop merger
        llm_out.put(None)
        merger.join()

    # 4. Stop DB worker
    db_queue.put(None)
    db_thread.join()

    if not raw_listings:
        log.error("No listings scraped.")
        raise typer.Exit(1)

    # 5. Mark inactive, dedup & market stats (single session, no contention)
    final_session = get_session()
    total_scraped = sum(len(v) for v in scraped_by_source.values())
    if total_scraped:
        log.info(
            "Marking inactive listings per source: %s",
            {s: len(v) for s, v in scraped_by_source.items()},
        )
        _mark_inactive_safely(final_session)
        final_session.commit()
    log.info("Final deduplication...")
    dedup_count = deduplicate_cross_platform(final_session)
    if dedup_count:
        log.info("Marked %d cross-platform duplicates", dedup_count)
    same_dedup = deduplicate_same_platform(final_session)
    if same_dedup:
        log.info("Marked %d same-platform duplicates", same_dedup)
    # Market stats roll up per day, so there's no point recomputing them on
    # every 8-hour scrape — only do the full pass if today hasn't been stamped
    # yet. The three-per-day full pass was the single slowest step for a
    # large DB.
    from datetime import date as _date
    from src.models.listing import MarketStats
    latest_stats_date = (
        final_session.query(MarketStats.date)
        .order_by(MarketStats.date.desc())
        .limit(1)
        .scalar()
    )
    if latest_stats_date == _date.today():
        log.info("Market stats already computed today — skipping full pass.")
    else:
        log.info("Final market stats...")
        compute_market_stats(final_session)
    final_session.commit()
    final_session.close()

    log.info("Saved %d listings (%d enriched), %d unmatched",
             db_result.get("saved", 0), db_result.get("enriched", 0),
             db_result.get("unmatched", 0))
    log.info("Done!")


@app.command()
def enrich(
    workers: int = typer.Option(6, help="Parallel Ollama workers (default 6)"),
    cheap_first: bool = typer.Option(True, help="Prioritize cheaper listings"),
    active_only: bool = typer.Option(
        False, help="Skip sold/expired listings (is_active=False) — useful "
                    "for fast schema-backfill runs that only matter to the "
                    "currently-trained model.",
    ),
):
    """Enrich unenriched listings with the local LLM (parallel).

    Processes listings that don't have llm_extras yet. Uses ThreadPoolExecutor
    so several Ollama requests are inflight at once; on M1 8 GB the model
    saturates around 6 workers.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from src.parser.llm_enrichment import (
        enrich_from_description, correct_listing_data,
        _llm_available, _get_config, _derive_damage_severity,
    )
    from src.models.listing import Listing

    init_db()
    session = get_session()

    cfg = _get_config()
    # Pending = no llm_extras yet, OR llm_extras lacks the v2 damage_severity
    # field (post-schema-bump backfill). The json_extract clause is SQLite-
    # specific but the project pins SQLite, so this is fine.
    from sqlalchemy import or_, func as sa_func, text as sa_text
    needs_damage_backfill = sa_func.json_extract(
        Listing.llm_extras, "$.damage_severity",
    ).is_(None)
    q = (
        session.query(Listing)
        .filter(
            or_(Listing.llm_extras.is_(None), needs_damage_backfill),
            Listing.description.isnot(None),
        )
    )
    if active_only:
        q = q.filter(Listing.is_active == True)  # noqa: E712 — SQLAlchemy needs ==
    pending = q.all()

    if not pending:
        log.info("All listings already enriched.")
        return

    # Two-stage split: rows that already have llm_extras only need the new
    # damage_severity field, which we derive deterministically from existing
    # extras + a keyword scan (validated 100% LLM-equivalent on the eval set).
    # Only rows with NO extras hit Ollama. Cuts ~80% of LLM calls in the
    # post-schema-bump backfill case.
    backfill_only = [l for l in pending if l.llm_extras]
    fresh = [l for l in pending if not l.llm_extras]

    if backfill_only:
        log.info("Backfill-only path: deriving damage_severity for %d listings (no LLM)...",
                 len(backfill_only))
        derived_count = 0
        for listing in backfill_only:
            try:
                extras = json.loads(listing.llm_extras) if listing.llm_extras else {}
            except (json.JSONDecodeError, TypeError):
                continue
            severity = _derive_damage_severity(
                extras, listing.title or "", listing.description or "",
            )
            extras["damage_severity"] = severity
            listing.llm_extras = json.dumps(extras, ensure_ascii=False)
            listing.damage_severity = severity
            derived_count += 1
            if derived_count % 500 == 0:
                session.commit()
                log.info("Derive progress: %d / %d", derived_count, len(backfill_only))
        session.commit()
        log.info("Derived damage_severity for %d listings without LLM.", derived_count)

    if not fresh:
        log.info("No fresh enrichment needed — all listings had existing extras.")
        return

    if not _llm_available():
        log.error("Ollama not reachable; %d listings still need fresh enrichment.", len(fresh))
        raise typer.Exit(1)

    if cheap_first:
        # Fetch min price per listing in one SQL query instead of firing a
        # lazy relationship lookup per listing (N+1 over price_snapshots).
        from sqlalchemy import func
        from src.models.listing import PriceSnapshot
        fresh_ids = [l.id for l in fresh]
        min_prices = dict(
            session.query(
                PriceSnapshot.listing_id,
                func.min(PriceSnapshot.price_eur),
            )
            .filter(PriceSnapshot.listing_id.in_(fresh_ids))
            .group_by(PriceSnapshot.listing_id)
            .all()
        )
        fresh.sort(key=lambda l: min_prices.get(l.id) or float("inf"))

    pending = fresh
    log.info("Fresh enrichment: %d listings with %d workers (Ollama %s)...",
             len(pending), workers, cfg['ollama_model'])

    enriched = 0
    failed = 0
    consecutive_failures = 0
    batch_size = 25

    def _enrich_one(listing, description, title):
        # Worker runs on a non-main thread — must not touch ORM attributes,
        # because the shared Session is not thread-safe and attributes get
        # expired after each main-thread commit, triggering lazy reloads.
        if not description or len(description.strip()) < 20:
            return listing, {}
        result = enrich_from_description(description, title or "")
        return listing, result

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for listing in pending:
            # Snapshot fields on the main thread before submitting, so the
            # worker never needs to hit the Session.
            desc = listing.description or ""
            title = listing.title or ""
            futures[pool.submit(_enrich_one, listing, desc, title)] = listing

        for fut in as_completed(futures):
            listing, result = fut.result()
            if result == {}:
                listing.llm_extras = "{}"
                continue
            if result:
                listing._llm_extras = result
                corrections = correct_listing_data(listing)
                # damage_severity is derived deterministically inside
                # correct_listing_data — fold it back into extras so the next
                # enrich run's needs_damage_backfill filter recognizes the
                # row as already done.
                if "damage_severity" in corrections:
                    result["damage_severity"] = corrections["damage_severity"]
                listing.llm_extras = json.dumps(result, ensure_ascii=False)
                for field, value in corrections.items():
                    if hasattr(listing, field):
                        setattr(listing, field, value)
                enriched += 1
                consecutive_failures = 0
                if enriched % batch_size == 0:
                    session.commit()
                    log.info("Enrich progress: %d / %d (failed: %d)",
                             enriched, len(pending), failed)
            else:
                failed += 1
                consecutive_failures += 1
                if consecutive_failures >= 10:
                    log.error("10 consecutive LLM failures, stopping.")
                    pool.shutdown(wait=False, cancel_futures=True)
                    break

    session.commit()
    log.info("Enriched %d listings (%d failed).", enriched, failed)


@app.command("verify-photos")
def verify_photos(
    threshold: float = typer.Option(0.20, help="P(damaged) threshold (0.20 = production default, F1=0.818 R=100%% on gold)."),
    workers: int = typer.Option(4, help="Concurrent listing workers — overlaps photo fetch I/O with classifier inference. Default 4; use 1 for sequential."),
    only_text_flagged: bool = typer.Option(
        False, help="Process only listings with text damage_severity >= 2 "
                    "(verifier mode). Off = scan all listings (full coverage)."),
    upgrade_legacy: bool = typer.Option(
        False,
        help="Re-run inference on rows whose photo_damage_flagged was "
             "written by the 2026-05-02 backfill script (legacy max-rule "
             "decision persisted as a schema-consistency stop-gap). "
             "Selects rows with photo_damage_flag_source = "
             "'legacy_max_rule_backfill', overwrites with proper "
             "multi-photo agreement, and clears the marker."),
    backfill_plates: bool = typer.Option(
        False,
        help="Plate-only pass over rows that already have damage scores "
             "but no plate_readable yet. Selects "
             "photo_damage_p IS NOT NULL AND plate_readable IS NULL, "
             "skips the damage classifier (preserves existing damage "
             "scores untouched), and runs only photo download + CLIP "
             "filter + plate OCR. Use to retro-fit the four ``plate_*`` "
             "fields onto listings verified before plate detection landed."),
    cache_dir: Path = typer.Option(
        Path("/tmp/photo_verify/cache"), help="Local photo cache directory."),
    dry_run: bool = typer.Option(
        False, help="Print what would change without writing to DB."),
    limit: int | None = typer.Option(
        None, help="Optional cap on listings processed (useful for staged rollouts)."),
):
    """Run the v2 damage classifier on listings' photos and store ``photo_damage_p`` in llm_extras.

    Non-destructive: keeps the text-derived ``damage_severity`` column intact.
    Adds four JSON keys:
      • ``photo_damage_p`` — max P(damaged) across photos
      • ``photo_damage_n_photos`` — photos checked
      • ``photo_damages`` — per-photo ``[{"idx": int, "p": float}, ...]``;
        ``idx`` is 1-based and matches the ``fetch_photos(url)`` ordering so
        the originating URL is recoverable by re-running fetch (issue #4).
      • ``photo_damage_flagged`` — listing-level damaged decision under the
        multi-photo agreement rule (issue #2): True iff at least
        ``FLAG_MIN_PHOTOS`` photos exceed ``FLAG_PHOTO_THRESHOLD``. Decoupled
        from ``photo_damage_p`` so alerts/dashboard threshold logic on the
        max-score keeps working untouched (additive field).

    Plus four plate-detection keys (PT license-plate OCR on the same
    exterior photos):
      • ``plate_texts`` — per-photo ``[{"idx": int, "text": "AA-00-AA",
        "confidence": float}, ...]`` for photos where a PT-formatted plate
        was readable. ``idx`` matches ``photo_damages`` so a listing's
        photos line up across both signals.
      • ``plate_n_readable`` — count of exterior photos with a readable
        plate. ``0`` means no photo in the listing showed a recognizable
        plate; useful as a soft "obscured listing" hint.
      • ``plate_readable`` — bool: ``plate_n_readable > 0``.
      • ``plate_text_primary`` — highest-confidence plate text across the
        listing, or ``None``. Stored on the listing for dedup / search.

    Coverage: both OLX (``apollo.olxcdn.com`` URL scrape) and StandVirtual
    (``__NEXT_DATA__`` JSON). Listings from other sources are skipped.

    Pipeline (issue #3): per-listing photos are first run through a CLIP
    zero-shot exterior / non-exterior filter; only exterior photos are scored
    by the damage classifier. The audit (#1) showed the v2 classifier was
    trained on full-vehicle exterior shots and confidently mis-fires on
    interiors / engine bays / wheel close-ups / dashboards / seats / trunks.
    Filtered photos contribute ``photo_damage_n_exterior`` (additive field).
    Listings whose every photo is filtered out persist as the same empty
    record as the no-photos path (``photo_damage_n_photos=len(original)``,
    ``photo_damage_n_exterior=0``, no flag). The plate reader runs on the
    same exterior set; non-exterior shots are skipped (plates rarely
    survive a wheel close-up or dashboard frame anyway).

    ``--backfill-plates``: one-shot retro-fit for rows verified before the
    plate reader landed. Selects ``photo_damage_p IS NOT NULL AND
    plate_readable IS NULL``, skips the damage classifier entirely (no
    inference cost, existing damage scores preserved bit-for-bit), and
    writes only the four ``plate_*`` fields. Mutually exclusive with
    ``--upgrade-legacy``.
    """
    # Flag validation happens before any heavy imports / DB init so an
    # operator typo aborts in milliseconds rather than after loading torch.
    if upgrade_legacy and backfill_plates:
        raise typer.BadParameter(
            "--upgrade-legacy and --backfill-plates are mutually exclusive: "
            "the first re-runs damage inference, the second deliberately skips it."
        )

    import sqlite3
    import time
    from src.parser.photo_damage import DamageClassifier
    from src.parser.photo_viewpoint import ExteriorFilter
    from src.parser.photo_plate import PlateReader
    from src.parser.photo_fetch import fetch_photos, download_photo

    init_db()
    session = get_session()
    from src.models.listing import Listing
    from sqlalchemy import and_, or_, func as sa_func

    cache_dir.mkdir(parents=True, exist_ok=True)
    if backfill_plates:
        log.info("Backfill-plates mode: skipping damage classifier load.")
        clf = None
    else:
        log.info("Loading classifier (threshold=%.2f)...", threshold)
        clf = DamageClassifier(threshold=threshold)
        log.info("Device: %s, classes: %s", clf.device, clf.classes)
    log.info("Loading CLIP exterior filter (issue #3 OOD pre-filter)...")
    exterior_filter = ExteriorFilter()
    log.info("CLIP filter device: %s", exterior_filter.device)
    log.info("Loading EasyOCR plate reader (CPU)...")
    plate_reader = PlateReader()

    # Default selection: active OLX/StandVirtual listings missing
    # photo_damage_p in llm_extras (steady-state cron). The
    # ``--upgrade-legacy`` flag swaps this for rows the 2026-05-02
    # backfill stamped with photo_damage_flag_source = 'legacy_max_rule_backfill'
    # — those have photo_damage_p but their photo_damage_flagged came from
    # the v2 max-rule (~32.8% over-flag rate), not real multi-photo
    # inference. Re-running inference replaces the boolean with the
    # multi-photo decision and clears the marker.
    from sqlalchemy import or_
    if upgrade_legacy:
        legacy_marker = sa_func.json_extract(
            Listing.llm_extras, "$.photo_damage_flag_source",
        ) == "legacy_max_rule_backfill"
        selection_filter = legacy_marker
    elif backfill_plates:
        # One-shot plate retro-fit: pick rows already verified for damage
        # (so we don't re-process listings still in the steady-state
        # damage queue) but missing the plate fields entirely.
        has_damage = sa_func.json_extract(
            Listing.llm_extras, "$.photo_damage_p",
        ).isnot(None)
        needs_plate = sa_func.json_extract(
            Listing.llm_extras, "$.plate_readable",
        ).is_(None)
        selection_filter = and_(has_damage, needs_plate)
    else:
        needs_photo = sa_func.json_extract(
            Listing.llm_extras, "$.photo_damage_p",
        ).is_(None)
        selection_filter = needs_photo
    # Priority order:
    #  1. Listings with text-derived damage_severity >= 2 — alerts already
    #     see them as suspect; photo verifier confirms or downgrades.
    #  2. desc_mentions_accident — same logic, smaller bucket.
    #  3. Newest first — drain the steady-state backlog of normal listings.
    # On the production DB (2698 pending) this floats ~120 high-signal
    # rows to the front, so a single cron's --limit budget pays maximal
    # alert-quality dividend before grinding through clean dealer photos.
    text_sev_ge2_order = sa_func.json_extract(
        Listing.llm_extras, "$.damage_severity",
    ) >= 2
    q = (
        session.query(Listing)
        .filter(
            Listing.is_active == True,  # noqa: E712
            or_(
                Listing.url.like("%standvirtual%"),
                Listing.url.like("%olx.pt%"),
            ),
            Listing.llm_extras.isnot(None),
            selection_filter,
        )
        .order_by(
            text_sev_ge2_order.desc(),
            Listing.desc_mentions_accident.desc(),
            Listing.first_seen_at.desc(),
        )
    )
    if only_text_flagged:
        text_sev_ge2 = sa_func.json_extract(
            Listing.llm_extras, "$.damage_severity",
        ) >= 2
        q = q.filter(text_sev_ge2)
    if limit is not None and limit > 0:
        q = q.limit(limit)
    pending = q.all()
    if not pending:
        log.info("Nothing to verify.")
        return
    log.info("Pending: %d listings.", len(pending))

    # Worker pool: each thread fetches photos + downloads + runs classifier
    # for one listing at a time, then returns the result. The main thread
    # owns the SQLAlchemy session — listings are looked up by olx_id and
    # llm_extras updated in batch commits, never inside a worker (Session
    # isn't thread-safe and ORM attribute lazy-loads expire on commit).
    #
    # Snapshot (olx_id, url) on the main thread so workers don't touch
    # ORM state at all.
    from concurrent.futures import ThreadPoolExecutor, as_completed
    listing_by_id = {l.olx_id: l for l in pending}
    work_items = [(l.olx_id, l.url) for l in pending]

    def _verify_one(
        olx_id: str, url: str
    ) -> tuple[
        str, float, int, int, list[dict], bool,
        list[dict], str | None, str | None,
    ]:
        """Returns the per-listing tuple consumed by the main loop.

        Tuple fields (in order):
          ``olx_id, max_p, n_photos, n_exterior, per_photo, flagged,
           plate_per_photo, plate_primary, error_msg``

        ``n_photos`` is the count of photos that successfully downloaded —
        same as the legacy semantics so ``photo_damage_n_photos`` keeps
        meaning "how many photos did we try to score on this listing".
        ``n_exterior`` is the subset of those that passed the CLIP filter
        and were actually fed to the damage classifier (issue #3).

        ``per_photo`` is a list of ``{"idx": int, "p": float}`` ordered by
        ``idx`` ascending. ``idx`` is 1-based and aligned to the
        ``fetch_photos(url)`` enumeration so consumers can recover the source
        URL by re-running fetch (deterministic ordering, see issue #4). The
        array contains entries only for *exterior* photos — non-exterior
        ones are dropped before scoring. Their original ``idx`` is preserved
        (not renumbered), so a re-fetch can still join URLs by idx.

        ``flagged`` is the listing-level multi-photo agreement decision
        (``ListingPrediction.is_damaged``, see issue #2). Decoupled from
        ``max_p`` so existing consumers that threshold on the max-score keep
        working unchanged.

        ``plate_per_photo`` is a list of ``{"idx", "text", "confidence"}``
        for exterior photos with a recognized PT plate (same idx convention
        as ``per_photo``; photos without a readable plate are absent).
        ``plate_primary`` is the single highest-confidence plate text across
        the listing, or ``None``.
        """
        try:
            photo_urls = fetch_photos(url)
            listing_dir = cache_dir / olx_id
            # Track the (idx, path) pairs we successfully downloaded so we
            # can stitch per-photo scores back to their original position
            # even when some downloads fail in the middle of the sequence.
            indexed_paths: list[tuple[int, Path]] = []
            for j, purl in enumerate(photo_urls, 1):
                p = listing_dir / f"{olx_id}_{j}.jpg"
                if download_photo(purl, p):
                    indexed_paths.append((j, p))
            if not indexed_paths:
                return olx_id, 0.0, 0, 0, [], False, [], None, None
            # CLIP pre-filter: drop OOD viewpoints before damage scoring.
            # Single forward pass over all of the listing's photos — same
            # CLIP model instance is reused across listings.
            all_paths = [p for _, p in indexed_paths]
            mask = exterior_filter.is_exterior_batch(all_paths)
            exterior_indexed = [
                ip for ip, keep in zip(indexed_paths, mask) if keep
            ]
            n_total = len(indexed_paths)
            n_exterior = len(exterior_indexed)
            if not exterior_indexed:
                # Every photo was OOD — same persistence shape as no_photos
                # but ``n_photos`` keeps the original count so the listing
                # records "we did look at N photos, none were exterior".
                return olx_id, 0.0, n_total, 0, [], False, [], None, None
            photo_paths = [p for _, p in exterior_indexed]
            if clf is None:
                # Backfill-plates mode: skip damage inference. We still
                # need n_total / n_exterior accounting and the exterior
                # photo set for the plate reader, but the damage tuple
                # fields are placeholders — the persistence layer below
                # is wired to leave existing damage_* values untouched
                # in this mode.
                pred_max_p = 0.0
                pred_is_damaged = False
                per_photo: list[dict] = []
            else:
                pred = clf.predict_listing(olx_id, photo_paths)
                pred_max_p = pred.max_p
                pred_is_damaged = bool(pred.is_damaged)
                per_photo = [
                    {"idx": idx, "p": round(float(photo.p_damaged), 4)}
                    for (idx, _), photo in zip(exterior_indexed, pred.photos)
                ]
            # Plate OCR on the same exterior set. ``read_photos`` preserves
            # ordering and emits ``None`` for photos with no readable PT
            # plate; we pair those back with the original idx and drop
            # blanks before persisting.
            plate_results = plate_reader.read_photos(photo_paths)
            plate_per_photo: list[dict] = []
            plate_primary: str | None = None
            best_conf: float = -1.0
            for (idx, _), pr in zip(exterior_indexed, plate_results):
                if pr is None:
                    continue
                plate_per_photo.append({
                    "idx": idx,
                    "text": pr.text,
                    "confidence": round(float(pr.confidence), 4),
                })
                if pr.confidence > best_conf:
                    best_conf = float(pr.confidence)
                    plate_primary = pr.text
            plate_per_photo.sort(key=lambda d: d["idx"])
            return (
                olx_id, pred_max_p, n_total, n_exterior,
                per_photo, pred_is_damaged,
                plate_per_photo, plate_primary, None,
            )
        except Exception as exc:  # noqa: BLE001
            return olx_id, 0.0, 0, 0, [], False, [], None, str(exc)

    flagged = downgraded = no_photos = errors = 0
    processed = 0
    updated_count = 0
    t0 = time.monotonic()

    with ThreadPoolExecutor(max_workers=max(workers, 1)) as pool:
        futures = {
            pool.submit(_verify_one, olx_id, url): olx_id
            for olx_id, url in work_items
        }
        for fut in as_completed(futures):
            (olx_id, max_p, n_photos, n_exterior,
             per_photo, flagged_pred,
             plate_per_photo, plate_primary, err) = fut.result()
            processed += 1
            if err:
                log.warning("Classifier failed on %s: %s", olx_id, err)
                errors += 1
                continue
            if n_photos == 0:
                no_photos += 1
            listing = listing_by_id[olx_id]
            try:
                extras = json.loads(listing.llm_extras) if listing.llm_extras else {}
            except (json.JSONDecodeError, TypeError):
                extras = {}
            text_sev = extras.get("damage_severity") or 0
            if not backfill_plates:
                extras["photo_damage_p"] = round(max_p, 4)
                extras["photo_damage_n_photos"] = n_photos
                # CLIP exterior pre-filter count (issue #3) — how many of the
                # ``n_photos`` downloaded actually got fed to the damage
                # classifier. ``n_exterior <= n_photos``; equal means no
                # filtering happened on this listing, zero means every photo
                # was OOD (interiors / engine bay / wheel close-up / etc.).
                extras["photo_damage_n_exterior"] = n_exterior
                # Per-photo scores (issue #4) — keep listing-level fields above
                # untouched for backward compat with alerts/dashboard. Sorted by
                # idx ascending so downstream consumers don't have to. After
                # issue #3 only exterior photos appear here; their ``idx`` is
                # the original 1-based position in ``fetch_photos(url)``.
                extras["photo_damages"] = sorted(per_photo, key=lambda d: d["idx"])
                # Listing-level multi-photo agreement decision (issue #2). New,
                # additive field — alerts/dashboard still threshold on
                # ``photo_damage_p`` (max across photos) and keep working
                # unchanged.
                extras["photo_damage_flagged"] = bool(flagged_pred)
                # Drop the 2026-05-02 backfill marker once a row gets real
                # multi-photo inference — the boolean above is now authoritative.
                extras.pop("photo_damage_flag_source", None)
            # PT plate OCR — same exterior set as the damage classifier.
            # ``plate_per_photo`` already sorted by idx in _verify_one. Always
            # written, including in --backfill-plates mode (that's the whole
            # point of the mode — retro-fit these four fields onto rows that
            # already have damage scores from a pre-plate run).
            extras["plate_texts"] = plate_per_photo
            extras["plate_n_readable"] = len(plate_per_photo)
            extras["plate_readable"] = bool(plate_per_photo)
            extras["plate_text_primary"] = plate_primary
            if not dry_run:
                listing.llm_extras = json.dumps(extras, ensure_ascii=False)
            # Count real writes (vs. error/skip rows). Used by the
            # post-loop sanity guard below to detect "step ran but produced
            # zero output" failure modes that ``continue-on-error: true``
            # would otherwise hide from the workflow's run summary
            # (see issue #7, masking pattern from runs 25220681021 /
            # 25222655513 — fixed in b13e6c8 / 2e649c8).
            updated_count += 1
            # Count under the new rule so the on-screen tally matches what
            # downstream consumers see in ``photo_damage_flagged``.
            if flagged_pred:
                flagged += 1
            elif text_sev >= 2:
                downgraded += 1

            if processed % 25 == 0:
                if not dry_run:
                    session.commit()
                rate = processed / max(time.monotonic() - t0, 1e-3)
                log.info("Verify progress: %d/%d  flagged=%d  no_photos=%d  errs=%d  (%.1f listing/s)",
                         processed, len(pending), flagged, no_photos, errors, rate)

    if not dry_run:
        session.commit()
    elapsed = time.monotonic() - t0
    log.info(
        "Done in %.1f min  flagged=%d  text_overcalls_downgrade_candidates=%d  "
        "no_photos=%d  errors=%d  (%.1fs/listing)",
        elapsed / 60, flagged, downgraded, no_photos, errors,
        elapsed / max(len(pending), 1),
    )
    if dry_run:
        log.info("Dry-run — no DB writes. Re-run without --dry-run to persist.")

    # Sanity guard (issue #7): the workflow step has ``continue-on-error:
    # true`` because verify-photos is a slow, opt-in enrichment that
    # shouldn't block train/upload. The downside is it also masks real
    # zero-output failures — runs 25220681021 (transformers ImportError,
    # fixed in b13e6c8) and 25222655513 (MPS thread-safety SIGTRAP, fixed
    # in 2e649c8) both completed "successfully" while writing nothing,
    # caught only by manually inspecting llm_extras counts in the DB.
    # When there's a non-trivial pending queue but every listing produced
    # an error, raise typer.Exit(2) and emit a workflow ``::warning::`` so
    # the failure surfaces in the run summary even though the step keeps
    # going. Skip the check on dry-run (zero writes is the point).
    if not dry_run and len(pending) >= 50 and updated_count == 0:
        print(
            f"::warning::verify-photos processed 0 of {len(pending)} pending "
            "listings — likely a startup crash; check logs."
        )
        raise typer.Exit(2)


@app.command()
def stats():
    """Show current market stats."""
    init_db()
    session = get_session()
    df = get_listings_df(session)

    if df.empty:
        console.print("[yellow]No data yet. Run 'scrape' first.[/yellow]")
        raise typer.Exit()

    table = Table(title="OLX.pt Market Overview")
    table.add_column("Brand")
    table.add_column("Model")
    table.add_column("Count", justify="right")
    table.add_column("Median", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("Max", justify="right")

    active = df[df["is_active"] & df["price_eur"].notna()]
    if active.empty:
        console.print("[yellow]No active listings with prices.[/yellow]")
        raise typer.Exit()

    grouped = (
        active.groupby(["brand", "model"])
        .agg(count=("price_eur", "size"), median=("price_eur", "median"),
             min_p=("price_eur", "min"), max_p=("price_eur", "max"))
        .reset_index()
        .sort_values("count", ascending=False)
    )

    for _, row in grouped.head(30).iterrows():
        table.add_row(
            row["brand"], row["model"], str(row["count"]),
            f"{row['median']:,.0f} EUR", f"{row['min_p']:,.0f} EUR", f"{row['max_p']:,.0f} EUR",
        )

    console.print(table)

    city_table = Table(title="Listings by District")
    city_table.add_column("District")
    city_table.add_column("Count", justify="right")
    city_counts = active["district"].value_counts().head(15)
    for district, count in city_counts.items():
        if district:
            city_table.add_row(str(district), str(count))
    console.print(city_table)


@app.command()
def dashboard():
    """Launch Streamlit dashboard."""
    import subprocess
    from pathlib import Path
    app_path = Path(__file__).parent / "dashboard" / "app.py"
    console.print("[bold]Launching dashboard...[/bold]")
    subprocess.run(["streamlit", "run", str(app_path)])


@app.command()
def alerts():
    """Check for deal alerts and send to Telegram."""
    init_db()
    from src.alerts.telegram_bot import check_and_send_alerts
    count = check_and_send_alerts()
    if count:
        console.print(f"[green]Sent {count} deal alerts to Telegram.[/green]")
    else:
        console.print("[yellow]No new alerts to send.[/yellow]")


@app.command("train-model")
def train_model():
    """Train price model and save to data/price_model.joblib.

    Intended for CI: runs after scrape+enrich, uploads to the release alongside
    the DB. Dashboard loads the shipped model and never trains inline.
    """
    from src.storage.repository import get_listings_df
    from src.analytics.computed_columns import enrich_listings
    from src.analytics.turnover import compute_turnover_stats
    from src.analytics.price_model import (
        train_price_model, save_model, save_importance,
        save_grouped_importance, save_shap_importance,
    )
    from src.dashboard.data_loader import prepare_active_for_model

    init_db()
    session = get_session()
    listings = get_listings_df(session)
    session.close()

    if listings.empty:
        console.print("[yellow]No listings in DB — nothing to train on.[/yellow]")
        raise typer.Exit(1)

    listings = enrich_listings(listings)
    if "real_mileage_km" in listings.columns:
        listings["mileage_km"] = listings["real_mileage_km"].fillna(listings["mileage_km"])

    turnover = compute_turnover_stats(listings)
    active = prepare_active_for_model(listings, turnover=turnover, include_sold=True)

    n_active_only = int(active["is_active"].sum()) if "is_active" in active.columns else len(active)
    n_sold = len(active) - n_active_only
    console.print(
        f"Training on {len(active)} listings ({n_active_only} active + "
        f"{n_sold} sold; sold rows get a days-on-market target/weight haircut)..."
    )
    result = train_price_model(active)
    if result is None:
        console.print("[red]Training failed: insufficient data after filtering.[/red]")
        raise typer.Exit(1)

    (
        models, cat_maps, metrics, oof_preds, calibrator, uncertainty,
        importance_df, grouped_importance_df, shap_importance_df,
    ) = result
    save_model(
        models, cat_maps, metrics,
        oof_preds=oof_preds,
        median_calibrator=calibrator,
        uncertainty_bundle=uncertainty,
    )

    # All three importance frames are CV-honest — computed on val folds
    # inside _cv_metrics during the same 5-fold CV that produced OOF preds.
    # No extra fits, no in-sample bias to disclaim.
    save_importance(importance_df)
    save_grouped_importance(grouped_importance_df)
    save_shap_importance(shap_importance_df)

    console.print(
        f"[green]Model saved.[/green] MAE={metrics['mae']:.0f} € · "
        f"MAPE={metrics['mape']:.1f}% · R²={metrics['r2']:.3f} · "
        f"n={metrics['n_samples']} · "
        f"importance rows={len(importance_df)} · "
        f"grouped rows={len(grouped_importance_df)} · "
        f"shap rows={len(shap_importance_df)}"
    )


@app.command(name="eval-model")
def eval_model(
    time_backtest: bool = typer.Option(
        False, "--time-backtest",
        help="Run rolling-window time-aware backtest (slow: retrains 4 models).",
    ),
    backtest_splits: int = typer.Option(
        5, "--splits", help="Number of folds for time backtest.",
    ),
    top_n_worst: int = typer.Option(
        20, "--top-n", help="Number of worst residuals to print.",
    ),
):
    """Print quality diagnostics for the saved price model.

    Reads ``data/price_model.joblib`` (no retraining) and reports:
      - global MAE/MAPE/R²/coverage on out-of-fold predictions
      - per-bucket slices (price tier, year, brand) so you can see *where*
        the model is wrong, not just the headline number
      - top-N listings with the largest absolute residual %
      - reliability curve: empirical 80% coverage by predicted-price decile

    With ``--time-backtest`` it also retrains on rolling windows (~30 min on
    5k listings) and persists the result to ``data/price_backtest.json``.
    """
    from src.storage.repository import get_listings_df
    from src.analytics.computed_columns import enrich_listings
    from src.analytics.turnover import compute_turnover_stats
    from src.analytics.price_model import load_model
    from src.analytics.model_eval import (
        evaluate_oof, worst_residuals, reliability_curve,
        time_backtest as run_time_backtest, save_backtest,
    )
    from src.dashboard.data_loader import prepare_active_for_model

    saved = load_model(max_age_hours=14 * 24)
    if saved is None:
        console.print(
            "[red]No fresh model bundle found.[/red] "
            "Run [bold]train-model[/bold] first."
        )
        raise typer.Exit(1)
    _models, _maps, metrics, oof_preds, _calibrator = saved

    if not oof_preds:
        console.print(
            "[yellow]Bundle has no oof_preds — likely trained with an older "
            "schema. Retrain to populate them.[/yellow]"
        )
        raise typer.Exit(1)

    init_db()
    session = get_session()
    listings = get_listings_df(session)
    session.close()
    if listings.empty:
        console.print("[red]No listings in DB.[/red]")
        raise typer.Exit(1)

    listings = enrich_listings(listings)
    if "real_mileage_km" in listings.columns:
        listings["mileage_km"] = listings["real_mileage_km"].fillna(listings["mileage_km"])
    turnover = compute_turnover_stats(listings)
    # Match the training-side row set so eval_oof can join sold-row OOF
    # predictions back to the rows they were trained on.
    active = prepare_active_for_model(listings, turnover=turnover, include_sold=True)

    # --- Global ---
    report = evaluate_oof(active, oof_preds)
    g = report["global"]
    if g["n"] == 0:
        console.print(
            "[yellow]No overlap between active listings and bundled OOF preds. "
            "DB may have rotated since the model was trained.[/yellow]"
        )
        raise typer.Exit(1)

    console.print()
    console.print(f"[bold cyan]OOF diagnostics[/bold cyan] (n={g['n']:,})")
    console.print(
        f"  MAE: €{g['mae']:,.0f} · MAPE: {g['mape']:.1f}% · "
        f"R²: {g['r2']:.3f}"
    )
    console.print(
        f"  Coverage(80%): {g['coverage_80']:.1%} · "
        f"Bias: {g['bias_pct']:+.2f}%"
    )
    if g["n_inverted_band"]:
        console.print(
            f"  [red]⚠ {g['n_inverted_band']} inverted bands found "
            f"(should be 0 — bundle may be from before the crossing-repair fix)[/red]"
        )

    # --- Bucket tables ---
    def _print_bucket_table(title: str, df: pd.DataFrame) -> None:
        if df.empty:
            return
        t = Table(title=title, show_header=True, header_style="bold")
        t.add_column("Bucket")
        t.add_column("n", justify="right")
        t.add_column("MAE €", justify="right")
        t.add_column("MAPE %", justify="right")
        t.add_column("Bias %", justify="right")
        t.add_column("Cov 80%", justify="right")
        for _, r in df.iterrows():
            bias = r["bias_pct"]
            bias_color = "green" if abs(bias) < 2 else ("yellow" if abs(bias) < 5 else "red")
            cov = r["coverage_80"]
            cov_color = "green" if 0.75 <= cov <= 0.85 else "yellow"
            t.add_row(
                str(r["bucket"]),
                f"{int(r['n']):,}",
                f"{r['mae']:,.0f}",
                f"{r['mape']:.1f}",
                f"[{bias_color}]{bias:+.2f}[/{bias_color}]",
                f"[{cov_color}]{cov:.1%}[/{cov_color}]",
            )
        console.print(t)

    _print_bucket_table("By price bucket", report["by_price"])
    _print_bucket_table("By year bucket", report["by_year"])
    _print_bucket_table("By brand (top 10)", report["by_brand"])

    # --- Reliability curve ---
    rel = reliability_curve(active, oof_preds, n_bins=10)
    if not rel.empty:
        t = Table(title="Reliability curve (target 80% coverage per decile)",
                  show_header=True, header_style="bold")
        t.add_column("Pred range €")
        t.add_column("n", justify="right")
        t.add_column("Empirical cov", justify="right")
        t.add_column("Gap", justify="right")
        for _, r in rel.iterrows():
            gap = r["calibration_gap"]
            gap_color = "green" if abs(gap) < 0.05 else ("yellow" if abs(gap) < 0.10 else "red")
            t.add_row(
                f"{r['predicted_min']:,.0f} – {r['predicted_max']:,.0f}",
                f"{int(r['n']):,}",
                f"{r['empirical_coverage']:.1%}",
                f"[{gap_color}]{gap:+.3f}[/{gap_color}]",
            )
        console.print(t)

    # --- Worst residuals ---
    worst = worst_residuals(active, oof_preds, n=top_n_worst)
    if not worst.empty:
        t = Table(
            title=f"Top {top_n_worst} worst residuals (by |residual %|)",
            show_header=True, header_style="bold",
        )
        t.add_column("olx_id")
        t.add_column("Brand")
        t.add_column("Model")
        t.add_column("Year", justify="right")
        t.add_column("Price €", justify="right")
        t.add_column("Pred €", justify="right")
        t.add_column("Δ %", justify="right")
        t.add_column("In band")
        for _, r in worst.iterrows():
            pct = r["abs_residual_pct"]
            pct_color = "red" if pct > 50 else ("yellow" if pct > 25 else "white")
            sign = "−" if r["residual"] < 0 else "+"
            t.add_row(
                str(r.get("olx_id", "")),
                str(r.get("brand", "")),
                str(r.get("model", "")),
                f"{int(r['year']) if pd.notna(r.get('year')) else '–'}",
                f"{r['price_eur']:,.0f}",
                f"{r['oof_median']:,.0f}",
                f"[{pct_color}]{sign}{pct:.1f}[/{pct_color}]",
                "✓" if r.get("in_band") else "✗",
            )
        console.print(t)

    # --- Time backtest (opt-in, slow) ---
    if time_backtest:
        console.print()
        console.print("[bold]Running time-aware backtest...[/bold]")
        n_per_q = metrics.get("best_n_estimators_per_q") or {
            name: 400 for name in ("low", "median", "high")
        }
        bt = run_time_backtest(
            listings,
            n_splits=backtest_splits,
            n_estimators_per_q=n_per_q,
            # Apply the bundle's active CQR widening so backtest coverage
            # reflects the deployed model, not raw uncalibrated bands.
            conformal_q=float(metrics.get("conformal_q", 0.0)),
        )
        if bt.empty:
            console.print("[yellow]Backtest returned no folds — too little data.[/yellow]")
        else:
            t = Table(title="Time backtest (rolling window, train→next slice)",
                      show_header=True, header_style="bold")
            t.add_column("Fold", justify="right")
            t.add_column("Train until")
            t.add_column("Test from → to")
            t.add_column("n_train", justify="right")
            t.add_column("n_test", justify="right")
            t.add_column("MAE €", justify="right")
            t.add_column("MAPE %", justify="right")
            t.add_column("Bias %", justify="right")
            t.add_column("Cov 80%", justify="right")
            for _, r in bt.iterrows():
                t.add_row(
                    str(r["fold"]),
                    str(r["train_until"])[:10],
                    f"{str(r['test_from'])[:10]} → {str(r['test_to'])[:10]}",
                    f"{int(r['n_train']):,}",
                    f"{int(r['n_test']):,}",
                    f"{r['mae']:,.0f}",
                    f"{r['mape']:.1f}",
                    f"{r['bias_pct']:+.2f}",
                    f"{r['coverage_80']:.1%}",
                )
            console.print(t)
            save_backtest(bt)
            console.print(
                "[green]Backtest saved to data/price_backtest.json — "
                "dashboard will pick it up.[/green]"
            )


def _load_predictions_for_model_consumers(active: "pd.DataFrame"):
    """Helper: get aligned price-model predictions for the active set.

    Returns a DataFrame with predicted_price / fair_price_low /
    fair_price_high indexed to ``active.index``, or None when no fresh
    price model bundle exists. Both ``train-anomaly`` and
    ``train-hazard`` need this because residual_pct + band_pct are
    their dominant features and they have to be aligned to the same
    index the trainer iterates.
    """
    from src.analytics.price_model import load_model as load_price_model
    from src.analytics.price_model import predict_prices

    saved = load_price_model(max_age_hours=14 * 24)
    if saved is None:
        return None
    models, cat_maps, metrics, oof_preds, calibrator, uncertainty = saved
    edges_raw = metrics.get("conformal_q_bucket_edges")
    bucket_edges = [tuple(e) for e in edges_raw] if edges_raw else None
    return predict_prices(
        models, cat_maps, active,
        conformal_q=metrics.get("conformal_q", 0.0),
        oof_preds=oof_preds,
        median_calibrator=calibrator,
        conformal_q_per_bucket=metrics.get("conformal_q_per_bucket"),
        conformal_q_bucket_edges=bucket_edges,
        uncertainty_bundle=uncertainty,
    )


@app.command("train-anomaly")
def train_anomaly():
    """Train IsolationForest anomaly model and save to
    ``data/anomaly_model.joblib``.

    Mirrors ``train-model``: runs after scrape+enrich, intended for CI
    so the dashboard always loads a fresh bundle. Pulls price-model
    predictions when available so residual_pct + band_pct enter the
    feature set; falls back to base features otherwise.
    """
    from src.storage.repository import get_listings_df
    from src.analytics.computed_columns import enrich_listings
    from src.analytics.turnover import compute_turnover_stats
    from src.analytics.anomaly import save_model, train_anomaly_detector
    from src.dashboard.data_loader import prepare_active_for_model

    init_db()
    session = get_session()
    listings = get_listings_df(session)
    session.close()
    if listings.empty:
        console.print("[yellow]No listings in DB — nothing to train on.[/yellow]")
        raise typer.Exit(1)

    listings = enrich_listings(listings)
    if "real_mileage_km" in listings.columns:
        listings["mileage_km"] = (
            listings["real_mileage_km"].fillna(listings["mileage_km"])
        )
    turnover = compute_turnover_stats(listings)
    active = prepare_active_for_model(listings, turnover=turnover, include_sold=True)

    predictions_df = _load_predictions_for_model_consumers(active)
    if predictions_df is None:
        console.print(
            "[yellow]No fresh price model — anomaly trains on base "
            "features only. Run train-model first to enable "
            "residual_pct / band_pct.[/yellow]"
        )

    bundle = train_anomaly_detector(active, predictions_df)
    if bundle is None:
        console.print(
            "[red]Training failed: insufficient data after NaN filter.[/red]"
        )
        raise typer.Exit(1)
    save_model(bundle)
    console.print(
        f"[green]Anomaly model saved.[/green] schema=v{bundle['schema_version']} · "
        f"contamination={bundle['contamination']} · "
        f"uses_predictions={bundle['uses_predictions']} · "
        f"n_samples={bundle['n_samples']}"
    )


@app.command("train-hazard")
def train_hazard(
    horizon_days: int = typer.Option(
        30, "--horizon", help="Sale-window horizon in days (default 30).",
    ),
):
    """Train per-listing hazard model (P(sold within horizon)) and
    save to ``data/hazard_model.joblib``.

    Mirrors ``train-model`` for the binary classifier. Pulls price-model
    predictions for residual_pct + band_pct features when available.
    """
    from src.storage.repository import get_listings_df
    from src.analytics.computed_columns import enrich_listings
    from src.analytics.turnover import compute_turnover_stats
    from src.analytics.hazard import save_model, train_hazard_model
    from src.dashboard.data_loader import prepare_active_for_model

    init_db()
    session = get_session()
    listings = get_listings_df(session)
    session.close()
    if listings.empty:
        console.print("[yellow]No listings in DB — nothing to train on.[/yellow]")
        raise typer.Exit(1)

    listings = enrich_listings(listings)
    if "real_mileage_km" in listings.columns:
        listings["mileage_km"] = (
            listings["real_mileage_km"].fillna(listings["mileage_km"])
        )
    turnover = compute_turnover_stats(listings)
    active = prepare_active_for_model(listings, turnover=turnover, include_sold=True)

    predictions_df = _load_predictions_for_model_consumers(active)
    if predictions_df is None:
        console.print(
            "[yellow]No fresh price model — hazard trains without "
            "residual_pct / band_pct features.[/yellow]"
        )

    bundle = train_hazard_model(
        active, predictions_df, horizon_days=horizon_days,
    )
    if bundle is None:
        console.print(
            "[red]Training failed: insufficient labeled data after "
            "censoring (active+young rows are dropped) and price-NaN "
            "filter.[/red]"
        )
        raise typer.Exit(1)
    save_model(bundle)
    m = bundle["metrics"]
    console.print(
        f"[green]Hazard model saved.[/green] schema=v{bundle['schema_version']} · "
        f"horizon={bundle['horizon_days']}d · AUC={m['auc']:.3f} · "
        f"logloss={m['logloss']:.3f} · split={m['split_mode']} · "
        f"n_train={m['n_train']} · n_val={m['n_val']} · "
        f"censored={m['n_dropped_censored']}"
    )


@app.command()
def init():
    """Initialize database (create tables)."""
    init_db()
    console.print("[green]Database initialized.[/green]")





if __name__ == "__main__":
    app()
