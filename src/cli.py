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
from src.models.generations import get_generation
from src.storage.repository import (
    add_price_snapshot, compute_market_stats, deduplicate_cross_platform,
    get_duplicate_ids, get_listings_df, mark_inactive, upsert_listing,
    upsert_unmatched,
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
        # of descriptions in RAM. 500 ≈ one hour of LLM headroom on M1 8 GB.
        llm_in = multiprocessing.Queue(maxsize=500)
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

    def _on_batch(batch):
        nonlocal sent_to_llm, skipped_llm
        for listing in batch:
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
            h = _desc_hash(listing.description)
            if enriched_hashes.get(listing.olx_id) == h:
                skipped_llm += 1
                db_queue.put((listing, None))
                continue
            llm_in.put((listing.olx_id, listing.title, listing.description))
            sent_to_llm += 1
        log.info("Page done: %d listings -> %d sent to LLM, %d skipped", len(batch), sent_to_llm, skipped_llm)

    with OlxScraper(config) as scraper:
        raw_listings = scraper.scrape_all(
            on_batch_ready=_on_batch,
            skip_enrichment_ids=duplicate_ids,
            known_ids=known_ids,
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
            skip_enrichment_ids=duplicate_ids,
            known_ids=known_ids,
        )
    raw_listings.extend(sv_listings)

    # Collect scraped IDs now; mark_inactive runs after DB worker finishes
    # to avoid "database is locked" from concurrent SQLite writers.
    scraped_ids = set(raw_by_id.keys())

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
                    olx_id, _ = item
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
        if scraped_ids:
            try:
                s = get_session()
                mark_inactive(s, scraped_ids)
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
                olx_id, _ = item
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
    if scraped_ids:
        log.info("Marking inactive listings (%d scraped IDs)...", len(scraped_ids))
        mark_inactive(final_session, scraped_ids)
        final_session.commit()
    log.info("Final deduplication...")
    dedup_count = deduplicate_cross_platform(final_session)
    if dedup_count:
        log.info("Marked %d cross-platform duplicates", dedup_count)
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
                listing.llm_extras = json.dumps(result, ensure_ascii=False)
                listing._llm_extras = result
                corrections = correct_listing_data(listing)
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
        compute_permutation_importance,
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
    active = prepare_active_for_model(listings, turnover=turnover)

    console.print(f"Training on {len(active)} active listings...")
    result = train_price_model(active)
    if result is None:
        console.print("[red]Training failed: insufficient data after filtering.[/red]")
        raise typer.Exit(1)

    models, cat_maps, metrics, oof_preds, calibrator, text_pipeline = result
    save_model(
        models, cat_maps, metrics,
        oof_preds=oof_preds,
        median_calibrator=calibrator,
        text_pipeline=text_pipeline,
    )

    console.print("Computing permutation importance...")
    importance_df = compute_permutation_importance(
        models, cat_maps, active, text_pipeline=text_pipeline,
    )
    save_importance(importance_df)

    console.print(
        f"[green]Model saved.[/green] MAE={metrics['mae']:.0f} € · "
        f"MAPE={metrics['mape']:.1f}% · R²={metrics['r2']:.3f} · "
        f"n={metrics['n_samples']} · "
        f"importance rows={len(importance_df)}"
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
    _models, _maps, metrics, oof_preds, _calibrator, _text_pipeline = saved

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
    active = prepare_active_for_model(listings, turnover=turnover)

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


@app.command()
def init():
    """Initialize database (create tables)."""
    init_db()
    console.print("[green]Database initialized.[/green]")





if __name__ == "__main__":
    app()
