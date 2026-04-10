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
    from src.parser.llm_enrichment import enrich_from_description, _ollama_available, _get_config
    from queue import Empty

    log = logging.getLogger("llm_worker")
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])

    cfg = _get_config()
    if not _ollama_available(cfg["ollama_url"]):
        log.warning("Ollama not available at %s, worker exiting.", cfg["ollama_url"])
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
        olx_id, description = item
        if not description or len(description.strip()) < 20:
            out_q.put((olx_id, None))
            continue
        result = enrich_from_description(description)
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
    changed_pairs: set[tuple[str, str]] = set()  # (brand, model) pairs touched since last stats
    processed = 0
    last_maintenance_at = 0  # saved count at last dedup/stats run

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
            changed_pairs.add((raw.brand, raw.model or ""))
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

        # Periodic dedup + market stats every 500 saves (incremental)
        if saved - last_maintenance_at >= 500:
            session.commit()
            try:
                deduplicate_cross_platform(session)
                compute_market_stats(session, changed_pairs=changed_pairs)
                session.commit()
                log.info("Periodic maintenance done at %d saved (%d pairs updated)",
                         saved, len(changed_pairs))
            except Exception as e:
                log.warning("Periodic maintenance failed: %s", e)
                session.rollback()
            changed_pairs.clear()
            last_maintenance_at = saved

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
    skip_llm: bool = typer.Option(True, help="Skip LLM during scrape, use 'enrich' command after"),
):
    """Scrape OLX.pt car listings, enrich with LLM, save to database.

    Streaming pipeline: Scraper -> LLM -> DB all run in parallel.
    Listings are saved to DB as soon as LLM finishes (or immediately if
    no enrichment needed), without waiting for the full scrape to complete.
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
    session.close()
    log.info("Already enriched: %d listings, %d duplicates in DB",
             len(enriched_hashes), len(duplicate_ids))

    # --- Streaming pipeline: Scraper -> [LLM] -> DB ---

    db_queue: Queue = Queue()
    raw_by_id: dict = {}

    # LLM pipeline (optional — skipped by default for speed)
    llm_in: multiprocessing.Queue | None = None
    llm_out: multiprocessing.Queue | None = None
    llm_shutdown = None
    llm_procs = []
    merger = None

    if not skip_llm:
        llm_in = multiprocessing.Queue()
        llm_out = multiprocessing.Queue()
        llm_shutdown = multiprocessing.Event()
        from src.parser.llm_enrichment import _get_config as _get_llm_config
        llm_cfg = _get_llm_config()
        num_workers = llm_workers if llm_workers is not None else llm_cfg.get("llm_workers", 1)
        for _ in range(num_workers):
            p = multiprocessing.Process(target=_llm_worker, args=(llm_in, llm_out, llm_shutdown), daemon=True)
            p.start()
            llm_procs.append(p)
        merger = threading.Thread(target=_llm_to_db, args=(llm_out, raw_by_id, db_queue), daemon=True)
        merger.start()
    else:
        log.info("LLM skipped during scrape (use 'enrich' command after)")

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
            if skip_llm:
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
            llm_in.put((listing.olx_id, listing.description))
            sent_to_llm += 1
        log.info("Page done: %d listings -> %d sent to LLM, %d skipped", len(batch), sent_to_llm, skipped_llm)

    with OlxScraper(config) as scraper:
        raw_listings = scraper.scrape_all(on_batch_ready=_on_batch)

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
        sv_listings = sv_scraper.scrape_all(on_batch_ready=_on_batch)
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

    if not skip_llm:
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
    workers: int = typer.Option(2, help="Parallel LLM workers (default 2)"),
    cheap_first: bool = typer.Option(True, help="Prioritize cheaper listings"),
):
    """Enrich unenriched listings with LLM (parallel).

    Processes listings that don't have llm_extras yet.
    Uses ThreadPoolExecutor for parallel Ollama requests.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from src.parser.llm_enrichment import (
        enrich_from_description, correct_listing_data,
        _ollama_available, _get_config,
    )
    from src.models.listing import Listing

    init_db()
    session = get_session()

    cfg = _get_config()
    if not _ollama_available(cfg["ollama_url"]):
        log.error("Ollama is not running.")
        raise typer.Exit(1)

    pending = (
        session.query(Listing)
        .filter(Listing.llm_extras.is_(None), Listing.description.isnot(None))
        .all()
    )

    if not pending:
        log.info("All listings already enriched.")
        return

    if cheap_first:
        pending.sort(key=lambda l: l.price_snapshots.first().price_eur
                     if l.price_snapshots.first() else float("inf"))

    log.info("Enriching %d listings with %d workers (%s)...",
             len(pending), workers, cfg['ollama_model'])

    enriched = 0
    failed = 0
    consecutive_failures = 0
    batch_size = 25

    def _enrich_one(listing):
        if not listing.description or len(listing.description.strip()) < 20:
            return listing, {}
        result = enrich_from_description(listing.description)
        return listing, result

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for listing in pending:
            futures[pool.submit(_enrich_one, listing)] = listing

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


@app.command()
def init():
    """Initialize database (create tables)."""
    init_db()
    console.print("[green]Database initialized.[/green]")





if __name__ == "__main__":
    app()
