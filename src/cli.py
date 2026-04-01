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
            "condition": raw.condition, "origin": raw.origin,
            "registration_month": raw.registration_month,
            "registration_plate": raw.registration_plate,
            "city": raw.city, "district": raw.district,
            "seller_type": raw.seller_type, "description": raw.description,
            "llm_extras": json.dumps(llm_data, ensure_ascii=False) if llm_data else None,
            "llm_description_hash": _desc_hash(raw.description) if llm_data and raw.description else None,
            "desc_mentions_repair": corrections.get("desc_mentions_repair"),
            "desc_mentions_accident": corrections.get("desc_mentions_accident"),
            "real_mileage_km": corrections.get("real_mileage_km"),
            "desc_mentions_num_owners": corrections.get("desc_mentions_num_owners"),
            "desc_mentions_customs_cleared": corrections.get("desc_mentions_customs_cleared"),
            "right_hand_drive": corrections.get("right_hand_drive"),
            "urgency": corrections.get("urgency"),
            "warranty": corrections.get("warranty"),
            "tuning_or_mods": json.dumps(corrections["tuning_or_mods"], ensure_ascii=False) if corrections.get("tuning_or_mods") else None,
            "taxi_fleet_rental": corrections.get("taxi_fleet_rental"),
            "tires_condition": corrections.get("tires_condition"),
            "first_owner_selling": corrections.get("first_owner_selling"),
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

    # --- Streaming pipeline: Scraper -> LLM -> DB (all run in parallel) ---

    # Queues
    llm_in: multiprocessing.Queue = multiprocessing.Queue()
    llm_out: multiprocessing.Queue = multiprocessing.Queue()
    db_queue: Queue = Queue()

    # LLM workers (separate processes — no GIL contention)
    llm_shutdown = multiprocessing.Event()
    from src.parser.llm_enrichment import _get_config as _get_llm_config
    llm_cfg = _get_llm_config()
    num_workers = llm_workers if llm_workers is not None else llm_cfg.get("llm_workers", 1)
    llm_procs = []
    for _ in range(num_workers):
        p = multiprocessing.Process(target=_llm_worker, args=(llm_in, llm_out, llm_shutdown), daemon=True)
        p.start()
        llm_procs.append(p)

    # Merger thread: pairs LLM results with raw listings -> db_queue
    raw_by_id: dict = {}
    merger = threading.Thread(target=_llm_to_db, args=(llm_out, raw_by_id, db_queue), daemon=True)
    merger.start()

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
        llm_shutdown.set()
        # Drain LLM input queue: save remaining listings without enrichment
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
        # Stop merger and DB worker
        llm_out.put(None)
        merger.join(timeout=10)
        db_queue.put(None)
        db_thread.join(timeout=30)
        # Safe to write now — db_worker is done
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

    # 1. Stop LLM workers
    for _ in llm_procs:
        llm_in.put(None)
    log.info("Scraping done (%d OLX + %d SV = %d total). Waiting for pipeline to drain...",
             len(raw_listings) - len(sv_listings), len(sv_listings), len(raw_listings))
    llm_shutdown.set()
    for p in llm_procs:
        p.join()

    # 2. Drain leftover llm_in (workers may have exited early on failures)
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

    # 3. Stop merger (all LLM results flushed)
    llm_out.put(None)
    merger.join()

    # 4. Stop DB worker (all items queued)
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
def enrich():
    """Enrich unenriched listings already in DB with LLM.

    Processes only listings that don't have llm_extras yet.
    Use this to backfill listings scraped before LLM was set up.
    """
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

    log.info("Enriching %d listings with Ollama (%s)...", len(pending), cfg['ollama_model'])

    enriched = 0
    failures = 0
    for listing in pending:
        if not listing.description or len(listing.description.strip()) < 20:
            listing.llm_extras = "{}"
            continue

        result = enrich_from_description(listing.description)
        if result:
            listing.llm_extras = json.dumps(result, ensure_ascii=False)
            listing._llm_extras = result
            corrections = correct_listing_data(listing)
            for field, value in corrections.items():
                if hasattr(listing, field):
                    setattr(listing, field, value)
            enriched += 1
            failures = 0
            if enriched % 25 == 0:
                session.commit()
                log.info("Enrich progress: %d / %d", enriched, len(pending))
        else:
            failures += 1
            if failures >= 5:
                log.error("5 consecutive LLM failures, stopping.")
                break

    session.commit()
    log.info("Enriched %d listings.", enriched)


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


@app.command()
def export_training_data(
    output: str = typer.Option("data/training_data.jsonl", help="Output JSONL path"),
    min_desc_len: int = typer.Option(50, help="Min description length to include"),
):
    """Export enriched listings as JSONL training data for fine-tuning."""
    init_db()
    session = get_session()
    df = get_listings_df(session)

    if df.empty:
        console.print("[yellow]No data. Run 'scrape' first.[/yellow]")
        raise typer.Exit()

    from src.parser.llm_enrichment import EXTRACTION_PROMPT, normalize_llm_extras

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(out_path, "w") as f:
        for _, row in df.iterrows():
            desc = row.get("description") or ""
            extras_raw = row.get("llm_extras")
            if len(desc.strip()) < min_desc_len or not extras_raw:
                continue
            try:
                extras = json.loads(extras_raw) if isinstance(extras_raw, str) else extras_raw
            except (json.JSONDecodeError, TypeError):
                continue
            extras = normalize_llm_extras(extras)

            for col in ("desc_mentions_repair", "desc_mentions_accident", "desc_mentions_num_owners",
                        "desc_mentions_customs_cleared", "real_mileage_km",
                        "urgency", "warranty", "taxi_fleet_rental", "tires_condition", "first_owner_selling"):
                val = row.get(col)
                if val is not None and not (isinstance(val, float) and pd.isna(val)):
                    extras[col] = val

            entry = {
                "messages": [
                    {"role": "user", "content": EXTRACTION_PROMPT + desc[:3000]},
                    {"role": "assistant", "content": json.dumps(extras, ensure_ascii=False)},
                ],
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            count += 1

    console.print(f"[green]Exported {count} training examples to {out_path}[/green]")
    if count < 200:
        console.print(f"[yellow]Tip: need ~500+ examples for good fine-tuning. Keep scraping![/yellow]")


@app.command()
def train_tos(
    target_days: int = typer.Option(30, help="Target sale window in days"),
):
    """Train time-to-sale model and show weights."""
    from src.analytics.time_to_sale import build_tos_dataset, fit_tos_model, compute_tos_stats
    from src.storage.repository import backfill_deactivated_at

    init_db()
    session = get_session()

    # Backfill deactivated_at for old listings
    backfilled = backfill_deactivated_at(session)
    if backfilled:
        console.print(f"[yellow]Backfilled deactivated_at for {backfilled} listings[/yellow]")

    df = get_listings_df(session)
    session.close()

    if df.empty:
        console.print("[yellow]No data. Run 'scrape' first.[/yellow]")
        raise typer.Exit()

    dataset = build_tos_dataset(df)
    sold_count = (~dataset["censored"]).sum() if not dataset.empty else 0
    console.print(f"Dataset: {len(dataset)} listings ({sold_count} sold, {len(dataset) - sold_count} active)")

    if dataset.empty:
        console.print("[yellow]No training data available.[/yellow]")
        raise typer.Exit()

    weights = fit_tos_model(dataset, target_days=target_days)

    table = Table(title=f"Time-to-Sale Weights (target: {target_days}d)")
    table.add_column("Feature")
    table.add_column("Weight", justify="right")
    for k, v in weights.items():
        table.add_row(k, f"{v:.3f}")
    console.print(table)

    # Show per-model stats
    tos_stats = compute_tos_stats(df)
    if not tos_stats.empty:
        stats_table = Table(title="Time-to-Sale by Model (top 20)")
        stats_table.add_column("Brand")
        stats_table.add_column("Model")
        stats_table.add_column("Sold", justify="right")
        stats_table.add_column("Median days", justify="right")
        stats_table.add_column("% under 30d", justify="right")
        stats_table.add_column("Sale rate", justify="right")

        for _, row in tos_stats.head(20).iterrows():
            stats_table.add_row(
                row["brand"], row["model"],
                str(int(row["sold_count"])),
                f"{row['median_days']:.0f}",
                f"{row['pct_under_30d']:.0f}%",
                f"{row['sale_rate_pct']:.0f}%",
            )
        console.print(stats_table)

    # Save weights
    import yaml
    weights_path = Path(__file__).resolve().parent.parent / "data" / "tos_weights.yaml"
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    with open(weights_path, "w") as f:
        yaml.dump({"target_days": target_days, "weights": weights}, f)
    console.print(f"[green]Weights saved to {weights_path}[/green]")


@app.command()
def optimize_price_cmd(
    brand: str = typer.Argument(..., help="Car brand"),
    model: str = typer.Argument(..., help="Car model"),
    price: float = typer.Argument(..., help="Current asking price"),
    year: int = typer.Option(2015, help="Car year"),
    mileage: int = typer.Option(150000, help="Mileage in km"),
    fuel: str = typer.Option("Gasolina", help="Fuel type"),
    target_days: int = typer.Option(30, help="Target sale window in days"),
    min_price: float = typer.Option(None, help="Minimum acceptable price"),
):
    """Optimize price for fastest sale within margin target."""
    import yaml
    from src.analytics.time_to_sale import (
        build_tos_dataset, optimize_price,
        DEFAULT_TOS_WEIGHTS, TOS_FEATURE_COLUMNS,
    )

    init_db()
    session = get_session()
    df = get_listings_df(session)
    session.close()

    # Load trained weights if available
    weights_path = Path(__file__).resolve().parent.parent / "data" / "tos_weights.yaml"
    if weights_path.exists():
        with open(weights_path) as f:
            saved = yaml.safe_load(f) or {}
        weights = saved.get("weights", DEFAULT_TOS_WEIGHTS)
        console.print(f"[dim]Using trained weights from {weights_path}[/dim]")
    else:
        weights = DEFAULT_TOS_WEIGHTS
        console.print("[dim]Using default weights (run 'train-tos' first for better results)[/dim]")

    # Get market median for this brand+model
    matching = df[(df["brand"].str.lower() == brand.lower()) &
                  (df["model"].str.lower() == model.lower()) &
                  df["price_eur"].notna()]
    if matching.empty:
        console.print(f"[red]No listings found for {brand} {model}[/red]")
        raise typer.Exit(1)

    market_median = float(matching["price_eur"].median())
    console.print(f"Market median for {brand} {model}: {market_median:,.0f} EUR ({len(matching)} listings)")

    # Build feature dict
    features = {
        "price_ratio": price / market_median,
        "mileage_norm": min(mileage, 500000) / 500000,
        "year_norm": (year - 2000) / 25,
        "is_diesel": 1.0 if "diesel" in fuel.lower() else 0.0,
        "is_professional": 0.0,
        "price_drop_rate": 0.0,
        "has_accident": 0.0,
        "has_repair": 0.0,
    }

    result = optimize_price(
        features, market_median, weights,
        target_days=target_days,
        min_acceptable_price=min_price,
    )

    console.print()
    console.print(f"[bold]Current price:[/bold] {price:,.0f} EUR (ratio: {price/market_median:.2f})")
    console.print(f"[bold green]Optimal price:[/bold green] {result['optimal_price']:,.0f} EUR "
                  f"(ratio: {result['optimal_ratio']:.2f})")
    console.print(f"[bold]P(sold in {target_days}d):[/bold] {result['sale_probability']:.0%}")
    console.print(f"[bold]Expected value:[/bold] {result['expected_value']:,.0f} EUR")

    # Show price curve
    table = Table(title="Price vs Sale Probability")
    table.add_column("Price", justify="right")
    table.add_column("P(sale)", justify="right")
    table.add_column("Expected €", justify="right")
    table.add_column("", justify="left")
    for point in result["price_curve"][::5]:  # every 5th point
        marker = " ← optimal" if point["price"] == result["optimal_price"] else ""
        table.add_row(
            f"{point['price']:,}",
            f"{point['probability']:.0%}",
            f"{point['expected_value']:,}",
            marker,
        )
    console.print(table)


if __name__ == "__main__":
    app()
