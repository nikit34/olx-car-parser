"""CLI for OLX.pt Car Parser."""

import fcntl
import json
import logging
import multiprocessing
import sys
from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

_LOCK_PATH = Path(__file__).resolve().parent.parent / "data" / "scrape.lock"

from src.parser.scraper import OlxScraper, ScraperConfig
from src.storage.database import get_session, init_db
from src.models.generations import get_generation
from src.storage.repository import (
    add_price_snapshot, compute_market_stats, get_listings_df,
    mark_inactive, upsert_listing, upsert_unmatched,
)

app = typer.Typer(help="OLX.pt Car Parser — scrape, store, analyze")
console = Console()

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
            continue
        result = enrich_from_description(description)
        if result:
            out_q.put((olx_id, result))
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


@app.command()
def scrape(
    pages: int = typer.Option(None, help="Max pages to scrape (default from config)"),
    delay_min: float = typer.Option(None, help="Min delay between requests (sec)"),
    delay_max: float = typer.Option(None, help="Max delay between requests (sec)"),
    private_only: bool = typer.Option(None, help="Only private sellers (Particular)"),
    concurrency: int = typer.Option(None, help="Parallel detail page workers (default 8)"),
):
    """Scrape OLX.pt car listings, enrich with LLM, save to database.

    Scraper and LLM run in parallel: scraper feeds listings directly to
    LLM worker process via queue (no DB round-trip).
    """
    _LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    lock_file = open(_LOCK_PATH, "w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        console.print("[red]Another scrape is already running. Exiting.[/red]")
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

    console.print(f"[bold]Starting scrape of OLX.pt: up to {config.max_pages} pages...[/bold]")

    # Load existing enrichment hashes to skip unchanged descriptions
    from hashlib import md5
    from src.storage.repository import get_enriched_hashes
    enriched_hashes = get_enriched_hashes(session)
    console.print(f"Already enriched: {len(enriched_hashes)} listings in DB")

    # Start LLM workers as separate processes (no GIL contention with scraper threads)
    # 2 workers = 2 parallel Ollama requests (OLLAMA_NUM_PARALLEL=2)
    num_workers = 2
    llm_in: multiprocessing.Queue = multiprocessing.Queue()
    llm_out: multiprocessing.Queue = multiprocessing.Queue()
    llm_shutdown = multiprocessing.Event()
    llm_procs = []
    for _ in range(num_workers):
        p = multiprocessing.Process(target=_llm_worker, args=(llm_in, llm_out, llm_shutdown), daemon=True)
        p.start()
        llm_procs.append(p)

    # Scraper sends each listing directly to LLM via queue
    sent_to_llm = 0
    skipped_llm = 0

    def _desc_hash(text: str) -> str:
        return md5(text.strip().encode()).hexdigest()

    def _on_batch(batch):
        nonlocal sent_to_llm, skipped_llm
        for listing in batch:
            if not listing.description or len(listing.description.strip()) < 20:
                continue
            h = _desc_hash(listing.description)
            if enriched_hashes.get(listing.olx_id) == h:
                skipped_llm += 1
                continue
            llm_in.put((listing.olx_id, listing.description))
            sent_to_llm += 1
        console.print(f"  Page done: {len(batch)} listings → {sent_to_llm} sent to LLM, {skipped_llm} skipped")

    with OlxScraper(config) as scraper:
        raw_listings = scraper.scrape_all(on_batch_ready=_on_batch)

    # Signal LLM workers: poison pills to drain queue, then shutdown event as fallback
    for _ in llm_procs:
        llm_in.put(None)
    console.print(f"Scraping done ({len(raw_listings)} listings). Waiting for LLM to finish...")
    llm_shutdown.set()
    for p in llm_procs:
        p.join()

    # Collect LLM results
    llm_results: dict[str, dict] = {}
    while not llm_out.empty():
        olx_id, result = llm_out.get_nowait()
        llm_results[olx_id] = result

    console.print(f"LLM enriched [bold]{len(llm_results)}[/bold] / {sent_to_llm} listings.")

    if not raw_listings:
        console.print("[red]No listings scraped.[/red]")
        raise typer.Exit(1)

    # Apply corrections and save to DB
    from src.parser.llm_enrichment import correct_listing_data

    console.print("Saving to database...")
    saved = 0
    unmatched_count = 0
    active_ids: set[str] = set()

    for raw in raw_listings:
        if not raw.brand and not raw.title:
            continue

        # Attach LLM result if available
        llm_data = llm_results.get(raw.olx_id)
        if llm_data:
            raw._llm_extras = llm_data
            corrections = correct_listing_data(raw)
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
            "needs_repair": corrections.get("needs_repair"),
            "had_accident": corrections.get("had_accident"),
            "real_mileage_km": corrections.get("real_mileage_km"),
            "num_owners": corrections.get("num_owners"),
            "customs_cleared": corrections.get("customs_cleared"),
            "estimated_repair_cost_eur": corrections.get("estimated_repair_cost_eur"),
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
            unmatched_count += 1

    mark_inactive(session, active_ids)
    session.commit()

    console.print(f"[green]Saved {saved} listings ({len(llm_results)} enriched).[/green]")
    if unmatched_count:
        console.print(f"[yellow]{unmatched_count} listings unmatched (no generation).[/yellow]")

    console.print("Computing market stats...")
    compute_market_stats(session)
    console.print("[green]Done![/green]")


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
        console.print("[red]Ollama is not running.[/red]")
        raise typer.Exit(1)

    pending = (
        session.query(Listing)
        .filter(Listing.llm_extras.is_(None), Listing.description.isnot(None))
        .all()
    )

    if not pending:
        console.print("[green]All listings already enriched.[/green]")
        return

    console.print(f"[bold]Enriching {len(pending)} listings with Ollama ({cfg['ollama_model']})...[/bold]")

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
                console.print(f"  Progress: {enriched}/{len(pending)}")
        else:
            failures += 1
            if failures >= 5:
                console.print("[red]5 consecutive LLM failures, stopping.[/red]")
                break

    session.commit()
    console.print(f"[green]Enriched {enriched} listings.[/green]")


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

    from src.parser.llm_enrichment import EXTRACTION_PROMPT

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

            for col in ("needs_repair", "had_accident", "num_owners",
                        "customs_cleared", "real_mileage_km", "estimated_repair_cost_eur"):
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


if __name__ == "__main__":
    app()
