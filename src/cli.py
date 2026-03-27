"""CLI for OLX.pt Car Parser."""

import fcntl
import json
import logging
import sys
from pathlib import Path

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


@app.command()
def scrape(
    pages: int = typer.Option(None, help="Max pages to scrape (default from config)"),
    delay_min: float = typer.Option(None, help="Min delay between requests (sec)"),
    delay_max: float = typer.Option(None, help="Max delay between requests (sec)"),
    private_only: bool = typer.Option(None, help="Only private sellers (Particular)"),
    concurrency: int = typer.Option(None, help="Parallel detail page workers (default 5)"),
):
    """Scrape OLX.pt car listings and save to database."""
    # Prevent concurrent scrapes
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

    with OlxScraper(config) as scraper:
        raw_listings = scraper.scrape_all()

    if not raw_listings:
        console.print("[red]No listings scraped. OLX may have changed structure or blocked request.[/red]")
        raise typer.Exit(1)

    console.print(f"Scraped [bold]{len(raw_listings)}[/bold] listings.")

    # LLM enrichment (if API key configured)
    from src.parser.llm_enrichment import enrich_listings_batch
    llm_count = enrich_listings_batch(raw_listings)
    if llm_count:
        console.print(f"LLM-enriched [bold]{llm_count}[/bold] listings from descriptions.")

    console.print("Saving to database...")

    saved = 0
    unmatched_count = 0
    active_ids = set()
    for raw in raw_listings:
        if not raw.brand and not raw.title:
            continue

        data = {
            "olx_id": raw.olx_id,
            "url": raw.url,
            "title": raw.title,
            "brand": raw.brand,
            "model": raw.model or "",
            "year": raw.year,
            "mileage_km": raw.mileage_km,
            "engine_cc": raw.engine_cc,
            "fuel_type": raw.fuel_type,
            "horsepower": raw.horsepower,
            "transmission": raw.transmission,
            "segment": raw.segment,
            "doors": raw.doors,
            "seats": raw.seats,
            "color": raw.color,
            "condition": raw.condition,
            "origin": raw.origin,
            "registration_month": raw.registration_month,
            "registration_plate": raw.registration_plate,
            "city": raw.city,
            "district": raw.district,
            "seller_type": raw.seller_type,
            "description": raw.description,
            "llm_extras": json.dumps(raw._llm_extras) if hasattr(raw, "_llm_extras") and raw._llm_extras else None,
        }

        # Determine generation — route to main or unmatched table
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

    console.print(f"[green]Saved {saved} listings to database.[/green]")
    if unmatched_count:
        console.print(f"[yellow]{unmatched_count} listings unmatched (no generation) — see dashboard.[/yellow]")

    console.print("Computing market stats...")
    compute_market_stats(session)
    console.print("[green]Done![/green]")


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

    # City breakdown
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
