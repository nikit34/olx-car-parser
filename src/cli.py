"""CLI for OLX.pt Car Parser."""

import logging
import sys

import typer
from rich.console import Console
from rich.table import Table

from src.parser.scraper import OlxScraper, ScraperConfig
from src.storage.database import get_session, init_db
from src.storage.repository import (
    add_price_snapshot, compute_market_stats, get_listings_df,
    mark_inactive, upsert_listing,
)

app = typer.Typer(help="OLX.pt Car Parser — scrape, store, analyze")
console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


@app.command()
def scrape(
    pages: int = typer.Option(999, help="Max pages to scrape (stops automatically when no more)"),
    delay_min: float = typer.Option(2.0, help="Min delay between requests (sec)"),
    delay_max: float = typer.Option(5.0, help="Max delay between requests (sec)"),
    private_only: bool = typer.Option(True, help="Only private sellers (Particular)"),
):
    """Scrape OLX.pt car listings and save to database."""
    init_db()
    session = get_session()

    config = ScraperConfig(
        max_pages=pages,
        delay_min=delay_min,
        delay_max=delay_max,
        private_only=private_only,
    )

    console.print(f"[bold]Starting scrape of OLX.pt: up to {pages} pages...[/bold]")

    with OlxScraper(config) as scraper:
        raw_listings = scraper.scrape_all()

    if not raw_listings:
        console.print("[red]No listings scraped. OLX may have changed structure or blocked request.[/red]")
        raise typer.Exit(1)

    console.print(f"Scraped [bold]{len(raw_listings)}[/bold] listings. Saving to database...")

    saved = 0
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
        }

        listing = upsert_listing(session, data)

        if raw.price_eur is not None:
            add_price_snapshot(session, listing.id, raw.price_eur, raw.negotiable)

        active_ids.add(raw.olx_id)
        saved += 1

    mark_inactive(session, active_ids)
    session.commit()

    console.print(f"[green]Saved {saved} listings to database.[/green]")

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
