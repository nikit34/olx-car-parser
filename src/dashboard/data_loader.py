"""Load data from SQLite database, fall back to demo data if DB is empty."""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_from_db() -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Try loading real data from SQLite. Returns (listings_df, history_df) or None."""
    db_path = PROJECT_ROOT / "data" / "olx_cars.db"
    if not db_path.exists():
        return None

    try:
        from src.storage.database import init_db, get_session
        from src.storage.repository import get_listings_df, get_price_history_df

        init_db(str(db_path))
        session = get_session()

        listings = get_listings_df(session)
        history = get_price_history_df(session)

        if listings.empty:
            return None

        return listings, history
    except Exception as e:
        print(f"Warning: could not load DB data: {e}")
        return None


def compute_signals(listings_df: pd.DataFrame, history_df: pd.DataFrame) -> pd.DataFrame:
    """Find listings priced significantly below market median."""
    if history_df.empty or listings_df.empty:
        return pd.DataFrame()

    signals = []
    latest_stats = history_df.sort_values("date").groupby(["brand", "model"]).last().reset_index()
    active = listings_df[listings_df["is_active"]] if "is_active" in listings_df.columns else listings_df

    for _, listing in active.iterrows():
        if pd.isna(listing.get("price_eur")):
            continue
        stat = latest_stats[
            (latest_stats["brand"] == listing["brand"])
            & (latest_stats["model"] == listing["model"])
        ]
        if stat.empty:
            continue
        median = stat.iloc[0]["median_price_eur"]
        if median and listing["price_eur"] < median * 0.85:
            discount = round((1 - listing["price_eur"] / median) * 100, 1)
            signals.append({
                "olx_id": listing.get("olx_id", ""),
                "brand": listing["brand"],
                "model": listing["model"],
                "year": listing.get("year"),
                "price_eur": listing["price_eur"],
                "median_price_eur": round(median),
                "discount_pct": discount,
                "city": listing.get("city", ""),
                "district": listing.get("district", ""),
                "mileage_km": listing.get("mileage_km"),
                "fuel_type": listing.get("fuel_type", ""),
            })

    df = pd.DataFrame(signals)
    if not df.empty:
        df = df.sort_values("discount_pct", ascending=False)
    return df


def load_all():
    """Load listings, history, and signals from the database."""
    db_data = load_from_db()

    if db_data is not None:
        listings, history = db_data
        signals = compute_signals(listings, history)
    else:
        listings = pd.DataFrame()
        history = pd.DataFrame()
        signals = pd.DataFrame()

    brands_models = {}
    if not listings.empty:
        for _, row in listings[["brand", "model"]].drop_duplicates().iterrows():
            brand = row["brand"]
            if brand not in brands_models:
                brands_models[brand] = []
            if row["model"] not in brands_models[brand]:
                brands_models[brand].append(row["model"])

    return listings, history, signals, brands_models
