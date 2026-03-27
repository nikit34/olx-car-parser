"""Load data from SQLite database (local or downloaded from GitHub Releases)."""

import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DB_PATH = PROJECT_ROOT / "data" / "olx_cars.db"


_DB_MAX_AGE_SECONDS = 6 * 3600  # re-download every 6 hours on Cloud


def _ensure_db() -> bool:
    """Download DB from GitHub Releases if missing or stale."""
    import time

    needs_download = not DB_PATH.exists()
    if DB_PATH.exists():
        age = time.time() - DB_PATH.stat().st_mtime
        if age > _DB_MAX_AGE_SECONDS and os.environ.get("STREAMLIT_SHARING_MODE"):
            needs_download = True  # stale on Cloud — refresh

    if not needs_download:
        return True

    repo = os.environ.get("GITHUB_REPOSITORY", "nikit34/olx-car-parser")
    if not repo:
        return DB_PATH.exists()

    url = f"https://github.com/{repo}/releases/download/latest-data/olx_cars.db"
    try:
        import httpx

        resp = httpx.get(url, follow_redirects=True, timeout=30)
        if resp.status_code == 200:
            DB_PATH.parent.mkdir(parents=True, exist_ok=True)
            DB_PATH.write_bytes(resp.content)
            print(f"Downloaded database from release ({len(resp.content)} bytes)")
            return True
    except Exception as e:
        print(f"Warning: could not download DB from release: {e}")
    return DB_PATH.exists()


def load_from_db() -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Try loading real data from SQLite. Returns (listings_df, history_df) or None."""
    if not _ensure_db():
        return None

    try:
        from src.storage.database import init_db, get_session
        from src.storage.repository import get_listings_df, get_price_history_df

        init_db(str(DB_PATH))
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
    """Find listings priced significantly below market median (generation-aware)."""
    if history_df.empty or listings_df.empty:
        return pd.DataFrame()

    from src.models.generations import get_generation

    signals = []
    active = listings_df[listings_df["is_active"]].copy() if "is_active" in listings_df.columns else listings_df.copy()

    # Assign generation to each listing
    active["generation"] = active.apply(
        lambda r: get_generation(r["brand"], r["model"], r.get("year")), axis=1
    )

    # Precompute generation-level medians from active listings
    priced = active[active["price_eur"].notna() & active["generation"].notna()]
    gen_stats = (
        priced
        .groupby(["brand", "model", "generation"])["price_eur"]
        .agg(gen_median="median", gen_count="count")
        .reset_index()
    )

    for _, listing in active.iterrows():
        if pd.isna(listing.get("price_eur")):
            continue

        generation = listing.get("generation")
        if not generation:
            continue

        gs = gen_stats[
            (gen_stats["brand"] == listing["brand"])
            & (gen_stats["model"] == listing["model"])
            & (gen_stats["generation"] == generation)
        ]
        if gs.empty:
            continue
        median = gs.iloc[0]["gen_median"]
        sample = int(gs.iloc[0]["gen_count"])

        if median and listing["price_eur"] < median * 0.85:
            discount = round((1 - listing["price_eur"] / median) * 100, 1)
            sig = {
                "olx_id": listing.get("olx_id", ""),
                "url": listing.get("url", ""),
                "brand": listing["brand"],
                "model": listing["model"],
                "year": listing.get("year"),
                "generation": generation or "",
                "price_eur": listing["price_eur"],
                "median_price_eur": round(median),
                "discount_pct": discount,
                "sample_size": sample,
                "city": listing.get("city", ""),
                "district": listing.get("district", ""),
                "mileage_km": listing.get("mileage_km"),
                "fuel_type": listing.get("fuel_type", ""),
            }
            # Carry over computed columns if available
            for col in ("days_listed", "price_change_eur", "price_change_pct", "eur_per_km"):
                if col in listing.index:
                    sig[col] = listing[col]
            signals.append(sig)

    df = pd.DataFrame(signals)
    if not df.empty:
        df = df.sort_values("discount_pct", ascending=False)
    return df


def load_unmatched() -> pd.DataFrame:
    """Load unmatched listings from database."""
    if not _ensure_db():
        return pd.DataFrame()
    try:
        from src.storage.database import init_db, get_session
        from src.storage.repository import get_unmatched_df
        init_db(str(DB_PATH))
        session = get_session()
        return get_unmatched_df(session)
    except Exception as e:
        print(f"Warning: could not load unmatched: {e}")
        return pd.DataFrame()


def load_portfolio() -> pd.DataFrame:
    """Load portfolio deals from database."""
    if not _ensure_db():
        return pd.DataFrame()
    try:
        from src.storage.database import init_db, get_session
        from src.storage.repository import get_portfolio_df
        init_db(str(DB_PATH))
        session = get_session()
        return get_portfolio_df(session)
    except Exception as e:
        print(f"Warning: could not load portfolio: {e}")
        return pd.DataFrame()


def load_all():
    """Load listings, history, signals, brand map, turnover, and portfolio."""
    from src.analytics.computed_columns import enrich_listings
    from src.analytics.turnover import compute_turnover_stats

    db_data = load_from_db()

    if db_data is not None:
        listings, history = db_data
        listings = enrich_listings(listings)
        signals = compute_signals(listings, history)
        turnover = compute_turnover_stats(listings)
    else:
        listings = pd.DataFrame()
        history = pd.DataFrame()
        signals = pd.DataFrame()
        turnover = pd.DataFrame()

    portfolio = load_portfolio()

    brands_models = {}
    if not listings.empty:
        for _, row in listings[["brand", "model"]].drop_duplicates().iterrows():
            brand = row["brand"]
            if brand not in brands_models:
                brands_models[brand] = []
            if row["model"] not in brands_models[brand]:
                brands_models[brand].append(row["model"])

    unmatched = load_unmatched()

    return listings, history, signals, brands_models, turnover, portfolio, unmatched
