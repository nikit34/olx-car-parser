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
    """Find undervalued listings and rank by flip potential.

    Scoring factors:
    - Undervaluation: regression-based fair price vs actual (fallback to median)
    - Liquidity: how fast this model sells (avg_days_to_sell)
    - Trend: is the market price rising or falling
    """
    if listings_df.empty:
        return pd.DataFrame()

    import numpy as np
    from src.models.generations import get_generation
    from src.analytics.turnover import compute_turnover_stats

    signals = []
    active = listings_df[listings_df["is_active"]].copy() if "is_active" in listings_df.columns else listings_df.copy()

    active["generation"] = active.apply(
        lambda r: get_generation(r["brand"], r["model"], r.get("year")), axis=1
    )

    # --- Generation-level price stats ---
    priced = active[active["price_eur"].notna() & active["generation"].notna()]
    gen_stats = (
        priced
        .groupby(["brand", "model", "generation"])
        .agg(gen_median=("price_eur", "median"), gen_count=("price_eur", "count"))
        .reset_index()
    )

    # --- Regression per generation: price ~ year + mileage ---
    gen_regressions = {}
    for (brand, model, gen), group in priced.groupby(["brand", "model", "generation"]):
        subset = group[group["mileage_km"].notna() & group["year"].notna()]
        if len(subset) < 5:
            continue
        X = np.column_stack([
            subset["year"].values.astype(float),
            subset["mileage_km"].values.astype(float),
            np.ones(len(subset)),
        ])
        y = subset["price_eur"].values.astype(float)
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            gen_regressions[(brand, model, gen)] = coeffs
        except Exception:
            pass

    # --- Liquidity: avg days to sell per brand+model ---
    liquidity_map: dict[tuple, float] = {}
    turnover = compute_turnover_stats(listings_df)
    if not turnover.empty:
        for _, row in turnover.iterrows():
            days = row.get("avg_days_to_sell")
            if pd.notna(days):
                liquidity_map[(row["brand"], row["model"])] = float(days)

    # --- Price trend from history (last 60 days) ---
    trend_map: dict[tuple, float] = {}
    if not history_df.empty and "date" in history_df.columns:
        hist = history_df.copy()
        hist["date"] = pd.to_datetime(hist["date"])
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=60)
        recent = hist[hist["date"] >= cutoff]
        for (brand, model), group in recent.groupby(["brand", "model"]):
            sorted_g = group.sort_values("date")
            if len(sorted_g) < 2:
                continue
            old_med = sorted_g.iloc[0]["median_price_eur"]
            new_med = sorted_g.iloc[-1]["median_price_eur"]
            if old_med and old_med > 0:
                trend_map[(brand, model)] = round((new_med - old_med) / old_med * 100, 1)

    # --- Score each listing ---
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

        if not median or listing["price_eur"] >= median * 0.85:
            continue

        price = listing["price_eur"]
        year = listing.get("year")
        mileage = listing.get("mileage_km")
        brand = listing["brand"]
        model = listing["model"]

        # 1. Undervaluation (regression if ≥5 samples, else median)
        reg_key = (brand, model, generation)
        if reg_key in gen_regressions and pd.notna(year) and pd.notna(mileage):
            coeffs = gen_regressions[reg_key]
            predicted = max(coeffs[0] * float(year) + coeffs[1] * float(mileage) + coeffs[2], 0)
            if predicted > 0:
                undervaluation_pct = round((1 - price / predicted) * 100, 1)
            else:
                predicted = median
                undervaluation_pct = round((1 - price / median) * 100, 1)
        else:
            predicted = median
            undervaluation_pct = round((1 - price / median) * 100, 1)

        discount_pct = round((1 - price / median) * 100, 1)

        # 2. Liquidity multiplier (30 days = 1.0 baseline)
        days_to_sell = liquidity_map.get((brand, model))
        if days_to_sell and days_to_sell > 0:
            liquidity_mult = min(max(30 / days_to_sell, 0.5), 2.0)
        else:
            liquidity_mult = 1.0

        # 3. Trend multiplier (rising market = bonus)
        trend_pct = trend_map.get((brand, model), 0.0)
        trend_mult = min(max(1 + trend_pct / 100, 0.8), 1.2)

        flip_score = round(undervaluation_pct * liquidity_mult * trend_mult, 1)

        sig = {
            "olx_id": listing.get("olx_id", ""),
            "url": listing.get("url", ""),
            "brand": brand,
            "model": model,
            "year": year,
            "generation": generation or "",
            "price_eur": price,
            "predicted_price": round(predicted),
            "median_price_eur": round(median),
            "discount_pct": discount_pct,
            "undervaluation_pct": undervaluation_pct,
            "avg_days_to_sell": days_to_sell,
            "price_trend_pct": trend_pct,
            "flip_score": flip_score,
            "sample_size": sample,
            "city": listing.get("city", ""),
            "district": listing.get("district", ""),
            "mileage_km": mileage,
            "fuel_type": listing.get("fuel_type", ""),
        }
        for col in ("days_listed", "price_change_eur", "price_change_pct", "eur_per_km"):
            if col in listing.index:
                sig[col] = listing[col]
        signals.append(sig)

    df = pd.DataFrame(signals)
    if not df.empty:
        df = df.sort_values("flip_score", ascending=False)
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
