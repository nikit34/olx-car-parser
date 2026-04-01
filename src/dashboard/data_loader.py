"""Load data from SQLite database (local or downloaded from GitHub Releases)."""

import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DB_PATH = PROJECT_ROOT / "data" / "olx_cars.db"


_CHECK_INTERVAL_SECONDS = 2 * 3600  # check for new release every 2 hours


def _release_updated(repo: str) -> str | None:
    """Return download URL if the release asset is newer than local DB, else None."""
    import httpx

    api_url = f"https://api.github.com/repos/{repo}/releases/tags/latest-data"
    try:
        resp = httpx.get(api_url, timeout=10)
        if resp.status_code != 200:
            return None
        for asset in resp.json().get("assets", []):
            if asset["name"] == "olx_cars.db":
                remote_ts = asset["updated_at"]  # ISO 8601
                from datetime import datetime, timezone
                remote_dt = datetime.fromisoformat(remote_ts.replace("Z", "+00:00"))
                if DB_PATH.exists():
                    local_dt = datetime.fromtimestamp(DB_PATH.stat().st_mtime, tz=timezone.utc)
                    if remote_dt <= local_dt:
                        return None  # local is up to date
                return asset["browser_download_url"]
    except Exception:
        return None
    return None


def _force_next_check():
    """Reset the check timer so the next _ensure_db() call hits GitHub API."""
    if DB_PATH.exists():
        os.utime(DB_PATH, (0, 0))


def _ensure_db() -> bool:
    """Download DB from GitHub Releases if missing or newer version available."""
    import time

    if not DB_PATH.exists():
        needs_check = True
    else:
        age = time.time() - DB_PATH.stat().st_mtime
        needs_check = age > _CHECK_INTERVAL_SECONDS

    if not needs_check:
        return True

    repo = os.environ.get("GITHUB_REPOSITORY", "nikit34/olx-car-parser")
    if not repo:
        return DB_PATH.exists()

    url = _release_updated(repo)
    if not url:
        return DB_PATH.exists()

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


def _expected_lifespan_km(engine_cc: int | None, fuel_type: str | None) -> float:
    """Estimated km before major engine overhaul, based on displacement and fuel.

    Smaller engines wear faster per km due to higher RPM under load.
    Diesel engines typically last ~40% longer than petrol.
    """
    is_diesel = bool(fuel_type and "diesel" in str(fuel_type).lower())

    if engine_cc is None or engine_cc <= 0:
        return 350_000 if is_diesel else 250_000

    if engine_cc < 1000:
        base = 150_000
    elif engine_cc < 1400:
        base = 200_000
    elif engine_cc < 2000:
        base = 250_000
    elif engine_cc < 2500:
        base = 300_000
    else:
        base = 350_000

    return base * 1.4 if is_diesel else float(base)


def compute_signals(listings_df: pd.DataFrame, history_df: pd.DataFrame) -> pd.DataFrame:
    """Find undervalued listings and rank by flip potential.

    Scoring factors:
    - Undervaluation: regression-based fair price vs actual (fallback to median)
    - Liquidity: how fast this model sells (avg_days_to_sell)
    - Trend: is the market price rising or falling
    - Engine life: remaining engine lifespan based on mileage, displacement and fuel
    """
    if listings_df.empty:
        return pd.DataFrame()

    import numpy as np
    from src.models.generations import get_generation
    from src.analytics.turnover import compute_turnover_stats

    signals = []
    active = listings_df[listings_df["is_active"]].copy() if "is_active" in listings_df.columns else listings_df.copy()

    # Exclude cross-platform duplicates from stats
    if "duplicate_of" in active.columns:
        active = active[active["duplicate_of"].isna()].copy()

    active["generation"] = active.apply(
        lambda r: get_generation(r["brand"], r["model"], r.get("year")), axis=1
    )


    # --- Generation-level stats ---
    priced_gen = active[active["price_eur"].notna() & active["generation"].notna()]
    gen_stats = (
        priced_gen
        .groupby(["brand", "model", "generation"])
        .agg(gen_median=("price_eur", "median"), gen_count=("price_eur", "count"),
             gen_year_median=("year", "median"), gen_mileage_median=("mileage_km", "median"))
        .reset_index()
    )

    # --- Model-level fallback stats (includes cars without generation) ---
    priced_all = active[active["price_eur"].notna()]
    model_stats = (
        priced_all
        .groupby(["brand", "model"])
        .agg(model_median=("price_eur", "median"), model_count=("price_eur", "count"),
             model_year_median=("year", "median"), model_mileage_median=("mileage_km", "median"))
        .reset_index()
    )

    # --- Regression per generation: price ~ year + mileage + engine_cc ---
    gen_regressions = {}
    for (brand, model, gen), group in priced_gen.groupby(["brand", "model", "generation"]):
        subset = group[group["mileage_km"].notna() & group["year"].notna()]
        if len(subset) < 5:
            continue
        has_cc = "engine_cc" in subset.columns and subset["engine_cc"].notna().sum() >= len(subset) * 0.5
        if has_cc:
            cc_vals = subset["engine_cc"].fillna(subset["engine_cc"].median()).values.astype(float)
            X = np.column_stack([
                subset["year"].values.astype(float),
                subset["mileage_km"].values.astype(float),
                cc_vals,
                np.ones(len(subset)),
            ])
        else:
            X = np.column_stack([
                subset["year"].values.astype(float),
                subset["mileage_km"].values.astype(float),
                np.ones(len(subset)),
            ])
        y = subset["price_eur"].values.astype(float)
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            gen_regressions[(brand, model, gen)] = (coeffs, has_cc)
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

        price = listing["price_eur"]
        year = listing.get("year")
        mileage = listing.get("mileage_km")
        brand = listing["brand"]
        model = listing["model"]
        generation = listing.get("generation")

        # Resolve comparison group: generation-level if available, else model-level
        median = None
        sample = 0
        group_year_median = None
        group_mileage_median = None
        if generation:
            gs = gen_stats[
                (gen_stats["brand"] == brand)
                & (gen_stats["model"] == model)
                & (gen_stats["generation"] == generation)
            ]
            if not gs.empty:
                median = gs.iloc[0]["gen_median"]
                sample = int(gs.iloc[0]["gen_count"])
                group_year_median = gs.iloc[0]["gen_year_median"]
                group_mileage_median = gs.iloc[0]["gen_mileage_median"]

        if median is None:
            ms = model_stats[
                (model_stats["brand"] == brand) & (model_stats["model"] == model)
            ]
            if ms.empty:
                continue
            median = ms.iloc[0]["model_median"]
            sample = int(ms.iloc[0]["model_count"])
            group_year_median = ms.iloc[0]["model_year_median"]
            group_mileage_median = ms.iloc[0]["model_mileage_median"]

        if not median or price >= median * 0.85:
            continue

        # 1. Undervaluation (generation-level regression only — no fallbacks)
        predicted = None
        engine_cc = listing.get("engine_cc")
        fuel_type = listing.get("fuel_type")
        cc_float = float(engine_cc) if pd.notna(engine_cc) and engine_cc else None

        if generation and (brand, model, generation) in gen_regressions and pd.notna(year) and pd.notna(mileage):
            coeffs, has_cc = gen_regressions[(brand, model, generation)]
            if has_cc and cc_float:
                predicted = max(coeffs[0] * float(year) + coeffs[1] * float(mileage) + coeffs[2] * cc_float + coeffs[3], 0)
            elif not has_cc:
                predicted = max(coeffs[0] * float(year) + coeffs[1] * float(mileage) + coeffs[2], 0)

        if predicted and predicted > 0:
            undervaluation_pct = round((1 - price / predicted) * 100, 1)
        else:
            undervaluation_pct = 0.0
        discount_pct = round((1 - price / median) * 100, 1)

        # 2. Year multiplier — newer cars hold value better (15% per year above group median)
        if pd.notna(year) and pd.notna(group_year_median) and group_year_median > 0:
            year_diff = float(year) - float(group_year_median)
            year_mult = min(max(1 + year_diff * 0.15, 0.5), 2.0)
        else:
            year_mult = 1.0

        # 2b. Mileage multiplier — lower mileage = better flip (30% per median deviation)
        if pd.notna(mileage) and pd.notna(group_mileage_median) and group_mileage_median > 0:
            mileage_ratio = (float(group_mileage_median) - float(mileage)) / float(group_mileage_median)
            mileage_mult = min(max(1 + mileage_ratio * 0.3, 0.5), 2.0)
        else:
            mileage_mult = 1.0

        # 2c. Engine life multiplier — penalize cars near end-of-life for their engine size
        #     Small engines (< 1.2L) wear out faster, so 200k km on a 1.0L is worse than on a 2.0L
        if pd.notna(mileage) and mileage > 0:
            lifespan = _expected_lifespan_km(
                int(engine_cc) if pd.notna(engine_cc) and engine_cc else None,
                fuel_type if pd.notna(fuel_type) else None,
            )
            remaining_pct = max(0, 1 - float(mileage) / lifespan)
            # 0% remaining → 0.3x, 50% → 1.0x, 80%+ → 1.3x
            engine_life_mult = min(0.3 + remaining_pct * 1.4, 1.5)
        else:
            engine_life_mult = 1.0

        # 3. Liquidity multiplier (30 days = 1.0 baseline)
        days_to_sell = liquidity_map.get((brand, model))
        if days_to_sell and days_to_sell > 0:
            liquidity_mult = min(max(30 / days_to_sell, 0.5), 2.0)
        else:
            liquidity_mult = 1.0

        # 4. Trend multiplier (rising market = bonus)
        trend_pct = trend_map.get((brand, model), 0.0)
        trend_mult = min(max(1 + trend_pct / 100, 0.8), 1.2)

        # 5. Description-mention multiplier — extracted from listing text
        desc_mentions_accident = listing.get("desc_mentions_accident")
        desc_mentions_repair = listing.get("desc_mentions_repair")
        if pd.notna(desc_mentions_accident) and desc_mentions_accident:
            condition_mult = 0.3
        elif pd.notna(desc_mentions_repair) and desc_mentions_repair:
            condition_mult = 0.5
        else:
            condition_mult = 1.0

        # 6. Customs multiplier — based on description mention only
        desc_mentions_customs_cleared = listing.get("desc_mentions_customs_cleared")
        if pd.notna(desc_mentions_customs_cleared) and desc_mentions_customs_cleared is False:
            customs_mult = 0.7
        else:
            customs_mult = 1.0

        # 7. Motivated seller multiplier — long listing + price drops = negotiation room
        days_listed = listing.get("days_listed") if "days_listed" in listing.index else None
        price_change = listing.get("price_change_eur") if "price_change_eur" in listing.index else None
        if pd.notna(days_listed) and days_listed > 30 and pd.notna(price_change) and price_change < 0:
            # Seller already dropping price — more room to negotiate
            motivated_mult = min(1.0 + abs(float(price_change)) / float(price) * 0.5, 1.3)
        elif pd.notna(days_listed) and days_listed > 60:
            # Stale listing — some negotiation room even without explicit drops
            motivated_mult = 1.1
        else:
            motivated_mult = 1.0

        # 8. Right-hand drive penalty
        right_hand_drive = listing.get("right_hand_drive")
        if pd.notna(right_hand_drive) and right_hand_drive:
            rhd_mult = 0.6
        else:
            rhd_mult = 1.0

        # 9. Owner count mentioned in description
        desc_mentions_num_owners = listing.get("desc_mentions_num_owners")
        if pd.notna(desc_mentions_num_owners) and desc_mentions_num_owners and int(desc_mentions_num_owners) > 0:
            n = int(desc_mentions_num_owners)
            if n == 1:
                owners_mult = 1.15
            elif n == 2:
                owners_mult = 1.0
            elif n == 3:
                owners_mult = 0.9
            else:
                owners_mult = 0.75
        else:
            owners_mult = 1.0

        base_pct = undervaluation_pct if undervaluation_pct > 0 else discount_pct
        flip_score = round(
            base_pct * year_mult * mileage_mult * engine_life_mult
            * liquidity_mult * trend_mult * condition_mult * customs_mult
            * motivated_mult * owners_mult * rhd_mult, 1
        )

        sig = {
            "olx_id": listing.get("olx_id", ""),
            "url": listing.get("url", ""),
            "brand": brand,
            "model": model,
            "year": year,
            "generation": "" if pd.isna(generation) else (generation or ""),
            "price_eur": price,
            "predicted_price": round(predicted) if predicted and predicted > 0 else None,
            "median_price_eur": round(median),
            "discount_pct": discount_pct,
            "undervaluation_pct": undervaluation_pct,
            "engine_cc": int(engine_cc) if pd.notna(engine_cc) and engine_cc else None,
            "year_mult": round(year_mult, 2),
            "engine_life_mult": round(engine_life_mult, 2),
            "condition_mult": round(condition_mult, 2),
            "customs_mult": round(customs_mult, 2),
            "motivated_mult": round(motivated_mult, 2),
            "owners_mult": round(owners_mult, 2),
            "rhd_mult": round(rhd_mult, 2),
            "avg_days_to_sell": days_to_sell,
            "price_trend_pct": trend_pct,
            "flip_score": flip_score,
            "sample_size": sample,
            "city": listing.get("city", ""),
            "district": listing.get("district", ""),
            "mileage_km": mileage,
            "fuel_type": listing.get("fuel_type", ""),
            "desc_mentions_accident": bool(desc_mentions_accident) if pd.notna(desc_mentions_accident) else None,
            "desc_mentions_repair": bool(desc_mentions_repair) if pd.notna(desc_mentions_repair) else None,
            "desc_mentions_num_owners": (
                int(desc_mentions_num_owners)
                if pd.notna(desc_mentions_num_owners) and desc_mentions_num_owners
                else None
            ),
            "desc_mentions_customs_cleared": (
                bool(desc_mentions_customs_cleared)
                if pd.notna(desc_mentions_customs_cleared)
                else None
            ),
            "right_hand_drive": (
                bool(right_hand_drive)
                if pd.notna(right_hand_drive)
                else None
            ),
            "negotiable": bool(listing.get("negotiable")) if "negotiable" in listing.index and pd.notna(listing.get("negotiable")) else None,
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


def load_listing_feedback() -> pd.DataFrame:
    """Load manual shortlist feedback from database."""
    if not _ensure_db():
        return pd.DataFrame()
    try:
        from src.storage.database import init_db, get_session
        from src.storage.repository import get_listing_feedback_df

        init_db(str(DB_PATH))
        session = get_session()
        return get_listing_feedback_df(session)
    except Exception as e:
        print(f"Warning: could not load listing feedback: {e}")
        return pd.DataFrame()


def load_all():
    """Load listings, history, signals, brand map, turnover, and portfolio."""
    from src.analytics.computed_columns import enrich_listings
    from src.analytics.turnover import compute_turnover_stats

    db_data = load_from_db()

    if db_data is not None:
        listings, history = db_data
        listings = enrich_listings(listings)
        # Use LLM-corrected mileage everywhere (sellers game filters with fake low values)
        if "real_mileage_km" in listings.columns:
            listings["mileage_km"] = listings["real_mileage_km"].fillna(listings["mileage_km"])
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
