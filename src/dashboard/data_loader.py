"""Load data from SQLite database (local or downloaded from GitHub Releases)."""

import json
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DB_PATH = PROJECT_ROOT / "data" / "olx_cars.db"


_CHECK_INTERVAL_SECONDS = 2 * 3600  # check for new release every 2 hours


def _github_token() -> str | None:
    tok = os.environ.get("GITHUB_TOKEN")
    if tok:
        return tok
    try:
        import streamlit as st
        return st.secrets.get("GITHUB_TOKEN")
    except Exception:
        return None


def _list_release_assets(repo: str) -> dict[str, dict] | None:
    """Return {asset_name: asset_dict} for the latest-data release."""
    import httpx

    api_url = f"https://api.github.com/repos/{repo}/releases/tags/latest-data"
    headers = {"Accept": "application/vnd.github+json"}
    token = _github_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        resp = httpx.get(api_url, headers=headers, timeout=10)
        if resp.status_code != 200:
            return None
        return {a["name"]: a for a in resp.json().get("assets", [])}
    except Exception:
        return None


def _asset_url_if_newer(asset: dict, local_path: Path) -> str | None:
    """Return the asset API URL if remote is newer than local_path (or local is missing)."""
    from datetime import datetime, timezone
    remote_dt = datetime.fromisoformat(asset["updated_at"].replace("Z", "+00:00"))
    if local_path.exists():
        local_dt = datetime.fromtimestamp(local_path.stat().st_mtime, tz=timezone.utc)
        if remote_dt <= local_dt:
            return None
    # Use API URL (not browser_download_url) — required to auth private-repo assets.
    return asset["url"]


def _download_asset(url: str, dest: Path) -> bool:
    import httpx

    headers = {"Accept": "application/octet-stream"}
    token = _github_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        resp = httpx.get(url, headers=headers, follow_redirects=True, timeout=60)
        if resp.status_code == 200:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(resp.content)
            print(f"Downloaded {dest.name} from release ({len(resp.content)} bytes)")
            return True
    except Exception as e:
        print(f"Warning: could not download {dest.name} from release: {e}")
    return False


def _force_next_check():
    """Reset the check timer so the next _ensure_db() call hits GitHub API."""
    if DB_PATH.exists():
        os.utime(DB_PATH, (0, 0))


# Model + metrics live alongside the DB in the data release — shipped by CI
# (see .github/workflows/scrape.yml `train-model` step). The dashboard never
# trains locally; it just consumes what the pipeline produced.
_MODEL_PATH = PROJECT_ROOT / "data" / "price_model.joblib"
_METRICS_PATH = PROJECT_ROOT / "data" / "price_metrics.json"
_IMPORTANCE_PATH = PROJECT_ROOT / "data" / "price_importance.json"

_RELEASE_ASSETS: tuple[tuple[str, Path], ...] = (
    ("olx_cars.db", DB_PATH),
    ("price_model.joblib", _MODEL_PATH),
    ("price_metrics.json", _METRICS_PATH),
    ("price_importance.json", _IMPORTANCE_PATH),
)


def _ensure_release_assets() -> bool:
    """Sync DB + model + metrics from the latest-data release (once per TTL).

    Returns True if the local DB exists after the attempt (model/metrics are
    best-effort — dashboard falls back to "no predictions" if they're missing).
    """
    import time

    # TTL gate is keyed on the DB's mtime (same as before): if DB was
    # refreshed recently, skip the GitHub API call entirely.
    if DB_PATH.exists() and (time.time() - DB_PATH.stat().st_mtime) <= _CHECK_INTERVAL_SECONDS:
        return True

    repo = os.environ.get("GITHUB_REPOSITORY", "nikit34/olx-car-parser")
    if not repo:
        return DB_PATH.exists()

    assets = _list_release_assets(repo)
    if not assets:
        return DB_PATH.exists()

    for name, dest in _RELEASE_ASSETS:
        asset = assets.get(name)
        if not asset:
            continue
        url = _asset_url_if_newer(asset, dest)
        if url:
            _download_asset(url, dest)

    return DB_PATH.exists()


def _ensure_db() -> bool:
    """Back-compat alias — sync all release assets, report DB status."""
    return _ensure_release_assets()


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



def _sub_segment(fuel_type, engine_cc) -> str:
    """Market sub-segment from fuel type and engine displacement.

    E.g. "diesel_mid", "petrol_small", "electric".
    """
    import pandas as _pd
    fuel = "unk"
    if _pd.notna(fuel_type) and fuel_type:
        fl = str(fuel_type).lower()
        if "diesel" in fl:
            fuel = "diesel"
        elif "eléctrico" in fl or "electr" in fl:
            fuel = "electric"
        elif "híbrido" in fl or "hybrid" in fl:
            fuel = "hybrid"
        elif "gpl" in fl:
            fuel = "gpl"
        else:
            fuel = "petrol"
    if fuel == "electric":
        return "electric"
    if not _pd.notna(engine_cc) or not engine_cc or engine_cc <= 0:
        return fuel
    if engine_cc < 1400:
        return f"{fuel}_small"
    if engine_cc <= 2000:
        return f"{fuel}_mid"
    return f"{fuel}_large"


def _load_llm_extras(raw_extras) -> dict:
    """Best-effort parse of serialized LLM extras stored on a listing row."""
    if isinstance(raw_extras, dict):
        return raw_extras
    if not isinstance(raw_extras, str) or not raw_extras.strip():
        return {}
    try:
        parsed = json.loads(raw_extras)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _normalized_text_list(value) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip().lower() for item in value if item not in (None, "")]


def _blocking_deal_reason(listing: pd.Series) -> str | None:
    """Return a hard-stop reason for listings that should not be shown as deals.

    Relies on two signals: desc_mentions_accident (DB column) and
    mechanical_condition (from llm_extras JSON).  The LLM prompt sets
    mechanical_condition="poor" for parts cars / breakdowns, so the old
    free-text checks (suspicious_signs, reason_for_sale, issues) are
    no longer needed.
    """
    desc_mentions_accident = listing.get("desc_mentions_accident")
    if pd.notna(desc_mentions_accident) and bool(desc_mentions_accident):
        return "description mentions accident"

    extras = _load_llm_extras(listing.get("llm_extras"))
    if not extras:
        return None

    if str(extras.get("mechanical_condition") or "").strip().lower() == "poor":
        return "poor mechanical condition"

    return None


def prepare_active_for_model(
    listings_df: pd.DataFrame,
    turnover: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Active-listing subset enriched with generation, sub_segment, avg_days_to_sell.

    Shared by compute_signals (dashboard) and the `train-model` CLI so the
    model sees the exact same feature prep in both paths.
    """
    from src.models.generations import get_generation
    from src.analytics.turnover import compute_turnover_stats

    if listings_df.empty:
        return listings_df.copy()

    active = listings_df[listings_df["is_active"]].copy() if "is_active" in listings_df.columns else listings_df.copy()
    if "duplicate_of" in active.columns:
        active = active[active["duplicate_of"].isna()].copy()

    gen_keys = active[["brand", "model", "year"]].drop_duplicates()
    gen_map = {
        (b, m, y if pd.notna(y) else None): get_generation(b, m, y if pd.notna(y) else None)
        for b, m, y in gen_keys.itertuples(index=False, name=None)
    }
    active["generation"] = [
        gen_map.get((b, m, y if pd.notna(y) else None))
        for b, m, y in zip(active["brand"], active["model"], active["year"])
    ]

    def _sub_key(f, e):
        fk = f if (isinstance(f, str) and f) else None
        ek = float(e) if pd.notna(e) and e else None
        return (fk, ek)

    # Tolerate fixtures/tests that omit these optional columns — fall back
    # to None for the missing one so the original apply-based behavior holds.
    fuel_series = active["fuel_type"] if "fuel_type" in active.columns else pd.Series([None] * len(active), index=active.index)
    cc_series = active["engine_cc"] if "engine_cc" in active.columns else pd.Series([None] * len(active), index=active.index)

    sub_keys = pd.DataFrame({"fuel_type": fuel_series, "engine_cc": cc_series}).drop_duplicates()
    sub_map = {
        _sub_key(f, e): _sub_segment(f, e)
        for f, e in sub_keys.itertuples(index=False, name=None)
    }
    active["sub_segment"] = [
        sub_map.get(_sub_key(f, e))
        for f, e in zip(fuel_series, cc_series)
    ]

    if turnover is None:
        turnover = compute_turnover_stats(listings_df)
    liquidity_map: dict[tuple, float] = {}
    if not turnover.empty:
        for row in turnover.itertuples(index=False):
            days = getattr(row, "avg_days_to_sell", None)
            if pd.notna(days):
                gen = getattr(row, "generation", None)
                liquidity_map[(row.brand, row.model, gen if pd.notna(gen) else None)] = float(days)
    if liquidity_map:
        active["avg_days_to_sell"] = [
            liquidity_map.get((b, m, g if pd.notna(g) else None))
            for b, m, g in zip(active["brand"], active["model"], active["generation"])
        ]

    return active


def compute_signals(
    listings_df: pd.DataFrame,
    history_df: pd.DataFrame,
    turnover: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Find undervalued listings and rank by flip potential.

    Price model (gradient boosting) uses 20 features including LLM-extracted
    fields (accident, RHD, condition, etc.) to predict fair market value.

    Flip score = undervaluation % × opportunity multipliers:
    - Liquidity: how fast this model sells
    - Trend: market price direction
    - Motivated seller: long listing + price drops
    - Urgency: seller desperation signals
    - Warranty: easier resale with warranty
    - Velocity: fast-selling segments
    - Confidence: comparable listing count
    """
    if listings_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    import numpy as np

    signals = []
    active = prepare_active_for_model(listings_df, turnover=turnover)

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

    # --- Sub-segment stats (fuel + engine size within generation) ---
    sub_stats = (
        priced_gen
        .groupby(["brand", "model", "generation", "sub_segment"])
        .agg(sub_median=("price_eur", "median"), sub_count=("price_eur", "count"),
             sub_year_median=("year", "median"), sub_mileage_median=("mileage_km", "median"))
        .reset_index()
    )

    # --- Liquidity map (keyed by (brand, model, generation|None)) ---
    from src.analytics.turnover import compute_turnover_stats
    if turnover is None:
        turnover = compute_turnover_stats(listings_df)
    liquidity_map: dict[tuple, float] = {}
    if not turnover.empty:
        for row in turnover.itertuples(index=False):
            days = getattr(row, "avg_days_to_sell", None)
            if pd.notna(days):
                gen = getattr(row, "generation", None)
                liquidity_map[(row.brand, row.model, gen if pd.notna(gen) else None)] = float(days)

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

    # --- Sale velocity: fraction of recently deactivated listings sold within 21 days ---
    velocity_map: dict[tuple, float] = {}
    inactive = listings_df[~listings_df["is_active"]].copy() if "is_active" in listings_df.columns else pd.DataFrame()
    if not inactive.empty and {"deactivated_at", "first_seen_at", "brand", "model"}.issubset(inactive.columns):
        sold = inactive[inactive["deactivated_at"].notna() & inactive["first_seen_at"].notna()].copy()
        if not sold.empty:
            sold["_lifespan"] = (
                pd.to_datetime(sold["deactivated_at"]) - pd.to_datetime(sold["first_seen_at"])
            ).dt.days
            sold = sold[sold["_lifespan"] > 0]
            group_keys = ["brand", "model", "generation"] if "generation" in sold.columns else ["brand", "model"]
            for keys, group in sold.groupby(group_keys, dropna=False):
                if len(group) < 3:
                    continue
                velocity_map[keys] = float((group["_lifespan"] <= 21).mean())

    # avg_days_to_sell already populated by prepare_active_for_model.

    # --- Gradient boosting price model (uses LLM fields + market data) ---
    from src.analytics.price_model import (
        predict_prices,
        compute_feature_completeness,
        load_importance,
        load_model,
    )

    feature_fill = compute_feature_completeness(active)

    # Model is trained in CI (see `train-model` CLI) and shipped in the data
    # release. Dashboard never trains inline — if the model is missing or
    # too old, we skip predictions instead of blocking the user for minutes.
    saved = load_model(max_age_hours=14 * 24)  # tolerate up to 2 weeks
    gb_models = None
    gb_cat_maps: dict = {}
    _gb_metrics: dict | None = None
    _gb_oof_preds: dict[str, tuple[float, float, float]] = {}
    if saved is not None:
        gb_models, gb_cat_maps, _gb_metrics, _gb_oof_preds = saved
    else:
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "No fresh price model available — skipping predictions. "
            "Run `python -m src.cli train-model` or wait for the next CI run."
        )

    gb_predictions: dict[str, float] = {}
    gb_fair_low: dict[str, float] = {}
    gb_fair_high: dict[str, float] = {}
    # Importance is computed once during training (CI) and shipped in the
    # data release — loading it here costs one JSON read instead of a
    # 690-predict permutation loop per signal recompute.
    importance_df = load_importance()
    if gb_models is not None:
        _conformal_q = _gb_metrics.get("conformal_q", 0.0) if _gb_metrics else 0.0
        # OOF preds (built during CV training) override model.predict for any
        # listing the model was trained on, so the deal-scoring loop compares
        # asking price against an out-of-fold "fair price" rather than an
        # in-sample one.
        price_df = predict_prices(
            gb_models, gb_cat_maps, active,
            conformal_q=_conformal_q, oof_preds=_gb_oof_preds,
        )
        if "olx_id" in active.columns:
            olx_ids = active["olx_id"].reindex(price_df.index).values
            preds = price_df["predicted_price"].values
            lows = price_df["fair_price_low"].values
            highs = price_df["fair_price_high"].values
            for oid, pred, lo, hi in zip(olx_ids, preds, lows, highs):
                if oid and pred > 0:
                    gb_predictions[oid] = float(pred)
                    gb_fair_low[oid] = float(lo)
                    gb_fair_high[oid] = float(hi)

    # --- Precompute stat lookups (dict by tuple) to avoid per-row boolean
    # indexing over sub_stats/gen_stats/model_stats in the scoring loop.
    sub_lookup: dict[tuple, tuple] = {
        (r.brand, r.model, r.generation, r.sub_segment):
            (r.sub_median, int(r.sub_count), r.sub_year_median, r.sub_mileage_median)
        for r in sub_stats.itertuples(index=False)
    }
    gen_lookup: dict[tuple, tuple] = {
        (r.brand, r.model, r.generation):
            (r.gen_median, int(r.gen_count), r.gen_year_median, r.gen_mileage_median)
        for r in gen_stats.itertuples(index=False)
    }
    model_lookup: dict[tuple, tuple] = {
        (r.brand, r.model):
            (r.model_median, int(r.model_count), r.model_year_median, r.model_mileage_median)
        for r in model_stats.itertuples(index=False)
    }

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

        # Resolve comparison group: sub-segment → generation → model
        median = None
        sample = 0
        group_year_median = None
        group_mileage_median = None
        sub_seg = listing.get("sub_segment")

        if generation and sub_seg:
            hit = sub_lookup.get((brand, model, generation, sub_seg))
            if hit is not None and hit[1] >= 5:
                median, sample, group_year_median, group_mileage_median = hit

        if median is None and generation:
            hit = gen_lookup.get((brand, model, generation))
            if hit is not None:
                median, sample, group_year_median, group_mileage_median = hit

        if median is None:
            hit = model_lookup.get((brand, model))
            if hit is None:
                continue
            median, sample, group_year_median, group_mileage_median = hit

        if not median or price >= median * 0.85:
            continue

        if _blocking_deal_reason(listing):
            continue

        # 1. Undervaluation: gradient boosting predicted price (now includes LLM features)
        olx_id = listing.get("olx_id", "")
        predicted = gb_predictions.get(olx_id)

        if predicted and predicted > 0:
            undervaluation_pct = round((1 - price / predicted) * 100, 1)
        else:
            undervaluation_pct = 0.0
        discount_pct = round((1 - price / median) * 100, 1)

        # Price range from quantile regression
        fair_low = gb_fair_low.get(olx_id)
        fair_high = gb_fair_high.get(olx_id)
        fill_rate = float(feature_fill.loc[listing.name]) if listing.name in feature_fill.index else 0.0
        sample_conf = min(sample / 20, 1.0)
        completeness = round(0.6 * fill_rate + 0.4 * sample_conf, 3)

        # --- Opportunity multipliers (deal quality, not market value) ---

        # 2. Liquidity multiplier (30 days = 1.0 baseline).
        # Key matches how the map is built: (brand, model, generation|None),
        # with a 2-level fallback if the generation-specific stat is missing.
        gen_key = generation if pd.notna(generation) and generation else None
        days_to_sell = liquidity_map.get((brand, model, gen_key))
        if days_to_sell is None and gen_key is not None:
            days_to_sell = liquidity_map.get((brand, model, None))
        if days_to_sell and days_to_sell > 0:
            liquidity_mult = min(max(30 / days_to_sell, 0.5), 2.0)
        else:
            liquidity_mult = 1.0

        # 3. Trend multiplier (rising market = bonus)
        trend_pct = trend_map.get((brand, model), 0.0)
        trend_mult = min(max(1 + trend_pct / 100, 0.8), 1.2)

        # 4. Motivated seller — long listing + price drops = negotiation room
        days_listed = listing.get("days_listed") if "days_listed" in listing.index else None
        price_change = listing.get("price_change_eur") if "price_change_eur" in listing.index else None
        if pd.notna(days_listed) and days_listed > 30 and pd.notna(price_change) and price_change < 0:
            motivated_mult = min(1.0 + abs(float(price_change)) / float(price) * 0.5, 1.3)
        elif pd.notna(days_listed) and days_listed > 60:
            motivated_mult = 1.1
        else:
            motivated_mult = 1.0

        # 5. Urgency — desperate seller = negotiation opportunity
        urgency = listing.get("urgency")
        if pd.notna(urgency) and urgency == "high":
            urgency_mult = 1.3
        elif pd.notna(urgency) and urgency == "medium":
            urgency_mult = 1.1
        else:
            urgency_mult = 1.0

        # 6. Warranty — easier to resell with warranty
        has_warranty = listing.get("warranty")
        if pd.notna(has_warranty) and has_warranty:
            warranty_mult = 1.15
        else:
            warranty_mult = 1.0

        # 7. Sale velocity — fast-selling segments = better for flipping
        velocity = velocity_map.get((brand, model, listing.get("generation")))
        if velocity is not None:
            velocity_mult = min(max(0.7 + velocity * 0.8, 0.7), 1.5)
        else:
            velocity_mult = 1.0

        # 8. Confidence — more comparable listings = more reliable estimate
        if sample >= 10:
            confidence_mult = 1.2
        elif sample >= 7:
            confidence_mult = 1.1
        elif sample >= 5:
            confidence_mult = 1.0
        else:
            confidence_mult = 0.7

        base_pct = undervaluation_pct if undervaluation_pct > 0 else discount_pct
        flip_score = round(
            base_pct * liquidity_mult * trend_mult * motivated_mult
            * urgency_mult * warranty_mult * velocity_mult * confidence_mult, 1
        )

        # --- Build signal dict ---
        desc_mentions_accident = listing.get("desc_mentions_accident")
        desc_mentions_repair = listing.get("desc_mentions_repair")
        desc_mentions_num_owners = listing.get("desc_mentions_num_owners")
        desc_mentions_customs_cleared = listing.get("desc_mentions_customs_cleared")
        right_hand_drive = listing.get("right_hand_drive")
        engine_cc = listing.get("engine_cc")

        sig = {
            "olx_id": olx_id,
            "url": listing.get("url", ""),
            "brand": brand,
            "model": model,
            "year": year,
            "generation": "" if pd.isna(generation) else (generation or ""),
            "sub_segment": sub_seg or "",
            "price_eur": price,
            "predicted_price": round(predicted) if predicted and predicted > 0 else None,
            "fair_price_low": fair_low,
            "fair_price_high": fair_high,
            "data_completeness": completeness,
            "median_price_eur": round(median),
            "discount_pct": discount_pct,
            "undervaluation_pct": undervaluation_pct,
            "engine_cc": int(engine_cc) if pd.notna(engine_cc) and engine_cc else None,
            "liquidity_mult": round(liquidity_mult, 2),
            "trend_mult": round(trend_mult, 2),
            "motivated_mult": round(motivated_mult, 2),
            "urgency_mult": round(urgency_mult, 2),
            "warranty_mult": round(warranty_mult, 2),
            "velocity_mult": round(velocity_mult, 2),
            "confidence_mult": round(confidence_mult, 2),
            "avg_days_to_sell": days_to_sell,
            "price_trend_pct": trend_pct,
            "flip_score": flip_score,
            "sample_size": sample,
            "city": listing.get("city", ""),
            "district": listing.get("district", ""),
            "mileage_km": mileage,
            "fuel_type": listing.get("fuel_type", ""),
            # LLM fields for display/warnings
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
            "right_hand_drive": bool(right_hand_drive) if pd.notna(right_hand_drive) else None,
            "urgency": urgency if pd.notna(urgency) else None,
            "warranty": bool(has_warranty) if pd.notna(has_warranty) else None,
            "taxi_fleet_rental": bool(listing.get("taxi_fleet_rental")) if pd.notna(listing.get("taxi_fleet_rental")) else None,
            "first_owner_selling": bool(listing.get("first_owner_selling")) if pd.notna(listing.get("first_owner_selling")) else None,
        }
        for col in ("days_listed", "price_change_eur", "price_change_pct", "eur_per_km"):
            if col in listing.index:
                sig[col] = listing[col]
        signals.append(sig)

    df = pd.DataFrame(signals)
    if not df.empty:
        df = df.sort_values("flip_score", ascending=False)
    return df, importance_df



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
        # Use LLM-corrected mileage everywhere (sellers game filters with fake low values)
        if "real_mileage_km" in listings.columns:
            listings["mileage_km"] = listings["real_mileage_km"].fillna(listings["mileage_km"])
        turnover = compute_turnover_stats(listings)
        signals, importance = compute_signals(listings, history, turnover=turnover)
    else:
        listings = pd.DataFrame()
        history = pd.DataFrame()
        signals = pd.DataFrame()
        importance = pd.DataFrame()
        turnover = pd.DataFrame()

    portfolio = load_portfolio()

    brands_models: dict[str, list[str]] = {}
    if not listings.empty:
        pairs = listings[["brand", "model"]].drop_duplicates()
        for brand, grp in pairs.groupby("brand", sort=False):
            brands_models[brand] = grp["model"].tolist()

    unmatched = load_unmatched()

    return listings, history, signals, brands_models, turnover, portfolio, unmatched, importance
