"""LightGBM price model with quantile regression, cross-validation, and persistence.

Uses all available features including LLM-extracted fields (accident,
condition, RHD, etc.) and market data (avg_days_to_sell).

Three models: median (predicted price), 10th percentile (low), 90th percentile (high).
LightGBM handles NaN natively and supports native categorical features.

Cross-validation provides MAE/MAPE/R² quality metrics.  Model can be
saved/loaded to avoid retraining on every dashboard visit.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import make_scorer, mean_pinball_loss
from sklearn.model_selection import KFold


_MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_MODEL_PATH = _MODEL_DIR / "price_model.joblib"
_METRICS_PATH = _MODEL_DIR / "price_metrics.json"
_IMPORTANCE_PATH = _MODEL_DIR / "price_importance.json"
_MODEL_MAX_AGE_HOURS = 24
_MIN_CATEGORY_COUNT = 3
_OTHER_CATEGORY = "__other__"
# Bumped whenever the bundle on-disk shape changes (feature list, log target,
# CQR semantics, OOF preds, etc.). load_model rejects mismatches so the
# dashboard can't accidentally consume an artifact trained against a different
# feature set or target transform.
# v3: time-aware CQR + isotonic median calibrator
_SCHEMA_VERSION = 3
# Fraction of the dataset (newest rows by first_seen_at) used as the
# time-honest conformal calibration window. Random-KFold CQR mixes
# time-adjacent rows across folds and over-estimates coverage on real future
# listings — backtest showed 80% nominal coverage delivering 61–69% on
# tomorrow's data. Calibrating on a held-out time-tail closes that gap.
_TIME_CALIBRATION_FRAC = 0.2

NUMERIC_FEATURES = [
    "year", "mileage_km", "engine_cc", "horsepower",
    "desc_mentions_num_owners", "avg_days_to_sell",
    "photo_count", "description_length", "seats",
    "tuning_or_mods_count",
    # NOTE: price-history features (num_price_drops, max_drop_pct,
    # price_drop_velocity, days_since_last_drop) are intentionally excluded.
    # They are post-hoc — known only after observing the listing for days —
    # and create circular logic: model learns "listings that dropped are
    # cheaper" instead of explaining price from the car's own attributes.
    # That nukes the flip-score signal for exactly the deals we care about
    # (motivated sellers who slashed prices). Dashboard still displays them
    # as separate indicators; they just don't feed the model.
]

BOOL_FEATURES = [
    "desc_mentions_accident", "desc_mentions_repair",
    "desc_mentions_customs_cleared", "right_hand_drive",
    "taxi_fleet_rental", "warranty", "first_owner_selling",
]

CATEGORICAL_FEATURES = [
    "brand", "model", "fuel_type", "transmission", "segment",
    "urgency", "generation", "mechanical_condition",
    "color", "district", "drive_type", "sub_model", "trim_level",
    "doors",
]

_ALL_FEATURES = NUMERIC_FEATURES + BOOL_FEATURES + CATEGORICAL_FEATURES


def _build_categorical_mapping(vals: pd.Series) -> dict[str, int]:
    """Build a stable mapping, grouping rare categories into __other__."""
    non_missing = vals.dropna().astype(str)
    if non_missing.empty:
        return {}

    counts = non_missing.value_counts()
    frequent = sorted(counts[counts >= _MIN_CATEGORY_COUNT].index.tolist())

    if len(frequent) < len(counts):
        values = [*frequent, _OTHER_CATEGORY]
    else:
        values = frequent

    return {value: i for i, value in enumerate(values)}


def _encode_categoricals(
    df: pd.DataFrame,
    cat_maps: dict[str, dict[str, int]] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    """Ordinal-encode categoricals for LightGBM."""
    out = df.copy()
    maps: dict[str, dict[str, int]] = cat_maps or {}

    for col in CATEGORICAL_FEATURES:
        if col not in out.columns:
            out[col] = np.nan
            continue

        if col not in maps:
            maps[col] = _build_categorical_mapping(out[col])

        mapping = maps[col]
        encoded = pd.Series(np.nan, index=out.index, dtype=float)
        non_missing = out[col].notna()

        if non_missing.any():
            vals = out.loc[non_missing, col].astype(str)
            if _OTHER_CATEGORY in mapping:
                known = set(mapping)
                known.discard(_OTHER_CATEGORY)
                vals = vals.where(vals.isin(known), _OTHER_CATEGORY)
            encoded.loc[non_missing] = vals.map(mapping).astype(float)

        out[col] = encoded

    return out, maps


def _prepare_X(
    df: pd.DataFrame,
    cat_maps: dict[str, dict[str, int]] | None = None,
) -> tuple[np.ndarray, dict[str, dict[str, int]]]:
    """Prepare feature matrix from listings DataFrame."""
    X = df.reindex(columns=_ALL_FEATURES).copy()

    for col in BOOL_FEATURES:
        if col in X.columns:
            X[col] = X[col].astype(float)

    for col in NUMERIC_FEATURES:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    X, maps = _encode_categoricals(X, cat_maps)
    X_arr = X[_ALL_FEATURES].values.astype(float)
    return X_arr, maps


_LGB_PARAMS = dict(
    # Ceiling only — the actual tree count comes from early stopping in CV
    # (`best_n_estimators`), so this just has to be high enough not to cap
    # the search.  On ~8k rows early stopping was pegged at 699/700 before
    # this bump, meaning the old cap was the bottleneck, not overfitting.
    n_estimators=2000,
    # max_depth and num_leaves must be consistent: a tree of depth d can have
    # at most 2^d leaves. With max_depth=4, num_leaves was silently capped at
    # 16 — depth=6 lets num_leaves=31 actually be reached.
    max_depth=6,
    learning_rate=0.05,
    min_child_samples=10,
    reg_lambda=1.5,
    num_leaves=31,
    random_state=42,
    verbose=-1,
    n_jobs=-1,
)

_QUANTILES = {"low": 0.1, "median": 0.5, "high": 0.9}
_EARLY_STOPPING_ROUNDS = 40
_MIN_N_ESTIMATORS = 50


def _pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    """Quantile (pinball) loss — the native metric for quantile regression."""
    diff = y_true - y_pred
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1) * diff)))


def _filter_training_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop outliers that would poison the quantile targets.

    Returns (filtered_df, drop_stats) for logging.  Filters:
      1. Price that moved 10× in either direction (data-entry noise, wrong currency)
      2. Prices outside [1st, 99th] percentile (heavy-tail clip)
      3. km / max(age, 1) outside [500, 100 000] (implausible mileage)
    """
    start = len(df)
    out = df.copy()
    stats = {"start": start}

    # 1. 10× price swings from first known price
    if "first_price_eur" in out.columns:
        first = pd.to_numeric(out["first_price_eur"], errors="coerce")
        last = pd.to_numeric(out["price_eur"], errors="coerce")
        ratio = last / first.where(first > 0)
        extreme = (ratio >= 10) | (ratio <= 0.1)
        dropped = int(extreme.fillna(False).sum())
        if dropped:
            out = out[~extreme.fillna(False)]
        stats["dropped_10x_swing"] = dropped

    # 2. Price percentile clip
    prices = pd.to_numeric(out["price_eur"], errors="coerce")
    if len(prices) > 0:
        low_p, high_p = prices.quantile([0.01, 0.99])
        price_mask = (prices >= low_p) & (prices <= high_p)
        stats["dropped_price_percentile"] = int((~price_mask.fillna(False)).sum())
        out = out[price_mask.fillna(False)]

    # 3. Mileage sanity (km per year of age)
    current_year = datetime.now(timezone.utc).year
    years = pd.to_numeric(out["year"], errors="coerce")
    km = pd.to_numeric(out["mileage_km"], errors="coerce")
    age = (current_year - years).clip(lower=1)
    km_per_year = km / age
    implausible = (km_per_year < 500) | (km_per_year > 100_000)
    stats["dropped_mileage_sanity"] = int(implausible.fillna(False).sum())
    out = out[~implausible.fillna(False)]

    stats["kept"] = len(out)
    return out, stats


def _conformal_q_from_scores(
    scores: np.ndarray,
    target_coverage: float = 0.80,
) -> float:
    """The standard CQR quantile-of-scores formula, separated so both the
    random-KFold and time-aware paths use exactly the same statistic."""
    n = len(scores)
    if n == 0:
        return 0.0
    q_level = min(np.ceil(target_coverage * (n + 1)) / n, 1.0)
    return float(np.quantile(scores, q_level))


def _time_aware_conformal_q(
    df: pd.DataFrame,
    best_iters_per_q: dict[str, int],
    calibration_frac: float = _TIME_CALIBRATION_FRAC,
) -> float | None:
    """Compute conformal_q from a time-honest holdout.

    Trains the three quantile models on the oldest ``(1 - calibration_frac)``
    of the data (sorted by ``first_seen_at``), predicts the newest fraction,
    and computes ``q`` from the band-miss scores on that calibration slice.
    Honest answer to "how wide must the band be to actually cover 80% of
    *tomorrow's* listings", versus random-KFold CQR which mixes time-adjacent
    rows across folds and reports too small a ``q``.

    Returns None if the data lacks ``first_seen_at`` or is too small to split,
    in which case the caller should fall back to the random-KFold value.
    """
    if "first_seen_at" not in df.columns:
        return None
    sorted_df = df.copy()
    sorted_df["_fsa"] = pd.to_datetime(sorted_df["first_seen_at"], errors="coerce")
    sorted_df = sorted_df.dropna(subset=["_fsa"]).sort_values("_fsa").reset_index(drop=True)
    if len(sorted_df) < 200:
        return None

    cutoff = int(len(sorted_df) * (1 - calibration_frac))
    train = sorted_df.iloc[:cutoff]
    cal = sorted_df.iloc[cutoff:]
    if len(train) < 100 or len(cal) < 50:
        return None

    X_train, cat_maps = _prepare_X(train)
    X_cal, _ = _prepare_X(cal, cat_maps)
    y_train_log = np.log1p(np.maximum(train["price_eur"].astype(float).values, 0))
    y_cal_log = np.log1p(np.maximum(cal["price_eur"].astype(float).values, 0))
    cat_indices = [_ALL_FEATURES.index(c) for c in CATEGORICAL_FEATURES]

    log_low_pred: np.ndarray | None = None
    log_high_pred: np.ndarray | None = None
    for name, alpha in _QUANTILES.items():
        if name == "median":
            continue  # CQR scores only use the band edges
        params = {**_LGB_PARAMS, "n_estimators": best_iters_per_q[name]}
        model = lgb.LGBMRegressor(objective="quantile", alpha=alpha, **params)
        model.fit(X_train, y_train_log, categorical_feature=cat_indices)
        preds = model.predict(X_cal)
        if name == "low":
            log_low_pred = preds
        else:
            log_high_pred = preds

    assert log_low_pred is not None and log_high_pred is not None
    scores = np.maximum(log_low_pred - y_cal_log, y_cal_log - log_high_pred)
    return _conformal_q_from_scores(scores)


def _cv_metrics(
    df: pd.DataFrame,
    n_splits: int = 5,
) -> tuple[dict, dict[str, int], np.ndarray, IsotonicRegression | None]:
    """K-fold CV for all three quantile models with per-fold categorical
    encoding (no leakage) and early stopping to tune n_estimators.

    Models are trained in log1p(price) space (heavy-tailed targets bias raw-EUR
    pinball toward expensive listings; log-target evens the influence and
    typically improves MAPE on the cheap end). User-facing metrics (MAE, MAPE,
    R², pinball) are reported on the price scale by back-transforming OOF
    predictions with expm1.

    Returns:
      metrics: dict of CV metrics for the dashboard
      suggested: per-quantile early-stop iteration counts (log-pinball converges
                 at different rates for α=0.1/0.5/0.9, so each tail tunes its own)
      oof_band: (3, n) array of [low, median, high] OOF predictions in price
                space — already calibrated with conformal_q, sorted to repair
                crossings, and clamped at 0. Indexed by df row position.
    """
    y_all = df["price_eur"].values.astype(float)
    # Train on log1p — clamp at 0 first so log1p never sees a negative input
    # (filter step normally already enforces this, but be defensive)
    y_log = np.log1p(np.maximum(y_all, 0))
    n_splits = min(n_splits, max(2, len(df) // 20))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cat_indices = [_ALL_FEATURES.index(c) for c in CATEGORICAL_FEATURES]

    # OOF in log space (raw model output, before back-transform)
    oof_log = {name: np.full(len(y_all), np.nan) for name in _QUANTILES}
    best_iters: dict[str, list[int]] = {name: [] for name in _QUANTILES}

    for train_idx, val_idx in kf.split(df):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        # Encode categoricals on train only, apply mapping to val — no leakage
        X_train, cat_maps_fold = _prepare_X(train_df)
        X_val, _ = _prepare_X(val_df, cat_maps_fold)
        y_train_log = y_log[train_idx]
        y_val_log = y_log[val_idx]

        for name, alpha in _QUANTILES.items():
            model = lgb.LGBMRegressor(
                objective="quantile", alpha=alpha, **_LGB_PARAMS,
            )
            model.fit(
                X_train, y_train_log,
                eval_set=[(X_val, y_val_log)],
                categorical_feature=cat_indices,
                callbacks=[lgb.early_stopping(_EARLY_STOPPING_ROUNDS, verbose=False)],
            )
            oof_log[name][val_idx] = model.predict(X_val)
            best_iters[name].append(
                int(model.best_iteration_ or _LGB_PARAMS["n_estimators"])
            )

    # Per-quantile early-stop suggestion (median across folds). Tails converge
    # at different rates than median, so each gets its own count. Compute now
    # because the time-aware CQR helper needs them.
    suggested: dict[str, int] = {}
    for name in _QUANTILES:
        iters = best_iters[name]
        median_iters = int(np.median(iters)) if iters else _LGB_PARAMS["n_estimators"]
        suggested[name] = max(median_iters, _MIN_N_ESTIMATORS)

    # Random-KFold CQR — score = max(log_low − log_y, log_y − log_high). The
    # widening factor stays in log space and is applied to log predictions at
    # predict time, then expm1 back. This is asymmetric in price space (which
    # is what we want for heavy-tailed targets: a € band that grows with
    # price). Kept for diagnostic comparison vs the time-aware value below.
    scores = np.maximum(oof_log["low"] - y_log, y_log - oof_log["high"])
    conformal_q_log_random = _conformal_q_from_scores(scores)

    # Time-aware CQR — re-train on oldest 80% by first_seen_at, score the
    # newest 20%. Honest answer to "how wide must the band be to cover 80% of
    # *tomorrow's* listings". Falls back to random when the dataset has no
    # first_seen_at column or is too small.
    conformal_q_log_time = _time_aware_conformal_q(df, suggested)
    if conformal_q_log_time is not None:
        conformal_q_log = conformal_q_log_time
    else:
        conformal_q_log = conformal_q_log_random

    # Isotonic post-calibration of the median quantile. Fit
    # ``predicted → actual`` on out-of-fold pairs to remove systematic shifts
    # in the global pred-vs-actual relationship (locally we observed +€1.3k
    # under-prediction in the €30k+ tier). Only the median is calibrated this
    # way — for low/high the isotonic target (=actual) is the wrong reference
    # (those predict the 10th and 90th percentile, not the actual price).
    # Caveat: 1D isotonic cannot fix subgroup mispricing where the model
    # mis-reads features (e.g. the −27% bias on ``<€3k`` cars whose actual
    # price is depressed by undetected damage/salvage flags). That needs
    # feature-level work, not post-hoc calibration.
    median_oof_price_raw = np.maximum(np.expm1(oof_log["median"]), 0)
    median_calibrator = IsotonicRegression(
        out_of_bounds="clip", increasing=True,
    )
    median_calibrator.fit(median_oof_price_raw, y_all)
    calibrated_median_price = median_calibrator.predict(median_oof_price_raw)

    # Back-transform OOF band edges to price scale and assemble the band
    # around the isotonic-calibrated median. We can't naive-sort all three
    # because isotonic can move the median *outside* [raw_low, raw_high]
    # (e.g. for the cheap-segment over-prediction case where the model says
    # €4500 and isotonic compresses it to €1500). A blind sort would then put
    # raw_low into the median slot and the calibrated value into low —
    # silently dropping the calibration. Instead:
    #   1. Repair true low/high crossing (rare independent-quantile artifact)
    #   2. Bracket the band around the calibrated median by min/max
    raw_low_price = np.expm1(oof_log["low"] - conformal_q_log)
    raw_high_price = np.expm1(oof_log["high"] + conformal_q_log)
    fixed_low = np.minimum(raw_low_price, raw_high_price)
    fixed_high = np.maximum(raw_low_price, raw_high_price)
    band_low = np.minimum(fixed_low, calibrated_median_price)
    band_high = np.maximum(fixed_high, calibrated_median_price)
    oof_band = np.maximum(
        np.stack([band_low, calibrated_median_price, band_high]),
        0,
    )

    # User-facing metrics on price scale (back-transformed + calibrated)
    median_oof = oof_band[1]
    low_oof_calibrated = oof_band[0]
    high_oof_calibrated = oof_band[2]

    mae = float(np.mean(np.abs(y_all - median_oof)))
    mape = float(np.mean(np.abs((y_all - median_oof) / np.maximum(y_all, 1))) * 100)
    ss_res = float(np.sum((y_all - median_oof) ** 2))
    ss_tot = float(np.sum((y_all - np.mean(y_all)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Pinball on price scale uses the raw (uncalibrated) OOF quantile
    # predictions — that's the thing the model directly optimizes (after
    # expm1 the relative ordering is preserved for monotonic transforms).
    raw_low = np.maximum(np.expm1(oof_log["low"]), 0)
    raw_high = np.maximum(np.expm1(oof_log["high"]), 0)
    pinball = {
        "low": _pinball_loss(y_all, raw_low, _QUANTILES["low"]),
        "median": _pinball_loss(y_all, median_oof, _QUANTILES["median"]),
        "high": _pinball_loss(y_all, raw_high, _QUANTILES["high"]),
    }

    # Coverage: invariant under monotonic transforms, so log/price agree.
    coverage_raw = float(np.mean((y_all >= raw_low) & (y_all <= raw_high)))
    coverage_calibrated = float(np.mean(
        (y_all >= low_oof_calibrated) & (y_all <= high_oof_calibrated)
    ))

    metrics = {
        "mae": round(mae, 0),
        "mape": round(mape, 1),
        "r2": round(r2, 3),
        "pinball_low": round(pinball["low"], 1),
        "pinball_median": round(pinball["median"], 1),
        "pinball_high": round(pinball["high"], 1),
        "coverage_80": round(coverage_raw, 3),
        "coverage_80_calibrated": round(coverage_calibrated, 3),
        # conformal_q is in LOG space (the band-widening exponent). Apply it
        # to log predictions in predict_prices, then expm1 to price scale.
        # The active value is time-aware when first_seen_at is available,
        # otherwise random-KFold (older fallback path).
        "conformal_q": round(conformal_q_log, 4),
        "conformal_q_random": round(conformal_q_log_random, 4),
        "conformal_q_time": (
            round(conformal_q_log_time, 4)
            if conformal_q_log_time is not None else None
        ),
        "conformal_q_source": "time" if conformal_q_log_time is not None else "random",
        # Approximate ± multiplicative band widening for human display:
        # a row's price band stretches by roughly (exp(q) − 1) × 100 % on
        # each side relative to the raw model band.
        "conformal_q_pct": round(float(np.expm1(conformal_q_log) * 100), 1),
        # Back-compat single number for the dashboard's "Trees (early-stop)"
        # KPI; the actual fit uses the per-quantile dict below.
        "best_n_estimators": suggested["median"],
        "best_n_estimators_per_q": suggested,
        "n_samples": int(len(y_all)),
        "cv_folds": n_splits,
        "log_target": True,
        "median_calibrated": True,
    }
    return metrics, suggested, oof_band, median_calibrator


def train_price_model(
    listings_df: pd.DataFrame,
    min_samples: int = 50,
) -> tuple[
    dict[str, lgb.LGBMRegressor],
    dict[str, dict[str, int]],
    dict,
    dict[str, tuple[float, float, float]],
    IsotonicRegression | None,
] | None:
    """Train quantile regression models: median, low (10th), high (90th).

    Returns (models, category_maps, metrics, oof_preds, median_calibrator) or
    None if insufficient data.

    - ``oof_preds`` is a dict olx_id → (low, median, high) of cross-validated,
      calibrated, crossing-repaired predictions in price space.
    - ``median_calibrator`` is an IsotonicRegression mapping raw expm1(median)
      → calibrated price; applied by ``predict_prices`` to new rows. OOF preds
      already have it baked in.
    """
    needed = {"price_eur", "year", "mileage_km"}
    if not needed.issubset(listings_df.columns):
        return None

    df = listings_df[
        listings_df["price_eur"].notna()
        & listings_df["year"].notna()
        & listings_df["mileage_km"].notna()
    ].copy()

    df, filter_stats = _filter_training_data(df)

    if len(df) < min_samples:
        return None

    y_log = np.log1p(np.maximum(df["price_eur"].values.astype(float), 0))

    # CV with per-fold cat encoding + per-quantile early stopping → tuned
    # n_estimators dict + OOF predictions in price space + isotonic calibrator
    # for the median quantile.
    metrics, best_iters_per_q, oof_band, median_calibrator = _cv_metrics(df)
    metrics["filter_stats"] = filter_stats

    # Final models: fit on full filtered data with CV-tuned per-quantile iters
    X_arr, cat_maps = _prepare_X(df)
    cat_indices = [_ALL_FEATURES.index(c) for c in CATEGORICAL_FEATURES]

    models = {}
    for name, quantile in _QUANTILES.items():
        params = {**_LGB_PARAMS, "n_estimators": best_iters_per_q[name]}
        model = lgb.LGBMRegressor(
            objective="quantile", alpha=quantile, **params,
        )
        model.fit(X_arr, y_log, categorical_feature=cat_indices)
        models[name] = model

    # Build OOF map keyed by olx_id (the stable join key the dashboard uses).
    # Listings without an olx_id fall back to model.predict at score time.
    oof_preds: dict[str, tuple[float, float, float]] = {}
    if "olx_id" in df.columns:
        for i, oid in enumerate(df["olx_id"].values):
            if oid is None or (isinstance(oid, float) and np.isnan(oid)):
                continue
            oof_preds[str(oid)] = (
                float(oof_band[0, i]),
                float(oof_band[1, i]),
                float(oof_band[2, i]),
            )

    return models, cat_maps, metrics, oof_preds, median_calibrator


def predict_prices(
    models: dict[str, lgb.LGBMRegressor],
    cat_maps: dict[str, dict[str, int]],
    listings_df: pd.DataFrame,
    conformal_q: float = 0.0,
    oof_preds: dict[str, tuple[float, float, float]] | None = None,
    median_calibrator: IsotonicRegression | None = None,
) -> pd.DataFrame:
    """Predict fair price range for each listing.

    Models predict in log1p space; this function back-transforms with expm1.
    If *conformal_q* is provided (from CQR calibration in CV metrics), the
    [low, high] band is widened symmetrically in log space — asymmetric in
    price space, which is appropriate for heavy-tailed targets.

    If *oof_preds* is provided (a dict olx_id → (low, median, high) in price
    space, shipped with the model bundle), listings whose olx_id appears in
    that dict get their out-of-fold CV predictions instead of in-sample
    model.predict output. This prevents the deal-scoring loop from comparing a
    listing's price against a "fair price" the model already memorized.

    If *median_calibrator* is provided (an IsotonicRegression fit on
    OOF ``predicted → actual``), it is applied to the median quantile for new
    rows only — OOF predictions already have it baked in. Removes systematic
    bias the bucket diagnostic surfaces (e.g. −27% on the cheap segment).

    The returned [low, median, high] is sorted per row to guarantee
    low ≤ median ≤ high (Chernozhukov et al. 2010 — independent quantile
    regressors can cross; this is the standard non-crossing repair).
    """
    X_arr, _ = _prepare_X(listings_df, cat_maps)

    # Model output is in log1p(price) space.
    log_median = models["median"].predict(X_arr)
    log_low = models["low"].predict(X_arr) - conformal_q
    log_high = models["high"].predict(X_arr) + conformal_q

    median = np.expm1(log_median)
    low = np.expm1(log_low)
    high = np.expm1(log_high)

    # Apply isotonic calibration to the median for new rows. OOF preds (set
    # below) skip this since they were already calibrated at training time.
    # We track whether calibration ran so the band-assembly step below can
    # bracket the median by min/max instead of naive-sorting (which would
    # silently swap a calibrated median out of position when isotonic moves
    # it outside the raw [low, high] range).
    median_was_calibrated = median_calibrator is not None
    if median_was_calibrated:
        median = median_calibrator.predict(np.maximum(median, 0))

    # Override with OOF predictions for listings the model was trained on.
    if oof_preds and "olx_id" in listings_df.columns:
        olx_ids = listings_df["olx_id"].values
        for i, oid in enumerate(olx_ids):
            if oid is None or (isinstance(oid, float) and np.isnan(oid)):
                continue
            entry = oof_preds.get(str(oid))
            if entry is None:
                continue
            low[i], median[i], high[i] = entry

    # Repair quantile crossing without losing the calibrated median.
    # Two crossing sources: (1) independent quantile fits can produce raw
    # low > raw high; (2) isotonic calibration can move the median outside
    # [raw_low, raw_high]. A blind sort fixes (1) but breaks (2) — it would
    # bury the calibrated median in the low or high slot. So we fix the band
    # edges first, then bracket the median by min/max.
    if median_was_calibrated:
        fixed_low = np.minimum(low, high)
        fixed_high = np.maximum(low, high)
        low = np.minimum(fixed_low, median)
        high = np.maximum(fixed_high, median)
    else:
        # Pre-isotonic behaviour — naive sort is fine when median sits
        # naturally inside [low, high].
        stacked = np.sort(np.stack([low, median, high]), axis=0)
        low, median, high = stacked[0], stacked[1], stacked[2]

    return pd.DataFrame({
        "predicted_price": np.maximum(median, 0),
        "fair_price_low": np.maximum(low, 0),
        "fair_price_high": np.maximum(high, 0),
    }, index=listings_df.index).round(0)


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def save_model(
    models: dict[str, lgb.LGBMRegressor],
    cat_maps: dict[str, dict[str, int]],
    metrics: dict,
    oof_preds: dict[str, tuple[float, float, float]] | None = None,
    median_calibrator: IsotonicRegression | None = None,
) -> None:
    """Save trained model bundle to disk and append metrics to history."""
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "schema_version": _SCHEMA_VERSION,
        "feature_names": list(_ALL_FEATURES),
        "models": models,
        "cat_maps": cat_maps,
        "metrics": metrics,
        "oof_preds": oof_preds or {},
        "median_calibrator": median_calibrator,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    joblib.dump(bundle, _MODEL_PATH)
    _append_metrics(metrics)


def load_model(
    max_age_hours: float = _MODEL_MAX_AGE_HOURS,
) -> tuple[dict, dict, dict, dict, IsotonicRegression | None] | None:
    """Load saved model if it exists, is fresh, and matches current schema.

    Returns (models, cat_maps, metrics, oof_preds, median_calibrator) or None.

    Mismatches that cause rejection:
      - file age > max_age_hours
      - schema_version != _SCHEMA_VERSION (e.g. log target was added,
        isotonic calibrator was added)
      - feature_names != _ALL_FEATURES (training features were renamed/
        added/removed)

    Returning None falls through to "no fresh model" — the dashboard then logs
    a warning and skips price predictions until the next CI training run.
    """
    if not _MODEL_PATH.exists():
        return None
    age_hours = (time.time() - _MODEL_PATH.stat().st_mtime) / 3600
    if age_hours > max_age_hours:
        return None
    try:
        bundle = joblib.load(_MODEL_PATH)
        if bundle.get("schema_version") != _SCHEMA_VERSION:
            return None
        if bundle.get("feature_names") != list(_ALL_FEATURES):
            return None
        return (
            bundle["models"],
            bundle["cat_maps"],
            bundle.get("metrics", {}),
            bundle.get("oof_preds", {}),
            bundle.get("median_calibrator"),
        )
    except Exception:
        return None


def _append_metrics(metrics: dict) -> None:
    """Append metrics entry to the JSON history file."""
    history: list[dict] = []
    if _METRICS_PATH.exists():
        try:
            history = json.loads(_METRICS_PATH.read_text())
        except (json.JSONDecodeError, ValueError):
            pass
    history.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **metrics,
    })
    history = history[-100:]
    _METRICS_PATH.write_text(json.dumps(history, indent=2))


def load_metrics_history() -> list[dict]:
    """Load full metrics history for dashboard display."""
    if not _METRICS_PATH.exists():
        return []
    try:
        return json.loads(_METRICS_PATH.read_text())
    except (json.JSONDecodeError, ValueError):
        return []


def save_importance(importance_df: pd.DataFrame) -> None:
    """Persist permutation importance next to the model.

    Computed once at training time and shipped in the data release so the
    dashboard doesn't rerun a 690-predict permutation loop on every load.
    """
    if importance_df is None or importance_df.empty:
        return
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    _IMPORTANCE_PATH.write_text(importance_df.to_json(orient="records"))


def load_importance() -> pd.DataFrame:
    """Return the shipped importance frame, or empty if missing."""
    if not _IMPORTANCE_PATH.exists():
        return pd.DataFrame()
    try:
        return pd.read_json(_IMPORTANCE_PATH, orient="records")
    except (ValueError, json.JSONDecodeError):
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Feature analysis
# ---------------------------------------------------------------------------

def compute_feature_completeness(listings_df: pd.DataFrame) -> pd.Series:
    """Fraction of model features that are non-null per listing (0–1)."""
    cols = [c for c in _ALL_FEATURES if c in listings_df.columns]
    if not cols:
        return pd.Series(0.0, index=listings_df.index, name="feature_fill_rate")
    present = listings_df[cols].notna().sum(axis=1)
    return (present / len(_ALL_FEATURES)).rename("feature_fill_rate")


def compute_data_completeness(
    feature_fill_rate: pd.Series,
    sample_sizes: pd.Series,
) -> pd.Series:
    """Combined data completeness: feature fill (60%) + sample confidence (40%)."""
    sample_conf = (sample_sizes.fillna(0) / 20).clip(upper=1.0)
    return (0.6 * feature_fill_rate + 0.4 * sample_conf).rename("data_completeness")


def compute_permutation_importance(
    models: dict[str, lgb.LGBMRegressor],
    cat_maps: dict[str, dict[str, int]],
    listings_df: pd.DataFrame,
    n_repeats: int = 10,
) -> pd.DataFrame:
    """Permutation importance for each feature across all three quantile models.

    Scorer: alpha-aligned negative pinball loss (the metric each quantile model
    is actually optimizing) instead of the sklearn default R², which makes no
    sense for the tail quantiles. Targets are in log space because the models
    predict log1p(price).

    Note: importance is computed on the same data the model was trained on,
    which mildly overstates feature contributions. A cleaner version would
    score on a held-out fold, but that costs a re-fit per quantile and the
    current ranking is already stable enough for the dashboard's purpose.
    """
    df = listings_df[
        listings_df["price_eur"].notna()
        & listings_df["year"].notna()
        & listings_df["mileage_km"].notna()
    ].copy()

    y_log = np.log1p(np.maximum(df["price_eur"].values.astype(float), 0))
    X_arr, _ = _prepare_X(df, cat_maps)

    rows = []
    for name, alpha in _QUANTILES.items():
        scorer = make_scorer(
            mean_pinball_loss, alpha=alpha, greater_is_better=False,
        )
        result = permutation_importance(
            models[name], X_arr, y_log,
            n_repeats=n_repeats, random_state=42, n_jobs=-1,
            scoring=scorer,
        )
        rows.append(pd.Series(
            result.importances_mean, index=_ALL_FEATURES, name=f"{name}_importance",
        ))

    imp = pd.concat(rows, axis=1).reset_index()
    imp.columns = ["feature", "low_importance", "median_importance", "high_importance"]
    return imp.sort_values("median_importance", ascending=False).reset_index(drop=True)
