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
from sklearn.model_selection import KFold
from sklearn.inspection import permutation_importance


_MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_MODEL_PATH = _MODEL_DIR / "price_model.joblib"
_METRICS_PATH = _MODEL_DIR / "price_metrics.json"
_MODEL_MAX_AGE_HOURS = 24
_MIN_CATEGORY_COUNT = 3
_OTHER_CATEGORY = "__other__"

NUMERIC_FEATURES = [
    "year", "mileage_km", "engine_cc", "horsepower",
    "desc_mentions_num_owners", "avg_days_to_sell",
    "photo_count", "description_length", "seats",
    "tuning_or_mods_count",
    "num_price_drops", "max_drop_pct", "price_drop_velocity",
    "days_since_last_drop",
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
    n_estimators=700,
    max_depth=4,
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


def _cv_metrics(
    df: pd.DataFrame,
    n_splits: int = 5,
) -> tuple[dict, int]:
    """K-fold CV for all three quantile models with per-fold categorical
    encoding (no leakage) and early stopping to tune n_estimators.

    Returns (metrics_dict, suggested_n_estimators).
    """
    y_all = df["price_eur"].values.astype(float)
    n_splits = min(n_splits, max(2, len(df) // 20))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cat_indices = [_ALL_FEATURES.index(c) for c in CATEGORICAL_FEATURES]

    oof = {name: np.full(len(y_all), np.nan) for name in _QUANTILES}
    best_iters: list[int] = []

    for train_idx, val_idx in kf.split(df):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        # Encode categoricals on train only, apply mapping to val — no leakage
        X_train, cat_maps_fold = _prepare_X(train_df)
        X_val, _ = _prepare_X(val_df, cat_maps_fold)
        y_train = y_all[train_idx]
        y_val = y_all[val_idx]

        for name, alpha in _QUANTILES.items():
            model = lgb.LGBMRegressor(
                objective="quantile", alpha=alpha, **_LGB_PARAMS,
            )
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                categorical_feature=cat_indices,
                callbacks=[lgb.early_stopping(_EARLY_STOPPING_ROUNDS, verbose=False)],
            )
            oof[name][val_idx] = model.predict(X_val)
            if name == "median":
                best_iters.append(int(model.best_iteration_ or _LGB_PARAMS["n_estimators"]))

    for name in oof:
        oof[name] = np.maximum(oof[name], 0)

    median_oof = oof["median"]
    mae = float(np.mean(np.abs(y_all - median_oof)))
    mape = float(np.mean(np.abs((y_all - median_oof) / np.maximum(y_all, 1))) * 100)
    ss_res = float(np.sum((y_all - median_oof) ** 2))
    ss_tot = float(np.sum((y_all - np.mean(y_all)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    pinball = {
        name: _pinball_loss(y_all, oof[name], alpha)
        for name, alpha in _QUANTILES.items()
    }
    # Fraction of true prices inside the [p10, p90] band — should be ~0.80
    coverage_80 = float(np.mean((y_all >= oof["low"]) & (y_all <= oof["high"])))

    suggested = int(np.median(best_iters)) if best_iters else _LGB_PARAMS["n_estimators"]
    suggested = max(suggested, _MIN_N_ESTIMATORS)

    return {
        "mae": round(mae, 0),
        "mape": round(mape, 1),
        "r2": round(r2, 3),
        "pinball_low": round(pinball["low"], 1),
        "pinball_median": round(pinball["median"], 1),
        "pinball_high": round(pinball["high"], 1),
        "coverage_80": round(coverage_80, 3),
        "best_n_estimators": suggested,
        "n_samples": int(len(y_all)),
        "cv_folds": n_splits,
    }, suggested


def train_price_model(
    listings_df: pd.DataFrame,
    min_samples: int = 50,
) -> tuple[dict[str, lgb.LGBMRegressor], dict[str, dict[str, int]], dict] | None:
    """Train quantile regression models: median, low (10th), high (90th).

    Returns ({"median": model, "low": model, "high": model}, category_maps, metrics)
    or None if insufficient data.
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

    y = df["price_eur"].values.astype(float)

    # CV with per-fold cat encoding + early stopping → tuned n_estimators
    metrics, best_n_estimators = _cv_metrics(df)
    metrics["filter_stats"] = filter_stats

    # Final models: fit on full filtered data with CV-tuned n_estimators
    X_arr, cat_maps = _prepare_X(df)
    cat_indices = [_ALL_FEATURES.index(c) for c in CATEGORICAL_FEATURES]

    final_params = {**_LGB_PARAMS, "n_estimators": best_n_estimators}
    models = {}
    for name, quantile in _QUANTILES.items():
        model = lgb.LGBMRegressor(
            objective="quantile", alpha=quantile, **final_params,
        )
        model.fit(X_arr, y, categorical_feature=cat_indices)
        models[name] = model

    return models, cat_maps, metrics


def predict_prices(
    models: dict[str, lgb.LGBMRegressor],
    cat_maps: dict[str, dict[str, int]],
    listings_df: pd.DataFrame,
) -> pd.DataFrame:
    """Predict fair price range for each listing."""
    X_arr, _ = _prepare_X(listings_df, cat_maps)

    median = models["median"].predict(X_arr)
    low = models["low"].predict(X_arr)
    high = models["high"].predict(X_arr)

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
) -> None:
    """Save trained model bundle to disk and append metrics to history."""
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "models": models,
        "cat_maps": cat_maps,
        "metrics": metrics,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    joblib.dump(bundle, _MODEL_PATH)
    _append_metrics(metrics)


def load_model(
    max_age_hours: float = _MODEL_MAX_AGE_HOURS,
) -> tuple[dict, dict, dict] | None:
    """Load saved model if it exists and is fresh enough."""
    if not _MODEL_PATH.exists():
        return None
    age_hours = (time.time() - _MODEL_PATH.stat().st_mtime) / 3600
    if age_hours > max_age_hours:
        return None
    try:
        bundle = joblib.load(_MODEL_PATH)
        return bundle["models"], bundle["cat_maps"], bundle.get("metrics", {})
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
    """Permutation importance for each feature across all three quantile models."""
    df = listings_df[
        listings_df["price_eur"].notna()
        & listings_df["year"].notna()
        & listings_df["mileage_km"].notna()
    ].copy()

    y = df["price_eur"].values.astype(float)
    X_arr, _ = _prepare_X(df, cat_maps)

    rows = []
    for name in ("median", "low", "high"):
        result = permutation_importance(
            models[name], X_arr, y,
            n_repeats=n_repeats, random_state=42, n_jobs=-1,
        )
        rows.append(pd.Series(result.importances_mean, index=_ALL_FEATURES, name=f"{name}_importance"))

    imp = pd.concat(rows, axis=1).reset_index()
    imp.columns = ["feature", "median_importance", "low_importance", "high_importance"]
    return imp.sort_values("median_importance", ascending=False).reset_index(drop=True)
