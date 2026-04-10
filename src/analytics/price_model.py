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


def _cv_metrics(
    X: np.ndarray,
    y: np.ndarray,
    cat_indices: list[int],
    n_splits: int = 5,
) -> dict:
    """Run k-fold cross-validation on the median model and return quality metrics."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.full(len(y), np.nan)

    for train_idx, val_idx in kf.split(X):
        model = lgb.LGBMRegressor(
            objective="quantile", alpha=0.5, **_LGB_PARAMS,
        )
        model.fit(X[train_idx], y[train_idx], categorical_feature=cat_indices)
        oof[val_idx] = model.predict(X[val_idx])

    oof = np.maximum(oof, 0)
    mae = float(np.mean(np.abs(y - oof)))
    mape = float(np.mean(np.abs((y - oof) / np.maximum(y, 1))) * 100)
    ss_res = np.sum((y - oof) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "mae": round(mae, 0),
        "mape": round(mape, 1),
        "r2": round(r2, 3),
        "n_samples": int(len(y)),
        "cv_folds": n_splits,
    }


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

    if len(df) < min_samples:
        return None

    y = df["price_eur"].values.astype(float)
    X_arr, cat_maps = _prepare_X(df)
    cat_indices = [_ALL_FEATURES.index(c) for c in CATEGORICAL_FEATURES]

    # Cross-validation metrics (median model only — most important)
    metrics = _cv_metrics(X_arr, y, cat_indices)

    # Train final models on full data
    models = {}
    for name, quantile in [("low", 0.1), ("median", 0.5), ("high", 0.9)]:
        model = lgb.LGBMRegressor(
            objective="quantile", alpha=quantile, **_LGB_PARAMS,
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
