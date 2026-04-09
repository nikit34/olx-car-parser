"""Gradient boosting price model for fair market value estimation.

Uses all available features including LLM-extracted fields (accident,
condition, RHD, etc.) and market data (avg_days_to_sell).

HistGradientBoostingRegressor handles NaN natively — ideal for LLM fields
which are null for ~20% of listings.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


MAX_HGB_BINS = 255
_OTHER_CATEGORY = "__other__"

NUMERIC_FEATURES = [
    "year", "mileage_km", "engine_cc", "horsepower",
    "desc_mentions_num_owners", "avg_days_to_sell",
]

BOOL_FEATURES = [
    "desc_mentions_accident", "desc_mentions_repair",
    "desc_mentions_customs_cleared", "right_hand_drive",
    "taxi_fleet_rental", "warranty", "first_owner_selling",
]

CATEGORICAL_FEATURES = [
    "brand", "model", "fuel_type", "transmission", "segment",
    "urgency",
]

_ALL_FEATURES = NUMERIC_FEATURES + BOOL_FEATURES + CATEGORICAL_FEATURES


def _build_categorical_mapping(vals: pd.Series) -> dict[str, int]:
    """Build a stable mapping that fits HGBR's categorical bin limit."""
    non_missing = vals.dropna().astype(str)
    if non_missing.empty:
        return {}

    counts = non_missing.value_counts()
    ordered = sorted(counts.index.tolist(), key=lambda value: (-int(counts[value]), value))

    if len(ordered) > MAX_HGB_BINS:
        kept = sorted(ordered[: MAX_HGB_BINS - 1])
        values = [*kept, _OTHER_CATEGORY]
    else:
        values = sorted(ordered)

    return {value: i for i, value in enumerate(values)}


def _encode_categoricals(
    df: pd.DataFrame,
    cat_maps: dict[str, dict[str, int]] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    """Ordinal-encode categoricals while staying within HGBR's category limit."""
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


def train_price_model(
    listings_df: pd.DataFrame,
    min_samples: int = 50,
) -> tuple[HistGradientBoostingRegressor, dict[str, dict[str, int]]] | None:
    """Train a gradient boosting model on all active priced listings.

    Returns (model, category_maps) or None if insufficient data.
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

    model = HistGradientBoostingRegressor(
        max_iter=700,
        max_depth=4,
        learning_rate=0.05,
        min_samples_leaf=10,
        l2_regularization=1.5,
        max_bins=MAX_HGB_BINS,
        categorical_features=cat_indices,
        random_state=42,
    )
    model.fit(X_arr, y)

    return model, cat_maps


def predict_prices(
    model: HistGradientBoostingRegressor,
    cat_maps: dict[str, dict[str, int]],
    listings_df: pd.DataFrame,
) -> pd.Series:
    """Predict fair price for each listing. Returns Series aligned with listings_df index."""
    X_arr, _ = _prepare_X(listings_df, cat_maps)
    predictions = model.predict(X_arr)
    return pd.Series(np.maximum(predictions, 0), index=listings_df.index, name="predicted_price")
