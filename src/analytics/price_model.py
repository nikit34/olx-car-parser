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
    "tires_condition", "urgency",
]

_ALL_FEATURES = NUMERIC_FEATURES + BOOL_FEATURES + CATEGORICAL_FEATURES


def _encode_categoricals(
    df: pd.DataFrame,
    cat_maps: dict[str, dict[str, int]] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    """Ordinal-encode categorical columns. Unknown values become NaN (handled by HGBR)."""
    out = df.copy()
    maps: dict[str, dict[str, int]] = cat_maps or {}

    for col in CATEGORICAL_FEATURES:
        if col not in out.columns:
            out[col] = np.nan
            continue

        vals = out[col].fillna("__missing__").astype(str)

        if col not in maps:
            uniques = sorted(vals.unique())
            maps[col] = {v: i for i, v in enumerate(uniques)}

        mapping = maps[col]
        out[col] = vals.map(mapping).astype(float)

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

    # max_bins must be >= max cardinality of any categorical feature + 1
    max_cat = max((len(m) for m in cat_maps.values()), default=0)
    max_bins = max(255, max_cat + 1)

    model = HistGradientBoostingRegressor(
        max_iter=700,
        max_depth=4,
        learning_rate=0.05,
        min_samples_leaf=10,
        l2_regularization=1.5,
        max_bins=max_bins,
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
