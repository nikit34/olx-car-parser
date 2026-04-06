"""Gradient boosting price model for fair market value estimation.

Replaces per-generation linear regression with a single global model
that uses all available features: year, mileage, engine specs, fuel type,
transmission, segment, brand, model.

Uses HistGradientBoostingRegressor which handles NaN natively and is fast
on datasets of this size (~4k listings).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


NUMERIC_FEATURES = ["year", "mileage_km", "engine_cc", "horsepower"]
CATEGORICAL_FEATURES = ["brand", "model", "fuel_type", "transmission", "segment"]

_ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


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
        # Unseen categories → NaN (HGBR handles natively)

    return out, maps


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

    X = df.reindex(columns=_ALL_FEATURES).copy()
    y = df["price_eur"].values.astype(float)

    X, cat_maps = _encode_categoricals(X)
    X_arr = X[_ALL_FEATURES].values.astype(float)

    # Categorical feature indices for HGBR (tells it which columns are categorical)
    cat_indices = [_ALL_FEATURES.index(c) for c in CATEGORICAL_FEATURES]

    model = HistGradientBoostingRegressor(
        max_iter=300,
        max_depth=6,
        learning_rate=0.08,
        min_samples_leaf=5,
        l2_regularization=1.0,
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
    X = listings_df.reindex(columns=_ALL_FEATURES).copy()
    X, _ = _encode_categoricals(X, cat_maps)
    X_arr = X[_ALL_FEATURES].values.astype(float)

    predictions = model.predict(X_arr)
    return pd.Series(np.maximum(predictions, 0), index=listings_df.index, name="predicted_price")
