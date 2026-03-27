"""Regression helpers for depreciation curves and price estimation."""

import numpy as np
import pandas as pd


def depreciation_curve(df: pd.DataFrame, brand: str, model: str) -> dict | None:
    """Fit price ~ year (linear) for a given brand+model.

    Returns dict with slope, intercept, r_squared, and filtered data subset,
    or None if fewer than 3 data points.
    """
    subset = df[
        (df["brand"] == brand)
        & (df["model"] == model)
        & df["price_eur"].notna()
        & df["year"].notna()
    ].copy()
    if len(subset) < 3:
        return None

    x = subset["year"].values.astype(float)
    y = subset["price_eur"].values.astype(float)
    coeffs = np.polyfit(x, y, deg=1)
    predicted = np.polyval(coeffs, x)
    ss_res = np.sum((y - predicted) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return {
        "slope": coeffs[0],
        "intercept": coeffs[1],
        "r_squared": round(r2, 3),
        "data": subset,
    }


def estimate_price(
    df: pd.DataFrame,
    brand: str,
    model: str,
    year: int,
    mileage_km: int,
    fuel_type: str | None = None,
) -> dict | None:
    """Multivariate regression: price ~ year + mileage.

    Returns dict with predicted, p25, median, p75, sample_size,
    or None if fewer than 5 data points.
    """
    subset = df[
        (df["brand"] == brand)
        & (df["model"] == model)
        & df["price_eur"].notna()
        & df["year"].notna()
        & df["mileage_km"].notna()
    ].copy()

    if fuel_type:
        subset_fuel = subset[subset["fuel_type"] == fuel_type]
        if len(subset_fuel) >= 5:
            subset = subset_fuel

    if len(subset) < 5:
        return None

    X = np.column_stack([
        subset["year"].values.astype(float),
        subset["mileage_km"].values.astype(float),
        np.ones(len(subset)),
    ])
    y = subset["price_eur"].values.astype(float)
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    predicted = coeffs[0] * year + coeffs[1] * mileage_km + coeffs[2]
    residuals = y - X @ coeffs

    return {
        "predicted": max(predicted, 0),
        "p25": max(predicted + np.percentile(residuals, 25), 0),
        "median": max(predicted + np.median(residuals), 0),
        "p75": max(predicted + np.percentile(residuals, 75), 0),
        "sample_size": len(subset),
    }
