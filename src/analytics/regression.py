"""Regression helpers for depreciation curves and price estimation."""

import numpy as np
import pandas as pd


def estimate_price(
    df: pd.DataFrame,
    brand: str,
    model: str,
    year: int,
    mileage_km: int,
    fuel_type: str | None = None,
    engine_cc: int | None = None,
) -> dict | None:
    """Multivariate regression: price ~ year + year² + log(mileage) [+ engine_cc].

    year² captures accelerated depreciation in first years;
    log(mileage) captures diminishing impact at high km.

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

    year_vals = subset["year"].values.astype(float)
    log_mileage = np.log1p(subset["mileage_km"].values.astype(float))

    has_cc = (
        engine_cc is not None
        and "engine_cc" in subset.columns
        and subset["engine_cc"].notna().sum() >= len(subset) * 0.5
    )
    if has_cc:
        cc_vals = subset["engine_cc"].fillna(subset["engine_cc"].median()).values.astype(float)
        X = np.column_stack([year_vals, year_vals ** 2, log_mileage, cc_vals, np.ones(len(subset))])
    else:
        X = np.column_stack([year_vals, year_vals ** 2, log_mileage, np.ones(len(subset))])

    y = subset["price_eur"].values.astype(float)
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    y_val = float(year)
    lm_val = np.log1p(float(mileage_km))
    if has_cc:
        predicted = coeffs[0] * y_val + coeffs[1] * y_val ** 2 + coeffs[2] * lm_val + coeffs[3] * engine_cc + coeffs[4]
    else:
        predicted = coeffs[0] * y_val + coeffs[1] * y_val ** 2 + coeffs[2] * lm_val + coeffs[3]
    residuals = y - X @ coeffs

    return {
        "predicted": max(predicted, 0),
        "p25": max(predicted + np.percentile(residuals, 25), 0),
        "median": max(predicted + np.median(residuals), 0),
        "p75": max(predicted + np.percentile(residuals, 75), 0),
        "sample_size": len(subset),
    }
