"""Feature price premium calculations."""

import pandas as pd


def compute_premiums(df: pd.DataFrame, brand: str, model: str) -> dict | None:
    """For a given brand+model, compute price premiums per feature.

    Returns dict with keys: transmission, fuel_type, color, mileage_bracket, year.
    Each value is a DataFrame with columns: category, count, median_price, premium_eur, premium_pct.
    Returns None if fewer than 10 listings.
    """
    subset = df[
        (df["brand"] == brand) & (df["model"] == model) & df["price_eur"].notna()
    ].copy()

    if len(subset) < 10:
        return None

    baseline = subset["price_eur"].median()
    result = {}

    def _premium_for(col: str, min_group: int = 3) -> pd.DataFrame:
        if col not in subset.columns:
            return pd.DataFrame()
        valid = subset[subset[col].notna()]
        if valid.empty:
            return pd.DataFrame()
        grouped = (
            valid.groupby(col)["price_eur"]
            .agg(["median", "size"])
            .reset_index()
            .rename(columns={col: "category", "median": "median_price", "size": "count"})
        )
        grouped = grouped[grouped["count"] >= min_group]
        grouped["premium_eur"] = (grouped["median_price"] - baseline).round(0)
        grouped["premium_pct"] = ((grouped["median_price"] / baseline - 1) * 100).round(1)
        return grouped.sort_values("premium_eur", ascending=False)

    result["transmission"] = _premium_for("transmission", min_group=2)
    result["fuel_type"] = _premium_for("fuel_type", min_group=2)
    result["color"] = _premium_for("color", min_group=3)

    # Mileage brackets
    if "mileage_km" in subset.columns and subset["mileage_km"].notna().any():
        s = subset[subset["mileage_km"].notna()].copy()
        bins = [0, 50000, 100000, 150000, 200000, float("inf")]
        labels = ["0-50k", "50k-100k", "100k-150k", "150k-200k", "200k+"]
        s["mileage_bracket"] = pd.cut(s["mileage_km"], bins=bins, labels=labels)
        result["mileage_bracket"] = _premium_for.__wrapped__(s, "mileage_bracket", 2) if hasattr(_premium_for, '__wrapped__') else pd.DataFrame()
        # Compute manually since _premium_for uses subset
        grouped = (
            s.groupby("mileage_bracket", observed=True)["price_eur"]
            .agg(["median", "size"])
            .reset_index()
            .rename(columns={"mileage_bracket": "category", "median": "median_price", "size": "count"})
        )
        grouped = grouped[grouped["count"] >= 2]
        grouped["premium_eur"] = (grouped["median_price"] - baseline).round(0)
        grouped["premium_pct"] = ((grouped["median_price"] / baseline - 1) * 100).round(1)
        result["mileage_bracket"] = grouped.sort_values("premium_eur", ascending=False)

    # Year
    if "year" in subset.columns and subset["year"].notna().any():
        s = subset[subset["year"].notna()].copy()
        s["year_str"] = s["year"].astype(int).astype(str)
        grouped = (
            s.groupby("year_str")["price_eur"]
            .agg(["median", "size"])
            .reset_index()
            .rename(columns={"year_str": "category", "median": "median_price", "size": "count"})
        )
        grouped = grouped[grouped["count"] >= 2]
        grouped["premium_eur"] = (grouped["median_price"] - baseline).round(0)
        grouped["premium_pct"] = ((grouped["median_price"] / baseline - 1) * 100).round(1)
        result["year"] = grouped.sort_values("category")

    result["baseline_price"] = baseline
    result["sample_size"] = len(subset)
    return result
