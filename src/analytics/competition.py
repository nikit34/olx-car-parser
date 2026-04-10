"""Competition density analysis."""

import pandas as pd


def compute_competition_density(
    listings_df: pd.DataFrame, turnover_df: pd.DataFrame
) -> pd.DataFrame:
    """For each model+district, compute competition metrics.

    Returns DataFrame with columns:
    - brand, model, district, label
    - local_count: active listings in this district
    - national_count: total active listings nationally
    - districts_present: how many districts have this model
    - national_avg_per_district: national_count / districts_present
    - saturation: local_count / national_avg_per_district (>1 = oversaturated)
    - avg_days_to_sell: from turnover data
    """
    if listings_df.empty:
        return pd.DataFrame()

    active = listings_df[
        (listings_df["is_active"] == True)
        & listings_df["district"].notna()
        & listings_df["price_eur"].notna()
    ].copy()

    if active.empty:
        return pd.DataFrame()

    local = (
        active.groupby(["brand", "model", "district"])
        .agg(
            local_count=("price_eur", "size"),
            local_median=("price_eur", "median"),
        )
        .reset_index()
    )

    national = (
        active.groupby(["brand", "model"])
        .agg(
            national_count=("price_eur", "size"),
            districts_present=("district", "nunique"),
        )
        .reset_index()
    )

    result = local.merge(national, on=["brand", "model"])
    result["national_avg_per_district"] = (
        result["national_count"] / result["districts_present"]
    ).round(1)
    result["saturation"] = (
        result["local_count"] / result["national_avg_per_district"]
    ).round(2)

    if not turnover_df.empty:
        merge_keys = ["brand", "model"]
        turnover_cols = ["brand", "model", "avg_days_to_sell"]
        if "generation" in turnover_df.columns and "generation" in result.columns:
            merge_keys.append("generation")
            turnover_cols.append("generation")
        result = result.merge(
            turnover_df[turnover_cols],
            on=merge_keys,
            how="left",
        )

    result["label"] = result["brand"] + " " + result["model"]
    return result.sort_values("saturation", ascending=False)
