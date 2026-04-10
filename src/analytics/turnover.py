"""Sell speed and turnover analytics."""

import pandas as pd


def compute_turnover_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per brand+model+generation compute avg_days_to_sell and weekly_turnover.

    - avg_days_to_sell: mean duration (first_seen -> last_seen) for inactive listings
    - weekly_turnover: % of listings that went inactive in the last 7 days
    """
    group_keys = ["brand", "model", "generation"]
    out_cols = group_keys + ["avg_days_to_sell", "weekly_turnover"]

    if df.empty or "first_seen_at" not in df.columns:
        return pd.DataFrame(columns=out_cols)

    # Ensure generation column exists
    if "generation" not in df.columns:
        df = df.copy()
        df["generation"] = pd.NA

    inactive = df[df["is_active"] == False].copy()

    if inactive.empty:
        total = df.groupby(group_keys, dropna=False).size().reset_index(name="total_listings")
        total["avg_days_to_sell"] = pd.NA
        total["weekly_turnover"] = 0.0
        return total[out_cols]

    first = pd.to_datetime(inactive["first_seen_at"]).dt.tz_localize(None)
    last = pd.to_datetime(inactive["last_seen_at"]).dt.tz_localize(None)
    inactive["duration_days"] = (last - first).dt.days

    avg_sell = (
        inactive.groupby(group_keys, dropna=False)["duration_days"]
        .mean()
        .round(1)
        .reset_index(name="avg_days_to_sell")
    )

    one_week_ago = pd.Timestamp.now() - pd.Timedelta(days=7)
    recent = inactive[last >= one_week_ago]
    weekly = recent.groupby(group_keys, dropna=False).size().reset_index(name="sold_last_week")

    total = df.groupby(group_keys, dropna=False).size().reset_index(name="total_listings")

    result = avg_sell.merge(weekly, on=group_keys, how="left")
    result = result.merge(total, on=group_keys, how="left")
    result["sold_last_week"] = result["sold_last_week"].fillna(0)
    result["weekly_turnover"] = (
        result["sold_last_week"] / result["total_listings"] * 100
    ).round(1)

    return result[out_cols]
