"""Seasonal price patterns."""

import pandas as pd


def compute_seasonality(
    history_df: pd.DataFrame, listings_df: pd.DataFrame
) -> pd.DataFrame | None:
    """Compute price index by month and segment.

    Returns DataFrame with columns: segment, month, month_name, avg_price, price_index.
    price_index = 100 * (month_avg / yearly_avg).
    Returns None if insufficient data (< 3 months).
    """
    if history_df.empty or listings_df.empty:
        return None

    h = history_df.copy()
    h["date"] = pd.to_datetime(h["date"])
    h["month"] = h["date"].dt.month

    if h["month"].nunique() < 3:
        return None

    # Map brand+model to segment from listings
    seg_map = (
        listings_df[listings_df["segment"].notna()]
        .groupby(["brand", "model"])["segment"]
        .first()
        .reset_index()
    )
    h = h.merge(seg_map, on=["brand", "model"], how="left")
    h = h[h["segment"].notna() & h["median_price_eur"].notna()]

    if h.empty:
        return None

    monthly = (
        h.groupby(["segment", "month"])["median_price_eur"]
        .mean()
        .reset_index(name="avg_price")
    )
    yearly = (
        h.groupby("segment")["median_price_eur"]
        .mean()
        .reset_index(name="yearly_avg")
    )
    monthly = monthly.merge(yearly, on="segment")
    monthly["price_index"] = (monthly["avg_price"] / monthly["yearly_avg"] * 100).round(1)

    month_names = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
    }
    monthly["month_name"] = monthly["month"].map(month_names)

    return monthly


def best_buy_sell_months(seasonality_df: pd.DataFrame) -> pd.DataFrame:
    """Identify cheapest (buy) and most expensive (sell) month per segment."""
    if seasonality_df is None or seasonality_df.empty:
        return pd.DataFrame()

    results = []
    for segment, group in seasonality_df.groupby("segment"):
        best_buy = group.loc[group["price_index"].idxmin()]
        best_sell = group.loc[group["price_index"].idxmax()]
        results.append({
            "Segment": segment,
            "Best Buy Month": best_buy["month_name"],
            "Buy Index": best_buy["price_index"],
            "Best Sell Month": best_sell["month_name"],
            "Sell Index": best_sell["price_index"],
            "Spread %": round(best_sell["price_index"] - best_buy["price_index"], 1),
        })

    return pd.DataFrame(results).sort_values("Spread %", ascending=False)
