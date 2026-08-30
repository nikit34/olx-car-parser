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


def compute_sell_speed_by_model(df: pd.DataFrame, min_sample: int = 8) -> pd.DataFrame:
    """Per (brand, model): the MEDIAN days a listing stays up before it goes,
    plus the observed sample size ``sell_n``.

    Differs from ``compute_turnover_stats`` deliberately, for the public product:
    - keyed on (brand, model) only — the worker's deal rows carry brand+model
      but not a reliable generation, so a model-level number always joins;
    - MEDIAN not mean — listing durations are right-skewed (a few stale tails);
    - gated on ``min_sample`` observed endings so we never surface a number
      built from one or two data points. Segments below the floor are dropped
      (the caller then shows nothing rather than a noisy figure).

    The median comes from ``liquidity.build_liquidity`` — the same Kaplan-Meier
    curve the /liquidez pages publish — so the figure on a deal card, on a model
    page and on the liquidity page is one number and not three. Taking the plain
    median of the listings that had already ended (what this did until
    2026-08-30) ignored every listing still live and read ~10 days fast.
    "Ended" is a sold-OR-withdrawn proxy (a listing leaving the scrape), so
    treat the figure as indicative.
    """
    from src.analytics.liquidity import build_liquidity, sell_speed_frame

    cols = ["brand", "model", "sell_days", "sell_n"]
    if df.empty or "is_active" not in df.columns or "first_seen_at" not in df.columns:
        return pd.DataFrame(columns=cols)
    out = sell_speed_frame(build_liquidity(df), min_events=min_sample)
    if out.empty:
        return pd.DataFrame(columns=cols)
    out["sell_days"] = out["sell_days"].astype("Int64")
    out["sell_n"] = out["sell_n"].astype("int64")
    return out[cols].reset_index(drop=True)
