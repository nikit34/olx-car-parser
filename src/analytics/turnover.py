"""Sell speed and turnover analytics."""

import pandas as pd


def compute_turnover_stats(
    df: pd.DataFrame,
    relisted: set[str] | None = None,
    pairs: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Per brand+model+generation compute avg_days_to_sell and weekly_turnover.

    - avg_days_to_sell: median days on the market off the Kaplan-Meier curve in
      ``analytics.liquidity`` — the same number the public pages and the
      decision gates read. Until 2026-09-19 this was the mean of
      ``last_seen - first_seen`` over the listings that had already gone, which
      dropped every listing still on sale and counted a listing that came back
      a fortnight later as a sale. It fed the deal scorer's liquidity multiplier,
      so the segments that relist most — the cheap, OLX-heavy ones — were scored
      as the quickest to sell.
    - weekly_turnover: % of listings that went in the last 7 days. Given
      ``relisted`` (or ``pairs``), the ones that came back are not counted,
      because they did not sell.

    A segment thinner than the curve's floor gets no ``avg_days_to_sell`` at
    all rather than a mean over three listings; callers already treat the
    missing value as "no liquidity signal".
    """
    from src.analytics.liquidity import dom_by_segment

    group_keys = ["brand", "model", "generation"]
    out_cols = group_keys + ["avg_days_to_sell", "weekly_turnover"]

    if df.empty or "first_seen_at" not in df.columns:
        return pd.DataFrame(columns=out_cols)

    if "generation" not in df.columns:
        df = df.copy()
        df["generation"] = pd.NA

    if relisted is None and pairs is not None and not pairs.empty \
            and "original_olx_id" in pairs.columns:
        relisted = set(pairs["original_olx_id"].astype(str))

    curves = dom_by_segment(df, relisted=relisted, pairs=pairs)
    total = df.groupby(group_keys, dropna=False).size().reset_index(name="total_listings")

    def _days(row) -> float:
        gen = row.generation
        blank = gen is None or pd.isna(gen) or str(gen) == ""
        rec = None if blank else curves.get((row.brand, row.model, gen))
        rec = rec or curves.get((row.brand, row.model, None))
        return round(rec["md"], 1) if rec else float("nan")

    total["avg_days_to_sell"] = [_days(row) for row in total.itertuples(index=False)]

    inactive = df[df["is_active"] == False]
    if not inactive.empty and relisted:
        inactive = inactive[~inactive["olx_id"].astype(str).isin(relisted)]
    if inactive.empty:
        total["weekly_turnover"] = 0.0
        return total[out_cols]

    last = pd.to_datetime(inactive["last_seen_at"], errors="coerce").dt.tz_localize(None)
    recent = inactive[last >= pd.Timestamp.now() - pd.Timedelta(days=7)]
    weekly = recent.groupby(group_keys, dropna=False).size().reset_index(name="sold_last_week")

    result = total.merge(weekly, on=group_keys, how="left")
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
