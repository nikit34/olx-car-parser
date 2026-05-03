"""Segment-level resale metrics — one row per (brand, model, generation).

Built on top of the existing per-listing dashboard outputs (signals_df +
listings_df). The dashboard's deal feed answers "is this listing
under-priced?"; this module answers "which segments are worth my time at
all?" — by combining liquidity (how many sold recently, how fast),
under-valuation (current ask gap vs the model), market direction (price
delta over the last 30 days), and model trustworthiness (calibration of
predicted vs actual sold prices in this segment).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd


SEGMENT_KEYS = ("brand", "model", "generation")
_SOLD_REASON = "sold"


def _safe_median(s: pd.Series) -> float | None:
    if s.empty:
        return None
    m = s.median()
    return float(m) if pd.notna(m) else None


def _days_on_market(df: pd.DataFrame) -> pd.Series:
    """Days between first_seen_at and deactivated_at (sold lifetime).

    For active listings ``deactivated_at`` is NaT, returning NaN. For
    rows missing either timestamp we get NaN — the caller filters those
    out per metric.
    """
    deact = pd.to_datetime(df.get("deactivated_at"), errors="coerce", utc=True)
    first = pd.to_datetime(df.get("first_seen_at"), errors="coerce", utc=True)
    return (deact - first).dt.total_seconds() / 86400


def _is_sold(df: pd.DataFrame) -> pd.Series:
    if "is_active" not in df.columns:
        return pd.Series(False, index=df.index)
    inactive = ~df["is_active"].astype(bool)
    if "deactivation_reason" in df.columns:
        return inactive & (df["deactivation_reason"].astype(str) == _SOLD_REASON)
    return inactive


def compute_segment_metrics(
    listings: pd.DataFrame,
    signals: pd.DataFrame | None = None,
    now: datetime | None = None,
    sold_window_days: int = 60,
    trend_window_days: int = 30,
) -> pd.DataFrame:
    """One row per (brand, model, generation) with resale-relevant metrics.

    Inputs:
      - ``listings``: full DataFrame from get_listings_df + enrich_listings.
        Must include ``is_active``, ``deactivation_reason``, ``first_seen_at``,
        ``deactivated_at``, ``price_eur``, ``first_price_eur``,
        ``brand``, ``model``, and ``generation`` (NaN segments are bucketed
        as the empty-string generation).
      - ``signals``: optional output of compute_signals — used to compute
        ``avg_undervaluation`` only for actives that got a GB prediction.

    Output columns:
      - n_active: active non-duplicate count
      - n_sold_60d: count of sold (deactivation_reason='sold') with
        deactivated_at in the last ``sold_window_days``
      - median_dom: median days-on-market of those sold
      - avg_undervaluation: mean undervaluation_pct from signals over
        actives in this segment (NaN when no signals have it)
      - clearing_ratio: median(sold last ask) / median(active ask) — <1
        means the market lately cleared *below* what's currently asked
      - trend_30d: percent change in median active ask vs the same
        listings' first_price_eur (proxy for "did the segment ramp up
        or correct down recently"); falls back to NaN with insufficient
        data
      - calibration_residual: median(actual_last_ask − predicted) for
        sold listings whose OOF prediction is in the signals. Positive
        = model under-predicts the segment; negative = over-predicts.

    The composite ranker is intentionally NOT computed here — different
    callers want different weightings (deal-feed vs market-explorer).
    """
    if listings.empty:
        return pd.DataFrame()

    now = now or datetime.now(timezone.utc)
    df = listings.copy()
    if "duplicate_of" in df.columns:
        df = df[df["duplicate_of"].isna()]

    # Empty / NaN generation → bucket as "" so groupby keeps the row.
    if "generation" in df.columns:
        df["generation"] = df["generation"].fillna("").astype(str)
    else:
        df["generation"] = ""

    df["_dom"] = _days_on_market(df)
    df["_is_sold"] = _is_sold(df)
    df["_is_active"] = df["is_active"].astype(bool) if "is_active" in df.columns else True

    sold_cutoff = now - timedelta(days=sold_window_days)
    deact = pd.to_datetime(df.get("deactivated_at"), errors="coerce", utc=True)
    df["_recent_sold"] = df["_is_sold"] & (deact >= sold_cutoff)

    # Per-listing signal lookup so we can attach per-segment averages.
    if signals is not None and not signals.empty and "olx_id" in signals.columns:
        sig_uv = (
            signals.set_index("olx_id")["undervaluation_pct"]
            if "undervaluation_pct" in signals.columns else None
        )
        sig_pred = (
            signals.set_index("olx_id")["predicted_price"]
            if "predicted_price" in signals.columns else None
        )
    else:
        sig_uv = sig_pred = None

    rows: list[dict] = []
    for keys, group in df.groupby(list(SEGMENT_KEYS), dropna=False):
        brand, model, generation = keys
        active = group[group["_is_active"]]
        sold = group[group["_is_sold"]]
        recent_sold = group[group["_recent_sold"]]

        # Undervaluation — only over actives that the model scored.
        avg_uv = None
        if sig_uv is not None and not active.empty:
            joined = active["olx_id"].map(sig_uv)
            joined = joined[pd.notna(joined)]
            if not joined.empty:
                avg_uv = float(joined.mean())

        # Clearing ratio: how does median sold ask compare to median active ask?
        med_active_ask = _safe_median(active["price_eur"])
        med_sold_ask = _safe_median(sold["price_eur"])
        clearing_ratio = (
            med_sold_ask / med_active_ask
            if med_active_ask and med_sold_ask else None
        )

        # Trend: median active ask vs median first_price_eur on the same
        # actives. Diff/first → "by how much have currently-listed cars'
        # asks moved since they were originally posted". Reasonable proxy
        # for "is the segment under price pressure right now".
        trend_30d = None
        if "first_price_eur" in active.columns and not active.empty:
            first_p = pd.to_numeric(active["first_price_eur"], errors="coerce")
            cur_p = pd.to_numeric(active["price_eur"], errors="coerce")
            mask = (
                first_p.notna() & cur_p.notna() & (first_p > 0)
                & ((now - pd.to_datetime(active["first_seen_at"], errors="coerce", utc=True))
                   .dt.days <= trend_window_days * 3)  # within ~quarter
            )
            if mask.sum() >= 5:
                pct = (cur_p[mask] - first_p[mask]) / first_p[mask] * 100
                trend_30d = float(pct.median())

        # Calibration: for sold rows where signals had a predicted price,
        # how does the model compare to the actual last-ask? (Proxy — we
        # don't know the actual sold price, but last-ask is what the
        # transaction approximated.)
        calibration_residual = None
        if sig_pred is not None and not sold.empty:
            preds = sold["olx_id"].map(sig_pred)
            joined = pd.DataFrame({
                "actual": pd.to_numeric(sold["price_eur"], errors="coerce"),
                "pred": pd.to_numeric(preds, errors="coerce"),
            }).dropna()
            joined = joined[joined["pred"] > 0]
            if not joined.empty:
                calibration_residual = float(
                    (joined["actual"] - joined["pred"]).median()
                )

        rows.append({
            "brand": brand or "",
            "model": model or "",
            "generation": generation or "",
            "n_active": int(len(active)),
            "n_sold_60d": int(len(recent_sold)),
            "n_sold_total": int(len(sold)),
            "median_dom": _safe_median(recent_sold["_dom"]),
            "avg_undervaluation_pct": avg_uv,
            "median_active_ask_eur": med_active_ask,
            "median_sold_ask_eur": med_sold_ask,
            "clearing_ratio": clearing_ratio,
            "trend_30d_pct": trend_30d,
            "calibration_residual_eur": calibration_residual,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["n_active", "n_sold_60d"], ascending=False).reset_index(drop=True)
    return out


def compute_segment_time_series(
    snapshots: pd.DataFrame,
    sold_listings: pd.DataFrame | None = None,
    freq: str = "W",
) -> pd.DataFrame:
    """Real per-period (default weekly) median ask + volume per segment.

    ``snapshots`` is the output of ``get_price_snapshots_df`` — every
    scrape wrote a price observation, so the time series reflects the
    *actual* asking price observed at each scrape, not just the
    first-seen value. That matters because a listing posted in March
    at €10 k that dropped to €8 k in April should land in April's
    median, not March's.

    ``sold_listings``: optional listings DataFrame restricted to sold
    rows. Their ``deactivated_at`` + last ``price_eur`` snapshot are
    used to compute the "median sold last-ask per period" line that
    the user can compare against the active median ASK.

    Returns long-format DataFrame with columns:
      bucket, brand, model, generation, series, value, n
    where ``series`` is one of ``"active_ask_median"`` or
    ``"sold_lastask_median"``.
    """
    if snapshots is None or snapshots.empty:
        return pd.DataFrame(
            columns=["bucket", "brand", "model", "generation",
                     "series", "value", "n"]
        )

    df = snapshots.copy()
    if "duplicate_of" in df.columns:
        df = df[df["duplicate_of"].isna()]
    df["scraped_at"] = pd.to_datetime(df["scraped_at"], errors="coerce", utc=True)
    df["bucket"] = df["scraped_at"].dt.to_period(freq).dt.start_time
    df["generation"] = df.get("generation", "").fillna("").astype(str)

    # Active observations: each scrape = one observation. We take the
    # median of all observed prices in the bucket per segment.
    active_df = df[df.get("is_active", True).astype(bool)] if "is_active" in df.columns else df
    if not active_df.empty:
        active_agg = (
            active_df.groupby(["bucket", "brand", "model", "generation"])
            .agg(value=("price_eur", "median"), n=("price_eur", "size"))
            .reset_index()
        )
        active_agg["series"] = "active_ask_median"
    else:
        active_agg = pd.DataFrame(
            columns=["bucket", "brand", "model", "generation",
                     "value", "n", "series"]
        )

    # Sold observations: bucket by the deactivation week, value = the
    # last ASK we saw (latest snapshot per listing). Computed against
    # snapshots so we don't need the listings DataFrame for the price.
    sold_agg = pd.DataFrame(
        columns=["bucket", "brand", "model", "generation",
                 "value", "n", "series"]
    )
    if sold_listings is not None and not sold_listings.empty:
        sold = sold_listings.copy()
        sold["generation"] = sold.get("generation", "").fillna("").astype(str)
        sold["deactivated_at"] = pd.to_datetime(
            sold.get("deactivated_at"), errors="coerce", utc=True,
        )
        sold = sold.dropna(subset=["deactivated_at", "price_eur"])
        sold = sold[sold["price_eur"] > 0]
        if not sold.empty:
            sold["bucket"] = sold["deactivated_at"].dt.to_period(freq).dt.start_time
            sold_agg = (
                sold.groupby(["bucket", "brand", "model", "generation"])
                .agg(value=("price_eur", "median"), n=("price_eur", "size"))
                .reset_index()
            )
            sold_agg["series"] = "sold_lastask_median"

    return pd.concat([active_agg, sold_agg], ignore_index=True)


def composite_resale_score(metrics: pd.DataFrame) -> pd.Series:
    """Single-number ranking for "interesting segments to focus on".

    Combines four dimensions on a 0-1 scaled basis, then sums:
      - undervaluation_score: avg_undervaluation_pct, clipped to [0, 30]
      - liquidity_score:     n_sold_60d (log scale, capped)
      - velocity_score:      1 / median_dom (faster = better; floor 14d)
      - trend_score:         trend_30d_pct (rising = bonus, capped ±10)

    Rows with insufficient data on any dimension fall back to neutral
    (the dim contributes 0). The composite is intentionally simple —
    the goal is "at-a-glance ranking", not a tuned objective.
    """
    if metrics.empty:
        return pd.Series(dtype=float)

    uv = (metrics["avg_undervaluation_pct"].fillna(0).clip(lower=0, upper=30)) / 30
    liq = (np.log1p(metrics["n_sold_60d"].fillna(0))) / np.log1p(50)
    dom = metrics["median_dom"].fillna(60)
    velocity = (14 / dom.clip(lower=14)).clip(upper=1.0)
    trend = (metrics["trend_30d_pct"].fillna(0).clip(lower=-10, upper=10) + 10) / 20

    # Weights: under-valuation matters most (it's the literal flip thesis),
    # liquidity / velocity matter second (can you actually exit), trend
    # is a tie-breaker.
    return (
        0.40 * uv
        + 0.25 * liq
        + 0.20 * velocity
        + 0.15 * trend
    ).rename("composite")
