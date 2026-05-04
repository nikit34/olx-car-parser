"""Resale-decision algorithm — pure function over signals + context.

Consumes already-computed dashboard artefacts (``signals_df`` rows,
``listings_df``, ``price_snapshots`` for trend, sold subset for DoM and
calibration) and emits a verdict per listing:

    BUY        — open the wallet
    WATCH      — set a price-drop alert
    SKIP       — margin too thin / market softening
    REJECT     — hard-block (accident, salvage, RHD, …) or model says
                 "fair price"
    NO_OPINION — model has no basis to score this segment

Algorithm follows the 15-step decision tree described in the planning
discussion: hard gates → model trust → economics → market direction →
liquidity → seller signals → final score. Each step appends to
``Decision.reasons`` so the UI can show *why* a verdict came out.

Designed as a thin layer: every input that this module reads is already
materialised by ``data_loader.compute_signals`` or by the existing
``get_price_snapshots_df`` / sold-subset queries. There is no new data
collection, no new model training.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd


# Public types --------------------------------------------------------------


VERDICT_BUY = "BUY"
VERDICT_WATCH = "WATCH"
VERDICT_SKIP = "SKIP"
VERDICT_REJECT = "REJECT"
VERDICT_NO_OPINION = "NO_OPINION"

VERDICT_ICON = {
    VERDICT_BUY: "🟢",
    VERDICT_WATCH: "🟡",
    VERDICT_SKIP: "⚫",
    VERDICT_REJECT: "⛔",
    VERDICT_NO_OPINION: "❔",
}


@dataclass
class DecisionContext:
    """Segment-level state pre-computed once per dashboard load.

    All maps are keyed on (brand, model, generation_or_None) with a
    tuple-(brand, model) fallback handled inside the lookup helpers
    below — that mirrors how the deal scorer in ``data_loader`` already
    treats the generation key.
    """

    dom_median: Mapping[tuple, float] = field(default_factory=dict)
    """Median days-on-market for sold listings in the segment."""

    dom_fast_share: Mapping[tuple, float] = field(default_factory=dict)
    """Share of sold listings closed in ≤21 days — proxy for liquidity."""

    trend_90d_pct: Mapping[tuple, float] = field(default_factory=dict)
    """Δ% of segment median ask between first and last tercile of a 90d window."""

    calibration_resid_pct: Mapping[tuple, float] = field(default_factory=dict)
    """Median (sold_ask − predicted) / predicted × 100 per segment."""

    coverage_80: float | None = None
    """Latest CQR 80%-band empirical coverage; <0.7 → distrust the band."""


@dataclass
class Decision:
    verdict: str
    score: float
    reasons: list[str]
    components: dict[str, float | None]


# Context construction ------------------------------------------------------


def _segkey(brand, model, generation) -> tuple:
    gen = generation if (generation is not None and pd.notna(generation) and str(generation) != "") else None
    return (brand, model, gen)


def _lookup_with_fallback(table: Mapping[tuple, float], brand, model, gen):
    """Try (b,m,gen) then (b,m,None). Returns NaN when no entry."""
    v = table.get(_segkey(brand, model, gen))
    if v is not None:
        return v
    v = table.get((brand, model, None))
    return v if v is not None else float("nan")


def build_context(
    listings_df: pd.DataFrame,
    snapshots_df: pd.DataFrame | None = None,
    *,
    coverage_80: float | None = None,
    predicted_lookup: Mapping[str, float] | None = None,
) -> DecisionContext:
    """Compute segment-level context shared across all listings on a page.

    Parameters
    ----------
    listings_df
        Same shape as ``data_loader.load_all()[0]`` — must have
        ``brand``, ``model``, ``generation``, ``is_active``,
        ``first_seen_at``, ``deactivated_at``, ``deactivation_reason``,
        ``price_eur``.
    snapshots_df
        Output of ``get_price_snapshots_df`` (joined with listing meta).
        If ``None`` or empty the trend map stays empty and the algorithm
        skips the trend step.
    coverage_80
        Latest CQR 80%-band coverage from ``load_metrics_history()[-1]``.
    predicted_lookup
        ``olx_id → predicted_price`` for sold listings — used by the
        calibration map. Pass the full GB predictions dict, not the
        signals subset (signals only covers actives surfaced as deals).
    """
    dom_median: dict[tuple, float] = {}
    dom_fast_share: dict[tuple, float] = {}
    trend_90d_pct: dict[tuple, float] = {}
    calibration_resid_pct: dict[tuple, float] = {}

    if listings_df is None or listings_df.empty:
        return DecisionContext(coverage_80=coverage_80)

    df = listings_df.copy()
    if "generation" not in df.columns:
        df["generation"] = pd.NA
    if "duplicate_of" in df.columns:
        df = df[df["duplicate_of"].isna()].copy()

    # --- Sold-side DoM stats (algorithm step 8 + 9). -----------------------
    is_active = df["is_active"].astype(bool) if "is_active" in df.columns else pd.Series(True, index=df.index)
    reason = df.get("deactivation_reason", pd.Series("", index=df.index)).astype(str)
    sold = df[~is_active & (reason == "sold")].copy()
    if not sold.empty and "first_seen_at" in sold.columns and "deactivated_at" in sold.columns:
        first = pd.to_datetime(sold["first_seen_at"], errors="coerce", utc=True)
        last = pd.to_datetime(sold["deactivated_at"], errors="coerce", utc=True)
        dom = ((last - first).dt.total_seconds() / 86400)
        sold["__dom"] = dom
        sold = sold[(sold["__dom"].notna()) & (sold["__dom"] >= 0) & (sold["__dom"] <= 365)]
        if not sold.empty:
            grouped = sold.groupby(["brand", "model", "generation"], dropna=False)
            for (b, m, g), grp in grouped:
                key = _segkey(b, m, g)
                dom_median[key] = float(grp["__dom"].median())
                dom_fast_share[key] = float((grp["__dom"] <= 21).mean())
            # Brand+model fallback: collapse generations.
            grouped_bm = sold.groupby(["brand", "model"], dropna=False)
            for (b, m), grp in grouped_bm:
                key = (b, m, None)
                if key not in dom_median:
                    dom_median[key] = float(grp["__dom"].median())
                    dom_fast_share[key] = float((grp["__dom"] <= 21).mean())

    # --- Calibration residuals (algorithm step 4). -------------------------
    if predicted_lookup and not sold.empty:
        sold = sold.copy()
        sold["__pred"] = sold["olx_id"].map(predicted_lookup)
        cal = sold[(sold["__pred"].notna()) & (sold["__pred"] > 0) & (sold["price_eur"] > 0)].copy()
        if not cal.empty:
            cal["__resid_pct"] = (cal["price_eur"] - cal["__pred"]) / cal["__pred"] * 100
            grouped = cal.groupby(["brand", "model", "generation"], dropna=False)
            for (b, m, g), grp in grouped:
                if len(grp) >= 3:
                    calibration_resid_pct[_segkey(b, m, g)] = float(grp["__resid_pct"].median())
            grouped_bm = cal.groupby(["brand", "model"], dropna=False)
            for (b, m), grp in grouped_bm:
                key = (b, m, None)
                if key not in calibration_resid_pct and len(grp) >= 5:
                    calibration_resid_pct[key] = float(grp["__resid_pct"].median())

    # --- 90d trend (algorithm step 7). -------------------------------------
    if snapshots_df is not None and not snapshots_df.empty:
        snap = snapshots_df.copy()
        if "duplicate_of" in snap.columns:
            snap = snap[snap["duplicate_of"].isna()]
        snap = snap[snap["price_eur"].fillna(0) > 0].copy()
        snap["scraped_at"] = pd.to_datetime(snap["scraped_at"], errors="coerce", utc=True)
        cutoff = snap["scraped_at"].max() - pd.Timedelta(days=90)
        snap = snap[snap["scraped_at"] >= cutoff]
        if not snap.empty:
            snap["__bucket"] = snap["scraped_at"].dt.to_period("W").dt.start_time
            per_listing = (
                snap.groupby(["brand", "model", "generation", "olx_id", "__bucket"])["price_eur"]
                .median().reset_index()
            )
            grouped = per_listing.groupby(["brand", "model", "generation"], dropna=False)
            for (b, m, g), grp in grouped:
                buckets = grp.groupby("__bucket")["price_eur"].median().sort_index()
                if len(buckets) < 2:
                    continue
                n = len(buckets)
                early = float(buckets.iloc[: max(1, n // 3)].median())
                late = float(buckets.iloc[-max(1, n // 3):].median())
                if early > 0:
                    trend_90d_pct[_segkey(b, m, g)] = round((late - early) / early * 100, 2)
            # Brand+model fallback.
            grouped_bm = per_listing.groupby(["brand", "model"], dropna=False)
            for (b, m), grp in grouped_bm:
                key = (b, m, None)
                if key in trend_90d_pct:
                    continue
                buckets = grp.groupby("__bucket")["price_eur"].median().sort_index()
                if len(buckets) < 2:
                    continue
                n = len(buckets)
                early = float(buckets.iloc[: max(1, n // 3)].median())
                late = float(buckets.iloc[-max(1, n // 3):].median())
                if early > 0:
                    trend_90d_pct[key] = round((late - early) / early * 100, 2)

    return DecisionContext(
        dom_median=dom_median,
        dom_fast_share=dom_fast_share,
        trend_90d_pct=trend_90d_pct,
        calibration_resid_pct=calibration_resid_pct,
        coverage_80=coverage_80,
    )


# Decision algorithm --------------------------------------------------------


# Tunables grouped at module scope so future calibration is one diff.
# Re-tuned 2026-05-03 against the production active set (the original
# BUY_SCORE=25 / SAMPLE_CONF_DIVISOR=20 shut everything to NO_OPINION
# because most PT segments carry 5-12 comparables, not 20+).
_BAND_TIGHT = 0.25      # ≤25%: confident
_BAND_WIDE = 0.40       # >40%: NO_OPINION
_MIN_SAMPLE = 5         # comparables needed before model is trusted
_SAMPLE_CONF_DIVISOR = 10  # min(sample / divisor, 1.0)
_MIN_NET_MARGIN_PCT = 12.0
_MIN_NET_MARGIN_PCT_SLOW = 18.0     # if median DoM > 60d
_MAX_SOFTENING_PCT = -3.0
_DOM_LIMIT_DAYS = 120                # > this: REJECT regardless of margin
_FAST_SHARE_OK = 0.30
# Resale-cost model (private resident, intra-PT flip — no IVA, no broker).
# Replaces the previous 3% + €200 stub which over-penalised expensive cars
# and missed the holding cost that dominates slow segments. Components:
#   flat (one-off)   — registo automóvel ~€65 + detailing ~€50 + transport
#                      / ads ~€35 ≈ €150
#   holding cost/day — seguro 3rd party (€25-40/mo) + IUC prorated
#                      (€5-20/mo, cat. B) ≈ €30-50/mo ≈ €1.30/day
# Hold time uses the segment's median DoM (capped at _DOM_LIMIT_DAYS so a
# slow-segment lookup before the DoM gate doesn't double-count). When DoM
# is unknown we fall back to a 45-day default (rough median of the active
# set's resolved DoM histogram).
_FEES_FLAT_EUR = 150.0
_HOLDING_COST_EUR_PER_DAY = 1.30
_DEFAULT_HOLD_DAYS = 45
_BUY_SCORE = 18.0
_WATCH_SCORE = 15.0
# Anomaly hard-gate (feature-space outlier from
# ``src.analytics.anomaly``). 0.90 keeps v2's contamination=0.05
# top-tail but only rejects on the very tip — plenty of legitimately
# rare expensive cars score 0.85 and we don't want to lock them out.
_ANOMALY_REJECT_SCORE = 0.90
# Per-listing hazard signal (P(sold within horizon) from
# ``src.analytics.hazard``). Tightens / loosens velocity_conf on top
# of the segment-level dom_median / fast_share that already feed into
# step 8. Thresholds are conservative: only the tails (top / bottom
# quintile of the v2 distribution: median 0.52, p25 0.35, p75 0.71)
# trigger an adjustment so we don't double-count signals that already
# correlate with segment liquidity.
_HAZARD_FAST_PROB = 0.70
_HAZARD_SLOW_PROB = 0.25


def _is_truthy(v) -> bool:
    return pd.notna(v) and bool(v)


def decide(
    listing: pd.Series | Mapping,
    ctx: DecisionContext,
) -> Decision:
    """Run the 15-step decision tree on one listing.

    ``listing`` is a row from ``signals_df`` (i.e. it has ``predicted_price``,
    ``fair_price_low``, ``fair_price_high``, ``sample_size``,
    ``band_pct``, ``repair_cost_eur``, plus the descriptive flags
    ``desc_mentions_accident`` etc). Hard-block guards work even if
    ``signals_df`` already filtered most of those out — the row may
    still carry sev-2 / repair flags that affect economics.
    """
    g = listing.get if hasattr(listing, "get") else lambda k, d=None: listing[k] if k in listing else d
    reasons: list[str] = []
    components: dict[str, float | None] = {}

    brand = g("brand")
    model = g("model")
    gen = g("generation") or None
    price = float(g("price_eur") or 0)
    predicted = g("predicted_price")
    predicted = float(predicted) if predicted is not None and pd.notna(predicted) else None
    fair_low = g("fair_price_low")
    fair_low = float(fair_low) if fair_low is not None and pd.notna(fair_low) else None
    fair_high = g("fair_price_high")
    fair_high = float(fair_high) if fair_high is not None and pd.notna(fair_high) else None
    sample = int(g("sample_size") or 0)
    band_pct_value = g("band_pct")  # 0–100, *not* fraction. None when bundle missing.
    band_frac = float(band_pct_value) / 100 if band_pct_value is not None and pd.notna(band_pct_value) else None
    repair_cost = float(g("repair_cost_eur") or 0)

    # ---- Step 1: hard gates. signals_df already drops most of these,
    # but we re-check so the function works on raw listings_df rows too.
    accident = _is_truthy(g("desc_mentions_accident"))
    rhd = _is_truthy(g("right_hand_drive"))
    sev_raw = g("damage_severity")
    try:
        sev = int(sev_raw) if pd.notna(sev_raw) else None
    except (TypeError, ValueError):
        sev = None

    if accident:
        reasons.append("description mentions accident")
        return Decision(VERDICT_REJECT, 0.0, reasons, components)
    if sev is not None and sev >= 3:
        reasons.append(f"damage severity {sev} (salvage / parts-only)")
        return Decision(VERDICT_REJECT, 0.0, reasons, components)
    if rhd:
        reasons.append("right-hand drive — PT market mismatch")
        return Decision(VERDICT_REJECT, 0.0, reasons, components)

    # ---- Step 1b: feature-space anomaly gate (from src.analytics.anomaly).
    # Catches parser artefacts (engine_cc=10000, mileage 9_999_999) and
    # unflagged salvage that the explicit gates above miss. Only the
    # very tip of the contamination tail (≥0.90) blocks; legitimately
    # rare expensive cars sit at 0.80–0.88 and pass through.
    anomaly_score = g("anomaly_score")
    if anomaly_score is not None and pd.notna(anomaly_score):
        anomaly_score = float(anomaly_score)
        components["anomaly_score"] = round(anomaly_score, 2)
        if anomaly_score >= _ANOMALY_REJECT_SCORE:
            reasons.append(
                f"feature-space outlier (anomaly_score={anomaly_score:.2f}) "
                "— likely parser artefact / undocumented salvage"
            )
            return Decision(VERDICT_REJECT, 0.0, reasons, components)

    # ---- Step 2: model coverage.
    if predicted is None or predicted <= 0:
        reasons.append("model has no prediction for this listing")
        return Decision(VERDICT_NO_OPINION, 0.0, reasons, components)
    if sample < _MIN_SAMPLE:
        reasons.append(f"only {sample} comparables (need ≥{_MIN_SAMPLE})")
        return Decision(VERDICT_NO_OPINION, 0.0, reasons, components)

    # ---- Step 3: band confidence.
    band_conf = 1.0
    if band_frac is not None:
        components["band_pct"] = band_frac * 100
        if band_frac > _BAND_WIDE:
            reasons.append(f"model band ±{band_frac * 50:.0f}% — too uncertain")
            return Decision(VERDICT_NO_OPINION, 0.0, reasons, components)
        if band_frac > _BAND_TIGHT:
            band_conf = 0.7
            reasons.append(f"wide band (±{band_frac * 50:.0f}%) — lowering confidence")
        else:
            band_conf = 1.15
            reasons.append(f"tight band (±{band_frac * 50:.0f}%)")

    # ---- Step 4: calibration correction on predicted price.
    seg_resid_pct = _lookup_with_fallback(ctx.calibration_resid_pct, brand, model, gen)
    calibration_conf = 1.0
    predicted_corrected = predicted
    if not pd.isna(seg_resid_pct):
        components["calibration_resid_pct"] = seg_resid_pct
        predicted_corrected = predicted * (1 + seg_resid_pct / 100)
        if abs(seg_resid_pct) > 5:
            calibration_conf = 0.85
            direction = "under" if seg_resid_pct > 0 else "over"
            reasons.append(
                f"sold-side calibration: model {direction}-predicts "
                f"this segment by {seg_resid_pct:+.1f}% — corrected"
            )
        if predicted_corrected <= price:
            reasons.append("after calibration, ask is at/above fair value")
            return Decision(VERDICT_REJECT, 0.0, reasons, components)

    components["predicted_corrected"] = round(predicted_corrected, 0)

    # ---- Step 5: net margin after repair + fees.
    # Look up DoM here (the gate at step 8 reuses it) so the holding-cost
    # component of fees is segment-aware rather than a flat stub.
    dom_med = _lookup_with_fallback(ctx.dom_median, brand, model, gen)
    fast_share = _lookup_with_fallback(ctx.dom_fast_share, brand, model, gen)
    expected_hold_days = float(dom_med) if not pd.isna(dom_med) else float(_DEFAULT_HOLD_DAYS)
    expected_hold_days = min(expected_hold_days, float(_DOM_LIMIT_DAYS))
    fees = _FEES_FLAT_EUR + _HOLDING_COST_EUR_PER_DAY * expected_hold_days
    raw_margin = predicted_corrected - price
    net_margin = raw_margin - repair_cost - fees
    net_margin_pct = (net_margin / predicted_corrected) * 100 if predicted_corrected else 0.0
    components["net_margin_eur"] = round(net_margin, 0)
    components["net_margin_pct"] = round(net_margin_pct, 1)
    components["repair_cost_eur"] = round(repair_cost, 0)
    components["fees_eur"] = round(fees, 0)

    if net_margin_pct < _MIN_NET_MARGIN_PCT:
        reasons.append(
            f"net margin {net_margin_pct:.1f}% below {_MIN_NET_MARGIN_PCT:.0f}% floor"
        )
        return Decision(VERDICT_SKIP, max(0.0, net_margin_pct), reasons, components)

    # ---- Step 6: position vs P10/P50.
    if fair_low is not None and price <= fair_low:
        reasons.append(f"ask €{price:,.0f} ≤ P10 €{fair_low:,.0f} — outlier-buy zone")
        band_conf *= 1.15
    elif predicted is not None and price > predicted:
        reasons.append("ask above P50 even before correction")
        return Decision(VERDICT_REJECT, 0.0, reasons, components)
    else:
        reasons.append("ask between P10 and P50 — fair value zone")

    # ---- Step 7: market direction.
    trend_pct = _lookup_with_fallback(ctx.trend_90d_pct, brand, model, gen)
    components["trend_90d_pct"] = None if pd.isna(trend_pct) else round(trend_pct, 2)
    expected_drop_pct = 0.0
    if not pd.isna(trend_pct):
        if trend_pct < _MAX_SOFTENING_PCT:
            # Market falling fast: only proceed if margin is >2× expected drop.
            expected_drop_pct = abs(trend_pct)
            if net_margin_pct < 2 * expected_drop_pct:
                reasons.append(
                    f"segment softening {trend_pct:+.1f}%/90d, net margin "
                    f"{net_margin_pct:.1f}% < 2× drop"
                )
                return Decision(VERDICT_SKIP, net_margin_pct, reasons, components)
            reasons.append(f"segment softening {trend_pct:+.1f}%/90d but margin still covers 2× drop")
        elif trend_pct < 0:
            expected_drop_pct = abs(trend_pct)
            reasons.append(f"segment slightly soft {trend_pct:+.1f}%/90d — discounted")
        else:
            reasons.append(f"segment firming {trend_pct:+.1f}%/90d")

    # ---- Step 8 + 9: liquidity gates. (dom_med / fast_share were looked
    # up earlier in step 5 so the fees holding-cost can use them.)
    components["dom_median"] = None if pd.isna(dom_med) else round(dom_med, 0)
    components["dom_fast_share"] = None if pd.isna(fast_share) else round(fast_share, 2)

    velocity_conf = 1.0
    if not pd.isna(dom_med):
        if dom_med > _DOM_LIMIT_DAYS:
            reasons.append(f"median DoM {dom_med:.0f}d > {_DOM_LIMIT_DAYS}d — capital trap")
            return Decision(VERDICT_SKIP, net_margin_pct, reasons, components)
        if dom_med > 60:
            if net_margin_pct < _MIN_NET_MARGIN_PCT_SLOW:
                reasons.append(
                    f"slow segment (DoM {dom_med:.0f}d) → margin floor "
                    f"raised to {_MIN_NET_MARGIN_PCT_SLOW:.0f}%"
                )
                return Decision(VERDICT_SKIP, net_margin_pct, reasons, components)
            reasons.append(f"slow segment (DoM {dom_med:.0f}d), margin still clears")
            velocity_conf = 0.85
        elif dom_med <= 30:
            velocity_conf = 1.1
            reasons.append(f"fast segment (DoM {dom_med:.0f}d)")

    if not pd.isna(fast_share) and fast_share < _FAST_SHARE_OK:
        velocity_conf *= 0.9
        reasons.append(f"only {fast_share:.0%} of sold cleared in ≤21d")

    # ---- Step 9b: per-listing hazard signal (from src.analytics.hazard).
    # Refines the segment-level dom_median / fast_share above with a
    # per-listing P(sold ≤ horizon) — same Golf in Porto can score 0.45
    # at €18k and 0.92 at €11k because price-vs-fair-value is the
    # dominant feature. Conservative thresholds: only the tails of the
    # v2 production distribution (median 0.52) trigger a multiplier so
    # we don't double-count signals that already correlate with the
    # segment-level inputs.
    prob_sold = g("prob_sold_within_horizon")
    if prob_sold is not None and pd.notna(prob_sold):
        prob_sold = float(prob_sold)
        components["prob_sold_within_horizon"] = round(prob_sold, 2)
        if prob_sold >= _HAZARD_FAST_PROB:
            velocity_conf *= 1.10
            reasons.append(
                f"per-listing P(sold≤30d)={prob_sold:.2f} — fast (hazard)"
            )
        elif prob_sold <= _HAZARD_SLOW_PROB:
            velocity_conf *= 0.85
            reasons.append(
                f"per-listing P(sold≤30d)={prob_sold:.2f} — slow (hazard)"
            )

    # ---- Step 10–13: seller signals (deal-quality multipliers).
    motivated_conf = 1.0
    days_listed = g("days_listed")
    price_change = g("price_change_eur")
    if pd.notna(days_listed) and days_listed and float(days_listed) > 30 and \
       pd.notna(price_change) and float(price_change) < 0:
        motivated_conf *= 1.15
        reasons.append(f"motivated seller: {int(days_listed)}d listed, dropped €{abs(float(price_change)):,.0f}")
    elif pd.notna(days_listed) and days_listed and float(days_listed) > 60:
        motivated_conf *= 1.05
        reasons.append(f"long-listed ({int(days_listed)}d) — negotiation room likely")

    urgency = g("urgency")
    if isinstance(urgency, str):
        if urgency == "high":
            motivated_conf *= 1.15
            reasons.append("seller urgency: high")
        elif urgency == "medium":
            motivated_conf *= 1.05
            reasons.append("seller urgency: medium")

    exit_penalty = 1.0
    if _is_truthy(g("warranty")):
        exit_penalty *= 1.05
        reasons.append("warranty present — easier resale")
    if _is_truthy(g("first_owner_selling")):
        exit_penalty *= 1.05
        reasons.append("first-owner sale — easier resale")
    if _is_truthy(g("taxi_fleet_rental")):
        exit_penalty *= 0.92
        reasons.append("taxi/rental history — exit penalty")
    n_owners = g("desc_mentions_num_owners")
    if pd.notna(n_owners) and n_owners and int(n_owners) >= 3:
        exit_penalty *= 0.95
        reasons.append(f"{int(n_owners)} prior owners — exit penalty")

    # ---- Step 14: expected profit + score.
    expected_drop_eur = price * (expected_drop_pct / 100)
    expected_profit = (net_margin - expected_drop_eur) * exit_penalty
    expected_profit_pct = (expected_profit / predicted_corrected) * 100 if predicted_corrected else 0.0

    sample_conf = min(sample / _SAMPLE_CONF_DIVISOR, 1.0)
    confidence = band_conf * sample_conf * calibration_conf * motivated_conf * velocity_conf
    score = expected_profit_pct * confidence

    components["expected_profit_eur"] = round(expected_profit, 0)
    components["expected_profit_pct"] = round(expected_profit_pct, 1)
    components["confidence"] = round(confidence, 2)
    components["score"] = round(score, 1)

    # ---- Step 15: verdict bucket.
    if score >= _BUY_SCORE:
        verdict = VERDICT_BUY
    elif score >= _WATCH_SCORE:
        verdict = VERDICT_WATCH
    else:
        verdict = VERDICT_SKIP

    return Decision(verdict, round(score, 1), reasons, components)


def decide_many(
    signals_df: pd.DataFrame,
    ctx: DecisionContext,
) -> pd.DataFrame:
    """Vectorised wrapper — returns a frame aligned to ``signals_df.index``
    with verdict / score / top-3 reasons columns. Internally it still
    loops; profile shows ~3 ms per row on the production active set,
    which is negligible compared to the surrounding chart renders."""
    if signals_df is None or signals_df.empty:
        return pd.DataFrame(columns=["olx_id", "verdict", "score", "reasons"])
    out = []
    for _, row in signals_df.iterrows():
        d = decide(row, ctx)
        out.append({
            "olx_id": row.get("olx_id"),
            "verdict": d.verdict,
            "score": d.score,
            "reasons": d.reasons,
            "components": d.components,
        })
    return pd.DataFrame(out)
