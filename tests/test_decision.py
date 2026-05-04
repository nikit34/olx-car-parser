"""Tests for the resale-decision algorithm."""

import pandas as pd
import pytest

from src.analytics.decision import (
    DecisionContext,
    decide,
    decide_many,
    build_context,
    VERDICT_BUY,
    VERDICT_WATCH,
    VERDICT_SKIP,
    VERDICT_REJECT,
    VERDICT_NO_OPINION,
)


def _row(**kw) -> pd.Series:
    """Baseline ‘healthy’ signal row — overridden per test."""
    base = {
        "olx_id": "x1",
        "brand": "Volkswagen",
        "model": "Golf",
        "generation": "Mk7",
        "price_eur": 10000.0,
        "predicted_price": 13000.0,
        "fair_price_low": 11400.0,
        "fair_price_high": 14600.0,
        "sample_size": 12,
        "band_pct": 24.0,                 # band_frac = 0.24, "tight"
        "repair_cost_eur": 0,
        "desc_mentions_accident": False,
        "right_hand_drive": False,
        "damage_severity": 0,
        "days_listed": 10,
        "price_change_eur": 0,
        "urgency": None,
        "warranty": False,
        "first_owner_selling": False,
        "taxi_fleet_rental": False,
        "desc_mentions_num_owners": None,
    }
    base.update(kw)
    return pd.Series(base)


def _ctx(**kw) -> DecisionContext:
    """Healthy segment context — fast-selling, slightly firming, calibrated."""
    defaults = {
        "dom_median": {("Volkswagen", "Golf", "Mk7"): 25.0},
        "dom_fast_share": {("Volkswagen", "Golf", "Mk7"): 0.55},
        "trend_90d_pct": {("Volkswagen", "Golf", "Mk7"): 1.0},
        "calibration_resid_pct": {("Volkswagen", "Golf", "Mk7"): 0.5},
        "coverage_80": 0.81,
    }
    defaults.update(kw)
    return DecisionContext(**defaults)


# ---- Step 1 hard gates ----------------------------------------------------


def test_reject_on_accident():
    d = decide(_row(desc_mentions_accident=True), _ctx())
    assert d.verdict == VERDICT_REJECT
    assert any("accident" in r for r in d.reasons)


def test_reject_on_severity_3():
    d = decide(_row(damage_severity=3), _ctx())
    assert d.verdict == VERDICT_REJECT


def test_reject_on_rhd():
    d = decide(_row(right_hand_drive=True), _ctx())
    assert d.verdict == VERDICT_REJECT


# ---- Step 2 model trust ---------------------------------------------------


def test_no_opinion_on_low_sample():
    d = decide(_row(sample_size=3), _ctx())
    assert d.verdict == VERDICT_NO_OPINION
    assert any("comparables" in r for r in d.reasons)


def test_no_opinion_on_missing_prediction():
    d = decide(_row(predicted_price=None), _ctx())
    assert d.verdict == VERDICT_NO_OPINION


# ---- Step 3 band confidence -----------------------------------------------


def test_no_opinion_on_wide_band():
    d = decide(_row(band_pct=45.0), _ctx())
    assert d.verdict == VERDICT_NO_OPINION
    assert any("band" in r for r in d.reasons)


# ---- Step 5 economics -----------------------------------------------------


def test_skip_on_thin_margin():
    # Predicted barely above price → margin under floor.
    d = decide(_row(price_eur=12500, predicted_price=13000), _ctx())
    assert d.verdict == VERDICT_SKIP
    assert any("margin" in r.lower() for r in d.reasons)


def test_reject_when_ask_above_predicted():
    d = decide(_row(price_eur=14000, predicted_price=13000), _ctx())
    # Calibration is +0.5%, still puts predicted_corrected ≈ 13065 < price.
    assert d.verdict == VERDICT_REJECT


# ---- Step 7 market direction ----------------------------------------------


def test_skip_on_softening_market_thin_buffer():
    # Net margin ~20%, market falling 12%/90d → margin < 2× drop (24%)
    ctx = _ctx(trend_90d_pct={("Volkswagen", "Golf", "Mk7"): -12.0})
    d = decide(_row(price_eur=10500, predicted_price=13000), ctx)
    assert d.verdict == VERDICT_SKIP
    assert any("softening" in r for r in d.reasons)


def test_buy_in_firming_market():
    d = decide(_row(price_eur=8500, predicted_price=13000), _ctx())
    assert d.verdict == VERDICT_BUY
    # Sanity: reasons mention either undervaluation or fair-value zone.
    assert d.score > 0


# ---- Step 8 liquidity -----------------------------------------------------


def test_skip_on_capital_trap_dom():
    ctx = _ctx(dom_median={("Volkswagen", "Golf", "Mk7"): 200.0})
    d = decide(_row(price_eur=8500, predicted_price=13000), ctx)
    assert d.verdict == VERDICT_SKIP
    assert any("capital trap" in r for r in d.reasons)


def test_slow_segment_raises_margin_floor():
    # ~15% margin clears the 12% fast-segment floor but fails the 18%
    # slow-segment floor.
    ctx = _ctx(dom_median={("Volkswagen", "Golf", "Mk7"): 80.0})
    d = decide(_row(price_eur=10500, predicted_price=13000), ctx)
    assert d.verdict == VERDICT_SKIP
    assert any("margin floor" in r for r in d.reasons)


# ---- Calibration correction -----------------------------------------------


def test_calibration_overprediction_pulls_predicted_down():
    # Segment over-predicts by 20% → predicted_corrected = 13000 * 0.80 = 10400.
    # Ask 10000 → margin ~3% → SKIP (under 12% floor).
    ctx = _ctx(calibration_resid_pct={("Volkswagen", "Golf", "Mk7"): -20.0})
    d = decide(_row(price_eur=10000, predicted_price=13000), ctx)
    assert d.verdict == VERDICT_SKIP
    assert any("over-predicts" in r for r in d.reasons)


# ---- Watch bucket ---------------------------------------------------------


def test_watch_on_moderate_margin():
    # ~13% margin × partial confidence (sample 8 → sample_conf=0.8) puts
    # the score in the WATCH band [15, 18) under the calibrated tunables.
    d = decide(_row(price_eur=10800, predicted_price=13000, sample_size=8), _ctx())
    assert d.verdict == VERDICT_WATCH
    assert 15 <= d.score < 18


# ---- decide_many wrapper --------------------------------------------------


def test_decide_many_returns_aligned_frame():
    df = pd.DataFrame([_row(olx_id="a"), _row(olx_id="b", desc_mentions_accident=True)])
    out = decide_many(df, _ctx())
    assert len(out) == 2
    assert set(out.columns) >= {"olx_id", "verdict", "score", "reasons"}
    by_id = out.set_index("olx_id")
    assert by_id.loc["b", "verdict"] == VERDICT_REJECT


# ---- build_context smoke --------------------------------------------------


def test_build_context_handles_empty_inputs():
    ctx = build_context(pd.DataFrame(), pd.DataFrame())
    assert ctx.dom_median == {}
    assert ctx.trend_90d_pct == {}
    assert ctx.calibration_resid_pct == {}


def test_anomaly_score_above_threshold_rejects():
    """anomaly_score ≥ 0.90 → REJECT before model coverage check."""
    d = decide(_row(anomaly_score=0.95), _ctx())
    assert d.verdict == VERDICT_REJECT
    assert any("feature-space outlier" in r for r in d.reasons)


def test_anomaly_score_below_threshold_passes_through():
    """A high-but-sub-threshold anomaly_score (rare expensive car) shouldn't
    block — the rest of the decision tree decides."""
    d = decide(_row(anomaly_score=0.85), _ctx())
    assert d.verdict != VERDICT_REJECT
    # The score is recorded so the UI can surface "rare configuration" warnings.
    assert d.components.get("anomaly_score") == pytest.approx(0.85, abs=0.01)


def test_anomaly_score_missing_does_not_break():
    """Bundle missing → anomaly_score=None → gate is a no-op."""
    d = decide(_row(anomaly_score=None), _ctx())
    assert "anomaly_score" not in d.components


def test_hazard_fast_listing_boosts_velocity():
    """Per-listing P(sold within 30d) ≥ 0.70 should bump velocity_conf,
    raising the final score. Compare to a baseline same-row without
    the hazard signal."""
    base = decide(_row(prob_sold_within_horizon=None), _ctx())
    fast = decide(_row(prob_sold_within_horizon=0.85), _ctx())
    # Both should reach the score-bucket step; fast should outscore base.
    assert fast.score >= base.score
    assert any("fast (hazard)" in r for r in fast.reasons)
    assert fast.components.get("prob_sold_within_horizon") == pytest.approx(0.85, abs=0.01)


def test_hazard_slow_listing_dampens_velocity():
    """P(sold within 30d) ≤ 0.25 should reduce velocity_conf."""
    base = decide(_row(prob_sold_within_horizon=None), _ctx())
    slow = decide(_row(prob_sold_within_horizon=0.15), _ctx())
    assert slow.score <= base.score
    assert any("slow (hazard)" in r for r in slow.reasons)


def test_hazard_mid_listing_no_change():
    """A mid-distribution probability (0.50) shouldn't trigger
    either tail — velocity_conf stays as the segment-level signal
    set it."""
    d = decide(_row(prob_sold_within_horizon=0.50), _ctx())
    # Component still recorded for transparency.
    assert d.components.get("prob_sold_within_horizon") == pytest.approx(0.50, abs=0.01)
    # No fast/slow reason added.
    assert not any("(hazard)" in r for r in d.reasons)


def test_build_context_extracts_dom_from_sold():
    listings = pd.DataFrame([
        {
            "olx_id": "s1", "brand": "Volkswagen", "model": "Golf",
            "generation": "Mk7", "is_active": False,
            "deactivation_reason": "sold",
            "first_seen_at": "2026-04-01T00:00:00Z",
            "deactivated_at": "2026-04-15T00:00:00Z",  # 14d
            "price_eur": 12000.0,
        },
        {
            "olx_id": "s2", "brand": "Volkswagen", "model": "Golf",
            "generation": "Mk7", "is_active": False,
            "deactivation_reason": "sold",
            "first_seen_at": "2026-04-01T00:00:00Z",
            "deactivated_at": "2026-05-10T00:00:00Z",  # 39d
            "price_eur": 13000.0,
        },
    ])
    ctx = build_context(listings, pd.DataFrame())
    key = ("Volkswagen", "Golf", "Mk7")
    assert key in ctx.dom_median
    # Median of {14, 39} = 26.5
    assert 25 <= ctx.dom_median[key] <= 28
    # 1 of 2 sold within 21d → 0.5
    assert ctx.dom_fast_share[key] == pytest.approx(0.5)
