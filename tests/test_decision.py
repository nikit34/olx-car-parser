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


# ---- Missing-odometer penalty ---------------------------------------------


def test_mileage_missing_pushes_band_into_no_opinion():
    """Reproduction of the 2026-06-08 Mégane case: a would-be BUY (ask far
    under an inflated prediction, band just under the WIDE gate) loses its
    verdict once the odometer is flagged missing — the band is widened past
    the gate."""
    healthy = decide(
        _row(price_eur=8300, predicted_price=17457,
             fair_price_low=10950, fair_price_high=17789, band_pct=39.2),
        _ctx(calibration_resid_pct={}),
    )
    assert healthy.verdict == VERDICT_BUY  # without the flag it's a (phantom) BUY

    missing = decide(
        _row(price_eur=8300, predicted_price=17457,
             fair_price_low=10950, fair_price_high=17789, band_pct=39.2,
             mileage_missing=True),
        _ctx(calibration_resid_pct={}),
    )
    assert missing.verdict == VERDICT_NO_OPINION
    assert any("odometer" in r for r in missing.reasons)


def test_mileage_missing_inferred_from_nan_mileage_km():
    """A present-but-NaN mileage_km is treated the same as the explicit flag."""
    d = decide(
        _row(price_eur=8300, predicted_price=17457,
             fair_price_low=10950, fair_price_high=17789, band_pct=39.2,
             mileage_km=float("nan")),
        _ctx(calibration_resid_pct={}),
    )
    assert d.verdict == VERDICT_NO_OPINION


def test_present_mileage_is_not_penalised():
    """A healthy BUY with mileage present stays BUY — no penalty leaks in."""
    d = decide(_row(price_eur=8500, predicted_price=13000, mileage_km=90000), _ctx())
    assert d.verdict == VERDICT_BUY
    assert "mileage_missing" not in d.components


def test_mileage_far_past_segment_p90_downgrades_phantom_buy():
    """Odometer extrapolation guard: a car whose mileage sits far past its
    segment's p90 is being valued against lower-mileage comps (the 205k-km
    Smart 0.8 CDI case) — it must not stay a BUY."""
    ctx = _ctx(seg_mileage_p90={("Volkswagen", "Golf", "Mk7"): 120000.0})
    # In-distribution mileage (≤ p90): unaffected, still BUY.
    ok = decide(_row(price_eur=8500, predicted_price=13000, mileage_km=110000), ctx)
    assert ok.verdict == VERDICT_BUY
    assert "seg_mileage_p90" not in ok.components
    # Mileage ≫ p90 × factor: extrapolation → no longer BUY, reason explains why.
    ood = decide(_row(price_eur=8500, predicted_price=13000, mileage_km=210000), ctx)
    assert ood.verdict != VERDICT_BUY
    assert ood.components.get("seg_mileage_p90") == 120000
    assert any("p90" in r or "lower-mileage" in r for r in ood.reasons)


def test_mileage_ood_guard_inert_without_segment_p90():
    """No segment p90 (small/unknown segment) → guard can't fire, BUY stands."""
    d = decide(_row(price_eur=8500, predicted_price=13000, mileage_km=210000), _ctx())
    assert d.verdict == VERDICT_BUY


def test_mileage_missing_confidence_cut_without_band():
    """When no band shipped the widening is a no-op, so the confidence
    multiplier must still bite (penalty can't be bypassed by a missing band)."""
    base = decide(_row(price_eur=8500, predicted_price=13000, band_pct=None), _ctx())
    missing = decide(
        _row(price_eur=8500, predicted_price=13000, band_pct=None,
             mileage_missing=True),
        _ctx(),
    )
    assert missing.components["confidence"] < base.components["confidence"]
    assert missing.score < base.score


def test_low_spec_fill_triggers_penalty_even_with_mileage_present():
    """A stripped listing (only 1 of 4 specs) is abstained on even when the
    odometer itself is present — the spec-fill trigger, not just mileage."""
    d = decide(
        _row(price_eur=8300, predicted_price=17457,
             fair_price_low=10950, fair_price_high=17789, band_pct=39.2,
             mileage_km=120000, spec_fill=0.25),
        _ctx(calibration_resid_pct={}),
    )
    assert d.verdict == VERDICT_NO_OPINION
    assert any("specs" in r for r in d.reasons)
    assert d.components["spec_fill"] == 0.25


def test_adequate_spec_fill_is_not_penalised():
    """spec_fill at/above the threshold leaves a healthy BUY intact."""
    d = decide(
        _row(price_eur=8500, predicted_price=13000, mileage_km=90000, spec_fill=0.75),
        _ctx(),
    )
    assert d.verdict == VERDICT_BUY
    assert "spec_fill" not in d.components  # only recorded when it triggers


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


@pytest.mark.parametrize("missing_repair", [None, float("nan")])
def test_score_not_poisoned_by_missing_repair_cost(missing_repair):
    """Regression: `float(v or 0)` returned NaN when v was numpy.NaN
    because nan is truthy. NaN propagated into net_margin → score and
    forced every otherwise-BUY row into SKIP. signals_df gets NaN here
    in production whenever any row has a real cost (pandas float64
    promotion of the column). Both None and NaN must coerce to 0."""
    d = decide(_row(price_eur=8500, predicted_price=13000, repair_cost_eur=missing_repair), _ctx())
    assert pd.notna(d.score), f"score was NaN for repair_cost_eur={missing_repair!r}"
    assert d.verdict == VERDICT_BUY


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
    # ~14% margin × partial confidence (sample 8 → sample_conf=0.8) puts
    # the score in the WATCH band [15, 18) under the calibrated tunables.
    d = decide(_row(price_eur=11000, predicted_price=13000, sample_size=8), _ctx())
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


# ---- Cheap-tail value-trust guard + condition NLP (2026-06-25 audit) -------

import src.analytics.decision as _decision_mod  # noqa: E402


def _cheap_ctx() -> DecisionContext:
    """Neutral ctx — empty segment maps so the cheap guard is isolated."""
    return DecisionContext(coverage_80=0.81)


def _cheap_row(**kw) -> pd.Series:
    base = {
        "brand": "Citroen", "model": "C3", "generation": None,
        "price_eur": 2800.0, "predicted_price": 3517.0,
        "fair_price_low": 2000.0, "fair_price_high": 4000.0,
        "sample_size": 12, "band_pct": 20.0, "damage_severity": 0,
        "desc_mentions_accident": False, "right_hand_drive": False,
        "days_listed": 10, "price_change_eur": 0,
        "title": "Citroen C3", "description": "carro de familia, bom estado",
    }
    base.update(kw)
    return _row(**base)


def test_cheap_blend_suppresses_marginal_phantom(monkeypatch):
    # ask 2800 / model 3517 (1.26x) clears the margin floor WITHOUT the guard.
    monkeypatch.setattr(_decision_mod, "_CHEAP_PRED_W", 1.0)
    monkeypatch.setattr(_decision_mod, "_CHEAP_DIVERGENCE_CAP", 1e9)
    off = decide(_cheap_row(), _cheap_ctx())
    assert off.verdict in (VERDICT_BUY, VERDICT_WATCH)
    # With the guard at the shipped weight (0.30 model / 0.70 ask) the blended
    # value collapses the margin below the 12% floor -> SKIP.
    monkeypatch.setattr(_decision_mod, "_CHEAP_PRED_W", 0.30)
    monkeypatch.setattr(_decision_mod, "_CHEAP_DIVERGENCE_CAP", 2.0)
    on = decide(_cheap_row(), _cheap_ctx())
    assert on.verdict == VERDICT_SKIP
    assert any("cheap tier" in r and "blended" in r for r in on.reasons)


def test_cheap_divergence_cap_abstains():
    # C180-class: ask 875 / model 11131 = 12.7x ask -> untrustworthy -> abstain.
    d = decide(_cheap_row(price_eur=875.0, predicted_price=11131.0,
                          fair_price_low=6000.0, fair_price_high=14000.0,
                          band_pct=30.0), _cheap_ctx())
    assert d.verdict == VERDICT_NO_OPINION
    assert any("ask" in r and "implausible" in r for r in d.reasons)


def test_cheap_guard_not_applied_above_tier():
    # >= 4000 ask: the healthy mid-market baseline must be untouched.
    d = decide(_row(), _ctx())
    assert d.verdict in (VERDICT_BUY, VERDICT_WATCH)
    assert not any("cheap tier" in r for r in d.reasons)


def test_condition_fault_cost_reduces_margin(monkeypatch):
    # Disable the blend so we isolate the fault-cost effect; a disclosed
    # check-engine fault subtracts a repair provision -> reason recorded.
    monkeypatch.setattr(_decision_mod, "_CHEAP_PRED_W", 1.0)
    monkeypatch.setattr(_decision_mod, "_CHEAP_DIVERGENCE_CAP", 1e9)
    clean = decide(_cheap_row(description="bom estado, sempre na marca"), _cheap_ctx())
    faulty = decide(_cheap_row(description="luz da injeção acesa, catalisador a precisar"),
                    _cheap_ctx())
    assert faulty.components.get("condition_fault_cost_eur", 0) > 0
    assert any("disclosed fault" in r for r in faulty.reasons)
    # the fault provision lowers the score vs the clean twin
    assert faulty.score <= clean.score


# ---------------------------------------------------------------------------
# Photo damage: a ranking weight, not a veto (2026-08-24)
# ---------------------------------------------------------------------------

class TestPhotoDamageWeight:
    """The photo classifier orders well (ROC-AUC 0.74 held out) and classifies
    badly (precision 0.20 in production). So it may move a listing down the
    feed and must never remove it."""

    def _score(self, p):
        return decide(_row(photo_damage_p=p), _ctx()).score

    def test_high_photo_damage_scores_below_clean(self):
        assert self._score(0.95) < self._score(0.0)

    def test_penalty_is_monotone_in_the_score(self):
        scores = [self._score(p) for p in (0.0, 0.4, 0.7, 1.0)]
        assert scores == sorted(scores, reverse=True)

    def test_low_scores_do_nothing(self):
        """Below the floor the classifier's output is mostly reflections."""
        assert self._score(0.29) == self._score(0.0)

    def test_worst_case_is_milder_than_a_rental_history(self):
        """A signal right 3 times in 4 must not outweigh a fact we are sure of."""
        from src.analytics.decision import _PHOTO_DAMAGE_MAX_PENALTY
        assert _PHOTO_DAMAGE_MAX_PENALTY < 0.08 + 0.05      # taxi penalty is 0.92
        worst = self._score(1.0)
        clean = self._score(0.0)
        assert worst >= clean * (1 - _PHOTO_DAMAGE_MAX_PENALTY) - 1e-6

    def test_missing_score_is_not_a_penalty(self):
        """Most of the corpus predates the classifier; absence must cost nothing."""
        assert self._score(None) == self._score(0.0)
        assert self._score(pd.NA) == self._score(0.0)

    def test_garbage_score_is_ignored_not_fatal(self):
        assert self._score("very damaged") == self._score(0.0)

    def test_it_never_vetoes(self):
        """A maxed-out photo score must not turn a BUY into a SKIP on its own."""
        clean = decide(_row(photo_damage_p=0.0), _ctx())
        if clean.verdict != VERDICT_BUY:
            pytest.skip("base row is not a BUY; nothing to demote")
        assert decide(_row(photo_damage_p=1.0), _ctx()).verdict == VERDICT_BUY
