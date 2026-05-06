"""Tests for the flipper-detection composite (``src.analytics.flipper``).

The composite is a soft weighted vote across four primitives. Tests
cover: (a) each primitive's mapping is what we documented, (b) the
weighted-mean denominator skips missing primitives, (c) the breakdown
exposes per-primitive contributions, (d) tri-state plate_obscured is
honoured (None ≠ False).
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from src.analytics.flipper import (
    FlipperBreakdown,
    compute_flipper_score,
    score_listing,
)


def _all_strong_signals() -> dict:
    """Row where every primitive votes flipper at maximum strength."""
    return {
        "seller_listings_count_90d": 5,     # → 1.0 (post-recalibration)
        "seller_cars_count": 5,              # → 1.0
        "seller_pseudoprivate": True,        # → 1.0
        "plate_obscured": True,              # → 1.0
    }


def _all_clean_signals() -> dict:
    """Row where every primitive votes private-individual."""
    return {
        "seller_listings_count_90d": 1,     # → 0.0
        "seller_cars_count": 1,              # → 0.0
        "seller_pseudoprivate": False,       # → 0.0
        "plate_obscured": False,             # → 0.0
    }


class TestScoreListingPrimitives:
    def test_all_strong_signals_score_one(self):
        bd = score_listing(_all_strong_signals())
        assert bd.score == pytest.approx(1.0)
        assert bd.confidence == pytest.approx(1.0)
        # All four primitives contributed.
        assert set(bd.contributions) == {
            "seller_listings_count_90d", "seller_cars_count",
            "seller_pseudoprivate", "plate_obscured",
        }

    def test_all_clean_signals_score_zero(self):
        bd = score_listing(_all_clean_signals())
        assert bd.score == pytest.approx(0.0)
        assert bd.confidence == pytest.approx(1.0)

    def test_listings_90d_buckets(self):
        """1 → 0.0, 2 → 0.5, 3+ → 1.0. Confidence is 0.15 because only
        one primitive is supplied (recalibrated 2026-05-06: rotation
        weight reduced from 0.40 to 0.15 until backfill accumulates a
        full 90-day uuid-linked window)."""
        bd_low = score_listing({"seller_listings_count_90d": 1})
        bd_mid = score_listing({"seller_listings_count_90d": 2})
        bd_high = score_listing({"seller_listings_count_90d": 10})
        assert bd_low.score == pytest.approx(0.0)
        assert bd_mid.score == pytest.approx(0.5)
        assert bd_high.score == pytest.approx(1.0)
        for bd in (bd_low, bd_mid, bd_high):
            assert bd.confidence == pytest.approx(0.15)

    def test_partial_signals_use_only_available_weight(self):
        """Two primitives missing → composite normalises over the two
        available ones. Score is the weighted mean over what's present,
        confidence reflects the sum of those weights."""
        # Only listings_90d (0.15) and pseudoprivate (0.35) present.
        bd = score_listing({
            "seller_listings_count_90d": 5,   # → 1.0
            "seller_pseudoprivate": False,    # → 0.0
        })
        expected = (1.0 * 0.15 + 0.0 * 0.35) / (0.15 + 0.35)
        assert bd.score == pytest.approx(expected)
        assert bd.confidence == pytest.approx(0.50)

    def test_no_data_returns_none_score(self):
        """Freshly scraped listing — no seller_uuid backfill yet, plate
        OCR not run yet — every primitive is None. Score must be None
        (not zero!) so downstream consumers don't mistake "missing data"
        for "definitely not a flipper"."""
        bd = score_listing({})
        assert bd.score is None
        assert bd.confidence == 0.0
        assert bd.contributions == {}

    def test_plate_obscured_tri_state(self):
        """plate_obscured == None means below-threshold photo coverage,
        not signal-absent. It's omitted from the composite (doesn't
        contribute weight), distinct from plate_obscured == False."""
        # plate_obscured=None, only listings_90d (0.15) present.
        bd_undef = score_listing({
            "seller_listings_count_90d": 1,
            "plate_obscured": None,
        })
        # plate_obscured=False adds a 0.0 contribution worth 0.20 weight.
        bd_clean = score_listing({
            "seller_listings_count_90d": 1,
            "plate_obscured": False,
        })
        assert bd_undef.confidence == pytest.approx(0.15)
        assert bd_clean.confidence == pytest.approx(0.15 + 0.20)
        # Both score 0.0 (clean), but the latter has higher confidence.
        assert bd_undef.score == pytest.approx(0.0)
        assert bd_clean.score == pytest.approx(0.0)

    def test_breakdown_exposes_contributions(self):
        """Dashboard tooltip needs (raw, mapped, weight) per primitive."""
        bd = score_listing({
            "seller_listings_count_90d": 5,
            "seller_pseudoprivate": True,
        })
        c = bd.contributions
        assert c["seller_listings_count_90d"] == (5.0, 1.0, 0.15)
        assert c["seller_pseudoprivate"] == (1.0, 1.0, 0.35)
        assert "seller_cars_count" not in c

    def test_handles_nan_pandas_inputs(self):
        """pandas NaN must be treated identically to None. Real callers
        come from DataFrame rows where missing values arrive as NaN."""
        bd = score_listing({
            "seller_listings_count_90d": float("nan"),
            "seller_cars_count": float("nan"),
            "seller_pseudoprivate": False,    # → 0.0
            "plate_obscured": True,            # → 1.0
        })
        # Only the two non-NaN primitives count.
        expected = (0.0 * 0.35 + 1.0 * 0.20) / (0.35 + 0.20)
        assert bd.score == pytest.approx(expected)
        assert bd.confidence == pytest.approx(0.55)


class TestComputeFlipperScoreDataFrame:
    def test_adds_score_and_confidence_columns(self):
        df = pd.DataFrame([
            _all_strong_signals(),
            _all_clean_signals(),
        ])
        out = compute_flipper_score(df)
        assert "flipper_score" in out.columns
        assert "flipper_confidence" in out.columns
        assert out.iloc[0]["flipper_score"] == pytest.approx(1.0)
        assert out.iloc[1]["flipper_score"] == pytest.approx(0.0)

    def test_no_data_row_yields_nan_score(self):
        """A row that DOES have the input columns but all values are
        NaN (e.g. fresh listing pre-backfill, plate OCR not yet run).
        Score is NaN — distinct from the no-op path where input columns
        are missing entirely."""
        df = pd.DataFrame([{
            "seller_listings_count_90d": float("nan"),
            "seller_cars_count": float("nan"),
            "seller_pseudoprivate": None,
            "plate_obscured": None,
        }])
        out = compute_flipper_score(df)
        assert math.isnan(out.iloc[0]["flipper_score"])
        assert out.iloc[0]["flipper_confidence"] == 0.0

    def test_no_op_when_inputs_absent(self):
        """Unmatched-listings DataFrame shape — no seller / plate
        columns present at all. Function returns the frame unchanged."""
        df = pd.DataFrame([{"olx_id": "x", "year": 2020}])
        out = compute_flipper_score(df)
        assert "flipper_score" not in out.columns
        assert "flipper_confidence" not in out.columns

    def test_partial_columns_still_score(self):
        """Only ``plate_obscured`` available — confidence is 0.20 but
        score still computed. Dashboard can hide low-confidence rows."""
        df = pd.DataFrame([{"plate_obscured": True}])
        out = compute_flipper_score(df)
        assert out.iloc[0]["flipper_score"] == pytest.approx(1.0)
        assert out.iloc[0]["flipper_confidence"] == pytest.approx(0.20)
