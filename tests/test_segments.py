"""Tests for src.analytics.segments — per-(brand, model, generation) metrics."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.analytics.segments import (
    compute_segment_metrics,
    compute_segment_time_series,
    composite_resale_score,
)


_NOW = datetime(2026, 5, 3, tzinfo=timezone.utc)


def _row(**overrides) -> dict:
    base = {
        "olx_id": "x",
        "brand": "Volkswagen",
        "model": "Golf",
        "generation": "Mk7",
        "is_active": True,
        "deactivation_reason": None,
        "first_seen_at": _NOW - timedelta(days=20),
        "deactivated_at": None,
        "duplicate_of": None,
        "price_eur": 10000.0,
        "first_price_eur": 10000.0,
    }
    base.update(overrides)
    return base


def _build(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


class TestComputeSegmentMetrics:
    def test_returns_one_row_per_segment(self):
        listings = _build([
            _row(olx_id="a"),
            _row(olx_id="b"),
            _row(olx_id="c", model="Polo", generation="Mk5"),
        ])
        m = compute_segment_metrics(listings, now=_NOW)
        assert len(m) == 2
        assert {"Golf", "Polo"} == set(m["model"])

    def test_n_active_and_sold_split(self):
        listings = _build([
            _row(olx_id=f"a{i}") for i in range(3)
        ] + [
            _row(
                olx_id=f"s{i}",
                is_active=False,
                deactivation_reason="sold",
                first_seen_at=_NOW - timedelta(days=20),
                deactivated_at=_NOW - timedelta(days=10),
            ) for i in range(5)
        ])
        m = compute_segment_metrics(listings, now=_NOW)
        row = m.iloc[0]
        assert row["n_active"] == 3
        assert row["n_sold_60d"] == 5
        assert row["n_sold_total"] == 5

    def test_old_sold_not_in_60d_window(self):
        """Sold > 60 days ago counts in n_sold_total but not n_sold_60d."""
        listings = _build([
            _row(
                olx_id="old-sold",
                is_active=False,
                deactivation_reason="sold",
                first_seen_at=_NOW - timedelta(days=200),
                deactivated_at=_NOW - timedelta(days=120),
            ),
        ])
        m = compute_segment_metrics(listings, now=_NOW)
        assert m.iloc[0]["n_sold_60d"] == 0
        assert m.iloc[0]["n_sold_total"] == 1

    def test_expired_not_treated_as_sold(self):
        """deactivation_reason != 'sold' — counts as inactive but not sold.
        Same logic as elsewhere: only confirmed-sold rows feed clearing /
        time-on-market metrics."""
        listings = _build([
            _row(
                olx_id="expired",
                is_active=False,
                deactivation_reason="expired",
                first_seen_at=_NOW - timedelta(days=20),
                deactivated_at=_NOW - timedelta(days=5),
            ),
        ])
        m = compute_segment_metrics(listings, now=_NOW)
        assert m.iloc[0]["n_active"] == 0
        assert m.iloc[0]["n_sold_60d"] == 0

    def test_duplicates_filtered(self):
        listings = _build([
            _row(olx_id="canon"),
            _row(olx_id="dup", duplicate_of="canon"),
        ])
        m = compute_segment_metrics(listings, now=_NOW)
        assert m.iloc[0]["n_active"] == 1

    def test_median_dom_is_only_recent_sold(self):
        listings = _build([
            _row(
                olx_id="fast",
                is_active=False, deactivation_reason="sold",
                first_seen_at=_NOW - timedelta(days=20),
                deactivated_at=_NOW - timedelta(days=15),  # 5d on market
            ),
            _row(
                olx_id="slow",
                is_active=False, deactivation_reason="sold",
                first_seen_at=_NOW - timedelta(days=100),
                deactivated_at=_NOW - timedelta(days=10),  # 90d on market
            ),
        ])
        m = compute_segment_metrics(listings, now=_NOW)
        # Median of [5, 90] = 47.5
        assert m.iloc[0]["median_dom"] == pytest.approx(47.5)

    def test_clearing_ratio_below_one_when_sold_cheaper(self):
        """Sold listings cleared at lower asks than what's currently
        active → clearing_ratio < 1 → segment is under price pressure."""
        listings = _build(
            [_row(olx_id=f"a{i}", price_eur=10000.0) for i in range(5)]
            + [
                _row(
                    olx_id=f"s{i}",
                    is_active=False, deactivation_reason="sold",
                    price_eur=8000.0,
                    first_seen_at=_NOW - timedelta(days=20),
                    deactivated_at=_NOW - timedelta(days=5),
                ) for i in range(5)
            ]
        )
        m = compute_segment_metrics(listings, now=_NOW)
        assert m.iloc[0]["clearing_ratio"] == pytest.approx(0.8)

    def test_avg_undervaluation_from_signals(self):
        listings = _build([
            _row(olx_id="a", price_eur=8000.0),
            _row(olx_id="b", price_eur=9000.0),
        ])
        signals = pd.DataFrame([
            {"olx_id": "a", "undervaluation_pct": 20.0, "predicted_price": 10000},
            {"olx_id": "b", "undervaluation_pct": 10.0, "predicted_price": 10000},
        ])
        m = compute_segment_metrics(listings, signals=signals, now=_NOW)
        assert m.iloc[0]["avg_undervaluation_pct"] == pytest.approx(15.0)

    def test_calibration_residual_signed(self):
        """Sold listings whose actual last-ask sat *above* the model's
        prediction → residual positive → model under-predicts the
        segment. Negative the other way."""
        listings = _build([
            _row(
                olx_id="s1", price_eur=11000.0,
                is_active=False, deactivation_reason="sold",
                first_seen_at=_NOW - timedelta(days=20),
                deactivated_at=_NOW - timedelta(days=5),
            ),
            _row(
                olx_id="s2", price_eur=12000.0,
                is_active=False, deactivation_reason="sold",
                first_seen_at=_NOW - timedelta(days=20),
                deactivated_at=_NOW - timedelta(days=5),
            ),
        ])
        signals = pd.DataFrame([
            {"olx_id": "s1", "undervaluation_pct": 0, "predicted_price": 10000},
            {"olx_id": "s2", "undervaluation_pct": 0, "predicted_price": 10000},
        ])
        m = compute_segment_metrics(listings, signals=signals, now=_NOW)
        # actual − pred = [+1000, +2000] → median +1500
        assert m.iloc[0]["calibration_residual_eur"] == pytest.approx(1500.0)


class TestSegmentTimeSeries:
    """``compute_segment_time_series`` aggregates real per-scrape
    snapshots into weekly medians. The point of using snapshots
    (not just first_seen_at) is that listings whose price moved
    after first-seen show up at the right week, not stuck at week-1."""

    def _snap(self, **kw):
        base = {
            "olx_id": "x", "brand": "VW", "model": "Golf", "generation": "Mk7",
            "fuel_type": "Diesel", "year": 2018,
            "is_active": True, "deactivation_reason": None, "deactivated_at": None,
            "duplicate_of": None,
        }
        base.update(kw)
        return base

    def test_empty_returns_empty_frame(self):
        out = compute_segment_time_series(pd.DataFrame())
        assert out.empty
        assert list(out.columns) == [
            "bucket", "brand", "model", "generation", "series", "value", "n",
        ]

    def test_active_median_per_week(self):
        """Two listings, two snapshots each — median should average
        them inside each weekly bucket."""
        snaps = pd.DataFrame([
            self._snap(olx_id="a", price_eur=10000, scraped_at=_NOW - timedelta(days=10)),
            self._snap(olx_id="b", price_eur=12000, scraped_at=_NOW - timedelta(days=10)),
            self._snap(olx_id="a", price_eur=9500,  scraped_at=_NOW - timedelta(days=3)),
            self._snap(olx_id="b", price_eur=11500, scraped_at=_NOW - timedelta(days=3)),
        ])
        out = compute_segment_time_series(snaps)
        active = out[out["series"] == "active_ask_median"].sort_values("bucket")
        assert len(active) == 2
        # Older bucket: median of [10000, 12000] = 11000
        # Newer bucket: median of [9500, 11500]  = 10500
        # Order isn't guaranteed by sort, so check both buckets exist.
        assert {11000.0, 10500.0} == set(active["value"])

    def test_price_drop_lands_in_later_bucket(self):
        """Listing posted 3 weeks ago at €10 k that drops to €8 k
        last week: the €8 k should appear in last week's median, NOT
        be stuck at the first-seen week. That's the whole reason we
        use snapshots instead of first_price_eur."""
        snaps = pd.DataFrame([
            self._snap(olx_id="dropper", price_eur=10000,
                       scraped_at=_NOW - timedelta(days=21)),
            self._snap(olx_id="dropper", price_eur=8000,
                       scraped_at=_NOW - timedelta(days=2)),
        ])
        out = compute_segment_time_series(snaps)
        active = out[out["series"] == "active_ask_median"].sort_values("bucket")
        assert len(active) == 2
        early = active.iloc[0]["value"]
        late = active.iloc[1]["value"]
        assert early == 10000.0
        assert late == 8000.0

    def test_duplicates_filtered(self):
        snaps = pd.DataFrame([
            self._snap(olx_id="canon", price_eur=10000,
                       scraped_at=_NOW - timedelta(days=3)),
            self._snap(olx_id="dup", price_eur=8000, duplicate_of="canon",
                       scraped_at=_NOW - timedelta(days=3)),
        ])
        out = compute_segment_time_series(snaps)
        active = out[out["series"] == "active_ask_median"]
        assert active.iloc[0]["value"] == 10000.0  # dup ignored
        assert active.iloc[0]["n"] == 1

    def test_sold_lastask_series_from_listings(self):
        """The ``sold_listings`` arg adds a parallel ``sold_lastask_median``
        series bucketed by deactivation date."""
        snaps = pd.DataFrame([
            self._snap(olx_id="a", price_eur=10000, scraped_at=_NOW - timedelta(days=3)),
        ])
        sold = pd.DataFrame([
            {"olx_id": "s1", "brand": "VW", "model": "Golf", "generation": "Mk7",
             "price_eur": 7500,
             "deactivated_at": _NOW - timedelta(days=10),
             "deactivation_reason": "sold"},
            {"olx_id": "s2", "brand": "VW", "model": "Golf", "generation": "Mk7",
             "price_eur": 8500,
             "deactivated_at": _NOW - timedelta(days=10),
             "deactivation_reason": "sold"},
        ])
        out = compute_segment_time_series(snaps, sold_listings=sold)
        sold_rows = out[out["series"] == "sold_lastask_median"]
        assert len(sold_rows) == 1
        assert sold_rows.iloc[0]["value"] == 8000.0  # median of 7500 + 8500


class TestCompositeScore:
    def test_empty_input_returns_empty(self):
        assert composite_resale_score(pd.DataFrame()).empty

    def test_higher_undervaluation_ranks_higher(self):
        m = pd.DataFrame([
            {"avg_undervaluation_pct": 20, "n_sold_60d": 5, "median_dom": 30, "trend_30d_pct": 0},
            {"avg_undervaluation_pct": 5,  "n_sold_60d": 5, "median_dom": 30, "trend_30d_pct": 0},
        ])
        c = composite_resale_score(m)
        assert c[0] > c[1]

    def test_more_sold_ranks_higher(self):
        m = pd.DataFrame([
            {"avg_undervaluation_pct": 10, "n_sold_60d": 30, "median_dom": 30, "trend_30d_pct": 0},
            {"avg_undervaluation_pct": 10, "n_sold_60d": 1,  "median_dom": 30, "trend_30d_pct": 0},
        ])
        c = composite_resale_score(m)
        assert c[0] > c[1]

    def test_faster_velocity_ranks_higher(self):
        m = pd.DataFrame([
            {"avg_undervaluation_pct": 10, "n_sold_60d": 5, "median_dom": 14, "trend_30d_pct": 0},
            {"avg_undervaluation_pct": 10, "n_sold_60d": 5, "median_dom": 90, "trend_30d_pct": 0},
        ])
        c = composite_resale_score(m)
        assert c[0] > c[1]
