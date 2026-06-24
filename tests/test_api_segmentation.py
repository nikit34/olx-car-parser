"""Unit tests for the price-band bisection that gives OLX full coverage.

The OLX JSON API hard-caps any single query at ~1000 results, so full
coverage of the ~50k-car category comes from recursively splitting the price
axis until every band is under the cap. These tests pin the bisection's
three guarantees — it terminates, it tiles the whole range, and every leaf is
under the cap (or below the min-width floor) — without touching the network
(``_segment_count`` is stubbed).
"""

import logging

from src.parser.scraper import (
    OlxScraper, ScraperConfig, _API_OFFSET_CAP, _MIN_BAND_WIDTH_EUR,
)


def _scraper():
    return OlxScraper(ScraperConfig())


class TestPriceBandBisection:
    def test_single_band_when_under_cap(self):
        s = _scraper()
        s._segment_count = lambda lo, hi, cat: 500
        try:
            assert s._price_bands(0, 100_000, 378) == [(0, 100_000, 500)]
        finally:
            s.close()

    def test_empty_band_dropped(self):
        s = _scraper()
        s._segment_count = lambda lo, hi, cat: 0
        try:
            assert s._price_bands(0, 100_000, 378) == []
        finally:
            s.close()

    def test_bisects_and_tiles_range_contiguously(self):
        s = _scraper()
        # synthetic distribution: count scales with band width, so wide
        # bands exceed the cap and must split, narrow ones don't.
        s._segment_count = lambda lo, hi, cat: ((hi or 1_000_000) - lo) // 10
        try:
            bands = sorted(s._price_bands(0, 1_000_000, 378))
        finally:
            s.close()
        assert bands
        assert bands[0][0] == 0
        assert bands[-1][1] == 1_000_000
        # contiguous: each band starts where the previous ends (shared
        # inclusive boundary — scrape_full dedups across it).
        for (lo1, hi1, _c1), (lo2, hi2, _c2) in zip(bands, bands[1:]):
            assert hi1 == lo2
        # every leaf under the cap, or below the min-width floor
        for lo, hi, count in bands:
            assert count < _API_OFFSET_CAP or (hi - lo) <= _MIN_BAND_WIDTH_EUR

    def test_terminates_when_never_under_cap(self):
        # Pathological: every band reports exactly the cap. Must still
        # terminate via the depth / min-width guards rather than recurse
        # forever.
        s = _scraper()
        s._segment_count = lambda lo, hi, cat: _API_OFFSET_CAP
        try:
            bands = s._price_bands(0, 1_000_000, 378, max_depth=14)
        finally:
            s.close()
        assert bands
        assert len(bands) < 100_000  # finite

    def test_open_top_band_not_bisected(self):
        s = _scraper()
        s._segment_count = lambda lo, hi, cat: _API_OFFSET_CAP
        try:
            # hi=None (open-ended) can't be split → returned as-is
            assert s._price_bands(40_000, None, 378) == [(40_000, None, _API_OFFSET_CAP)]
        finally:
            s.close()

    def test_splits_a_saturated_band_below_the_old_100_floor(self):
        # A saturated sliver only 64 euros wide. The old hard €100 floor
        # accepted it as a single leaf and silently dropped its tail; now we
        # keep bisecting past €100 (down to where it falls under the cap).
        s = _scraper()
        # at-cap while wider than 4 euros, sub-cap once narrower
        s._segment_count = (
            lambda lo, hi, cat: _API_OFFSET_CAP if (hi - lo) > 4 else 5)
        try:
            bands = s._price_bands(1000, 1064, 378)
        finally:
            s.close()
        assert len(bands) > 1                       # actually split, not one capped leaf
        assert all(c < _API_OFFSET_CAP for _lo, _hi, c in bands)  # all leaves recovered under cap
        assert all((hi - lo) <= 4 for lo, hi, _c in bands)        # split finer than the old 100 floor

    def test_warns_when_capped_band_cannot_be_split(self, caplog):
        # Pathological: every band reports the cap, so depth runs out before
        # any leaf drops under it. Each unsplittable-yet-saturated leaf must
        # emit a loud ::warning:: so the coverage loss is never silent.
        s = _scraper()
        s._segment_count = lambda lo, hi, cat: _API_OFFSET_CAP
        try:
            with caplog.at_level(logging.WARNING):
                bands = s._price_bands(0, 8, 378, max_depth=2)
        finally:
            s.close()
        assert bands
        assert any("saturated" in r.getMessage() for r in caplog.records)
