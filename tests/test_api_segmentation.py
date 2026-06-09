"""Unit tests for the price-band bisection that gives OLX full coverage.

The OLX JSON API hard-caps any single query at ~1000 results, so full
coverage of the ~50k-car category comes from recursively splitting the price
axis until every band is under the cap. These tests pin the bisection's
three guarantees — it terminates, it tiles the whole range, and every leaf is
under the cap (or below the min-width floor) — without touching the network
(``_segment_count`` is stubbed).
"""

from src.parser.scraper import OlxScraper, ScraperConfig, _API_OFFSET_CAP


def _scraper():
    return OlxScraper(ScraperConfig())


class TestPriceBandBisection:
    def test_single_band_when_under_cap(self):
        s = _scraper()
        s._segment_count = lambda lo, hi, cat: 500
        try:
            assert s._price_bands(0, 100_000, 378) == [(0, 100_000)]
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
        for (lo1, hi1), (lo2, hi2) in zip(bands, bands[1:]):
            assert hi1 == lo2
        # every leaf under the cap, or below the min-width floor
        for lo, hi in bands:
            count = (hi - lo) // 10
            assert count < _API_OFFSET_CAP or (hi - lo) <= 100

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
            assert s._price_bands(40_000, None, 378) == [(40_000, None)]
        finally:
            s.close()
