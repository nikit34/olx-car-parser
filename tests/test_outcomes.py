"""The outcome table is the only place the corpus gets close to a realised
price, so the two things it must not do are lose the re-listing (which turns
one stubborn car into two happy ones) and read the ask off the wrong end of
the trajectory.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.analytics.outcomes import build_outcomes

NOW = pd.Timestamp("2026-09-20 12:00:00", tz="UTC")


def _ad(olx_id, start_days_ago, lived, price, active=False, brand="Volkswagen",
        model="Golf", year=2015):
    start = NOW - pd.Timedelta(days=start_days_ago)
    return {
        "olx_id": olx_id, "brand": brand, "model": model, "year": year,
        "price_eur": price, "is_active": active,
        "first_seen_at": start,
        "last_scraped_at": start + pd.Timedelta(days=lived),
        "deactivated_at": pd.NaT if active else start + pd.Timedelta(days=lived),
    }


def _snaps(olx_id, start_days_ago, prices, step=5):
    start = NOW - pd.Timedelta(days=start_days_ago)
    return [{"olx_id": olx_id, "price_eur": p,
             "scraped_at": start + pd.Timedelta(days=i * step)}
            for i, p in enumerate(prices)]


def test_one_car_with_two_adverts_is_one_row_that_spans_both():
    ads = pd.DataFrame([_ad("a1", 90, 30, 10000), _ad("a2", 50, 20, 9000)])
    pairs = pd.DataFrame([{"original_olx_id": "a1", "relist_olx_id": "a2",
                           "match_score": 0.9, "gap_days": 10.0}])
    out = build_outcomes(ads, None, pairs, now=NOW)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["car_id"] == "a1"
    assert row["n_ads"] == 2
    assert row["days"] == pytest.approx(60, abs=0.1)
    assert row["first_ask"] == 10000
    assert row["last_ask"] == 9000
    assert row["ask_change_pct"] == pytest.approx(-10.0, abs=0.01)
    assert bool(row["cut"]) is True


def test_without_the_pairs_the_same_car_reads_as_two_quick_sales():
    ads = pd.DataFrame([_ad("a1", 90, 30, 10000), _ad("a2", 50, 20, 9000)])
    out = build_outcomes(ads, None, None, now=NOW)
    assert len(out) == 2
    assert out["days"].max() <= 31
    assert not out["cut"].any()


def test_the_asks_come_from_the_whole_trajectory_not_the_last_row():
    ads = pd.DataFrame([_ad("a1", 60, 40, 8000)])
    snaps = pd.DataFrame(_snaps("a1", 60, [9000, 8600, 8000]))
    row = build_outcomes(ads, snaps, None, now=NOW).iloc[0]
    assert row["first_ask"] == 9000
    assert row["last_ask"] == 8000
    assert row["min_ask"] == 8000
    assert row["ask_change_pct"] == pytest.approx(-11.11, abs=0.01)


def test_a_car_still_on_sale_is_not_marked_sold_and_its_clock_runs_to_now():
    ads = pd.DataFrame([_ad("a1", 30, 30, 7000, active=True)])
    row = build_outcomes(ads, None, None, now=NOW).iloc[0]
    assert bool(row["sold"]) is False
    assert bool(row["still_active"]) is True
    assert row["days"] == pytest.approx(30, abs=0.1)


def test_a_car_whose_last_advert_is_live_counts_as_unsold_across_the_chain():
    ads = pd.DataFrame([_ad("a1", 90, 30, 10000),
                        _ad("a2", 40, 40, 9500, active=True)])
    pairs = pd.DataFrame([{"original_olx_id": "a1", "relist_olx_id": "a2",
                           "match_score": 0.9, "gap_days": 20.0}])
    row = build_outcomes(ads, None, pairs, now=NOW).iloc[0]
    assert bool(row["sold"]) is False
    assert row["n_ads"] == 2
    assert row["days"] == pytest.approx(90, abs=0.1)


def test_a_price_that_is_obviously_not_an_ask_is_ignored():
    ads = pd.DataFrame([_ad("a1", 60, 40, 8000)])
    snaps = pd.DataFrame(_snaps("a1", 60, [9000, 1, 8000]))
    row = build_outcomes(ads, snaps, None, now=NOW).iloc[0]
    assert row["min_ask"] == 8000


def test_an_empty_corpus_keeps_the_shape():
    out = build_outcomes(pd.DataFrame())
    assert list(out.columns)[:4] == ["car_id", "n_ads", "brand", "model"]
    assert out.empty


class TestFirstCutDay:
    """When the seller moved matters as much as whether: a norm read against a
    listing's own age needs to know who had already moved by day 30."""

    def test_the_day_counts_from_the_first_advert(self):
        ads = pd.DataFrame([_ad("a1", 60, 40, 8000)])
        snaps = pd.DataFrame(_snaps("a1", 60, [9000, 9000, 8400], step=10))
        row = build_outcomes(ads, snaps, None, now=NOW).iloc[0]
        assert row["first_cut_day"] == pytest.approx(20, abs=0.5)

    def test_a_seller_who_never_moved_has_no_day(self):
        ads = pd.DataFrame([_ad("a1", 60, 40, 9000)])
        snaps = pd.DataFrame(_snaps("a1", 60, [9000, 9000, 9000]))
        row = build_outcomes(ads, snaps, None, now=NOW).iloc[0]
        assert pd.isna(row["first_cut_day"])
        assert bool(row["cut"]) is False

    def test_a_relist_that_goes_back_up_is_not_a_cut(self):
        ads = pd.DataFrame([_ad("a1", 90, 20, 9000), _ad("a2", 50, 20, 9500)])
        pairs = pd.DataFrame([{"original_olx_id": "a1", "relist_olx_id": "a2",
                               "match_score": 0.9, "gap_days": 20.0}])
        snaps = pd.DataFrame(_snaps("a1", 90, [9000, 9000], step=10)
                             + _snaps("a2", 50, [9500, 9500], step=10))
        row = build_outcomes(ads, snaps, pairs, now=NOW).iloc[0]
        assert pd.isna(row["first_cut_day"])

    def test_the_cut_is_found_across_the_chain_not_only_inside_one_advert(self):
        ads = pd.DataFrame([_ad("a1", 90, 20, 9000), _ad("a2", 50, 20, 8000)])
        pairs = pd.DataFrame([{"original_olx_id": "a1", "relist_olx_id": "a2",
                               "match_score": 0.9, "gap_days": 20.0}])
        snaps = pd.DataFrame(_snaps("a1", 90, [9000, 9000], step=10)
                             + _snaps("a2", 50, [8000, 8000], step=10))
        row = build_outcomes(ads, snaps, pairs, now=NOW).iloc[0]
        assert row["first_cut_day"] == pytest.approx(40, abs=0.5)
        assert bool(row["cut"]) is True


class TestCrossPosting:
    """The same car on OLX and StandVirtual is one car. Counting it twice
    inflated every sample size published off this table by 19%."""

    def _both_sites(self):
        olx = _ad("o1", 60, 40, 9000)
        sv = _ad("s1", 55, 35, 9000)
        sv["duplicate_of"] = "o1"
        olx["duplicate_of"] = None
        return pd.DataFrame([olx, sv])

    def test_a_car_posted_on_two_sites_at_once_is_one_row(self):
        out = build_outcomes(self._both_sites(), None, None, now=NOW)
        assert len(out) == 1
        row = out.iloc[0]
        assert row["car_id"] == "o1"
        assert row["n_ads"] == 2
        assert row["n_relists"] == 0

    def test_without_the_duplicate_mark_it_reads_as_two_cars(self):
        ads = self._both_sites()
        ads["duplicate_of"] = None
        assert len(build_outcomes(ads, None, None, now=NOW)) == 2

    def test_a_cross_post_is_not_a_second_attempt_but_a_relist_is(self):
        ads = self._both_sites()
        later = _ad("o2", 20, 15, 8500)
        later["duplicate_of"] = None
        ads = pd.concat([ads, pd.DataFrame([later])], ignore_index=True)
        pairs = pd.DataFrame([{"original_olx_id": "o1", "relist_olx_id": "o2",
                               "match_score": 0.9, "gap_days": 5.0}])
        row = build_outcomes(ads, None, pairs, now=NOW).iloc[0]
        assert row["n_ads"] == 3
        assert row["n_relists"] == 1

    def test_a_duplicate_pointing_outside_the_corpus_is_left_alone(self):
        ads = self._both_sites()
        ads.loc[ads["olx_id"] == "s1", "duplicate_of"] = "gone-from-db"
        assert len(build_outcomes(ads, None, None, now=NOW)) == 2


class TestPriceStoryOfACrossPostedCar:
    """The two sites carry different numbers for the same car, so pouring both
    into one trajectory invents a cut the seller never made."""

    def _car_on_two_sites(self, olx_price, sv_price):
        olx = _ad("o1", 60, 40, olx_price)
        olx["duplicate_of"] = None
        sv = _ad("s1", 58, 38, sv_price)
        sv["duplicate_of"] = "o1"
        snaps = pd.DataFrame(_snaps("o1", 60, [olx_price, olx_price], step=10)
                             + _snaps("s1", 58, [sv_price, sv_price], step=10))
        return pd.DataFrame([olx, sv]), snaps

    def test_a_platform_gap_is_not_a_price_cut(self):
        ads, snaps = self._car_on_two_sites(7500, 5000)
        row = build_outcomes(ads, snaps, None, now=NOW).iloc[0]
        assert row["n_ads"] == 2
        assert bool(row["cut"]) is False
        assert row["first_ask"] == 7500 and row["last_ask"] == 7500

    def test_a_real_cut_on_the_surviving_advert_still_counts(self):
        ads, _ = self._car_on_two_sites(9000, 8000)
        snaps = pd.DataFrame(_snaps("o1", 60, [9000, 8100], step=20)
                             + _snaps("s1", 58, [8000, 8000], step=10))
        row = build_outcomes(ads, snaps, None, now=NOW).iloc[0]
        assert bool(row["cut"]) is True
        assert row["ask_change_pct"] == pytest.approx(-10.0, abs=0.01)

    def test_an_advert_that_replaced_a_dead_one_keeps_telling_the_story(self):
        first = _ad("a1", 90, 20, 10000)
        second = _ad("a2", 50, 20, 9000)
        for row in (first, second):
            row["duplicate_of"] = None
        pairs = pd.DataFrame([{"original_olx_id": "a1", "relist_olx_id": "a2",
                               "match_score": 0.9, "gap_days": 20.0}])
        snaps = pd.DataFrame(_snaps("a1", 90, [10000, 10000], step=10)
                             + _snaps("a2", 50, [9000, 9000], step=10))
        row = build_outcomes(pd.DataFrame([first, second]), snaps, pairs, now=NOW).iloc[0]
        assert row["first_ask"] == 10000 and row["last_ask"] == 9000
        assert bool(row["cut"]) is True
