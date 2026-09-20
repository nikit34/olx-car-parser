"""Tests for the /avaliar lookup blob (src.analytics.valuations)."""
from __future__ import annotations

import pandas as pd
import pytest

from src.analytics.valuations import build_valuations


NOW = pd.Timestamp.now(tz="UTC")


def _listings(**over):
    row = {
        "olx_id": "AAA", "is_active": True, "title": "VW Golf 1.6 TDI",
        "description": "", "brand": "Volkswagen", "model": "Golf", "year": 2015,
        "mileage_km": 150000, "fuel_type": "Diesel", "price_eur": 9000,
        "city": "Porto", "origin": None, "first_seen_at": NOW - pd.Timedelta(days=68),
        "text_minor_fault": float("nan"), "text_hard_block_phrase": float("nan"),
    }
    row.update(over)
    return pd.DataFrame([row])


def _predictions():
    return pd.DataFrame([{
        "olx_id": "AAA", "predicted_price": 11000,
        "fair_price_low": 9500, "fair_price_high": 12500,
    }])


def _snapshots(points):
    return pd.DataFrame([
        {"olx_id": "AAA", "price_eur": p, "scraped_at": NOW - pd.Timedelta(days=d)}
        for d, p in points
    ])


class TestPriceTrack:
    def test_a_seller_who_came_down_ships_the_track(self):
        snaps = _snapshots([(68, 10500), (40, 9900), (6, 9000)])
        car = build_valuations(_listings(), _predictions(), snapshots=snaps)["cars"]["AAA"]
        assert car["ph"] == [[68, 10500], [40, 9900], [6, 9000]]

    def test_a_price_that_never_moved_ships_nothing(self):
        snaps = _snapshots([(68, 9000)])
        car = build_valuations(_listings(), _predictions(), snapshots=snaps)["cars"]["AAA"]
        assert "ph" not in car

    def test_repeated_identical_prices_collapse(self):
        snaps = _snapshots([(68, 9000), (40, 9000), (6, 9000)])
        car = build_valuations(_listings(), _predictions(), snapshots=snaps)["cars"]["AAA"]
        assert "ph" not in car

    def test_only_the_last_six_points_travel(self):
        snaps = _snapshots([(90 - i * 5, 12000 - i * 100) for i in range(10)])
        car = build_valuations(_listings(), _predictions(), snapshots=snaps)["cars"]["AAA"]
        assert len(car["ph"]) == 6

    def test_no_snapshots_at_all_is_not_an_error(self):
        car = build_valuations(_listings(), _predictions())["cars"]["AAA"]
        assert "ph" not in car
        assert car["p"] == 9000


class TestDaysOnMarket:
    def test_age_comes_from_the_posting_date(self):
        car = build_valuations(_listings(), _predictions())["cars"]["AAA"]
        assert car["dom"] == 68

    def test_a_missing_posting_date_is_left_out(self):
        car = build_valuations(_listings(first_seen_at=None), _predictions())["cars"]["AAA"]
        assert "dom" not in car


class TestFaultPhrases:
    def test_a_missing_phrase_is_not_a_flag(self):
        """pandas-missing is a float NaN and NaN is truthy — the whole corpus
        came back flagged when this was a plain boolean test."""
        car = build_valuations(_listings(), _predictions())["cars"]["AAA"]
        assert "mf" not in car
        assert "hb" not in car

    def test_the_seller_own_words_travel(self):
        car = build_valuations(
            _listings(text_minor_fault="fuga de óleo",
                      text_hard_block_phrase="para peças"),
            _predictions())["cars"]["AAA"]
        assert car["mf"] == "fuga de óleo"
        assert car["hb"] == "para peças"


class TestAHoleInTheTextColumns:
    """A foreign corpus has descriptions on some rows and not others.

    A text column holds None only while every row is None; let one row carry a
    description and pandas turns the rest into NaN, which is truthy. Every
    country blob build died on that the first time the readers started filling
    descriptions in.
    """

    def test_a_nan_description_reads_as_no_text(self):
        rows = _listings().to_dict("records")[0]
        listings = pd.DataFrame([
            dict(rows, olx_id="AAA", description="importado da Alemanha"),
            dict(rows, olx_id="BBB", description=None),
        ])
        predictions = pd.concat([_predictions(),
                                 _predictions().assign(olx_id="BBB")],
                                ignore_index=True)
        assert isinstance(listings["description"].iloc[1], float), "the hole is a NaN"

        cars = build_valuations(listings, predictions)["cars"]

        assert set(cars) == {"AAA", "BBB"}

    def test_a_nan_title_does_not_stop_the_build(self):
        cars = build_valuations(_listings(title=float("nan")),
                                _predictions())["cars"]
        assert "AAA" in cars


class TestListingSource:
    def test_a_standvirtual_row_says_so(self):
        blob = build_valuations(_listings(source="standvirtual"), _predictions())
        assert blob["cars"]["AAA"]["sv"] == 1

    def test_an_olx_row_carries_no_flag(self):
        blob = build_valuations(_listings(source="olx"), _predictions())
        assert "sv" not in blob["cars"]["AAA"]

    def test_a_missing_source_reads_as_olx(self):
        rows = _listings().to_dict("records")[0]
        rows.pop("source", None)
        blob = build_valuations(pd.DataFrame([rows]), _predictions())
        assert "sv" not in blob["cars"]["AAA"]

    def test_the_blob_announces_the_version_that_carries_the_source(self):
        assert build_valuations(_listings(), _predictions())["v"] == 3
        assert build_valuations(pd.DataFrame(), pd.DataFrame())["v"] == 3


class TestCarHistoryAndNorms:
    """A car on its third advert has been for sale far longer than the advert
    admits, and a listing that has not moved still sits in a band where most
    sellers do. Both are facts the buyer can check; neither was in the blob."""

    def _two_ads(self):
        first = _listings().to_dict("records")[0]
        first.update({"olx_id": "OLD", "is_active": False,
                      "first_seen_at": NOW - pd.Timedelta(days=200)})
        second = _listings().to_dict("records")[0]
        second["first_seen_at"] = NOW - pd.Timedelta(days=68)
        return pd.DataFrame([first, second])

    def _pairs(self):
        return pd.DataFrame([{"original_olx_id": "OLD", "relist_olx_id": "AAA",
                              "match_score": 0.9, "gap_days": 12.0}])

    def test_the_car_clock_runs_from_the_first_advert_not_the_current_one(self):
        car = build_valuations(self._two_ads(), _predictions(),
                               pairs=self._pairs())["cars"]["AAA"]
        assert car["dom"] == 68
        assert car["dc"] == 200
        assert car["na"] == 2

    def test_a_car_with_one_advert_carries_no_chain_fields(self):
        car = build_valuations(_listings(), _predictions(),
                               pairs=self._pairs())["cars"]["AAA"]
        assert "dc" not in car and "na" not in car

    def test_without_the_pairs_the_relist_stays_invisible(self):
        car = build_valuations(self._two_ads(), _predictions())["cars"]["AAA"]
        assert "dc" not in car

    def test_the_bands_ride_along_so_a_quiet_listing_still_has_a_norm(self):
        norms = [{"lo": 8000, "hi": 15000, "lbl": "€8.000 a €15.000",
                  "n": 10305, "cu": 0.604, "cp": 4.8, "md": 69}]
        blob = build_valuations(_listings(), _predictions(), norms=norms)
        assert blob["neg"] == norms
        assert "neg" not in build_valuations(_listings(), _predictions())
