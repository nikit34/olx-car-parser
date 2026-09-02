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
