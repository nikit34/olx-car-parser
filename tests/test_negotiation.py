"""The negotiation norm is a number a buyer will quote at a seller, so the two
ways it could lie matter more than usual: counting a seller who never moved as
a mover, and publishing a band that rests on a handful of cars.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.analytics.negotiation import BANDS, norm_for_price, price_band_norms


def _cars(n, first_ask, change_pct, days=60, start=0):
    return [{
        "car_id": f"c{start + i}", "first_ask": first_ask,
        "ask_change_pct": change_pct, "cut": change_pct < -1.0, "days": days,
    } for i in range(n)]


def _df(*groups):
    return pd.DataFrame([r for g in groups for r in g])


def test_the_share_and_the_size_describe_only_the_sellers_who_moved():
    cars = _df(_cars(300, 10000, -8.0), _cars(700, 10000, 0.0, start=1000))
    band = [b for b in price_band_norms(cars) if b["lo"] == 8000][0]
    assert band["n"] == 1000
    assert band["cu"] == pytest.approx(0.30, abs=0.001)
    assert band["cp"] == pytest.approx(8.0, abs=0.01)


def test_a_thin_band_is_dropped_rather_than_published():
    cars = _df(_cars(199, 30000, -5.0), _cars(400, 10000, -5.0, start=500))
    bands = {b["lo"] for b in price_band_norms(cars, min_cars=200)}
    assert 25000 not in bands
    assert 8000 in bands


def test_a_band_where_nobody_moved_still_reports_the_share():
    cars = _df(_cars(400, 5000, 0.0))
    band = [b for b in price_band_norms(cars) if b["lo"] == 4000][0]
    assert band["cu"] == 0.0
    assert "cp" not in band


def test_the_bands_cover_every_price_without_overlapping():
    cars = _df(_cars(250, 1000, -3.0), _cars(250, 5000, -3.0, start=300),
               _cars(250, 9000, -3.0, start=600), _cars(250, 20000, -3.0, start=900),
               _cars(250, 40000, -3.0, start=1200))
    norms = price_band_norms(cars)
    assert [b["lo"] for b in norms] == [0, 4000, 8000, 15000, 25000]
    assert sum(b["n"] for b in norms) == 1250
    assert norms[-1]["hi"] is None


def test_a_price_lands_in_exactly_one_band():
    norms = price_band_norms(_df(*[_cars(250, p, -3.0, start=i * 300)
                                   for i, p in enumerate((1000, 5000, 9000, 20000, 40000))]))
    assert norm_for_price(norms, 3999)["lo"] == 0
    assert norm_for_price(norms, 4000)["lo"] == 4000
    assert norm_for_price(norms, 99000)["lo"] == 25000
    assert norm_for_price(norms, None) is None
    assert norm_for_price([], 5000) is None


def test_an_empty_corpus_publishes_nothing():
    assert price_band_norms(pd.DataFrame()) == []
    assert len(BANDS) == 5
