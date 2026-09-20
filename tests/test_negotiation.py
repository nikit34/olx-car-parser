"""The negotiation norm is a number a buyer will quote at a seller, so the two
ways it could lie matter more than usual: counting a seller who never moved as
a mover, and publishing a band that rests on a handful of cars.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.analytics.negotiation import (BANDS, age_cell, age_conditional_norms,
                                       merge_age_cells, norm_for_price,
                                       price_band_norms)


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


class TestAgeConditionalNorms:
    """A listing that has sat two months is not a random listing: whoever was
    going to move early already did. The cell must describe who is left."""

    @staticmethod
    def _car(i, first_ask, days, cut_day=None, change=-6.0):
        return {"car_id": f"a{i}", "first_ask": first_ask, "days": days,
                "cut": cut_day is not None, "first_cut_day": cut_day,
                "ask_change_pct": change if cut_day is not None else 0.0}

    def _corpus(self):
        """300 that moved early and went, 300 that held then moved at day 80,
        300 that never moved, and 300 that sold in three weeks untouched."""
        rows = []
        rows += [self._car(i, 10000, 20, cut_day=5) for i in range(300)]
        rows += [self._car(1000 + i, 10000, 120, cut_day=80) for i in range(300)]
        rows += [self._car(2000 + i, 10000, 120) for i in range(300)]
        rows += [self._car(3000 + i, 10000, 20) for i in range(300)]
        return pd.DataFrame(rows)

    def test_sellers_who_already_moved_are_not_counted_as_the_ones_left(self):
        band = age_conditional_norms(self._corpus(), steps=(60,))[0]
        day, share, size, n = band["ag"][0]
        assert day == 60
        assert n == 600
        assert share == pytest.approx(0.5, abs=0.01)
        assert size == pytest.approx(6.0, abs=0.01)

    def test_the_share_climbs_with_age_because_the_quick_sales_drop_out(self):
        cells = age_conditional_norms(self._corpus(), steps=(14, 60))[0]["ag"]
        early, late = cells[0], cells[1]
        assert early[0] == 14 and late[0] == 60
        assert early[3] == 900 and late[3] == 600
        assert early[1] == pytest.approx(1 / 3, abs=0.01)
        assert late[1] == pytest.approx(0.5, abs=0.01)

    def test_a_thin_cell_is_absent_and_a_band_without_cells_is_dropped(self):
        rows = [self._car(i, 30000, 120, cut_day=90) for i in range(199)]
        assert age_conditional_norms(pd.DataFrame(rows), steps=(60,)) == []

    def test_the_reader_gets_the_deepest_cell_the_listing_has_reached(self):
        band = age_conditional_norms(self._corpus(), steps=(14, 60))[0]
        assert age_cell(band, 5) is None
        assert age_cell(band, 20)[0] == 14
        assert age_cell(band, 200)[0] == 60
        assert age_cell(None, 90) is None
        assert age_cell(band, None) is None

    def test_without_the_cut_day_nobody_is_excluded_for_having_moved_already(self):
        with_day = age_conditional_norms(self._corpus(), steps=(14,))[0]["ag"][0]
        cars = self._corpus().drop(columns=["first_cut_day"])
        without = age_conditional_norms(cars, steps=(14,))[0]["ag"][0]
        assert with_day[3] == 900
        assert without[3] == 1200


def test_the_two_norms_ship_as_one_band_record():
    norms = [{"lo": 8000, "hi": 15000, "lbl": "€8.000 a €15.000", "n": 100, "cu": 0.46},
             {"lo": 25000, "hi": None, "lbl": "acima de €25.000", "n": 80, "cu": 0.45}]
    age = [{"lo": 8000, "hi": 15000, "lbl": "x", "ag": [[60, 0.63, 7.4, 4812]]}]
    merged = merge_age_cells(norms, age)
    assert merged[0]["cu"] == 0.46 and merged[0]["ag"][0][0] == 60
    assert "ag" not in merged[1]
    assert merge_age_cells([], age) == []


def test_a_car_still_on_sale_has_not_finished_its_story():
    rows = ([{"car_id": f"e{i}", "first_ask": 10000, "ask_change_pct": -6.0,
              "cut": True, "days": 90, "first_cut_day": 40, "still_active": False}
             for i in range(300)]
            + [{"car_id": f"a{i}", "first_ask": 10000, "ask_change_pct": 0.0,
                "cut": False, "days": 90, "first_cut_day": None, "still_active": True}
               for i in range(300)])
    cars = pd.DataFrame(rows)
    band = price_band_norms(cars, min_cars=200)[0]
    assert band["n"] == 300
    assert band["cu"] == 1.0
    cell = age_conditional_norms(cars, steps=(30,), min_cars=200)[0]["ag"][0]
    assert cell[3] == 300 and cell[1] == 1.0
