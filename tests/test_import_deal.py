"""The import comparison: what has to be true before a saving is published.

The claim on these pages is the strongest one on the site — buy there, pay the
tax, and you still come out ahead — so the gates matter more than the arithmetic.
Pinned here: both sides of the same model year or no cell; the ISV computed per
German listing rather than off a median car; the fee band carried through to both
ends of the answer; and nothing published for a model that only one market has.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.analytics.import_deal import (
    MIN_CELLS,
    MIN_DE_YEAR_N,
    MIN_ISV_N,
    MIN_PT_YEAR_N,
    build_import_pages,
    fees_band,
)

NOW_YEAR = 2026


def _de(n, *, year=2016, price=10000, brand="Volkswagen", model="Golf",
        co2=120, cc=1598, fuel="Diesel", km=150000, taxable=None):
    """n German listings; ``taxable`` limits how many carry a usable CO2."""
    taxable = n if taxable is None else taxable
    return [{
        "brand": brand, "model": model, "year": year,
        "price_eur": price + (i % 5) * 100,
        "co2_g_km": co2 if i < taxable else None,
        "engine_cc": cc, "fuel_type": fuel, "mileage_km": km,
    } for i in range(n)]


def _pt(n, *, year=2016, price=14000, brand="Volkswagen", model="Golf", km=180000):
    return [{
        "brand": brand, "model": model, "year": year,
        "price_eur": price + (i % 5) * 100, "mileage_km": km, "is_active": True,
    } for i in range(n)]


def _build(de_rows, pt_rows):
    return build_import_pages(pd.DataFrame(pt_rows), pd.DataFrame(de_rows),
                              now_year=NOW_YEAR)


def _two_good_years(**over):
    de = _de(MIN_DE_YEAR_N, year=2016, **over) + _de(MIN_DE_YEAR_N, year=2017, **over)
    pt = _pt(MIN_PT_YEAR_N, year=2016) + _pt(MIN_PT_YEAR_N, year=2017)
    return de, pt


class TestTheAnswer:
    def test_the_landed_cost_is_the_german_price_plus_tax_plus_fees(self):
        de, pt = _two_good_years()
        doc = _build(de, pt)
        rec = doc["models"]["volkswagen-golf"]
        lo, hi = fees_band()
        cell = rec["yr"][0]
        assert cell["ll"] == round(cell["dm"] + cell["isv"] + lo)
        assert cell["lh"] == round(cell["dm"] + cell["isv"] + hi)
        assert cell["isv"] > 0

    def test_the_saving_is_a_band_because_the_fees_are(self):
        de, pt = _two_good_years()
        cell = _build(de, pt)["models"]["volkswagen-golf"]["yr"][0]
        lo, hi = fees_band()
        assert cell["gh"] - cell["gl"] == pytest.approx(hi - lo, abs=1)
        assert cell["gl"] == cell["ptm"] - cell["lh"]

    def test_a_model_that_lands_dearer_is_published_as_a_loss(self):
        de, pt = _two_good_years(price=20000)
        rec = _build(de, pt)["models"]["volkswagen-golf"]
        assert rec["wins"] == 0
        assert rec["med_gap"] < 0

    def test_both_markets_carry_their_own_sample_size(self):
        de = _de(12, year=2016) + _de(11, year=2017)
        pt = _pt(9, year=2016) + _pt(7, year=2017)
        rec = _build(de, pt)["models"]["volkswagen-golf"]
        assert [c["nde"] for c in rec["yr"]] == [11, 12]
        assert [c["npt"] for c in rec["yr"]] == [7, 9]


class TestGates:
    def test_a_year_only_germany_has_is_not_a_cell(self):
        de = _de(MIN_DE_YEAR_N, year=2016) + _de(MIN_DE_YEAR_N, year=2017) \
            + _de(MIN_DE_YEAR_N, year=2011)
        pt = _pt(MIN_PT_YEAR_N, year=2016) + _pt(MIN_PT_YEAR_N, year=2017)
        rec = _build(de, pt)["models"]["volkswagen-golf"]
        assert {c["y"] for c in rec["yr"]} == {2016, 2017}

    def test_a_thin_german_year_is_not_a_cell(self):
        de = _de(MIN_DE_YEAR_N - 1, year=2016) + _de(MIN_DE_YEAR_N, year=2017)
        pt = _pt(MIN_PT_YEAR_N, year=2016) + _pt(MIN_PT_YEAR_N, year=2017)
        assert _build(de, pt)["models"] == {}

    def test_a_thin_portuguese_year_is_not_a_cell(self):
        de = _de(MIN_DE_YEAR_N, year=2016) + _de(MIN_DE_YEAR_N, year=2017)
        pt = _pt(MIN_PT_YEAR_N - 1, year=2016) + _pt(MIN_PT_YEAR_N, year=2017)
        assert _build(de, pt)["models"] == {}

    def test_a_year_where_too_few_cars_can_be_taxed_is_dropped(self):
        de = _de(MIN_DE_YEAR_N, year=2016, taxable=MIN_ISV_N - 1) \
            + _de(MIN_DE_YEAR_N, year=2017)
        pt = _pt(MIN_PT_YEAR_N, year=2016) + _pt(MIN_PT_YEAR_N, year=2017)
        assert _build(de, pt)["models"] == {}

    def test_one_good_year_is_not_a_page(self):
        assert MIN_CELLS == 2
        de = _de(MIN_DE_YEAR_N, year=2016)
        pt = _pt(MIN_PT_YEAR_N, year=2016)
        assert _build(de, pt)["models"] == {}

    def test_a_model_portugal_does_not_sell_never_appears(self):
        de, pt = _two_good_years()
        de += _de(MIN_DE_YEAR_N, year=2016, brand="Opel", model="Insignia") \
            + _de(MIN_DE_YEAR_N, year=2017, brand="Opel", model="Insignia")
        assert set(_build(de, pt)["models"]) == {"volkswagen-golf"}

    def test_an_empty_german_side_publishes_nothing_but_still_states_the_costs(self):
        doc = build_import_pages(pd.DataFrame(_pt(30)), pd.DataFrame(), now_year=NOW_YEAR)
        assert doc["models"] == {}
        assert doc["costs"]["lo"] > 0 and doc["costs"]["hi"] > doc["costs"]["lo"]


class TestSpellingAndTax:
    def test_the_two_markets_spell_the_same_car_differently(self):
        de = _de(MIN_DE_YEAR_N, year=2016, brand="Citroen", model="Megane") \
            + _de(MIN_DE_YEAR_N, year=2017, brand="Citroen", model="Megane")
        pt = _pt(MIN_PT_YEAR_N, year=2016, brand="Citroën", model="Mégane") \
            + _pt(MIN_PT_YEAR_N, year=2017, brand="Citroën", model="Mégane")
        assert "citroen-megane" in _build(de, pt)["models"]

    def test_an_electric_car_is_exempt_and_still_gets_a_cell(self):
        de, pt = _two_good_years(fuel="Eléctrico", co2=0)
        rec = _build(de, pt)["models"]["volkswagen-golf"]
        assert all(c["isv"] == 0 for c in rec["yr"])

    def test_a_plug_in_hybrid_cannot_be_taxed_here_so_it_is_not_published(self):
        de, pt = _two_good_years(fuel="Híbrido Plug-in")
        assert _build(de, pt)["models"] == {}

    def test_older_cars_pay_less_of_the_same_tax(self):
        de = _de(MIN_DE_YEAR_N, year=2013) + _de(MIN_DE_YEAR_N, year=2023)
        pt = _pt(MIN_PT_YEAR_N, year=2013) + _pt(MIN_PT_YEAR_N, year=2023)
        cells = {c["y"]: c["isv"] for c in _build(de, pt)["models"]["volkswagen-golf"]["yr"]}
        assert cells[2013] < cells[2023]

    def test_the_mileage_of_both_sides_is_carried_so_the_reader_can_see_the_mismatch(self):
        de = _de(MIN_DE_YEAR_N, year=2016, km=90000) + _de(MIN_DE_YEAR_N, year=2017, km=90000)
        pt = _pt(MIN_PT_YEAR_N, year=2016, km=180000) + _pt(MIN_PT_YEAR_N, year=2017, km=180000)
        rec = _build(de, pt)["models"]["volkswagen-golf"]
        assert all(c["dkm"] == 90000 and c["ptkm"] == 180000 for c in rec["yr"])
        assert rec["km_gap"] == pytest.approx(-0.5, abs=0.01)
