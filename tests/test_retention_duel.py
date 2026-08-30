"""Retention duels: the number, and everything they refuse to say.

A page built on this module makes exactly one claim — that one side of an
either/or loses value faster than the other in a named model — so these tests
pin the two ways that claim can be wrong. It must survive the mileage confound
(diesels on sale have far more kilometres, automatics far fewer, and a fit
without that control reports the mileage mix as if it were the fuel or the
gearbox), and it must not be published at all when the sample cannot separate
the two curves.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics.retention_duel import (
    DUELS,
    MAX_CI_HALF,
    MIN_SIDE_N,
    MIN_SPAN,
    all_duels,
    retention_duel,
)

NOW = 2026


def _fleet(side, rate, years, per_year, base=20000, km_per_year=15000, seed=0, price_noise=0.0):
    """Listings whose asking price falls at ``rate`` per year of age."""
    rng = np.random.default_rng(seed)
    rows = []
    for y in years:
        age = NOW - y
        for _ in range(per_year):
            price = base * (1 - rate) ** age
            if price_noise:
                price *= float(np.exp(rng.normal(0, price_noise)))
            rows.append({"fuel_type": side, "year": y, "price_eur": round(price),
                         "mileage_km": max(5000, int(km_per_year * age))})
    return rows


def _df(rows):
    return pd.DataFrame(rows)


YEARS = list(range(2006, 2023))


class TestTheNumber:
    def test_recovers_two_different_rates(self):
        rows = (_fleet("Diesel", 0.06, YEARS, 3, seed=1, price_noise=0.05)
                + _fleet("Gasolina", 0.10, YEARS, 3, seed=2, price_noise=0.05))
        out = _fit(_df(rows), NOW)
        assert out is not None
        assert abs(out["a"]["r"] - 0.06) < 0.015
        assert abs(out["b"]["r"] - 0.10) < 0.015
        assert out["t"] > 1.96

    def test_mileage_gap_alone_does_not_invent_a_fuel_effect(self):
        """Both fuels lose value at the same rate, but the diesels on sale have
        twice the kilometres. Without the mileage control the diesel curve would
        look far cheaper and the page would blame the fuel."""
        rows = (_fleet("Diesel", 0.08, YEARS, 3, km_per_year=28000, seed=3, price_noise=0.04)
                + _fleet("Gasolina", 0.08, YEARS, 3, km_per_year=12000, seed=4, price_noise=0.04))
        out = _fit(_df(rows), NOW)
        assert out is not None
        assert abs(out["a"]["r"] - out["b"]["r"]) < 0.02
        assert abs(out["t"]) < 1.96
        assert out["a"]["km"] > out["b"]["km"]

    def test_gap_is_a_premium_with_its_own_interval(self):
        rows = (_fleet("Diesel", 0.07, YEARS, 4, base=24000, seed=5, price_noise=0.04)
                + _fleet("Gasolina", 0.07, YEARS, 4, base=20000, seed=6, price_noise=0.04))
        out = _fit(_df(rows), NOW)
        assert out is not None
        for age, est, half in out["gap"]:
            assert 0 < age <= NOW - min(YEARS)
            assert half >= 0
            assert 0.05 < est < 0.35


class TestWhatItRefusesToPublish:
    def test_thin_side_gets_nothing(self):
        rows = (_fleet("Diesel", 0.07, YEARS, 3, seed=7)
                + _fleet("Gasolina", 0.10, YEARS[:2], 2, seed=8))
        assert len(_df(rows).query("fuel_type == 'Gasolina'")) < MIN_SIDE_N
        assert _fit(_df(rows), NOW) is None

    def test_short_year_span_gets_nothing(self):
        short = list(range(2020, 2024))
        assert len(short) - 1 < MIN_SPAN
        rows = (_fleet("Diesel", 0.07, short, 8, seed=9)
                + _fleet("Gasolina", 0.10, short, 8, seed=10))
        assert _fit(_df(rows), NOW) is None

    def test_a_wide_interval_is_not_published_as_a_draw(self):
        """A difference nobody can measure must not reach the page as "no
        difference" — that reads as a result and it is not one."""
        rows = (_fleet("Diesel", 0.07, YEARS, 2, seed=11, price_noise=0.55)
                + _fleet("Gasolina", 0.07, YEARS, 2, seed=12, price_noise=0.55))
        out = _fit(_df(rows), NOW)
        assert out is None or out["ci"] <= MAX_CI_HALF

    def test_implausible_rates_are_dropped(self):
        rows = (_fleet("Diesel", 0.45, YEARS, 4, seed=13)
                + _fleet("Gasolina", 0.42, YEARS, 4, seed=14))
        assert _fit(_df(rows), NOW) is None

    def test_hybrids_and_missing_mileage_never_enter_the_fit(self):
        rows = (_fleet("Diesel", 0.06, YEARS, 3, seed=15, price_noise=0.05)
                + _fleet("Gasolina", 0.10, YEARS, 3, seed=16, price_noise=0.05))
        noise = _fleet("Híbrido", 0.02, YEARS, 9, seed=17)
        blind = [{**r, "mileage_km": None} for r in _fleet("Diesel", 0.30, YEARS, 9, seed=18)]
        out = _fit(_df(rows + noise + blind), NOW)
        clean = _fit(_df(rows), NOW)
        assert out is not None and clean is not None
        assert out["a"]["n"] == clean["a"]["n"]
        assert abs(out["a"]["r"] - clean["a"]["r"]) < 1e-9

    def test_missing_columns_are_not_an_error(self):
        assert _fit(pd.DataFrame({"brand": ["VW"]}), NOW) is None
        assert _fit(pd.DataFrame(), NOW) is None


def _fit(df, now_year):
    return retention_duel(df, "fuel_type", "Diesel", "Gasolina", now_year)


class TestEveryDimension:
    """The gearbox duel is the same estimator on another column, and the DUELS
    table is what the Worker's own table has to agree with."""

    def test_gearbox_is_fitted_from_the_transmission_column(self):
        rows = (_fleet("Manual", 0.10, YEARS, 3, seed=21, price_noise=0.05)
                + _fleet("Automática", 0.06, YEARS, 3, seed=22, price_noise=0.05))
        df = pd.DataFrame(rows).rename(columns={"fuel_type": "transmission"})
        out = retention_duel(df, "transmission", "Manual", "Automática", NOW)
        assert out is not None
        assert abs(out["a"]["r"] - 0.10) < 0.015
        assert abs(out["b"]["r"] - 0.06) < 0.015

    def test_a_is_the_side_named_first(self):
        assert DUELS["dg"] == ("fuel_type", "Diesel", "Gasolina")
        assert DUELS["cx"] == ("transmission", "Manual", "Automática")

    def test_all_duels_emits_only_what_the_sample_carries(self):
        rows = (_fleet("Diesel", 0.06, YEARS, 3, seed=23, price_noise=0.05)
                + _fleet("Gasolina", 0.10, YEARS, 3, seed=24, price_noise=0.05))
        df = pd.DataFrame(rows)
        df["transmission"] = "Manual"
        out = all_duels(df, NOW)
        assert set(out) == {"dg"}
