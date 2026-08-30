"""Days-on-market: what the estimator must not get wrong.

Three failure modes are pinned here, because each one shipped a number that was
wrong in a direction nobody would notice on the page:

* dropping the listings still on sale reads fast (the corpus grows, so the ones
  that already ended are the short-lived ones);
* taking the exit time from ``deactivated_at`` reads slow after a scrape outage,
  since the sweep that follows stamps a fortnight of absences with one date;
* a cut with a handful of listings behind it looks exactly like a cut with four
  hundred once it is a row in a table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analytics.liquidity import (
    MIN_CELL_EVENTS,
    MIN_EVENTS,
    WINDOW_DAYS,
    build_liquidity,
    page_records,
    prepare,
    sell_speed_frame,
    survival,
)

NOW = pd.Timestamp("2026-08-30 12:00:00")


def _rows(n, days, active=False, brand="Volkswagen", model="Golf", price=7000,
          year=2014, district="Porto", start=0, outage_lag=0.0, first_price=None,
          spread=2):
    """n listings that lived about ``days`` days, ended (or not) by NOW."""
    out = []
    for i in range(n):
        lived = max(1, days + (i % (2 * spread + 1)) - spread)
        first = NOW - pd.Timedelta(days=lived)
        seen = first + pd.Timedelta(days=lived)
        out.append({
            "olx_id": f"id{start + i}",
            "brand": brand, "model": model, "price_eur": price, "year": year,
            "district": district, "is_active": active,
            "first_seen_at": first,
            "last_scraped_at": seen,
            "deactivated_at": pd.NaT if active else seen + pd.Timedelta(days=outage_lag),
            "first_price_eur": first_price if first_price is not None else price,
        })
    return out


def _watchers(n=8, days=150, **kw):
    """Listings still on sale after ``days``, which is what gives a group the
    follow-up its 30/60/90-day shares are read off. Every real group has them."""
    return _rows(n, days, active=True, start=kw.pop("start", 9000), **kw)


def _df(*groups):
    return pd.DataFrame([r for g in groups for r in g])


class TestSurvival:
    def test_curve_matches_kaplan_meier_by_hand(self):
        times, surv = survival(np.array([1.0, 2.0, 3.0, 4.0]),
                               np.array([True, False, True, True]))
        assert list(times) == [1.0, 3.0, 4.0]
        assert surv[0] == pytest.approx(0.75)
        assert surv[1] == pytest.approx(0.75 * 0.5)
        assert surv[2] == pytest.approx(0.0)

    def test_a_row_still_on_sale_counts_in_the_risk_set_but_never_as_an_event(self):
        _, censored = survival(np.array([5.0, 9.0]), np.array([True, False]))
        _, ended = survival(np.array([5.0, 9.0]), np.array([True, True]))
        assert len(censored) == 1 and censored[0] == pytest.approx(0.5)
        assert len(ended) == 2 and ended[-1] == pytest.approx(0.0)


class TestCensoring:
    def test_listings_still_on_sale_slow_the_measured_market_down(self):
        ended = _rows(60, days=10)
        live = _rows(60, days=25, active=True, start=100)
        key = ("Volkswagen", "Golf")
        fast = build_liquidity(_df(ended, _watchers()), now=NOW)["models"][key]
        both = build_liquidity(_df(ended, live, _watchers()), now=NOW)["models"][key]
        assert fast["md"] == 10
        assert both["s30"] < fast["s30"]
        assert both["n"] == fast["n"] == 60
        assert both["cn"] == 68

    def test_a_median_the_curve_never_reaches_is_absent_not_guessed(self):
        ended = _rows(60, days=10)
        live = _rows(120, days=25, active=True, start=100)
        rec = build_liquidity(_df(ended, live, _watchers()), now=NOW)["models"][("Volkswagen", "Golf")]
        assert "md" not in rec
        assert rec["s30"] < 0.5

    def test_the_share_gone_is_never_above_one_or_below_zero(self):
        rec = build_liquidity(_df(_rows(50, days=12), _rows(50, days=70, start=100),
                                  _watchers()), now=NOW)["models"][("Volkswagen", "Golf")]
        for key in ("s30", "s60", "s90"):
            assert 0.0 <= rec[key] <= 1.0
        assert rec["s30"] <= rec["s60"] <= rec["s90"]


class TestOutageInflation:
    def test_the_sweep_that_noticed_does_not_become_the_day_it_ended(self):
        clean = build_liquidity(_df(_rows(60, days=12), _watchers()), now=NOW)
        stamped = build_liquidity(_df(_rows(60, days=12, outage_lag=13), _watchers()), now=NOW)
        key = ("Volkswagen", "Golf")
        assert stamped["models"][key]["md"] == clean["models"][key]["md"] == 12
        assert stamped["models"][key]["s30"] == clean["models"][key]["s30"]

    def test_a_row_the_scraper_never_confirmed_falls_back_to_the_sweep(self):
        rows = _rows(50, days=20)
        for r in rows:
            r["last_scraped_at"] = pd.NaT
        rec = build_liquidity(_df(rows, _watchers()), now=NOW)["models"][("Volkswagen", "Golf")]
        assert rec["md"] == 20


class TestGates:
    def test_a_model_below_the_page_floor_gets_no_page(self):
        liq = build_liquidity(_df(_rows(MIN_EVENTS - 1, days=10), _watchers()), now=NOW)
        assert ("Volkswagen", "Golf") in liq["models"]
        assert page_records(liq) == {}

    def test_a_thin_cut_is_absent_rather_than_estimated(self):
        deep = _rows(MIN_CELL_EVENTS + 10, days=10, price=3000)
        thin = _rows(MIN_CELL_EVENTS - 1, days=40, price=30000, start=500)
        rec = build_liquidity(_df(deep, thin, _watchers(price=3000)),
                              now=NOW)["models"][("Volkswagen", "Golf")]
        keys = {c["k"] for c in rec.get("pb", [])}
        assert "lt5" in keys
        assert "gt20" not in keys

    def test_listings_older_than_the_window_are_not_the_market_of_today(self):
        old = _rows(60, days=10)
        for r in old:
            r["first_seen_at"] = NOW - pd.Timedelta(days=WINDOW_DAYS + 40)
            r["last_scraped_at"] = r["first_seen_at"] + pd.Timedelta(days=10)
            r["deactivated_at"] = r["last_scraped_at"]
        assert build_liquidity(_df(old), now=NOW)["models"] == {}

    def test_a_negative_or_absurd_duration_is_dropped(self):
        rows = _rows(40, days=10)
        rows[0]["last_scraped_at"] = rows[0]["first_seen_at"] - pd.Timedelta(days=3)
        rows[1]["first_seen_at"] = NOW - pd.Timedelta(days=5000)
        kept = prepare(_df(rows), now=NOW)
        assert len(kept) == 38


class TestCuts:
    def test_a_price_band_that_moves_slower_shows_as_slower(self):
        cheap = _rows(60, days=8, price=3000)
        dear = _rows(60, days=80, price=30000, start=500)
        rec = build_liquidity(_df(cheap, dear, _watchers(price=3000),
                                  _watchers(price=30000, start=9500)),
                              now=NOW)["models"][("Volkswagen", "Golf")]
        cells = {c["k"]: c for c in rec["pb"]}
        assert cells["lt5"]["s30"] > cells["gt20"]["s30"]
        assert cells["lt5"]["n"] == 60

    def test_a_district_without_a_name_never_becomes_a_cell(self):
        rows = _rows(60, days=10, district=None)
        rec = build_liquidity(_df(rows, _watchers(district=None)),
                              now=NOW)["models"][("Volkswagen", "Golf")]
        assert not rec.get("dt")


class TestRelistsAndDiscounts:
    def test_the_relist_share_counts_only_listings_that_came_back(self):
        rows = _rows(MIN_EVENTS + 10, days=10)
        back = {r["olx_id"] for r in rows[:10]}
        rec = build_liquidity(_df(rows, _watchers()), relisted=back,
                              now=NOW)["models"][("Volkswagen", "Golf")]
        assert rec["rb"] == pytest.approx(10 / len(rows), abs=0.01)

    def test_the_discount_is_measured_against_the_first_price_we_saw(self):
        cut = _rows(60, days=40, price=9000, first_price=10000)
        held = _rows(60, days=10, price=9000, first_price=9000, start=500)
        rec = build_liquidity(_df(cut, held, _watchers(price=9000)),
                              now=NOW)["models"][("Volkswagen", "Golf")]
        assert rec["cu"] == pytest.approx(0.5, abs=0.01)
        assert rec["cp"] == pytest.approx(0.1, abs=0.001)
        assert rec["cd"] > rec["hd"]


class TestSellSpeedFrame:
    def test_it_carries_the_same_median_the_pages_publish(self):
        liq = build_liquidity(_df(_rows(60, days=10), _watchers()), now=NOW)
        frame = sell_speed_frame(liq)
        assert list(frame.columns) == ["brand", "model", "sell_days", "sell_n"]
        assert frame.iloc[0]["sell_days"] == liq["models"][("Volkswagen", "Golf")]["md"]

    def test_a_segment_below_the_floor_is_dropped_not_shown_noisy(self):
        liq = build_liquidity(_df(_rows(4, days=10), _watchers()), now=NOW)
        assert sell_speed_frame(liq).empty

    def test_an_empty_corpus_returns_the_empty_shape(self):
        assert build_liquidity(pd.DataFrame(), now=NOW)["models"] == {}
        assert sell_speed_frame({}).empty
