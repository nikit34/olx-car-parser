"""Publishing your own error is only worth doing if the number is the real one,
so the sample must be the cars whose ask the market actually took — not the
ones still arguing, not the ones that gave up after half a year.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.analytics.accuracy import accepted_price_rows, accuracy_by_band


def _car(i, ask, err_pct=0.0, days=40, cut=False, active=False):
    return {"car_id": f"c{i}", "last_ask": ask, "first_ask": ask, "days": days,
            "cut": cut, "still_active": active, "ask_change_pct": -8.0 if cut else 0.0,
            "_pred": ask * (1 + err_pct / 100)}


def _corpus(rows):
    df = pd.DataFrame(rows)
    preds = dict(zip(df["car_id"], df["_pred"]))
    return df.drop(columns=["_pred"]), preds


def test_the_sample_is_the_cars_whose_ask_the_market_took():
    df, _ = _corpus([_car(1, 10000), _car(2, 10000, cut=True),
                     _car(3, 10000, active=True), _car(4, 10000, days=300)])
    kept = accepted_price_rows(df)
    assert list(kept["car_id"]) == ["c1"]


def test_the_error_is_the_median_gap_and_the_bias_keeps_its_sign():
    rows = ([_car(i, 10000, err_pct=20.0) for i in range(150)]
            + [_car(500 + i, 10000, err_pct=-20.0) for i in range(150)])
    df, preds = _corpus(rows)
    band = accuracy_by_band(df, preds, min_cars=200)[0]
    assert band["n"] == 300
    assert band["err"] == pytest.approx(20.0, abs=0.1)
    assert band["bias"] == pytest.approx(0.0, abs=0.1)
    assert band["within"] == 0.0


def test_a_band_that_lands_close_says_so():
    df, preds = _corpus([_car(i, 20000, err_pct=5.0) for i in range(250)])
    band = accuracy_by_band(df, preds, min_cars=200)[0]
    assert band["lbl"] == "€15.000 a €25.000"
    assert band["within"] == 1.0
    assert band["bias"] == pytest.approx(5.0, abs=0.1)


def test_bands_are_cut_by_the_accepted_price_not_by_the_estimate():
    rows = [_car(i, 3900, err_pct=60.0) for i in range(250)]
    df, preds = _corpus(rows)
    bands = accuracy_by_band(df, preds, min_cars=200)
    assert [b["lo"] for b in bands] == [0]
    assert bands[0]["err"] == pytest.approx(60.0, abs=0.1)


def test_a_thin_band_is_absent_and_a_car_without_an_estimate_is_dropped():
    df, preds = _corpus([_car(i, 30000) for i in range(199)])
    assert accuracy_by_band(df, preds, min_cars=200) == []
    df2, _ = _corpus([_car(i, 10000) for i in range(250)])
    assert accuracy_by_band(df2, {}, min_cars=200) == []


def test_an_empty_corpus_publishes_nothing():
    assert accuracy_by_band(pd.DataFrame(), {"a": 1.0}) == []
    assert accepted_price_rows(pd.DataFrame()).empty
