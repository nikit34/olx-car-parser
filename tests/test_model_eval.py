"""Tests for model_eval — diagnostics computed from bundled OOF predictions."""

import json
import numpy as np
import pandas as pd
import pytest

from src.analytics.model_eval import (
    evaluate_oof, worst_residuals, reliability_curve, time_backtest,
    save_backtest, load_backtest,
)
from src.analytics.price_model import train_price_model


def _sample_listings(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Synthetic listings shaped like the price-model fixture, with first_seen_at
    spread across 60 days so time_backtest has something to slice."""
    rng = np.random.RandomState(seed)
    years = rng.randint(2008, 2024, size=n)
    mileage = rng.randint(10000, 300000, size=n)
    engine_cc = rng.choice([1000, 1200, 1400, 1600, 2000], size=n)
    brands = rng.choice(["Volkswagen", "Renault", "BMW"], size=n)
    models = []
    for b in brands:
        if b == "Volkswagen":
            models.append(rng.choice(["Golf", "Polo"]))
        elif b == "Renault":
            models.append(rng.choice(["Clio", "Megane"]))
        else:
            models.append("320d")

    accident = rng.choice([True, False, None], size=n, p=[0.05, 0.75, 0.20])
    price = (
        (years - 2000) * 600
        - mileage * 0.02
        + engine_cc * 2
        + rng.normal(0, 1000, size=n)
    ).clip(min=500)
    for i in range(n):
        if accident[i] is True:
            price[i] *= 0.7

    base_ts = pd.Timestamp("2026-01-01", tz="UTC")
    first_seen = [base_ts + pd.Timedelta(days=int(d)) for d in rng.randint(0, 60, size=n)]

    return pd.DataFrame({
        "olx_id": [f"t{i}" for i in range(n)],
        "brand": brands,
        "model": models,
        "year": years,
        "mileage_km": mileage,
        "engine_cc": engine_cc,
        "price_eur": price.round(0),
        "fuel_type": rng.choice(["Diesel", "Gasoline"], size=n),
        "transmission": rng.choice(["Manual", "Automatic"], size=n),
        "segment": "Citadino",
        "horsepower": (engine_cc * 0.07 + rng.normal(0, 10, size=n)).clip(min=50).round(0),
        "is_active": True,
        "first_seen_at": first_seen,
        "desc_mentions_accident": accident,
        "desc_mentions_repair": rng.choice([True, False, None], size=n, p=[0.10, 0.70, 0.20]),
        "right_hand_drive": rng.choice([True, False, None], size=n, p=[0.02, 0.78, 0.20]),
        "desc_mentions_customs_cleared": rng.choice([True, False, None], size=n, p=[0.15, 0.65, 0.20]),
        "desc_mentions_num_owners": rng.choice([1, 2, 3, None], size=n, p=[0.3, 0.2, 0.1, 0.4]),
        "taxi_fleet_rental": rng.choice([True, False, None], size=n, p=[0.03, 0.77, 0.20]),
        "warranty": rng.choice([True, False, None], size=n, p=[0.15, 0.65, 0.20]),
        "first_owner_selling": rng.choice([True, False, None], size=n, p=[0.20, 0.60, 0.20]),
        "urgency": rng.choice(["high", "medium", "low", None], size=n, p=[0.05, 0.15, 0.50, 0.30]),
    })


@pytest.fixture(scope="module")
def trained():
    """Train once per module — train_price_model takes ~15s on the fixture.

    600 rows is enough that filter losses + 4-fold backtest still leave each
    fold above the 100-row train minimum that time_backtest requires.
    """
    df = _sample_listings(600)
    models, cat_maps, metrics, oof_preds, calibrator = train_price_model(df)
    return df, models, cat_maps, metrics, oof_preds, calibrator


def test_evaluate_oof_global_metrics(trained):
    df, _models, _maps, _metrics, oof, _calib = trained
    report = evaluate_oof(df, oof)
    g = report["global"]
    # Every training row appears in oof_preds (filter dropped <1%), so
    # global n should be close to len(df).
    assert g["n"] > 0
    assert g["n"] <= len(df)
    assert g["mae"] >= 0
    assert g["mape"] >= 0
    assert -1.0 <= g["r2"] <= 1.0
    assert 0.0 <= g["coverage_80"] <= 1.0
    # Sort step in train_price_model already repaired any crossings.
    assert g["n_inverted_band"] == 0


def test_evaluate_oof_buckets_sum_to_global(trained):
    """Sum of bucket sample counts must equal global n (each row → exactly one
    bucket; there are no gaps in the price/year ranges)."""
    df, _models, _maps, _metrics, oof, _calib = trained
    report = evaluate_oof(df, oof)
    g = report["global"]
    assert int(report["by_price"]["n"].sum()) == g["n"]
    assert int(report["by_year"]["n"].sum()) == g["n"]


def test_evaluate_oof_returns_empty_on_no_overlap():
    """Listings with olx_ids that don't appear in oof_preds get filtered out."""
    df = _sample_listings(100)
    oof = {"unrelated_id": (1.0, 2.0, 3.0)}
    report = evaluate_oof(df, oof)
    assert report["global"]["n"] == 0
    assert report["by_price"].empty


def test_worst_residuals_sorted_desc(trained):
    df, _models, _maps, _metrics, oof, _calib = trained
    worst = worst_residuals(df, oof, n=10)
    assert len(worst) <= 10
    assert len(worst) > 0
    # Strictly non-increasing in |residual %|
    pcts = worst["abs_residual_pct"].values
    assert all(pcts[i] >= pcts[i + 1] for i in range(len(pcts) - 1))


def test_reliability_curve_bin_count(trained):
    df, _models, _maps, _metrics, oof, _calib = trained
    rel = reliability_curve(df, oof, n_bins=10)
    # qcut may collapse duplicate edges, so n_bins is an upper bound.
    assert 1 <= len(rel) <= 10
    # Every bin's empirical_coverage is in [0,1]
    assert ((rel["empirical_coverage"] >= 0) & (rel["empirical_coverage"] <= 1)).all()
    # calibration_gap = empirical - 0.80
    assert np.allclose(rel["calibration_gap"], rel["empirical_coverage"] - 0.80)


def test_time_backtest_returns_one_row_per_fold(trained):
    df, _models, _maps, metrics, _oof, _calib = trained
    n_per_q = metrics.get("best_n_estimators_per_q", {n: 100 for n in ("low", "median", "high")})
    bt = time_backtest(df, n_splits=4, n_estimators_per_q=n_per_q)
    # 4 splits → 3 folds (skip the first slice, which has no train data)
    assert len(bt) == 3
    assert set(bt.columns) >= {
        "fold", "train_until", "test_from", "test_to",
        "n_train", "n_test", "mae", "mape", "bias_pct", "coverage_80",
    }
    # Folds use a growing train set
    n_trains = bt["n_train"].tolist()
    assert all(n_trains[i] < n_trains[i + 1] for i in range(len(n_trains) - 1))


def test_time_backtest_requires_first_seen_at():
    df = _sample_listings(100).drop(columns=["first_seen_at"])
    with pytest.raises(ValueError, match="first_seen_at"):
        time_backtest(df)


def test_save_load_backtest_round_trip(tmp_path, monkeypatch, trained):
    monkeypatch.setattr(
        "src.analytics.model_eval._BACKTEST_PATH",
        tmp_path / "price_backtest.json",
    )
    df, _models, _maps, metrics, _oof, _calib = trained
    n_per_q = metrics.get("best_n_estimators_per_q", {n: 100 for n in ("low", "median", "high")})
    bt = time_backtest(df, n_splits=3, n_estimators_per_q=n_per_q)

    save_backtest(bt)
    loaded = load_backtest()
    assert loaded is not None
    assert "generated_at" in loaded
    assert len(loaded["folds"]) == len(bt)
    if not bt.empty:
        assert loaded["folds"][0]["fold"] == int(bt.iloc[0]["fold"])


def test_load_backtest_returns_none_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.analytics.model_eval._BACKTEST_PATH",
        tmp_path / "nonexistent.json",
    )
    assert load_backtest() is None
