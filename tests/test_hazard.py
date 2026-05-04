"""Tests for the per-listing hazard model."""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.analytics.hazard import (
    DEFAULT_HORIZON_DAYS,
    NUMERIC_FEATURES,
    PREDICTION_FEATURES,
    _build_features,
    _build_target,
    predict_sold_probability,
    train_hazard_model,
)


# A fixed "now" anchor for deterministic age-based labelling.
_NOW = pd.Timestamp("2026-05-04 00:00:00", tz="UTC")


def _row(**kw) -> dict:
    """Baseline listing — overridden per test."""
    base = {
        "olx_id": "x1",
        "brand": "Volkswagen",
        "model": "Golf",
        "year": 2018,
        "mileage_km": 80_000,
        "engine_cc": 1968,
        "horsepower": 150,
        "price_eur": 14_000.0,
        "fuel_type": "Diesel",
        "transmission": "Manual",
        "photo_count": 8,
        "description_length": 500,
        "damage_severity": 0,
        "is_active": True,
        "deactivated_at": None,
        "deactivation_reason": None,
        "first_seen_at": _NOW - pd.Timedelta(days=60),
    }
    base.update(kw)
    return base


# ---------------------------------------------------------------------------
# _build_target
# ---------------------------------------------------------------------------


def test_target_sold_within_horizon_is_positive():
    df = pd.DataFrame([_row(
        is_active=False, deactivation_reason="sold",
        first_seen_at=_NOW - pd.Timedelta(days=60),
        deactivated_at=_NOW - pd.Timedelta(days=50),  # sold 10d in
    )])
    y, labeled = _build_target(df, horizon_days=30, now=_NOW)
    assert labeled[0]
    assert y[0] == 1


def test_target_sold_past_horizon_is_negative():
    df = pd.DataFrame([_row(
        is_active=False, deactivation_reason="sold",
        first_seen_at=_NOW - pd.Timedelta(days=120),
        deactivated_at=_NOW - pd.Timedelta(days=30),  # sold 90d in
    )])
    y, labeled = _build_target(df, horizon_days=30, now=_NOW)
    assert labeled[0]
    assert y[0] == 0


def test_target_active_old_listing_is_negative():
    """Still listed and survived past horizon → observably not sold."""
    df = pd.DataFrame([_row(
        is_active=True,
        first_seen_at=_NOW - pd.Timedelta(days=90),
    )])
    y, labeled = _build_target(df, horizon_days=30, now=_NOW)
    assert labeled[0]
    assert y[0] == 0


def test_target_active_young_listing_is_censored():
    """Active and only 10 days old < 30-day horizon → can't label yet."""
    df = pd.DataFrame([_row(
        is_active=True,
        first_seen_at=_NOW - pd.Timedelta(days=10),
    )])
    y, labeled = _build_target(df, horizon_days=30, now=_NOW)
    assert not labeled[0]


def test_target_deactivated_within_horizon_non_sold_reason_is_censored():
    """Removed for unknown reason within horizon — could be a parser
    misclassification or genuine removal. Drop to be safe."""
    df = pd.DataFrame([_row(
        is_active=False, deactivation_reason="expired",
        first_seen_at=_NOW - pd.Timedelta(days=60),
        deactivated_at=_NOW - pd.Timedelta(days=50),  # 10d in
    )])
    y, labeled = _build_target(df, horizon_days=30, now=_NOW)
    assert not labeled[0]


def test_target_deactivated_past_horizon_non_sold_reason_is_negative():
    """If the listing visibly survived past horizon, it didn't sell
    within horizon — the eventual reason for removal doesn't matter."""
    df = pd.DataFrame([_row(
        is_active=False, deactivation_reason="expired",
        first_seen_at=_NOW - pd.Timedelta(days=120),
        deactivated_at=_NOW - pd.Timedelta(days=30),  # 90d in
    )])
    y, labeled = _build_target(df, horizon_days=30, now=_NOW)
    assert labeled[0]
    assert y[0] == 0


def test_target_missing_first_seen_is_censored():
    df = pd.DataFrame([_row(first_seen_at=None)])
    y, labeled = _build_target(df, horizon_days=30, now=_NOW)
    assert not labeled[0]


def test_target_negative_days_to_deact_dropped():
    """deactivated_at < first_seen_at is parser nonsense; treat as
    censored (don't trust as positive even though reason='sold')."""
    df = pd.DataFrame([_row(
        is_active=False, deactivation_reason="sold",
        first_seen_at=_NOW - pd.Timedelta(days=10),
        deactivated_at=_NOW - pd.Timedelta(days=20),  # before first_seen
    )])
    y, labeled = _build_target(df, horizon_days=30, now=_NOW)
    assert not labeled[0]


# ---------------------------------------------------------------------------
# _build_features
# ---------------------------------------------------------------------------


def test_features_base_only():
    df = pd.DataFrame([_row()])
    X, features = _build_features(df, predictions_df=None)
    assert features == NUMERIC_FEATURES
    assert list(X.columns) == NUMERIC_FEATURES
    assert X["log_price"].iloc[0] == pytest.approx(np.log1p(14_000), rel=0.01)


def test_features_with_predictions():
    df = pd.DataFrame([_row()])
    preds = pd.DataFrame({
        "predicted_price": [15_000.0],
        "fair_price_low": [13_000.0],
        "fair_price_high": [17_000.0],
    }, index=df.index)
    X, features = _build_features(df, predictions_df=preds)
    assert features == NUMERIC_FEATURES + PREDICTION_FEATURES
    # residual_pct = (14000 - 15000) / 15000 × 100 ≈ -6.67%
    assert X["residual_pct"].iloc[0] == pytest.approx(-6.67, abs=0.1)
    # band_pct = 4000/15000 × 100 ≈ 26.7%
    assert X["band_pct"].iloc[0] == pytest.approx(26.67, abs=0.1)


def test_features_zero_predicted_price_yields_nan():
    df = pd.DataFrame([_row()])
    preds = pd.DataFrame({
        "predicted_price": [0.0], "fair_price_low": [0.0], "fair_price_high": [0.0],
    }, index=df.index)
    X, _ = _build_features(df, predictions_df=preds)
    assert pd.isna(X["residual_pct"].iloc[0])
    assert pd.isna(X["band_pct"].iloc[0])


# ---------------------------------------------------------------------------
# train_hazard_model
# ---------------------------------------------------------------------------


def _synth_corpus(n: int = 400) -> pd.DataFrame:
    """Synthetic corpus with a price-driven sale-speed signal:
    listings priced below median sell within horizon; above median
    survive past horizon. Spread first_seen_at across 90 days so the
    time-aware split has both folds."""
    rng = np.random.default_rng(42)
    rows = []
    for i in range(n):
        # Time anchor — spread across last 90 days
        first_seen = _NOW - pd.Timedelta(days=int(rng.integers(35, 120)))
        # Cheap half (price < 12000) sells in 5-25 days; expensive
        # half (price > 16000) survives past horizon. Add a touch of
        # noise so the model can't memorise.
        if i % 2 == 0:
            price = float(rng.uniform(8_000, 12_000))
            days_to_sell = int(rng.uniform(5, 25))
            deactivated_at = first_seen + pd.Timedelta(days=days_to_sell)
            is_active = False
            reason = "sold"
        else:
            price = float(rng.uniform(16_000, 22_000))
            deactivated_at = None
            is_active = True
            reason = None
        rows.append({
            "olx_id": f"r{i}",
            "brand": "Volkswagen", "model": "Golf",
            "year": 2018, "mileage_km": 80_000 + int(rng.normal(0, 5_000)),
            "engine_cc": 1968, "horsepower": 150,
            "price_eur": price,
            "fuel_type": "Diesel", "transmission": "Manual",
            "photo_count": int(rng.integers(5, 12)),
            "description_length": int(rng.integers(300, 800)),
            "damage_severity": 0,
            "is_active": is_active,
            "deactivated_at": deactivated_at,
            "deactivation_reason": reason,
            "first_seen_at": first_seen,
        })
    return pd.DataFrame(rows)


def test_train_returns_none_on_empty():
    assert train_hazard_model(pd.DataFrame()) is None


def test_train_returns_none_under_min_samples():
    df = _synth_corpus(50)
    assert train_hazard_model(df, min_samples=200) is None


def test_train_produces_valid_bundle():
    df = _synth_corpus(400)
    bundle = train_hazard_model(df)
    assert bundle is not None
    assert bundle["schema_version"] == 1
    assert bundle["horizon_days"] == DEFAULT_HORIZON_DAYS
    assert bundle["uses_predictions"] is False
    assert bundle["feature_names"] == NUMERIC_FEATURES
    metrics = bundle["metrics"]
    assert "auc" in metrics and "logloss" in metrics
    # Synth corpus has a clean price signal — model should trivially
    # separate the classes (AUC ≈ 1.0). Allow some slack for the
    # 80/20 time split corner cases.
    assert metrics["auc"] >= 0.85
    assert metrics["base_rate_train"] == pytest.approx(0.5, abs=0.15)


def test_train_with_predictions_extends_features():
    df = _synth_corpus(400)
    preds = pd.DataFrame({
        "predicted_price": np.full(len(df), 14_000.0),
        "fair_price_low": np.full(len(df), 12_000.0),
        "fair_price_high": np.full(len(df), 16_000.0),
    }, index=df.index)
    bundle = train_hazard_model(df, predictions_df=preds)
    assert bundle is not None
    assert bundle["uses_predictions"] is True
    assert bundle["feature_names"] == NUMERIC_FEATURES + PREDICTION_FEATURES


# ---------------------------------------------------------------------------
# predict_sold_probability
# ---------------------------------------------------------------------------


def test_predict_returns_probabilities_in_unit_interval():
    df = _synth_corpus(400)
    bundle = train_hazard_model(df)
    out = predict_sold_probability(bundle, df)
    valid = out["prob_sold_within_horizon"].dropna()
    assert (valid >= 0).all() and (valid <= 1).all()


def test_predict_cheap_listings_get_higher_probability():
    """The synthetic corpus encodes 'cheap → fast sale' — a fresh
    cheap listing should score higher than a fresh expensive one."""
    df = _synth_corpus(400)
    bundle = train_hazard_model(df)
    cheap = pd.DataFrame([_row(price_eur=9_000.0, olx_id="cheap")])
    pricey = pd.DataFrame([_row(price_eur=20_000.0, olx_id="pricey")])
    p_cheap = predict_sold_probability(bundle, cheap).iloc[0]["prob_sold_within_horizon"]
    p_pricey = predict_sold_probability(bundle, pricey).iloc[0]["prob_sold_within_horizon"]
    assert p_cheap > p_pricey


def test_predict_returns_nan_for_unfeaturisable_rows():
    df = _synth_corpus(400)
    bundle = train_hazard_model(df)
    bad = pd.DataFrame([_row(year=None, olx_id="bad")])
    out = predict_sold_probability(bundle, bad)
    assert pd.isna(out["prob_sold_within_horizon"].iloc[0])


def test_predict_feature_mismatch_raises():
    """Bundle trained with predictions cannot be applied without them."""
    df = _synth_corpus(400)
    preds = pd.DataFrame({
        "predicted_price": np.full(len(df), 14_000.0),
        "fair_price_low": np.full(len(df), 12_000.0),
        "fair_price_high": np.full(len(df), 16_000.0),
    }, index=df.index)
    bundle = train_hazard_model(df, predictions_df=preds)
    with pytest.raises(ValueError, match="feature mismatch"):
        predict_sold_probability(bundle, df, predictions_df=None)


def test_predict_empty_input_returns_empty():
    df = _synth_corpus(400)
    bundle = train_hazard_model(df)
    out = predict_sold_probability(bundle, pd.DataFrame())
    assert out.empty
    assert list(out.columns) == ["olx_id", "prob_sold_within_horizon"]
