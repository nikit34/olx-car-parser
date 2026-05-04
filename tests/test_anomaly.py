"""Tests for IsolationForest anomaly detection."""

import numpy as np
import pandas as pd
import pytest

from src.analytics.anomaly import (
    BASE_FEATURES,
    PREDICTION_FEATURES,
    _build_features,
    score_anomalies,
    train_anomaly_detector,
)


def _normal_listing(i: int, **kw) -> dict:
    """Build a 'normal' VW Golf listing — small variation across rows
    keeps IsolationForest from collapsing to a degenerate fit."""
    base = {
        "olx_id": f"normal_{i}",
        "brand": "Volkswagen",
        "model": "Golf",
        "year": 2018 + (i % 3),
        "mileage_km": 80_000 + i * 500,
        "price_eur": 14_000.0 + i * 50,
        "engine_cc": 1968,
        "horsepower": 150,
        "fuel_type": "Diesel",
        "transmission": "Manual",
        "color": "Cinzento",
        "district": "Porto",
        "photo_count": 8 + (i % 5),
        "description_length": 500 + i * 10,
        "seats": 5,
    }
    base.update(kw)
    return base


def _make_corpus(n_normal: int = 150) -> pd.DataFrame:
    return pd.DataFrame([_normal_listing(i) for i in range(n_normal)])


# ---------------------------------------------------------------------------
# _build_features
# ---------------------------------------------------------------------------


def test_build_features_base_only():
    df = _make_corpus(20)
    X, features = _build_features(df, predictions_df=None)
    assert features == BASE_FEATURES
    assert list(X.columns) == BASE_FEATURES
    assert len(X) == len(df)
    # log_price should be reasonable (log1p(14k) ≈ 9.5)
    assert X["log_price"].mean() == pytest.approx(np.log1p(14_500), rel=0.1)


def test_build_features_with_predictions():
    df = _make_corpus(20)
    preds = pd.DataFrame({
        "predicted_price": np.full(20, 15_000.0),
        "fair_price_low": np.full(20, 13_000.0),
        "fair_price_high": np.full(20, 17_000.0),
    }, index=df.index)
    X, features = _build_features(df, predictions_df=preds)
    assert features == BASE_FEATURES + PREDICTION_FEATURES
    assert "residual_pct" in X.columns
    assert "band_pct" in X.columns
    # band_pct should be ~26.7 % (=4000/15000*100), uniform across rows
    assert X["band_pct"].iloc[0] == pytest.approx(26.67, abs=0.1)


def test_build_features_handles_zero_predicted_price():
    """A row with predicted_price=0 should yield NaN residual/band, not /0."""
    df = pd.DataFrame([_normal_listing(0)])
    preds = pd.DataFrame({
        "predicted_price": [0.0],
        "fair_price_low": [0.0],
        "fair_price_high": [0.0],
    }, index=df.index)
    X, _ = _build_features(df, predictions_df=preds)
    assert pd.isna(X["residual_pct"].iloc[0])
    assert pd.isna(X["band_pct"].iloc[0])


# ---------------------------------------------------------------------------
# train_anomaly_detector
# ---------------------------------------------------------------------------


def test_train_returns_none_on_empty():
    assert train_anomaly_detector(pd.DataFrame()) is None


def test_train_returns_none_under_min_samples():
    df = _make_corpus(10)
    assert train_anomaly_detector(df, min_samples=100) is None


def test_train_produces_valid_bundle():
    df = _make_corpus(150)
    bundle = train_anomaly_detector(df, contamination=0.05)
    assert bundle is not None
    assert bundle["schema_version"] == 1
    assert bundle["contamination"] == 0.05
    assert bundle["uses_predictions"] is False
    assert bundle["n_samples"] == 150
    assert bundle["feature_names"] == BASE_FEATURES
    assert "model" in bundle and "scaler" in bundle
    assert bundle["raw_min"] < bundle["raw_max"]


def test_train_with_predictions_extends_features():
    df = _make_corpus(150)
    preds = pd.DataFrame({
        "predicted_price": np.full(150, 14_500.0),
        "fair_price_low": np.full(150, 12_500.0),
        "fair_price_high": np.full(150, 16_500.0),
    }, index=df.index)
    bundle = train_anomaly_detector(df, predictions_df=preds)
    assert bundle["uses_predictions"] is True
    assert bundle["feature_names"] == BASE_FEATURES + PREDICTION_FEATURES


# ---------------------------------------------------------------------------
# score_anomalies
# ---------------------------------------------------------------------------


def test_score_flags_extreme_outlier():
    """150 normal Golfs + 1 obvious outlier (price=100€, engine=6000cc,
    1 photo) should be flagged."""
    rows = [_normal_listing(i) for i in range(150)]
    rows.append({
        "olx_id": "weird",
        "brand": "Volkswagen", "model": "Golf",
        "year": 2018, "mileage_km": 80_000,
        "price_eur": 100.0,            # 1000× cheaper than others
        "engine_cc": 6000,             # absurd
        "horsepower": 600,             # absurd
        "fuel_type": "Diesel", "transmission": "Manual",
        "color": "Cinzento", "district": "Porto",
        "photo_count": 1,              # almost no photos
        "description_length": 5,       # tiny
        "seats": 2,                    # uncommon for Golf
    })
    df = pd.DataFrame(rows)
    bundle = train_anomaly_detector(df, contamination=0.05)
    scores = score_anomalies(bundle, df)
    weird = scores[scores["olx_id"] == "weird"].iloc[0]
    assert weird["is_anomaly"]
    assert weird["anomaly_score"] >= 0.8


def test_score_returns_nan_for_unfeaturisable_rows():
    df = _make_corpus(150)
    bundle = train_anomaly_detector(df, contamination=0.05)
    # New listing with NaN year — can't be featurised
    bad = pd.DataFrame([_normal_listing(999, year=None)])
    scores = score_anomalies(bundle, bad)
    assert pd.isna(scores["anomaly_score"].iloc[0])
    assert scores["is_anomaly"].iloc[0] == False  # noqa: E712


def test_score_respects_feature_contract():
    """A bundle trained with predictions errors loudly when scored
    without — silent feature mismatch would produce garbage scores."""
    df = _make_corpus(150)
    preds = pd.DataFrame({
        "predicted_price": np.full(150, 14_500.0),
        "fair_price_low": np.full(150, 12_500.0),
        "fair_price_high": np.full(150, 16_500.0),
    }, index=df.index)
    bundle = train_anomaly_detector(df, predictions_df=preds)
    with pytest.raises(ValueError, match="feature mismatch"):
        score_anomalies(bundle, df, predictions_df=None)


def test_score_anomaly_score_range_clipped():
    """Test-time outliers worse than the training extreme should clip
    to 1.0, not exceed it."""
    df = _make_corpus(150)
    bundle = train_anomaly_detector(df, contamination=0.05)
    # Construct a row that's far weirder than anything in training
    extreme = pd.DataFrame([_normal_listing(
        999, price_eur=1.0, engine_cc=99_000, horsepower=10_000,
        photo_count=0, description_length=0,
    )])
    scores = score_anomalies(bundle, extreme)
    assert 0.0 <= scores["anomaly_score"].iloc[0] <= 1.0


def test_score_normal_listings_get_low_scores():
    df = _make_corpus(150)
    bundle = train_anomaly_detector(df, contamination=0.05)
    scores = score_anomalies(bundle, df)
    # Almost all normal rows should be below the threshold (=
    # is_anomaly=False); the contamination=0.05 threshold flags ~5 %.
    flagged_share = scores["is_anomaly"].mean()
    assert flagged_share <= 0.10  # generous bound for stochastic IF
