"""Tests for the per-listing hazard model (schema v2)."""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.analytics.hazard import (
    CATEGORICAL_FEATURES,
    DEFAULT_HORIZON_DAYS,
    NUMERIC_FEATURES,
    PREDICTION_FEATURES,
    _apply_segment_imputation,
    _build_features,
    _build_segment_lookups,
    _build_target,
    _segment_lookup,
    predict_sold_probability,
    train_hazard_model,
)


_NOW = pd.Timestamp("2026-05-04 00:00:00", tz="UTC")


def _row(**kw) -> dict:
    base = {
        "olx_id": "x1",
        "brand": "Volkswagen",
        "model": "Golf",
        "generation": "Mk7",
        "year": 2018,
        "mileage_km": 80_000,
        "engine_cc": 1968,
        "horsepower": 150,
        "price_eur": 14_000.0,
        "fuel_type": "Diesel",
        "transmission": "Manual",
        "segment": "Citadino",
        "district": "Porto",
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
    assert labeled[0] and y[0] == 1


def test_target_sold_past_horizon_is_negative():
    df = pd.DataFrame([_row(
        is_active=False, deactivation_reason="sold",
        first_seen_at=_NOW - pd.Timedelta(days=120),
        deactivated_at=_NOW - pd.Timedelta(days=30),  # sold 90d in
    )])
    y, labeled = _build_target(df, horizon_days=30, now=_NOW)
    assert labeled[0] and y[0] == 0


def test_target_active_old_listing_is_negative():
    df = pd.DataFrame([_row(
        is_active=True, first_seen_at=_NOW - pd.Timedelta(days=90),
    )])
    y, labeled = _build_target(df, horizon_days=30, now=_NOW)
    assert labeled[0] and y[0] == 0


def test_target_active_young_listing_is_censored():
    df = pd.DataFrame([_row(
        is_active=True, first_seen_at=_NOW - pd.Timedelta(days=10),
    )])
    y, labeled = _build_target(df, horizon_days=30, now=_NOW)
    assert not labeled[0]


def test_target_deactivated_within_horizon_non_sold_reason_is_censored():
    df = pd.DataFrame([_row(
        is_active=False, deactivation_reason="expired",
        first_seen_at=_NOW - pd.Timedelta(days=60),
        deactivated_at=_NOW - pd.Timedelta(days=50),
    )])
    y, labeled = _build_target(df, horizon_days=30, now=_NOW)
    assert not labeled[0]


def test_target_deactivated_past_horizon_non_sold_reason_is_negative():
    df = pd.DataFrame([_row(
        is_active=False, deactivation_reason="expired",
        first_seen_at=_NOW - pd.Timedelta(days=120),
        deactivated_at=_NOW - pd.Timedelta(days=30),
    )])
    y, labeled = _build_target(df, horizon_days=30, now=_NOW)
    assert labeled[0] and y[0] == 0


def test_target_missing_first_seen_is_censored():
    df = pd.DataFrame([_row(first_seen_at=None)])
    y, labeled = _build_target(df, horizon_days=30, now=_NOW)
    assert not labeled[0]


def test_target_negative_days_to_deact_dropped():
    df = pd.DataFrame([_row(
        is_active=False, deactivation_reason="sold",
        first_seen_at=_NOW - pd.Timedelta(days=10),
        deactivated_at=_NOW - pd.Timedelta(days=20),
    )])
    y, labeled = _build_target(df, horizon_days=30, now=_NOW)
    assert not labeled[0]


# ---------------------------------------------------------------------------
# Per-segment imputation
# ---------------------------------------------------------------------------


def test_segment_lookups_layered_fallback():
    """A 5+ row brand/model bucket gets its own median; a single
    odd row falls through to a coarser key."""
    rows = []
    # 6 Golfs at 1968 cc → solid (brand, model) imputation source
    for i in range(6):
        rows.append(_row(olx_id=f"vw_g_{i}", engine_cc=1968))
    # One stray BMW alone — too few for a brand bucket → falls through
    rows.append(_row(
        olx_id="bmw_alone", brand="BMW", model="320", engine_cc=1995,
    ))
    df = pd.DataFrame(rows)
    X = pd.DataFrame({
        col: pd.to_numeric(df[col], errors="coerce") if col in df.columns else np.nan
        for col in (
            "year", "mileage_km", "engine_cc", "horsepower",
            "photo_count", "description_length", "damage_severity",
        )
    })
    lookups = _build_segment_lookups(df, X)
    # VW + Golf qualifies as a brand_model bucket (6 rows)
    assert ("Volkswagen", "Golf") in lookups["brand_model"]
    # BMW alone — not enough rows; brand_model bucket absent
    assert ("BMW", "320") not in lookups["brand_model"]


def test_segment_lookup_walks_chain():
    lookups = {
        "global": {"engine_cc": 1500.0},
        "brand": {("Volkswagen",): {"engine_cc": 1800.0}},
        "brand_model": {("Volkswagen", "Golf"): {"engine_cc": 1968.0}},
        "segment": {},
    }
    # Most specific available → 1968
    assert _segment_lookup(lookups, "Volkswagen", "Golf", "Mk7", "engine_cc") == 1968.0
    # Falls to brand
    assert _segment_lookup(lookups, "Volkswagen", "Polo", "Mk6", "engine_cc") == 1800.0
    # Falls to global
    assert _segment_lookup(lookups, "BMW", "320", "E90", "engine_cc") == 1500.0
    # Missing feature → 0.0
    assert _segment_lookup(lookups, "Volkswagen", "Golf", "Mk7", "horsepower") == 0.0


def test_apply_imputation_fills_per_segment():
    rows = [_row(olx_id=f"g_{i}", engine_cc=1968) for i in range(6)]
    rows.append(_row(olx_id="missing", engine_cc=None))
    df = pd.DataFrame(rows)
    X = pd.DataFrame({
        col: pd.to_numeric(df[col], errors="coerce") if col in df.columns else np.nan
        for col in (
            "year", "mileage_km", "engine_cc", "horsepower",
            "photo_count", "description_length", "damage_severity",
        )
    })
    lookups = _build_segment_lookups(df, X)
    out = _apply_segment_imputation(df, X, lookups)
    # The NaN row picked up the brand_model median = 1968
    assert out.loc[6, "engine_cc"] == 1968.0


# ---------------------------------------------------------------------------
# _build_features
# ---------------------------------------------------------------------------


def test_features_shape_includes_categoricals():
    df = pd.DataFrame([_row()])
    X, info = _build_features(df, predictions_df=None)
    assert info["feature_names"] == NUMERIC_FEATURES + CATEGORICAL_FEATURES
    assert set(info["categorical_features"]) == set(CATEGORICAL_FEATURES)
    for col in CATEGORICAL_FEATURES:
        assert isinstance(X[col].dtype, pd.CategoricalDtype)


def test_features_with_predictions_extends_columns():
    df = pd.DataFrame([_row()])
    preds = pd.DataFrame({
        "predicted_price": [15_000.0],
        "fair_price_low": [13_000.0],
        "fair_price_high": [17_000.0],
    }, index=df.index)
    X, info = _build_features(df, predictions_df=preds)
    assert info["feature_names"] == NUMERIC_FEATURES + CATEGORICAL_FEATURES + PREDICTION_FEATURES
    assert X["residual_pct"].iloc[0] == pytest.approx(-6.67, abs=0.1)
    assert X["band_pct"].iloc[0] == pytest.approx(26.67, abs=0.1)


def test_features_zero_predicted_price_yields_nan():
    df = pd.DataFrame([_row()])
    preds = pd.DataFrame({
        "predicted_price": [0.0], "fair_price_low": [0.0], "fair_price_high": [0.0],
    }, index=df.index)
    X, _ = _build_features(df, predictions_df=preds)
    assert pd.isna(X["residual_pct"].iloc[0])
    assert pd.isna(X["band_pct"].iloc[0])


def test_features_imputes_missing_numeric():
    """v2 contract: missing photo_count / engine_cc gets segment-imputed
    rather than dropping the row from the matrix."""
    rows = [_row(olx_id=f"g_{i}") for i in range(6)]
    rows.append(_row(olx_id="missing_specs", engine_cc=None, photo_count=None))
    df = pd.DataFrame(rows)
    X, info = _build_features(df)
    # Both filled from per-segment median
    assert pd.notna(X.loc[6, "engine_cc"])
    assert pd.notna(X.loc[6, "photo_count"])
    assert info["lookups"] is not None


def test_features_uses_supplied_categorical_levels():
    """Predict-time: unknown category values should not produce a new
    level — they map to NaN, preserving the train-time encoding."""
    train_df = pd.DataFrame([_row(olx_id=f"t_{i}", brand="Volkswagen") for i in range(6)])
    _, train_info = _build_features(train_df)
    levels = train_info["categorical_levels"]
    # New row with an unseen brand
    new_df = pd.DataFrame([_row(olx_id="new", brand="ZZZ_NewMake")])
    X_new, _ = _build_features(
        new_df, lookups=train_info["lookups"], categorical_levels=levels,
    )
    # The unseen brand drops to NaN under the fixed level set
    assert pd.isna(X_new["brand"].iloc[0])


# ---------------------------------------------------------------------------
# Training / prediction
# ---------------------------------------------------------------------------


def _synth_corpus(n: int = 400) -> pd.DataFrame:
    """Synthetic corpus where price drives sale speed: cheap → fast."""
    rng = np.random.default_rng(42)
    rows = []
    for i in range(n):
        first_seen = _NOW - pd.Timedelta(days=int(rng.integers(35, 120)))
        if i % 2 == 0:
            price = float(rng.uniform(8_000, 12_000))
            deactivated_at = first_seen + pd.Timedelta(days=int(rng.uniform(5, 25)))
            is_active = False
            reason = "sold"
        else:
            price = float(rng.uniform(16_000, 22_000))
            deactivated_at = None
            is_active = True
            reason = None
        rows.append({
            "olx_id": f"r{i}",
            "brand": "Volkswagen", "model": "Golf", "generation": "Mk7",
            "year": 2018, "mileage_km": 80_000 + int(rng.normal(0, 5_000)),
            "engine_cc": 1968, "horsepower": 150,
            "price_eur": price,
            "fuel_type": "Diesel", "transmission": "Manual",
            "segment": "Citadino", "district": "Porto",
            "photo_count": int(rng.integers(5, 12)),
            "description_length": int(rng.integers(300, 800)),
            "damage_severity": 0,
            "is_active": is_active, "deactivated_at": deactivated_at,
            "deactivation_reason": reason, "first_seen_at": first_seen,
        })
    return pd.DataFrame(rows)


def test_train_returns_none_on_empty():
    assert train_hazard_model(pd.DataFrame()) is None


def test_train_returns_none_under_min_samples():
    df = _synth_corpus(50)
    assert train_hazard_model(df, min_samples=200) is None


def test_train_produces_v2_bundle():
    df = _synth_corpus(400)
    bundle = train_hazard_model(df)
    assert bundle is not None
    assert bundle["schema_version"] == 2
    assert bundle["horizon_days"] == DEFAULT_HORIZON_DAYS
    assert bundle["uses_predictions"] is False
    assert bundle["feature_names"] == NUMERIC_FEATURES + CATEGORICAL_FEATURES
    assert "lookups" in bundle and "categorical_levels" in bundle
    metrics = bundle["metrics"]
    # Synth corpus has a clean price signal — model should easily separate.
    assert metrics["auc"] >= 0.85


def test_train_with_predictions_extends_features():
    df = _synth_corpus(400)
    preds = pd.DataFrame({
        "predicted_price": np.full(len(df), 14_000.0),
        "fair_price_low": np.full(len(df), 12_000.0),
        "fair_price_high": np.full(len(df), 16_000.0),
    }, index=df.index)
    bundle = train_hazard_model(df, predictions_df=preds)
    assert bundle["uses_predictions"] is True
    assert bundle["feature_names"] == (
        NUMERIC_FEATURES + CATEGORICAL_FEATURES + PREDICTION_FEATURES
    )


def test_predict_returns_probabilities_in_unit_interval():
    df = _synth_corpus(400)
    bundle = train_hazard_model(df)
    out = predict_sold_probability(bundle, df)
    valid = out["prob_sold_within_horizon"].dropna()
    assert (valid >= 0).all() and (valid <= 1).all()


def test_predict_cheap_listings_get_higher_probability():
    df = _synth_corpus(400)
    bundle = train_hazard_model(df)
    cheap = pd.DataFrame([_row(price_eur=9_000.0, olx_id="cheap")])
    pricey = pd.DataFrame([_row(price_eur=20_000.0, olx_id="pricey")])
    p_cheap = predict_sold_probability(bundle, cheap).iloc[0]["prob_sold_within_horizon"]
    p_pricey = predict_sold_probability(bundle, pricey).iloc[0]["prob_sold_within_horizon"]
    assert p_cheap > p_pricey


def test_predict_imputes_missing_numerics():
    """v2 contract: a fresh listing with missing photo_count or
    engine_cc still gets a probability — features come from the
    bundle's per-segment imputation table."""
    df = _synth_corpus(400)
    bundle = train_hazard_model(df)
    # Brand-new row missing the two NaN-prone fields from production
    bad = pd.DataFrame([_row(
        olx_id="bad", price_eur=10_000.0,
        photo_count=None, engine_cc=None,
    )])
    out = predict_sold_probability(bundle, bad)
    assert pd.notna(out["prob_sold_within_horizon"].iloc[0])


def test_predict_returns_nan_when_price_missing():
    """Without price, no log_price → can't score. Other NaN fields
    get imputed; price NaN is the only remaining drop reason."""
    df = _synth_corpus(400)
    bundle = train_hazard_model(df)
    bad = pd.DataFrame([_row(price_eur=None, olx_id="no_price")])
    out = predict_sold_probability(bundle, bad)
    assert pd.isna(out["prob_sold_within_horizon"].iloc[0])


def test_predict_feature_mismatch_raises():
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
