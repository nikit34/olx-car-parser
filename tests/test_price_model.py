"""Tests for LightGBM price model."""

import json
import numpy as np
import pandas as pd
import pytest

from src.analytics.price_model import (
    train_price_model, predict_prices, save_model, load_model,
    load_metrics_history, _METRICS_PATH, _MODEL_PATH, _SCHEMA_VERSION,
)


def _sample_listings(n: int = 200) -> pd.DataFrame:
    """Generate synthetic listings with realistic price patterns."""
    rng = np.random.RandomState(42)
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
    repair = rng.choice([True, False, None], size=n, p=[0.10, 0.70, 0.20])
    rhd = rng.choice([True, False, None], size=n, p=[0.02, 0.78, 0.20])

    # Price = f(year, mileage, engine, condition) + noise
    price = (
        (years - 2000) * 600
        - mileage * 0.02
        + engine_cc * 2
        + rng.normal(0, 1000, size=n)
    ).clip(min=500)
    # Accident cars are cheaper
    for i in range(n):
        if accident[i] is True:
            price[i] *= 0.7
        if rhd[i] is True:
            price[i] *= 0.8

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
        # LLM fields
        "desc_mentions_accident": accident,
        "desc_mentions_repair": repair,
        "right_hand_drive": rhd,
        "desc_mentions_customs_cleared": rng.choice([True, False, None], size=n, p=[0.15, 0.65, 0.20]),
        "desc_mentions_num_owners": rng.choice([1, 2, 3, None], size=n, p=[0.3, 0.2, 0.1, 0.4]),
        "taxi_fleet_rental": rng.choice([True, False, None], size=n, p=[0.03, 0.77, 0.20]),
        "warranty": rng.choice([True, False, None], size=n, p=[0.15, 0.65, 0.20]),
        "first_owner_selling": rng.choice([True, False, None], size=n, p=[0.20, 0.60, 0.20]),
        "urgency": rng.choice(["high", "medium", "low", None], size=n, p=[0.05, 0.15, 0.50, 0.30]),
    })


def test_train_returns_model_and_metrics():
    df = _sample_listings()
    result = train_price_model(df)
    assert result is not None
    models, cat_maps, metrics, oof_preds = result
    assert "brand" in cat_maps
    assert "model" in cat_maps
    assert "low" in models and "median" in models and "high" in models
    # Metrics from cross-validation
    assert "mae" in metrics and metrics["mae"] > 0
    assert "mape" in metrics and metrics["mape"] > 0
    assert "r2" in metrics
    assert "n_samples" in metrics
    assert "cv_folds" in metrics and metrics["cv_folds"] == 5
    # Per-quantile early-stop iters are tracked separately
    assert "best_n_estimators_per_q" in metrics
    assert set(metrics["best_n_estimators_per_q"]) == {"low", "median", "high"}
    # Log target on by default; conformal_q is in log space and a
    # human-readable percent is also exposed.
    assert metrics.get("log_target") is True
    assert "conformal_q_pct" in metrics
    # OOF preds keyed by olx_id, one entry per training listing
    assert isinstance(oof_preds, dict)
    assert len(oof_preds) > 0
    sample_id = next(iter(oof_preds))
    lo, med, hi = oof_preds[sample_id]
    # OOF entries are already crossing-repaired and clamped
    assert lo <= med <= hi
    assert lo >= 0


def test_train_returns_none_for_small_data():
    df = _sample_listings(10)
    result = train_price_model(df, min_samples=50)
    assert result is None


def test_predictions_are_positive():
    df = _sample_listings()
    models, cat_maps, _metrics, _oof = train_price_model(df)
    preds = predict_prices(models, cat_maps, df)
    assert (preds["predicted_price"] >= 0).all()
    assert (preds["fair_price_low"] >= 0).all()
    assert (preds["fair_price_high"] >= 0).all()
    assert len(preds) == len(df)


def test_predictions_never_cross():
    """low ≤ median ≤ high must hold for every row, with or without OOF override."""
    df = _sample_listings(300)
    models, cat_maps, metrics, oof_preds = train_price_model(df)

    # In-sample (uses OOF preds for every row)
    in_sample = predict_prices(
        models, cat_maps, df,
        conformal_q=metrics.get("conformal_q", 0.0),
        oof_preds=oof_preds,
    )
    assert (in_sample["fair_price_low"] <= in_sample["predicted_price"]).all()
    assert (in_sample["predicted_price"] <= in_sample["fair_price_high"]).all()

    # Out-of-sample (synthetic rows the model has never seen, no OOF override).
    # Build new olx_ids so nothing collides with the OOF dict.
    fresh = df.copy()
    fresh["olx_id"] = [f"new_{i}" for i in range(len(fresh))]
    oos = predict_prices(
        models, cat_maps, fresh,
        conformal_q=metrics.get("conformal_q", 0.0),
        oof_preds=oof_preds,
    )
    assert (oos["fair_price_low"] <= oos["predicted_price"]).all()
    assert (oos["predicted_price"] <= oos["fair_price_high"]).all()


def test_oof_preds_used_for_known_olx_ids():
    """Listings present in oof_preds get exactly the OOF values, not model.predict."""
    df = _sample_listings(150)
    models, cat_maps, _metrics, oof_preds = train_price_model(df)

    # Pick an olx_id that's in the OOF dict and predict on a single-row df.
    target_id = next(iter(oof_preds))
    row = df[df["olx_id"] == target_id].head(1).copy()
    if row.empty:  # all training rows survived filtering
        target_id = df["olx_id"].iloc[0]
        row = df.head(1).copy()

    expected_low, expected_median, expected_high = oof_preds[str(target_id)]
    preds = predict_prices(
        models, cat_maps, row,
        conformal_q=0.0, oof_preds=oof_preds,
    )
    # Round-trip through DataFrame.round(0) — compare with rounded expected too
    assert preds["predicted_price"].iloc[0] == round(expected_median)
    assert preds["fair_price_low"].iloc[0] == round(expected_low)
    assert preds["fair_price_high"].iloc[0] == round(expected_high)


def test_newer_cars_predicted_higher():
    df = _sample_listings(500)
    models, cat_maps, _metrics, _oof = train_price_model(df)

    old_car = pd.DataFrame([{
        "year": 2010, "mileage_km": 150000, "engine_cc": 1600,
        "brand": "Volkswagen", "model": "Golf",
        "fuel_type": "Diesel", "transmission": "Manual",
        "segment": "Citadino", "horsepower": 110,
    }])
    new_car = pd.DataFrame([{
        "year": 2022, "mileage_km": 30000, "engine_cc": 1600,
        "brand": "Volkswagen", "model": "Golf",
        "fuel_type": "Diesel", "transmission": "Manual",
        "segment": "Citadino", "horsepower": 110,
    }])

    pred_old = predict_prices(models, cat_maps, old_car)["predicted_price"].iloc[0]
    pred_new = predict_prices(models, cat_maps, new_car)["predicted_price"].iloc[0]
    assert pred_new > pred_old


def test_accident_cars_predicted_lower():
    df = _sample_listings(500)
    models, cat_maps, _metrics, _oof = train_price_model(df)

    base = {
        "year": 2018, "mileage_km": 100000, "engine_cc": 1600,
        "brand": "Volkswagen", "model": "Golf",
        "fuel_type": "Diesel", "transmission": "Manual",
        "segment": "Citadino", "horsepower": 110,
    }
    clean = pd.DataFrame([{**base, "desc_mentions_accident": False}])
    damaged = pd.DataFrame([{**base, "desc_mentions_accident": True}])

    pred_clean = predict_prices(models, cat_maps, clean)["predicted_price"].iloc[0]
    pred_damaged = predict_prices(models, cat_maps, damaged)["predicted_price"].iloc[0]
    assert pred_clean > pred_damaged


def test_handles_missing_features():
    """Model should handle NaN in optional features."""
    df = _sample_listings()
    # Drop some optional columns
    df.loc[df.index[:50], "engine_cc"] = np.nan
    df.loc[df.index[:30], "horsepower"] = np.nan

    result = train_price_model(df)
    assert result is not None
    models, cat_maps, _metrics, _oof = result

    # Predict on row with missing features
    sparse = pd.DataFrame([{
        "year": 2018, "mileage_km": 100000,
        "brand": "Volkswagen", "model": "Golf",
    }])
    preds = predict_prices(models, cat_maps, sparse)
    assert len(preds) == 1
    assert preds["predicted_price"].iloc[0] > 0


def test_rare_categories_grouped():
    """Rare categories should be grouped into __other__."""
    df = _sample_listings(600)
    # Each model appears exactly once → all below _MIN_CATEGORY_COUNT
    df["model"] = [f"Model_{i}" for i in range(len(df))]

    result = train_price_model(df)
    assert result is not None
    models, cat_maps, _metrics, _oof = result
    assert "__other__" in cat_maps["model"]

    unseen = pd.DataFrame([{
        "year": 2018, "mileage_km": 100000, "engine_cc": 1600,
        "brand": "Volkswagen", "model": "Future_Model",
        "fuel_type": "Diesel", "transmission": "Manual",
        "segment": "Citadino", "horsepower": 110,
    }])
    pred = predict_prices(models, cat_maps, unseen)["predicted_price"].iloc[0]
    assert pred > 0


def test_save_and_load_model(tmp_path, monkeypatch):
    """Save/load round-trip produces identical predictions and oof_preds."""
    monkeypatch.setattr("src.analytics.price_model._MODEL_PATH", tmp_path / "model.joblib")
    monkeypatch.setattr("src.analytics.price_model._METRICS_PATH", tmp_path / "metrics.json")

    df = _sample_listings()
    models, cat_maps, metrics, oof_preds = train_price_model(df)
    save_model(models, cat_maps, metrics, oof_preds=oof_preds)

    loaded = load_model(max_age_hours=1)
    assert loaded is not None
    l_models, l_cat_maps, l_metrics, l_oof = loaded

    preds_orig = predict_prices(models, cat_maps, df, oof_preds=oof_preds)
    preds_loaded = predict_prices(l_models, l_cat_maps, df, oof_preds=l_oof)
    pd.testing.assert_frame_equal(preds_orig, preds_loaded)
    assert l_metrics["mae"] == metrics["mae"]
    assert l_oof == oof_preds


def test_load_model_rejects_schema_mismatch(tmp_path, monkeypatch):
    """A bundle with a stale schema_version is rejected, not silently used."""
    import joblib
    monkeypatch.setattr("src.analytics.price_model._MODEL_PATH", tmp_path / "model.joblib")
    monkeypatch.setattr("src.analytics.price_model._METRICS_PATH", tmp_path / "metrics.json")

    df = _sample_listings()
    models, cat_maps, metrics, oof_preds = train_price_model(df)
    save_model(models, cat_maps, metrics, oof_preds=oof_preds)

    # Hand-corrupt the schema_version field, simulating an artifact trained
    # against an older feature list.
    bundle = joblib.load(tmp_path / "model.joblib")
    bundle["schema_version"] = _SCHEMA_VERSION - 1
    joblib.dump(bundle, tmp_path / "model.joblib")

    assert load_model(max_age_hours=1) is None


def test_load_model_rejects_feature_mismatch(tmp_path, monkeypatch):
    """A bundle whose feature_names list differs from current is rejected."""
    import joblib
    monkeypatch.setattr("src.analytics.price_model._MODEL_PATH", tmp_path / "model.joblib")
    monkeypatch.setattr("src.analytics.price_model._METRICS_PATH", tmp_path / "metrics.json")

    df = _sample_listings()
    models, cat_maps, metrics, oof_preds = train_price_model(df)
    save_model(models, cat_maps, metrics, oof_preds=oof_preds)

    bundle = joblib.load(tmp_path / "model.joblib")
    bundle["feature_names"] = bundle["feature_names"] + ["fictional_feature"]
    joblib.dump(bundle, tmp_path / "model.joblib")

    assert load_model(max_age_hours=1) is None


def test_metrics_history(tmp_path, monkeypatch):
    """Each save appends to metrics history."""
    monkeypatch.setattr("src.analytics.price_model._MODEL_PATH", tmp_path / "model.joblib")
    monkeypatch.setattr("src.analytics.price_model._METRICS_PATH", tmp_path / "metrics.json")

    df = _sample_listings()
    models, cat_maps, metrics, oof_preds = train_price_model(df)

    save_model(models, cat_maps, metrics, oof_preds=oof_preds)
    save_model(models, cat_maps, metrics, oof_preds=oof_preds)

    history = load_metrics_history()
    assert len(history) == 2
    assert "timestamp" in history[0]
    assert history[0]["mae"] == metrics["mae"]
