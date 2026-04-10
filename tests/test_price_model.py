"""Tests for gradient boosting price model."""

import numpy as np
import pandas as pd

from src.analytics.price_model import train_price_model, predict_prices


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
        "tires_condition": rng.choice(["new", "good", "fair", "poor", None], size=n, p=[0.1, 0.3, 0.2, 0.1, 0.3]),
        "urgency": rng.choice(["high", "medium", "low", None], size=n, p=[0.05, 0.15, 0.50, 0.30]),
    })


def test_train_returns_model():
    df = _sample_listings()
    result = train_price_model(df)
    assert result is not None
    models, cat_maps = result
    assert "brand" in cat_maps
    assert "model" in cat_maps
    assert "low" in models and "median" in models and "high" in models


def test_train_returns_none_for_small_data():
    df = _sample_listings(10)
    result = train_price_model(df, min_samples=50)
    assert result is None


def test_predictions_are_positive():
    df = _sample_listings()
    models, cat_maps = train_price_model(df)
    preds = predict_prices(models, cat_maps, df)
    assert (preds["predicted_price"] >= 0).all()
    assert (preds["fair_price_low"] >= 0).all()
    assert (preds["fair_price_high"] >= 0).all()
    assert len(preds) == len(df)


def test_newer_cars_predicted_higher():
    df = _sample_listings(500)
    models, cat_maps = train_price_model(df)

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
    models, cat_maps = train_price_model(df)

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
    models, cat_maps = result

    # Predict on row with missing features
    sparse = pd.DataFrame([{
        "year": 2018, "mileage_km": 100000,
        "brand": "Volkswagen", "model": "Golf",
    }])
    preds = predict_prices(models, cat_maps, sparse)
    assert len(preds) == 1
    assert preds["predicted_price"].iloc[0] > 0


def test_caps_high_cardinality_categories():
    """Rare categories should be grouped so HGBR stays within its 255-bin limit."""
    df = _sample_listings(600)
    df["model"] = [f"Model {i}" for i in range(len(df))]

    result = train_price_model(df)

    assert result is not None
    models, cat_maps = result
    assert len(cat_maps["model"]) <= 255
    assert "__other__" in cat_maps["model"]

    unseen = pd.DataFrame([{
        "year": 2018,
        "mileage_km": 100000,
        "engine_cc": 1600,
        "brand": "Volkswagen",
        "model": "Future Model",
        "fuel_type": "Diesel",
        "transmission": "Manual",
        "segment": "Citadino",
        "horsepower": 110,
    }])
    pred = predict_prices(models, cat_maps, unseen)["predicted_price"].iloc[0]
    assert pred > 0
