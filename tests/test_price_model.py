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

    # Price = f(year, mileage, engine) + noise
    price = (
        (years - 2000) * 600
        - mileage * 0.02
        + engine_cc * 2
        + rng.normal(0, 1000, size=n)
    ).clip(min=500)

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
    })


def test_train_returns_model():
    df = _sample_listings()
    result = train_price_model(df)
    assert result is not None
    model, cat_maps = result
    assert "brand" in cat_maps
    assert "model" in cat_maps


def test_train_returns_none_for_small_data():
    df = _sample_listings(10)
    result = train_price_model(df, min_samples=50)
    assert result is None


def test_predictions_are_positive():
    df = _sample_listings()
    model, cat_maps = train_price_model(df)
    preds = predict_prices(model, cat_maps, df)
    assert (preds >= 0).all()
    assert len(preds) == len(df)


def test_newer_cars_predicted_higher():
    df = _sample_listings(500)
    model, cat_maps = train_price_model(df)

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

    pred_old = predict_prices(model, cat_maps, old_car).iloc[0]
    pred_new = predict_prices(model, cat_maps, new_car).iloc[0]
    assert pred_new > pred_old


def test_handles_missing_features():
    """Model should handle NaN in optional features."""
    df = _sample_listings()
    # Drop some optional columns
    df.loc[df.index[:50], "engine_cc"] = np.nan
    df.loc[df.index[:30], "horsepower"] = np.nan

    result = train_price_model(df)
    assert result is not None
    model, cat_maps = result

    # Predict on row with missing features
    sparse = pd.DataFrame([{
        "year": 2018, "mileage_km": 100000,
        "brand": "Volkswagen", "model": "Golf",
    }])
    preds = predict_prices(model, cat_maps, sparse)
    assert len(preds) == 1
    assert preds.iloc[0] > 0
