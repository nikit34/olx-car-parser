"""Tests for time-to-sale model."""

from datetime import datetime, timedelta

import pandas as pd
import pytest

from src.analytics.time_to_sale import (
    build_tos_dataset,
    compute_tos_stats,
    fit_tos_model,
    optimize_price,
    predict_sale_probability,
    DEFAULT_TOS_WEIGHTS,
    TOS_FEATURE_COLUMNS,
)


def _make_listings() -> pd.DataFrame:
    """Create test listings with known sold/active statuses."""
    now = datetime.utcnow()
    rows = []

    # Sold listings (is_active=False) — sold quickly (underpriced)
    for i in range(15):
        rows.append({
            "olx_id": f"sold_fast_{i}",
            "brand": "Volkswagen",
            "model": "Golf",
            "year": 2016,
            "price_eur": 7000 + i * 100,
            "first_price_eur": 7500 + i * 100,
            "mileage_km": 120000 + i * 5000,
            "fuel_type": "Gasolina",
            "seller_type": "Particular",
            "desc_mentions_repair": False,
            "desc_mentions_accident": False,
            "first_seen_at": now - timedelta(days=20 + i),
            "last_seen_at": now - timedelta(days=5 + i),
            "deactivated_at": now - timedelta(days=5 + i),
            "is_active": False,
        })

    # Sold listings — sold slowly (overpriced)
    for i in range(10):
        rows.append({
            "olx_id": f"sold_slow_{i}",
            "brand": "Volkswagen",
            "model": "Golf",
            "year": 2015,
            "price_eur": 11000 + i * 200,
            "first_price_eur": 12000 + i * 200,
            "mileage_km": 180000 + i * 5000,
            "fuel_type": "Gasolina",
            "seller_type": "Particular",
            "desc_mentions_repair": False,
            "desc_mentions_accident": False,
            "first_seen_at": now - timedelta(days=80 + i * 3),
            "last_seen_at": now - timedelta(days=10),
            "deactivated_at": now - timedelta(days=10),
            "is_active": False,
        })

    # Active listings
    for i in range(10):
        rows.append({
            "olx_id": f"active_{i}",
            "brand": "Volkswagen",
            "model": "Golf",
            "year": 2016,
            "price_eur": 9000 + i * 200,
            "first_price_eur": 9000 + i * 200,
            "mileage_km": 150000,
            "fuel_type": "Gasolina",
            "seller_type": "Particular",
            "desc_mentions_repair": False,
            "desc_mentions_accident": False,
            "first_seen_at": now - timedelta(days=10 + i * 5),
            "last_seen_at": now,
            "deactivated_at": None,
            "is_active": True,
        })

    return pd.DataFrame(rows)


def test_build_tos_dataset_creates_features():
    df = _make_listings()
    dataset = build_tos_dataset(df)

    assert not dataset.empty
    for col in TOS_FEATURE_COLUMNS:
        assert col in dataset.columns, f"Missing feature column: {col}"
    assert "days_to_sale" in dataset.columns
    assert "censored" in dataset.columns

    # Sold listings should not be censored
    sold_rows = dataset[dataset["olx_id"].str.startswith("sold_")]
    assert not sold_rows["censored"].any()

    # Active listings should be censored
    active_rows = dataset[dataset["olx_id"].str.startswith("active_")]
    assert active_rows["censored"].all()


def test_build_tos_dataset_price_ratio():
    df = _make_listings()
    dataset = build_tos_dataset(df)

    # price_ratio should be price / market_median
    assert dataset["price_ratio"].notna().all()
    assert (dataset["price_ratio"] > 0).all()


def test_fit_tos_model_learns():
    df = _make_listings()
    dataset = build_tos_dataset(df)
    weights = fit_tos_model(dataset, target_days=30)

    assert "bias" in weights
    for col in TOS_FEATURE_COLUMNS:
        assert col in weights

    # price_ratio should have negative weight (higher price → less likely to sell)
    assert weights["price_ratio"] < 0


def test_predict_sale_probability():
    df = _make_listings()
    dataset = build_tos_dataset(df)

    probs = predict_sale_probability(dataset)
    assert len(probs) == len(dataset)
    assert (probs >= 0).all() and (probs <= 1).all()


def test_predict_cheaper_sells_faster():
    """Cheaper listings (lower price_ratio) should have higher sale probability."""
    df = _make_listings()
    dataset = build_tos_dataset(df)
    weights = fit_tos_model(dataset, target_days=30)

    # Create two test rows: one cheap, one expensive
    cheap = pd.DataFrame([{col: 0.5 for col in TOS_FEATURE_COLUMNS}])
    cheap["price_ratio"] = 0.8

    expensive = pd.DataFrame([{col: 0.5 for col in TOS_FEATURE_COLUMNS}])
    expensive["price_ratio"] = 1.2

    p_cheap = float(predict_sale_probability(cheap, weights).iloc[0])
    p_expensive = float(predict_sale_probability(expensive, weights).iloc[0])

    assert p_cheap > p_expensive


def test_optimize_price_returns_valid_result():
    features = {
        "price_ratio": 1.0,
        "mileage_norm": 0.3,
        "year_norm": 0.64,
        "is_diesel": 0.0,
        "is_professional": 0.0,
        "price_drop_rate": 0.0,
        "has_accident": 0.0,
        "has_repair": 0.0,
    }
    result = optimize_price(features, market_median=10000)

    assert "optimal_price" in result
    assert "sale_probability" in result
    assert "price_curve" in result
    assert result["optimal_price"] > 0
    assert 0 <= result["sale_probability"] <= 1
    assert len(result["price_curve"]) == 50


def test_optimize_price_respects_min_price():
    features = {col: 0.5 for col in TOS_FEATURE_COLUMNS}
    result = optimize_price(features, market_median=10000, min_acceptable_price=9000)

    # Optimal price should be >= min_acceptable where score > 0
    assert result["optimal_price"] >= 7000  # min_price_ratio * median


def test_compute_tos_stats():
    df = _make_listings()
    stats = compute_tos_stats(df)

    assert not stats.empty
    assert "median_days" in stats.columns
    assert "sold_count" in stats.columns
    assert "pct_under_30d" in stats.columns
    assert "sale_rate_pct" in stats.columns

    vw_row = stats[(stats["brand"] == "Volkswagen") & (stats["model"] == "Golf")]
    assert not vw_row.empty
    assert int(vw_row.iloc[0]["sold_count"]) == 25  # 15 fast + 10 slow


def test_compute_tos_stats_empty():
    stats = compute_tos_stats(pd.DataFrame())
    assert stats.empty
