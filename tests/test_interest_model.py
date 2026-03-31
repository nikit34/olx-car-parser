"""Tests for interesting listing ranking."""

import pandas as pd

from src.analytics.interest_model import score_interest_candidates


def _active_listings() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "olx_id": "a1",
                "url": "https://olx.pt/a1",
                "brand": "Volkswagen",
                "model": "Golf",
                "year": 2016,
                "price_eur": 8200,
                "days_listed": 12,
                "needs_repair": False,
                "had_accident": False,
                "num_owners": 2,
            },
            {
                "olx_id": "a2",
                "url": "https://olx.pt/a2",
                "brand": "Volkswagen",
                "model": "Golf",
                "year": 2017,
                "price_eur": 9100,
                "days_listed": 18,
                "needs_repair": False,
                "had_accident": False,
                "num_owners": 1,
            },
            {
                "olx_id": "a3",
                "url": "https://olx.pt/a3",
                "brand": "BMW",
                "model": "320d",
                "year": 2014,
                "price_eur": 12500,
                "days_listed": 4,
                "needs_repair": False,
                "had_accident": False,
                "num_owners": 2,
            },
            {
                "olx_id": "a4",
                "url": "https://olx.pt/a4",
                "brand": "Renault",
                "model": "Clio",
                "year": 2012,
                "price_eur": 6500,
                "days_listed": 42,
                "needs_repair": True,
                "had_accident": False,
                "num_owners": 4,
            },
            {
                "olx_id": "a5",
                "url": "https://olx.pt/a5",
                "brand": "Peugeot",
                "model": "208",
                "year": 2013,
                "price_eur": 7100,
                "days_listed": 37,
                "needs_repair": False,
                "had_accident": True,
                "num_owners": 5,
            },
        ]
    )


def _deal_signals() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "olx_id": "a1",
                "predicted_price": 9800,
                "undervaluation_pct": 16.3,
                "sample_size": 11,
                "avg_days_to_sell": 24,
                "price_drop_per_day": -32,
                "est_profit_eur": 1600,
                "est_roi_pct": 19.5,
            },
            {
                "olx_id": "a2",
                "predicted_price": 10800,
                "undervaluation_pct": 15.7,
                "sample_size": 10,
                "avg_days_to_sell": 26,
                "price_drop_per_day": -18,
                "est_profit_eur": 1700,
                "est_roi_pct": 18.7,
            },
            {
                "olx_id": "a3",
                "predicted_price": 13600,
                "undervaluation_pct": 8.1,
                "sample_size": 8,
                "avg_days_to_sell": 19,
                "price_drop_per_day": -8,
                "est_profit_eur": 1100,
                "est_roi_pct": 8.8,
            },
            {
                "olx_id": "a4",
                "predicted_price": 7600,
                "undervaluation_pct": 6.2,
                "sample_size": 4,
                "avg_days_to_sell": 40,
                "price_drop_per_day": -4,
                "est_profit_eur": 700,
                "est_roi_pct": 10.8,
            },
        ]
    )


def test_interest_scoring_works_without_portfolio_labels():
    scored = score_interest_candidates(_active_listings(), _deal_signals(), pd.DataFrame())

    assert not scored.empty
    assert {"interest_probability", "interest_class", "interest_reason", "model_source"}.issubset(
        scored.columns
    )
    assert set(scored["model_source"]) == {"prior"}
    assert scored.iloc[0]["olx_id"] == "a1"


def test_interest_scoring_can_learn_from_portfolio_examples():
    portfolio_df = pd.DataFrame(
        [
            {"olx_listing_id": "a1"},
            {"olx_listing_id": "a2"},
        ]
    )

    scored = score_interest_candidates(
        _active_listings(),
        _deal_signals(),
        portfolio_df,
        min_positive_labels=2,
    )

    assert not scored.empty
    assert set(scored["model_source"]) == {"portfolio-trained"}
    assert int(scored["portfolio_positive"].sum()) == 2

    positive_probs = scored.loc[scored["portfolio_positive"], "interest_probability"]
    negative_probs = scored.loc[~scored["portfolio_positive"], "interest_probability"]
    assert positive_probs.min() > negative_probs.median()
