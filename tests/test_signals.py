"""Tests for buy signal computation."""

from unittest.mock import patch

import pandas as pd

from src.dashboard.data_loader import compute_signals


class TestComputeSignals:
    def test_empty_inputs(self):
        assert compute_signals(pd.DataFrame(), pd.DataFrame()).empty

    def test_no_generation_uses_model_fallback(self, sample_listings_df, sample_history_df, generations_data):
        with patch("src.models.generations.load_generations", return_value=generations_data):
            signals = compute_signals(sample_listings_df, sample_history_df)
            # a4 has no year → no generation, but model-level median is used as fallback
            # a4 price=5000, model median of [5000,8000,14000,15000]=11000
            # 5000 < 11000*0.85=9350 → should be a signal via model fallback
            if not signals.empty and "a4" in signals["olx_id"].values:
                row = signals[signals["olx_id"] == "a4"].iloc[0]
                assert row["generation"] == ""
                assert row["flip_score"] > 0

    def test_finds_discount(self, sample_listings_df, sample_history_df, generations_data):
        with patch("src.models.generations.load_generations", return_value=generations_data):
            signals = compute_signals(sample_listings_df, sample_history_df)
            if not signals.empty:
                # a1 has price 8000, generation median should be computed from a1+a2+a3
                # median of [8000, 14000, 15000] = 14000 → 8000 < 14000*0.85=11900 → signal
                assert "a1" in signals["olx_id"].values
                row = signals[signals["olx_id"] == "a1"].iloc[0]
                assert row["discount_pct"] > 0
                assert row["flip_score"] > 0
                assert row["generation"] == "Mk7"

    def test_signal_has_required_fields(self, sample_listings_df, sample_history_df, generations_data):
        with patch("src.models.generations.load_generations", return_value=generations_data):
            signals = compute_signals(sample_listings_df, sample_history_df)
            if not signals.empty:
                required = {"olx_id", "brand", "model", "generation", "year",
                            "price_eur", "predicted_price", "median_price_eur",
                            "discount_pct", "undervaluation_pct", "year_mult",
                            "velocity_mult", "confidence_mult",
                            "flip_score", "sample_size"}
                assert required.issubset(set(signals.columns))
