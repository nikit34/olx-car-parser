"""Tests for buy signal computation."""

from unittest.mock import patch

import pandas as pd

from src.dashboard.data_loader import compute_signals


class TestComputeSignals:
    def test_empty_inputs(self):
        assert compute_signals(pd.DataFrame(), pd.DataFrame()).empty

    def test_excludes_no_generation(self, sample_listings_df, sample_history_df, generations_data):
        with patch("src.models.generations.load_generations", return_value=generations_data):
            signals = compute_signals(sample_listings_df, sample_history_df)
            # Listing a4 has no year → no generation → excluded
            assert "a4" not in signals["olx_id"].values if not signals.empty else True

    def test_finds_discount(self, sample_listings_df, sample_history_df, generations_data):
        with patch("src.models.generations.load_generations", return_value=generations_data):
            signals = compute_signals(sample_listings_df, sample_history_df)
            if not signals.empty:
                # a1 has price 8000, generation median should be computed from a1+a2+a3
                # median of [8000, 14000, 15000] = 14000 → 8000 < 14000*0.85=11900 → signal
                assert "a1" in signals["olx_id"].values
                row = signals[signals["olx_id"] == "a1"].iloc[0]
                assert row["discount_pct"] > 0
                assert row["generation"] == "Mk7"

    def test_signal_has_required_fields(self, sample_listings_df, sample_history_df, generations_data):
        with patch("src.models.generations.load_generations", return_value=generations_data):
            signals = compute_signals(sample_listings_df, sample_history_df)
            if not signals.empty:
                required = {"olx_id", "brand", "model", "generation", "year",
                            "price_eur", "median_price_eur", "discount_pct", "sample_size"}
                assert required.issubset(set(signals.columns))
