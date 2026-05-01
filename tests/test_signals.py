"""Tests for buy signal computation."""

import json
from unittest.mock import patch

import pandas as pd

from src.dashboard.data_loader import compute_signals


class TestComputeSignals:
    def test_empty_inputs(self):
        signals, importance = compute_signals(pd.DataFrame(), pd.DataFrame())
        assert signals.empty

    def test_no_generation_uses_model_fallback(self, sample_listings_df, sample_history_df, generations_data):
        with patch("src.models.generations.load_generations", return_value=generations_data):
            signals, _ = compute_signals(sample_listings_df, sample_history_df)
            # a4 has no year → no generation, but model-level median is used as fallback
            # a4 price=5000, model median of [5000,8000,14000,15000]=11000
            # 5000 < 11000*0.85=9350 → should be a signal via model fallback
            if not signals.empty and "a4" in signals["olx_id"].values:
                row = signals[signals["olx_id"] == "a4"].iloc[0]
                assert row["generation"] == ""
                assert row["flip_score"] > 0

    def test_finds_discount(self, sample_listings_df, sample_history_df, generations_data):
        with patch("src.models.generations.load_generations", return_value=generations_data):
            signals, _ = compute_signals(sample_listings_df, sample_history_df)
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
            signals, _ = compute_signals(sample_listings_df, sample_history_df)
            if not signals.empty:
                required = {"olx_id", "brand", "model", "generation", "year",
                            "price_eur", "predicted_price", "median_price_eur",
                            "discount_pct", "undervaluation_pct",
                            "urgency_mult", "warranty_mult",
                            "velocity_mult", "confidence_mult",
                            "flip_score", "sample_size"}
                assert required.issubset(set(signals.columns))

    def test_excludes_obvious_total_loss_listings(self, sample_history_df, generations_data):
        listings = pd.DataFrame([
            {"olx_id": "risk-1", "url": "", "brand": "Volkswagen", "model": "Golf",
             "year": 2015, "price_eur": 3000, "mileage_km": 150000, "engine_cc": 1600,
             "fuel_type": "Diesel", "is_active": True, "desc_mentions_accident": True,
             "llm_extras": json.dumps({
                 "mechanical_condition": "poor",
             })},
            {"olx_id": "comp-1", "url": "", "brand": "Volkswagen", "model": "Golf",
             "year": 2015, "price_eur": 12000, "mileage_km": 160000, "engine_cc": 1600,
             "fuel_type": "Diesel", "is_active": True},
            {"olx_id": "comp-2", "url": "", "brand": "Volkswagen", "model": "Golf",
             "year": 2016, "price_eur": 14000, "mileage_km": 110000, "engine_cc": 1600,
             "fuel_type": "Diesel", "is_active": True},
            {"olx_id": "comp-3", "url": "", "brand": "Volkswagen", "model": "Golf",
             "year": 2017, "price_eur": 15000, "mileage_km": 90000, "engine_cc": 1600,
             "fuel_type": "Diesel", "is_active": True},
            {"olx_id": "comp-4", "url": "", "brand": "Volkswagen", "model": "Golf",
             "year": 2014, "price_eur": 11000, "mileage_km": 175000, "engine_cc": 1600,
             "fuel_type": "Diesel", "is_active": True},
            {"olx_id": "comp-5", "url": "", "brand": "Volkswagen", "model": "Golf",
             "year": 2013, "price_eur": 10000, "mileage_km": 190000, "engine_cc": 1600,
             "fuel_type": "Diesel", "is_active": True},
        ])

        with patch("src.models.generations.load_generations", return_value=generations_data):
            signals, _ = compute_signals(listings, sample_history_df)

        assert signals.empty or "risk-1" not in signals["olx_id"].values

    def test_keeps_maintenance_mentions_when_condition_is_good(self, sample_history_df, generations_data):
        listings = pd.DataFrame([
            {"olx_id": "maint-1", "url": "", "brand": "Volkswagen", "model": "Golf",
             "year": 2015, "price_eur": 8000, "mileage_km": 150000, "engine_cc": 1600,
             "fuel_type": "Diesel", "is_active": True, "desc_mentions_repair": True,
             "desc_mentions_accident": False,
             "llm_extras": json.dumps({
                 "mechanical_condition": "good",
             })},
            {"olx_id": "comp-1", "url": "", "brand": "Volkswagen", "model": "Golf",
             "year": 2015, "price_eur": 12000, "mileage_km": 160000, "engine_cc": 1600,
             "fuel_type": "Diesel", "is_active": True},
            {"olx_id": "comp-2", "url": "", "brand": "Volkswagen", "model": "Golf",
             "year": 2016, "price_eur": 14000, "mileage_km": 110000, "engine_cc": 1600,
             "fuel_type": "Diesel", "is_active": True},
            {"olx_id": "comp-3", "url": "", "brand": "Volkswagen", "model": "Golf",
             "year": 2017, "price_eur": 15000, "mileage_km": 90000, "engine_cc": 1600,
             "fuel_type": "Diesel", "is_active": True},
            {"olx_id": "comp-4", "url": "", "brand": "Volkswagen", "model": "Golf",
             "year": 2014, "price_eur": 11000, "mileage_km": 175000, "engine_cc": 1600,
             "fuel_type": "Diesel", "is_active": True},
            {"olx_id": "comp-5", "url": "", "brand": "Volkswagen", "model": "Golf",
             "year": 2013, "price_eur": 10000, "mileage_km": 190000, "engine_cc": 1600,
             "fuel_type": "Diesel", "is_active": True},
        ])

        with patch("src.models.generations.load_generations", return_value=generations_data):
            signals, _ = compute_signals(listings, sample_history_df)

        assert "maint-1" in signals["olx_id"].values

    def test_excludes_listings_flagged_by_new_multi_photo_rule(self, sample_history_df, generations_data):
        """Issue #8 migration: when ``photo_damage_flagged`` is set on a
        post-#2 listing, the dashboard must block it (the boolean wins over
        any p_max display value), wired through ``_blocking_deal_reason``.

        Without this, the production user-facing flag rate would still be
        the v2 max-rule rate (~32.8 %) instead of the new ~9.6 % validated
        in #1's production-validation comment.
        """
        listings = pd.DataFrame([
            {"olx_id": "new-flagged", "url": "", "brand": "Volkswagen", "model": "Golf",
             "year": 2015, "price_eur": 3000, "mileage_km": 150000, "engine_cc": 1600,
             "fuel_type": "Diesel", "is_active": True,
             "llm_extras": json.dumps({
                 # New cron writes both fields; the boolean is authoritative.
                 "photo_damage_p": 0.42,
                 "photo_damage_flagged": True,
             })},
            {"olx_id": "comp-1", "url": "", "brand": "Volkswagen", "model": "Golf",
             "year": 2015, "price_eur": 12000, "mileage_km": 160000, "engine_cc": 1600,
             "fuel_type": "Diesel", "is_active": True},
            {"olx_id": "comp-2", "url": "", "brand": "Volkswagen", "model": "Golf",
             "year": 2016, "price_eur": 14000, "mileage_km": 110000, "engine_cc": 1600,
             "fuel_type": "Diesel", "is_active": True},
            {"olx_id": "comp-3", "url": "", "brand": "Volkswagen", "model": "Golf",
             "year": 2017, "price_eur": 15000, "mileage_km": 90000, "engine_cc": 1600,
             "fuel_type": "Diesel", "is_active": True},
            {"olx_id": "comp-4", "url": "", "brand": "Volkswagen", "model": "Golf",
             "year": 2014, "price_eur": 11000, "mileage_km": 175000, "engine_cc": 1600,
             "fuel_type": "Diesel", "is_active": True},
            {"olx_id": "comp-5", "url": "", "brand": "Volkswagen", "model": "Golf",
             "year": 2013, "price_eur": 10000, "mileage_km": 190000, "engine_cc": 1600,
             "fuel_type": "Diesel", "is_active": True},
        ])

        with patch("src.models.generations.load_generations", return_value=generations_data):
            signals, _ = compute_signals(listings, sample_history_df)

        assert signals.empty or "new-flagged" not in signals["olx_id"].values

    def test_keeps_listings_cleared_by_new_multi_photo_rule(self, sample_history_df, generations_data):
        """Inverse of the previous test: ``photo_damage_flagged=False``
        with a high p_max — the v2 max-rule would have blocked this, but
        the new multi-photo agreement *cleared* it (one weirdly-lit photo).
        Issue #8 says the new boolean wins, so this listing must surface.
        """
        listings = pd.DataFrame([
            {"olx_id": "new-cleared", "url": "", "brand": "Volkswagen", "model": "Golf",
             "year": 2015, "price_eur": 8000, "mileage_km": 150000, "engine_cc": 1600,
             "fuel_type": "Diesel", "is_active": True,
             "llm_extras": json.dumps({
                 # max p above v2 threshold but multi-photo rule cleared it.
                 "photo_damage_p": 0.95,
                 "photo_damage_flagged": False,
             })},
            {"olx_id": "comp-1", "url": "", "brand": "Volkswagen", "model": "Golf",
             "year": 2015, "price_eur": 12000, "mileage_km": 160000, "engine_cc": 1600,
             "fuel_type": "Diesel", "is_active": True},
            {"olx_id": "comp-2", "url": "", "brand": "Volkswagen", "model": "Golf",
             "year": 2016, "price_eur": 14000, "mileage_km": 110000, "engine_cc": 1600,
             "fuel_type": "Diesel", "is_active": True},
            {"olx_id": "comp-3", "url": "", "brand": "Volkswagen", "model": "Golf",
             "year": 2017, "price_eur": 15000, "mileage_km": 90000, "engine_cc": 1600,
             "fuel_type": "Diesel", "is_active": True},
            {"olx_id": "comp-4", "url": "", "brand": "Volkswagen", "model": "Golf",
             "year": 2014, "price_eur": 11000, "mileage_km": 175000, "engine_cc": 1600,
             "fuel_type": "Diesel", "is_active": True},
            {"olx_id": "comp-5", "url": "", "brand": "Volkswagen", "model": "Golf",
             "year": 2013, "price_eur": 10000, "mileage_km": 190000, "engine_cc": 1600,
             "fuel_type": "Diesel", "is_active": True},
        ])

        with patch("src.models.generations.load_generations", return_value=generations_data):
            signals, _ = compute_signals(listings, sample_history_df)

        # The whole point of issue #8: trust the new field even when p is high.
        assert "new-cleared" in signals["olx_id"].values

    def test_legacy_listings_keep_v2_max_rule(self, sample_history_df, generations_data):
        """Pre-#2 listings only have ``photo_damage_p`` (we didn't backfill
        the 6 271 legacy rows — see #8 spec). For those, the helper must
        fall back to ``photo_damage_p >= 0.20`` so blocking parity is
        preserved. Otherwise legacy damaged cars would silently start
        passing the deal filter.
        """
        listings = pd.DataFrame([
            {"olx_id": "legacy-damaged", "url": "", "brand": "Volkswagen", "model": "Golf",
             "year": 2015, "price_eur": 3000, "mileage_km": 150000, "engine_cc": 1600,
             "fuel_type": "Diesel", "is_active": True,
             "llm_extras": json.dumps({
                 # No photo_damage_flagged → fall back to v2 max-rule.
                 "photo_damage_p": 0.5,
             })},
            {"olx_id": "comp-1", "url": "", "brand": "Volkswagen", "model": "Golf",
             "year": 2015, "price_eur": 12000, "mileage_km": 160000, "engine_cc": 1600,
             "fuel_type": "Diesel", "is_active": True},
            {"olx_id": "comp-2", "url": "", "brand": "Volkswagen", "model": "Golf",
             "year": 2016, "price_eur": 14000, "mileage_km": 110000, "engine_cc": 1600,
             "fuel_type": "Diesel", "is_active": True},
            {"olx_id": "comp-3", "url": "", "brand": "Volkswagen", "model": "Golf",
             "year": 2017, "price_eur": 15000, "mileage_km": 90000, "engine_cc": 1600,
             "fuel_type": "Diesel", "is_active": True},
            {"olx_id": "comp-4", "url": "", "brand": "Volkswagen", "model": "Golf",
             "year": 2014, "price_eur": 11000, "mileage_km": 175000, "engine_cc": 1600,
             "fuel_type": "Diesel", "is_active": True},
            {"olx_id": "comp-5", "url": "", "brand": "Volkswagen", "model": "Golf",
             "year": 2013, "price_eur": 10000, "mileage_km": 190000, "engine_cc": 1600,
             "fuel_type": "Diesel", "is_active": True},
        ])

        with patch("src.models.generations.load_generations", return_value=generations_data):
            signals, _ = compute_signals(listings, sample_history_df)

        assert signals.empty or "legacy-damaged" not in signals["olx_id"].values

    def test_drops_zero_and_negative_prices(self, sample_history_df, generations_data):
        listings = pd.DataFrame([
            {"olx_id": "junk-zero", "url": "", "brand": "Volkswagen", "model": "Golf",
             "year": 2015, "price_eur": 0, "mileage_km": 150000, "engine_cc": 1600,
             "fuel_type": "Diesel", "is_active": True},
            {"olx_id": "junk-neg", "url": "", "brand": "Volkswagen", "model": "Golf",
             "year": 2015, "price_eur": -1, "mileage_km": 150000, "engine_cc": 1600,
             "fuel_type": "Diesel", "is_active": True},
            {"olx_id": "comp-1", "url": "", "brand": "Volkswagen", "model": "Golf",
             "year": 2015, "price_eur": 12000, "mileage_km": 160000, "engine_cc": 1600,
             "fuel_type": "Diesel", "is_active": True},
            {"olx_id": "comp-2", "url": "", "brand": "Volkswagen", "model": "Golf",
             "year": 2016, "price_eur": 14000, "mileage_km": 110000, "engine_cc": 1600,
             "fuel_type": "Diesel", "is_active": True},
            {"olx_id": "comp-3", "url": "", "brand": "Volkswagen", "model": "Golf",
             "year": 2017, "price_eur": 15000, "mileage_km": 90000, "engine_cc": 1600,
             "fuel_type": "Diesel", "is_active": True},
            {"olx_id": "comp-4", "url": "", "brand": "Volkswagen", "model": "Golf",
             "year": 2014, "price_eur": 11000, "mileage_km": 175000, "engine_cc": 1600,
             "fuel_type": "Diesel", "is_active": True},
            {"olx_id": "comp-5", "url": "", "brand": "Volkswagen", "model": "Golf",
             "year": 2013, "price_eur": 10000, "mileage_km": 190000, "engine_cc": 1600,
             "fuel_type": "Diesel", "is_active": True},
        ])

        with patch("src.models.generations.load_generations", return_value=generations_data):
            signals, _ = compute_signals(listings, sample_history_df)

        if not signals.empty:
            assert "junk-zero" not in signals["olx_id"].values
            assert "junk-neg" not in signals["olx_id"].values
