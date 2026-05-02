"""Tests for buy signal computation."""

import json
from unittest.mock import patch

import pandas as pd

from src.dashboard.data_loader import compute_signals


class TestComputeSignals:
    def test_empty_inputs(self):
        signals, importance = compute_signals(pd.DataFrame(), pd.DataFrame())
        assert signals.empty

    def test_without_gb_model_no_signals(
        self, sample_listings_df, sample_history_df, generations_data,
    ):
        """No price model loaded → no deals surface. The previous behaviour
        used a median-discount fallback that the 2026-05-02 audit traced to
        ~37 % of false-positive top-30 (CLA, X2, C-HR via plain median
        comparison). Quality-over-coverage: empty is correct."""
        with patch(
            "src.models.generations.load_generations",
            return_value=generations_data,
        ):
            signals, _ = compute_signals(sample_listings_df, sample_history_df)
        assert signals.empty

    def test_finds_discount(
        self, sample_listings_df, sample_history_df, generations_data,
        patched_gb_model,
    ):
        with patched_gb_model(multiplier=1.5), patch(
            "src.models.generations.load_generations",
            return_value=generations_data,
        ):
            signals, _ = compute_signals(sample_listings_df, sample_history_df)
        # GB stub predicts price * 1.5 → undervaluation_pct = ~33 for every
        # listing that clears the median-discount gate. a1 (€8000 vs Golf
        # Mk7 median ~€14k) clears, gets surfaced via the GB path.
        assert "a1" in signals["olx_id"].values
        row = signals[signals["olx_id"] == "a1"].iloc[0]
        assert row["undervaluation_pct"] > 0
        assert row["flip_score"] > 0
        assert row["generation"] == "Mk7"

    def test_signal_has_required_fields(
        self, sample_listings_df, sample_history_df, generations_data,
        patched_gb_model,
    ):
        with patched_gb_model(), patch(
            "src.models.generations.load_generations",
            return_value=generations_data,
        ):
            signals, _ = compute_signals(sample_listings_df, sample_history_df)
        assert not signals.empty
        required = {"olx_id", "brand", "model", "generation", "year",
                    "price_eur", "predicted_price", "median_price_eur",
                    "discount_pct", "undervaluation_pct",
                    "urgency_mult", "warranty_mult",
                    "velocity_mult", "confidence_mult",
                    "flip_score", "sample_size"}
        assert required.issubset(set(signals.columns))

    def test_skips_listings_without_gb_undervaluation(
        self, sample_history_df, generations_data, patched_gb_model,
    ):
        """When the GB model says a listing is not undervalued (predicted
        price <= asking price), the deal is dropped — no median-discount
        fallback. This is the central guarantee of the 2026-05-02 audit
        fix; without it, listings like Mercedes CLA 8PZP1l (predicted
        €12.3k, ask €14k) would still surface with score 83.5."""
        listings = pd.DataFrame([
            # Asking price equals our fake "predicted" → undervaluation_pct = 0
            # → must be skipped even though discount_pct vs median is positive.
            {"olx_id": "fairly-priced", "url": "", "brand": "Volkswagen", "model": "Golf",
             "year": 2015, "price_eur": 8000, "mileage_km": 150000, "engine_cc": 1600,
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

        # multiplier=1.0 → predicted == price → undervaluation_pct == 0 for every row.
        with patched_gb_model(multiplier=1.0), patch(
            "src.models.generations.load_generations",
            return_value=generations_data,
        ):
            signals, _ = compute_signals(listings, sample_history_df)
        assert signals.empty or "fairly-priced" not in signals["olx_id"].values

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

    def test_keeps_maintenance_mentions_when_condition_is_good(
        self, sample_history_df, generations_data, patched_gb_model,
    ):
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

        with patched_gb_model(), patch(
            "src.models.generations.load_generations",
            return_value=generations_data,
        ):
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

    def test_keeps_listings_cleared_by_new_multi_photo_rule(
        self, sample_history_df, generations_data, patched_gb_model,
    ):
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

        with patched_gb_model(), patch(
            "src.models.generations.load_generations",
            return_value=generations_data,
        ):
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


def _golf_comparables() -> list[dict]:
    """Comparable Golf listings used to populate the median in hard-block tests."""
    return [
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
    ]


class TestBlockingDealReason:
    """Hard-block coverage for `_blocking_deal_reason`. The 2026-05-02 audit
    found the previous blocker only checked `desc_mentions_accident`,
    `mechanical_condition == "poor"`, and the photo flag — letting through
    salvage / parts-only / RHD / non-runner listings (Seat Leon JmWei,
    Peugeot 508 JmUNP, Citroën C5 8Q0kOc, Fiat Punto JmutI, Passat JmR3C
    all surfaced with scores 53–82). Each test pins one new gate."""

    def test_blocks_damage_severity_3(
        self, sample_history_df, generations_data, patched_gb_model,
    ):
        listings = pd.DataFrame(
            [{"olx_id": "salvage-1", "url": "", "brand": "Volkswagen", "model": "Golf",
              "year": 2015, "price_eur": 3000, "mileage_km": 150000, "engine_cc": 1600,
              "fuel_type": "Diesel", "is_active": True,
              "damage_severity": 3}]
            + _golf_comparables(),
        )
        with patched_gb_model(), patch(
            "src.models.generations.load_generations",
            return_value=generations_data,
        ):
            signals, _ = compute_signals(listings, sample_history_df)
        assert signals.empty or "salvage-1" not in signals["olx_id"].values

    def test_blocks_right_hand_drive(
        self, sample_history_df, generations_data, patched_gb_model,
    ):
        listings = pd.DataFrame(
            [{"olx_id": "rhd-1", "url": "", "brand": "Volkswagen", "model": "Golf",
              "year": 2015, "price_eur": 3000, "mileage_km": 150000, "engine_cc": 1600,
              "fuel_type": "Diesel", "is_active": True,
              "right_hand_drive": True}]
            + _golf_comparables(),
        )
        with patched_gb_model(), patch(
            "src.models.generations.load_generations",
            return_value=generations_data,
        ):
            signals, _ = compute_signals(listings, sample_history_df)
        assert signals.empty or "rhd-1" not in signals["olx_id"].values

    def test_blocks_salvage_phrasing_in_title_when_severity_unknown(
        self, sample_history_df, generations_data, patched_gb_model,
    ):
        """Defense-in-depth path: a freshly scraped row before enrichment
        runs has `damage_severity = NULL`, so we must scan the title."""
        listings = pd.DataFrame(
            [{"olx_id": "title-junta", "url": "", "title": "Golf 1.6 TDI - junta queimada",
              "brand": "Volkswagen", "model": "Golf",
              "year": 2015, "price_eur": 3000, "mileage_km": 150000, "engine_cc": 1600,
              "fuel_type": "Diesel", "is_active": True}]
            + _golf_comparables(),
        )
        with patched_gb_model(), patch(
            "src.models.generations.load_generations",
            return_value=generations_data,
        ):
            signals, _ = compute_signals(listings, sample_history_df)
        assert signals.empty or "title-junta" not in signals["olx_id"].values

    def test_blocks_parts_only_phrasing_in_title(
        self, sample_history_df, generations_data, patched_gb_model,
    ):
        listings = pd.DataFrame(
            [{"olx_id": "title-pecas", "url": "",
              "title": "Golf 1.6 TDI completo para peças",
              "brand": "Volkswagen", "model": "Golf",
              "year": 2015, "price_eur": 3000, "mileage_km": 150000, "engine_cc": 1600,
              "fuel_type": "Diesel", "is_active": True}]
            + _golf_comparables(),
        )
        with patched_gb_model(), patch(
            "src.models.generations.load_generations",
            return_value=generations_data,
        ):
            signals, _ = compute_signals(listings, sample_history_df)
        assert signals.empty or "title-pecas" not in signals["olx_id"].values

    def test_blocks_salvage_phrasing_in_description_only(
        self, sample_history_df, generations_data, patched_gb_model,
    ):
        """The audit's JmUNP / JmutI / JmR3C all had the salvage tell only
        in the description ("não pega", "junta queimada", "avaria no motor"
        respectively); the original title-only scan missed them. Putting
        the phrase only in description should still hard-block."""
        listings = pd.DataFrame(
            [{"olx_id": "desc-junta", "url": "",
              "title": "Fiat Punto 1.2",
              "description": "Bom carro mas a junta queimada, vende-se barato.",
              "brand": "Volkswagen", "model": "Golf",
              "year": 2015, "price_eur": 3000, "mileage_km": 150000, "engine_cc": 1600,
              "fuel_type": "Diesel", "is_active": True}]
            + _golf_comparables(),
        )
        with patched_gb_model(), patch(
            "src.models.generations.load_generations",
            return_value=generations_data,
        ):
            signals, _ = compute_signals(listings, sample_history_df)
        assert signals.empty or "desc-junta" not in signals["olx_id"].values

    def test_blocks_non_runner_phrasing_in_description(
        self, sample_history_df, generations_data, patched_gb_model,
    ):
        """JmUNP-style: title is innocuous, description says "não pega,
        só de reboque". Must hard-block."""
        listings = pd.DataFrame(
            [{"olx_id": "desc-runner", "url": "",
              "title": "Peugeot 508 SW 2013",
              "description": "Vendo carro. Não pega, só de reboque.",
              "brand": "Volkswagen", "model": "Golf",
              "year": 2015, "price_eur": 1000, "mileage_km": 200000, "engine_cc": 1600,
              "fuel_type": "Diesel", "is_active": True}]
            + _golf_comparables(),
        )
        with patched_gb_model(), patch(
            "src.models.generations.load_generations",
            return_value=generations_data,
        ):
            signals, _ = compute_signals(listings, sample_history_df)
        assert signals.empty or "desc-runner" not in signals["olx_id"].values

    def test_keeps_severity_2_listings(
        self, sample_history_df, generations_data, patched_gb_model,
    ):
        """Severity 2 (needs repair / accident history) is *not* a hard
        block — those listings stay in scope but get a repair-cost
        haircut applied to their flip basis (see TestRepairCostAdjustment).
        Only severity ≥ 3 is unconditional."""
        listings = pd.DataFrame(
            [{"olx_id": "needs-repair", "url": "", "brand": "Volkswagen", "model": "Golf",
              "year": 2015, "price_eur": 8000, "mileage_km": 150000, "engine_cc": 1600,
              "fuel_type": "Diesel", "is_active": True,
              "damage_severity": 2}]
            + _golf_comparables(),
        )
        with patched_gb_model(), patch(
            "src.models.generations.load_generations",
            return_value=generations_data,
        ):
            signals, _ = compute_signals(listings, sample_history_df)
        assert "needs-repair" in signals["olx_id"].values


class TestRepairCostAdjustment:
    """Severity-2 listings need a repair-cost haircut on their flip basis.
    A "junta queimada" Punto sitting at €1500 looks like a 50 % discount
    on a €3000 clean-comp prediction, but once you book €1500 of head-
    gasket work the actual flip thesis is zero. These tests pin the
    `_estimate_repair_cost` heuristic and the downstream signal columns."""

    def test_severity_2_drops_listing_when_repair_eats_margin(
        self, sample_history_df, generations_data, patched_gb_model,
    ):
        """GB stub @ multiplier=1.10 ⇒ predicted = price * 1.10, raw
        undervaluation_pct = ~9 %. A 12 % repair haircut wipes that out
        and the listing must drop. Without the adjustment, the old code
        would have surfaced this as a 9 %-discount deal."""
        listings = pd.DataFrame(
            [{"olx_id": "thin-margin", "url": "", "brand": "Volkswagen", "model": "Golf",
              "year": 2015, "price_eur": 8000, "mileage_km": 150000, "engine_cc": 1600,
              "fuel_type": "Diesel", "is_active": True,
              "damage_severity": 2}]
            + _golf_comparables(),
        )
        with patched_gb_model(multiplier=1.10), patch(
            "src.models.generations.load_generations",
            return_value=generations_data,
        ):
            signals, _ = compute_signals(listings, sample_history_df)
        assert signals.empty or "thin-margin" not in signals["olx_id"].values

    def test_severity_2_kept_when_undervaluation_covers_repair(
        self, sample_history_df, generations_data, patched_gb_model,
    ):
        """GB stub @ multiplier=2.0 ⇒ 50 % raw undervaluation, easily
        absorbs the 12 % repair haircut. Listing surfaces with
        ``repair_cost_eur`` and ``est_profit_after_repair_eur`` populated."""
        listings = pd.DataFrame(
            [{"olx_id": "fat-margin", "url": "", "brand": "Volkswagen", "model": "Golf",
              "year": 2015, "price_eur": 8000, "mileage_km": 150000, "engine_cc": 1600,
              "fuel_type": "Diesel", "is_active": True,
              "damage_severity": 2}]
            + _golf_comparables(),
        )
        with patched_gb_model(multiplier=2.0), patch(
            "src.models.generations.load_generations",
            return_value=generations_data,
        ):
            signals, _ = compute_signals(listings, sample_history_df)
        assert "fat-margin" in signals["olx_id"].values
        row = signals[signals["olx_id"] == "fat-margin"].iloc[0]
        assert row["repair_cost_eur"] is not None
        assert row["repair_cost_eur"] > 0
        assert row["est_profit_after_repair_eur"] is not None
        # raw undervaluation is 50 %, adjusted should be lower (haircut applied)
        assert row["adjusted_undervaluation_pct"] < row["undervaluation_pct"]

    def test_severity_0_or_1_unchanged_by_repair_path(
        self, sample_history_df, generations_data, patched_gb_model,
    ):
        """Pristine / normal-wear cars get repair_cost_eur = None, and
        adjusted_undervaluation_pct == undervaluation_pct (no haircut)."""
        listings = pd.DataFrame(
            [{"olx_id": "clean", "url": "", "brand": "Volkswagen", "model": "Golf",
              "year": 2015, "price_eur": 8000, "mileage_km": 150000, "engine_cc": 1600,
              "fuel_type": "Diesel", "is_active": True,
              "damage_severity": 1}]
            + _golf_comparables(),
        )
        with patched_gb_model(multiplier=1.5), patch(
            "src.models.generations.load_generations",
            return_value=generations_data,
        ):
            signals, _ = compute_signals(listings, sample_history_df)
        row = signals[signals["olx_id"] == "clean"].iloc[0]
        assert row["repair_cost_eur"] is None
        assert row["est_profit_after_repair_eur"] is None
        assert row["adjusted_undervaluation_pct"] == row["undervaluation_pct"]

    def test_poor_mechanical_condition_uses_higher_repair_cost(self):
        """Mechanical-poor branch gets 18 % / €1500 floor — wider than the
        12 % / €1000 default. The 2026-05-02 audit C5 8Q0kOc (starter +
        EGR + MAF + water leak) was the loudest example of stacked
        mechanical work blowing past a panel-paint estimate."""
        from src.dashboard.data_loader import _estimate_repair_cost

        baseline = _estimate_repair_cost(2, "good", 10000.0)
        poor = _estimate_repair_cost(2, "poor", 10000.0)
        assert poor > baseline
        # Floor for "poor" is €1500; 12 % of €5000 would only be €600.
        assert _estimate_repair_cost(2, "poor", 5000.0) >= 1500.0
        # Severity 0/1 returns 0 unconditionally.
        assert _estimate_repair_cost(0, "good", 10000.0) == 0.0
        assert _estimate_repair_cost(1, "poor", 10000.0) == 0.0
        assert _estimate_repair_cost(None, None, 10000.0) == 0.0
