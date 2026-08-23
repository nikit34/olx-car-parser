"""Tests for the LLM-enrichment domain rules: corrections, validation, export.

Transport lives in test_cloud_enrichment.py — this module is about what a
listing looks like AFTER an extraction comes back, and the rules that keep a
model's guess from writing nonsense into a column: brand-family validation of
sub_model, mileage sanity bounds, and the deterministic damage_severity
derivation that needs no model at all.
"""

from dataclasses import dataclass
from unittest.mock import patch

from src.parser.llm_enrichment import (
    correct_listing_data,
    apply_corrections,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class FakeListing:
    olx_id: str = "test-001"
    url: str = "https://olx.pt/test"
    description: str = "Vendo BMW 320d com 180.000km"
    mileage_km: int | None = 150000
    origin: str | None = None
    brand: str = "BMW"
    title: str = ""


VALID_LLM_JSON = {
    "sub_model": "320d",
    "trim_level": None,
    "mileage_in_description_km": 180000,
}


# ---------------------------------------------------------------------------
# Data correction — cross-checking an extraction against the listing
# ---------------------------------------------------------------------------

class TestCorrectListingData:
    def test_mileage_mismatch_uses_description(self):
        listing = FakeListing(mileage_km=100000)
        listing._llm_extras = {"mileage_in_description_km": 180000}
        corrections = correct_listing_data(listing)
        assert corrections["real_mileage_km"] == 180000

    def test_mileage_close_uses_description(self):
        listing = FakeListing(mileage_km=150000)
        listing._llm_extras = {"mileage_in_description_km": 155000}
        corrections = correct_listing_data(listing)
        assert corrections["real_mileage_km"] == 155000

    def test_attribute_higher_uses_description(self):
        listing = FakeListing(mileage_km=300000)
        listing._llm_extras = {"mileage_in_description_km": 100000}
        corrections = correct_listing_data(listing)
        assert corrections["real_mileage_km"] == 100000

    def test_description_too_low_falls_back_to_attribute(self):
        # JltT9 (2026-05) — title polluted with price ("9.000 €") made the
        # LLM emit 9000 km against an OLX attr of 355000. 10×-or-more gap
        # downward is treated as a parse error, like the symmetric upward case.
        listing = FakeListing(mileage_km=355000)
        listing._llm_extras = {"mileage_in_description_km": 9000}
        corrections = correct_listing_data(listing)
        assert corrections["real_mileage_km"] == 355000

    def test_no_attribute_mileage_uses_description(self):
        listing = FakeListing(mileage_km=0)
        listing._llm_extras = {"mileage_in_description_km": 120000}
        corrections = correct_listing_data(listing)
        assert corrections["real_mileage_km"] == 120000

    def test_no_description_mileage_falls_back_to_attribute(self):
        listing = FakeListing(mileage_km=95000)
        listing._llm_extras = {}
        corrections = correct_listing_data(listing)
        assert corrections["real_mileage_km"] == 95000

    def test_sub_model_and_trim_passed_through(self):
        listing = FakeListing()
        listing._llm_extras = {"sub_model": "320d", "trim_level": "M Sport"}
        corrections = correct_listing_data(listing)
        assert corrections["sub_model"] == "320d"
        assert corrections["trim_level"] == "M Sport"

    # 2026-05-10 audit regression — see _validate_sub_model docstring.

    def test_sub_model_drops_psa_tag_on_vag_brand(self):
        listing = FakeListing(brand="Audi")
        listing._llm_extras = {"sub_model": "2.0 HDi"}
        corrections = correct_listing_data(listing)
        assert "sub_model" not in corrections

    def test_sub_model_drops_psa_tag_on_mercedes(self):
        listing = FakeListing(brand="Mercedes-Benz")
        listing._llm_extras = {"sub_model": "2.0 HDi"}
        corrections = correct_listing_data(listing)
        assert "sub_model" not in corrections

    def test_sub_model_drops_gm_tag_on_fiat(self):
        listing = FakeListing(brand="Fiat")
        listing._llm_extras = {"sub_model": "1.3 CDTI"}
        corrections = correct_listing_data(listing)
        assert "sub_model" not in corrections

    def test_sub_model_drops_vag_tag_on_bmw(self):
        listing = FakeListing(brand="BMW")
        listing._llm_extras = {"sub_model": "1.6 TDI"}
        corrections = correct_listing_data(listing)
        assert "sub_model" not in corrections

    def test_sub_model_keeps_correct_family_tag(self):
        listing = FakeListing(brand="Audi")
        listing._llm_extras = {"sub_model": "2.0 TDI"}
        corrections = correct_listing_data(listing)
        assert corrections["sub_model"] == "2.0 TDI"

    def test_sub_model_keeps_mercedes_cdi(self):
        listing = FakeListing(brand="Mercedes-Benz")
        listing._llm_extras = {"sub_model": "220 CDI"}
        corrections = correct_listing_data(listing)
        assert corrections["sub_model"] == "220 CDI"

    def test_sub_model_keeps_renault_dci(self):
        listing = FakeListing(brand="Renault")
        listing._llm_extras = {"sub_model": "1.5 dCi"}
        corrections = correct_listing_data(listing)
        assert corrections["sub_model"] == "1.5 dCi"

    def test_sub_model_keeps_bmw_xxxd(self):
        listing = FakeListing(brand="BMW")
        listing._llm_extras = {"sub_model": "320d"}
        corrections = correct_listing_data(listing)
        assert corrections["sub_model"] == "320d"

    def test_sub_model_passes_through_unmapped_brand(self):
        # Opel straddles GM/PSA eras — validator must not reject either.
        listing = FakeListing(brand="Opel")
        listing._llm_extras = {"sub_model": "1.6 CDTI"}
        corrections = correct_listing_data(listing)
        assert corrections["sub_model"] == "1.6 CDTI"

    def test_sub_model_no_tech_tag_unaffected(self):
        # Bare displacement / Mercedes class names have no recognized tech
        # tag — validator must pass them through unchanged regardless of brand.
        listing = FakeListing(brand="Audi")
        listing._llm_extras = {"sub_model": "2.0"}
        corrections = correct_listing_data(listing)
        assert corrections["sub_model"] == "2.0"

    def test_damage_severity_derived_from_text(self):
        # Parts-car phrasing in description → severity 3 even when extras are empty.
        listing = FakeListing(description="Vendo unicamente para peças, motor avariado")
        listing._llm_extras = {}
        corrections = correct_listing_data(listing)
        assert corrections["damage_severity"] == 3

    def test_damage_severity_default_normal_wear(self):
        listing = FakeListing(description="Vendo Honda Civic 2018 com 90000km, sempre assistido")
        listing._llm_extras = {}
        corrections = correct_listing_data(listing)
        assert corrections["damage_severity"] == 1

    def test_no_extras_returns_empty(self):
        listing = FakeListing()
        corrections = correct_listing_data(listing)
        assert corrections == {}

    def test_implausible_mileage_falls_back_to_attribute(self):
        """The 2026-05-02 audit found Honda Civic JmuYR with
        ``real_mileage_km = 278_000_000`` because the LLM mis-parsed
        "278 mil km" as ``278000 * 1000``. Anything > 1M km is a parse
        error — fall back to the structured attribute."""
        listing = FakeListing(mileage_km=210000)
        listing._llm_extras = {"mileage_in_description_km": 278_000_000}
        corrections = correct_listing_data(listing)
        assert corrections["real_mileage_km"] == 210000

    def test_mileage_more_than_10x_attribute_falls_back(self):
        """Catches narrower unit-suffix mis-reads (e.g. "120 mil km" parsed
        as 1_200_000) where the absolute cap doesn't fire but the LLM
        value is still implausible relative to the OLX attribute."""
        listing = FakeListing(mileage_km=120000)
        listing._llm_extras = {"mileage_in_description_km": 1_200_001}
        corrections = correct_listing_data(listing)
        assert corrections["real_mileage_km"] == 120000

    def test_no_attribute_caps_implausible_mileage(self):
        """Without an attribute baseline, the absolute cap still applies —
        we'd rather drop the LLM value than write 278M km to the DB."""
        listing = FakeListing(mileage_km=0)
        listing._llm_extras = {"mileage_in_description_km": 5_000_000}
        corrections = correct_listing_data(listing)
        # No attribute, no plausible LLM read → no real_mileage_km correction.
        assert "real_mileage_km" not in corrections


# ---------------------------------------------------------------------------
# apply_corrections
# ---------------------------------------------------------------------------

class TestApplyCorrections:
    def test_applies_to_listings_with_extras(self):
        listing = FakeListing(description="BMW 320d 2018 com 180000km, sempre assistido")
        listing._llm_extras = {
            "sub_model": "320d",
            "trim_level": "M Sport",
            "mileage_in_description_km": 180000,
        }
        count = apply_corrections([listing])
        assert count == 1
        assert listing._corrections["sub_model"] == "320d"
        assert listing._corrections["trim_level"] == "M Sport"
        assert listing._corrections["real_mileage_km"] == 180000

    def test_skips_listings_without_extras(self):
        listing = FakeListing()
        count = apply_corrections([listing])
        assert count == 0
        assert not hasattr(listing, "_corrections")


# ---------------------------------------------------------------------------
# Pipeline: multiprocessing-based LLM worker (uses actual _llm_worker from CLI)
# ---------------------------------------------------------------------------
# scraper on_detail_ready callback
# ---------------------------------------------------------------------------

class TestScraperCallback:
    def test_enrich_one_calls_callback(self):
        """_enrich_one calls on_detail_ready when listing has description."""
        from src.parser.scraper import OlxScraper, ScraperConfig, RawListing

        scraper = OlxScraper(ScraperConfig(delay_min=0, delay_max=0))
        callback_received = []

        listing = RawListing(olx_id="cb-1", url="https://olx.pt/test")

        fake_details = {"description": "Carro em bom estado, 120000km"}
        with patch.object(scraper, "scrape_listing_detail", return_value=fake_details):
            with patch.object(scraper, "_delay"):
                scraper._enrich_one(listing, on_ready=lambda l: callback_received.append(l))

        assert len(callback_received) == 1
        assert callback_received[0].description == "Carro em bom estado, 120000km"

    def test_enrich_one_no_callback_without_description(self):
        """_enrich_one does NOT call callback when listing has no description."""
        from src.parser.scraper import OlxScraper, ScraperConfig, RawListing

        scraper = OlxScraper(ScraperConfig(delay_min=0, delay_max=0))
        callback_received = []

        listing = RawListing(olx_id="cb-2", url="https://olx.pt/test")

        with patch.object(scraper, "scrape_listing_detail", return_value={}):
            with patch.object(scraper, "_delay"):
                scraper._enrich_one(listing, on_ready=lambda l: callback_received.append(l))

        assert len(callback_received) == 0


class TestDeriveDamageSeverity:
    """Rule-based derivation for the backfill path. Validated 100%
    LLM-equivalent on data/eval/qwen3_4b-instruct.jsonl."""

    def test_parts_only_returns_3(self):
        from src.parser.llm_enrichment import _derive_damage_severity
        assert _derive_damage_severity(
            {}, "Honda Civic 2009", "Vendo para peças, sucata.",
        ) == 3

    def test_no_plates_returns_3(self):
        from src.parser.llm_enrichment import _derive_damage_severity
        assert _derive_damage_severity(
            {}, "Nissan Qashqai", "SEM MATRICULA, para exportação.",
        ) == 3

    def test_severe_damage_returns_2_or_3(self):
        from src.parser.llm_enrichment import _derive_damage_severity
        # Whole car, broken — severity 2
        assert _derive_damage_severity(
            {"mechanical_condition": "fair"}, "BMW", "Motor fundido.",
        ) == 2
        # And condition=poor on top → severity 3
        assert _derive_damage_severity(
            {"mechanical_condition": "poor"}, "BMW", "Motor fundido.",
        ) == 3

    def test_nao_liga_returns_3(self):
        """Audit case 8Q0kOc (Citroën C5): description literally says
        "O carro não liga devido a essas avarias" — the original regex
        only caught "não pega", missing this Portuguese variant."""
        from src.parser.llm_enrichment import _derive_damage_severity
        assert _derive_damage_severity(
            {}, "Citroën C5", "O carro não liga devido a essas avarias.",
        ) == 3
        assert _derive_damage_severity(
            {}, "Citroën C5",
            "Não é possível testar. Vendido no estado em que se encontra.",
        ) == 3

    def test_non_runner_returns_3_unconditionally(self):
        """``não pega`` / ``só reboque`` are non-runner — severity 3 even
        when mechanical_condition is "fair" or "good" (the body might be
        fine, but a car you have to tow has no flip thesis). Audit
        cases: Peugeot 508 JmUNP ("não pega, só de reboque", condition
        "fair") and Citroën C5 8Q0kOc ("não pega").
        """
        from src.parser.llm_enrichment import _derive_damage_severity
        assert _derive_damage_severity(
            {"mechanical_condition": "fair"}, "Peugeot 508 SW", "Não pega.",
        ) == 3
        assert _derive_damage_severity(
            {"mechanical_condition": "good"}, "Citroën C5", "Só de reboque.",
        ) == 3
        assert _derive_damage_severity(
            {}, "BMW", "Engine seized, parted out engine.",
        ) == 3

    def test_junta_queimada_returns_2(self):
        """Blown head gasket — fixable with money, so severity 2 by
        default (3 only if condition is also "poor"). Fiat Punto JmutI
        from the audit."""
        from src.parser.llm_enrichment import _derive_damage_severity
        assert _derive_damage_severity(
            {}, "Fiat Punto", "Junta queimada, vende-se barato.",
        ) == 2
        assert _derive_damage_severity(
            {"mechanical_condition": "poor"}, "Fiat Punto",
            "Junta queimada.",
        ) == 3

    def test_avaria_no_motor_returns_2(self):
        """Passat JmR3C: "avaria no motor" — severity 2."""
        from src.parser.llm_enrichment import _derive_damage_severity
        assert _derive_damage_severity(
            {}, "VW Passat", "Avaria no motor, vende-se a peças ou inteiro.",
        ) == 3  # "vende-se a peças" hits parts-only path first
        assert _derive_damage_severity(
            {}, "VW Passat", "Avaria no motor.",
        ) == 2

    def test_accident_or_repair_flag_returns_2(self):
        from src.parser.llm_enrichment import _derive_damage_severity
        assert _derive_damage_severity(
            {"desc_mentions_accident": True}, "VW Golf", "Sofreu sinistro.",
        ) == 2
        assert _derive_damage_severity(
            {"desc_mentions_repair": True}, "Renault", "Precisa de reparações.",
        ) == 2

    def test_excellent_condition_returns_0(self):
        from src.parser.llm_enrichment import _derive_damage_severity
        assert _derive_damage_severity(
            {"mechanical_condition": "excellent"}, "Audi", "Boa máquina.",
        ) == 0

    def test_pristine_keywords_return_0(self):
        from src.parser.llm_enrichment import _derive_damage_severity
        assert _derive_damage_severity(
            {}, "Mercedes", "Veículo como novo, estado impecável.",
        ) == 0
        assert _derive_damage_severity(
            {}, "Porsche Cayenne", "FULL EXTRAS, todas as opções.",
        ) == 0

    def test_warranty_flag_returns_0(self):
        from src.parser.llm_enrichment import _derive_damage_severity
        assert _derive_damage_severity(
            {"warranty": True}, "VW Golf 2022", "Carro normal de família.",
        ) == 0

    def test_default_normal_wear_returns_1(self):
        from src.parser.llm_enrichment import _derive_damage_severity
        assert _derive_damage_severity(
            {}, "Peugeot 208", "Vendo Peugeot 208 de 2018, 90000 km.",
        ) == 1

    def test_legacy_aliases_for_accident_repair(self):
        """Old llm_extras dicts use had_accident / needs_repair instead of
        the current desc_mentions_* names — the rule must read both."""
        from src.parser.llm_enrichment import _derive_damage_severity
        assert _derive_damage_severity(
            {"had_accident": True}, "BMW", "Carro nacional.",
        ) == 2
        assert _derive_damage_severity(
            {"needs_repair": True}, "Audi", "Vende-se.",
        ) == 2


