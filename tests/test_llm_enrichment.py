"""Tests for LLM enrichment: Claude API, pipeline, corrections, export."""

import json
import queue
import threading
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

import src.parser.llm_enrichment as llm_mod
from src.parser.llm_enrichment import (
    _call_llm,
    _get_config,
    correct_listing_data,
    apply_corrections,
    enrich_from_description,
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


VALID_LLM_JSON = {
    "sub_model": "320d",
    "trim_level": None,
    "desc_mentions_num_owners": 2,
    "desc_mentions_accident": False,
    "desc_mentions_repair": True,
    "mileage_in_description_km": 180000,
    "desc_mentions_customs_cleared": None,
    "right_hand_drive": None,
    "mechanical_condition": "good",
    "urgency": "low",
    "warranty": None,
    "tuning_or_mods": None,
    "taxi_fleet_rental": None,
    "first_owner_selling": None,
}


# ---------------------------------------------------------------------------
# _call_llm — Claude tool-use round-trip
# ---------------------------------------------------------------------------

def _make_claude_resp(tool_input: dict | None):
    """Build a fake `messages.create()` return value with a single tool_use block."""
    block = MagicMock()
    block.type = "tool_use"
    block.name = "record_listing_features"
    block.input = tool_input or {}
    resp = MagicMock()
    resp.content = [block] if tool_input is not None else []
    return resp


class TestCallLlm:
    def setup_method(self):
        llm_mod._client = None
        llm_mod._token_status = None

    def test_success(self):
        cfg = _get_config()
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_claude_resp(VALID_LLM_JSON)

        with patch("src.parser.llm_enrichment._get_client", return_value=mock_client):
            result = _call_llm("Vendo carro com 100km", cfg)

        assert result is not None
        assert result["desc_mentions_accident"] is False
        # The system prompt must be sent with cache_control so prompt caching
        # actually engages — this is how the per-call cost stays low.
        kwargs = mock_client.messages.create.call_args.kwargs
        assert kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}
        assert kwargs["tool_choice"]["name"] == "record_listing_features"

    def test_no_tool_use_block(self):
        cfg = _get_config()
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_claude_resp(None)

        with patch("src.parser.llm_enrichment._get_client", return_value=mock_client):
            result = _call_llm("Vendo carro", cfg)

        assert result is None

    def test_no_client_returns_none(self):
        cfg = _get_config()
        with patch("src.parser.llm_enrichment._get_client", return_value=None):
            result = _call_llm("Vendo carro", cfg)
        assert result is None


# ---------------------------------------------------------------------------
# enrich_from_description
# ---------------------------------------------------------------------------

class TestEnrichFromDescription:
    def test_empty_description_returns_none(self):
        assert enrich_from_description("") is None
        assert enrich_from_description("short") is None

    @patch("src.parser.llm_enrichment._llm_available", return_value=True)
    @patch("src.parser.llm_enrichment._call_llm", return_value=VALID_LLM_JSON)
    def test_calls_claude(self, mock_call, mock_avail):
        result = enrich_from_description("Vendo BMW 320d com 180.000km reais")
        assert result == VALID_LLM_JSON
        mock_call.assert_called_once()

    @patch("src.parser.llm_enrichment._llm_available", return_value=True)
    @patch("src.parser.llm_enrichment._call_llm", return_value=None)
    def test_returns_none_on_claude_failure(self, mock_call, mock_avail):
        result = enrich_from_description("Vendo BMW 320d com 180.000km reais")
        assert result is None

    @patch("src.parser.llm_enrichment._llm_available", return_value=False)
    def test_returns_none_when_token_missing(self, mock_avail):
        result = enrich_from_description("Vendo BMW 320d com 180.000km reais")
        assert result is None


# ---------------------------------------------------------------------------
# correct_listing_data — cross-check logic
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

    def test_num_owners_from_description(self):
        listing = FakeListing()
        listing._llm_extras = {"desc_mentions_num_owners": 2}
        corrections = correct_listing_data(listing)
        assert corrections["desc_mentions_num_owners"] == 2

    def test_num_owners_missing(self):
        listing = FakeListing()
        listing._llm_extras = {"desc_mentions_num_owners": None}
        corrections = correct_listing_data(listing)
        assert "desc_mentions_num_owners" not in corrections

    def test_needs_repair_from_extras(self):
        # Post-rules require a damage keyword in the description for
        # desc_mentions_repair=True to pass through — otherwise the flag is
        # assumed to be an over-flag on routine maintenance phrases.
        listing = FakeListing(description="BMW 320d avariado, precisa reparar")
        listing._llm_extras = {"desc_mentions_repair": True}
        corrections = correct_listing_data(listing)
        assert corrections["desc_mentions_repair"] is True

    def test_needs_repair_reverted_without_damage_keywords(self):
        # Same flag but a clean description — post-rules assume the LLM
        # conflated maintenance with damage and revert the flag.
        listing = FakeListing(description="BMW 320d em estado impecável, revisão feita")
        listing._llm_extras = {"desc_mentions_repair": True}
        corrections = correct_listing_data(listing)
        assert corrections["desc_mentions_repair"] is False

    def test_parts_car_override_forces_triple(self):
        # Parts-car listings must always come back with all three flags set,
        # regardless of what the LLM returned.
        listing = FakeListing(description="Vendo unicamente para peças, motor avariado")
        listing._llm_extras = {
            "desc_mentions_repair": False,
            "desc_mentions_accident": False,
            "mechanical_condition": "good",
        }
        corrections = correct_listing_data(listing)
        assert corrections["desc_mentions_repair"] is True
        assert corrections["desc_mentions_accident"] is True
        assert corrections["mechanical_condition"] == "poor"

    def test_rhd_reverted_without_explicit_phrase(self):
        # Generic "importado" must not be accepted as RHD evidence.
        listing = FakeListing(description="Nissan Qashqai importado da Bélgica, legalizado")
        listing._llm_extras = {"right_hand_drive": True}
        corrections = correct_listing_data(listing)
        assert corrections["right_hand_drive"] is False

    def test_rhd_kept_with_explicit_phrase(self):
        listing = FakeListing(description="Carro com matrícula inglesa, documentação em dia")
        listing._llm_extras = {"right_hand_drive": True}
        corrections = correct_listing_data(listing)
        assert corrections["right_hand_drive"] is True

    def test_needs_repair_not_set_when_null(self):
        listing = FakeListing()
        listing._llm_extras = {"desc_mentions_repair": None}
        corrections = correct_listing_data(listing)
        assert "desc_mentions_repair" not in corrections

    def test_accident_explicit_false(self):
        listing = FakeListing()
        listing._llm_extras = {"desc_mentions_accident": False}
        corrections = correct_listing_data(listing)
        assert corrections["desc_mentions_accident"] is False

    def test_customs_cleared(self):
        listing = FakeListing(origin="Importado")
        listing._llm_extras = {"desc_mentions_customs_cleared": True}
        corrections = correct_listing_data(listing)
        assert corrections["desc_mentions_customs_cleared"] is True

    def test_no_extras_returns_empty(self):
        listing = FakeListing()
        corrections = correct_listing_data(listing)
        assert corrections == {}


# ---------------------------------------------------------------------------
# apply_corrections
# ---------------------------------------------------------------------------

class TestApplyCorrections:
    def test_applies_to_listings_with_extras(self):
        # Description has a damage keyword ("partido") but NOT a parts-car
        # trigger — so desc_mentions_repair stays True while accident=False
        # is preserved (the parts-car override would otherwise force both).
        listing = FakeListing(description="BMW 320d com para-choques partido, resto bem")
        listing._llm_extras = {"desc_mentions_repair": True, "desc_mentions_accident": False}
        count = apply_corrections([listing])
        assert count == 1
        assert listing._corrections["desc_mentions_repair"] is True
        assert listing._corrections["desc_mentions_accident"] is False

    def test_skips_listings_without_extras(self):
        listing = FakeListing()
        count = apply_corrections([listing])
        assert count == 0
        assert not hasattr(listing, "_corrections")


# ---------------------------------------------------------------------------
# Pipeline: multiprocessing-based LLM worker (uses actual _llm_worker from CLI)
# ---------------------------------------------------------------------------

class TestLlmPipeline:
    @patch("src.parser.llm_enrichment._llm_available", return_value=True)
    @patch("src.parser.llm_enrichment.enrich_from_description", return_value=VALID_LLM_JSON)
    def test_queue_feeds_llm_worker(self, mock_enrich, mock_avail):
        """Simulate the CLI pipeline: scraper puts (olx_id, desc) in queue, worker processes."""
        import multiprocessing
        import queue
        from src.cli import _llm_worker

        # Use queue.Queue (not multiprocessing.Queue) since worker runs as a
        # thread here — avoids the race where mp.Queue's internal feeder daemon
        # hasn't flushed the pipe by the time we check empty().
        in_q = queue.Queue()
        out_q = queue.Queue()
        shutdown = multiprocessing.Event()

        worker = threading.Thread(target=_llm_worker, args=(in_q, out_q, shutdown))
        worker.start()

        for i in range(5):
            in_q.put((f"test-{i}", f"Test Car {i}", f"Vendo carro {i} com {i*50000}km muitos detalhes"))

        in_q.put(None)  # poison pill
        worker.join(timeout=10)

        results = []
        while not out_q.empty():
            results.append(out_q.get_nowait())
        assert len(results) == 5
        assert all(r[1] == VALID_LLM_JSON for r in results)

    @patch("src.parser.llm_enrichment._llm_available", return_value=True)
    @patch("src.parser.llm_enrichment.enrich_from_description", return_value=None)
    def test_worker_handles_failures(self, mock_enrich, mock_avail):
        """Worker sends None results and exits after 5 consecutive failures."""
        import multiprocessing
        import queue
        from src.cli import _llm_worker

        in_q = queue.Queue()
        out_q = queue.Queue()
        shutdown = multiprocessing.Event()

        for i in range(7):
            in_q.put((f"fail-{i}", f"Car {i}", f"Vendo carro numero {i} com muitos quilometros"))

        worker = threading.Thread(target=_llm_worker, args=(in_q, out_q, shutdown))
        worker.start()
        worker.join(timeout=10)

        results = []
        while not out_q.empty():
            results.append(out_q.get_nowait())
        # Exits after 5 consecutive failures
        assert len(results) == 5
        assert all(r[1] is None for r in results)


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


