"""Tests for LLM enrichment: Ollama, pipeline, corrections, export."""

import json
import queue
import threading
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from src.parser.llm_enrichment import (
    _call_ollama,
    _parse_llm_json,
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
    "num_owners": 2,
    "accident_free": True,
    "had_accident": False,
    "accident_details": None,
    "service_history": True,
    "needs_repair": True,
    "repair_details": "precisa de embraiagem",
    "estimated_repair_cost_eur": 800,
    "mileage_in_description_km": 180000,
    "customs_cleared": None,
    "imported": None,
    "legal_issues": [],
    "mechanical_condition": "good",
    "paint_condition": "good",
    "suspicious_signs": [],
    "extras": ["GPS", "bancos em pele"],
    "issues": ["embraiagem"],
    "reason_for_sale": None,
}


# ---------------------------------------------------------------------------
# _parse_llm_json
# ---------------------------------------------------------------------------

class TestParseLlmJson:
    def test_plain_json(self):
        result = _parse_llm_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_markdown_code_block(self):
        text = '```json\n{"key": "value"}\n```'
        result = _parse_llm_json(text)
        assert result == {"key": "value"}

    def test_markdown_without_language(self):
        text = '```\n{"key": 123}\n```'
        result = _parse_llm_json(text)
        assert result == {"key": 123}

    def test_invalid_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            _parse_llm_json("not json at all")


# ---------------------------------------------------------------------------
# _call_ollama
# ---------------------------------------------------------------------------

class TestCallOllama:
    def test_success(self):
        cfg = {"ollama_url": "http://localhost:11434", "ollama_model": "qwen2.5:1.5b"}
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "response": json.dumps(VALID_LLM_JSON)
        }

        with patch("src.parser.llm_enrichment.httpx.post", return_value=mock_resp):
            result = _call_ollama("Vendo carro com 100km", cfg)

        assert result is not None
        assert result["had_accident"] is False

    def test_api_error(self):
        cfg = {"ollama_url": "http://localhost:11434", "ollama_model": "qwen2.5:1.5b"}
        mock_resp = MagicMock()
        mock_resp.status_code = 500

        with patch("src.parser.llm_enrichment.httpx.post", return_value=mock_resp):
            result = _call_ollama("Vendo carro", cfg)

        assert result is None

    def test_invalid_json_response(self):
        cfg = {"ollama_url": "http://localhost:11434", "ollama_model": "qwen2.5:1.5b"}
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"response": "Sorry I cannot help"}

        with patch("src.parser.llm_enrichment.httpx.post", return_value=mock_resp):
            result = _call_ollama("Vendo carro", cfg)

        assert result is None


# ---------------------------------------------------------------------------
# enrich_from_description
# ---------------------------------------------------------------------------

class TestEnrichFromDescription:
    def test_empty_description_returns_none(self):
        assert enrich_from_description("") is None
        assert enrich_from_description("short") is None

    @patch("src.parser.llm_enrichment._ollama_available", return_value=True)
    @patch("src.parser.llm_enrichment._call_ollama", return_value=VALID_LLM_JSON)
    def test_calls_ollama(self, mock_ollama, mock_avail):
        result = enrich_from_description("Vendo BMW 320d com 180.000km reais")
        assert result == VALID_LLM_JSON
        mock_ollama.assert_called_once()

    @patch("src.parser.llm_enrichment._ollama_available", return_value=True)
    @patch("src.parser.llm_enrichment._call_ollama", return_value=None)
    def test_returns_none_on_ollama_failure(self, mock_ollama, mock_avail):
        result = enrich_from_description("Vendo BMW 320d com 180.000km reais")
        assert result is None

    @patch("src.parser.llm_enrichment._ollama_available", return_value=False)
    def test_returns_none_when_ollama_unavailable(self, mock_avail):
        result = enrich_from_description("Vendo BMW 320d com 180.000km reais")
        assert result is None


# ---------------------------------------------------------------------------
# correct_listing_data — cross-check logic
# ---------------------------------------------------------------------------

class TestCorrectListingData:
    def test_mileage_mismatch_flags_suspect(self):
        listing = FakeListing(mileage_km=100000)
        listing._llm_extras = {"mileage_in_description_km": 180000}
        corrections = correct_listing_data(listing)
        assert corrections["mileage_suspect"] is True
        assert corrections["real_mileage_km"] == 180000

    def test_mileage_close_no_flag(self):
        listing = FakeListing(mileage_km=150000)
        listing._llm_extras = {"mileage_in_description_km": 155000}
        corrections = correct_listing_data(listing)
        assert "mileage_suspect" not in corrections

    def test_mileage_typo_detection(self):
        listing = FakeListing(mileage_km=300000)
        listing._llm_extras = {"mileage_in_description_km": 100000}
        corrections = correct_listing_data(listing)
        assert corrections["mileage_suspect"] is True
        assert corrections["real_mileage_km"] == 100000

    def test_no_attribute_mileage_uses_description(self):
        listing = FakeListing(mileage_km=0)
        listing._llm_extras = {"mileage_in_description_km": 120000}
        corrections = correct_listing_data(listing)
        assert corrections["real_mileage_km"] == 120000
        assert corrections["mileage_suspect"] is False

    def test_needs_repair_from_extras(self):
        listing = FakeListing()
        listing._llm_extras = {"needs_repair": True}
        corrections = correct_listing_data(listing)
        assert corrections["needs_repair"] is True

    def test_needs_repair_inferred_from_issues(self):
        listing = FakeListing()
        listing._llm_extras = {"needs_repair": None, "issues": ["motor faz barulho"]}
        corrections = correct_listing_data(listing)
        assert corrections["needs_repair"] is True

    def test_accident_from_accident_free(self):
        listing = FakeListing()
        listing._llm_extras = {"had_accident": None, "accident_free": True}
        corrections = correct_listing_data(listing)
        assert corrections["had_accident"] is False

    def test_customs_cleared(self):
        listing = FakeListing(origin="Importado")
        listing._llm_extras = {"customs_cleared": True}
        corrections = correct_listing_data(listing)
        assert corrections["customs_cleared"] is True

    def test_repair_cost(self):
        listing = FakeListing()
        listing._llm_extras = {"estimated_repair_cost_eur": 1500}
        corrections = correct_listing_data(listing)
        assert corrections["estimated_repair_cost_eur"] == 1500

    def test_no_extras_returns_empty(self):
        listing = FakeListing()
        corrections = correct_listing_data(listing)
        assert corrections == {}


# ---------------------------------------------------------------------------
# apply_corrections
# ---------------------------------------------------------------------------

class TestApplyCorrections:
    def test_applies_to_listings_with_extras(self):
        listing = FakeListing()
        listing._llm_extras = {"needs_repair": True, "had_accident": False}
        count = apply_corrections([listing])
        assert count == 1
        assert listing._corrections["needs_repair"] is True
        assert listing._corrections["had_accident"] is False

    def test_skips_listings_without_extras(self):
        listing = FakeListing()
        count = apply_corrections([listing])
        assert count == 0
        assert not hasattr(listing, "_corrections")


# ---------------------------------------------------------------------------
# Pipeline: queue-based LLM worker
# ---------------------------------------------------------------------------

class TestLlmPipeline:
    @patch("src.parser.llm_enrichment.enrich_from_description", return_value=VALID_LLM_JSON)
    def test_queue_feeds_llm_worker(self, mock_enrich):
        """Simulate the CLI pipeline: scraper puts listings in queue, worker processes."""
        from src.parser.llm_enrichment import enrich_from_description

        llm_queue: queue.Queue = queue.Queue()
        llm_done = threading.Event()
        results = []

        def _worker():
            while True:
                try:
                    listing = llm_queue.get(timeout=1)
                except queue.Empty:
                    if llm_done.is_set():
                        break
                    continue
                if listing is None:
                    break
                result = enrich_from_description(listing.description)
                if result:
                    listing._llm_extras = result
                    results.append(listing)
                llm_queue.task_done()

        worker = threading.Thread(target=_worker, daemon=True)
        worker.start()

        # Simulate scraper producing listings
        listings = [FakeListing(olx_id=f"test-{i}",
                                description=f"Vendo carro {i} com {i*50000}km")
                    for i in range(5)]
        for l in listings:
            llm_queue.put(l)

        llm_done.set()
        llm_queue.put(None)
        worker.join(timeout=10)

        assert len(results) == 5
        assert all(hasattr(l, "_llm_extras") for l in results)
        assert mock_enrich.call_count == 5

    @patch("src.parser.llm_enrichment.enrich_from_description", return_value=None)
    def test_worker_handles_failures(self, mock_enrich):
        """Worker continues processing even if LLM returns None."""
        llm_queue: queue.Queue = queue.Queue()
        llm_done = threading.Event()
        processed = []

        def _worker():
            while True:
                try:
                    listing = llm_queue.get(timeout=1)
                except queue.Empty:
                    if llm_done.is_set():
                        break
                    continue
                if listing is None:
                    break
                from src.parser.llm_enrichment import enrich_from_description
                enrich_from_description(listing.description)
                processed.append(listing)
                llm_queue.task_done()

        worker = threading.Thread(target=_worker, daemon=True)
        worker.start()

        for i in range(3):
            llm_queue.put(FakeListing(olx_id=f"fail-{i}"))

        llm_done.set()
        llm_queue.put(None)
        worker.join(timeout=10)

        assert len(processed) == 3


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


# ---------------------------------------------------------------------------
# export-training-data CLI command
# ---------------------------------------------------------------------------

class TestExportTrainingData:
    def test_export_creates_jsonl(self, db_session, sample_listing_data, tmp_path):
        from src.storage.repository import upsert_listing
        from src.parser.llm_enrichment import EXTRACTION_PROMPT

        data = {**sample_listing_data,
                "description": "Vendo Golf VII 2015, 100.000km, diesel, sem acidentes.",
                "llm_extras": json.dumps(VALID_LLM_JSON)}
        upsert_listing(db_session, data)
        db_session.commit()

        out_path = tmp_path / "train.jsonl"

        # Call export logic directly (avoid CLI runner complexity)
        from src.storage.repository import get_listings_df
        df = get_listings_df(db_session)

        count = 0
        with open(out_path, "w") as f:
            for _, row in df.iterrows():
                desc = row.get("description") or ""
                extras_raw = row.get("llm_extras")
                if len(desc.strip()) < 50 or not extras_raw:
                    continue
                extras = json.loads(extras_raw) if isinstance(extras_raw, str) else extras_raw
                entry = {
                    "messages": [
                        {"role": "user", "content": EXTRACTION_PROMPT + desc[:3000]},
                        {"role": "assistant", "content": json.dumps(extras, ensure_ascii=False)},
                    ],
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                count += 1

        assert count == 1
        line = json.loads(out_path.read_text().strip())
        assert "messages" in line
        assert len(line["messages"]) == 2
        assert line["messages"][0]["role"] == "user"
        assert line["messages"][1]["role"] == "assistant"
        # Verify the completion is valid JSON
        completion = json.loads(line["messages"][1]["content"])
        assert completion["needs_repair"] is True
