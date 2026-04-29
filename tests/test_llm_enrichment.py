"""Tests for LLM enrichment: Ollama call, pipeline, corrections, export."""

import json
import queue
import threading
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import httpx as httpx_mod
import pytest

import src.parser.llm_enrichment as llm_mod
from src.parser.llm_enrichment import (
    _call_llm,
    _call_ollama,
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
    "mileage_in_description_km": 180000,
}


# ---------------------------------------------------------------------------
# _call_llm / _call_ollama — Ollama JSON-mode round-trip
# ---------------------------------------------------------------------------

def _make_ollama_resp(content: str, status: int = 200):
    """Fake httpx.post() return value matching Ollama's /api/generate shape."""
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = {"response": content}
    return resp


class TestCallLlm:
    def setup_method(self):
        llm_mod._ollama_status = None
        llm_mod._resolved_ollama_url = None
        llm_mod._resolved_ollama_urls = None
        llm_mod._resolved_assignment_pool = None
        llm_mod._thread_backend.clear()
        llm_mod._next_backend_idx[0] = 0
        # _get_client memoises onto thread-local; reset between tests so the
        # mock client we install here actually replaces it.
        if hasattr(llm_mod._thread_local, "http_clients"):
            del llm_mod._thread_local.http_clients
        if hasattr(llm_mod._thread_local, "http_client"):
            del llm_mod._thread_local.http_client

    def test_success(self):
        cfg = _get_config()
        mock_resp = _make_ollama_resp(json.dumps(VALID_LLM_JSON))
        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp

        with patch("src.parser.llm_enrichment._get_client", return_value=mock_client):
            result = _call_llm("Vendo carro com 100km", cfg)

        assert result is not None
        assert result["sub_model"] == "320d"
        assert result["mileage_in_description_km"] == 180000
        # Confirm we hit /api/generate (NOT /api/chat) so the system prompt
        # stays byte-identical across calls and Ollama can reuse its KV-cache
        # slot for the instruction prefix. format=json keeps the output
        # parseable; the latency-tuned options below are also part of the
        # contract — regressing them silently would slow every batch.
        call_args = mock_client.post.call_args
        assert call_args.args[0] == "/api/generate"
        body = call_args.kwargs["json"]
        assert body["format"] == "json"
        assert body["system"] == llm_mod._SYSTEM_PROMPT
        assert body["keep_alive"] == "30m"
        opts = body["options"]
        assert opts["temperature"] == 0.0
        assert opts["top_k"] == 1
        assert opts["num_ctx"] == 2048
        assert opts["stop"] == ["}\n{", "} {"]

    def test_http_error_returns_none(self):
        cfg = _get_config()
        mock_resp = _make_ollama_resp("", status=500)
        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp
        with patch("src.parser.llm_enrichment._get_client", return_value=mock_client):
            result = _call_llm("Vendo carro", cfg)
        assert result is None

    def test_invalid_json_returns_none(self):
        cfg = _get_config()
        mock_resp = _make_ollama_resp("not json at all")
        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp
        with patch("src.parser.llm_enrichment._get_client", return_value=mock_client):
            result = _call_llm("Vendo carro", cfg)
        assert result is None

    def test_markdown_wrapped_json_recovers(self):
        # Some fine-tuned checkpoints occasionally wrap output in ```json … ```;
        # the strip pass should still recover the payload.
        cfg = _get_config()
        wrapped = "```json\n" + json.dumps(VALID_LLM_JSON) + "\n```"
        mock_resp = _make_ollama_resp(wrapped)
        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp
        with patch("src.parser.llm_enrichment._get_client", return_value=mock_client):
            result = _call_llm("Vendo carro", cfg)
        assert result == VALID_LLM_JSON

    def test_call_llm_delegates_to_ollama(self):
        # _call_llm is a thin alias; both should resolve to the same payload.
        cfg = _get_config()
        mock_resp = _make_ollama_resp(json.dumps(VALID_LLM_JSON))
        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp
        with patch("src.parser.llm_enrichment._get_client", return_value=mock_client):
            assert _call_llm("x", cfg) == _call_ollama("x", cfg)

    def test_resolve_picks_first_reachable_backend(self):
        # First URL fails, second succeeds → resolver returns the second.
        # This is the failover path used when localhost Ollama is down and
        # the LAN backend (Windows) takes over.
        ok = MagicMock()
        ok.status_code = 200

        def fake_get_client(url):
            client = MagicMock()
            if "192.168.1.69" in url:
                client.get.return_value = ok
            else:
                client.get.side_effect = httpx_mod.RequestError("boom")
            return client

        with patch("src.parser.llm_enrichment._get_config",
                   return_value={"ollama_urls": [
                       "http://localhost:11434",
                       "http://192.168.1.69:11434",
                   ]}), \
             patch("src.parser.llm_enrichment._get_client", side_effect=fake_get_client):
            picked = llm_mod._resolve_ollama_url()
        assert picked == "http://192.168.1.69:11434"

    def test_pick_distributes_across_healthy_backends(self):
        # With two healthy backends, parallel threads must NOT all land on
        # one URL — that defeats the load-balancing point and leaves the
        # second host idle. Sticky-per-thread keeps each thread on the same
        # backend (so KV-cache stays warm) but the overall distribution
        # across threads should hit both URLs.
        ok = MagicMock()
        ok.status_code = 200
        with patch("src.parser.llm_enrichment._get_config",
                   return_value={"ollama_urls": [
                       "http://192.168.1.77:11434",
                       "http://192.168.1.69:11434",
                   ]}), \
             patch("src.parser.llm_enrichment._get_client",
                   return_value=MagicMock(get=MagicMock(return_value=ok))):
            results = []
            barriers = threading.Barrier(8)

            def worker():
                barriers.wait()  # max parallelism
                results.append(llm_mod._pick_ollama_url())

            ts = [threading.Thread(target=worker) for _ in range(8)]
            for t in ts: t.start()
            for t in ts: t.join()

        unique = set(results)
        assert unique == {"http://192.168.1.77:11434", "http://192.168.1.69:11434"}, \
            f"expected both backends to receive traffic, got {unique}"

    def test_pick_sticky_per_thread(self):
        # Same thread must always pick the same backend (so prompt-cache
        # stays warm on its assigned host, instead of bouncing every call).
        ok = MagicMock()
        ok.status_code = 200
        with patch("src.parser.llm_enrichment._get_config",
                   return_value={"ollama_urls": [
                       "http://a:11434", "http://b:11434", "http://c:11434",
                   ]}), \
             patch("src.parser.llm_enrichment._get_client",
                   return_value=MagicMock(get=MagicMock(return_value=ok))):
            picks = [llm_mod._pick_ollama_url() for _ in range(20)]
        assert len(set(picks)) == 1, f"same thread bounced backends: {set(picks)}"


# ---------------------------------------------------------------------------
# enrich_from_description
# ---------------------------------------------------------------------------

class TestEnrichFromDescription:
    def test_empty_description_returns_none(self):
        assert enrich_from_description("") is None
        assert enrich_from_description("short") is None

    @patch("src.parser.llm_enrichment._llm_available", return_value=True)
    @patch("src.parser.llm_enrichment._call_llm", return_value=VALID_LLM_JSON)
    def test_calls_llm(self, mock_call, mock_avail):
        result = enrich_from_description("Vendo BMW 320d com 180.000km reais")
        assert result == VALID_LLM_JSON
        mock_call.assert_called_once()

    @patch("src.parser.llm_enrichment._llm_available", return_value=True)
    @patch("src.parser.llm_enrichment._call_llm", return_value=None)
    def test_returns_none_on_llm_failure(self, mock_call, mock_avail):
        result = enrich_from_description("Vendo BMW 320d com 180.000km reais")
        assert result is None

    @patch("src.parser.llm_enrichment._llm_available", return_value=False)
    def test_returns_none_when_ollama_unavailable(self, mock_avail):
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

    def test_sub_model_and_trim_passed_through(self):
        listing = FakeListing()
        listing._llm_extras = {"sub_model": "320d", "trim_level": "M Sport"}
        corrections = correct_listing_data(listing)
        assert corrections["sub_model"] == "320d"
        assert corrections["trim_level"] == "M Sport"

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
            {"mechanical_condition": "fair"}, "BMW", "Motor fundido, não anda.",
        ) == 2
        # And condition=poor on top → severity 3
        assert _derive_damage_severity(
            {"mechanical_condition": "poor"}, "BMW", "Motor fundido, não anda.",
        ) == 3

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


