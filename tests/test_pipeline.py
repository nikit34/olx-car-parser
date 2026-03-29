"""Tests for the streaming pipeline: Scraper -> LLM -> DB."""

import json
import multiprocessing
import threading
from queue import Queue, Empty
from unittest.mock import patch, MagicMock

import pytest

from src.parser.scraper import RawListing
from src.cli import _llm_worker, _llm_to_db, _db_worker, _desc_hash


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_LLM_RESULT = {
    "num_owners": 2,
    "accident_free": True,
    "had_accident": False,
    "needs_repair": False,
    "mileage_in_description_km": 120000,
    "estimated_repair_cost_eur": None,
    "customs_cleared": None,
}


def _make_raw(olx_id="test-001", brand="Volkswagen", model="Golf",
              year=2015, price_eur=10000, description="Vendo carro em bom estado com 120000km reais"):
    return RawListing(
        olx_id=olx_id,
        url=f"https://olx.pt/{olx_id}",
        title=f"{brand} {model}",
        brand=brand,
        model=model,
        year=year,
        price_eur=price_eur,
        city="Porto",
        district="Porto",
        description=description,
    )


# ---------------------------------------------------------------------------
# _llm_to_db (merger thread)
# ---------------------------------------------------------------------------

class TestLlmToDb:
    def test_forwards_results_to_db_queue(self):
        """Merger pairs LLM results with raw listings and sends to db_queue."""
        llm_out = multiprocessing.Queue()
        db_queue = Queue()
        raw_by_id = {}

        raws = [_make_raw(olx_id=f"m-{i}") for i in range(3)]
        for r in raws:
            raw_by_id[r.olx_id] = r

        merger = threading.Thread(target=_llm_to_db, args=(llm_out, raw_by_id, db_queue))
        merger.start()

        for r in raws:
            llm_out.put((r.olx_id, VALID_LLM_RESULT))
        llm_out.put(None)  # poison pill
        merger.join(timeout=5)

        items = []
        while not db_queue.empty():
            items.append(db_queue.get_nowait())
        assert len(items) == 3
        for raw, llm_data in items:
            assert llm_data == VALID_LLM_RESULT
            assert raw.brand == "Volkswagen"

    def test_forwards_none_results(self):
        """Merger forwards LLM failures (None result) to db_queue."""
        llm_out = multiprocessing.Queue()
        db_queue = Queue()
        raw = _make_raw(olx_id="fail-1")
        raw_by_id = {raw.olx_id: raw}

        merger = threading.Thread(target=_llm_to_db, args=(llm_out, raw_by_id, db_queue))
        merger.start()

        llm_out.put((raw.olx_id, None))
        llm_out.put(None)
        merger.join(timeout=5)

        item = db_queue.get_nowait()
        assert item == (raw, None)

    def test_ignores_unknown_olx_id(self):
        """Merger skips results for olx_ids not in raw_by_id."""
        llm_out = multiprocessing.Queue()
        db_queue = Queue()

        merger = threading.Thread(target=_llm_to_db, args=(llm_out, {}, db_queue))
        merger.start()

        llm_out.put(("unknown-id", VALID_LLM_RESULT))
        llm_out.put(None)
        merger.join(timeout=5)

        assert db_queue.empty()

    def test_stops_on_poison_pill(self):
        """Merger exits cleanly on poison pill."""
        llm_out = multiprocessing.Queue()
        db_queue = Queue()

        merger = threading.Thread(target=_llm_to_db, args=(llm_out, {}, db_queue))
        merger.start()

        llm_out.put(None)
        merger.join(timeout=5)
        assert not merger.is_alive()


# ---------------------------------------------------------------------------
# _db_worker
# ---------------------------------------------------------------------------

class TestDbWorker:
    @patch("src.cli.get_generation", return_value="Mk7")
    @patch("src.cli.add_price_snapshot")
    @patch("src.cli.upsert_listing")
    @patch("src.cli.get_session")
    def test_saves_listing_with_llm_data(self, mock_session_fn, mock_upsert,
                                         mock_snapshot, mock_gen):
        mock_session = MagicMock()
        mock_session_fn.return_value = mock_session
        mock_upsert.return_value = MagicMock(id=1)

        db_queue = Queue()
        result = {}

        raw = _make_raw()
        db_queue.put((raw, VALID_LLM_RESULT))
        db_queue.put(None)

        _db_worker(db_queue, result)

        assert result["saved"] == 1
        assert result["enriched"] == 1
        assert result["unmatched"] == 0
        assert raw.olx_id in result["active_ids"]
        mock_upsert.assert_called_once()
        mock_snapshot.assert_called_once()
        mock_session.commit.assert_called()

        # Verify llm_extras in data dict
        call_data = mock_upsert.call_args[0][1]
        assert call_data["llm_extras"] is not None
        assert json.loads(call_data["llm_extras"])["num_owners"] == 2

    @patch("src.cli.get_generation", return_value="Mk7")
    @patch("src.cli.add_price_snapshot")
    @patch("src.cli.upsert_listing")
    @patch("src.cli.get_session")
    def test_saves_listing_without_llm_data(self, mock_session_fn, mock_upsert,
                                            mock_snapshot, mock_gen):
        mock_session = MagicMock()
        mock_session_fn.return_value = mock_session
        mock_upsert.return_value = MagicMock(id=1)

        db_queue = Queue()
        result = {}

        raw = _make_raw()
        db_queue.put((raw, None))
        db_queue.put(None)

        _db_worker(db_queue, result)

        assert result["saved"] == 1
        assert result["enriched"] == 0
        call_data = mock_upsert.call_args[0][1]
        assert call_data["llm_extras"] is None

    @patch("src.cli.get_generation", return_value=None)
    @patch("src.cli.upsert_unmatched")
    @patch("src.cli.get_session")
    def test_unmatched_listing(self, mock_session_fn, mock_unmatched, mock_gen):
        mock_session = MagicMock()
        mock_session_fn.return_value = mock_session

        db_queue = Queue()
        result = {}

        raw = _make_raw()
        db_queue.put((raw, None))
        db_queue.put(None)

        _db_worker(db_queue, result)

        assert result["saved"] == 0
        assert result["unmatched"] == 1
        mock_unmatched.assert_called_once()

    @patch("src.cli.get_generation", return_value="Mk7")
    @patch("src.cli.add_price_snapshot")
    @patch("src.cli.upsert_listing")
    @patch("src.cli.get_session")
    def test_skips_listing_without_brand_and_title(self, mock_session_fn, mock_upsert,
                                                   mock_snapshot, mock_gen):
        mock_session = MagicMock()
        mock_session_fn.return_value = mock_session

        db_queue = Queue()
        result = {}

        raw = _make_raw(brand="", model="")
        raw.title = ""
        db_queue.put((raw, None))
        db_queue.put(None)

        _db_worker(db_queue, result)

        assert result["saved"] == 0
        assert result["unmatched"] == 0
        mock_upsert.assert_not_called()

    @patch("src.cli.get_generation", return_value="Mk7")
    @patch("src.cli.add_price_snapshot")
    @patch("src.cli.upsert_listing")
    @patch("src.cli.get_session")
    def test_processes_multiple_listings(self, mock_session_fn, mock_upsert,
                                        mock_snapshot, mock_gen):
        mock_session = MagicMock()
        mock_session_fn.return_value = mock_session
        mock_upsert.return_value = MagicMock(id=1)

        db_queue = Queue()
        result = {}

        for i in range(5):
            raw = _make_raw(olx_id=f"multi-{i}")
            llm = VALID_LLM_RESULT if i % 2 == 0 else None
            db_queue.put((raw, llm))
        db_queue.put(None)

        _db_worker(db_queue, result)

        assert result["saved"] == 5
        assert result["enriched"] == 3  # indices 0, 2, 4
        assert len(result["active_ids"]) == 5

    @patch("src.cli.get_generation", return_value="Mk7")
    @patch("src.cli.add_price_snapshot")
    @patch("src.cli.upsert_listing")
    @patch("src.cli.get_session")
    def test_no_snapshot_when_no_price(self, mock_session_fn, mock_upsert,
                                      mock_snapshot, mock_gen):
        mock_session = MagicMock()
        mock_session_fn.return_value = mock_session
        mock_upsert.return_value = MagicMock(id=1)

        db_queue = Queue()
        result = {}

        raw = _make_raw(price_eur=None)
        db_queue.put((raw, None))
        db_queue.put(None)

        _db_worker(db_queue, result)

        assert result["saved"] == 1
        mock_snapshot.assert_not_called()


# ---------------------------------------------------------------------------
# _llm_worker (multiprocessing-based)
# ---------------------------------------------------------------------------

class TestLlmWorker:
    @patch("src.parser.llm_enrichment._ollama_available", return_value=True)
    @patch("src.parser.llm_enrichment.enrich_from_description", return_value=VALID_LLM_RESULT)
    def test_processes_items_and_sends_results(self, mock_enrich, mock_avail):
        in_q = multiprocessing.Queue()
        out_q = multiprocessing.Queue()
        shutdown = multiprocessing.Event()

        for i in range(3):
            in_q.put((f"item-{i}", f"Vendo carro em bom estado numero {i} com muitos km"))

        in_q.put(None)  # poison pill

        # Run in thread to avoid subprocess pickling issues with mocks
        t = threading.Thread(target=_llm_worker, args=(in_q, out_q, shutdown))
        t.start()
        t.join(timeout=10)

        results = []
        while not out_q.empty():
            results.append(out_q.get_nowait())
        assert len(results) == 3
        for olx_id, result in results:
            assert result == VALID_LLM_RESULT

    @patch("src.parser.llm_enrichment._ollama_available", return_value=True)
    @patch("src.parser.llm_enrichment.enrich_from_description", return_value=None)
    def test_sends_none_on_failure(self, mock_enrich, mock_avail):
        in_q = multiprocessing.Queue()
        out_q = multiprocessing.Queue()
        shutdown = multiprocessing.Event()

        in_q.put(("fail-1", "Vendo carro com problemas e muitos km percorridos"))
        in_q.put(None)

        t = threading.Thread(target=_llm_worker, args=(in_q, out_q, shutdown))
        t.start()
        t.join(timeout=10)

        olx_id, result = out_q.get_nowait()
        assert olx_id == "fail-1"
        assert result is None

    @patch("src.parser.llm_enrichment._ollama_available", return_value=True)
    @patch("src.parser.llm_enrichment.enrich_from_description", return_value=None)
    def test_exits_after_5_consecutive_failures(self, mock_enrich, mock_avail):
        in_q = multiprocessing.Queue()
        out_q = multiprocessing.Queue()
        shutdown = multiprocessing.Event()

        for i in range(10):
            in_q.put((f"fail-{i}", f"Vendo carro numero {i} com muitos quilometros reais"))

        t = threading.Thread(target=_llm_worker, args=(in_q, out_q, shutdown))
        t.start()
        t.join(timeout=10)

        results = []
        while not out_q.empty():
            results.append(out_q.get_nowait())
        # Worker exits after 5 consecutive failures, processes exactly 5
        assert len(results) == 5

    @patch("src.parser.llm_enrichment._ollama_available", return_value=True)
    @patch("src.parser.llm_enrichment.enrich_from_description", return_value=VALID_LLM_RESULT)
    def test_short_description_sends_none(self, mock_enrich, mock_avail):
        in_q = multiprocessing.Queue()
        out_q = multiprocessing.Queue()
        shutdown = multiprocessing.Event()

        in_q.put(("short-1", "too short"))
        in_q.put(None)

        t = threading.Thread(target=_llm_worker, args=(in_q, out_q, shutdown))
        t.start()
        t.join(timeout=10)

        olx_id, result = out_q.get_nowait()
        assert olx_id == "short-1"
        assert result is None
        mock_enrich.assert_not_called()

    @patch("src.parser.llm_enrichment._ollama_available", return_value=True)
    @patch("src.parser.llm_enrichment.enrich_from_description", return_value=VALID_LLM_RESULT)
    def test_shutdown_event_exits_worker(self, mock_enrich, mock_avail):
        in_q = multiprocessing.Queue()
        out_q = multiprocessing.Queue()
        shutdown = multiprocessing.Event()

        shutdown.set()
        # Queue is empty + shutdown set → worker exits after timeout

        t = threading.Thread(target=_llm_worker, args=(in_q, out_q, shutdown))
        t.start()
        # Worker has 60s timeout on get, override for test by putting poison pill
        in_q.put(None)
        t.join(timeout=5)
        assert not t.is_alive()

    @patch("src.parser.llm_enrichment._ollama_available", return_value=False)
    def test_exits_when_ollama_unavailable(self, mock_avail):
        in_q = multiprocessing.Queue()
        out_q = multiprocessing.Queue()
        shutdown = multiprocessing.Event()

        t = threading.Thread(target=_llm_worker, args=(in_q, out_q, shutdown))
        t.start()
        t.join(timeout=5)

        assert not t.is_alive()
        assert out_q.empty()


# ---------------------------------------------------------------------------
# Shutdown & drain sequence
# ---------------------------------------------------------------------------

class TestShutdownDrain:
    def test_drain_rescues_unprocessed_items(self):
        """Items left in llm_in after workers exit get sent to db_queue.

        Uses queue.Queue here because multiprocessing.Queue.put() is async
        and items may not be immediately available for get_nowait().
        The drain logic is identical for both queue types (both raise Empty).
        """
        from queue import Queue as StdQueue
        llm_in = StdQueue()
        db_queue = Queue()
        raw_by_id = {}

        # Simulate: 3 items left unprocessed in llm_in
        for i in range(3):
            raw = _make_raw(olx_id=f"drain-{i}")
            raw_by_id[raw.olx_id] = raw
            llm_in.put((raw.olx_id, raw.description))

        # Also a leftover poison pill (worker exited early without consuming it)
        llm_in.put(None)

        # Drain logic (same as in scrape())
        drained = 0
        while True:
            try:
                item = llm_in.get_nowait()
            except Empty:
                break
            if item is not None:
                olx_id, _ = item
                raw = raw_by_id.get(olx_id)
                if raw:
                    db_queue.put((raw, None))
                    drained += 1

        assert drained == 3
        assert db_queue.qsize() == 3

    def test_drain_handles_empty_queue(self):
        """Drain on empty queue does nothing."""
        from queue import Queue as StdQueue
        llm_in = StdQueue()
        db_queue = Queue()

        drained = 0
        while True:
            try:
                item = llm_in.get_nowait()
            except Empty:
                break
            if item is not None:
                drained += 1

        assert drained == 0
        assert db_queue.empty()


# ---------------------------------------------------------------------------
# Full pipeline integration (Scraper -> LLM -> Merger -> DB)
# ---------------------------------------------------------------------------

class TestFullPipeline:
    @patch("src.parser.llm_enrichment._ollama_available", return_value=True)
    @patch("src.parser.llm_enrichment.enrich_from_description", return_value=VALID_LLM_RESULT)
    @patch("src.cli.get_generation", return_value="Mk7")
    @patch("src.cli.add_price_snapshot")
    @patch("src.cli.upsert_listing")
    @patch("src.cli.get_session")
    def test_end_to_end_pipeline(self, mock_session_fn, mock_upsert,
                                 mock_snapshot, mock_gen, mock_enrich, mock_avail):
        """Full pipeline: items flow from llm_in through LLM worker -> merger -> DB worker."""
        mock_session = MagicMock()
        mock_session_fn.return_value = mock_session
        mock_upsert.return_value = MagicMock(id=1)

        llm_in = multiprocessing.Queue()
        llm_out = multiprocessing.Queue()
        db_queue = Queue()
        shutdown = multiprocessing.Event()
        raw_by_id = {}
        db_result = {}

        # Start all workers
        llm_thread = threading.Thread(target=_llm_worker, args=(llm_in, llm_out, shutdown))
        merger_thread = threading.Thread(target=_llm_to_db, args=(llm_out, raw_by_id, db_queue))
        db_thread = threading.Thread(target=_db_worker, args=(db_queue, db_result))

        llm_thread.start()
        merger_thread.start()
        db_thread.start()

        # Feed listings (simulating _on_batch)
        listings = [_make_raw(olx_id=f"e2e-{i}") for i in range(5)]
        for raw in listings:
            raw_by_id[raw.olx_id] = raw
            llm_in.put((raw.olx_id, raw.description))

        # Shutdown sequence (same order as scrape())
        llm_in.put(None)  # poison pill for LLM worker
        shutdown.set()
        llm_thread.join(timeout=10)

        llm_out.put(None)  # poison pill for merger
        merger_thread.join(timeout=10)

        db_queue.put(None)  # poison pill for DB worker
        db_thread.join(timeout=10)

        assert db_result["saved"] == 5
        assert db_result["enriched"] == 5
        assert len(db_result["active_ids"]) == 5
        assert mock_upsert.call_count == 5
        assert mock_snapshot.call_count == 5

    @patch("src.parser.llm_enrichment._ollama_available", return_value=True)
    @patch("src.parser.llm_enrichment.enrich_from_description", return_value=VALID_LLM_RESULT)
    @patch("src.cli.get_generation", return_value="Mk7")
    @patch("src.cli.add_price_snapshot")
    @patch("src.cli.upsert_listing")
    @patch("src.cli.get_session")
    def test_mixed_llm_and_direct_saves(self, mock_session_fn, mock_upsert,
                                        mock_snapshot, mock_gen, mock_enrich, mock_avail):
        """Some listings go through LLM, others go directly to DB."""
        mock_session = MagicMock()
        mock_session_fn.return_value = mock_session
        mock_upsert.return_value = MagicMock(id=1)

        llm_in = multiprocessing.Queue()
        llm_out = multiprocessing.Queue()
        db_queue = Queue()
        shutdown = multiprocessing.Event()
        raw_by_id = {}
        db_result = {}

        llm_thread = threading.Thread(target=_llm_worker, args=(llm_in, llm_out, shutdown))
        merger_thread = threading.Thread(target=_llm_to_db, args=(llm_out, raw_by_id, db_queue))
        db_thread = threading.Thread(target=_db_worker, args=(db_queue, db_result))

        llm_thread.start()
        merger_thread.start()
        db_thread.start()

        # 3 listings sent to LLM
        for i in range(3):
            raw = _make_raw(olx_id=f"llm-{i}")
            raw_by_id[raw.olx_id] = raw
            llm_in.put((raw.olx_id, raw.description))

        # 2 listings sent directly to DB (skip LLM)
        for i in range(2):
            raw = _make_raw(olx_id=f"direct-{i}", description="short")
            db_queue.put((raw, None))

        # Shutdown
        llm_in.put(None)
        shutdown.set()
        llm_thread.join(timeout=10)

        llm_out.put(None)
        merger_thread.join(timeout=10)

        db_queue.put(None)
        db_thread.join(timeout=10)

        assert db_result["saved"] == 5
        assert db_result["enriched"] == 3  # only LLM ones

    @patch("src.parser.llm_enrichment._ollama_available", return_value=True)
    @patch("src.parser.llm_enrichment.enrich_from_description", return_value=None)
    @patch("src.cli.get_generation", return_value="Mk7")
    @patch("src.cli.add_price_snapshot")
    @patch("src.cli.upsert_listing")
    @patch("src.cli.get_session")
    def test_pipeline_with_llm_failures_and_drain(self, mock_session_fn, mock_upsert,
                                                  mock_snapshot, mock_gen,
                                                  mock_enrich, mock_avail):
        """LLM worker exits after 5 failures, remaining items are drained."""
        mock_session = MagicMock()
        mock_session_fn.return_value = mock_session
        mock_upsert.return_value = MagicMock(id=1)

        llm_in = multiprocessing.Queue()
        llm_out = multiprocessing.Queue()
        db_queue = Queue()
        shutdown = multiprocessing.Event()
        raw_by_id = {}
        db_result = {}

        llm_thread = threading.Thread(target=_llm_worker, args=(llm_in, llm_out, shutdown))
        merger_thread = threading.Thread(target=_llm_to_db, args=(llm_out, raw_by_id, db_queue))
        db_thread = threading.Thread(target=_db_worker, args=(db_queue, db_result))

        llm_thread.start()
        merger_thread.start()
        db_thread.start()

        # Feed 8 listings — LLM will fail on all, exit after 5
        for i in range(8):
            raw = _make_raw(olx_id=f"drain-{i}")
            raw_by_id[raw.olx_id] = raw
            llm_in.put((raw.olx_id, raw.description))

        # Wait for LLM worker to exit (5 failures)
        llm_in.put(None)
        shutdown.set()
        llm_thread.join(timeout=10)

        # Drain leftover items from llm_in
        drained = 0
        while True:
            try:
                item = llm_in.get_nowait()
            except Empty:
                break
            if item is not None:
                olx_id, _ = item
                raw = raw_by_id.get(olx_id)
                if raw:
                    db_queue.put((raw, None))
                    drained += 1

        # Stop merger and DB
        llm_out.put(None)
        merger_thread.join(timeout=10)
        db_queue.put(None)
        db_thread.join(timeout=10)

        # 5 processed by LLM (all with None result) + drained items
        total = db_result["saved"]
        assert total == 5 + drained
        assert db_result["enriched"] == 0  # all LLM calls failed
