"""Tests for the streaming pipeline: Scraper -> DB.

The pipeline is raw-only: nothing between the scraper and the database calls a
language model. Description NLP happens later, on the ranked top deals only —
see test_cloud_enrichment.py. What is pinned here is that the DB writer saves
what it is given, routes unmatched generations to the unmatched table, and
never blocks the scrape.
"""

from queue import Queue
from unittest.mock import patch, MagicMock

from src.parser.scraper import RawListing
from src.cli import _db_worker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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

class TestDbWorker:
    @patch("src.cli.get_generation", return_value="Mk7")
    @patch("src.cli.add_price_snapshot")
    @patch("src.cli.upsert_listing")
    @patch("src.cli.get_session")
    def test_saves_listing(self, mock_session_fn, mock_upsert,
                                            mock_snapshot, mock_gen):
        mock_session = MagicMock()
        mock_session_fn.return_value = mock_session
        mock_upsert.return_value = MagicMock(id=1)

        db_queue = Queue()
        result = {}

        raw = _make_raw()
        db_queue.put(raw)
        db_queue.put(None)

        _db_worker(db_queue, result)

        assert result["saved"] == 1
        assert result["unmatched"] == 0
        assert raw.olx_id in result["active_ids"]
        mock_upsert.assert_called_once()
        mock_snapshot.assert_called_once()
        mock_session.commit.assert_called()
        # The scrape writes no llm columns at all — the cloud step owns them,
        # and upsert_listing skips None, so a re-scrape can't blank them.
        call_data = mock_upsert.call_args[0][1]
        assert "llm_extras" not in call_data
        assert "llm_description_hash" not in call_data

    @patch("src.cli.get_generation", return_value=None)
    @patch("src.cli.upsert_unmatched")
    @patch("src.cli.get_session")
    def test_unmatched_listing(self, mock_session_fn, mock_unmatched, mock_gen):
        mock_session = MagicMock()
        mock_session_fn.return_value = mock_session

        db_queue = Queue()
        result = {}

        raw = _make_raw()
        db_queue.put(raw)
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
        db_queue.put(raw)
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
            db_queue.put(_make_raw(olx_id=f"multi-{i}"))
        db_queue.put(None)

        _db_worker(db_queue, result)

        assert result["saved"] == 5
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
        db_queue.put(raw)
        db_queue.put(None)

        _db_worker(db_queue, result)

        assert result["saved"] == 1
        mock_snapshot.assert_not_called()
