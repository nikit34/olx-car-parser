"""Tests for ``scripts/backfill_sellers.py``.

The selector and link-back logic are pure SQL against a freshly-built
schema; the fetch path is exercised separately in test_seller_profile.
Here we wire ``_process_one`` against a real in-memory DB and a mock
scraper to cover the full upsert/link round-trip without going near a
real network or the global module-level engine cache in
``src.storage.database``.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from scripts.backfill_sellers import (
    _link_listings,
    _process_one,
    _select_targets,
    _upsert_seller,
)
from src.parser.seller_profile import SellerProfile


# ---------------------------------------------------------------------------
# Fresh schema fixture
# ---------------------------------------------------------------------------


def _reset_module_engine_cache():
    import src.storage.database as db_mod
    db_mod._engine = None
    db_mod._Session = None


@pytest.fixture
def db(tmp_path: Path) -> sqlite3.Connection:
    """A fresh DB at the on-disk path the script expects, with the v3
    schema applied (so ``sellers`` exists and ``listings`` has the
    seller_* columns). Tests open their own raw connection on top —
    matches the script's own access pattern."""
    _reset_module_engine_cache()
    db_file = tmp_path / "olx_cars.db"

    # Point both the script's DB_PATH constant and database.get_db_path()
    # at the temp file so init_db() builds schema there and the script's
    # _open_db() targets the same file.
    import scripts.backfill_sellers as backfill_mod
    backfill_mod.DB_PATH = db_file
    import src.storage.database as db_mod
    db_mod.get_db_path = lambda: str(db_file)

    from src.storage.database import init_db
    init_db(str(db_file))

    conn = sqlite3.connect(str(db_file), timeout=30, isolation_level=None)
    yield conn
    conn.close()


def _insert_listing(conn, olx_id: str, profile_url: str | None,
                    seller_uuid: str | None = None) -> None:
    conn.execute(
        "INSERT INTO listings (olx_id, url, brand, model, "
        "seller_profile_url, seller_uuid) VALUES (?, ?, ?, ?, ?, ?)",
        (olx_id, f"https://olx.pt/{olx_id}", "VW", "Golf",
         profile_url, seller_uuid),
    )


# ---------------------------------------------------------------------------
# _select_targets
# ---------------------------------------------------------------------------


class TestSelectTargets:
    def test_picks_up_listings_without_seller_uuid(self, db):
        _insert_listing(db, "L1", "https://www.olx.pt/ads/user/abc/")
        _insert_listing(db, "L2", "https://www.olx.pt/ads/user/def/")
        targets = _select_targets(db, ttl_days=14, limit=None)
        assert sorted(targets) == [
            "https://www.olx.pt/ads/user/abc/",
            "https://www.olx.pt/ads/user/def/",
        ]

    def test_dedups_url_across_multiple_listings(self, db):
        url = "https://www.olx.pt/ads/user/abc/"
        _insert_listing(db, "L1", url)
        _insert_listing(db, "L2", url)
        _insert_listing(db, "L3", url)
        targets = _select_targets(db, ttl_days=14, limit=None)
        assert targets == [url]

    def test_skips_listings_without_profile_url(self, db):
        _insert_listing(db, "L1", None)
        targets = _select_targets(db, ttl_days=14, limit=None)
        assert targets == []

    def test_skips_already_linked_listings_with_fresh_seller(self, db):
        # Listing already linked to a seller fetched 1 day ago — skip it
        # under a 14-day TTL.
        url = "https://www.olx.pt/ads/user/abc/"
        recent = (datetime.now(timezone.utc).replace(tzinfo=None)
                  - timedelta(days=1)).isoformat(sep=" ")
        db.execute(
            "INSERT INTO sellers (uuid, profile_url, profile_fetched_at) "
            "VALUES (?, ?, ?)",
            ("u-abc", url, recent),
        )
        _insert_listing(db, "L1", url, seller_uuid="u-abc")
        targets = _select_targets(db, ttl_days=14, limit=None)
        assert targets == []

    def test_includes_linked_listings_with_stale_seller(self, db):
        # Same as above but seller fetched 30 days ago — must refresh.
        url = "https://www.olx.pt/ads/user/abc/"
        stale = (datetime.now(timezone.utc).replace(tzinfo=None)
                 - timedelta(days=30)).isoformat(sep=" ")
        db.execute(
            "INSERT INTO sellers (uuid, profile_url, profile_fetched_at) "
            "VALUES (?, ?, ?)",
            ("u-abc", url, stale),
        )
        _insert_listing(db, "L1", url, seller_uuid="u-abc")
        targets = _select_targets(db, ttl_days=14, limit=None)
        assert targets == [url]


# ---------------------------------------------------------------------------
# _upsert_seller + _link_listings
# ---------------------------------------------------------------------------


def _profile(
    *,
    uuid: str = "u-1",
    is_business: bool = False,
    facets: dict[int, int] | None = None,
    categories_list: dict | None = None,
    social_account_type: str | None = "facebook",
    has_user_photo: bool = False,
    position_lat: float | None = 41.46,
    position_lon: float | None = -8.14,
) -> SellerProfile:
    return SellerProfile(
        uuid=uuid,
        profile_url="https://www.olx.pt/ads/user/abc/",
        short_id="abc",
        shop_slug=None,
        name="Rui",
        is_business=is_business,
        business_type=None,
        created_at=datetime(2019, 11, 14),
        last_seen_at=datetime(2026, 5, 4),
        last_login_at=datetime(2026, 5, 4),
        total_ads=8,
        facets=facets or {362: 7, 378: 5, 416: 1, 379: 1, 4918: 1},
        categories_list=categories_list or {},
        social_account_type=social_account_type,
        has_user_photo=has_user_photo,
        position_lat=position_lat,
        position_lon=position_lon,
    )


class TestUpsertAndLink:
    def test_inserts_seller_with_derived_counts(self, db):
        _upsert_seller(db, _profile())
        row = db.execute(
            "SELECT uuid, name, is_business, total_ads, "
            "cars_count, commercial_count, motos_count, non_auto_count, "
            "tools_industrial_count "
            "FROM sellers"
        ).fetchone()
        assert row[0] == "u-1"
        assert row[1] == "Rui"
        assert row[2] == 0  # is_business stored as 0/1 in SQLite
        assert row[3] == 8
        assert row[4] == 5  # cars
        assert row[5] == 1  # commercial
        assert row[6] == 1  # motos
        assert row[7] == 1  # 4918 → non_auto rollup
        assert row[8] == 1  # 4918 → tools_industrial sub-bucket too

    def test_inserts_seller_with_identity_fields(self, db):
        _upsert_seller(db, _profile(
            social_account_type="facebook",
            has_user_photo=False,
            position_lat=41.46008,
            position_lon=-8.144,
        ))
        row = db.execute(
            "SELECT social_account_type, has_user_photo, "
            "position_lat, position_lon FROM sellers"
        ).fetchone()
        assert row[0] == "facebook"
        assert row[1] == 0
        assert row[2] == pytest.approx(41.46008)
        assert row[3] == pytest.approx(-8.144)

    def test_upsert_updates_existing_row(self, db):
        _upsert_seller(db, _profile(facets={362: 1, 378: 1}))
        # Re-fetch with different counts (seller listed two more cars)
        _upsert_seller(db, _profile(facets={362: 3, 378: 3}))
        rows = db.execute("SELECT cars_count, total_ads FROM sellers").fetchall()
        assert len(rows) == 1
        assert rows[0] == (3, 8)  # total_ads always 8 in helper; cars updated

    def test_link_listings_sets_seller_uuid_for_matching_url(self, db):
        url = "https://www.olx.pt/ads/user/abc/"
        _insert_listing(db, "L1", url)
        _insert_listing(db, "L2", url)
        _insert_listing(db, "L3", "https://www.olx.pt/ads/user/other/")
        n = _link_listings(db, url, "u-1")
        assert n == 2
        rows = db.execute(
            "SELECT olx_id, seller_uuid FROM listings ORDER BY olx_id"
        ).fetchall()
        assert rows == [("L1", "u-1"), ("L2", "u-1"), ("L3", None)]

    def test_link_listings_does_not_clobber_existing_uuid(self, db):
        url = "https://www.olx.pt/ads/user/abc/"
        _insert_listing(db, "L1", url, seller_uuid="u-old")
        _insert_listing(db, "L2", url)
        n = _link_listings(db, url, "u-new")
        assert n == 1  # only L2 was unlinked
        rows = db.execute(
            "SELECT olx_id, seller_uuid FROM listings ORDER BY olx_id"
        ).fetchall()
        assert rows == [("L1", "u-old"), ("L2", "u-new")]


# ---------------------------------------------------------------------------
# _process_one — round trip against a mock scraper
# ---------------------------------------------------------------------------


class TestProcessOne:
    def test_happy_path_inserts_and_links(self, db):
        url = "https://www.olx.pt/ads/user/abc/"
        _insert_listing(db, "L1", url)
        _insert_listing(db, "L2", url)
        scraper = MagicMock()
        scraper.scrape_seller_profile.return_value = _profile()
        status, linked = _process_one(db, scraper, url, dry_run=False)
        assert status == "ok"
        assert linked == 2

    def test_dry_run_skips_writes(self, db):
        url = "https://www.olx.pt/ads/user/abc/"
        _insert_listing(db, "L1", url)
        scraper = MagicMock()
        scraper.scrape_seller_profile.return_value = _profile()
        status, linked = _process_one(db, scraper, url, dry_run=True)
        assert status == "ok"
        assert linked == 0
        assert db.execute("SELECT COUNT(*) FROM sellers").fetchone()[0] == 0
        assert db.execute(
            "SELECT seller_uuid FROM listings WHERE olx_id='L1'"
        ).fetchone()[0] is None

    def test_fetch_returning_none_reports_fetch_err(self, db):
        url = "https://www.olx.pt/ads/user/abc/"
        _insert_listing(db, "L1", url)
        scraper = MagicMock()
        scraper.scrape_seller_profile.return_value = None
        status, linked = _process_one(db, scraper, url, dry_run=False)
        assert status == "fetch_err"
        assert linked == 0

    def test_fetch_raising_reports_fetch_err(self, db):
        url = "https://www.olx.pt/ads/user/abc/"
        scraper = MagicMock()
        scraper.scrape_seller_profile.side_effect = RuntimeError("boom")
        status, linked = _process_one(db, scraper, url, dry_run=False)
        assert status == "fetch_err"
        assert linked == 0
