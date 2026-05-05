"""Migration smoke tests for ``src/storage/database.py``.

Each test starts with a freshly-named SQLite file so the module-level
``_engine`` cache in ``database.py`` doesn't leak engine state across
tests. We poke through PRAGMA / sqlite_master directly because the
migration's contract is "make these columns/tables/indexes exist on an
existing DB" — going through the ORM would hide the case where
``ALTER TABLE`` silently failed and ``create_all`` covered for it on a
fresh file.
"""

from __future__ import annotations

import sqlite3

from sqlalchemy import create_engine, text


def _reset_module_engine_cache():
    """``database.py`` holds a global ``_engine``; clear it so each test
    can target its own DB file without inheriting the previous one."""
    import src.storage.database as db_mod

    db_mod._engine = None
    db_mod._Session = None


def _table_columns(path: str, table: str) -> set[str]:
    conn = sqlite3.connect(path)
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        return {row[1] for row in rows}
    finally:
        conn.close()


def _table_exists(path: str, table: str) -> bool:
    conn = sqlite3.connect(path)
    try:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
        return row is not None
    finally:
        conn.close()


def _index_exists(path: str, index: str) -> bool:
    conn = sqlite3.connect(path)
    try:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name=?",
            (index,),
        ).fetchone()
        return row is not None
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Fresh DB — create_all path
# ---------------------------------------------------------------------------


def test_fresh_db_has_seller_table_and_columns(tmp_path):
    _reset_module_engine_cache()
    from src.storage.database import init_db

    db_path = str(tmp_path / "fresh.db")
    init_db(db_path)

    assert _table_exists(db_path, "sellers")
    assert _table_exists(db_path, "listings")
    listing_cols = _table_columns(db_path, "listings")
    assert "seller_uuid" in listing_cols
    assert "seller_displayed_as" in listing_cols
    seller_cols = _table_columns(db_path, "sellers")
    # Verify the schema-level fields the scraper will write — ORM ↔ DB
    # drift here would surface as mysterious upsert failures at runtime.
    for col in ["uuid", "short_id", "shop_slug", "name", "is_business",
                "created_at", "total_ads", "ads_by_category",
                "cars_count", "parts_count", "distinct_car_brands",
                # Non-auto sub-buckets
                "family_lifestyle_count", "electronics_count",
                "realestate_count", "tools_industrial_count",
                "pets_hobby_count", "services_jobs_count",
                # Identity / trust
                "social_account_type", "has_user_photo",
                "position_lat", "position_lon",
                "profile_fetched_at"]:
        assert col in seller_cols, f"missing seller.{col}"


# ---------------------------------------------------------------------------
# Existing DB — ALTER TABLE path
# ---------------------------------------------------------------------------


def _build_legacy_db(path: str) -> None:
    """Create a pre-v3 DB by running ``create_all`` (which produces the
    current schema), then dropping the v3-introduced surface so the
    column-level migration has something to do. The migration also runs
    a SELECT over ``llm_extras`` unconditionally, so we need the rest of
    the schema intact — building it ad-hoc with handwritten SQL would be
    a maintenance trap when new columns get added."""
    from sqlalchemy import create_engine, text as sa_text
    from src.models.listing import Base
    import src.models.portfolio  # noqa: F401
    import src.models.relist  # noqa: F401
    import src.models.seller  # noqa: F401

    engine = create_engine(f"sqlite:///{path}")
    Base.metadata.create_all(engine)

    with engine.begin() as conn:
        conn.execute(sa_text("DROP TABLE IF EXISTS sellers"))
        for col in ("seller_uuid", "seller_displayed_as"):
            try:
                conn.execute(sa_text(f"ALTER TABLE listings DROP COLUMN {col}"))
            except Exception:
                pass
        try:
            conn.execute(sa_text("DROP INDEX IF EXISTS ix_listings_seller_uuid"))
        except Exception:
            pass
        # _schema_meta is created lazily by init_db's _read_schema_version;
        # we don't need to seed it. Absence == version 0 == run migrations.
    engine.dispose()


def test_existing_db_gets_seller_columns(tmp_path):
    _reset_module_engine_cache()
    db_path = str(tmp_path / "legacy.db")
    _build_legacy_db(db_path)

    from src.storage.database import init_db
    init_db(db_path)

    cols = _table_columns(db_path, "listings")
    assert "seller_uuid" in cols
    assert "seller_displayed_as" in cols
    assert _table_exists(db_path, "sellers")
    assert _index_exists(db_path, "ix_listings_seller_uuid")


def test_migration_is_idempotent(tmp_path):
    _reset_module_engine_cache()
    db_path = str(tmp_path / "twice.db")
    _build_legacy_db(db_path)

    from src.storage.database import init_db
    init_db(db_path)
    # Second invocation is the production-realistic case (every CLI run
    # re-enters init_db); it must short-circuit on the schema-version
    # gate without raising on the already-existing column / index.
    _reset_module_engine_cache()
    init_db(db_path)

    cols = _table_columns(db_path, "listings")
    assert "seller_uuid" in cols
    assert _index_exists(db_path, "ix_listings_seller_uuid")


def test_partial_v3_db_gets_new_seller_columns(tmp_path):
    """A dev DB that ran an early v3 migration may already have
    ``sellers`` but be missing the bucketing/identity columns we added
    later. Re-running init_db must ALTER those in idempotently."""
    _reset_module_engine_cache()
    db_path = str(tmp_path / "partial_v3.db")
    _build_legacy_db(db_path)

    # Build the v3-shape sellers table by hand, missing the new columns
    # (family_lifestyle_count etc.). Mimics a dev DB that ran the
    # migration before this PR landed.
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("""
            CREATE TABLE sellers (
                uuid TEXT PRIMARY KEY,
                short_id TEXT,
                shop_slug TEXT,
                profile_url TEXT NOT NULL,
                name TEXT,
                is_business BOOLEAN,
                business_type TEXT,
                created_at DATETIME,
                last_seen_at DATETIME,
                last_login_at DATETIME,
                total_ads INTEGER,
                ads_by_category TEXT,
                cars_count INTEGER,
                parts_count INTEGER,
                commercial_count INTEGER,
                motos_count INTEGER,
                boats_count INTEGER,
                other_auto_count INTEGER,
                non_auto_count INTEGER,
                distinct_car_brands INTEGER,
                profile_fetched_at DATETIME
            )
        """)
        conn.commit()
    finally:
        conn.close()

    from src.storage.database import init_db
    init_db(db_path)

    cols = _table_columns(db_path, "sellers")
    for col in ["family_lifestyle_count", "electronics_count",
                "realestate_count", "tools_industrial_count",
                "pets_hobby_count", "services_jobs_count",
                "social_account_type", "has_user_photo",
                "position_lat", "position_lon"]:
        assert col in cols, f"migration didn't add seller.{col}"


def test_listing_row_can_reference_seller(tmp_path):
    """End-to-end: insert a Seller and a Listing pointing at it via FK."""
    _reset_module_engine_cache()
    db_path = str(tmp_path / "fk.db")

    from src.storage.database import init_db
    init_db(db_path)

    engine = create_engine(f"sqlite:///{db_path}")
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO sellers (uuid, profile_url, name, is_business, total_ads)
            VALUES (:uuid, :url, :name, 0, 3)
        """), {"uuid": "u-1", "url": "https://www.olx.pt/ads/user/abc/",
               "name": "Rui"})
        conn.execute(text("""
            INSERT INTO listings (olx_id, url, brand, model, seller_uuid,
                                  seller_displayed_as)
            VALUES (:o, :u, :b, :m, :s, :d)
        """), {"o": "L1", "u": "https://x", "b": "VW", "m": "Golf",
               "s": "u-1", "d": "Utilizador"})
    with engine.connect() as conn:
        row = conn.execute(text(
            "SELECT l.olx_id, s.name, l.seller_displayed_as "
            "FROM listings l JOIN sellers s ON s.uuid = l.seller_uuid"
        )).fetchone()
        assert row == ("L1", "Rui", "Utilizador")
