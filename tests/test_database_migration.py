"""Migration smoke tests for ``src/storage/database.py``.

Each test gets its own PostgreSQL schema so ``init_db`` runs against a
clean namespace without a second database. We read ``information_schema``
directly rather than going through the ORM: the migration's contract is
"make these columns/tables exist on an existing database", and the ORM
would hide an ``ALTER TABLE`` that silently failed while ``create_all``
covered for it.
"""

from __future__ import annotations

from sqlalchemy import create_engine, text

from tests.conftest import reset_module_engine_cache


def _table_columns(url: str, table: str) -> set[str]:
    engine = create_engine(url)
    try:
        with engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = :t AND table_schema = current_schema()"
            ), {"t": table}).fetchall()
        return {r[0] for r in rows}
    finally:
        engine.dispose()


def _table_exists(url: str, table: str) -> bool:
    engine = create_engine(url)
    try:
        with engine.connect() as conn:
            return conn.execute(text(
                "SELECT to_regclass(current_schema() || '.' || :t) IS NOT NULL"
            ), {"t": table}).scalar_one()
    finally:
        engine.dispose()


def _build_legacy_db(url: str) -> None:
    """A pre-v3 database: the current schema minus the v3 surface, so the
    column-level migration has something to do. Built with ``create_all``
    rather than handwritten DDL — the latter is a maintenance trap every
    time a column is added."""
    from src.models.listing import Base
    import src.models.portfolio  # noqa: F401
    import src.models.relist  # noqa: F401
    import src.models.seller  # noqa: F401

    engine = create_engine(url)
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS sellers CASCADE"))
        for col in ("seller_uuid", "seller_displayed_as"):
            conn.execute(text(f"ALTER TABLE listings DROP COLUMN IF EXISTS {col}"))
        conn.execute(text("DROP INDEX IF EXISTS ix_listings_seller_uuid"))
    engine.dispose()


def test_fresh_db_has_seller_table_and_columns(fresh_schema):
    from src.storage.database import init_db

    init_db(fresh_schema)

    assert _table_exists(fresh_schema, "sellers")
    assert _table_exists(fresh_schema, "listings")
    listing_cols = _table_columns(fresh_schema, "listings")
    assert "seller_uuid" in listing_cols
    assert "seller_displayed_as" in listing_cols
    seller_cols = _table_columns(fresh_schema, "sellers")
    for col in ["uuid", "profile_url", "is_business", "total_ads",
                "cars_count", "distinct_car_brands", "profile_fetched_at"]:
        assert col in seller_cols, f"sellers.{col} missing on a fresh DB"


def test_existing_db_gets_seller_columns(fresh_schema):
    _build_legacy_db(fresh_schema)
    reset_module_engine_cache()
    from src.storage.database import init_db

    init_db(fresh_schema)

    assert _table_exists(fresh_schema, "sellers")
    listing_cols = _table_columns(fresh_schema, "listings")
    assert "seller_uuid" in listing_cols
    assert "seller_displayed_as" in listing_cols


def test_migration_is_idempotent(fresh_schema):
    from src.storage.database import init_db

    init_db(fresh_schema)
    reset_module_engine_cache()
    init_db(fresh_schema)

    assert "seller_uuid" in _table_columns(fresh_schema, "listings")


def test_partial_v3_db_gets_new_seller_columns(fresh_schema):
    """A dev database that ran an early v3 migration may already have
    ``sellers`` but lack the bucketing/identity columns added later.
    Re-running init_db must ALTER those in idempotently."""
    _build_legacy_db(fresh_schema)
    engine = create_engine(fresh_schema)
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE sellers (
                uuid TEXT PRIMARY KEY,
                short_id TEXT,
                shop_slug TEXT,
                profile_url TEXT NOT NULL,
                name TEXT,
                is_business BOOLEAN,
                business_type TEXT,
                created_at TIMESTAMP,
                last_seen_at TIMESTAMP,
                last_login_at TIMESTAMP,
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
                profile_fetched_at TIMESTAMP
            )
        """))
    engine.dispose()
    reset_module_engine_cache()

    from src.storage.database import init_db
    init_db(fresh_schema)

    cols = _table_columns(fresh_schema, "sellers")
    for col in ["family_lifestyle_count", "electronics_count",
                "realestate_count", "tools_industrial_count",
                "pets_hobby_count", "services_jobs_count",
                "social_account_type", "has_user_photo",
                "position_lat", "position_lon"]:
        assert col in cols, f"migration didn't add seller.{col}"


def test_listing_row_can_reference_seller(fresh_schema):
    """End-to-end: insert a Seller and a Listing pointing at it via FK."""
    from src.storage.database import init_db

    init_db(fresh_schema)

    engine = create_engine(fresh_schema)
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO sellers (uuid, profile_url, name, is_business, total_ads)
            VALUES (:uuid, :url, :name, FALSE, 3)
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
    engine.dispose()
    assert tuple(row) == ("L1", "Rui", "Utilizador")
