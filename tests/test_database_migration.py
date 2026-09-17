"""Migration smoke tests for ``src/storage/database.py``.

Each test gets its own PostgreSQL schema so ``init_db`` runs against a
clean namespace without a second database. We read ``information_schema``
directly rather than going through the ORM: the migration's contract is
"make these columns/tables exist on an existing database", and the ORM
would hide an ``ALTER TABLE`` that silently failed while ``create_all``
covered for it.
"""

from __future__ import annotations

import json

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


_V7_IMPORT_COLUMNS = (
    "region", "photo_count", "version", "price_label", "image_url",
    "offer_type", "body_type", "is_active", "deactivated_at",
)


def _index_exists(url: str, name: str) -> bool:
    engine = create_engine(url)
    try:
        with engine.connect() as conn:
            return bool(conn.execute(text(
                "SELECT COUNT(*) FROM pg_indexes "
                "WHERE schemaname = current_schema() AND indexname = :n"
            ), {"n": name}).scalar_one())
    finally:
        engine.dispose()


def _build_legacy_import_db(url: str) -> None:
    """A database from before the merge: a separate ``import_listings`` table
    with two German rows in it, one of them carrying the fields that have no
    column of their own in ``listings`` and must survive as JSON."""
    from src.models.listing import Base
    import src.models.portfolio  # noqa: F401
    import src.models.relist  # noqa: F401
    import src.models.seller  # noqa: F401

    engine = create_engine(url)
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE import_listings (
                id SERIAL PRIMARY KEY,
                source TEXT NOT NULL,
                external_id TEXT NOT NULL,
                url TEXT NOT NULL,
                brand TEXT NOT NULL,
                model TEXT NOT NULL,
                model_group TEXT, variant TEXT, motor_type TEXT, version TEXT,
                offer_type TEXT, body_type TEXT, is_damaged BOOLEAN,
                price_eur DOUBLE PRECISION, price_label TEXT,
                vat_label TEXT, vat_reclaimable BOOLEAN,
                year INTEGER, registration_month TEXT, mileage_km INTEGER,
                engine_cc INTEGER, horsepower INTEGER, power_kw INTEGER,
                fuel_type TEXT, transmission TEXT, co2_g_km INTEGER,
                seller_type TEXT, country_code TEXT, region TEXT, city TEXT,
                zip_code TEXT, photo_count INTEGER, image_url TEXT,
                is_active BOOLEAN DEFAULT TRUE, deactivated_at TIMESTAMP,
                first_seen_at TIMESTAMP, last_seen_at TIMESTAMP
            )"""))
        conn.execute(text(
            "INSERT INTO import_listings (source, external_id, url, brand, model, "
            "year, region, motor_type, variant, price_eur, country_code) VALUES "
            "('autoscout24', 'legacy-1', 'https://x', 'BMW', '320', 2016, "
            "'Bayern', '2.0d', 'Touring', 21000, 'DE')"))
        conn.execute(text(
            "INSERT INTO import_listings (source, external_id, url, brand, model, "
            "year, country_code) VALUES "
            "('as24_fr', 'legacy-2', 'https://y', 'Renault', 'Clio', 2019, 'FR')"))
    engine.dispose()


def test_fresh_db_has_the_market_columns(fresh_schema):
    """One table now holds every market, so the columns a foreign card fills
    have to be on ``listings`` itself."""
    from src.storage.database import init_db

    init_db(fresh_schema)

    cols = _table_columns(fresh_schema, "listings")
    for col in ("country_code", "external_id", "extras", "image_url",
                "price_label", "zip_code", "body_type", "power_kw",
                "vat_label", "vat_reclaimable", "is_damaged"):
        assert col in cols, f"listings.{col} missing on a fresh DB"
    assert _index_exists(fresh_schema, "ix_listings_country_code")


def test_the_old_import_table_is_folded_into_listings(fresh_schema):
    """The rows move, keyed by source so a German id cannot collide with an
    OLX one, and the fields with no column land in ``extras`` rather than
    being dropped."""
    _build_legacy_import_db(fresh_schema)
    reset_module_engine_cache()
    from src.storage.database import init_db

    init_db(fresh_schema)

    engine = create_engine(fresh_schema)
    with engine.connect() as conn:
        assert not conn.execute(text(
            "SELECT to_regclass('import_listings') IS NOT NULL")).scalar_one()
        row = conn.execute(text(
            "SELECT olx_id, source, external_id, country_code, district, extras "
            "FROM listings WHERE external_id = 'legacy-1'")).fetchone()
        codes = conn.execute(text(
            "SELECT country_code FROM listings ORDER BY country_code")).scalars().all()
    engine.dispose()

    assert row is not None, "the legacy row did not survive the merge"
    olx_id, source, external_id, country_code, district, extras = row
    assert olx_id == "autoscout24:legacy-1"
    assert (source, external_id, country_code) == ("autoscout24", "legacy-1", "DE")
    assert district == "Bayern"
    assert json.loads(extras) == {"motor_type": "2.0d", "variant": "Touring"}
    assert codes == ["DE", "FR"]


def test_the_merge_is_idempotent(fresh_schema):
    """Running it twice must not duplicate a car: the second pass finds the
    old table gone and the rows already carrying their namespaced ids."""
    _build_legacy_import_db(fresh_schema)
    reset_module_engine_cache()
    from src.storage.database import init_db

    init_db(fresh_schema)
    reset_module_engine_cache()
    init_db(fresh_schema)

    engine = create_engine(fresh_schema)
    with engine.connect() as conn:
        count = conn.execute(text(
            "SELECT COUNT(*) FROM listings WHERE olx_id LIKE '%:%'")).scalar_one()
    engine.dispose()
    assert count == 2
