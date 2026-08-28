"""Database connection and initialization."""

import json
import os
from pathlib import Path

from sqlalchemy import create_engine, event, inspect
from sqlalchemy.orm import sessionmaker

from sqlalchemy import text

from src.models.listing import Base
import src.models.portfolio  # noqa: F401 — register PortfolioDeal with Base
import src.models.relist  # noqa: F401 — register RelistEvent with Base
import src.models.seller  # noqa: F401 — register Seller with Base

_engine = None
_Session = None


def get_db_path() -> str:
    project_root = Path(__file__).resolve().parent.parent.parent
    return str(project_root / "data" / "olx_cars.db")


def resolve_db_url(db_path: str | None = None) -> str:
    """Pick the engine URL for this process.

    ``OLX_DB_URL`` (e.g. ``postgresql+psycopg://olx@localhost/olx_cars``) is
    the production setting. A caller that passes an explicit path still wins
    when that path is not the legacy default — that keeps ``--db /tmp/copy.db``
    honest instead of silently reading production.
    """
    if db_path and "://" in db_path:
        return db_path
    if db_path and os.path.abspath(db_path) != os.path.abspath(get_db_path()):
        return f"sqlite:///{db_path}"
    env_url = os.environ.get("OLX_DB_URL", "").strip()
    if env_url:
        return env_url
    return f"sqlite:///{db_path or get_db_path()}"


def get_engine(db_path: str | None = None):
    global _engine
    if _engine is None:
        url = resolve_db_url(db_path)
        if url.startswith("sqlite:///"):
            path = url[len("sqlite:///"):]
            if path and path != ":memory:":
                os.makedirs(os.path.dirname(path), exist_ok=True)
            _engine = create_engine(url, echo=False)
        else:
            _engine = create_engine(url, echo=False, pool_pre_ping=True)
            return _engine

        # Enable WAL mode — allows reads while writing (no lock conflicts)
        @event.listens_for(_engine, "connect")
        def _set_sqlite_pragmas(dbapi_conn, connection_record):
            cursor = dbapi_conn.cursor()
            # WAL + a relaxed fsync policy — durable enough for a scraper
            # (worst case we lose the last ~second on kernel panic, which we
            # recover from the next run anyway).
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            # 5 min: a writer waits out the lock instead of crashing. The
            # scrape worker holds the write lock for minutes during the
            # market_stats commit (longer under full coverage), and batch jobs
            # like detect_relists used to die instantly on the old 30 s.
            cursor.execute("PRAGMA busy_timeout=300000")
            # 64 MB page cache — keeps the hot working set (active listings,
            # indexes, recent snapshots) in memory instead of paging from disk.
            cursor.execute("PRAGMA cache_size=-65536")
            # ORDER BY / GROUP BY / CREATE INDEX scratch space stays in RAM.
            cursor.execute("PRAGMA temp_store=MEMORY")
            # 256 MB memory-mapped reads — saves a syscall per page on SELECTs.
            cursor.execute("PRAGMA mmap_size=268435456")
            cursor.close()
    return _engine


def open_conn(db_path: str | None = None):
    """Statement-autocommitting connection for scripts that hand-write SQL."""
    return get_engine(db_path).connect().execution_options(
        isolation_level="AUTOCOMMIT"
    )


def get_session():
    global _Session
    if _Session is None:
        _Session = sessionmaker(bind=get_engine())
    return _Session()


def _get_table_columns(conn, table_name: str) -> set[str]:
    return {col["name"] for col in inspect(conn).get_columns(table_name)}


_SCHEMA_VERSION = 6  # bump when _migrate_columns or _dead_json_keys changes


def _read_schema_version(conn) -> int:
    conn.execute(text(
        "CREATE TABLE IF NOT EXISTS _schema_meta (version INTEGER NOT NULL)"
    ))
    # pysqlite autocommits DDL; PostgreSQL does not, and the later
    # ``_write_schema_version`` runs on a fresh connection — without this
    # commit the table it writes to was never created.
    conn.commit()
    row = conn.execute(text("SELECT version FROM _schema_meta LIMIT 1")).fetchone()
    return int(row[0]) if row else 0


def _write_schema_version(conn, version: int):
    conn.execute(text("DELETE FROM _schema_meta"))
    conn.execute(text("INSERT INTO _schema_meta (version) VALUES (:v)"), {"v": version})


_PG_TYPE_OVERRIDES = {"DATETIME": "TIMESTAMP"}


def _portable_type(engine, col_type: str) -> str:
    if engine.dialect.name == "sqlite":
        return col_type
    head, _, tail = col_type.partition(" ")
    return f"{_PG_TYPE_OVERRIDES.get(head.upper(), head)} {tail}".strip()


def init_db(db_path: str | None = None):
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)

    # Schema migrations are idempotent but expensive on large DBs (full
    # SELECT over llm_extras + one ALTER per added column).  Gate on a
    # persisted schema_version so startup cost for scrape/enrich/dashboard
    # is a single integer read once the migration has been applied.
    with engine.connect() as conn:
        current = _read_schema_version(conn)
        if current >= _SCHEMA_VERSION:
            conn.commit()
            return engine

    # Migrate: add columns to existing listings table
    _migrate_columns = [
        ("generation", "TEXT"),
        ("desc_mentions_repair", "BOOLEAN"),
        ("desc_mentions_accident", "BOOLEAN"),
        ("real_mileage_km", "INTEGER"),
        ("desc_mentions_num_owners", "INTEGER"),
        ("desc_mentions_customs_cleared", "BOOLEAN"),
        ("llm_description_hash", "TEXT"),
        ("source", "TEXT DEFAULT 'olx'"),
        ("duplicate_of", "TEXT"),
        ("right_hand_drive", "BOOLEAN"),
        ("deactivated_at", "DATETIME"),
        ("deactivation_reason", "TEXT"),
        ("urgency", "TEXT"),
        ("warranty", "BOOLEAN"),
        ("tuning_or_mods", "TEXT"),
        ("taxi_fleet_rental", "BOOLEAN"),
        ("first_owner_selling", "BOOLEAN"),
        ("mechanical_condition", "TEXT"),
        ("drive_type", "TEXT"),
        ("sub_model", "TEXT"),
        ("trim_level", "TEXT"),
        ("photo_count", "INTEGER"),
        ("description_length", "INTEGER"),
        # v2: LLM-inferred damage severity (0=pristine, 3=salvage/parts).
        # Backfilled by `python -m src.cli enrich` (the pending query
        # re-runs LLM on rows whose llm_extras has no damage_severity yet).
        ("damage_severity", "INTEGER"),
        # v3: seller-profile FK + per-listing trader-title claim +
        # seller profile URL pointer (used by the backfill job to resolve
        # seller_uuid from the profile page after scrape time).
        ("seller_uuid", "TEXT"),
        ("seller_displayed_as", "TEXT"),
        ("seller_profile_url", "TEXT"),
        # v4: structured origin (national/imported) from the OLX/SV "origin"
        # param — distinct from the dropped dead-LLM "imported" key.
        ("origin", "TEXT"),
        # v5: CO₂ emissions g/km (StandVirtual detail "co2_emissions") — ISV input.
        ("co2_g_km", "INTEGER"),
        # v6: actual scrape wall-clock, distinct from last_seen_at (= OLX
        # posted date). NULL on existing rows until the next scrape re-sees
        # them (~all within one deep run); lets us measure real scrape
        # freshness/coverage instead of misreading posted-date as staleness.
        ("last_scraped_at", "DATETIME"),
    ]
    _migrate_unmatched_columns = [
        ("source", "TEXT DEFAULT 'olx'"),
    ]
    # Sellers-table additions. The table itself is created by ``create_all``
    # for fresh DBs; these ALTERs cover dev DBs that ran an earlier v3
    # migration before the bucketing/identity expansion landed. Production
    # has no v3 yet, so these are belt-and-suspenders.
    _migrate_seller_columns = [
        ("family_lifestyle_count", "INTEGER"),
        ("electronics_count", "INTEGER"),
        ("realestate_count", "INTEGER"),
        ("tools_industrial_count", "INTEGER"),
        ("pets_hobby_count", "INTEGER"),
        ("services_jobs_count", "INTEGER"),
        ("social_account_type", "TEXT"),
        ("has_user_photo", "BOOLEAN"),
        ("position_lat", "REAL"),
        ("position_lon", "REAL"),
    ]
    # Columns removed from ORM — drop from DB if present
    _drop_columns = [
        # old heuristic columns (replaced by desc_mentions_* equivalents)
        "needs_repair", "had_accident", "num_owners", "customs_cleared",
        "mileage_suspect", "estimated_repair_cost_eur",
        # never used in src/ (NB: "origin" was here as a dead column but is now a
        # live structured field — captured from the OLX/SV param, see v4 migrate)
        "registration_plate", "tires_condition",
        # removed LLM fields (zero price-model importance)
        "accident_details", "imported", "paint_condition", "service_history",
        "repair_details", "suspicious_signs", "extras", "issues",
        "reason_for_sale", "recent_maintenance",
    ]
    # Keys to strip from llm_extras JSON
    _dead_json_keys = {
        "accident_details", "imported", "paint_condition", "service_history",
        "repair_details", "suspicious_signs", "extras", "issues",
        "reason_for_sale", "recent_maintenance", "tires_condition",
        "accident_free", "legal_issues",
    }
    # Indexes that ADD COLUMN doesn't create automatically. ``create_all``
    # builds them on fresh DBs, but existing rows added via ALTER TABLE
    # need an explicit ``CREATE INDEX IF NOT EXISTS`` to match the ORM.
    _migrate_indexes = [
        ("ix_listings_seller_uuid", "listings", "seller_uuid"),
        ("ix_listings_last_scraped_at", "listings", "last_scraped_at"),
    ]
    with engine.connect() as conn:
        existing_listing_columns = _get_table_columns(conn, "listings")
        for col_name, col_type in _migrate_columns:
            if col_name in existing_listing_columns:
                continue
            try:
                conn.execute(text(
                    f"ALTER TABLE listings ADD COLUMN {col_name} "
                    f"{_portable_type(engine, col_type)}"
                ))
                conn.commit()
            except Exception:
                conn.rollback()
        for idx_name, table, column in _migrate_indexes:
            try:
                conn.execute(text(
                    f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table} ({column})"
                ))
                conn.commit()
            except Exception:
                conn.rollback()
        existing_unmatched_columns = _get_table_columns(conn, "unmatched_listings")
        for col_name, col_type in _migrate_unmatched_columns:
            if col_name in existing_unmatched_columns:
                continue
            try:
                conn.execute(text(
                    f"ALTER TABLE unmatched_listings ADD COLUMN {col_name} "
                    f"{_portable_type(engine, col_type)}"
                ))
                conn.commit()
            except Exception:
                conn.rollback()
        existing_seller_columns = _get_table_columns(conn, "sellers")
        for col_name, col_type in _migrate_seller_columns:
            if col_name in existing_seller_columns:
                continue
            try:
                conn.execute(text(
                    f"ALTER TABLE sellers ADD COLUMN {col_name} "
                    f"{_portable_type(engine, col_type)}"
                ))
                conn.commit()
            except Exception:
                conn.rollback()
        # Drop dead columns
        listing_columns = _get_table_columns(conn, "listings")
        for col_name in _drop_columns:
            if col_name in listing_columns:
                try:
                    conn.execute(text(f"ALTER TABLE listings DROP COLUMN {col_name}"))
                    conn.commit()
                except Exception:
                    conn.rollback()
        # Clean llm_extras JSON: strip removed keys
        rows = conn.execute(
            text("SELECT id, llm_extras FROM listings WHERE llm_extras IS NOT NULL")
        ).fetchall()
        updated = 0
        for row_id, raw in rows:
            try:
                data = json.loads(raw)
                keys_present = set(data) & _dead_json_keys
                if not keys_present:
                    continue
                for k in keys_present:
                    del data[k]
                conn.execute(
                    text("UPDATE listings SET llm_extras = :extras WHERE id = :id"),
                    {"extras": json.dumps(data, ensure_ascii=False), "id": row_id},
                )
                updated += 1
            except (json.JSONDecodeError, TypeError):
                continue
        if updated:
            conn.commit()
        _write_schema_version(conn, _SCHEMA_VERSION)
        conn.commit()
    return engine
