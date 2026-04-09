"""Database connection and initialization."""

import os
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from sqlalchemy import text

from src.models.listing import Base
import src.models.portfolio  # noqa: F401 — register PortfolioDeal with Base

_engine = None
_Session = None


_LISTING_COLUMN_ALIASES = {
    "needs_repair": "desc_mentions_repair",
    "had_accident": "desc_mentions_accident",
    "num_owners": "desc_mentions_num_owners",
    "customs_cleared": "desc_mentions_customs_cleared",
}


def get_db_path() -> str:
    project_root = Path(__file__).resolve().parent.parent.parent
    return str(project_root / "data" / "olx_cars.db")


def get_engine(db_path: str | None = None):
    global _engine
    if _engine is None:
        path = db_path or get_db_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        _engine = create_engine(f"sqlite:///{path}", echo=False)
        # Enable WAL mode — allows reads while writing (no lock conflicts)
        @event.listens_for(_engine, "connect")
        def _set_sqlite_pragmas(dbapi_conn, connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=30000")  # wait up to 30s instead of failing immediately
            cursor.close()
    return _engine


def get_session():
    global _Session
    if _Session is None:
        _Session = sessionmaker(bind=get_engine())
    return _Session()


def _get_table_columns(conn, table_name: str) -> set[str]:
    rows = conn.execute(text(f"PRAGMA table_info({table_name})")).fetchall()
    return {row[1] for row in rows}


def init_db(db_path: str | None = None):
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)
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
    ]
    _migrate_unmatched_columns = [
        ("source", "TEXT DEFAULT 'olx'"),
    ]
    with engine.connect() as conn:
        for col_name, col_type in _migrate_columns:
            try:
                conn.execute(text(f"ALTER TABLE listings ADD COLUMN {col_name} {col_type}"))
                conn.commit()
            except Exception:
                conn.rollback()
        for col_name, col_type in _migrate_unmatched_columns:
            try:
                conn.execute(text(f"ALTER TABLE unmatched_listings ADD COLUMN {col_name} {col_type}"))
                conn.commit()
            except Exception:
                conn.rollback()
        listing_columns = _get_table_columns(conn, "listings")
        for old_name, new_name in _LISTING_COLUMN_ALIASES.items():
            if old_name in listing_columns and new_name in listing_columns:
                try:
                    conn.execute(
                        text(
                            f"UPDATE listings SET {new_name} = {old_name} "
                            f"WHERE {new_name} IS NULL AND {old_name} IS NOT NULL"
                        )
                    )
                    conn.commit()
                except Exception:
                    conn.rollback()
    return engine
