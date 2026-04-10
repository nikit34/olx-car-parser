"""Database connection and initialization."""

import json
import os
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from sqlalchemy import text

from src.models.listing import Base
import src.models.portfolio  # noqa: F401 — register PortfolioDeal with Base

_engine = None
_Session = None


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
        ("mechanical_condition", "TEXT"),
        ("drive_type", "TEXT"),
        ("sub_model", "TEXT"),
        ("trim_level", "TEXT"),
        ("photo_count", "INTEGER"),
        ("description_length", "INTEGER"),
    ]
    _migrate_unmatched_columns = [
        ("source", "TEXT DEFAULT 'olx'"),
    ]
    # Columns removed from ORM — drop from DB if present
    _drop_columns = [
        # old heuristic columns (replaced by desc_mentions_* equivalents)
        "needs_repair", "had_accident", "num_owners", "customs_cleared",
        "mileage_suspect", "estimated_repair_cost_eur",
        # never used in src/
        "origin", "registration_plate", "tires_condition",
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
    return engine
