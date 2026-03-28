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
        def _set_sqlite_wal(dbapi_conn, connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.close()
    return _engine


def get_session():
    global _Session
    if _Session is None:
        _Session = sessionmaker(bind=get_engine())
    return _Session()


def init_db(db_path: str | None = None):
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)
    # Migrate: add generation column to existing listings table
    with engine.connect() as conn:
        try:
            conn.execute(text("ALTER TABLE listings ADD COLUMN generation TEXT"))
            conn.commit()
        except Exception:
            conn.rollback()
    return engine
