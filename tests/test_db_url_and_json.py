"""Engine selection and the dialect-portable ``llm_extras`` accessor.

Both exist so the same code runs on the dev machine's SQLite file and on
the scrape host's PostgreSQL. The SQL text each dialect gets is asserted
here because a wrong-dialect expression fails only at query time, on the
host, in the middle of a scrape.
"""
from __future__ import annotations

import pytest
from sqlalchemy import select
from sqlalchemy.dialects import postgresql, sqlite

import src.storage.database as db_mod
from src.models.listing import Listing
from src.storage.jsonsql import json_field, json_sql


@pytest.fixture(autouse=True)
def _no_env_url(monkeypatch):
    monkeypatch.delenv("OLX_DB_URL", raising=False)


class TestResolveDbUrl:
    def test_defaults_to_the_repo_sqlite_file(self):
        url = db_mod.resolve_db_url()
        assert url.startswith("sqlite:///")
        assert url.endswith("olx_cars.db")

    def test_env_url_wins_over_the_default(self, monkeypatch):
        monkeypatch.setenv("OLX_DB_URL", "postgresql+psycopg://olx@localhost/olx_cars")
        assert db_mod.resolve_db_url() == "postgresql+psycopg://olx@localhost/olx_cars"

    def test_env_url_wins_when_caller_passes_the_default_path(self, monkeypatch):
        """Callers like ``init_db(str(DB_PATH))`` pass the legacy default —
        that is not a request for SQLite, it predates the env var."""
        monkeypatch.setenv("OLX_DB_URL", "postgresql+psycopg://olx@localhost/olx_cars")
        assert db_mod.resolve_db_url(db_mod.get_db_path()).startswith("postgresql")

    def test_explicit_other_path_beats_env_url(self, monkeypatch, tmp_path):
        """``--db /tmp/copy.db`` must read that copy, not production."""
        monkeypatch.setenv("OLX_DB_URL", "postgresql+psycopg://olx@localhost/olx_cars")
        other = str(tmp_path / "copy.db")
        assert db_mod.resolve_db_url(other) == f"sqlite:///{other}"

    def test_explicit_url_is_passed_through(self):
        assert db_mod.resolve_db_url("postgresql+psycopg://u@h/d") == \
            "postgresql+psycopg://u@h/d"


class TestJsonField:
    def _sql(self, expr, dialect):
        return str(select(Listing.olx_id).where(expr).compile(dialect=dialect))

    def test_sqlite_uses_json_extract(self):
        sql = self._sql(json_field(Listing.llm_extras, "vlm_damage").isnot(None),
                        sqlite.dialect())
        assert "json_extract(listings.llm_extras, '$.vlm_damage')" in sql

    def test_postgres_uses_jsonb_arrow(self):
        sql = self._sql(json_field(Listing.llm_extras, "vlm_damage").isnot(None),
                        postgresql.dialect())
        assert "(listings.llm_extras)::jsonb ->> 'vlm_damage'" in sql

    def test_numeric_comparison_casts_on_postgres(self):
        """``->>`` yields text; comparing it to 2 without the cast either
        errors or compares lexically."""
        sql = self._sql(
            json_field(Listing.llm_extras, "damage_severity", numeric=True) >= 2,
            postgresql.dialect(),
        )
        assert "::numeric >=" in sql

    def test_key_may_be_written_with_the_sqlite_path_prefix(self):
        assert json_field(Listing.llm_extras, "$.damage_severity").key == \
            "damage_severity"

    def test_rejects_a_key_that_could_break_out_of_the_literal(self):
        with pytest.raises(ValueError):
            json_field(Listing.llm_extras, "a' OR '1'='1")


class TestJsonSql:
    def test_sqlite_text_and_numeric(self):
        engine = db_mod.create_engine("sqlite://")
        assert json_sql(engine, "llm_extras", "k") == "json_extract(llm_extras, '$.k')"
        assert json_sql(engine, "llm_extras", "k", numeric=True) == \
            "CAST(json_extract(llm_extras, '$.k') AS REAL)"

    def test_postgres_text_and_numeric(self):
        engine = db_mod.create_engine("postgresql+psycopg://u@h/d")
        assert json_sql(engine, "llm_extras", "k") == "(llm_extras)::jsonb ->> 'k'"
        assert json_sql(engine, "llm_extras", "k", numeric=True) == \
            "((llm_extras)::jsonb ->> 'k')::numeric"

    def test_rejects_an_injectable_key(self):
        engine = db_mod.create_engine("sqlite://")
        with pytest.raises(ValueError):
            json_sql(engine, "llm_extras", "k'); DROP TABLE listings; --")


class TestDashboardGate:
    """``_looks_like_real_db`` sizes a local file; with PostgreSQL there is
    no file, and the old gate would have reported "no data" forever."""

    def test_remote_engine_passes_the_gate_without_a_local_file(self, monkeypatch):
        from src.dashboard import data_loader as dl

        monkeypatch.setenv("OLX_DB_URL", "postgresql+psycopg://olx@localhost/olx_cars")
        assert dl._looks_like_real_db() is True

    def test_sqlite_still_requires_a_real_file(self, monkeypatch, tmp_path):
        from src.dashboard import data_loader as dl

        monkeypatch.setattr(dl, "DB_PATH", tmp_path / "olx_cars.db")
        assert dl._looks_like_real_db() is False
