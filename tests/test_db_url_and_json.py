"""Engine selection and the ``llm_extras`` accessor.

Both exist so query code stays readable while emitting the SQL PostgreSQL
actually needs. The SQL text is asserted here because a wrong expression
fails only at query time, on the host, in the middle of a scrape.
"""
from __future__ import annotations

import pytest
from sqlalchemy import select
from sqlalchemy.dialects import postgresql

import src.storage.database as db_mod
from src.models.listing import Listing
from src.storage.database import DatabaseNotConfigured
from src.storage.jsonsql import json_field, json_sql


@pytest.fixture(autouse=True)
def _no_env_url(monkeypatch):
    monkeypatch.delenv("OLX_DB_URL", raising=False)


class TestResolveDbUrl:
    def test_env_url_is_used_when_no_argument_is_given(self, monkeypatch):
        monkeypatch.setenv("OLX_DB_URL", "postgresql+psycopg://olx@localhost/olx_cars")
        assert db_mod.resolve_db_url() == "postgresql+psycopg://olx@localhost/olx_cars"

    def test_explicit_url_wins(self, monkeypatch):
        """Tests and one-off tooling point at their own database this way."""
        monkeypatch.setenv("OLX_DB_URL", "postgresql+psycopg://olx@localhost/olx_cars")
        assert db_mod.resolve_db_url("postgresql+psycopg://u@h/other") == \
            "postgresql+psycopg://u@h/other"

    def test_missing_configuration_is_loud(self):
        """There is no local-file fallback to silently drift onto, so an
        unset variable has to say so rather than open something else."""
        with pytest.raises(DatabaseNotConfigured) as exc:
            db_mod.resolve_db_url()
        assert "OLX_DB_URL" in str(exc.value)


class TestJsonField:
    def _sql(self, expr) -> str:
        return str(select(Listing.olx_id).where(expr).compile(
            dialect=postgresql.dialect()))

    def test_reads_a_key_through_jsonb(self):
        sql = self._sql(json_field(Listing.llm_extras, "vlm_damage").isnot(None))
        assert "(listings.llm_extras)::jsonb ->> 'vlm_damage'" in sql

    def test_numeric_comparison_casts(self):
        """``->>`` yields text; comparing it to 2 without the cast either
        errors or compares lexically."""
        sql = self._sql(
            json_field(Listing.llm_extras, "damage_severity", numeric=True) >= 2)
        assert "::numeric >=" in sql

    def test_key_may_carry_the_legacy_path_prefix(self):
        assert json_field(Listing.llm_extras, "$.damage_severity").key == \
            "damage_severity"

    def test_rejects_a_key_that_could_break_out_of_the_literal(self):
        with pytest.raises(ValueError):
            json_field(Listing.llm_extras, "a' OR '1'='1")


class TestJsonSql:
    def test_text_and_numeric_forms(self):
        assert json_sql("llm_extras", "k") == "(llm_extras)::jsonb ->> 'k'"
        assert json_sql("llm_extras", "k", numeric=True) == \
            "((llm_extras)::jsonb ->> 'k')::numeric"

    def test_rejects_an_injectable_key(self):
        with pytest.raises(ValueError):
            json_sql("llm_extras", "k'); DROP TABLE listings; --")


class TestDashboardGate:
    def test_configured_engine_passes_the_gate(self, monkeypatch):
        from src.dashboard import data_loader as dl

        monkeypatch.setenv("OLX_DB_URL", "postgresql+psycopg://olx@localhost/olx_cars")
        assert dl._database_is_configured() is True

    def test_unconfigured_engine_fails_the_gate(self):
        from src.dashboard import data_loader as dl

        assert dl._database_is_configured() is False
