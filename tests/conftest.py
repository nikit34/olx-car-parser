"""Shared test fixtures."""

# Note: the dashboard's blocking-deal logic now imports ``is_listing_flagged``
# from ``src.parser.damage_decision`` (a torch-free sibling of
# ``photo_damage``), so the session-wide torchvision shim that used to live
# here is no longer needed. ``tests/test_photo_damage.py`` and
# ``tests/test_cli_verify_photos.py`` still install local shims because they
# explicitly exercise the heavy classifier path.

import os
import uuid
from contextlib import contextmanager
from unittest.mock import patch, MagicMock

import pytest
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker

from src.models.listing import Base
import src.models.portfolio  # noqa: F401 — register with Base
import src.models.relist  # noqa: F401 — register with Base
import src.models.seller  # noqa: F401 — register with Base


@contextmanager
def _patched_gb_model(multiplier: float = 1.4):
    """Stub the GB price model so compute_signals produces predictions.

    The dashboard's deal scorer requires a fresh GB bundle to surface any
    listing — the median-discount fallback was removed (2026-05-02 audit
    found ~37 % of false-positive top-30 came in via that path). Tests that
    assert a deal IS surfaced therefore need the model layer mocked; this
    helper wires a synthetic ``predicted_price = price_eur * multiplier``
    so every input row reads as undervalued by ``(multiplier - 1) * 100 %``.
    """
    fake_bundle = (
        {"median": MagicMock(), "low": MagicMock(), "high": MagicMock()},
        {},  # cat_maps
        {"conformal_q": 0.0, "conformal_q_per_bucket": {},
         "conformal_q_bucket_edges": None},
        {},  # oof_preds
        None,  # calibrator
        None,  # uncertainty_bundle (option C — None means fallback to per-bucket q)
    )

    def _fake_predict(models, cat_maps, df, **_kw):
        preds = df["price_eur"].astype(float) * multiplier
        return pd.DataFrame(
            {
                "predicted_price": preds.values,
                "fair_price_low": (preds * 0.85).values,
                "fair_price_high": (preds * 1.15).values,
            },
            index=df.index,
        )

    with patch(
        "src.analytics.price_model.load_model", return_value=fake_bundle,
    ), patch(
        "src.analytics.price_model.predict_prices", side_effect=_fake_predict,
    ), patch(
        "src.analytics.price_model.load_importance",
        return_value=pd.DataFrame(),
    ):
        yield


@pytest.fixture
def patched_gb_model():
    """Surface ``_patched_gb_model`` to tests as a context-manager fixture."""
    return _patched_gb_model


# Tests run against a real PostgreSQL because production does: the
# parameter-limit and foreign-key-order failures that reached production on
# 2026-08-28 were both invisible on SQLite, which enforces neither.
_DEFAULT_TEST_DB_URL = "postgresql+psycopg://postgres:postgres@localhost:5432/olx_test"
TEST_DB_URL = os.environ.get("TEST_DB_URL") or _DEFAULT_TEST_DB_URL
_TEST_DB_URL_IS_EXPLICIT = bool(os.environ.get("TEST_DB_URL"))


@pytest.fixture(scope="session")
def _test_engine():
    engine = create_engine(TEST_DB_URL, pool_pre_ping=True)
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except OperationalError as exc:
        # An explicit TEST_DB_URL means someone expects that database — CI
        # sets it, and a skip there would be a green build that tested
        # nothing. Only the local-dev default is allowed to skip.
        if _TEST_DB_URL_IS_EXPLICIT:
            raise RuntimeError(f"TEST_DB_URL is set but unreachable: {exc}") from exc
        pytest.skip(f"no PostgreSQL at {TEST_DB_URL}: {exc}")
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


def reset_module_engine_cache():
    """``database.py`` caches one global engine; clear it so a test can
    point at its own schema without inheriting the previous one."""
    import src.storage.database as db_mod

    db_mod._engine = None
    db_mod._Session = None


@pytest.fixture
def fresh_schema(_test_engine):
    """A throwaway PostgreSQL schema plus the URL that resolves inside it.

    For tests that need to build a database from nothing — schema
    migrations, CLI end-to-end runs — without disturbing the shared
    fixture schema the rest of the suite works in.
    """
    name = f"t_{uuid.uuid4().hex[:12]}"
    with _test_engine.connect() as conn:
        conn.execution_options(isolation_level="AUTOCOMMIT").execute(
            text(f'CREATE SCHEMA "{name}"'))
    sep = "&" if "?" in TEST_DB_URL else "?"
    url = f"{TEST_DB_URL}{sep}options=-csearch_path%3D{name}"
    reset_module_engine_cache()
    try:
        yield url
    finally:
        reset_module_engine_cache()
        with _test_engine.connect() as conn:
            conn = conn.execution_options(isolation_level="AUTOCOMMIT")
            # A CLI entry point under test leaves its session open, and an
            # idle-in-transaction backend holds locks that make DROP SCHEMA
            # wait forever. Tests are serial, so nothing else is mid-query.
            conn.execute(text(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                "WHERE datname = current_database() "
                "  AND pid <> pg_backend_pid() "
                "  AND state = 'idle in transaction'"
            ))
            conn.execute(text(f'DROP SCHEMA "{name}" CASCADE'))


@pytest.fixture
def db_session(_test_engine):
    """A session inside a transaction that is rolled back after each test.

    The schema is built once per session; each test runs in its own
    transaction on a dedicated connection, so a test's ``commit()`` lands in
    a SAVEPOINT and vanishes on rollback. That keeps tests isolated without
    recreating seven tables per test.
    """
    connection = _test_engine.connect()
    transaction = connection.begin()
    session = sessionmaker(bind=connection, join_transaction_mode="create_savepoint")()
    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()


@pytest.fixture
def sample_listing_data():
    """Minimal listing dict for upsert."""
    return {
        "olx_id": "test-001",
        "url": "https://olx.pt/test-001",
        "title": "Test Car",
        "brand": "Volkswagen",
        "model": "Golf",
        "year": 2015,
        "generation": "Mk7",
        "mileage_km": 100000,
        "fuel_type": "Diesel",
        "city": "Porto",
        "district": "Porto",
    }


@pytest.fixture
def generations_data():
    """Deterministic generations dict for testing."""
    return {
        "Volkswagen": {
            "Golf": [
                {"name": "Mk4", "year_from": 1997, "year_to": 2003},
                {"name": "Mk5", "year_from": 2003, "year_to": 2008},
                {"name": "Mk6", "year_from": 2008, "year_to": 2012},
                {"name": "Mk7", "year_from": 2012, "year_to": 2019},
                {"name": "Mk8", "year_from": 2019, "year_to": 2026},
            ],
            "Polo": [
                {"name": "Mk4", "year_from": 2001, "year_to": 2009},
                {"name": "Mk5", "year_from": 2009, "year_to": 2017},
            ],
        },
        "BMW": {
            "3 Series": [
                {"name": "E90", "year_from": 2005, "year_to": 2011},
                {"name": "F30", "year_from": 2012, "year_to": 2019},
                {"name": "G20", "year_from": 2019, "year_to": 2026},
            ],
        },
        "Mercedes-Benz": {
            "E-Class": [
                {"name": "W212", "year_from": 2009, "year_to": 2016},
                {"name": "W213", "year_from": 2016, "year_to": 2023},
            ],
        },
    }


@pytest.fixture
def sample_listings_df():
    """Active listings DataFrame for signal tests.

    The Mk7 generation has 6 priced listings so the comparable lookup
    clears the ≥5 sample gate uniformly applied across sub/gen/model
    fallbacks (compute_signals). a1 is the obvious below-median deal.
    a4 has no year → no generation → excluded by upstream prep.
    """
    return pd.DataFrame([
        {"olx_id": "a1", "url": "", "brand": "Volkswagen", "model": "Golf",
         "year": 2015, "price_eur": 8000, "mileage_km": 150000,
         "fuel_type": "Diesel", "city": "Porto", "district": "Porto", "is_active": True},
        {"olx_id": "a2", "url": "", "brand": "Volkswagen", "model": "Golf",
         "year": 2016, "price_eur": 14000, "mileage_km": 100000,
         "fuel_type": "Diesel", "city": "Lisboa", "district": "Lisboa", "is_active": True},
        {"olx_id": "a3", "url": "", "brand": "Volkswagen", "model": "Golf",
         "year": 2017, "price_eur": 15000, "mileage_km": 80000,
         "fuel_type": "Diesel", "city": "Faro", "district": "Faro", "is_active": True},
        {"olx_id": "a5", "url": "", "brand": "Volkswagen", "model": "Golf",
         "year": 2014, "price_eur": 13000, "mileage_km": 130000,
         "fuel_type": "Diesel", "city": "Braga", "district": "Braga", "is_active": True},
        {"olx_id": "a6", "url": "", "brand": "Volkswagen", "model": "Golf",
         "year": 2018, "price_eur": 16000, "mileage_km": 60000,
         "fuel_type": "Diesel", "city": "Coimbra", "district": "Coimbra", "is_active": True},
        {"olx_id": "a7", "url": "", "brand": "Volkswagen", "model": "Golf",
         "year": 2016, "price_eur": 14500, "mileage_km": 110000,
         "fuel_type": "Diesel", "city": "Aveiro", "district": "Aveiro", "is_active": True},
        # No year → no generation → should be excluded
        {"olx_id": "a4", "url": "", "brand": "Volkswagen", "model": "Golf",
         "year": None, "price_eur": 5000, "mileage_km": 200000,
         "fuel_type": "Diesel", "city": "Porto", "district": "Porto", "is_active": True},
    ])


@pytest.fixture
def sample_history_df():
    """Market stats history for signal tests."""
    return pd.DataFrame([
        {"brand": "Volkswagen", "model": "Golf", "date": "2024-01-01",
         "median_price_eur": 14000, "avg_price_eur": 13000,
         "min_price_eur": 8000, "max_price_eur": 18000, "listing_count": 10},
    ])
