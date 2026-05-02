"""Shared test fixtures."""

import sys
import types

# torchvision is a heavy GPU dep that production runners install (verify-photos
# needs it) but the minimal CI test env does not. ``src.parser.photo_damage``
# imports torchvision at module-load time. Tests that touch dashboard
# blocking-deal logic (issue #8) lazy-import that module via
# ``is_listing_flagged`` and would otherwise fail collection on a clean
# venv. Mirror the same shim shape ``tests/test_photo_damage.py`` uses, but
# install it once for the whole test session so any import order works.
if "torchvision" not in sys.modules:
    _tv = types.ModuleType("torchvision")
    _tv_models = types.ModuleType("torchvision.models")
    _tv_transforms = types.ModuleType("torchvision.transforms")

    def _noop(*_a, **_kw):  # pragma: no cover - never invoked under tests
        return None

    for _name in ("resnet50", "efficientnet_b0", "efficientnet_b3"):
        setattr(_tv_models, _name, _noop)
    for _name in ("Compose", "Resize", "ToTensor", "Normalize"):
        setattr(_tv_transforms, _name, _noop)

    _tv.models = _tv_models
    _tv.transforms = _tv_transforms
    sys.modules["torchvision"] = _tv
    sys.modules["torchvision.models"] = _tv_models
    sys.modules["torchvision.transforms"] = _tv_transforms

from contextlib import contextmanager
from unittest.mock import patch, MagicMock

import pytest
import pandas as pd
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from src.models.listing import Base


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


@pytest.fixture
def db_session():
    """In-memory SQLite session, rolled back after each test."""
    engine = create_engine("sqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def _set_wal(dbapi_conn, _):
        dbapi_conn.cursor().execute("PRAGMA journal_mode=WAL")

    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.rollback()
    session.close()


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
    """Active listings DataFrame for signal tests."""
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
