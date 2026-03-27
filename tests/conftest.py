"""Shared test fixtures."""

import pytest
import pandas as pd
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from src.models.listing import Base


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
