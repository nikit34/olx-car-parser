"""Integration tests for repository CRUD (in-memory SQLite)."""

from datetime import date

from src.models.listing import Listing, PriceSnapshot, MarketStats, UnmatchedListing
from src.storage.repository import (
    upsert_listing,
    add_price_snapshot,
    upsert_unmatched,
    mark_inactive,
    compute_market_stats,
    get_listings_df,
    get_unmatched_df,
)


class TestUpsertListing:
    def test_insert_new(self, db_session, sample_listing_data):
        listing = upsert_listing(db_session, sample_listing_data)
        db_session.commit()
        assert listing.id is not None
        assert listing.olx_id == "test-001"
        assert listing.brand == "Volkswagen"
        assert listing.generation == "Mk7"
        assert listing.is_active is True

    def test_update_existing(self, db_session, sample_listing_data):
        upsert_listing(db_session, sample_listing_data)
        db_session.commit()

        updated = {**sample_listing_data, "mileage_km": 110000}
        listing = upsert_listing(db_session, updated)
        db_session.commit()

        assert listing.mileage_km == 110000
        assert db_session.query(Listing).count() == 1

    def test_reactivates_on_update(self, db_session, sample_listing_data):
        listing = upsert_listing(db_session, sample_listing_data)
        listing.is_active = False
        db_session.commit()

        upsert_listing(db_session, sample_listing_data)
        db_session.commit()
        assert listing.is_active is True


class TestUpsertUnmatched:
    def test_insert_unmatched(self, db_session):
        data = {"olx_id": "um-001", "url": "https://olx.pt/um", "brand": "Ligier", "model": "JS 50", "year": 2020}
        row = upsert_unmatched(db_session, data, reason="no_generation_match")
        db_session.commit()
        assert row.reason == "no_generation_match"
        assert row.brand == "Ligier"

    def test_update_unmatched(self, db_session):
        data = {"olx_id": "um-002", "url": "https://olx.pt/um2", "brand": "DS", "model": "DS7", "year": 2021}
        upsert_unmatched(db_session, data, reason="no_generation_match")
        db_session.commit()

        data2 = {**data, "price_eur": 25000}
        row = upsert_unmatched(db_session, data2, reason="no_generation_match")
        db_session.commit()
        assert row.price_eur == 25000
        assert db_session.query(UnmatchedListing).count() == 1


class TestPriceSnapshot:
    def test_add_snapshot(self, db_session, sample_listing_data):
        listing = upsert_listing(db_session, sample_listing_data)
        db_session.commit()

        add_price_snapshot(db_session, listing.id, 12500.0, negotiable=True)
        db_session.commit()

        snap = db_session.query(PriceSnapshot).filter_by(listing_id=listing.id).first()
        assert snap.price_eur == 12500.0
        assert snap.negotiable is True


class TestMarkInactive:
    def test_marks_unseen_inactive(self, db_session, sample_listing_data):
        upsert_listing(db_session, sample_listing_data)
        upsert_listing(db_session, {**sample_listing_data, "olx_id": "test-002", "url": "https://olx.pt/test-002"})
        db_session.commit()

        mark_inactive(db_session, {"test-001"})
        db_session.commit()

        l1 = db_session.query(Listing).filter_by(olx_id="test-001").one()
        l2 = db_session.query(Listing).filter_by(olx_id="test-002").one()
        assert l1.is_active is True
        assert l2.is_active is False


class TestComputeMarketStats:
    def test_computes_stats(self, db_session, sample_listing_data):
        for i, price in enumerate([10000, 12000, 14000]):
            data = {**sample_listing_data, "olx_id": f"ms-{i}", "url": f"https://olx.pt/ms-{i}"}
            listing = upsert_listing(db_session, data)
            add_price_snapshot(db_session, listing.id, price)
        db_session.commit()

        compute_market_stats(db_session, target_date=date(2024, 1, 15))

        stats = db_session.query(MarketStats).first()
        assert stats is not None
        assert stats.brand == "Volkswagen"
        assert stats.model == "Golf"
        assert stats.listing_count == 3
        assert stats.median_price_eur == 12000.0
        assert stats.min_price_eur == 10000.0
        assert stats.max_price_eur == 14000.0


class TestGetDataFrames:
    def test_get_listings_df(self, db_session, sample_listing_data):
        listing = upsert_listing(db_session, sample_listing_data)
        add_price_snapshot(db_session, listing.id, 11000)
        db_session.commit()

        df = get_listings_df(db_session)
        assert len(df) == 1
        assert df.iloc[0]["brand"] == "Volkswagen"
        assert df.iloc[0]["price_eur"] == 11000

    def test_get_unmatched_df_empty(self, db_session):
        df = get_unmatched_df(db_session)
        assert df.empty

    def test_get_unmatched_df(self, db_session):
        upsert_unmatched(db_session, {
            "olx_id": "u1", "url": "https://olx.pt/u1", "brand": "DS", "model": "DS7", "year": 2021,
        }, reason="no_generation_match")
        db_session.commit()

        df = get_unmatched_df(db_session)
        assert len(df) == 1
        assert df.iloc[0]["reason"] == "no_generation_match"
