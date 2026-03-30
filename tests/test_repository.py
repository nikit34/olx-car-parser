"""Integration tests for repository CRUD (in-memory SQLite)."""

from datetime import date

from src.models.listing import Listing, PriceSnapshot, MarketStats, UnmatchedListing
from src.storage.repository import (
    upsert_listing,
    add_price_snapshot,
    upsert_unmatched,
    mark_inactive,
    compute_market_stats,
    deduplicate_cross_platform,
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


class TestDeduplicateCrossPlatform:
    def _make_listing(self, db_session, olx_id, source, brand="Volkswagen",
                      model="Golf", year=2015, mileage=100000, price=12000,
                      district="Porto"):
        data = {
            "olx_id": olx_id,
            "url": f"https://{'standvirtual.com' if source == 'standvirtual' else 'olx.pt'}/{olx_id}",
            "brand": brand, "model": model, "year": year, "generation": "Mk7",
            "mileage_km": mileage, "city": district, "district": district,
            "source": source,
        }
        listing = upsert_listing(db_session, data)
        add_price_snapshot(db_session, listing.id, price)
        return listing

    def test_marks_duplicate_when_same_car_on_both(self, db_session):
        self._make_listing(db_session, "olx-1", "olx")
        self._make_listing(db_session, "sv-1", "standvirtual")
        db_session.commit()

        count = deduplicate_cross_platform(db_session)
        assert count == 1

        olx = db_session.query(Listing).filter_by(olx_id="olx-1").one()
        sv = db_session.query(Listing).filter_by(olx_id="sv-1").one()
        # One should be marked as duplicate of the other
        assert (olx.duplicate_of is not None) or (sv.duplicate_of is not None)

    def test_no_dedup_when_different_year(self, db_session):
        self._make_listing(db_session, "olx-2", "olx", year=2015)
        self._make_listing(db_session, "sv-2", "standvirtual", year=2018)
        db_session.commit()

        count = deduplicate_cross_platform(db_session)
        assert count == 0

    def test_no_dedup_when_mileage_too_different(self, db_session):
        self._make_listing(db_session, "olx-3", "olx", mileage=100000)
        self._make_listing(db_session, "sv-3", "standvirtual", mileage=200000)
        db_session.commit()

        count = deduplicate_cross_platform(db_session)
        assert count == 0

    def test_no_dedup_when_price_too_different(self, db_session):
        self._make_listing(db_session, "olx-4", "olx", price=10000)
        self._make_listing(db_session, "sv-4", "standvirtual", price=20000)
        db_session.commit()

        count = deduplicate_cross_platform(db_session)
        assert count == 0

    def test_no_dedup_within_same_platform(self, db_session):
        self._make_listing(db_session, "olx-5a", "olx")
        self._make_listing(db_session, "olx-5b", "olx")
        db_session.commit()

        count = deduplicate_cross_platform(db_session)
        assert count == 0

    def test_dedup_with_close_price_and_mileage(self, db_session):
        """5% difference in price and mileage should still match."""
        self._make_listing(db_session, "olx-6", "olx", price=10000, mileage=100000)
        self._make_listing(db_session, "sv-6", "standvirtual", price=10500, mileage=105000)
        db_session.commit()

        count = deduplicate_cross_platform(db_session)
        assert count == 1
