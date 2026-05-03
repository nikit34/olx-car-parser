"""Integration tests for repository CRUD (in-memory SQLite)."""

from datetime import date, datetime, timedelta, timezone

from src.models.listing import Listing, PriceSnapshot, MarketStats, UnmatchedListing
from src.storage.repository import (
    upsert_listing,
    add_price_snapshot,
    upsert_unmatched,
    mark_inactive,
    heal_mass_sweeps,
    compute_market_stats,
    deduplicate_cross_platform,
    deduplicate_same_platform,
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

    def test_drops_future_posted_at(self, db_session, sample_listing_data):
        """A posted_at far in the future (warranty/inspection date that
        slipped past the parser) must not become first_seen_at — fall back
        to scrape time instead."""
        future = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(days=365 * 5)
        listing = upsert_listing(db_session, {**sample_listing_data, "posted_at": future})
        db_session.commit()
        assert listing.first_seen_at < future
        assert listing.last_seen_at < future


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
        upsert_listing(db_session, {**sample_listing_data, "source": "olx"})
        upsert_listing(db_session, {
            **sample_listing_data, "olx_id": "test-002",
            "url": "https://olx.pt/test-002", "source": "olx",
        })
        db_session.commit()

        mark_inactive(db_session, "olx", {"test-001"}, verify_via_url=False)
        db_session.commit()

        l1 = db_session.query(Listing).filter_by(olx_id="test-001").one()
        l2 = db_session.query(Listing).filter_by(olx_id="test-002").one()
        assert l1.is_active is True
        assert l2.is_active is False

    def test_does_not_touch_other_sources(self, db_session, sample_listing_data):
        """An OLX scrape must not deactivate StandVirtual rows even if
        the SV ids aren't in active_olx_ids — bug from 2026-05 where a
        single global mark_inactive wiped one source whenever the other
        scrape returned 0 results."""
        upsert_listing(db_session, {**sample_listing_data, "olx_id": "olx-1",
                                    "url": "https://olx.pt/olx-1", "source": "olx"})
        upsert_listing(db_session, {**sample_listing_data, "olx_id": "sv-1",
                                    "url": "https://standvirtual.com/sv-1",
                                    "source": "standvirtual"})
        db_session.commit()

        # Simulate a successful OLX scrape that just didn't see olx-1.
        mark_inactive(db_session, "olx", {"olx-other"}, verify_via_url=False)
        db_session.commit()

        olx_row = db_session.query(Listing).filter_by(olx_id="olx-1").one()
        sv_row = db_session.query(Listing).filter_by(olx_id="sv-1").one()
        assert olx_row.is_active is False
        # SV must remain active — its scrape wasn't part of this call.
        assert sv_row.is_active is True

    def test_empty_scraped_ids_is_noop(self, db_session, sample_listing_data):
        """If a source's scrape returned zero ids the call must not
        deactivate everything — empty set means "scrape failed", not
        "everything sold"."""
        upsert_listing(db_session, {**sample_listing_data, "source": "olx"})
        db_session.commit()

        updated = mark_inactive(db_session, "olx", set(), verify_via_url=False)
        db_session.commit()

        assert updated == 0
        l = db_session.query(Listing).filter_by(olx_id="test-001").one()
        assert l.is_active is True

    def test_legacy_null_source_treated_as_olx(self, db_session, sample_listing_data):
        """Old rows predate the source column and stored NULL. They must
        be deactivated by an OLX scrape (NULL-source listings only ever
        came from OLX before SV support was added)."""
        listing = upsert_listing(db_session, sample_listing_data)
        listing.source = None  # legacy row
        db_session.commit()

        mark_inactive(db_session, "olx", {"olx-other"}, verify_via_url=False)
        db_session.commit()

        legacy = db_session.query(Listing).filter_by(olx_id="test-001").one()
        assert legacy.is_active is False


class TestMarkInactiveURLVerify:
    """The 2026-05-03 audit found Alfa Romeo 147 ``JmEjz`` flagged as
    ``deactivation_reason='sold'`` while still live on OLX — the
    listing dropped out of one scrape's promoted-slot rotation but
    nothing actually changed. URL verify probes each candidate before
    marking it sold so a transient miss can't false-positive."""

    def _seed(self, db_session, sample_listing_data, n: int):
        """Seed N listings, return [(olx_id, url), ...]."""
        from src.storage.repository import upsert_listing
        rows = []
        for i in range(n):
            oid = f"v-{i:03d}"
            url = f"https://olx.pt/d/anuncio/{oid}.html"
            upsert_listing(db_session, {
                **sample_listing_data, "olx_id": oid, "url": url,
                "source": "olx",
            })
            rows.append((oid, url))
        db_session.commit()
        return rows

    def test_alive_listing_is_NOT_marked_sold(
        self, db_session, sample_listing_data,
    ):
        """The literal JmEjz scenario: scrape didn't see the listing
        (passed empty active_olx_ids), but the URL still returns 200
        with valid HTML → listing stays active."""
        from unittest.mock import patch, Mock
        rows = self._seed(db_session, sample_listing_data, 1)

        fake_resp = Mock(status_code=200, text="<html>Alfa Romeo 147 ...</html>")
        with patch("src.storage.repository.httpx.get", return_value=fake_resp):
            mark_inactive(db_session, "olx", {"some-other-id"})
        db_session.commit()
        l = db_session.query(Listing).filter_by(olx_id=rows[0][0]).one()
        assert l.is_active is True
        assert l.deactivation_reason is None

    def test_404_marks_sold(self, db_session, sample_listing_data):
        from unittest.mock import patch, Mock
        rows = self._seed(db_session, sample_listing_data, 1)

        fake_resp = Mock(status_code=404, text="")
        with patch("src.storage.repository.httpx.get", return_value=fake_resp):
            mark_inactive(db_session, "olx", {"some-other-id"})
        db_session.commit()
        l = db_session.query(Listing).filter_by(olx_id=rows[0][0]).one()
        assert l.is_active is False
        assert l.deactivation_reason == "sold"

    def test_dead_marker_in_html_marks_sold(
        self, db_session, sample_listing_data,
    ):
        """OLX/SV sometimes return 200 with a 'this advert no longer
        exists' page instead of 404. We pattern-match the message."""
        from unittest.mock import patch, Mock
        rows = self._seed(db_session, sample_listing_data, 1)

        fake_resp = Mock(
            status_code=200,
            text="<html>O anúncio que tentas aceder não existe ou foi eliminado</html>",
        )
        with patch("src.storage.repository.httpx.get", return_value=fake_resp):
            mark_inactive(db_session, "olx", {"some-other-id"})
        db_session.commit()
        l = db_session.query(Listing).filter_by(olx_id=rows[0][0]).one()
        assert l.is_active is False
        assert l.deactivation_reason == "sold"

    def test_network_error_defers_decision(
        self, db_session, sample_listing_data,
    ):
        """Timeout / connection error → don't deactivate, wait for next
        cycle. Better one false negative than a misclassified sold."""
        from unittest.mock import patch
        import httpx
        rows = self._seed(db_session, sample_listing_data, 1)

        with patch(
            "src.storage.repository.httpx.get",
            side_effect=httpx.TimeoutException("timeout"),
        ):
            mark_inactive(db_session, "olx", {"some-other-id"})
        db_session.commit()
        l = db_session.query(Listing).filter_by(olx_id=rows[0][0]).one()
        assert l.is_active is True
        assert l.deactivation_reason is None

    def test_mixed_population(self, db_session, sample_listing_data):
        """Three candidates, one per outcome (alive / dead / deferred):
        only the dead one gets marked sold."""
        from unittest.mock import patch, Mock
        import httpx
        rows = self._seed(db_session, sample_listing_data, 3)
        alive_url = rows[0][1]
        dead_url = rows[1][1]

        def _fake(url, **_kw):
            if url == alive_url:
                return Mock(status_code=200, text="<html>listing alive</html>")
            if url == dead_url:
                return Mock(status_code=404, text="")
            raise httpx.TimeoutException("timeout")

        with patch("src.storage.repository.httpx.get", side_effect=_fake):
            mark_inactive(db_session, "olx", {"some-other-id"}, max_workers=1)
        db_session.commit()

        states = {
            r[0]: db_session.query(Listing).filter_by(olx_id=r[0]).one().is_active
            for r in rows
        }
        assert states[rows[0][0]] is True   # alive
        assert states[rows[1][0]] is False  # confirmed dead
        assert states[rows[2][0]] is True   # deferred (network error)


class TestHealMassSweeps:
    def _seed(self, db_session, n, source, ts, reason="sold"):
        from src.models.listing import Listing
        for i in range(n):
            db_session.add(Listing(
                olx_id=f"{source}-{i}",
                url=f"https://example.com/{source}-{i}",
                brand="VW", model="Golf", year=2015,
                source=source, is_active=False,
                deactivated_at=ts, deactivation_reason=reason,
            ))
        db_session.commit()

    def test_restores_cluster_above_threshold(self, db_session):
        from datetime import datetime
        from src.models.listing import Listing
        sweep_ts = datetime(2026, 4, 1, 12, 0, 0, 123456)
        self._seed(db_session, 600, "olx", sweep_ts)

        restored = heal_mass_sweeps(db_session, threshold=500)
        assert restored == 600

        rows = db_session.query(Listing).filter_by(source="olx").all()
        assert all(r.is_active for r in rows)
        assert all(r.deactivated_at is None for r in rows)
        assert all(r.deactivation_reason is None for r in rows)

    def test_leaves_normal_churn_alone(self, db_session):
        """A normal mark_inactive batch (~50–200 rows / cycle) must not
        trip the healer."""
        from datetime import datetime
        from src.models.listing import Listing
        normal_ts = datetime(2026, 4, 1, 12, 0, 0, 123456)
        self._seed(db_session, 80, "olx", normal_ts)

        restored = heal_mass_sweeps(db_session, threshold=500)
        assert restored == 0
        assert db_session.query(Listing).filter_by(is_active=False).count() == 80

    def test_idempotent(self, db_session):
        from datetime import datetime
        sweep_ts = datetime(2026, 4, 1, 12, 0, 0, 123456)
        self._seed(db_session, 600, "olx", sweep_ts)

        first = heal_mass_sweeps(db_session, threshold=500)
        second = heal_mass_sweeps(db_session, threshold=500)
        assert first == 600
        assert second == 0

    def test_only_heals_one_source_per_bucket(self, db_session):
        """Two different sources happening to share the same
        deactivated_at must each be evaluated separately."""
        from datetime import datetime
        from src.models.listing import Listing
        ts = datetime(2026, 4, 1, 12, 0, 0, 123456)
        self._seed(db_session, 600, "olx", ts)
        self._seed(db_session, 50, "standvirtual", ts)

        restored = heal_mass_sweeps(db_session, threshold=500)
        # Only the OLX cluster (600) is above threshold
        assert restored == 600
        sv_inactive = db_session.query(Listing).filter_by(
            source="standvirtual", is_active=False,
        ).count()
        assert sv_inactive == 50

    def test_skips_sweeps_after_cutoff(self, db_session):
        """Post-fix sweeps must be left alone — once mark_inactive is
        source-scoped, a 500+ row batch is genuine churn (or a separate
        bug), not the source-blind regression. Auto-reverting it would
        silently resurrect truly sold rows."""
        from datetime import datetime
        from src.models.listing import Listing
        from src.storage.repository import _HEAL_CUTOFF
        # Anything at-or-after the cutoff is out of scope for the healer.
        post_fix_ts = _HEAL_CUTOFF + timedelta(days=14)
        self._seed(db_session, 600, "olx", post_fix_ts)

        restored = heal_mass_sweeps(db_session, threshold=500)
        assert restored == 0
        still_inactive = db_session.query(Listing).filter_by(is_active=False).count()
        assert still_inactive == 600


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

    def test_merges_attributes_into_canonical(self, db_session):
        """Canonical listing gets missing fields filled from the duplicate."""
        from datetime import datetime

        # OLX listing: more recently updated, but no color
        olx = self._make_listing(db_session, "olx-7", "olx")
        olx.last_seen_at = datetime(2026, 3, 30)
        # SV listing: older, but has color and engine details
        sv = self._make_listing(db_session, "sv-7", "standvirtual")
        sv.last_seen_at = datetime(2026, 3, 20)
        sv.color = "Vermelho"
        sv.horsepower = 130
        sv.engine_cc = 1598
        db_session.commit()

        deduplicate_cross_platform(db_session)

        # OLX is canonical (more recent), gets SV's attributes
        canonical = db_session.query(Listing).filter_by(olx_id="olx-7").one()
        assert canonical.duplicate_of is None
        assert canonical.color == "Vermelho"
        assert canonical.engine_cc == 1598

    def test_most_recently_updated_is_canonical(self, db_session):
        """The listing with the latest last_seen_at becomes canonical."""
        from datetime import datetime, timedelta

        olx = self._make_listing(db_session, "olx-8", "olx")
        sv = self._make_listing(db_session, "sv-8", "standvirtual")
        # SV was updated more recently
        olx.last_seen_at = datetime(2026, 3, 1)
        sv.last_seen_at = datetime(2026, 3, 28)
        # SV has a fresher description, OLX has color
        sv.description = "Carro recém-revisado, tudo em ordem"
        olx.color = "Preto"
        db_session.commit()

        deduplicate_cross_platform(db_session)

        sv = db_session.query(Listing).filter_by(olx_id="sv-8").one()
        olx = db_session.query(Listing).filter_by(olx_id="olx-8").one()
        # SV is canonical (more recent), OLX is duplicate
        assert olx.duplicate_of == "sv-8"
        assert sv.duplicate_of is None
        # SV got OLX's color
        assert sv.color == "Preto"
        # SV kept its own description
        assert "recém-revisado" in sv.description


class TestDeduplicateSamePlatform:
    """Same-platform near-duplicate dedup added 2026-05-02 to catch the
    Peugeot 206 ``8Q0ll0`` / ``8Q0ll4`` case (both StandVirtual, identical
    €700 / 43200 km / 2008 / district). Cross-platform dedup explicitly
    skips groups that don't have one OLX + one SV side, so these were
    silently surfacing as twin signals."""

    def _make(self, db_session, olx_id, source="standvirtual",
              brand="Peugeot", model="206", year=2008, mileage=43200,
              price=700, district="Porto"):
        from datetime import datetime
        data = {
            "olx_id": olx_id,
            "url": f"https://standvirtual.com/{olx_id}" if source == "standvirtual"
                   else f"https://olx.pt/{olx_id}",
            "brand": brand, "model": model, "year": year,
            "mileage_km": mileage, "city": district, "district": district,
            "source": source,
        }
        listing = upsert_listing(db_session, data)
        listing.first_seen_at = datetime(2026, 1, 1)
        add_price_snapshot(db_session, listing.id, price)
        return listing

    def test_marks_identical_same_source_listings(self, db_session):
        """The audit case: both StandVirtual, exact mileage + price match."""
        from datetime import datetime
        canon = self._make(db_session, "8Q0ll0")
        canon.first_seen_at = datetime(2026, 4, 1)
        dup = self._make(db_session, "8Q0ll4")
        dup.first_seen_at = datetime(2026, 4, 15)
        db_session.commit()

        count = deduplicate_same_platform(db_session)
        assert count == 1
        canon_row = db_session.query(Listing).filter_by(olx_id="8Q0ll0").one()
        dup_row = db_session.query(Listing).filter_by(olx_id="8Q0ll4").one()
        assert canon_row.duplicate_of is None
        assert dup_row.duplicate_of == "8Q0ll0"

    def test_does_not_merge_different_mileage(self, db_session):
        """Even on the same platform, different mileage = different unit
        from the same dealer's inventory."""
        self._make(db_session, "lst-a", mileage=43200)
        self._make(db_session, "lst-b", mileage=85000)
        db_session.commit()
        assert deduplicate_same_platform(db_session) == 0

    def test_does_not_merge_different_price(self, db_session):
        """Different prices on identical specs probably means two real
        units (or one was repriced) — refuse to merge."""
        self._make(db_session, "lst-c", price=700)
        self._make(db_session, "lst-d", price=900)
        db_session.commit()
        assert deduplicate_same_platform(db_session) == 0

    def test_does_not_cross_platforms(self, db_session):
        """Cross-platform pairs are deduplicate_cross_platform's job —
        the same-platform pass must skip them so we don't double-mark."""
        self._make(db_session, "olx-a", source="olx")
        self._make(db_session, "sv-a", source="standvirtual")
        db_session.commit()
        assert deduplicate_same_platform(db_session) == 0
