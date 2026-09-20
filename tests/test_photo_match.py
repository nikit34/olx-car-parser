"""Photo fingerprints: extraction, hashing, matching, and what they decide.

The measurements these tests encode were taken on 2026-09-20 against live
listings: CDN file ids are minted per upload and never shared, whereas the
pixels of a re-posted ad come back at Hamming 0-2.
"""

import json
from datetime import datetime

import numpy as np
from PIL import Image

from src.analytics.photo_match import (
    find_pairs, load_photo_hashes, photo_overlap, stock_hashes, usable_hashes,
)
from src.models.listing import Listing
from src.models.photo import ListingPhoto
from src.parser.photo_fetch import (
    photo_id_from_url, photo_refs_olx_api, photo_refs_standvirtual,
)
from src.parser.photo_hash import (
    dhash, hamming, is_degenerate, thumb_url,
)
from src.storage.repository import (
    add_price_snapshot, deduplicate_same_platform, save_listing_photos,
    upsert_listing,
)


SV_URL = (
    "https://ireland.apollo.olxcdn.com/v1/files/"
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJmbiI6InYxZWc4MzNsMTNtcTEtU1REVlRMUFQiLCJ3IjpbeyJmbiI6IjZtZ2p3bHA3a2dkYjIt"
    "U1REVlRMUFQiLCJzIjoiMTYiLCJhIjoiMCIsInAiOiIxMCwtMTAifV19."
    "dZdNJWpYrIn6HUtprHhC-jdjvyegVe1YvtUafKwpOZQ/image"
)
OLX_URL = "https://ireland.apollo.olxcdn.com:443/v1/files/7oxyyq6pe7fu1-PT/image;s=1000x700"


def _photo(seed: int, size=(120, 90)) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (*size[::-1], 3), dtype=np.uint8))


class TestPhotoIdExtraction:
    def test_olx_id_is_the_path_segment(self):
        assert photo_id_from_url(OLX_URL) == "7oxyyq6pe7fu1-PT"

    def test_standvirtual_id_comes_out_of_the_signed_payload(self):
        """The URL is a JWT; the file it names is in ``fn``, the watermark in
        ``w`` is the same overlay on every advert and must not be returned."""
        assert photo_id_from_url(SV_URL) == "v1eg833l13mq1-STDVTLPT"

    def test_junk_url_yields_nothing(self):
        assert photo_id_from_url("") is None
        assert photo_id_from_url("https://example.com/car.jpg") is None

    def test_olx_api_offer_gives_ids_in_gallery_order(self):
        offers = json.load(open("tests/fixtures/api/olx_offers.json"))
        offer = offers["data"][0] if isinstance(offers, dict) else offers[0]
        refs = photo_refs_olx_api(offer)
        assert len(refs) == len(offer["photos"])
        assert [r[0] for r in refs] == [p["filename"] for p in offer["photos"]]
        assert all(r[1] is None for r in refs), "OLX URLs are rebuilt, not stored"

    def test_standvirtual_advert_keeps_the_signed_url(self):
        """It cannot be rebuilt from the id, and the photo still has to be
        fetched once to be hashed."""
        advert = json.load(open("tests/fixtures/api/sv_advert.json"))
        advert = advert.get("props", {}).get("pageProps", {}).get("advert", advert)
        refs = photo_refs_standvirtual(advert)
        assert refs and all(fid.endswith("-STDVTLPT") for fid, _ in refs)
        assert all(url.startswith("https://") for _, url in refs)


class TestHashing:
    def test_same_image_same_hash(self):
        assert dhash(_photo(1)) == dhash(_photo(1))

    def test_rescaling_does_not_move_the_hash_far(self):
        """The CDN re-encodes at every requested size; a hash that drifted
        across renditions would match nothing."""
        big = _photo(7, size=(400, 300))
        small = big.resize((200, 150), Image.LANCZOS)
        assert hamming(dhash(big), dhash(small)) <= 2

    def test_different_images_are_far_apart(self):
        assert hamming(dhash(_photo(1)), dhash(_photo(2))) > 8

    def test_uniform_frames_are_refused_as_evidence(self):
        blank = Image.new("RGB", (120, 90), (18, 18, 18))
        assert is_degenerate(dhash(blank))
        assert not is_degenerate(dhash(_photo(3)))

    def test_thumb_url_rebuilds_olx_and_respects_a_signed_one(self):
        assert thumb_url("7oxyyq6pe7fu1-PT").endswith(
            "/v1/files/7oxyyq6pe7fu1-PT/image;s=200x150")
        assert thumb_url("x-STDVTLPT", SV_URL) == SV_URL + ";s=200x150"
        assert thumb_url("x", OLX_URL) == OLX_URL


class TestOverlap:
    A = "0f1e2d3c4b5a6978"
    A_NEAR = "0f1e2d3c4b5a6979"
    B = "7a3b19e5c2d40816"

    def test_one_shared_photo_is_a_match(self):
        matched, score = photo_overlap([self.A, self.B], [self.A_NEAR])
        assert matched == 1
        assert score == 1.0

    def test_score_is_the_share_of_the_smaller_gallery(self):
        matched, score = photo_overlap([self.A, self.B], [self.A_NEAR, "ffff0000ffff0000"])
        assert matched == 1
        assert score == 0.5

    def test_unrelated_galleries_do_not_match(self):
        assert photo_overlap([self.A], [self.B]) == (0, 0.0)

    def test_degenerate_hashes_never_carry_a_match(self):
        blank = "0000000000000000"
        assert photo_overlap([blank], [blank]) == (0, 0.0)
        assert usable_hashes([blank, self.A]) == [self.A]


class TestFindPairs:
    def test_finds_the_pair_and_leaves_the_rest_alone(self):
        data = {
            "a": ["0f1e2d3c4b5a6978", "7a3b19e5c2d40816"],
            "b": ["0f1e2d3c4b5a6979"],
            "c": ["13579bdf02468ace"],
        }
        pairs = find_pairs(data)
        assert set(pairs) == {("a", "b")}
        assert pairs[("a", "b")][0] == 1

    def test_stock_photo_links_nothing(self):
        """A frame that turns up in more listings than a car could be in is a
        dealer template or a banner, and must not join them all together."""
        banner = "0f1e2d3c4b5a6978"
        data = {f"l{i}": [banner, f"{i:016x}" + ""] for i in range(6)}
        for k in data:
            data[k][1] = f"{hash(k) & 0x0f0f0f0f0f0f0f0f:016x}"
        assert banner in stock_hashes(data)
        assert find_pairs(data) == {}

    def test_a_listing_never_pairs_with_itself(self):
        assert find_pairs({"a": ["0f1e2d3c4b5a6978", "0f1e2d3c4b5a6979"]}) == {}


class TestSaveListingPhotos:
    def test_stores_gallery_order_and_is_idempotent(self, db_session):
        save_listing_photos(db_session, "abc", [("p1-PT", None), ("p2-PT", None)])
        db_session.flush()
        assert save_listing_photos(db_session, "abc", [("p1-PT", None)]) == 0
        rows = (db_session.query(ListingPhoto)
                .filter_by(olx_id="abc").order_by(ListingPhoto.pos).all())
        assert [r.photo_id for r in rows] == ["p1-PT", "p2-PT"]
        assert [r.pos for r in rows] == [0, 1]
        assert all(r.phash is None for r in rows)

    def test_keeps_photos_the_seller_later_removed(self, db_session):
        """The question these rows answer is whether this car has been
        advertised before, so evidence is never withdrawn."""
        save_listing_photos(db_session, "abc", [("p1-PT", None), ("p2-PT", None)])
        db_session.flush()
        save_listing_photos(db_session, "abc", [("p2-PT", None), ("p3-PT", None)])
        db_session.flush()
        ids = {r.photo_id for r in db_session.query(ListingPhoto).filter_by(olx_id="abc")}
        assert ids == {"p1-PT", "p2-PT", "p3-PT"}

    def test_empty_gallery_writes_nothing(self, db_session):
        assert save_listing_photos(db_session, "abc", []) == 0
        assert save_listing_photos(db_session, "abc", None) == 0

    def test_load_photo_hashes_skips_unhashed_rows(self, db_session):
        save_listing_photos(db_session, "abc", [("p1-PT", None), ("p2-PT", None)])
        db_session.flush()
        row = db_session.query(ListingPhoto).filter_by(photo_id="p1-PT").one()
        row.phash = "0f1e2d3c4b5a6978"
        db_session.flush()
        assert load_photo_hashes(db_session) == {"abc": ["0f1e2d3c4b5a6978"]}


class TestDedupDecidedByPhotos:
    """The 2026-09-20 sample: of 163 candidate pairs on live OLX listings, 34
    were one car — and every confirmed pair carried two different prices."""

    def _listing(self, db_session, olx_id, mileage=180000, price=2500,
                 seen=datetime(2026, 4, 1), hashes=()):
        listing = upsert_listing(db_session, {
            "olx_id": olx_id,
            "url": f"https://olx.pt/{olx_id}",
            "brand": "Opel", "model": "Astra", "year": 2008,
            "mileage_km": mileage, "city": "Porto", "district": "Porto",
            "source": "olx",
        })
        listing.first_seen_at = seen
        add_price_snapshot(db_session, listing.id, price)
        save_listing_photos(db_session, olx_id,
                            [(f"{olx_id}-{i}", None) for i in range(len(hashes))])
        db_session.flush()
        for row, h in zip(db_session.query(ListingPhoto)
                          .filter_by(olx_id=olx_id).order_by(ListingPhoto.pos), hashes):
            row.phash = h
        db_session.flush()
        return listing

    def test_shared_photos_beat_a_price_that_moved(self, db_session):
        """The old rule wanted the prices within 1 %, which every confirmed
        duplicate in the sample failed."""
        self._listing(db_session, "dup-a", price=2500,
                      hashes=["0f1e2d3c4b5a6978", "7a3b19e5c2d40816"],
                      seen=datetime(2026, 4, 1))
        self._listing(db_session, "dup-b", price=1750,
                      hashes=["0f1e2d3c4b5a6978"], seen=datetime(2026, 4, 9))
        db_session.commit()

        assert deduplicate_same_platform(db_session) == 1
        assert db_session.query(Listing).filter_by(olx_id="dup-b").one().duplicate_of == "dup-a"

    def test_shared_photos_beat_mileage_that_drifted(self, db_session):
        self._listing(db_session, "km-a", mileage=265000,
                      hashes=["0f1e2d3c4b5a6978"], seen=datetime(2026, 4, 1))
        self._listing(db_session, "km-b", mileage=275000,
                      hashes=["0f1e2d3c4b5a6979"], seen=datetime(2026, 4, 9))
        db_session.commit()
        assert deduplicate_same_platform(db_session) == 1

    def test_mileage_further_than_five_percent_apart_is_not_considered(self, db_session):
        """Nothing beyond that window was ever confirmed, control included."""
        self._listing(db_session, "far-a", mileage=200000,
                      hashes=["0f1e2d3c4b5a6978"])
        self._listing(db_session, "far-b", mileage=240000,
                      hashes=["0f1e2d3c4b5a6978"])
        db_session.commit()
        assert deduplicate_same_platform(db_session) == 0

    def test_photos_that_disagree_block_a_merge_the_attributes_allowed(self, db_session):
        """A white five-door and a red three-door shared brand, model, year,
        district, mileage and price closely enough for the attribute rule, and
        were marked one car. Photographs say otherwise and now win."""
        self._listing(db_session, "diff-a", mileage=180000, price=2500,
                      hashes=["0f1e2d3c4b5a6978"])
        self._listing(db_session, "diff-b", mileage=180000, price=2500,
                      hashes=["7a3b19e5c2d40816"])
        db_session.commit()
        assert deduplicate_same_platform(db_session) == 0

    def test_listings_without_fingerprints_keep_the_old_rule(self, db_session):
        """Everything recorded before the fingerprint table must still
        deduplicate exactly as it did."""
        self._listing(db_session, "old-a", mileage=180000, price=2500,
                      seen=datetime(2026, 4, 1))
        self._listing(db_session, "old-b", mileage=180000, price=2500,
                      seen=datetime(2026, 4, 9))
        db_session.commit()
        assert deduplicate_same_platform(db_session) == 1

    def test_one_side_without_fingerprints_falls_back_too(self, db_session):
        self._listing(db_session, "half-a", price=2500,
                      hashes=["0f1e2d3c4b5a6978"], seen=datetime(2026, 4, 1))
        self._listing(db_session, "half-b", price=1750, seen=datetime(2026, 4, 9))
        db_session.commit()
        assert deduplicate_same_platform(db_session) == 0
