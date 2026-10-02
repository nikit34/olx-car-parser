"""A country feed card with no photo is not a deal.

The Portuguese feed opens each listing and collects its gallery, so
``_resolve_photos`` decides there. The DE/FR/IT feeds never open anything —
the AutoScout crawl stored the gallery at crawl time — so nothing upstream
ever looked at whether a gallery exists, and ``_format_deal`` happily
shipped ``"photo_urls": []``.

Measured on the German corpus 2026-10-01: 6 349 active rows with no photo
in either place the reader looks (``extras.photo_urls`` and ``image_url``),
of which 5 912 come from the aggregates crawler that has not run since
1 September. Exactly one had reached the published feed, which is a small
number and not the reason to close the gate — the reason is that the
corpus grows and nothing stops the next one.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


def _load():
    spec = importlib.util.spec_from_file_location(
        "build_country_blobs",
        Path(__file__).resolve().parent.parent / "scripts" / "build_country_blobs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bcb = _load()


def _row(**over) -> dict:
    base = {
        "olx_id": "as24_de:aaa",
        "url": "https://www.autoscout24.de/angebote/vw-golf-IDaaa.html",
        "title": "VW Golf",
        "brand": "Volkswagen",
        "model": "Golf",
        "year": 2019,
        "price_eur": 18000.0,
        "predicted_price": 22000.0,
        "fair_price_low": 20000.0,
        "fair_price_high": 24000.0,
        "sample_size": 12,
        "band_pct": 20.0,
        "repair_cost_eur": 0,
        "flip_score": 40.0,
        "verdict": "BUY",
        "decision_score": 30.0,
        "is_active": True,
        "first_seen_at": pd.Timestamp("2026-10-01"),
        "last_scraped_at": pd.Timestamp("2026-10-01"),
        "district": "Bayern",
        "generation": "",
        "damage_severity": None,
    }
    base.update(over)
    return base


def _deals(rows: list[dict]) -> list[dict]:
    """Run the real ``_hot_deals`` over *rows*.

    The frame carries the country's whole corpus, not just the deals: the
    year-cell gate (``_cell_deep_enough``) holds back a car whose model-year
    holds fewer than ``MIN_YEAR_CELL_FOR_DEAL`` listings, which is the same
    gate the published DE feed passed through. Nine filler rows per car keep
    the cell deep enough for the photo gate to be what is under test.
    """
    deals = pd.DataFrame(rows)
    corpus = deals.to_dict("records")
    for index in range(9):
        filler = deals.iloc[index % len(deals)].to_dict()
        filler["olx_id"] = f"filler-{index}"
        filler["price_eur"] = float(filler["price_eur"]) + 100 * (index + 1)
        filler["first_seen_at"] = pd.Timestamp("2026-10-01")
        filler["is_active"] = True
        corpus.append(filler)
    listings = pd.DataFrame(corpus).drop_duplicates("olx_id")
    return bcb._hot_deals(deals, listings, pd.DataFrame(), pd.DataFrame(), "DE")


class TestRowPhotos:
    def test_prefers_the_stored_gallery(self):
        row = _row(extras=json.dumps({"photo_urls": ["https://cdn/a.webp"]}),
                   image_url="https://cdn/single.webp")
        assert bcb._row_photos(row) == ["https://cdn/a.webp"]

    def test_falls_back_to_the_single_image(self):
        """Rows written before the crawl kept a list still have one photo."""
        row = _row(extras=json.dumps({"model_group": "Golf"}),
                   image_url="https://cdn/single.webp")
        assert bcb._row_photos(row) == ["https://cdn/single.webp"]

    def test_empty_gallery_list_falls_through_to_the_single_image(self):
        row = _row(extras=json.dumps({"photo_urls": []}),
                   image_url="https://cdn/single.webp")
        assert bcb._row_photos(row) == ["https://cdn/single.webp"]

    def test_both_missing_is_empty(self):
        """The shape the German corpus actually holds: no photo_urls key and
        no image_url. Returns [] — which the feed must then treat as a
        drop, not ship."""
        row = _row(extras=json.dumps({"model_group": "Golf", "variant": "TSI"}),
                   image_url=None)
        assert bcb._row_photos(row) == []

    def test_malformed_extras_is_empty(self):
        row = _row(extras="{not json", image_url=None)
        assert bcb._row_photos(row) == []


class TestFeedWithholdsPhotolessDeals:
    def test_a_deal_with_a_photo_ships(self):
        deals = _deals([_row(extras=json.dumps({"photo_urls": ["https://cdn/a.webp"]}))])
        assert len(deals) == 1
        assert deals[0]["photo_urls"] == ["https://cdn/a.webp"]

    def test_a_deal_with_no_photo_is_withheld(self):
        deals = _deals([_row(extras=json.dumps({"variant": "TSI"}), image_url=None)])
        assert deals == []

    def test_the_photoless_row_only_loses_itself(self):
        """One bad row must not take the feed with it — the DE feed shipped
        68 of 69 cards while exactly one row had no gallery."""
        rows = [
            _row(olx_id="a", extras=json.dumps({"photo_urls": ["https://cdn/a.webp"]})),
            _row(olx_id="b", extras=json.dumps({"variant": "TSI"}), image_url=None),
            _row(olx_id="c", extras=None, image_url="https://cdn/c.webp"),
        ]
        deals = _deals(rows)
        assert [d["olx_id"] for d in deals] == ["a", "c"]

    def test_every_shipped_deal_carries_a_photo(self):
        """The invariant the gate exists to hold, asserted over a mix."""
        rows = [
            _row(olx_id="a", extras=json.dumps({"photo_urls": ["https://cdn/a.webp"]})),
            _row(olx_id="b", extras=json.dumps({"variant": "TSI"}), image_url=None),
            _row(olx_id="c", extras=None, image_url="https://cdn/c.webp"),
            _row(olx_id="d", extras="{broken", image_url=None),
            _row(olx_id="e", extras=json.dumps({"photo_urls": []}), image_url="https://cdn/e.webp"),
        ]
        deals = _deals(rows)
        assert deals, "some deals must survive"
        assert all(d.get("photo_urls") for d in deals)


class TestNoPhotoCountZeroReachesTheFeed:
    """A foreign row whose photo_count is 0 is the site saying there are no
    photos — as real a zero as the Portuguese shape, and it must not ship."""

    def test_zero_photo_count_with_no_urls_is_withheld(self, capsys):
        row = _row(photo_count=0, extras=json.dumps({"variant": "TSI"}), image_url=None)
        deals = _deals([row])
        assert deals == []

    def test_the_withholding_is_logged(self, capsys):
        """A silent drop reads exactly like a quiet market — the reason the
        Portuguese funnel prints its counts is the outage of 2026-08-25."""
        row = _row(photo_count=0, extras=None, image_url=None)
        _deals([row])
        assert "withheld" in capsys.readouterr().out