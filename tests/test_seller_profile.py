"""Tests for the seller-profile parser (src/parser/seller_profile.py)."""

from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import patch, MagicMock

import pytest
from bs4 import BeautifulSoup

from src.parser.olx_categories import categorise_facets
from src.parser.seller_profile import (
    SellerProfile,
    extract_prerendered_state,
    parse_seller_link,
    parse_seller_profile_html,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wrap_state(state_dict: dict) -> str:
    """Build the exact ``window.__PRERENDERED_STATE__= "..."`` shape OLX
    serves: a JSON-encoded JSON string assigned in a script tag, then a
    second statement so our terminator regex has somewhere to anchor.
    """
    encoded = json.dumps(json.dumps(state_dict, ensure_ascii=False))
    return (
        "<html><body></body>\n"
        "<script>\n"
        f"        window.__PRERENDERED_STATE__= {encoded};\n"
        '        window.__INITIAL_STATE__= "";\n'
        "</script></html>"
    )


def _user_listing_state(
    *,
    seller_uuid: str = "uuid-1",
    name: str = "Rui",
    is_business: bool = False,
    created: str = "2019-11-14T01:02:25+00:00",
    last_seen: str = "2026-05-04T22:44:58+01:00",
    total: int = 3,
    facets_category: list[dict] | None = None,
    categories_list: dict | None = None,
    user_ads_url: str = "https://www.olx.pt/ads/user/7lr1l/",
    social_network_account_type: str | None = None,
    user_photo: str | None = None,
    map_lat: float | None = 38.70,
    map_lon: float | None = -9.18,
) -> dict:
    return {
        "userListing": {
            "seller": {
                "data": {
                    "id": 108543111,
                    "uuid": seller_uuid,
                    "name": name,
                    "is_business": is_business,
                    "created": created,
                    "last_seen": last_seen,
                    "last_login": last_seen,
                    "user_ads_url": user_ads_url,
                    "type": "user",
                    "social_network_account_type": social_network_account_type,
                    "show_photo": True,
                    "user_photo": user_photo,
                    "position": {"map_lat": map_lat, "map_lon": map_lon},
                }
            },
            "adsOffers": {
                "metadata": {
                    "total_elements": total,
                    "facets": {"category": facets_category or []},
                }
            },
        },
        "categories": {"list": categories_list or {}},
    }


# ---------------------------------------------------------------------------
# parse_seller_link
# ---------------------------------------------------------------------------


LISTING_DETAIL_FRAGMENT = """\
<html><body>
<div data-testid="seller_card">
  <p data-testid="trader-title">Utilizador</p>
  <a data-testid="user-profile-link" href="/ads/user/7lr1l/">
    <h4 data-testid="user-profile-user-name">Rui</h4>
    <p data-testid="member-since">No OLX desde <span>novembro de 2019</span></p>
  </a>
</div>
</body></html>
"""


class TestParseSellerLink:
    def test_extracts_user_profile_short_id_and_name(self):
        soup = BeautifulSoup(LISTING_DETAIL_FRAGMENT, "lxml")
        link = parse_seller_link(soup)
        assert link is not None
        assert link.profile_url == "https://www.olx.pt/ads/user/7lr1l/"
        assert link.short_id == "7lr1l"
        assert link.shop_slug is None
        assert link.display_name == "Rui"
        assert link.displayed_as == "Utilizador"
        assert link.member_since == datetime(2019, 11, 1)

    def test_extracts_business_shop_subdomain(self):
        html = """
        <html><body><a data-testid="user-profile-link"
        href="https://vendasmarcobarros.olx.pt/home/">x</a></body></html>"""
        link = parse_seller_link(BeautifulSoup(html, "lxml"))
        assert link is not None
        assert link.shop_slug == "vendasmarcobarros"
        assert link.short_id is None
        assert link.profile_url == "https://vendasmarcobarros.olx.pt/home/"

    def test_returns_none_when_link_missing(self):
        soup = BeautifulSoup("<html><body></body></html>", "lxml")
        assert parse_seller_link(soup) is None

    def test_member_since_handles_concatenated_text(self):
        # The live page renders "No OLX desdesetembro de 2015" because the
        # static text and the inner <span> sit flush against each other —
        # we need to strip and still match the month name.
        html = """
        <a data-testid="user-profile-link" href="/ads/user/abc/">
        <p data-testid="member-since">No OLX desdesetembro de 2015</p>
        </a>"""
        link = parse_seller_link(BeautifulSoup(html, "lxml"))
        assert link.member_since == datetime(2015, 9, 1)

    def test_member_since_marco_with_cedilha(self):
        html = """
        <a data-testid="user-profile-link" href="/ads/user/abc/">
        <p data-testid="member-since">No OLX desde março de 2020</p>
        </a>"""
        link = parse_seller_link(BeautifulSoup(html, "lxml"))
        assert link.member_since == datetime(2020, 3, 1)

    def test_displayed_as_can_disagree_with_business_flag(self):
        # Real-world case: a seller whose JSON says is_business=True still
        # gets rendered as "Utilizador" on individual listings. We surface
        # both fields so the modelling layer can derive a "pseudo-private"
        # flag from the disagreement.
        soup = BeautifulSoup(LISTING_DETAIL_FRAGMENT, "lxml")
        link = parse_seller_link(soup)
        assert link.displayed_as == "Utilizador"


# ---------------------------------------------------------------------------
# extract_prerendered_state
# ---------------------------------------------------------------------------


class TestExtractPrerenderedState:
    def test_returns_decoded_state_dict(self):
        html = _wrap_state({"userListing": {"x": 1}})
        state = extract_prerendered_state(html)
        assert state == {"userListing": {"x": 1}}

    def test_returns_none_when_blob_missing(self):
        assert extract_prerendered_state("<html></html>") is None
        assert extract_prerendered_state("") is None

    def test_handles_unicode_inside_state(self):
        html = _wrap_state({"userListing": {"name": "Sérgio"}})
        state = extract_prerendered_state(html)
        assert state["userListing"]["name"] == "Sérgio"

    def test_handles_escaped_quotes_inside_strings(self):
        # The state contains real quotes inside string values that get
        # double-escaped when JSON-encoded. The walker must not stop at
        # the first \" it sees inside the value.
        html = _wrap_state({"userListing": {"q": 'a"b"c'}})
        state = extract_prerendered_state(html)
        assert state["userListing"]["q"] == 'a"b"c'

    def test_returns_none_on_truncated_blob(self):
        # Closing quote never arrives — we should bail without raising.
        broken = '<script>window.__PRERENDERED_STATE__= "{\\"a\\":1}'
        assert extract_prerendered_state(broken) is None


# ---------------------------------------------------------------------------
# parse_seller_profile_html — userListing shape
# ---------------------------------------------------------------------------


class TestUserListingProfile:
    def test_parses_core_seller_fields(self):
        state = _user_listing_state()
        profile = parse_seller_profile_html(_wrap_state(state),
                                            "https://www.olx.pt/ads/user/7lr1l/")
        assert isinstance(profile, SellerProfile)
        assert profile.uuid == "uuid-1"
        assert profile.short_id == "7lr1l"
        assert profile.shop_slug is None
        assert profile.name == "Rui"
        assert profile.is_business is False
        assert profile.created_at == datetime.fromisoformat("2019-11-14T01:02:25+00:00")
        assert profile.total_ads == 3

    def test_facets_decoded_into_int_keys(self):
        state = _user_listing_state(
            facets_category=[
                {"id": 362, "count": 7},
                {"id": 378, "count": 5},
                {"id": 741, "count": 2},
            ]
        )
        profile = parse_seller_profile_html(_wrap_state(state))
        assert profile.facets == {362: 7, 378: 5, 741: 2}

    def test_returns_none_when_uuid_missing(self):
        state = {
            "userListing": {
                "seller": {"data": {"name": "Anon"}},
                "adsOffers": {"metadata": {"total_elements": 0}},
            }
        }
        assert parse_seller_profile_html(_wrap_state(state)) is None

    def test_returns_none_when_state_missing(self):
        assert parse_seller_profile_html("<html></html>") is None

    def test_identity_fields_extracted(self):
        # Sérgio's real shape: facebook-linked, no avatar uploaded,
        # northern Portugal coordinates. The has_user_photo flag must
        # NOT be True just because show_photo is — the fallback to the
        # default OLX SVG is what we saw on 3/4 private accounts.
        state = _user_listing_state(
            social_network_account_type="facebook",
            user_photo=None,
            map_lat=41.46008,
            map_lon=-8.144,
        )
        profile = parse_seller_profile_html(_wrap_state(state))
        assert profile.social_account_type == "facebook"
        assert profile.has_user_photo is False
        assert profile.position_lat == pytest.approx(41.46008)
        assert profile.position_lon == pytest.approx(-8.144)

    def test_identity_fields_default_when_missing(self):
        state = _user_listing_state(
            social_network_account_type=None,
            user_photo="https://avatars.olx.cdn/abc.jpg",
            map_lat=None,
            map_lon=None,
        )
        profile = parse_seller_profile_html(_wrap_state(state))
        assert profile.social_account_type is None
        assert profile.has_user_photo is True
        assert profile.position_lat is None
        assert profile.position_lon is None


# ---------------------------------------------------------------------------
# parse_seller_profile_html — shop shape
# ---------------------------------------------------------------------------


def _shop_state(
    *,
    owner_uuid: str = "shop-uuid",
    name: str = "Marco Barros",
    domain: str = "vendasMarcoBarros.olx.pt",
    user_ads_url: str = "https://www.olx.pt/ads/user/abc123/",
    business_type: str = "PartsDealer",
    total: int = 256,
    facets_category: list[dict] | None = None,
) -> dict:
    return {
        "shop": {
            "shop": {
                "data": {
                    "name": name,
                    "domain": domain,
                    "owner_id": 2462239,
                    "owner_uuid": owner_uuid,
                    "details": {
                        "user_ads_url": user_ads_url,
                        "business_type": business_type,
                        "created": "2018-04-19T00:00:00+00:00",
                        "last_seen": "2026-05-04T20:00:00+01:00",
                    },
                }
            },
            "adsOffers": {
                "metadata": {
                    "total_elements": total,
                    "facets": {"category": facets_category or []},
                }
            },
        },
        "categories": {"list": {}},
    }


class TestShopProfile:
    def test_parses_shop_shape(self):
        state = _shop_state(
            facets_category=[
                {"id": 362, "count": 256},
                {"id": 377, "count": 256},
            ],
        )
        profile = parse_seller_profile_html(
            _wrap_state(state),
            "https://vendasmarcobarros.olx.pt/home/",
        )
        assert profile.uuid == "shop-uuid"
        assert profile.shop_slug == "vendasmarcobarros"
        assert profile.short_id == "abc123"
        assert profile.is_business is True
        assert profile.business_type == "PartsDealer"
        assert profile.total_ads == 256
        assert profile.facets[377] == 256

    def test_returns_none_when_owner_uuid_missing(self):
        state = {"shop": {"shop": {"data": {"name": "x"}},
                           "adsOffers": {"metadata": {}}}}
        assert parse_seller_profile_html(_wrap_state(state)) is None


# ---------------------------------------------------------------------------
# categorise_facets
# ---------------------------------------------------------------------------


class TestCategoriseFacets:
    def test_reads_top_level_rollups_directly(self):
        # Sergio's real facet table: 5 carros, 1 truck, 1 scooter, 1 industrial.
        facets = [
            {"id": 362, "count": 7},
            {"id": 378, "count": 5},
            {"id": 416, "count": 1},
            {"id": 379, "count": 1},
            {"id": 4918, "count": 1},
        ]
        out = categorise_facets(facets)
        assert out["cars"] == 5
        assert out["commercial"] == 1
        assert out["motos"] == 1
        # 4918 is a top-level non-auto bucket (Equipamentos e Ferramentas)
        assert out["non_auto"] == 1
        assert out["parts"] == 0

    def test_parts_groups_three_dismantler_categories(self):
        facets = [
            {"id": 377, "count": 4},   # Peças e Acessórios
            {"id": 5240, "count": 2},  # Carros para Peças
            {"id": 418, "count": 1},   # Salvados
        ]
        out = categorise_facets(facets)
        assert out["parts"] == 7

    def test_distinct_car_brands_uses_categories_list(self):
        facets = [
            {"id": 378, "count": 8},
            {"id": 741, "count": 2},   # BMW
            {"id": 777, "count": 1},   # VW
            {"id": 809, "count": 1},   # Seat
            {"id": 1497, "count": 1},  # Peugeot — but parent is 416, NOT 378
        ]
        cats = {
            "378": {"parentId": 362},
            "741": {"parentId": 378},
            "777": {"parentId": 378},
            "809": {"parentId": 378},
            "1497": {"parentId": 416},
        }
        out = categorise_facets(facets, categories_list=cats)
        # only BMW + VW + Seat are direct children of 378 → 3 brands
        assert out["distinct_car_brands"] == 3

    def test_distinct_car_brands_zero_without_categories_list(self):
        facets = [{"id": 741, "count": 2}, {"id": 777, "count": 1}]
        out = categorise_facets(facets)
        assert out["distinct_car_brands"] == 0

    def test_unknown_top_level_does_not_crash(self):
        # Future-proofing: an unmapped top-level id just gets dropped on
        # the floor when categories_list is missing — better than raising
        # mid-batch and losing the rest of the seller's signal.
        out = categorise_facets([{"id": 999999, "count": 4}])
        assert out["non_auto"] == 0
        assert out["cars"] == 0

    def test_empty_facets(self):
        out = categorise_facets([])
        assert all(v == 0 for v in out.values())

    def test_non_auto_subbuckets(self):
        # Sérgio's industrial-equipment ad sits under 4918 → tools_industrial.
        # Plus a fictitious tech-reseller with phones and electronics.
        facets = [
            {"id": 99, "count": 4},     # Bebé → family_lifestyle
            {"id": 13, "count": 2},     # Móveis → family_lifestyle
            {"id": 11, "count": 5},     # Tecnologia → electronics
            {"id": 25, "count": 3},     # Telemóveis → electronics
            {"id": 16, "count": 1},     # Imóveis → realestate
            {"id": 4918, "count": 1},   # Equipamentos → tools_industrial
        ]
        out = categorise_facets(facets)
        assert out["family_lifestyle"] == 6
        assert out["electronics"] == 8
        assert out["realestate"] == 1
        assert out["tools_industrial"] == 1
        assert out["pets_hobby"] == 0
        assert out["services_jobs"] == 0
        # The non_auto rollup still sums everything for back-compat.
        assert out["non_auto"] == 4 + 2 + 5 + 3 + 1 + 1  # 16

    def test_auto_facets_do_not_leak_into_subbuckets(self):
        # 378 (Carros) is automotive — must NOT show up in any non_auto bucket.
        out = categorise_facets([{"id": 378, "count": 5}])
        assert out["cars"] == 5
        assert out["family_lifestyle"] == 0
        assert out["non_auto"] == 0


# ---------------------------------------------------------------------------
# Wiring: scrape_listing_detail must surface seller fields in its dict.
# ---------------------------------------------------------------------------

OLX_DETAIL_WITH_SELLER_HTML = """\
<html><body>
<div data-testid="seller_card">
  <p data-testid="trader-title">Utilizador</p>
  <a data-testid="user-profile-link" href="/ads/user/7lr1l/">
    <h4 data-testid="user-profile-user-name">Rui</h4>
    <p data-testid="member-since">No OLX desde novembro de 2019</p>
  </a>
</div>
<div data-cy="ad_description"><div>Bom carro.</div></div>
</body></html>
"""


class TestScrapeListingDetailSellerWiring:
    """``scrape_listing_detail`` must populate the seller_* fields it
    parses from the same soup. The detail dict feeds straight into
    ``_merge_details`` / ``upsert_listing``, so a missing key here
    silently NULLs the listing's seller pointer at scrape time.

    We can't construct ``OlxScraper`` directly — its ``__init__``
    instantiates ``httpx.Client(http2=True)`` which requires the
    optional ``h2`` package not present in this CI env. Patching the
    client to a MagicMock keeps construction happy; ``_fetch`` is
    stubbed to bypass any actual network/HTTP path.
    """

    def _make_scraper_with_mocked_client(self, html: str):
        from src.parser.scraper import OlxScraper, ScraperConfig
        with patch("src.parser.scraper.httpx.Client", return_value=MagicMock()):
            scraper = OlxScraper(ScraperConfig())
        scraper._fetch = lambda url, retries=3: (url, html)
        return scraper

    def test_user_profile_link_lands_in_details_dict(self):
        scraper = self._make_scraper_with_mocked_client(OLX_DETAIL_WITH_SELLER_HTML)
        d = scraper.scrape_listing_detail(
            "https://www.olx.pt/d/anuncio/test-IDx.html"
        )
        assert d["seller_profile_url"] == "https://www.olx.pt/ads/user/7lr1l/"
        assert d["seller_short_id"] == "7lr1l"
        assert d["seller_shop_slug"] is None
        assert d["seller_display_name"] == "Rui"
        assert d["seller_displayed_as"] == "Utilizador"

    def test_seller_keys_absent_when_link_missing(self):
        # Page without a user-profile-link element — pre-2026 layout, or
        # a one-off A/B variant. We must not write spurious keys; the
        # listing's seller_* columns stay NULL until the next scrape
        # captures the link.
        html = '<html><body><div data-cy="ad_description"><div>x</div></div></body></html>'
        scraper = self._make_scraper_with_mocked_client(html)
        d = scraper.scrape_listing_detail(
            "https://www.olx.pt/d/anuncio/test-IDx.html"
        )
        assert "seller_profile_url" not in d
        assert "seller_displayed_as" not in d

    def test_business_shop_link_yields_slug(self):
        html = """\
<html><body>
<a data-testid="user-profile-link" href="https://vendasmarcobarros.olx.pt/home/">x</a>
<div data-cy="ad_description"><div>x</div></div>
</body></html>"""
        scraper = self._make_scraper_with_mocked_client(html)
        d = scraper.scrape_listing_detail(
            "https://www.olx.pt/d/anuncio/test-IDx.html"
        )
        assert d["seller_shop_slug"] == "vendasmarcobarros"
        assert d["seller_short_id"] is None
