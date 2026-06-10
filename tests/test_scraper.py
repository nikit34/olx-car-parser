"""Tests for OLX + StandVirtual scraper parsing logic."""

import re

import pytest

from datetime import datetime

from src.parser.scraper import (
    OlxScraper,
    RawListing,
    ScraperConfig,
    ScraperParseError,
    StandVirtualScraper,
    _extract_brand_from_title,
    _fix_mileage,
    _merge_details,
    _offer_to_raw,
    _parse_pt_date,
    _strip_desc_chrome,
)


# ---------------------------------------------------------------------------
# StandVirtual detail page parsing
# ---------------------------------------------------------------------------

import json as _json


def _sv_detail_html(advert: dict) -> str:
    """Wrap a SV advert dict in a page with a ``__NEXT_DATA__`` blob."""
    payload = {"props": {"pageProps": {"advert": advert}}}
    return (
        '<html><body><script id="__NEXT_DATA__" type="application/json">'
        + _json.dumps(payload, ensure_ascii=False)
        + "</script></body></html>"
    )


# Controlled advert mirroring the real StandVirtual ``advert`` JSON shape
# (props.pageProps.advert): detail rows under ``details`` with the human
# display string in ``value``; ``price.value`` a string + negotiable flagged
# in ``price.labels``; ``seller.type``; ``createdAt`` ISO; ``images.photos``.
SV_ADVERT = {
    "url": "https://www.standvirtual.com/carros/anuncio/nissan-qashqai-ID8PZUgg.html",
    "title": "Nissan Qashqai",
    "createdAt": "2026-03-29T22:17:00Z",
    "price": {"value": "2900", "currency": "EUR",
              "labels": ["ad-page-negotiable-tag", "Aceita retoma"]},
    "description": "<p>NISSAN QASHQAI importada de 2010, 130000 km reais.</p>",
    "seller": {"type": "PRIVATE", "uuid": "abc-123", "name": "João"},
    "images": {"photos": [{"id": "p1"}, {"id": "p2"}, {"id": "p3"}]},
    "details": [
        {"key": "make", "value": "Nissan"},
        {"key": "model", "value": "Qashqai"},
        {"key": "mileage", "value": "130 000 km"},
        {"key": "fuel_type", "value": "Diesel"},
        {"key": "gearbox", "value": "Manual"},
        {"key": "first_registration_year", "value": "2010"},
        {"key": "first_registration_month", "value": "Julho"},
        {"key": "engine_capacity", "value": "1 598 cm3"},
        {"key": "engine_power", "value": "130 cv"},
        {"key": "door_count", "value": "5"},
        {"key": "color", "value": "Vermelho"},
        {"key": "body_type", "value": "Carrinha"},
        {"key": "new_used", "value": "Usado"},
        {"key": "transmission", "value": "Tracção integral"},
    ],
}


class TestStandVirtualDetailParsing:
    """scrape_standvirtual_detail now reads the embedded __NEXT_DATA__ JSON."""

    @pytest.fixture()
    def scraper(self):
        s = OlxScraper(ScraperConfig())
        yield s
        s.close()

    def _patch_fetch(self, scraper, html):
        scraper._fetch = lambda url, retries=3: (url, html)

    def test_parses_all_fields(self, scraper):
        self._patch_fetch(scraper, _sv_detail_html(SV_ADVERT))
        d = scraper.scrape_standvirtual_detail(SV_ADVERT["url"])

        assert d["brand"] == "Nissan"
        assert d["model"] == "Qashqai"
        assert d["year"] == 2010
        assert d["mileage_km"] == 130000
        assert d["fuel_type"] == "Diesel"
        assert d["transmission"] == "Manual"          # gearbox
        assert d["drive_type"] == "Tracção integral"  # transmission
        assert d["engine_cc"] == 1598
        assert d["horsepower"] == 130
        assert d["doors"] == "5"
        assert d["color"] == "Vermelho"
        assert d["segment"] == "Carrinha"
        assert d["condition"] == "Usado"
        assert d["registration_month"] == "Julho"
        assert d["seller_type"] == "Particular"
        assert d["photo_count"] == 3

    def test_parses_price(self, scraper):
        self._patch_fetch(scraper, _sv_detail_html(SV_ADVERT))
        d = scraper.scrape_standvirtual_detail(SV_ADVERT["url"])
        assert d["price_eur"] == 2900.0

    def test_parses_negotiable(self, scraper):
        self._patch_fetch(scraper, _sv_detail_html(SV_ADVERT))
        d = scraper.scrape_standvirtual_detail(SV_ADVERT["url"])
        assert d["negotiable"] is True

    def test_parses_description(self, scraper):
        self._patch_fetch(scraper, _sv_detail_html(SV_ADVERT))
        d = scraper.scrape_standvirtual_detail(SV_ADVERT["url"])
        assert "NISSAN QASHQAI" in d["description"]
        assert "130000 km" in d["description"]
        assert "<p>" not in d["description"]  # tags stripped

    def test_extracts_olx_id_from_url(self, scraper):
        self._patch_fetch(scraper, _sv_detail_html(SV_ADVERT))
        d = scraper.scrape_standvirtual_detail(SV_ADVERT["url"])
        assert d["olx_id"] == "8PZUgg"

    def test_returns_empty_on_fetch_failure(self, scraper):
        scraper._fetch = lambda url, retries=3: None
        d = scraper.scrape_standvirtual_detail("https://www.standvirtual.com/carros/anuncio/x-IDfail.html")
        assert d == {}

    def test_returns_empty_when_no_next_data(self, scraper):
        self._patch_fetch(scraper, "<html><body>no json here</body></html>")
        d = scraper.scrape_standvirtual_detail(SV_ADVERT["url"])
        assert d == {}

    def test_professional_seller(self, scraper):
        adv = {**SV_ADVERT, "seller": {"type": "PROFESSIONAL"}}
        self._patch_fetch(scraper, _sv_detail_html(adv))
        d = scraper.scrape_standvirtual_detail(adv["url"])
        assert d["seller_type"] == "Profissional"

    def test_posted_at_from_created_at(self, scraper):
        self._patch_fetch(scraper, _sv_detail_html(SV_ADVERT))
        d = scraper.scrape_standvirtual_detail(SV_ADVERT["url"])
        assert d["posted_at"] == datetime(2026, 3, 29, 22, 17)

    def test_posted_at_absent_when_no_created_at(self, scraper):
        adv = {k: v for k, v in SV_ADVERT.items() if k != "createdAt"}
        self._patch_fetch(scraper, _sv_detail_html(adv))
        d = scraper.scrape_standvirtual_detail(adv["url"])
        assert "posted_at" not in d

    def test_parses_real_fixture(self, scraper):
        """Real captured advert payload maps without error (BMW i4)."""
        import pathlib
        adv = _json.loads((pathlib.Path(__file__).parent
                           / "fixtures/api/sv_advert.json").read_text())
        self._patch_fetch(scraper, _sv_detail_html(adv))
        d = scraper.scrape_standvirtual_detail(adv["url"])
        assert d["color"] == "Cinzento"
        assert d["drive_type"] == "Tracção traseira"
        assert d["brand"] == "BMW"


# ---------------------------------------------------------------------------
# _enrich_one routes by domain
# ---------------------------------------------------------------------------

class TestEnrichRouting:
    def test_routes_standvirtual_to_sv_parser(self):
        scraper = OlxScraper(ScraperConfig())
        calls = []

        def mock_sv_detail(url):
            calls.append(("sv", url))
            return {"brand": "Nissan"}

        def mock_olx_detail(url):
            calls.append(("olx", url))
            return {"brand": "VW"}

        scraper.scrape_standvirtual_detail = mock_sv_detail
        scraper.scrape_listing_detail = mock_olx_detail
        scraper._delay = lambda: None

        sv_listing = RawListing(
            olx_id="sv1",
            url="https://www.standvirtual.com/carros/anuncio/test-IDsv1.html",
        )
        scraper._enrich_one(sv_listing)
        assert calls[-1][0] == "sv"

        olx_listing = RawListing(
            olx_id="olx1",
            url="https://www.olx.pt/d/anuncio/test-IDolx1.html",
        )
        scraper._enrich_one(olx_listing)
        assert calls[-1][0] == "olx"

        scraper.close()


# ---------------------------------------------------------------------------
# StandVirtual search page parsing
# ---------------------------------------------------------------------------

def _sv_node(url, title, units, year, mileage, fuel, gearbox, seller="ProfessionalSeller"):
    return {
        "url": url, "title": title, "createdAt": "2026-06-01T10:00:00Z",
        "price": {"amount": {"units": units}},
        "seller": {"__typename": seller},
        "location": {"city": {"name": "Porto"}, "region": {"name": "Porto"}},
        "parameters": [
            {"key": "make", "displayValue": title.split()[0], "value": title.split()[0].lower()},
            {"key": "model", "displayValue": title.split()[1] if len(title.split()) > 1 else "", "value": "m"},
            {"key": "mileage", "value": str(mileage), "displayValue": f"{mileage} km"},
            {"key": "fuel_type", "displayValue": fuel, "value": fuel.lower()},
            {"key": "gearbox", "displayValue": gearbox, "value": gearbox.lower()},
            {"key": "first_registration_year", "value": str(year), "displayValue": str(year)},
        ],
    }


def _sv_search_html(nodes, total=None) -> str:
    """Wrap SV search nodes in a __NEXT_DATA__ urqlState advertSearch blob."""
    advert_search = {
        "advertSearch": {
            "totalCount": total if total is not None else len(nodes),
            "pageInfo": {"pageSize": 32, "currentOffset": 0},
            "edges": [{"node": n} for n in nodes],
        }
    }
    urql = {"k1": {"data": _json.dumps(advert_search, ensure_ascii=False)}}
    payload = {"props": {"pageProps": {"urqlState": urql}}}
    return (
        '<html><body><script id="__NEXT_DATA__" type="application/json">'
        + _json.dumps(payload, ensure_ascii=False)
        + "</script></body></html>"
    )


SV_NODE_BMW = _sv_node(
    "https://www.standvirtual.com/carros/anuncio/bmw-x1-ver-18-d-sdrive-auto-ID8PZGI9.html",
    "BMW X1 18 d sDrive Auto", 31900, 2020, 33163, "Diesel", "Automática")
SV_NODE_FIAT = _sv_node(
    "https://www.standvirtual.com/carros/anuncio/fiat-bravo-ID8PZUgy.html",
    "Fiat Bravo", 3200, 2010, 168000, "Gasolina", "Manual", seller="PrivateSeller")
SV_SEARCH_HTML = _sv_search_html([SV_NODE_BMW, SV_NODE_FIAT])


class TestStandVirtualSearchParsing:
    def test_parses_listings_from_json(self):
        sv = StandVirtualScraper(ScraperConfig())
        listings = sv._parse_search_page(SV_SEARCH_HTML)
        sv.close()
        assert len(listings) == 2

    def test_extracts_fields_correctly(self):
        sv = StandVirtualScraper(ScraperConfig())
        listings = sv._parse_search_page(SV_SEARCH_HTML)
        sv.close()

        bmw = listings[0]
        assert bmw.olx_id == "8PZGI9"
        assert bmw.title == "BMW X1 18 d sDrive Auto"
        assert bmw.price_eur == 31900.0
        assert bmw.year == 2020
        assert bmw.mileage_km == 33163
        assert bmw.fuel_type == "Diesel"
        assert bmw.transmission == "Automática"
        assert bmw.brand == "BMW"
        assert bmw.seller_type == "Profissional"
        assert bmw.source == "standvirtual"

    def test_extracts_second_listing(self):
        sv = StandVirtualScraper(ScraperConfig())
        listings = sv._parse_search_page(SV_SEARCH_HTML)
        sv.close()

        fiat = listings[1]
        assert fiat.olx_id == "8PZUgy"
        assert fiat.brand == "Fiat"
        assert fiat.price_eur == 3200.0
        assert fiat.year == 2010
        assert fiat.mileage_km == 168000
        assert fiat.seller_type == "Particular"

    def test_skips_nodes_without_valid_id(self):
        bad = _sv_node("https://www.standvirtual.com/carros/novos/catalogo",
                       "Catalogo X", 0, 2020, 0, "Diesel", "Manual")
        listings = StandVirtualScraper(ScraperConfig())._parse_search_page(
            _sv_search_html([SV_NODE_BMW, bad]))
        # the catalog url has no ID<slug>.html → dropped
        assert len(listings) == 1
        assert listings[0].olx_id == "8PZGI9"

    def test_empty_when_no_advert_search(self):
        listings = StandVirtualScraper(ScraperConfig())._parse_search_page(
            "<html><body>nothing</body></html>")
        assert listings == []

    def test_source_is_standvirtual(self):
        sv = StandVirtualScraper(ScraperConfig())
        listings = sv._parse_search_page(SV_SEARCH_HTML)
        sv.close()
        assert all(l.source == "standvirtual" for l in listings)

    def test_price_on_request_sentinel_is_none(self):
        # SV uses units=1 for "price on request"; must not become a €1 car.
        from src.parser.scraper import _sv_node_to_raw
        node = _sv_node(
            "https://www.standvirtual.com/carros/anuncio/bmw-330-IDs9.html",
            "BMW 330", 1, 2019, 50000, "Gasolina", "Manual")
        raw = _sv_node_to_raw(node)
        assert raw.price_eur is None
        # a real price still maps through
        node2 = _sv_node(
            "https://www.standvirtual.com/carros/anuncio/bmw-x1-IDs10.html",
            "BMW X1", 14900, 2018, 60000, "Diesel", "Auto")
        assert _sv_node_to_raw(node2).price_eur == 14900.0


# ---------------------------------------------------------------------------
# OLX JSON-API offer parsing
# ---------------------------------------------------------------------------

def _olx_offer(**over):
    base = {
        "id": 670000001,  # numeric API id — must NOT become olx_id
        "url": "https://www.olx.pt/d/anuncio/bmw-320d-IDABCxy.html",
        "title": "BMW 320d",
        "created_time": "2026-05-01T10:00:00+01:00",
        "business": False,
        "location": {"city": {"name": "Lisboa"}, "region": {"name": "Lisboa"}},
        "photos": [{"link": "https://x/v1/files/a-PT/image;s={width}x{height}"},
                   {"link": "https://x/v1/files/b-PT/image;s={width}x{height}"}],
        "description": "Carro impecável<br/>sem acidentes &amp; revisões",
        "params": [
            {"key": "price", "value": {"value": 15000, "label": "15.000 €", "negotiable": True}},
            {"key": "year", "value": {"key": "2018", "label": "2018 "}},
            {"key": "quilometros", "value": {"key": "120000", "label": "120.000 km"}},
            {"key": "modelo", "value": {"key": "320d", "label": "320d"}},
            {"key": "combustivel", "value": {"key": "diesel", "label": "Diesel"}},
            {"key": "gearbox", "value": {"key": "manual", "label": "Manual"}},
            {"key": "body_type", "value": {"key": "sedan", "label": "Sedan"}},
            {"key": "engine_capacity", "value": {"key": "1995", "label": "1.995 "}},
            {"key": "engine_power", "value": {"key": "190", "label": "190 "}},
            {"key": "portas", "value": {"key": "4-5", "label": "4-5"}},
            {"key": "nr_seats", "value": {"key": "5", "label": "5"}},
            {"key": "condicao", "value": {"key": "usado", "label": "Usado"}},
            {"key": "first_registration_month", "value": {"key": "03", "label": "Março"}},
        ],
    }
    base.update(over)
    return base


class TestOlxApiOfferParsing:
    def test_olx_id_from_url_not_numeric_id(self):
        raw = _offer_to_raw(_olx_offer())
        # Continuity rule: dedup key is the URL slug, never the numeric id.
        assert raw.olx_id == "ABCxy"
        assert raw.olx_id != "670000001"

    def test_maps_all_fields(self):
        raw = _offer_to_raw(_olx_offer())
        assert raw.brand == "BMW"
        assert raw.model == "320d"
        assert raw.year == 2018
        assert raw.mileage_km == 120000
        assert raw.price_eur == 15000.0
        assert raw.negotiable is True
        assert raw.fuel_type == "Diesel"
        assert raw.transmission == "Manual"
        assert raw.segment == "Sedan"
        assert raw.engine_cc == 1995
        assert raw.horsepower == 190
        assert raw.doors == "4-5"
        assert raw.seats == 5
        assert raw.condition == "Usado"
        assert raw.registration_month == "Março"
        assert raw.city == "Lisboa"
        assert raw.district == "Lisboa"
        assert raw.seller_type == "Particular"
        assert raw.photo_count == 2
        assert raw.source == "olx"

    def test_business_maps_to_profissional(self):
        raw = _offer_to_raw(_olx_offer(business=True))
        assert raw.seller_type == "Profissional"

    def test_description_cleaned(self):
        raw = _offer_to_raw(_olx_offer())
        assert "<br" not in raw.description
        assert "&amp;" not in raw.description  # entity unescaped
        assert "impecável" in raw.description

    def test_description_br_plus_newline_not_doubled(self):
        # OLX's API HTML carries a <br> AND the author's literal newline per
        # line break; the cleaner must collapse them to a single \n instead of
        # the blank line that previously made ~37% of descriptions double-spaced.
        offer = _olx_offer(description="113.500<br/>\nKms<br />\nTesla Model 3")
        raw = _offer_to_raw(offer)
        assert raw.description == "113.500\nKms\nTesla Model 3"
        assert "\n\n" not in raw.description

    def test_description_double_br_keeps_paragraph_gap(self):
        # A genuine <br><br> paragraph break still survives as one blank line.
        offer = _olx_offer(description="Parágrafo um.<br/><br/>\nParágrafo dois.")
        raw = _offer_to_raw(offer)
        assert raw.description == "Parágrafo um.\n\nParágrafo dois."

    def test_posted_at_is_naive_datetime(self):
        raw = _offer_to_raw(_olx_offer())
        posted = getattr(raw, "_posted_at", None)
        assert posted is not None
        assert posted.tzinfo is None  # repository compares against naive utcnow

    def test_sparse_offer_does_not_crash(self):
        offer = _olx_offer(params=[{"key": "price",
                                    "value": {"value": 5000, "label": "5.000 €"}}])
        raw = _offer_to_raw(offer)
        assert raw.price_eur == 5000.0
        assert raw.year is None
        assert raw.model == ""

    def test_real_fixtures_map_without_error(self):
        import pathlib
        offers = _json.loads((pathlib.Path(__file__).parent
                             / "fixtures/api/olx_offers.json").read_text())
        assert offers
        for o in offers:
            raw = _offer_to_raw(o)
            assert raw.olx_id  # never blank
            assert not raw.olx_id.isdigit()  # slug, not numeric id


class TestDescriptionChromeStrip:
    def test_strips_anotacoes_reportar_prefix(self):
        # The dominant pattern: detail-page get_text() captures the icon-button
        # labels as the first two lines before the real description.
        out = _strip_desc_chrome("Anotações\nReportar\nRENAULT CLIO 2014\n5 portas")
        assert out == "RENAULT CLIO 2014\n5 portas"

    def test_strips_descricao_heading_too(self):
        out = _strip_desc_chrome("Descrição\nAnotações\nReportar\nVW Golf")
        assert out == "VW Golf"

    def test_keeps_text_when_no_chrome(self):
        body = "VW Polo 1.0\nÚnico dono\nManutenção em dia"
        assert _strip_desc_chrome(body) == body

    def test_only_strips_from_top(self):
        # A real line that merely contains a label word is never touched, and
        # stripping stops at the first non-chrome line.
        out = _strip_desc_chrome("Anotações\nCarro impecável\nReportar avarias: nenhuma")
        assert out == "Carro impecável\nReportar avarias: nenhuma"

    def test_empty_and_none_safe(self):
        assert _strip_desc_chrome("") == ""
        assert _strip_desc_chrome(None) is None


class TestOlxApiSearchPage:
    @pytest.fixture()
    def scraper(self):
        s = OlxScraper(ScraperConfig())
        yield s
        s.close()

    def test_private_only_filters_dealers(self, scraper):
        dealer = _olx_offer(url="https://www.olx.pt/d/anuncio/x-IDdeal1.html", business=True)
        private = _olx_offer(url="https://www.olx.pt/d/anuncio/y-IDpriv1.html", business=False)
        scraper._fetch_json = lambda url, retries=3: {"data": [dealer, private]}
        listings = scraper.scrape_search_page(1)
        ids = [l.olx_id for l in listings]
        assert ids == ["priv1"]  # dealer dropped under private_only

    def test_keeps_dealers_when_not_private_only(self):
        s = OlxScraper(ScraperConfig(private_only=False))
        dealer = _olx_offer(url="https://www.olx.pt/d/anuncio/x-IDdeal1.html", business=True)
        private = _olx_offer(url="https://www.olx.pt/d/anuncio/y-IDpriv1.html", business=False)
        s._fetch_json = lambda url, retries=3: {"data": [dealer, private]}
        listings = s.scrape_search_page(1)
        s.close()
        assert {l.olx_id for l in listings} == {"deal1", "priv1"}

    def test_empty_deep_page_returns_none(self, scraper):
        scraper._fetch_json = lambda url, retries=3: {"data": []}
        assert scraper.scrape_search_page(2) is None

    def test_empty_first_page_returns_list(self, scraper):
        scraper._fetch_json = lambda url, retries=3: {"data": []}
        assert scraper.scrape_search_page(1) == []

    def test_offset_cap_stops_paging(self, scraper):
        # page 27 → offset 1040 > cap → None without any fetch
        scraper._fetch_json = lambda url, retries=3: pytest.fail("should not fetch past cap")
        assert scraper.scrape_search_page(27) is None


# ---------------------------------------------------------------------------
# Portuguese date parsing
# ---------------------------------------------------------------------------

class TestParsePtDate:
    def test_olx_format(self):
        d = _parse_pt_date("Para o topo a 29 de março de 2026")
        assert d == datetime(2026, 3, 29, 0, 0)

    def test_sv_format_with_time(self):
        d = _parse_pt_date("29 de março de 2026 às 22:17")
        assert d == datetime(2026, 3, 29, 22, 17)

    def test_all_months(self):
        for month_name, month_num in [
            ("janeiro", 1), ("fevereiro", 2), ("março", 3), ("abril", 4),
            ("maio", 5), ("junho", 6), ("julho", 7), ("agosto", 8),
            ("setembro", 9), ("outubro", 10), ("novembro", 11), ("dezembro", 12),
        ]:
            d = _parse_pt_date(f"1 de {month_name} de 2025")
            assert d.month == month_num, f"Failed for {month_name}"

    def test_returns_none_for_garbage(self):
        assert _parse_pt_date("hello world") is None
        assert _parse_pt_date("") is None

    def test_olx_detail_posted_at(self):
        """OLX detail page stores posted_at in _posted_at after merge."""
        scraper = OlxScraper(ScraperConfig())
        html = '''<html><body>
        <div data-testid="ad-posted-at">Para o topo a 15 de fevereiro de 2026</div>
        </body></html>'''
        scraper._fetch = lambda url, retries=3: (url, html)
        d = scraper.scrape_listing_detail("https://www.olx.pt/d/anuncio/test-IDxyz.html")
        assert d["posted_at"] == datetime(2026, 2, 15, 0, 0)
        scraper.close()


# ---------------------------------------------------------------------------
# Loud-failure detection in scrape_all
# ---------------------------------------------------------------------------

def _olx_raw(oid, price=1000):
    return RawListing(olx_id=oid, url=f"https://www.olx.pt/d/anuncio/x-ID{oid}.html",
                      title=oid, price_eur=price, source="olx")


class TestScrapeFullLoudFailure:
    """Full-coverage scrape must loud-fail (cron exits != 0) when the source
    returns nothing across the whole sweep — modelled on the 2026-04 outage
    where a silent 0-result scrape went unnoticed for ten days."""

    def test_olx_scrape_full_zero_offers_raises(self):
        s = OlxScraper(ScraperConfig(delay_min=0, delay_max=0))
        s._price_bands = lambda lo, hi, cat, max_depth=14: [(0, None, 100)]
        s._scrape_search_page_api = lambda page, category_id=None, extra_params=None: []
        with pytest.raises(ScraperParseError, match="0 offers across all price bands"):
            s.scrape_full()
        s.close()

    def test_olx_scrape_full_dedups_within_and_across_pages(self):
        s = OlxScraper(ScraperConfig(delay_min=0, delay_max=0))
        s._price_bands = lambda lo, hi, cat, max_depth=14: [(0, None, 80)]

        def fake_api(page, category_id=None, extra_params=None):
            if page == 1:
                return [_olx_raw("x1"), _olx_raw("x2"), _olx_raw("x1")]  # promoted dup
            if page == 2:
                return [_olx_raw("x2"), _olx_raw("x3")]  # cross-page dup
            return []
        s._scrape_search_page_api = fake_api
        result = s.scrape_full()
        s.close()
        assert sorted(l.olx_id for l in result) == ["x1", "x2", "x3"]

    def test_olx_scrape_full_partial_empty_does_not_raise(self):
        s = OlxScraper(ScraperConfig(delay_min=0, delay_max=0))
        s._price_bands = lambda lo, hi, cat, max_depth=14: [(0, None, 80)]
        s._scrape_search_page_api = (
            lambda page, category_id=None, extra_params=None:
            [_olx_raw("y1")] if page == 1 else [])
        result = s.scrape_full()
        s.close()
        assert [l.olx_id for l in result] == ["y1"]

    def test_sv_graphql_failure_falls_back_to_ssr(self):
        sv = StandVirtualScraper(ScraperConfig(max_pages=3, delay_min=0, delay_max=0))
        sv._sv_listing_screen = lambda page: None  # force SSR fallback
        sv.scrape_search_page = (
            lambda page=1:
            [RawListing(olx_id="s1", url="https://www.standvirtual.com/carros/anuncio/x-IDs1.html",
                        source="standvirtual")] if page == 1 else None)
        result = sv.scrape_full(enrich_details=False)
        sv.close()
        assert [l.olx_id for l in result] == ["s1"]

    def test_sv_graphql_and_ssr_both_fail_raises(self):
        sv = StandVirtualScraper(ScraperConfig(max_pages=5, delay_min=0, delay_max=0))
        sv._sv_listing_screen = lambda page: None
        sv.scrape_search_page = lambda page=1: []
        with pytest.raises(ScraperParseError, match="GraphQL . SSR both failed"):
            sv.scrape_full(enrich_details=False)
        sv.close()


# ---------------------------------------------------------------------------
# OLX detail page — photo-count selector regression
# ---------------------------------------------------------------------------

# Snippet mirrors the current OLX layout (verified 2026-05-04). Each photo
# is wrapped in [data-testid="ad-photo"]; the surrounding container uses
# the typo "image-galery-container".
OLX_DETAIL_NEW_LAYOUT_HTML = """\
<html><body>
<div data-testid="image-galery-container">
  <div data-testid="ad-photo"><img src="https://ireland.apollo.olxcdn.com/v1/files/a-PT/image"/></div>
  <div data-testid="ad-photo"><img src="https://ireland.apollo.olxcdn.com/v1/files/b-PT/image"/></div>
  <div data-testid="ad-photo"><img src="https://ireland.apollo.olxcdn.com/v1/files/c-PT/image"/></div>
</div>
<div data-cy="ad_description"><div>Carro em bom estado, sem acidentes, ITV em dia.</div></div>
</body></html>
"""

# Pre-2026 layout — kept around so the fallback path is exercised too.
OLX_DETAIL_LEGACY_HTML = """\
<html><body>
<div data-testid="photo-gallery">
  <img src="https://x.example/1.jpg"/>
  <img src="https://x.example/2.jpg"/>
</div>
<div data-cy="ad_description"><div>Bom estado.</div></div>
</body></html>
"""


class TestOlxDetailPhotoCount:
    """Regression: 2026-05-04 audit found photo_count=None on 4436/4438
    active OLX rows because the old [data-testid="photo-gallery"] selector
    no longer matches. The new selector is [data-testid="ad-photo"]."""

    @pytest.fixture()
    def scraper(self):
        s = OlxScraper(ScraperConfig())
        yield s
        s.close()

    def _patch_fetch(self, scraper, html):
        scraper._fetch = lambda url, retries=3: (url, html)

    def test_counts_ad_photo_containers_on_new_layout(self, scraper):
        self._patch_fetch(scraper, OLX_DETAIL_NEW_LAYOUT_HTML)
        d = scraper.scrape_listing_detail(
            "https://www.olx.pt/d/anuncio/test-IDxYz12.html",
        )
        assert d.get("photo_count") == 3

    def test_falls_back_to_legacy_gallery_selector(self, scraper):
        self._patch_fetch(scraper, OLX_DETAIL_LEGACY_HTML)
        d = scraper.scrape_listing_detail(
            "https://www.olx.pt/d/anuncio/test-IDxYz12.html",
        )
        assert d.get("photo_count") == 2

    def test_returns_no_photo_count_when_neither_selector_matches(self, scraper):
        self._patch_fetch(
            scraper, "<html><body><p>nothing here</p></body></html>",
        )
        d = scraper.scrape_listing_detail(
            "https://www.olx.pt/d/anuncio/test-IDxYz12.html",
        )
        assert "photo_count" not in d
