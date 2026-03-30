"""Tests for OLX + StandVirtual scraper parsing logic."""

import re

import pytest

from datetime import datetime

from src.parser.scraper import (
    OlxScraper,
    RawListing,
    ScraperConfig,
    StandVirtualScraper,
    _extract_brand_from_title,
    _extract_brand_from_url,
    _fix_mileage,
    _merge_details,
    _parse_eur_price,
    _parse_pt_date,
)


# ---------------------------------------------------------------------------
# StandVirtual detail page parsing
# ---------------------------------------------------------------------------

SV_DETAIL_HTML = """\
<html><body>
<div data-testid="ad-price">2 900EUR</div>
<div data-testid="make">MarcaNissan</div>
<div data-testid="model">ModeloQashqai</div>
<div data-testid="mileage">Quilómetros130 000 km</div>
<div data-testid="fuel_type">CombustívelDiesel</div>
<div data-testid="gearbox">Tipo de CaixaManual</div>
<div data-testid="first_registration_year">Ano2010</div>
<div data-testid="first_registration_month">Mês de RegistoJulho</div>
<div data-testid="engine_capacity">Cilindrada1 598 cm3</div>
<div data-testid="engine_power">Potência130 cv</div>
<div data-testid="door_count">Nº de portas5</div>
<div data-testid="color">CorVermelho</div>
<div data-testid="body_type">SegmentoCarrinha</div>
<div data-testid="new_used">CondiçãoUsado</div>
<div data-testid="transmission">TracçãoIntegral</div>
<div data-testid="seller-header">ParticularNo Standvirtual desde 2023</div>
<div data-testid="summary-info-area">Nissan Qashqai Negociável2 900EUR</div>
<div data-testid="content-description-section">
NISSAN QASHQAI importada de 2010, 130000 km reais, revisão feita.
</div>
</body></html>
"""


class TestStandVirtualDetailParsing:
    """Test scrape_standvirtual_detail with mocked HTTP."""

    @pytest.fixture()
    def scraper(self):
        s = OlxScraper(ScraperConfig())
        yield s
        s.close()

    def _patch_fetch(self, scraper, html):
        scraper._fetch = lambda url, retries=3: (url, html)

    def test_parses_all_fields(self, scraper):
        self._patch_fetch(scraper, SV_DETAIL_HTML)
        url = "https://www.standvirtual.com/carros/anuncio/nissan-qashqai-ID8PZUgg.html"
        d = scraper.scrape_standvirtual_detail(url)

        assert d["brand"] == "Nissan"
        assert d["model"] == "Qashqai"
        assert d["year"] == 2010
        assert d["mileage_km"] == 130000
        assert d["fuel_type"] == "Diesel"
        assert d["transmission"] == "Manual"
        assert d["engine_cc"] == 1598
        assert d["horsepower"] == 130
        assert d["doors"] == "5"
        assert d["color"] == "Vermelho"
        assert d["segment"] == "Carrinha"
        assert d["condition"] == "Usado"
        assert d["registration_month"] == "Julho"
        assert d["seller_type"] == "Particular"

    def test_parses_price(self, scraper):
        self._patch_fetch(scraper, SV_DETAIL_HTML)
        url = "https://www.standvirtual.com/carros/anuncio/nissan-qashqai-ID8PZUgg.html"
        d = scraper.scrape_standvirtual_detail(url)
        assert d["price_eur"] == 2900.0

    def test_parses_negotiable(self, scraper):
        self._patch_fetch(scraper, SV_DETAIL_HTML)
        url = "https://www.standvirtual.com/carros/anuncio/nissan-qashqai-ID8PZUgg.html"
        d = scraper.scrape_standvirtual_detail(url)
        assert d["negotiable"] is True

    def test_parses_description(self, scraper):
        self._patch_fetch(scraper, SV_DETAIL_HTML)
        url = "https://www.standvirtual.com/carros/anuncio/nissan-qashqai-ID8PZUgg.html"
        d = scraper.scrape_standvirtual_detail(url)
        assert "NISSAN QASHQAI" in d["description"]
        assert "130000 km" in d["description"]

    def test_extracts_olx_id_from_url(self, scraper):
        self._patch_fetch(scraper, SV_DETAIL_HTML)
        url = "https://www.standvirtual.com/carros/anuncio/nissan-qashqai-ID8PZUgg.html"
        d = scraper.scrape_standvirtual_detail(url)
        assert d["olx_id"] == "8PZUgg"

    def test_returns_empty_on_fetch_failure(self, scraper):
        scraper._fetch = lambda url, retries=3: None
        d = scraper.scrape_standvirtual_detail("https://www.standvirtual.com/carros/anuncio/x-IDfail.html")
        assert d == {}

    def test_professional_seller(self, scraper):
        html = SV_DETAIL_HTML.replace("ParticularNo Standvirtual", "ProfissionalNo Standvirtual")
        self._patch_fetch(scraper, html)
        url = "https://www.standvirtual.com/carros/anuncio/test-IDabc.html"
        d = scraper.scrape_standvirtual_detail(url)
        assert d["seller_type"] == "Profissional"


# ---------------------------------------------------------------------------
# Search page: standvirtual URLs accepted
# ---------------------------------------------------------------------------

OLX_SEARCH_WITH_SV = """\
<html><body>
<div data-testid="l-card">
  <a href="https://www.standvirtual.com/carros/anuncio/alfa-romeo-mito-ID8PZbca.html">
    <h6 data-cy="ad-card-title">Alfa Romeo MiTo</h6>
  </a>
  <p data-testid="ad-price">6.700 €</p>
  <p data-testid="location-date">Porto - 28 de março de 2026</p>
  <span data-nx-name="P5">2014 - 160.000 km</span>
</div>
<div data-testid="l-card">
  <a href="https://www.olx.pt/d/anuncio/volkswagen-golf-IDtest123.html">
    <h6 data-cy="ad-card-title">Volkswagen Golf</h6>
  </a>
  <p data-testid="ad-price">12.500 €</p>
  <p data-testid="location-date">Lisboa - 27 de março de 2026</p>
  <span data-nx-name="P5">2018 - 95.000 km</span>
</div>
</body></html>
"""


class TestSearchPageWithStandVirtual:
    def test_accepts_standvirtual_urls(self):
        scraper = OlxScraper(ScraperConfig())
        listings = scraper._parse_search_page(OLX_SEARCH_WITH_SV)
        scraper.close()

        urls = [l.url for l in listings]
        assert any("standvirtual.com" in u for u in urls)
        assert any("olx.pt" in u for u in urls)
        assert len(listings) == 2

    def test_standvirtual_card_fields(self):
        scraper = OlxScraper(ScraperConfig())
        listings = scraper._parse_search_page(OLX_SEARCH_WITH_SV)
        scraper.close()

        sv = [l for l in listings if "standvirtual" in l.url][0]
        assert sv.olx_id == "8PZbca"
        assert sv.title == "Alfa Romeo MiTo"
        assert sv.price_eur == 6700.0
        assert sv.year == 2014
        assert sv.mileage_km == 160000


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

SV_SEARCH_HTML = """\
<html><body>
<article>
  <a href="https://www.standvirtual.com/carros/anuncio/bmw-x1-ver-18-d-sdrive-auto-ID8PZGI9.html">link</a>
  <h2>BMW X1 18 d sDrive Auto</h2>
  <h3>31 900</h3>
  <p>EUR</p>
  <dl>
    <dt>mileage</dt><dd>33 163 km</dd>
    <dt>fuel_type</dt><dd>Diesel</dd>
    <dt>gearbox</dt><dd>Automática</dd>
    <dt>first_registration_year</dt><dd>2020</dd>
  </dl>
</article>
<article>
  <a href="https://www.standvirtual.com/carros/anuncio/fiat-bravo-ID8PZUgy.html">link</a>
  <h2>Fiat Bravo</h2>
  <h3>3 200</h3>
  <p>EUR</p>
  <dl>
    <dt>mileage</dt><dd>168 000 km</dd>
    <dt>fuel_type</dt><dd>Gasolina</dd>
    <dt>gearbox</dt><dd>Manual</dd>
    <dt>first_registration_year</dt><dd>2010</dd>
  </dl>
</article>
<article>
  <a href="https://www.standvirtual.com/carros/novos/catalogo">Not a listing</a>
</article>
</body></html>
"""


class TestStandVirtualSearchParsing:
    def test_parses_listings_from_articles(self):
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

    def test_skips_non_listing_articles(self):
        sv = StandVirtualScraper(ScraperConfig())
        listings = sv._parse_search_page(SV_SEARCH_HTML)
        sv.close()
        # 3rd article has /novos/catalogo link, not /anuncio/
        assert len(listings) == 2

    def test_source_is_standvirtual(self):
        sv = StandVirtualScraper(ScraperConfig())
        listings = sv._parse_search_page(SV_SEARCH_HTML)
        sv.close()
        assert all(l.source == "standvirtual" for l in listings)


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
