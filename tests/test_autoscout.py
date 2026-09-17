"""Reading AutoScout24: the fields that decide a number, and the ones that lie.

The German card is the only place the ISV inputs come from, and two of them are
free text the seller fills in. So this file pins what the parser must refuse as
much as what it must read: a CO2 nobody measured, a price without its VAT label,
a listing with no first registration.

It also pins the crawler's manners, because those are a promise made in the
module docstring and a promise nobody tests is a comment: the robots rules it
obeys, and that a 429 or 403 raises instead of retrying.

Since the reader grew a French and an Italian market, it pins the translation
too, against saved pages from all three sites. That part is not cosmetic: the
same Golf reads ``Benzin``, ``Essence`` and ``Benzina``, quotes its power in
``PS``, ``Ch`` and ``CV``, and writes sixty-five thousand kilometres with a
narrow no-break space in France and a dot in Italy. Each of those reaching the
database untranslated is a silent country-shaped hole in the price model, so
the fixtures are real cards and the assertions are the real values.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from src.parser.autoscout import (
    AutoScoutBlocked,
    AutoScoutClient,
    AutoScoutConfig,
    base_url,
    make_path,
    parse_search,
    robots_allows,
    search_path,
)

FIXTURES = Path(__file__).parent / "fixtures" / "as24"


def _saved(name: str) -> str:
    """A saved ``__NEXT_DATA__`` payload back inside the tag the page ships it in."""
    payload = (FIXTURES / name).read_text(encoding="utf-8")
    return ('<html><body><script id="__NEXT_DATA__" type="application/json">'
            + payload + "</script></body></html>")


def _card(**over) -> dict:
    card = {
        "id": "abc-123",
        "url": "/angebote/bmw-3er-320d-abc",
        "price": {"priceRaw": 23900, "vatLabel": "inkl. MwSt."},
        "vehicle": {
            "make": "BMW", "model": "320", "modelGroup": "3er", "variant": "Limousine",
            "motorTypeName": "320d", "transmission": "Automatik", "fuel": "Diesel",
            "mileageInKm": "94.184 km", "engineDisplacementInCCM": "1.995 cm³",
            "isCurrentlyDamaged": False,
        },
        "vehicleDetails": [
            {"data": "94.184 km", "ariaLabel": "Kilometerstand"},
            {"data": "03/2022", "ariaLabel": "Erstzulassung"},
            {"data": "140 kW (190 PS)", "ariaLabel": "Leistung"},
            {"data": "114 g/km (komb.)", "ariaLabel": "CO₂-Emissionen"},
        ],
        "tracking": {"firstRegistration": "03-2022", "mileage": "94184", "price": "23900"},
        "location": {"countryCode": "DE", "zip": "41469", "city": "Neuss"},
        "seller": {"type": "Dealer"},
        "wltpValues": ["4,3 l/100 km (komb.)", "114 g/km (komb.)"],
    }
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(card.get(key), dict):
            card[key] = {**card[key], **value}
        else:
            card[key] = value
    return card


def _page(cards, results=120, pages=6) -> str:
    doc = {"props": {"pageProps": {
        "numberOfResults": results, "numberOfPages": pages, "listings": cards}}}
    return ('<html><body><script id="__NEXT_DATA__" type="application/json">'
            + json.dumps(doc) + "</script></body></html>")


class TestParse:
    def test_a_card_becomes_our_vocabulary(self):
        (l,), meta = parse_search(_page([_card()]))
        assert (meta["results"], meta["pages"]) == (120, 6)
        assert meta["make_models"] == []
        assert (l.brand, l.model, l.motor_type) == ("BMW", "320", "320d")
        assert l.price_eur == 23900
        assert l.year == 2022 and l.registration_month == "03/2022"
        assert l.mileage_km == 94184
        assert l.engine_cc == 1995
        assert l.power_kw == 140 and l.horsepower == 190
        assert l.fuel_type == "Diesel"
        assert l.transmission == "Automática"
        assert l.co2_g_km == 114
        assert l.country_code == "DE"
        assert l.url.startswith("https://www.autoscout24.de/angebote/")

    def test_german_thousands_and_decimals_survive(self):
        (l,), _ = parse_search(_page([_card(vehicle={"engineDisplacementInCCM": "2.993 cm³"})]))
        assert l.engine_cc == 2993

    def test_the_vat_label_rides_with_the_price(self):
        (gross,), _ = parse_search(_page([_card(price={"priceRaw": 1, "vatLabel": "inkl. MwSt."})]))
        (net,), _ = parse_search(_page([_card(price={"priceRaw": 1, "vatLabel": "MwSt. ausweisbar"})]))
        (none,), _ = parse_search(_page([_card(price={"priceRaw": 1, "vatLabel": None})]))
        assert gross.vat_reclaimable is False
        assert net.vat_reclaimable is True
        assert none.vat_label is None and none.vat_reclaimable is None

    def test_a_co2_nobody_measured_is_dropped(self):
        junk = _card(wltpValues=[], tracking={"firstRegistration": "03-2016"},
                     vehicleDetails=[
                         {"data": "03/2016", "ariaLabel": "Erstzulassung"},
                         {"data": "5 g/km (komb.)", "ariaLabel": "CO₂-Emissionen"}])
        (l,), _ = parse_search(_page([junk]))
        assert l.co2_g_km is None
        assert l.year == 2016

    def test_an_empty_co2_field_is_absent_not_zero(self):
        blank = _card(wltpValues=[], vehicleDetails=[
            {"data": "- (g/km)", "ariaLabel": "CO₂-Emissionen"},
            {"data": "03/2016", "ariaLabel": "Erstzulassung"}])
        (l,), _ = parse_search(_page([blank]))
        assert l.co2_g_km is None

    def test_the_registration_the_site_tracks_wins_over_the_printed_label(self):
        (l,), _ = parse_search(_page([_card(
            tracking={"firstRegistration": "07-2019"},
            vehicleDetails=[{"data": "03/2022", "ariaLabel": "Erstzulassung"}])]))
        assert l.year == 2019 and l.registration_month == "07/2019"

    def test_an_electric_car_keeps_its_zero(self):
        ev = _card(vehicle={"fuel": "Elektro"}, wltpValues=["0 g/km (komb.)"])
        (l,), _ = parse_search(_page([ev]))
        assert l.fuel_type == "Eléctrico"
        assert l.co2_g_km == 0

    def test_a_card_without_an_id_or_a_make_is_skipped(self):
        listings, _ = parse_search(_page([_card(id=""), _card(vehicle={"make": ""}), _card()]))
        assert len(listings) == 1

    def test_a_page_that_is_not_the_search_page_yields_nothing(self):
        assert parse_search("<html>no next data</html>") == ([], {})
        assert parse_search("") == ([], {})

    def test_broken_json_is_not_an_exception(self):
        html = ('<script id="__NEXT_DATA__" type="application/json">{oops</script>')
        assert parse_search(html) == ([], {})


class TestManners:
    def test_the_search_form_robots_leaves_open_is_the_one_we_use(self):
        path = search_path("bmw", "320", year=2016)
        assert path.startswith("/lst/bmw/320?")
        assert robots_allows(path)

    def test_the_query_only_search_form_is_refused(self):
        assert not robots_allows("/lst?make=bmw")
        assert not robots_allows("/lst/?make=bmw")
        assert not robots_allows("/dealerarea/x")

    def test_a_disallowed_path_is_skipped_without_a_request(self):
        client = AutoScoutClient(config=AutoScoutConfig(budget=5))
        client._client = httpx.Client(
            transport=httpx.MockTransport(lambda r: pytest.fail("requested a disallowed path")))
        assert client.fetch("/lst?make=bmw") is None
        assert client.spent == 0

    def test_being_asked_to_go_away_stops_the_run(self):
        client = AutoScoutClient(config=AutoScoutConfig(budget=5))
        client._client = httpx.Client(
            transport=httpx.MockTransport(lambda r: httpx.Response(429)))
        with pytest.raises(AutoScoutBlocked):
            client.fetch("/lst/bmw/320?x=1")

    def test_the_budget_is_a_hard_stop(self):
        calls = []

        def handler(request):
            calls.append(request.url.path)
            return httpx.Response(200, text=_page([_card()]))

        client = AutoScoutClient(config=AutoScoutConfig(budget=2, delay_min=0, delay_max=0))
        client._client = httpx.Client(transport=httpx.MockTransport(handler))
        for _ in range(5):
            client.fetch("/lst/bmw/320?x=1")
        assert len(calls) == 2

    def test_paging_stops_at_the_last_page_the_site_reports(self):
        def handler(request):
            return httpx.Response(200, text=_page([_card()], results=20, pages=1))

        client = AutoScoutClient(config=AutoScoutConfig(budget=10, delay_min=0, delay_max=0))
        client._client = httpx.Client(transport=httpx.MockTransport(handler))
        found = client.model_year("bmw", "320", 2016, max_pages=5)
        assert len(found) == 1 and client.spent == 1


class TestPortugueseNamesOnAutoScout:
    """Portugal names a body type as a model; AutoScout24 does not."""

    @staticmethod
    def _q(brand, model):
        from scripts.crawl_autoscout import as24_query
        return as24_query(brand, model)

    def test_an_estate_becomes_the_base_model_plus_a_body_filter(self):
        assert self._q("Peugeot", "308 SW") == ("peugeot", "308", "bt_kombi")
        assert self._q("Seat", "Leon ST") == ("seat", "leon", "bt_kombi")
        assert self._q("Renault", "Mégane Sport Tourer") == ("renault", "megane", "bt_kombi")
        assert self._q("Opel", "Astra Caravan") == ("opel", "astra", "bt_kombi")

    def test_a_coupe_does_the_same(self):
        assert self._q("BMW", "420 Gran Coupé") == ("bmw", "420", "bt_coupe")
        assert self._q("Smart", "ForTwo Coupé") == ("smart", "fortwo", "bt_coupe")
        assert self._q("Mini", "Cabrio Sport Tourer")[2] == "bt_kombi"

    def test_a_model_that_is_only_a_body_word_is_left_alone(self):
        """Stripping "Cabrio" off "Mini Cabrio" would ask for a model with no
        name; AutoScout24 has a Mini called Cabrio, so it goes through as-is."""
        assert self._q("Mini", "Cabrio") == ("mini", "cabrio", None)

    def test_everything_else_goes_through_untouched(self):
        assert self._q("Volkswagen", "Golf") == ("volkswagen", "golf", None)
        assert self._q("Mercedes-Benz", "C 220") == ("mercedes-benz", "c-220", None)
        assert self._q("Citroën", "C3") == ("citroen", "c3", None)

    def test_the_body_filter_lands_in_the_path_robots_allows(self):
        path = search_path("peugeot", "308", year=2016, body="bt_kombi")
        assert path.startswith("/lst/peugeot/308/bt_kombi?")
        assert robots_allows(path)


class TestTheGermanPage:
    """A saved page from autoscout24.de, read the way the crawler reads it."""

    @staticmethod
    def _first():
        listings, meta = parse_search(_saved("search_de.json"), "de")
        return listings, meta

    def test_the_page_reports_its_own_size(self):
        _, meta = self._first()
        assert (meta["results"], meta["pages"]) == (1047, 53)

    def test_a_german_card_arrives_in_portuguese(self):
        listings, _ = self._first()
        first = listings[0]
        assert (first.brand, first.model) == ("Volkswagen", "Golf")
        assert first.fuel_type == "Diesel"
        assert first.transmission == "Automática"
        assert (first.power_kw, first.horsepower) == (85, 116)
        assert first.mileage_km == 120000
        assert (first.year, first.registration_month) == (2018, "10/2018")
        assert first.co2_g_km == 102
        assert first.seller_type == "Profissional"
        assert first.offer_type == "U"
        assert first.version == "Join  Automatik LED"

    def test_the_petrol_manual_is_not_translated_as_the_diesel_automatic(self):
        listings, _ = self._first()
        petrol = listings[1]
        assert petrol.fuel_type == "Gasolina"
        assert petrol.transmission == "Manual"
        assert (petrol.power_kw, petrol.horsepower) == (110, 150)

    def test_the_plz_becomes_a_bundesland(self):
        listings, _ = self._first()
        assert [x.region for x in listings] == [
            "Nordrhein-Westfalen", "Hamburg", "Hessen", "Nordrhein-Westfalen"]

    def test_the_card_links_to_the_german_site_and_a_photo_worth_showing(self):
        first = self._first()[0][0]
        assert first.url.startswith("https://www.autoscout24.de/angebote/")
        assert first.image_url.endswith("/720x540.webp")
        assert "/250x188.webp" not in first.image_url
        assert first.photo_count == 2

    def test_the_sites_own_price_verdict_rides_along_stripped(self):
        listings, _ = self._first()
        assert [x.price_label for x in listings] == [
            "unknown", "fair-price", "fair-price", "top-price"]


class TestTheFrenchPage:
    """The same Golf on autoscout24.fr: another vocabulary, another space."""

    @staticmethod
    def _first():
        return parse_search(_saved("search_fr.json"), "fr")

    def test_the_page_reports_its_own_size(self):
        _, meta = self._first()
        assert (meta["results"], meta["pages"]) == (80, 4)

    def test_essence_is_gasolina_and_boite_automatique_is_automatica(self):
        listings, _ = self._first()
        first = listings[0]
        assert first.fuel_type == "Gasolina"
        assert first.transmission == "Automática"
        assert listings[1].transmission == "Manual"
        assert listings[1].fuel_type == "Diesel"

    def test_power_is_read_off_the_french_label_and_its_ch(self):
        first = self._first()[0][0]
        assert (first.power_kw, first.horsepower) == (81, 110)

    def test_the_narrow_space_in_the_mileage_is_a_thousands_separator(self):
        """A French card writes 65 000 km with U+202F; read naively that is 65."""
        listings, _ = self._first()
        assert [x.mileage_km for x in listings] == [65000, 132000, 126300, 133000]

    def test_the_registration_survives_the_french_label(self):
        listings, _ = self._first()
        assert [(x.year, x.registration_month) for x in listings][:2] == [
            (2018, "04/2018"), (2018, "10/2018")]

    def test_the_code_postal_becomes_a_region(self):
        listings, _ = self._first()
        assert [x.region for x in listings] == [
            "Provence-Alpes-Côte d'Azur", "Occitanie", "Bourgogne-Franche-Comté",
            "Grand Est"]

    def test_the_card_links_to_the_french_site(self):
        first = self._first()[0][0]
        assert first.url.startswith("https://www.autoscout24.fr/offres/")
        assert first.image_url.endswith("/720x540.webp")


class TestTheItalianPage:
    """autoscout24.it, where the province hides in the city string."""

    @staticmethod
    def _first():
        return parse_search(_saved("search_it.json"), "it")

    def test_the_page_reports_its_own_size(self):
        _, meta = self._first()
        assert (meta["results"], meta["pages"]) == (294, 15)

    def test_benzina_and_the_three_italian_gearboxes(self):
        listings, _ = self._first()
        assert [x.fuel_type for x in listings] == ["Diesel", "Gasolina", "Diesel", "Diesel"]
        assert [x.transmission for x in listings] == [
            "Automática", "Manual", "Manual", "Automática"]

    def test_power_is_read_off_the_italian_label_and_its_cv(self):
        listings, _ = self._first()
        assert (listings[0].power_kw, listings[0].horsepower) == (110, 150)
        assert (listings[1].power_kw, listings[1].horsepower) == (85, 116)

    def test_the_italian_thousands_dot_and_the_italian_date_label(self):
        listings, _ = self._first()
        assert [x.mileage_km for x in listings] == [81000, 86646, 145548, 120000]
        assert (listings[0].year, listings[0].registration_month) == (2018, "03/2018")

    def test_the_province_in_the_city_string_becomes_a_regione(self):
        listings, _ = self._first()
        assert [x.region for x in listings] == ["Emilia-Romagna", "Toscana", "Lazio", "Lazio"]

    def test_the_card_links_to_the_italian_site(self):
        first = self._first()[0][0]
        assert first.url.startswith("https://www.autoscout24.it/annunci/")


class TestTheMakePage:
    """A make page is twenty cars plus the site's own model vocabulary."""

    def test_the_taxonomy_comes_back_flattened(self):
        listings, meta = parse_search(_saved("make_de.json"), "de")
        assert meta["results"] == 146807
        models = meta["make_models"]
        assert {"label": "Golf", "value": 2084, "makeId": 74} in models
        assert {"label": "Polo", "value": 2090, "makeId": 74} in models
        assert all(set(m) == {"label", "value", "makeId"} for m in models)
        assert len({m["label"] for m in models}) == len(models)

    def test_the_cars_on_it_are_read_like_any_other_page(self):
        listings, _ = parse_search(_saved("make_de.json"), "de")
        assert [x.model for x in listings][:3] == ["T-Roc", "Passat Variant", "Polo"]
        assert listings[0].offer_type == "A"


class TestTheVocabularies:
    """Every label the three sites actually print, mapped onto the PT corpus."""

    @staticmethod
    def _fuel(raw, tld):
        (listing,), _ = parse_search(_page([_card(vehicle={"fuel": raw})]), tld)
        return listing.fuel_type

    @staticmethod
    def _gearbox(raw, tld):
        (listing,), _ = parse_search(_page([_card(vehicle={"transmission": raw})]), tld)
        return listing.transmission

    @pytest.mark.parametrize("tld,raw,expected", [
        ("de", "Diesel", "Diesel"), ("de", "Benzin", "Gasolina"),
        ("de", "Elektro", "Eléctrico"), ("de", "Elektro/Benzin", "Híbrido (Gasolina)"),
        ("de", "Elektro/Diesel", "Híbrido (Diesel)"), ("de", "Autogas (LPG)", "GPL"),
        ("de", "Erdgas (CNG)", "GNC"), ("de", "Wasserstoff", "Hidrogénio"),
        ("de", "Ethanol", "Gasolina"), ("de", "Sonstige", None),
        ("fr", "Diesel", "Diesel"), ("fr", "Essence", "Gasolina"),
        ("fr", "Electrique", "Eléctrico"), ("fr", "Électrique/Essence", "Híbrido (Gasolina)"),
        ("fr", "Électrique/Diesel", "Híbrido (Diesel)"), ("fr", "GPL", "GPL"),
        ("fr", "CNG", "GNC"), ("fr", "Hydrogène", "Hidrogénio"),
        ("fr", "Ethanol", "Gasolina"), ("fr", "Autres", None),
        ("it", "Diesel", "Diesel"), ("it", "Benzina", "Gasolina"),
        ("it", "Elettrica", "Eléctrico"), ("it", "Elettrica/Benzina", "Híbrido (Gasolina)"),
        ("it", "Elettrica/Diesel", "Híbrido (Diesel)"), ("it", "GPL", "GPL"),
        ("it", "Metano", "GNC"), ("it", "Idrogeno", "Hidrogénio"),
        ("it", "Etanolo", "Gasolina"), ("it", "Altro", None),
    ])
    def test_every_live_fuel_label(self, tld, raw, expected):
        assert self._fuel(raw, tld) == expected

    @pytest.mark.parametrize("tld,raw,expected", [
        ("de", "Schaltgetriebe", "Manual"), ("de", "Automatik", "Automática"),
        ("de", "Halbautomatik", "Automática"),
        ("fr", "Boîte manuelle", "Manual"), ("fr", "Boîte automatique", "Automática"),
        ("fr", "Semi-automatique", "Automática"),
        ("it", "Manuale", "Manual"), ("it", "Automatico", "Automática"),
        ("it", "Semiautomatico", "Automática"),
    ])
    def test_every_live_gearbox_label(self, tld, raw, expected):
        assert self._gearbox(raw, tld) == expected

    def test_a_label_nobody_has_seen_survives_as_itself(self):
        assert self._fuel("Kernfusion", "de") == "Kernfusion"
        assert self._gearbox("Direttissimo", "it") == "Direttissimo"

    def test_the_seller_kind_is_canonical_or_absent(self):
        def seller(raw):
            (listing,), _ = parse_search(_page([_card(seller={"type": raw})]))
            return listing.seller_type

        assert seller("Dealer") == "Profissional"
        assert seller("PrivateSeller") == "Particular"
        assert seller("") is None
        assert seller("Something") is None


class TestThreeSites:

    def test_each_tld_has_its_own_origin(self):
        assert base_url() == "https://www.autoscout24.de"
        assert base_url("de") == "https://www.autoscout24.de"
        assert base_url("fr") == "https://www.autoscout24.fr"
        assert base_url("it") == "https://www.autoscout24.it"

    def test_a_market_we_do_not_read_is_an_error_not_a_default(self):
        with pytest.raises(ValueError):
            base_url("es")
        with pytest.raises(ValueError):
            parse_search(_page([_card()]), "es")

    def test_the_language_header_follows_the_site(self):
        seen = []

        def handler(request):
            seen.append((str(request.url), request.headers.get("Accept-Language")))
            return httpx.Response(200, text=_page([_card()]))

        for tld, prefix in (("de", "de-DE"), ("fr", "fr-FR"), ("it", "it-IT")):
            config = AutoScoutConfig(tld=tld, budget=1, delay_min=0, delay_max=0)
            with AutoScoutClient(config=config) as client:
                client._client._transport = httpx.MockTransport(handler)
                client.fetch("/lst/volkswagen/golf?x=1")
        assert [lang.split(",")[0] for _url, lang in seen] == ["de-DE", "fr-FR", "it-IT"]
        assert [url.split("/lst")[0] for url, _lang in seen] == [
            "https://www.autoscout24.de", "https://www.autoscout24.fr",
            "https://www.autoscout24.it"]


class TestTheQueriesTheCountryCrawlMakes:

    def test_newest_first_is_still_a_path_robots_leaves_open(self):
        path = search_path("volkswagen", "golf", year=2018, page=2, country="F",
                           sort="age", desc=True, ustate="U")
        assert path.startswith("/lst/volkswagen/golf?")
        assert "sort=age" in path and "desc=1" in path
        assert "ustate=U" in path and "cy=F" in path
        assert "fregfrom=2018" in path and "fregto=2018" in path and "page=2" in path
        assert robots_allows(path)

    def test_left_alone_the_query_is_the_one_the_import_benchmark_always_sent(self):
        assert search_path("bmw", "320", year=2016) == (
            "/lst/bmw/320?atype=C&cy=D&damaged_listing=exclude&powertype=kw"
            "&sort=standard&ustate=N%2CU&fregfrom=2016&fregto=2016")

    def test_a_make_page_is_a_path_robots_leaves_open(self):
        path = make_path("volkswagen")
        assert path.startswith("/lst/volkswagen?")
        assert robots_allows(path)

    def test_search_asks_for_the_configured_country_and_returns_meta(self):
        seen = []

        def handler(request):
            seen.append(str(request.url))
            return httpx.Response(200, text=_saved("search_it.json"))

        config = AutoScoutConfig(tld="it", country="I", budget=5, delay_min=0, delay_max=0)
        client = AutoScoutClient(config=config)
        client._client = httpx.Client(transport=httpx.MockTransport(handler))
        listings, meta = client.search("volkswagen", "golf", year=2018, ustate="U",
                                       sort="age", desc=True)
        assert len(listings) == 4 and meta["results"] == 294
        assert seen[0].startswith("https://www.autoscout24.it/lst/volkswagen/golf?")
        assert "cy=I" in seen[0] and "sort=age" in seen[0] and "desc=1" in seen[0]

    def test_make_page_reads_the_make_path_and_brings_back_the_models(self):
        seen = []

        def handler(request):
            seen.append(request.url.path)
            return httpx.Response(200, text=_saved("make_de.json"))

        client = AutoScoutClient(config=AutoScoutConfig(budget=5, delay_min=0, delay_max=0))
        client._client = httpx.Client(transport=httpx.MockTransport(handler))
        listings, meta = client.make_page("volkswagen")
        assert seen == ["/lst/volkswagen"]
        assert len(listings) == 4
        assert any(m["label"] == "Golf" for m in meta["make_models"])

    def test_a_search_that_read_nothing_says_so_with_an_empty_meta(self):
        client = AutoScoutClient(config=AutoScoutConfig(budget=0))
        client._client = httpx.Client(transport=httpx.MockTransport(
            lambda r: pytest.fail("a spent budget must not reach the site")))
        assert client.search("volkswagen", "golf") == ([], {})
        assert client.make_page("volkswagen") == ([], {})


def _next_data(payload: dict) -> str:
    import json as _json
    return ('<html><script id="__NEXT_DATA__" type="application/json">'
            + _json.dumps(payload) + "</script></html>")


class TestTheAdvertPage:
    """The advert is read on the two markets whose robots.txt leaves it open."""

    def test_germany_closes_its_advert_path_and_the_others_do_not(self):
        """``autoscout24.de`` disallows ``/angebote/`` outright; ``.fr`` closes
        only the malformed ``/offres/-`` and ``.it`` says nothing about
        ``/annunci/``. One shared list would get two of the three wrong."""
        assert not robots_allows("/angebote/vw-golf-abc", "de")
        assert robots_allows("/offres/vw-golf-abc", "fr")
        assert not robots_allows("/offres/-vw-golf", "fr")
        assert robots_allows("/annunci/vw-golf-abc", "it")

    def test_an_unnamed_market_gets_the_strictest_answer(self):
        assert not robots_allows("/angebote/x")

    def test_the_search_rules_still_apply_to_every_market(self):
        for tld in ("de", "fr", "it"):
            assert not robots_allows("/lst?make=bmw", tld)
            assert robots_allows("/lst/volkswagen/golf", tld)

    def test_the_advert_yields_what_no_card_carries(self):
        from src.parser.autoscout import parse_detail

        page = _next_data({"props": {"pageProps": {"listingDetails": {
            "description": "Scheckheft<br />gepflegt &#x27;24",
            "vehicle": {
                "rawData": {"bodyType": {"raw": "StationWagon", "formatted": "Kombi"},
                            "bodyColor": {"raw": "Black", "formatted": "Schwarz"}},
                "driveTrain": "Anteriore",
                "numberOfDoors": 5, "numberOfSeats": 5,
                "co2emissionInGramPerKmWithFallback": {"raw": 118},
                "motorTypeName": "2.0 TDI", "noOfPreviousOwners": 2,
            },
            "seller": {"isDealer": True, "companyName": "Naz Auto"},
            "location": {"zip": "25030", "city": "Barbariga"},
            "createdTimestampWithOffset": "2026-09-15T11:10:28.688Z",
        }}}})
        patch = parse_detail(page, "it")
        assert patch["body_type"] == "Carrinha" and patch["segment"] == "Carrinha"
        assert patch["color"] == "Preto", "colours arrive in Portuguese, not as raw codes"
        assert patch["drive_type"] == "Dianteira"
        assert (patch["doors"], patch["seats"]) == ("4-5", 5)
        assert patch["co2_g_km"] == 118
        assert patch["seller_type"] == "Profissional"
        assert patch["seller_displayed_as"] == "Naz Auto"
        assert (patch["zip_code"], patch["city"]) == ("25030", "Barbariga")
        assert patch["sub_model"] == "2.0 TDI"
        assert patch["description"] == "Scheckheft\ngepflegt '24", "entities are unescaped"
        assert patch["extras"]["previous_owners"] == 2
        assert patch["extras"]["posted_at"].startswith("2026-09-15")

    def test_a_page_without_a_listing_is_a_parse_error(self):
        from src.parser.autoscout import AutoScoutUnreadable, parse_detail

        with pytest.raises(AutoScoutUnreadable):
            parse_detail("<html>blocked</html>", "it")
        with pytest.raises(AutoScoutUnreadable):
            parse_detail(_next_data({"props": {"pageProps": {}}}), "it")

    def test_the_card_is_never_corrected_by_the_advert(self):
        """Card and advert come from one database, so where they overlap the
        card is not the one to doubt."""
        from src.parser.autoscout import CORRECTS

        assert CORRECTS == ()
