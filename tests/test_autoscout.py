"""Reading AutoScout24: the fields that decide a number, and the ones that lie.

The German card is the only place the ISV inputs come from, and two of them are
free text the seller fills in. So this file pins what the parser must refuse as
much as what it must read: a CO2 nobody measured, a price without its VAT label,
a listing with no first registration.

It also pins the crawler's manners, because those are a promise made in the
module docstring and a promise nobody tests is a comment: the robots rules it
obeys, and that a 429 or 403 raises instead of retrying.
"""

from __future__ import annotations

import json

import httpx
import pytest

from src.parser.autoscout import (
    AutoScoutBlocked,
    AutoScoutClient,
    AutoScoutConfig,
    parse_search,
    robots_allows,
    search_path,
)


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
        assert meta == {"results": 120, "pages": 6}
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
