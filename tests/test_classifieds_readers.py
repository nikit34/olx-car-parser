"""The two new readers: what they take off a card, and what the advert adds.

Both are pinned against payloads shaped like the real ones rather than against
the live sites, so a failure here means the parser changed and not that Italy
had a slow morning. What the tests care about is the boundary the rest of the
project relies on: values arrive in this project's Portuguese vocabulary, the
robots limits are enforced in code, and an advert fills gaps without overwriting
what the card already got right.
"""

from __future__ import annotations

import json

import pytest

from src.parser import automobile_it as am
from src.parser import kleinanzeigen_de as ka


def _next_data(payload: dict) -> str:
    return ('<html><script id="__NEXT_DATA__" type="application/json">'
            + json.dumps(payload) + "</script></html>")


def _italian_search(**over) -> str:
    card = {
        "id": 180506605,
        "title": "Fiat Panda 1.2 EasyPower Lounge",
        "formattedPrice": "€ 4.950",
        "location": "Uboldo (VA)",
        "url": "/uboldo-fiat-panda/180506605",
        "rental": False,
        "sellerType": "DEALER",
        "detailsType": "USED",
        "totalNumberOfPictures": 16,
        "dealerName": "A.F. AUTO",
        "carfaxAllowed": True,
        "firstMediumSizeImage": "https://media-ys.automobile.it/x.jpeg",
        "details": {
            "registration": "Maggio 2018",
            "fuelEmissions": "GPL - Euro 6",
            "formattedKm": "290.000 km",
            "formattedPower": "69 CV (51 KW)",
            "shift": "Manuale",
            "formattedEngineCapacity": "1242 cc",
        },
    }
    card.update(over)
    return _next_data({"props": {"pageProps": {
        "apiResults": {"result": {
            "resultList": [card],
            "page": {"size": 20, "totalElements": 33416, "totalPages": 500, "number": 1},
        }},
        "brands": {"all": [{"slugLabel": "fiat"}, {"slugLabel": "alfa_romeo"}]},
    }}})


class TestAutomobileItCard:

    def test_the_card_arrives_in_portuguese(self):
        cards, meta = am.parse_search(_italian_search(), "fiat")
        assert meta["total"] == 33416 and meta["pages"] == 500
        card = cards[0]
        assert (card.brand, card.model) == ("Fiat", "Panda")
        assert card.price_eur == 4950.0
        assert card.fuel_type == "GPL"
        assert card.transmission == "Manual"
        assert card.seller_type == "Profissional"
        assert (card.year, card.registration_month) == (2018, "05/2018")
        assert (card.mileage_km, card.engine_cc) == (290000, 1242)
        assert (card.horsepower, card.power_kw) == (69, 51)
        assert (card.city, card.region) == ("Uboldo", "VA")
        assert card.country_code == "IT"
        assert card.url.startswith("https://www.automobile.it/")

    def test_a_site_only_field_rides_in_extras(self):
        card = am.parse_search(_italian_search(), "fiat")[0][0]
        row = card.as_row()
        assert json.loads(row["extras"])["dealer_name"] == "A.F. AUTO"
        assert "dealer_name" not in row

    def test_a_new_car_and_a_rental_are_not_this_corpus(self):
        assert am.parse_search(_italian_search(detailsType="NEW"), "fiat")[0] == []
        assert am.parse_search(_italian_search(rental=True), "fiat")[0] == []

    def test_a_page_that_is_not_a_result_list_is_a_refusal(self):
        with pytest.raises(am.AutomobileItBlocked):
            am.parse_search("<html>nothing here</html>")

    def test_robots_closes_the_json_the_page_is_built_on(self):
        assert am.robots_allows("/fiat/page-2")
        assert not am.robots_allows("/api/v1/ads")
        assert not am.robots_allows("/rest/search")

    def test_pagination_stays_in_the_path(self):
        assert am.search_path("fiat") == "/fiat"
        assert am.search_path("fiat", 3) == "/fiat/page-3"

    def test_the_make_list_comes_off_the_page(self):
        assert am.brand_slugs(_italian_search()) == ["fiat", "alfa_romeo"]


def _german_card(price="16.600 € VB", km="160.581 km", ez="EZ 03/2016") -> str:
    ld = json.dumps({"creditText": "Kleinanzeigen", "title": "VW Sharan 2.0 TDI",
                     "description": "gepflegt", "contentUrl": "https://img/x.jpg",
                     "representativeOfPage": False,
                     "@context": "https://schema.org", "@type": "ImageObject"})
    return (f'<article class="aditem" data-adid="3470578973">'
            f'<div data-href="/s-anzeige/vw-sharan/3470578973-216-3466">{ld}</div>'
            f'<div>20</div><div>13125 Pankow</div>'
            f'<div>{price}</div><div>{km}</div><div>{ez}</div>'
            f'</article>')


def _german_advert(**over) -> str:
    attrs = {
        "Aussenfarbe": "weiss", "Erstzulassungsjahr": "2018",
        "Erstzulassungsmonat": "10", "Leistung": "95", "Kraftstoffart": "diesel",
        "Fahrzeugtyp": "limousine", "Getriebe": "manuell", "Marke": "volkswagen",
        "Modell": "polo", "Kilometerstand": "150000", "Schadstoffklasse": "euro6",
    }
    attrs.update(over)
    body = ", ".join(f'"{k}":"{v}"' for k, v in attrs.items())
    return ('<html><script>var p = {' + body + '};</script>'
            '<p id="viewad-description-text">Scheckheft <br>gepflegt</p></html>')


class TestKleinanzeigenCard:

    def test_the_card_gives_price_mileage_and_registration(self):
        cards, meta = ka.parse_search(_german_card(), "volkswagen")
        assert meta["articles"] == 1
        card = cards[0]
        assert card.external_id == "3470578973"
        assert card.price_eur == 16600.0
        assert card.mileage_km == 160581
        assert (card.year, card.registration_month) == (2016, "03/2016")
        assert (card.zip_code, card.city) == ("13125", "Pankow")
        assert card.country_code == "DE"
        assert card.extras["negotiable"] is True

    def test_seller_type_is_left_unknown_rather_than_guessed(self):
        """robots closes the seller-type filter, so this source cannot know it,
        and a wrong label poisons the private-price comparison it exists for."""
        card = ka.parse_search(_german_card(), "volkswagen")[0][0]
        assert card.seller_type is None

    def test_a_page_with_no_cards_is_a_refusal_not_an_empty_make(self):
        with pytest.raises(ka.KleinanzeigenBlocked):
            ka.parse_search("<html>200 but empty</html>", "volkswagen")

    def test_robots_limits_are_enforced_in_code(self):
        assert ka.robots_allows("/s-autos/volkswagen/c216")
        assert ka.robots_allows("/s-autos/seite:5/c216")
        assert not ka.robots_allows("/s-autos/seite:6/c216")
        assert not ka.robots_allows("/s-autos/anbieter:privat/c216")
        with pytest.raises(ValueError):
            ka.search_path("volkswagen", 6)


class TestKleinanzeigenAdvert:

    def test_the_advert_supplies_what_the_card_has_no_room_for(self):
        patch = ka.parse_detail(_german_advert())
        assert patch["fuel_type"] == "Diesel"
        assert patch["transmission"] == "Manual"
        assert patch["body_type"] == "Sedan"
        assert patch["color"] == "Branco"
        assert patch["horsepower"] == 95
        assert patch["description"] == "Scheckheft gepflegt"

    def test_it_corrects_the_make_and_the_model(self):
        """The make path is a text search, so the card's make is only a word the
        advert mentioned; these two fields are the site's own."""
        patch = ka.parse_detail(_german_advert())
        assert (patch["brand"], patch["model"]) == ("Volkswagen", "Polo")
        assert ka.CORRECTS == ("brand", "model")

    def test_the_rounded_mileage_is_kept_apart_from_the_exact_one(self):
        """The advert rounds to a bracket and the card does not, so the bracket
        travels in extras and the merge never lets it win."""
        patch = ka.parse_detail(_german_advert())
        assert patch["extras"]["mileage_bucket_km"] == 150000
        assert "mileage_km" not in ka.CORRECTS

    def test_an_advert_with_no_attribute_block_is_a_refusal(self):
        with pytest.raises(ka.KleinanzeigenBlocked):
            ka.parse_detail("<html>blocked</html>")


class TestTheMerge:

    def test_an_advert_fills_gaps_without_overwriting_the_card(self):
        from src.parser.market_card import merge_patch

        row = ka.parse_search(_german_card(), "volkswagen")[0][0].as_row()
        assert row["mileage_km"] == 160581
        merge_patch(row, ka.parse_detail(_german_advert()), ka.CORRECTS)

        assert row["mileage_km"] == 160581, "the advert's bracket overwrote the card"
        assert row["brand"] == "Volkswagen" and row["model"] == "Polo"
        assert row["fuel_type"] == "Diesel"
        assert row["extras"]["negotiable"] is True
        assert row["extras"]["mileage_bucket_km"] == 150000
