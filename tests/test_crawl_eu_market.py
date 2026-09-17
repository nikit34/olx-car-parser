"""The EU market crawler's bookkeeping, with AutoScout24 replaced by a fake client.

What is pinned here is the part that decides how the request budget is spent
and when a row is retired: the stalest-first queue and its freshness gate, the
"fully enumerated" test that gates deactivation, the state file that lets a
killed run resume, and that one blocked country does not take the others down.
Nothing in this file makes a request; the fake client only counts them.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from scripts import crawl_eu_market as crawl
from src.countries import country
from src.parser import autoscout
from src.parser.autoscout import AutoScoutBlocked, DeListing

NOW = datetime(2026, 9, 13, 12, 0, 0)


def _listing(ext: str, year: int = 2018) -> DeListing:
    """A French card, because France is a market whose adverts are ours to open."""
    return DeListing(external_id=ext, url=f"https://www.autoscout24.fr/offres/{ext}",
                     brand="Volkswagen", model="Golf", year=year, price_eur=15000.0)


def _de_listing(ext: str, year: int = 2018) -> DeListing:
    """A German card. Its advert is closed, so its body comes from the filter."""
    return DeListing(external_id=ext, url=f"https://www.autoscout24.de/angebote/{ext}",
                     brand="Volkswagen", model="Golf", year=year, price_eur=15000.0)


class FakeClient:
    """Answers make pages and searches from dicts; raises where told to."""

    def __init__(self, *, make_pages=None, searches=None, budget=1000, block_on_make=None,
                 adverts=None):
        self.make_pages = make_pages or {}
        self.searches = searches or {}
        self.block_on_make = block_on_make
        self.adverts = adverts or {}
        self.spent = 0
        self.config = SimpleNamespace(budget=budget)
        self.calls: list[tuple] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def advert(self, path):
        """None is an ordinary answer: a page that would not parse costs that
        car its extra fields, and a reader that cannot cope with None breaks."""
        self.spent += 1
        self.calls.append(("advert", path))
        return self.adverts.get(path)

    def make_page(self, make, *, page=1):
        self.spent += 1
        self.calls.append(("make", make))
        if make == self.block_on_make:
            raise AutoScoutBlocked(f"403 on /lst/{make}")
        return self.make_pages.get(make, ([], {}))

    def search(self, make, model, *, year=None, page=1, body=None, sort=None, desc=False,
               ustate="U"):
        """A body-filtered search is keyed by its filter; one nobody wrote down
        answers the way the site does, with a page that holds no cars."""
        self.spent += 1
        self.calls.append(("search", make, model, year, page, sort, desc, ustate, body))
        if body is not None:
            return self.searches.get((make, model, year, page, body),
                                     ([], {"results": 0, "pages": 1}))
        return self.searches.get((make, model, year, page), ([], {}))


def _cfg(**over) -> crawl.MarketConfig:
    base = dict(makes=["volkswagen"], years_back=2, pages_per_cell=2, min_model_results=40,
                daily_budget=100, cell_max_age_days=7, discovery_max_age_days=30,
                expire_after_days=21)
    base.update(over)
    return crawl.MarketConfig(**base)


def _state(tmp_path, code="DE", discovered=None, inventory=None) -> crawl.CrawlState:
    state = crawl.CrawlState(tmp_path / "state.json")
    entry = state.country(code)
    entry["discovered"].update(discovered or {})
    entry["inventory"].update(inventory or {})
    return state


def _golf_discovered(at=NOW):
    return {"volkswagen": {"at": crawl._iso(at), "label": "Volkswagen", "models": [
        {"label": "Golf", "slug": "golf", "model_id": 2084},
        {"label": "Polo", "slug": "polo", "model_id": 2090},
    ]}}


@pytest.fixture
def repo_spy(monkeypatch):
    """Replace the repository writes with recorders so no test here needs a database."""
    calls = {"upsert": [], "deactivate": [], "expire": []}

    def fake_upsert(session, listings):
        items = list(listings)
        calls["upsert"].append(items)
        return len(items), 0

    def fake_deactivate(session, source, cells, seen_ids, now=None):
        calls["deactivate"].append((source, list(cells), set(seen_ids)))
        return 1

    def fake_expire(session, source, max_age_days=21, now=None):
        calls["expire"].append((source, max_age_days))
        return 0

    monkeypatch.setattr(crawl, "upsert_import_listings", fake_upsert)
    monkeypatch.setattr(crawl, "deactivate_import_missing", fake_deactivate)
    monkeypatch.setattr(crawl, "expire_import_listings", fake_expire)
    monkeypatch.setattr(crawl, "cell_last_seen", lambda session, source: {})
    monkeypatch.setattr(crawl, "listings_with_body", lambda session, source, ids: set())
    return calls


class TestConfig:

    def test_the_shipped_yaml_has_three_markets_ordered_by_popularity(self):
        configs = crawl.load_config(crawl.DEFAULT_CONFIG)
        assert set(configs) == {"DE", "FR", "IT"}
        assert configs["DE"].makes[:4] == ["volkswagen", "mercedes-benz", "bmw", "audi"]
        assert configs["FR"].makes[:3] == ["renault", "peugeot", "citroen"]
        assert configs["IT"].makes[:2] == ["fiat", "volkswagen"]
        for cfg in configs.values():
            assert len(cfg.makes) >= 30 and len(set(cfg.makes)) == len(cfg.makes)
            assert (cfg.years_back, cfg.pages_per_cell, cfg.min_model_results) == (10, 2, 40)
            assert (cfg.daily_budget, cfg.cell_max_age_days) == (900, 14)
            assert cfg.enrich_share == 0.5
            assert (cfg.discovery_max_age_days, cfg.expire_after_days) == (30, 60)
            assert (cfg.delay_min, cfg.delay_max) == (3.0, 6.0)

    def test_a_country_block_overrides_the_defaults(self, tmp_path):
        path = tmp_path / "m.yaml"
        path.write_text("defaults:\n  daily_budget: 100\n  years_back: 5\n"
                        "countries:\n  DE:\n    makes: [bmw]\n    daily_budget: 20\n",
                        encoding="utf-8")
        cfg = crawl.load_config(path)["DE"]
        assert (cfg.makes, cfg.daily_budget, cfg.years_back) == (["bmw"], 20, 5)


class TestQueue:

    @staticmethod
    def _cells():
        return crawl.build_cells([("volkswagen", "Volkswagen",
                                   {"label": "Golf", "slug": "golf", "model_id": 1})],
                                 now_year=2020, years_back=3)

    def test_cells_cover_every_year_back_newest_first(self):
        cells = self._cells()
        assert [c.year for c in cells] == [2020, 2019, 2018, 2017]
        assert cells[0].key == ("Volkswagen", "Golf", 2020)
        assert (cells[0].make_slug, cells[0].model_slug) == ("volkswagen", "golf")

    def test_never_seen_first_then_oldest(self):
        cells = self._cells()
        seen = {
            ("Volkswagen", "Golf", 2020): NOW - timedelta(days=10),
            ("Volkswagen", "Golf", 2019): NOW - timedelta(days=40),
            ("Volkswagen", "Golf", 2017): NOW - timedelta(days=20),
        }
        ordered = crawl.order_cells(cells, seen, now=NOW, cell_max_age_days=7)
        assert [c.year for c in ordered] == [2018, 2019, 2017, 2020]

    def test_a_cell_refreshed_inside_the_window_is_skipped(self):
        cells = self._cells()
        seen = {
            ("Volkswagen", "Golf", 2020): NOW - timedelta(days=2),
            ("Volkswagen", "Golf", 2019): NOW - timedelta(days=6, hours=23),
            ("Volkswagen", "Golf", 2018): NOW - timedelta(days=7, hours=1),
        }
        ordered = crawl.order_cells(cells, seen, now=NOW, cell_max_age_days=7)
        assert [c.year for c in ordered] == [2017, 2018]

    def test_only_models_with_enough_inventory_are_kept(self, tmp_path):
        state = _state(tmp_path, discovered=_golf_discovered(), inventory={
            "volkswagen/golf": {"results": 120, "at": crawl._iso(NOW)},
            "volkswagen/polo": {"results": 12, "at": crawl._iso(NOW)},
        })
        kept = crawl.kept_models(_cfg(), state, "DE")
        assert [(make, brand, m["label"]) for make, brand, m in kept] == [
            ("volkswagen", "Volkswagen", "Golf")]


class TestState:

    def test_round_trip(self, tmp_path):
        state = _state(tmp_path, discovered=_golf_discovered(),
                       inventory={"volkswagen/golf": {"results": 120, "at": crawl._iso(NOW)}})
        state.save()
        again = crawl.CrawlState.load(tmp_path / "state.json")
        assert again.data == state.data
        assert again.country("DE")["discovered"]["volkswagen"]["models"][0]["slug"] == "golf"
        assert json.loads((tmp_path / "state.json").read_text(encoding="utf-8")) == state.data

    def test_a_missing_or_broken_file_starts_empty(self, tmp_path):
        assert crawl.CrawlState.load(tmp_path / "none.json").data == {}
        (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
        state = crawl.CrawlState.load(tmp_path / "bad.json")
        assert state.country("FR") == {"discovered": {}, "inventory": {}}


class TestDiscovery:

    def test_models_are_learned_from_the_make_page_and_others_dropped(self, tmp_path):
        client = FakeClient(make_pages={"volkswagen": ([_listing("a")], {"results": 1000,
            "pages": 20, "make_models": [
                {"label": "Golf", "value": 2084, "makeId": 74},
                {"label": "Sonstige", "value": 999, "makeId": 74},
                {"label": "T-Roc", "value": 3001, "makeId": 74}]})})
        state = _state(tmp_path)
        read, empty = crawl.discover(client, _cfg(), state, "DE", now=NOW, log=lambda *a, **k: 0)
        assert (read, empty) == (1, 0)
        entry = state.country("DE")["discovered"]["volkswagen"]
        assert entry["label"] == "Volkswagen" and entry["at"] == crawl._iso(NOW)
        assert entry["models"] == [{"label": "Golf", "slug": "golf", "model_id": 2084},
                                   {"label": "T-Roc", "slug": "t-roc", "model_id": 3001}]
        assert (tmp_path / "state.json").exists()

    def test_a_fresh_discovery_costs_no_request(self, tmp_path):
        client = FakeClient()
        state = _state(tmp_path, discovered=_golf_discovered(at=NOW - timedelta(days=5)))
        assert crawl.discover(client, _cfg(), state, "DE", now=NOW) == (0, 0)
        assert client.spent == 0

    def test_a_stale_discovery_is_read_again(self, tmp_path):
        client = FakeClient(make_pages={"volkswagen": ([], {"make_models": [
            {"label": "Golf", "value": 2084, "makeId": 74}]})})
        state = _state(tmp_path, discovered=_golf_discovered(at=NOW - timedelta(days=31)))
        assert crawl.discover(client, _cfg(), state, "DE", now=NOW) == (1, 0)
        assert client.spent == 1

    def test_a_page_that_did_not_parse_is_not_recorded(self, tmp_path):
        client = FakeClient(make_pages={"volkswagen": ([], {})})
        state = _state(tmp_path)
        crawl.discover(client, _cfg(), state, "DE", now=NOW, log=lambda *a, **k: 0)
        assert "volkswagen" not in state.country("DE")["discovered"]


class TestInventory:

    def test_probe_records_results_and_stores_the_page(self, tmp_path, repo_spy):
        client = FakeClient(searches={
            ("volkswagen", "golf", None, 1): ([_listing("g1"), _listing("g2")],
                                              {"results": 120, "pages": 6}),
            ("volkswagen", "polo", None, 1): ([_listing("p1")], {"results": 12, "pages": 1}),
        })
        state = _state(tmp_path, "FR", discovered=_golf_discovered())
        result = crawl.CountryResult(code="FR", source="as24_fr")
        crawl.probe_inventory(client, _cfg(), state, country("FR"), object(), now=NOW,
                              result=result)
        inv = state.country("FR")["inventory"]
        assert inv["volkswagen/golf"]["results"] == 120
        assert inv["volkswagen/polo"]["results"] == 12
        assert result.models_probed == 2 and result.inserted == 3
        stored = repo_spy["upsert"][0]
        assert {(r["source"], r["brand"], r["model"]) for r in stored} == {
            ("as24_fr", "Volkswagen", "Golf")}
        assert client.calls[0][5:8] == ("age", True, "U")

    def test_the_probe_page_gets_its_adverts_too(self, tmp_path, repo_spy):
        """A model too small to be walked year by year is never harvested, so
        if the probe did not open its adverts its rows would stay card-shallow
        for as long as the model stays small."""
        client = FakeClient(
            searches={("volkswagen", "golf", None, 1): ([_listing("g1")],
                                                        {"results": 12, "pages": 1})},
            adverts={"/offres/g1": {"segment": "Carrinha", "co2_g_km": 118}})
        state = _state(tmp_path, "FR", discovered=_golf_discovered())
        result = crawl.CountryResult(code="FR", source="as24_fr")
        crawl.probe_inventory(client, _cfg(), state, country("FR"), object(), now=NOW,
                              result=result)
        row = repo_spy["upsert"][0][0]
        assert row["segment"] == "Carrinha" and row["co2_g_km"] == 118
        assert result.adverts_read == 1

    def test_a_fresh_probe_is_skipped(self, tmp_path, repo_spy):
        client = FakeClient()
        state = _state(tmp_path, "FR", discovered=_golf_discovered(), inventory={
            "volkswagen/golf": {"results": 120, "at": crawl._iso(NOW - timedelta(days=3))},
            "volkswagen/polo": {"results": 12, "at": crawl._iso(NOW - timedelta(days=3))},
        })
        result = crawl.CountryResult(code="FR", source="as24_fr")
        crawl.probe_inventory(client, _cfg(), state, country("FR"), object(), now=NOW,
                              result=result)
        assert client.spent == 0 and result.models_probed == 0


class TestHarvest:

    @staticmethod
    def _cell(year=2018):
        return crawl.Cell("volkswagen", "golf", "Volkswagen", "Golf", year)

    def test_a_fully_enumerated_cell_retires_what_it_no_longer_lists(self, tmp_path, repo_spy):
        client = FakeClient(searches={
            ("volkswagen", "golf", 2018, 1): ([_listing("a"), _listing("b")],
                                              {"results": 2, "pages": 1}),
        })
        result = crawl.CountryResult(code="FR", source="as24_fr")
        crawl.harvest(client, _cfg(), [self._cell()], _state(tmp_path, "FR"), country("FR"),
                      object(), pages_per_cell=2, result=result)
        assert client.spent == 3, "one search plus one advert per car"
        assert repo_spy["deactivate"] == [("as24_fr", [("Volkswagen", "Golf", 2018)], {"a", "b"})]
        assert result.deactivated == 1 and result.cells_read == 1
        stored = repo_spy["upsert"][0]
        assert {(r["source"], r["brand"], r["model"]) for r in stored} == {
            ("as24_fr", "Volkswagen", "Golf")}

    def test_a_second_page_is_read_only_when_it_exists(self, tmp_path, repo_spy):
        client = FakeClient(searches={
            ("volkswagen", "golf", 2018, 1): ([_listing("a")], {"results": 25, "pages": 2}),
            ("volkswagen", "golf", 2018, 2): ([_listing("b")], {"results": 25, "pages": 2}),
            ("volkswagen", "golf", 2017, 1): ([_listing("c")], {"results": 1, "pages": 1}),
        })
        result = crawl.CountryResult(code="FR", source="as24_fr")
        crawl.harvest(client, _cfg(), [self._cell(2018), self._cell(2017)], _state(tmp_path, "FR"),
                      country("FR"), object(), pages_per_cell=2, result=result)
        pages = [(c[3], c[4]) for c in client.calls if c[0] == "search"]
        assert pages == [(2018, 1), (2018, 2), (2017, 1)]
        assert [d[1][0][2] for d in repo_spy["deactivate"]] == [2018, 2017]
        assert repo_spy["deactivate"][0][2] == {"a", "b"}

    def test_a_cell_deeper_than_the_pages_read_is_not_retired(self, tmp_path, repo_spy):
        client = FakeClient(searches={
            ("volkswagen", "golf", 2018, 1): ([_listing("a")], {"results": 90, "pages": 5}),
            ("volkswagen", "golf", 2018, 2): ([_listing("b")], {"results": 90, "pages": 5}),
        })
        result = crawl.CountryResult(code="FR", source="as24_fr")
        crawl.harvest(client, _cfg(), [self._cell()], _state(tmp_path, "FR"), country("FR"),
                      object(), pages_per_cell=2, result=result)
        assert client.spent == 4, "two searches plus one advert per car"
        assert repo_spy["deactivate"] == []
        assert len(repo_spy["upsert"][0]) == 2

    def test_the_advert_fills_what_the_card_had_no_room_for(self, tmp_path, repo_spy):
        """The body type is the point: no AutoScout24 search card carries one,
        which is why ``segment`` was null for every foreign row."""
        client = FakeClient(
            searches={("volkswagen", "golf", 2018, 1): ([_listing("a")],
                                                        {"results": 1, "pages": 1})},
            adverts={"/offres/a": {"body_type": "Carrinha", "segment": "Carrinha",
                                     "color": "Preto", "co2_g_km": 118,
                                     "description": "Scheckheft"}})
        result = crawl.CountryResult(code="FR", source="as24_fr")
        crawl.harvest(client, _cfg(), [self._cell()], _state(tmp_path, "FR"), country("FR"),
                      object(), pages_per_cell=2, result=result)
        row = repo_spy["upsert"][0][0]
        assert row["segment"] == "Carrinha" and row["color"] == "Preto"
        assert row["co2_g_km"] == 118 and row["description"] == "Scheckheft"
        assert result.adverts_read == 1 and result.adverts_missed == 0

    def test_an_advert_we_may_not_read_still_stores_the_car(self, tmp_path, repo_spy):
        """Germany's robots.txt disallows ``/offres/``, so the client answers
        None there every time. A car is never dropped for want of its advert."""
        client = FakeClient(searches={
            ("volkswagen", "golf", 2018, 1): ([_listing("a")], {"results": 1, "pages": 1})})
        result = crawl.CountryResult(code="FR", source="as24_fr")
        crawl.harvest(client, _cfg(), [self._cell()], _state(tmp_path, "FR"), country("FR"),
                      object(), pages_per_cell=2, result=result)
        assert len(repo_spy["upsert"][0]) == 1
        assert result.adverts_read == 0 and result.adverts_missed == 1

    def test_the_advert_never_overwrites_the_card(self, tmp_path, repo_spy):
        """AutoScout24 names no ``CORRECTS``: card and advert come from one
        database, so where they overlap the card is not the one to doubt."""
        client = FakeClient(
            searches={("volkswagen", "golf", 2018, 1): ([_listing("a")],
                                                        {"results": 1, "pages": 1})},
            adverts={"/offres/a": {"year": 1999, "brand": "Wrong",
                                     "mileage_km": 88000}})
        result = crawl.CountryResult(code="FR", source="as24_fr")
        crawl.harvest(client, _cfg(), [self._cell()], _state(tmp_path, "FR"), country("FR"),
                      object(), pages_per_cell=2, result=result)
        row = repo_spy["upsert"][0][0]
        assert row["brand"] == "Volkswagen" and row["year"] == 2018
        assert row["mileage_km"] == 88000, "a gap the card left is still filled"

    def test_cards_are_still_stored_when_adverts_are_off(self, tmp_path, repo_spy):
        client = FakeClient(searches={
            ("volkswagen", "golf", 2018, 1): ([_listing("a")], {"results": 1, "pages": 1})})
        result = crawl.CountryResult(code="FR", source="as24_fr")
        crawl.harvest(client, _cfg(), [self._cell()], _state(tmp_path, "FR"), country("FR"),
                      object(), pages_per_cell=2, result=result, adverts=False)
        assert client.spent == 1 and len(repo_spy["upsert"][0]) == 1
        assert result.adverts_read == 0 and result.adverts_missed == 0

    def test_a_page_that_did_not_parse_retires_nothing(self, tmp_path, repo_spy):
        client = FakeClient(searches={("volkswagen", "golf", 2018, 1): ([], {})})
        result = crawl.CountryResult(code="FR", source="as24_fr")
        crawl.harvest(client, _cfg(), [self._cell()], _state(tmp_path, "FR"), country("FR"),
                      object(), pages_per_cell=2, result=result)
        assert repo_spy["deactivate"] == [] and repo_spy["upsert"] == []
        assert result.cells_failed == 1 and result.cells_read == 0

    def test_a_car_whose_advert_was_read_is_not_opened_again(self, tmp_path, repo_spy,
                                                             monkeypatch):
        """A body already in the database says the advert has been read, and an
        advert does not change while the car is for sale. On a mature corpus
        half a pass is cars it has already seen."""
        monkeypatch.setattr(crawl, "listings_with_body",
                            lambda session, source, ids: {"a"})
        client = FakeClient(
            searches={("volkswagen", "golf", 2018, 1): ([_listing("a"), _listing("b")],
                                                        {"results": 2, "pages": 1})},
            adverts={"/offres/b": {"body_type": "Carrinha"}})
        result = crawl.CountryResult(code="FR", source="as24_fr")
        crawl.harvest(client, _cfg(), [self._cell()], _state(tmp_path, "FR"), country("FR"),
                      object(), pages_per_cell=2, result=result)
        assert [c for c in client.calls if c[0] == "advert"] == [("advert", "/offres/b")]
        assert result.adverts_read == 1 and result.adverts_missed == 0

    def test_a_card_the_page_repeats_is_deepened_once(self, tmp_path, repo_spy):
        """AutoScout24 prints a promoted listing twice on the same page. Both
        copies are the same car, so the advert is bought once and the second
        copy is given what the first learned."""
        client = FakeClient(
            searches={("volkswagen", "golf", 2018, 1): ([_listing("a"), _listing("a")],
                                                        {"results": 2, "pages": 1})},
            adverts={"/offres/a": {"body_type": "Carrinha", "color": "Preto"}})
        result = crawl.CountryResult(code="FR", source="as24_fr")
        crawl.harvest(client, _cfg(), [self._cell()], _state(tmp_path, "FR"), country("FR"),
                      object(), pages_per_cell=2, result=result)
        assert [c for c in client.calls if c[0] == "advert"] == [("advert", "/offres/a")]
        assert [row["color"] for row in repo_spy["upsert"][0]] == ["Preto", "Preto"]

    def test_the_deepening_may_not_eat_the_whole_budget(self, tmp_path, repo_spy):
        """A cell of twenty cars costs twenty-one requests without a ceiling,
        and then the cells stop coming round before their rows expire."""
        rows = [_listing(f"c{i}") for i in range(5)]
        client = FakeClient(
            searches={("volkswagen", "golf", 2018, 1): (rows, {"results": 5, "pages": 1})},
            adverts={f"/offres/c{i}": {"body_type": "Carrinha"} for i in range(5)})
        result = crawl.CountryResult(code="FR", source="as24_fr")
        crawl.harvest(client, _cfg(), [self._cell()], _state(tmp_path, "FR"), country("FR"),
                      object(), pages_per_cell=2, result=result, ceiling=2)
        assert (result.adverts_read, result.adverts_missed) == (2, 3)
        assert client.spent == 3, "the search, and the two adverts the ceiling left"
        assert len(repo_spy["upsert"][0]) == 5, "every car is stored either way"

    def test_the_budget_stops_the_harvest(self, tmp_path, repo_spy):
        client = FakeClient(budget=1, searches={
            ("volkswagen", "golf", 2018, 1): ([_listing("a")], {"results": 1, "pages": 1}),
        })
        result = crawl.CountryResult(code="FR", source="as24_fr")
        crawl.harvest(client, _cfg(), [self._cell(2018), self._cell(2017)], _state(tmp_path, "FR"),
                      country("FR"), object(), pages_per_cell=2, result=result)
        assert client.spent == 1 and result.cells_read == 1


class TestTheMarketWithNoAdvert:
    """Germany, where ``robots.txt`` disallows the advert and the body type has
    to come off a page we are allowed to ask for.

    A search filtered to one body returns only cars of that body, so the filter
    labels every car it returns. Nothing here is inferred: a car no filter
    named keeps no body at all.
    """

    @staticmethod
    def _cell(year=2018):
        return crawl.Cell("volkswagen", "golf", "Volkswagen", "Golf", year)

    def test_the_body_comes_off_the_filter_and_the_advert_is_never_asked_for(
            self, tmp_path, repo_spy):
        client = FakeClient(searches={
            ("volkswagen", "golf", 2018, 1): (
                [_de_listing("a"), _de_listing("b")],
                {"results": 2, "pages": 1, "body_types": ["bt_kombi", "bt_limousine"]}),
            ("volkswagen", "golf", 2018, 1, "bt_kombi"): ([_de_listing("a")],
                                                          {"results": 1, "pages": 1}),
            ("volkswagen", "golf", 2018, 1, "bt_limousine"): ([_de_listing("b")],
                                                              {"results": 1, "pages": 1}),
        })
        result = crawl.CountryResult(code="DE", source="as24_de")
        crawl.harvest(client, _cfg(), [self._cell()], _state(tmp_path), country("DE"),
                      object(), pages_per_cell=2, result=result)
        stored = {row["external_id"]: row.get("body_type") for row in repo_spy["upsert"][0]}
        assert stored == {"a": "Carrinha", "b": "Sedan"}
        assert not [c for c in client.calls if c[0] == "advert"]
        assert (result.bodies_labelled, result.adverts_read) == (2, 0)

    def test_the_walk_stops_once_every_car_is_accounted_for(self, tmp_path, repo_spy):
        client = FakeClient(searches={
            ("volkswagen", "golf", 2018, 1): (
                [_de_listing("a")],
                {"results": 1, "pages": 1,
                 "body_types": ["bt_kombi", "bt_limousine", "bt_cabrio"]}),
            ("volkswagen", "golf", 2018, 1, "bt_kombi"): ([_de_listing("a")],
                                                          {"results": 1, "pages": 1}),
        })
        result = crawl.CountryResult(code="DE", source="as24_de")
        crawl.harvest(client, _cfg(), [self._cell()], _state(tmp_path), country("DE"),
                      object(), pages_per_cell=2, result=result)
        assert [c[8] for c in client.calls if c[0] == "search" and c[8]] == ["bt_kombi"]
        assert client.spent == 2, "the cell, and the one filter that answered for it"

    def test_a_car_no_filter_returned_keeps_no_body(self, tmp_path, repo_spy):
        client = FakeClient(searches={
            ("volkswagen", "golf", 2018, 1): ([_de_listing("a")],
                                              {"results": 1, "pages": 1,
                                               "body_types": ["bt_kombi"]}),
        })
        result = crawl.CountryResult(code="DE", source="as24_de")
        crawl.harvest(client, _cfg(), [self._cell()], _state(tmp_path), country("DE"),
                      object(), pages_per_cell=2, result=result)
        row = repo_spy["upsert"][0][0]
        assert row.get("body_type") is None and result.bodies_labelled == 0

    def test_a_page_that_named_no_bodies_leaves_the_whole_vocabulary(self):
        """A model with one body ships no facet list, and that is the model
        whose body the corpus most wants; the walk asks the vocabulary."""
        filters = autoscout.market("de").body_filters
        assert crawl.body_order([], filters) == list(filters)
        assert crawl.body_order(["bt_kombi", "bt_nonsense"], filters) == ["bt_kombi"]
        assert crawl.body_order(None, autoscout.market("fr").body_filters) == []


class TestHarvestAgainstTheDatabase:

    def test_the_missing_row_is_retired_and_the_seen_one_kept(self, tmp_path, db_session,
                                                              monkeypatch):
        from src.storage.repository import get_country_listings_df, upsert_import_listings

        gone = _listing("gone")
        stays = _listing("stays")
        crawl.stamp([gone, stays], "as24_de", "Volkswagen", "Golf")
        upsert_import_listings(db_session, [gone, stays])
        client = FakeClient(searches={
            ("volkswagen", "golf", 2018, 1): ([_listing("stays"), _listing("new")],
                                              {"results": 2, "pages": 1}),
        })
        result = crawl.CountryResult(code="DE", source="as24_de")
        cell = crawl.Cell("volkswagen", "golf", "Volkswagen", "Golf", 2018)
        crawl.harvest(client, _cfg(), [cell], _state(tmp_path), country("DE"), db_session,
                      pages_per_cell=2, result=result)
        df = get_country_listings_df(db_session, "DE").set_index("external_id")
        assert {ext: bool(a) for ext, a in df["is_active"].items()} == {
            "gone": False, "stays": True, "new": True}
        assert result.deactivated == 1 and result.inserted == 1 and result.updated == 1

    def test_the_queue_reads_last_seen_from_the_database(self, db_session):
        from src.models.listing import Listing
        from src.storage.repository import upsert_import_listings

        old = _listing("old", year=2017)
        fresh = _listing("fresh", year=2018)
        crawl.stamp([old, fresh], "as24_de", "Volkswagen", "Golf")
        upsert_import_listings(db_session, [old, fresh])
        now = crawl._utcnow()
        (db_session.query(Listing).filter(Listing.external_id == "old")
         .update({Listing.last_seen_at: now - timedelta(days=30)},
                 synchronize_session="fetch"))
        db_session.commit()
        seen = crawl.cell_last_seen(db_session, "as24_de")
        assert set(seen) == {("Volkswagen", "Golf", 2017), ("Volkswagen", "Golf", 2018)}
        assert crawl.cell_last_seen(db_session, "as24_fr") == {}
        cells = crawl.build_cells([("volkswagen", "Volkswagen",
                                    {"label": "Golf", "slug": "golf", "model_id": 1})],
                                  now_year=2019, years_back=2)
        ordered = crawl.order_cells(cells, seen, now=now, cell_max_age_days=7)
        assert [c.year for c in ordered] == [2019, 2017]


class TestRun:

    def test_a_blocked_country_does_not_stop_the_others(self, tmp_path, repo_spy):
        state = _state(tmp_path)
        golf_page = ([_listing("a")], {"make_models": [{"label": "Golf", "value": 1,
                                                        "makeId": 74}]})
        year = crawl._utcnow().year
        fr_searches = {
            ("volkswagen", "golf", None, 1): ([_listing("f1")], {"results": 50, "pages": 3}),
        }
        for y in (year, year - 1, year - 2):
            fr_searches[("volkswagen", "golf", y, 1)] = (
                [_listing(f"f{y}", year=y)], {"results": 1, "pages": 1})
        clients = {
            "DE": FakeClient(block_on_make="volkswagen"),
            "FR": FakeClient(make_pages={"volkswagen": golf_page}, searches=fr_searches),
        }
        results = crawl.run([country("DE"), country("FR")], {"DE": _cfg(), "FR": _cfg()}, state,
                            client_factory=lambda cty, cfg: clients[cty.code],
                            session_factory=lambda: SimpleNamespace(close=lambda: None),
                            log=lambda *a, **k: 0)
        by_code = {r.code: r for r in results}
        assert by_code["DE"].blocked is True and by_code["DE"].makes_read == 0
        assert by_code["FR"].blocked is False
        assert by_code["FR"].makes_read == 1 and by_code["FR"].models_probed == 1
        assert by_code["FR"].models_kept == 1 and by_code["FR"].cells_read == 3
        assert state.country("FR")["inventory"]["volkswagen/golf"]["results"] == 50
        assert "volkswagen" not in state.country("DE")["discovered"]
        assert {c[0] for c in repo_spy["expire"]} == {"as24_de", "as24_fr"}

    def test_a_crash_in_one_country_is_reported_not_raised(self, tmp_path, repo_spy):
        class Exploding(FakeClient):
            def make_page(self, make, *, page=1):
                raise RuntimeError("boom")

        results = crawl.run([country("IT")], {"IT": _cfg()}, _state(tmp_path, "IT"),
                            client_factory=lambda cty, cfg: Exploding(),
                            session_factory=lambda: SimpleNamespace(close=lambda: None),
                            log=lambda *a, **k: 0)
        assert results[0].error == "RuntimeError: boom" and results[0].blocked is False
        assert crawl.summary(results[0]).startswith("[as24_it] failed:")

    def test_the_summary_line_reads_like_the_other_crawler(self):
        result = crawl.CountryResult(code="DE", source="as24_de", makes_read=3, models_probed=40,
                                     models_kept=12, cells_pending=180, cells_read=90,
                                     inserted=900, updated=300, deactivated=7, expired=2,
                                     requests=133, seconds=800.4)
        line = crawl.summary(result)
        assert line.startswith("[as24_de] 3 makes read")
        assert "90/180 cells read" in line and "133 requests in 800s" in line
        assert "blocked" not in line
        result.blocked = True
        assert crawl.summary(result).endswith("blocked")


class TestDryRun:

    def test_prints_the_queue_without_a_client(self, fresh_schema, tmp_path, monkeypatch, capsys):
        from sqlalchemy.orm import Session
        from src.models.listing import Listing
        from src.storage.database import init_db
        from src.storage.repository import upsert_import_listings

        year = crawl._utcnow().year
        engine = init_db(fresh_schema)
        with Session(engine) as session:
            rows = [_listing("fresh", year=year - 1), _listing("old", year=year - 3)]
            crawl.stamp(rows, "as24_de", "Volkswagen", "Golf")
            upsert_import_listings(session, rows)
            (session.query(Listing).filter(Listing.external_id == "old")
             .update({Listing.last_seen_at: crawl._utcnow() - timedelta(days=30)},
                     synchronize_session="fetch"))
            session.commit()
        state = _state(tmp_path, discovered=_golf_discovered(), inventory={
            "volkswagen/golf": {"results": 120, "at": crawl._iso(crawl._utcnow())}})
        state.save()

        def no_client(*a, **k):
            raise AssertionError("dry run must not build a client")

        monkeypatch.setattr(crawl, "_client_factory", no_client)
        config = tmp_path / "m.yaml"
        config.write_text("defaults:\n  years_back: 3\n  daily_budget: 50\n"
                          "countries:\n  DE:\n    makes: [volkswagen]\n", encoding="utf-8")
        rc = crawl.main(["--dry-run", "--db", fresh_schema, "--state", str(state.path),
                         "--config", str(config), "--country", "de"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "[as24_de] discovery: 0/1 makes to read; inventory: 1 models to probe; " in out
        assert "harvest: 1 models kept, 3 cells stale enough to fetch; budget 50" in out
        lines = [line for line in out.splitlines() if "→ /lst/volkswagen/golf" in line]
        assert [line.split()[2] for line in lines] == [str(year), str(year - 2), str(year - 3)]
        assert all("(last seen never)" in line for line in lines[:2])
        assert "(last seen never)" not in lines[2]


class TestNewCarsStayOut:
    """A new car's asking price is not a used-market price.

    ``autoscout24.de`` accepts ``ustate=U`` and then echoes back no ``ustate``
    at all, so the query cannot be trusted to have filtered anything; the offer
    type printed on the card is what decides.
    """

    class _Card:
        def __init__(self, external_id, offer_type):
            self.external_id = external_id
            self.offer_type = offer_type
            self.source = self.brand = self.model = None

    def test_the_offer_type_on_the_card_decides(self):
        cards = [self._Card("a", "U"), self._Card("b", "N"),
                 self._Card("c", "J"), self._Card("d", None)]
        kept = crawl.used_only(cards)
        assert [c.external_id for c in kept] == ["a", "c", "d"]

    def test_case_and_padding_do_not_smuggle_one_through(self):
        assert crawl.used_only([self._Card("a", " n ")]) == []
