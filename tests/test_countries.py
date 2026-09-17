"""The one table three subsystems agree on, so they cannot disagree quietly.

A country's ``source`` is written by the crawler and read back by the frame
builder, the price-model artifact paths and the blob builder. If any two of
them disagree about the string, nothing raises: the crawler fills a table
nobody queries and a country ships an empty page. So what is pinned here is the
exact strings — ``as24_de``, tld ``fr``, ``cy`` code ``I`` — and the two rules
that keep a mistake loud: an unknown code raises, an unknown source does not.
"""

from __future__ import annotations

import dataclasses

import pytest

from src.countries import (
    COUNTRIES,
    EU_COUNTRIES,
    Country,
    country,
    country_for_source,
    source_for,
    code_for_source,
)


class TestTheTable:

    def test_the_markets_this_project_reads(self):
        assert set(COUNTRIES) == {"PT", "DE", "FR", "IT"}
        assert all(code == code.upper() for code in COUNTRIES)
        assert all(code == COUNTRIES[code].code for code in COUNTRIES)

    def test_the_crawled_markets_are_the_three_foreign_ones(self):
        assert EU_COUNTRIES == ("DE", "FR", "IT")
        assert "PT" not in EU_COUNTRIES

    def test_every_source_string_is_its_own(self):
        sources = [c.source for c in COUNTRIES.values()]
        assert len(set(sources)) == len(sources)
        assert {code: COUNTRIES[code].source for code in EU_COUNTRIES} == {
            "DE": "as24_de", "FR": "as24_fr", "IT": "as24_it"}
        assert COUNTRIES["PT"].source == "olx"

    def test_each_market_carries_its_language_and_its_own_name(self):
        assert [(c.lang, c.name) for c in COUNTRIES.values()] == [
            ("pt", "Portugal"), ("de", "Deutschland"), ("fr", "France"),
            ("it", "Italia")]

    def test_the_autoscout_halves_of_a_request(self):
        assert [(COUNTRIES[c].as24_tld, COUNTRIES[c].as24_cy) for c in EU_COUNTRIES] == [
            ("de", "D"), ("fr", "F"), ("it", "I")]
        assert (COUNTRIES["PT"].as24_tld, COUNTRIES["PT"].as24_cy) == (None, None)

    def test_the_host_a_market_is_read_from(self):
        assert [c.market_host for c in COUNTRIES.values()] == [
            "olx.pt", "autoscout24.de", "autoscout24.fr", "autoscout24.it"]

    def test_a_country_cannot_be_edited_in_place(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            country("DE").source = "as24_xx"


class TestLookups:

    def test_the_code_is_case_insensitive(self):
        assert country("de") is country("DE") is COUNTRIES["DE"]
        assert country(" it ").code == "IT"

    def test_an_unknown_code_raises_rather_than_returning_nothing(self):
        with pytest.raises(ValueError) as exc:
            country("XX")
        assert "XX" in str(exc.value)
        with pytest.raises(ValueError):
            country("")

    def test_source_for_is_what_the_frame_builder_filters_on(self):
        assert source_for("DE") == "as24_de"
        assert source_for("fr") == "as24_fr"
        assert source_for("IT") == "as24_it"
        assert source_for("PT") == "olx"

    def test_a_source_maps_back_to_its_country(self):
        assert country_for_source("as24_it") is COUNTRIES["IT"]
        assert country_for_source("olx").code == "PT"

    def test_the_retired_benchmark_source_still_resolves_to_its_market(self):
        """``autoscout24`` is the weekly German benchmark, which predates
        per-country sources. Its rows are German cars and share one table with
        every other market now, so it has to resolve to Germany: a row the
        writer cannot place a market on is dropped, and dropping the benchmark
        would empty the import comparison."""
        assert country_for_source("autoscout24").code == "DE"
        assert code_for_source("autoscout24") == "DE"

    def test_a_source_nobody_registered_is_none_not_an_error(self):
        """Sources arrive from the database, where a value no reader writes any
        more is data, not a programming mistake. None is also a refusal: the
        writer drops such a row rather than guessing it into a corpus."""
        assert country_for_source("hood_de") is None
        assert code_for_source("hood_de") is None
        assert country_for_source("") is None
        assert country_for_source(None) is None


def test_the_dataclass_carries_every_field_the_callers_read():
    assert {f.name for f in dataclasses.fields(Country)} == {
        "code", "lang", "name", "source", "as24_tld", "as24_cy", "market_host"}
