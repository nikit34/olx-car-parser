"""Tests for car generation lookup logic (no network)."""

from unittest.mock import patch

from src.models.generations import (
    _parse_dbpedia_label,
    _fix_year_ranges,
    _BRAND_NORMALIZE,
    get_generation,
    _MODEL_ALIASES,
)


class TestParseDbpediaLabel:
    def test_parenthetical_chassis_code(self):
        assert _parse_dbpedia_label("BMW 3 Series (E90)", "BMW") == ("3 Series", "E90")

    def test_parenthetical_w_code(self):
        assert _parse_dbpedia_label("Mercedes-Benz C-Class (W205)", "Mercedes-Benz") == ("C-Class", "W205")

    def test_parenthetical_platform_code(self):
        assert _parse_dbpedia_label("Toyota Yaris (XP150)", "Toyota") == ("Yaris", "XP150")

    def test_mk_pattern(self):
        assert _parse_dbpedia_label("Volkswagen Golf Mk7", "Volkswagen") == ("Golf", "Mk7")

    def test_mk_with_space(self):
        assert _parse_dbpedia_label("Volkswagen Polo Mk 4", "Volkswagen") == ("Polo", "Mk4")

    def test_ordinal_generation(self):
        assert _parse_dbpedia_label("Ford Fiesta (fifth generation)", "Ford") == ("Fiesta", "V")

    def test_ordinal_eighth(self):
        assert _parse_dbpedia_label("Honda Civic (eighth generation)", "Honda") == ("Civic", "VIII")

    def test_no_generation_suffix(self):
        assert _parse_dbpedia_label("Renault Clio", "Renault") is None

    def test_no_generation_plain(self):
        assert _parse_dbpedia_label("BMW 1 Series", "BMW") is None

    def test_raw_brand_fallback(self):
        assert _parse_dbpedia_label("Volkswagen Group Golf Mk7", "Volkswagen Group") == ("Golf", "Mk7")


class TestFixYearRanges:
    def test_fixes_equal_years(self):
        data = {"VW": {"Golf": [
            {"name": "Mk1", "year_from": 1974, "year_to": 1974},
            {"name": "Mk2", "year_from": 1983, "year_to": 1992},
        ]}}
        _fix_year_ranges(data)
        assert data["VW"]["Golf"][0]["year_to"] == 1982  # next gen - 1

    def test_fixes_last_gen_to_2026(self):
        data = {"VW": {"Golf": [
            {"name": "Mk8", "year_from": 2019, "year_to": 2019},
        ]}}
        _fix_year_ranges(data)
        assert data["VW"]["Golf"][0]["year_to"] == 2026

    def test_no_change_if_valid(self):
        data = {"VW": {"Golf": [
            {"name": "Mk7", "year_from": 2012, "year_to": 2019},
        ]}}
        _fix_year_ranges(data)
        assert data["VW"]["Golf"][0]["year_to"] == 2019


class TestBrandNormalize:
    def test_volkswagen_group(self):
        assert _BRAND_NORMALIZE.get("Volkswagen Group") == "Volkswagen"

    def test_mercedes_benz_group(self):
        assert _BRAND_NORMALIZE.get("Mercedes-Benz Group") == "Mercedes-Benz"

    def test_passthrough(self):
        assert _BRAND_NORMALIZE.get("Toyota", "Toyota") == "Toyota"

    def test_ford_motor_company(self):
        assert _BRAND_NORMALIZE.get("Ford Motor Company") == "Ford"


class TestGetGeneration:
    def test_direct_lookup(self, generations_data):
        with patch("src.models.generations.load_generations", return_value=generations_data):
            assert get_generation("Volkswagen", "Golf", 2015) == "Mk7"

    def test_alias_lookup(self, generations_data):
        with patch("src.models.generations.load_generations", return_value=generations_data):
            assert get_generation("BMW", "320", 2015) == "F30"

    def test_mercedes_alias(self, generations_data):
        with patch("src.models.generations.load_generations", return_value=generations_data):
            assert get_generation("Mercedes-Benz", "E 220", 2017) == "W213"

    def test_no_year(self, generations_data):
        with patch("src.models.generations.load_generations", return_value=generations_data):
            assert get_generation("Volkswagen", "Golf", None) is None

    def test_unknown_brand(self, generations_data):
        with patch("src.models.generations.load_generations", return_value=generations_data):
            assert get_generation("Unknown", "Car", 2020) is None

    def test_year_out_of_range(self, generations_data):
        with patch("src.models.generations.load_generations", return_value=generations_data):
            assert get_generation("Volkswagen", "Golf", 1990) is None

    def test_boundary_year_overlap(self, generations_data):
        """2019 falls in both Mk7 (2012-2019) and Mk8 (2019-2026), first match wins."""
        with patch("src.models.generations.load_generations", return_value=generations_data):
            result = get_generation("Volkswagen", "Golf", 2019)
            assert result in ("Mk7", "Mk8")


class TestModelAliases:
    def test_bmw_series_aliases_exist(self):
        for model in ("116", "118", "120", "318", "320", "520", "530"):
            assert model in _MODEL_ALIASES["BMW"]

    def test_mercedes_class_aliases_exist(self):
        for model in ("C 220", "E 220", "CLA 180"):
            assert model in _MODEL_ALIASES["Mercedes-Benz"]
