"""Tests for car generation lookup logic."""

from unittest.mock import patch

from src.models.generations import get_generation, _MODEL_ALIASES


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
