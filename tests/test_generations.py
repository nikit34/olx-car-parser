"""Tests for car generation lookup logic."""

from unittest.mock import patch

from src.models.generations import (
    brand_for_model,
    get_generation,
    infer_model_from_title,
    _get_model_aliases,
)


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


class TestNormalizedFallback:
    def test_case_insensitive_brand(self, generations_data):
        with patch("src.models.generations.load_generations", return_value=generations_data):
            assert get_generation("VOLKSWAGEN", "Golf", 2015) == "Mk7"

    def test_case_insensitive_model(self, generations_data):
        with patch("src.models.generations.load_generations", return_value=generations_data):
            assert get_generation("Volkswagen", "GOLF", 2015) == "Mk7"

    def test_punctuation_and_spacing_ignored(self, generations_data):
        with patch("src.models.generations.load_generations", return_value=generations_data):
            assert get_generation("Mercedes-Benz", "E Class", 2017) == "W213"
            assert get_generation("mercedes benz", "e-class", 2017) == "W213"

    def test_exact_spelling_still_wins(self, generations_data):
        with patch("src.models.generations.load_generations", return_value=generations_data):
            assert get_generation("BMW", "3 Series", 2015) == "F30"

    def test_still_none_for_unknown_model(self, generations_data):
        with patch("src.models.generations.load_generations", return_value=generations_data):
            assert get_generation("Volkswagen", "Amarok", 2015) is None

    def test_index_rebuilds_when_table_changes(self, generations_data):
        with patch("src.models.generations.load_generations", return_value=generations_data):
            assert get_generation("volkswagen", "golf", 2015) == "Mk7"
        other = {"Renault": {"Clio": [{"name": "Mk4", "year_from": 2012, "year_to": 2019}]}}
        with patch("src.models.generations.load_generations", return_value=other):
            assert get_generation("renault", "clio", 2015) == "Mk4"
            assert get_generation("volkswagen", "golf", 2015) is None

    def test_portuguese_spellings_resolve_on_real_config(self):
        assert get_generation("BMW", "Série 3", 2015) is not None
        assert get_generation("Mercedes-Benz", "Classe C", 2015) is not None
        assert get_generation("SEAT", "Ibiza", 2015) is not None


class TestReferenceTableCoverage:
    def test_pre_2008_ibiza_resolves(self):
        assert get_generation("Seat", "Ibiza", 1995) is not None

    def test_pre_2006_civic_resolves(self):
        assert get_generation("Honda", "Civic", 1998) is not None

    def test_classic_defender_resolves(self):
        assert get_generation("Land Rover", "Defender", 1998) is not None

    def test_skoda_spellings_agree(self):
        with_hacek = get_generation("Škoda", "Fabia", 2003)
        without = get_generation("Skoda", "Fabia", 2003)
        assert with_hacek is not None
        assert with_hacek == without


class TestBodyVariantAliases:
    def test_body_suffix_resolves_to_base_model(self):
        assert get_generation("Peugeot", "206 CC", 2002) is not None
        assert get_generation("Nissan", "Qashqai +2", 2011) is not None
        assert get_generation("Opel", "Vectra Caravan", 2004) is not None

    def test_portuguese_beetle_name_resolves(self):
        assert get_generation("Volkswagen", "Carocha", 2003) is not None

    def test_newly_added_models_resolve(self):
        assert get_generation("Hyundai", "Getz", 2004) is not None
        assert get_generation("Peugeot", "306", 1999) is not None
        assert get_generation("Alfa Romeo", "156", 2002) is not None
        assert get_generation("Mercedes-Benz", "190", 1990) is not None


class TestModelAliases:
    def test_bmw_series_aliases_exist(self):
        aliases = _get_model_aliases()
        for model in ("116", "118", "120", "318", "320", "520", "530"):
            assert model in aliases["BMW"]

    def test_mercedes_class_aliases_exist(self):
        aliases = _get_model_aliases()
        for model in ("C 220", "E 220", "CLA 180"):
            assert model in aliases["Mercedes-Benz"]


class TestBrandsBeyondTheOriginalThirtyEight:
    def test_lexus_trim_names_resolve(self):
        assert get_generation("Lexus", "IS 220", 2008) is not None
        assert get_generation("Lexus", "CT 200h", 2015) is not None
        assert get_generation("Lexus", "RX 450h", 2011) is not None

    def test_mg_split_generations(self):
        assert get_generation("MG", "ZS", 2004) is not None
        assert get_generation("MG", "ZS", 2022) is not None
        assert get_generation("MG", "MG4", 2024) is not None

    def test_rover_numeric_models_fold_into_families(self):
        assert get_generation("Rover", "214", 1994) is not None
        assert get_generation("Rover", "620", 1996) is not None
        assert get_generation("Rover", "75 Tourer", 2002) is not None

    def test_saab_body_variants_resolve(self):
        assert get_generation("Saab", "9-3 Cabriolet", 2004) is not None
        assert get_generation("Saab", "9-5 SportWagon", 2001) is not None

    def test_recent_electric_brands_resolve(self):
        assert get_generation("BYD", "Atto 3", 2023) is not None
        assert get_generation("Polestar", "2", 2022) is not None
        assert get_generation("Xpeng", "G6", 2024) is not None

    def test_chrysler_european_line_up_resolves(self):
        assert get_generation("Chrysler", "Grand Voyager", 2005) is not None
        assert get_generation("Chrysler", "300 C", 2007) is not None
        assert get_generation("Chrysler", "PT Cruiser", 2002) is not None

    def test_abarth_is_not_fiat(self):
        assert get_generation("Abarth", "595", 2019) is not None
        assert get_generation("Abarth", "500", 2011) is not None


class TestYearAwareTitleInference:
    def test_price_digits_do_not_win_over_the_real_model(self):
        assert infer_model_from_title("Audi", "Audi A4 1.9 TDI m63.100 €", 2001) == "A4"
        assert infer_model_from_title("BMW", "Bmw 316 i gasolina3.700 €", 2002) == "316"

    def test_year_in_the_title_does_not_win_over_the_real_model(self):
        assert infer_model_from_title("BMW", "BMW 320 D 2000 DIESEL", 2000) == "320"

    def test_numeric_classic_still_wins_inside_its_own_range(self):
        assert infer_model_from_title("BMW", "BMW 700 Coupé - restauro", 1961) == "700"

    def test_without_a_year_the_first_match_is_returned(self):
        assert infer_model_from_title("Audi", "Audi A4 1.9 TDI m63.100 €") == "100"

    def test_falls_back_when_no_candidate_resolves(self):
        assert infer_model_from_title("Audi", "Audi A4 1.9 TDI", 1890) == "A4"


class TestModelsInsideKnownBrands:
    def test_mercedes_trim_designations_resolve(self):
        assert get_generation("Mercedes-Benz", "A 45 AMG", 2016) is not None
        assert get_generation("Mercedes-Benz", "ML 320", 2005) is not None
        assert get_generation("Mercedes-Benz", "Classe M", 2009) is not None
        assert get_generation("Mercedes-Benz", "W124 (1984-1997)", 1990) is not None

    def test_mercedes_vans_are_their_own_models(self):
        assert get_generation("Mercedes-Benz", "Sprinter", 2015) is not None
        assert get_generation("Mercedes-Benz", "Citan", 2016) is not None
        assert get_generation("Mercedes-Benz", "Viano", 2008) is not None

    def test_c5_aircross_is_not_a_c5(self):
        assert get_generation("Citroën", "C5 Aircross", 2020) is not None
        assert get_generation("Citroën", "C5 Aircross", 2005) is None

    def test_classics_resolve_only_inside_their_production_years(self):
        assert get_generation("Renault", "19", 1994) is not None
        assert get_generation("Renault", "19", 2016) is None
        assert get_generation("Fiat", "127", 1978) is not None
        assert get_generation("Opel", "Kadett", 1990) is not None
        assert get_generation("Volkswagen", "Carocha", 1968) is not None

    def test_truncated_ranges_reach_the_first_generation(self):
        assert get_generation("Toyota", "Corolla", 1974) is not None
        assert get_generation("Volvo", "S80", 2001) is not None
        assert get_generation("Hyundai", "Tucson", 2006) is not None
        assert get_generation("Suzuki", "Ignis", 2003) is not None

    def test_newly_covered_models_resolve(self):
        assert get_generation("Porsche", "Cayman", 2010) is not None
        assert get_generation("Porsche", "718 Cayman", 2018) is not None
        assert get_generation("Jaguar", "F-Type", 2016) is not None
        assert get_generation("Hyundai", "Galloper", 1999) is not None
        assert get_generation("Jeep", "Avenger", 2024) is not None


class TestBrandForModel:
    def test_a_model_owned_by_one_brand_names_it(self):
        assert brand_for_model("Golf") == "Volkswagen"
        assert brand_for_model("Sprinter") == "Mercedes-Benz"
        assert brand_for_model("C5 Aircross") == "Citroën"

    def test_a_model_name_two_brands_share_stays_unresolved(self):
        assert brand_for_model("Corsa") is None
        assert brand_for_model("Ibiza") is None
        assert brand_for_model("220") is None

    def test_empty_input(self):
        assert brand_for_model("") is None

    def test_spelling_of_the_model_does_not_matter(self):
        assert brand_for_model("c5 aircross") == "Citroën"
        assert brand_for_model("C5-Aircross") == "Citroën"


class TestSpacelessModelNames:
    def test_the_space_inside_a_model_name_is_optional(self):
        assert infer_model_from_title("Mercedes-Benz", "Mercedes E270 Avantgard", 2004) == "E 270"
        assert infer_model_from_title("Mercedes-Benz", "Mercedes ml320 cdi", 2005) == "ML 320"
        assert infer_model_from_title("Mercedes-Benz", "Mercedes A200 cdi", 2010) == "A 200"

    def test_the_spelled_out_name_still_matches(self):
        assert infer_model_from_title("Mercedes-Benz", "Mercedes E 270 CDI", 2004) == "E 270"

    def test_it_does_not_match_across_a_word_boundary(self):
        assert infer_model_from_title("Mercedes-Benz", "SE270 qualquer coisa", 2004) is None
