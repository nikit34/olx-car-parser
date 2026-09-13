"""Placing a foreign listing, and the three different ways its country hides it.

These are real postcodes and real AutoScout24 city strings, because the failure
mode this guards against is not a crash: a wrong region parses fine, stores
fine and shows up as a car in somebody else's market median. The cases picked
are the ones where a naive two-digit table is wrong — Potsdam against Berlin,
Neu-Ulm against Ulm, Ramstein against Saarbrücken — plus the two Italian city
shapes and Corsica's shared prefix.
"""

from __future__ import annotations

import pytest

from src.parser.regions import french_departement, italian_province_region, region_for


class TestGermany:

    @pytest.mark.parametrize("plz,state", [
        ("01067", "Sachsen"), ("04109", "Sachsen"), ("08056", "Sachsen"),
        ("06108", "Sachsen-Anhalt"), ("39104", "Sachsen-Anhalt"),
        ("07743", "Thüringen"), ("99084", "Thüringen"), ("98527", "Thüringen"),
        ("03046", "Brandenburg"), ("15230", "Brandenburg"),
        ("10115", "Berlin"), ("12043", "Berlin"), ("13353", "Berlin"),
        ("18055", "Mecklenburg-Vorpommern"), ("19053", "Mecklenburg-Vorpommern"),
        ("20095", "Hamburg"), ("24103", "Schleswig-Holstein"),
        ("28195", "Bremen"), ("30159", "Niedersachsen"), ("26123", "Niedersachsen"),
        ("33098", "Nordrhein-Westfalen"), ("40213", "Nordrhein-Westfalen"),
        ("44141", "Nordrhein-Westfalen"), ("53757", "Nordrhein-Westfalen"),
        ("54290", "Rheinland-Pfalz"), ("56068", "Rheinland-Pfalz"),
        ("60311", "Hessen"), ("64569", "Hessen"),
        ("66111", "Saarland"),
        ("67059", "Rheinland-Pfalz"), ("70173", "Baden-Württemberg"),
        ("79098", "Baden-Württemberg"), ("80331", "Bayern"), ("90402", "Bayern"),
        ("97070", "Bayern"),
    ])
    def test_the_leitregion_decides_the_bundesland(self, plz, state):
        assert region_for("DE", plz) == state

    @pytest.mark.parametrize("plz,state", [
        ("14109", "Berlin"), ("14169", "Berlin"),
        ("14467", "Brandenburg"), ("14770", "Brandenburg"),
        ("21073", "Hamburg"), ("21147", "Hamburg"),
        ("21335", "Niedersachsen"), ("21614", "Niedersachsen"),
        ("21502", "Schleswig-Holstein"),
        ("22769", "Hamburg"), ("22844", "Schleswig-Holstein"),
        ("28779", "Bremen"), ("28832", "Niedersachsen"),
        ("36037", "Hessen"), ("36404", "Thüringen"),
        ("37073", "Niedersachsen"), ("37213", "Hessen"), ("37308", "Thüringen"),
        ("38100", "Niedersachsen"), ("38820", "Sachsen-Anhalt"),
        ("53111", "Nordrhein-Westfalen"), ("53474", "Rheinland-Pfalz"),
        ("53604", "Nordrhein-Westfalen"),
        ("57072", "Nordrhein-Westfalen"), ("57518", "Rheinland-Pfalz"),
        ("66424", "Saarland"), ("66877", "Rheinland-Pfalz"),
        ("68159", "Baden-Württemberg"), ("68623", "Hessen"),
        ("76133", "Baden-Württemberg"), ("76744", "Rheinland-Pfalz"),
        ("88212", "Baden-Württemberg"), ("88131", "Bayern"),
        ("89073", "Baden-Württemberg"), ("89231", "Bayern"),
        ("89522", "Baden-Württemberg"),
        ("97816", "Bayern"), ("97941", "Baden-Württemberg"),
    ])
    def test_a_prefix_that_straddles_two_states_is_split_at_the_third_digit(
            self, plz, state):
        assert region_for("DE", plz) == state

    def test_something_that_is_not_a_plz_is_not_a_state(self):
        assert region_for("DE", None) is None
        assert region_for("DE", "") is None
        assert region_for("DE", "1234") is None
        assert region_for("DE", "05123") is None


class TestFrance:

    @pytest.mark.parametrize("cp,region", [
        ("75001", "Île-de-France"), ("93100", "Île-de-France"),
        ("13001", "Provence-Alpes-Côte d'Azur"), ("83210", "Provence-Alpes-Côte d'Azur"),
        ("69001", "Auvergne-Rhône-Alpes"), ("38000", "Auvergne-Rhône-Alpes"),
        ("33000", "Nouvelle-Aquitaine"), ("87000", "Nouvelle-Aquitaine"),
        ("59000", "Hauts-de-France"), ("80000", "Hauts-de-France"),
        ("67000", "Grand Est"), ("68000", "Grand Est"), ("51100", "Grand Est"),
        ("44000", "Pays de la Loire"), ("85000", "Pays de la Loire"),
        ("35000", "Bretagne"), ("29200", "Bretagne"),
        ("76000", "Normandie"), ("14000", "Normandie"),
        ("21000", "Bourgogne-Franche-Comté"), ("25620", "Bourgogne-Franche-Comté"),
        ("45000", "Centre-Val de Loire"), ("37000", "Centre-Val de Loire"),
        ("31000", "Occitanie"), ("30900", "Occitanie"), ("34000", "Occitanie"),
    ])
    def test_the_departement_rolls_up_to_its_region(self, cp, region):
        assert region_for("FR", cp) == region

    def test_both_halves_of_corsica_are_one_region(self):
        assert region_for("FR", "20000") == "Corse"
        assert region_for("FR", "20137") == "Corse"
        assert region_for("FR", "20200") == "Corse"
        assert region_for("FR", "20620") == "Corse"

    def test_corsica_still_knows_which_departement_it_is(self):
        assert french_departement("20000") == "Corse-du-Sud"
        assert french_departement("20137") == "Corse-du-Sud"
        assert french_departement("20200") == "Haute-Corse"
        assert french_departement("20250") == "Haute-Corse"
        assert french_departement("75015") == "Paris"
        assert french_departement("90000") == "Territoire de Belfort"

    @pytest.mark.parametrize("cp,region", [
        ("97110", "Guadeloupe"), ("97200", "Martinique"), ("97300", "Guyane"),
        ("97400", "La Réunion"), ("97600", "Mayotte"),
        ("98800", "Nouvelle-Calédonie"), ("98700", "Polynésie française"),
    ])
    def test_an_overseas_departement_is_its_own_region(self, cp, region):
        assert region_for("FR", cp) == region

    def test_a_code_postal_that_is_no_departement_places_nothing(self):
        assert region_for("FR", "96000") is None
        assert region_for("FR", "99999") is None
        assert region_for("FR", "7500") is None
        assert region_for("FR", None) is None


class TestItaly:

    @pytest.mark.parametrize("city,region", [
        ("Poviglio - Reggio Emilia - RE", "Emilia-Romagna"),
        ("Ladispoli - Roma - RM", "Lazio"),
        ("Bra - Cuneo - CN", "Piemonte"),
        ("Bolzano - BZ", "Trentino-Alto Adige"),
        ("Reggio Calabria - RC", "Calabria"),
    ])
    def test_the_sigla_at_the_end_of_the_city_string_wins(self, city, region):
        assert region_for("IT", None, city) == region

    @pytest.mark.parametrize("city,region", [
        ("Carrara - Massa Carrara", "Toscana"),
        ("Ardea - Roma", "Lazio"),
        ("Milano", "Lombardia"),
        ("Napoli", "Campania"),
        ("Forlì - Forlì-Cesena", "Emilia-Romagna"),
        ("L'Aquila", "Abruzzo"),
        ("Monza", "Lombardia"),
    ])
    def test_a_province_written_out_is_read_by_name(self, city, region):
        assert region_for("IT", None, city) == region

    @pytest.mark.parametrize("cap,region", [
        ("00185", "Lazio"), ("20121", "Lombardia"), ("35100", "Veneto"),
        ("10121", "Piemonte"), ("11100", "Valle d'Aosta"), ("16121", "Liguria"),
        ("30100", "Veneto"), ("33100", "Friuli-Venezia Giulia"),
        ("38122", "Trentino-Alto Adige"), ("40121", "Emilia-Romagna"),
        ("50122", "Toscana"), ("60121", "Marche"), ("65121", "Abruzzo"),
        ("70121", "Puglia"), ("75100", "Basilicata"), ("86100", "Molise"),
        ("80121", "Campania"), ("88100", "Calabria"), ("90133", "Sicilia"),
        ("09121", "Sardegna"), ("06121", "Umbria"),
    ])
    def test_the_cap_is_the_fallback_for_a_bare_city_name(self, cap, region):
        assert region_for("IT", cap, "Qualche Paese") == region
        assert region_for("IT", cap, None) == region

    def test_the_city_string_beats_the_cap_when_it_names_a_province(self):
        """The CAP of Ardea is Roman anyway; Poviglio's is not the point — a
        card whose city names its province should never need the fallback."""
        assert region_for("IT", "00040", "Ardea - Roma") == "Lazio"
        assert italian_province_region("Poviglio - Reggio Emilia - RE") == "Emilia-Romagna"
        assert italian_province_region("Ardea") is None

    def test_a_cap_nobody_uses_places_nothing(self):
        assert region_for("IT", "99999", None) is None
        assert region_for("IT", "49000", None) is None
        assert region_for("IT", None, None) is None


class TestEverywhereElse:

    def test_a_country_this_module_does_not_know_returns_nothing(self):
        assert region_for("AT", "1010", "Wien") is None
        assert region_for("ES", "28001", "Madrid") is None
        assert region_for("", "10115", None) is None
        assert region_for(None, "10115", None) is None

    def test_the_code_is_case_insensitive(self):
        assert region_for("de", "10115") == "Berlin"
        assert region_for("fr", "75001") == "Île-de-France"
        assert region_for("it", None, "Milano") == "Lombardia"
