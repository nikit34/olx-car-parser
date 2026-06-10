"""Tests for fuel_type canonicalisation."""

from src.parser.fuel_normalize import normalize_fuel_type


def test_electric_variants_collapse():
    assert normalize_fuel_type("Eléctrico") == "Eléctrico"   # OLX spelling
    assert normalize_fuel_type("Elétrico") == "Eléctrico"    # SV spelling -> canon
    assert normalize_fuel_type("electrico") == "Eléctrico"
    # all electric labels land on a single category
    assert len({normalize_fuel_type(v) for v in ("Eléctrico", "Elétrico")}) == 1


def test_plugin_case_collapses():
    assert normalize_fuel_type("Híbrido Plug-in") == "Híbrido Plug-in"
    assert normalize_fuel_type("Híbrido Plug-In") == "Híbrido Plug-in"
    assert len({normalize_fuel_type(v)
                for v in ("Híbrido Plug-in", "Híbrido Plug-In")}) == 1


def test_distinct_powertrains_preserved():
    # Plain labels and the genuinely-distinct hybrid sub-types are untouched.
    for v in ("Diesel", "Gasolina", "GPL", "GNC", "Híbrido",
              "Híbrido (Gasolina)", "Híbrido (Diesel)"):
        assert normalize_fuel_type(v) == v


def test_empty_and_none():
    assert normalize_fuel_type(None) is None
    assert normalize_fuel_type("") == ""
    assert normalize_fuel_type("  Diesel  ") == "Diesel"  # stripped


def test_full_corpus_distincts_collapse():
    # The 12 distinct values seen in prod collapse to the expected canon set.
    raw = ["Diesel", "Gasolina", "", "Eléctrico", "Híbrido Plug-in",
           "Híbrido Plug-In", "Elétrico", "Híbrido", "GPL",
           "Híbrido (Gasolina)", "Híbrido (Diesel)", "GNC"]
    canon = {normalize_fuel_type(v) for v in raw}
    assert "Eléctrico" in canon
    assert "Elétrico" not in canon            # merged away
    assert "Híbrido Plug-In" not in canon     # merged away
    assert "Híbrido Plug-in" in canon
