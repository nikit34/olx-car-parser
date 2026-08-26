"""Facet cells (fuel, district) and the national district rollup.

The rule these pin is the one that makes a 1 000-page expansion safe rather than
reckless: a facet exists only where its own sample can carry a median. A thin
facet is ABSENT, never merged into an unrelated bucket and never estimated —
same contract as the per-year cells (see test_model_pages_gbm).
"""

from __future__ import annotations

import pandas as pd

from src.analytics.model_pages import (
    MIN_DISTRICT_N,
    MIN_FACET_N,
    MIN_MODEL_N,
    build_model_pages,
    slugify,
)


def _listings(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["is_active"] = True
    return df


def _make(brand="Volkswagen", model="Golf", n=MIN_MODEL_N, price=7000,
          fuel="Diesel", district="Porto", year=2014, km=180000, start=0):
    return [{
        "brand": brand, "model": model, "price_eur": price + (i % 7) * 100,
        "fuel_type": fuel, "district": district, "year": year,
        "mileage_km": km + (i % 5) * 1000, "olx_id": f"x{start + i}",
    } for i in range(n)]


class TestFuelFacets:
    def test_deep_fuel_gets_a_cell_with_its_own_range(self):
        rows = _make(n=30, fuel="Diesel") + _make(n=20, fuel="Gasolina", price=9000, start=100)
        doc = build_model_pages(_listings(rows))
        rec = doc["models"]["volkswagen-golf"]
        keys = {c["k"]: c for c in rec["fx"]}
        assert set(keys) == {"diesel", "gasolina"}
        assert keys["diesel"]["n"] == 30 and keys["gasolina"]["n"] == 20
        # Every median ships with its range — never a lone number.
        for cell in rec["fx"]:
            assert cell["fl"] <= cell["fm"] <= cell["fh"]
            assert cell["lbl"]

    def test_thin_fuel_is_absent_not_merged(self):
        thin = MIN_FACET_N - 1
        rows = _make(n=30, fuel="Diesel") + _make(n=thin, fuel="Gasolina", price=9000, start=100)
        rec = build_model_pages(_listings(rows))["models"]["volkswagen-golf"]
        assert [c["k"] for c in rec["fx"]] == ["diesel"]

    def test_ambiguous_fuels_get_no_facet(self):
        """Near-duplicate vocabulary ("Hibrido Plug-in"/"Plug-In") would slug
        into competing pages for one thing, so those fuels get no facet at all —
        they still appear in the ``fu`` mix."""
        rows = _make(n=25, fuel="Diesel") + _make(n=25, fuel="Híbrido Plug-in", price=15000, start=100)
        rec = build_model_pages(_listings(rows))["models"]["volkswagen-golf"]
        assert [c["k"] for c in rec["fx"]] == ["diesel"]
        assert any(f[0].startswith("Híbrido") for f in rec["fu"])

    def test_no_facet_key_when_nothing_qualifies(self):
        rows = _make(n=25, fuel="Eléctrico")
        rec = build_model_pages(_listings(rows))["models"]["volkswagen-golf"]
        assert "fx" not in rec


class TestDistrictFacets:
    def test_per_model_district_cells(self):
        rows = _make(n=20, district="Porto") + _make(n=20, district="Lisboa", price=8000, start=100)
        rec = build_model_pages(_listings(rows))["models"]["volkswagen-golf"]
        assert {c["k"] for c in rec["dt"]} == {"porto", "lisboa"}
        assert all(c["n"] >= MIN_FACET_N for c in rec["dt"])

    def test_accented_district_slugs_match_the_js_rule(self):
        rows = _make(n=20, district="Setúbal")
        rec = build_model_pages(_listings(rows))["models"]["volkswagen-golf"]
        assert rec["dt"][0]["k"] == slugify("Setúbal") == "setubal"
        assert rec["dt"][0]["lbl"] == "Setúbal"

    def test_thin_district_is_absent(self):
        rows = _make(n=20, district="Porto") + _make(n=MIN_FACET_N - 1, district="Beja", start=100)
        rec = build_model_pages(_listings(rows))["models"]["volkswagen-golf"]
        assert {c["k"] for c in rec["dt"]} == {"porto"}


class TestDistrictRollup:
    def test_rollup_appears_only_above_the_national_floor(self):
        rows = _make(n=MIN_DISTRICT_N + 10, district="Porto")
        doc = build_model_pages(_listings(rows))
        assert "porto" in doc["districts"]
        d = doc["districts"]["porto"]
        assert d["n"] == MIN_DISTRICT_N + 10
        assert d["fl"] <= d["fm"] <= d["fh"]
        assert d["lbl"] == "Porto"

    def test_small_district_gets_no_page(self):
        rows = _make(n=MIN_DISTRICT_N - 1, district="Beja")
        doc = build_model_pages(_listings(rows))
        assert "districts" not in doc or "beja" not in doc.get("districts", {})

    def test_rollup_only_links_models_that_have_a_page(self):
        """A district page must not link a model we never published — that is a
        404 shipped in the sitemap."""
        rows = (_make(n=MIN_DISTRICT_N, brand="Volkswagen", model="Golf", district="Porto")
                + _make(n=MIN_MODEL_N - 1, brand="Rara", model="Coisa", district="Porto", start=900))
        doc = build_model_pages(_listings(rows))
        linked = {t[0] for t in doc["districts"]["porto"]["top"]}
        assert linked == {"volkswagen-golf"}
        assert "rara-coisa" not in doc["models"]

    def test_document_shape_is_unchanged_when_there_is_no_geo(self):
        rows = [{**r, "district": None} for r in _make(n=MIN_MODEL_N)]
        doc = build_model_pages(_listings(rows))
        assert doc["v"] == 1 and "volkswagen-golf" in doc["models"]
        assert "districts" not in doc
