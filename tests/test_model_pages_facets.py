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
    MIN_MATCH_YEARS,
    MIN_MODEL_N,
    build_model_pages,
    slugify,
)


def _listings(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["is_active"] = True
    return df


def _make(brand="Volkswagen", model="Golf", n=MIN_MODEL_N, price=7000,
          fuel="Diesel", district="Porto", year=2014, km=180000, start=0,
          transmission="Manual"):
    return [{
        "brand": brand, "model": model, "price_eur": price + (i % 7) * 100,
        "fuel_type": fuel, "district": district, "year": year,
        "transmission": transmission,
        "mileage_km": km + (i % 5) * 1000, "olx_id": f"x{start + i}",
    } for i in range(n)]


def _spread(years, per_year, **kw):
    """One block per year, so a facet can be compared year by year."""
    rows = []
    for i, y in enumerate(years):
        rows += _make(n=per_year, year=y, start=kw.pop("start", 0) + i * 1000, **kw)
    return rows


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


class TestTransmissionFacets:
    def test_both_gearboxes_get_a_cell(self):
        rows = (_make(n=30, transmission="Manual")
                + _make(n=20, transmission="Automática", price=15000, start=100))
        rec = build_model_pages(_listings(rows))["models"]["volkswagen-golf"]
        keys = {c["k"]: c for c in rec["tx"]}
        assert set(keys) == {"manual", "automatica"}
        assert keys["automatica"]["lbl"] == "Automática"
        for cell in rec["tx"]:
            assert cell["fl"] <= cell["fm"] <= cell["fh"]

    def test_thin_gearbox_is_absent(self):
        rows = (_make(n=30, transmission="Manual")
                + _make(n=MIN_FACET_N - 1, transmission="Automática", price=15000, start=100))
        rec = build_model_pages(_listings(rows))["models"]["volkswagen-golf"]
        assert [c["k"] for c in rec["tx"]] == ["manual"]

    def test_unknown_gearbox_vocabulary_gets_no_facet(self):
        rows = _make(n=25, transmission="Semi-automática")
        rec = build_model_pages(_listings(rows))["models"]["volkswagen-golf"]
        assert "tx" not in rec


class TestMatchedRatios:
    """The number a facet page is allowed to print.

    Two facets of one model are two different mixes of model years, so their
    medians cannot be subtracted: the automatics on sale are newer than the
    manuals, and the raw ratio reports that as a gearbox premium. Every facet
    therefore carries the ratio measured WITHIN each year and then pooled.
    """

    def test_matched_ratio_ignores_the_age_mix(self):
        years = [2012, 2014, 2016, 2018]
        rows = (_spread(years, 8, transmission="Manual", price=10000)
                + _spread(years, 8, transmission="Automática", price=12000, start=500))
        rec = build_model_pages(_listings(rows))["models"]["volkswagen-golf"]
        auto = next(c for c in rec["tx"] if c["k"] == "automatica")
        ratio, shared = auto["vs"]["manual"]
        assert shared == len(years)
        assert 1.15 <= ratio <= 1.25

    def test_no_shared_years_means_no_ratio(self):
        """Where the two sides never overlap in year there is nothing to match,
        and the page must be left with no percentage to print."""
        rows = (_spread([2006, 2008, 2010, 2012], 8, transmission="Manual", price=4000)
                + _spread([2020, 2021, 2022, 2023], 8, transmission="Automática",
                          price=22000, start=500))
        rec = build_model_pages(_listings(rows))["models"]["volkswagen-golf"]
        auto = next(c for c in rec["tx"] if c["k"] == "automatica")
        assert "vs" not in auto or "manual" not in auto.get("vs", {})

    def test_one_shared_year_is_not_enough(self):
        rows = (_spread([2010, 2012, 2014], 8, transmission="Manual", price=6000)
                + _spread([2014], 20, transmission="Automática", price=6300, start=500))
        rec = build_model_pages(_listings(rows))["models"]["volkswagen-golf"]
        auto = next(c for c in rec["tx"] if c["k"] == "automatica")
        assert MIN_MATCH_YEARS > 1
        assert "vs" not in auto or "manual" not in auto.get("vs", {})

    def test_fuel_cells_carry_it_too(self):
        years = [2012, 2014, 2016, 2018]
        rows = (_spread(years, 8, fuel="Diesel", price=10000)
                + _spread(years, 8, fuel="Gasolina", price=8000, start=500))
        rec = build_model_pages(_listings(rows))["models"]["volkswagen-golf"]
        diesel = next(c for c in rec["fx"] if c["k"] == "diesel")
        assert diesel["vs"]["gasolina"][0] > 1
        assert diesel["vsm"][1] >= MIN_MATCH_YEARS


class TestCompositionGuard:
    """A cut whose median is far from the model's, with no way to control for
    age, is publishing its age mix as a price. It is dropped, not disclaimed:
    the number lands in the title, the meta description and the AggregateOffer,
    where a disclaimer three paragraphs down cannot reach it."""

    def test_uncontrolled_outlier_cut_is_dropped(self):
        rows = (_spread([2004, 2006, 2008], 10, transmission="Manual", price=3000)
                + _make(n=16, year=2022, transmission="Automática", price=19000, start=900))
        rec = build_model_pages(_listings(rows))["models"]["volkswagen-golf"]
        assert not any(c["k"] == "automatica" for c in rec.get("tx", [])), \
            "published a cut whose whole gap is the age mix"
        assert any(c["k"] == "manual" for c in rec.get("tx", []))

    def test_a_cut_that_can_be_age_controlled_survives_the_same_gap(self):
        years = [2016, 2018, 2020, 2022]
        rows = (_spread(years, 10, transmission="Manual", price=8000)
                + _spread(years, 10, transmission="Automática", price=19000, start=900))
        rec = build_model_pages(_listings(rows))["models"]["volkswagen-golf"]
        auto = next(c for c in rec["tx"] if c["k"] == "automatica")
        assert "vsm" in auto or "dr" in auto

    def test_a_cut_that_dominates_its_own_reference_gets_no_ratio(self):
        """The age control divides each listing by the model's median for its
        year. Where the cut IS most of that year it divides itself by itself and
        lands on 1.00 whatever the real gap — so those years do not count, and a
        cut left without enough of them keeps no ratio and no page."""
        rows = []
        for i, y in enumerate(range(2000, 2008)):
            rows += _make(n=40, year=y, price=2000, district="Porto", start=i * 100)
        for i, y in enumerate(range(2015, 2023)):
            rows += _make(n=4, year=y, price=15000, district="Faro", start=5000 + i * 100)
            rows += _make(n=1, year=y, price=14000, district="Lisboa", start=8000 + i * 100)
        rec = build_model_pages(_listings(rows))["models"]["volkswagen-golf"]
        faro = next((c for c in rec.get("dt", []) if c["k"] == "faro"), None)
        assert faro is None, \
            f"published a cut at {faro['fm'] / rec['fm']:.1f}x the model on a ratio it computed against itself"

    def test_the_age_controlled_ratio_reaches_thin_cells(self):
        rows = []
        for i, y in enumerate([2010, 2012, 2014, 2016]):
            rows += _make(n=20, year=y, district="Porto", price=5000 + i * 2000,
                          start=i * 100)
        rows += [{
            "brand": "Volkswagen", "model": "Golf", "price_eur": 6000 + (i % 4) * 2000,
            "fuel_type": "Diesel", "district": "Faro",
            "year": [2010, 2012, 2014, 2016][i % 4], "transmission": "Manual",
            "mileage_km": 180000, "olx_id": f"f{i}",
        } for i in range(16)]
        rec = build_model_pages(_listings(rows))["models"]["volkswagen-golf"]
        faro = next(c for c in rec["dt"] if c["k"] == "faro")
        assert "dr" in faro, "a district too thin for vsm got no age control at all"
        assert 0.8 <= faro["dr"][0] <= 1.2
        assert faro["dr"][1] >= 10


class TestPublishHysteresis:
    """The page set is a function of live inventory, so a model or year sitting
    on its floor flips out on a normal dip and takes an indexed, ranking URL
    with it — measured at 12% of impression-earning URLs over six days. Entry
    stays where it was; leaving got harder."""

    def test_a_published_model_survives_a_dip_below_the_entry_floor(self):
        rows = _make(n=16, price=5000)
        assert "volkswagen-golf" not in build_model_pages(_listings(rows))["models"]
        prev = {"models": {"volkswagen-golf": {"b": "Volkswagen", "m": "Golf", "yr": []}}}
        kept = build_model_pages(_listings(rows), published=prev)["models"]
        assert "volkswagen-golf" in kept
        assert kept["volkswagen-golf"]["n"] == 16

    def test_it_still_leaves_when_the_sample_really_goes(self):
        rows = _make(n=10, price=5000)
        prev = {"models": {"volkswagen-golf": {"b": "Volkswagen", "m": "Golf", "yr": []}}}
        assert "volkswagen-golf" not in build_model_pages(_listings(rows), published=prev)["models"]

    def test_a_published_year_survives_a_dip(self):
        rows = _spread([2012, 2014, 2016], 8, price=6000) + _make(n=3, year=2018, start=800)
        plain = build_model_pages(_listings(rows))["models"]["volkswagen-golf"]
        assert 2018 not in {c["y"] for c in plain["yr"]}
        prev = {"models": {"volkswagen-golf": {"yr": [{"y": 2018}]}}}
        kept = build_model_pages(_listings(rows), published=prev)["models"]["volkswagen-golf"]
        cell = next((c for c in kept["yr"] if c["y"] == 2018), None)
        assert cell is not None and cell["n"] == 3

    def test_a_year_that_had_its_own_url_keeps_it_through_a_dip(self):
        """The floor that decides a /preco/{slug}/{ano} URL is 10, not 5, and it
        is read in the Worker, which has no memory of the previous build. So the
        decision ships in the blob as `pg`: 38% of year pages sit within three
        listings of that floor."""
        rows = _spread([2012, 2014], 20, price=6000) + _make(n=8, year=2018, start=800)
        plain = build_model_pages(_listings(rows))["models"]["volkswagen-golf"]
        cell = next(c for c in plain["yr"] if c["y"] == 2018)
        assert cell["n"] == 8 and "pg" not in cell, "a year under the page floor got a URL"

        prev = {"models": {"volkswagen-golf": {"yr": [{"y": 2018, "n": 11, "pg": 1}]}}}
        kept = build_model_pages(_listings(rows), published=prev)["models"]["volkswagen-golf"]
        held = next(c for c in kept["yr"] if c["y"] == 2018)
        assert held.get("pg") == 1, "a published year page died on a two-listing dip"

    def test_a_year_page_still_goes_when_the_sample_really_goes(self):
        rows = _spread([2012, 2014], 20, price=6000) + _make(n=5, year=2018, start=800)
        prev = {"models": {"volkswagen-golf": {"yr": [{"y": 2018, "n": 11, "pg": 1}]}}}
        kept = build_model_pages(_listings(rows), published=prev)["models"]["volkswagen-golf"]
        held = next(c for c in kept["yr"] if c["y"] == 2018)
        assert "pg" not in held, "a year page outlived its retirement floor"

    def test_a_previous_blob_of_the_wrong_shape_does_not_abort_the_build(self):
        rows = _make(n=30, price=5000)
        for junk in (None, {}, {"models": None}, {"models": {"volkswagen-golf": None}},
                     {"models": {"volkswagen-golf": {"yr": "not-a-list"}}},
                     {"models": {"volkswagen-golf": {"yr": [{"y": "2010-2011"}, None, 7]}}}):
            doc = build_model_pages(_listings(rows), published=junk)
            assert "volkswagen-golf" in doc["models"], f"build died on published={junk!r}"

    def test_an_unpublished_thin_year_is_still_refused(self):
        rows = _spread([2012, 2014, 2016], 8, price=6000) + _make(n=3, year=2018, start=800)
        prev = {"models": {"volkswagen-golf": {"yr": [{"y": 2012}]}}}
        kept = build_model_pages(_listings(rows), published=prev)["models"]["volkswagen-golf"]
        assert 2018 not in {c["y"] for c in kept["yr"]}

    def test_no_previous_blob_means_the_old_behaviour(self):
        rows = _make(n=16, price=5000)
        assert build_model_pages(_listings(rows), published=None)["models"] == {}
