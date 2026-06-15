"""ISV estimator regression tests.

The expected values are the official Autoridade Tributária worked figures /
Código do ISV Art. 7.º+11.º (2026 tables, == 2024/2025). Verified 2026-06-15.
If a future OE changes the rates, these are the canonical numbers to re-derive.
"""

from src.analytics.isv import compute_isv


def _isv(**kw):
    return compute_isv(as_of_year=2026, **kw)


def test_example_a_golf_petrol_wltp():
    # VW Golf 1.5 TSI, petrol, 1498 cm3, 130 g/km WLTP, 1st reg 2021 (age 5 → 43%)
    r = _isv(co2_g_km=130, engine_cc=1498, fuel_type="Gasolina", first_reg_year=2021)
    assert r["components"]["cilindrada"] == 2208.90
    assert r["components"]["co2"] == 65.93
    assert r["cycle"] == "WLTP"
    assert r["reduction_pct"] == 0.43
    assert r["isv_eur"] == 1296.65


def test_example_b_320d_diesel_nedc():
    # BMW 320d, diesel, 1995 cm3, 120 g/km NEDC, 1st reg 2016 (age 10 → 75%)
    r = _isv(co2_g_km=120, engine_cc=1995, fuel_type="Diesel", first_reg_year=2016)
    assert r["components"]["cilindrada"] == 4997.07
    assert r["components"]["co2"] == 2310.77
    assert r["components"]["particulas"] == 500.0
    assert r["cycle"] == "NEDC"
    assert r["isv_eur"] == 1951.96


def test_example_c_clio_petrol_small_engine():
    # Renault Clio 1.0 TCe, petrol, 999 cm3, 120 g/km WLTP, 1st reg 2022 (age 4 → 35%)
    r = _isv(co2_g_km=120, engine_cc=999, fuel_type="Gasolina", first_reg_year=2022)
    assert r["isv_eur"] == 167.50


def test_cilindrada_sanity():
    # CGD's published 998cc example: 998×1.09 − 849.03 = 238.79
    assert _isv(co2_g_km=100, engine_cc=998, fuel_type="Gasolina",
                first_reg_year=2024)["components"]["cilindrada"] == 238.79


def test_co2_nedc_petrol_sanity():
    # 105 g/km NEDC petrol: 105×8.09 − 750.99 = 98.46
    assert _isv(co2_g_km=105, engine_cc=1400, fuel_type="Gasolina",
                first_reg_year=2015)["components"]["co2"] == 98.46


def test_bev_exempt():
    r = _isv(co2_g_km=0, engine_cc=1500, fuel_type="Eléctrico", first_reg_year=2022)
    assert r["isv_eur"] == 0.0


def test_phev_abstains():
    assert _isv(co2_g_km=40, engine_cc=1500, fuel_type="Híbrido Plug-in",
                first_reg_year=2022) is None


def test_abstain_on_missing_inputs():
    assert _isv(co2_g_km=None, engine_cc=1500, fuel_type="Diesel", first_reg_year=2018) is None
    assert _isv(co2_g_km=120, engine_cc=None, fuel_type="Diesel", first_reg_year=2018) is None
    assert _isv(co2_g_km=120, engine_cc=1500, fuel_type=None, first_reg_year=2018) is None
    assert _isv(co2_g_km=120, engine_cc=1500, fuel_type="Diesel", first_reg_year=None) is None


def test_2018_2019_cycle_confidence_medium():
    assert _isv(co2_g_km=120, engine_cc=1500, fuel_type="Gasolina",
                first_reg_year=2018)["confidence"] == "medium"
    assert _isv(co2_g_km=120, engine_cc=1500, fuel_type="Gasolina",
                first_reg_year=2021)["confidence"] == "high"


def test_non_eu_no_age_reduction():
    eu = _isv(co2_g_km=130, engine_cc=1498, fuel_type="Gasolina", first_reg_year=2021)
    non_eu = compute_isv(co2_g_km=130, engine_cc=1498, fuel_type="Gasolina",
                         first_reg_year=2021, as_of_year=2026, is_eu=False)
    assert non_eu["isv_eur"] == non_eu["gross_eur"]      # no reduction
    assert non_eu["isv_eur"] > eu["isv_eur"]
