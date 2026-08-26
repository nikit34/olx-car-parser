"""Portuguese ISV (Imposto Sobre Veículos) estimator for used imported cars.

Estimates the tax to nationalize a USED imported categoria-A passenger car in
Portugal. Tables verified 2026-06-15 against the official Autoridade Tributária
source (Portal das Finanças, FAQ ISV-06) + Código do ISV Art. 7.º/11.º; the 2026
rates are unchanged from 2024/2025 (OE 2026 made no ISV-rate change).

ISV = [ componente_cilindrada + componente_CO2 + (€500 partículas if diesel) ]
      × (1 − redução_por_anos_de_uso)        # EU-registered used cars

The BRACKETS BELOW ARE LOCK-STEP with ISV_TABLES in
flipper-club/src/seo-pages.js, which powers the public /isv calculator
(paired-comment pact, like slugify <-> model_pages.slugify). Change one, change
both: a public simulator quoting last year's tables is worse than no simulator,
and tests/worker/render_smoke.mjs spot-checks one bracket from each table to
catch the drift.

Honesty (project rule "flag, don't fake"): returns None when we lack a mandatory
input (CO2, engine_cc, fuel, first-registration year) or when the car is a PHEV
(special reduced regime that needs per-car eligibility we don't have). A BEV is
exempt → 0. CO2 cycle (WLTP/NEDC) is inferred from the registration year, which
is the one genuine source of imprecision (2018–2019 is ambiguous; only ≤2017
NEDC / ≥2020 WLTP are safe) — surfaced via `confidence`.
"""

from __future__ import annotations

import datetime

# ── Componente cilindrada (cm³ → €). Cycle- and fuel-independent. ─────────────
# (upper_bound_cm3, €/cm³, parcela_a_abater). Floored at 0.
_CILINDRADA = [
    (1000, 1.09, 849.03),
    (1250, 1.18, 850.69),
    (float("inf"), 5.61, 6194.88),
]

# ── Componente ambiental (CO2 g/km → €). (upper_bound_gkm, €/g, parcela). ─────
_CO2 = {
    ("WLTP", "petrol"): [
        (110, 0.44, 43.02), (115, 1.10, 115.80), (120, 1.38, 147.79),
        (130, 5.27, 619.17), (145, 6.38, 762.73), (175, 41.54, 5819.56),
        (195, 51.38, 7247.39), (235, 193.01, 34190.52), (float("inf"), 233.81, 41910.96),
    ],
    ("WLTP", "diesel"): [
        (110, 1.72, 11.50), (120, 18.96, 1906.19), (140, 65.04, 7360.85),
        (150, 127.40, 16080.57), (160, 160.81, 21176.06), (170, 221.69, 29227.38),
        (190, 274.08, 36987.98), (float("inf"), 282.35, 38271.32),
    ],
    ("NEDC", "petrol"): [
        (99, 4.62, 427.00), (115, 8.09, 750.99), (145, 52.56, 5903.94),
        (175, 61.24, 7140.17), (195, 155.97, 23627.27), (float("inf"), 205.65, 33390.12),
    ],
    ("NEDC", "diesel"): [
        (79, 5.78, 439.04), (95, 23.45, 1848.58), (120, 79.22, 7195.63),
        (140, 175.73, 18924.92), (160, 195.43, 21720.92), (float("inf"), 268.42, 33447.90),
    ],
}

_DIESEL_PARTICULAS_EUR = 500.0  # flat surcharge for diesel cat-A cars

# ── Tabela D — redução por anos de uso (since 1 Jan 2025: applies to the WHOLE
# ISV). (max_age_years_inclusive, reduction_fraction). ────────────────────────
_REDUCAO = [
    (1, 0.10), (2, 0.20), (3, 0.28), (4, 0.35), (5, 0.43), (6, 0.52),
    (7, 0.60), (8, 0.65), (9, 0.70), (10, 0.75), (float("inf"), 0.80),
]


def _bracket(value: float, table: list[tuple]) -> tuple[float, float]:
    for upper, taxa, parcela in table:
        if value <= upper:
            return taxa, parcela
    return table[-1][1], table[-1][2]


def _reducao(age_years: float) -> float:
    for upper, frac in _REDUCAO:
        if age_years <= upper:
            return frac
    return _REDUCAO[-1][1]


def _fuel_class(fuel_type: str | None) -> str | None:
    """Map canonicalised fuel_type → 'bev' | 'phev' | 'diesel' | 'petrol' | None."""
    f = (fuel_type or "").strip().lower()
    if not f:
        return None
    if "plug" in f:                       # Híbrido Plug-in (PHEV)
        return "phev"
    if "ele" in f or "elé" in f:          # Eléctrico / Elétrico (BEV)
        return "bev"
    if "diesel" in f or "gasol" in f and "eo" in f or "gasóleo" in f or "gasoleo" in f:
        return "diesel"
    # Gasolina, Híbrido (non-plug-in), GPL, Gás Natural → petrol CO2 table
    return "petrol"


def co2_cycle(first_reg_year: int | None) -> str | None:
    """Infer the CO2 test cycle from first-registration year. Only ≤2017 (NEDC)
    and ≥2020 (WLTP) are safe; 2018–2019 defaults to NEDC (per-car ambiguous)."""
    if first_reg_year is None:
        return None
    return "NEDC" if first_reg_year <= 2019 else "WLTP"


def compute_isv(co2_g_km, engine_cc, fuel_type, first_reg_year, as_of_year=None,
                is_eu=True) -> dict | None:
    """Estimate ISV (€) to nationalize a used imported cat-A passenger car.

    Returns a dict {isv_eur, gross_eur, components, cycle, reduction_pct,
    age_years, confidence} or None when uncomputable (per the honesty rule).
    BEV → isv_eur 0. PHEV → None (special regime, needs unavailable eligibility).
    """
    fclass = _fuel_class(fuel_type)
    if fclass is None:
        return None
    if fclass == "bev":
        return {"isv_eur": 0.0, "gross_eur": 0.0, "components": {"exempt": True},
                "cycle": None, "reduction_pct": 0.0, "age_years": None, "confidence": "high"}
    if fclass == "phev":
        return None  # reduced regime depends on range+CO2 eligibility we don't have

    # Mandatory numeric inputs for a petrol/diesel ISV.
    try:
        co2 = float(co2_g_km); cc = float(engine_cc); ry = int(first_reg_year)
    except (TypeError, ValueError):
        return None
    if co2 <= 0 or cc <= 0:
        return None

    cycle = co2_cycle(ry)
    if cycle is None:
        return None

    # 1. Cilindrada (floored at 0)
    taxa_cc, parc_cc = _bracket(cc, _CILINDRADA)
    cilindrada = max(0.0, cc * taxa_cc - parc_cc)

    # 2. CO2 (componente ambiental)
    taxa_co2, parc_co2 = _bracket(co2, _CO2[(cycle, fclass)])
    co2_comp = max(0.0, co2 * taxa_co2 - parc_co2)

    # 3. Partículas (diesel only)
    particulas = _DIESEL_PARTICULAS_EUR if fclass == "diesel" else 0.0

    gross = cilindrada + co2_comp + particulas

    # 4. Tabela D age reduction (EU-registered used cars only)
    as_of = as_of_year or datetime.date.today().year
    age = max(0, as_of - ry)
    red = _reducao(age) if is_eu else 0.0
    isv = round(gross * (1 - red), 2)

    # Confidence: the only structural imprecision is the cycle inference for
    # 2018–2019 cars (year-only) → MEDIUM there, HIGH otherwise.
    confidence = "medium" if 2018 <= ry <= 2019 else "high"

    return {
        "isv_eur": isv,
        "gross_eur": round(gross, 2),
        "components": {
            "cilindrada": round(cilindrada, 2),
            "co2": round(co2_comp, 2),
            "particulas": particulas,
        },
        "cycle": cycle,
        "reduction_pct": red,
        "age_years": age,
        "confidence": confidence,
    }
