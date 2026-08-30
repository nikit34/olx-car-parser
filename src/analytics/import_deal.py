"""Does importing this model from Germany beat buying it in Portugal?

Everyone in this market sells the same half-answer: an ISV calculator. An ISV
number on its own decides nothing — the question a buyer actually has is
whether the German price plus the tax plus getting the car here plus the
paperwork lands under what the same car asks in Portugal today. The Portuguese
half of that equation is this project's own corpus, the German half arrives via
``parser.autoscout``, and the tax is ``analytics.isv``. This module is where the
three meet.

The comparison is made **year by year**, never model against model: a German
sample skewed two years newer than the Portuguese one would show a saving that
is just the age difference. Each cell therefore needs both sides of the same
model year, and a cell missing either side is absent rather than estimated.

The ISV is computed per German listing and then taken as a median, for the same
reason ``model_pages`` prices real configurations instead of one archetype: a
median car's CO2 and cilindrada, run through a progressive bracket table, is not
the median of the tax. Only listings carrying a plausible CO2 can be taxed at
all (AutoScout24 leaves that field to the seller), so the ISV sample is smaller
than the price sample and is published with its own count.

What this deliberately does NOT do is quote one number for "legalização". The
fees are a range with named parts (``COST_ITEMS``), the comparison is carried
through at both ends of that range, and the page shows the band. A single
"custos: 900 €" would be the most confident-looking number on the page and the
least defensible.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src.analytics.isv import compute_isv
from src.analytics.model_pages import slugify

MIN_DE_YEAR_N = 10
MIN_PT_YEAR_N = 5
MIN_ISV_N = 6
MIN_CELLS = 2
MAX_CELLS = 14
MIN_YEAR = 1990

COST_SOURCE_YEAR = 2026

COST_ITEMS = (
    {"k": "transporte", "lbl": "Transporte da Alemanha", "lo": 400.0, "hi": 800.0,
     "src": "orçamentos correntes de transportadoras 2026"},
    {"k": "coc", "lbl": "Certificado de Conformidade (COC)", "lo": 160.0, "hi": 370.0,
     "src": "varia com a marca"},
    {"k": "inspecao", "lbl": "Inspeção tipo B", "lo": 93.52, "hi": 93.52,
     "src": "tarifa IMT 2026"},
    {"k": "matricula", "lbl": "Matrícula (modelo 9, IMT)", "lo": 45.0, "hi": 45.0,
     "src": "taxa IMT"},
    {"k": "registo", "lbl": "Registo de propriedade (IRN, online)", "lo": 55.30, "hi": 55.30,
     "src": "Automóvel Online"},
)


def fees_band() -> tuple[float, float]:
    """(low, high) euros of everything except the ISV and the car itself."""
    return (round(sum(i["lo"] for i in COST_ITEMS), 2),
            round(sum(i["hi"] for i in COST_ITEMS), 2))


def _q(values: pd.Series) -> tuple[int, int, int] | None:
    v = pd.to_numeric(values, errors="coerce").dropna()
    if v.empty:
        return None
    p25, p50, p75 = np.percentile(v, [25, 50, 75])
    return int(round(p25)), int(round(p50)), int(round(p75))


def _isv_values(rows: pd.DataFrame, as_of_year: int) -> list[float]:
    """One ISV per German listing that carries enough to be taxed."""
    out: list[float] = []
    for r in rows.itertuples():
        est = compute_isv(
            getattr(r, "co2_g_km", None),
            getattr(r, "engine_cc", None),
            getattr(r, "fuel_type", None),
            getattr(r, "year", None),
            as_of_year=as_of_year,
        )
        if est is not None:
            out.append(float(est["isv_eur"]))
    return out


def _pt_year_medians(pt: pd.DataFrame) -> dict[str, dict[int, tuple[int, int, int | None]]]:
    """{slug: {year: (median asking, n, median km)}} over active priced PT listings.

    Keyed on the slug rather than on the raw pair because the two markets spell
    the same car differently — AutoScout24 has "Citroen Megane" where OLX has
    "Citroën Mégane" — and a join on raw strings silently matches nothing, which
    looks exactly like "Germany does not sell this model".
    """
    if pt is None or pt.empty:
        return {}
    df = pt
    if "is_active" in df.columns:
        df = df[df["is_active"] == True]  # noqa: E712
    price = pd.to_numeric(df.get("price_eur"), errors="coerce")
    year = pd.to_numeric(df.get("year"), errors="coerce")
    km = pd.to_numeric(df.get("mileage_km"), errors="coerce")
    df = df.assign(_p=price, _y=year, _km=km).dropna(subset=["_p", "_y"])
    out: dict[str, dict[int, tuple[int, int, int | None]]] = {}
    for (brand, model, y), rows in df.groupby(["brand", "model", "_y"]):
        if y < MIN_YEAR:
            continue
        slug = slugify(f"{brand}-{model}")
        if not slug:
            continue
        cell = out.setdefault(slug, {})
        prev = cell.get(int(y))
        if prev is not None and prev[1] >= len(rows):
            continue
        kmv = rows["_km"].dropna()
        cell[int(y)] = (int(round(float(rows["_p"].median()))), int(len(rows)),
                        int(round(float(kmv.median()))) if len(kmv) else None)
    return out


def build_import_pages(pt_listings: pd.DataFrame, de_listings: pd.DataFrame,
                       now_year: int | None = None) -> dict:
    """``{"v":1, "costs": {...}, "models": {slug: rec}}`` for the /importar pages.

    A model reaches ``models`` only with ``MIN_CELLS`` model-years that have a
    German sample, a Portuguese sample and a taxable subsample on the German
    side. Everything else — a model Germany sells and we do not, a year we have
    and Germany does not — is simply absent.
    """
    now_year = now_year or time.gmtime().tm_year
    lo_fees, hi_fees = fees_band()
    doc: dict = {"v": 1, "models": {},
                 "costs": {"items": [dict(i) for i in COST_ITEMS],
                           "lo": lo_fees, "hi": hi_fees, "year": COST_SOURCE_YEAR}}
    if de_listings is None or de_listings.empty:
        return doc
    pt_med = _pt_year_medians(pt_listings)
    if not pt_med:
        return doc

    de = de_listings.copy()
    de["_p"] = pd.to_numeric(de.get("price_eur"), errors="coerce")
    de["_y"] = pd.to_numeric(de.get("year"), errors="coerce")
    de = de.dropna(subset=["_p", "_y"])
    de = de[(de["_p"] > 0) & (de["_y"] >= MIN_YEAR)]
    if de.empty:
        return doc

    de["_slug"] = [slugify(f"{b}-{m}") for b, m in zip(de["brand"], de["model"])]
    for slug, grp in de.groupby("_slug"):
        years = pt_med.get(slug)
        if not slug or not years:
            continue
        brand = str(grp["brand"].iloc[0])
        model = str(grp["model"].iloc[0])
        cells = []
        for y, rows in grp.groupby("_y"):
            year = int(y)
            pt_cell = years.get(year)
            if pt_cell is None or pt_cell[1] < MIN_PT_YEAR_N or len(rows) < MIN_DE_YEAR_N:
                continue
            q = _q(rows["_p"])
            if not q:
                continue
            isvs = _isv_values(rows, now_year)
            if len(isvs) < MIN_ISV_N:
                continue
            isv_med = int(round(float(np.median(isvs))))
            landed_lo = q[1] + isv_med + lo_fees
            landed_hi = q[1] + isv_med + hi_fees
            pt_price = pt_cell[0]
            de_km = pd.to_numeric(rows.get("mileage_km"), errors="coerce").dropna()
            cell = {
                "y": year, "nde": int(len(rows)), "npt": int(pt_cell[1]),
                "dl": q[0], "dm": q[1], "dh": q[2],
                "isv": isv_med, "isvn": len(isvs),
                "ll": int(round(landed_lo)), "lh": int(round(landed_hi)),
                "ptm": pt_price,
                "gl": int(round(pt_price - landed_hi)),
                "gh": int(round(pt_price - landed_lo)),
            }
            if len(de_km):
                cell["dkm"] = int(round(float(de_km.median())))
            if pt_cell[2] is not None:
                cell["ptkm"] = pt_cell[2]
            cells.append(cell)
        if len(cells) < MIN_CELLS:
            continue
        cells.sort(key=lambda c: -c["y"])
        cells = cells[:MAX_CELLS]
        wins = [c for c in cells if c["gl"] > 0]
        rec = {
            "b": str(brand), "m": str(model),
            "nde": int(sum(c["nde"] for c in cells)),
            "npt": int(sum(c["npt"] for c in cells)),
            "yr": cells,
            "wins": len(wins),
            "med_gap": int(round(float(np.median([c["gl"] for c in cells])))),
            "isv_med": int(round(float(np.median([c["isv"] for c in cells])))),
        }
        km_pairs = [(c["dkm"], c["ptkm"]) for c in cells
                    if c.get("dkm") and c.get("ptkm")]
        if km_pairs:
            rec["km_gap"] = round(float(np.median([d / p for d, p in km_pairs if p])) - 1, 3)
        doc["models"][slug] = rec
    return doc
