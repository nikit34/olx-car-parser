"""Build the per-model SEO valuation blob (Tier-3).

Evergreen per-(brand, model) pages target "quanto vale <model>" search queries.
The Worker can't run the model, so we precompute, per model, the ASKING-price
quantiles (p25/median/p75) over its ACTIVE listings — overall and per year — plus
median mileage, fuel mix, and the (Tier-1) median days-to-sell. The Worker fetches
``models.json`` and renders /preco/{slug} + /precos + /sitemap.xml.

Honesty (see feedback_quality_over_coverage): every median ships WITH its p25/p75
range (never a lone number); prices are ASKING prices on live listings ("pedido"),
not closed-sale prices; per-year cells are gated on sample size, thin years merged
into bands or honestly omitted (yrs_thin). Predicted/fair bands stay in
valuations.json for the per-listing /avaliar tool — this file is asking-price only.

slugify() MUST stay byte-identical to the JS slugify() in
flipper-club/src/templates.js (paired-comment pact, like
valuations.py::_import_flags ↔ templates.js::importInfo). If they drift, the
Worker's /preco/{slug} lookup and the live-deal bridge silently miss.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Callable

import numpy as np
import pandas as pd

from src.analytics.valuations import _i

# Page-worthy floor + per-year-cell gate (validated: 271 models clear >=20).
MIN_MODEL_N = 20
MIN_YEAR_N = 5
MAX_YEAR_ROWS = 25          # cap emitted year rows (most recent) to bound size

# ── GBM fair-value display guards ────────────────────────────────────────────
# The asking quantiles above always ship. The MODEL's fair-value band (gl/gm/gh)
# only ships for a cell when ALL guards below pass — else we fall back to
# asking-only. Rationale (validated 2026-07-01 against the shipped v13 model,
# see project_pseo_feasibility / project_price_model_cheap_tail_audit):
#   • cheap tail (<€5k): the model over-predicts (coarse-baseline collapse).
#   • high end (>€45k): the model SATURATES — distinct exotics collapse to one
#     value (e.g. Porsche 911, BMW M4 both floored at GBM €59900), unreliable.
#   • heterogeneous groups (fair estimate outside the asking IQR context) — the
#     model can't be trusted where it disagrees wildly with 20+ real asks.
# Publishing a wrong number on a public SEO page is a trust liability, so we
# drop rather than mask (feedback_quality_over_coverage). ~68% of model pages
# clear these guards; the rest show asking-only exactly as before.
GBM_ASK_MIN = 5000          # €: below this the model over-predicts the cheap tail
GBM_ASK_MAX = 45000         # €: above this the model saturates (ceiling artifact)
GBM_RATIO_LO = 0.70         # suppress if GBM median < 0.70× the asking median
GBM_RATIO_HI = 1.40         # or > 1.40× the asking median (implausible disagreement)
GBM_CTX_LO = 0.85           # GBM median must sit >= asking P25 × 0.85 …
GBM_CTX_HI = 1.10           # … and <= asking P75 × 1.10 (consistent with real asks)
GBM_MIN_SPEC_FILL = 0.5     # need >=2 of 4 discriminative specs (== decision._MIN_SPEC_FILL)


def _mode(s: pd.Series | None):
    """Most-common non-null value, or None (also None when the column is absent)."""
    if s is None:
        return None
    v = s.dropna()
    return v.mode().iloc[0] if len(v) else None


def _med(s: pd.Series | None):
    """Median of numeric-coerced non-null values, or None (None when absent)."""
    if s is None:
        return None
    v = pd.to_numeric(s, errors="coerce").dropna()
    return float(v.median()) if len(v) else None


def _group_profile(grp: pd.DataFrame) -> dict:
    """Representative (modal/median) spec profile for a (brand, model) group.

    A synthetic "typical car of this model" the GBM can value — dominant fuel/
    gearbox/generation/segment + median engine/power/seats/mileage/year. Filling
    the discriminative specs (mileage, engine_cc, horsepower, fuel) pushes
    spec_fill to ~1.0 so the model prices off real attributes, not a baseline.
    """
    return {
        "year": (int(round(_med(grp.get("year")))) if _med(grp.get("year")) is not None else None),
        "mileage_km": _med(grp.get("mileage_km")),
        "engine_cc": _med(grp.get("engine_cc")),
        "horsepower": _med(grp.get("horsepower")),
        "seats": _med(grp.get("seats")),
        "fuel_type": _mode(grp.get("fuel_type")),
        "transmission": _mode(grp.get("transmission")),
        "generation": _mode(grp.get("generation")),
        "segment": _mode(grp.get("segment")),
        "sub_model": _mode(grp.get("sub_model")),
        "trim_level": _mode(grp.get("trim_level")),
        "district": _mode(grp.get("district")),
    }


def _gbm_passes(pred: float, spec_fill: float, vocab_ok: bool,
                ask: float, ap25: float, ap75: float) -> bool:
    """Whether a GBM fair-value estimate is trustworthy enough to publish."""
    if not vocab_ok or ask is None or ask <= 0 or pred is None or pred <= 0:
        return False
    if not (spec_fill is not None and spec_fill >= GBM_MIN_SPEC_FILL):
        return False
    if not (GBM_ASK_MIN <= ask <= GBM_ASK_MAX):
        return False
    ratio = pred / ask
    if not (GBM_RATIO_LO <= ratio <= GBM_RATIO_HI):
        return False
    if ap25 and ap75 and not (pred >= ap25 * GBM_CTX_LO and pred <= ap75 * GBM_CTX_HI):
        return False
    return True


def _cell_year(y) -> int | None:
    """Representative year for a cell: the year itself, or a band's latest year."""
    if isinstance(y, (int, np.integer)):
        return int(y)
    m = re.match(r"^(\d{4})-(\d{4})$", str(y))
    return int(m.group(2)) if m else None


def slugify(s: str) -> str:
    """NFD-strip accents → lowercase → non-alnum runs to '-' → trim.
    LOCK-STEP with flipper-club/src/templates.js::slugify — keep byte-identical."""
    s = (s or "")
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn").lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s


def _quantiles(prices: pd.Series):
    """Return (p25, median, p75) as ints over non-null asking prices, or None."""
    v = pd.to_numeric(prices, errors="coerce").dropna()
    if len(v) == 0:
        return None
    q = np.percentile(v, [25, 50, 75])
    return _i(q[0]), _i(q[1]), _i(q[2])


def _year_cells(grp: pd.DataFrame) -> tuple[list[dict], int]:
    """Per-year asking-price cells (year DESC), gating thin years and merging
    consecutive sub-gate years into 2+-year bands. Returns (cells, yrs_thin)."""
    yr = pd.to_numeric(grp.get("year"), errors="coerce")
    g = grp.assign(_y=yr).dropna(subset=["_y"])
    g = g[(g["_y"] >= 1980) & (g["_y"] <= 2026)]
    if g.empty:
        return [], 0
    by_year = {int(y): sub for y, sub in g.groupby("_y")}
    years_asc = sorted(by_year)

    cells: list[dict] = []
    band: list[int] = []          # accumulator of consecutive sub-gate years
    yrs_thin = 0

    def flush_band():
        nonlocal yrs_thin
        if not band:
            return
        rows = pd.concat([by_year[y] for y in band])
        if len(rows) >= MIN_YEAR_N and len(band) >= 2:
            q = _quantiles(rows["price_eur"])
            if q:
                cells.append({"y": f"{band[0]}-{band[-1]}", "n": int(len(rows)),
                              "fl": q[0], "fm": q[1], "fh": q[2]})
            else:
                yrs_thin += len(band)
        else:
            yrs_thin += len(band)
        band.clear()

    for y in years_asc:
        sub = by_year[y]
        if len(sub) >= MIN_YEAR_N:
            flush_band()
            q = _quantiles(sub["price_eur"])
            if not q:
                yrs_thin += 1
                continue
            cell = {"y": y, "n": int(len(sub)), "fl": q[0], "fm": q[1], "fh": q[2]}
            kmv = pd.to_numeric(sub.get("mileage_km"), errors="coerce").dropna()
            if len(kmv) >= MIN_YEAR_N:
                cell["km"] = _i(kmv.median())
            cells.append(cell)
        else:
            band.append(y)
            # close a band as soon as it reaches the sample floor
            if sum(len(by_year[b]) for b in band) >= MIN_YEAR_N and len(band) >= 2:
                flush_band()
    flush_band()

    cells.sort(key=lambda c: (int(str(c["y"]).split("-")[-1])), reverse=True)
    return cells[:MAX_YEAR_ROWS], yrs_thin


def build_model_pages(
    listings: pd.DataFrame,
    sell_speed: pd.DataFrame | None = None,
    valuator: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> dict:
    """Return ``{"v":1, "models": {slug: {...}}}`` for models with >=MIN_MODEL_N
    active, asking-priced listings.

    When ``valuator`` is given (a callable taking a configs DataFrame → the
    ``price_model.value_configs`` output), each page and per-year cell also gets
    the MODEL's fair-value band (``gl``/``gm``/``gh``) — but only where it passes
    the cheap-tail/ceiling/agreement guards (``_gbm_passes``); otherwise the cell
    stays asking-only. ``valuator`` stays a callable so this module needs no
    LightGBM import (it's built host-side; the Worker only reads the blob).
    """
    models: dict[str, dict] = {}
    profiles: dict[str, dict] = {}     # slug → representative spec profile (for GBM)
    if listings.empty:
        return {"v": 1, "models": models}

    active = (listings[listings["is_active"] == True]  # noqa: E712
              if "is_active" in listings.columns else listings).copy()
    active = active[pd.to_numeric(active.get("price_eur"), errors="coerce").notna()]
    active = active[active.get("brand").notna() & active.get("model").notna()]
    if active.empty:
        return {"v": 1, "models": models}

    sell_lookup: dict[tuple, tuple[int, int]] = {}
    if sell_speed is not None and not sell_speed.empty:
        sell_lookup = {(r.brand, r.model): (int(r.sell_days), int(r.sell_n))
                       for r in sell_speed.itertuples()}

    for (brand, model), grp in active.groupby(["brand", "model"]):
        if len(grp) < MIN_MODEL_N:
            continue
        slug = slugify(f"{brand}-{model}")
        if not slug or slug in models:   # zero collisions verified; skip dup defensively
            continue
        q = _quantiles(grp["price_eur"])
        if not q:
            continue
        yr_cells, yrs_thin = _year_cells(grp)
        years = pd.to_numeric(grp.get("year"), errors="coerce").dropna()
        rec: dict = {
            "b": str(brand), "m": str(model), "n": int(len(grp)),
            "fl": q[0], "fm": q[1], "fh": q[2],
        }
        kmv = pd.to_numeric(grp.get("mileage_km"), errors="coerce").dropna()
        if len(kmv):
            rec["kmm"] = _i(kmv.median())
        if len(years):
            rec["y0"], rec["y1"] = int(years.min()), int(years.max())
        # top-3 fuel mix (already canonicalised upstream)
        fu = grp.get("fuel_type")
        if fu is not None:
            vc = fu.dropna().value_counts()
            tot = int(vc.sum())
            if tot:
                rec["fu"] = [[str(k), round(int(v) / tot, 2)] for k, v in vc.head(3).items()]
        sd = sell_lookup.get((brand, model))
        if sd is not None:
            rec["sd"], rec["sn"] = sd[0], sd[1]
        if yr_cells:
            rec["yr"] = yr_cells
        if yrs_thin:
            rec["yt"] = int(yrs_thin)
        models[slug] = {k: v for k, v in rec.items() if v is not None}
        if valuator is not None:
            profiles[slug] = _group_profile(grp)

    if valuator is not None and models:
        _apply_gbm_bands(models, profiles, valuator)

    return {"v": 1, "models": models}


def _apply_gbm_bands(
    models: dict[str, dict],
    profiles: dict[str, dict],
    valuator: Callable[[pd.DataFrame], pd.DataFrame],
) -> None:
    """Batch-value one synthetic config per page (overall) + per year cell, then
    attach the MODEL fair band (gl/gm/gh) wherever ``_gbm_passes`` — in place.

    One valuator call for the whole corpus (~2k rows) keeps the CI build cheap.
    Keys encode where each result lands: ``"{slug}"`` for the page, ``"{slug}|{y}"``
    for a year cell (y is the cell's original label, int or "y0-y1" band)."""
    rows: list[dict] = []
    keys: list[str] = []

    def _cfg(prof: dict, brand: str, model: str, year, mileage) -> dict:
        c = dict(prof)
        c["brand"], c["model"] = brand, model
        if year is not None:
            c["year"] = year
        if mileage is not None:
            c["mileage_km"] = mileage
        return c

    for slug, rec in models.items():
        prof = profiles.get(slug)
        if not prof:
            continue
        brand, model = rec["b"], rec["m"]
        # Page-level config: representative specs + median year/mileage.
        rows.append(_cfg(prof, brand, model, prof.get("year"), rec.get("kmm", prof.get("mileage_km"))))
        keys.append(slug)
        # Per-year-cell configs: cell's year + cell's median mileage.
        for cell in rec.get("yr", []):
            cy = _cell_year(cell.get("y"))
            if cy is None:
                continue
            rows.append(_cfg(prof, brand, model, cy, cell.get("km", prof.get("mileage_km"))))
            keys.append(f"{slug}|{cell.get('y')}")

    if not rows:
        return
    configs = pd.DataFrame(rows, index=pd.Index(keys, name="k"))
    valued = valuator(configs)   # predicted_price/fair_price_low/fair_price_high/spec_fill/vocab_ok

    def _band(key: str):
        try:
            r = valued.loc[key]
        except KeyError:
            return None
        # A duplicate key (shouldn't happen — slugs unique, cell labels unique
        # per slug) would yield a frame; take the first row defensively.
        if isinstance(r, pd.DataFrame):
            r = r.iloc[0]
        return r

    for slug, rec in models.items():
        if slug not in profiles:
            continue
        r = _band(slug)
        if r is not None and _gbm_passes(
            float(r["predicted_price"]), float(r["spec_fill"]), bool(r["vocab_ok"]),
            rec.get("fm"), rec.get("fl"), rec.get("fh"),
        ):
            rec["gl"] = _i(r["fair_price_low"])
            rec["gm"] = _i(r["predicted_price"])
            rec["gh"] = _i(r["fair_price_high"])
        for cell in rec.get("yr", []):
            cr = _band(f"{slug}|{cell.get('y')}")
            if cr is not None and _gbm_passes(
                float(cr["predicted_price"]), float(cr["spec_fill"]), bool(cr["vocab_ok"]),
                cell.get("fm"), cell.get("fl"), cell.get("fh"),
            ):
                cell["gl"] = _i(cr["fair_price_low"])
                cell["gm"] = _i(cr["predicted_price"])
                cell["gh"] = _i(cr["fair_price_high"])
