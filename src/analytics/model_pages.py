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

import numpy as np
import pandas as pd

from src.analytics.valuations import _i

# Page-worthy floor + per-year-cell gate (validated: 271 models clear >=20).
MIN_MODEL_N = 20
MIN_YEAR_N = 5
MAX_YEAR_ROWS = 25          # cap emitted year rows (most recent) to bound size


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


def build_model_pages(listings: pd.DataFrame, sell_speed: pd.DataFrame | None = None) -> dict:
    """Return ``{"v":1, "models": {slug: {...}}}`` for models with >=MIN_MODEL_N
    active, asking-priced listings."""
    models: dict[str, dict] = {}
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

    return {"v": 1, "models": models}
