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

Beyond the per-year cells, each model also carries FACET cells where the sample
allows: ``fx`` (fuel: diesel/gasolina/GPL), ``tx`` (gearbox: manual/automática)
and ``dt`` (district), plus a top-level ``districts`` rollup for the cross-model
geo pages. Same gate as everything else - a facet with a thin sample is absent,
not estimated.

Days on market arrive ready-made from ``analytics.liquidity`` and ride along as
``lq``: how many of this model's listings leave OLX inside 30, 60 and 90 days,
the median and quartiles behind it, the same cuts by price band, age and
district, and the share that came back later as a new listing (the only hard
evidence we have that a disappearance was not a sale). A model carries the key
only above that module's page floor, which is what tells the Worker a
/liquidez/{slug} page exists. The market-wide row is written next to
``districts`` as ``lqm`` by the build script.

Where a model has enough of both sides of an either/or to fit them separately,
it also carries the mileage-controlled retention duel from ``retention_duel``:
``dg`` (diesel vs gasolina) and ``cx`` (caixa manual vs automática) — the
per-year rate of each side and the asking premium at concrete ages, each with
the interval it was measured at.

slugify() MUST stay byte-identical to the JS slugify() in
flipper-club/src/templates.js (paired-comment pact, like
valuations.py::_import_flags ↔ templates.js::importInfo). If they drift, the
Worker's /preco/{slug} lookup and the live-deal bridge silently miss.
"""

from __future__ import annotations

import re
import time
import unicodedata
from typing import Callable

import numpy as np
import pandas as pd

from src.analytics.retention_duel import all_duels
from src.analytics.valuations import _i

# Page-worthy floor + per-year-cell gate (validated: 271 models clear >=20).
MIN_MODEL_N = 20
MIN_YEAR_N = 5
RETIRE_MODEL_N = 14
MIN_YEAR_PAGE_N = 10
RETIRE_YEAR_PAGE_N = 7
RETIRE_FACET_N = 11
MAX_YEAR_ROWS = 25          # cap emitted year rows (most recent) to bound size

# ── Facet cells (fuel, district) ─────────────────────────────────────────────
# Higher floor than a year row: a facet page is a page, and "quanto vale um Golf
# diesel" is only worth answering separately if the diesel sample can carry its
# own median. 15 is where the facet median stops crossing the model median in
# the wrong direction on the current corpus.
MIN_FACET_N = 15
MAX_DISTRICT_ROWS = 8       # per model, deepest districts only
MIN_MATCH_YEAR_N = 5
MIN_MATCH_YEARS = 3
_DR_MIN_LISTINGS = 10
_RAW_GAP_LO = 0.70
_RAW_GAP_HI = 1.40
_FACET_SOLO_MAX_SHARE = 0.85

# Only three fuels get facets. The raw fuel_type vocabulary carries near-
# duplicates that would slug into competing pages for the same thing
# ("Hibrido Plug-in" vs "Plug-In", "Electrico" vs "Eletrico"), and the query
# cluster that actually exists is diesel-vs-gasolina anyway. The full mix still
# ships in ``fu`` for the chart; this is only about which facets get a URL.
_FUEL_FACETS = {"diesel": "Diesel", "gasolina": "Gasolina", "gpl": "GPL"}

_TRANSMISSION_FACETS = {"manual": "Manual", "automatica": "Automática"}

# National floor for a district to get its own cross-model page.
MIN_DISTRICT_N = 100
MAX_DISTRICT_TOP_MODELS = 40

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

# Columns fed to the valuator when pricing a page's REAL listings. Everything
# else is NaN-filled by price_model._prepare_X; brand/model also drive the vocab
# gate. Pricing the actual configs (not one modal archetype) is what stops
# distinct models collapsing onto the same coarse GBM bucket — the page number
# is the MEDIAN of per-listing fair values, riding each model's real spec spread.
_GBM_COLS = ("brand", "model", "year", "mileage_km", "engine_cc", "horsepower",
             "seats", "fuel_type", "transmission", "generation", "segment",
             "sub_model", "trim_level", "district")


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


def _cell_year_range(y) -> tuple[int, int] | None:
    """Inclusive (lo, hi) year span a cell covers — a single year or a band —
    used to reselect that cell's real listings for per-cell GBM aggregation."""
    if isinstance(y, (int, np.integer)):
        return int(y), int(y)
    m = re.match(r"^(\d{4})-(\d{4})$", str(y))
    return (int(m.group(1)), int(m.group(2))) if m else None


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


def _year_cells(grp: pd.DataFrame, keep_pages: set | None = None) -> tuple[list[dict], int]:
    """Per-year asking-price cells (year DESC), gating thin years and merging
    consecutive sub-gate years into 2+-year bands. Returns (cells, yrs_thin).

    ``keep_pages`` are the years that had their OWN URL last build. They keep the
    ``pg`` flag down to RETIRE_YEAR_PAGE_N instead of MIN_YEAR_PAGE_N, because the
    page set is a function of live inventory: a year sitting on that floor flips
    between builds and takes an indexed, ranking URL with it — measured at 12% of
    impression-earning URLs over six days, with 38% of year pages within three
    listings of the floor. Entry stays where it was; only leaving got harder.

    There is deliberately no hysteresis on the CELL floor. A row below
    MIN_YEAR_N carries no URL, so nothing can churn out of the index; retaining
    it only breaks the band it would have merged into and strands the neighbour,
    which omits more years than it saves. RETIRE_YEAR_PAGE_N sits above
    MIN_YEAR_N, so a retained page always clears the cell floor on its own.

    ``pg`` is what the Worker reads to decide a /preco/{slug}/{ano} URL exists.
    It has to be decided here, not there: the Worker has no memory of the
    previous build. A cell without the key falls back to the Worker's own
    MIN_YEAR_PAGE_N, which is what an older blob gets."""
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

    kept_pages = keep_pages or set()
    for y in years_asc:
        sub = by_year[y]
        if len(sub) >= MIN_YEAR_N:
            flush_band()
            q = _quantiles(sub["price_eur"])
            if not q:
                yrs_thin += 1
                continue
            cell = {"y": y, "n": int(len(sub)), "fl": q[0], "fm": q[1], "fh": q[2]}
            if len(sub) >= MIN_YEAR_PAGE_N or (y in kept_pages and len(sub) >= RETIRE_YEAR_PAGE_N):
                cell["pg"] = 1
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
    pages = [c for c in cells if c.get("pg")]
    rest = [c for c in cells if not c.get("pg")]
    kept = pages + rest[:max(0, MAX_YEAR_ROWS - len(pages))]
    kept.sort(key=lambda c: (int(str(c["y"]).split("-")[-1])), reverse=True)
    return kept, yrs_thin


def _fuel_facet_key(value) -> str | None:
    """Canonical facet key for a raw fuel_type, or None if it gets no facet."""
    k = slugify(str(value or ""))
    return k if k in _FUEL_FACETS else None


def _transmission_facet_key(value) -> str | None:
    """Canonical facet key for a raw transmission, or None if it gets no facet."""
    k = slugify(str(value or ""))
    return k if k in _TRANSMISSION_FACETS else None


def _year_medians(sub: pd.DataFrame) -> dict[int, tuple[float, int]]:
    """{year: (median asking price, n)} over years carrying MIN_MATCH_YEAR_N."""
    yr = pd.to_numeric(sub.get("year"), errors="coerce")
    out: dict[int, tuple[float, int]] = {}
    if yr is None:
        return out
    for y, rows in sub.assign(_y=yr).dropna(subset=["_y"]).groupby("_y"):
        if len(rows) < MIN_MATCH_YEAR_N:
            continue
        v = pd.to_numeric(rows["price_eur"], errors="coerce").dropna()
        if v.empty:
            continue
        out[int(y)] = (float(v.median()), int(len(rows)))
    return out


def _weighted_median(points: list[tuple[float, int]]) -> float | None:
    if not points:
        return None
    pts = sorted(points)
    total = sum(w for _, w in pts)
    acc = 0
    for value, w in pts:
        acc += w
        if acc * 2 >= total:
            return value
    return pts[-1][0]


def _matched_ratio(a: dict[int, tuple[float, int]],
                   b: dict[int, tuple[float, int]]) -> list | None:
    """[ratio, shared_years] comparing two samples year by year, or None.

    The raw medians of two facets cannot be subtracted: they describe different
    cars. A Golf automatic asks 4.3x a Golf manual, which reads as "the gearbox
    is worth 330%" and is not what the corpus says - the automatics on sale are
    simply much newer. So the ratio here is the sample-weighted median of the
    PER-YEAR ratios (a five-listing year cannot outvote a forty-listing one),
    which holds the age mix fixed by construction. Same method as the
    model-vs-model gap on the comparison pages. None below MIN_MATCH_YEARS
    shared years, so the page is left with no percentage to print rather than a
    percentage that means the age difference."""
    shared = [(a[y][0] / b[y][0], min(a[y][1], b[y][1]))
              for y in a.keys() & b.keys() if b[y][0] > 0]
    if len(shared) < MIN_MATCH_YEARS:
        return None
    r = _weighted_median(shared)
    return None if r is None else [round(r, 4), len(shared)]


def _year_normalized_ratio(sub: pd.DataFrame, rest: dict[int, tuple[float, int]]) -> list | None:
    """[ratio, listings_used] for a cut against the REST of the model, at equal age.

    ``_matched_ratio`` needs MIN_MATCH_YEARS years carrying MIN_MATCH_YEAR_N on
    BOTH sides, which a 15-35 listing district cell spread over twenty model
    years almost never reaches - so the cells that most need the age control are
    the ones that cannot get it, and they fall back to raw medians whose age mix
    swamps the effect. This estimates the same quantity per LISTING instead of
    per shared year: each listing is divided by the reference median for its own
    registration year, and the median of those ratios is the answer.

    The reference is the model MINUS this cut. Dividing by a median the cut
    itself helped set pulls the answer toward 1 by construction, and capping the
    cut's share of the year does not fix it - at a 50% cap the attenuation was
    still about half the true effect, and 28% of cells landed on exactly 1.0000,
    which the page then printed as "asks the same". Comparing against the rest
    removes the bias outright rather than bounding it. ``_year_medians`` already
    refuses a year with fewer than MIN_MATCH_YEAR_N on the reference side, so a
    year where the cut is nearly everything simply has no reference and does not
    count. None below ``_DR_MIN_LISTINGS`` usable listings.
    """
    if "year" not in sub.columns or "price_eur" not in sub.columns:
        return None
    yr = pd.to_numeric(sub["year"], errors="coerce")
    price = pd.to_numeric(sub["price_eur"], errors="coerce")
    ratios: list[float] = []
    for y, p in zip(yr, price):
        if pd.isna(y) or pd.isna(p) or p <= 0:
            continue
        ref = rest.get(int(y))
        if not ref or ref[0] <= 0:
            continue
        ratios.append(float(p) / ref[0])
    if len(ratios) < _DR_MIN_LISTINGS:
        return None
    return [round(float(np.median(ratios)), 4), len(ratios)]


def _facet_cells(grp: pd.DataFrame, col: str, keyer, labeller,
                 min_n: int, limit: int | None = None,
                 keep_keys: set | None = None) -> list[dict]:
    """Asking-price cells for one facet dimension (fuel, gearbox or district).

    Same shape and same honesty rules as ``_year_cells``: median WITH its
    P25-P75, gated on sample size, and simply absent when the sample is thin —
    never merged with an unrelated bucket to reach the floor.

    Each cell also carries the year-matched ratios ``vsm`` (this facet against
    the whole model) and ``vs`` (against each sibling facet) — see
    ``_matched_ratio`` for why the raw medians may not be compared directly.
    Both are absent where too few model years carry a sample on both sides.
    """
    if col not in grp.columns:
        return []
    keys = grp[col].map(keyer)
    out: list[dict] = []
    by_key: dict[str, dict[int, tuple[float, int]]] = {}
    sub_of: dict[str, pd.DataFrame] = {}
    kept = keep_keys or set()
    for key, sub in grp.assign(_k=keys).dropna(subset=["_k"]).groupby("_k"):
        if len(sub) < (RETIRE_FACET_N if str(key) in kept else min_n):
            continue
        q = _quantiles(sub["price_eur"])
        if not q:
            continue
        cell = {"k": str(key), "lbl": labeller(sub[col].iloc[0]),
                "n": int(len(sub)), "fl": q[0], "fm": q[1], "fh": q[2]}
        kmv = pd.to_numeric(sub.get("mileage_km"), errors="coerce").dropna()
        if len(kmv) >= min_n:
            cell["km"] = _i(kmv.median())
        yrs = pd.to_numeric(sub.get("year"), errors="coerce").dropna()
        if len(yrs):
            cell["y0"], cell["y1"] = int(yrs.min()), int(yrs.max())
        out.append(cell)
        by_key[str(key)] = _year_medians(sub)
        sub_of[str(key)] = sub
    out.sort(key=lambda c: -c["n"])
    out = out[:limit] if limit else out

    parent = _year_medians(grp)
    siblings = {c["k"] for c in out}
    model_median = pd.to_numeric(grp["price_eur"], errors="coerce").dropna().median()
    for cell in out:
        mine = by_key[cell["k"]]
        vsm = _matched_ratio(mine, parent)
        if vsm:
            cell["vsm"] = vsm
        dr = _year_normalized_ratio(sub_of[cell["k"]],
                                    _year_medians(grp.drop(index=sub_of[cell["k"]].index)))
        if dr:
            cell["dr"] = dr
        vs = {}
        for other in siblings:
            if other == cell["k"]:
                continue
            r = _matched_ratio(mine, by_key[other])
            if r:
                vs[other] = r
        if vs:
            cell["vs"] = vs
    if pd.notna(model_median) and model_median > 0:
        def explained(c):
            return ("dr" in c or "vsm" in c or str(c["k"]) in kept
                    or _RAW_GAP_LO <= c["fm"] / model_median <= _RAW_GAP_HI)
        survivors = [c for c in out if explained(c)]
        model_n = len(grp)
        orphaned = (len(survivors) == 1 and len(out) > 1 and col != "district"
                    and model_n and survivors[0]["n"] / model_n >= _FACET_SOLO_MAX_SHARE)
        if not orphaned:
            out = survivors
    return out


def _district_rollup(active: pd.DataFrame, models: dict) -> dict:
    """Cross-model district cut: what a car costs in Porto vs Lisboa.

    Separate from the per-model district cells because the query is different -
    "carros usados Lisboa precos" is about the market, not about one model - and
    because the sample only supports it at the national level for most places.

    The floor is what decides how much of the country has a page. At 200 active
    listings only 13 of the 18 mainland districts cleared it, and the five that
    did not - Vila Real, Beja, Guarda, Portalegre, Braganca, between 107 and 188
    listings - are exactly where a local median is worth most, because there the
    national number is furthest from what a buyer sees. At 100 all 18 have one.

    What those five cannot carry is the per-model table: none of them has more
    than two models with five listings, so ``top`` comes back nearly empty and
    the page has to stand on the district median, its interquartile range and
    the comparison against the rest of the country. The Worker is told that by
    the length of ``top`` and says it in words rather than rendering an empty
    table. The islands stay out on their own numbers (Madeira 63, every Azores
    island under 35), which is the floor working rather than a decision about
    them.
    """
    if "district" not in active.columns:
        return {}
    out: dict[str, dict] = {}
    slug_of = {}
    for slug, rec in models.items():
        slug_of[(rec["b"], rec["m"])] = slug
    _keys = active["district"].map(lambda v: None if pd.isna(v) else (slugify(str(v)) or None))
    for raw, sub in active.assign(_d=_keys).dropna(subset=["_d"]).groupby("_d"):
        if not raw or len(sub) < MIN_DISTRICT_N:
            continue
        q = _quantiles(sub["price_eur"])
        if not q:
            continue
        rec: dict = {"lbl": str(sub["district"].iloc[0]), "n": int(len(sub)),
                     "fl": q[0], "fm": q[1], "fh": q[2]}
        kmv = pd.to_numeric(sub.get("mileage_km"), errors="coerce").dropna()
        if len(kmv):
            rec["kmm"] = _i(kmv.median())
        # Deepest models in this district, but only those that HAVE a page —
        # a link to a model page we never published is a 404 in the making.
        top = []
        for (brand, model), g in sub.groupby(["brand", "model"]):
            slug = slug_of.get((str(brand), str(model)))
            if slug is None or len(g) < MIN_YEAR_N:
                continue
            gq = _quantiles(g["price_eur"])
            if not gq:
                continue
            top.append([slug, int(len(g)), gq[1]])
        top.sort(key=lambda t: -t[1])
        rec["top"] = top[:MAX_DISTRICT_TOP_MODELS]
        out[str(raw)] = rec
    return out


def _published_page_set(published: dict | None) -> dict[str, dict]:
    """What the previous build published, per slug, or empty.

    ``{slug: {"pages": {years}, "fx"/"tx"/"dt": {keys}}}``. Total by construction:
    any shape it does not recognise yields no entry, so a malformed or foreign
    blob degrades to no hysteresis instead of aborting the build.

    A cell without ``pg`` falls back to the same rule the Worker applies when the
    key is absent, so the first build after ``pg`` is introduced already knows
    which years are being served as pages. Without that fallback the retirement
    floor would be dead on exactly the build that needs it.
    """
    out: dict[str, dict] = {}
    if not isinstance(published, dict):
        return out
    models = published.get("models")
    if not isinstance(models, dict):
        return out
    for slug, rec in models.items():
        if not isinstance(rec, dict):
            continue
        cells = rec.get("yr") if isinstance(rec.get("yr"), list) else []
        pages = set()
        for c in cells:
            if not isinstance(c, dict) or not isinstance(c.get("y"), int):
                continue
            n = c.get("n")
            served = c["pg"] if "pg" in c else (isinstance(n, int) and n >= MIN_YEAR_PAGE_N)
            if served:
                pages.add(c["y"])
        entry = {"pages": pages}
        for kind in ("fx", "tx", "dt"):
            arr = rec.get(kind)
            entry[kind] = {str(c["k"]) for c in arr
                           if isinstance(c, dict) and c.get("k") is not None} \
                if isinstance(arr, list) else set()
        out[str(slug)] = entry
    return out


def build_model_pages(
    listings: pd.DataFrame,
    sell_speed: pd.DataFrame | None = None,
    valuator: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    now_year: int | None = None,
    liquidity: dict | None = None,
    published: dict | None = None,
) -> dict:
    """Return ``{"v":1, "models": {slug: {...}}}`` for models with >=MIN_MODEL_N
    active, asking-priced listings.

    ``published`` is the previous build's own output (the models.json currently
    live). A model or year-cell it already carries survives down to
    RETIRE_MODEL_N / RETIRE_YEAR_PAGE_N instead of the entry floor: the page set is a
    function of live inventory, so anything sitting on the floor otherwise flips
    out on a normal dip and takes an indexed, ranking URL with it. Omitted, the
    entry floors apply to everything, which is the pre-hysteresis behaviour.

    ``liquidity`` is ``analytics.liquidity.page_records`` output keyed by
    (brand, model) — the days-on-market curve, its cuts and the relist floor.
    It arrives ready-gated: a model that has one gets a ``lq`` key and, with it,
    a /liquidez page; a model that does not simply has neither.

    When ``valuator`` is given (a callable taking a configs DataFrame → the
    ``price_model.value_configs`` output), each page and per-year cell also gets
    the MODEL's fair-value band (``gl``/``gm``/``gh``) — but only where it passes
    the cheap-tail/ceiling/agreement guards (``_gbm_passes``); otherwise the cell
    stays asking-only. ``valuator`` stays a callable so this module needs no
    LightGBM import (it's built host-side; the Worker only reads the blob).

    ``now_year`` is the reference year ages are counted from in the ``dg`` /
    ``cx`` retention duels; it defaults to the build year.
    """
    now_year = now_year or time.gmtime().tm_year
    models: dict[str, dict] = {}
    page_groups: dict[str, pd.DataFrame] = {}   # slug → its real listings (for GBM)
    if listings.empty:
        return {"v": 1, "models": models}

    active = (listings[listings["is_active"] == True]  # noqa: E712
              if "is_active" in listings.columns else listings).copy()
    active = active[pd.to_numeric(active.get("price_eur"), errors="coerce").notna()]
    active = active[active.get("brand").notna() & active.get("model").notna()]
    if active.empty:
        return {"v": 1, "models": models}

    # Brand and model arrive free-texted, so one car reaches us under several
    # spellings: "SEAT"/"Seat", "MiTo"/"Mito", "C-MAX"/"C-Max", "Mégane"/"Megane".
    # They all slugify to ONE page address. Grouping on the raw pair therefore
    # produced two groups competing for the same slug, and the loop below kept
    # whichever came first and dropped the other outright — publishing a median
    # over part of the sample with nothing on the page to say so. Today the
    # MIN_MODEL_N gate hides it (the minority spelling rarely clears 20 active
    # listings), which is luck, not a guarantee: Alfa Romeo sits at 13 vs 59.
    # Collapse the variants first, so everything downstream sees one pair per
    # slug and the sample is whole.
    _slug_key = (active["brand"].astype(str) + "-" + active["model"].astype(str)).map(slugify)
    _canon: dict[str, tuple[str, str]] = {}
    for _s, _g in active.assign(__s=_slug_key).groupby("__s"):
        if not _s:
            continue
        # The spelling most sellers actually use wins the display label.
        _b, _m = _g.groupby([_g["brand"].astype(str), _g["model"].astype(str)]).size().idxmax()
        _canon[_s] = (str(_b), str(_m))
    if _canon:
        _pairs = [_canon.get(k, (b, m)) for k, b, m
                  in zip(_slug_key, active["brand"].astype(str), active["model"].astype(str))]
        active["brand"] = [p[0] for p in _pairs]
        active["model"] = [p[1] for p in _pairs]

    sell_lookup: dict[tuple, tuple[int, int]] = {}
    if sell_speed is not None and not sell_speed.empty:
        # Keys come from the pre-collapse frame, so map them through the same
        # canon; when both spellings carried stats the deeper sample wins.
        for r in sell_speed.itertuples():
            _b, _m = str(r.brand), str(r.model)
            _key = _canon.get(slugify(f"{_b}-{_m}"), (_b, _m))
            _prev = sell_lookup.get(_key)
            if _prev is None or int(r.sell_n) > _prev[1]:
                sell_lookup[_key] = (int(r.sell_days), int(r.sell_n))

    liq_lookup: dict[tuple, dict] = {}
    for (_b, _m), _rec in (liquidity or {}).items():
        _key = _canon.get(slugify(f"{_b}-{_m}"), (str(_b), str(_m)))
        _prev = liq_lookup.get(_key)
        if _prev is None or int(_rec.get("n", 0)) > int(_prev.get("n", 0)):
            liq_lookup[_key] = _rec

    was_published = _published_page_set(published)
    for (brand, model), grp in active.groupby(["brand", "model"]):
        slug = slugify(f"{brand}-{model}")
        prev = was_published.get(slug)
        if len(grp) < (RETIRE_MODEL_N if prev else MIN_MODEL_N):
            continue
        # Collapsing above makes one group per slug, so this can no longer drop
        # a real sample; it stays as a guard against an empty slug.
        if not slug or slug in models:
            continue
        q = _quantiles(grp["price_eur"])
        if not q:
            continue
        yr_cells, yrs_thin = _year_cells(grp, prev and prev["pages"])
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
        # Facet cells — the Worker turns each into /preco/{slug}/{facet} and
        # simply serves no such page while the key is absent, so shipping this
        # ahead of the pages is safe.
        fx = _facet_cells(grp, "fuel_type", _fuel_facet_key,
                          lambda v: _FUEL_FACETS[_fuel_facet_key(v)], MIN_FACET_N,
                          keep_keys=prev and prev["fx"])
        if fx:
            rec["fx"] = fx
        tx = _facet_cells(grp, "transmission", _transmission_facet_key,
                          lambda v: _TRANSMISSION_FACETS[_transmission_facet_key(v)],
                          MIN_FACET_N, keep_keys=prev and prev["tx"])
        if tx:
            rec["tx"] = tx
        rec.update(all_duels(grp, now_year))
        dt = _facet_cells(grp, "district",
                          lambda v: None if pd.isna(v) else (slugify(str(v)) or None),
                          lambda v: str(v), MIN_FACET_N, limit=MAX_DISTRICT_ROWS,
                          keep_keys=prev and prev["dt"])
        if dt:
            rec["dt"] = dt
        lq = liq_lookup.get((brand, model))
        if lq:
            rec["lq"] = lq
        models[slug] = {k: v for k, v in rec.items() if v is not None}
        if valuator is not None:
            keep = [c for c in _GBM_COLS if c in grp.columns]
            page_groups[slug] = grp[keep].copy()

    if valuator is not None and models:
        _apply_gbm_bands(models, page_groups, valuator)

    doc = {"v": 1, "models": models}
    districts = _district_rollup(active, models)
    if districts:
        doc["districts"] = districts
    return doc


def _apply_gbm_bands(
    models: dict[str, dict],
    page_groups: dict[str, pd.DataFrame],
    valuator: Callable[[pd.DataFrame], pd.DataFrame],
) -> None:
    """Value each page's REAL listings through the GBM and attach the model fair
    band (gl/gm/gh) as the MEDIAN of the per-listing fair values — overall and
    per year cell — wherever ``_gbm_passes``. In place.

    Pricing the actual configs (not one modal archetype) is what stops distinct
    models collapsing onto the same coarse bucket: the median rides each model's
    real mileage/year/trim spread. One valuator call over the whole qualifying
    corpus keeps the CI build to a single LightGBM pass."""
    if not page_groups:
        return
    # One batched valuation of every qualifying listing. Clean RangeIndex so the
    # per-row result aligns back by index (predict_prices preserves the index).
    frames: list[pd.DataFrame] = []
    for slug, grp in page_groups.items():
        if slug not in models or grp is None or grp.empty:
            continue
        f = grp.reset_index(drop=True)
        f["__slug__"] = slug
        frames.append(f)
    if not frames:
        return
    big = pd.concat(frames, ignore_index=True)
    valued = valuator(big.drop(columns="__slug__"))
    big = big.join(valued[["predicted_price", "fair_price_low",
                           "fair_price_high", "spec_fill", "vocab_ok"]])
    big["_yr"] = (pd.to_numeric(big["year"], errors="coerce")
                  if "year" in big.columns else np.nan)

    def _agg(rows: pd.DataFrame):
        """(pred, low, high, spec_fill, vocab_ok) = medians over real listings."""
        p = pd.to_numeric(rows.get("predicted_price"), errors="coerce").dropna()
        if p.empty:
            return None
        return (
            float(p.median()),
            float(pd.to_numeric(rows["fair_price_low"], errors="coerce").median()),
            float(pd.to_numeric(rows["fair_price_high"], errors="coerce").median()),
            float(pd.to_numeric(rows["spec_fill"], errors="coerce").median()),
            bool(rows["vocab_ok"].all()),
        )

    for slug, grp_big in big.groupby("__slug__"):
        rec = models[slug]
        a = _agg(grp_big)
        if a and _gbm_passes(a[0], a[3], a[4], rec.get("fm"), rec.get("fl"), rec.get("fh")):
            rec["gl"], rec["gm"], rec["gh"] = _i(a[1]), _i(a[0]), _i(a[2])
        for cell in rec.get("yr", []):
            span = _cell_year_range(cell.get("y"))
            if span is None:
                continue
            crows = grp_big[(grp_big["_yr"] >= span[0]) & (grp_big["_yr"] <= span[1])]
            ca = _agg(crows)
            if ca and _gbm_passes(ca[0], ca[3], ca[4],
                                  cell.get("fm"), cell.get("fl"), cell.get("fh")):
                cell["gl"], cell["gm"], cell["gh"] = _i(ca[1]), _i(ca[0]), _i(ca[2])
