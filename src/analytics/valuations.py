"""Build the public "value any listing" lookup blob (Tier-2).

The flipper-club Worker can't run the price model, so the /avaliar paste-a-link
tool looks up a precomputed fair-price band by ``olx_id``. This module turns the
active listings ⋈ price-model predictions into a compact JSON map the Worker
fetches once (≈0.9 MB gzipped for ~18k cars) and caches at the edge.

Per car we ship only what a verdict page needs — asking price, fair band, a few
specs, the (precomputed) imported-car flag, and the segment sell-speed. We do
NOT ship descriptions (the import flag is computed here, where the text lives,
so the Worker never needs it). The Worker derives the verdict (below/within/
above the fair band) and the "poupas/pagas a mais" framing at render time.

Import-detection regex is intentionally kept in lock-step with the JS detector
in flipper-club/src/templates.js (importInfo). Keep both in sync.
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd


def _strip_accents(s: str) -> str:
    s = (s or "").lower()
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


# Mirrors flipper-club/src/templates.js IMPORT_POS / IMPORT_NEG / IMPORT_LEGAL.
_IMPORT_POS = re.compile(
    r"\b(importad[ao]s?|importacao|nacionaliz\w*|legaliza(?:r|cao|do|da)|por\s+legalizar"
    r"|matricul(?:ar|a(?:do|da)?\s+(?:na|nos|em)\s+(?:alemanha|franca|belgica|holanda|espanha|italia|suica))"
    r"|matricula\s+(?:nl|de|be|fr|es|it|alem\w*|estrangeira|holandesa|alema|francesa|belga)"
    r"|ainda\s+(?:com|por)\s+matricula\s+estrangeira|vindo\s+d[ao]\s+estrangeiro)\b"
)
_IMPORT_NEG = re.compile(
    r"matricula\s+(?:portuguesa|nacional)|nacional\s+desde\s+novo|sempre\s+(?:em\s+)?portugal"
    r"|documentacao\s+(?:regularizada|portuguesa)|matriculado\s+em\s+portugal"
    r"|nao\s+(?:e\s+)?importad|sem\s+importacao"
)
_IMPORT_LEGAL = re.compile(
    r"\bja\s+(?:legalizad[oa]|nacionalizad[oa])|legalizacao\s+(?:feita|concluida|paga)|isv\s+pag"
)


def _import_flags(title: str, description: str) -> tuple[int, int]:
    hay = _strip_accents(title) + " " + _strip_accents(description)
    if not (_IMPORT_POS.search(hay) and not _IMPORT_NEG.search(hay)):
        return 0, 0
    return 1, (1 if _IMPORT_LEGAL.search(hay) else 0)


def _i(v):
    try:
        return int(v) if v is not None and pd.notna(v) else None
    except (TypeError, ValueError):
        return None


def _s(v):
    # Coerce to a clean string or None. Critically, a pandas-missing value is a
    # float NaN (truthy!), so `nan or None` would leak `nan` and json.dumps would
    # emit the literal `NaN` — valid for Python's json.load but NOT valid JSON,
    # so the Worker's JSON.parse throws and the whole blob is unusable.
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip()
    return s or None


def build_valuations(listings: pd.DataFrame, predictions: pd.DataFrame,
                     sell_speed: pd.DataFrame | None = None) -> dict:
    """Return ``{"v":1, "cars": {olx_id: {...}}}`` for active, priced listings.

    - ``listings``: enriched listings DataFrame (needs olx_id, is_active, title,
      description, brand, model, year, mileage_km, fuel_type, price_eur, city).
    - ``predictions``: per-olx_id ``predicted_price`` + ``fair_price_{low,high}``.
    - ``sell_speed``: optional output of ``compute_sell_speed_by_model`` for the
      median days-to-sell per (brand, model).
    """
    cars: dict[str, dict] = {}
    if listings.empty or predictions.empty:
        return {"v": 1, "cars": cars}

    pred = predictions.set_index("olx_id")
    sell_lookup: dict[tuple, int] = {}
    if sell_speed is not None and not sell_speed.empty:
        sell_lookup = {(r.brand, r.model): int(r.sell_days)
                       for r in sell_speed.itertuples()}

    active = (listings[listings["is_active"] == True]  # noqa: E712
              if "is_active" in listings.columns else listings)
    for r in active.itertuples():
        oid = getattr(r, "olx_id", None)
        if oid is None or oid not in pred.index:
            continue
        prow = pred.loc[oid]
        fm = _i(prow.get("predicted_price"))
        price = _i(getattr(r, "price_eur", None))
        if fm is None or price is None:
            continue
        imp, leg = _import_flags(getattr(r, "title", "") or "",
                                 getattr(r, "description", "") or "")
        title = _s(getattr(r, "title", None))
        rec = {
            "t": title[:90] if title else None,
            "y": _i(getattr(r, "year", None)),
            "km": _i(getattr(r, "mileage_km", None)),
            "fu": _s(getattr(r, "fuel_type", None)),
            "p": price,
            "fl": _i(prow.get("fair_price_low")),
            "fm": fm,
            "fh": _i(prow.get("fair_price_high")),
            "ct": _s(getattr(r, "city", None)) or _s(getattr(r, "district", None)),
        }
        if imp:
            rec["imp"] = 1
            if leg:
                rec["il"] = 1
        sd = sell_lookup.get((getattr(r, "brand", None), getattr(r, "model", None)))
        if sd is not None:
            rec["sd"] = sd
        # Drop None values to keep the blob small.
        cars[str(oid)] = {k: v for k, v in rec.items() if v is not None}

    return {"v": 1, "cars": cars}
