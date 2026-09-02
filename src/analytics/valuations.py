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

_PRICE_TRACK_MAX = 6


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


def _import_flags(title: str, description: str, origin: str | None = None) -> tuple[int, int]:
    # The structured `origin` field reinforces both sides: "imported" is a
    # positive even if the text is silent; "national" clears a text
    # false-positive. Text still supplies the catch-all + legalized nuance.
    # LOCK-STEP with flipper-club/src/templates.js::importInfo.
    hay = _strip_accents(title) + " " + _strip_accents(description)
    pos = (origin == "imported") or bool(_IMPORT_POS.search(hay))
    neg = (origin == "national") or bool(_IMPORT_NEG.search(hay))
    if not (pos and not neg):
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


def _price_track(snapshots: pd.DataFrame | None, now: pd.Timestamp,
                 wanted: set[str]) -> dict[str, list[list[int]]]:
    """Per-listing price track as ``[[days_ago, price], ...]``, oldest first.

    Snapshots are written on change only, so a listing that never moved has a
    single row and is left out entirely — the track is shipped precisely for
    the ads whose seller has already come down, which is the one fact about an
    ad that the ad's own page does not show.
    """
    if snapshots is None or snapshots.empty or not wanted:
        return {}
    need = ["olx_id", "price_eur", "scraped_at"]
    if not set(need).issubset(snapshots.columns):
        return {}
    snap = snapshots[need].dropna()
    snap = snap[snap["olx_id"].astype(str).isin(wanted)]
    if snap.empty:
        return {}
    snap = snap.copy()
    snap["olx_id"] = snap["olx_id"].astype(str)
    snap["scraped_at"] = pd.to_datetime(snap["scraped_at"], errors="coerce", utc=True)
    snap = snap.dropna(subset=["scraped_at"])
    counts = snap["olx_id"].value_counts()
    snap = snap[snap["olx_id"].isin(counts[counts >= 2].index)]
    if snap.empty:
        return {}
    snap = snap.sort_values(["olx_id", "scraped_at"])
    snap = snap.groupby("olx_id", sort=False).tail(_PRICE_TRACK_MAX)
    days = (now - snap["scraped_at"]).dt.days.astype("int64").to_numpy()
    prices = snap["price_eur"].to_numpy()
    ids = snap["olx_id"].to_numpy()
    out: dict[str, list[list[int]]] = {}
    for oid, day, price in zip(ids, days, prices):
        if day < 0:
            continue
        value = _i(price)
        if value is None:
            continue
        pts = out.setdefault(oid, [])
        if pts and pts[-1][1] == value:
            continue
        pts.append([int(day), value])
    return {k: v for k, v in out.items() if len(v) >= 2}


def build_valuations(listings: pd.DataFrame, predictions: pd.DataFrame,
                     sell_speed: pd.DataFrame | None = None,
                     snapshots: pd.DataFrame | None = None) -> dict:
    """Return ``{"v":1, "cars": {olx_id: {...}}}`` for active, priced listings.

    - ``listings``: enriched listings DataFrame (needs olx_id, is_active, title,
      description, brand, model, year, mileage_km, fuel_type, price_eur, city).
    - ``predictions``: per-olx_id ``predicted_price`` + ``fair_price_{low,high}``.
    - ``sell_speed``: optional output of ``compute_sell_speed_by_model`` for the
      median days-to-sell per (brand, model).
    - ``snapshots``: optional price-snapshot frame; supplies the per-ad price
      track and, with it, the "the seller has already come down" line.
    """
    cars: dict[str, dict] = {}
    if listings.empty or predictions.empty:
        return {"v": 1, "cars": cars}

    now = pd.Timestamp.now(tz="UTC")
    pred = predictions.set_index("olx_id")
    track = _price_track(snapshots, now, set(pred.index.astype(str)))

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
                                 getattr(r, "description", "") or "",
                                 getattr(r, "origin", None))
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
        brand, model = getattr(r, "brand", None), getattr(r, "model", None)
        sd = sell_lookup.get((brand, model))
        if sd is not None:
            rec["sd"] = sd
        # Model slug (→ contextual /preco/{slug} link on the /avaliar verdict).
        # Lazy import avoids a module-load circular (model_pages imports _i here).
        if brand and model:
            from src.analytics.model_pages import slugify
            rec["ms"] = slugify(f"{brand}-{model}")
        posted = getattr(r, "first_seen_at", None)
        if posted is not None and pd.notna(posted):
            posted_ts = pd.Timestamp(posted)
            if posted_ts.tzinfo is None:
                posted_ts = posted_ts.tz_localize("UTC")
            dom = int((now - posted_ts).days)
            if 0 <= dom <= 3650:
                rec["dom"] = dom
        pts = track.get(str(oid))
        if pts:
            rec["ph"] = pts
        fault = _s(getattr(r, "text_minor_fault", None))
        if fault:
            rec["mf"] = fault[:40]
        blocker = _s(getattr(r, "text_hard_block_phrase", None))
        if blocker:
            rec["hb"] = blocker[:40]
        # Drop None values to keep the blob small.
        cars[str(oid)] = {k: v for k, v in rec.items() if v is not None}

    return {"v": 1, "cars": cars}
