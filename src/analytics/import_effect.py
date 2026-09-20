"""What the import flag is worth, measured instead of assumed.

The site has always shown an import flag next to an estimate of the
legalisation tax, which told the reader an imported car is worth less. Nothing
had ever checked that. This module checks it against the outcomes table, inside
cells that hold the car fixed — same platform, same brand, same model, same
three-year age bucket — so the answer is not the composition of the two
populations (imports skew newer, pricier and higher-mileage, and the two
platforms have different advert lifecycles) read back as an effect of
importing.

Two numbers come out, both as a difference between an imported car and a
national one in the same cell:

``s30``     percentage points of 30-day sales. Negative means the import is
            less likely to be gone inside the window.
``price``   percentage points of model error on the accepted price, over cars
            that left without ever touching the ask. The price model has no
            origin feature, so its prediction is a control that already holds
            mileage and equipment: a positive gap means the import went for
            less than its specs alone predict.

Each carries a 90% interval bootstrapped over cells. A block whose interval
straddles zero is still returned — the caller decides what to say about it —
but ``publishable`` is True only when the interval excludes zero, because a
number that could be zero has no business being printed as a fact.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

YEAR_BUCKET = 3
MIN_ARM = 8
MIN_CELLS = 20
MIN_CELLS_PRICE = 12
WINDOW_DAYS = 30
MAX_SOLD_DAYS = 120.0
BOOT = 1000
SEED = 20260920


def import_arm(listings: pd.DataFrame) -> pd.Series:
    """'imported' / 'national' / None per advert, the way the page flags it."""
    from src.analytics.valuations import _IMPORT_NEG, _import_flags, _strip_accents

    titles = listings.get("title", pd.Series("", index=listings.index))
    descs = listings.get("description", pd.Series("", index=listings.index))
    origins = listings.get("origin", pd.Series(None, index=listings.index))
    out: list[str | None] = []
    for title, desc, origin in zip(titles, descs, origins, strict=False):
        t = title if isinstance(title, str) else ""
        d = desc if isinstance(desc, str) else ""
        o = origin if isinstance(origin, str) else None
        if _import_flags(t, d, o)[0]:
            out.append("imported")
        elif o == "national":
            out.append("national")
        elif o == "imported":
            out.append(None)
        elif _IMPORT_NEG.search(_strip_accents(t) + " " + _strip_accents(d)):
            out.append("national")
        else:
            out.append(None)
    return pd.Series(out, index=listings.index, dtype=object)


def _boot(cells: list[tuple[float, int]], rng: np.random.Generator):
    if len(cells) < 2:
        return None
    d = np.array([c[0] for c in cells], dtype=float)
    w = np.array([c[1] for c in cells], dtype=float)
    idx = rng.integers(0, len(d), size=(BOOT, len(d)))
    draws = np.array([np.average(d[i], weights=w[i]) for i in idx])
    return (float(np.average(d, weights=w)),
            float(np.percentile(draws, 5)), float(np.percentile(draws, 95)))


def _cells(df: pd.DataFrame, keys: list[str], value) -> list[tuple[float, int]]:
    out: list[tuple[float, int]] = []
    for _, g in df.groupby(keys, observed=True, dropna=True):
        imp = g[g["_arm"] == "imported"]
        nat = g[g["_arm"] == "national"]
        if len(imp) < MIN_ARM or len(nat) < MIN_ARM:
            continue
        a, b = value(imp), value(nat)
        if a is None or b is None or not np.isfinite(a) or not np.isfinite(b):
            continue
        out.append((float(a - b), int(min(len(imp), len(nat)))))
    return out


def _block(cells, rng, min_cells: int) -> dict | None:
    if len(cells) < min_cells:
        return None
    got = _boot(cells, rng)
    if got is None:
        return None
    point, lo, hi = got
    return {"v": round(point, 1), "lo": round(lo, 1), "hi": round(hi, 1),
            "cells": len(cells), "n": int(sum(c[1] for c in cells)),
            "publishable": bool(lo > 0 or hi < 0)}


def import_effect(listings: pd.DataFrame, outcomes: pd.DataFrame,
                  predictions: dict | None = None,
                  now=None, seed: int = SEED) -> dict | None:
    """Matched import-vs-national gaps, or None when the corpus is too thin."""
    if outcomes is None or not len(outcomes) or listings is None or not len(listings):
        return None
    ids = listings["olx_id"].astype(str)
    arm = import_arm(listings)
    df = outcomes.copy()
    df["_arm"] = df["car_id"].map(dict(zip(ids, arm)))
    df = df[df["_arm"].notna()]
    if not len(df):
        return None
    df["_y3"] = pd.to_numeric(df["year"], errors="coerce") // YEAR_BUCKET
    df["_src"] = df["car_id"].map(dict(zip(
        ids, listings.get("source", pd.Series("", index=listings.index)))))

    if now is None:
        now = pd.to_datetime(listings.get("last_seen_at"), utc=True,
                             errors="coerce").max()
    now = pd.Timestamp(now)
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    age = (now - pd.to_datetime(df["first_seen_at"], utc=True, errors="coerce")
           ).dt.total_seconds() / 86400.0
    obs = df[age >= WINDOW_DAYS].copy()
    obs["_gone"] = (~obs["still_active"].astype(bool)) & (obs["days"] <= WINDOW_DAYS)

    rng = np.random.default_rng(seed)
    keys = ["_src", "brand", "model", "_y3"]
    s30 = _block(_cells(obs, keys,
                        lambda g: float(g["_gone"].mean()) * 100), rng, MIN_CELLS)

    price = None
    if predictions:
        clean = df[(~df["still_active"].astype(bool)) & (~df["cut"].astype(bool))
                   & df["last_ask"].notna() & (df["last_ask"] > 0)
                   & (df["days"] <= MAX_SOLD_DAYS)].copy()
        clean["_pred"] = clean["car_id"].astype(str).map(predictions)
        clean = clean[clean["_pred"].notna() & (clean["_pred"] > 0)]
        clean["_err"] = (clean["_pred"] / clean["last_ask"] - 1) * 100
        price = _block(_cells(clean, keys, lambda g: float(g["_err"].median())),
                       rng, MIN_CELLS_PRICE)

    if s30 is None and price is None:
        return None
    out: dict = {"window": WINDOW_DAYS}
    if s30 is not None:
        out["s30"] = s30
    if price is not None:
        out["price"] = price
    return out
