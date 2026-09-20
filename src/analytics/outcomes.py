"""What actually happened to a car, as opposed to what an advert claimed.

Everything the decision engine believes is measured against asks: the model
predicts an ask, the discount is against a median of asks, the verdict is a
claim about an ask. Nothing in the corpus records the price a car changed
hands at, so the engine has never been checked against an outcome — and when
it finally was, on 2026-09-19, nothing it surfaced looked different from the
market it was drawn from (see ``project_dom_signal_validity``).

This is the closest thing to an outcome the data does hold. A car that sells
near its ask leaves once and never comes back. A car priced above what anyone
will pay comes back as a new advert, usually cheaper, and the pair of asks —
the first one and the one it eventually accepted — brackets the real number
from above and below. That is a realised price move for an identified car, not
a snapshot of what sellers hope for.

One row per car, not per advert: adverts of the same car are glued together by
``liquidity.relist_roots``. The columns are deliberately raw (first ask, last
ask, days, whether it ended) so the caller decides what counts as a good
outcome — holding the ask, selling inside a month, not being relisted — rather
than having that judgement baked in here.

Two limits worth keeping in view. The matcher only finds re-listings it can
match, so ``n_ads`` is a floor and a car that quietly gave up looks the same as
one that sold. And an ask the seller never cut is evidence that nobody
argued them down in public, not proof of the price on the invoice.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics.liquidity import relist_roots

MIN_ASK_EUR = 100.0


def build_outcomes(
    listings: pd.DataFrame,
    snapshots: pd.DataFrame | None = None,
    pairs: pd.DataFrame | None = None,
    now: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """One row per car: how long it took, what it asked, whether it ended.

    ``snapshots`` is ``price_snapshots`` (``olx_id``, ``price_eur``,
    ``scraped_at``); without it the asks come from the listing rows and the
    trajectory columns stay flat. ``pairs`` is ``relist_events``; without it
    every advert is its own car.

    Columns: ``car_id`` (first advert), ``n_ads``, ``first_seen_at``,
    ``ended_at``, ``days`` (calendar, the gaps between adverts included),
    ``sold`` (it ended and did not come back), ``first_ask``, ``last_ask``,
    ``min_ask``, ``ask_change_pct`` (negative = the seller came down),
    ``cut`` (came down by more than 1%), ``first_cut_day`` (days from the first
    advert to the first time the ask stepped down, NaN when it never did), plus
    brand/model/year carried from the first advert.
    """
    cols = ["car_id", "n_ads", "brand", "model", "year", "first_seen_at", "ended_at",
            "days", "sold", "first_ask", "last_ask", "min_ask", "ask_change_pct", "cut",
            "first_cut_day"]
    if listings is None or listings.empty or "olx_id" not in listings.columns:
        return pd.DataFrame(columns=cols)

    df = listings.copy()
    df["olx_id"] = df["olx_id"].astype(str)
    roots = relist_roots(df, pairs) if pairs is not None else pd.Series(dtype=object)
    df["car_id"] = df["olx_id"].map(roots).fillna(df["olx_id"]) if len(roots) else df["olx_id"]

    start = pd.to_datetime(df.get("first_seen_at"), errors="coerce", utc=True)
    end = pd.to_datetime(df.get("last_scraped_at"), errors="coerce", utc=True)
    for col in ("deactivated_at", "last_seen_at"):
        if col in df.columns:
            end = end.fillna(pd.to_datetime(df[col], errors="coerce", utc=True))
    df["_start"], df["_end"] = start, end
    df["_active"] = df["is_active"].astype(bool) if "is_active" in df.columns else False

    if now is None:
        now = end.max()
    df.loc[df["_active"], "_end"] = now

    by_car = df.sort_values("_end", na_position="first").groupby("car_id", sort=False)
    out = pd.DataFrame({
        "n_ads": by_car.size(),
        "first_seen_at": by_car["_start"].min(),
        "ended_at": by_car["_end"].max(),
        "still_active": by_car["_active"].any(),
    })
    first_ad = df.sort_values("_start").groupby("car_id", sort=False).head(1).set_index("car_id")
    for col in ("brand", "model", "year"):
        if col in first_ad.columns:
            out[col] = first_ad[col]

    out["days"] = (out["ended_at"] - out["first_seen_at"]).dt.total_seconds() / 86400.0
    out["sold"] = ~out["still_active"]

    asks = _ask_trajectory(df, snapshots)
    out = out.join(asks)
    change = (out["last_ask"] - out["first_ask"]) / out["first_ask"].replace(0, np.nan)
    out["ask_change_pct"] = (change * 100).round(2)
    out["cut"] = out["ask_change_pct"] < -1.0

    out = out.reset_index().rename(columns={"index": "car_id"})
    out = out[out["days"].notna() & (out["days"] >= 0)]
    return out[[c for c in cols if c in out.columns] + ["still_active"]]


def _ask_trajectory(df: pd.DataFrame, snapshots: pd.DataFrame | None) -> pd.DataFrame:
    """First, last and lowest ask per car, plus the day it first came down.

    ``first_cut_day`` counts from the car's first advert, so a seller who held
    for two months and then moved is distinguishable from one who moved in the
    first week — that is what lets the norm be read against a listing's own
    age instead of against the whole market. A step counts as a cut when it is
    more than 1% below the ask before it, which is the same threshold the
    price track on the card uses; a relist that goes back UP resets nothing and
    is simply not a cut.
    """
    if (snapshots is not None and not snapshots.empty
            and {"olx_id", "price_eur", "scraped_at"}.issubset(snapshots.columns)):
        snap = snapshots.copy()
        snap["olx_id"] = snap["olx_id"].astype(str)
        car_start = df.groupby("car_id", sort=False)["_start"].min().rename("_car_start")
        snap = snap.merge(df[["olx_id", "car_id"]], on="olx_id", how="inner")
        snap = snap.merge(car_start, left_on="car_id", right_index=True, how="left")
        snap["price_eur"] = pd.to_numeric(snap["price_eur"], errors="coerce")
        snap = snap[snap["price_eur"] >= MIN_ASK_EUR]
        snap["scraped_at"] = pd.to_datetime(snap["scraped_at"], errors="coerce", utc=True)
        snap = snap.dropna(subset=["price_eur", "scraped_at"]).sort_values(
            ["car_id", "scraped_at"])
        if not snap.empty:
            g = snap.groupby("car_id", sort=False)["price_eur"]
            prev = g.shift(1)
            cut_rows = snap[snap["price_eur"] < prev * 0.99]
            first_cut = cut_rows.groupby("car_id", sort=False).first()
            cut_day = ((first_cut["scraped_at"] - first_cut["_car_start"])
                       .dt.total_seconds() / 86400.0).round(0)
            return pd.DataFrame({"first_ask": g.first(), "last_ask": g.last(),
                                 "min_ask": g.min()}).join(
                cut_day.rename("first_cut_day"))
    price = pd.to_numeric(df.get("price_eur"), errors="coerce")
    flat = df.assign(_p=price).dropna(subset=["_p"])
    flat = flat[flat["_p"] >= MIN_ASK_EUR].sort_values("_start")
    if flat.empty:
        return pd.DataFrame(columns=["first_ask", "last_ask", "min_ask"])
    g = flat.groupby("car_id", sort=False)["_p"]
    return pd.DataFrame({"first_ask": g.first(), "last_ask": g.last(), "min_ask": g.min()})
