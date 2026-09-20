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


def car_ids(listings: pd.DataFrame, pairs: pd.DataFrame | None) -> tuple[pd.Series, pd.Series]:
    """``(advert → car id, car id → how many of its edges were re-listings)``.

    A car is a connected set of adverts, and the adverts connect two ways that
    must not be confused. A RE-LISTING is sequential: the advert died and the
    seller posted again, which is evidence the car did not sell. A DUPLICATE
    (``duplicate_of``, written by the scrape-time dedup) is simultaneous: the
    same car on OLX and StandVirtual at once, which is evidence of nothing
    except that the seller wanted reach.

    Counting only re-listings, as this did until 2026-09-20, made one
    cross-posted car into two: 21 800 of 111 885 "cars" in the Portuguese
    corpus were the same car twice, and every sample size published off this
    table was inflated by 19%. Counting both kinds as one car fixes that, and
    the second return value keeps the distinction alive so ``rb`` — "did not
    sell first time" — can still be read off sequential links alone.

    Both sources are imperfect and in opposite directions. The dedup key
    demands the price match to 1%, so it misses same-platform duplicates whose
    seller moved the price (about 150 live ones on 2026-09-20); and roughly one
    mark in ten joins two different cars. Under-merging costs a double count,
    over-merging costs an observation, and neither is worth waiting for: both
    are smaller than the 19% they replace.
    """
    ids = listings["olx_id"].astype(str)
    parent = {i: i for i in ids}

    def find(x):
        root = x
        while parent.get(root, root) != root:
            root = parent[root]
        while parent.get(x, x) != root:
            parent[x], x = root, parent[x]
        return root

    def union(a, b):
        ra, rb_ = find(a), find(b)
        if ra != rb_:
            parent[ra] = rb_
        return ra != rb_

    relist_edges: dict[str, int] = {}
    if pairs is not None and not getattr(pairs, "empty", True):
        roots = relist_roots(listings, pairs)
        for child, root in roots.items():
            if child != root and child in parent and root in parent:
                union(child, root)
    if "duplicate_of" in listings.columns:
        dup = listings[["olx_id", "duplicate_of"]].dropna()
        for advert, target in zip(dup["olx_id"].astype(str), dup["duplicate_of"].astype(str)):
            if target in parent and advert in parent:
                union(advert, target)

    start = pd.to_datetime(listings.get("first_seen_at"), errors="coerce", utc=True)
    order = pd.DataFrame({"olx_id": ids.values, "_s": start.values})
    order["_root"] = [find(i) for i in order["olx_id"]]
    earliest = order.sort_values("_s", na_position="last").groupby("_root", sort=False).head(1)
    name_of = dict(zip(earliest["_root"], earliest["olx_id"]))
    car_of = {i: name_of.get(r, r) for i, r in zip(order["olx_id"], order["_root"])}

    if pairs is not None and not getattr(pairs, "empty", True):
        roots = relist_roots(listings, pairs)
        for child, root in roots.items():
            if child != root and child in car_of:
                car = car_of[child]
                relist_edges[car] = relist_edges.get(car, 0) + 1
    return pd.Series(car_of, dtype=object), pd.Series(relist_edges, dtype="int64")


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
    advert to the first time the ask stepped down, NaN when it never did),
    ``n_relists`` (how many of the car's adverts followed a dead one, as
    opposed to running beside it), plus brand/model/year from the first advert.
    """
    cols = ["car_id", "n_ads", "brand", "model", "year", "first_seen_at", "ended_at",
            "days", "sold", "first_ask", "last_ask", "min_ask", "ask_change_pct", "cut",
            "first_cut_day"]
    if listings is None or listings.empty or "olx_id" not in listings.columns:
        return pd.DataFrame(columns=cols)

    df = listings.copy()
    df["olx_id"] = df["olx_id"].astype(str)
    ids, relists = car_ids(df, pairs)
    df["car_id"] = df["olx_id"].map(ids).fillna(df["olx_id"])

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

    out["n_relists"] = out.index.map(relists).fillna(0).astype(int) if len(relists) else 0
    out = out.reset_index().rename(columns={"index": "car_id"})
    out = out[out["days"].notna() & (out["days"] >= 0)]
    return out[[c for c in cols if c in out.columns] + ["still_active", "n_relists"]]


def _one_advert_at_a_time(df: pd.DataFrame) -> pd.DataFrame:
    """The adverts that tell the car's price story, one at a time.

    Identity and price history need different rules. A car posted on OLX and
    StandVirtual at once is ONE car, but it is not one price series: the two
    sites routinely carry different numbers for the same car — a peer session
    confirmed 27 duplicate pairs by photo on 2026-09-20 and every one of them
    differed, from €6 500 against €6 399 to €7 500 against €5 000. Pour both
    into one trajectory and the gap between platforms reads as a seller cutting
    a third off the price, which is a cut nobody made.

    So overlapping adverts collapse to the one that started first, and what
    survives is a sequence: advert, then the advert that replaced it. That is
    the series a "the seller came down" claim can be made from.
    """
    if "car_id" not in df.columns:
        return df
    ordered = df.sort_values(["car_id", "_start"], na_position="last")
    keep: list[bool] = []
    current_car, open_until = None, None
    for car, start, end in zip(ordered["car_id"], ordered["_start"], ordered["_end"]):
        if car != current_car:
            current_car, open_until = car, end
            keep.append(True)
            continue
        if pd.notna(start) and pd.notna(open_until) and start < open_until:
            keep.append(False)
            continue
        keep.append(True)
        if pd.notna(end) and (open_until is None or pd.isna(open_until) or end > open_until):
            open_until = end
    return ordered[pd.Series(keep, index=ordered.index)]


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
        snap = snap.merge(_one_advert_at_a_time(df)[["olx_id", "car_id"]],
                          on="olx_id", how="inner")
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
