"""How much sellers actually come down, by price band.

Every buyer arrives at a listing with the same question and no way to answer
it: is there room here, and how much? The advice online is folklore — "offer
ten percent below" — and the marketplaces have no reason to publish the real
number, because it is an argument against their sellers.

We can measure it, and the measurement is not a model. Take every car whose
whole story we watched, first advert to last, and count how many came down
before they went and by how much. Done per car rather than per advert, because
a seller who withdraws and reposts cheaper has come down, and counting adverts
would score that as two sellers who never moved.

The bands are by asking price because the two halves of the answer move in
opposite directions along it. Measured on the Portuguese corpus on 2026-09-20:
under €4.000 only 26 sellers in 100 ever come down, but the ones who do give up
14.3%; at €8.000-15.000 it is 46 in 100 giving up 8.9%; above €25.000, 45 in 100
giving up 6.2%. Dear cars move often and a little, cheap cars rarely and a lot,
and a single number across all of them is true of nobody.

The base is every car we watched to the end, quick sales included. A car that
went in a week never had the chance to come down, and that belongs in the
number: the question a buyer is asking is how often a seller of this kind of
car ends up moving, not how often the stubborn ones do. It does mean the share
reads lower than it would among cars that have already sat a month.

Read against a listing's own age it says something sharper still. A car that
has sat two months without moving is not a random car: the sellers who were
going to move early already have, and the median first cut lands on day 15. So
``age_conditional_norms`` asks the question a buyer actually has — of the cars
that reached day 60 with the ask untouched, how many ended up moving, and by
how much.

The answer surprised the hypothesis that prompted it. Holding out longer barely
changes the odds: in the €8.000-15.000 band 37 in 100 of those untouched at day
14 eventually moved, 39 at day 30, 39 at day 60, 39 at day 90. A seller who has
held for three months is no closer to breaking than one who has held a
fortnight — but in the cheap band, the ones who do break late break harder,
from 14.6% at day 14 to 16.7% at day 60.

What this is not: the discount on the invoice. We see the asking price, so a
car that never changed its ask may still have been haggled down at the kerb.
Read it as the public part of the negotiation, which is the part a buyer can
check before ringing the seller.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_BAND_CARS = 200
MIN_AGE_CELL_CARS = 200
AGE_STEPS: tuple[int, ...] = (14, 30, 60, 90)
BANDS: tuple[tuple[float, float, str], ...] = (
    (0.0, 4000.0, "até €4.000"),
    (4000.0, 8000.0, "€4.000 a €8.000"),
    (8000.0, 15000.0, "€8.000 a €15.000"),
    (15000.0, 25000.0, "€15.000 a €25.000"),
    (25000.0, float("inf"), "acima de €25.000"),
)


def _finished(outcomes: pd.DataFrame) -> pd.DataFrame:
    """Only cars whose story is over — an advert still up has not finished it.

    A car on sale today may come down tomorrow, so counting it as one that
    never moved understates the share. The gap is small on this corpus (0 to 4
    points, measured 2026-09-20) but it points the wrong way and it is free to
    remove: drop the unfinished stories rather than publish them as decided.
    """
    if "still_active" in outcomes.columns:
        return outcomes[~outcomes["still_active"].astype(bool)]
    if "sold" in outcomes.columns:
        return outcomes[outcomes["sold"].astype(bool)]
    return outcomes


def price_band_norms(outcomes: pd.DataFrame, min_cars: int = MIN_BAND_CARS) -> list[dict]:
    """``[{lo, hi, lbl, n, cu, cp, md}]`` — one row per price band.

    ``cu`` is the share of cars that came down before they left, ``cp`` the
    median size of that move among the ones that moved, ``md`` the median days
    on the market. A band under ``min_cars`` is dropped rather than published
    thin, the same rule the liquidity cells follow.
    """
    need = {"first_ask", "ask_change_pct", "cut"}
    if outcomes is None or outcomes.empty or not need.issubset(outcomes.columns):
        return []
    df = _finished(outcomes).dropna(subset=["first_ask", "ask_change_pct"]).copy()
    if df.empty:
        return []

    out: list[dict] = []
    for lo, hi, label in BANDS:
        band = df[(df["first_ask"] >= lo) & (df["first_ask"] < hi)]
        if len(band) < min_cars:
            continue
        movers = band[band["cut"]]
        rec = {
            "lo": int(lo),
            "hi": None if np.isinf(hi) else int(hi),
            "lbl": label,
            "n": int(len(band)),
            "cu": round(float(band["cut"].mean()), 3),
        }
        if len(movers) >= min_cars // 4:
            rec["cp"] = round(float(-movers["ask_change_pct"].median()), 1)
        if "days" in band.columns and band["days"].notna().any():
            rec["md"] = int(round(float(band["days"].median())))
        out.append(rec)
    return out


def norm_for_price(norms: list[dict], price: float | None) -> dict | None:
    """The band a given asking price falls into, or None."""
    if not norms or price is None or not np.isfinite(price):
        return None
    for rec in norms:
        hi = rec.get("hi")
        if price >= rec["lo"] and (hi is None or price < hi):
            return rec
    return None


def age_conditional_norms(outcomes: pd.DataFrame, steps: tuple[int, ...] = AGE_STEPS,
                          min_cars: int = MIN_AGE_CELL_CARS) -> list[dict]:
    """``[{lo, hi, lbl, ag: [[day, cu, cp, n], ...]}]`` — the norm by age.

    Each cell answers one question: among the cars of this band that were
    still on the market on ``day``, had not yet touched the ask, and have since
    left it, what share came down before going, and how far. Cars that had already moved are excluded
    rather than counted as movers — they are not the listing the reader is
    looking at. A cell under ``min_cars`` is absent; a band left with no cells
    at all does not appear.
    """
    need = {"first_ask", "days", "cut"}
    if outcomes is None or outcomes.empty or not need.issubset(outcomes.columns):
        return []
    df = _finished(outcomes).dropna(subset=["first_ask", "days"]).copy()
    if df.empty:
        return []
    cut_day = (pd.to_numeric(df["first_cut_day"], errors="coerce")
               if "first_cut_day" in df.columns else pd.Series(np.nan, index=df.index))
    df["_cut_day"] = cut_day
    df["_moved"] = df["cut"].astype(bool)

    out: list[dict] = []
    for lo, hi, label in BANDS:
        band = df[(df["first_ask"] >= lo) & (df["first_ask"] < hi)]
        if band.empty:
            continue
        cells: list[list] = []
        for day in steps:
            untouched = band[(band["days"] >= day)
                             & (band["_cut_day"].isna() | (band["_cut_day"] >= day))]
            if len(untouched) < min_cars:
                continue
            later = untouched[untouched["_moved"]]
            share = round(float(len(later) / len(untouched)), 3)
            size = (round(float(-later["ask_change_pct"].median()), 1)
                    if len(later) >= min_cars // 4 and "ask_change_pct" in later.columns
                    else None)
            cells.append([int(day), share, size, int(len(untouched))])
        if cells:
            out.append({"lo": int(lo), "hi": None if np.isinf(hi) else int(hi),
                        "lbl": label, "ag": cells})
    return out


def age_cell(band: dict | None, age_days: float | None) -> list | None:
    """The deepest age cell a listing of ``age_days`` has actually reached."""
    if not band or age_days is None or not np.isfinite(age_days):
        return None
    reached = [c for c in band.get("ag", []) if age_days >= c[0]]
    return reached[-1] if reached else None


def merge_age_cells(norms: list[dict], age_norms: list[dict]) -> list[dict]:
    """One band record carrying both the market-wide norm and the age cells."""
    by_lo = {rec["lo"]: rec.get("ag") for rec in (age_norms or [])}
    out = []
    for rec in norms or []:
        cells = by_lo.get(rec["lo"])
        out.append({**rec, "ag": cells} if cells else dict(rec))
    return out
