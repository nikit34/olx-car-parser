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
under €4.000 only 27 sellers in 100 ever come down, but the ones who do give up
13.9%; at €8.000-15.000 it is 46 in 100 giving up 9.1%; above €25.000, 45 in 100
giving up 6.2%. Dear cars move often and a little, cheap cars rarely and a lot,
and a single number across all of them is true of nobody.

The base is every car we watched to the end, quick sales included. A car that
went in a week never had the chance to come down, and that belongs in the
number: the question a buyer is asking is how often a seller of this kind of
car ends up moving, not how often the stubborn ones do. It does mean the share
reads lower than it would among cars that have already sat a month.

What this is not: the discount on the invoice. We see the asking price, so a
car that never changed its ask may still have been haggled down at the kerb.
Read it as the public part of the negotiation, which is the part a buyer can
check before ringing the seller.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_BAND_CARS = 200
BANDS: tuple[tuple[float, float, str], ...] = (
    (0.0, 4000.0, "até €4.000"),
    (4000.0, 8000.0, "€4.000 a €8.000"),
    (8000.0, 15000.0, "€8.000 a €15.000"),
    (15000.0, 25000.0, "€15.000 a €25.000"),
    (25000.0, float("inf"), "acima de €25.000"),
)


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
    df = outcomes.dropna(subset=["first_ask", "ask_change_pct"]).copy()
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
