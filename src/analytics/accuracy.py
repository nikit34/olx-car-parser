"""How wrong the valuation is, measured against prices the market accepted.

The error we publish today is cross-validated over listings, so its target is
what a seller wanted, not what a buyer paid. It is also a single number for the
whole corpus, and the corpus is not one market: measured on 2026-09-20 the
estimate lands within 11-13% of the accepted price above €8.000 and within 25%
below €4.000, while the published figure was 27.5 for everything. That number
flatters the cheap tail and slanders the rest.

There is one subset where an ask is a price: a car that left the market having
never touched its number and never came back as a new advert. Nobody argued the
seller down in public and the car went at what was written. Half of the
finished stories in the corpus qualify, which is enough to cut by price band.

Publishing this is the point. Standvirtual and AutoUncle both give away a
valuation and neither says how far off it usually is; a band-by-band error
measured against accepted prices is a claim they are not making, and it is
checkable — the sample size sits next to every row.

What it is not: a transaction log. A seller can still be haggled down at the
kerb after the advert comes down, and a car that vanished without selling looks
the same as one that sold. ``max_days`` exists to keep the second kind out.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

MIN_BAND_CARS = 200
MAX_DAYS = 120.0
WITHIN_PCT = 10.0
BANDS: tuple[tuple[float, float, str], ...] = (
    (0.0, 4000.0, "até €4.000"),
    (4000.0, 8000.0, "€4.000 a €8.000"),
    (8000.0, 15000.0, "€8.000 a €15.000"),
    (15000.0, 25000.0, "€15.000 a €25.000"),
    (25000.0, float("inf"), "acima de €25.000"),
)


def accepted_price_rows(outcomes: pd.DataFrame, max_days: float = MAX_DAYS) -> pd.DataFrame:
    """Cars whose last ask is the price the market took: left, never cut, quick."""
    need = {"still_active", "cut", "last_ask", "days"}
    if outcomes is None or outcomes.empty or not need.issubset(outcomes.columns):
        return pd.DataFrame()
    return outcomes[(~outcomes["still_active"].astype(bool))
                    & (~outcomes["cut"].astype(bool))
                    & outcomes["last_ask"].notna() & (outcomes["last_ask"] > 0)
                    & outcomes["days"].notna() & (outcomes["days"] <= max_days)].copy()


def accuracy_by_band(
    outcomes: pd.DataFrame,
    predictions: Mapping[str, float],
    min_cars: int = MIN_BAND_CARS,
    max_days: float = MAX_DAYS,
) -> list[dict]:
    """``[{lo, hi, lbl, n, err, within, bias}]`` — one row per price band.

    ``err`` is the median absolute gap between the estimate and the accepted
    price in percent, ``within`` the share that landed inside ``WITHIN_PCT``,
    ``bias`` the median signed gap (positive = the estimate reads high). A band
    under ``min_cars`` is absent rather than published thin.
    """
    rows = accepted_price_rows(outcomes, max_days)
    if rows.empty or not predictions:
        return []
    rows["pred"] = rows["car_id"].astype(str).map(
        {str(k): v for k, v in predictions.items()})
    rows = rows[rows["pred"].notna() & (rows["pred"] > 0)]
    if rows.empty:
        return []
    rows["err_pct"] = (rows["pred"] / rows["last_ask"] - 1) * 100

    out: list[dict] = []
    for lo, hi, label in BANDS:
        band = rows[(rows["last_ask"] >= lo) & (rows["last_ask"] < hi)]
        if len(band) < min_cars:
            continue
        err = band["err_pct"]
        out.append({
            "lo": int(lo),
            "hi": None if np.isinf(hi) else int(hi),
            "lbl": label,
            "n": int(len(band)),
            "err": round(float(err.abs().median()), 1),
            "within": round(float((err.abs() <= WITHIN_PCT).mean()), 3),
            "bias": round(float(err.median()), 1),
        })
    return out
