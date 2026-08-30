"""Which of two variants of one model holds its price better, fitted on our data.

"Diesel ou gasolina", "manual ou automática" — these are answered everywhere
with opinion and nowhere with a number for a named car. This module produces
that number: for one model and one either/or, how much of its value each side
loses per year of age, with mileage held equal, plus the asking premium one side
carries over the other at a few concrete ages.

The estimator is a single OLS per model per dimension::

    log(price) = a + b*age + c*log(km) + d*isA + e*age*isA

``b`` is the B-side depreciation rate, ``b + e`` the A-side one, and ``e`` is the
whole claim of the page — the distance between the two curves. ``log(km)``
carries the confound that makes the raw comparison useless: diesels are sold with
far more mileage and automatics with far less, so a slope fitted without it
reports the mileage mix as if it were the fuel or the gearbox. Because both sides
are evaluated at the same km, mileage cancels out of the premium
``exp(d + e*age) - 1`` entirely.

Everything ships with the uncertainty it was measured at. A model reaches a page
only when the 95% interval on the rate difference is narrower than
``MAX_CI_HALF`` — otherwise "no difference" would mean "we could not see one",
and those are not the same statement to put in front of a buyer.

``DUELS`` names the dimensions. Its keys are the blob keys the Worker reads, and
its (column, a, b) values MUST stay in lock-step with the DUELS table in
flipper-club/src/seo-pages.js: A is the side named first in the URL, so the sign
of every difference on the page is the sign produced here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_SIDE_N = 20
MIN_SPAN = 8
MIN_DOF = 15
MIN_R2 = 0.50
MIN_RATE = 0.03
MAX_RATE = 0.22
MAX_CI_HALF = 0.03
MAX_GAP_CI_HALF = 0.15
SIG_T = 1.96

DUELS = {
    "dg": ("fuel_type", "Diesel", "Gasolina"),
    "cx": ("transmission", "Manual", "Automática"),
}

_GAP_QUANTILES = (0.2, 0.5, 0.8)


def _clean(grp: pd.DataFrame, column: str, a: str, b: str, now_year: int) -> pd.DataFrame:
    need = {column, "price_eur", "year", "mileage_km"}
    if not need.issubset(grp.columns):
        return pd.DataFrame()
    d = pd.DataFrame({
        "side": grp[column].astype("string"),
        "price": pd.to_numeric(grp["price_eur"], errors="coerce"),
        "year": pd.to_numeric(grp["year"], errors="coerce"),
        "km": pd.to_numeric(grp["mileage_km"], errors="coerce"),
    }).dropna()
    d = d[d["side"].isin([a, b])]
    d = d[(d["price"] > 0) & (d["km"] > 0)]
    d = d[(d["year"] >= 1980) & (d["year"] <= now_year)]
    if len(d) < 2 * MIN_SIDE_N:
        return pd.DataFrame()
    lo, hi = np.percentile(d["price"], [1, 99])
    d = d[(d["price"] >= lo) & (d["price"] <= hi)]
    d["age"] = (now_year - d["year"]).clip(lower=0.5)
    return d


def retention_duel(grp: pd.DataFrame, column: str, a: str, b: str,
                   now_year: int) -> dict | None:
    """Fitted A-vs-B retention for one model, or None when the sample cannot
    carry it.

    Returns ``{"a": {...}, "b": {...}, "ci", "t", "r2", "y0", "y1", "gap"}``
    where each side holds its sample size, annual rate, median mileage and median
    asking price, ``ci`` is the 95% half-width on the rate difference, and ``gap``
    is ``[[age, premium, half_width], ...]`` — the A-over-B asking premium at
    that age with mileage held equal, carrying only the ages whose own interval
    is tight enough to say anything (``MAX_GAP_CI_HALF``).
    """
    d = _clean(grp, column, a, b, now_year)
    if d.empty:
        return None
    is_a = (d["side"] == a).to_numpy(dtype=float)
    n_a, n_b = int(is_a.sum()), int(len(d) - is_a.sum())
    if n_a < MIN_SIDE_N or n_b < MIN_SIDE_N:
        return None
    span = int(d["year"].max() - d["year"].min())
    if span < MIN_SPAN:
        return None

    age = d["age"].to_numpy(dtype=float)
    y = np.log(d["price"].to_numpy(dtype=float))
    X = np.column_stack([np.ones_like(age), age,
                         np.log(d["km"].to_numpy(dtype=float)), is_a, age * is_a])
    dof = len(y) - X.shape[1]
    if dof < MIN_DOF or np.linalg.matrix_rank(X) < X.shape[1]:
        return None
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    if ss_tot <= 0:
        return None
    r2 = 1.0 - ss_res / ss_tot
    if r2 < MIN_R2:
        return None
    cov = (ss_res / dof) * np.linalg.pinv(X.T @ X)

    rate_b = 1.0 - float(np.exp(beta[1]))
    rate_a = 1.0 - float(np.exp(beta[1] + beta[4]))
    if not (MIN_RATE <= rate_b <= MAX_RATE and MIN_RATE <= rate_a <= MAX_RATE):
        return None

    se_e = float(np.sqrt(max(cov[4, 4], 0.0)))
    hw = SIG_T * se_e
    ci = abs(float(np.exp(beta[1] + beta[4] - hw))
             - float(np.exp(beta[1] + beta[4] + hw))) / 2
    if ci > MAX_CI_HALF:
        return None

    ages = sorted({int(round(v)) for v in np.quantile(age, _GAP_QUANTILES)})
    lo_age = max(float(d.loc[d["side"] == a, "age"].min()),
                 float(d.loc[d["side"] == b, "age"].min()))
    hi_age = min(float(d.loc[d["side"] == a, "age"].max()),
                 float(d.loc[d["side"] == b, "age"].max()))
    gap = []
    for at in ages:
        if at < lo_age or at > hi_age:
            continue
        est = float(np.exp(beta[3] + beta[4] * at)) - 1.0
        var = cov[3, 3] + at * at * cov[4, 4] + 2 * at * cov[3, 4]
        se = float(np.sqrt(max(var, 0.0)))
        half = abs(float(np.exp(beta[3] + beta[4] * at + SIG_T * se))
                   - float(np.exp(beta[3] + beta[4] * at - SIG_T * se))) / 2
        if half > MAX_GAP_CI_HALF:
            continue
        gap.append([at, round(est, 4), round(half, 4)])

    def side(mask: pd.Series) -> dict:
        sub = d[mask]
        return {"n": int(len(sub)), "km": int(round(sub["km"].median())),
                "fm": int(round(sub["price"].median()))}

    out = {
        "a": {**side(d["side"] == a), "r": round(rate_a, 4)},
        "b": {**side(d["side"] == b), "r": round(rate_b, 4)},
        "ci": round(ci, 4),
        "t": round(float(beta[4] / se_e), 2) if se_e > 0 else 0.0,
        "r2": round(r2, 3),
        "y0": int(d["year"].min()), "y1": int(d["year"].max()),
    }
    if gap:
        out["gap"] = gap
    return out


def all_duels(grp: pd.DataFrame, now_year: int) -> dict[str, dict]:
    """Every dimension this model's sample can carry, keyed by blob key."""
    out = {}
    for key, (column, a, b) in DUELS.items():
        fit = retention_duel(grp, column, a, b, now_year)
        if fit:
            out[key] = fit
    return out
