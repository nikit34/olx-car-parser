"""How long a listing stays on the market, measured with the censoring kept in.

The question is the seller's ("how long will this take to sell?") and the
buyer's ("does this one sit, so can I push the price?"). Nobody in Portugal
publishes it: Standvirtual and the valuation books describe a snapshot, and a
snapshot cannot see time. Watching listings appear and disappear can.

Three things make the naive answer — median of ``last_seen - first_seen`` over
the listings that already ended — wrong, and all three are fixed here:

1. **Censoring.** Listings still live have no end date, so dropping them keeps
   only the ones that ended, which over-represents the short-lived ones and
   pulls the median down. The estimator is Kaplan-Meier over BOTH: an active
   listing contributes the days it has survived so far and then leaves the risk
   set. On the current corpus that is a third of the sample.

2. **Outage inflation.** ``deactivated_at`` is the wall-clock of the sweep that
   noticed the listing was gone, not of it going. When a scrape is blocked for
   days (OLX answered 403 from 2026-08-10 to 08-23) the sweep that follows
   stamps thousands of listings with one date, and the median lag between the
   last time we saw a listing alive and that stamp was 13 days. The exit time
   here is ``last_scraped_at`` — the last cycle that confirmed the listing was
   still up — which is the honest lower edge of the interval it died in.

3. **The 30-day wall.** An OLX ad runs in 30-day cycles, and the duration
   histogram has a hard spike at exactly 30 days on both sources. A listing that
   ends there may have sold or may simply have run out, and we cannot tell the
   two apart. So the median is a poor summary (it lands inside that spike for
   most models) and the headline figure is ``s30`` — the share gone by day 30 —
   which is what actually separates a Clio (0.70) from a Mini Cooper (0.40).

What we can say about "sold" versus "gave up" is bounded, not guessed: ``rb`` is
the share of ended listings that reappeared later as a NEW listing of the same
car (``relist_events``). Those definitely did not sell. It is a floor, not a
rate — a relist is only counted when the matcher finds it, and a listing that
ended last week has not had time to come back.

Everything is gated on sample size and simply absent below it, like every other
public figure here (see feedback_quality_over_coverage).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics.model_pages import slugify

MIN_EVENTS = 40
MIN_CELL_EVENTS = 40
MIN_SELL_EVENTS = 8
WINDOW_DAYS = 365
MAX_DAYS = 400
HORIZONS = (30, 60, 90)
MAX_DISTRICT_CELLS = 6

_PRICE_BANDS = (
    ("lt5", "Até €5.000", 0.0, 5000.0),
    ("5-10", "€5.000 a €10.000", 5000.0, 10000.0),
    ("10-20", "€10.000 a €20.000", 10000.0, 20000.0),
    ("gt20", "Acima de €20.000", 20000.0, float("inf")),
)

_AGE_BANDS = (
    ("0-5", "Até 5 anos", 0, 5),
    ("6-10", "6 a 10 anos", 6, 10),
    ("11-15", "11 a 15 anos", 11, 15),
    ("16+", "16 anos ou mais", 16, 120),
)


def survival(durations: np.ndarray, events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kaplan-Meier curve as (times, S(t)) at the days something ended.

    ``events`` is True where the listing left the market and False where it is
    still live and merely leaves the risk set at that duration.
    """
    durations = np.asarray(durations, dtype=float)
    events = np.asarray(events, dtype=bool)
    order = np.argsort(durations, kind="mergesort")
    durations, events = durations[order], events[order]
    n = len(durations)
    times: list[float] = []
    surv: list[float] = []
    s, at_risk, i = 1.0, n, 0
    while i < n:
        t = durations[i]
        j, ended, total = i, 0, 0
        while j < n and durations[j] == t:
            ended += int(events[j])
            total += 1
            j += 1
        if ended and at_risk > 0:
            s *= 1.0 - ended / at_risk
            times.append(float(t))
            surv.append(s)
        at_risk -= total
        i = j
    return np.asarray(times), np.asarray(surv)


def _quantile(times: np.ndarray, surv: np.ndarray, p: float) -> float | None:
    """First day where the curve has dropped to ``1 - p``, or None if never."""
    hit = np.nonzero(surv <= 1.0 - p)[0]
    return float(times[hit[0]]) if len(hit) else None


def _gone_by(times: np.ndarray, surv: np.ndarray, day: float,
             horizon: float) -> float | None:
    """Share gone by ``day``, or None when nothing was watched that long.

    The test is the follow-up, not the last event: once every listing in the
    group has been observed past ``day``, the curve is flat and known out to
    there even if nothing happened in between. Below it the number would be an
    extrapolation, and this file does not extrapolate.
    """
    if horizon < day:
        return None
    if not len(times):
        return 0.0
    idx = np.nonzero(times <= day)[0]
    return float(1.0 - (surv[idx[-1]] if len(idx) else 1.0))


def _curve_stats(sub: pd.DataFrame) -> dict | None:
    """Median, quartiles and the horizon shares for one group of listings."""
    events = sub["_event"].to_numpy(dtype=bool)
    if int(events.sum()) < 1:
        return None
    durations = sub["_dur"].to_numpy(dtype=float)
    times, surv = survival(durations, events)
    horizon = float(durations.max()) if len(durations) else 0.0
    out: dict = {"n": int(events.sum()), "cn": int((~events).sum())}
    md = _quantile(times, surv, 0.5)
    if md is not None:
        out["md"] = int(round(md))
    q1 = _quantile(times, surv, 0.25)
    q3 = _quantile(times, surv, 0.75)
    if q1 is not None:
        out["q1"] = int(round(q1))
    if q3 is not None:
        out["q3"] = int(round(q3))
    for h in HORIZONS:
        share = _gone_by(times, surv, h, horizon)
        if share is not None:
            out[f"s{h}"] = round(share, 3)
    return out


def _cell(sub: pd.DataFrame, key: str, label: str) -> dict | None:
    """One cut of a model's listings, gated on its own ended sample."""
    if int(sub["_event"].sum()) < MIN_CELL_EVENTS:
        return None
    stats = _curve_stats(sub)
    if stats is None or "s30" not in stats:
        return None
    cell = {"k": key, "lbl": label, "n": stats["n"], "s30": stats["s30"]}
    if "md" in stats:
        cell["md"] = stats["md"]
    return cell


def _price_cells(grp: pd.DataFrame) -> list[dict]:
    prices = pd.to_numeric(grp.get("price_eur"), errors="coerce")
    out = []
    for key, label, lo, hi in _PRICE_BANDS:
        cell = _cell(grp[(prices >= lo) & (prices < hi)], key, label)
        if cell:
            out.append(cell)
    return out


def _age_cells(grp: pd.DataFrame, now_year: int) -> list[dict]:
    age = now_year - pd.to_numeric(grp.get("year"), errors="coerce")
    out = []
    for key, label, lo, hi in _AGE_BANDS:
        cell = _cell(grp[(age >= lo) & (age <= hi)], key, label)
        if cell:
            out.append(cell)
    return out


def _district_cells(grp: pd.DataFrame) -> list[dict]:
    if "district" not in grp.columns:
        return []
    keys = grp["district"].map(lambda v: None if pd.isna(v) else (slugify(str(v)) or None))
    out = []
    for key, sub in grp.assign(_k=keys).dropna(subset=["_k"]).groupby("_k"):
        cell = _cell(sub, str(key), str(sub["district"].iloc[0]))
        if cell:
            out.append(cell)
    out.sort(key=lambda c: -c["n"])
    return out[:MAX_DISTRICT_CELLS]


def _discount_stats(ended: pd.DataFrame) -> dict:
    """How often the asking price came down before the listing ended, by how
    much, and how long each side stayed up.

    The two medians are descriptive and must be read in that direction: a
    listing is cut BECAUSE it is not moving, so the longer life of the cut ones
    is the symptom, never evidence that cutting slows a sale down.
    """
    first = pd.to_numeric(ended.get("first_price_eur"), errors="coerce")
    last = pd.to_numeric(ended.get("price_eur"), errors="coerce")
    ok = first.notna() & last.notna() & (first > 0)
    if int(ok.sum()) < MIN_CELL_EVENTS:
        return {}
    cut = ok & (last < first)
    out = {"cu": round(float(cut[ok].mean()), 3), "dn": int(ok.sum())}
    drops = ((first - last) / first)[cut]
    if len(drops):
        out["cp"] = round(float(drops.median()), 3)
    dur = ended["_dur"]
    if int(cut.sum()) >= MIN_CELL_EVENTS:
        out["cd"] = int(round(float(dur[cut].median())))
    held = ok & ~cut
    if int(held.sum()) >= MIN_CELL_EVENTS:
        out["hd"] = int(round(float(dur[held].median())))
    return out


def prepare(listings: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Listings with ``_dur`` (days on the market) and ``_event`` (it ended).

    The clock starts at ``first_seen_at`` (the date OLX shows the ad as posted
    or last refreshed) and stops at the last cycle that saw it alive. Rows older
    than ``WINDOW_DAYS`` are dropped: this describes the market as it is now,
    not as it was two years ago.
    """
    need = {"first_seen_at", "is_active"}
    if listings is None or listings.empty or not need.issubset(listings.columns):
        return pd.DataFrame()
    df = listings.copy()
    start = pd.to_datetime(df["first_seen_at"], utc=True, errors="coerce").dt.tz_localize(None)
    seen = (pd.to_datetime(df["last_scraped_at"], utc=True, errors="coerce").dt.tz_localize(None)
            if "last_scraped_at" in df.columns else pd.Series(pd.NaT, index=df.index))
    gone = (pd.to_datetime(df["deactivated_at"], utc=True, errors="coerce").dt.tz_localize(None)
            if "deactivated_at" in df.columns else pd.Series(pd.NaT, index=df.index))
    if now is None:
        now = seen.max()
    if pd.isna(now):
        now = pd.Timestamp.now("UTC").tz_localize(None)
    now = pd.Timestamp(now)

    active = df["is_active"] == True  # noqa: E712
    end = seen.fillna(gone)
    end = end.where(~active, seen.fillna(now))
    df["_event"] = ~active
    df["_dur"] = (end - start).dt.total_seconds() / 86400.0
    df = df[df["_dur"].notna() & (df["_dur"] >= 0) & (df["_dur"] <= MAX_DAYS)]
    df = df[start.reindex(df.index) >= now - pd.Timedelta(days=WINDOW_DAYS)]
    return df


def build_liquidity(
    listings: pd.DataFrame,
    relisted: set[str] | None = None,
    now: pd.Timestamp | None = None,
    now_year: int | None = None,
) -> dict:
    """``{"models": {(brand, model): rec}, "market": rec}`` for the public pages.

    A model reaches ``models`` with ``MIN_SELL_EVENTS`` listings observed all
    the way to their end, which is enough for the single median the deal cards
    and the model pages print. A page of its own needs ``MIN_EVENTS`` — see
    ``page_records`` — and every cut inside a record carries its own gate, so a
    thin district or price band is missing rather than estimated.
    """
    df = prepare(listings, now=now)
    out: dict = {"models": {}, "market": {}}
    if df.empty or not {"brand", "model"}.issubset(df.columns):
        return out
    if now_year is None:
        now_year = int(pd.Timestamp.now("UTC").year)
    relisted = relisted or set()
    back = df["olx_id"].astype(str).isin(relisted) if "olx_id" in df.columns else None

    market = _curve_stats(df)
    if market:
        market.update(_discount_stats(df[df["_event"]]))
        if back is not None:
            ended = df["_event"]
            if int(ended.sum()) >= MIN_EVENTS:
                market["rb"] = round(float(back[ended].mean()), 3)
        out["market"] = market

    for (brand, model), grp in df.groupby(["brand", "model"]):
        ended = grp[grp["_event"]]
        if len(ended) < MIN_SELL_EVENTS:
            continue
        rec = _curve_stats(grp)
        if not rec or "s30" not in rec:
            continue
        rec.update(_discount_stats(ended))
        if back is not None and len(ended) >= MIN_EVENTS:
            rec["rb"] = round(float(back.loc[ended.index].mean()), 3)
        pb = _price_cells(grp)
        if pb:
            rec["pb"] = pb
        ab = _age_cells(grp, now_year)
        if ab:
            rec["ab"] = ab
        dt = _district_cells(grp)
        if dt:
            rec["dt"] = dt
        out["models"][(str(brand), str(model))] = rec
    return out


def page_records(liquidity: dict) -> dict:
    """The subset deep enough to carry a page of its own — the gate the Worker
    then reads as "this model has a /liquidez page" (an absent key is a 404)."""
    return {key: rec for key, rec in (liquidity or {}).get("models", {}).items()
            if rec.get("n", 0) >= MIN_EVENTS and "s30" in rec}


def sell_speed_frame(liquidity: dict, min_events: int = MIN_SELL_EVENTS) -> pd.DataFrame:
    """The ``(brand, model, sell_days, sell_n)`` frame the rest of the pipeline
    already consumes, taken from the same curve the liquidity pages publish so
    the two can never disagree."""
    cols = ["brand", "model", "sell_days", "sell_n"]
    rows = [
        {"brand": brand, "model": model, "sell_days": rec["md"], "sell_n": rec["n"]}
        for (brand, model), rec in (liquidity or {}).get("models", {}).items()
        if rec.get("md") is not None and rec.get("n", 0) >= min_events
    ]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)[cols]
