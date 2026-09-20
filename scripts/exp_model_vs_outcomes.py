#!/usr/bin/env python3
"""Is the model's number right, measured against prices the market accepted?

Model error has only ever been reported against asks — cross-validated over
listings, where the target is what a seller wanted. That is the number in
``price_metrics.json`` and it says nothing about whether the car was worth it.

There is a subset where the ask and the accepted price are the same thing: a
car that left the market having never touched its price. Nobody argued the
seller down in public, the listing did not come back as a new advert, and the
car went at the number on the page. It is the closest the corpus gets to a
transaction, and on 2026-09-20 there were tens of thousands of them.

Two questions, then. Does the model agree with those prices, or does it read
high? And how often does it call a car that sold at its ask 15% underpriced —
each of those is a "cheap!" the engine would have said about a car priced
exactly right, which is the phantom-BUY mechanism measured on outcomes rather
than on a held-out fold.

Predictions come from the bundle's out-of-fold CV where the row was in
training, so the model is not graded on rows it memorised.

    OLX_DB_URL=postgresql+psycopg://olx@localhost:5432/olx_cars \\
        .venv/bin/python scripts/exp_model_vs_outcomes.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.exp_realised_outcomes import _load  # noqa: E402

PRICE_BANDS = [0, 4000, 8000, 15000, 25000, 1e9]
PRICE_LABELS = ["under 4k", "4-8k", "8-15k", "15-25k", "25k+"]
CHEAP_CALL_PCT = 15.0
MIN_CELL = 100


def _summary(d: pd.DataFrame, label: str) -> dict:
    err = d["err_pct"]
    return {
        "group": label,
        "n": len(d),
        "median_err": round(float(err.median()), 1),
        "mape": round(float(err.abs().median()), 1),
        "reads_high": round(float((err > 0).mean()), 3),
        f"calls_{int(CHEAP_CALL_PCT)}pct_cheap": round(
            float((err >= CHEAP_CALL_PCT).mean()), 3),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=None)
    ap.add_argument("--max-days", type=float, default=120.0,
                    help="a car that sat this long before vanishing was probably "
                         "given up on, not sold")
    args = ap.parse_args(argv)

    from src.analytics.computed_columns import enrich_listings
    from src.analytics.outcomes import build_outcomes
    from src.analytics.price_model import load_model, predict_prices

    listings, snaps, pairs = _load(args.db)
    listings = listings[listings["source"].isin(["olx", "standvirtual"])].copy()
    listings = enrich_listings(listings)
    out = build_outcomes(listings, snaps, pairs)

    clean = out[(~out["still_active"].astype(bool)) & (~out["cut"].astype(bool))
                & out["last_ask"].notna() & (out["last_ask"] > 0)].copy()
    print(f"cars {len(out):,} → left without ever touching the ask: {len(clean):,} "
          f"({len(clean) / max(len(out), 1) * 100:.0f}%)")
    sold = clean[clean["days"] <= args.max_days]
    print(f"of those, gone within {args.max_days:.0f} days (read as sold): {len(sold):,}")

    saved = load_model(max_age_hours=90 * 24)
    if saved is None:
        print("no model bundle — run the training step first")
        return 1
    models, cat_maps, metrics, oof, calibrator, uncertainty = saved
    cv = metrics.get("mape") or metrics.get("median_ape") or metrics.get("mape_median")
    print(f"model bundle: {len(oof):,} out-of-fold rows"
          + (f", cross-validated error on asks: {cv}" if cv else ""))

    movers = out[(~out["still_active"].astype(bool))
                 & (out["ask_change_pct"] <= -10.0) & out["first_ask"].notna()].copy()
    wanted = set(sold["car_id"]) | set(movers["car_id"])
    first_ads = listings[listings["olx_id"].astype(str).isin(wanted)]
    first_ads = first_ads.drop_duplicates("olx_id").reset_index(drop=True)
    preds = predict_prices(models, cat_maps, first_ads, oof_preds=oof,
                           median_calibrator=calibrator, uncertainty_bundle=uncertainty)
    pred_by_id = dict(zip(first_ads["olx_id"].astype(str),
                          preds["predicted_price"].reindex(first_ads.index).values))

    d = sold.copy()
    d["pred"] = d["car_id"].map(pred_by_id)
    d["oof"] = d["car_id"].isin(set(map(str, oof)))
    d = d[d["pred"].notna() & (d["pred"] > 0)]
    d["err_pct"] = (d["pred"] / d["last_ask"] - 1) * 100
    d["band"] = pd.cut(d["last_ask"], PRICE_BANDS, labels=PRICE_LABELS)
    print(f"priced by the model: {len(d):,} ({d['oof'].mean() * 100:.0f}% out-of-fold)\n")

    rows = [_summary(d, "all clean exits")]
    if d["oof"].any():
        rows.append(_summary(d[d["oof"]], "out-of-fold only"))
    for band, g in d.groupby("band", observed=True):
        if len(g) >= MIN_CELL:
            rows.append(_summary(g, f"  band {band}"))
    print(pd.DataFrame(rows).to_string(index=False))

    print("\nthe converse — cars whose seller had to come down by 10% or more:")
    movers["pred"] = movers["car_id"].map(pred_by_id)
    movers = movers[movers["pred"].notna() & (movers["pred"] > 0)]
    if len(movers) >= MIN_CELL:
        over = (movers["pred"] / movers["first_ask"] - 1) * 100
        print(f"  n={len(movers):,}  median model-vs-first-ask {over.median():+.1f}%  "
              f"called them underpriced: {float((over >= CHEAP_CALL_PCT).mean()):.3f}  "
              f"called them overpriced: {float((over <= -CHEAP_CALL_PCT).mean()):.3f}")
        print("  (the ask was too high by the market's own verdict, so a model that "
              "knew would read BELOW it)")
    else:
        print(f"  only {len(movers)} priced — too thin to read")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
