#!/usr/bin/env python3
"""Does the model's "underpriced" survive contact with what the car did next?

The decision engine has only ever been checked against asks. On 2026-09-19 it
was finally checked against an outcome — whether a listing left the market —
and what it surfaced was indistinguishable from the market it was drawn from.
That measure is weak, though: leaving the market says nothing about the price.

This asks the sharper question the corpus can answer. Take each car at its
first advert, ask the model what it thinks the car is worth, and then look at
what the seller ended up asking. A car the model calls cheap should not have to
come down; a car that comes down 10% before it goes was not cheap. Predictions
come from the bundle's out-of-fold CV where the row was in training, so the
model is not being graded on rows it memorised.

Run it against the host:

    OLX_DB_URL=postgresql+psycopg://olx@localhost:5432/olx_cars \\
        .venv/bin/python scripts/exp_realised_outcomes.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

BUCKETS = [-np.inf, -10, 0, 5, 10, 15, 20, 30, np.inf]
BUCKET_LABELS = ["ask above model", "at model", "0-5%", "5-10%", "10-15%",
                 "15-20%", "20-30%", "30%+"]
MIN_BUCKET_N = 30
PRICE_BANDS = [0, 4000, 8000, 15000, 25000, 1e9]
PRICE_LABELS = ["under 4k", "4-8k", "8-15k", "15-25k", "25k+"]


def _load(db_url: str | None):
    from sqlalchemy import text
    from src.storage.database import init_db, get_engine, get_session
    from src.storage.repository import get_listings_df, get_relist_events_df
    init_db(db_url)
    session = get_session()
    listings = get_listings_df(session)
    pairs = get_relist_events_df(session)
    snaps = pd.read_sql(
        text("SELECT l.olx_id, ps.price_eur, ps.scraped_at FROM price_snapshots ps "
             "JOIN listings l ON l.id = ps.listing_id WHERE l.source IN "
             "('olx','standvirtual')"),
        get_engine(db_url),
    )
    return listings, snaps, pairs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=None, help="database URL (defaults to OLX_DB_URL)")
    ap.add_argument("--min-days", type=float, default=30.0,
                    help="ignore cars whose story is shorter than this")
    args = ap.parse_args(argv)

    from src.analytics.computed_columns import enrich_listings
    from src.analytics.outcomes import build_outcomes
    from src.analytics.price_model import load_model, predict_prices

    listings, snaps, pairs = _load(args.db)
    listings = listings[listings["source"].isin(["olx", "standvirtual"])].copy()
    listings = enrich_listings(listings)
    print(f"listings {len(listings):,}  snapshots {len(snaps):,}  relist pairs {len(pairs):,}")

    out = build_outcomes(listings, snaps, pairs)
    print(f"cars {len(out):,}  (more than one advert: {int((out['n_ads'] > 1).sum()):,}, "
          f"still on sale: {int(out['still_active'].sum()):,})")

    saved = load_model(max_age_hours=90 * 24)
    if saved is None:
        print("no model bundle — run the training step first")
        return 1
    models, cat_maps, _metrics, oof, calibrator, uncertainty = saved
    print(f"model bundle loaded, out-of-fold rows: {len(oof):,}")

    first_ads = listings[listings["olx_id"].astype(str).isin(set(out["car_id"]))].copy()
    first_ads = first_ads.drop_duplicates("olx_id")
    preds = predict_prices(models, cat_maps, first_ads, oof_preds=oof,
                           median_calibrator=calibrator, uncertainty_bundle=uncertainty)
    first_ads = first_ads.reset_index(drop=True)
    pred_by_id = dict(zip(first_ads["olx_id"].astype(str),
                          preds["predicted_price"].reindex(first_ads.index).values))
    in_oof = set(map(str, oof))

    d = out.copy()
    d["pred"] = d["car_id"].map(pred_by_id)
    d["oof"] = d["car_id"].isin(in_oof)
    d = d[d["pred"].notna() & d["first_ask"].notna() & (d["pred"] > 0)]
    d["underpriced_pct"] = (1 - d["first_ask"] / d["pred"]) * 100
    d = d[d["days"] >= args.min_days]
    print(f"cars with a prediction and at least {args.min_days:.0f} days of story: {len(d):,} "
          f"({d['oof'].mean() * 100:.0f}% scored out-of-fold)")

    d["bucket"] = pd.cut(d["underpriced_pct"], BUCKETS, labels=BUCKET_LABELS)
    report = d.groupby("bucket", observed=True).agg(
        cars=("car_id", "size"),
        sold=("sold", "mean"),
        relisted=("n_ads", lambda s: float((s > 1).mean())),
        median_days=("days", "median"),
        median_ask_change=("ask_change_pct", "median"),
        cut_share=("cut", "mean"),
        held_and_sold=("car_id", lambda idx: float(
            (d.loc[idx.index, "sold"] & ~d.loc[idx.index, "cut"]).mean())),
    ).round(3)
    report = report[report["cars"] >= MIN_BUCKET_N]
    print("\nby how far the first ask sat below the model:")
    print(report.to_string())

    print("\ncorrelations (Spearman, over cars with a prediction):")
    for col, label in (("ask_change_pct", "ask change %"), ("days", "days on the market")):
        rho = d["underpriced_pct"].corr(d[col], method="spearman")
        print(f"  underpriced% vs {label:<20}: {rho:+.3f}")
    for flag in ("sold", "cut"):
        rho = d["underpriced_pct"].corr(d[flag].astype(float), method="spearman")
        print(f"  underpriced% vs {flag:<20}: {rho:+.3f}")

    print("\nthe same over out-of-fold rows only:")
    o = d[d["oof"]]
    if len(o) >= MIN_BUCKET_N:
        print(f"  n={len(o):,}  ask change {o['underpriced_pct'].corr(o['ask_change_pct'], method='spearman'):+.3f}"
              f"  cut {o['underpriced_pct'].corr(o['cut'].astype(float), method='spearman'):+.3f}")
    print(f"\nbaseline: {d['sold'].mean():.3f} sold, {d['cut'].mean():.3f} cut the ask, "
          f"median ask change {d['ask_change_pct'].median():+.2f}%, "
          f"median {d['days'].median():.0f} days")

    print("\nthe same inside each price band, because 'cheap' and 'below the model' "
          "are the same cars until proven otherwise:")
    d["price_band"] = pd.cut(d["first_ask"], PRICE_BANDS, labels=PRICE_LABELS)
    for band, g in d.groupby("price_band", observed=True):
        if len(g) < MIN_BUCKET_N * 4:
            continue
        lo = g[g["underpriced_pct"] <= 0]
        hi = g[g["underpriced_pct"] >= 20]
        rho = g["underpriced_pct"].corr(g["cut"].astype(float), method="spearman")
        line = (f"  {band:<12} n={len(g):>6}  cut {g['cut'].mean():.3f}  "
                f"median change {g['ask_change_pct'].median():+.2f}%  "
                f"spearman(underpriced, cut) {rho:+.3f}")
        if len(lo) >= MIN_BUCKET_N and len(hi) >= MIN_BUCKET_N:
            line += (f"  | ask above model {lo['cut'].mean():.3f} vs 20%+ below "
                     f"{hi['cut'].mean():.3f} (n={len(lo)}/{len(hi)})")
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
