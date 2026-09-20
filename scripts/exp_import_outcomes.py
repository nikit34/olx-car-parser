#!/usr/bin/env python3
"""Does an imported car actually go cheaper, or slower, than a national one?

The site already shows an import flag, and the ISV line next to it implies a
cost the buyer should price in. That is an argument, not a measurement: nothing
in the corpus has ever been checked for whether import-flagged cars end up
accepting less money or sitting longer than comparable national ones.

``origin`` is not a feature of the price model — the full feature list is
year/mileage/engine/horsepower/photos/description/seats/seller-count/plate plus
brand, model, fuel, transmission, segment, generation, district, sub-model and
trim. So the model's prediction is a clean control: it knows the car's specs
and nothing about where it was first registered. If imports systematically
leave at a price below what the specs alone predict, the model reads high on
them, and the gap is the import discount.

Three questions, each against the outcomes table rather than against asks:

1. accepted price — over cars that left without ever touching the ask, is the
   model's error different for imports than for nationals?
2. time on market — of cars old enough for a 30- and 60-day answer, does the
   import share that has gone lag the national one inside the same cell?
3. haggling — do imports cut their ask more often, and by more?
4. the ask itself — do import sellers start above what the specs predict, which
   would explain a slower sale without any buyer preference at all?

Composition is the trap: imports skew old and cheap, and both of those move
every one of these numbers on their own. So each comparison is also run inside
(brand, model, year) or (brand, model, price band) cells that hold both arms,
with a bootstrap over cells to say whether the gap is bigger than the noise.

    OLX_DB_URL=postgresql+psycopg://olx@localhost:5432/olx_cars \\
        .venv/bin/python scripts/exp_import_outcomes.py
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(os.environ.get("OLX_REPO_ROOT", Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(REPO_ROOT))

PRICE_BANDS = [0, 4000, 8000, 15000, 25000, 1e9]
PRICE_LABELS = ["under 4k", "4-8k", "8-15k", "15-25k", "25k+"]
KM_BANDS = [0, 50_000, 100_000, 150_000, 200_000, 250_000, 1e9]
KM_LABELS = ["<50k", "50-100k", "100-150k", "150-200k", "200-250k", "250k+"]
ASK_BANDS = [-1e9, -10, 0, 10, 20, 1e9]
ASK_LABELS = ["10%+ under model", "0-10% under", "0-10% over", "10-20% over", "20%+ over"]
MIN_ARM = 8
MIN_GROUP = 100
BOOT = 2000
RNG = np.random.default_rng(20260920)


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


def _arm(listings: pd.DataFrame, text_flags: bool) -> pd.Series:
    """Import status per advert: 'imported', 'national', or NaN for unknown."""
    from src.analytics.valuations import _IMPORT_NEG, _IMPORT_POS, _strip_accents

    origin = listings.get("origin")
    out = pd.Series(np.nan, index=listings.index, dtype=object)
    if origin is not None:
        out[origin == "imported"] = "imported"
        out[origin == "national"] = "national"
    if not text_flags:
        return out
    hay = (listings.get("title", pd.Series("", index=listings.index)).map(_strip_accents)
           + " "
           + listings.get("description", pd.Series("", index=listings.index)).map(_strip_accents))
    pos = hay.str.contains(_IMPORT_POS, regex=True, na=False)
    neg = hay.str.contains(_IMPORT_NEG, regex=True, na=False)
    out[out.isna() & pos & ~neg] = "imported"
    out[out.isna() & neg & ~pos] = "national"
    return out


def _boot_delta(cells: list[tuple[float, int]]) -> tuple[float, float, float]:
    """Weighted mean of per-cell deltas, with a percentile interval over cells."""
    if not cells:
        return float("nan"), float("nan"), float("nan")
    d = np.array([c[0] for c in cells], dtype=float)
    w = np.array([c[1] for c in cells], dtype=float)
    point = float(np.average(d, weights=w))
    idx = RNG.integers(0, len(d), size=(BOOT, len(d)))
    draws = np.array([np.average(d[i], weights=w[i]) for i in idx])
    return point, float(np.percentile(draws, 5)), float(np.percentile(draws, 95))


def _paired(df: pd.DataFrame, keys: list[str], value, label: str) -> None:
    cells = []
    n_imp = n_nat = 0
    for _, g in df.groupby(keys, observed=True):
        imp = g[g["arm"] == "imported"]
        nat = g[g["arm"] == "national"]
        if len(imp) < MIN_ARM or len(nat) < MIN_ARM:
            continue
        cells.append((value(imp) - value(nat), min(len(imp), len(nat))))
        n_imp += len(imp)
        n_nat += len(nat)
    point, lo, hi = _boot_delta(cells)
    if not cells:
        print(f"  {label}: no cell holds {MIN_ARM}+ of each arm")
        return
    verdict = "real" if (lo > 0) == (hi > 0) else "noise"
    print(f"  {label}: {point:+.2f} [{lo:+.2f}, {hi:+.2f}]  "
          f"{len(cells)} cells, {n_imp:,} imported vs {n_nat:,} national → {verdict}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=None)
    ap.add_argument("--max-days", type=float, default=120.0,
                    help="a car that sat this long before vanishing was probably "
                         "given up on, not sold")
    ap.add_argument("--source", default=None,
                    help="restrict to one platform; the two carry different advert "
                         "lifecycles and different origin coverage")
    ap.add_argument("--text-flags", action="store_true",
                    help="also read import status out of the advert text, the way "
                         "the published flag does; off means structured origin only")
    args = ap.parse_args(argv)

    from src.analytics.computed_columns import enrich_listings
    from src.analytics.outcomes import build_outcomes
    from src.analytics.price_model import load_model, predict_prices

    listings, snaps, pairs = _load(args.db)
    keep = [args.source] if args.source else ["olx", "standvirtual"]
    listings = listings[listings["source"].isin(keep)].copy()
    listings = enrich_listings(listings)
    listings["arm"] = _arm(listings, args.text_flags)
    seen = listings["arm"].value_counts(dropna=False)
    print(f"listings {len(listings):,}  imported {int(seen.get('imported', 0)):,}  "
          f"national {int(seen.get('national', 0)):,}  unknown "
          f"{len(listings) - int(seen.get('imported', 0)) - int(seen.get('national', 0)):,}")

    out = build_outcomes(listings, snaps, pairs)
    arm_by_id = dict(zip(listings["olx_id"].astype(str), listings["arm"]))
    out["arm"] = out["car_id"].map(arm_by_id)
    out = out[out["arm"].notna()].copy()
    out["band"] = pd.cut(out["first_ask"], PRICE_BANDS, labels=PRICE_LABELS)
    out["year3"] = (out["year"] // 3 * 3)
    print(f"cars with a known origin: {len(out):,}  "
          f"({int((out['arm'] == 'imported').sum()):,} imported)")

    saved = load_model(max_age_hours=90 * 24)
    if saved is None:
        print("   no model bundle — run the training step first")
        return 1
    models, cat_maps, metrics, oof, calibrator, uncertainty = saved
    first_ads = listings[listings["olx_id"].astype(str).isin(set(out["car_id"]))]
    first_ads = first_ads.drop_duplicates("olx_id").reset_index(drop=True)
    preds = predict_prices(models, cat_maps, first_ads, oof_preds=oof,
                           median_calibrator=calibrator, uncertainty_bundle=uncertainty)
    pred_by_id = dict(zip(first_ads["olx_id"].astype(str),
                          preds["predicted_price"].reindex(first_ads.index).values))
    out["pred"] = out["car_id"].map(pred_by_id)
    for col in ("mileage_km", "fuel_type", "transmission", "district"):
        if col in listings.columns:
            out[col] = out["car_id"].map(dict(zip(listings["olx_id"].astype(str),
                                                  listings[col])))
    out["kmband"] = pd.cut(pd.to_numeric(out.get("mileage_km"), errors="coerce"),
                           KM_BANDS, labels=KM_LABELS)
    out["ask_vs_model"] = (out["first_ask"] / out["pred"] - 1) * 100
    out.loc[out["pred"].isna() | (out["pred"] <= 0), "ask_vs_model"] = np.nan
    out["askband"] = pd.cut(out["ask_vs_model"], ASK_BANDS, labels=ASK_LABELS)

    print("\n1. accepted price, over cars that left without ever touching the ask")
    clean = out[(~out["still_active"].astype(bool)) & (~out["cut"].astype(bool))
                & out["last_ask"].notna() & (out["last_ask"] > 0)
                & (out["days"] <= args.max_days)].copy()
    clean = clean[clean["pred"].notna() & (clean["pred"] > 0)]
    clean["err_pct"] = (clean["pred"] / clean["last_ask"] - 1) * 100
    print(f"   priced by the model: {len(clean):,}")
    rows = []
    for arm, g in clean.groupby("arm", observed=True):
        rows.append({"arm": arm, "n": len(g),
                     "median_err": round(float(g["err_pct"].median()), 1),
                     "median_ask": int(g["last_ask"].median()),
                     "median_year": int(g["year"].median()) if g["year"].notna().any() else 0,
                     "median_days": round(float(g["days"].median()), 1)})
    print(pd.DataFrame(rows).to_string(index=False))
    print("   (a higher median_err means the model reads high — the car went for "
          "less than its specs alone predict)")
    print("   model error gap, imported minus national, in pp:")
    med = lambda g: float(g["err_pct"].median())  # noqa: E731
    _paired(clean, ["brand", "model", "year3"], med, "by brand+model+year bucket")
    _paired(clean, ["brand", "model", "band"], med, "by brand+model+price band")
    _paired(clean, ["band"], med, "by price band only")

    print("\n2. time on market, share gone inside the window")
    now = pd.to_datetime(listings["last_seen_at"], utc=True, errors="coerce").max()
    first = pd.to_datetime(out["first_seen_at"], utc=True, errors="coerce")
    age = (now - first).dt.total_seconds() / 86400.0
    for window in (30, 60):
        obs = out[(age >= window)].copy()
        obs["gone"] = (~obs["still_active"].astype(bool)) & (obs["days"] <= window)
        share = lambda g: float(g["gone"].mean()) * 100  # noqa: E731
        rows = [{"arm": a, "n": len(g), f"gone_by_{window}d_pct": round(share(g), 1)}
                for a, g in obs.groupby("arm", observed=True) if len(g) >= MIN_GROUP]
        print(f"   window {window}d, cars old enough to answer: {len(obs):,}")
        print(pd.DataFrame(rows).to_string(index=False))
        print(f"   gap in pp, imported minus national:")
        _paired(obs, ["brand", "model", "year3"], share, "by brand+model+year bucket")
        _paired(obs, ["brand", "model", "band"], share, "by brand+model+price band")
        _paired(obs, ["band"], share, "by price band only")
        print("   and the same gap once the ask itself is held:")
        _paired(obs[obs["askband"].notna()], ["brand", "model", "askband"], share,
                "by brand+model+ask vs model")
        _paired(obs[obs["askband"].notna()], ["band", "askband"], share,
                "by price band+ask vs model")
        print("   and once the odometer is held too:")
        _paired(obs[obs["kmband"].notna()], ["brand", "model", "year3", "kmband"], share,
                "by brand+model+year bucket+km bucket")
        _paired(obs[obs["kmband"].notna()], ["brand", "model", "kmband"], share,
                "by brand+model+km bucket")

    print("\n3. haggling, over cars that have finished")
    fin = out[~out["still_active"].astype(bool)].copy()
    cut_share = lambda g: float(g["cut"].astype(bool).mean()) * 100  # noqa: E731
    drop = lambda g: float(g.loc[g["cut"].astype(bool), "ask_change_pct"].median())  # noqa: E731
    rows = []
    for arm, g in fin.groupby("arm", observed=True):
        c = g[g["cut"].astype(bool)]
        rows.append({"arm": arm, "n": len(g), "cut_pct": round(cut_share(g), 1),
                     "median_drop_pct": round(drop(g), 1) if len(c) else float("nan")})
    print(pd.DataFrame(rows).to_string(index=False))
    print("   cut-share gap in pp, imported minus national:")
    _paired(fin, ["brand", "model", "year3"], cut_share, "by brand+model+year bucket")
    _paired(fin, ["band"], cut_share, "by price band only")
    cutters = fin[fin["cut"].astype(bool)]
    print("   size-of-cut gap in pp, imported minus national:")
    _paired(cutters, ["band"], drop, "by price band only")

    print("\n4. the ask itself, first advert against the model's number")
    asked = out[out["ask_vs_model"].notna()].copy()
    rows = []
    for arm, g in asked.groupby("arm", observed=True):
        rows.append({"arm": arm, "n": len(g),
                     "median_ask_vs_model": round(float(g["ask_vs_model"].median()), 1),
                     "asks_10pct_over": round(float((g["ask_vs_model"] >= 10).mean()) * 100, 1)})
    print(pd.DataFrame(rows).to_string(index=False))
    over = lambda g: float(g["ask_vs_model"].median())  # noqa: E731
    print("   gap in pp, imported minus national:")
    _paired(asked, ["brand", "model", "year3"], over, "by brand+model+year bucket")
    _paired(asked, ["brand", "model", "band"], over, "by brand+model+price band")
    _paired(asked, ["band"], over, "by price band only")

    print("\n5. what an import actually is, inside the cells the gaps are measured in")
    rows = []
    for _, g in out.groupby(["brand", "model", "year3"], observed=True):
        imp = g[g["arm"] == "imported"]
        nat = g[g["arm"] == "national"]
        if len(imp) < MIN_ARM or len(nat) < MIN_ARM:
            continue
        w = min(len(imp), len(nat))
        km_i = pd.to_numeric(imp.get("mileage_km"), errors="coerce").median()
        km_n = pd.to_numeric(nat.get("mileage_km"), errors="coerce").median()
        rows.append({
            "w": w,
            "d_km": (km_i - km_n) if pd.notna(km_i) and pd.notna(km_n) else np.nan,
            "d_ask_pct": (imp["first_ask"].median() / nat["first_ask"].median() - 1) * 100,
            "d_diesel_pp": (float(imp["fuel_type"].astype(str).str.lower()
                                  .str.startswith("dies").mean())
                            - float(nat["fuel_type"].astype(str).str.lower()
                                    .str.startswith("dies").mean())) * 100,
        })
    comp = pd.DataFrame(rows)
    if len(comp):
        w = comp["w"]
        print(f"   {len(comp)} cells with both arms; weighted median difference, "
              f"imported minus national:")
        for col, unit in (("d_km", "km on the clock"), ("d_ask_pct", "% on the ask"),
                          ("d_diesel_pp", "pp diesel")):
            v = comp[col].dropna()
            if len(v):
                ww = w.loc[v.index]
                print(f"     {float(np.average(v, weights=ww)):+.0f} {unit}")

    print("\n6. two ways the time gap could be an artefact")
    window = 30
    now = pd.to_datetime(listings["last_seen_at"], utc=True, errors="coerce").max()
    age = (now - pd.to_datetime(out["first_seen_at"], utc=True, errors="coerce")
           ).dt.total_seconds() / 86400.0
    obs = out[age >= window].copy()
    obs["gone"] = (~obs["still_active"].astype(bool)) & (obs["days"] <= window)
    share = lambda g: float(g["gone"].mean()) * 100  # noqa: E731
    for arm, g in obs.groupby("arm", observed=True):
        print(f"   {arm}: {float((g['n_ads'] > 1).mean()) * 100:.1f}% of cars were "
              f"glued from more than one advert")
    print("   so the same gap over cars that only ever had one advert, where no "
          "gluing could have stretched the clock:")
    single = obs[obs["n_ads"] == 1]
    _paired(single, ["brand", "model", "year3"], share, "by brand+model+year bucket")
    _paired(single[single["kmband"].notna()], ["brand", "model", "kmband"], share,
            "by brand+model+km bucket")

    print("   and how much a bigger odometer alone is worth, measured on nationals "
          "only inside the same cells:")
    deltas = []
    for _, g in obs[obs["arm"] == "national"].groupby(["brand", "model", "year3"],
                                                      observed=True):
        km = pd.to_numeric(g.get("mileage_km"), errors="coerce")
        g = g[km.notna()]
        km = km[km.notna()]
        if len(g) < 2 * MIN_ARM:
            continue
        hi, lo = g[km >= km.median()], g[km < km.median()]
        if len(hi) < MIN_ARM or len(lo) < MIN_ARM:
            continue
        dkm = float(km[km >= km.median()].median() - km[km < km.median()].median())
        if dkm <= 0:
            continue
        deltas.append(((share(hi) - share(lo)) / dkm * 10_000, min(len(hi), len(lo))))
    point, lo_ci, hi_ci = _boot_delta(deltas)
    print(f"     {point:+.2f} pp of 30-day sales per 10,000 km  "
          f"[{lo_ci:+.2f}, {hi_ci:+.2f}], {len(deltas)} cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
