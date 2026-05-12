#!/usr/bin/env python3
"""Precompute dashboard witnesses for the stlite static dashboard.

The browser-side dashboard (stlite on Cloudflare Pages) can't run the
LightGBM / sklearn inference pipeline that ``src.dashboard.data_loader.load_all``
fires on every cold start. This script runs that pipeline ONCE in CI
against the local SQLite, then serialises every artifact the dashboard
needs into ``data/dashboard/`` as parquet / JSON.

Outputs (uploaded to the ``latest-data`` GitHub Release by scrape-ci):

  listings.parquet              full enriched listings DataFrame
  history.parquet               aggregated daily market stats (trend charts)
  snapshots.parquet             per-listing price_snapshots (deal cards: "dropped €X")
  signals.parquet               compute_signals output — the deal feed
  predictions.parquet           per-olx_id predicted_price + bands
  contributions.parquet         long-form TreeSHAP deltas (olx_id, label, delta_eur)
  importance.parquet            feature importance
  grouped_importance.parquet    grouped feature importance
  shap_importance.parquet       SHAP-based feature importance
  turnover.parquet              compute_turnover_stats output
  portfolio.parquet             portfolio_deals (currently empty)
  unmatched.parquet             unmatched_listings
  brands_models.json            {brand: [model, ...]} for filter dropdowns
  manifest.json                 build timestamp + row counts + file sizes

Use:
    python scripts/build_dashboard_data.py
    python scripts/build_dashboard_data.py --db data/olx_cars.db --out data/dashboard
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _to_parquet(df: pd.DataFrame, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, compression="zstd", index=False)
    return path.stat().st_size


def _contributions_to_long(
    contributions: dict[str, dict],
) -> pd.DataFrame:
    """Flatten ``{olx_id: {baseline_eur, predicted_eur, deltas: [(label, eur), ...]}}``
    into a long-form table with one row per (olx_id, feature)."""
    rows: list[dict] = []
    for olx_id, payload in contributions.items():
        if not isinstance(payload, dict):
            continue
        baseline = payload.get("baseline_eur")
        predicted = payload.get("predicted_eur")
        for rank, item in enumerate(payload.get("deltas") or []):
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            label, delta = item[0], item[1]
            rows.append({
                "olx_id": str(olx_id),
                "rank": rank,
                "feature_label": str(label),
                "delta_eur": float(delta) if delta is not None else None,
                "baseline_eur": float(baseline) if baseline is not None else None,
                "predicted_eur": float(predicted) if predicted is not None else None,
            })
    if not rows:
        return pd.DataFrame(columns=[
            "olx_id", "rank", "feature_label", "delta_eur",
            "baseline_eur", "predicted_eur",
        ])
    return pd.DataFrame(rows)


def _build(db_path: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Import here so the ``--help`` path doesn't pay the cost of loading
    # the whole analytics stack (LightGBM, sklearn, etc.).
    from src.storage.database import init_db, get_session
    from src.storage.repository import (
        get_listings_df, get_price_history_df,
        get_price_snapshots_df,
        get_unmatched_df, get_portfolio_df,
    )
    from src.analytics.computed_columns import enrich_listings
    from src.analytics.turnover import compute_turnover_stats
    from src.dashboard.data_loader import compute_signals

    print(f"[build] loading DB {db_path}", flush=True)
    init_db(str(db_path))
    session = get_session()

    t0 = time.perf_counter()
    listings = get_listings_df(session)
    history = get_price_history_df(session)
    print(f"[build]   listings: {len(listings):>6}  history: {len(history):>6}  "
          f"({time.perf_counter() - t0:.1f}s)", flush=True)

    if listings.empty:
        raise SystemExit("DB has no listings — nothing to build")

    listings = enrich_listings(listings)

    # Read-time mileage sanity gate — mirrors the dashboard's load_all logic.
    _SANITY_MAX_MILEAGE_KM = 1_000_000
    if "real_mileage_km" in listings.columns:
        real_km = listings["real_mileage_km"]
        plausible = (real_km > 0) & (real_km <= _SANITY_MAX_MILEAGE_KM)
        listings["mileage_km"] = real_km.where(plausible).fillna(listings["mileage_km"])

    t0 = time.perf_counter()
    turnover = compute_turnover_stats(listings)
    print(f"[build]   turnover stats: {len(turnover):>6}  "
          f"({time.perf_counter() - t0:.1f}s)", flush=True)

    t0 = time.perf_counter()
    (
        signals, importance, grouped_importance, predictions,
        contributions, shap_importance,
    ) = compute_signals(listings, history, turnover=turnover)
    print(f"[build]   compute_signals: signals={len(signals):>5}  "
          f"predictions={len(predictions):>5}  contributions={len(contributions):>5}  "
          f"({time.perf_counter() - t0:.1f}s)", flush=True)

    # Per-listing price snapshots — pages 2/3 query with since_days=365
    # and the deal-card "dropped €X" widget reads it via since_days=120.
    # 530k rows fit in ~5 MB zstd parquet — small enough to ship the full
    # year and let the browser filter, beats shipping multiple windows.
    snapshots = get_price_snapshots_df(session, since_days=365)
    portfolio = get_portfolio_df(session)
    unmatched = get_unmatched_df(session)

    brands_models: dict[str, list[str]] = {}
    pairs = listings[["brand", "model"]].drop_duplicates()
    for brand, grp in pairs.groupby("brand", sort=False):
        brands_models[str(brand)] = grp["model"].dropna().astype(str).tolist()

    contributions_df = _contributions_to_long(contributions)

    sizes: dict[str, int] = {}
    sizes["listings.parquet"] = _to_parquet(listings, out_dir / "listings.parquet")
    sizes["history.parquet"] = _to_parquet(history, out_dir / "history.parquet")
    sizes["snapshots.parquet"] = _to_parquet(snapshots, out_dir / "snapshots.parquet")
    sizes["signals.parquet"] = _to_parquet(signals, out_dir / "signals.parquet")
    sizes["predictions.parquet"] = _to_parquet(predictions, out_dir / "predictions.parquet")
    sizes["contributions.parquet"] = _to_parquet(contributions_df, out_dir / "contributions.parquet")
    sizes["importance.parquet"] = _to_parquet(importance, out_dir / "importance.parquet")
    sizes["grouped_importance.parquet"] = _to_parquet(grouped_importance, out_dir / "grouped_importance.parquet")
    sizes["shap_importance.parquet"] = _to_parquet(shap_importance, out_dir / "shap_importance.parquet")
    sizes["turnover.parquet"] = _to_parquet(turnover, out_dir / "turnover.parquet")
    sizes["portfolio.parquet"] = _to_parquet(portfolio, out_dir / "portfolio.parquet")
    sizes["unmatched.parquet"] = _to_parquet(unmatched, out_dir / "unmatched.parquet")

    brands_path = out_dir / "brands_models.json"
    brands_path.write_text(json.dumps(brands_models, ensure_ascii=False))
    sizes["brands_models.json"] = brands_path.stat().st_size

    manifest = {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rows": {
            "listings": len(listings),
            "history": len(history),
            "snapshots": len(snapshots),
            "signals": len(signals),
            "predictions": len(predictions),
            "contributions": len(contributions_df),
            "turnover": len(turnover),
            "portfolio": len(portfolio),
            "unmatched": len(unmatched),
        },
        "files_bytes": sizes,
        "total_bytes": sum(sizes.values()),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    total_mb = manifest["total_bytes"] / 1e6
    print(f"[build] DONE — {total_mb:.2f} MB across {len(sizes)} files", flush=True)
    for name, sz in sorted(sizes.items(), key=lambda kv: -kv[1]):
        print(f"           {sz/1e6:>6.2f} MB  {name}")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--db", type=Path,
        default=REPO_ROOT / "data" / "olx_cars.db",
        help="SQLite database path (default: data/olx_cars.db)",
    )
    ap.add_argument(
        "--out", type=Path,
        default=REPO_ROOT / "data" / "dashboard",
        help="Output directory for parquet/json artifacts (default: data/dashboard)",
    )
    args = ap.parse_args()
    if not args.db.exists():
        raise SystemExit(f"DB not found: {args.db}")
    _build(args.db, args.out)


if __name__ == "__main__":
    main()
