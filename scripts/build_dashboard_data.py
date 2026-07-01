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
    # One build timestamp reused by the manifest AND models.json (the public
    # "preços atualizados em …" freshness signal the Worker renders).
    built_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # Import here so the ``--help`` path doesn't pay the cost of loading
    # the whole analytics stack (LightGBM, sklearn, etc.).
    from src.storage.database import init_db, get_session
    from src.storage.repository import (
        get_listings_df, get_price_history_df,
        get_price_snapshots_df,
        get_unmatched_df, get_portfolio_df,
    )
    from src.analytics.computed_columns import enrich_listings
    from src.analytics.turnover import compute_turnover_stats, compute_sell_speed_by_model
    from src.analytics.valuations import build_valuations
    from src.analytics.model_pages import build_model_pages
    from src.analytics.price_model import load_model, value_configs
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

    # Sanity-gate the LLM mileage read against the structured attribute
    # before letting it override (absolute cap + 10× relative gate). Without
    # the relative gate, pre-2026-05-11 dirty-title rows like JltT9 (price
    # leaked as mileage: 9000 vs the real 355000) would render as 9 km on
    # the deal card.
    from src.parser.llm_enrichment import merge_real_mileage
    listings = merge_real_mileage(listings)

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

    # valuations.json — the public "value any listing" lookup (Tier-2), fetched
    # by the Worker's /avaliar paste-a-link tool. Active+priced listings only;
    # ~0.9 MB gzipped for ~18k cars. Uploaded to the Release by the existing
    # ``data/dashboard/*.json`` glob in scrape-ci (no workflow change needed).
    sell_speed = compute_sell_speed_by_model(listings)
    valuations = build_valuations(listings, predictions, sell_speed)
    val_path = out_dir / "valuations.json"
    # allow_nan=False: a non-finite value (pandas NaN leaking through) emits the
    # literal `NaN`, which is valid for Python's json.load but breaks the Worker's
    # JSON.parse. Fail LOUDLY and skip the file rather than ship an unparseable
    # blob (or crash the other witnesses).
    try:
        blob = json.dumps(valuations, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        val_path.write_text(blob)
        sizes["valuations.json"] = val_path.stat().st_size
        print(f"[build]   valuations: {len(valuations.get('cars', {})):>6} cars  "
              f"({sizes['valuations.json']/1e6:.2f} MB)", flush=True)
    except ValueError as e:
        print(f"[build]   valuations.json SKIPPED — non-finite value leaked: {e}", flush=True)

    # models.json — evergreen per-model SEO pages (Tier-3): /preco/{slug}, /precos,
    # /sitemap.xml. Asking-price quantiles per model + per year, PLUS the model's
    # fair-value band (gl/gm/gh) wherever it clears the cheap-tail/ceiling guards.
    # ~50 KB gzipped for ~264 models. Same Release glob upload + allow_nan guard.
    #
    # The GBM valuator runs host-side here (LightGBM can't run in the Worker /
    # Pyodide). One shared bundle load feeds one batched value_configs call over
    # ~2k synthetic model/year configs. If no fresh model, pages ship asking-only.
    _loaded = load_model(max_age_hours=14 * 24)
    _valuator = None
    if _loaded is not None:
        _m, _cm, _mt, _of, _cal, _unc = _loaded
        _bundle = {"models": _m, "cat_maps": _cm, "metrics": _mt,
                   "median_calibrator": _cal, "uncertainty_bundle": _unc}
        _valuator = lambda cfg: value_configs(cfg, bundle=_bundle)  # noqa: E731
    else:
        print("[build]   model pages: no fresh price model — shipping asking-only", flush=True)
    model_pages = build_model_pages(listings, sell_speed, valuator=_valuator)
    model_pages["built_at"] = built_at   # freshness signal the Worker renders ("atualizado em")
    _n_models = len(model_pages.get("models", {}))
    _n_gbm = sum(1 for r in model_pages.get("models", {}).values() if "gm" in r)
    models_path = out_dir / "models.json"
    # Collapse guard: refuse to overwrite a healthy blob with a gutted one (a
    # data/query regression that halves the corpus would silently 404 hundreds of
    # SEO pages). <50 = catastrophic → skip the write, keep the live Release asset.
    if _n_models < 50:
        print(f"[build]   models.json SKIPPED — collapsed to {_n_models} models (<50); "
              f"keeping the previously published blob", flush=True)
    else:
        if _n_models < 200:
            print(f"[build]   ⚠ models.json has {_n_models} models (<200, usual ~264) — "
                  f"check the corpus/query", flush=True)
        try:
            mblob = json.dumps(model_pages, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            models_path.write_text(mblob)
            sizes["models.json"] = models_path.stat().st_size
            print(f"[build]   model pages: {_n_models:>6} models  ({_n_gbm} with GBM band)  "
                  f"({sizes['models.json']/1e3:.0f} KB)", flush=True)
        except ValueError as e:
            print(f"[build]   models.json SKIPPED — non-finite value leaked: {e}", flush=True)

    manifest = {
        "built_at": built_at,
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
            "valuations": len(valuations.get("cars", {})),
            "model_pages": len(model_pages.get("models", {})),
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
