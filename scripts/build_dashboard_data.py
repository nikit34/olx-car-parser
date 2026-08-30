#!/usr/bin/env python3
"""Precompute dashboard witnesses for the stlite static dashboard.

The browser-side dashboard (stlite on Cloudflare Pages) can't run the
LightGBM / sklearn inference pipeline that ``src.dashboard.data_loader.load_all``
fires on every cold start. This script runs that pipeline ONCE in CI
against the database, then serialises every artifact the dashboard
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
    python scripts/build_dashboard_data.py --out data/dashboard
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _model_quality(metrics: dict | None) -> dict | None:
    if not metrics:
        return None
    out: dict = {}
    for src, dst, nd in (("mae", "mae", 0), ("mape", "mape", 1), ("r2", "r2", 3),
                         ("coverage_80_calibrated", "cov", 3)):
        v = metrics.get(src)
        if v is None or not math.isfinite(float(v)):
            continue
        out[dst] = round(float(v), nd) if nd else int(round(float(v)))
    n = metrics.get("n_samples")
    if n:
        out["n"] = int(n)
    folds = metrics.get("cv_folds")
    if folds:
        out["folds"] = int(folds)
    ts = metrics.get("timestamp")
    if ts:
        out["ts"] = str(ts)[:10]
    return out if {"mae", "mape", "cov"} <= set(out) else None


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


def _build(db_url: str | None, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    # One build timestamp reused by the manifest AND models.json (the public
    # "preços atualizados em …" freshness signal the Worker renders).
    built_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # Import here so the ``--help`` path doesn't pay the cost of loading
    # the whole analytics stack (LightGBM, sklearn, etc.).
    from src.storage.database import init_db, get_session
    from src.storage.repository import (
        get_listings_df, get_price_history_df,
        get_price_snapshots_df, get_relist_events_df,
        get_unmatched_df, get_portfolio_df,
    )
    from src.analytics.computed_columns import enrich_listings
    from src.analytics.text_signals import TEXT_SIGNAL_COLUMNS, add_text_signals
    from src.analytics.turnover import compute_turnover_stats
    from src.analytics.liquidity import build_liquidity, page_records, sell_speed_frame
    from src.analytics.valuations import build_valuations
    from src.analytics.model_pages import build_model_pages
    from src.analytics.price_model import load_model, value_configs
    from src.dashboard.data_loader import compute_signals

    print("[build] loading DB", flush=True)
    init_db(db_url)
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
    # Two lossless passes that together cut this witness from 22.96 MiB to
    # 8.24 MiB on the 2.09M-row year window — it was the next file due to hit
    # Cloudflare's 25 MiB asset cap, which is what broke deploys on 08-24.
    #
    # Sorting groups each listing's history together, so the repeated brand /
    # model / generation values and the near-adjacent timestamps compress as
    # runs instead of as noise (-37% on its own). Flooring to the second drops
    # microseconds that only record when the scraper happened to write the row
    # (another -27pp). No consumer depends on row order or on sub-second
    # precision: they all group by week or by segment.
    if not snapshots.empty:
        for _col in ("scraped_at", "deactivated_at"):
            if _col in snapshots.columns:
                snapshots[_col] = pd.to_datetime(
                    snapshots[_col], errors="coerce").dt.floor("s")
        _sort_by = [c for c in ("olx_id", "scraped_at") if c in snapshots.columns]
        if _sort_by:
            snapshots = snapshots.sort_values(_sort_by).reset_index(drop=True)
    portfolio = get_portfolio_df(session)
    unmatched = get_unmatched_df(session)

    # The filter dropdowns are built from whatever spellings the sellers used, so
    # "SEAT" and "Seat" arrived as two separate brands with two partial model
    # lists — picking either one analysed half the sample. Canonicalise the brand
    # and fold model spellings that differ only by case or accent ("MiTo"/"Mito",
    # "C-MAX"/"C-Max"), keeping the spelling that appears most often.
    from src.analytics.model_pages import slugify as _slugify
    from src.parser.brand_normalize import normalize_brand as _norm_brand

    brands_models: dict[str, list[str]] = {}
    _pairs = listings[["brand", "model"]].dropna()
    if not _pairs.empty:
        _pairs = _pairs.assign(
            _b=_pairs["brand"].astype(str).map(_norm_brand),
            _m=_pairs["model"].astype(str),
        )
        _counts = _pairs.groupby(["_b", "_m"]).size()
        _best: dict[tuple[str, str], tuple[int, str]] = {}
        for (_b, _m), _n in _counts.items():
            _key = (_b, _slugify(_m))
            if _key[1] and (_key not in _best or _n > _best[_key][0]):
                _best[_key] = (int(_n), _m)
        for (_b, _), (_, _m) in sorted(_best.items()):
            brands_models.setdefault(_b, []).append(_m)
        for _b in brands_models:
            brands_models[_b] = sorted(set(brands_models[_b]))

    contributions_df = _contributions_to_long(contributions)

    sizes: dict[str, int] = {}
    # The witness ships without the raw prose. ``description`` was 57% of the
    # file (15.75 MB zstd of 27.6 MB) and pushed it past Cloudflare's 25 MiB
    # per-asset cap, breaking every Worker deploy from 2026-08-24. Nothing
    # renders it: the three consumers that scanned it (ISV import flags,
    # condition-NLP faults, salvage hard block) read the precomputed columns
    # instead. Dropped only on the way to parquet — build_valuations and
    # build_model_pages below still get the full frame.
    #
    # The scans cost ~26 s over the full corpus (47 MB of text, four regexes),
    # which is why they run here and not in enrich_listings: every CLI command
    # calls that, and none of them need the columns — they still hold the prose
    # and scan it inline.
    # Scans mutate ``listings`` in place (four narrow columns, harmless to the
    # valuations/model-pages consumers below) rather than copying a 100k×90
    # frame — the scrape host has 8 GB. The drop is copy-on-write cheap.
    listings = add_text_signals(listings)
    listings_witness = listings.drop(columns=["description"], errors="ignore")
    missing_signals = [c for c in TEXT_SIGNAL_COLUMNS if c not in listings_witness.columns]
    if missing_signals:
        raise SystemExit(
            "text-signal columns missing from the witness — the browser would "
            f"silently lose these scans: {', '.join(missing_signals)}"
        )
    sizes["listings.parquet"] = _to_parquet(listings_witness, out_dir / "listings.parquet")
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

    # Asset-size guard. Cloudflare refuses any single static asset over 25 MiB,
    # and the failure surfaces as a red Workers build with no hint that data
    # size is the cause — production sat on stale code for a day and a half in
    # August before anyone connected the two. Warn with room to act, fail
    # before the deploy does.
    _CF_ASSET_LIMIT = 25 * 1024 * 1024
    _WARN_AT = int(_CF_ASSET_LIMIT * 0.8)
    _oversized = []
    for _name, _bytes in sorted(sizes.items(), key=lambda kv: -kv[1]):
        if _bytes > _CF_ASSET_LIMIT:
            _oversized.append(f"{_name} ({_bytes / 1048576:.1f} MiB)")
        elif _bytes > _WARN_AT:
            print(f"::warning::{_name} is {_bytes / 1048576:.1f} MiB — approaching "
                  f"Cloudflare's 25 MiB per-asset limit; the Worker deploy fails "
                  f"outright once it crosses.", flush=True)
    if _oversized:
        raise SystemExit(
            "witness over Cloudflare's 25 MiB asset limit: " + ", ".join(_oversized)
            + " — the Worker deploy would fail and production would keep serving "
              "the previous build."
        )

    brands_path = out_dir / "brands_models.json"
    brands_path.write_text(json.dumps(brands_models, ensure_ascii=False))
    sizes["brands_models.json"] = brands_path.stat().st_size

    # valuations.json — the public "value any listing" lookup (Tier-2), fetched
    # by the Worker's /avaliar paste-a-link tool. Active+priced listings only;
    # ~0.9 MB gzipped for ~18k cars. Uploaded to the Release by the existing
    # ``data/dashboard/*.json`` glob in scrape-ci (no workflow change needed).
    _relisted = set()
    try:
        _rel = get_relist_events_df(session)
        if not _rel.empty:
            _relisted = set(_rel["original_olx_id"].astype(str))
    except Exception as e:
        print(f"[build]   relist events unavailable ({e}) — liquidity ships without them",
              flush=True)
    liquidity = build_liquidity(listings, relisted=_relisted)
    liq_pages = page_records(liquidity)
    sell_speed = sell_speed_frame(liquidity)
    print(f"[build]   liquidity: {len(liquidity.get('models', {})):>6} models  "
          f"({len(liq_pages)} deep enough for a page)", flush=True)
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
    model_pages = build_model_pages(listings, sell_speed, valuator=_valuator,
                                    liquidity=liq_pages)
    if liquidity.get("market"):
        model_pages["lqm"] = liquidity["market"]
    model_pages["built_at"] = built_at   # freshness signal the Worker renders ("atualizado em")
    _mq = _model_quality(_mt if _loaded is not None else None)
    if _mq:
        model_pages["mq"] = _mq
    _n_models = len(model_pages.get("models", {}))
    _n_gbm = sum(1 for r in model_pages.get("models", {}).values() if "gm" in r)
    _n_facets = sum(len(r.get(k, [])) for r in model_pages.get("models", {}).values()
                    for k in ("fx", "tx", "dt"))
    _n_duel = sum(1 for r in model_pages.get("models", {}).values() if "dg" in r)
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
                  f"{_n_facets} facet cells, {_n_duel} fuel-retention fits  "
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
        "--db", default=None,
        help="Engine URL (default: OLX_DB_URL)",
    )
    ap.add_argument(
        "--out", type=Path,
        default=REPO_ROOT / "data" / "dashboard",
        help="Output directory for parquet/json artifacts (default: data/dashboard)",
    )
    args = ap.parse_args()
    _build(args.db, args.out)


if __name__ == "__main__":
    main()
