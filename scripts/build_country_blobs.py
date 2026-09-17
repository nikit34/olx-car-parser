#!/usr/bin/env python3
"""Build the per-country Worker blobs for the AutoScout24 markets (DE/FR/IT).

The Portuguese site is rendered from ``models.json`` (per-model price pages),
``valuations.json`` (paste-a-link valuation) and ``hot_deals_{zone}.json``
(deal feed), all produced from the OLX listings table by
``scripts/build_dashboard_data.py`` and ``scripts/build_hot_deals.py``. The
foreign markets live in ``import_listings`` and reach this script through
``repository.get_country_listings_df``, which shapes them exactly like the
Portuguese frame, so the same analytics stack runs unchanged per country.

Outputs, in ``--out`` (default ``data/intl``), ``cc`` lower-case:

  models_{cc}.json          per-model asking quantiles + GBM band + regions
  valuations_{cc}.json      per-listing fair value (only with a fresh model)
  hot_deals_{cc}_all.json   BUY/WATCH deals, zone "all" (only with a model)
  brands_models_{cc}.json   {brand: [model, ...]} for the filter dropdowns
  manifest_{cc}.json        built_at + row counts + file sizes

Every file has the shape of its Portuguese counterpart. A country whose page
set collapses under ``--min-models`` keeps the previously published blob:
the write is skipped, never replaced by a gutted one.

Use:
    python scripts/build_country_blobs.py
    python scripts/build_country_blobs.py --country DE --no-model
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import contextmanager
from functools import partial
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

RELEASE_BASE_URL = ("https://github.com/nikit34/olx-car-parser/releases/"
                    "download/latest-data")
MODEL_MAX_AGE_HOURS = 14 * 24
DEFAULT_MIN_MODELS = 30
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "intl"

_DEAL_EXTRA_COLS = (
    "olx_id", "title", "description", "llm_extras", "first_seen_at",
    "seller_type", "transmission", "is_active", "engine_cc", "origin",
    "co2_g_km", "image_url", "last_scraped_at", "extras",
)

MAX_UNSEEN_DAYS = 14
MIN_YEAR_CELL_FOR_DEAL = 10


def _log(cc: str, msg: str) -> None:
    print(f"[intl:{cc.lower()}] {msg}", flush=True)


def _utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _dump_json(path: Path, doc: dict, cc: str) -> int | None:
    """Write *doc* as compact JSON; an unshippable value skips the file loudly.

    ``allow_nan=False`` turns a leaked pandas NaN into a ``ValueError`` instead
    of the literal ``NaN`` the Worker's ``JSON.parse`` cannot read, and a leaked
    Timestamp or numpy scalar raises ``TypeError``. Either way the run loses one
    blob and says which, rather than the country's whole build — the manifest
    and the other files still land.
    """
    try:
        blob = json.dumps(doc, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    except (ValueError, TypeError) as e:
        _log(cc, f"{path.name} SKIPPED — value the Worker cannot read: {e}")
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(blob)
    return path.stat().st_size


def _published_models(out_dir: Path, cc: str) -> dict | None:
    """The country's models.json currently live, for page-set hysteresis.

    Same role as ``build_dashboard_data._published_models``: prefers the copy
    this runner wrote last time, falls back to the Release asset the Worker
    serves, and returns None (no hysteresis) when neither exists.
    """
    name = f"models_{cc.lower()}.json"
    local = out_dir / name
    if local.exists():
        try:
            return json.loads(local.read_text())
        except (OSError, ValueError) as e:
            _log(cc, f"previous {name} unreadable ({e}); trying the Release")
    try:
        import urllib.request
        with urllib.request.urlopen(f"{RELEASE_BASE_URL}/{name}", timeout=30) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception as e:
        _log(cc, f"no previous {name} ({e}); publishing without hysteresis")
        return None


def _brands_models(listings: pd.DataFrame) -> dict[str, list[str]]:
    """``{brand: [model, ...]}`` with brand spellings canonicalised and model
    spellings that differ only by case or accent folded onto the most common
    one, exactly as the Portuguese build does."""
    from src.analytics.model_pages import slugify
    from src.parser.brand_normalize import normalize_brand

    out: dict[str, list[str]] = {}
    if not {"brand", "model"}.issubset(listings.columns):
        return out
    pairs = listings[["brand", "model"]].dropna()
    if pairs.empty:
        return out
    pairs = pairs.assign(
        _b=pairs["brand"].astype(str).map(normalize_brand),
        _m=pairs["model"].astype(str),
    )
    counts = pairs.groupby(["_b", "_m"]).size()
    best: dict[tuple[str, str], tuple[int, str]] = {}
    for (b, m), n in counts.items():
        key = (b, slugify(m))
        if key[1] and (key not in best or n > best[key][0]):
            best[key] = (int(n), m)
    for (b, _), (_, m) in sorted(best.items()):
        out.setdefault(b, []).append(m)
    return {b: sorted(set(ms)) for b, ms in out.items()}


@contextmanager
def _country_model_scope(cc: str):
    """Point the loaders ``compute_signals`` reaches for at this country.

    ``compute_signals`` calls ``price_model.load_model()`` and the importance
    loaders without a country argument, so for the duration of the block they
    are bound to ``country=cc``. The anomaly and hazard bundles are Portuguese
    fits over Portuguese feature distributions; scoring a German corpus with
    them would veto and rank on noise, so both read as "no bundle" here.
    """
    from src.analytics import anomaly, hazard, price_model

    saved = {
        (price_model, "load_model"): price_model.load_model,
        (price_model, "load_importance"): price_model.load_importance,
        (price_model, "load_grouped_importance"): price_model.load_grouped_importance,
        (price_model, "load_shap_importance"): price_model.load_shap_importance,
        (anomaly, "load_model"): anomaly.load_model,
        (hazard, "load_model"): hazard.load_model,
    }

    def _none(*_a, **_k):
        return None

    try:
        for (mod, name), fn in saved.items():
            if mod is price_model:
                setattr(mod, name, partial(fn, country=cc))
            else:
                setattr(mod, name, _none)
        yield
    finally:
        for (mod, name), fn in saved.items():
            setattr(mod, name, fn)


def _load_bundle(cc: str) -> dict | None:
    """The country's fresh price-model bundle as ``value_configs`` wants it."""
    from src.analytics import price_model

    loaded = price_model.load_model(max_age_hours=MODEL_MAX_AGE_HOURS, country=cc)
    if loaded is None:
        return None
    models, cat_maps, metrics, _oof, calibrator, uncertainty = loaded
    return {"models": models, "cat_maps": cat_maps, "metrics": metrics,
            "median_calibrator": calibrator, "uncertainty_bundle": uncertainty}


def _model_pages(listings: pd.DataFrame, sell_speed: pd.DataFrame, liq_pages: dict,
                 bundle: dict | None, published: dict | None, cc: str) -> dict:
    """``build_model_pages`` with the GBM band when a bundle is available.

    A valuator that fails leaves the pages asking-only rather than losing
    the whole blob: the asking quantiles are the page, the band is a bonus.
    """
    from src.analytics.model_pages import build_model_pages
    from src.analytics.price_model import value_configs

    if bundle is not None:
        def _valuator(cfg: pd.DataFrame) -> pd.DataFrame:
            return value_configs(cfg, bundle=bundle)
        try:
            return build_model_pages(listings, sell_speed, valuator=_valuator,
                                     liquidity=liq_pages, published=published)
        except Exception as e:
            _log(cc, f"GBM band failed ({type(e).__name__}: {e}) — shipping asking-only")
    else:
        _log(cc, "no fresh price model — shipping asking-only")
    return build_model_pages(listings, sell_speed, valuator=None,
                             liquidity=liq_pages, published=published)


def _signals(listings: pd.DataFrame, turnover: pd.DataFrame, cc: str) -> tuple | None:
    """``compute_signals`` bound to the country's model; None when it fails.

    The deal artefacts depend on it, the model pages do not, so a failure
    here is printed loudly and the build carries on without deals.
    """
    from src.dashboard.data_loader import compute_signals

    try:
        with _country_model_scope(cc):
            return compute_signals(listings, pd.DataFrame(), turnover=turnover)
    except Exception as e:
        _log(cc, f"compute_signals FAILED ({type(e).__name__}: {e}) — "
                 f"no valuations/hot_deals this run; model pages still build")
        return None


def _coverage_80(bundle: dict | None) -> float | None:
    """The band-coverage the decision engine's confidence step wants.

    The Portuguese feed reads it from the metrics history; here it comes off
    the bundle that actually produced these predictions, which is the same
    number. None leaves the step neutral, as it does there.
    """
    metrics = (bundle or {}).get("metrics") or {}
    return metrics.get("coverage_80_calibrated") or metrics.get("coverage_80")


def _cell_deep_enough(merged: pd.DataFrame, listings: pd.DataFrame, cc: str) -> pd.DataFrame:
    """Deals whose own model year the corpus has actually looked at.

    ``sample_size`` cannot catch this on a young market. The crawl reads twenty
    listings per model before it ever walks that model year by year, so every
    car of a covered model reports twenty comparables while its own year may
    hold exactly one. The band comes out tight and wrong, and the card then
    offers a saving against a value nobody measured: the first Italian run put
    a 2007 van with 420 000 km at 13 321 EUR against an asking price of 5 500.

    So the gate is the depth of the car's own (brand, model, year) cell, which
    is what the harvest pass fills. The floor is the one the project already
    defends for a model year to earn a page of its own
    (``model_pages.MIN_YEAR_PAGE_N``): a year too thin to carry a published
    median is too thin to price one car against. Until that pass has been round, a market
    publishes its medians and no per-car claim, and the feed opens by itself as
    the cells fill.
    """
    need = {"brand", "model", "year"}
    if merged.empty or not need <= set(merged.columns) or not need <= set(listings.columns):
        return merged
    active = listings
    if "is_active" in listings.columns:
        active = listings[listings["is_active"].fillna(False).astype(bool)]
    counts = active.groupby(["brand", "model", "year"]).size()
    keys = list(zip(merged["brand"], merged["model"], merged["year"]))
    deep = pd.Series([int(counts.get(k, 0)) >= MIN_YEAR_CELL_FOR_DEAL for k in keys],
                     index=merged.index)
    kept = merged[deep]
    if len(kept) != len(merged):
        _log(cc, f"deal feed: {len(merged) - len(kept)} of {len(merged)} signals sit in a model "
                 f"year holding under {MIN_YEAR_CELL_FOR_DEAL} listings — no per-car claim yet")
    return kept


def _seen_recently(merged: pd.DataFrame, cc: str) -> pd.DataFrame:
    """Deals whose listing we have actually confirmed is still up.

    The Portuguese feed re-reads every listing several times a day, so a card on
    it is current. A country corpus is walked model-year by model-year inside a
    politeness budget, and a cell waits weeks for its turn, so ``is_active``
    there means "still up when we last looked" — which can be a fortnight ago.
    Offering a fortnight-old price as a live deal is a promise the data cannot
    keep, so a row nobody has seen inside ``MAX_UNSEEN_DAYS`` does not reach the
    feed. It stays in the corpus: the medians it feeds are a statement about a
    period, not about this minute.
    """
    if merged.empty or "last_scraped_at" not in merged.columns:
        return merged
    seen = pd.to_datetime(merged["last_scraped_at"], utc=True, errors="coerce")
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=MAX_UNSEEN_DAYS)
    fresh = merged[seen >= cutoff]
    if len(fresh) != len(merged):
        _log(cc, f"deal feed: {len(merged) - len(fresh)} of {len(merged)} signals "
                 f"last confirmed over {MAX_UNSEEN_DAYS}d ago — held back")
    return fresh


def _row_photos(row) -> list[str]:
    """The gallery a stored row carries, newest reader first.

    ``extras`` is where a foreign row keeps what has no column of its own, and
    the crawl puts the whole gallery there. ``image_url`` stays the fallback:
    rows written before the crawl started keeping the list have only that one.
    """
    extras = row.get("extras")
    if isinstance(extras, str):
        try:
            extras = json.loads(extras)
        except ValueError:
            extras = None
    photos = (extras or {}).get("photo_urls") if isinstance(extras, dict) else None
    out = [str(u) for u in photos if str(u or "").strip()] if isinstance(photos, list) else []
    if out:
        return out
    image = row.get("image_url")
    return [str(image)] if isinstance(image, str) and image else []


def _hot_deals(signals: pd.DataFrame, listings: pd.DataFrame, predictions: pd.DataFrame,
               sell_speed: pd.DataFrame, cc: str, bundle: dict | None = None) -> list[dict]:
    """BUY/WATCH deals in the Worker card shape, photo from the listing card.

    Same field list and gates as the Portuguese feed (``_format_deal``,
    ``_pick_zone_deals``), with the decision engine's context built from this
    country's frame alone and no HTTP fetch. The Portuguese feed opens each
    listing to collect its photos; here the crawl already stored them, three
    off the search card and the whole set off the advert where robots leaves it
    open, so the gallery costs this build nothing.
    """
    from scripts.build_hot_deals import (
        MAX_LISTING_AGE_DAYS, _format_deal, _pick_zone_deals,
    )
    from src.analytics.decision import build_context, decide

    if signals is None or signals.empty:
        return []
    extra = listings[[c for c in _DEAL_EXTRA_COLS if c in listings.columns]]
    extra = extra.drop_duplicates("olx_id")
    merged = signals.merge(extra, on="olx_id", how="left", suffixes=("", "_l"))
    if sell_speed is not None and not sell_speed.empty:
        merged = merged.merge(sell_speed, on=["brand", "model"], how="left")
    if "first_seen_at" in merged.columns:
        merged["first_seen_at"] = pd.to_datetime(
            merged["first_seen_at"], utc=True, errors="coerce").dt.tz_localize(None)

    predicted_lookup: dict = {}
    if predictions is not None and {"olx_id", "predicted_price"}.issubset(predictions.columns):
        predicted_lookup = dict(zip(predictions["olx_id"], predictions["predicted_price"]))
    ctx = build_context(listings, pd.DataFrame(), coverage_80=_coverage_80(bundle),
                        predicted_lookup=predicted_lookup)
    decisions = [decide(row, ctx) for _, row in merged.iterrows()]
    merged["verdict"] = [d.verdict for d in decisions]
    merged["decision_score"] = [d.score for d in decisions]
    merged["isv_eur"] = [(d.components or {}).get("isv_eur") or None for d in decisions]
    _log(cc, f"decisions over {len(merged)} signals: "
             f"{merged['verdict'].value_counts().to_dict()}")

    merged = _seen_recently(merged, cc)
    merged = _cell_deep_enough(merged, listings, cc)
    stages: dict[str, int] = {}
    picked = _pick_zone_deals(merged, "all", None, 0, MAX_LISTING_AGE_DAYS, stages)
    _log(cc, f"funnel: {stages.get('signals', 0)} signals → {stages.get('active', 0)} active"
             f" → {stages.get('fresh', 0)} posted <={MAX_LISTING_AGE_DAYS}d ago"
             f" → {stages.get('vetted', 0)} BUY/WATCH")
    deals: list[dict] = []
    for _, row in picked.iterrows():
        deals.append(_format_deal(row.to_dict(), _row_photos(row)))
    return deals


def build_country(cc: str, session, out_dir: Path, *, min_models: int = DEFAULT_MIN_MODELS,
                  with_model: bool = True, built_at: str | None = None,
                  with_liquidity: bool = False) -> dict:
    """Build every blob for one country into *out_dir*; returns its manifest.

    ``with_liquidity`` is off by design rather than by caution. Days-to-sell is
    measured from when a listing stops being seen, so it is only a fact about
    the market when the crawl revisits every listing far more often than the
    thing it claims to measure. The Portuguese corpus is re-read several times a
    day; a country corpus is walked cell by cell inside a politeness budget and
    a given car waits weeks for its turn, so the curve would be a picture of our
    request budget wearing the market's name. Turn it on for a market once that
    market's crawl demonstrably comes round faster than the horizon the number
    claims — until then the pages simply carry no such number.
    """
    from src.storage import repository
    from src.analytics.computed_columns import enrich_listings
    from src.analytics.liquidity import build_liquidity, page_records, sell_speed_frame
    from src.analytics.turnover import compute_turnover_stats
    from src.analytics.valuations import build_valuations
    from scripts.build_dashboard_data import _model_quality

    cc = cc.upper()
    lc = cc.lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    built_at = built_at or _utc_now_iso()
    sizes: dict[str, int] = {}
    rows: dict[str, int] = {}

    t0 = time.perf_counter()
    listings = repository.get_country_listings_df(session, cc)
    rows["listings"] = int(len(listings))
    _log(cc, f"listings: {len(listings)}  ({time.perf_counter() - t0:.1f}s)")
    if listings.empty:
        _log(cc, "no listings — nothing to build")
        manifest = {"built_at": built_at, "country": cc, "rows": rows,
                    "files_bytes": sizes, "total_bytes": 0}
        (out_dir / f"manifest_{lc}.json").write_text(json.dumps(manifest, indent=2))
        return manifest

    listings = enrich_listings(listings)
    active_mask = (listings["is_active"].fillna(False).astype(bool)
                   if "is_active" in listings.columns
                   else pd.Series(True, index=listings.index))
    rows["active"] = int(active_mask.sum())
    turnover = compute_turnover_stats(listings)
    rows["turnover"] = int(len(turnover))

    bundle = _load_bundle(cc) if with_model else None
    signals = pd.DataFrame()
    predictions = pd.DataFrame()
    signals_ok = False
    if with_model and bundle is not None:
        result = _signals(listings, turnover, cc)
        if result is not None:
            signals, _imp, _gimp, predictions, _contrib, _simp = result
            signals_ok = True
            _log(cc, f"compute_signals: signals={len(signals)}  predictions={len(predictions)}")
    rows["signals"] = int(len(signals))
    rows["predictions"] = int(len(predictions))

    if with_liquidity:
        liquidity = build_liquidity(listings)
        liq_pages = page_records(liquidity)
        sell_speed = sell_speed_frame(liquidity)
        _log(cc, f"liquidity: {len(liquidity.get('models', {}))} models "
                 f"({len(liq_pages)} deep enough for a page)")
    else:
        liquidity, liq_pages = {}, {}
        sell_speed = pd.DataFrame(columns=["brand", "model", "sell_days", "sell_n"])
        _log(cc, "liquidity withheld — the crawl revisits a listing far less often "
                 "than the horizon a days-to-sell figure would claim to measure")

    doc = _model_pages(listings, sell_speed, liq_pages, bundle,
                       _published_models(out_dir, cc), cc)
    if liquidity.get("market"):
        doc["lqm"] = liquidity["market"]
    doc["built_at"] = built_at
    mq = _model_quality((bundle or {}).get("metrics")) if bundle else None
    if mq:
        doc["mq"] = mq
    n_models = len(doc.get("models", {}))
    rows["model_pages"] = int(n_models)
    n_gbm = sum(1 for r in doc.get("models", {}).values() if "gm" in r)
    if n_models < min_models:
        _log(cc, f"models_{lc}.json SKIPPED — collapsed to {n_models} models "
                 f"(<{min_models}); keeping the previously published blob")
    else:
        size = _dump_json(out_dir / f"models_{lc}.json", doc, cc)
        if size is not None:
            sizes[f"models_{lc}.json"] = size
            _log(cc, f"model pages: {n_models} models ({n_gbm} with GBM band, "
                     f"{len(doc.get('districts', {}))} regions)  ({size / 1e3:.0f} KB)")

    brands = _brands_models(listings[active_mask])
    size = _dump_json(out_dir / f"brands_models_{lc}.json", brands, cc)
    if size is not None:
        sizes[f"brands_models_{lc}.json"] = size

    rows["valuations"] = 0
    if signals_ok and not predictions.empty:
        valuations = build_valuations(listings, predictions, sell_speed)
        rows["valuations"] = len(valuations.get("cars", {}))
        size = _dump_json(out_dir / f"valuations_{lc}.json", valuations, cc)
        if size is not None:
            sizes[f"valuations_{lc}.json"] = size
            _log(cc, f"valuations: {rows['valuations']} cars ({size / 1e6:.2f} MB)")

    rows["hot_deals"] = 0
    if signals_ok:
        deals = _hot_deals(signals, listings, predictions, sell_speed, cc, bundle)
        rows["hot_deals"] = len(deals)
        payload = {"zone": "all", "built_at": built_at, "deals": deals}
        size = _dump_json(out_dir / f"hot_deals_{lc}_all.json", payload, cc)
        if size is not None:
            sizes[f"hot_deals_{lc}_all.json"] = size
            _log(cc, f"hot deals: {len(deals)} → hot_deals_{lc}_all.json")

    manifest = {
        "built_at": built_at,
        "country": cc,
        "rows": rows,
        "files_bytes": sizes,
        "total_bytes": sum(sizes.values()),
    }
    (out_dir / f"manifest_{lc}.json").write_text(json.dumps(manifest, indent=2))
    _log(cc, f"DONE — {manifest['total_bytes'] / 1e6:.2f} MB across {len(sizes)} files")
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--country", action="append", default=None,
                    help="ISO code, repeatable (default: every AutoScout24 market)")
    ap.add_argument("--db", default=None, help="Engine URL (default: OLX_DB_URL)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR,
                    help="Output directory (default: data/intl)")
    ap.add_argument("--min-models", type=int, default=DEFAULT_MIN_MODELS,
                    help="Skip writing models_{cc}.json under this many models")
    ap.add_argument("--no-model", dest="with_model", action="store_false", default=True,
                    help="Skip the GBM: asking-only pages, no valuations or deals")
    ap.add_argument("--with-liquidity", action="store_true", default=False,
                    help="Publish days-to-sell for this market — only once its crawl "
                         "revisits listings faster than the horizon that number claims")
    args = ap.parse_args(argv)

    countries = args.country
    if not countries:
        from src.countries import EU_COUNTRIES
        countries = list(EU_COUNTRIES)

    from src.storage.database import init_db, get_session

    init_db(args.db)
    session = get_session()
    built_at = _utc_now_iso()
    failures = 0
    try:
        for cc in countries:
            try:
                build_country(cc, session, args.out, min_models=args.min_models,
                              with_model=args.with_model, built_at=built_at,
                              with_liquidity=args.with_liquidity)
            except Exception as e:
                failures += 1
                _log(cc, f"BUILD FAILED ({type(e).__name__}: {e})")
    finally:
        session.close()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
