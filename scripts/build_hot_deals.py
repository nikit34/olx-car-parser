#!/usr/bin/env python3
"""Build hot_deals_{zone}.json — flip candidates per geographic zone,
scored by the production price model and filtered through the production
veto rules (``_blocking_deal_reason`` is already applied inside
``compute_signals``).

Uploaded to the ``latest-data`` GitHub Release by ``scrape.yml``; the
flipper-club Cloudflare Worker fetches each zone's JSON at request time
and caches it in KV for 5 minutes.

Zones map districts → group:
  norte:  Porto, Braga, Aveiro, Viana do Castelo, Vila Real, Bragança
  centro: Coimbra, Leiria, Viseu, Guarda, Castelo Branco, Santarém
  sul:    Lisboa, Setúbal, Évora, Beja, Portalegre, Faro, Ilha da Madeira
  all:    union of the above (admin / observer PINs see this)

Use:
    python scripts/build_hot_deals.py \
        --db data/olx_cars.db --out-dir data/hot_deals
    # or in dev with --no-fetch-photos to skip the OLX og:image hits
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

ZONE_DISTRICTS = {
    "norte":  ["Porto", "Braga", "Aveiro", "Viana do Castelo", "Vila Real", "Bragança"],
    "centro": ["Coimbra", "Leiria", "Viseu", "Guarda", "Castelo Branco", "Santarém"],
    "sul":    ["Lisboa", "Setúbal", "Évora", "Beja", "Portalegre", "Faro",
               "Ilha da Madeira", "Ilha de São Miguel"],
}
# Per-zone cap on number of deals. 0 = no cap: ship every candidate that
# passes the active + freshness + zone + verdict filters. The feed is already
# bounded by MAX_LISTING_AGE_DAYS + the BUY/WATCH gate, so no runaway risk.
TOP_N_PER_ZONE = 0
MAX_LISTING_AGE_DAYS = 120
# Quality gate: only surface deals the production decision engine rates BUY or
# WATCH (same engine + thresholds the /analytics dashboard uses). SKIP/REJECT/
# NO_OPINION are dropped — we'd rather show fewer, vetted deals than pad the
# feed with marginal ones.
ALLOWED_VERDICTS = ("BUY", "WATCH")

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
      "(KHTML, like Gecko) Version/17.0 Safari/605.1.15")
# Same pattern src/parser/photo_fetch.py uses — apollo CDN URL + optional
# size variant. We filter out related-listing thumbnails by requiring at
# least one ≥1000-px variant per photo id, then emit the 1000x700 URL.
_OLX_PHOTO_RE = re.compile(
    r"apollo\.olxcdn\.com[:\d]*/v1/files/([\w-]+)-PT/image"
    r"(?:;s=(\d+)x(\d+))?"
)

_PHOTO_CACHE: dict[str, list[str] | None] = {}

# Relay egress, same Worker route the scraper uses. OLX's WAF blocks this
# machine, so a direct listing-page fetch 403s and every card ships without an
# image. Unset → direct fetch, exactly as before.
_RELAY_URL = (os.environ.get("OLX_RELAY_URL") or "").strip() or None
_RELAY_TOKEN = (os.environ.get("OLX_RELAY_TOKEN") or "").strip() or None
if _RELAY_URL and not _RELAY_TOKEN:
    print("[hot_deals]   OLX_RELAY_URL set without OLX_RELAY_TOKEN — relay disabled",
          file=sys.stderr, flush=True)
    _RELAY_URL = None


def _relay(url: str) -> tuple[str, dict]:
    """Rewrite an OLX listing URL to go through the Worker relay.

    Only the path is forwarded; the Worker pins the origin itself and refuses
    anything outside its allowed prefixes, so a URL that is not an OLX listing
    page simply goes out direct and fails the same way it would have.
    """
    if not _RELAY_URL:
        return url, {}
    parts = urlsplit(url)
    if parts.netloc not in ("www.olx.pt", "olx.pt"):
        return url, {}
    path_q = parts.path + (("?" + parts.query) if parts.query else "")
    return (f"{_RELAY_URL}?path={quote(path_q, safe='')}",
            {"X-Relay-Token": _RELAY_TOKEN})


def fetch_photo_urls(url: str, timeout: int = 10) -> list[str] | None:
    """Fetch the OLX listing page and return all gallery photo URLs in page
    order.

    Three outcomes, and the caller must tell them apart:

    * ``[...]`` — page fetched, photos found.
    * ``[]``    — page fetched and it has no usable photos, i.e. the listing is
      dead (410/redirect) and its link would be broken anyway. Drop the deal.
    * ``None``  — we could not reach OLX at all (403 block, timeout, DNS).
      That says nothing about the listing, so the deal must survive without an
      image. Returning ``[]`` here is what emptied the whole feed on
      2026-08-25: OLX started 403ing our address, every fetch raised, and all
      21 BUY/WATCH deals were discarded as "dead" while the site showed
      "Sem negócios".

    Cached in-memory per process so the same listing-URL across zones triggers
    one HTTP request.
    """
    if url in _PHOTO_CACHE:
        return _PHOTO_CACHE[url]
    try:
        target, relay_headers = _relay(url)
        req = Request(target, headers={
            "User-Agent": UA,
            "Accept-Language": "pt-PT,pt;q=0.9,en;q=0.7",
            **relay_headers,
        })
        with urlopen(req, timeout=timeout) as r:
            html = r.read().decode("utf-8", errors="ignore")
        sizes_by_id: dict[str, set[int]] = {}
        order: list[str] = []
        for m in _OLX_PHOTO_RE.finditer(html):
            pid = m.group(1)
            if pid not in sizes_by_id:
                sizes_by_id[pid] = set()
                order.append(pid)
            if m.group(2):
                sizes_by_id[pid].add(int(m.group(2)))
        photos = [
            f"https://ireland.apollo.olxcdn.com:443/v1/files/{pid}-PT/image;s=1000x700"
            for pid in order
            if any(w >= 1000 for w in sizes_by_id[pid])
        ]
    except HTTPError as e:
        # 404/410 — the listing really is gone. Anything else (403 block, 5xx)
        # is our problem, not the listing's.
        dead = e.code in (404, 410)
        print(f"[hot_deals]   photos {'gone' if dead else 'unreachable'} "
              f"(HTTP {e.code}): {url[:60]}…", file=sys.stderr, flush=True)
        photos = [] if dead else None
    except Exception as e:
        print(f"[hot_deals]   photos unreachable ({type(e).__name__}): {url[:60]}…",
              file=sys.stderr, flush=True)
        photos = None
    _PHOTO_CACHE[url] = photos
    return photos


def _format_deal(row: dict, photo_urls: list[str]) -> dict:
    """Shape one signals/listings-merged row into the worker's JSON contract."""
    extras: dict = {}
    raw_extras = row.get("llm_extras")
    if raw_extras and isinstance(raw_extras, str):
        try:
            extras = json.loads(raw_extras)
        except json.JSONDecodeError:
            pass

    # Full description — the card's expanded view shows it in full. Preserve
    # per-line breaks (OLX descriptions are commonly a "• ..." list) so the
    # card's <div class="desc"> (white-space: pre-wrap) renders each item on its
    # own line. But OLX's API HTML doubled every break into a blank line (the
    # <br>+literal-newline bug, fixed in scraper._clean_html_description); legacy
    # rows scraped before that fix still carry \n\n. Collapse ALL blank lines to
    # a single break so cards match the source's tight, single-spaced layout.
    # Legacy rows from the detail path also lead with OLX UI chrome
    # ("Anotações"/"Reportar"); strip it here so the ≈12k rows scraped before
    # the scraper fix render clean without a DB migration.
    from src.parser.scraper import _strip_desc_chrome
    desc = (row.get("description") or "").replace("\r\n", "\n").replace("\r", "\n")
    desc = _strip_desc_chrome(desc)
    desc = re.sub(r"[ \t]+\n", "\n", desc)
    desc = re.sub(r"\n{2,}", "\n", desc).strip()

    first_seen_raw = row.get("first_seen_at")
    first_seen_iso = None
    days_on_market = None
    if first_seen_raw:
        try:
            ts = pd.Timestamp(first_seen_raw)
            if ts.tz is not None:
                ts = ts.tz_convert("UTC").tz_localize(None)
            first_seen_iso = ts.isoformat(timespec="seconds") + "Z"
            # first_seen_at is stored naive-UTC (src/models/listing.py:_utcnow).
            # Diff against naive-UTC now, NOT pd.Timestamp.now() (naive LOCAL) —
            # the latter skews days_on_market by the runner's TZ offset and
            # flips the int-rounded day count near midnight.
            now_utc = pd.Timestamp.now("UTC").tz_localize(None)
            days_on_market = max(0, int((now_utc - ts).days))
        except Exception:
            pass

    def _i(v):
        try:
            return int(v) if v is not None and pd.notna(v) else None
        except (TypeError, ValueError):
            return None

    def _f(v):
        try:
            return float(v) if v is not None and pd.notna(v) else None
        except (TypeError, ValueError):
            return None

    def _s(v):
        # String fields straight off pandas rows can be float NaN (missing).
        # json.dumps emits those as the literal `NaN`, which is valid Python
        # but NOT valid JSON — the worker's JSON.parse then throws and we
        # fall back to mock for the whole zone. Coerce to None up front.
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        return v

    discount_pct_raw = _f(row.get("adjusted_undervaluation_pct"))
    discount_pct = round(discount_pct_raw / 100, 4) if discount_pct_raw is not None else None
    # Fallback: if est_profit_after_repair_eur is None (no repair cost), the
    # naive profit is fair_median - price. The dashboard uses the same logic.
    profit = _i(row.get("est_profit_after_repair_eur"))
    if profit is None:
        price = _f(row.get("price_eur"))
        median = _f(row.get("predicted_price"))
        if price is not None and median is not None:
            profit = int(round(median - price))

    return {
        "olx_id": _s(row.get("olx_id")),
        "url": _s(row.get("url")),
        "title": _s(row.get("title")) or f"{_s(row.get('brand')) or ''} {_s(row.get('model')) or ''}".strip(),
        "brand": _s(row.get("brand")),
        "model": _s(row.get("model")),
        "year": _i(row.get("year")),
        "mileage_km": _i(row.get("mileage_km")),
        "fuel_type": _s(row.get("fuel_type")),
        "transmission": _s(row.get("transmission")),
        "price_eur": _i(row.get("price_eur")),
        "fair_low": _i(row.get("fair_price_low")),
        "fair_median": _i(row.get("predicted_price")),
        "fair_high": _i(row.get("fair_price_high")),
        "discount_pct": discount_pct,
        "est_profit_eur": profit,
        "flip_score": _f(row.get("flip_score")),
        "verdict": _s(row.get("verdict")),
        "decision_score": _f(row.get("decision_score")),
        "sample_size": _i(row.get("sample_size")),   # comparable count behind the fair median (provenance)
        "first_seen_at": first_seen_iso,
        "days_on_market": days_on_market,
        "district": _s(row.get("district")),
        "city": _s(row.get("city")),
        "seller_type": _s(row.get("seller_type")),
        "origin": _s(row.get("origin")),   # structured import signal for importInfo()
        "co2_g_km": _i(row.get("co2_g_km")),
        "isv_eur": _i(row.get("isv_eur")),  # computed nationalisation tax (imports w/ CO2)
        "damage_severity": _i(row.get("damage_severity")) or 0,
        "photo_damage_p": float(extras.get("photo_damage_p") or 0),
        "photo_damage_flagged": bool(extras.get("photo_damage_flagged")),
        "photo_urls": photo_urls,
        "description": desc,
        # Sell-speed (median days-to-inactive for this brand+model; only present
        # when the segment cleared the >=8 sample gate). None ⇒ worker shows none.
        "sell_days": _i(row.get("sell_days")),
        "sell_n": _i(row.get("sell_n")),
    }


def _build_signals(db_url: str | None) -> pd.DataFrame:
    """Run the production pipeline end-to-end and return a (signals ⋈ listings)
    DataFrame with every column the worker cards need."""
    from src.storage.database import init_db, get_session
    from src.storage.repository import get_listings_df, get_price_history_df
    from src.analytics.computed_columns import enrich_listings
    from src.analytics.turnover import compute_turnover_stats, compute_sell_speed_by_model
    from src.dashboard.data_loader import compute_signals
    from src.parser.llm_enrichment import merge_real_mileage

    print("[hot_deals] init DB", flush=True)
    init_db(db_url)
    session = get_session()

    t0 = time.perf_counter()
    listings = get_listings_df(session)
    history = get_price_history_df(session)
    print(f"[hot_deals]   listings: {len(listings):,}  history: {len(history):,}  "
          f"({time.perf_counter()-t0:.1f}s)", flush=True)
    if listings.empty:
        raise SystemExit("DB has no listings")

    listings = enrich_listings(listings)
    listings = merge_real_mileage(listings)
    turnover = compute_turnover_stats(listings)

    t0 = time.perf_counter()
    signals_tuple = compute_signals(listings, history, turnover=turnover)
    signals = signals_tuple[0]
    print(f"[hot_deals]   compute_signals → {len(signals):,} rows (vetoes already applied) "
          f"({time.perf_counter()-t0:.1f}s)", flush=True)

    if signals.empty:
        return signals

    # Merge in the columns compute_signals doesn't carry over but the worker
    # cards need: title, description, llm_extras (photo_damage signals),
    # first_seen_at, seller_type, transmission, is_active.
    extra_cols = ["olx_id", "title", "description", "llm_extras",
                  "first_seen_at", "seller_type", "transmission", "is_active",
                  # ISV inputs for the decision engine (import nationalisation tax)
                  "engine_cc", "origin", "co2_g_km"]
    extra = listings[[c for c in extra_cols if c in listings.columns]].drop_duplicates("olx_id")
    merged = signals.merge(extra, on="olx_id", how="left", suffixes=("", "_l"))

    # Sell-speed (seller lens "vende em ~Nd" + liquidity context for all lenses).
    # Median days-to-inactive per (brand, model), gated on sample size — computed
    # from the FULL listings df (incl. inactive), joined onto the active deals.
    sell_speed = compute_sell_speed_by_model(listings)
    if not sell_speed.empty and {"brand", "model"}.issubset(merged.columns):
        merged = merged.merge(sell_speed, on=["brand", "model"], how="left")
        print(f"[hot_deals]   sell-speed: {len(sell_speed):,} model segments "
              f"(>=8 sold); matched {merged['sell_days'].notna().sum():,}/{len(merged):,} signals",
              flush=True)

    merged = _annotate_decisions(merged, listings)
    return merged


def _annotate_decisions(signals: pd.DataFrame, listings: pd.DataFrame) -> pd.DataFrame:
    """Attach ``verdict`` + ``decision_score`` columns using the same decision
    engine the /analytics dashboard runs (``src.analytics.decision``).

    The segment context (sold-side DoM, 90d trend, calibration residuals)
    needs snapshots + the full GB predictions; we reuse the dashboard witnesses
    ``data/dashboard/{snapshots,predictions}.parquet`` built by the preceding
    ``build_dashboard_data.py`` step instead of recomputing. If a witness is
    missing (e.g. this script run standalone), the corresponding step degrades
    to neutral (no trend / no calibration nudge) rather than failing."""
    if signals.empty:
        return signals
    from src.analytics.decision import build_context, decide

    dash_dir = REPO_ROOT / "data" / "dashboard"

    snapshots = pd.DataFrame()
    snap_path = dash_dir / "snapshots.parquet"
    if snap_path.exists():
        try:
            snapshots = pd.read_parquet(snap_path)
        except Exception as e:  # noqa: BLE001 — witness optional, degrade loudly
            print(f"[hot_deals]   snapshots witness unreadable ({e}) — trend step skipped", flush=True)

    predicted_lookup: dict = {}
    pred_path = dash_dir / "predictions.parquet"
    if pred_path.exists():
        try:
            pred = pd.read_parquet(pred_path)
            if {"olx_id", "predicted_price"}.issubset(pred.columns):
                predicted_lookup = dict(zip(pred["olx_id"], pred["predicted_price"]))
        except Exception as e:  # noqa: BLE001
            print(f"[hot_deals]   predictions witness unreadable ({e}) — calibration step skipped", flush=True)

    coverage_80 = None
    try:
        from src.analytics.price_model import load_metrics_history
        mh = load_metrics_history()
        if mh:
            last = mh[-1]
            coverage_80 = last.get("coverage_80_calibrated") or last.get("coverage_80")
    except Exception as e:  # noqa: BLE001
        print(f"[hot_deals]   coverage history unavailable ({e}) — band-confidence neutral", flush=True)

    ctx = build_context(listings, snapshots, coverage_80=coverage_80,
                        predicted_lookup=predicted_lookup)
    decisions = [decide(row, ctx) for _, row in signals.iterrows()]
    signals = signals.copy()
    signals["verdict"] = [d.verdict for d in decisions]
    signals["decision_score"] = [d.score for d in decisions]
    signals["isv_eur"] = [(d.components or {}).get("isv_eur") or None for d in decisions]
    counts = signals["verdict"].value_counts().to_dict()
    n_isv = int(signals["isv_eur"].notna().sum())
    if n_isv:
        print(f"[hot_deals]   ISV applied to {n_isv} imported deals", flush=True)
    print(f"[hot_deals]   decisions over {len(signals):,} signals: {counts}", flush=True)
    return signals


def _pick_zone_deals(signals: pd.DataFrame, zone: str, districts: list[str] | None,
                     top_n: int, max_age_days: int,
                     stages: dict[str, int] | None = None) -> pd.DataFrame:
    """Deals for one zone, recording the size after every gate into *stages*.

    The run log used to jump from "decisions over 190 signals: BUY 20, WATCH 8"
    straight to "all 14 deals", so a feed that halved between those two lines
    told nobody which gate ate it. That is the shape of the 2026-08-25 outage
    (see tests/test_hot_deals_photos.py): every deal was discarded downstream
    and the only visible symptom was a small number, which reads exactly like a
    quiet market. Counting each gate is what tells the two apart.
    """
    df = signals
    rec = stages if stages is not None else {}
    rec["signals"] = len(df)
    if "is_active" in df.columns:
        df = df[df["is_active"].fillna(True).astype(bool)]
    rec["active"] = len(df)
    if max_age_days and "first_seen_at" in df.columns:
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=max_age_days)
        ts = pd.to_datetime(df["first_seen_at"], errors="coerce")
        df = df[ts >= cutoff]
    rec["fresh"] = len(df)
    if districts is not None:
        df = df[df["district"].isin(districts)]
    rec["in_zone"] = len(df)
    # Quality gate — only BUY/WATCH verdicts from the decision engine.
    if "verdict" in df.columns:
        df = df[df["verdict"].isin(ALLOWED_VERDICTS)]
    rec["vetted"] = len(df)
    if df.empty:
        return df
    # Rank by the decision engine's risk-adjusted score (falls back to the
    # simpler flip_score / undervaluation if decisions weren't annotated).
    sort_col = next((c for c in ("decision_score", "flip_score",
                                 "adjusted_undervaluation_pct") if c in df.columns), None)
    if sort_col:
        df = df.sort_values(sort_col, ascending=False)
    if top_n and top_n > 0:
        df = df.head(top_n)
    return df.copy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--db", default=None,
                    help="Engine URL (default: OLX_DB_URL)")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "hot_deals")
    ap.add_argument("--top-n", type=int, default=TOP_N_PER_ZONE,
                    help="per-zone deal cap; 0 = no cap (default)")
    ap.add_argument("--max-age-days", type=int, default=MAX_LISTING_AGE_DAYS)
    ap.add_argument("--fetch-photos", dest="fetch_photos", action="store_true",
                    default=True, help="(default) fetch og:image from OLX per deal")
    ap.add_argument("--no-fetch-photos", dest="fetch_photos", action="store_false",
                    help="skip OLX HTTP calls — JSON ships without photos")
    ap.add_argument("--photo-sleep-sec", type=float, default=0.3,
                    help="polite sleep between OLX HTTP calls")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    signals = _build_signals(args.db)

    # All zones, plus the "all" aggregate (admin / "observer" PINs).
    zone_plan: list[tuple[str, list[str] | None]] = list(ZONE_DISTRICTS.items())
    zone_plan.append(("all", None))

    built_at = pd.Timestamp.now("UTC").tz_localize(None).isoformat(timespec="seconds") + "Z"
    overall_counts: dict[str, int] = {}

    for zone, districts in zone_plan:
        stages: dict[str, int] = {}
        picked = _pick_zone_deals(signals, zone, districts, args.top_n,
                                  args.max_age_days, stages)
        unreachable = 0
        dead = 0
        deals: list[dict] = []
        for _, row in picked.iterrows():
            photos: list[str] = []
            if args.fetch_photos and row.get("url"):
                fetched = fetch_photo_urls(row["url"])
                if fetched:
                    photos = fetched
                    time.sleep(args.photo_sleep_sec)
                elif fetched is None:
                    # Couldn't reach OLX. Ship the deal without an image
                    # rather than pretend it doesn't exist — a blocked
                    # scraper must not look like an empty market.
                    unreachable += 1
                else:
                    # Page fetched and carries no usable photo: the listing is
                    # 410/redirected and its link would be dead anyway.
                    dead += 1
                    continue
            deals.append(_format_deal(row.to_dict(), photos))

        out_path = args.out_dir / f"hot_deals_{zone}.json"
        payload = {
            "zone": zone,
            "built_at": built_at,
            "deals": deals,
        }
        # allow_nan=False makes json.dumps raise ValueError on NaN/Infinity
        # instead of emitting the literal `NaN` (invalid JSON, kills the
        # worker's JSON.parse). _format_deal already sanitises all fields,
        # so this is belt-and-braces — if it ever fires, fix _format_deal.
        out_path.write_text(json.dumps(payload, ensure_ascii=False, default=str,
                                       indent=2, allow_nan=False))
        overall_counts[zone] = len(deals)
        print(f"[hot_deals]   {zone:<6} funnel: {stages['signals']} signals"
              f" → {stages['active']} active"
              f" → {stages['fresh']} posted <={args.max_age_days}d ago"
              f" → {stages['in_zone']} in zone"
              f" → {stages['vetted']} BUY/WATCH"
              + (f" → -{dead} dead links" if dead else ""), flush=True)
        print(f"[hot_deals]   {zone:<6} {len(deals):>3} deals → {out_path.name}"
              + (f"  ({unreachable} shipped without a photo — OLX unreachable)"
                 if unreachable else ""), flush=True)

    print(f"[hot_deals] DONE  built_at={built_at}  zones={overall_counts}", flush=True)


if __name__ == "__main__":
    main()
