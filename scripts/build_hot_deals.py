#!/usr/bin/env python3
"""Build hot_deals_{zone}.json — top flip candidates per geographic zone,
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
import re
import sys
import time
from pathlib import Path
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
TOP_N_PER_ZONE = 30
MAX_LISTING_AGE_DAYS = 14

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
      "(KHTML, like Gecko) Version/17.0 Safari/605.1.15")
OG_IMAGE_RE = re.compile(r'<meta[^>]*property="og:image"[^>]*content="([^"]+)"')

_PHOTO_CACHE: dict[str, str | None] = {}


def fetch_og_image(url: str, timeout: int = 10) -> str | None:
    """Fetch the OLX listing page and extract its og:image. Cached in-memory
    per process so the same listing-URL appearing in multiple zones (e.g. "all"
    + "norte") only triggers one HTTP request."""
    if url in _PHOTO_CACHE:
        return _PHOTO_CACHE[url]
    try:
        req = Request(url, headers={
            "User-Agent": UA,
            "Accept-Language": "pt-PT,pt;q=0.9,en;q=0.7",
        })
        with urlopen(req, timeout=timeout) as r:
            html = r.read().decode("utf-8", errors="ignore")
        m = OG_IMAGE_RE.search(html)
        photo = m.group(1) if m else None
    except Exception as e:
        print(f"[hot_deals]   og:image fail ({type(e).__name__}): {url[:60]}…",
              file=sys.stderr, flush=True)
        photo = None
    _PHOTO_CACHE[url] = photo
    return photo


def _format_deal(row: dict, photo_url: str | None) -> dict:
    """Shape one signals/listings-merged row into the worker's JSON contract."""
    extras: dict = {}
    raw_extras = row.get("llm_extras")
    if raw_extras and isinstance(raw_extras, str):
        try:
            extras = json.loads(raw_extras)
        except json.JSONDecodeError:
            pass

    desc = (row.get("description") or "").strip().replace("\r", " ").replace("\n", " ")
    if len(desc) > 280:
        desc = desc[:280].rsplit(" ", 1)[0] + "..."

    first_seen_raw = row.get("first_seen_at")
    first_seen_iso = None
    days_on_market = None
    if first_seen_raw:
        try:
            ts = pd.Timestamp(first_seen_raw)
            if ts.tz is not None:
                ts = ts.tz_convert("UTC").tz_localize(None)
            first_seen_iso = ts.isoformat(timespec="seconds") + "Z"
            days_on_market = max(0, int((pd.Timestamp.now() - ts).days))
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
        "olx_id": row.get("olx_id"),
        "url": row.get("url"),
        "title": row.get("title") or f"{row.get('brand', '')} {row.get('model', '')}".strip(),
        "brand": row.get("brand"),
        "model": row.get("model"),
        "year": _i(row.get("year")),
        "mileage_km": _i(row.get("mileage_km")),
        "fuel_type": row.get("fuel_type"),
        "transmission": row.get("transmission"),
        "price_eur": _i(row.get("price_eur")),
        "fair_low": _i(row.get("fair_price_low")),
        "fair_median": _i(row.get("predicted_price")),
        "fair_high": _i(row.get("fair_price_high")),
        "discount_pct": discount_pct,
        "est_profit_eur": profit,
        "flip_score": _f(row.get("flip_score")),
        "first_seen_at": first_seen_iso,
        "days_on_market": days_on_market,
        "district": row.get("district"),
        "city": row.get("city"),
        "seller_type": row.get("seller_type"),
        "damage_severity": _i(row.get("damage_severity")) or 0,
        "photo_damage_p": float(extras.get("photo_damage_p") or 0),
        "photo_damage_flagged": bool(extras.get("photo_damage_flagged")),
        "photo_urls": [photo_url] if photo_url else [],
        "description_excerpt": desc,
    }


def _build_signals(db_path: Path) -> pd.DataFrame:
    """Run the production pipeline end-to-end and return a (signals ⋈ listings)
    DataFrame with every column the worker cards need."""
    from src.storage.database import init_db, get_session
    from src.storage.repository import get_listings_df, get_price_history_df
    from src.analytics.computed_columns import enrich_listings
    from src.analytics.turnover import compute_turnover_stats
    from src.dashboard.data_loader import compute_signals
    from src.parser.llm_enrichment import merge_real_mileage

    print(f"[hot_deals] init DB {db_path}", flush=True)
    init_db(str(db_path))
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
                  "first_seen_at", "seller_type", "transmission", "is_active"]
    extra = listings[[c for c in extra_cols if c in listings.columns]].drop_duplicates("olx_id")
    merged = signals.merge(extra, on="olx_id", how="left", suffixes=("", "_l"))
    return merged


def _pick_zone_deals(signals: pd.DataFrame, zone: str, districts: list[str] | None,
                     top_n: int, max_age_days: int) -> pd.DataFrame:
    df = signals
    if "is_active" in df.columns:
        df = df[df["is_active"].fillna(True).astype(bool)]
    if max_age_days and "first_seen_at" in df.columns:
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=max_age_days)
        ts = pd.to_datetime(df["first_seen_at"], errors="coerce")
        df = df[ts >= cutoff]
    if districts is not None:
        df = df[df["district"].isin(districts)]
    if df.empty:
        return df
    sort_col = "flip_score" if "flip_score" in df.columns else "adjusted_undervaluation_pct"
    return df.sort_values(sort_col, ascending=False).head(top_n).copy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--db", type=Path, default=REPO_ROOT / "data" / "olx_cars.db")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "hot_deals")
    ap.add_argument("--top-n", type=int, default=TOP_N_PER_ZONE)
    ap.add_argument("--max-age-days", type=int, default=MAX_LISTING_AGE_DAYS)
    ap.add_argument("--fetch-photos", dest="fetch_photos", action="store_true",
                    default=True, help="(default) fetch og:image from OLX per deal")
    ap.add_argument("--no-fetch-photos", dest="fetch_photos", action="store_false",
                    help="skip OLX HTTP calls — JSON ships without photos")
    ap.add_argument("--photo-sleep-sec", type=float, default=0.3,
                    help="polite sleep between OLX HTTP calls")
    args = ap.parse_args()

    if not args.db.exists():
        raise SystemExit(f"DB not found: {args.db}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    signals = _build_signals(args.db)

    # All zones, plus the "all" aggregate (admin / "observer" PINs).
    zone_plan: list[tuple[str, list[str] | None]] = list(ZONE_DISTRICTS.items())
    zone_plan.append(("all", None))

    built_at = pd.Timestamp.now("UTC").tz_localize(None).isoformat(timespec="seconds") + "Z"
    overall_counts: dict[str, int] = {}

    for zone, districts in zone_plan:
        picked = _pick_zone_deals(signals, zone, districts, args.top_n, args.max_age_days)
        deals: list[dict] = []
        for _, row in picked.iterrows():
            photo: str | None = None
            if args.fetch_photos and row.get("url"):
                photo = fetch_og_image(row["url"])
                if photo:
                    time.sleep(args.photo_sleep_sec)
                else:
                    # Skip listings whose OLX page is 410/redirected — we
                    # can't show them without a working image, and the link
                    # would be dead anyway.
                    continue
            deals.append(_format_deal(row.to_dict(), photo))

        out_path = args.out_dir / f"hot_deals_{zone}.json"
        payload = {
            "zone": zone,
            "built_at": built_at,
            "deals": deals,
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, default=str, indent=2))
        overall_counts[zone] = len(deals)
        print(f"[hot_deals]   {zone:<6} {len(deals):>3} deals → {out_path.name}", flush=True)

    print(f"[hot_deals] DONE  built_at={built_at}  zones={overall_counts}", flush=True)


if __name__ == "__main__":
    main()
