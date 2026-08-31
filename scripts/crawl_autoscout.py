#!/usr/bin/env python3
"""Read the German asking prices the import pages compare against.

Runs a small, budgeted pass over AutoScout24 search pages and writes what it
finds into ``import_listings``. Nothing here touches the Portuguese corpus.

The work queue is derived from our own data rather than from a hand-kept list:
a (model, year) pair is worth a request only where Portugal has enough active
listings to put a median against it, so the crawl follows the pages that exist.
Pairs already refreshed inside ``--max-age-days`` are skipped, so consecutive
runs walk forward through the queue instead of re-reading the same models. A
model whose name AutoScout24 does not know comes back empty once and is then
dropped for the rest of the run instead of spending a request on every one of
its years. The common case of that — Portugal naming an estate or a coupé as a
model ("Peugeot 308 SW", "Seat Leon ST") — is handled by ``as24_query``, which
moves the suffix into AutoScout24's body filter.

Rows are stamped with the Portuguese (brand, model) they were fetched for, not
with the German model string: a body-filtered query returns cars AutoScout24
calls "508" and Portugal calls "508 SW", and the import pages join on our
vocabulary. What AutoScout24 called them survives in ``variant`` and
``model_group``.

Politeness is the point, not a side effect — see ``src.parser.autoscout`` for
what this crawler does and does not do on someone else's site. The two knobs
that matter are ``--budget`` (hard cap on requests per run) and ``--delay-min``
/ ``--delay-max``; a 403 or 429 ends the run immediately.

Use:
    python scripts/crawl_autoscout.py --budget 200
    python scripts/crawl_autoscout.py --models 60 --years 12 --dry-run
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

_BODY_SUFFIXES = (
    (" sport tourer", "bt_kombi"), (" sports tourer", "bt_kombi"),
    (" sw", "bt_kombi"), (" st", "bt_kombi"), (" break", "bt_kombi"),
    (" variant", "bt_kombi"), (" caravan", "bt_kombi"), (" avant", "bt_kombi"),
    (" touring", "bt_kombi"), (" estate", "bt_kombi"),
    (" gran coupe", "bt_coupe"), (" gran coupé", "bt_coupe"),
    (" coupe", "bt_coupe"), (" coupé", "bt_coupe"),
    (" cabrio", "bt_cabrio"), (" cabriolet", "bt_cabrio"),
)


def as24_query(brand: str, model: str) -> tuple[str, str, str | None]:
    """(make token, model token, body segment) for a Portuguese (brand, model).

    Portugal names estates and coupés as models — "Peugeot 308 SW", "Seat Leon
    ST", "Renault Mégane Sport Tourer" — where AutoScout24 has one model and a
    body type. Asking for the Portuguese name gets a 404, so the suffix is moved
    into the body filter and the base model is queried instead. Everything else
    goes through unchanged: AutoScout24 canonicalises "/mercedes-benz/c-220" to
    its own path on a redirect, so no per-model table is needed.
    """
    from src.analytics.model_pages import slugify

    low = f" {model.strip().lower()}"
    for suffix, body in _BODY_SUFFIXES:
        if low.endswith(suffix):
            base = model.strip()[: -len(suffix.strip())].strip()
            if base:
                return slugify(brand), slugify(base), body
    return slugify(brand), slugify(model), None


MIN_PT_YEAR_N = 5
MIN_PT_MODEL_N = 20
DEFAULT_MODELS = 60
DEFAULT_YEARS = 12
DEFAULT_BUDGET = 200
DEFAULT_MAX_AGE_DAYS = 7


def _targets(session, *, models: int, years: int, now_year: int) -> list[tuple]:
    """[(brand, model, year, pt_n)] worth a request, deepest Portuguese sample first."""
    import pandas as pd
    from src.storage.repository import get_listings_df

    df = get_listings_df(session)
    if df.empty:
        return []
    active = df[df["is_active"] == True]  # noqa: E712
    active = active[pd.to_numeric(active.get("price_eur"), errors="coerce").notna()]
    yr = pd.to_numeric(active.get("year"), errors="coerce")
    active = active.assign(_y=yr).dropna(subset=["_y"])
    active = active[(active["_y"] >= now_year - years) & (active["_y"] <= now_year)]

    deep = (active.groupby(["brand", "model"]).size()
            .sort_values(ascending=False))
    deep = deep[deep >= MIN_PT_MODEL_N].head(models)
    keep = set(deep.index)

    out = []
    for (brand, model, year), n in active.groupby(["brand", "model", "_y"]).size().items():
        if (brand, model) not in keep or n < MIN_PT_YEAR_N:
            continue
        out.append((str(brand), str(model), int(year), int(n)))
    out.sort(key=lambda t: (-t[3], t[0], t[1], -t[2]))
    return out


def _fresh_pairs(session, max_age_days: int) -> set[tuple]:
    """(brand, model, year) refreshed recently enough to skip this run."""
    from datetime import timedelta
    from sqlalchemy import func
    from src.models.import_listing import ImportListing
    from src.storage.repository import _utcnow

    cutoff = _utcnow() - timedelta(days=max_age_days)
    rows = (session.query(ImportListing.brand, ImportListing.model, ImportListing.year,
                          func.max(ImportListing.last_seen_at))
            .group_by(ImportListing.brand, ImportListing.model, ImportListing.year)
            .all())
    return {(b, m, y) for b, m, y, seen in rows if seen is not None and seen >= cutoff}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=None)
    ap.add_argument("--budget", type=int, default=DEFAULT_BUDGET,
                    help="hard cap on requests this run")
    ap.add_argument("--models", type=int, default=DEFAULT_MODELS)
    ap.add_argument("--years", type=int, default=DEFAULT_YEARS)
    ap.add_argument("--pages", type=int, default=1, help="pages per model-year")
    ap.add_argument("--max-age-days", type=int, default=DEFAULT_MAX_AGE_DAYS)
    ap.add_argument("--delay-min", type=float, default=None)
    ap.add_argument("--delay-max", type=float, default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the queue and exit without a single request")
    args = ap.parse_args()

    from src.analytics.model_pages import slugify
    from src.parser.autoscout import (
        AutoScoutBlocked, AutoScoutClient, AutoScoutConfig, DELAY_MAX, DELAY_MIN,
    )
    from src.storage.database import init_db, get_session
    from src.storage.repository import upsert_import_listings

    init_db(args.db)
    session = get_session()
    now_year = time.gmtime().tm_year

    queue = _targets(session, models=args.models, years=args.years, now_year=now_year)
    if not queue:
        print("[as24] no Portuguese sample deep enough to benchmark — nothing to do")
        return 0
    skip = _fresh_pairs(session, args.max_age_days)
    pending = [t for t in queue if (t[0], t[1], t[2]) not in skip]
    print(f"[as24] queue: {len(queue)} model-years, {len(pending)} stale enough to fetch, "
          f"budget {args.budget}", flush=True)

    if args.dry_run:
        for brand, model, year, n in pending[:40]:
            make_token, model_token, body = as24_query(brand, model)
            print(f"       {brand} {model} {year}  (PT n={n})  "
                  f"→ /lst/{make_token}/{model_token}{'/' + body if body else ''} re_{year}")
        return 0

    config = AutoScoutConfig(
        budget=args.budget,
        delay_min=args.delay_min if args.delay_min is not None else DELAY_MIN,
        delay_max=args.delay_max if args.delay_max is not None else DELAY_MAX,
    )
    seen_models: set[tuple] = set()
    unmapped: set[tuple] = set()
    inserted = updated = fetched = empty = 0
    blocked = False
    t0 = time.perf_counter()
    with AutoScoutClient(config=config) as client:
        for brand, model, year, _n in pending:
            if client.spent >= config.budget:
                break
            if (brand, model) in unmapped:
                continue
            make_token, model_token, body = as24_query(brand, model)
            try:
                listings = client.model_year(make_token, model_token, year,
                                             max_pages=args.pages, body=body)
            except AutoScoutBlocked as exc:
                print(f"[as24] stopped: {exc} — the site asked us to back off", flush=True)
                blocked = True
                break
            fetched += 1
            if not listings:
                empty += 1
                unmapped.add((brand, model))
                continue
            seen_models.add((brand, model))
            for item in listings:
                item.brand, item.model = brand, model
            ins, upd = upsert_import_listings(session, listings)
            inserted += ins
            updated += upd

    print(f"[as24] {fetched} model-years read ({empty} with nothing), "
          f"{len(seen_models)} models, {inserted} new listings, {updated} refreshed, "
          f"{client.spent} requests in {time.perf_counter() - t0:.0f}s", flush=True)
    if unmapped:
        names = ", ".join(f"{b} {m}" for b, m in sorted(unmapped))
        print(f"[as24] no AutoScout24 model behind these, skipped for the rest of "
              f"the run: {names}", flush=True)
    return 2 if blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())
