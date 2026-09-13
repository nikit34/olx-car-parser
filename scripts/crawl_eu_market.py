#!/usr/bin/env python3
"""Read the German, French and Italian used-car markets from AutoScout24.

Each country becomes its own corpus in ``import_listings`` under the source
``src.countries`` assigns it (``as24_de``, ``as24_fr``, ``as24_it``), from which
it gets its own price model and its own pages. Nothing here touches the
Portuguese corpus, and nothing here touches the weekly German benchmark crawl
(``scripts/crawl_autoscout.py``, source ``autoscout24``), which keeps stamping
its rows with Portuguese model names for the import pages.

The work is three passes per country, in this order, all inside one request
budget:

1. Discovery. A make page (``/lst/{make}``) carries AutoScout24's own model
   list for that make, so the crawl learns its vocabulary from the site instead
   of a hand-kept table. Refreshed once a month.
2. Inventory. One newest-first page per (make, model) says how many cars the
   model has for sale. Models under ``min_model_results`` are not worth a year
   by year walk and are left out of the harvest; the page's twenty cars are
   still real data and are stored.
3. Harvest. The (make, model, year) cells of kept models, stalest first, never
   seen first of all, skipping what was refreshed inside ``cell_max_age_days``.
   A cell whose result count fits in the pages read was enumerated in full, and
   the rows it no longer lists are retired then and there; everything else is
   retired by age at the end of the pass.

Discovery and inventory live in a JSON state file so a run that is killed or
runs out of budget loses at most one make of work. Harvest progress lives in
the database itself (``last_seen_at``), which is what makes the queue stalest
first without a second bookkeeping.

Countries run in parallel threads because each is a different host and the
per-host pacing is what has to stay polite; the wall clock does not. Every
request still goes through ``AutoScoutClient``, so the robots rules, the delay,
the budget and the 403/429 stop are in one place. A blocked country stops that
country only; the run exits 2 if any country was blocked.

Use:
    python scripts/crawl_eu_market.py --dry-run
    python scripts/crawl_eu_market.py --country DE --budget 300
    python scripts/crawl_eu_market.py --pages 1 --years 10
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.analytics.model_pages import slugify  # noqa: E402
from src.countries import EU_COUNTRIES, Country, country  # noqa: E402
from src.parser.autoscout import AutoScoutBlocked  # noqa: E402
from src.parser.brand_normalize import normalize_brand  # noqa: E402
from src.storage.repository import (  # noqa: E402
    _utcnow,
    deactivate_import_missing,
    expire_import_listings,
    upsert_import_listings,
)

DEFAULT_CONFIG = REPO_ROOT / "config" / "eu_markets.yaml"
DEFAULT_STATE = REPO_ROOT / "data" / "eu_crawl_state.json"
PAGE_SIZE = 20
STATE_FLUSH_EVERY = 20
DRY_RUN_CELLS = 30

_OTHER_MODEL_LABELS = {"sonstige", "andere", "autres", "autre", "altro", "altri", "other", "others"}


@dataclass
class MarketConfig:
    """One country's block of ``config/eu_markets.yaml`` merged over ``defaults``."""

    makes: list[str]
    years_back: int = 15
    pages_per_cell: int = 2
    min_model_results: int = 40
    daily_budget: int = 450
    delay_min: float = 3.0
    delay_max: float = 6.0
    cell_max_age_days: int = 7
    discovery_max_age_days: int = 30
    expire_after_days: int = 21


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, MarketConfig]:
    """{country code: MarketConfig}. Unknown keys fail loudly rather than silently."""
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    defaults = dict(doc.get("defaults") or {})
    out: dict[str, MarketConfig] = {}
    for code, block in (doc.get("countries") or {}).items():
        block = dict(block or {})
        makes = [str(m).strip() for m in (block.pop("makes", None) or []) if str(m).strip()]
        out[str(code).upper()] = MarketConfig(makes=makes, **{**defaults, **block})
    return out


@dataclass(frozen=True)
class Cell:
    """One (make, model, year) search, with the names the database stores it under."""

    make_slug: str
    model_slug: str
    brand: str
    model: str
    year: int

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.brand, self.model, self.year)


class CrawlState:
    """The JSON file of what discovery and inventory learned, per country.

    Shape: ``{code: {"discovered": {make_slug: {"at", "label", "models": [
    {"label", "slug", "model_id"}]}}, "inventory": {"make/model": {"results",
    "at"}}}}``. One lock covers reads, writes and the save, since each country
    runs in its own thread and the file is shared.
    """

    def __init__(self, path: str | Path, data: dict | None = None):
        self.path = Path(path)
        self.data: dict = data if data is not None else {}
        self.lock = threading.RLock()

    @classmethod
    def load(cls, path: str | Path) -> "CrawlState":
        path = Path(path)
        data: dict = {}
        if path.exists():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                data = loaded if isinstance(loaded, dict) else {}
            except json.JSONDecodeError:
                data = {}
        return cls(path, data)

    def country(self, code: str) -> dict:
        with self.lock:
            entry = self.data.setdefault(code.upper(), {})
            entry.setdefault("discovered", {})
            entry.setdefault("inventory", {})
            return entry

    def save(self) -> None:
        with self.lock:
            payload = json.dumps(self.data, ensure_ascii=False, indent=1, sort_keys=True)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(payload, encoding="utf-8")
            os.replace(tmp, self.path)


def _iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat()


def _parse_iso(value) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return parsed.replace(tzinfo=None) if parsed.tzinfo is not None else parsed


def _fresh(at_iso, now: datetime, max_age_days: int) -> bool:
    at = _parse_iso(at_iso)
    return at is not None and at >= now - timedelta(days=max_age_days)


def _budget_left(client) -> bool:
    return client.spent < client.config.budget


def make_label(make_slug: str, listings) -> str:
    """The brand the rows are stored under: the site's own spelling, canonicalised."""
    for item in listings:
        brand = getattr(item, "brand", None)
        if brand:
            return normalize_brand(brand)
    return normalize_brand(make_slug.replace("-", " ").title())


def model_entries(make_models) -> list[dict]:
    """State entries for a make page's model list; the "other" bucket is dropped."""
    out = []
    for item in make_models or []:
        label = str(item.get("label") or "").strip()
        if not label or label.lower() in _OTHER_MODEL_LABELS:
            continue
        slug = slugify(label)
        if not slug:
            continue
        out.append({"label": label, "slug": slug, "model_id": item.get("value")})
    return out


def used_only(listings):
    """Drop the cars offered as new.

    The used-only filter cannot be left to the URL: ``autoscout24.de`` accepts
    ``ustate=U`` and then echoes back no ``ustate`` at all, so a query that
    looks filtered may not be. A new car's asking price is not a used-market
    price, and the recent model years are exactly where one would leak in, so
    the offer type printed on the card is what decides.
    """
    return [item for item in listings
            if str(getattr(item, "offer_type", "") or "").strip().upper() != "N"]


def stamp(listings, source: str, brand: str, model: str) -> None:
    """Store rows under the discovery vocabulary, not the card's free-text names."""
    for item in listings:
        item.source = source
        item.brand = brand
        item.model = model


@dataclass
class CountryResult:
    code: str
    source: str
    makes_read: int = 0
    makes_empty: int = 0
    models_probed: int = 0
    models_kept: int = 0
    cells_pending: int = 0
    cells_read: int = 0
    cells_empty: int = 0
    cells_failed: int = 0
    inserted: int = 0
    updated: int = 0
    deactivated: int = 0
    expired: int = 0
    requests: int = 0
    seconds: float = 0.0
    blocked: bool = False
    error: str | None = None


def discover(client, cfg: MarketConfig, state: CrawlState, code: str, *,
             now: datetime, log=print) -> tuple[int, int]:
    """Refresh the model list of every make whose discovery is older than a month.

    Returns (makes read, makes that came back without models). A page that did
    not parse at all is not recorded, so a transient error is retried next run;
    a page that parsed with no taxonomy is recorded empty and left alone for
    ``discovery_max_age_days``.
    """
    cstate = state.country(code)
    read = empty = 0
    for make in cfg.makes:
        with state.lock:
            entry = cstate["discovered"].get(make)
        if entry and _fresh(entry.get("at"), now, cfg.discovery_max_age_days):
            continue
        if not _budget_left(client):
            break
        listings, meta = client.make_page(make)
        read += 1
        if not meta:
            log(f"[{code}] make page for {make} did not parse, will retry next run")
            continue
        models = model_entries(meta.get("make_models"))
        if not models:
            empty += 1
            log(f"[{code}] make page for {make} lists no models, skipped")
        with state.lock:
            cstate["discovered"][make] = {
                "at": _iso(now), "label": make_label(make, listings), "models": models,
            }
        state.save()
    return read, empty


def probe_inventory(client, cfg: MarketConfig, state: CrawlState, cty: Country, session, *,
                    now: datetime, result: CountryResult, log=print) -> None:
    """One newest-first page per (make, model) not probed inside a month."""
    cstate = state.country(cty.code)
    since_flush = 0
    for make in cfg.makes:
        with state.lock:
            entry = cstate["discovered"].get(make)
            models = list((entry or {}).get("models") or [])
            brand = normalize_brand((entry or {}).get("label") or make)
        if not models:
            continue
        for model in models:
            key = f"{make}/{model['slug']}"
            with state.lock:
                inv = cstate["inventory"].get(key)
            if inv and _fresh(inv.get("at"), now, cfg.discovery_max_age_days):
                continue
            if not _budget_left(client):
                state.save()
                return
            listings, meta = client.search(make, model["slug"], page=1, ustate="U",
                                           sort="age", desc=True)
            listings = used_only(listings)
            result.models_probed += 1
            if not meta or meta.get("results") is None:
                continue
            with state.lock:
                cstate["inventory"][key] = {"results": int(meta["results"]), "at": _iso(now)}
            if listings:
                stamp(listings, cty.source, brand, model["label"])
                ins, upd = upsert_import_listings(session, listings)
                result.inserted += ins
                result.updated += upd
            since_flush += 1
            if since_flush >= STATE_FLUSH_EVERY:
                state.save()
                since_flush = 0
        state.save()


def kept_models(cfg: MarketConfig, state: CrawlState, code: str) -> list[tuple[str, str, dict]]:
    """[(make_slug, brand, model entry)] with enough cars for sale to walk year by year."""
    cstate = state.country(code)
    out = []
    with state.lock:
        for make in cfg.makes:
            entry = cstate["discovered"].get(make)
            if not entry:
                continue
            brand = normalize_brand(entry.get("label") or make)
            for model in entry.get("models") or []:
                inv = cstate["inventory"].get(f"{make}/{model['slug']}")
                if inv and int(inv.get("results") or 0) >= cfg.min_model_results:
                    out.append((make, brand, dict(model)))
    return out


def build_cells(models: list[tuple[str, str, dict]], *, now_year: int,
                years_back: int) -> list[Cell]:
    """Every (make, model, year) of the kept models, newest year first."""
    return [
        Cell(make, model["slug"], brand, model["label"], year)
        for make, brand, model in models
        for year in range(now_year, now_year - years_back - 1, -1)
    ]


def order_cells(cells: list[Cell], last_seen: dict[tuple, datetime | None], *,
                now: datetime, cell_max_age_days: int) -> list[Cell]:
    """Stalest first, never seen first of all; freshly refreshed cells drop out."""
    cutoff = now - timedelta(days=cell_max_age_days)
    pending = [c for c in cells
               if last_seen.get(c.key) is None or last_seen[c.key] < cutoff]
    pending.sort(key=lambda c: (last_seen.get(c.key) is not None,
                                last_seen.get(c.key) or datetime.min))
    return pending


def cell_last_seen(session, source: str) -> dict[tuple, datetime]:
    """{(brand, model, year): MAX(last_seen_at)} for one source."""
    from sqlalchemy import func
    from src.models.import_listing import ImportListing

    rows = (session.query(ImportListing.brand, ImportListing.model, ImportListing.year,
                          func.max(ImportListing.last_seen_at))
            .filter(ImportListing.source == source)
            .group_by(ImportListing.brand, ImportListing.model, ImportListing.year)
            .all())
    return {(b, m, y): seen for b, m, y, seen in rows if seen is not None}


def harvest(client, cfg: MarketConfig, cells: list[Cell], state: CrawlState, cty: Country,
            session, *, pages_per_cell: int, result: CountryResult, log=print) -> None:
    """Read each cell's newest pages, store them, retire what a full read no longer lists."""
    for index, cell in enumerate(cells, start=1):
        if not _budget_left(client):
            break
        batch = []
        pages_fetched = 0
        results = None
        total_pages = 1
        for page in range(1, max(1, pages_per_cell) + 1):
            if page > 1 and (page > total_pages or not _budget_left(client)):
                break
            listings, meta = client.search(cell.make_slug, cell.model_slug, year=cell.year,
                                           page=page, ustate="U", sort="age", desc=True)
            if not meta:
                break
            pages_fetched += 1
            if page == 1:
                results = meta.get("results")
                total_pages = int(meta.get("pages") or 1)
            batch.extend(used_only(listings))
        if pages_fetched == 0:
            result.cells_failed += 1
            continue
        result.cells_read += 1
        if not batch:
            result.cells_empty += 1
        stamp(batch, cty.source, cell.brand, cell.model)
        seen_ids = {str(item.external_id) for item in batch}
        if batch:
            ins, upd = upsert_import_listings(session, batch)
            result.inserted += ins
            result.updated += upd
        if results is not None and int(results) <= PAGE_SIZE * pages_fetched:
            result.deactivated += deactivate_import_missing(
                session, cty.source, [cell.key], seen_ids)
        if index % STATE_FLUSH_EVERY == 0:
            state.save()


def crawl_country(cty: Country, cfg: MarketConfig, state: CrawlState, client, session, *,
                  now: datetime | None = None, pages_per_cell: int | None = None,
                  years_back: int | None = None, log=print) -> CountryResult:
    """The three passes for one country, then the age-based expiry."""
    now = now or _utcnow()
    pages = pages_per_cell if pages_per_cell is not None else cfg.pages_per_cell
    years = years_back if years_back is not None else cfg.years_back
    result = CountryResult(code=cty.code, source=cty.source)
    t0 = time.perf_counter()
    try:
        result.makes_read, result.makes_empty = discover(client, cfg, state, cty.code,
                                                         now=now, log=log)
        probe_inventory(client, cfg, state, cty, session, now=now, result=result, log=log)
        kept = kept_models(cfg, state, cty.code)
        result.models_kept = len(kept)
        cells = order_cells(build_cells(kept, now_year=now.year, years_back=years),
                            cell_last_seen(session, cty.source),
                            now=now, cell_max_age_days=cfg.cell_max_age_days)
        result.cells_pending = len(cells)
        harvest(client, cfg, cells, state, cty, session, pages_per_cell=pages,
                result=result, log=log)
    except AutoScoutBlocked as exc:
        result.blocked = True
        log(f"[{cty.source}] stopped: {exc} — the site asked us to back off", flush=True)
    finally:
        state.save()
    result.expired = expire_import_listings(session, cty.source, cfg.expire_after_days, now=now)
    result.requests = client.spent
    result.seconds = time.perf_counter() - t0
    return result


def summary(result: CountryResult) -> str:
    if result.error:
        return f"[{result.source}] failed: {result.error}"
    return (f"[{result.source}] {result.makes_read} makes read ({result.makes_empty} empty), "
            f"{result.models_probed} models probed, {result.models_kept} kept, "
            f"{result.cells_read}/{result.cells_pending} cells read "
            f"({result.cells_empty} with nothing, {result.cells_failed} failed), "
            f"{result.inserted} new listings, {result.updated} refreshed, "
            f"{result.deactivated} retired, {result.expired} expired, "
            f"{result.requests} requests in {result.seconds:.0f}s"
            + (" — blocked" if result.blocked else ""))


def run(countries: list[Country], configs: dict[str, MarketConfig], state: CrawlState, *,
        client_factory, session_factory, pages_per_cell: int | None = None,
        years_back: int | None = None, log=print) -> list[CountryResult]:
    """One thread per country; a blocked or crashed country leaves the others running."""
    results: dict[str, CountryResult] = {}
    for cty in countries:
        state.country(cty.code)

    def worker(cty: Country) -> None:
        cfg = configs[cty.code]
        session = session_factory()
        try:
            with client_factory(cty, cfg) as client:
                results[cty.code] = crawl_country(cty, cfg, state, client, session,
                                                  pages_per_cell=pages_per_cell,
                                                  years_back=years_back, log=log)
        except Exception as exc:
            results[cty.code] = CountryResult(code=cty.code, source=cty.source,
                                              error=f"{type(exc).__name__}: {exc}")
            log(f"[{cty.source}] crashed: {exc!r}", flush=True)
        finally:
            close = getattr(session, "close", None)
            if callable(close):
                close()

    threads = [threading.Thread(target=worker, args=(cty,), name=f"crawl-{cty.code}")
               for cty in countries]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return [results[cty.code] for cty in countries]


def _client_factory(budget: int | None, delay_min: float | None, delay_max: float | None):
    from src.parser.autoscout import AutoScoutClient, AutoScoutConfig

    def make(cty: Country, cfg: MarketConfig):
        config = AutoScoutConfig(
            tld=cty.as24_tld, country=cty.as24_cy,
            budget=budget if budget is not None else cfg.daily_budget,
            delay_min=delay_min if delay_min is not None else cfg.delay_min,
            delay_max=delay_max if delay_max is not None else cfg.delay_max,
        )
        return AutoScoutClient(config=config)

    return make


def dry_run(countries: list[Country], configs: dict[str, MarketConfig], state: CrawlState,
            session, *, budget: int | None = None, years_back: int | None = None,
            now: datetime | None = None, log=print) -> None:
    """Print each country's queue as the run would see it, without one request."""
    now = now or _utcnow()
    for cty in countries:
        cfg = configs[cty.code]
        cstate = state.country(cty.code)
        stale_makes = [m for m in cfg.makes
                       if not _fresh((cstate["discovered"].get(m) or {}).get("at"), now,
                                     cfg.discovery_max_age_days)]
        to_probe = 0
        for make in cfg.makes:
            for model in (cstate["discovered"].get(make) or {}).get("models") or []:
                inv = cstate["inventory"].get(f"{make}/{model['slug']}")
                if not (inv and _fresh(inv.get("at"), now, cfg.discovery_max_age_days)):
                    to_probe += 1
        kept = kept_models(cfg, state, cty.code)
        years = years_back if years_back is not None else cfg.years_back
        cells = order_cells(build_cells(kept, now_year=now.year, years_back=years),
                            cell_last_seen(session, cty.source),
                            now=now, cell_max_age_days=cfg.cell_max_age_days)
        log(f"[{cty.source}] discovery: {len(stale_makes)}/{len(cfg.makes)} makes to read; "
            f"inventory: {to_probe} models to probe; harvest: {len(kept)} models kept, "
            f"{len(cells)} cells stale enough to fetch; "
            f"budget {budget if budget is not None else cfg.daily_budget}", flush=True)
        seen = cell_last_seen(session, cty.source)
        for cell in cells[:DRY_RUN_CELLS]:
            last = seen.get(cell.key)
            log(f"       {cell.brand} {cell.model} {cell.year}  "
                f"→ /lst/{cell.make_slug}/{cell.model_slug} fregfrom={cell.year}  "
                f"(last seen {_iso(last) if last else 'never'})", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--country", action="append", default=None,
                    help="country code, repeatable (default: every EU country)")
    ap.add_argument("--budget", type=int, default=None,
                    help="hard cap on requests per country this run (default: yaml)")
    ap.add_argument("--pages", type=int, default=None, help="pages per cell (default: yaml)")
    ap.add_argument("--years", type=int, default=None,
                    help="how many years back to harvest (default: yaml)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the queue per country and exit without a single request")
    ap.add_argument("--db", default=None)
    ap.add_argument("--delay-min", type=float, default=None)
    ap.add_argument("--delay-max", type=float, default=None)
    ap.add_argument("--state", default=str(DEFAULT_STATE))
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = ap.parse_args(argv)

    codes = [c.upper() for c in (args.country or list(EU_COUNTRIES))]
    countries = [country(code) for code in codes]
    configs = load_config(args.config)
    missing = [c for c in codes if c not in configs]
    if missing:
        print(f"[eu] no block in {args.config} for: {', '.join(missing)}", flush=True)
        return 1
    state = CrawlState.load(args.state)

    from src.storage.database import get_session, init_db

    init_db(args.db)
    if args.dry_run:
        session = get_session()
        try:
            dry_run(countries, configs, state, session, budget=args.budget,
                    years_back=args.years)
        finally:
            session.close()
        return 0

    results = run(countries, configs, state,
                  client_factory=_client_factory(args.budget, args.delay_min, args.delay_max),
                  session_factory=get_session, pages_per_cell=args.pages,
                  years_back=args.years)
    for result in results:
        print(summary(result), flush=True)
    if any(r.blocked for r in results):
        return 2
    return 1 if any(r.error for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
