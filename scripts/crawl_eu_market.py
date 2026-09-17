#!/usr/bin/env python3
"""Read the German, French and Italian used-car markets from AutoScout24.

Each country becomes its own corpus in ``listings`` under the source
``src.countries`` assigns it (``as24_de``, ``as24_fr``, ``as24_it``), from which
it gets its own price model and its own pages. Every market shares one table
now, kept apart by ``country_code``; nothing here touches the Portuguese
corpus, and nothing here touches the weekly German benchmark crawl
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
4. Deepening. A search card carries no body type, no colour, no drive train,
   no seller text and — outside Germany — no CO2, and those are the difference
   between a row that can be priced beside a Portuguese one and a row that
   cannot. Each market is deepened by whichever road its ``robots.txt`` leaves
   open, and the crawler asks the file rather than remembering the answer.

   France and Italy leave the advert open, so every card that needs it is
   opened. Germany does not — ``autoscout24.de`` disallows ``/angebote/`` — so
   its cars get their body type from the site's own body filter instead: the
   same cell asked again through ``bt_kombi`` and friends, where everything
   that comes back has that body. It is the one advert field reachable without
   the advert, and the rest is simply not had there.

   Two things keep this from eating the pass. A car is deepened once: the body
   already in the database says the advert has been read, and a cell that comes
   round again costs nothing for the cars it already knows. And the deepening
   may spend at most ``enrich_share`` of the country's budget, so the harvest
   keeps the rest and a cell still comes round inside ``expire_after_days`` —
   without that cap a cell of twenty cars costs twenty-one requests instead of
   one, and the corpus starts retiring rows faster than the crawl re-reads
   them. ``--no-adverts`` skips the deepening entirely.

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
    python scripts/crawl_eu_market.py --country IT --no-adverts
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.analytics.model_pages import slugify  # noqa: E402
from src.countries import EU_COUNTRIES, Country, country  # noqa: E402
from src.parser import autoscout  # noqa: E402
from src.parser.autoscout import AutoScoutBlocked, adverts_allowed  # noqa: E402
from src.parser.market_card import merge_patch  # noqa: E402
from src.parser.brand_normalize import normalize_brand  # noqa: E402
from src.storage.repository import (  # noqa: E402
    _utcnow,
    deactivate_import_missing,
    expire_import_listings,
    listings_with_body,
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
    enrich_share: float = 0.5
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


def _ceiling(client, ceiling: int | None) -> int:
    """How many requests this pass may spend on deepening rather than coverage."""
    return client.config.budget if ceiling is None else max(0, int(ceiling))


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
    adverts_read: int = 0
    adverts_missed: int = 0
    bodies_labelled: int = 0
    enrich_requests: int = 0
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


def advert_path(url: str) -> str:
    """A card's advert as a path the client can ask for.

    The card stores an absolute URL and ``AutoScoutClient.fetch`` prepends the
    national host, so the origin has to come off or the request is built twice.
    """
    text = str(url or "")
    if not text.startswith("http"):
        return text if text.startswith("/") else "/" + text
    rest = text.split("//", 1)[-1]
    slash = rest.find("/")
    return rest[slash:] if slash >= 0 else "/"


def deepen(client, rows: list[dict], result: CountryResult, *, session, cty: Country,
           make_slug: str, model_slug: str, year: int | None, bodies, ceiling: int) -> None:
    """Fill what the card could not, by whichever road this market leaves open.

    A car is deepened once. The body type already stored says its advert has
    been read — or, where adverts are closed, that it has already been through
    the body filter — and nothing here is spent on it again. On a mature corpus
    that is half the rows of a pass, which is the difference between a crawl
    that keeps up with its own expiry and one that falls behind it.

    Once, too, within the page: AutoScout24 repeats a promoted listing further
    down its own result page, and both copies are the same car. One of them is
    deepened and the other is given what it learned, rather than the same
    advert being bought twice.
    """
    if not rows:
        return
    known = listings_with_body(session, cty.source,
                               [row.get("external_id") for row in rows])
    copies: dict[str, list[dict]] = {}
    for row in rows:
        copies.setdefault(str(row.get("external_id")), []).append(row)
    pending = [group[0] for ext, group in copies.items()
               if ext not in known and not group[0].get("body_type")]
    if pending:
        if adverts_allowed(cty.as24_tld):
            read_adverts(client, pending, result, ceiling=ceiling)
        else:
            label_bodies(client, pending, result, tld=cty.as24_tld, make_slug=make_slug,
                         model_slug=model_slug, year=year, bodies=bodies, ceiling=ceiling)
    for group in copies.values():
        for other in group[1:]:
            for key, value in group[0].items():
                if value is not None and other.get(key) is None:
                    other[key] = value


def read_adverts(client, rows: list[dict], result: CountryResult, *, ceiling: int) -> None:
    """Open each card's advert and fold what it says into the row.

    What it buys is the fields no search card carries: the body type, which is
    why ``segment`` was null for every foreign row; CO2, which the ISV formula
    needs and which the French and Italian cards omit; and the colour, the
    doors, the drive train and the seller's own text, which is what a
    Portuguese row has always had.

    A missing advert is never a missing car: the row is stored either way, and
    ``adverts_missed`` counts the difference — budget gone, ceiling reached, or
    a page that would not parse — so a pass that quietly stopped enriching shows
    up in the summary rather than only in the data.
    """
    for row in rows:
        if result.enrich_requests >= ceiling or not _budget_left(client):
            result.adverts_missed += 1
            continue
        before = client.spent
        patch = client.advert(advert_path(row.get("url")))
        result.enrich_requests += client.spent - before
        if patch:
            merge_patch(row, patch, autoscout.CORRECTS)
            result.adverts_read += 1
        else:
            result.adverts_missed += 1


def body_order(bodies, filters: dict) -> list[str]:
    """Which body filters to ask through, in which order.

    The page's own facet list is the short answer: it is the bodies that model
    actually has, so anything outside it is a request that can only come back
    empty. A page that shipped no list — which is what a model with a single
    body does — leaves the whole vocabulary, walked until every car is
    accounted for.
    """
    listed = [slug for slug in (bodies or []) if slug in filters]
    return listed or list(filters)


def label_bodies(client, rows: list[dict], result: CountryResult, *, tld: str,
                 make_slug: str, model_slug: str, year: int | None, bodies,
                 ceiling: int) -> None:
    """Give a market with no advert the one advert field it can still have.

    ``autoscout24.de`` disallows ``/angebote/``, so the body type — the field
    ``segment`` is built from, and the reason every German row had none — has
    to come from a page we are allowed to ask for. A search filtered to one
    body returns only cars of that body, so the filter labels every car it
    returns, and the walk stops as soon as the cell's cars are all accounted
    for. Nothing is inferred: a car no filter returned keeps no body at all.
    """
    filters = autoscout.market(tld).body_filters
    if not filters:
        return
    waiting = {str(row.get("external_id")): row for row in rows}
    for slug in body_order(bodies, filters):
        if not waiting or result.enrich_requests >= ceiling or not _budget_left(client):
            break
        before = client.spent
        listings, meta = client.search(make_slug, model_slug, year=year, body=slug,
                                       ustate="U", sort="age", desc=True)
        result.enrich_requests += client.spent - before
        if not meta:
            continue
        body = filters.get(slug)
        for item in used_only(listings):
            row = waiting.pop(str(item.external_id), None)
            if row is not None and body:
                row["body_type"] = body
                result.bodies_labelled += 1


def probe_inventory(client, cfg: MarketConfig, state: CrawlState, cty: Country, session, *,
                    now: datetime, result: CountryResult, adverts: bool = True,
                    ceiling: int | None = None, log=print) -> None:
    """One newest-first page per (make, model) not probed inside a month.

    The page is asked for its result count, but its twenty cars are real and
    are stored, so their adverts are opened here too. Without that, a model
    under ``min_model_results`` — one this pass will never walk year by year —
    would keep its rows card-shallow for as long as it stays small, which is
    exactly the corner where nobody would think to look for the gap.
    """
    cstate = state.country(cty.code)
    ceiling = _ceiling(client, ceiling)
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
                rows = [asdict(item) for item in listings]
                if adverts:
                    deepen(client, rows, result, session=session, cty=cty,
                           make_slug=make, model_slug=model["slug"], year=None,
                           bodies=meta.get("body_types"), ceiling=ceiling)
                ins, upd = upsert_import_listings(session, rows)
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
    from src.models.listing import Listing

    rows = (session.query(Listing.brand, Listing.model, Listing.year,
                          func.max(Listing.last_seen_at))
            .filter(Listing.source == source)
            .group_by(Listing.brand, Listing.model, Listing.year)
            .all())
    return {(b, m, y): seen for b, m, y, seen in rows if seen is not None}


def harvest(client, cfg: MarketConfig, cells: list[Cell], state: CrawlState, cty: Country,
            session, *, pages_per_cell: int, result: CountryResult, adverts: bool = True,
            ceiling: int | None = None, log=print) -> None:
    """Read each cell's newest pages, store them, retire what a full read no longer lists."""
    ceiling = _ceiling(client, ceiling)
    for index, cell in enumerate(cells, start=1):
        if not _budget_left(client):
            break
        batch = []
        pages_fetched = 0
        results = None
        total_pages = 1
        bodies = None
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
                bodies = meta.get("body_types")
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
            rows = [asdict(item) for item in batch]
            if adverts:
                deepen(client, rows, result, session=session, cty=cty,
                       make_slug=cell.make_slug, model_slug=cell.model_slug,
                       year=cell.year, bodies=bodies, ceiling=ceiling)
            ins, upd = upsert_import_listings(session, rows)
            result.inserted += ins
            result.updated += upd
        if results is not None and int(results) <= PAGE_SIZE * pages_fetched:
            result.deactivated += deactivate_import_missing(
                session, cty.source, [cell.key], seen_ids)
        if index % STATE_FLUSH_EVERY == 0:
            state.save()


def crawl_country(cty: Country, cfg: MarketConfig, state: CrawlState, client, session, *,
                  now: datetime | None = None, pages_per_cell: int | None = None,
                  years_back: int | None = None, adverts: bool = True,
                  log=print) -> CountryResult:
    """The three passes for one country, then the age-based expiry."""
    now = now or _utcnow()
    pages = pages_per_cell if pages_per_cell is not None else cfg.pages_per_cell
    years = years_back if years_back is not None else cfg.years_back
    result = CountryResult(code=cty.code, source=cty.source)
    t0 = time.perf_counter()
    try:
        ceiling = int(client.config.budget * cfg.enrich_share)
        result.makes_read, result.makes_empty = discover(client, cfg, state, cty.code,
                                                         now=now, log=log)
        probe_inventory(client, cfg, state, cty, session, now=now, result=result,
                        adverts=adverts, ceiling=ceiling, log=log)
        kept = kept_models(cfg, state, cty.code)
        result.models_kept = len(kept)
        cells = order_cells(build_cells(kept, now_year=now.year, years_back=years),
                            cell_last_seen(session, cty.source),
                            now=now, cell_max_age_days=cfg.cell_max_age_days)
        result.cells_pending = len(cells)
        harvest(client, cfg, cells, state, cty, session, pages_per_cell=pages,
                result=result, adverts=adverts, ceiling=ceiling, log=log)
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
            f"{result.adverts_read} adverts read "
            f"({result.adverts_missed} not read), "
            f"{result.bodies_labelled} bodies from the filter, "
            f"{result.deactivated} retired, {result.expired} expired, "
            f"{result.requests} requests in {result.seconds:.0f}s"
            + (" — blocked" if result.blocked else ""))


def run(countries: list[Country], configs: dict[str, MarketConfig], state: CrawlState, *,
        client_factory, session_factory, pages_per_cell: int | None = None,
        years_back: int | None = None, adverts: bool = True,
        log=print) -> list[CountryResult]:
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
                                                  years_back=years_back,
                                                  adverts=adverts, log=log)
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
    ap.add_argument("--no-adverts", action="store_true",
                    help="Read search cards only, with no deepening at all: no "
                         "advert opened where robots allows it, and no body "
                         "filter asked where it does not. Cheap and shallow - "
                         "no body type, no CO2, no colour, no description.")
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
                  years_back=args.years, adverts=not args.no_adverts)
    for result in results:
        print(summary(result), flush=True)
    if any(r.blocked for r in results):
        return 2
    return 1 if any(r.error for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
