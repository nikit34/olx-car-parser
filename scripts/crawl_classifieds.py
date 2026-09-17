"""Read automobile.it and Kleinanzeigen, advert by advert, into ``listings``.

Separate from ``crawl_eu_market`` on purpose. That script is AutoScout24's, and
its shape — discover makes, build one cell per make/model/year, enumerate a cell
fully so that "not on the page" can mean "sold" — is a shape AutoScout24 allows
and these two sites do not. automobile.it caps a query at five hundred pages and
Kleinanzeigen at five, so neither can be enumerated; what a pass here holds is a
refreshing sample, and nothing in it may be read as a sale.

What the two do share is the sequence, and it is the expensive one: read a page
of cards, then open every advert on it. The advert is where the fields that make
a row comparable to a Portuguese one live — fuel, gearbox, colour, the seller's
own text — and without them a row is a price with a year on it. So one car costs
one request, and the budget is the only thing standing between that and a flood.
It is counted in requests rather than cars, checked before every fetch, and a
403 or 429 ends the run for that platform then and there.

Rows are written in batches as they are read rather than at the end, so a run cut
short by a budget, a block or a laptop lid keeps what it had already read.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.parser import automobile_it as am  # noqa: E402
from src.parser import kleinanzeigen_de as ka  # noqa: E402
from src.parser.market_card import merge_patch  # noqa: E402
from src.storage.database import get_session, init_db  # noqa: E402
from src.storage.repository import upsert_import_listings  # noqa: E402

BATCH = 40


@dataclass
class Reader:
    """One platform, reduced to what the runner needs to drive it."""

    name: str
    source: str
    host: str
    accept_language: str
    module: object
    max_page: int
    brands: tuple[str, ...] = ()
    needs_advert: bool = False


@dataclass
class Result:
    source: str
    requests: int = 0
    cards: int = 0
    adverts: int = 0
    inserted: int = 0
    updated: int = 0
    dropped: int = 0
    stopped: str | None = None
    errors: list[str] = field(default_factory=list)


class Budget:
    """A hard ceiling on requests, shared by every fetch in one platform's run."""

    def __init__(self, total: int):
        self.left = int(total)

    def spend(self) -> bool:
        if self.left <= 0:
            return False
        self.left -= 1
        return True


def _sleep(reader: Reader) -> None:
    time.sleep(random.uniform(reader.module.DELAY_MIN, reader.module.DELAY_MAX))


def _fetch(client: httpx.Client, reader: Reader, path: str, budget: Budget,
           result: Result) -> str | None:
    """One polite GET, or None when the budget, robots or the site says no."""
    if not reader.module.robots_allows(path):
        return None
    if not budget.spend():
        result.stopped = "budget"
        return None
    response = client.get(f"https://{reader.host}{path}")
    result.requests += 1
    if response.status_code in (403, 429):
        result.stopped = f"HTTP {response.status_code}"
        return None
    if response.status_code != 200:
        result.errors.append(f"{path}: HTTP {response.status_code}")
        return None
    return response.text


def _flush(session, rows: list[dict], result: Result) -> None:
    if not rows:
        return
    inserted, updated = upsert_import_listings(session, rows)
    result.inserted += inserted
    result.updated += updated
    rows.clear()


def crawl(reader: Reader, session, budget: Budget, *, brands: int,
          pages: int, log=print) -> Result:
    """Walk a platform's makes, reading each advert, until the budget runs out."""
    result = Result(source=reader.source)
    pending: list[dict] = []
    client = httpx.Client(
        headers={"User-Agent": reader.module.USER_AGENT,
                 "Accept-Language": reader.accept_language},
        timeout=reader.module.TIMEOUT, follow_redirects=True)
    try:
        for brand in list(reader.brands)[:brands]:
            for page in range(1, min(pages, reader.max_page) + 1):
                if result.stopped:
                    return result
                html = _fetch(client, reader, reader.module.search_path(brand, page),
                              budget, result)
                if html is None:
                    if result.stopped:
                        return result
                    break
                _sleep(reader)
                try:
                    cards, _meta = reader.module.parse_search(html, brand)
                except Exception as exc:
                    result.errors.append(f"{brand} p{page}: {exc}")
                    break
                if not cards:
                    break
                result.cards += len(cards)

                for card in cards:
                    if result.stopped:
                        _flush(session, pending, result)
                        return result
                    row = card.as_row()
                    path = card.url.split(reader.host, 1)[-1] or "/"
                    advert = _fetch(client, reader, path, budget, result)
                    if advert is None:
                        if result.stopped:
                            if not reader.needs_advert:
                                pending.append(row)
                            _flush(session, pending, result)
                            return result
                        if reader.needs_advert:
                            result.dropped += 1
                            continue
                    else:
                        _sleep(reader)
                        try:
                            merge_patch(row, reader.module.parse_detail(advert),
                                        getattr(reader.module, "CORRECTS", ()))
                            result.adverts += 1
                        except Exception as exc:
                            result.errors.append(f"{card.external_id}: {exc}")
                            if reader.needs_advert:
                                result.dropped += 1
                                continue
                    pending.append(row)
                    if len(pending) >= BATCH:
                        _flush(session, pending, result)
                log(f"[{reader.source}] {brand} p{page}: "
                    f"{len(cards)} cards, {budget.left} requests left", flush=True)
        _flush(session, pending, result)
        return result
    finally:
        _flush(session, pending, result)
        client.close()


def discover_italian_brands(client: httpx.Client) -> tuple[str, ...]:
    """automobile.it publishes its make list on the search page; read it there."""
    response = client.get(f"https://{am.HOST}/annunci")
    if response.status_code != 200:
        return ()
    return tuple(am.brand_slugs(response.text))


GERMAN_BRANDS: tuple[str, ...] = (
    "volkswagen", "bmw", "mercedes-benz", "audi", "opel", "ford", "skoda",
    "renault", "seat", "toyota", "fiat", "hyundai", "kia", "peugeot", "mazda",
    "nissan", "volvo", "citroen", "dacia", "mini",
)


def readers(italian_brands: tuple[str, ...]) -> list[Reader]:
    return [
        Reader(name="automobile.it", source=am.SOURCE, host=am.HOST,
               accept_language=am.ACCEPT_LANGUAGE, module=am,
               max_page=am.MAX_PAGES, brands=italian_brands),
        Reader(name="Kleinanzeigen", source=ka.SOURCE, host=ka.HOST,
               accept_language=ka.ACCEPT_LANGUAGE, module=ka,
               max_page=ka.MAX_PAGE, brands=GERMAN_BRANDS, needs_advert=True),
    ]


def summary(result: Result) -> str:
    head = (f"[{result.source}] {result.requests} requests, {result.cards} cards, "
            f"{result.adverts} adverts read, "
            f"{result.inserted} new, {result.updated} refreshed")
    if result.dropped:
        head += f", {result.dropped} dropped without an advert"
    if result.stopped:
        head += f" — stopped: {result.stopped}"
    if result.errors:
        head += f" ({len(result.errors)} errors, first: {result.errors[0]})"
    return head


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append",
                        choices=[am.SOURCE, ka.SOURCE],
                        help="Platform to read; repeatable. Default: both.")
    parser.add_argument("--budget", type=int, default=400,
                        help="Hard ceiling on HTTP requests per platform.")
    parser.add_argument("--brands", type=int, default=6,
                        help="How many makes to walk this pass.")
    parser.add_argument("--pages", type=int, default=2,
                        help="Search pages per make, within the site's own ceiling.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Read one search page per platform and print what it parsed.")
    args = parser.parse_args(argv)

    with httpx.Client(headers={"User-Agent": am.USER_AGENT},
                      timeout=am.TIMEOUT, follow_redirects=True) as client:
        italian = discover_italian_brands(client)
    if not italian:
        print("automobile.it did not answer with a make list; using a short default",
              file=sys.stderr)
        italian = ("fiat", "volkswagen", "audi", "bmw", "ford", "opel")

    wanted = set(args.source or [am.SOURCE, ka.SOURCE])
    chosen = [r for r in readers(italian) if r.source in wanted]

    if args.dry_run:
        for reader in chosen:
            with httpx.Client(
                    headers={"User-Agent": reader.module.USER_AGENT,
                             "Accept-Language": reader.accept_language},
                    timeout=reader.module.TIMEOUT, follow_redirects=True) as client:
                brand = reader.brands[0] if reader.brands else ""
                path = reader.module.search_path(brand, 1)
                response = client.get(f"https://{reader.host}{path}")
                cards, meta = reader.module.parse_search(response.text, brand)
                print(f"[{reader.source}] {path} -> {len(cards)} cards {meta}")
                for card in cards[:3]:
                    print(f"   {card.brand} {card.model} {card.year} "
                          f"{card.price_eur} {card.mileage_km}km {card.city}")
        return 0

    init_db()
    session = get_session()
    try:
        for reader in chosen:
            result = crawl(reader, session, Budget(args.budget),
                           brands=args.brands, pages=args.pages)
            print(summary(result), flush=True)
    finally:
        session.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
