"""The markets this project reads, and the one name each of them answers to.

Portugal is the original corpus and comes off OLX; Germany, France and Italy
are read from AutoScout24's national sites. Everything downstream needs to
agree on three things per market — the ISO code the pages and the blobs are
keyed by, the ``source`` string its rows carry in ``import_listings``, and the
AutoScout24 host and country filter the crawler has to use — and when those
three live in three different modules they drift. A German row written under
``as24_de`` but read back under ``autoscout24_de`` is not a bug that shows up
in a test; it is a country that quietly has no cars.

So the mapping lives here once, keyed by ISO code, and everything else asks:
``repository.get_country_listings_df`` for the source of a code, the crawler
for the tld and the ``cy`` filter, the blob builder for the list of markets.

A country is no longer one source. Portugal was always two platforms, and
Germany and Italy are now two each; ``PLATFORMS`` is the full list and
``sources_for`` is what a per-country read filters on. It is also how a row
gets its ``country_code``: the writer resolves the market from the source
rather than trusting a reader to remember, because a row that arrives without
a market reads as Portuguese and a German price inside a Portuguese median is
a bug that raises nothing.

Portugal is in the table for completeness — it is a country this project has
a corpus for — but it is not in ``EU_COUNTRIES``: that tuple is the set of
markets the AutoScout24 crawler and the per-country model pipeline run over,
and Portugal is the home market, read by a different scraper into a different
table under different rules.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Country:
    """One market: its code, its language, and where its listings come from.

    ``source`` is the value that lands in ``import_listings.source``; for
    Portugal it is ``olx``, which is not an import source at all and exists so
    that code asking "which source is this country" never has to special-case
    the home market with a ``None``.

    ``as24_tld`` and ``as24_cy`` are the two halves of an AutoScout24 request:
    the national host (``autoscout24.de``) and the ``cy`` query parameter the
    site filters sellers by (``D``, ``F``, ``I`` — AutoScout24's own codes,
    which are not the ISO ones). Both are None for a market AutoScout24 is not
    read for.
    """

    code: str
    lang: str
    name: str
    source: str
    as24_tld: str | None
    as24_cy: str | None
    market_host: str


COUNTRIES: dict[str, Country] = {
    "PT": Country(code="PT", lang="pt", name="Portugal", source="olx",
                  as24_tld=None, as24_cy=None, market_host="olx.pt"),
    "DE": Country(code="DE", lang="de", name="Deutschland", source="as24_de",
                  as24_tld="de", as24_cy="D", market_host="autoscout24.de"),
    "FR": Country(code="FR", lang="fr", name="France", source="as24_fr",
                  as24_tld="fr", as24_cy="F", market_host="autoscout24.fr"),
    "IT": Country(code="IT", lang="it", name="Italia", source="as24_it",
                  as24_tld="it", as24_cy="I", market_host="autoscout24.it"),
}

@dataclass(frozen=True)
class Platform:
    """One site we read, and the market its cars are for sale in.

    ``host`` is what the reader requests and what the public pages credit, so
    it stays a bare hostname rather than a URL. ``legacy`` marks a source
    string that is still in the database but is not a reader any more — the
    weekly German benchmark predates per-country sources and its rows must
    still resolve to a country.
    """

    source: str
    code: str
    name: str
    host: str
    legacy: bool = False


PLATFORMS: dict[str, Platform] = {
    "olx": Platform("olx", "PT", "OLX", "olx.pt"),
    "standvirtual": Platform("standvirtual", "PT", "StandVirtual", "standvirtual.com"),
    "as24_de": Platform("as24_de", "DE", "AutoScout24", "autoscout24.de"),
    "as24_fr": Platform("as24_fr", "FR", "AutoScout24", "autoscout24.fr"),
    "as24_it": Platform("as24_it", "IT", "AutoScout24", "autoscout24.it"),
    "ka_de": Platform("ka_de", "DE", "Kleinanzeigen", "kleinanzeigen.de"),
    "am_it": Platform("am_it", "IT", "automobile.it", "automobile.it"),
    "autoscout24": Platform("autoscout24", "DE", "AutoScout24", "autoscout24.de",
                            legacy=True),
}

EU_COUNTRIES: tuple[str, ...] = ("DE", "FR", "IT")

_BY_SOURCE: dict[str, Country] = {c.source: c for c in COUNTRIES.values()}


def platform(source: str) -> Platform | None:
    """The platform a source string names, or None for one we do not read."""
    return PLATFORMS.get(str(source or "").strip())


def platforms_for(code: str) -> tuple[Platform, ...]:
    """Every platform serving a market, including retired source strings."""
    key = country(code).code
    return tuple(p for p in PLATFORMS.values() if p.code == key)


def sources_for(code: str) -> tuple[str, ...]:
    """Every ``listings.source`` value that belongs to a market."""
    return tuple(p.source for p in platforms_for(code))


def code_for_source(source: str) -> str | None:
    """The ISO market code a source belongs to, or None if it is unknown.

    This is what the writer stamps rows with. None is a refusal rather than a
    default: a source nobody registered has no market, and guessing one would
    put a foreign car in whichever corpus the guess named.
    """
    known = PLATFORMS.get(str(source or "").strip())
    return known.code if known else None


def country(code: str) -> Country:
    """The market for an ISO code, case-insensitively. Unknown codes raise.

    Raising rather than returning None is deliberate: every caller here turns
    the answer into a source string or a host, and a silent None would become
    a query against source ``None`` that finds nothing and reports no error.
    """
    key = str(code or "").strip().upper()
    try:
        return COUNTRIES[key]
    except KeyError:
        raise ValueError(f"unknown country code: {code!r} "
                         f"(known: {', '.join(sorted(COUNTRIES))})") from None


def source_for(code: str) -> str:
    """The ``import_listings.source`` value that belongs to a country code."""
    return country(code).source


def country_for_source(source: str) -> Country | None:
    """The market a source string belongs to, or None for an unknown source.

    None here rather than a raise: sources arrive from the database, where
    anything a future reader writes is a legitimate value that is simply not a
    registered platform yet.
    """
    key = str(source or "").strip()
    known = PLATFORMS.get(key)
    if known is not None:
        return COUNTRIES.get(known.code)
    return _BY_SOURCE.get(key)
