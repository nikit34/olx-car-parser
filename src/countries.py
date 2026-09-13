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

EU_COUNTRIES: tuple[str, ...] = ("DE", "FR", "IT")

_BY_SOURCE: dict[str, Country] = {c.source: c for c in COUNTRIES.values()}


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

    None here rather than a raise: sources arrive from the database, where the
    old German benchmark crawl (``autoscout24``) and anything a future reader
    writes are legitimate values that simply are not a country corpus.
    """
    return _BY_SOURCE.get(str(source or "").strip())
