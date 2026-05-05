"""Backfill the ``sellers`` table from listings' ``seller_profile_url``.

Background
----------
``scrape_listing_detail`` records the seller's profile URL on every
listing it ingests, but linking that listing to a canonical ``Seller``
row needs a second HTTP request — the profile page carries the
``__PRERENDERED_STATE__`` blob with the seller's UUID, registration
date, total ad count and per-category facets. We fetch profiles
asynchronously rather than inline during scrape so a multi-car seller
hits the network once instead of N times per scrape cycle.

Usage
-----
First-pass backfill (everything missing a seller_uuid):

    .venv/bin/python -m scripts.backfill_sellers

Refresh stale snapshots (>14 days old):

    .venv/bin/python -m scripts.backfill_sellers --ttl-days 14

Smoke test:

    .venv/bin/python -m scripts.backfill_sellers --limit 20 --dry-run

What it does
------------
1. Collects distinct ``seller_profile_url`` values from ``listings``
   that either don't yet have a ``seller_uuid`` or whose linked seller
   row's ``profile_fetched_at`` is older than ``--ttl-days``.
2. For each URL, fetches the profile page through ``OlxScraper`` (so we
   inherit its throttle, UA rotation, and 403 cascade).
3. Upserts the parsed :class:`SellerProfile` into the ``sellers`` table
   plus derived feature buckets via :func:`categorise_facets`.
4. Updates every listing pointing at that URL to set ``seller_uuid``.

Concurrency / DB locking
~~~~~~~~~~~~~~~~~~~~~~~~
Same recipe as ``backfill_olx_photo_count``: a single raw
``sqlite3.Connection`` with ``timeout=30s`` and ``isolation_level=None``,
so each statement runs in its own transaction and waits politely for
the live scrape worker's market-stats commit (which can hold the write
lock for 3-5 minutes).

Resumable
~~~~~~~~~
The selector queries "still missing or stale" on every run, so an
interrupted backfill picks up where it left off on the next invocation.
"""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.parser.olx_categories import categorise_facets  # noqa: E402
from src.parser.scraper import OlxScraper, ScraperConfig  # noqa: E402
from src.parser.seller_profile import SellerProfile  # noqa: E402

DB_PATH = _REPO_ROOT / "data" / "olx_cars.db"

logger = logging.getLogger("backfill_sellers")


def _open_db(timeout_s: float = 30.0) -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), timeout=timeout_s, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _utcnow_str() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat(sep=" ")


def _select_targets(
    conn: sqlite3.Connection,
    ttl_days: int,
    limit: int | None,
) -> list[str]:
    """Return distinct seller_profile_urls needing a (re)fetch.

    A URL is a target when at least one of the following holds:

    * No listing pointing at it has had ``seller_uuid`` resolved yet
      (first-pass backfill).
    * The seller row it would link to has ``profile_fetched_at`` older
      than ``ttl_days`` (refresh — total_ads / facets drift over time).

    De-dup at the SQL layer so a 256-ad dealer turns into one fetch,
    not 256.
    """
    cutoff = (
        datetime.now(timezone.utc).replace(tzinfo=None)
        - timedelta(days=ttl_days)
    ).isoformat(sep=" ")
    sql = """
        SELECT DISTINCT l.seller_profile_url
        FROM listings l
        LEFT JOIN sellers s ON s.uuid = l.seller_uuid
        WHERE l.seller_profile_url IS NOT NULL
          AND (
            l.seller_uuid IS NULL
            OR s.profile_fetched_at IS NULL
            OR s.profile_fetched_at < ?
          )
        ORDER BY l.seller_profile_url
    """
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    return [row[0] for row in conn.execute(sql, (cutoff,)).fetchall()]


def _upsert_seller(
    conn: sqlite3.Connection,
    profile: SellerProfile,
) -> None:
    """Insert or update a sellers row keyed by uuid.

    Stores the raw category facets as JSON for diagnostics and the
    derived feature buckets as separate columns so dashboards can read
    them without re-rolling-up on every query.
    """
    facets_list = [{"id": k, "count": v} for k, v in profile.facets.items()]
    derived = categorise_facets(facets_list, categories_list=profile.categories_list)
    facets_json = json.dumps(profile.facets, ensure_ascii=False)
    now = _utcnow_str()
    fields = {
        "uuid": profile.uuid,
        "short_id": profile.short_id,
        "shop_slug": profile.shop_slug,
        "profile_url": profile.profile_url,
        "name": profile.name,
        "is_business": int(profile.is_business) if profile.is_business is not None else None,
        "business_type": profile.business_type,
        "created_at": profile.created_at.isoformat(sep=" ") if profile.created_at else None,
        "last_seen_at": profile.last_seen_at.isoformat(sep=" ") if profile.last_seen_at else None,
        "last_login_at": profile.last_login_at.isoformat(sep=" ") if profile.last_login_at else None,
        "total_ads": profile.total_ads,
        "ads_by_category": facets_json,
        "cars_count": derived["cars"],
        "parts_count": derived["parts"],
        "commercial_count": derived["commercial"],
        "motos_count": derived["motos"],
        "boats_count": derived["boats"],
        "other_auto_count": derived["other_auto"],
        "non_auto_count": derived["non_auto"],
        "distinct_car_brands": derived["distinct_car_brands"],
        "family_lifestyle_count": derived["family_lifestyle"],
        "electronics_count": derived["electronics"],
        "realestate_count": derived["realestate"],
        "tools_industrial_count": derived["tools_industrial"],
        "pets_hobby_count": derived["pets_hobby"],
        "services_jobs_count": derived["services_jobs"],
        "social_account_type": profile.social_account_type,
        "has_user_photo": int(profile.has_user_photo) if profile.has_user_photo is not None else None,
        "position_lat": profile.position_lat,
        "position_lon": profile.position_lon,
        "profile_fetched_at": now,
    }
    cols = ", ".join(fields.keys())
    placeholders = ", ".join("?" for _ in fields)
    updates = ", ".join(f"{k}=excluded.{k}" for k in fields if k != "uuid")
    conn.execute(
        f"INSERT INTO sellers ({cols}) VALUES ({placeholders}) "
        f"ON CONFLICT(uuid) DO UPDATE SET {updates}",
        tuple(fields.values()),
    )


def _link_listings(conn: sqlite3.Connection, profile_url: str, uuid: str) -> int:
    """Set seller_uuid on every listing that points at *profile_url*.

    A seller can have multiple listings; one fetch resolves all of
    them. We don't touch listings that already carry a different
    seller_uuid — that would clobber a prior link if the listing's URL
    happened to change format (private → business shop subdomain),
    which the dedicated refresh path is meant to handle, not this one.
    """
    cursor = conn.execute(
        "UPDATE listings SET seller_uuid = ? "
        "WHERE seller_profile_url = ? AND seller_uuid IS NULL",
        (uuid, profile_url),
    )
    return cursor.rowcount or 0


def _process_one(
    conn: sqlite3.Connection,
    scraper: OlxScraper,
    url: str,
    dry_run: bool,
) -> tuple[str, int]:
    """Fetch + parse + persist one seller. Returns (status, listings_linked).

    Status values:
      * ``ok``       — row upserted, listings linked
      * ``skip``     — page fetched but profile blob missing/malformed
      * ``fetch_err``— HTTP fetch returned None (404 / network)
      * ``write_err``— DB write blocked beyond busy_timeout
    """
    try:
        profile = scraper.scrape_seller_profile(url)
    except Exception as exc:  # noqa: BLE001
        logger.warning("fetch raised for %s: %s", url, exc)
        return "fetch_err", 0
    if profile is None:
        return "fetch_err", 0
    if not profile.uuid:
        return "skip", 0
    if dry_run:
        return "ok", 0
    try:
        _upsert_seller(conn, profile)
        linked = _link_listings(conn, url, profile.uuid)
    except sqlite3.OperationalError as exc:
        logger.warning("write blocked for %s: %s", url, exc)
        return "write_err", 0
    return "ok", linked


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ttl-days", type=int, default=14,
        help="Refresh sellers whose snapshot is older than N days (default 14).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap the number of sellers processed (smoke-test mode).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Fetch + parse but skip DB writes. Useful for selector audits.",
    )
    parser.add_argument(
        "--progress-every", type=int, default=20,
        help="Log a progress line every N sellers.",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    conn = _open_db()
    try:
        targets = _select_targets(conn, args.ttl_days, args.limit)
        if not targets:
            logger.info("Nothing to backfill — no listings need a seller fetch.")
            return 0
        logger.info("Backfill targets: %d distinct seller URLs", len(targets))

        scraper = OlxScraper(ScraperConfig())
        try:
            counts = {"ok": 0, "skip": 0, "fetch_err": 0, "write_err": 0}
            total_linked = 0
            for i, url in enumerate(targets, start=1):
                status, linked = _process_one(conn, scraper, url, args.dry_run)
                counts[status] += 1
                total_linked += linked
                if i % args.progress_every == 0:
                    logger.info(
                        "progress %d/%d  · ok=%d skip=%d fetch_err=%d "
                        "write_err=%d  listings_linked=%d",
                        i, len(targets), counts["ok"], counts["skip"],
                        counts["fetch_err"], counts["write_err"], total_linked,
                    )
            logger.info(
                "Done. ok=%d skip=%d fetch_err=%d write_err=%d  "
                "listings_linked=%d  total=%d",
                counts["ok"], counts["skip"], counts["fetch_err"],
                counts["write_err"], total_linked, len(targets),
            )
        finally:
            scraper.close()
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
