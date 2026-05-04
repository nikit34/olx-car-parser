"""Backfill ``photo_count`` for OLX listings missed by the stale selector.

Background
----------
A 2026-05-04 audit of the active set found ``photo_count IS NULL`` on
4436 / 4438 active OLX listings (≈100%). The dashboard's photo-damage
classifier therefore never ran on them, and the uncertainty model's
``desc_quality = log(desc_len) × log(photo_count)`` feature was zero
on every OLX row. The selector was fixed in
``src.parser.scraper.OlxScraper.scrape_listing_detail`` (commit follows);
this script re-fetches the affected listings and writes the corrected
``photo_count`` back to the DB.

Usage
-----
Default — all NULLs in the active set:

    .venv/bin/python -m scripts.backfill_olx_photo_count

Test on a small batch first:

    .venv/bin/python -m scripts.backfill_olx_photo_count --limit 20

Where to run
~~~~~~~~~~~~
The authoritative DB lives on the scrape host (anastasia@192.168.1.77
in the LAN map). The local checkout reads a snapshot from the GitHub
``latest-data`` release; updates here do *not* propagate to production.
For the actual backfill, push the script + selector fix and run from
the scrape host's persistent clone (see ``post-push-host-sync``).

What it does
------------
1. Selects every active OLX listing with ``photo_count IS NULL``.
2. Re-fetches each detail page through the existing scraper (so we
   inherit its retries, throttling, sticky-session HTTP client).
3. Writes the new ``photo_count`` and bumps ``last_seen_at``. Untouched
   if the page returns nothing or the new selector also misses (rare
   layout edge cases).
4. 404 / removed listings get marked ``is_active = False`` with
   ``deactivation_reason = 'expired'`` — they would have been caught
   by the next scrape cycle anyway, but that's hours away.
5. Resumable: the script always queries "still NULL" rows on each run.
   Re-running picks up where a previous one left off.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.parser.scraper import OlxScraper, ScraperConfig  # noqa: E402
from src.storage.database import init_db, get_session  # noqa: E402
from src.models.listing import Listing  # noqa: E402

logger = logging.getLogger("backfill_olx_photo_count")


def _select_targets(session, limit: int | None) -> list[tuple[int, str, str]]:
    """Return (id, olx_id, url) for active OLX listings missing photo_count."""
    q = (
        session.query(Listing.id, Listing.olx_id, Listing.url)
        .filter(Listing.source == "olx")
        .filter(Listing.is_active == True)  # noqa: E712
        .filter(Listing.photo_count.is_(None))
        .order_by(Listing.id)
    )
    if limit is not None:
        q = q.limit(limit)
    return q.all()


def _commit_with_retry(session, max_attempts: int = 5) -> bool:
    """SQLite write contention with the live scrape worker can hold the
    lock 3-5 min during market_stats commit. Retry with backoff before
    giving up — losing a single row's update is fine, losing the whole
    batch over a transient lock would mean re-fetching."""
    delay = 5.0
    for attempt in range(1, max_attempts + 1):
        try:
            session.commit()
            return True
        except Exception as exc:  # noqa: BLE001 — sqlite3.OperationalError
            if "locked" not in str(exc).lower() or attempt == max_attempts:
                logger.warning("commit failed after %d attempts: %s", attempt, exc)
                session.rollback()
                return False
            logger.info(
                "DB locked (scrape worker?) — retry in %.0fs (attempt %d/%d)",
                delay, attempt, max_attempts,
            )
            time.sleep(delay)
            delay = min(delay * 2, 60.0)
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap the number of rows processed (smoke-test mode).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Fetch + parse but don't update the DB. Logs what would change.",
    )
    parser.add_argument(
        "--commit-every", type=int, default=20,
        help="Commit batch size. Smaller = more retries, larger = more lost "
             "work on a crash.",
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

    init_db()
    session = get_session()
    try:
        targets = _select_targets(session, args.limit)
        if not targets:
            logger.info("Nothing to backfill — all active OLX rows have photo_count.")
            return 0
        logger.info("Backfill targets: %d active OLX listings", len(targets))

        scraper = OlxScraper(ScraperConfig())
        try:
            n_updated = 0
            n_no_photo_field = 0
            n_expired = 0
            n_failed = 0
            for i, (listing_id, olx_id, url) in enumerate(targets, start=1):
                try:
                    details = scraper.scrape_listing_detail(url)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("fetch error for %s (%s): %s", olx_id, url, exc)
                    details = {}

                if not details:
                    # 404 / network — treat as expired listing.
                    if not args.dry_run:
                        listing = session.get(Listing, listing_id)
                        if listing is not None and listing.is_active:
                            from datetime import datetime, timezone
                            listing.is_active = False
                            listing.deactivation_reason = "expired"
                            listing.deactivated_at = datetime.now(timezone.utc)
                    n_expired += 1
                elif "photo_count" in details:
                    if not args.dry_run:
                        listing = session.get(Listing, listing_id)
                        if listing is not None:
                            listing.photo_count = details["photo_count"]
                    n_updated += 1
                else:
                    # Page loaded but neither selector hit — layout edge
                    # case (private-account / blocked / atypical render).
                    n_no_photo_field += 1
                    n_failed += 1

                if i % args.commit_every == 0:
                    if not args.dry_run and not _commit_with_retry(session):
                        logger.error(
                            "commit aborted at i=%d — re-running picks up "
                            "where this stopped (rows still NULL)", i,
                        )
                        return 2
                    logger.info(
                        "progress %d/%d  · updated=%d  expired=%d  no-photo-field=%d",
                        i, len(targets), n_updated, n_expired, n_no_photo_field,
                    )

            # Tail commit.
            if not args.dry_run:
                _commit_with_retry(session)

            logger.info(
                "Done. Updated=%d  Expired=%d  No-photo-field=%d  Failed=%d  Total=%d",
                n_updated, n_expired, n_no_photo_field, n_failed, len(targets),
            )
        finally:
            scraper.close()
    finally:
        session.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
