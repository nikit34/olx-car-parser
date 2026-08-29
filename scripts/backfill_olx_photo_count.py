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

Concurrency
~~~~~~~~~~~
This script never opens an ORM session: it holds one connection from
``src.storage.database`` in statement-autocommit mode, so each row lands
in its own transaction and the pending-write window stays milliseconds
long. Readers never block a writer, and a row contended by the scrape
worker waits only for that transaction's commit.

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
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from sqlalchemy import DateTime, bindparam, text as sa_text  # noqa: E402
from sqlalchemy.exc import DBAPIError, OperationalError  # noqa: E402

from src.storage.database import get_engine  # noqa: E402
from src.parser.scraper import OlxScraper, ScraperConfig  # noqa: E402

logger = logging.getLogger("backfill_olx_photo_count")


def _open_db():
    """One statement-autocommitting connection on the configured engine.

    Pooling lives in ``src.storage.database``; a row contended by the
    live scrape worker waits for its commit, not for a database-wide lock.
    """
    engine = get_engine()
    return engine.connect().execution_options(isolation_level="AUTOCOMMIT")


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _select_targets(conn, limit: int | None) -> list[tuple[int, str, str]]:
    """Return (id, olx_id, url) for active OLX listings missing photo_count."""
    sql = (
        "SELECT id, olx_id, url FROM listings "
        "WHERE source='olx' AND is_active = TRUE AND photo_count IS NULL "
        "ORDER BY id"
    )
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    return [tuple(row) for row in conn.execute(sa_text(sql)).fetchall()]


def _update_photo_count(conn, listing_id: int, photo_count: int) -> bool:
    """Single-row UPDATE under statement autocommit. True on success.

    Logs a driver-level failure without aborting the whole run."""
    try:
        conn.execute(
            sa_text(
                "UPDATE listings SET photo_count=:photo_count, "
                "last_seen_at=:now WHERE id=:listing_id"
            ).bindparams(bindparam("now", type_=DateTime())),
            {"photo_count": photo_count, "now": _now(), "listing_id": listing_id},
        )
        return True
    except (OperationalError, DBAPIError) as exc:
        logger.warning("UPDATE locked for id=%d: %s", listing_id, exc)
        return False


def _mark_expired(conn, listing_id: int) -> bool:
    try:
        conn.execute(
            sa_text(
                "UPDATE listings SET is_active = FALSE, "
                "deactivation_reason='expired', deactivated_at=:now "
                "WHERE id=:listing_id AND is_active = TRUE"
            ).bindparams(bindparam("now", type_=DateTime())),
            {"now": _now(), "listing_id": listing_id},
        )
        return True
    except (OperationalError, DBAPIError) as exc:
        logger.warning("UPDATE-expired locked for id=%d: %s", listing_id, exc)
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
        "--progress-every", type=int, default=20,
        help="Log a progress line every N rows.",
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
        targets = _select_targets(conn, args.limit)
        if not targets:
            logger.info("Nothing to backfill — all active OLX rows have photo_count.")
            return 0
        logger.info("Backfill targets: %d active OLX listings", len(targets))

        scraper = OlxScraper(ScraperConfig())
        try:
            n_updated = 0
            n_no_photo_field = 0
            n_expired = 0
            n_failed_write = 0

            for i, (listing_id, olx_id, url) in enumerate(targets, start=1):
                try:
                    details = scraper.scrape_listing_detail(url)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("fetch error for %s (%s): %s", olx_id, url, exc)
                    details = {}

                if not details:
                    if args.dry_run:
                        n_expired += 1
                    elif _mark_expired(conn, listing_id):
                        n_expired += 1
                    else:
                        n_failed_write += 1
                elif "photo_count" in details:
                    pc = int(details["photo_count"])
                    if args.dry_run:
                        n_updated += 1
                    elif _update_photo_count(conn, listing_id, pc):
                        n_updated += 1
                    else:
                        n_failed_write += 1
                else:
                    # Page loaded but neither selector hit — layout edge
                    # case (private-account / blocked / atypical render).
                    n_no_photo_field += 1

                if i % args.progress_every == 0:
                    logger.info(
                        "progress %d/%d  · updated=%d  expired=%d  "
                        "no-photo-field=%d  failed-write=%d",
                        i, len(targets), n_updated, n_expired,
                        n_no_photo_field, n_failed_write,
                    )

            logger.info(
                "Done. Updated=%d  Expired=%d  No-photo-field=%d  "
                "Failed-write=%d  Total=%d",
                n_updated, n_expired, n_no_photo_field,
                n_failed_write, len(targets),
            )
        finally:
            scraper.close()
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
