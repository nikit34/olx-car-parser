"""Recovery for the 2026-05 source-blind mark_inactive bug.

Until this fix, ``mark_inactive`` deactivated every active row not present
in the just-scraped id set, regardless of source. So any cycle where one
source returned 0 (anti-bot, broken selector, network) swept that source's
entire active inventory to ``is_active=False, deactivation_reason='sold'``.

This script:

1. Prints a histogram of OLX deactivations per UTC day (so you can spot
   the sweep — it shows up as a single day with hundreds–thousands of
   "sold" stamps where steady state is a few dozen).
2. Optionally restores OLX rows marked sold on/after a chosen cutoff
   (clears ``is_active``, ``deactivated_at``, ``deactivation_reason``).

Restored rows are *candidates*, not confirmed-live. The next successful
scrape will either re-confirm them via ``upsert_listing`` (which sets
``is_active=True``) or the next ``mark_inactive`` will deactivate truly
sold ones again — but this time correctly, scoped to the source.

Run on the host that owns the authoritative DB (anastasia@…).
"""
from __future__ import annotations

import argparse
import logging
from collections import Counter
from datetime import datetime, timezone

from sqlalchemy import or_

from src.storage.database import init_db, get_session
from src.models.listing import Listing


def _parse_date(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=None)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source", default="olx", choices=["olx", "standvirtual"],
        help="Which source to inspect/restore (default: olx).",
    )
    p.add_argument(
        "--since", type=_parse_date, default=None,
        help="ISO date (YYYY-MM-DD). Restore rows with deactivated_at >= this. "
             "If omitted, only the histogram is shown.",
    )
    p.add_argument(
        "--apply", action="store_true",
        help="Write changes. Without this flag, --since runs in dry-run.",
    )
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("restore_falsely_deactivated")

    init_db()
    session = get_session()

    if args.source == "olx":
        src_filter = or_(Listing.source == "olx", Listing.source.is_(None))
    else:
        src_filter = Listing.source == args.source

    # Histogram of "sold" stamps per UTC day for the chosen source.
    rows = (
        session.query(Listing.deactivated_at)
        .filter(
            Listing.is_active == False,  # noqa: E712
            Listing.deactivation_reason == "sold",
            Listing.deactivated_at.isnot(None),
            src_filter,
        )
        .all()
    )
    by_day: Counter[str] = Counter()
    for (ts,) in rows:
        by_day[ts.date().isoformat()] += 1
    log.info("Deactivations-per-day for source=%s (top 30 days):", args.source)
    for day, n in sorted(by_day.items(), key=lambda kv: kv[1], reverse=True)[:30]:
        log.info("  %s  %5d", day, n)
    log.info("Total deactivated rows for %s: %d", args.source, sum(by_day.values()))

    if args.since is None:
        log.info("\nNo --since given — histogram only. To restore, re-run with "
                 "--since=YYYY-MM-DD (sweep day) and --apply.")
        return 0

    candidates = (
        session.query(Listing)
        .filter(
            Listing.is_active == False,  # noqa: E712
            Listing.deactivation_reason == "sold",
            Listing.deactivated_at >= args.since,
            src_filter,
        )
        .all()
    )
    log.info("\nWould restore %d %s rows deactivated on/after %s",
             len(candidates), args.source, args.since.date().isoformat())

    if not args.apply:
        log.info("Dry run. Re-run with --apply to write.")
        return 0

    for l in candidates:
        l.is_active = True
        l.deactivated_at = None
        l.deactivation_reason = None
    session.commit()
    log.info("Committed: restored %d rows.", len(candidates))
    log.info("Next successful scrape will re-confirm live rows; truly sold "
             "ones will be re-deactivated by the now source-scoped mark_inactive.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
