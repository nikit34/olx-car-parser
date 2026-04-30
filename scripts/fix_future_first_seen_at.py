"""One-shot cleanup: replace future-dated first_seen_at on listings.

The StandVirtual detail parser used to grab the first <p> matching the
Portuguese date pattern, which sometimes was a warranty/inspection date in
the future — that value got written through to first_seen_at and broke
turnover, days-on-market, and time-backtest. The parser is fixed; this
script repairs the rows that already landed.

For each row with first_seen_at > now, we replace it with the earliest
PriceSnapshot.scraped_at for that listing (a real scrape time), falling
back to last_seen_at if the listing has no snapshots.

Run on the host that owns the authoritative DB (anastasia@192.168.1.77).
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone

from sqlalchemy import func, text

from src.storage.database import init_db, get_session
from src.models.listing import Listing, PriceSnapshot


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--apply", action="store_true",
                   help="Write changes. Without this flag, dry-run only.")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("fix_future_first_seen_at")

    init_db()
    session = get_session()
    now = datetime.now(timezone.utc).replace(tzinfo=None)

    bad = (
        session.query(Listing)
        .filter(Listing.first_seen_at > now)
        .order_by(Listing.first_seen_at)
        .all()
    )
    log.info("Found %d listings with first_seen_at > %s", len(bad), now.isoformat())
    if not bad:
        return 0

    earliest_rows = (
        session.query(
            PriceSnapshot.listing_id,
            func.min(PriceSnapshot.scraped_at).label("earliest"),
        )
        .filter(PriceSnapshot.listing_id.in_([l.id for l in bad]))
        .group_by(PriceSnapshot.listing_id)
        .all()
    )
    earliest_by_listing: dict[int, datetime] = {lid: ts for lid, ts in earliest_rows}

    fixed = 0
    skipped = 0
    for l in bad:
        snap_ts = earliest_by_listing.get(l.id)
        candidates = [t for t in (snap_ts, l.last_seen_at) if t and t <= now]
        if not candidates:
            log.warning("Listing id=%s olx_id=%s has no usable fallback (snap=%s last_seen=%s) — skipping",
                        l.id, l.olx_id, snap_ts, l.last_seen_at)
            skipped += 1
            continue
        new_ts = min(candidates)
        log.info("listing %s (%s %s %s): %s -> %s",
                 l.olx_id, l.brand, l.model, l.year,
                 l.first_seen_at.isoformat(), new_ts.isoformat())
        if args.apply:
            # Raw SQL on purpose: ORM UPDATE would trigger
            # Listing.last_seen_at's onupdate=_utcnow and bump every fixed
            # row to "seen now", breaking the sold/inactive logic.
            session.execute(
                text("UPDATE listings SET first_seen_at = :ts WHERE id = :id"),
                {"ts": new_ts, "id": l.id},
            )
        fixed += 1

    if args.apply:
        session.commit()
        log.info("Committed %d updates (%d skipped).", fixed, skipped)
    else:
        log.info("Dry run: %d would be fixed, %d skipped. Re-run with --apply.", fixed, skipped)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
