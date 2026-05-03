"""One-shot: probe every ``deactivation_reason='sold'`` listing's URL
and restore those whose page still exists on OLX/SV (HTTP 200 with
no dead-marker phrase). Backfills the false-positive sweeps the old
mark_inactive produced before the URL-verify guard landed.

The 2026-05-03 audit found Alfa Romeo 147 ``JmEjz`` flagged "sold"
while clearly still live. Same root cause for an unknown number of
historical rows; this script reverses them.

Run on the host that owns the authoritative DB
(``anastasia@192.168.1.74``).
"""
from __future__ import annotations

import argparse
import logging
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from sqlalchemy.exc import OperationalError

from src.storage.database import init_db, get_session
from src.storage.repository import _verify_listing_alive
from src.models.listing import Listing


_BATCH_SIZE = 200
_RETRY_MAX = 12
_RETRY_BASE_S = 2.0
_RETRY_MAX_WAIT_S = 60.0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--apply", action="store_true",
                   help="Write changes. Without this flag the run is dry-run only.")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=None,
                   help="Cap on rows probed (useful for staged rollout).")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("heal_false_sold")

    init_db()
    session = get_session()
    rows = (
        session.query(Listing)
        .filter(Listing.deactivation_reason == "sold")
        .order_by(Listing.deactivated_at.desc())
        .all()
    )
    if args.limit:
        rows = rows[: args.limit]
    log.info("Probing %d 'sold' listings", len(rows))

    stats: Counter = Counter()
    to_restore: list[int] = []

    def _check(listing):
        return listing.id, _verify_listing_alive(listing.url)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, (lid, status) in enumerate(pool.map(_check, rows)):
            if status is True:
                to_restore.append(lid)
                stats["false_positive_alive"] += 1
            elif status is False:
                stats["confirmed_dead"] += 1
            else:
                stats["deferred_unknown"] += 1
            if (i + 1) % 100 == 0:
                log.info("  probed %d / %d  (alive=%d dead=%d deferred=%d)",
                         i + 1, len(rows),
                         stats["false_positive_alive"],
                         stats["confirmed_dead"],
                         stats["deferred_unknown"])

    log.info("Probe complete: %s", dict(stats))
    log.info("Will %s %d false-positive 'sold' rows back to active",
             "RESTORE" if args.apply else "(dry-run) restore",
             len(to_restore))

    if not args.apply or not to_restore:
        return 0

    written = 0
    for i in range(0, len(to_restore), _BATCH_SIZE):
        chunk = to_restore[i:i + _BATCH_SIZE]
        for attempt in range(_RETRY_MAX):
            try:
                session.query(Listing).filter(Listing.id.in_(chunk)).update(
                    {
                        "is_active": True,
                        "deactivated_at": None,
                        "deactivation_reason": None,
                    },
                    synchronize_session="evaluate",
                )
                session.commit()
                written += len(chunk)
                break
            except OperationalError as e:
                if "locked" not in str(e).lower():
                    raise
                session.rollback()
                wait = min(_RETRY_BASE_S * (2 ** attempt), _RETRY_MAX_WAIT_S)
                log.warning("DB locked on batch %d, retry %d/%d in %.1fs",
                            i, attempt + 1, _RETRY_MAX, wait)
                time.sleep(wait)
        else:
            log.error("Gave up on batch %d", i)
            return 1
        if (i // _BATCH_SIZE) % 5 == 0:
            log.info("Restored %d / %d", written, len(to_restore))

    log.info("Restored %d rows total", written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
