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


_RETRY_MAX = 12
_RETRY_BASE_S = 2.0
_RETRY_MAX_WAIT_S = 60.0
# Probes are committed every N restores so killing the script mid-run
# (rate-limit, SIGTERM, OLX going funny) doesn't lose accumulated work
# the way the v1 "scan-then-bulk-update" design did.
_FLUSH_EVERY = 50


def _commit_restores(session, ids: list[int], log) -> int:
    if not ids:
        return 0
    for attempt in range(_RETRY_MAX):
        try:
            session.query(Listing).filter(Listing.id.in_(ids)).update(
                {
                    "is_active": True,
                    "deactivated_at": None,
                    "deactivation_reason": None,
                },
                synchronize_session="evaluate",
            )
            session.commit()
            return len(ids)
        except OperationalError as e:
            if "locked" not in str(e).lower():
                raise
            session.rollback()
            wait = min(_RETRY_BASE_S * (2 ** attempt), _RETRY_MAX_WAIT_S)
            log.warning(
                "DB locked, retry %d/%d in %.1fs (%d ids)",
                attempt + 1, _RETRY_MAX, wait, len(ids),
            )
            time.sleep(wait)
    log.error("Gave up committing %d ids", len(ids))
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--apply", action="store_true",
                   help="Write changes. Without this flag the run is dry-run only.")
    p.add_argument("--workers", type=int, default=4,
                   help="Concurrent HTTP probes. v1 used 8 and OLX rate-"
                        "limited; 2-4 stays under the radar.")
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
    log.info("Probing %d 'sold' listings (workers=%d, flush every %d restores)",
             len(rows), args.workers, _FLUSH_EVERY)

    stats: Counter = Counter()
    pending_ids: list[int] = []
    written = 0

    def _check(listing):
        return listing.id, _verify_listing_alive(listing.url)

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for i, (lid, status) in enumerate(pool.map(_check, rows)):
                if status is True:
                    pending_ids.append(lid)
                    stats["alive"] += 1
                elif status is False:
                    stats["dead"] += 1
                else:
                    stats["deferred"] += 1

                # Flush every _FLUSH_EVERY confirmed alive — keeps work
                # safe against mid-run aborts (rate-limit, SIGTERM,
                # OLX outage). Commits via batched UPDATE with retry.
                if args.apply and len(pending_ids) >= _FLUSH_EVERY:
                    n = _commit_restores(session, pending_ids, log)
                    written += n
                    pending_ids.clear()

                if (i + 1) % 100 == 0:
                    log.info(
                        "  probed %d / %d  (alive=%d dead=%d deferred=%d, restored=%d)",
                        i + 1, len(rows),
                        stats["alive"], stats["dead"], stats["deferred"],
                        written,
                    )
    except KeyboardInterrupt:
        log.warning("interrupted by user — flushing pending restores")

    # Final flush of anything that didn't reach the threshold.
    if args.apply and pending_ids:
        written += _commit_restores(session, pending_ids, log)
        pending_ids.clear()

    log.info(
        "DONE: %s  (apply=%s, restored=%d)",
        dict(stats), args.apply, written,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
