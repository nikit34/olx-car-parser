"""One-shot: re-canonicalise the ``brand`` column on every listing
that was written before the upsert-time normalisation hook landed.

Idempotent — applying ``normalize_brand`` to an already-canonical
value is a no-op. Safe to re-run.

Run on the host that owns the authoritative DB
(``anastasia@192.168.1.74``).
"""
from __future__ import annotations

import argparse
import logging
import time
from collections import Counter

from sqlalchemy.exc import OperationalError

from src.parser.brand_normalize import normalize_brand
from src.storage.database import init_db, get_session
from src.models.listing import Listing


_BATCH_SIZE = 200
_RETRY_MAX = 12
_RETRY_BASE_S = 2.0
_RETRY_MAX_WAIT_S = 60.0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--apply", action="store_true",
                   help="Write changes. Without this flag the run is dry-run only.")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("normalize_brands")

    init_db()
    session = get_session()
    rows = session.query(Listing).all()
    log.info("Loaded %d listings", len(rows))

    transitions: Counter = Counter()
    to_write: list[tuple[Listing, str]] = []
    for listing in rows:
        old = listing.brand or ""
        new = normalize_brand(old)
        if new == old:
            continue
        transitions[f"{old!r} -> {new!r}"] += 1
        to_write.append((listing, new))

    log.info("Brand transitions:")
    for k in sorted(transitions, key=lambda s: (-transitions[s], s)):
        log.info("  %s: %d", k, transitions[k])
    log.info("Will %s %d rows",
             "WRITE" if args.apply else "(dry-run) update",
             len(to_write))

    if not args.apply or not to_write:
        return 0

    written = 0
    for i in range(0, len(to_write), _BATCH_SIZE):
        chunk = to_write[i:i + _BATCH_SIZE]
        for listing, new in chunk:
            listing.brand = new
        for attempt in range(_RETRY_MAX):
            try:
                session.commit()
                written += len(chunk)
                break
            except OperationalError as e:
                if "locked" not in str(e).lower():
                    raise
                session.rollback()
                wait = min(_RETRY_BASE_S * (2 ** attempt), _RETRY_MAX_WAIT_S)
                log.warning(
                    "DB locked on batch %d, retry %d/%d in %.1fs",
                    i, attempt + 1, _RETRY_MAX, wait,
                )
                time.sleep(wait)
                for listing, new in chunk:
                    listing.brand = new
        else:
            log.error("Gave up on batch %d", i)
            return 1
        if (i // _BATCH_SIZE) % 5 == 0:
            log.info("Committed %d / %d", written, len(to_write))
    log.info("Committed %d rows total", written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
