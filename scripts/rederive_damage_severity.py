"""One-shot: re-run rule-based ``_derive_damage_severity`` on every
listing that already has ``llm_extras``, propagate the result to the
``damage_severity`` column.

The 2026-05-02 audit showed JmUNP / 8Q0kOc / JmutI / JmR3C all carrying
stale ``damage_severity`` values written under an earlier regex set. The
LLM call doesn't need to repeat — ``_derive_damage_severity`` is rule-
based and runs in microseconds — so this script just re-applies it
across the corpus.

Run on the host that owns the authoritative DB
(``anastasia@192.168.1.74``).
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter

from sqlalchemy.exc import OperationalError

from src.parser.llm_enrichment import _derive_damage_severity
from src.storage.database import init_db, get_session
from src.models.listing import Listing


_BATCH_SIZE = 200
# Total retry budget: 2+4+8+16+32+60+60+60+60+60+60+60 ≈ 8 minutes. The
# scrape worker's "compute_market_stats" phase can hold the write lock for
# 3-5 minutes on the production DB; busy_timeout is 30 s so we need
# enough retries to outlast it. _RETRY_BASE_S * 2**attempt is capped at 60.
_RETRY_MAX = 12
_RETRY_BASE_S = 2.0
_RETRY_MAX_WAIT_S = 60.0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--apply", action="store_true",
        help="Write changes. Without this flag the run is dry-run only.",
    )
    p.add_argument(
        "--active-only", action="store_true",
        help="Limit to active listings (default: all listings with llm_extras).",
    )
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("rederive_damage_severity")

    init_db()
    session = get_session()

    q = session.query(Listing).filter(Listing.llm_extras.isnot(None))
    if args.active_only:
        q = q.filter(Listing.is_active == True)  # noqa: E712
    rows = q.all()
    log.info("Loaded %d listings with llm_extras (active-only=%s)",
             len(rows), args.active_only)

    transitions = Counter()
    to_write: list[tuple[Listing, int]] = []
    for listing in rows:
        try:
            extras = json.loads(listing.llm_extras) if listing.llm_extras else {}
        except (json.JSONDecodeError, TypeError):
            transitions["skip_unparseable_extras"] += 1
            continue
        if not isinstance(extras, dict):
            transitions["skip_non_dict_extras"] += 1
            continue

        new_sev = _derive_damage_severity(
            extras, listing.title or "", listing.description or "",
        )
        old_sev = listing.damage_severity
        if old_sev == new_sev:
            transitions["unchanged"] += 1
            continue
        transitions[f"{old_sev} -> {new_sev}"] += 1
        to_write.append((listing, new_sev))

    log.info("Severity transitions:")
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
        for listing, new_sev in chunk:
            listing.damage_severity = new_sev
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
                for listing, new_sev in chunk:
                    listing.damage_severity = new_sev
        else:
            log.error("Gave up on batch %d after %d retries", i, _RETRY_MAX)
            return 1
        if (i // _BATCH_SIZE) % 5 == 0:
            log.info("Committed %d / %d", written, len(to_write))
    log.info("Committed %d rows total", written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
