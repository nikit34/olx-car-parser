"""One-shot backfill: persist ``photo_damage_flagged`` on active listings
that have ``photo_damage_p`` but not the multi-photo decision.

Two paths, picked per listing:

1. **multi-photo rule** when the listing already carries ``photo_damages``
   (per-photo array, written by post-issue-#4 verify-photos runs). We
   reapply ``FLAG_MIN_PHOTOS`` / ``FLAG_PHOTO_THRESHOLD`` from
   ``src.parser.damage_decision`` to recover the same decision the cron
   would have written. No torch / no photo download — pure JSON math.

2. **legacy max-rule** when ``photo_damages`` is missing (pre-#4 rows).
   We write ``photo_damage_flagged = (photo_damage_p >= DEFAULT_THRESHOLD)``,
   which is identical to what ``is_listing_flagged`` returns as fallback
   today. Persisting it removes the schema branch in ``llm_extras`` so
   the dashboard / alerts / blocker code stops needing the fallback.

Adds an audit field ``photo_damage_flag_source`` set to
``"multi_photo_backfill"`` or ``"legacy_max_rule_backfill"`` so a later
analysis can tell which path each row took.

Run on the host that owns the authoritative DB
(``anastasia@192.168.1.74`` — see ``release-db`` skill).
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter

from sqlalchemy.exc import OperationalError

from src.parser.damage_decision import (
    DEFAULT_THRESHOLD,
    FLAG_MIN_PHOTOS,
    FLAG_PHOTO_THRESHOLD,
)
from src.storage.database import init_db, get_session
from src.models.listing import Listing


_BATCH_SIZE = 200
_RETRY_MAX = 6
_RETRY_BASE_S = 2.0


def _decide_from_per_photo(per_photo: list[dict]) -> bool:
    """Apply the multi-photo agreement rule to a stored per-photo array."""
    above = 0
    for entry in per_photo:
        try:
            p = float(entry.get("p", 0.0))
        except (TypeError, ValueError):
            continue
        if p >= FLAG_PHOTO_THRESHOLD:
            above += 1
            if above >= FLAG_MIN_PHOTOS:
                return True
    return False


def _decide_legacy(extras: dict) -> bool:
    p = extras.get("photo_damage_p") or 0.0
    try:
        return float(p) >= DEFAULT_THRESHOLD
    except (TypeError, ValueError):
        return False


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--apply", action="store_true",
        help="Write changes. Without this flag the run is dry-run only.",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Optional cap on rows processed (useful for staged rollout).",
    )
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("backfill_photo_damage_flagged")

    init_db()
    session = get_session()

    rows = (
        session.query(Listing)
        .filter(
            Listing.is_active == True,  # noqa: E712
            Listing.llm_extras.isnot(None),
        )
        .all()
    )
    log.info("Loaded %d active listings with llm_extras", len(rows))

    stats = Counter()
    to_write: list[tuple[Listing, dict]] = []
    for listing in rows:
        try:
            extras = json.loads(listing.llm_extras) if listing.llm_extras else {}
        except (json.JSONDecodeError, TypeError):
            stats["skip_unparseable_extras"] += 1
            continue
        if not isinstance(extras, dict):
            stats["skip_non_dict_extras"] += 1
            continue
        if "photo_damage_p" not in extras:
            stats["skip_no_photo_damage_p"] += 1
            continue
        if "photo_damage_flagged" in extras:
            stats["skip_already_set"] += 1
            continue

        per_photo = extras.get("photo_damages")
        if isinstance(per_photo, list) and per_photo:
            decision = _decide_from_per_photo(per_photo)
            extras["photo_damage_flagged"] = bool(decision)
            extras["photo_damage_flag_source"] = "multi_photo_backfill"
            stats[
                "multi_photo_flagged" if decision else "multi_photo_cleared"
            ] += 1
        else:
            decision = _decide_legacy(extras)
            extras["photo_damage_flagged"] = bool(decision)
            extras["photo_damage_flag_source"] = "legacy_max_rule_backfill"
            stats[
                "legacy_flagged" if decision else "legacy_cleared"
            ] += 1

        to_write.append((listing, extras))
        if args.limit and len(to_write) >= args.limit:
            log.info("Hit --limit=%d, stopping selection", args.limit)
            break

    log.info("Decision tally:")
    for k in sorted(stats):
        log.info("  %s: %d", k, stats[k])
    log.info("Will %s %d rows",
             "WRITE" if args.apply else "(dry-run) update",
             len(to_write))

    if not args.apply or not to_write:
        return 0

    # Commit in batches with exponential-backoff retries on "database is
    # locked" — the scrape worker on this host writes through the same
    # SQLite file, and a single 5k-row UPDATE blocks new listings from
    # landing for the duration of the commit. 200-row batches keep each
    # transaction short enough that scrape inserts can interleave.
    written = 0
    for i in range(0, len(to_write), _BATCH_SIZE):
        chunk = to_write[i:i + _BATCH_SIZE]
        for listing, extras in chunk:
            listing.llm_extras = json.dumps(extras, ensure_ascii=False)
        for attempt in range(_RETRY_MAX):
            try:
                session.commit()
                written += len(chunk)
                break
            except OperationalError as e:
                if "locked" not in str(e).lower():
                    raise
                session.rollback()
                wait = _RETRY_BASE_S * (2 ** attempt)
                log.warning(
                    "DB locked on batch %d-%d, retry %d/%d in %.1fs",
                    i, i + len(chunk), attempt + 1, _RETRY_MAX, wait,
                )
                time.sleep(wait)
                # Re-stage the updates after rollback (rollback drops the
                # pending UPDATEs from the unit-of-work).
                for listing, extras in chunk:
                    listing.llm_extras = json.dumps(extras, ensure_ascii=False)
        else:
            log.error("Gave up on batch %d after %d retries", i, _RETRY_MAX)
            return 1
        if (i // _BATCH_SIZE) % 5 == 0:
            log.info("Committed %d / %d", written, len(to_write))
    log.info("Committed %d rows total", written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
