"""Record galleries for StandVirtual listings the scrape will never revisit.

OLX needs no backfill: its offers API carries the gallery on every page of
every full-coverage run, so each active listing picks up its fingerprints on
the next scrape at no cost. StandVirtual does not — the search GraphQL node has
no photos at all, and only listings the scrape considers new get the detail
fetch that does. Left alone, the ads already live when the fingerprint table
landed would never get one.

So this walks the active StandVirtual listings that have no stored photos and
opens each advert once. Resumable and rate-limited; a listing whose page no
longer parses is left for the next run rather than recorded as empty.

    .venv/bin/python -m scripts.backfill_photo_refs [--limit 1000] [--rps 2]
"""
from __future__ import annotations

import argparse
import logging
import random
import time

from sqlalchemy import func

from src.models.listing import Listing
from src.models.photo import ListingPhoto
from src.parser.photo_fetch import fetch_standvirtual_advert, photo_refs_standvirtual
from src.storage.database import get_session, init_db
from src.storage.repository import save_listing_photos


def pending(session, limit: int, source: str = "standvirtual"):
    """Active listings of *source* with nothing in ``listing_photos`` yet."""
    have = session.query(ListingPhoto.olx_id).distinct().scalar_subquery()
    return (
        session.query(Listing.olx_id, Listing.url)
        .filter(Listing.source == source)
        .filter(Listing.is_active.is_(True))
        .filter(Listing.olx_id.notin_(have))
        .order_by(func.coalesce(Listing.last_seen_at, Listing.first_seen_at).desc())
        .limit(limit)
        .all()
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill StandVirtual photo fingerprints.")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--rps", type=float, default=2.0,
                        help="Advert fetches per second.")
    parser.add_argument("--source", default="standvirtual")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("backfill_photos")

    init_db()
    session = get_session()
    rows = pending(session, args.limit, args.source)
    if not rows:
        log.info("Nothing pending for %s", args.source)
        return 0

    log.info("Opening %d %s adverts at ~%.1f/s", len(rows), args.source, args.rps)
    delay = 1.0 / max(args.rps, 0.1)
    saved = empty = 0
    for i, (olx_id, url) in enumerate(rows, 1):
        advert = fetch_standvirtual_advert(url)
        refs = photo_refs_standvirtual(advert) if advert else []
        if refs:
            save_listing_photos(session, olx_id, refs)
            saved += 1
        else:
            empty += 1
        if i % 100 == 0:
            session.commit()
            log.info("  %d/%d (%d with photos, %d without)",
                     i, len(rows), saved, empty)
        time.sleep(delay + random.uniform(0, delay * 0.3))
    session.commit()
    log.info("Done: %d listings fingerprinted, %d yielded nothing", saved, empty)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
