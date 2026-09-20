"""Check existing ``duplicate_of`` marks against the photographs.

The marks come from attribute rules — same brand, model, year, district, close
mileage, close price — and on 2026-09-20 a hand check of 20 live cross-platform
pairs found 2 of them joined two different cars: a white five-door BMW 116 to a
red three-door, and a black Alfa Giulietta to a grey one. Every such mark costs
a real observation, because a listing flagged as a duplicate is dropped from the
segment statistics.

This reads the pairs both of whose listings are still live (a dead ad's photos
can no longer be fetched, so its mark cannot be audited) and sorts them:

  confirmed    — the galleries share a frame; the mark stands
  contradicted — both sides are fingerprinted and share nothing
  unchecked    — at least one side has no stored hashes yet

Contradicted is not the same as wrong. Of the three mismatches in that sample,
one was a genuine duplicate whose StandVirtual side posted 2x2 collages, which
share no whole frame with the single photos on OLX. So roughly a third of
contradictions are the hash failing rather than the mark, and ``--apply`` trades
those against the marks that really do merge two cars. It is off by default.

    .venv/bin/python -m scripts.audit_duplicate_marks [--apply] [--limit N]
"""
from __future__ import annotations

import argparse
import logging

from src.analytics.photo_match import load_photo_hashes, photo_overlap
from src.models.listing import Listing
from src.storage.database import get_session, init_db


def audit(session, limit: int | None = None) -> dict:
    """Bucket every auditable duplicate mark. Returns the three lists."""
    canonical = Listing.__table__.alias("canonical")
    q = (
        session.query(
            Listing.olx_id, Listing.source, Listing.brand, Listing.model,
            Listing.year, canonical.c.olx_id, canonical.c.source,
        )
        .join(canonical, canonical.c.olx_id == Listing.duplicate_of)
        .filter(Listing.duplicate_of.isnot(None))
        .filter(Listing.is_active.is_(True))
        .filter(canonical.c.is_active.is_(True))
    )
    if limit:
        q = q.limit(limit)
    rows = q.all()

    ids = {r[0] for r in rows} | {r[5] for r in rows}
    hashes = load_photo_hashes(session, ids)

    out = {"confirmed": [], "contradicted": [], "unchecked": []}
    for dup_id, dup_src, brand, model, year, canon_id, canon_src in rows:
        a, b = hashes.get(canon_id), hashes.get(dup_id)
        label = (dup_id, canon_id, f"{brand} {model} {year}",
                 f"{canon_src}->{dup_src}")
        if not a or not b:
            out["unchecked"].append(label)
            continue
        matched, _score = photo_overlap(a, b)
        out["confirmed" if matched else "contradicted"].append(label)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit duplicate_of marks against photo fingerprints.")
    parser.add_argument("--apply", action="store_true",
                        help="Clear the marks the photographs contradict.")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("audit_duplicates")

    init_db()
    session = get_session()
    buckets = audit(session, args.limit)

    checked = len(buckets["confirmed"]) + len(buckets["contradicted"])
    log.info("Auditable marks: %d (both sides live)",
             checked + len(buckets["unchecked"]))
    log.info("  confirmed by photos:    %d", len(buckets["confirmed"]))
    log.info("  contradicted by photos: %d", len(buckets["contradicted"]))
    log.info("  not yet fingerprinted:  %d", len(buckets["unchecked"]))
    if checked:
        log.info("  contradiction rate among checked: %.1f%%",
                 100 * len(buckets["contradicted"]) / checked)

    for dup_id, canon_id, car, direction in buckets["contradicted"][:40]:
        log.info("    %s is marked a duplicate of %s (%s, %s)",
                 dup_id, canon_id, car, direction)

    if not args.apply:
        log.info("(dry run: nothing written; pass --apply to clear them)")
        return 0

    cleared = 0
    for dup_id, _canon_id, _car, _direction in buckets["contradicted"]:
        listing = session.query(Listing).filter_by(olx_id=dup_id).one_or_none()
        if listing is not None:
            listing.duplicate_of = None
            cleared += 1
    session.commit()
    log.info("Cleared %d marks", cleared)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
