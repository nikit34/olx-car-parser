"""Deciding whether two listings show the same physical car, from the pixels.

Text cannot separate "the same car posted twice" from "two cars of the same
model with a round odometer reading in the same district" — the 2026-09-20
sample had 163 such candidate pairs and only 34 were one car. Photos can, and
at a threshold that leaves no room for argument: matching pairs come back at
Hamming 0-2 out of 64 bits, while the first false pair in 14 million
comparisons appeared at distance 6.

Two guards keep the precision:

*degenerate hashes* — near-uniform frames match each other whatever they show,
so :func:`src.parser.photo_hash.is_degenerate` drops them before matching;

*stock photos* — a hash that turns up in more listings than
:data:`MAX_LISTINGS_PER_HASH` is a dealer template, a banner or a placeholder,
not a car, and is excluded rather than allowed to link everything it touches.

What this cannot see: a re-shoot. A reseller who photographs the car again
shares no pixels with the ad they bought from, so photo evidence covers
"the seller re-posted" and leaves "someone else re-sold it" to the text
matcher.
"""

from __future__ import annotations

from collections import defaultdict

from src.parser.photo_hash import MATCH_DISTANCE, hamming, is_degenerate


MAX_LISTINGS_PER_HASH = 4

BANDS = 4
BAND_HEX = 16 // BANDS


def usable_hashes(hashes) -> list[str]:
    """The hashes of one listing that may stand as evidence, deduplicated."""
    out, seen = [], set()
    for h in hashes or []:
        if not h or h in seen:
            continue
        seen.add(h)
        if not is_degenerate(h):
            out.append(h)
    return out


def stock_hashes(by_listing: dict[str, list[str]],
                 max_listings: int = MAX_LISTINGS_PER_HASH) -> set[str]:
    """Hashes shared by too many listings to mean "the same car"."""
    counts: dict[str, int] = defaultdict(int)
    for hashes in by_listing.values():
        for h in set(hashes or []):
            counts[h] += 1
    return {h for h, n in counts.items() if n > max_listings}


def photo_overlap(a_hashes, b_hashes,
                  max_distance: int = MATCH_DISTANCE,
                  exclude: set[str] | None = None) -> tuple[int, float]:
    """``(matching photos, share of the smaller gallery)`` for one pair.

    The share is the score the callers store: two ads that share four of their
    five photos are a stronger claim than two that share one of twenty.
    """
    a = [h for h in usable_hashes(a_hashes) if not exclude or h not in exclude]
    b = [h for h in usable_hashes(b_hashes) if not exclude or h not in exclude]
    if not a or not b:
        return 0, 0.0
    matched = 0
    taken: set[int] = set()
    for x in a:
        for j, y in enumerate(b):
            if j in taken:
                continue
            if hamming(x, y) <= max_distance:
                taken.add(j)
                matched += 1
                break
    return matched, matched / min(len(a), len(b))


def _bands(h: str) -> list[tuple[int, str]]:
    return [(i, h[i * BAND_HEX:(i + 1) * BAND_HEX]) for i in range(BANDS)]


def find_pairs(by_listing: dict[str, list[str]],
               max_distance: int = MATCH_DISTANCE,
               max_listings_per_hash: int = MAX_LISTINGS_PER_HASH,
               min_photos: int = 1,
               max_bucket: int = 2000) -> dict[tuple[str, str], tuple[int, float]]:
    """Every pair of listings sharing photos, keyed by sorted ``(a, b)``.

    Candidates come from a banded index rather than all-pairs: two 64-bit
    hashes within Hamming 2 must agree on at least two of four 16-bit bands, so
    grouping by band value finds every true pair while comparing a small
    fraction of the stored hashes.

    ``max_bucket`` bounds the damage from a band value thousands of photos
    happen to share — comparing such a bucket costs millions of operations and
    yields pairs the distance check would throw away anyway.
    """
    stock = stock_hashes(by_listing, max_listings_per_hash)
    clean = {k: [h for h in usable_hashes(v) if h not in stock]
             for k, v in by_listing.items()}
    clean = {k: v for k, v in clean.items() if v}
    sizes = {k: len(v) for k, v in clean.items()}

    buckets: dict[tuple[int, str], list[tuple[str, str]]] = defaultdict(list)
    for olx_id, hashes in clean.items():
        for h in hashes:
            for band in _bands(h):
                buckets[band].append((olx_id, h))

    hits: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    for entries in buckets.values():
        if len(entries) < 2 or len(entries) > max_bucket:
            continue
        for i, (a, ha) in enumerate(entries):
            for b, hb in entries[i + 1:]:
                if a == b or hamming(ha, hb) > max_distance:
                    continue
                key = (a, b) if a < b else (b, a)
                pair = (ha, hb) if a < b else (hb, ha)
                hits[key].add(pair)

    out: dict[tuple[str, str], tuple[int, float]] = {}
    for (a, b), pairs in hits.items():
        matched = len({p[0] for p in pairs})
        if matched >= min_photos:
            out[(a, b)] = (matched, matched / min(sizes[a], sizes[b]))
    return out


def load_photo_hashes(session, olx_ids=None) -> dict[str, list[str]]:
    """``{olx_id: [phash, ...]}`` for hashed photos, gallery order preserved."""
    from src.models.photo import ListingPhoto

    q = (session.query(ListingPhoto.olx_id, ListingPhoto.phash)
         .filter(ListingPhoto.phash.isnot(None))
         .order_by(ListingPhoto.olx_id, ListingPhoto.pos))
    if olx_ids is not None:
        ids = list(olx_ids)
        if not ids:
            return {}
        q = q.filter(ListingPhoto.olx_id.in_(ids))
    out: dict[str, list[str]] = defaultdict(list)
    for olx_id, phash in q:
        out[olx_id].append(phash)
    return dict(out)
