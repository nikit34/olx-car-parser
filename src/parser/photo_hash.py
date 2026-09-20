"""Perceptual hashing of listing photos.

The CDN gives every upload a fresh file id, so two ads carrying the same
photograph share nothing but the pixels. This module turns a photo into a
64-bit dHash of its 200x150 rendition — 7 KB on the wire per photo, which is
what makes hashing the whole active corpus affordable. Renditions below about
200 px lose the detail that separates two similar cars; above it the bytes grow
without changing the 8x8 grid the hash reduces to.

Measured on 2026-09-20: at Hamming 0 the hash produced one cross-listing match
in 14 million comparisons and it was a real duplicate; false pairs start at
distance 6. Distances 1-2 still carry a third of the true matches (11 of 34 in
the OLX sample), so :data:`MATCH_DISTANCE` is 2, not 0.

The watermark StandVirtual burns into its renditions does not move the hash —
matching pairs come back at distance 0 with or without cropping the strip — so
nothing here special-cases it. Collages do defeat it: a seller who posts four
photos as one 2x2 image shares no whole frame with the ad that posts them
singly, and that pair is simply missed.
"""

from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO

import numpy as np
from PIL import Image

from src.parser.photo_fetch import _CLIENT


HASH_SIZE = 8
MATCH_DISTANCE = 2
THUMB_SIZE = "200x150"

OLX_CDN = "https://ireland.apollo.olxcdn.com/v1/files"

_MIN_BITS = 12
_MAX_BITS = 52


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def dhash(img: Image.Image) -> str:
    """64-bit difference hash as 16 hex characters."""
    grey = img.convert("L").resize((HASH_SIZE + 1, HASH_SIZE), Image.LANCZOS)
    px = np.asarray(grey, dtype=np.int16)
    bits = (px[:, :-1] < px[:, 1:]).flatten()
    return f"{int(''.join('1' if b else '0' for b in bits), 2):016x}"


def hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def is_degenerate(phash: str) -> bool:
    """True for a near-uniform frame — a blank, a dark interior, a placeholder.

    Those hashes collide with each other whatever the subject, so they are
    never allowed to stand as match evidence.
    """
    bits = bin(int(phash, 16)).count("1")
    return bits < _MIN_BITS or bits > _MAX_BITS


def thumb_url(photo_id: str, url: str | None = None,
              size: str = THUMB_SIZE) -> str:
    """Fetchable URL for a stored photo.

    StandVirtual's URL is a signed JWT and must be the one the advert gave us;
    OLX's is a plain path around the file id, so it costs nothing to rebuild.
    """
    if url:
        return url if ";s=" in url else f"{url};s={size}"
    return f"{OLX_CDN}/{photo_id}/image;s={size}"


def fetch_hash(photo_id: str, url: str | None = None) -> str | None:
    """Download one photo and hash it. None on any HTTP or decode failure."""
    try:
        resp = _CLIENT.get(thumb_url(photo_id, url))
        resp.raise_for_status()
        return dhash(Image.open(BytesIO(resp.content)))
    except Exception:
        return None
