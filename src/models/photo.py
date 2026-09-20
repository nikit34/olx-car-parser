"""Per-photo fingerprints — the only identity signal that survives a re-post.

One row per (listing, photo). Written at scrape time from the gallery both
scrapers already receive and used to throw away: the OLX offers API carries
``photos[].filename`` and the StandVirtual advert carries
``images.photos[].url``, whose JWT payload names the file.

``photo_id`` is the CDN's own file id. It is minted fresh on every upload —
measured 2026-09-20 over 26 258 StandVirtual photos and 163 OLX pairs, not one
id was ever shared by two listings, including by pairs whose photos were
byte-identical. So the id is a within-listing key, never a cross-listing one,
and ``phash`` is what actually matches: a 64-bit dHash of the 200x150 rendition,
stored as 16 hex characters.

``url`` exists because StandVirtual's photo URL is a signed JWT that cannot be
rebuilt from the file id, and the image still has to be fetched once to be
hashed. It is cleared as soon as ``phash`` lands. OLX URLs are reconstructable
from ``photo_id`` alone and are never stored.

Rows outlive the listing on purpose. A car that is re-posted next month is only
recognisable if the fingerprints of the ad that died are still here.
"""

from sqlalchemy import (
    Column, DateTime, Integer, String, Text, UniqueConstraint,
)

from src.models.listing import Base


class ListingPhoto(Base):
    __tablename__ = "listing_photos"

    id = Column(Integer, primary_key=True)
    olx_id = Column(String, nullable=False, index=True)
    pos = Column(Integer, nullable=False)
    photo_id = Column(String, nullable=False, index=True)
    url = Column(Text)
    phash = Column(String(16), index=True)
    hashed_at = Column(DateTime)

    __table_args__ = (
        UniqueConstraint("olx_id", "photo_id"),
    )
