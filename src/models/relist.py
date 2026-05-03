"""Re-listing event tracking — same physical car re-posted under a new olx_id.

Populated by ``scripts/detect_relists.py`` running the matcher in
``src.analytics.relist``. One row per detected (original, relist) pair.
The unique constraint blocks duplicate inserts on re-runs of the
detection job — re-runs update the score in place instead.

Downstream consumer: ``decision.calibrate_thresholds`` via
``relist.build_outcomes_df``, which converts these events into the
``outcomes_df`` shape (olx_id / buy_price / sell_price / days_held /
fees_eur) that the threshold grid-search expects. ``original_olx_id``
is the buy-side listing — that's what a hypothetical flipper would
have purchased.
"""

from sqlalchemy import (
    Column, DateTime, Float, Integer, String, UniqueConstraint,
)

from src.models.listing import Base, _utcnow


class RelistEvent(Base):
    __tablename__ = "relist_events"

    id = Column(Integer, primary_key=True)
    original_olx_id = Column(String, nullable=False, index=True)
    relist_olx_id = Column(String, nullable=False, index=True)
    gap_days = Column(Float, nullable=False)
    match_score = Column(Float, nullable=False)
    original_price_eur = Column(Float)
    relist_price_eur = Column(Float)
    price_delta_eur = Column(Float)
    price_delta_pct = Column(Float)
    mileage_delta_km = Column(Integer)
    detected_at = Column(DateTime, default=_utcnow)

    __table_args__ = (
        UniqueConstraint("original_olx_id", "relist_olx_id"),
    )
