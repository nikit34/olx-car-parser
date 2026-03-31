"""User feedback on listing quality for shortlist training."""

from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, Text

from src.models.listing import Base


INTEREST_FEEDBACK_LABELS = ("interesting", "skipped", "bought")


class ListingFeedback(Base):
    __tablename__ = "listing_feedback"

    id = Column(Integer, primary_key=True)
    olx_id = Column(String, unique=True, nullable=False, index=True)
    url = Column(Text)
    title = Column(Text)
    brand = Column(String)
    model = Column(String)
    year = Column(Integer)
    price_eur = Column(Float)
    label = Column(String, nullable=False, index=True)
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
