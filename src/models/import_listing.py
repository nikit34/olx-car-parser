"""Foreign-market listings, kept in their own table on purpose.

These are cars for sale abroad, read from AutoScout24. The original reader
(``source == "autoscout24"``) answers one question: does importing this model
from Germany beat buying it here. The country readers (``as24_de``,
``as24_fr``, ``as24_it``) hold whole national corpora that get their own price
models and their own pages. None of them is part of the Portuguese corpus and
none may leak into it: the price model, the market index, the deal feed and
every median on the public pages describe what a car asks in Portugal, and a
German price is a different quantity in a different market with a different
tax on it.

Hence a separate table rather than a ``source`` value on ``listings``: every
query that reads ``listings`` would otherwise have to remember to exclude them,
and the first one that forgets ships a market average that is quietly part
German.

Rows carry a lifecycle of their own. ``is_active`` flips off when a fully
enumerated model-year no longer lists the car, or when nothing has seen it for
``expire_import_listings``'s window; ``deactivated_at`` records when. A row that
comes back is simply re-activated by the next upsert.
"""

from sqlalchemy import (
    Boolean, Column, DateTime, Float, Integer, String, Text, UniqueConstraint,
)

from src.models.listing import Base, _utcnow


class ImportListing(Base):
    __tablename__ = "import_listings"

    id = Column(Integer, primary_key=True)
    source = Column(String, nullable=False, index=True)
    external_id = Column(String, nullable=False, index=True)
    url = Column(Text, nullable=False)

    brand = Column(String, nullable=False, index=True)
    model = Column(String, nullable=False, index=True)
    model_group = Column(String)
    variant = Column(String)
    motor_type = Column(String)
    version = Column(String)
    body_type = Column(String)
    offer_type = Column(String)

    price_eur = Column(Float)
    price_label = Column(String)
    vat_label = Column(String)
    vat_reclaimable = Column(Boolean)

    year = Column(Integer, index=True)
    registration_month = Column(String)
    mileage_km = Column(Integer)
    engine_cc = Column(Integer)
    horsepower = Column(Integer)
    power_kw = Column(Integer)
    fuel_type = Column(String)
    transmission = Column(String)
    co2_g_km = Column(Integer)

    seller_type = Column(String)
    country_code = Column(String)
    region = Column(String)
    city = Column(String)
    zip_code = Column(String)
    is_damaged = Column(Boolean)
    photo_count = Column(Integer)
    image_url = Column(Text)

    is_active = Column(Boolean, default=True)
    deactivated_at = Column(DateTime)

    first_seen_at = Column(DateTime, default=_utcnow)
    last_seen_at = Column(DateTime, default=_utcnow, index=True)

    __table_args__ = (
        UniqueConstraint("source", "external_id"),
    )
