"""SQLAlchemy models for OLX.pt car listings and price history."""

from datetime import datetime

from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text, Date,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Listing(Base):
    __tablename__ = "listings"

    id = Column(Integer, primary_key=True)
    olx_id = Column(String, unique=True, nullable=False, index=True)
    url = Column(Text, nullable=False)
    title = Column(Text)
    brand = Column(String, nullable=False, index=True)
    model = Column(String, nullable=False, index=True)
    year = Column(Integer)
    mileage_km = Column(Integer)
    engine_cc = Column(Integer)            # Cilindrada (cc)
    fuel_type = Column(String)             # Combustível: Diesel, Gasolina, Híbrido, Híbrido Plug-in, Eléctrico, GPL
    horsepower = Column(Integer)           # Potência (cv)
    transmission = Column(String)          # Tipo de Caixa: Automática, Manual
    segment = Column(String)               # Segmento: Citadino, Carrinha, SUV/TT, Sedan, Coupé, Cabrio, Pequeno Citadino, Utilitário, Monovolume
    doors = Column(String)                 # Portas: 1-3, 4-5
    seats = Column(Integer)               # Lugares
    color = Column(String)                 # Cor
    condition = Column(String)             # Condição: Usado, Novo
    drive_type = Column(String)            # Tração: Dianteira, Traseira, Integral
    photo_count = Column(Integer)          # Number of photos in listing
    description_length = Column(Integer)   # Length of description text

    registration_month = Column(String)    # Mes de Registo
    city = Column(String, index=True)      # Cidade/freguesia
    district = Column(String, index=True)  # Distrito (Porto, Lisboa, Faro...)
    seller_type = Column(String)           # Particular / Profissional
    description = Column(Text)              # Listing description text
    llm_extras = Column(Text)               # JSON: LLM-extracted structured data from description
    llm_description_hash = Column(String)    # Hash of description used for LLM enrichment
    first_seen_at = Column(DateTime, default=datetime.utcnow)
    last_seen_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    deactivated_at = Column(DateTime)          # When listing went inactive (sold/removed)
    deactivation_reason = Column(String)       # "sold", "expired", "removed", "unknown"

    generation = Column(String)               # Car generation (e.g. "Golf VII", "E90")

    # Enrichment columns (extracted from listing description text)
    sub_model = Column(String)                 # Engine/body variant: "320d", "1.6 TDI"
    trim_level = Column(String)                # Equipment line: "AMG Line", "M Sport", "GTI"
    desc_mentions_repair = Column(Boolean)
    desc_mentions_accident = Column(Boolean)
    accident_details = Column(String)
    real_mileage_km = Column(Integer)           # Mileage from description (may differ from attribute)
    desc_mentions_num_owners = Column(Integer)
    desc_mentions_customs_cleared = Column(Boolean)
    imported = Column(Boolean)
    right_hand_drive = Column(Boolean)
    mechanical_condition = Column(String)       # "excellent", "good", "fair", "poor"
    paint_condition = Column(String)            # "excellent", "good", "fair", "poor"
    service_history = Column(Boolean)
    repair_details = Column(String)
    suspicious_signs = Column(Text)             # JSON list
    extras = Column(Text)                       # JSON list
    issues = Column(Text)                       # JSON list
    reason_for_sale = Column(String)
    urgency = Column(String)                   # "high", "medium", "low"
    warranty = Column(Boolean)
    tuning_or_mods = Column(Text)              # JSON list
    taxi_fleet_rental = Column(Boolean)
    recent_maintenance = Column(Text)           # JSON list
    first_owner_selling = Column(Boolean)

    source = Column(String, default="olx")     # "olx" or "standvirtual"
    duplicate_of = Column(String)              # olx_id of the canonical listing (fuzzy dedup)

    price_snapshots = relationship("PriceSnapshot", back_populates="listing", lazy="dynamic")


class PriceSnapshot(Base):
    __tablename__ = "price_snapshots"

    id = Column(Integer, primary_key=True)
    listing_id = Column(Integer, ForeignKey("listings.id"), nullable=False, index=True)
    price_eur = Column(Float, nullable=False)
    negotiable = Column(Boolean, default=False)
    scraped_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    listing = relationship("Listing", back_populates="price_snapshots")


class MarketStats(Base):
    __tablename__ = "market_stats"

    id = Column(Integer, primary_key=True)
    brand = Column(String, nullable=False)
    model = Column(String, nullable=False)
    year_from = Column(Integer)
    year_to = Column(Integer)
    date = Column(Date, nullable=False)
    median_price_eur = Column(Float)
    avg_price_eur = Column(Float)
    min_price_eur = Column(Float)
    max_price_eur = Column(Float)
    listing_count = Column(Integer)

    __table_args__ = (
        UniqueConstraint("brand", "model", "year_from", "year_to", "date"),
    )


class UnmatchedListing(Base):
    """Listings where car generation could not be determined."""
    __tablename__ = "unmatched_listings"

    id = Column(Integer, primary_key=True)
    olx_id = Column(String, unique=True, nullable=False, index=True)
    url = Column(Text, nullable=False)
    title = Column(Text)
    brand = Column(String, nullable=False)
    model = Column(String, nullable=False)
    year = Column(Integer)
    price_eur = Column(Float)
    mileage_km = Column(Integer)
    fuel_type = Column(String)
    city = Column(String)
    district = Column(String)
    seller_type = Column(String)
    description = Column(Text)
    reason = Column(String)              # "no_year", "no_generation_match"
    source = Column(String, default="olx")     # "olx" or "standvirtual"
    first_seen_at = Column(DateTime, default=datetime.utcnow)
    last_seen_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
