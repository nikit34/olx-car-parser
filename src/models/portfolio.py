"""Portfolio deal tracking model for car flipping."""

from datetime import datetime

from sqlalchemy import Column, Date, DateTime, Float, Integer, String, Text

from src.models.listing import Base


class PortfolioDeal(Base):
    __tablename__ = "portfolio_deals"

    id = Column(Integer, primary_key=True)
    brand = Column(String, nullable=False)
    model = Column(String, nullable=False)
    year = Column(Integer)
    mileage_km = Column(Integer)
    fuel_type = Column(String)
    transmission = Column(String)
    color = Column(String)
    district = Column(String)
    buy_date = Column(Date, nullable=False)
    buy_price_eur = Column(Float, nullable=False)
    repair_cost_eur = Column(Float, default=0.0)
    registration_cost_eur = Column(Float, default=0.0)
    sell_date = Column(Date)
    sell_price_eur = Column(Float)
    notes = Column(Text)
    olx_listing_id = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
