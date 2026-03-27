"""CRUD operations for listings and price snapshots."""

from datetime import datetime, date

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.models.listing import Listing, PriceSnapshot, MarketStats
from src.models.portfolio import PortfolioDeal


def upsert_listing(session: Session, data: dict) -> Listing:
    """Insert or update a listing by olx_id. Returns the Listing object."""
    listing = session.query(Listing).filter_by(olx_id=data["olx_id"]).first()
    now = datetime.utcnow()

    if listing:
        for key, value in data.items():
            if key != "olx_id" and value is not None:
                setattr(listing, key, value)
        listing.last_seen_at = now
        listing.is_active = True
    else:
        listing = Listing(**data, first_seen_at=now, last_seen_at=now, is_active=True)
        session.add(listing)

    session.flush()
    return listing


def add_price_snapshot(session: Session, listing_id: int, price_eur: float,
                       negotiable: bool = False):
    snap = PriceSnapshot(
        listing_id=listing_id,
        price_eur=price_eur,
        negotiable=negotiable,
        scraped_at=datetime.utcnow(),
    )
    session.add(snap)


def mark_inactive(session: Session, active_olx_ids: set[str]):
    """Mark listings not seen in this scrape as inactive."""
    session.query(Listing).filter(
        Listing.is_active == True,
        ~Listing.olx_id.in_(active_olx_ids),
    ).update({"is_active": False}, synchronize_session="fetch")


def compute_market_stats(session: Session, target_date: date | None = None):
    """Compute and store daily aggregate stats per brand+model."""
    target_date = target_date or date.today()

    stmt = (
        select(
            Listing.brand,
            Listing.model,
            func.min(Listing.year).label("year_from"),
            func.max(Listing.year).label("year_to"),
            func.count(PriceSnapshot.id).label("listing_count"),
            func.avg(PriceSnapshot.price_eur).label("avg_price_eur"),
            func.min(PriceSnapshot.price_eur).label("min_price_eur"),
            func.max(PriceSnapshot.price_eur).label("max_price_eur"),
        )
        .join(PriceSnapshot, PriceSnapshot.listing_id == Listing.id)
        .where(Listing.is_active == True)
        .where(PriceSnapshot.price_eur.isnot(None))
        .group_by(Listing.brand, Listing.model)
    )

    rows = session.execute(stmt).all()

    for row in rows:
        prices = [
            p for (p,) in session.query(PriceSnapshot.price_eur)
            .join(Listing)
            .filter(
                Listing.brand == row.brand,
                Listing.model == row.model,
                Listing.is_active == True,
                PriceSnapshot.price_eur.isnot(None),
            ).all()
        ]
        median = float(pd.Series(prices).median()) if prices else None

        existing = session.query(MarketStats).filter_by(
            brand=row.brand, model=row.model,
            year_from=row.year_from, year_to=row.year_to,
            date=target_date,
        ).first()

        if existing:
            existing.median_price_eur = median
            existing.avg_price_eur = row.avg_price_eur
            existing.min_price_eur = row.min_price_eur
            existing.max_price_eur = row.max_price_eur
            existing.listing_count = row.listing_count
        else:
            session.add(MarketStats(
                brand=row.brand, model=row.model,
                year_from=row.year_from, year_to=row.year_to,
                date=target_date,
                median_price_eur=median,
                avg_price_eur=row.avg_price_eur,
                min_price_eur=row.min_price_eur,
                max_price_eur=row.max_price_eur,
                listing_count=row.listing_count,
            ))

    session.commit()


# ---------------------------------------------------------------------------
# Queries for dashboard
# ---------------------------------------------------------------------------

def get_listings_df(session: Session) -> pd.DataFrame:
    """All listings as DataFrame."""
    q = session.query(Listing).all()
    if not q:
        return pd.DataFrame()
    rows = []
    for l in q:
        snaps = l.price_snapshots.order_by(PriceSnapshot.scraped_at.desc()).all()
        latest_price = snaps[0].price_eur if snaps else None
        first_price = snaps[-1].price_eur if snaps else None
        rows.append({
            "olx_id": l.olx_id, "url": l.url, "title": l.title,
            "brand": l.brand, "model": l.model, "year": l.year,
            "price_eur": latest_price,
            "first_price_eur": first_price,
            "mileage_km": l.mileage_km, "engine_cc": l.engine_cc,
            "fuel_type": l.fuel_type, "horsepower": l.horsepower,
            "transmission": l.transmission, "segment": l.segment,
            "doors": l.doors, "seats": l.seats, "color": l.color,
            "condition": l.condition, "origin": l.origin,
            "city": l.city, "district": l.district,
            "seller_type": l.seller_type, "is_active": l.is_active,
            "first_seen_at": l.first_seen_at,
            "last_seen_at": l.last_seen_at,
        })
    return pd.DataFrame(rows)


def get_price_history_df(session: Session) -> pd.DataFrame:
    """Market stats history as DataFrame."""
    q = session.query(MarketStats).order_by(MarketStats.date).all()
    if not q:
        return pd.DataFrame()
    return pd.DataFrame([{
        "brand": s.brand, "model": s.model, "date": s.date,
        "median_price_eur": s.median_price_eur, "avg_price_eur": s.avg_price_eur,
        "min_price_eur": s.min_price_eur, "max_price_eur": s.max_price_eur,
        "listing_count": s.listing_count,
    } for s in q])


# ---------------------------------------------------------------------------
# Portfolio CRUD
# ---------------------------------------------------------------------------

def add_portfolio_deal(session: Session, data: dict) -> PortfolioDeal:
    deal = PortfolioDeal(**data)
    session.add(deal)
    session.commit()
    return deal


def update_portfolio_deal(session: Session, deal_id: int, data: dict):
    deal = session.query(PortfolioDeal).get(deal_id)
    if deal:
        for k, v in data.items():
            setattr(deal, k, v)
        session.commit()


def delete_portfolio_deal(session: Session, deal_id: int):
    deal = session.query(PortfolioDeal).get(deal_id)
    if deal:
        session.delete(deal)
        session.commit()


def get_portfolio_df(session: Session) -> pd.DataFrame:
    q = session.query(PortfolioDeal).order_by(PortfolioDeal.buy_date.desc()).all()
    if not q:
        return pd.DataFrame()
    rows = []
    for d in q:
        total_cost = (d.buy_price_eur or 0) + (d.repair_cost_eur or 0) + (d.registration_cost_eur or 0)
        gross_profit = (d.sell_price_eur - total_cost) if d.sell_price_eur else None
        roi = (gross_profit / total_cost * 100) if gross_profit is not None and total_cost > 0 else None
        days_inv = (d.sell_date - d.buy_date).days if d.sell_date and d.buy_date else None
        rows.append({
            "id": d.id,
            "brand": d.brand, "model": d.model, "year": d.year,
            "mileage_km": d.mileage_km, "fuel_type": d.fuel_type,
            "transmission": d.transmission, "color": d.color, "district": d.district,
            "buy_date": d.buy_date, "buy_price_eur": d.buy_price_eur,
            "repair_cost_eur": d.repair_cost_eur or 0,
            "registration_cost_eur": d.registration_cost_eur or 0,
            "total_cost_eur": total_cost,
            "sell_date": d.sell_date, "sell_price_eur": d.sell_price_eur,
            "gross_profit_eur": gross_profit,
            "roi_pct": round(roi, 1) if roi is not None else None,
            "days_in_inventory": days_inv,
            "notes": d.notes, "olx_listing_id": d.olx_listing_id,
        })
    return pd.DataFrame(rows)
