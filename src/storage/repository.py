"""CRUD operations for listings and price snapshots."""

from datetime import datetime, date, timezone


def _utcnow() -> datetime:
    """Timezone-aware UTC now, then stripped to naive for schema compat.

    SQLAlchemy columns were defined as naive datetime, so we keep writes
    naive but use timezone-aware arithmetic to avoid Python 3.12+
    DeprecationWarning on `datetime.utcnow()`.
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.models.listing import Listing, PriceSnapshot, MarketStats, UnmatchedListing
from src.models.portfolio import PortfolioDeal


def upsert_listing(session: Session, data: dict) -> Listing:
    """Insert or update a listing by olx_id. Returns the Listing object."""
    listing = session.query(Listing).filter_by(olx_id=data["olx_id"]).first()
    now = _utcnow()
    # Use the site-parsed date if available, fall back to scrape time
    posted_at = data.pop("posted_at", None)
    seen_at = posted_at or now

    if listing:
        for key, value in data.items():
            if key != "olx_id" and value is not None:
                setattr(listing, key, value)
        listing.last_seen_at = seen_at
        listing.is_active = True
        listing.deactivated_at = None
        listing.deactivation_reason = None
    else:
        listing = Listing(**data, first_seen_at=seen_at, last_seen_at=seen_at, is_active=True)
        session.add(listing)

    session.flush()
    return listing


def add_price_snapshot(session: Session, listing_id: int, price_eur: float,
                       negotiable: bool = False):
    snap = PriceSnapshot(
        listing_id=listing_id,
        price_eur=price_eur,
        negotiable=negotiable,
        scraped_at=_utcnow(),
    )
    session.add(snap)


def upsert_unmatched(session: Session, data: dict, reason: str) -> UnmatchedListing:
    """Insert or update an unmatched listing."""
    row = session.query(UnmatchedListing).filter_by(olx_id=data["olx_id"]).first()
    now = _utcnow()
    fields = {
        "url": data.get("url", ""),
        "title": data.get("title"),
        "brand": data.get("brand", ""),
        "model": data.get("model", ""),
        "year": data.get("year"),
        "price_eur": data.get("price_eur"),
        "mileage_km": data.get("mileage_km"),
        "fuel_type": data.get("fuel_type"),
        "city": data.get("city"),
        "district": data.get("district"),
        "seller_type": data.get("seller_type"),
        "description": data.get("description"),
        "reason": reason,
        "source": data.get("source", "olx"),
    }
    if row:
        for k, v in fields.items():
            if v is not None:
                setattr(row, k, v)
        row.last_seen_at = now
        row.is_active = True
    else:
        row = UnmatchedListing(olx_id=data["olx_id"], **fields,
                               first_seen_at=now, last_seen_at=now, is_active=True)
        session.add(row)
    session.flush()
    return row


def get_enriched_hashes(session: Session) -> dict[str, str]:
    """Return {olx_id: description_hash} for listings that have llm_extras."""
    rows = session.query(Listing.olx_id, Listing.llm_description_hash).filter(
        Listing.llm_extras.isnot(None),
        Listing.llm_description_hash.isnot(None),
    ).all()
    return {r[0]: r[1] for r in rows}


def get_duplicate_ids(session: Session) -> set[str]:
    """Return olx_ids of listings marked as duplicates."""
    rows = session.query(Listing.olx_id).filter(
        Listing.duplicate_of.isnot(None),
    ).all()
    return {r[0] for r in rows}


def mark_inactive(session: Session, active_olx_ids: set[str]):
    """Mark listings not seen in this scrape as inactive."""
    import logging
    log = logging.getLogger(__name__)
    now = _utcnow()
    count = session.query(Listing).filter(
        Listing.is_active == True,
        ~Listing.olx_id.in_(active_olx_ids),
    ).update({
        "is_active": False,
        "deactivated_at": now,
        "deactivation_reason": "sold",
    }, synchronize_session="evaluate")
    log.info("Marked %d listings as inactive", count)


def backfill_deactivated_at(session: Session) -> int:
    """Backfill deactivated_at from last_seen_at for old inactive listings."""
    import logging
    log = logging.getLogger(__name__)
    rows = session.query(Listing).filter(
        Listing.is_active == False,
        Listing.deactivated_at.is_(None),
    ).all()
    for listing in rows:
        listing.deactivated_at = listing.last_seen_at
        listing.deactivation_reason = "sold"
    if rows:
        session.commit()
    log.info("Backfilled deactivated_at for %d listings", len(rows))
    return len(rows)


_MERGE_FIELDS = [
    "model", "engine_cc", "fuel_type", "horsepower", "transmission",
    "segment", "doors", "seats", "color", "condition", "drive_type",
    "registration_month", "city", "district",
    "seller_type", "description", "llm_extras", "llm_description_hash",
    "desc_mentions_repair", "desc_mentions_accident", "real_mileage_km", "desc_mentions_num_owners",
    "desc_mentions_customs_cleared", "right_hand_drive", "generation",
]


def _merge_into_canonical(canonical: "Listing", duplicate: "Listing"):
    """Fill empty fields on the canonical listing from the duplicate."""
    for field in _MERGE_FIELDS:
        canon_val = getattr(canonical, field, None)
        dup_val = getattr(duplicate, field, None)
        if (canon_val is None or canon_val == "") and dup_val not in (None, ""):
            setattr(canonical, field, dup_val)


def deduplicate_cross_platform(session: Session) -> int:
    """Detect likely duplicates between OLX and StandVirtual.

    When the same car is manually posted on both platforms, match by
    (brand, model, year, mileage ±10%, price ±10%, district).
    The earlier-seen listing is canonical; the duplicate gets ``duplicate_of`` set.
    Missing attributes on the canonical listing are filled from the duplicate.
    Returns the number of newly marked duplicates.
    """
    import logging
    log = logging.getLogger("dedup")

    active = session.query(Listing).filter(
        Listing.is_active == True,
        Listing.duplicate_of.is_(None),
        Listing.brand != "",
        Listing.model != "",
        Listing.year.isnot(None),
        Listing.mileage_km.isnot(None),
        Listing.mileage_km != 0,
    ).all()

    log.info("Dedup: loaded %d active candidates", len(active))

    # Pre-fetch latest price per listing in one query (avoid N+1 lazy loads)
    latest_sub = (
        session.query(
            PriceSnapshot.listing_id,
            func.max(PriceSnapshot.scraped_at).label("max_at"),
        )
        .group_by(PriceSnapshot.listing_id)
        .subquery()
    )
    price_rows = (
        session.query(PriceSnapshot.listing_id, PriceSnapshot.price_eur)
        .join(latest_sub,
              (PriceSnapshot.listing_id == latest_sub.c.listing_id)
              & (PriceSnapshot.scraped_at == latest_sub.c.max_at))
        .all()
    )
    latest_prices: dict[int, float | None] = {lid: price for lid, price in price_rows}

    by_key: dict[tuple, list[Listing]] = {}
    for l in active:
        # Group by (brand, model, year, district) for candidate pairs
        key = (l.brand.lower(), (l.model or "").lower(), l.year, (l.district or "").lower())
        by_key.setdefault(key, []).append(l)

    marked = 0
    for key, group in by_key.items():
        if len(group) < 2:
            continue
        # Only check cross-platform pairs
        olx_listings = [l for l in group if (l.source or "olx") == "olx"]
        sv_listings = [l for l in group if (l.source or "olx") == "standvirtual"]
        if not olx_listings or not sv_listings:
            continue

        for sv in sv_listings:
            for olx in olx_listings:
                if sv.duplicate_of or olx.duplicate_of:
                    continue
                # Mileage within 10% (require both to have mileage)
                if not sv.mileage_km or not olx.mileage_km:
                    continue
                ratio = sv.mileage_km / olx.mileage_km
                if not (0.9 <= ratio <= 1.1):
                    continue
                # Price within 10% (use pre-fetched latest prices)
                sv_price = latest_prices.get(sv.id)
                olx_price = latest_prices.get(olx.id)
                if sv_price and olx_price:
                    p_ratio = sv_price / olx_price
                    if not (0.9 <= p_ratio <= 1.1):
                        continue
                # Match! Keep the most recently updated as canonical
                if (olx.last_seen_at or datetime.min) >= (sv.last_seen_at or datetime.min):
                    canonical, duplicate = olx, sv
                else:
                    canonical, duplicate = sv, olx
                duplicate.duplicate_of = canonical.olx_id
                _merge_into_canonical(canonical, duplicate)
                log.info("Dedup: %s %s is duplicate of %s %s (%s %s %s)",
                         duplicate.source, duplicate.olx_id,
                         canonical.source, canonical.olx_id,
                         sv.brand, sv.model, sv.year)
                marked += 1
                break  # each SV listing matches at most one OLX listing

    if marked:
        session.commit()
        log.info("Dedup: marked %d cross-platform duplicates", marked)
    return marked


def compute_market_stats(
    session: Session,
    target_date: date | None = None,
    changed_pairs: set[tuple[str, str]] | None = None,
):
    """Compute and store daily aggregate stats per brand+model.

    If *changed_pairs* is provided, only recompute stats for those
    (brand, model) pairs — much faster during incremental scraping.
    Pass None (or omit) for a full recompute.
    """
    import logging
    from statistics import median as _median
    log = logging.getLogger(__name__)

    target_date = target_date or date.today()

    query = (
        session.query(
            Listing.brand, Listing.model, Listing.year, PriceSnapshot.price_eur,
        )
        .join(PriceSnapshot, PriceSnapshot.listing_id == Listing.id)
        .filter(Listing.is_active == True, PriceSnapshot.price_eur.isnot(None))
    )
    if changed_pairs:
        from sqlalchemy import tuple_
        query = query.filter(
            tuple_(Listing.brand, Listing.model).in_(changed_pairs)
        )

    rows = query.all()
    log.info("Market stats: fetched %d price rows%s", len(rows),
             f" for {len(changed_pairs)} pairs" if changed_pairs else " (full)")

    # Aggregate in Python — no N+1 queries
    groups: dict[tuple[str, str], dict] = {}
    for brand, model, year, price in rows:
        key = (brand, model)
        if key not in groups:
            groups[key] = {"prices": [], "years": []}
        groups[key]["prices"].append(float(price))
        if year is not None:
            groups[key]["years"].append(year)

    # Pre-fetch existing MarketStats for target_date (only relevant pairs)
    existing_query = session.query(MarketStats).filter_by(date=target_date)
    if changed_pairs:
        existing_query = existing_query.filter(
            tuple_(MarketStats.brand, MarketStats.model).in_(changed_pairs)
        )
    existing_map: dict[tuple, MarketStats] = {}
    for ms in existing_query.all():
        existing_map[(ms.brand, ms.model, ms.year_from, ms.year_to)] = ms

    log.info("Market stats: computing for %d brand+model groups...", len(groups))

    for (brand, model), data in groups.items():
        prices = data["prices"]
        years = data["years"]

        median_price = _median(prices) if prices else None
        avg_price = sum(prices) / len(prices) if prices else None
        min_price = min(prices) if prices else None
        max_price = max(prices) if prices else None
        count = len(prices)
        year_from = min(years) if years else None
        year_to = max(years) if years else None

        stats_key = (brand, model, year_from, year_to)
        existing = existing_map.get(stats_key)

        if existing:
            existing.median_price_eur = median_price
            existing.avg_price_eur = avg_price
            existing.min_price_eur = min_price
            existing.max_price_eur = max_price
            existing.listing_count = count
        else:
            ms = MarketStats(
                brand=brand, model=model,
                year_from=year_from, year_to=year_to,
                date=target_date,
                median_price_eur=median_price,
                avg_price_eur=avg_price,
                min_price_eur=min_price,
                max_price_eur=max_price,
                listing_count=count,
            )
            session.add(ms)
            existing_map[stats_key] = ms

    session.commit()
    log.info("Market stats: saved %d groups for %s", len(groups), target_date)


# ---------------------------------------------------------------------------
# Queries for dashboard
# ---------------------------------------------------------------------------

def get_listings_df(session: Session) -> pd.DataFrame:
    """All listings as DataFrame.

    Batches the price-snapshot load into one query keyed by listing id,
    replacing the old N+1 lazy-relationship access that fired one SELECT
    per listing on the hot dashboard-load path.
    """
    q = session.query(Listing).all()
    if not q:
        return pd.DataFrame()

    listing_ids = [l.id for l in q]
    snap_rows = (
        session.query(PriceSnapshot)
        .filter(PriceSnapshot.listing_id.in_(listing_ids))
        .order_by(PriceSnapshot.listing_id, PriceSnapshot.scraped_at.desc())
        .all()
    )
    snaps_by_listing: dict[int, list] = {}
    for s in snap_rows:
        snaps_by_listing.setdefault(s.listing_id, []).append(s)

    rows = []
    for l in q:
        snaps = snaps_by_listing.get(l.id, [])
        latest_price = snaps[0].price_eur if snaps else None
        first_price = snaps[-1].price_eur if snaps else None

        # --- Price dynamics from snapshot history ---
        num_price_drops = 0
        max_drop_pct = 0.0
        price_drop_velocity = None
        days_since_last_drop = None
        if len(snaps) >= 2:
            last_drop_at = None
            # snaps are newest-first; walk from oldest to newest
            for i in range(len(snaps) - 1, 0, -1):
                older_price = snaps[i].price_eur
                newer_price = snaps[i - 1].price_eur
                if newer_price < older_price and older_price > 0:
                    num_price_drops += 1
                    drop_pct = (older_price - newer_price) / older_price * 100
                    if drop_pct > max_drop_pct:
                        max_drop_pct = drop_pct
                    last_drop_at = snaps[i - 1].scraped_at
            if num_price_drops > 0 and first_price and first_price > 0:
                total_drop_pct = max((first_price - latest_price) / first_price * 100, 0)
                first_ts = snaps[-1].scraped_at
                last_ts = snaps[0].scraped_at
                span_days = max((last_ts - first_ts).total_seconds() / 86400, 1)
                price_drop_velocity = round(total_drop_pct / span_days, 3)
            if last_drop_at is not None:
                days_since_last_drop = max(
                    (snaps[0].scraped_at - last_drop_at).total_seconds() / 86400, 0
                )

        rows.append({
            "olx_id": l.olx_id, "url": l.url, "title": l.title,
            "brand": l.brand, "model": l.model, "year": l.year,
            "price_eur": latest_price,
            "first_price_eur": first_price,
            "num_price_drops": num_price_drops,
            "max_drop_pct": round(max_drop_pct, 1),
            "price_drop_velocity": price_drop_velocity,
            "days_since_last_drop": round(days_since_last_drop, 1) if days_since_last_drop is not None else None,
            "mileage_km": l.mileage_km, "engine_cc": l.engine_cc,
            "fuel_type": l.fuel_type, "horsepower": l.horsepower,
            "transmission": l.transmission, "segment": l.segment,
            "doors": l.doors, "seats": l.seats, "color": l.color,
            "condition": l.condition, "drive_type": l.drive_type,
            "photo_count": l.photo_count, "description_length": l.description_length,
            "city": l.city, "district": l.district,
            "seller_type": l.seller_type, "is_active": l.is_active,
            # description text itself is intentionally excluded — it can be
            # hundreds of KB × thousands of rows and no consumer of this df
            # needs the raw text. Anything that wants to gate on presence
            # uses the separately-stored description_length column.
            "llm_extras": l.llm_extras,
            "first_seen_at": l.first_seen_at,
            "last_seen_at": l.last_seen_at,
            # Enrichment columns
            "sub_model": l.sub_model,
            "trim_level": l.trim_level,
            "desc_mentions_repair": l.desc_mentions_repair,
            "desc_mentions_accident": l.desc_mentions_accident,
            "real_mileage_km": l.real_mileage_km,
            "desc_mentions_num_owners": l.desc_mentions_num_owners,
            "desc_mentions_customs_cleared": l.desc_mentions_customs_cleared,
            "right_hand_drive": l.right_hand_drive,
            "mechanical_condition": l.mechanical_condition,
            "urgency": l.urgency,
            "warranty": l.warranty,
            "tuning_or_mods": l.tuning_or_mods,
            "taxi_fleet_rental": l.taxi_fleet_rental,
            "first_owner_selling": l.first_owner_selling,
            "source": l.source or "olx",
            "duplicate_of": l.duplicate_of,
            "deactivated_at": l.deactivated_at,
            "deactivation_reason": l.deactivation_reason,
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


def get_unmatched_df(session: Session) -> pd.DataFrame:
    """Unmatched listings as DataFrame."""
    q = session.query(UnmatchedListing).filter(UnmatchedListing.is_active == True).all()
    if not q:
        return pd.DataFrame()
    return pd.DataFrame([{
        "olx_id": u.olx_id, "url": u.url, "title": u.title,
        "brand": u.brand, "model": u.model, "year": u.year,
        "price_eur": u.price_eur, "mileage_km": u.mileage_km,
        "fuel_type": u.fuel_type, "city": u.city, "district": u.district,
        "reason": u.reason, "source": u.source or "olx",
        "first_seen_at": u.first_seen_at,
    } for u in q])


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
