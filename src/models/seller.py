"""Seller account model for OLX user-profile features.

One row per OLX seller (private "Utilizador" or business shop). The same
seller is reused across all of their car listings via ``Listing.seller_uuid``.

Refresh model: ``profile_fetched_at`` gates re-fetching the profile page
in the scraper — multi-car sellers don't need a fresh hit per listing.
The TTL is enforced in the scraper, not here, so the table stays a pure
snapshot of the last successful fetch.
"""

from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, Text

from src.models.listing import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Seller(Base):
    __tablename__ = "sellers"

    uuid = Column(String, primary_key=True)
    short_id = Column(String, index=True)         # /ads/user/{short_id}/ slug
    shop_slug = Column(String, index=True)        # {slug}.olx.pt for businesses
    profile_url = Column(Text, nullable=False)

    name = Column(String)
    is_business = Column(Boolean)                  # JSON-truth, NOT page label
    business_type = Column(String)                 # only set on shop pages
    created_at = Column(DateTime)                  # account registration
    last_seen_at = Column(DateTime)
    last_login_at = Column(DateTime)

    total_ads = Column(Integer)                    # total_elements snapshot
    ads_by_category = Column(Text)                 # JSON: {cat_id: count}

    # Derived feature buckets (from olx_categories.categorise_facets).
    # Stored alongside the raw facets so dashboards/scoring can read them
    # without re-rolling up on every query — they only change when the
    # profile is re-fetched.
    cars_count = Column(Integer)
    parts_count = Column(Integer)
    commercial_count = Column(Integer)
    motos_count = Column(Integer)
    boats_count = Column(Integer)
    other_auto_count = Column(Integer)
    non_auto_count = Column(Integer)
    distinct_car_brands = Column(Integer)

    # Non-automotive sub-buckets — see olx_categories.NON_AUTO_BUCKETS for
    # the OLX top-level ids each one rolls up. These let modelling code
    # distinguish "family selling personal stuff" (family_lifestyle) from
    # "tech reseller dabbling in cars" (electronics) etc., instead of
    # collapsing every non-car ad into a single ``non_auto_count``.
    family_lifestyle_count = Column(Integer)
    electronics_count = Column(Integer)
    realestate_count = Column(Integer)
    tools_industrial_count = Column(Integer)
    pets_hobby_count = Column(Integer)
    services_jobs_count = Column(Integer)

    # Identity / trust signals from the profile JSON.
    social_account_type = Column(String)        # facebook / google / apple / NULL
    has_user_photo = Column(Boolean)
    position_lat = Column(Float)
    position_lon = Column(Float)

    profile_fetched_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)
