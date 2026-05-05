"""Seller-profile extraction for olx.pt.

Two surfaces are parsed:

* The **listing detail page** (already fetched by ``OlxScraper``). The
  link to the seller and a few cheap-to-grab fields live there:

    * ``data-testid="user-profile-link"`` → href
    * ``data-testid="user-profile-user-name"`` → display name
    * ``data-testid="member-since"`` → "No OLX desde {month} de {year}"
    * ``data-testid="trader-title"`` → "Utilizador" / "Empresa" /
      "Particular" — what the seller advertises as on this listing.
      NB: this DOES NOT match the JSON ``is_business`` flag for some
      sellers (verified 2026-05-05: a business with ``is_business=True``
      still renders ``trader-title="Utilizador"``); the discrepancy is
      itself a useful "pseudo-private" signal at modelling time.

* The **seller profile page** (``/ads/user/{id}/`` or the business-shop
  variant ``{slug}.olx.pt/home/``). The body is empty SSR'd HTML; all
  data lives in ``window.__PRERENDERED_STATE__`` as a JSON-encoded
  string. ``userListing`` and ``shop`` are the two top-level shapes;
  this module flattens them into a single :class:`SellerProfile`.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


_PT_MONTHS = {
    "janeiro": 1, "fevereiro": 2, "março": 3, "marco": 3, "abril": 4,
    "maio": 5, "junho": 6, "julho": 7, "agosto": 8, "setembro": 9,
    "outubro": 10, "novembro": 11, "dezembro": 12,
}


@dataclass
class SellerLink:
    """Seller pointer extracted from a listing detail page."""
    profile_url: str
    short_id: str | None = None        # /ads/user/{short_id}/
    shop_slug: str | None = None       # {slug}.olx.pt
    display_name: str | None = None
    displayed_as: str | None = None    # raw trader-title text on listing
    member_since_text: str | None = None
    member_since: datetime | None = None


@dataclass
class SellerProfile:
    """Aggregated seller data parsed from the profile page."""
    uuid: str
    profile_url: str
    short_id: str | None = None
    shop_slug: str | None = None
    name: str | None = None
    is_business: bool | None = None
    business_type: str | None = None   # only set on business-shop pages
    created_at: datetime | None = None
    last_seen_at: datetime | None = None
    last_login_at: datetime | None = None
    total_ads: int = 0
    facets: dict[int, int] = field(default_factory=dict)
    categories_list: dict[str, dict] = field(default_factory=dict)

    # Identity / trust signals. ``social_account_type`` is the strongest
    # of the three — a Facebook-linked account is harder to spin up for
    # one-off scams than a phone-only registration. ``has_user_photo``
    # is a weaker but still present-vs-absent signal: ``show_photo=True``
    # alone doesn't mean a real avatar was uploaded (we saw 3/4 private
    # sellers with show_photo=True and user_photo=None — i.e. default
    # OLX SVG avatar), so we key off ``user_photo is not None`` instead.
    # Position is stored as separate float columns so distance-based
    # features can join against listing/district coordinates later.
    social_account_type: str | None = None  # "facebook" / "google" / "apple" / None
    has_user_photo: bool = False
    position_lat: float | None = None
    position_lon: float | None = None


# ---------------------------------------------------------------------------
# Listing-detail extraction
# ---------------------------------------------------------------------------

_USER_ADS_RE = re.compile(r"/ads/user/([^/]+)/?")
_SHOP_HOST_RE = re.compile(r"^https?://([a-z0-9-]+)\.olx\.pt/", re.I)
_MEMBER_SINCE_RE = re.compile(
    r"desde\s*"
    r"(?P<month>janeiro|fevereiro|mar[çc]o|abril|maio|junho|julho|agosto|"
    r"setembro|outubro|novembro|dezembro)"
    r"\s*de\s*(?P<year>\d{4})",
    re.I,
)


def parse_seller_link(detail_soup: BeautifulSoup) -> SellerLink | None:
    """Extract the seller pointer from a parsed listing detail page.

    Returns ``None`` if no ``user-profile-link`` element is present —
    every recent OLX detail page carries one, so absence is a parser
    breakage signal callers should log rather than silently skip.
    """
    link = detail_soup.find(attrs={"data-testid": "user-profile-link"})
    if not link or not link.get("href"):
        return None

    href = link["href"]
    profile_url = href if href.startswith("http") else f"https://www.olx.pt{href}"

    short_id = None
    shop_slug = None
    m = _USER_ADS_RE.search(profile_url)
    if m:
        short_id = m.group(1)
    else:
        h = _SHOP_HOST_RE.match(profile_url)
        if h and h.group(1) != "www":
            shop_slug = h.group(1).lower()

    name_el = detail_soup.find(attrs={"data-testid": "user-profile-user-name"})
    display_name = name_el.get_text(strip=True) if name_el else None

    trader_el = detail_soup.find(attrs={"data-testid": "trader-title"})
    displayed_as = trader_el.get_text(strip=True) if trader_el else None

    member_el = detail_soup.find(attrs={"data-testid": "member-since"})
    member_text = member_el.get_text(" ", strip=True) if member_el else None
    member_since = _parse_member_since(member_text) if member_text else None

    return SellerLink(
        profile_url=profile_url,
        short_id=short_id,
        shop_slug=shop_slug,
        display_name=display_name,
        displayed_as=displayed_as,
        member_since_text=member_text,
        member_since=member_since,
    )


def _parse_member_since(text: str) -> datetime | None:
    """Parse 'No OLX desde novembro de 2019' → datetime(2019, 11, 1).

    Day defaults to the 1st — the listing detail page only shows month
    granularity. For exact account-age use :attr:`SellerProfile.created_at`
    parsed from the profile JSON instead.
    """
    if not text:
        return None
    m = _MEMBER_SINCE_RE.search(text)
    if not m:
        return None
    month_name = m.group("month").lower().replace("ç", "c")
    month = _PT_MONTHS.get(month_name)
    if not month:
        return None
    return datetime(int(m.group("year")), month, 1)


# ---------------------------------------------------------------------------
# Profile-page extraction (__PRERENDERED_STATE__)
# ---------------------------------------------------------------------------

_STATE_PREFIX_RE = re.compile(r'window\.__PRERENDERED_STATE__\s*=\s*"')


def extract_prerendered_state(html: str) -> dict | None:
    """Pull and decode the ``window.__PRERENDERED_STATE__`` JSON blob.

    The blob is a JSON-encoded JSON string assigned in a ``<script>``
    tag: ``window.__PRERENDERED_STATE__= "{\\"userListing\\":{...}}";``.
    Walk the source character-by-character to find the closing quote
    while honouring backslash escapes, then ``json.loads`` twice (once
    to unwrap the string literal, once to parse the inner JSON).

    Returns ``None`` when the blob is missing or malformed — the caller
    decides whether that's a hard failure or a "skip this seller and
    move on" event. We never raise, because OLX has been known to ship
    A/B variants with subtly different SSR shapes; we'd rather lose one
    seller than crash the whole enrichment batch.
    """
    if not html:
        return None
    m = _STATE_PREFIX_RE.search(html)
    if not m:
        return None
    quote_start = m.end() - 1
    quote_end = _find_closing_quote(html, quote_start)
    if quote_end is None:
        return None
    inner_literal = html[quote_start: quote_end + 1]
    try:
        decoded_str = json.loads(inner_literal)
        return json.loads(decoded_str)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Failed to decode __PRERENDERED_STATE__: %s", e)
        return None


def _find_closing_quote(s: str, start: int) -> int | None:
    """Return the index of the unescaped closing ``"`` matching ``s[start]``."""
    if start >= len(s) or s[start] != '"':
        return None
    i = start + 1
    n = len(s)
    while i < n:
        c = s[i]
        if c == "\\":
            i += 2
            continue
        if c == '"':
            return i
        i += 1
    return None


def parse_seller_profile_html(
    html: str,
    profile_url: str = "",
) -> SellerProfile | None:
    """Parse a seller profile / business shop page into a SellerProfile.

    Handles both shapes that OLX ships under ``__PRERENDERED_STATE__``:

    * ``userListing`` — private and most professional sellers, accessed
      via ``/ads/user/{short_id}/``.
    * ``shop`` — branded business shops with their own subdomain
      (``{slug}.olx.pt/home/``). Field layout differs but the ad-count
      facets live at the same path inside ``adsOffers.metadata.facets``.
    """
    state = extract_prerendered_state(html)
    if not state:
        return None

    if "userListing" in state:
        return _from_user_listing(state, profile_url)
    if "shop" in state:
        return _from_shop(state, profile_url)
    logger.warning(
        "Unknown __PRERENDERED_STATE__ shape (top keys=%s)",
        list(state.keys()),
    )
    return None


def _short_id_from_url(url: str) -> str | None:
    if not url:
        return None
    m = _USER_ADS_RE.search(url)
    return m.group(1) if m else None


def _shop_slug_from_url(url: str) -> str | None:
    if not url:
        return None
    m = _SHOP_HOST_RE.match(url)
    if m and m.group(1) != "www":
        return m.group(1).lower()
    return None


def _facets_from_metadata(metadata: dict) -> dict[int, int]:
    raw = (metadata or {}).get("facets", {}) or {}
    cat = raw.get("category") or []
    out: dict[int, int] = {}
    for f in cat:
        cid = f.get("id")
        cnt = f.get("count")
        if isinstance(cid, int) and isinstance(cnt, int):
            out[cid] = cnt
    return out


def _parse_iso(value) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _from_user_listing(state: dict, profile_url: str) -> SellerProfile | None:
    ul = state.get("userListing") or {}
    seller_data = ((ul.get("seller") or {}).get("data")) or {}
    if not seller_data.get("uuid"):
        return None
    md = ((ul.get("adsOffers") or {}).get("metadata")) or {}
    position = seller_data.get("position") or {}
    return SellerProfile(
        uuid=seller_data["uuid"],
        profile_url=profile_url or seller_data.get("user_ads_url", ""),
        short_id=_short_id_from_url(
            profile_url or seller_data.get("user_ads_url", "")
        ),
        shop_slug=None,
        name=seller_data.get("name"),
        is_business=bool(seller_data.get("is_business")),
        business_type=None,
        created_at=_parse_iso(seller_data.get("created")),
        last_seen_at=_parse_iso(seller_data.get("last_seen")),
        last_login_at=_parse_iso(seller_data.get("last_login")),
        total_ads=int(md.get("total_elements") or 0),
        facets=_facets_from_metadata(md),
        categories_list=(state.get("categories") or {}).get("list") or {},
        social_account_type=seller_data.get("social_network_account_type"),
        has_user_photo=bool(seller_data.get("user_photo")),
        position_lat=_safe_float(position.get("map_lat")),
        position_lon=_safe_float(position.get("map_lon")),
    )


def _from_shop(state: dict, profile_url: str) -> SellerProfile | None:
    shop = state.get("shop") or {}
    sd = ((shop.get("shop") or {}).get("data")) or {}
    if not sd.get("owner_uuid"):
        return None
    details = sd.get("details") or {}
    md = ((shop.get("adsOffers") or {}).get("metadata")) or {}
    user_ads_url = details.get("user_ads_url") or profile_url
    address = sd.get("address") or {}
    return SellerProfile(
        uuid=sd["owner_uuid"],
        profile_url=profile_url or user_ads_url,
        short_id=_short_id_from_url(user_ads_url),
        shop_slug=_shop_slug_from_url(profile_url) or (sd.get("domain") or "").lower() or None,
        name=sd.get("name"),
        is_business=True,  # shop pages are by definition business sellers
        business_type=details.get("business_type"),
        created_at=_parse_iso(details.get("created")),
        last_seen_at=_parse_iso(details.get("last_seen")),
        last_login_at=None,
        total_ads=int(md.get("total_elements") or 0),
        facets=_facets_from_metadata(md),
        categories_list=(state.get("categories") or {}).get("list") or {},
        # Business shops don't expose social_network_account_type or a
        # standalone user_photo (the logo is at sd['logo'] but is the
        # company brand, not personal identity). Position is sometimes
        # absent — only fill if we actually got coordinates back.
        social_account_type=None,
        has_user_photo=bool(sd.get("logo")),
        position_lat=_safe_float(address.get("map_lat")),
        position_lon=_safe_float(address.get("map_lon")),
    )
