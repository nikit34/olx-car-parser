"""OLX.pt car listings scraper.

Uses httpx (NOT Playwright) for HTTP requests.
DO NOT replace httpx with Playwright — Playwright requires browser binaries
that are not available on CI runners, and OLX blocks datacenter IPs regardless.
"""

import json
import logging
import random
import re
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import httpx
from bs4 import BeautifulSoup

from src.parser.seller_profile import (
    SellerProfile,
    parse_seller_link,
    parse_seller_profile_html,
)

logger = logging.getLogger(__name__)


class ScraperParseError(RuntimeError):
    """Raised when the SERP parser returns 0 cards across multiple pages
    and we haven't collected any listings yet — almost certainly a source-
    side change (HTML restructured, encoding flipped, bot wall added).

    The 2026-04 OLX outage was exactly this failure mode: ``Accept-Encoding:
    br`` started getting Brotli responses that httpx couldn't decode, the
    parser silently saw binary garbage, and the loop returned ``[]`` page
    after page for ten days while StandVirtual kept working. Loud-fail so
    the cron exits non-zero and a human notices the next morning instead
    of when the OLX-side database has gone fully stale.
    """

USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0",
]

BASE_URL = "https://www.olx.pt/carros-motos-e-barcos/carros/"

# Internal JSON API the OLX frontend itself calls. Undocumented but
# auth-free and far more robust than HTML scraping: the list payload
# already carries every structured param, the full description, the
# photo list, the seller, lifecycle timestamps and the cross-post flag,
# so no per-listing detail fetch is needed. ``v2`` does not exist.
OLX_API_URL = "https://www.olx.pt/api/v1/offers/"
# Category id for "Carros" (cars). Brands are *sub-categories* of this
# (e.g. BMW = 741); price bands use the bare ``filter_float_price:from/to``
# query params. A single query is hard-capped at ~1000 results regardless
# of HTML-vs-API, which is why full coverage needs query segmentation.
CARS_CATEGORY_ID = 378
_API_PAGE_SIZE = 40
# OLX caps offset at ~1000 (offset 1040 → HTTP 400). Stop paging a segment
# once we reach it; full coverage comes from price-band bisection instead.
_API_OFFSET_CAP = 1000
# Concurrency for parallel page fetching (OLX bands + SV GraphQL pages).
# 12 concurrent OLX API requests verified to return all 200 (no rate-limit);
# the bounded pool is the rate limit, so no per-request delay is applied.
_PARALLEL_FETCH_WORKERS = 8

# StandVirtual GraphQL "listingScreen" — focused JSON (id, url, price,
# sellerUUID, createdAt, params + totalCount), far lighter than the 2.4 MB
# SSR page, and paginates with no per-query cap (verified to page 800+).
# The persisted-query hash rotates on SV frontend deploys; on miss we fall
# back to SSR ``__NEXT_DATA__`` parsing.
SV_GRAPHQL_URL = "https://www.standvirtual.com/graphql"
SV_LISTING_SCREEN_HASH = "5f9903c01d8e8b50a496ef5b10ce0ca397c85f795b158449db3492e6e8acb364"
SV_CARS_CATEGORY_ID = "29"
SV_PAGE_SIZE = 32
SV_LISTING_PARAMS = [
    "origin", "make", "version", "model", "engine_code", "fuel_type",
    "gearbox", "mileage", "engine_capacity", "engine_power",
    "first_registration_year", "year",
]

KNOWN_BRANDS = [
    "Alfa Romeo", "Audi", "BMW", "Chevrolet", "Chrysler", "Citroen", "Citroën",
    "Cupra", "Dacia", "DS", "Fiat", "Ford", "Honda", "Hyundai", "Jaguar", "Jeep",
    "Kia", "Land Rover", "Lexus", "Mazda", "Mercedes-Benz", "Mini", "Mitsubishi",
    "Nissan", "Opel", "Peugeot", "Porsche", "Renault", "Seat", "Skoda",
    "Smart", "Subaru", "Suzuki", "Tesla", "Toyota", "Volkswagen", "Volvo",
]

PARAM_LABEL_MAP = {
    "Segmento": "segment",
    "Ano": "year",
    "Modelo": "model",
    "Mês de Registo": "registration_month",
    "Cilindrada": "engine_cc",
    "Combustível": "fuel_type",
    "Potência": "horsepower",
    "Quilómetros": "mileage_km",
    "Tipo de Caixa": "transmission",
    "Condição": "condition",
    "Portas": "doors",
    "Lugares": "seats",
    "Cor": "color",
    "Tração": "drive_type",
    "Marca": "brand",
}


@dataclass
class ScraperConfig:
    base_url: str = BASE_URL
    max_pages: int = 50
    delay_min: float = 3.0
    delay_max: float = 7.0
    private_only: bool = True
    timeout: float = 30.0
    concurrency: int = 8


@dataclass
class RawListing:
    olx_id: str
    url: str
    title: str = ""
    price_eur: float | None = None
    negotiable: bool = False
    brand: str = ""
    model: str = ""
    year: int | None = None
    mileage_km: int | None = None
    engine_cc: int | None = None
    fuel_type: str | None = None
    horsepower: int | None = None
    transmission: str | None = None
    segment: str | None = None
    doors: str | None = None
    seats: int | None = None
    color: str | None = None
    condition: str | None = None
    drive_type: str | None = None
    photo_count: int | None = None
    registration_month: str | None = None
    city: str = ""
    district: str = ""
    seller_type: str = "Particular"
    description: str = ""
    source: str = "olx"  # "olx" or "standvirtual"

    # Seller pointer — extracted cheaply from the detail-page HTML during
    # ``scrape_listing_detail``. ``seller_uuid`` stays None at scrape time;
    # it's resolved by a follow-up profile-page fetch (see
    # ``scripts/backfill_sellers.py``) which upserts ``sellers`` and links
    # listings to the canonical seller record.
    seller_profile_url: str | None = None
    seller_short_id: str | None = None
    seller_shop_slug: str | None = None
    seller_display_name: str | None = None
    seller_displayed_as: str | None = None


class OlxScraper:
    """Scraper using httpx. No browser dependencies."""

    def __init__(self, config: ScraperConfig | None = None):
        self.config = config or ScraperConfig()
        self.client = httpx.Client(
            timeout=self.config.timeout,
            follow_redirects=True,
            http2=True,
            headers={
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "pt-PT,pt;q=0.9,en;q=0.5",
                # NB: do NOT advertise `br`. OLX flipped its encoder to
                # Brotli around 2026-04-20 for clients that opt in, and
                # httpx without the optional `brotli` extra silently
                # returns the compressed bytes as `r.text` — BeautifulSoup
                # then sees binary garbage and reports "no listing cards
                # found", which the production scraper logged for ~10
                # days while OLX listings quietly went stale. gzip +
                # deflate cover everything we actually need.
                "Accept-Encoding": "gzip, deflate",
            },
        )
        self._consecutive_403 = 0
        self._lock_403 = threading.Lock()
        self._stop_event = threading.Event()

    def _random_headers(self) -> dict:
        return {"User-Agent": random.choice(USER_AGENTS)}

    def _delay(self):
        time.sleep(random.uniform(self.config.delay_min, self.config.delay_max))

    def _fetch(self, url: str, retries: int = 3) -> tuple[str, str] | None:
        """Fetch *url* and return ``(final_url, html)`` or *None*."""
        for attempt in range(retries):
            if self._stop_event.is_set():
                return None
            try:
                resp = self.client.get(url, headers=self._random_headers())
                resp.raise_for_status()
                with self._lock_403:
                    self._consecutive_403 = 0
                return str(resp.url), resp.text
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 403:
                    with self._lock_403:
                        self._consecutive_403 += 1
                        if self._consecutive_403 >= 5:
                            logger.error("Too many 403s. IP blocked. Wait 15min.")
                            self._stop_event.set()
                            return None
                    wait = min(30 * (2 ** attempt), 120) + random.uniform(5, 15)
                    logger.warning("403 blocked (attempt %d/%d). Waiting %.0fs...",
                                   attempt + 1, retries, wait)
                    time.sleep(wait)
                else:
                    logger.warning("HTTP %s for %s", e.response.status_code, url)
                    return None
            except httpx.RequestError as e:
                logger.warning("Request error for %s: %s", url, e)
                return None
        return None

    def _fetch_json(self, url: str, retries: int = 3) -> dict | None:
        """Fetch *url* and return parsed JSON, or *None*.

        Shares the 403-cascade / backoff behaviour of :meth:`_fetch` so a
        long API run inherits the rate-limiting we validated against OLX.
        """
        for attempt in range(retries):
            if self._stop_event.is_set():
                return None
            try:
                headers = {**self._random_headers(), "Accept": "application/json"}
                resp = self.client.get(url, headers=headers)
                resp.raise_for_status()
                with self._lock_403:
                    self._consecutive_403 = 0
                return resp.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 403:
                    with self._lock_403:
                        self._consecutive_403 += 1
                        if self._consecutive_403 >= 5:
                            logger.error("Too many 403s. IP blocked. Wait 15min.")
                            self._stop_event.set()
                            return None
                    wait = min(30 * (2 ** attempt), 120) + random.uniform(5, 15)
                    logger.warning("403 blocked (attempt %d/%d). Waiting %.0fs...",
                                   attempt + 1, retries, wait)
                    time.sleep(wait)
                else:
                    logger.warning("HTTP %s for %s", e.response.status_code, url)
                    return None
            except httpx.RequestError as e:
                # Transient under parallel load: the shared HTTP/2 connection
                # hits the server's per-connection stream cap (GOAWAY /
                # ConnectionTerminated) or the local socket pool momentarily
                # exhausts (EAGAIN / "Resource temporarily unavailable"). A
                # retry opens a fresh connection, so back off and try again
                # rather than dropping the page (which would lose listings).
                if attempt < retries - 1:
                    time.sleep(0.5 * (attempt + 1) + random.uniform(0, 0.5))
                    continue
                logger.warning("Request error for %s: %s", url, e)
                return None
            except json.JSONDecodeError as e:
                logger.warning("Bad JSON from %s: %s", url, e)
                return None
        return None

    # ------------------------------------------------------------------
    # Search results page
    # ------------------------------------------------------------------

    def scrape_search_page(self, page: int = 1) -> list[RawListing] | None:
        """Return listings for *page*, or ``None`` if there are no more pages."""
        return self._scrape_search_page_api(page)

    def _scrape_search_page_api(
        self,
        page: int = 1,
        category_id: int | None = None,
        extra_params: list[str] | None = None,
    ) -> list[RawListing] | None:
        """Fetch one page of listings from the JSON API.

        Returns ``None`` past the end of results (mirrors the HTML
        redirect-stop semantics), ``[]`` on an empty first page or a fetch
        failure. ``category_id`` selects a brand sub-category; ``extra_params``
        carries price-band filters for segmented full-coverage runs.
        """
        offset = (page - 1) * _API_PAGE_SIZE
        if offset > _API_OFFSET_CAP:
            return None
        cat = category_id if category_id is not None else CARS_CATEGORY_ID
        params = [f"offset={offset}", f"limit={_API_PAGE_SIZE}",
                  f"category_id={cat}"]
        if extra_params:
            params += extra_params
        url = OLX_API_URL + "?" + "&".join(params)

        logger.info("Scraping API page %d (cat=%s): %s", page, cat, url)
        data = self._fetch_json(url)
        if data is None:
            return []
        offers = data.get("data") or []
        if not offers:
            # Empty deep page = end of this segment; empty first page is a
            # zero-result query (or a shape change — scrape_all loud-fails).
            return None if page > 1 else []

        listings: list[RawListing] = []
        for offer in offers:
            try:
                if self.config.private_only and offer.get("business"):
                    continue  # dealer listing — keep only private sellers
                raw = _offer_to_raw(offer)
                if raw.olx_id:
                    listings.append(raw)
            except Exception as e:  # noqa: BLE001 - one bad offer must not kill the page
                logger.debug("Error parsing API offer: %s", e)

        logger.info("Parsed %d listings from API page %d", len(listings), page)
        return listings

    # ------------------------------------------------------------------
    # Detail page (HTML) — retained for the alert-refresh probe
    # (src/alerts/telegram_bot.py) and the photo-count backfill, which
    # fetch a single listing by URL. Not part of the scrape pipeline:
    # scrape_all sources complete records from the JSON API.
    # ------------------------------------------------------------------

    def scrape_listing_detail(self, url: str) -> dict:
        result = self._fetch(url)
        if not result:
            return {}

        _final_url, html = result
        soup = BeautifulSoup(html, "lxml")
        details = {}

        # Limit JSON-LD scan: Vehicle block is always among the first scripts,
        # and `find_all(limit=N)` stops DOM traversal once N hits accumulate.
        for script in soup.find_all("script", type="application/ld+json", limit=5):
            try:
                data = json.loads(script.string)
                if isinstance(data, dict) and data.get("@type") == "Vehicle":
                    details["brand"] = data.get("brand", "")
                    details["model"] = data.get("model", "")
                    details["year"] = _safe_int(data.get("productionDate"))
                    # JSON-LD ``Vehicle.name`` is the canonical headline; the
                    # search-card title can carry price residue when the OLX
                    # card layout fuses title+price under one wrapper (the
                    # pre-2026-05-11 cards stored "BMW-520-f10 20129.000 €"
                    # — year and price glued without a separator). Pull from
                    # the detail page so any re-fetch heals the bad rows.
                    title_clean = (data.get("name") or "").strip()
                    if title_clean:
                        details["title"] = title_clean
                    offers = data.get("offers", {})
                    details["price_eur"] = _safe_float(offers.get("price"))
                    area = offers.get("areaServed", {})
                    if isinstance(area, dict):
                        details["city"] = area.get("name", "")
                    details["olx_id"] = str(data.get("sku", ""))
                    break
            except (json.JSONDecodeError, TypeError):
                continue

        params_container = soup.select_one("[data-testid='ad-parameters-container']")
        if params_container:
            for p in params_container.find_all("p"):
                text = p.get_text(strip=True)
                if text in ("Particular", "Profissional"):
                    details["seller_type"] = text
                    continue
                if ":" in text:
                    label, _, value = text.partition(":")
                    field_name = PARAM_LABEL_MAP.get(label.strip())
                    if not field_name:
                        continue
                    value = value.strip()
                    if field_name in ("year", "mileage_km", "engine_cc", "horsepower", "seats"):
                        details[field_name] = _safe_int(value)
                    else:
                        details[field_name] = value

        if "price_eur" not in details:
            price_el = soup.select_one("[data-testid='ad-price-container']")
            if price_el:
                details["price_eur"] = _parse_eur_price(price_el.get_text(strip=True))

        prices_wrapper = soup.select_one("[data-testid='prices-wrapper']")
        if prices_wrapper:
            details["negotiable"] = "negociável" in prices_wrapper.get_text(strip=True).lower()

        breadcrumbs = soup.select_one("[data-testid='breadcrumbs']")
        if breadcrumbs:
            items = [el.get_text(strip=True) for el in breadcrumbs.select("[data-testid='breadcrumb-item']")]
            loc_items = [it for it in items if " - " in it]
            if len(loc_items) >= 2:
                details["district"] = loc_items[-2].split(" - ", 1)[-1].strip()
                city_from_bc = loc_items[-1].split(" - ", 1)[-1].strip()
                if city_from_bc:
                    details["city"] = city_from_bc
            elif len(loc_items) == 1:
                details["district"] = loc_items[0].split(" - ", 1)[-1].strip()

        # Photo count. Current OLX layout (verified 2026-05-04) wraps each
        # gallery photo in [data-testid="ad-photo"]; counting those is the
        # simplest stable signal. The old [data-testid="photo-gallery"] /
        # [data-cy="ad-photos"] selectors no longer match — they returned
        # photo_count=None for 4436 / 4438 active OLX listings (≈100 %),
        # leaving the photo-damage classifier nothing to score and
        # zeroing out the uncertainty model's desc_quality feature.
        ad_photos = soup.select('[data-testid="ad-photo"]')
        if ad_photos:
            details["photo_count"] = len(ad_photos)
        else:
            gallery = soup.select_one("[data-testid='photo-gallery']") \
                or soup.select_one("[data-cy='ad-photos']") \
                or soup.select_one("[data-testid='image-galery-container']")
            if gallery:
                details["photo_count"] = len(gallery.find_all("img"))

        # Description text. OLX injects "Anotações"/"Reportar" icon-button
        # labels and a "Descrição" heading as bare leading text inside this
        # container, so strip that chrome off the top before storing.
        desc_el = soup.select_one("[data-cy='ad_description'] div") or soup.select_one("[data-testid='ad-description']")
        if desc_el:
            details["description"] = _strip_desc_chrome(
                desc_el.get_text(separator="\n", strip=True)
            )

        # Posted/updated date
        posted_el = soup.select_one("[data-testid='ad-posted-at']")
        if posted_el:
            details["posted_at"] = _parse_pt_date(posted_el.get_text(strip=True))

        if "olx_id" not in details:
            footer = soup.select_one("[data-testid='ad-footer-bar-section']")
            if footer:
                id_match = re.search(r"ID:\s*(\d+)", footer.get_text())
                if id_match:
                    details["olx_id"] = id_match.group(1)

        # Seller pointer — cheap to grab from the same soup. We do NOT
        # fetch the profile page here; that's a separate batch step
        # (scripts/backfill_sellers.py) so multi-car sellers only get
        # one HTTP hit per refresh window instead of one per listing.
        seller_link = parse_seller_link(soup)
        if seller_link:
            details["seller_profile_url"] = seller_link.profile_url
            details["seller_short_id"] = seller_link.short_id
            details["seller_shop_slug"] = seller_link.shop_slug
            details["seller_display_name"] = seller_link.display_name
            details["seller_displayed_as"] = seller_link.displayed_as

        return details

    # ------------------------------------------------------------------
    # Seller profile page
    # ------------------------------------------------------------------

    def scrape_seller_profile(self, url: str) -> SellerProfile | None:
        """Fetch and parse an OLX seller-profile or business-shop page.

        Reuses the same throttle / retry / 403-cascade logic as listing-
        detail fetches, so a backfill that processes thousands of sellers
        in a row inherits the rate limiting we already validated against
        OLX. Returns ``None`` if the page can't be fetched or its
        ``__PRERENDERED_STATE__`` blob is missing/malformed — the caller
        decides whether that's a skip or a hard fail.
        """
        if not url:
            return None
        result = self._fetch(url)
        if not result:
            return None
        _final_url, html = result
        return parse_seller_profile_html(html, profile_url=url)

    # ------------------------------------------------------------------
    # StandVirtual detail page
    # ------------------------------------------------------------------

    def scrape_standvirtual_detail(self, url: str) -> dict:
        """Parse a standvirtual.com listing detail page via its embedded JSON.

        Reads ``__NEXT_DATA__.props.pageProps.advert`` instead of scraping
        ``data-testid`` nodes — robust against markup changes and the source
        of colour / drive_type / sub-model that OLX simply doesn't carry.
        """
        result = self._fetch(url)
        if not result:
            return {}
        _final_url, html = result
        advert = _sv_advert_from_html(html)
        if not advert:
            return {}
        return _sv_advert_to_details(advert)

    # ------------------------------------------------------------------
    # Full scrape
    # ------------------------------------------------------------------

    def _enrich_one(self, listing: "RawListing",
                    on_ready=None) -> bool:
        """Enrich a single listing with detail page data. Returns True on success."""
        if self._stop_event.is_set() or not listing.url:
            return False
        self._delay()
        logger.debug("Fetching detail: %s", listing.url)
        if "standvirtual.com" in listing.url:
            details = self.scrape_standvirtual_detail(listing.url)
        else:
            details = self.scrape_listing_detail(listing.url)
        _merge_details(listing, details)
        if on_ready and listing.description:
            on_ready(listing)
        return True

    _ENRICH_TIMEOUT = 90  # seconds — max time per detail page (incl. retries)

    def _enrich_batch(
        self,
        listings: list[RawListing],
        skip_ids: set[str] | None = None,
    ) -> tuple[int, int]:
        """Fetch detail pages for a batch of listings. Returns (ok, failed).

        *skip_ids* lists olx_ids that already have a canonical twin enriched
        elsewhere (e.g. cross-platform duplicates) — their detail page is a
        wasted HTTP request, so we just keep card-level fields.
        """
        workers = self.config.concurrency
        enriched = 0
        failed = 0
        to_fetch = (
            [l for l in listings if l.olx_id not in skip_ids]
            if skip_ids else listings
        )
        skipped = len(listings) - len(to_fetch)
        if skipped:
            logger.info("Skipped detail fetch for %d known duplicates", skipped)
        if not to_fetch:
            return 0, 0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_listing = {
                executor.submit(self._enrich_one, listing): listing
                for listing in to_fetch
            }
            try:
                for future in as_completed(future_to_listing,
                                           timeout=self._ENRICH_TIMEOUT):
                    try:
                        if future.result():
                            enriched += 1
                        else:
                            failed += 1
                    except Exception:
                        failed += 1
                    if self._stop_event.is_set():
                        executor.shutdown(wait=False, cancel_futures=True)
                        break
            except TimeoutError:
                not_done = [f for f in future_to_listing if not f.done()]
                for f in not_done:
                    url = future_to_listing[f].url
                    logger.warning("Detail fetch timed out (>%ds): %s",
                                   self._ENRICH_TIMEOUT, url)
                    f.cancel()
                    failed += 1
        return enriched, failed

    # ------------------------------------------------------------------
    # Full-coverage scrape (price-band segmentation)
    # ------------------------------------------------------------------

    def _segment_count(self, lo: int, hi: int | None, category_id: int) -> int:
        """Return ``metadata.total_elements`` for a price band.

        OLX caps this at 1000: a band reporting < 1000 is fully pageable;
        a band reporting exactly the cap still hides listings and must be
        split further. Uses the verified bare ``filter_float_price:from/to``
        query params (the ``search[...]`` wrapper is silently ignored).
        """
        params = ["offset=0", "limit=1", f"category_id={category_id}",
                  f"filter_float_price%3Afrom={lo}"]
        if hi is not None:
            params.append(f"filter_float_price%3Ato={hi}")
        data = self._fetch_json(OLX_API_URL + "?" + "&".join(params))
        if not data:
            return 0
        return int(data.get("metadata", {}).get("total_elements", 0) or 0)

    def _price_bands(self, lo: int, hi: int | None, category_id: int,
                     max_depth: int = 14) -> list[tuple[int, int | None, int]]:
        """Recursively bisect ``[lo, hi]`` into ``(lo, hi, count)`` bands.

        Terminates when a band reports < the cap, the recursion depth is
        exhausted, or the band is too narrow to split — so every leaf is
        fully pageable and their union covers the whole range. ``count`` is
        the band's reported size (capped at 1000), used to bound paging.
        """
        count = self._segment_count(lo, hi, category_id)
        if count == 0:
            return []
        if count < _API_OFFSET_CAP or max_depth <= 0:
            return [(lo, hi, count)]
        if hi is None:
            return [(lo, hi, count)]  # open-ended top band can't be bisected
        if hi - lo <= 100:
            return [(lo, hi, count)]  # narrow enough; accept residual cap loss
        mid = (lo + hi) // 2
        return (self._price_bands(lo, mid, category_id, max_depth - 1)
                + self._price_bands(mid, hi, category_id, max_depth - 1))

    def _fetch_pages_parallel(self, specs: list, fetch_one) -> list:
        """Fetch many independent pages concurrently, preserving input order.

        ``fetch_one(spec)`` runs in a worker thread (httpx.Client is safe for
        concurrent requests; ``_fetch_json`` keeps the 403-cascade). The
        bounded pool *is* the rate limit — verified safe at 12 concurrent
        against the OLX API — so no per-request delay is applied. Returns a
        list aligned with ``specs``; a spec that errors yields ``[]``.
        """
        results: list = [None] * len(specs)
        if not specs:
            return results
        with ThreadPoolExecutor(max_workers=_PARALLEL_FETCH_WORKERS) as ex:
            fut_to_idx = {ex.submit(fetch_one, s): i for i, s in enumerate(specs)}
            for fut in as_completed(fut_to_idx):
                idx = fut_to_idx[fut]
                try:
                    results[idx] = fut.result()
                except Exception as e:  # noqa: BLE001
                    logger.debug("Parallel page fetch error: %s", e)
                    results[idx] = []
                if self._stop_event.is_set():
                    ex.shutdown(wait=False, cancel_futures=True)
                    break
        return results

    def scrape_full(self, on_batch_ready=None, known_ids: set[str] | None = None,
                    category_id: int | None = None,
                    max_price: int = 1_000_000) -> list[RawListing]:
        """Full-coverage scrape that escapes the ~1000-per-query ceiling.

        A single OLX query (HTML or API) is hard-capped at ~1000 results,
        but the category holds ~50k cars. We bisect the price axis until
        each band is under the cap, then page every band, deduping across
        the (inclusive) band boundaries. API path only.
        """
        cat = category_id if category_id is not None else CARS_CATEGORY_ID
        bands = self._price_bands(0, max_price, cat)
        est = sum(c for *_, c in bands)
        logger.info("Full coverage: %d price bands (~%d offers) for category %s",
                    len(bands), est, cat)

        # Build one flat list of (page, extra_params) specs across all bands,
        # bounding each band's page count by its reported size and the
        # offset cap, then fetch them all concurrently.
        max_page = _API_OFFSET_CAP // _API_PAGE_SIZE + 1
        specs: list[tuple[int, list[str]]] = []
        for lo, hi, count in bands:
            extra = [f"filter_float_price%3Afrom={lo}"]
            if hi is not None:
                extra.append(f"filter_float_price%3Ato={hi}")
            pages = min(-(-count // _API_PAGE_SIZE), max_page)
            specs.extend((page, extra) for page in range(1, pages + 1))

        pages_results = self._fetch_pages_parallel(
            specs,
            lambda s: self._scrape_search_page_api(
                s[0], category_id=cat, extra_params=s[1]),
        )

        all_listings: list[RawListing] = []
        seen: set[str] = set()
        got_any = False
        for res in pages_results:
            if res:
                got_any = True
            # Dedup within the page too — OLX prepends promoted ads that also
            # appear in the regular results, so the same olx_id can show up
            # twice on one page (and across band boundaries).
            fresh = []
            for l in (res or []):
                if l.olx_id and l.olx_id not in seen:
                    seen.add(l.olx_id)
                    fresh.append(l)
            all_listings.extend(fresh)
            if on_batch_ready and fresh:
                on_batch_ready(fresh)

        # Loud-fail (cron exits non-zero) if the API returned nothing at all
        # across every band — almost certainly a shape/endpoint change, not a
        # genuinely empty market.
        if not got_any and not self._stop_event.is_set():
            msg = ("OLX JSON API returned 0 offers across all price bands — "
                   "source likely changed (API shape/endpoint change or bot wall)")
            logger.error("::error::%s", msg)
            raise ScraperParseError(msg)

        logger.info("Full scrape: %d unique listings across %d bands",
                    len(all_listings), len(bands))
        return all_listings

    def close(self):
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PT_MONTHS = {
    "janeiro": 1, "fevereiro": 2, "março": 3, "abril": 4,
    "maio": 5, "junho": 6, "julho": 7, "agosto": 8,
    "setembro": 9, "outubro": 10, "novembro": 11, "dezembro": 12,
}


def _parse_pt_date(text: str):
    """Parse a Portuguese date string like '29 de março de 2026 às 22:17'.

    Returns a ``datetime`` or *None*.
    """
    from datetime import datetime
    m = re.search(r"(\d{1,2})\s+de\s+(\w+)\s+de\s+(\d{4})", text)
    if not m:
        return None
    day, month_name, year = int(m.group(1)), m.group(2).lower(), int(m.group(3))
    month = _PT_MONTHS.get(month_name)
    if not month:
        return None
    hour, minute = 0, 0
    time_m = re.search(r"(\d{1,2}):(\d{2})", text)
    if time_m:
        hour, minute = int(time_m.group(1)), int(time_m.group(2))
    try:
        return datetime(year, month, day, hour, minute)
    except ValueError:
        return None

def _merge_details(listing: RawListing, details: dict):
    # Store posted_at separately (not a RawListing field — handled in DB layer)
    if "posted_at" in details:
        listing._posted_at = details.pop("posted_at")
    for key, value in details.items():
        if value is not None and hasattr(listing, key):
            current = getattr(listing, key)
            # Title from JSON-LD ``Vehicle.name`` is canonical, but the
            # search-card title can carry richer descriptive text. Overwrite
            # only when the existing title shows price residue — "€" or a
            # 5+ digit run (clean titles never have either; year ≤ 4 digits).
            if key == "title" and current:
                dirty = "€" in current or bool(re.search(r"\d{5,}", current))
                if dirty:
                    setattr(listing, key, value)
                continue
            # Canonical detail-page fields (JSON-LD ``Vehicle`` + the
            # ``ad-parameters-container`` rows) override whatever the
            # search card produced. Pre-2026-05 search cards sometimes
            # rendered the *price* number where the parser expected
            # mileage ("2012 - 9.000 km" patterns), so 377 / 19,308
            # listings ended up with ``mileage_km == price_eur``. The
            # detail page's "Quilómetros: 355.000 km" param is the
            # ground truth and must win.
            if key in _DETAIL_AUTHORITATIVE_FIELDS:
                setattr(listing, key, value)
                continue
            if not current or current == "" or current == 0:
                setattr(listing, key, value)
    # Fix mileage after all fields are populated
    listing.mileage_km = _fix_mileage(listing.mileage_km, listing.year)


# Fields written by ``scrape_listing_detail`` / ``scrape_standvirtual_detail``
# from canonical sources (JSON-LD ``Vehicle`` + the ad-parameters-container).
# When detail has a non-None value for one of these, it must override the
# search-card value — the card is a preview, the detail page is the ground
# truth.
_DETAIL_AUTHORITATIVE_FIELDS = frozenset({
    "brand", "model", "year", "price_eur", "mileage_km", "engine_cc",
    "fuel_type", "horsepower", "transmission", "doors", "seats", "color",
    "drive_type", "condition", "segment", "registration_month", "city",
})


def _parse_eur_price(text: str) -> float | None:
    if not text:
        return None
    text = re.split(r"[a-zA-Zà-ÿ]", text)[0]
    cleaned = re.sub(r"[^\d,.]", "", text.replace(" ", ""))
    if not cleaned:
        return None
    if "." in cleaned and "," not in cleaned:
        parts = cleaned.split(".")
        if len(parts[-1]) == 3:
            cleaned = cleaned.replace(".", "")
    elif "," in cleaned:
        cleaned = cleaned.replace(".", "").replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _extract_brand_from_title(title: str) -> str:
    # Word-boundary match: "ds" used to substring-match in "DSG" (the dual-
    # clutch transmission), and since DS is a real brand sorted at the end
    # of KNOWN_BRANDS, every Cupra/VW listing with DSG was getting tagged
    # brand=DS. We saw "DS Formentor" / "DS Passat Variant" etc. piling up
    # in unmatched_listings as a result.
    title_lower = title.lower()
    for brand in sorted(KNOWN_BRANDS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(brand.lower())}\b", title_lower):
            return brand
    abbrevs = {"vw": "Volkswagen", "merc": "Mercedes-Benz", "mb": "Mercedes-Benz"}
    for abbrev, brand in abbrevs.items():
        if re.search(rf"\b{abbrev}\b", title_lower):
            return brand
    return ""


def _fix_mileage(km: int | None, year: int | None) -> int | None:
    """Detect and fix mileage entered without thousands (e.g. 150 instead of 150000).

    Heuristic: a car driven ~10k-20k km/year. If mileage is suspiciously low
    for the car's age, it's likely missing *1000.
    """
    if km is None or km == 0:
        return km
    if year is None:
        # No year to cross-check — only fix obvious cases
        if km < 1000:
            return km * 1000
        return km

    import datetime
    age = max(datetime.date.today().year - year, 1)
    avg_per_year = km / age

    # A car averaging < 200 km/year is almost certainly missing *1000
    # (real minimum is ~3000 km/year for a parked car)
    if km < 1000 and avg_per_year < 500:
        return km * 1000

    # Values like 50-999 with reasonable age → multiply
    if km < 1000:
        corrected = km * 1000
        corrected_avg = corrected / age
        if 3000 <= corrected_avg <= 40000:
            return corrected

    return km


def _safe_int(val) -> int | None:
    if val is None:
        return None
    try:
        return int(re.sub(r"[^\d]", "", str(val)))
    except (ValueError, TypeError):
        return None


def _safe_float(val) -> float | None:
    if val is None:
        return None
    try:
        cleaned = str(val).replace(",", ".")
        return float(re.sub(r"[^\d.]", "", cleaned))
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# OLX JSON-API offer -> RawListing
# ---------------------------------------------------------------------------

# API param ``key`` -> (RawListing field, which sub-field to read).
# OLX params come as ``{"key": "<machine>", "label": "<human>"}``; numeric
# fields carry the clean integer in ``key`` (e.g. quilometros key "133000",
# label "133.000 km"), categorical fields carry the display text in ``label``
# (e.g. combustivel label "Diesel"). ``cor``/``tração`` are absent from the
# OLX cars vertical entirely (they only ever populate from StandVirtual).
_API_PARAM_MAP = {
    "modelo": ("model", "label"),
    "body_type": ("segment", "label"),
    "combustivel": ("fuel_type", "label"),
    "gearbox": ("transmission", "label"),
    "condicao": ("condition", "label"),
    "portas": ("doors", "label"),
    "first_registration_month": ("registration_month", "label"),
    "year": ("year", "key"),
    "quilometros": ("mileage_km", "key"),
    "engine_capacity": ("engine_cc", "key"),
    "engine_power": ("horsepower", "key"),
    "nr_seats": ("seats", "key"),
}
_API_INT_FIELDS = frozenset({"year", "mileage_km", "engine_cc", "horsepower", "seats"})


def _clean_html_description(desc: str) -> str:
    """Turn the API's HTML description into plain text (unescape + drop tags).

    OLX's API HTML carries BOTH a ``<br>`` and the author's literal newline for
    every line break, so a naive ``<br>``→``\\n`` doubled each break into a blank
    line (≈37% of stored descriptions came out double-spaced). Collapse each
    ``<br>`` together with one adjacent newline so a single visual break maps to
    a single ``\\n``; a genuine ``<br><br>`` paragraph gap still yields ``\\n\\n``.
    """
    import html as _html
    text = _html.unescape(desc).replace("\r\n", "\n").replace("\r", "\n")
    # <br> + one adjacent literal newline (either side) collapses to one break.
    text = re.sub(r"\n?[ \t]*<br\s*/?>[ \t]*\n?", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    # Collapse trailing whitespace and runs of blank lines.
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# OLX renders icon-only action buttons (save-to-notes, report) and the
# "Descrição" heading as bare label text at the very top of the description
# container, so the detail page's get_text() captures them as leading lines
# ("Anotações", "Reportar", …). They are never part of the author's text;
# ≈12k stored descriptions begin with this chrome. Drop any run of them from
# the top, stopping at the first real line.
_OLX_DESC_CHROME = frozenset({
    "Descrição", "Anotações", "Reportar", "Observar", "Denunciar",
    "Partilhar", "Guardar",
})


def _strip_desc_chrome(text: str) -> str:
    """Drop leading OLX UI-chrome labels from a scraped description."""
    if not text:
        return text
    lines = text.split("\n")
    i = 0
    while i < len(lines) and lines[i].strip() in _OLX_DESC_CHROME:
        i += 1
    return "\n".join(lines[i:]).lstrip("\n")


def _parse_iso_dt(value: str | None):
    """Parse an ISO-8601 timestamp to a *naive* UTC datetime.

    The repository compares ``posted_at`` against a naive ``utcnow()`` and
    rejects future dates, so we must hand it a tz-naive value or the
    comparison raises ``TypeError``.
    """
    if not value:
        return None
    from datetime import datetime, timezone
    try:
        # ``fromisoformat`` only learned to parse a trailing ``Z`` in 3.11;
        # StandVirtual stamps ``...:10Z`` so normalise it for older runtimes.
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _offer_to_raw(offer: dict) -> RawListing:
    """Build a :class:`RawListing` from one OLX JSON-API offer dict.

    Single source of truth for OLX listings — replaces both the SERP-card
    parser and the per-listing detail-page parser, since the list payload
    already carries every field they extracted.
    """
    url = offer.get("url", "") or ""
    # olx_id MUST come from the URL slug (the historical key the DB dedups
    # on), NOT the numeric ``offer['id']`` — using the numeric id would make
    # every existing listing look new and break lifecycle/relist tracking.
    m = re.search(r"ID(\w+)\.html", url)
    olx_id = m.group(1) if m else ""

    params = {p["key"]: p.get("value") for p in offer.get("params", [])
              if isinstance(p, dict) and "key" in p}

    price_v = params.get("price")
    price_eur = None
    negotiable = False
    if isinstance(price_v, dict):
        if price_v.get("value") is not None:
            price_eur = float(price_v["value"])
        negotiable = bool(price_v.get("negotiable"))

    title = offer.get("title", "") or ""
    loc = offer.get("location") or {}
    raw = RawListing(
        olx_id=olx_id,
        url=url,
        title=title,
        price_eur=price_eur,
        negotiable=negotiable,
        brand=_extract_brand_from_title(title),
        seller_type="Profissional" if offer.get("business") else "Particular",
        city=((loc.get("city") or {}).get("name")) or "",
        district=((loc.get("region") or {}).get("name")) or "",
        photo_count=len(offer.get("photos") or []),
        source="olx",
    )

    for key, (field, attr) in _API_PARAM_MAP.items():
        v = params.get(key)
        if not isinstance(v, dict):
            continue
        val = v.get(attr)
        if val is None:
            continue
        if field in _API_INT_FIELDS:
            setattr(raw, field, _safe_int(val))
        else:
            setattr(raw, field, str(val).strip())

    raw.mileage_km = _fix_mileage(raw.mileage_km, raw.year)

    desc = offer.get("description") or ""
    if desc:
        raw.description = _clean_html_description(desc)

    posted = _parse_iso_dt(offer.get("created_time"))
    if posted:
        raw._posted_at = posted

    return raw


# ---------------------------------------------------------------------------
# StandVirtual JSON (__NEXT_DATA__) parsing
# ---------------------------------------------------------------------------

# StandVirtual is a Next.js app over a GraphQL (urql) backend; every field
# the data-testid selectors used to scrape is also embedded as clean JSON in
# the page's ``__NEXT_DATA__``. Parsing that JSON is robust against CSS/markup
# changes and recovers richer fields (sub-model/version, origin, VIN, the
# seller UUID). It does NOT cut the request count — SV detail specs
# (colour/drive-type) still need the per-listing page — but it kills the
# fragile selector layer and is where colour/drive_type actually come from
# (98.5% / 82% of SV rows vs ~1% of OLX rows).
_NEXT_DATA_RE = re.compile(
    r'<script id="__NEXT_DATA__"[^>]*>(.+?)</script>', re.DOTALL
)

# advert.details[key] -> (RawListing field, parser). SV exposes BOTH
# ``gearbox`` (Tipo de Caixa -> transmission) and ``transmission`` (Tracção
# -> drive_type); detail ``value`` is already the human display string.
_SV_DETAIL_MAP = {
    "make": ("brand", None),
    "model": ("model", None),
    "mileage": ("mileage_km", _safe_int),
    "fuel_type": ("fuel_type", None),
    "gearbox": ("transmission", None),
    "first_registration_year": ("year", _safe_int),
    "first_registration_month": ("registration_month", None),
    "engine_capacity": ("engine_cc", _safe_int),
    "engine_power": ("horsepower", _safe_int),
    "door_count": ("doors", None),
    "nr_seats": ("seats", _safe_int),
    "color": ("color", None),
    "body_type": ("segment", None),
    "new_used": ("condition", None),
    "transmission": ("drive_type", None),
}


def _next_data(html: str) -> dict | None:
    m = _NEXT_DATA_RE.search(html or "")
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def _sv_advert_from_html(html: str) -> dict:
    """Return ``props.pageProps.advert`` from a SV detail page, or ``{}``."""
    data = _next_data(html)
    if not data:
        return {}
    try:
        return data["props"]["pageProps"]["advert"] or {}
    except (KeyError, TypeError):
        return {}


def _sv_advert_search_from_html(html: str) -> dict | None:
    """Return the ``advertSearch`` result object from a SV search page.

    The urql client caches each GraphQL result as a JSON string under
    ``props.pageProps.urqlState[*].data``; we scan those for the one that
    holds ``advertSearch`` (the listing results + pageInfo + totalCount).
    """
    data = _next_data(html)
    if not data:
        return None
    try:
        urql = data["props"]["pageProps"]["urqlState"]
    except (KeyError, TypeError):
        return None
    for entry in (urql or {}).values():
        raw = (entry or {}).get("data")
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict) and "advertSearch" in parsed:
            return parsed["advertSearch"]
    return None


def _sv_advert_to_details(advert: dict) -> dict:
    """Map a SV advert JSON object to the detail dict ``_merge_details`` wants."""
    details: dict = {}
    by_key = {d.get("key"): d for d in advert.get("details", [])
              if isinstance(d, dict)}
    for key, (field, parser) in _SV_DETAIL_MAP.items():
        d = by_key.get(key)
        if not d or d.get("value") is None:
            continue
        val = d["value"]
        if parser:
            # Strip trailing units before int-casting — ``cm3`` in
            # particular leaves a stray "3" digit that _safe_int would glue
            # onto the displacement (1 598 cm3 -> 15983).
            val = re.sub(r"\s*(cm3|km|cv|l)\s*$", "", str(val), flags=re.IGNORECASE)
            details[field] = parser(val)
        else:
            details[field] = str(val).strip()

    price = advert.get("price") or {}
    if price.get("value") is not None:
        details["price_eur"] = _safe_float(price.get("value"))
        labels = price.get("labels") or []
        details["negotiable"] = any("negotiable" in str(l).lower() for l in labels)

    stype = ((advert.get("seller") or {}).get("type") or "").upper()
    if stype == "PROFESSIONAL":
        details["seller_type"] = "Profissional"
    elif stype == "PRIVATE":
        details["seller_type"] = "Particular"

    photos = (advert.get("images") or {}).get("photos") or []
    if photos:
        details["photo_count"] = len(photos)

    desc = advert.get("description")
    if isinstance(desc, str) and desc:
        details["description"] = _clean_html_description(desc)

    posted = _parse_iso_dt(advert.get("createdAt"))
    if posted:
        details["posted_at"] = posted

    m = re.search(r"ID(\w+)\.html", advert.get("url", "") or "")
    if m:
        details["olx_id"] = m.group(1)
    return details


def _sv_node_to_raw(node: dict) -> RawListing | None:
    """Build a card-level :class:`RawListing` from a SV ``advertSearch`` node."""
    url = node.get("url", "") or ""
    m = re.search(r"ID(\w+)\.html", url)
    if not m:
        return None
    params = {p.get("key"): p for p in node.get("parameters", [])
              if isinstance(p, dict)}

    def disp(k):
        p = params.get(k)
        return p.get("displayValue") if p else None

    def val(k):
        p = params.get(k)
        return p.get("value") if p else None

    title = node.get("title", "") or ""
    units = ((node.get("price") or {}).get("amount") or {}).get("units")
    # SV uses a sentinel of 1 for "price on request" / "sob consulta"; no real
    # car costs < 100 €, so treat such values as no-price (the old data-testid
    # parser produced None for these too).
    price_eur = float(units) if units is not None and units >= 100 else None
    stype = (node.get("seller") or {}).get("__typename", "") or ""
    loc = node.get("location") or {}
    raw = RawListing(
        olx_id=m.group(1),
        url=url,
        title=title,
        price_eur=price_eur,
        brand=_extract_brand_from_title(title),
        model=disp("model") or "",
        year=_safe_int(val("first_registration_year")),
        mileage_km=_safe_int(val("mileage")),
        fuel_type=disp("fuel_type"),
        transmission=disp("gearbox"),
        engine_cc=_safe_int(val("engine_capacity")),
        horsepower=_safe_int(val("engine_power")),
        city=((loc.get("city") or {}).get("name")) or "",
        district=((loc.get("region") or {}).get("name")) or "",
        seller_type="Profissional" if "Professional" in stype else "Particular",
        source="standvirtual",
    )
    raw.mileage_km = _fix_mileage(raw.mileage_km, raw.year)
    posted = _parse_iso_dt(node.get("createdAt"))
    if posted:
        raw._posted_at = posted
    return raw


# ---------------------------------------------------------------------------
# StandVirtual search page scraper
# ---------------------------------------------------------------------------

SV_BASE_URL = "https://www.standvirtual.com/carros"


class StandVirtualScraper:
    """Scraper for standvirtual.com search pages + detail pages."""

    def __init__(self, config: ScraperConfig | None = None):
        self.config = config or ScraperConfig(base_url=SV_BASE_URL)
        self._olx_scraper = OlxScraper(config)  # reuse HTTP client + detail parser

    @property
    def _stop_event(self):
        return self._olx_scraper._stop_event

    def _delay(self):
        self._olx_scraper._delay()

    def _fetch(self, url: str) -> tuple[str, str] | None:
        return self._olx_scraper._fetch(url)

    # ------------------------------------------------------------------
    # Search results page
    # ------------------------------------------------------------------

    def scrape_search_page(self, page: int = 1) -> list[RawListing] | None:
        params = [f"page={page}"]
        if self.config.private_only:
            params.append("search%5Bprivate_business%5D=private")
        url = self.config.base_url + "?" + "&".join(params)
        logger.info("Scraping StandVirtual page %d: %s", page, url)

        result = self._fetch(url)
        if not result:
            return []

        final_url, html = result

        if page > 1 and f"page={page}" not in final_url:
            logger.info("SV page %d redirected to %s — no more pages", page, final_url)
            return None

        return self._parse_search_page(html)

    def _parse_search_page(self, html: str) -> list[RawListing]:
        """Parse SV search results from the embedded ``advertSearch`` JSON."""
        advert_search = _sv_advert_search_from_html(html)
        if advert_search is None:
            logger.warning("SV: advertSearch not found in __NEXT_DATA__")
            return []
        listings = []
        for edge in advert_search.get("edges") or []:
            node = edge.get("node") if isinstance(edge, dict) else None
            if not node:
                continue
            try:
                raw = _sv_node_to_raw(node)
                if raw and raw.olx_id:
                    listings.append(raw)
            except Exception as e:  # noqa: BLE001
                logger.debug("Error parsing SV node: %s", e)
        logger.info("Parsed %d listings from SV search page", len(listings))
        return listings

    # ------------------------------------------------------------------
    # Detail & enrichment (delegate to OlxScraper)
    # ------------------------------------------------------------------

    def scrape_listing_detail(self, url: str) -> dict:
        return self._olx_scraper.scrape_standvirtual_detail(url)

    def _enrich_one(self, listing: RawListing, on_ready=None) -> bool:
        if self._stop_event.is_set() or not listing.url:
            return False
        self._delay()
        details = self.scrape_listing_detail(listing.url)
        _merge_details(listing, details)
        if on_ready and listing.description:
            on_ready(listing)
        return True

    def _enrich_batch(
        self,
        listings: list[RawListing],
        skip_ids: set[str] | None = None,
    ) -> tuple[int, int]:
        return self._olx_scraper._enrich_batch(listings, skip_ids=skip_ids)

    # ------------------------------------------------------------------
    # Full scrape (GraphQL listingScreen + parallel pagination)
    # ------------------------------------------------------------------

    _SV_ENRICH_CHUNK = 64

    def _sv_listing_screen(self, page: int) -> dict | None:
        """Fetch one page of SV results via the ``listingScreen`` persisted
        GraphQL query. Returns the ``advertSearch`` object, or ``None`` on any
        GraphQL/transport failure (caller falls back to SSR)."""
        filters = [{"name": "category_id", "value": SV_CARS_CATEGORY_ID}]
        if self.config.private_only:
            filters.append({"name": "private_business", "value": "private"})
        variables = {
            "after": None, "filters": filters,
            "includeCepik": False, "includeFiltersCounters": False,
            "includeNewPromotedAds": False, "includePremiumTopAd": False,
            "includePriceDrop": False, "includePriceEvaluation": False,
            "includePromotedAds": False, "includeSortOptions": False,
            "includeSuggestedFilters": False, "maxAge": 60, "page": page,
            "parameters": SV_LISTING_PARAMS, "promotedInput": {},
            "searchTerms": [], "sortBy": "relevance_web",
        }
        ext = {"persistedQuery": {"sha256Hash": SV_LISTING_SCREEN_HASH, "version": 1}}
        qs = urllib.parse.urlencode({
            "operationName": "listingScreen",
            "variables": json.dumps(variables, separators=(",", ":")),
            "extensions": json.dumps(ext, separators=(",", ":")),
        })
        data = self._olx_scraper._fetch_json(SV_GRAPHQL_URL + "?" + qs)
        if not data or data.get("errors"):
            return None
        return (data.get("data") or {}).get("advertSearch")

    def _sv_page_raws(self, page: int) -> list[RawListing] | None:
        """Fetch + map one GraphQL page. ``None`` signals a fetch failure
        (distinct from an empty page)."""
        adv = self._sv_listing_screen(page)
        if adv is None:
            return None
        out: list[RawListing] = []
        for edge in adv.get("edges") or []:
            node = edge.get("node") if isinstance(edge, dict) else None
            if not node:
                continue
            try:
                raw = _sv_node_to_raw(node)
            except Exception as e:  # noqa: BLE001
                logger.debug("Error parsing SV node: %s", e)
                continue
            if raw and raw.olx_id:
                out.append(raw)
        return out

    def scrape_full(self, on_batch_ready=None,
                    known_ids: set[str] | None = None,
                    skip_enrichment_ids: set[str] | None = None,
                    enrich_details: bool = True) -> list[RawListing]:
        """Full-coverage StandVirtual scrape via the ``listingScreen`` GraphQL
        API with parallel pagination. New listings (not in
        ``skip_enrichment_ids``) get a detail-page fetch for colour/drive_type;
        known ones keep card-level fields. Falls back to SSR ``__NEXT_DATA__``
        parsing if the persisted-query hash has rotated."""
        first = self._sv_listing_screen(1)
        if first is None:
            logger.warning("SV listingScreen GraphQL unavailable — "
                           "falling back to SSR pagination")
            return self._scrape_full_ssr(on_batch_ready, skip_enrichment_ids,
                                         enrich_details)

        total = int(first.get("totalCount") or 0)
        pages = max(1, -(-total // SV_PAGE_SIZE))
        logger.info("SV full coverage: %d offers, %d GraphQL pages", total, pages)

        results = self._olx_scraper._fetch_pages_parallel(
            list(range(1, pages + 1)), self._sv_page_raws)

        # Flatten + dedup (promoted ads repeat within/across pages).
        cards: list[RawListing] = []
        seen: set[str] = set()
        for res in results:
            for l in (res or []):
                if l.olx_id and l.olx_id not in seen:
                    seen.add(l.olx_id)
                    cards.append(l)

        # Enrich NEW listings with detail (colour/drive_type) + stream to the
        # caller in chunks, so the per-batch enrich timeout applies per chunk
        # rather than capping the whole run, and the DB/LLM pipeline drains
        # concurrently.
        for i in range(0, len(cards), self._SV_ENRICH_CHUNK):
            if self._stop_event.is_set():
                break
            chunk = cards[i:i + self._SV_ENRICH_CHUNK]
            if enrich_details:
                self._enrich_batch(chunk, skip_ids=skip_enrichment_ids)
            if on_batch_ready:
                on_batch_ready(chunk)

        logger.info("SV full scrape: %d unique listings", len(cards))
        return cards

    def _scrape_full_ssr(self, on_batch_ready=None,
                         skip_enrichment_ids: set[str] | None = None,
                         enrich_details: bool = True) -> list[RawListing]:
        """Fallback: walk SSR search pages sequentially (parses ``urqlState``).
        Used only when the GraphQL persisted-query hash has rotated."""
        all_listings: list[RawListing] = []
        seen: set[str] = set()
        consecutive_empty = 0
        for page in range(1, self.config.max_pages + 1):
            page_listings = self.scrape_search_page(page)
            if page_listings is None:
                break
            if not page_listings:
                consecutive_empty += 1
                if all_listings:
                    break
                if consecutive_empty >= 2:
                    msg = ("StandVirtual returned 0 listings on pages "
                           f"1-{page} (GraphQL + SSR both failed) — source "
                           "likely changed")
                    logger.error("::error::%s", msg)
                    raise ScraperParseError(msg)
                self._delay()
                continue
            consecutive_empty = 0
            fresh = []
            for l in page_listings:
                if l.olx_id and l.olx_id not in seen:
                    seen.add(l.olx_id)
                    fresh.append(l)
            if enrich_details and fresh:
                self._enrich_batch(fresh, skip_ids=skip_enrichment_ids)
            all_listings.extend(fresh)
            if on_batch_ready and fresh:
                on_batch_ready(fresh)
            if self._stop_event.is_set():
                break
            self._delay()
        logger.info("SV SSR fallback: %d listings", len(all_listings))
        return all_listings

    def close(self):
        self._olx_scraper.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
