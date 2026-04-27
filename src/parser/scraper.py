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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0",
]

BASE_URL = "https://www.olx.pt/carros-motos-e-barcos/carros/"

KNOWN_BRANDS = [
    "Alfa Romeo", "Audi", "BMW", "Chevrolet", "Citroen", "Citroën", "Dacia", "DS",
    "Fiat", "Ford", "Honda", "Hyundai", "Jaguar", "Jeep", "Kia",
    "Land Rover", "Lexus", "Mazda", "Mercedes-Benz", "Mini", "Mitsubishi",
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
                "Accept-Encoding": "gzip, deflate, br",
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

    # ------------------------------------------------------------------
    # Search results page
    # ------------------------------------------------------------------

    def scrape_search_page(self, page: int = 1) -> list[RawListing] | None:
        """Return listings for *page*, or ``None`` if redirected (no more pages)."""
        params = [f"page={page}"]
        if self.config.private_only:
            params.append("search%5Bprivate_business%5D=private")
        url = self.config.base_url + "?" + "&".join(params)

        logger.info("Scraping search page %d: %s", page, url)
        result = self._fetch(url)
        if not result:
            return []

        final_url, html = result

        # OLX redirects out-of-range pages back to the last valid page.
        if page > 1 and f"page={page}" not in final_url:
            logger.info("Page %d redirected to %s — no more pages", page, final_url)
            return None

        return self._parse_search_page(html)

    def _parse_search_page(self, html: str) -> list[RawListing]:
        soup = BeautifulSoup(html, "lxml")
        listings = []

        cards = soup.select("[data-testid='l-card']")
        if not cards:
            logger.warning("No listing cards found on page")
            return []

        for card in cards:
            try:
                link = card.find("a", href=True)
                if not link:
                    continue
                url = link.get("href", "")
                if not url.startswith("http"):
                    url = "https://www.olx.pt" + url
                # Accept OLX listings (/d/anuncio/) and StandVirtual cross-posts
                if "/d/anuncio/" not in url and "standvirtual.com" not in url:
                    continue

                olx_id_match = re.search(r"ID(\w+)\.html", url)
                olx_id = olx_id_match.group(1) if olx_id_match else ""
                if not olx_id:
                    continue

                title_el = card.select_one("[data-cy='ad-card-title']")
                if not title_el:
                    title_el = card.find("h6") or card.find("h4") or card.find("h5")
                title = title_el.get_text(strip=True) if title_el else ""

                price_eur = None
                negotiable = False
                price_el = card.select_one("[data-testid='ad-price']")
                if price_el:
                    price_text = price_el.get_text(strip=True)
                    negotiable = "negociável" in price_text.lower()
                    price_eur = _parse_eur_price(price_text)

                city = ""
                loc_el = card.select_one("[data-testid='location-date']")
                if loc_el:
                    loc_text = loc_el.get_text(strip=True)
                    dash_idx = loc_text.find(" - ")
                    city = loc_text[:dash_idx].strip() if dash_idx > 0 else loc_text.strip()

                year = None
                mileage_km = None
                year_km_el = card.select_one("span[data-nx-name='P5']")
                if year_km_el:
                    ykm_text = year_km_el.get_text(strip=True)
                    ykm_match = re.match(r"(\d{4})\s*-\s*([\d.]+)\s*km", ykm_text)
                    if ykm_match:
                        year = int(ykm_match.group(1))
                        mileage_km = _safe_int(ykm_match.group(2))

                brand = _extract_brand_from_url(url) or _extract_brand_from_title(title)

                listings.append(RawListing(
                    olx_id=olx_id, url=url, title=title,
                    price_eur=price_eur, negotiable=negotiable,
                    brand=brand, model="", year=year, mileage_km=mileage_km,
                    city=city,
                ))
            except Exception as e:
                logger.debug("Error parsing card: %s", e)

        logger.info("Parsed %d listings from search page", len(listings))
        return listings

    # ------------------------------------------------------------------
    # Detail page
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

        # Photo count
        gallery = soup.select_one("[data-testid='photo-gallery']") or soup.select_one("[data-cy='ad-photos']")
        if gallery:
            details["photo_count"] = len(gallery.find_all("img"))

        # Description text
        desc_el = soup.select_one("[data-cy='ad_description'] div") or soup.select_one("[data-testid='ad-description']")
        if desc_el:
            details["description"] = desc_el.get_text(separator="\n", strip=True)

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

        return details

    # ------------------------------------------------------------------
    # StandVirtual detail page
    # ------------------------------------------------------------------

    def scrape_standvirtual_detail(self, url: str) -> dict:
        """Parse a standvirtual.com listing detail page."""
        result = self._fetch(url)
        if not result:
            return {}

        _final_url, html = result
        soup = BeautifulSoup(html, "lxml")
        details: dict = {}

        # Mapping: data-testid -> (field_name, parser)
        SV_FIELDS = {
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

        # StandVirtual labels baked into text (e.g. "MarcaNissan", "Quilómetros130 000 km")
        SV_LABEL_PREFIXES = {
            "make": "Marca",
            "model": "Modelo",
            "mileage": "Quilómetros",
            "fuel_type": "Combustível",
            "gearbox": "Tipo de Caixa",
            "first_registration_year": "Ano",
            "first_registration_month": "Mês de Registo",
            "engine_capacity": "Cilindrada",
            "engine_power": "Potência",
            "door_count": "Nº de portas",
            "nr_seats": "Lotação",
            "color": "Cor",
            "body_type": "Segmento",
            "new_used": "Condição",
            "transmission": "Tracção",
        }

        for testid, (field, parser) in SV_FIELDS.items():
            el = soup.find(attrs={"data-testid": testid})
            if not el:
                continue
            text = el.get_text(strip=True)
            prefix = SV_LABEL_PREFIXES.get(testid, "")
            if prefix and text.startswith(prefix):
                text = text[len(prefix):].strip()
            # Strip trailing units (km, cm3, cv)
            text = re.sub(r"\s*(km|cm3|cv)$", "", text).strip()
            if parser:
                details[field] = parser(text)
            else:
                details[field] = text

        # Price
        price_el = soup.find(attrs={"data-testid": "ad-price"})
        if price_el:
            details["price_eur"] = _parse_eur_price(price_el.get_text(strip=True))

        # Negotiable (check summary area)
        summary = soup.find(attrs={"data-testid": "summary-info-area"})
        if summary:
            details["negotiable"] = "negociável" in summary.get_text(strip=True).lower()

        # Seller type
        seller = soup.find(attrs={"data-testid": "seller-header"})
        if seller:
            seller_text = seller.get_text(strip=True)
            if "Particular" in seller_text:
                details["seller_type"] = "Particular"
            elif "Profissional" in seller_text or "Stand" in seller_text:
                details["seller_type"] = "Profissional"

        # Photo count
        photo_gallery = soup.find(attrs={"data-testid": "photo-gallery"})
        if photo_gallery:
            details["photo_count"] = len(photo_gallery.find_all("img"))
        else:
            counter = soup.find(attrs={"data-testid": "photo-counter"})
            if counter:
                # Format: "1/27"
                match = re.search(r"/(\d+)", counter.get_text(strip=True))
                if match:
                    details["photo_count"] = int(match.group(1))

        # Description
        desc_el = soup.find(attrs={"data-testid": "content-description-section"})
        if desc_el:
            details["description"] = desc_el.get_text(separator="\n", strip=True)

        # Breadcrumbs for location (if available)
        breadcrumb = soup.find(attrs={"data-testid": "breadcrumb-section"})
        if breadcrumb:
            items = [el.get_text(strip=True) for el in breadcrumb.find_all("a")]
            # Breadcrumbs: Carros > Brand > Model (no location in standvirtual breadcrumbs)

        # Posted/updated date (SV has it as plain text: "29 de março de 2026 às 22:17").
        # Pre-filter by the Portuguese "de" separator to skip the vast majority
        # of <p> tags without parsing each one.
        for p in soup.find_all("p", string=re.compile(r"\d+\s+de\s+\w+\s+de\s+\d{4}"), limit=5):
            parsed = _parse_pt_date(p.get_text(strip=True))
            if parsed:
                details["posted_at"] = parsed
                break

        # Extract olx_id from URL
        id_match = re.search(r"ID(\w+)\.html", url)
        if id_match:
            details["olx_id"] = id_match.group(1)

        return details

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

    def scrape_all(self, enrich_details: bool = True,
                   on_batch_ready=None,
                   skip_enrichment_ids: set[str] | None = None,
                   known_ids: set[str] | None = None,
                   early_stop_known_ratio: float = 0.95,
                   early_stop_consecutive: int = 3) -> list[RawListing]:
        """Scrape all listings.

        Args:
            enrich_details: Fetch detail pages for each listing.
            on_batch_ready: Optional callback ``fn(batch: list[RawListing])``
                called after each page's detail pages are fetched and saved.
                Allows the caller to persist listings to DB incrementally.
            skip_enrichment_ids: olx_ids whose detail page we already have
                covered via a canonical twin (cross-platform duplicates).
            known_ids: olx_ids already in our DB. When *most* of a search
                page is already-known listings (per ``early_stop_known_ratio``)
                across ``early_stop_consecutive`` pages in a row, we stop —
                OLX shows newest first, so the deep pages are 100% revisits.
                On the production run this turns ~200 page fetches into ~30,
                which is the difference between a 6-hour scrape and a
                30-minute one.
            early_stop_known_ratio: ≥ this fraction of a page being already
                in ``known_ids`` counts as "this page is mostly known".
            early_stop_consecutive: number of consecutive mostly-known pages
                before we trigger early-stop (single-page false positives
                happen when OLX surfaces older listings on top).
        """
        all_listings = []
        consecutive_known = 0
        for page in range(1, self.config.max_pages + 1):
            page_listings = self.scrape_search_page(page)
            if page_listings is None:
                break
            if not page_listings:
                logger.info("No more listings at page %d, stopping", page)
                break

            if enrich_details:
                ok, fail = self._enrich_batch(page_listings, skip_ids=skip_enrichment_ids)
                logger.info("Page %d: %d listings, details ok=%d fail=%d",
                            page, len(page_listings), ok, fail)
            else:
                logger.info("Page %d: %d listings", page, len(page_listings))

            all_listings.extend(page_listings)

            if on_batch_ready:
                on_batch_ready(page_listings)

            if self._stop_event.is_set():
                logger.warning("Stopping early — IP blocked. Got %d listings.", len(all_listings))
                break

            # Early-stop: when the deep pages are mostly already-known, OLX
            # has nothing new to give us this cycle.
            if known_ids:
                known_count = sum(1 for l in page_listings if l.olx_id in known_ids)
                ratio = known_count / max(len(page_listings), 1)
                if ratio >= early_stop_known_ratio:
                    consecutive_known += 1
                    if consecutive_known >= early_stop_consecutive:
                        logger.info(
                            "Early-stop after page %d: %d pages in a row had "
                            "≥%.0f%% already-known listings.",
                            page, consecutive_known, early_stop_known_ratio * 100,
                        )
                        break
                else:
                    consecutive_known = 0

            self._delay()

        logger.info("Scraping complete: %d total listings", len(all_listings))
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
            if not current or current == "" or current == 0:
                setattr(listing, key, value)
    # Fix mileage after all fields are populated
    listing.mileage_km = _fix_mileage(listing.mileage_km, listing.year)


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


def _extract_brand_from_url(url: str) -> str:
    match = re.search(r"/carros/([^/]+)/", url)
    if match:
        slug = match.group(1).lower()
        for brand in KNOWN_BRANDS:
            if brand.lower().replace(" ", "-") == slug:
                return brand
    return ""


def _extract_brand_from_title(title: str) -> str:
    title_lower = title.lower()
    for brand in sorted(KNOWN_BRANDS, key=len, reverse=True):
        if brand.lower() in title_lower:
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
# StandVirtual search page scraper
# ---------------------------------------------------------------------------

SV_BASE_URL = "https://www.standvirtual.com/carros"

# dt text -> RawListing field
SV_CARD_FIELDS = {
    "mileage": "mileage_km",
    "fuel_type": "fuel_type",
    "gearbox": "transmission",
    "first_registration_year": "year",
}


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
        soup = BeautifulSoup(html, "lxml")
        listings = []

        for article in soup.find_all("article"):
            try:
                link = article.find("a", href=True)
                if not link:
                    continue
                url = link.get("href", "")
                if "/anuncio/" not in url:
                    continue

                id_match = re.search(r"ID(\w+)\.html", url)
                if not id_match:
                    continue
                olx_id = id_match.group(1)

                title_el = article.find("h2") or article.find("h3") or article.find("h1")
                title = title_el.get_text(strip=True) if title_el else ""

                # Price: h3 that contains digits (not the title h2)
                price_eur = None
                for h3 in article.find_all("h3"):
                    text = h3.get_text(strip=True).replace(" ", "")
                    if re.match(r"^[\d.]+$", text):
                        price_eur = _safe_float(text)
                        break

                # Specs from dt/dd pairs
                specs: dict = {}
                dts = article.find_all("dt")
                dds = article.find_all("dd")
                for dt, dd in zip(dts, dds):
                    key = dt.get_text(strip=True)
                    val = dd.get_text(strip=True)
                    field = SV_CARD_FIELDS.get(key)
                    if field:
                        specs[field] = val

                year = _safe_int(specs.get("year"))
                mileage_raw = specs.get("mileage_km", "")
                mileage_km = _safe_int(mileage_raw.replace("km", "").strip()) if mileage_raw else None

                brand = _extract_brand_from_title(title)

                listings.append(RawListing(
                    olx_id=olx_id,
                    url=url,
                    title=title,
                    price_eur=price_eur,
                    brand=brand,
                    year=year,
                    mileage_km=mileage_km,
                    fuel_type=specs.get("fuel_type"),
                    transmission=specs.get("transmission"),
                    source="standvirtual",
                ))
            except Exception as e:
                logger.debug("Error parsing SV article: %s", e)

        logger.info("Parsed %d listings from StandVirtual search page", len(listings))
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
    # Full scrape
    # ------------------------------------------------------------------

    def scrape_all(self, enrich_details: bool = True,
                   on_batch_ready=None,
                   skip_enrichment_ids: set[str] | None = None,
                   known_ids: set[str] | None = None,
                   early_stop_known_ratio: float = 0.95,
                   early_stop_consecutive: int = 3) -> list[RawListing]:
        """See OlxScraper.scrape_all for the early-stop semantics — this
        method shares the same shape so the CLI can pass `known_ids` to both
        scrapers uniformly."""
        all_listings = []
        consecutive_known = 0
        for page in range(1, self.config.max_pages + 1):
            page_listings = self.scrape_search_page(page)
            if page_listings is None:
                break
            if not page_listings:
                logger.info("SV: no more listings at page %d, stopping", page)
                break

            if enrich_details:
                ok, fail = self._enrich_batch(page_listings, skip_ids=skip_enrichment_ids)
                logger.info("SV page %d: %d listings, details ok=%d fail=%d",
                            page, len(page_listings), ok, fail)
            else:
                logger.info("SV page %d: %d listings", page, len(page_listings))

            all_listings.extend(page_listings)
            if on_batch_ready:
                on_batch_ready(page_listings)

            if self._stop_event.is_set():
                logger.warning("SV stopping early — blocked. Got %d listings.", len(all_listings))
                break

            if known_ids:
                known_count = sum(1 for l in page_listings if l.olx_id in known_ids)
                ratio = known_count / max(len(page_listings), 1)
                if ratio >= early_stop_known_ratio:
                    consecutive_known += 1
                    if consecutive_known >= early_stop_consecutive:
                        logger.info(
                            "SV early-stop after page %d: %d consecutive pages "
                            "with ≥%.0f%% already-known listings.",
                            page, consecutive_known, early_stop_known_ratio * 100,
                        )
                        break
                else:
                    consecutive_known = 0

            self._delay()

        logger.info("StandVirtual scraping complete: %d total listings", len(all_listings))
        return all_listings

    def close(self):
        self._olx_scraper.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
