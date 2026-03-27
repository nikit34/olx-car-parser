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
    "Matrícula": "registration_plate",
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
    "Origem": "origin",
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
    concurrency: int = 5


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
    origin: str | None = None
    registration_month: str | None = None
    registration_plate: str | None = None
    city: str = ""
    district: str = ""
    seller_type: str = "Particular"
    description: str = ""


class OlxScraper:
    """Scraper using httpx. No browser dependencies."""

    def __init__(self, config: ScraperConfig | None = None):
        self.config = config or ScraperConfig()
        self.client = httpx.Client(
            timeout=self.config.timeout,
            follow_redirects=True,
            headers={
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "pt-PT,pt;q=0.9,en;q=0.5",
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
                            logger.error("Too many 403s. IP blocked. Set OLX_PROXY env var or wait 15min.")
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
                if "/d/anuncio/" not in url:
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

        for script in soup.find_all("script", type="application/ld+json"):
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

        # Description text
        desc_el = soup.select_one("[data-cy='ad_description'] div") or soup.select_one("[data-testid='ad-description']")
        if desc_el:
            details["description"] = desc_el.get_text(separator="\n", strip=True)

        if "olx_id" not in details:
            footer = soup.select_one("[data-testid='ad-footer-bar-section']")
            if footer:
                id_match = re.search(r"ID:\s*(\d+)", footer.get_text())
                if id_match:
                    details["olx_id"] = id_match.group(1)

        return details

    # ------------------------------------------------------------------
    # Full scrape
    # ------------------------------------------------------------------

    def _enrich_one(self, listing: "RawListing") -> bool:
        """Enrich a single listing with detail page data. Returns True on success."""
        if self._stop_event.is_set() or not listing.url:
            return False
        self._delay()
        details = self.scrape_listing_detail(listing.url)
        _merge_details(listing, details)
        return True

    def scrape_all(self, enrich_details: bool = True) -> list[RawListing]:
        all_listings = []
        for page in range(1, self.config.max_pages + 1):
            page_listings = self.scrape_search_page(page)
            if page_listings is None:
                # Redirect detected — we've passed the last real page.
                break
            if not page_listings:
                logger.info("No more listings at page %d, stopping", page)
                break
            all_listings.extend(page_listings)
            logger.info("Total so far: %d listings", len(all_listings))
            self._delay()

        if enrich_details:
            total = len(all_listings)
            workers = self.config.concurrency
            logger.info("Enriching %d listings with detail pages (concurrency=%d)...",
                        total, workers)
            enriched = 0
            failed = 0
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_idx = {
                    executor.submit(self._enrich_one, listing): i
                    for i, listing in enumerate(all_listings)
                }
                for future in as_completed(future_to_idx):
                    try:
                        if future.result():
                            enriched += 1
                        else:
                            failed += 1
                    except Exception:
                        idx = future_to_idx[future]
                        logger.debug("Error enriching listing %d: %s",
                                     idx, all_listings[idx].url, exc_info=True)
                        failed += 1
                    done = enriched + failed
                    if done % 50 == 0 or done == total:
                        logger.info("Enrichment progress: %d / %d (ok=%d, fail=%d)",
                                    done, total, enriched, failed)
                    if self._stop_event.is_set():
                        logger.warning("Stopping enrichment early — IP appears blocked. "
                                       "Enriched %d / %d before stop.", enriched, total)
                        executor.shutdown(wait=False, cancel_futures=True)
                        break
            logger.info("Enrichment complete: %d succeeded, %d failed out of %d",
                        enriched, failed, total)

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

def _merge_details(listing: RawListing, details: dict):
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
