"""OLX.pt car listings scraper.

OLX.pt is a server-rendered React app. Data sources:
  - Search page cards: data-testid selectors for price/location/year/mileage,
    brand extracted from URL breadcrumb path
  - Detail pages: JSON-LD (schema.org Vehicle) + data-testid selectors for
    all car parameters (14+ fields)
"""

import json
import logging
import random
import re
import time
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
    "Alfa Romeo", "Audi", "BMW", "Chevrolet", "Citroen", "Dacia", "DS",
    "Fiat", "Ford", "Honda", "Hyundai", "Jaguar", "Jeep", "Kia",
    "Land Rover", "Lexus", "Mazda", "Mercedes-Benz", "Mini", "Mitsubishi",
    "Nissan", "Opel", "Peugeot", "Porsche", "Renault", "Seat", "Skoda",
    "Smart", "Subaru", "Suzuki", "Tesla", "Toyota", "Volkswagen", "Volvo",
]

# Map Portuguese parameter labels to our field names
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
    delay_min: float = 2.0
    delay_max: float = 5.0
    private_only: bool = True
    timeout: float = 30.0


@dataclass
class RawListing:
    """Raw listing data extracted from OLX.pt page."""
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


class OlxScraper:
    def __init__(self, config: ScraperConfig | None = None):
        self.config = config or ScraperConfig()
        self.client = httpx.Client(
            timeout=self.config.timeout,
            follow_redirects=True,
            headers={
                "Accept-Language": "pt-PT,pt;q=0.9,en;q=0.5",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            },
        )

    def _random_headers(self) -> dict:
        return {"User-Agent": random.choice(USER_AGENTS)}

    def _delay(self):
        time.sleep(random.uniform(self.config.delay_min, self.config.delay_max))

    def _fetch(self, url: str) -> str | None:
        try:
            resp = self.client.get(url, headers=self._random_headers())
            resp.raise_for_status()
            return resp.text
        except httpx.HTTPStatusError as e:
            logger.warning("HTTP %s for %s", e.response.status_code, url)
            return None
        except httpx.RequestError as e:
            logger.warning("Request error for %s: %s", url, e)
            return None

    # ------------------------------------------------------------------
    # Search results page
    # ------------------------------------------------------------------

    def scrape_search_page(self, page: int = 1) -> list[RawListing]:
        """Scrape a single search results page."""
        params = [f"page={page}"]
        if self.config.private_only:
            params.append("search%5Bprivate_business%5D=private")
        url = self.config.base_url + "?" + "&".join(params)

        logger.info("Scraping search page %d: %s", page, url)
        html = self._fetch(url)
        if not html:
            return []

        return self._parse_search_page(html)

    def _parse_search_page(self, html: str) -> list[RawListing]:
        """Extract listings from search page HTML cards."""
        soup = BeautifulSoup(html, "lxml")
        listings = []

        cards = soup.select("[data-testid='l-card']")
        if not cards:
            logger.warning("No listing cards found on page")
            return []

        for card in cards:
            try:
                # Link + OLX ID
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

                # Title
                title_el = card.select_one("[data-cy='ad-card-title']")
                if not title_el:
                    title_el = card.find("h6") or card.find("h4") or card.find("h5")
                title = title_el.get_text(strip=True) if title_el else ""

                # Price
                price_eur = None
                negotiable = False
                price_el = card.select_one("[data-testid='ad-price']")
                if price_el:
                    price_text = price_el.get_text(strip=True)
                    negotiable = "negociável" in price_text.lower()
                    price_eur = _parse_eur_price(price_text)

                # Location: "City - Date" in data-testid="location-date"
                city = ""
                loc_el = card.select_one("[data-testid='location-date']")
                if loc_el:
                    loc_text = loc_el.get_text(strip=True)
                    # Format: "City - Date" or "City - Para o topo ..."
                    dash_idx = loc_text.find(" - ")
                    if dash_idx > 0:
                        city = loc_text[:dash_idx].strip()
                    else:
                        city = loc_text.strip()

                # Year and mileage: span[data-nx-name="P5"] contains "2018 - 105.000 km"
                year = None
                mileage_km = None
                year_km_el = card.select_one("span[data-nx-name='P5']")
                if year_km_el:
                    ykm_text = year_km_el.get_text(strip=True)
                    ykm_match = re.match(r"(\d{4})\s*-\s*([\d.]+)\s*km", ykm_text)
                    if ykm_match:
                        year = int(ykm_match.group(1))
                        mileage_km = _safe_int(ykm_match.group(2))

                # Brand from URL: /carros/bmw/ or /carros/mercedes-benz/
                brand = _extract_brand_from_url(url)

                # Try to extract brand from title if not in URL
                if not brand:
                    brand = _extract_brand_from_title(title)

                listings.append(RawListing(
                    olx_id=olx_id, url=url, title=title,
                    price_eur=price_eur, negotiable=negotiable,
                    brand=brand, model="",
                    year=year, mileage_km=mileage_km,
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
        """Scrape full details from an individual listing page."""
        html = self._fetch(url)
        if not html:
            return {}

        soup = BeautifulSoup(html, "lxml")
        details = {}

        # 1) JSON-LD for brand, model, year, price
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

        # 2) Parameters from data-testid="ad-parameters-container"
        params_container = soup.select_one("[data-testid='ad-parameters-container']")
        if params_container:
            paragraphs = params_container.find_all("p")
            for p in paragraphs:
                text = p.get_text(strip=True)

                # Seller type (no colon)
                if text in ("Particular", "Profissional"):
                    details["seller_type"] = text
                    continue

                # "Label: Value" pairs
                if ":" in text:
                    label, _, value = text.partition(":")
                    label = label.strip()
                    value = value.strip()

                    field_name = PARAM_LABEL_MAP.get(label)
                    if not field_name:
                        continue

                    if field_name in ("year", "mileage_km", "engine_cc", "horsepower", "seats"):
                        details[field_name] = _safe_int(value)
                    else:
                        details[field_name] = value

        # 3) Price from price container (fallback)
        if "price_eur" not in details:
            price_el = soup.select_one("[data-testid='ad-price-container']")
            if price_el:
                details["price_eur"] = _parse_eur_price(price_el.get_text(strip=True))

        # Negotiable flag
        prices_wrapper = soup.select_one("[data-testid='prices-wrapper']")
        if prices_wrapper:
            details["negotiable"] = "negociável" in prices_wrapper.get_text(strip=True).lower()

        # 4) Location from map section
        map_section = soup.select_one("[data-testid='map-aside-section']")
        if map_section:
            loc_paragraphs = map_section.find_all("p")
            loc_texts = [p.get_text(strip=True) for p in loc_paragraphs
                         if p.get_text(strip=True) and p.get_text(strip=True) != "Localização"]
            if len(loc_texts) >= 2:
                details["city"] = loc_texts[0]
                details["district"] = loc_texts[1]
            elif len(loc_texts) == 1:
                details["city"] = loc_texts[0]

        # 5) Ad ID from footer (fallback)
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

    def scrape_all(self, enrich_details: bool = False) -> list[RawListing]:
        """Scrape all pages. Optionally visit each listing for full details."""
        all_listings = []
        for page in range(1, self.config.max_pages + 1):
            page_listings = self.scrape_search_page(page)
            if not page_listings:
                logger.info("No more listings at page %d, stopping", page)
                break

            all_listings.extend(page_listings)
            logger.info("Total so far: %d listings", len(all_listings))
            self._delay()

        if enrich_details:
            logger.info("Enriching %d listings with detail pages...", len(all_listings))
            for i, listing in enumerate(all_listings):
                if not listing.url:
                    continue
                details = self.scrape_listing_detail(listing.url)
                _merge_details(listing, details)
                if (i + 1) % 10 == 0:
                    logger.info("Enriched %d / %d", i + 1, len(all_listings))
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

def _merge_details(listing: RawListing, details: dict):
    """Merge scraped detail page data into a RawListing."""
    for key, value in details.items():
        if value is not None and hasattr(listing, key):
            current = getattr(listing, key)
            if not current or current == "" or current == 0:
                setattr(listing, key, value)


def _parse_eur_price(text: str) -> float | None:
    """Parse '15.400 €' or '15 400€' into 15400.0"""
    if not text:
        return None
    # Remove "Negociável" and other non-price text
    text = re.split(r"[a-zA-Zà-ÿ]", text)[0]
    cleaned = re.sub(r"[^\d,.]", "", text.replace(" ", ""))
    if not cleaned:
        return None
    # Portuguese: dot = thousands separator (15.400), comma = decimal (rare for cars)
    if "." in cleaned and "," not in cleaned:
        parts = cleaned.split(".")
        if len(parts[-1]) == 3:  # 15.400 = thousands
            cleaned = cleaned.replace(".", "")
    elif "," in cleaned:
        cleaned = cleaned.replace(".", "").replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _extract_brand_from_url(url: str) -> str:
    """Extract brand from OLX.pt URL: /carros/bmw/... or breadcrumb category."""
    # URL pattern: /carros-motos-e-barcos/carros/bmw/
    match = re.search(r"/carros/([^/]+)/", url)
    if match:
        slug = match.group(1).lower()
        for brand in KNOWN_BRANDS:
            if brand.lower().replace(" ", "-").replace("-", "-") == slug:
                return brand
            if brand.lower().replace(" ", "-") == slug:
                return brand
    return ""


def _extract_brand_from_title(title: str) -> str:
    """Try to find a known brand in the listing title."""
    title_lower = title.lower()
    # Sort by length descending so "Mercedes-Benz" matches before "Mercedes"
    for brand in sorted(KNOWN_BRANDS, key=len, reverse=True):
        # Match brand name or common abbreviation
        if brand.lower() in title_lower:
            return brand
    # Common abbreviations
    abbrevs = {"vw": "Volkswagen", "merc": "Mercedes-Benz", "mb": "Mercedes-Benz"}
    for abbrev, brand in abbrevs.items():
        if re.search(rf"\b{abbrev}\b", title_lower):
            return brand
    return ""


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
