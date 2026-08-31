"""AutoScout24.de search reader — the German side of the import question.

The one thing this project cannot answer from its own corpus is whether buying
a given model in Germany and nationalising it beats buying it here. The
Portuguese half we own; the German half is one number we do not have — what the
same car asks in Germany today — and this module is how it arrives.

Why the search page and not a detail page: AutoScout24 is a Next.js app and its
result page ships ``__NEXT_DATA__`` with twenty fully-formed listings, exactly
like Standvirtual (see ``scraper._sv_advert_from_html``). Every input the ISV
formula needs is already on the card — CO2 in g/km, Erstzulassung, fuel,
cilindrada, plus the price and, crucially, its VAT label. So one request per
twenty cars, and no detail fetches at all.

**How this behaves on someone else's site**, because that is a decision and not
an implementation detail:

* It identifies itself. ``USER_AGENT`` is ours, with a URL to the site that
  runs it — no pretending to be Chrome. If AutoScout24 wants to refuse or
  throttle this crawler, the name is right there to do it with.
* It reads what ``robots.txt`` leaves open to ``User-agent: *``. That file
  disallows ``/lst?`` and ``/lst/?`` — the query-only form of search — and
  closes the whole site to the named AI crawlers (GPTBot, ClaudeBot, CCBot).
  We are none of those, and the path form used here (``/lst/{make}/{model}``)
  is not among the disallowed prefixes. ``robots_allows`` keeps that judgement
  in code rather than in a comment nobody re-reads.
* It is slow on purpose: ``DELAY_MIN``/``DELAY_MAX`` seconds between requests,
  serial, one request per model-year, and a hard ``budget`` per run so a bug
  cannot turn into a flood. A 429 or a 403 stops the run then and there instead
  of retrying into a ban.
* It takes aggregates, not inventory. What ships to the public pages is the
  median of a model-year, never a copy of somebody's listing.
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from dataclasses import dataclass, field
from urllib.parse import urlencode

import httpx

from src.parser.fuel_normalize import normalize_fuel_type

logger = logging.getLogger(__name__)

BASE_URL = "https://www.autoscout24.de"
SEARCH_PATH = "/lst"
PAGE_SIZE = 20
MAX_PAGE = 20
USER_AGENT = ("Mozilla/5.0 (compatible; CarsbuyerBot/1.0; "
              "+https://carsbuyer.org/sobre)")
DELAY_MIN = 6.0
DELAY_MAX = 10.0
TIMEOUT = 30.0

_DISALLOWED_PREFIXES = (
    "/private-feedback/", "/dealerarea/", "/entry/", "/ergebnisse?", "/i/",
    "/modelle/page/", "/regional/page/", "/lst?", "/lst/?", "/lst-moto?",
    "/lst-moto/?", "/Partner/", "/partner/", "/favorites",
)

_NEXT_DATA_RE = re.compile(
    r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', re.S)

_FUEL_DE_TO_PT = {
    "diesel": "Diesel",
    "benzin": "Gasolina",
    "elektro": "Eléctrico",
    "elektro/benzin": "Híbrido (Gasolina)",
    "elektro/diesel": "Híbrido (Diesel)",
    "autogas (lpg)": "GPL",
    "lpg": "GPL",
    "erdgas (cng)": "GNC",
    "cng": "GNC",
    "wasserstoff": "Hidrogénio",
    "ethanol": "Gasolina",
}

_TRANSMISSION_DE_TO_PT = {
    "automatik": "Automática",
    "schaltgetriebe": "Manual",
    "halbautomatik": "Automática",
}


class AutoScoutBlocked(RuntimeError):
    """The site asked us to stop (429/403). Callers must not retry in-run."""


def robots_allows(path: str) -> bool:
    """Whether ``User-agent: *`` in AutoScout24's robots.txt leaves this path open.

    Kept as code so the crawler cannot drift away from what the file says: the
    runner asks before every fetch, and a path that lands on a disallowed
    prefix is skipped rather than requested.
    """
    p = path if path.startswith("/") else "/" + path
    return not any(p.startswith(pre) for pre in _DISALLOWED_PREFIXES)


@dataclass
class DeListing:
    """One German listing, in this project's vocabulary rather than AS24's."""

    external_id: str
    url: str
    brand: str
    model: str
    price_eur: float | None = None
    vat_label: str | None = None
    vat_reclaimable: bool | None = None
    model_group: str | None = None
    variant: str | None = None
    motor_type: str | None = None
    year: int | None = None
    registration_month: str | None = None
    mileage_km: int | None = None
    engine_cc: int | None = None
    horsepower: int | None = None
    power_kw: int | None = None
    fuel_type: str | None = None
    transmission: str | None = None
    co2_g_km: int | None = None
    seller_type: str | None = None
    country_code: str | None = None
    city: str | None = None
    zip_code: str | None = None
    is_damaged: bool | None = None
    source: str = "autoscout24"


@dataclass
class AutoScoutConfig:
    delay_min: float = DELAY_MIN
    delay_max: float = DELAY_MAX
    timeout: float = TIMEOUT
    budget: int = 200
    user_agent: str = USER_AGENT
    country: str = "D"


def _num(text: str | None) -> float | None:
    """German-formatted number out of a label: '1.995 cm³' → 1995.0."""
    if not text:
        return None
    m = re.search(r"-?\d[\d.]*(?:,\d+)?", str(text))
    if not m:
        return None
    raw = m.group(0).replace(".", "").replace(",", ".")
    try:
        return float(raw)
    except ValueError:
        return None


def _int(text) -> int | None:
    v = _num(text) if not isinstance(text, (int, float)) else float(text)
    return int(round(v)) if v is not None else None


def _detail(listing: dict, aria: str) -> str | None:
    for item in listing.get("vehicleDetails") or []:
        if item.get("ariaLabel") == aria:
            return item.get("data")
    return None


CO2_MIN_G_KM = 50
CO2_MAX_G_KM = 500


def _co2(listing: dict, fuel_pt: str | None = None) -> int | None:
    """CO2 in g/km off the card, or None when the seller typed something impossible.

    The field is free text on AutoScout24 and sellers fill it with anything: a
    2016 320d in this sample claims 5 g/km, and another shows "- (g/km)". CO2 is
    the largest term in the ISV, so a number nobody measured would land on the
    page as a tax bill nobody will pay. Anything outside a plausible combustion
    band is treated as absent, which costs that listing its ISV and nothing else.
    Electric cars legitimately read 0 and are exempt anyway, so they keep it.
    """
    raw = None
    for value in (listing.get("wltpValues") or []):
        if "g/km" in str(value):
            raw = _int(value)
            break
    if raw is None:
        raw = _int(_detail(listing, "CO₂-Emissionen"))
    if raw is None:
        return None
    electric = (fuel_pt or "").lower().startswith("elé")
    if electric:
        return raw if 0 <= raw <= CO2_MAX_G_KM else None
    return raw if CO2_MIN_G_KM <= raw <= CO2_MAX_G_KM else None


def _registration(listing: dict) -> tuple[int | None, str | None]:
    """(year, 'MM/YYYY') from Erstzulassung, tracking first, label second."""
    raw = (listing.get("tracking") or {}).get("firstRegistration")
    if raw and re.match(r"^\d{2}-\d{4}$", str(raw)):
        month, year = str(raw).split("-")
        return int(year), f"{month}/{year}"
    label = _detail(listing, "Erstzulassung")
    if label and re.match(r"^\d{2}/\d{4}$", str(label).strip()):
        month, year = str(label).strip().split("/")
        return int(year), f"{month}/{year}"
    return None, None


def _power(listing: dict) -> tuple[int | None, int | None]:
    """(kW, PS) from '140 kW (190 PS)'."""
    label = _detail(listing, "Leistung") or ""
    kw = re.search(r"(\d[\d.]*)\s*kW", label)
    ps = re.search(r"(\d[\d.]*)\s*PS", label)
    return (_int(kw.group(1)) if kw else None, _int(ps.group(1)) if ps else None)


def _vat(price: dict) -> tuple[str | None, bool | None]:
    """(label, reclaimable) — the 19% that decides whether an import is worth it.

    A German dealer price marked ``inkl. MwSt.`` is what a Portuguese private
    buyer actually pays; ``zzgl. MwSt.`` / ``MwSt. ausweisbar`` means the VAT
    is stated separately and only a VAT-registered buyer gets it back. Publishing
    one as if it were the other moves the comparison by more than the whole
    saving, so the label rides along with every price and the aggregation keeps
    the two apart.
    """
    label = price.get("vatLabel")
    if not label:
        return None, None
    low = str(label).lower()
    if "inkl" in low:
        return str(label), False
    if "ausweisbar" in low or "zzgl" in low or "exkl" in low:
        return str(label), True
    return str(label), None


def parse_search(html: str) -> tuple[list[DeListing], dict]:
    """(listings, meta) from a search page's ``__NEXT_DATA__``.

    ``meta`` carries ``results`` and ``pages`` so the runner can stop paging
    instead of guessing. A page whose JSON is missing or reshaped returns an
    empty list and an empty meta — the caller treats that as "stop", never as
    "no cars in Germany".
    """
    m = _NEXT_DATA_RE.search(html or "")
    if not m:
        return [], {}
    try:
        doc = json.loads(m.group(1))
    except json.JSONDecodeError:
        logger.warning("autoscout: __NEXT_DATA__ did not parse")
        return [], {}
    props = (doc.get("props") or {}).get("pageProps") or {}
    raw = props.get("listings")
    if not isinstance(raw, list):
        return [], {}
    meta = {"results": props.get("numberOfResults"), "pages": props.get("numberOfPages")}
    out = []
    for item in raw:
        parsed = _to_listing(item)
        if parsed is not None:
            out.append(parsed)
    return out, meta


def _to_listing(item: dict) -> DeListing | None:
    vehicle = item.get("vehicle") or {}
    price = item.get("price") or {}
    tracking = item.get("tracking") or {}
    external_id = str(item.get("id") or item.get("crossReferenceId") or "").strip()
    brand = str(vehicle.get("make") or "").strip()
    model = str(vehicle.get("model") or "").strip()
    if not external_id or not brand or not model:
        return None
    price_eur = price.get("priceRaw")
    price_eur = float(price_eur) if isinstance(price_eur, (int, float)) else _num(tracking.get("price"))
    vat_label, vat_reclaimable = _vat(price)
    year, reg_month = _registration(item)
    power_kw, power_ps = _power(item)
    location = item.get("location") or {}
    seller = item.get("seller") or {}
    url = str(item.get("url") or "")
    fuel_raw = str(vehicle.get("fuel") or "").strip()
    gearbox_raw = str(vehicle.get("transmission") or "").strip()
    fuel_pt = normalize_fuel_type(_FUEL_DE_TO_PT.get(fuel_raw.lower(), fuel_raw or None))
    return DeListing(
        external_id=external_id,
        url=(BASE_URL + url) if url.startswith("/") else url,
        brand=brand,
        model=model,
        price_eur=price_eur,
        vat_label=vat_label,
        vat_reclaimable=vat_reclaimable,
        model_group=str(vehicle.get("modelGroup") or "").strip() or None,
        variant=str(vehicle.get("variant") or "").strip() or None,
        motor_type=str(vehicle.get("motorTypeName") or "").strip() or None,
        year=year,
        registration_month=reg_month,
        mileage_km=_int(tracking.get("mileage")) or _int(vehicle.get("mileageInKm")),
        engine_cc=_int(vehicle.get("engineDisplacementInCCM")),
        horsepower=power_ps,
        power_kw=power_kw,
        fuel_type=fuel_pt,
        transmission=_TRANSMISSION_DE_TO_PT.get(gearbox_raw.lower(), gearbox_raw or None),
        co2_g_km=_co2(item, fuel_pt),
        seller_type=str(seller.get("type") or "").strip() or None,
        country_code=str(location.get("countryCode") or "").strip() or None,
        city=str(location.get("city") or "").strip() or None,
        zip_code=str(location.get("zip") or "").strip() or None,
        is_damaged=vehicle.get("isCurrentlyDamaged"),
    )


def search_path(make: str, model: str, *, year: int | None = None, page: int = 1,
                country: str = "D", body: str | None = None) -> str:
    """The path+query for one model-year page, in the form robots.txt leaves open.

    ``body`` is AutoScout24's body-type segment (``bt_kombi`` and friends). It
    exists because half the Portuguese estate vocabulary — "308 SW", "Leon ST",
    "Mégane Sport Tourer" — is a body type there rather than a model, so those
    models are unreachable without it and were coming back 404.
    """
    params = {
        "atype": "C",
        "cy": country,
        "damaged_listing": "exclude",
        "powertype": "kw",
        "sort": "standard",
        "ustate": "N,U",
    }
    if year:
        params["fregfrom"] = year
        params["fregto"] = year
    if page and page > 1:
        params["page"] = page
    tail = f"/{body}" if body else ""
    return f"{SEARCH_PATH}/{make}/{model}{tail}?{urlencode(params)}"


@dataclass
class AutoScoutClient:
    """Serial, self-identifying, budgeted reader of AutoScout24 search pages."""

    config: AutoScoutConfig = field(default_factory=AutoScoutConfig)
    spent: int = 0
    _client: httpx.Client | None = None

    def __enter__(self):
        self._client = httpx.Client(
            timeout=self.config.timeout,
            follow_redirects=True,
            headers={"User-Agent": self.config.user_agent,
                     "Accept-Language": "de-DE,de;q=0.9"},
        )
        return self

    def __exit__(self, *exc):
        if self._client is not None:
            self._client.close()
            self._client = None
        return False

    def _sleep(self):
        time.sleep(random.uniform(self.config.delay_min, self.config.delay_max))

    def fetch(self, path: str) -> str | None:
        """One page, or None when we are out of budget or the path is closed.

        Raises ``AutoScoutBlocked`` on 429/403 so the caller stops the whole run:
        the polite response to being asked to go away is to go away, not to
        rotate a header and try again.
        """
        if not robots_allows(path):
            logger.warning("autoscout: robots.txt disallows %s — skipped", path)
            return None
        if self.spent >= self.config.budget:
            return None
        if self._client is None:
            raise RuntimeError("AutoScoutClient must be used as a context manager")
        if self.spent:
            self._sleep()
        self.spent += 1
        resp = self._client.get(BASE_URL + path)
        if resp.status_code in (403, 429):
            raise AutoScoutBlocked(f"{resp.status_code} on {path}")
        if resp.status_code >= 400:
            logger.warning("autoscout: %s on %s", resp.status_code, path)
            return None
        return resp.text

    def model_year(self, make: str, model: str, year: int, *, max_pages: int = 1,
                   body: str | None = None) -> list[DeListing]:
        """Every listing we are willing to read for one model in one year."""
        found: list[DeListing] = []
        for page in range(1, max(1, min(max_pages, MAX_PAGE)) + 1):
            html = self.fetch(search_path(make, model, year=year, page=page,
                                          country=self.config.country, body=body))
            if html is None:
                break
            listings, meta = parse_search(html)
            if not listings:
                break
            found.extend(listings)
            if page >= (meta.get("pages") or 1):
                break
        return found
