"""automobile.it — the second Italian market reader, beside AutoScout24.

Italy is the first market this project reads through two classifieds at once,
and the reason is coverage rather than tidiness: AutoScout24's Italian site and
automobile.it do not hold the same inventory, and a per-country price model
fitted on half a market is fitted on whichever half we happened to read.

The payload arrives the same way it does on AutoScout24 and Standvirtual — a
Next.js page with the whole result list in ``__NEXT_DATA__`` — and the advert
page carries a second one. Both are read, because a row here goes in the same
table as a Portuguese one and has to be able to answer the same questions: the
card gives price, registration, mileage, fuel and gearbox, and the advert adds
the seller's own text, the dealer behind it and the colour. Every value is
mapped onto this project's Portuguese vocabulary before it becomes a card, and
the Italian half of that vocabulary already exists for AutoScout24's ``.it``
site, so this module borrows ``autoscout.market("it")`` rather than writing a
second, drifting copy of the same fuel table.

Reading the advert is what makes these rows comparable, and it is also what
makes them expensive: twenty cars used to cost one request and now cost
twenty-one. The crawler's budget is the only thing holding that down, so it is
a number to set deliberately rather than raise when a run comes back short.

**How this behaves on someone else's site.** The site publishes a JSON API and
``robots.txt`` closes it: ``/rest/*`` and ``/api/*`` are disallowed for
``User-agent: *``. The same payload is embedded in the search page, which is
not disallowed, so this reader takes the long way round on purpose and asks for
the HTML. It also keeps off ``?b=`` and ``?d=`` query forms, which the file
disallows, by paginating through the path (``/fiat/page-2``). It identifies
itself with our own User-Agent, waits ``DELAY_MIN``-``DELAY_MAX`` seconds
between requests, and stops the run on a 403 or 429 instead of retrying into a
ban.

One limit worth stating rather than discovering: a query caps at 500 pages of
twenty, so a brand with more than ten thousand cars for sale is not fully
enumerable from one path. Fiat alone lists thirty-three thousand. The crawler
reads brands in rotation and sorts nothing, so what we hold is a sample that
refreshes rather than a census, and no code downstream should treat a missing
car as a sold one for this source.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from src.parser.autoscout import _label_key, market
from src.parser.market_card import MarketCard

SOURCE = "am_it"
COUNTRY = "IT"
HOST = "www.automobile.it"
BASE_URL = f"https://{HOST}"
USER_AGENT = ("Mozilla/5.0 (compatible; CarsbuyerBot/1.0; "
              "+https://carsbuyer.org/pt/sobre)")
ACCEPT_LANGUAGE = "it-IT,it;q=0.9"
DELAY_MIN = 6.0
DELAY_MAX = 10.0
TIMEOUT = 30.0
PAGE_SIZE = 20
MAX_PAGES = 500

CORRECTS: tuple[str, ...] = ()

_DISALLOWED_PREFIXES = (
    "/mappa-province/", "/area-privata", "/fb/", "/rest/", "/api/", "/post/",
)

_NEXT_DATA_RE = re.compile(
    r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', re.S)

_MONTHS = {
    "gennaio": 1, "febbraio": 2, "marzo": 3, "aprile": 4, "maggio": 5,
    "giugno": 6, "luglio": 7, "agosto": 8, "settembre": 9, "ottobre": 10,
    "novembre": 11, "dicembre": 12,
}

_SELLERS = {"dealer": "Profissional", "private": "Particular"}

_EXTRA_FLAGS = ("dekraAllowed", "carfaxAllowed", "hasQualitySeal",
                "hasGeneralWarranty", "hasScarcityTag")


class AutomobileItBlocked(RuntimeError):
    """The site asked us to stop: 403, 429, or a page that is not a result list."""


def robots_allows(path: str) -> bool:
    """Whether ``User-agent: *`` leaves this path open.

    Kept as code rather than a comment so the reader cannot drift away from
    what the file says. The JSON endpoints this page is built on are among the
    disallowed prefixes, which is why the reader asks for the HTML instead.
    """
    p = path if path.startswith("/") else "/" + path
    if "?b=" in p or "?d=" in p:
        return False
    return not any(p.startswith(pre) for pre in _DISALLOWED_PREFIXES)


def search_path(brand_slug: str, page: int = 1) -> str:
    """The path for one brand's results, paginated through the path itself."""
    slug = str(brand_slug or "").strip().strip("/").lower()
    if not slug:
        return "/annunci" if page <= 1 else f"/annunci/page-{page}"
    return f"/{slug}" if page <= 1 else f"/{slug}/page-{page}"


def _price(formatted: str | None) -> float | None:
    """'€ 4.950' → 4950.0. A price on request has no number and stays None."""
    if not formatted:
        return None
    digits = re.sub(r"[^\d]", "", str(formatted))
    return float(digits) if digits else None


def _int(text) -> int | None:
    if text is None:
        return None
    digits = re.sub(r"[^\d]", "", str(text))
    return int(digits) if digits else None


def _registration(raw: str | None) -> tuple[int | None, str | None]:
    """'Maggio 2018' → (2018, '05/2018'); a bare year keeps the month empty."""
    if not raw:
        return None, None
    text = str(raw).strip().lower()
    year_match = re.search(r"(19|20)\d{2}", text)
    if not year_match:
        return None, None
    year = int(year_match.group(0))
    for name, num in _MONTHS.items():
        if name in text:
            return year, f"{num:02d}/{year}"
    return year, None


def _power(raw: str | None) -> tuple[int | None, int | None]:
    """'69 CV (51 KW)' → (69, 51). Either half may be missing."""
    if not raw:
        return None, None
    cv = re.search(r"(\d+)\s*CV", str(raw), re.I)
    kw = re.search(r"(\d+)\s*KW", str(raw), re.I)
    return (int(cv.group(1)) if cv else None), (int(kw.group(1)) if kw else None)


def _fuel(raw: str | None) -> str | None:
    """'GPL - Euro 6' → 'GPL', through the Italian table AutoScout24 already has."""
    if not raw:
        return None
    head = str(raw).split("-")[0]
    table = market("it").fuels
    key = _label_key(head)
    if key in table:
        return table[key]
    return head.strip() or None


def _transmission(raw: str | None) -> str | None:
    if not raw:
        return None
    table = market("it").gearboxes
    key = _label_key(raw)
    if key in table:
        return table[key]
    return str(raw).strip() or None


def _location(raw: str | None) -> tuple[str | None, str | None]:
    """'Uboldo (VA)' → ('Uboldo', 'VA'). The parenthesis is the province code."""
    if not raw:
        return None, None
    text = str(raw).strip()
    m = re.match(r"^(.*?)\s*\(([A-Za-z]{2})\)\s*$", text)
    if m:
        return m.group(1).strip() or None, m.group(2).upper()
    return text or None, None


def _brand_model(title: str, brand_hint: str | None) -> tuple[str, str]:
    """Split 'Fiat Panda 1.2 EasyPower Lounge' into make and model.

    The brand we already know, because the crawler asked for one brand's page;
    the model is the next word. Falling back to the title's own first word
    keeps an unfiltered ``/annunci`` page readable, at the cost of getting a
    two-word make like Alfa Romeo wrong — which is why the crawler does not
    use that page for anything it stores.
    """
    words = str(title or "").split()
    if brand_hint:
        brand = brand_hint.replace("_", " ").strip()
        lowered = [w.lower() for w in words]
        parts = brand.lower().split()
        if lowered[:len(parts)] == parts:
            rest = words[len(parts):]
        else:
            rest = words[1:] if words else []
        return brand.title(), (rest[0] if rest else "")
    if not words:
        return "", ""
    return words[0], (words[1] if len(words) > 1 else "")


def _card(item: dict, brand_hint: str | None) -> MarketCard | None:
    """One result-list entry as a card, or None when it is not a used car."""
    external_id = str(item.get("id") or "").strip()
    url = str(item.get("url") or "").strip()
    if not external_id or not url:
        return None
    if str(item.get("detailsType") or "").upper() not in ("USED", ""):
        return None
    if item.get("rental"):
        return None

    details = item.get("details") or {}
    year, reg_month = _registration(details.get("registration"))
    horsepower, power_kw = _power(details.get("formattedPower"))
    city, region = _location(item.get("location"))
    brand, model = _brand_model(item.get("title"), brand_hint)

    extras = {k: item[k] for k in _EXTRA_FLAGS if item.get(k)}
    if item.get("dealerName"):
        extras["dealer_name"] = item["dealerName"]

    return MarketCard(
        source=SOURCE,
        external_id=external_id,
        url=url if url.startswith("http") else BASE_URL + url,
        brand=brand,
        model=model,
        country_code=COUNTRY,
        price_eur=_price(item.get("formattedPrice")),
        price_label=item.get("formattedPrice"),
        version=item.get("title"),
        year=year,
        registration_month=reg_month,
        mileage_km=_int(details.get("formattedKm")),
        engine_cc=_int(details.get("formattedEngineCapacity")),
        horsepower=horsepower,
        power_kw=power_kw,
        fuel_type=_fuel(details.get("fuelEmissions")),
        transmission=_transmission(details.get("shift")),
        seller_type=_SELLERS.get(str(item.get("sellerType") or "").lower()),
        city=city,
        region=region,
        photo_count=item.get("totalNumberOfPictures"),
        image_url=item.get("firstMediumSizeImage"),
        extras=extras,
    )


def parse_search(html: str, brand_slug: str | None = None
                 ) -> tuple[list[MarketCard], dict]:
    """Cards and paging info out of one search page.

    Returns the page's own ``total``/``pages`` alongside the cards so the
    crawler can stop at the real end of a brand rather than walking into empty
    pages, and can see when a brand is larger than the 500-page ceiling.
    """
    match = _NEXT_DATA_RE.search(html or "")
    if not match:
        raise AutomobileItBlocked("no __NEXT_DATA__ in page")
    try:
        props = json.loads(match.group(1))["props"]["pageProps"]
    except (KeyError, ValueError) as exc:
        raise AutomobileItBlocked(f"unreadable __NEXT_DATA__: {exc}") from exc

    result = (props.get("apiResults") or {}).get("result")
    if not isinstance(result, dict):
        raise AutomobileItBlocked("page carries no result list")

    page = result.get("page") or {}
    meta = {
        "total": page.get("totalElements"),
        "pages": page.get("totalPages"),
        "number": page.get("number"),
        "size": page.get("size") or PAGE_SIZE,
    }
    cards = [c for c in (_card(item, brand_slug)
                         for item in result.get("resultList") or [])
             if c is not None]
    return cards, meta


def brand_slugs(html: str) -> list[str]:
    """Every brand the search page offers, as path slugs.

    Read from the page the crawler is already fetching rather than kept in a
    config file, so a make that appears or disappears on the site does so here
    too without an edit.
    """
    match = _NEXT_DATA_RE.search(html or "")
    if not match:
        return []
    try:
        brands = json.loads(match.group(1))["props"]["pageProps"]["brands"]["all"]
    except (KeyError, ValueError, TypeError):
        return []
    slugs = []
    for entry in brands:
        slug = str((entry or {}).get("slugLabel") or "").strip().lower()
        if slug:
            slugs.append(slug)
    return slugs


def parse_detail(html: str) -> dict:
    """What the advert page adds to a card: the seller's own text and the colour.

    The card is a summary the site writes; this is what the seller wrote. The
    description is the input every downstream language step reads, the dealer
    block is the only seller identity this source publishes, and the colour
    lives here and nowhere on the card.

    Returns a patch to lay over a card rather than a whole card, so a detail
    fetch that fails costs the extra fields and not the listing.
    """
    match = _NEXT_DATA_RE.search(html or "")
    if not match:
        raise AutomobileItBlocked("no __NEXT_DATA__ on advert page")
    try:
        props = json.loads(match.group(1))["props"]["pageProps"]
    except (KeyError, ValueError) as exc:
        raise AutomobileItBlocked(f"unreadable advert page: {exc}") from exc

    result = props.get("result")
    if not isinstance(result, dict):
        raise AutomobileItBlocked("advert page carries no result")

    patch: dict = {}
    extras: dict = {}
    description = str(result.get("description") or "").strip()
    if description:
        patch["description"] = description
        patch["description_length"] = len(description)

    dealer = result.get("dealer") or {}
    if isinstance(dealer, dict) and dealer.get("name"):
        patch["seller_displayed_as"] = dealer["name"]
        if dealer.get("canonicalUrl"):
            extras["dealer_url"] = dealer["canonicalUrl"]

    info = props.get("vehicleInformation") or {}
    for entry in (info.get("aesthetic") or []):
        title = _label_key(entry.get("title"))
        values = [v for v in (entry.get("values") or []) if v]
        if not values:
            continue
        if "colore esterno" in title:
            patch["color"] = values[0]
        elif "colore interno" in title:
            extras["interior_colour"] = values[0]
    for group in ("accessories", "consumption"):
        for entry in (info.get(group) or []):
            values = [v for v in (entry.get("values") or []) if v]
            if entry.get("title") and values:
                extras.setdefault(group, {})[str(entry["title"])] = values

    details = result.get("details") or {}
    if details.get("numberOfPreviousOwners") is not None:
        extras["previous_owners"] = details["numberOfPreviousOwners"]

    if extras:
        patch["extras"] = extras
    return patch


def detail_path(url: str) -> str:
    """The advert's path, whether the card gave a path or a full URL."""
    text = str(url or "")
    if text.startswith("http"):
        return "/" + text.split("/", 3)[-1] if text.count("/") >= 3 else "/"
    return text if text.startswith("/") else "/" + text


@dataclass
class AutomobileItConfig:
    delay_min: float = DELAY_MIN
    delay_max: float = DELAY_MAX
    timeout: float = TIMEOUT
    budget: int = 120
    user_agent: str = USER_AGENT
