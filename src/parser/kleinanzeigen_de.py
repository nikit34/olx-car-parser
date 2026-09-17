"""Kleinanzeigen — the private half of the German market, which AutoScout24 is not.

AutoScout24 is a dealer's site. Kleinanzeigen is where German private sellers
advertise, and the two price the same car differently; a German corpus read
only through AutoScout24 is a corpus of forecourt prices. This reader is here
for that gap, not for volume.

There is no JSON payload: the search page ships the cards as server-rendered
markup, and what this module reads is the tokens the site prints for humans —
``16.600 € VB``, ``160.581 km``, ``EZ 03/2016``. That sounds fragile and is in
fact the opposite: those are the site's own user-facing conventions and outlive
the Tailwind class names around them, which change between deploys. Each card
also embeds a small ``ld+json`` object with the title and image, and that is
taken as given rather than scraped out of the markup.

**What robots.txt allows, and what it therefore costs.** Two limits shape this
reader and both come from the file:

* Only the first five pages of any search are open — ``/*/seite:6*`` through
  ``/*/seite:59*`` are disallowed. Five pages of twenty-seven is a hundred and
  thirty-five cars per path, so the only way to a useful sample is one path per
  make, and even then what we hold refreshes rather than accumulates. Nothing
  downstream may read a car's absence here as a sale.
* ``/*/anbieter:*`` is disallowed, which is the seller-type filter. So this
  source cannot say whether an ad is a dealer's or a private seller's, and
  ``seller_type`` stays None rather than being guessed from the wording of the
  description. Guessing it would be worse than not knowing: the whole reason
  this reader exists is the private-seller price level, and a wrong label there
  poisons exactly the comparison it was added for.

The card carries no fuel, gearbox or engine — those live on the advert page,
so the advert page is read too. It is worth the request: behind it sits a flat
key-value block with ``Kraftstoffart``, ``Getriebe``, ``Leistung``,
``Fahrzeugtyp``, ``Aussenfarbe``, ``Anzahl_Tueren`` and the seller's own text,
which is the difference between a row that can be priced against a Portuguese
one and a row that only has a number on it.

That costs one request per car on a site where five pages is the ceiling, so a
pass is small by construction: a hundred and thirty-five cars a make, each one
a fetch. The budget is what bounds it, and nothing here retries.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from src.parser.market_card import MarketCard

SOURCE = "ka_de"
COUNTRY = "DE"
HOST = "www.kleinanzeigen.de"
BASE_URL = f"https://{HOST}"
USER_AGENT = ("Mozilla/5.0 (compatible; CarsbuyerBot/1.0; "
              "+https://carsbuyer.org/pt/sobre)")
ACCEPT_LANGUAGE = "de-DE,de;q=0.9"
DELAY_MIN = 6.0
DELAY_MAX = 10.0
TIMEOUT = 30.0
CATEGORY = "c216"
MAX_PAGE = 5

_ARTICLE_RE = re.compile(r'<article[^>]*data-adid="(\d+)"(.*?)</article>', re.S)
_LD_RE = re.compile(r'\{"creditText".*?"@type":"ImageObject"\}', re.S)
_HREF_RE = re.compile(r'data-href="(/s-anzeige/[^"]+)"')
_TAG_RE = re.compile(r'<[^>]+>')
_PRICE_RE = re.compile(r'([\d.]+)\s*€(\s*VB)?')
_KM_RE = re.compile(r'([\d.]+)\s*km\b', re.I)
_EZ_RE = re.compile(r'EZ\s*(\d{1,2})/(\d{4})')
_ZIP_RE = re.compile(r'\b(\d{5})\s+([A-Za-zÄÖÜäöüß][^|]*?)\s*(?:\||$)')


class KleinanzeigenBlocked(RuntimeError):
    """The site asked us to stop: 403, 429, or a page with no cards on it."""


def robots_allows(path: str) -> bool:
    """Whether ``User-agent: *`` leaves this path open.

    The page-depth ceiling and the seller-type filter are both robots rules
    rather than conventions, so they are enforced here and the crawler asks
    before every fetch. A path this returns False for is skipped, not retried.
    """
    p = path if path.startswith("/") else "/" + path
    if re.search(r'/anbieter:', p):
        return False
    page = re.search(r'/seite:(\d+)', p)
    if page and int(page.group(1)) > MAX_PAGE:
        return False
    for blocked in ("/s-feed.rss", "/s-kategorie-baum.html", "/s-bestandsliste.html",
                    "/s-suchanfrage.html", "/s-oac", "/s-poac", "/s-direktkaufen:aktiv"):
        if p.startswith(blocked):
            return False
    if re.search(r'/s-anzeige:(gesuche|angebote)', p) or "/anzeige:" in p:
        return False
    return True


def search_path(brand_slug: str | None = None, page: int = 1) -> str:
    """The path for one make's car listings, at most five pages deep.

    Pages beyond the ceiling are not silently clamped: asking for one is a bug
    in the caller, and returning page five under a page-nine label would make
    the crawler re-read the same cars believing it had moved on.
    """
    if page > MAX_PAGE or page < 1:
        raise ValueError(f"page {page} is outside the crawlable range 1..{MAX_PAGE}")
    slug = str(brand_slug or "").strip().strip("/").lower()
    parts = ["/s-autos"]
    if slug:
        parts.append(slug)
    if page > 1:
        parts.append(f"seite:{page}")
    parts.append(CATEGORY)
    return "/".join(parts)


def _int(text) -> int | None:
    if text is None:
        return None
    digits = re.sub(r"[^\d]", "", str(text))
    return int(digits) if digits else None


def _ld(body: str) -> dict:
    match = _LD_RE.search(body)
    if not match:
        return {}
    try:
        doc = json.loads(match.group(0))
    except ValueError:
        return {}
    return doc if isinstance(doc, dict) else {}


def _brand_model(title: str, brand_hint: str | None) -> tuple[str, str]:
    """Make and model out of a free-text headline.

    Both are provisional. The make path is a text search, so the make asked for
    is only what the advert mentions, and the headline is whatever the seller
    felt like typing. ``parse_detail`` replaces both from the advert's own
    fields, and the crawler drops any row it could not read an advert for
    rather than storing this guess.
    """
    words = [w for w in re.split(r"[\s,]+", str(title or "")) if w]
    if brand_hint:
        brand = brand_hint.replace("-", " ").replace("_", " ").strip()
        parts = brand.lower().split()
        lowered = [w.lower() for w in words]
        rest = words[len(parts):] if lowered[:len(parts)] == parts else words[1:]
        return brand.title(), (rest[0] if rest else "")
    if not words:
        return "", ""
    return words[0], (words[1] if len(words) > 1 else "")


def _card(adid: str, body: str, brand_hint: str | None) -> MarketCard | None:
    href_match = _HREF_RE.search(body)
    if not href_match:
        return None
    ld = _ld(body)
    text = _TAG_RE.sub(" | ", _LD_RE.sub(" ", body))
    text = re.sub(r"(\s*\|\s*)+", " | ", text).strip()

    prices = _PRICE_RE.findall(text)
    price_eur = _int(prices[0][0]) if prices else None
    negotiable = bool(prices and prices[0][1])

    km_match = _KM_RE.search(text)
    ez_match = _EZ_RE.search(text)
    zip_match = _ZIP_RE.search(text)

    year = int(ez_match.group(2)) if ez_match else None
    reg_month = f"{int(ez_match.group(1)):02d}/{year}" if ez_match else None

    title = str(ld.get("title") or "").strip()
    brand, model = _brand_model(title, brand_hint)

    extras = {}
    if negotiable:
        extras["negotiable"] = True
    if len(prices) > 1:
        extras["price_before_eur"] = _int(prices[1][0])

    return MarketCard(
        source=SOURCE,
        external_id=str(adid),
        url=BASE_URL + href_match.group(1),
        brand=brand,
        model=model,
        country_code=COUNTRY,
        price_eur=float(price_eur) if price_eur else None,
        version=title or None,
        year=year,
        registration_month=reg_month,
        mileage_km=_int(km_match.group(1)) if km_match else None,
        zip_code=zip_match.group(1) if zip_match else None,
        city=zip_match.group(2).strip() if zip_match else None,
        image_url=str(ld.get("contentUrl") or "").strip() or None,
        extras=extras,
    )


def parse_search(html: str, brand_slug: str | None = None
                 ) -> tuple[list[MarketCard], dict]:
    """Cards out of one search page, with what the page says about itself.

    An empty page raises rather than returning nothing: the site answers a
    blocked request with a 200 and a page that simply has no articles on it, so
    "no cards" is the shape a refusal arrives in and must not be mistaken for
    a make with no cars for sale.
    """
    blocks = _ARTICLE_RE.findall(html or "")
    if not blocks:
        raise KleinanzeigenBlocked("page carries no ad cards")
    cards = [c for c in (_card(adid, body, brand_slug) for adid, body in blocks)
             if c is not None]
    return cards, {"cards": len(cards), "articles": len(blocks)}


CORRECTS: tuple[str, ...] = ("brand", "model")

_ATTR_RE = re.compile(r'"([A-Za-z_][A-Za-z0-9_%]*)"\s*:\s*"([^"]*)"')
_DESC_RE = re.compile(
    r'id="viewad-description-text"[^>]*>(.*?)</p>', re.S)

_ATTR_KEYS = {
    "kraftstoffart", "getriebe", "leistung", "fahrzeugtyp", "aussenfarbe",
    "anzahl_tueren", "kilometerstand", "erstzulassungsjahr",
    "erstzulassungsmonat", "marke", "modell", "schadstoffklasse",
    "fahrzeugzustand", "material_innenausstattung", "umweltplakette",
}

_FUELS = {
    "diesel": "Diesel",
    "benzin": "Gasolina",
    "elektro": "Eléctrico",
    "hybrid": "Híbrido (Gasolina)",
    "autogas": "GPL",
    "lpg": "GPL",
    "erdgas": "GNC",
    "cng": "GNC",
    "wasserstoff": "Hidrogénio",
    "andere": None,
    "sonstige": None,
}

_GEARBOXES = {"manuell": "Manual", "automatik": "Automática"}

_BODY_TYPES = {
    "limousine": "Sedan", "kleinwagen": "Pequeno Citadino", "kombi": "Carrinha",
    "suv": "SUV/TT", "gelaendewagen": "SUV/TT", "cabrio": "Cabrio",
    "sportwagen": "Coupé", "coupe": "Coupé", "van": "Monovolume",
    "bus": "Monovolume", "andere": None,
}

_COLOURS = {
    "weiss": "Branco", "schwarz": "Preto", "grau": "Cinzento", "silber": "Prateado",
    "blau": "Azul", "rot": "Vermelho", "gruen": "Verde", "gelb": "Amarelo",
    "braun": "Castanho", "beige": "Bege", "orange": "Laranja", "gold": "Dourado",
}


def _attributes(html: str) -> dict:
    """The advert's own key-value block, lowercased and deduplicated.

    The block is repeated on the page with the same keys, so the first value
    for a key wins and later copies are ignored rather than fighting over it.
    Only the keys this reader knows are kept: the page carries a few hundred,
    most of them equipment booleans and ad-targeting noise.
    """
    found: dict[str, str] = {}
    for key, value in _ATTR_RE.findall(html or ""):
        slug = key.lower()
        if slug in _ATTR_KEYS and slug not in found and value:
            found[slug] = value
    return found


def _lookup(table: dict[str, str | None], raw: str | None) -> str | None:
    """A site word through a table, passed through unchanged when unknown.

    An unmapped value stays itself rather than becoming None, so a body style
    or a colour the site adds next month shows up in the data as itself instead
    of disappearing; a value the table maps to None is the seller declining to
    answer and is dropped on purpose.
    """
    if not raw:
        return None
    key = str(raw).strip().lower()
    if key in table:
        return table[key]
    for name, mapped in table.items():
        if key.startswith(name):
            return mapped
    return str(raw).strip() or None


def parse_detail(html: str) -> dict:
    """What the advert page adds to a card: specs, colour and the seller's text.

    Only ``CORRECTS`` overwrites what the card already knew; everything else
    fills a gap. The mileage is why: the card prints the exact figure the seller
    typed and the advert's own field is rounded to a bracket, so taking the
    advert's would quietly turn 160 581 km into 150 000 for every German row.

    It also corrects the make and the model, and that is not a refinement. The
    make path on this site is a text search rather than a filter — asking for
    Volkswagen returns any advert that says the word, including a Renault whose
    seller mentioned their old Golf — and a private seller's headline is not a
    field. ``Marke`` and ``Modell`` here are the site's own structured values,
    so they are the only ones worth storing, and a listing whose advert we
    could not read has no trustworthy make at all.
    """
    attrs = _attributes(html)
    if not attrs:
        raise KleinanzeigenBlocked("advert page carries no attribute block")

    patch: dict = {}
    extras: dict = {}

    fuel = _lookup(_FUELS, attrs.get("kraftstoffart"))
    if fuel:
        patch["fuel_type"] = fuel
    gearbox = _lookup(_GEARBOXES, attrs.get("getriebe"))
    if gearbox:
        patch["transmission"] = gearbox
    body = _lookup(_BODY_TYPES, attrs.get("fahrzeugtyp"))
    if body:
        patch["body_type"] = body
        patch["segment"] = body
    colour = _lookup(_COLOURS, attrs.get("aussenfarbe"))
    if colour:
        patch["color"] = colour

    if attrs.get("leistung"):
        patch["horsepower"] = _int(attrs["leistung"])
    if attrs.get("kilometerstand"):
        patch["mileage_km"] = _int(attrs["kilometerstand"])
        extras["mileage_bucket_km"] = _int(attrs["kilometerstand"])
    if attrs.get("anzahl_tueren"):
        patch["doors"] = attrs["anzahl_tueren"].replace("_", "-")
    if attrs.get("marke"):
        patch["brand"] = attrs["marke"].replace("_", " ").title()
    if attrs.get("modell"):
        patch["model"] = attrs["modell"].replace("_", " ").title()

    year = _int(attrs.get("erstzulassungsjahr"))
    month = _int(attrs.get("erstzulassungsmonat"))
    if year:
        patch["year"] = year
        if month:
            patch["registration_month"] = f"{month:02d}/{year}"

    for key in ("schadstoffklasse", "fahrzeugzustand", "material_innenausstattung",
                "umweltplakette"):
        if attrs.get(key):
            extras[key] = attrs[key]

    match = _DESC_RE.search(html or "")
    if match:
        text = _TAG_RE.sub(" ", match.group(1))
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            patch["description"] = text
            patch["description_length"] = len(text)

    if extras:
        patch["extras"] = extras
    return patch


@dataclass
class KleinanzeigenConfig:
    delay_min: float = DELAY_MIN
    delay_max: float = DELAY_MAX
    timeout: float = TIMEOUT
    budget: int = 120
    user_agent: str = USER_AGENT
