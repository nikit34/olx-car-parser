"""AutoScout24 search reader — the German, French and Italian sides of the market.

This started as one question the Portuguese corpus cannot answer from itself:
does buying a given model in Germany and nationalising it beat buying it here.
The Portuguese half we own; the German half is one number we do not have — what
the same car asks in Germany today — and this module is how it arrives.

It now reads three national sites rather than one. AutoScout24 runs the same
Next.js application on ``.de``, ``.fr`` and ``.it``, with the same card shape
and the same ``__NEXT_DATA__`` payload; what changes between them is the
vocabulary printed on the card (``Benzin`` / ``Essence`` / ``Benzina``), the
aria labels those values hang off, and the unit horsepower is quoted in
(``PS`` / ``Ch`` / ``CV``). So the reader is one parser with a per-market
vocabulary table, not three parsers: a field that moves on one site moves on
all three, and a translation that is wrong is wrong in one visible place.

Everything a card yields is mapped into this project's Portuguese vocabulary on
the way out — ``Diesel``/``Gasolina``/``Eléctrico``, ``Manual``/``Automática``,
``Profissional``/``Particular`` — because the price model, the deal builders and
the pages are all written against that vocabulary and a per-country dialect in
the database is a per-country bug in every one of them.

A card is most of a car and not all of it. The result page ships twenty cards,
exactly like Standvirtual (see ``scraper._sv_advert_from_html``), and that is
where the price and its VAT label, the mileage, the power and the first
registration come from. What no card carries on any of the three sites is the
body type, the colour, the drive train and the seller's own text, and outside
Germany it carries no CO2 either; those live on the advert, which is opened per
car where robots.txt leaves it open and never where it does not. Germany is the
market where it does not, so its body type is recovered from the site's own
body filter instead — see ``Market.body_filters`` — and the rest of what the
advert would have said is simply not had.

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
  in code rather than in a comment nobody re-reads. The three national files
  agree on the search prefixes and part ways on the advert, which is why that
  judgement takes a market.
* It is slow on purpose: ``DELAY_MIN``/``DELAY_MAX`` seconds between requests,
  serial, one request per model-year, and a hard ``budget`` per run so a bug
  cannot turn into a flood. A 429 or a 403 stops the run then and there instead
  of retrying into a ban.
* What it publishes differs by page, and the difference is worth stating
  rather than leaving for someone to discover. The import comparison
  (``scripts/crawl_autoscout.py``, source ``autoscout24``) takes aggregates
  only: the median of a model-year, never a copy of somebody's listing. The
  country corpora (``scripts/crawl_eu_market.py``) publish those medians too,
  and on top of them a feed that shows individual cars, each with its photo,
  price, specs and a link to the advert itself. That is a republication of
  someone else's inventory, and calling it anything softer here would be a
  comfortable lie. What bounds it is real, though: the seller's own text is
  never stored for these corpora (``get_country_listings_df`` drops the
  description), the photo is displayed from AutoScout24's own servers rather
  than copied onto ours, every card sends the reader out to the source, and a
  listing nobody has re-confirmed lately is withheld instead of being passed
  off as live.
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
import unicodedata
from dataclasses import dataclass, field
from html import unescape as _unescape
from urllib.parse import urlencode

import httpx

from src.parser import regions
from src.parser.fuel_normalize import normalize_fuel_type

logger = logging.getLogger(__name__)

BASE_URL = "https://www.autoscout24.de"
SEARCH_PATH = "/lst"
PAGE_SIZE = 20
MAX_PAGE = 20
USER_AGENT = ("Mozilla/5.0 (compatible; CarsbuyerBot/1.0; "
              "+https://carsbuyer.org/pt/sobre)")
DELAY_MIN = 6.0
DELAY_MAX = 10.0
TIMEOUT = 30.0

_DISALLOWED_PREFIXES = (
    "/private-feedback/", "/dealerarea/", "/entry/", "/ergebnisse?", "/i/",
    "/modelle/page/", "/regional/page/", "/lst?", "/lst/?", "/lst-moto?",
    "/lst-moto/?", "/Partner/", "/partner/", "/favorites",
)

_DISALLOWED_BY_TLD: dict[str, tuple[str, ...]] = {
    "de": ("/angebote/", "/auto-abo/angebote/"),
    "fr": ("/offres/-",),
    "it": (),
}

CORRECTS: tuple[str, ...] = ("photo_urls",)

_COLOURS: dict[str, str | None] = {
    "white": "Branco", "black": "Preto", "grey": "Cinzento", "gray": "Cinzento",
    "silver": "Prateado", "blue": "Azul", "red": "Vermelho", "green": "Verde",
    "yellow": "Amarelo", "brown": "Castanho", "beige": "Bege",
    "orange": "Laranja", "gold": "Dourado", "violet": "Roxo", "purple": "Roxo",
    "bronze": "Bronze", "other": None,
}

_BODY_TYPES: dict[str, str | None] = {
    "sedan": "Sedan",
    "stationwagon": "Carrinha",
    "suv": "SUV/TT",
    "offroad": "SUV/TT",
    "compact": "Citadino",
    "smallcar": "Pequeno Citadino",
    "coupe": "Coupé",
    "convertible": "Cabrio",
    "cabrio": "Cabrio",
    "van": "Monovolume",
    "transporter": "Comercial",
    "other": None,
}

_DE_BODY_FILTERS: dict[str, str | None] = {
    "bt_limousine": "Sedan",
    "bt_kombi": "Carrinha",
    "bt_suv-gelaendewagen-pickup": "SUV/TT",
    "bt_kleinwagen": "Pequeno Citadino",
    "bt_van-kleinbus": "Monovolume",
    "bt_cabrio": "Cabrio",
    "bt_coupe": "Coupé",
    "bt_transporter": "Comercial",
}

_NEXT_DATA_RE = re.compile(
    r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', re.S)

_HOSTS = {
    "de": BASE_URL,
    "fr": "https://www.autoscout24.fr",
    "it": "https://www.autoscout24.it",
}

_SELLER_TO_PT = {"dealer": "Profissional", "privateseller": "Particular"}

_THOUSANDS_SPACE = re.compile(r"(?<=\d)[\s   ](?=\d)")
_POWER_KW_RE = re.compile(r"(\d[\d.]*)\s*kW")
_POWER_HP_RE = re.compile(r"(\d[\d.]*)\s*(?:PS|Ch|CV)\b", re.I)


def base_url(tld: str = "de") -> str:
    """The national AutoScout24 origin for a tld. Unknown tlds raise."""
    try:
        return _HOSTS[str(tld or "").lower()]
    except KeyError:
        raise ValueError(f"no AutoScout24 market for tld {tld!r} "
                         f"(known: {', '.join(sorted(_HOSTS))})") from None


def _label_key(text) -> str:
    """Accent- and punctuation-free lookup key: 'Boîte manuelle' → 'boite manuelle'.

    The same powertrain is spelled ``Electrique`` and ``Électrique`` on the same
    French page, and ``Autogas (LPG)`` carries punctuation that means nothing.
    Folding both away keeps the vocabulary tables one entry per concept instead
    of one per spelling the site happens to ship this month.
    """
    decomposed = unicodedata.normalize("NFKD", str(text or ""))
    plain = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return " ".join(re.sub(r"[^0-9a-z]+", " ", plain.lower()).split())


@dataclass(frozen=True)
class Market:
    """One national site: its host, its language header, and what its cards say.

    ``advert_prefix`` is where this site keeps its adverts, and it is asked of
    ``robots.txt`` rather than assumed: ``adverts_allowed`` is what decides
    whether a market's cars are deepened by opening them. ``body_filters`` is
    the fallback for a market where they are not — AutoScout24's own body-type
    filter segments, mapped onto the same Portuguese vocabulary, so a body type
    can be learned from a search the crawler is allowed to make. The filter
    vocabulary is the coarser of the two: ``bt_kleinwagen`` holds what an advert
    would call a small car and a compact both, so a German row lands in one
    segment where a French one, read off its advert, could land in either.

    ``fuels`` and ``gearboxes`` map the site's own label onto this project's
    Portuguese vocabulary. A value of None is a deliberate "this tells us
    nothing" — ``Sonstige``/``Autres``/``Altro`` is the seller declining to
    answer, and storing it as a fuel type would invent a category the price
    model then splits the corpus on. A label that is in neither the table nor
    the None set passes through unchanged, so a new powertrain shows up in the
    data as itself rather than disappearing.
    """

    tld: str
    accept_language: str
    mileage_label: str
    gearbox_label: str
    registration_label: str
    fuel_label: str
    power_label: str
    co2_label: str | None
    fuels: dict[str, str | None]
    gearboxes: dict[str, str | None]
    drive_trains: dict[str, str | None] = field(default_factory=dict)
    advert_prefix: str = "/angebote/"
    body_filters: dict[str, str | None] = field(default_factory=dict)


_MARKETS: dict[str, Market] = {
    "de": Market(
        tld="de",
        accept_language="de-DE,de;q=0.9",
        mileage_label="Kilometerstand",
        gearbox_label="Getriebe",
        registration_label="Erstzulassung",
        fuel_label="Kraftstoff",
        power_label="Leistung",
        co2_label="CO₂-Emissionen",
        fuels={
            "diesel": "Diesel",
            "benzin": "Gasolina",
            "elektro": "Eléctrico",
            "elektro benzin": "Híbrido (Gasolina)",
            "elektro diesel": "Híbrido (Diesel)",
            "autogas lpg": "GPL",
            "lpg": "GPL",
            "erdgas cng": "GNC",
            "cng": "GNC",
            "wasserstoff": "Hidrogénio",
            "ethanol": "Gasolina",
            "sonstige": None,
            "andere": None,
        },
        gearboxes={
            "schaltgetriebe": "Manual",
            "automatik": "Automática",
            "halbautomatik": "Automática",
        },
        drive_trains={
            "vorderrad": "Dianteira",
            "hinterrad": "Traseira",
            "allrad": "Integral",
        },
        advert_prefix="/angebote/",
        body_filters=_DE_BODY_FILTERS,
    ),
    "fr": Market(
        tld="fr",
        accept_language="fr-FR,fr;q=0.9",
        mileage_label="Kilométrage",
        gearbox_label="Boîte",
        registration_label="1ère immatriculation",
        fuel_label="Carburant",
        power_label="Puissance kW (CH)",
        co2_label=None,
        fuels={
            "diesel": "Diesel",
            "essence": "Gasolina",
            "electrique": "Eléctrico",
            "electrique essence": "Híbrido (Gasolina)",
            "electrique diesel": "Híbrido (Diesel)",
            "gpl": "GPL",
            "cng": "GNC",
            "gnv": "GNC",
            "hydrogene": "Hidrogénio",
            "ethanol": "Gasolina",
            "autres": None,
            "autre": None,
        },
        gearboxes={
            "boite manuelle": "Manual",
            "boite automatique": "Automática",
            "semi automatique": "Automática",
        },
        drive_trains={
            "traction avant": "Dianteira",
            "propulsion": "Traseira",
            "transmission integrale": "Integral",
            "4 roues motrices": "Integral",
        },
        advert_prefix="/offres/",
    ),
    "it": Market(
        tld="it",
        accept_language="it-IT,it;q=0.9",
        mileage_label="Chilometraggio",
        gearbox_label="Cambio",
        registration_label="Anno",
        fuel_label="Carburante",
        power_label="Potenza",
        co2_label=None,
        fuels={
            "diesel": "Diesel",
            "benzina": "Gasolina",
            "elettrica": "Eléctrico",
            "elettrica benzina": "Híbrido (Gasolina)",
            "elettrica diesel": "Híbrido (Diesel)",
            "gpl": "GPL",
            "metano": "GNC",
            "idrogeno": "Hidrogénio",
            "etanolo": "Gasolina",
            "altro": None,
            "altri": None,
        },
        gearboxes={
            "manuale": "Manual",
            "automatico": "Automática",
            "semiautomatico": "Automática",
        },
        drive_trains={
            "anteriore": "Dianteira",
            "posteriore": "Traseira",
            "integrale": "Integral",
            "4x4": "Integral",
        },
        advert_prefix="/annunci/",
    ),
}


def market(tld: str = "de") -> Market:
    """The vocabulary table for a tld. Unknown tlds raise rather than default."""
    try:
        return _MARKETS[str(tld or "").lower()]
    except KeyError:
        raise ValueError(f"no AutoScout24 market for tld {tld!r} "
                         f"(known: {', '.join(sorted(_MARKETS))})") from None


class AutoScoutUnreadable(RuntimeError):
    """A page answered but did not carry what it was asked for.

    Kept apart from ``AutoScoutBlocked`` because the right response differs: a
    block is the site asking us to go away and stops the whole run, while an
    advert that will not parse costs that one car its extra fields and nothing
    else.
    """


class AutoScoutBlocked(RuntimeError):
    """The site asked us to stop (429/403). Callers must not retry in-run."""


def robots_allows(path: str, tld: str = "de") -> bool:
    """Whether ``User-agent: *`` in AutoScout24's robots.txt leaves this path open.

    Kept as code so the crawler cannot drift away from what the file says: the
    runner asks before every fetch, and a path that lands on a disallowed
    prefix is skipped rather than requested.

    The three national files agree on the search prefixes, so that list is
    shared. They do **not** agree on the advert page, and the difference is the
    whole reason this takes a ``tld``: ``autoscout24.de`` disallows
    ``/angebote/`` outright, ``.fr`` closes only the malformed ``/offres/-``,
    and ``.it`` says nothing about ``/annunci/``. So the German adverts are not
    ours to read and the French and Italian ones are, and that judgement lives
    here rather than in whichever caller remembers it.

    The default is ``de`` because it is the strictest: a caller that forgets to
    say which market it is reading gets the answer that refuses more.
    """
    p = path if path.startswith("/") else "/" + path
    if any(p.startswith(pre) for pre in _DISALLOWED_PREFIXES):
        return False
    return not any(p.startswith(pre)
                   for pre in _DISALLOWED_BY_TLD.get(str(tld or "de"), ()))


def adverts_allowed(tld: str = "de") -> bool:
    """Whether this market's advert pages are ours to open.

    Asked of the same robots rules rather than remembered as a constant, so a
    file that changes changes the crawler with it. Today the answer is no for
    ``.de`` and yes for the other two, and it is what decides whether a car is
    deepened by opening its advert or by asking the search again through a body
    filter.
    """
    return robots_allows(market(tld).advert_prefix + "x", tld)


@dataclass
class DeListing:
    """One foreign listing, in this project's vocabulary rather than AS24's."""

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
    version: str | None = None
    body_type: str | None = None
    offer_type: str | None = None
    year: int | None = None
    registration_month: str | None = None
    mileage_km: int | None = None
    engine_cc: int | None = None
    horsepower: int | None = None
    power_kw: int | None = None
    fuel_type: str | None = None
    transmission: str | None = None
    co2_g_km: int | None = None
    price_label: str | None = None
    seller_type: str | None = None
    country_code: str | None = None
    region: str | None = None
    city: str | None = None
    zip_code: str | None = None
    is_damaged: bool | None = None
    photo_count: int | None = None
    image_url: str | None = None
    photo_urls: list[str] | None = None
    source: str = "autoscout24"


@dataclass
class AutoScoutConfig:
    delay_min: float = DELAY_MIN
    delay_max: float = DELAY_MAX
    timeout: float = TIMEOUT
    budget: int = 200
    user_agent: str = USER_AGENT
    country: str = "D"
    tld: str = "de"


def _num(text: str | None) -> float | None:
    """A card's number out of its label: '1.995 cm³' → 1995, '65 000 km' → 65000.

    Three markets, three thousands separators: Germany and Italy print a dot,
    France prints a narrow no-break space. The space forms have to go before
    the digits are read or ``65 000 km`` comes back as sixty-five.
    """
    if not text:
        return None
    cleaned = _THOUSANDS_SPACE.sub("", str(text))
    m = re.search(r"-?\d[\d.]*(?:,\d+)?", cleaned)
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


def _detail(listing: dict, aria: str | None) -> str | None:
    if not aria:
        return None
    for item in listing.get("vehicleDetails") or []:
        if item.get("ariaLabel") == aria:
            return item.get("data")
    return None


CO2_MIN_G_KM = 50
CO2_MAX_G_KM = 500


def _co2(listing: dict, fuel_pt: str | None = None, mk: Market | None = None) -> int | None:
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
        raw = _int(_detail(listing, (mk or _MARKETS["de"]).co2_label))
    if raw is None:
        return None
    electric = (fuel_pt or "").lower().startswith("elé")
    if electric:
        return raw if 0 <= raw <= CO2_MAX_G_KM else None
    return raw if CO2_MIN_G_KM <= raw <= CO2_MAX_G_KM else None


def _registration(listing: dict, mk: Market) -> tuple[int | None, str | None]:
    """(year, 'MM/YYYY') from first registration, tracking first, label second.

    ``tracking.firstRegistration`` is the same ``MM-YYYY`` on all three sites
    while the printed label is whatever the locale prints, so the tracked value
    wins and the label is only the fallback.
    """
    raw = (listing.get("tracking") or {}).get("firstRegistration")
    if raw and re.match(r"^\d{2}-\d{4}$", str(raw)):
        month, year = str(raw).split("-")
        return int(year), f"{month}/{year}"
    label = _detail(listing, mk.registration_label)
    if label and re.match(r"^\d{2}/\d{4}$", str(label).strip()):
        month, year = str(label).strip().split("/")
        return int(year), f"{month}/{year}"
    return None, None


def _power(listing: dict, mk: Market) -> tuple[int | None, int | None]:
    """(kW, hp) from '140 kW (190 PS)', '81 kW (110 Ch)', '110 kW (150 CV)'."""
    label = _detail(listing, mk.power_label) or ""
    kw = _POWER_KW_RE.search(label)
    hp = _POWER_HP_RE.search(label)
    return (_int(kw.group(1)) if kw else None, _int(hp.group(1)) if hp else None)


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


def _translate(raw: str, table: dict[str, str | None]) -> str | None:
    """A card label in our vocabulary; unknown labels survive as themselves."""
    key = _label_key(raw)
    if key in table:
        return table[key]
    return raw.strip() or None


def _make_models(props: dict) -> list[dict]:
    """The make page's own model list, flattened out of the taxonomy.

    ``taxonomy.models`` is keyed by make id and each value is that make's list,
    so a make page carries exactly one key and a filtered search page carries
    the make it was filtered to. Flattening keeps the caller from having to
    know the make id it asked for.
    """
    models = ((props.get("taxonomy") or {}).get("models")) or {}
    entries: list[dict] = []
    groups = models.values() if isinstance(models, dict) else [models]
    for group in groups:
        for item in group or []:
            if not isinstance(item, dict):
                continue
            entries.append({"label": item.get("label"), "value": item.get("value"),
                            "makeId": item.get("makeId")})
    return entries


def _body_types(props: dict) -> list[str]:
    """The body-type filters AutoScout24 offers for the model on this page.

    The page interlinks its own facets, and the body-type group is the list of
    bodies that model actually has for sale. It costs nothing — it rides on a
    page the crawler already asked for — and it is what makes a body type
    reachable on a market whose adverts are closed: ask the same cell again
    through one filter, and every car that comes back has that body.

    A model with a single body ships no group at all, so an empty list means
    "the page did not say", not "this model has no body".
    """
    for group in props.get("interlinking") or []:
        if not isinstance(group, dict) or group.get("id") != "bodyTypes":
            continue
        slugs = []
        for link in group.get("links") or []:
            slug = str((link or {}).get("url") or "").rstrip("/").rsplit("/", 1)[-1]
            if slug.startswith("bt_"):
                slugs.append(slug)
        return slugs
    return []


def parse_search(html: str, tld: str = "de") -> tuple[list[DeListing], dict]:
    """(listings, meta) from a search page's ``__NEXT_DATA__``.

    ``meta`` carries ``results`` and ``pages`` so the runner can stop paging
    instead of guessing, ``make_models`` so a make page doubles as the
    discovery of that make's model vocabulary, and ``body_types`` so a market
    that may not open adverts still knows which body filters this model is worth
    asking through. A page whose JSON is missing or reshaped returns an empty
    list and an empty meta — the caller treats that as "stop", never as "no cars
    in this country".
    """
    mk = market(tld)
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
    meta = {"results": props.get("numberOfResults"), "pages": props.get("numberOfPages"),
            "make_models": _make_models(props), "body_types": _body_types(props)}
    out = []
    for item in raw:
        parsed = _to_listing(item, mk)
        if parsed is not None:
            out.append(parsed)
    return out, meta


def _to_listing(item: dict, mk: Market) -> DeListing | None:
    vehicle = item.get("vehicle") or {}
    price = item.get("price") or {}
    tracking = item.get("tracking") or {}
    external_id = str(item.get("id") or item.get("crossReferenceId") or "").strip()
    brand = str(vehicle.get("make") or "").strip()
    model = str(vehicle.get("model") or "").strip()
    if not external_id or not brand or not model:
        return None
    price_eur = price.get("priceRaw")
    price_eur = (float(price_eur) if isinstance(price_eur, (int, float))
                 else _num(tracking.get("price")))
    vat_label, vat_reclaimable = _vat(price)
    year, reg_month = _registration(item, mk)
    power_kw, power_hp = _power(item, mk)
    location = item.get("location") or {}
    seller = item.get("seller") or {}
    url = str(item.get("url") or "")
    fuel_raw = str(vehicle.get("fuel") or "").strip()
    gearbox_raw = str(vehicle.get("transmission") or "").strip()
    fuel_pt = normalize_fuel_type(_translate(fuel_raw, mk.fuels)) if fuel_raw else None
    images = item.get("images")
    images = images if isinstance(images, list) else None
    country_code = str(location.get("countryCode") or "").strip() or None
    city = str(location.get("city") or "").strip() or None
    zip_code = str(location.get("zip") or "").strip() or None
    return DeListing(
        external_id=external_id,
        url=(base_url(mk.tld) + url) if url.startswith("/") else url,
        brand=brand,
        model=model,
        price_eur=price_eur,
        vat_label=vat_label,
        vat_reclaimable=vat_reclaimable,
        model_group=str(vehicle.get("modelGroup") or "").strip() or None,
        variant=str(vehicle.get("variant") or "").strip() or None,
        motor_type=str(vehicle.get("motorTypeName") or "").strip() or None,
        version=str(vehicle.get("modelVersionInput") or "").strip() or None,
        body_type=str(vehicle.get("bodyType") or "").strip() or None,
        offer_type=str(vehicle.get("offerType") or "").strip() or None,
        year=year,
        registration_month=reg_month,
        mileage_km=_int(tracking.get("mileage")) or _int(vehicle.get("mileageInKm")),
        engine_cc=_int(vehicle.get("engineDisplacementInCCM")),
        horsepower=power_hp,
        power_kw=power_kw,
        fuel_type=fuel_pt,
        transmission=_translate(gearbox_raw, mk.gearboxes) if gearbox_raw else None,
        co2_g_km=_co2(item, fuel_pt, mk),
        price_label=str(tracking.get("priceLabel") or "").strip() or None,
        seller_type=_SELLER_TO_PT.get(_label_key(seller.get("type"))),
        country_code=country_code,
        region=regions.region_for(country_code or "", zip_code, city),
        city=city,
        zip_code=zip_code,
        is_damaged=vehicle.get("isCurrentlyDamaged"),
        photo_count=len(images) if images is not None else None,
        image_url=_cover_image(images),
        photo_urls=_photo_list(images),
    )


PHOTO_LIMIT = 8


def _photo_list(images: list | None, limit: int = PHOTO_LIMIT) -> list[str] | None:
    """A gallery at the size a page can actually show it.

    The search payload links thumbnails (``/250x188.webp``), which look like
    mud at card width; the same object is served at ``/720x540.webp`` and the
    swap costs one string replacement instead of a detail fetch. The card
    carries three photos and the advert the whole set, so the cap is what keeps
    a row from growing without bound.
    """
    if not images:
        return None
    out: list[str] = []
    for item in images:
        url = str(item or "").strip().replace("/250x188.webp", "/720x540.webp")
        if url and url not in out:
            out.append(url)
        if len(out) >= limit:
            break
    return out or None


def _cover_image(images: list | None) -> str | None:
    """The first photo of the gallery, for readers that want exactly one."""
    photos = _photo_list(images, 1)
    return photos[0] if photos else None


def _first(*values):
    """The first value that is neither None nor an empty string."""
    for value in values:
        if value not in (None, ""):
            return value
    return None


def _raw(node, key: str):
    """``{"raw": "Sedan", "formatted": "Berlina"}`` → the raw code, else the value.

    AutoScout24 ships some fields twice, once translated for the page and once
    as a language-independent code. The code is what this reader wants: one
    table for three markets instead of three tables that drift apart.
    """
    value = (node or {}).get(key)
    if isinstance(value, dict):
        return _first(value.get("raw"), value.get("formatted"))
    return value


def parse_detail(html: str, tld: str = "de") -> dict:
    """What the advert adds to a card: the body, the colour, and the seller's text.

    Returns a patch to lay over a card rather than a whole listing, so an
    advert that cannot be read costs the extra fields and not the car.

    This is where two long-standing holes in the foreign corpora close.
    ``segment`` was null for every AutoScout24 row because no search card
    carries a body type, and CO2 was missing outside Germany for the same
    reason — both are on the advert, and CO2 is an input the ISV formula needs.
    The rest is what a Portuguese row has and a card never did: the seller's
    own description, the colour, the doors and seats, the drive train, the
    dealer behind the advert and the exact place it sits in.

    Nothing here overwrites a card but the gallery. ``CORRECTS`` names only
    ``photo_urls``: unlike a generalist classified, an AutoScout24 card is
    structured data from the same database as the advert, so where the two
    overlap they agree — except for the photos, where the card links three of
    them and the advert the whole set, and more of the same car is strictly
    better than fewer.
    """
    match = _NEXT_DATA_RE.search(html or "")
    if not match:
        raise AutoScoutUnreadable("no __NEXT_DATA__ on advert page")
    try:
        props = json.loads(match.group(1))["props"]["pageProps"]
    except (KeyError, ValueError) as exc:
        raise AutoScoutUnreadable(f"unreadable advert page: {exc}") from exc

    details = props.get("listingDetails")
    if not isinstance(details, dict):
        raise AutoScoutUnreadable("advert page carries no listing")

    mk = market(tld)
    vehicle = details.get("vehicle") or {}
    raw = vehicle.get("rawData") or {}
    patch: dict = {}
    extras: dict = {}

    description = re.sub(r"<br\s*/?>", "\n", str(details.get("description") or ""))
    description = re.sub(r"<[^>]+>", " ", description)
    description = _unescape(description)
    description = re.sub(r"[ \t]+", " ", description).strip()
    if description:
        patch["description"] = description
        patch["description_length"] = len(description)

    body = _translate(str(_raw(raw, "bodyType") or ""), _BODY_TYPES)
    if body:
        patch["body_type"] = body
        patch["segment"] = body

    photos = _photo_list(details.get("images"))
    if photos:
        patch["photo_urls"] = photos

    colour = _translate(str(_raw(raw, "bodyColor") or ""), _COLOURS)
    if colour:
        patch["color"] = colour

    drive = _translate(str(vehicle.get("driveTrain") or ""), mk.drive_trains)
    if drive:
        patch["drive_type"] = drive

    doors = vehicle.get("numberOfDoors")
    if doors:
        patch["doors"] = "1-3" if _int(doors) and _int(doors) <= 3 else "4-5"
    if vehicle.get("numberOfSeats"):
        patch["seats"] = _int(vehicle["numberOfSeats"])

    co2 = vehicle.get("co2emissionInGramPerKmWithFallback")
    co2 = _int(co2.get("raw")) if isinstance(co2, dict) else _int(co2)
    if co2:
        patch["co2_g_km"] = co2

    seller = details.get("seller") or {}
    if seller.get("companyName") or seller.get("contactName"):
        patch["seller_displayed_as"] = _first(seller.get("companyName"),
                                              seller.get("contactName"))
    if seller.get("isDealer") is not None:
        patch["seller_type"] = "Profissional" if seller["isDealer"] else "Particular"

    location = details.get("location") or {}
    if location.get("zip"):
        patch["zip_code"] = str(location["zip"])
    if location.get("city"):
        patch["city"] = str(location["city"])

    version = _first(vehicle.get("variant"), vehicle.get("modelVersionInput"))
    if version:
        extras["version"] = str(version)
    if vehicle.get("motorTypeName"):
        patch["sub_model"] = str(vehicle["motorTypeName"])
    for key, name in (("noOfPreviousOwners", "previous_owners"),
                      ("hasFullServiceHistory", "full_service_history"),
                      ("hadAccident", "had_accident"),
                      ("nonSmoking", "non_smoking"),
                      ("originalMarket", "original_market"),
                      ("gears", "gears")):
        if vehicle.get(key) is not None:
            extras[name] = vehicle[key]
    if details.get("createdTimestampWithOffset"):
        extras["posted_at"] = details["createdTimestampWithOffset"]

    damage = _first(vehicle.get("hadAccident"), vehicle.get("damageConditions"))
    if isinstance(damage, bool):
        patch["is_damaged"] = damage

    if extras:
        patch["extras"] = extras
    return patch


def _list_path(segments, *, year: int | None = None, page: int = 1, country: str = "D",
               sort: str | None = None, desc: bool = False, ustate: str = "N,U") -> str:
    params = {
        "atype": "C",
        "cy": country,
        "damaged_listing": "exclude",
        "powertype": "kw",
        "sort": sort or "standard",
        "ustate": ustate,
    }
    if desc:
        params["desc"] = 1
    if year:
        params["fregfrom"] = year
        params["fregto"] = year
    if page and page > 1:
        params["page"] = page
    tail = "/".join(str(s).strip("/") for s in segments if s)
    return f"{SEARCH_PATH}/{tail}?{urlencode(params)}"


def search_path(make: str, model: str, *, year: int | None = None, page: int = 1,
                country: str = "D", body: str | None = None, sort: str | None = None,
                desc: bool = False, ustate: str = "N,U") -> str:
    """The path+query for one model-year page, in the form robots.txt leaves open.

    ``body`` is AutoScout24's body-type segment (``bt_kombi`` and friends). It
    exists because half the Portuguese estate vocabulary — "308 SW", "Leon ST",
    "Mégane Sport Tourer" — is a body type there rather than a model, so those
    models are unreachable without it and were coming back 404.

    ``sort``/``desc`` exist for the country crawl, which walks a model newest
    first (``sort=age&desc=1``) so that the twenty cars a single request buys
    are the twenty that changed. Left alone it keeps the site's own relevance
    order, which is what the import benchmark has always asked for.
    """
    return _list_path([make, model, body], year=year, page=page, country=country,
                      sort=sort, desc=desc, ustate=ustate)


def make_path(make: str, *, page: int = 1, country: str = "D", sort: str | None = None,
              desc: bool = False, ustate: str = "N,U") -> str:
    """The path+query for a make's own page, which carries its model taxonomy."""
    return _list_path([make], page=page, country=country, sort=sort, desc=desc,
                      ustate=ustate)


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
                     "Accept-Language": market(self.config.tld).accept_language},
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
        if not robots_allows(path, self.config.tld):
            logger.warning("autoscout: robots.txt disallows %s — skipped", path)
            return None
        if self.spent >= self.config.budget:
            return None
        if self._client is None:
            raise RuntimeError("AutoScoutClient must be used as a context manager")
        if self.spent:
            self._sleep()
        self.spent += 1
        resp = self._client.get(base_url(self.config.tld) + path)
        if resp.status_code in (403, 429):
            raise AutoScoutBlocked(f"{resp.status_code} on {path}")
        if resp.status_code >= 400:
            logger.warning("autoscout: %s on %s", resp.status_code, path)
            return None
        return resp.text

    def search(self, make: str, model: str, *, year: int | None = None, page: int = 1,
               body: str | None = None, sort: str | None = None, desc: bool = False,
               ustate: str = "U") -> tuple[list[DeListing], dict]:
        """One search page as (listings, meta); ``([], {})`` when nothing was read.

        The empty meta is the caller's signal to stop: a budget that ran out, a
        path robots closed and a 404 are all "no answer", and none of them is
        the same as a page that answered with zero cars.
        """
        html = self.fetch(search_path(make, model, year=year, page=page,
                                      country=self.config.country, body=body,
                                      sort=sort, desc=desc, ustate=ustate))
        if html is None:
            return [], {}
        return parse_search(html, self.config.tld)

    def advert(self, path: str) -> dict | None:
        """One advert as a patch, or None when it is not ours to read.

        None covers every reason not to have it and they are all the same to
        the caller: robots closed the path — which is the standing answer on
        ``autoscout24.de``, whose ``/angebote/`` is disallowed — the budget ran
        out, or the page did not parse. The card is stored either way; a listing
        is never dropped for want of its advert.
        """
        html = self.fetch(path)
        if html is None:
            return None
        try:
            return parse_detail(html, self.config.tld)
        except AutoScoutUnreadable as exc:
            logger.warning("autoscout: advert %s unreadable: %s", path, exc)
            return None

    def make_page(self, make: str, *, page: int = 1) -> tuple[list[DeListing], dict]:
        """A make's own page: twenty real cars and, in meta, its model vocabulary."""
        html = self.fetch(make_path(make, page=page, country=self.config.country))
        if html is None:
            return [], {}
        return parse_search(html, self.config.tld)

    def model_year(self, make: str, model: str, year: int, *, max_pages: int = 1,
                   body: str | None = None) -> list[DeListing]:
        """Every listing we are willing to read for one model in one year."""
        found: list[DeListing] = []
        for page in range(1, max(1, min(max_pages, MAX_PAGE)) + 1):
            html = self.fetch(search_path(make, model, year=year, page=page,
                                          country=self.config.country, body=body))
            if html is None:
                break
            listings, meta = parse_search(html, self.config.tld)
            if not listings:
                break
            found.extend(listings)
            if page >= (meta.get("pages") or 1):
                break
        return found
