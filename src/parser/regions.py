"""Where a foreign listing is, in the unit its own country thinks in.

The Portuguese corpus has ``district`` and every page, facet and model feature
built on top of it assumes one: a geography coarse enough that a model-year has
several dozen cars in it, fine enough that "near me" means something. The
AutoScout24 card has no such field. It has a postcode, a city string, and a
country, and the three countries hide the answer in a different one of those.

So this module is three small conversions, not one:

* **Germany** publishes Bundesland nowhere on the card, but the PLZ decides it
  almost everywhere — the first two digits are a Leitregion and Leitregionen
  respect state borders far more often than not. Where a two-digit prefix
  genuinely straddles two states (14 between Berlin and Brandenburg, 89 between
  Baden-Württemberg and Bayern, 66 between Saarland and Rheinland-Pfalz) a
  three-digit entry decides it; where it straddles them only at the edges, the
  majority state wins and a handful of border towns are labelled by their
  neighbour. That is the right trade: the alternative is an 8200-row postcode
  table to move perhaps one listing in five hundred.
* **France** is the easy one. The first two digits of a code postal are the
  département, and the 2016 reform grouped the départements into thirteen
  régions that are the natural unit for a market page. Corsica is the exception
  the numbering forces — 2A and 2B share the prefix 20 — and both halves come
  back as one région, which is what a buyer means by "Corse" anyway.
* **Italy** does not encode the province in the CAP in any usable way (Rome's
  00xxx runs from the centre to the coast), but the site writes it into the
  city string instead: "Poviglio - Reggio Emilia - RE". So the province is
  parsed out of that, by two-letter sigla first and by name second, and the CAP
  is only the fallback for a card that gives a bare city name.

Unknown is unknown: a postcode this module cannot place returns None rather
than a guess, because a wrong region is worse than a missing one — it moves a
car into somebody else's market median instead of leaving it out of both.
"""

from __future__ import annotations

import re
import unicodedata

_DIGITS = re.compile(r"\d+")


def _fold(text: str) -> str:
    """Lowercase, accent-free, punctuation-free key for matching a place name."""
    decomposed = unicodedata.normalize("NFKD", str(text))
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return " ".join(re.sub(r"[^0-9a-z]+", " ", stripped.lower()).split())


def _zip_digits(zip_code: str | None) -> str:
    """The leading run of digits of a postcode, or '' when there is none."""
    if not zip_code:
        return ""
    match = _DIGITS.search(str(zip_code))
    return match.group(0) if match else ""


_DE_BY_PREFIX3 = {
    "140": "Berlin", "141": "Berlin",
    "210": "Hamburg", "211": "Hamburg", "215": "Schleswig-Holstein",
    "228": "Schleswig-Holstein", "229": "Schleswig-Holstein",
    "288": "Niedersachsen", "289": "Niedersachsen",
    "364": "Thüringen",
    "372": "Hessen", "373": "Thüringen",
    "388": "Sachsen-Anhalt", "389": "Sachsen-Anhalt",
    "534": "Rheinland-Pfalz", "535": "Rheinland-Pfalz",
    "575": "Rheinland-Pfalz", "576": "Rheinland-Pfalz",
    "668": "Rheinland-Pfalz", "669": "Rheinland-Pfalz",
    "685": "Hessen", "686": "Hessen",
    "767": "Rheinland-Pfalz", "768": "Rheinland-Pfalz",
    "881": "Bayern",
    "890": "Baden-Württemberg", "891": "Baden-Württemberg",
    "895": "Baden-Württemberg", "896": "Baden-Württemberg",
    "979": "Baden-Württemberg",
}

_DE_BY_PREFIX2 = {
    "01": "Sachsen", "02": "Sachsen", "03": "Brandenburg", "04": "Sachsen",
    "06": "Sachsen-Anhalt", "07": "Thüringen", "08": "Sachsen", "09": "Sachsen",
    "10": "Berlin", "11": "Berlin", "12": "Berlin", "13": "Berlin",
    "14": "Brandenburg", "15": "Brandenburg", "16": "Brandenburg",
    "17": "Mecklenburg-Vorpommern", "18": "Mecklenburg-Vorpommern",
    "19": "Mecklenburg-Vorpommern",
    "20": "Hamburg", "21": "Niedersachsen", "22": "Hamburg",
    "23": "Schleswig-Holstein", "24": "Schleswig-Holstein", "25": "Schleswig-Holstein",
    "26": "Niedersachsen", "27": "Niedersachsen", "28": "Bremen", "29": "Niedersachsen",
    "30": "Niedersachsen", "31": "Niedersachsen",
    "32": "Nordrhein-Westfalen", "33": "Nordrhein-Westfalen",
    "34": "Hessen", "35": "Hessen", "36": "Hessen",
    "37": "Niedersachsen", "38": "Niedersachsen", "39": "Sachsen-Anhalt",
    "40": "Nordrhein-Westfalen", "41": "Nordrhein-Westfalen",
    "42": "Nordrhein-Westfalen", "43": "Nordrhein-Westfalen",
    "44": "Nordrhein-Westfalen", "45": "Nordrhein-Westfalen",
    "46": "Nordrhein-Westfalen", "47": "Nordrhein-Westfalen",
    "48": "Nordrhein-Westfalen", "49": "Niedersachsen",
    "50": "Nordrhein-Westfalen", "51": "Nordrhein-Westfalen",
    "52": "Nordrhein-Westfalen", "53": "Nordrhein-Westfalen",
    "54": "Rheinland-Pfalz", "55": "Rheinland-Pfalz", "56": "Rheinland-Pfalz",
    "57": "Nordrhein-Westfalen", "58": "Nordrhein-Westfalen",
    "59": "Nordrhein-Westfalen",
    "60": "Hessen", "61": "Hessen", "62": "Hessen", "63": "Hessen",
    "64": "Hessen", "65": "Hessen",
    "66": "Saarland", "67": "Rheinland-Pfalz",
    "68": "Baden-Württemberg", "69": "Baden-Württemberg",
    "70": "Baden-Württemberg", "71": "Baden-Württemberg",
    "72": "Baden-Württemberg", "73": "Baden-Württemberg",
    "74": "Baden-Württemberg", "75": "Baden-Württemberg",
    "76": "Baden-Württemberg", "77": "Baden-Württemberg",
    "78": "Baden-Württemberg", "79": "Baden-Württemberg",
    "80": "Bayern", "81": "Bayern", "82": "Bayern", "83": "Bayern",
    "84": "Bayern", "85": "Bayern", "86": "Bayern", "87": "Bayern",
    "88": "Baden-Württemberg", "89": "Bayern",
    "90": "Bayern", "91": "Bayern", "92": "Bayern", "93": "Bayern",
    "94": "Bayern", "95": "Bayern", "96": "Bayern", "97": "Bayern",
    "98": "Thüringen", "99": "Thüringen",
}

_FR_DEPARTEMENTS = {
    "01": "Ain", "02": "Aisne", "03": "Allier", "04": "Alpes-de-Haute-Provence",
    "05": "Hautes-Alpes", "06": "Alpes-Maritimes", "07": "Ardèche", "08": "Ardennes",
    "09": "Ariège", "10": "Aube", "11": "Aude", "12": "Aveyron",
    "13": "Bouches-du-Rhône", "14": "Calvados", "15": "Cantal", "16": "Charente",
    "17": "Charente-Maritime", "18": "Cher", "19": "Corrèze",
    "2A": "Corse-du-Sud", "2B": "Haute-Corse",
    "21": "Côte-d'Or", "22": "Côtes-d'Armor", "23": "Creuse", "24": "Dordogne",
    "25": "Doubs", "26": "Drôme", "27": "Eure", "28": "Eure-et-Loir",
    "29": "Finistère", "30": "Gard", "31": "Haute-Garonne", "32": "Gers",
    "33": "Gironde", "34": "Hérault", "35": "Ille-et-Vilaine", "36": "Indre",
    "37": "Indre-et-Loire", "38": "Isère", "39": "Jura", "40": "Landes",
    "41": "Loir-et-Cher", "42": "Loire", "43": "Haute-Loire",
    "44": "Loire-Atlantique", "45": "Loiret", "46": "Lot", "47": "Lot-et-Garonne",
    "48": "Lozère", "49": "Maine-et-Loire", "50": "Manche", "51": "Marne",
    "52": "Haute-Marne", "53": "Mayenne", "54": "Meurthe-et-Moselle", "55": "Meuse",
    "56": "Morbihan", "57": "Moselle", "58": "Nièvre", "59": "Nord", "60": "Oise",
    "61": "Orne", "62": "Pas-de-Calais", "63": "Puy-de-Dôme",
    "64": "Pyrénées-Atlantiques", "65": "Hautes-Pyrénées", "66": "Pyrénées-Orientales",
    "67": "Bas-Rhin", "68": "Haut-Rhin", "69": "Rhône", "70": "Haute-Saône",
    "71": "Saône-et-Loire", "72": "Sarthe", "73": "Savoie", "74": "Haute-Savoie",
    "75": "Paris", "76": "Seine-Maritime", "77": "Seine-et-Marne", "78": "Yvelines",
    "79": "Deux-Sèvres", "80": "Somme", "81": "Tarn", "82": "Tarn-et-Garonne",
    "83": "Var", "84": "Vaucluse", "85": "Vendée", "86": "Vienne",
    "87": "Haute-Vienne", "88": "Vosges", "89": "Yonne", "90": "Territoire de Belfort",
    "91": "Essonne", "92": "Hauts-de-Seine", "93": "Seine-Saint-Denis",
    "94": "Val-de-Marne", "95": "Val-d'Oise",
}

_FR_REGION_MEMBERS = {
    "Auvergne-Rhône-Alpes": ("01", "03", "07", "15", "26", "38", "42", "43", "63",
                             "69", "73", "74"),
    "Bourgogne-Franche-Comté": ("21", "25", "39", "58", "70", "71", "89", "90"),
    "Bretagne": ("22", "29", "35", "56"),
    "Centre-Val de Loire": ("18", "28", "36", "37", "41", "45"),
    "Corse": ("2A", "2B"),
    "Grand Est": ("08", "10", "51", "52", "54", "55", "57", "67", "68", "88"),
    "Hauts-de-France": ("02", "59", "60", "62", "80"),
    "Île-de-France": ("75", "77", "78", "91", "92", "93", "94", "95"),
    "Normandie": ("14", "27", "50", "61", "76"),
    "Nouvelle-Aquitaine": ("16", "17", "19", "23", "24", "33", "40", "47", "64",
                           "79", "86", "87"),
    "Occitanie": ("09", "11", "12", "30", "31", "32", "34", "46", "48", "65", "66",
                  "81", "82"),
    "Pays de la Loire": ("44", "49", "53", "72", "85"),
    "Provence-Alpes-Côte d'Azur": ("04", "05", "06", "13", "83", "84"),
}

_FR_REGION_BY_DEPARTEMENT = {
    dep: region for region, deps in _FR_REGION_MEMBERS.items() for dep in deps
}

_FR_OVERSEAS = {
    "971": "Guadeloupe", "972": "Martinique", "973": "Guyane",
    "974": "La Réunion", "975": "Saint-Pierre-et-Miquelon", "976": "Mayotte",
    "977": "Saint-Barthélemy", "978": "Saint-Martin",
    "986": "Wallis-et-Futuna", "987": "Polynésie française",
    "988": "Nouvelle-Calédonie",
}

_IT_PROVINCES = {
    "Abruzzo": (("L'Aquila", "AQ"), ("Teramo", "TE"), ("Pescara", "PE"),
                ("Chieti", "CH")),
    "Basilicata": (("Potenza", "PZ"), ("Matera", "MT")),
    "Calabria": (("Cosenza", "CS"), ("Catanzaro", "CZ"), ("Reggio Calabria", "RC"),
                 ("Crotone", "KR"), ("Vibo Valentia", "VV")),
    "Campania": (("Napoli", "NA"), ("Salerno", "SA"), ("Avellino", "AV"),
                 ("Benevento", "BN"), ("Caserta", "CE")),
    "Emilia-Romagna": (("Bologna", "BO"), ("Modena", "MO"), ("Reggio Emilia", "RE"),
                       ("Parma", "PR"), ("Piacenza", "PC"), ("Ferrara", "FE"),
                       ("Ravenna", "RA"), ("Forlì-Cesena", "FC"), ("Rimini", "RN")),
    "Friuli-Venezia Giulia": (("Trieste", "TS"), ("Udine", "UD"), ("Gorizia", "GO"),
                              ("Pordenone", "PN")),
    "Lazio": (("Roma", "RM"), ("Latina", "LT"), ("Frosinone", "FR"),
              ("Viterbo", "VT"), ("Rieti", "RI")),
    "Liguria": (("Genova", "GE"), ("Savona", "SV"), ("La Spezia", "SP"),
                ("Imperia", "IM")),
    "Lombardia": (("Milano", "MI"), ("Bergamo", "BG"), ("Brescia", "BS"),
                  ("Como", "CO"), ("Cremona", "CR"), ("Mantova", "MN"),
                  ("Pavia", "PV"), ("Sondrio", "SO"), ("Varese", "VA"),
                  ("Lecco", "LC"), ("Lodi", "LO"), ("Monza e della Brianza", "MB")),
    "Marche": (("Ancona", "AN"), ("Pesaro e Urbino", "PU"), ("Macerata", "MC"),
               ("Ascoli Piceno", "AP"), ("Fermo", "FM")),
    "Molise": (("Campobasso", "CB"), ("Isernia", "IS")),
    "Piemonte": (("Torino", "TO"), ("Alessandria", "AL"), ("Asti", "AT"),
                 ("Biella", "BI"), ("Cuneo", "CN"), ("Novara", "NO"),
                 ("Verbano-Cusio-Ossola", "VB"), ("Vercelli", "VC")),
    "Puglia": (("Bari", "BA"), ("Brindisi", "BR"), ("Foggia", "FG"), ("Lecce", "LE"),
               ("Taranto", "TA"), ("Barletta-Andria-Trani", "BT")),
    "Sardegna": (("Cagliari", "CA"), ("Nuoro", "NU"), ("Oristano", "OR"),
                 ("Sassari", "SS"), ("Sud Sardegna", "SU")),
    "Sicilia": (("Palermo", "PA"), ("Catania", "CT"), ("Messina", "ME"),
                ("Agrigento", "AG"), ("Caltanissetta", "CL"), ("Enna", "EN"),
                ("Ragusa", "RG"), ("Siracusa", "SR"), ("Trapani", "TP")),
    "Toscana": (("Firenze", "FI"), ("Arezzo", "AR"), ("Grosseto", "GR"),
                ("Livorno", "LI"), ("Lucca", "LU"), ("Massa-Carrara", "MS"),
                ("Pisa", "PI"), ("Pistoia", "PT"), ("Prato", "PO"), ("Siena", "SI")),
    "Trentino-Alto Adige": (("Trento", "TN"), ("Bolzano", "BZ")),
    "Umbria": (("Perugia", "PG"), ("Terni", "TR")),
    "Valle d'Aosta": (("Aosta", "AO"),),
    "Veneto": (("Venezia", "VE"), ("Verona", "VR"), ("Padova", "PD"),
               ("Vicenza", "VI"), ("Treviso", "TV"), ("Rovigo", "RO"),
               ("Belluno", "BL")),
}

_IT_REGION_BY_SIGLA = {
    sigla: region for region, provinces in _IT_PROVINCES.items()
    for _name, sigla in provinces
}

_IT_REGION_BY_NAME = {
    _fold(name): region for region, provinces in _IT_PROVINCES.items()
    for name, _sigla in provinces
}

_IT_NAME_ALIASES = {
    "monza": "Lombardia", "monza e brianza": "Lombardia",
    "monza brianza": "Lombardia", "pesaro urbino": "Marche",
    "pesaro": "Marche", "urbino": "Marche", "forli": "Emilia-Romagna",
    "cesena": "Emilia-Romagna", "massa": "Toscana", "carrara": "Toscana",
    "aquila": "Abruzzo", "bozen": "Trentino-Alto Adige",
    "sudtirol": "Trentino-Alto Adige", "verbania": "Piemonte",
    "reggio nell emilia": "Emilia-Romagna", "reggio di calabria": "Calabria",
    "aosta": "Valle d'Aosta", "vibo": "Calabria",
    "barletta andria trani": "Puglia", "andria": "Puglia", "trani": "Puglia",
}

_IT_SIGLA_ALIASES = {
    "FO": "Emilia-Romagna", "PS": "Marche", "VS": "Sardegna", "CI": "Sardegna",
    "OG": "Sardegna", "OT": "Sardegna", "MB": "Lombardia",
}

_IT_BY_CAP2 = {
    "00": "Lazio", "01": "Lazio", "02": "Lazio", "03": "Lazio", "04": "Lazio",
    "05": "Umbria", "06": "Umbria",
    "07": "Sardegna", "08": "Sardegna", "09": "Sardegna",
    "10": "Piemonte", "11": "Valle d'Aosta", "12": "Piemonte", "13": "Piemonte",
    "14": "Piemonte", "15": "Piemonte",
    "16": "Liguria", "17": "Liguria", "18": "Liguria", "19": "Liguria",
    "20": "Lombardia", "21": "Lombardia", "22": "Lombardia", "23": "Lombardia",
    "24": "Lombardia", "25": "Lombardia", "26": "Lombardia", "27": "Lombardia",
    "28": "Piemonte", "29": "Emilia-Romagna",
    "30": "Veneto", "31": "Veneto", "32": "Veneto",
    "33": "Friuli-Venezia Giulia", "34": "Friuli-Venezia Giulia",
    "35": "Veneto", "36": "Veneto", "37": "Veneto",
    "38": "Trentino-Alto Adige", "39": "Trentino-Alto Adige",
    "40": "Emilia-Romagna", "41": "Emilia-Romagna", "42": "Emilia-Romagna",
    "43": "Emilia-Romagna", "44": "Emilia-Romagna", "45": "Veneto",
    "46": "Lombardia", "47": "Emilia-Romagna", "48": "Emilia-Romagna",
    "50": "Toscana", "51": "Toscana", "52": "Toscana", "53": "Toscana",
    "54": "Toscana", "55": "Toscana", "56": "Toscana", "57": "Toscana",
    "58": "Toscana", "59": "Toscana",
    "60": "Marche", "61": "Marche", "62": "Marche", "63": "Marche",
    "64": "Abruzzo", "65": "Abruzzo", "66": "Abruzzo", "67": "Abruzzo",
    "70": "Puglia", "71": "Puglia", "72": "Puglia", "73": "Puglia", "74": "Puglia",
    "75": "Basilicata", "76": "Puglia",
    "80": "Campania", "81": "Campania", "82": "Campania", "83": "Campania",
    "84": "Campania", "85": "Basilicata", "86": "Molise",
    "87": "Calabria", "88": "Calabria", "89": "Calabria",
    "90": "Sicilia", "91": "Sicilia", "92": "Sicilia", "93": "Sicilia",
    "94": "Sicilia", "95": "Sicilia", "96": "Sicilia", "97": "Sicilia",
    "98": "Sicilia",
}


def _german_state(zip_code: str | None) -> str | None:
    """Bundesland from a PLZ: the three-digit exceptions first, then the Leitregion."""
    digits = _zip_digits(zip_code)
    if len(digits) < 5:
        return None
    return _DE_BY_PREFIX3.get(digits[:3]) or _DE_BY_PREFIX2.get(digits[:2])


def _french_departement_code(zip_code: str | None) -> str | None:
    """The département number a code postal falls in: '75', '2B', '974'."""
    digits = _zip_digits(zip_code)
    if len(digits) < 5:
        return None
    if digits[:2] in ("97", "98"):
        return digits[:3] if digits[:3] in _FR_OVERSEAS else None
    if digits.startswith("20"):
        return "2A" if digits[:3] in ("200", "201") else "2B"
    return digits[:2] if digits[:2] in _FR_DEPARTEMENTS else None


def french_departement(zip_code: str | None) -> str | None:
    """The département a code postal belongs to, Corsica's two halves included."""
    code = _french_departement_code(zip_code)
    if code is None:
        return None
    return _FR_OVERSEAS.get(code) or _FR_DEPARTEMENTS.get(code)


def _french_region(zip_code: str | None) -> str | None:
    """Région from a code postal; an overseas département is its own région."""
    code = _french_departement_code(zip_code)
    if code is None:
        return None
    if code in _FR_OVERSEAS:
        return _FR_OVERSEAS[code]
    return _FR_REGION_BY_DEPARTEMENT.get(code)


def italian_province_region(city: str | None) -> str | None:
    """Regione out of an AutoScout24 city string, by sigla first and name second.

    The site writes the province into the city field in three shapes —
    "Poviglio - Reggio Emilia - RE", "Carrara - Massa Carrara", "Ardea - Roma"
    — so the segments are read right to left: the rightmost is the province
    when there is one, and a bare city name is only a province when the city
    and the province share a name, which is the common case for the big ones.
    """
    if not city:
        return None
    segments = [s.strip() for s in re.split(r"\s[-–]\s|,", str(city)) if s.strip()]
    for segment in reversed(segments):
        upper = segment.upper()
        if len(segment) == 2 and segment.isalpha():
            region = _IT_REGION_BY_SIGLA.get(upper) or _IT_SIGLA_ALIASES.get(upper)
            if region:
                return region
        key = _fold(segment)
        region = _IT_REGION_BY_NAME.get(key) or _IT_NAME_ALIASES.get(key)
        if region:
            return region
    return None


def _italian_region(zip_code: str | None, city: str | None) -> str | None:
    """Regione from the city string, falling back to the CAP's leading digits."""
    region = italian_province_region(city)
    if region:
        return region
    digits = _zip_digits(zip_code)
    if len(digits) < 5:
        return None
    return _IT_BY_CAP2.get(digits[:2])


_RESOLVERS = {
    "DE": lambda zip_code, city: _german_state(zip_code),
    "FR": lambda zip_code, city: _french_region(zip_code),
    "IT": _italian_region,
}


def region_for(country_code: str, zip_code: str | None,
               city: str | None = None) -> str | None:
    """The listing's region in its own country's vocabulary, or None if unplaceable.

    *country_code* is the ISO code AutoScout24 puts on the card's location, so
    a German page selling an Austrian car returns None rather than pretending
    ``1010`` is a PLZ. Callers store whatever comes back straight into
    ``import_listings.region``, which the country frame then serves as
    ``district``.
    """
    resolver = _RESOLVERS.get(str(country_code or "").strip().upper())
    return resolver(zip_code, city) if resolver else None
