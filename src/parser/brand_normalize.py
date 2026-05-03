"""Brand-name canonicalisation.

OLX and StandVirtual let sellers free-text the brand, so the same make
shows up under multiple spellings — "VW" vs "Volkswagen", "Citroen"
(no diacritic) vs "Citroën", etc. The dashboard's segment ranker and
market trend pages then split that bucket in two and under-count
liquidity.

Applied at write time in ``upsert_listing`` so new rows land canonical;
the ``scripts/normalize_brands.py`` one-shot fixes the back-catalogue.
"""
from __future__ import annotations


# (alias_lowercase → canonical brand). Keys are lowercased for matching.
_ALIASES: dict[str, str] = {
    # Volkswagen
    "vw": "Volkswagen",
    "v.w.": "Volkswagen",
    "volks wagen": "Volkswagen",
    # Citroën — accent canonical
    "citroen": "Citroën",
    "citroën": "Citroën",
    "citröen": "Citroën",
    # Mercedes — Mercedes-Benz canonical (matches what most listings already use)
    "mercedes": "Mercedes-Benz",
    "mercedes benz": "Mercedes-Benz",
    "mercedesbenz": "Mercedes-Benz",
    # Skoda / Škoda
    "skoda": "Skoda",
    "škoda": "Skoda",
    # Renault sometimes "RENAULT" full caps after some scrape paths
    "renault": "Renault",
    # BMW occasionally lowercase
    "bmw": "BMW",
    # Land Rover / Range Rover
    "land-rover": "Land Rover",
    "landrover": "Land Rover",
    # Alfa Romeo
    "alfa-romeo": "Alfa Romeo",
    "alfaromeo": "Alfa Romeo",
}


def normalize_brand(value: str | None) -> str:
    """Return the canonical spelling for a free-text brand string.

    Unknown brands pass through with whitespace stripped — we don't
    auto-titlecase, since it'd mangle multi-word names ("DS" → "Ds")
    and lower-the-bar to introducing new dups.
    """
    if value is None:
        return ""
    s = str(value).strip()
    if not s:
        return ""
    key = s.lower().strip()
    return _ALIASES.get(key, s)
