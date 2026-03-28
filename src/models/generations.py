"""Car generation lookup from config/generations.json."""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_GENERATIONS_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "generations.json"
_generations: dict | None = None

# =========================================================================
# Brand & model normalization for lookup
# =========================================================================

# Listing brand → canonical brand used in generations.json
_BRAND_LOOKUP: dict[str, str] = {
    "VW": "Volkswagen",
    "Citroën": "Citroen",
    "SEAT": "Seat",
}

_MODEL_ALIASES: dict[str, dict[str, str]] = {
    "BMW": {
        "116": "1 Series", "118": "1 Series", "120": "1 Series",
        "125": "1 Series", "130": "1 Series", "135": "1 Series",
        "218": "2 Series", "220": "2 Series", "225": "2 Series",
        "230": "2 Series",
        "316": "3 Series", "318": "3 Series", "320": "3 Series",
        "325": "3 Series", "328": "3 Series", "330": "3 Series",
        "335": "3 Series", "340": "3 Series",
        "418 Gran Coupé": "4 Series Gran Coupé",
        "420": "4 Series", "420 Gran Coupé": "4 Series Gran Coupé",
        "425": "4 Series", "430": "4 Series", "435": "4 Series",
        "440": "4 Series",
        "520": "5 Series", "525": "5 Series", "530": "5 Series",
        "535": "5 Series", "540": "5 Series", "550": "5 Series",
        "630 Gran Turismo": "6 Series Gran Turismo",
        "640": "6 Series", "650": "6 Series",
        "730": "7 Series", "740": "7 Series", "750": "7 Series",
    },
    "Mercedes-Benz": {
        "180": "C-Class", "220": "E-Class",
        "A 160": "A-Class", "A 180": "A-Class", "A 200": "A-Class",
        "A 220": "A-Class", "A 250": "A-Class",
        "B 180": "B-Class", "B 200": "B-Class", "B 220": "B-Class",
        "C 180": "C-Class", "C 200": "C-Class", "C 220": "C-Class",
        "C 250": "C-Class", "C 300": "C-Class", "C 350": "C-Class",
        "E 200": "E-Class", "E 220": "E-Class", "E 250": "E-Class",
        "E 300": "E-Class", "E 350": "E-Class",
        "CLA 180": "CLA-Class", "CLA 200": "CLA-Class",
        "CLA 220": "CLA-Class", "CLA 250": "CLA-Class",
        "CLA 45 AMG": "CLA-Class", "CLC 220": "CLC-Class",
        "GLA 180": "GLA-Class", "GLA 200": "GLA-Class",
        "GLA 220": "GLA-Class",
        "GLB 180": "GLB-Class", "GLB 200": "GLB-Class",
        "GLC 220": "GLC-Class", "GLC 250": "GLC-Class",
        "GLC 300": "GLC-Class",
    },
    "Audi": {
        "A1 Sportback": "A1", "A3 Sportback": "A3",
        "A4 Avant": "A4", "A5 Cabrio": "A5",
        "S3 Limousine": "S3", "Q4 Sportback e-tron": "Q4 e-tron",
    },
    "Opel": {"Astra Sports Tourer": "Astra"},
    "Renault": {
        "Clio Sport Tourer": "Clio", "Mégane Break": "Mégane",
        "Mégane E-Tech": "Mégane",
        "Grand Scénic": "Scénic", "Grand Modus": "Modus",
    },
    "Peugeot": {"e-208": "208", "307 SW": "307", "407 SW": "407", "508 SW": "508"},
    "Volkswagen": {"Golf Variant": "Golf", "ID.7": "ID.7"},
    "VW": {"Golf": "Golf"},
    "Seat": {"Ibiza ST": "Ibiza"},
    "Citroen": {
        "C5 Break": "C5", "DS4": "DS4",
        "e-Mehari": "e-Mehari",
    },
    "Porsche": {"Panamera Sport Turismo": "Panamera", "718 Boxster": "Boxster"},
    "Volvo": {"XC 90": "XC90"},
    "Land Rover": {"Evoque": "Range Rover Evoque"},
    "Smart": {"ForTwo Coupé": "ForTwo"},
    "Toyota": {"107": "Aygo"},
}


# =========================================================================
# Public API
# =========================================================================

def load_generations() -> dict:
    """Load generations from config/generations.json."""
    global _generations
    if _generations is not None:
        return _generations
    if _GENERATIONS_PATH.exists():
        with open(_GENERATIONS_PATH, encoding="utf-8") as f:
            _generations = json.load(f)
    else:
        logger.warning("generations.json not found at %s", _GENERATIONS_PATH)
        _generations = {}
    return _generations


def _lookup_gens(data: dict, brand: str, model: str) -> list | None:
    """Look up generation list, trying brand aliases and model aliases."""
    for b in (brand, _BRAND_LOOKUP.get(brand, brand)):
        gens = data.get(b, {}).get(model)
        if gens:
            return gens
        alias = _MODEL_ALIASES.get(b, {}).get(model)
        if alias:
            gens = data.get(b, {}).get(alias)
            if gens:
                return gens
    return None


def get_generation(brand: str, model: str, year: int | None) -> str | None:
    """Return generation name for a given car, or None if unknown."""
    if not year:
        return None
    data = load_generations()
    gens = _lookup_gens(data, brand, model)
    if not gens:
        return None
    for g in gens:
        if g["year_from"] <= year <= g["year_to"]:
            return g["name"]
    return None
