"""Car generation lookup via Wikidata SPARQL API with local seed fallback."""

import json
import logging
import time
from pathlib import Path
import re
import urllib.parse
import urllib.request

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_CACHE_PATH = _DATA_DIR / "generations.json"
_SEED_PATH = _DATA_DIR / "generations_seed.json"
_generations: dict | None = None
_CACHE_MAX_AGE = 30 * 24 * 3600  # refresh from Wikidata every 30 days

WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"

_SPARQL_QUERY = """\
SELECT ?brandLabel ?seriesLabel ?genLabel
       (YEAR(?start) AS ?yearFrom) (YEAR(?end) AS ?yearTo)
WHERE {
  ?gen wdt:P31 wd:Q3231690 .
  ?gen wdt:P179 ?series .
  ?gen wdt:P580 ?start .
  OPTIONAL { ?gen wdt:P582 ?end . }
  { ?gen wdt:P176 ?brand . } UNION { ?gen wdt:P8720 ?brand . }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en,mul". }
}
ORDER BY ?brandLabel ?seriesLabel ?yearFrom
"""

# Wikidata brand labels → OLX brand names
_BRAND_NORMALIZE: dict[str, str] = {
    "Volkswagen Group": "Volkswagen",
    "Mercedes-Benz Group": "Mercedes-Benz",
    "Audi AG": "Audi",
    "Ford Motor Company": "Ford",
    "Ford of Europe": "Ford",
    "BMW Group": "BMW",
    "Hyundai Motor Company": "Hyundai",
    "Jaguar Cars": "Jaguar",
    "Lotus Cars": "Lotus",
    "General Motors do Brasil": "Chevrolet",
    "Chrysler Fevre Argentina S.A.": "Chrysler",
    "GM CAMI Assembly": "GM",
    "British Motor Corporation": "BMC",
    "Morris Motors": "Morris",
    "Chery Automobile": "Chery",
    "BYD Auto": "BYD",
    "Mitsubishi Motors": "Mitsubishi",
}

# OLX model name → Wikidata-extracted series name
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
        "535": "5 Series", "540": "5 Series", "545": "5 Series",
        "550": "5 Series",
        "630 Gran Turismo": "6 Series Gran Turismo",
        "640": "6 Series", "650": "6 Series",
        "730": "7 Series", "740": "7 Series", "750": "7 Series",
    },
    "Mercedes-Benz": {
        "A 160": "A-Class", "A 180": "A-Class", "A 200": "A-Class",
        "A 220": "A-Class", "A 250": "A-Class",
        "B 180": "B-Class", "B 200": "B-Class", "B 220": "B-Class",
        "C 180": "C-Class", "C 200": "C-Class", "C 220": "C-Class",
        "C 250": "C-Class", "C 300": "C-Class", "C 350": "C-Class",
        "E 200": "E-Class", "E 220": "E-Class", "E 250": "E-Class",
        "E 300": "E-Class", "E 350": "E-Class",
        "CLA 180": "CLA-Class", "CLA 200": "CLA-Class",
        "CLA 220": "CLA-Class", "CLA 250": "CLA-Class",
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
}

_Q_CODE_RE = re.compile(r"^Q\d+$")


def _normalize_brand(label: str) -> str:
    """Normalize Wikidata brand label to OLX brand name."""
    if _Q_CODE_RE.match(label):
        return label  # unresolved — will be skipped
    return _BRAND_NORMALIZE.get(label, label)


def _strip_prefix(text: str, prefix: str) -> str:
    """Strip brand prefix from series/gen labels."""
    if text.startswith(prefix + " "):
        return text[len(prefix) + 1:]
    return text


def fetch_generations() -> dict:
    """Fetch car generation data from Wikidata SPARQL and merge with seed."""
    logger.info("Fetching car generations from Wikidata SPARQL...")
    body = urllib.parse.urlencode({"query": _SPARQL_QUERY}).encode()
    req = urllib.request.Request(
        WIKIDATA_ENDPOINT,
        data=body,
        headers={
            "Accept": "application/sparql-results+json",
            "User-Agent": "olx-car-parser/1.0 (https://github.com/nikit34/olx-car-parser)",
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        raw = json.loads(resp.read())

    seen: set[tuple] = set()
    result: dict[str, dict[str, list]] = {}

    for row in raw["results"]["bindings"]:
        raw_brand = row["brandLabel"]["value"]
        brand = _normalize_brand(raw_brand)
        if _Q_CODE_RE.match(brand):
            continue  # skip unresolved brands

        series_label = row["seriesLabel"]["value"]
        gen_label = row["genLabel"]["value"]

        year_from_raw = row.get("yearFrom", {}).get("value")
        year_to_raw = row.get("yearTo", {}).get("value")
        if not year_from_raw:
            continue
        year_from = int(year_from_raw)
        year_to = int(year_to_raw) if year_to_raw else 2025

        # Extract model / gen name by stripping brand prefix
        # Try both raw and normalized brand as prefix
        model = series_label
        for prefix in (raw_brand, brand):
            model = _strip_prefix(model, prefix)
        gen_name = gen_label
        for prefix in (raw_brand, brand):
            gen_name = _strip_prefix(gen_name, prefix)

        key = (brand, model, gen_name, year_from)
        if key in seen:
            continue
        seen.add(key)

        result.setdefault(brand, {}).setdefault(model, []).append({
            "name": gen_name,
            "year_from": year_from,
            "year_to": year_to,
        })

    # Sort generations by year_from
    for brand_data in result.values():
        for model_gens in brand_data.values():
            model_gens.sort(key=lambda g: g["year_from"])

    wikidata_count = sum(len(g) for m in result.values() for g in m.values())
    logger.info("Wikidata returned %d brands, %d generations", len(result), wikidata_count)

    # Merge seed data — seed fills gaps, Wikidata takes priority
    if _SEED_PATH.exists():
        with open(_SEED_PATH, encoding="utf-8") as f:
            seed = json.load(f)
        seed_added = 0
        for brand, models in seed.items():
            for model, gens in models.items():
                if model not in result.get(brand, {}):
                    result.setdefault(brand, {})[model] = gens
                    seed_added += len(gens)
        if seed_added:
            logger.info("Merged %d generations from seed file", seed_added)

    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    total = sum(len(g) for m in result.values() for g in m.values())
    logger.info("Total: %d brands, %d generations saved to %s", len(result), total, _CACHE_PATH)

    global _generations
    _generations = result
    return result


def _cache_is_stale() -> bool:
    """Check if cache file is missing or older than _CACHE_MAX_AGE."""
    if not _CACHE_PATH.exists():
        return True
    age = time.time() - _CACHE_PATH.stat().st_mtime
    return age > _CACHE_MAX_AGE


def load_generations() -> dict:
    """Load generations, auto-fetching from Wikidata if cache is stale."""
    global _generations
    if _generations is not None:
        return _generations

    if _cache_is_stale():
        try:
            _generations = fetch_generations()
            return _generations
        except Exception as e:
            logger.warning("Wikidata fetch failed, using local data: %s", e)

    if _CACHE_PATH.exists():
        with open(_CACHE_PATH, encoding="utf-8") as f:
            _generations = json.load(f)
    elif _SEED_PATH.exists():
        with open(_SEED_PATH, encoding="utf-8") as f:
            _generations = json.load(f)
    else:
        _generations = {}

    return _generations


def get_generation(brand: str, model: str, year: int | None) -> str | None:
    """Return generation name for a given car, or None if unknown."""
    if not year:
        return None
    data = load_generations()

    # Direct lookup
    gens = data.get(brand, {}).get(model)

    # Try alias
    if not gens:
        alias = _MODEL_ALIASES.get(brand, {}).get(model)
        if alias:
            gens = data.get(brand, {}).get(alias)

    if not gens:
        return None

    for g in gens:
        if g["year_from"] <= year <= g["year_to"]:
            return g["name"]
    return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data = fetch_generations()
    total = sum(len(g) for m in data.values() for g in m.values())
    print(f"Total: {len(data)} brands, {total} generations")
