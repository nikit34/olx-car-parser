"""Car generation lookup via DBpedia SPARQL + LLM fallback for full coverage."""

import json
import logging
import os
import time
from pathlib import Path
import re
import urllib.parse
import urllib.request

import httpx
import yaml

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_CACHE_PATH = _DATA_DIR / "generations.json"
_generations: dict | None = None
_CACHE_MAX_AGE = 30 * 24 * 3600  # refresh every 30 days

DBPEDIA_ENDPOINT = "https://dbpedia.org/sparql"

_SPARQL_QUERY = """\
PREFIX dbo: <http://dbpedia.org/ontology/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT DISTINCT ?genLabel ?mfgLabel ?sy ?ey
WHERE {
  ?gen a dbo:Automobile .
  ?gen dbo:manufacturer ?mfg .
  ?gen dbo:productionStartYear ?sy .
  OPTIONAL { ?gen dbo:productionEndYear ?ey . }
  ?gen rdfs:label ?genLabel . FILTER(LANG(?genLabel) = "en")
  ?mfg rdfs:label ?mfgLabel . FILTER(LANG(?mfgLabel) = "en")
}
ORDER BY ?mfgLabel ?genLabel
"""

# DBpedia manufacturer labels → OLX brand names
_BRAND_NORMALIZE: dict[str, str] = {
    "Volkswagen Group": "Volkswagen",
    "FAW-Volkswagen": "Volkswagen",
    "Mercedes-Benz Group": "Mercedes-Benz",
    "Daimler-Benz": "Mercedes-Benz",
    "Audi AG": "Audi",
    "Ford Motor Company": "Ford",
    "Ford of Europe": "Ford",
    "Hyundai Motor Company": "Hyundai",
    "Jaguar Cars": "Jaguar",
    "Jaguar Land Rover": "Jaguar",
    "Lotus Cars": "Lotus",
    "Mitsubishi Motors": "Mitsubishi",
    "Renault Sport": "Renault",
    "Peugeot Sport": "Peugeot",
    "Dongfeng Peugeot-Citroën": "Citroën",
    "Nissan Motor Company": "Nissan",
    "Honda Motor Company": "Honda",
    "Toyota Motor Corporation": "Toyota",
    "Stellantis": "Stellantis",
}

# OLX model name → DBpedia-extracted series name
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

# Regex patterns to split "Series (Generation)" from label
_PAREN_RE = re.compile(r"^(.+?)\s*\(([^)]+)\)\s*$")          # "3 Series (E90)"
_MK_RE = re.compile(r"^(.+?)\s+(Mk\s*\.?\s*\d+)\s*$", re.I)  # "Golf Mk7"
_ORDINAL_RE = re.compile(                                      # "Fiesta (fifth generation)"
    r"^(.+?)\s*\((first|second|third|fourth|fifth|sixth|seventh|"
    r"eighth|ninth|tenth|eleventh|twelfth)\s+generation.*\)\s*$", re.I
)
_ORDINAL_MAP = {
    "first": "I", "second": "II", "third": "III", "fourth": "IV",
    "fifth": "V", "sixth": "VI", "seventh": "VII", "eighth": "VIII",
    "ninth": "IX", "tenth": "X", "eleventh": "XI", "twelfth": "XII",
}


def _parse_label(label: str, brand: str) -> tuple[str, str] | None:
    """Parse generation label → (series, gen_name). None if not a generation item."""
    name = label
    # Strip brand prefix (try various forms)
    for prefix in (brand, brand.split()[0]):
        if name.startswith(prefix + " "):
            name = name[len(prefix) + 1:]
            break

    # "Fiesta (fifth generation)" → ("Fiesta", "V")
    m = _ORDINAL_RE.match(name)
    if m:
        return m.group(1).strip(), _ORDINAL_MAP[m.group(2).lower()]

    # "3 Series (E90)" / "C-Class (W205)" / "Yaris (XP150)"
    m = _PAREN_RE.match(name)
    if m:
        return m.group(1).strip(), m.group(2).strip()

    # "Golf Mk7" / "Polo Mk4"
    m = _MK_RE.match(name)
    if m:
        series = m.group(1).strip()
        gen = m.group(2).replace(" ", "").replace(".", "")
        return series, gen

    return None


def _normalize_brand(label: str) -> str:
    return _BRAND_NORMALIZE.get(label, label)


def fetch_generations() -> dict:
    """Fetch car generation data from DBpedia SPARQL and save to cache."""
    logger.info("Fetching car generations from DBpedia...")
    body = urllib.parse.urlencode({"query": _SPARQL_QUERY}).encode()
    req = urllib.request.Request(
        DBPEDIA_ENDPOINT,
        data=body,
        headers={
            "Accept": "application/sparql-results+json",
            "User-Agent": "olx-car-parser/1.0 (https://github.com/nikit34/olx-car-parser)",
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = json.loads(resp.read())

    seen: set[tuple] = set()
    result: dict[str, dict[str, list]] = {}

    for row in raw["results"]["bindings"]:
        raw_brand = row["mfgLabel"]["value"]
        brand = _normalize_brand(raw_brand)

        gen_label = row["genLabel"]["value"]

        try:
            year_from = int(row["sy"]["value"])
        except (ValueError, KeyError):
            continue
        year_to_raw = row.get("ey", {}).get("value")
        try:
            year_to = int(year_to_raw) if year_to_raw else 2026
        except ValueError:
            year_to = 2026

        # Validate years
        if year_from < 1950 or year_from > 2030:
            continue

        # Parse label into (series, generation)
        parsed = _parse_label(gen_label, brand)
        if not parsed:
            # Also try with raw brand name
            parsed = _parse_label(gen_label, raw_brand)
        if not parsed:
            continue

        series, gen_name = parsed

        key = (brand, series, gen_name)
        if key in seen:
            continue
        seen.add(key)

        result.setdefault(brand, {}).setdefault(series, []).append({
            "name": gen_name,
            "year_from": year_from,
            "year_to": year_to,
        })

    # Sort and fix year ranges: if year_to <= year_from, infer from next gen
    for brand_data in result.values():
        for model_gens in brand_data.values():
            model_gens.sort(key=lambda g: g["year_from"])
            for i, gen in enumerate(model_gens):
                if gen["year_to"] <= gen["year_from"]:
                    if i + 1 < len(model_gens):
                        gen["year_to"] = model_gens[i + 1]["year_from"] - 1
                    else:
                        gen["year_to"] = 2026

    dbpedia_total = sum(len(g) for m in result.values() for g in m.values())
    logger.info("DBpedia: %d brands, %d generations", len(result), dbpedia_total)

    # Fill gaps with LLM for models in our database but missing from DBpedia
    llm_filled = fill_missing_from_llm(result)

    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    total = sum(len(g) for m in result.values() for g in m.values())
    logger.info("Total: %d brands, %d generations (%d from LLM) saved to %s",
                len(result), total, llm_filled, _CACHE_PATH)

    global _generations
    _generations = result
    return result


# ---------------------------------------------------------------------------
# LLM fallback for models not covered by DBpedia
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "settings.yaml"
_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

_LLM_PROMPT = """\
List ALL generations of the car model "{brand} {model}" with production year ranges.
Return ONLY a JSON array, no extra text. Example format:
[{{"name": "Mk1", "year_from": 1974, "year_to": 1983}}, {{"name": "Mk2", "year_from": 1983, "year_to": 1992}}]
If the model has only one generation or you are unsure, return a single entry covering the full production span.
If the model does not exist, return an empty array []."""


def _get_llm_config() -> dict:
    cfg = {}
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH) as f:
            data = yaml.safe_load(f) or {}
        cfg = data.get("llm", {})
    return {
        "api_key": os.environ.get("OPENROUTER_API_KEY", cfg.get("openrouter_api_key", "")),
        "model": cfg.get("model", "nvidia/nemotron-3-super-120b-a12b:free"),
    }


def _llm_query_generations(brand: str, model: str) -> list[dict] | None:
    """Ask LLM for generation data of a single brand+model."""
    cfg = _get_llm_config()
    if not cfg["api_key"]:
        return None
    try:
        resp = httpx.post(
            _OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {cfg['api_key']}",
                "Content-Type": "application/json",
            },
            json={
                "model": cfg["model"],
                "messages": [{"role": "user", "content": _LLM_PROMPT.format(brand=brand, model=model)}],
                "max_tokens": 800,
                "temperature": 0.1,
            },
            timeout=30,
        )
        if resp.status_code != 200:
            return None
        content = resp.json()["choices"][0]["message"]["content"]
        # Extract JSON from possible markdown
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        gens = json.loads(content.strip())
        if not isinstance(gens, list):
            return None
        # Validate structure
        valid = []
        for g in gens:
            if isinstance(g, dict) and "name" in g and "year_from" in g and "year_to" in g:
                valid.append({"name": str(g["name"]), "year_from": int(g["year_from"]), "year_to": int(g["year_to"])})
        return valid if valid else None
    except Exception as e:
        logger.debug("LLM generation query failed for %s %s: %s", brand, model, e)
        return None


def fill_missing_from_llm(result: dict, db_path: str | None = None) -> int:
    """Query LLM for brand+model combos present in DB but missing from result.

    Modifies result in-place, returns count of models filled.
    """
    cfg = _get_llm_config()
    if not cfg["api_key"]:
        logger.info("No OpenRouter API key — skipping LLM generation fill.")
        return 0

    # Get unique brand+model from database
    import sqlite3
    if db_path is None:
        db_path = str(_DATA_DIR / "olx_cars.db")
    if not Path(db_path).exists():
        return 0

    conn = sqlite3.connect(db_path)
    models = conn.execute("SELECT DISTINCT brand, model FROM listings ORDER BY brand, model").fetchall()
    conn.close()

    filled = 0
    failures = 0
    for brand, model in models:
        # Check if already covered (direct or via alias)
        if result.get(brand, {}).get(model):
            continue
        alias = _MODEL_ALIASES.get(brand, {}).get(model)
        if alias and result.get(brand, {}).get(alias):
            continue

        gens = _llm_query_generations(brand, model)
        if gens:
            result.setdefault(brand, {})[model] = sorted(gens, key=lambda g: g["year_from"])
            filled += 1
            failures = 0
            logger.debug("LLM filled %s %s: %d generations", brand, model, len(gens))
        else:
            failures += 1
            if failures >= 3:
                logger.warning("3 consecutive LLM failures, stopping generation fill.")
                break

    if filled:
        logger.info("LLM filled %d missing models", filled)
    return filled


def _cache_is_stale() -> bool:
    if not _CACHE_PATH.exists():
        return True
    return (time.time() - _CACHE_PATH.stat().st_mtime) > _CACHE_MAX_AGE


def load_generations() -> dict:
    """Load generations, auto-fetching from DBpedia if cache is stale."""
    global _generations
    if _generations is not None:
        return _generations

    if _cache_is_stale():
        try:
            _generations = fetch_generations()
            return _generations
        except Exception as e:
            logger.warning("DBpedia fetch failed, using local cache: %s", e)

    if _CACHE_PATH.exists():
        with open(_CACHE_PATH, encoding="utf-8") as f:
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
    # Show sample
    for brand in sorted(data)[:30]:
        for model in sorted(data[brand]):
            gens = [g["name"] for g in data[brand][model]]
            print(f"  {brand:20s} {model:25s} → {gens}")
