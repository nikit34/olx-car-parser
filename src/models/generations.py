"""Car generation lookup: DBpedia SPARQL + LLM fallback for full coverage."""

import json
import logging
import os
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import httpx
import yaml

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_CACHE_PATH = _DATA_DIR / "generations.json"
_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "settings.yaml"
_generations: dict | None = None
_CACHE_MAX_AGE = 24 * 3600  # refresh daily

# =========================================================================
# Brand normalization & model aliases
# =========================================================================

_BRAND_NORMALIZE: dict[str, str] = {
    "Volkswagen Group": "Volkswagen", "FAW-Volkswagen": "Volkswagen",
    "Mercedes-Benz Group": "Mercedes-Benz", "Daimler-Benz": "Mercedes-Benz",
    "Audi AG": "Audi",
    "Ford Motor Company": "Ford", "Ford of Europe": "Ford",
    "Hyundai Motor Company": "Hyundai",
    "Jaguar Cars": "Jaguar", "Jaguar Land Rover": "Jaguar",
    "Lotus Cars": "Lotus", "Mitsubishi Motors": "Mitsubishi",
    "Renault Sport": "Renault", "Peugeot Sport": "Peugeot",
    "Dongfeng Peugeot-Citroën": "Citroën",
    "Nissan Motor Company": "Nissan",
    "Honda Motor Company": "Honda", "Toyota Motor Corporation": "Toyota",
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
        "Grand Scénic": "Scénic",
    },
    "Peugeot": {"e-208": "208", "307 SW": "307", "407 SW": "407", "508 SW": "508"},
    "Volkswagen": {"Golf Variant": "Golf"},
    "VW": {"Golf": "Golf"},
    "Seat": {"Ibiza ST": "Ibiza"},
}

# =========================================================================
# Provider 1: DBpedia SPARQL
# =========================================================================

_DBPEDIA_ENDPOINT = "https://dbpedia.org/sparql"
_DBPEDIA_QUERY = """\
PREFIX dbo: <http://dbpedia.org/ontology/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT DISTINCT ?genLabel ?mfgLabel ?sy ?ey WHERE {
  ?gen a dbo:Automobile .
  ?gen dbo:manufacturer ?mfg .
  ?gen dbo:productionStartYear ?sy .
  OPTIONAL { ?gen dbo:productionEndYear ?ey . }
  ?gen rdfs:label ?genLabel . FILTER(LANG(?genLabel) = "en")
  ?mfg rdfs:label ?mfgLabel . FILTER(LANG(?mfgLabel) = "en")
} ORDER BY ?mfgLabel ?genLabel
"""

_PAREN_RE = re.compile(r"^(.+?)\s*\(([^)]+)\)\s*$")
_MK_RE = re.compile(r"^(.+?)\s+(Mk\s*\.?\s*\d+)\s*$", re.I)
_ORDINAL_RE = re.compile(
    r"^(.+?)\s*\((first|second|third|fourth|fifth|sixth|seventh|"
    r"eighth|ninth|tenth|eleventh|twelfth)\s+generation.*\)\s*$", re.I,
)
_ORDINAL_MAP = {
    "first": "I", "second": "II", "third": "III", "fourth": "IV",
    "fifth": "V", "sixth": "VI", "seventh": "VII", "eighth": "VIII",
    "ninth": "IX", "tenth": "X", "eleventh": "XI", "twelfth": "XII",
}


def _parse_dbpedia_label(label: str, brand: str) -> tuple[str, str] | None:
    name = label
    for prefix in (brand, brand.split()[0]):
        if name.startswith(prefix + " "):
            name = name[len(prefix) + 1:]
            break
    m = _ORDINAL_RE.match(name)
    if m:
        return m.group(1).strip(), _ORDINAL_MAP[m.group(2).lower()]
    m = _PAREN_RE.match(name)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    m = _MK_RE.match(name)
    if m:
        return m.group(1).strip(), m.group(2).replace(" ", "").replace(".", "")
    return None


def _fetch_dbpedia() -> dict[str, dict[str, list]]:
    logger.info("Provider: DBpedia — fetching...")
    body = urllib.parse.urlencode({"query": _DBPEDIA_QUERY}).encode()
    req = urllib.request.Request(
        _DBPEDIA_ENDPOINT, data=body,
        headers={
            "Accept": "application/sparql-results+json",
            "User-Agent": "olx-car-parser/1.0",
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = json.loads(resp.read())

    seen: set[tuple] = set()
    result: dict[str, dict[str, list]] = {}
    for row in raw["results"]["bindings"]:
        raw_brand = row["mfgLabel"]["value"]
        brand = _BRAND_NORMALIZE.get(raw_brand, raw_brand)
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
        if year_from < 1950 or year_from > 2030:
            continue
        parsed = _parse_dbpedia_label(gen_label, brand)
        if not parsed:
            parsed = _parse_dbpedia_label(gen_label, raw_brand)
        if not parsed:
            continue
        series, gen_name = parsed
        key = (brand, series, gen_name)
        if key in seen:
            continue
        seen.add(key)
        result.setdefault(brand, {}).setdefault(series, []).append({
            "name": gen_name, "year_from": year_from, "year_to": year_to,
        })

    _fix_year_ranges(result)
    total = sum(len(g) for m in result.values() for g in m.values())
    logger.info("DBpedia: %d brands, %d generations", len(result), total)
    return result


# =========================================================================
# Provider 2: LLM (OpenRouter)
# =========================================================================

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_LLM_PROMPT = """\
List ALL generations of the car "{brand} {model}" with production year ranges.
Return ONLY a JSON array: [{{"name": "Mk1", "year_from": 1974, "year_to": 1983}}]
If only one generation exists, return one entry. If model unknown, return []."""


def _get_llm_config() -> dict:
    cfg = {}
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH) as f:
            data = yaml.safe_load(f) or {}
        cfg = data.get("llm", {})
    return {
        "api_key": os.environ.get("OPENROUTER_API_KEY", cfg.get("openrouter_api_key", "")),
        "model": cfg.get("generation_model", "google/gemma-3-12b-it:free"),
    }


def _llm_query(brand: str, model: str) -> list[dict] | None:
    cfg = _get_llm_config()
    if not cfg["api_key"]:
        return None
    try:
        resp = httpx.post(
            _OPENROUTER_URL,
            headers={"Authorization": f"Bearer {cfg['api_key']}", "Content-Type": "application/json"},
            json={
                "model": cfg["model"],
                "messages": [{"role": "user", "content": _LLM_PROMPT.format(brand=brand, model=model)}],
                "max_tokens": 600, "temperature": 0.1,
            },
            timeout=30,
        )
        if resp.status_code != 200:
            return None
        content = resp.json()["choices"][0]["message"].get("content")
        if not content:
            return None
        text = content.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        start, end = text.find("["), text.rfind("]")
        if start >= 0 and end > start:
            text = text[start:end + 1]
        gens = json.loads(text)
        if not isinstance(gens, list):
            return None
        return [
            {"name": str(g["name"]), "year_from": int(g["year_from"]), "year_to": int(g["year_to"])}
            for g in gens
            if isinstance(g, dict) and "name" in g and "year_from" in g and "year_to" in g
        ] or None
    except Exception as e:
        logger.debug("LLM failed for %s %s: %s", brand, model, e)
        return None


def _fill_from_llm(result: dict, db_path: str | None = None) -> int:
    cfg = _get_llm_config()
    if not cfg["api_key"]:
        logger.info("No OpenRouter API key — skipping LLM fill.")
        return 0
    import sqlite3
    db_path = db_path or str(_DATA_DIR / "olx_cars.db")
    if not Path(db_path).exists():
        return 0
    conn = sqlite3.connect(db_path)
    models = conn.execute("SELECT DISTINCT brand, model FROM listings ORDER BY brand, model").fetchall()
    conn.close()

    filled = 0
    failures = 0
    for brand, model in models:
        if _lookup_gens(result, brand, model):
            continue
        gens = _llm_query(brand, model)
        if gens:
            result.setdefault(brand, {})[model] = sorted(gens, key=lambda g: g["year_from"])
            filled += 1
            failures = 0
        else:
            failures += 1
            if failures >= 5:
                logger.warning("5 consecutive LLM failures — stopping.")
                break
        time.sleep(1)

    if filled:
        logger.info("LLM filled %d missing models", filled)
    return filled


# =========================================================================
# Helpers
# =========================================================================

def _fix_year_ranges(data: dict):
    for brand_data in data.values():
        for model_gens in brand_data.values():
            model_gens.sort(key=lambda g: g["year_from"])
            for i, gen in enumerate(model_gens):
                if gen["year_to"] <= gen["year_from"]:
                    if i + 1 < len(model_gens):
                        gen["year_to"] = model_gens[i + 1]["year_from"] - 1
                    else:
                        gen["year_to"] = 2026


def _merge(base: dict, extra: dict):
    for brand, models in extra.items():
        for model, gens in models.items():
            if model not in base.get(brand, {}):
                base.setdefault(brand, {})[model] = gens


def _lookup_gens(data: dict, brand: str, model: str) -> list | None:
    gens = data.get(brand, {}).get(model)
    if gens:
        return gens
    alias = _MODEL_ALIASES.get(brand, {}).get(model)
    if alias:
        return data.get(brand, {}).get(alias)
    return None


# =========================================================================
# Public API
# =========================================================================

def fetch_generations() -> dict:
    """Fetch from DBpedia → LLM fallback."""
    result = {}

    try:
        _merge(result, _fetch_dbpedia())
    except Exception as e:
        logger.warning("DBpedia failed: %s", e)

    try:
        _fill_from_llm(result)
    except Exception as e:
        logger.warning("LLM fill failed: %s", e)

    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    total = sum(len(g) for m in result.values() for g in m.values())
    logger.info("Total: %d brands, %d generations saved to %s", len(result), total, _CACHE_PATH)

    global _generations
    _generations = result
    return result


def _cache_is_stale() -> bool:
    if not _CACHE_PATH.exists():
        return True
    return (time.time() - _CACHE_PATH.stat().st_mtime) > _CACHE_MAX_AGE


def load_generations() -> dict:
    """Load generations, auto-fetching if cache is stale (daily)."""
    global _generations
    if _generations is not None:
        return _generations
    if _cache_is_stale():
        try:
            _generations = fetch_generations()
            return _generations
        except Exception as e:
            logger.warning("Fetch failed, using cache: %s", e)
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
    gens = _lookup_gens(data, brand, model)
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
    print(f"\nTotal: {len(data)} brands, {total} generations")
