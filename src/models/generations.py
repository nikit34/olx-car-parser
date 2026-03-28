"""Car generation lookup from config JSON files."""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"
_generations: dict | None = None
_brand_aliases: dict | None = None
_model_aliases: dict | None = None


def _load_json(path: Path) -> dict:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    logger.warning("%s not found", path)
    return {}


def load_generations() -> dict:
    """Load generations from config/generations.json."""
    global _generations
    if _generations is None:
        _generations = _load_json(_CONFIG_DIR / "generations.json")
    return _generations


def _get_brand_aliases() -> dict:
    global _brand_aliases
    if _brand_aliases is None:
        _brand_aliases = _load_json(_CONFIG_DIR / "brand_aliases.json")
    return _brand_aliases


def _get_model_aliases() -> dict:
    global _model_aliases
    if _model_aliases is None:
        _model_aliases = _load_json(_CONFIG_DIR / "model_aliases.json")
    return _model_aliases


def _lookup_gens(data: dict, brand: str, model: str) -> list | None:
    """Look up generation list, trying brand aliases and model aliases."""
    brand_aliases = _get_brand_aliases()
    model_aliases = _get_model_aliases()

    for b in (brand, brand_aliases.get(brand, brand)):
        gens = data.get(b, {}).get(model)
        if gens:
            return gens
        alias = model_aliases.get(b, {}).get(model)
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
