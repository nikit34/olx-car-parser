"""Car generation lookup from config JSON files."""

import json
import logging
import re
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
    """Return generation name for a given car, or None if unknown.

    On overlap (adjacent generations sharing a boundary year, e.g. Mk1
    1996-2008 and Mk2 2008-2018), prefer the generation with the latest
    ``year_from`` — the new generation has already started by that
    calendar year. Without this, a 2008 listing would pick Mk1 just
    because it's listed first in the JSON.
    """
    if not year:
        return None
    data = load_generations()
    gens = _lookup_gens(data, brand, model)
    if not gens:
        return None
    best = None
    for g in gens:
        if g["year_from"] <= year <= g["year_to"]:
            if best is None or g["year_from"] > best["year_from"]:
                best = g
    return best["name"] if best else None


_known_models_cache: dict[str, list[str]] = {}


def get_known_models_for_brand(brand: str) -> list[str]:
    """All canonical + alias model names known for *brand*, longest-first.

    Used as a last-resort lexicon when the scraper detail page leaves
    ``model`` empty (StandVirtual frequently does), so we can scan the
    title for a known model name and recover the row.
    """
    if not brand:
        return []
    if brand in _known_models_cache:
        return _known_models_cache[brand]
    data = load_generations()
    brand_aliases = _get_brand_aliases()
    model_aliases = _get_model_aliases()
    models: set[str] = set()
    for b in (brand, brand_aliases.get(brand, brand)):
        models.update(data.get(b, {}).keys())
        models.update(model_aliases.get(b, {}).keys())
    out = sorted(models, key=len, reverse=True)
    _known_models_cache[brand] = out
    return out


def infer_model_from_title(brand: str, title: str) -> str | None:
    """Return a known *brand* model name found in *title*, or None.

    Word-boundary match so short model codes like ``"320"`` don't fire
    inside ``"2.0"`` or ``"3000"``. Longest-first so ``"Mégane Sport
    Tourer"`` wins over ``"Mégane"``.
    """
    if not brand or not title:
        return None
    for m in get_known_models_for_brand(brand):
        if re.search(rf"\b{re.escape(m)}\b", title, flags=re.IGNORECASE):
            return m
    return None
