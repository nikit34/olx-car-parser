"""Car generation lookup from config JSON files."""

import json
import logging
import re
import unicodedata
from pathlib import Path

logger = logging.getLogger(__name__)

_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"
_generations: dict | None = None
_brand_aliases: dict | None = None
_model_aliases: dict | None = None
_norm_index_src: dict | None = None
_norm_index: dict[str, dict[str, tuple[str, str]]] = {}


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


def _norm_key(value: str) -> str:
    """Case-, accent- and punctuation-insensitive form of a brand or model name."""
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = text.encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _normalized_index(data: dict) -> dict[str, dict[str, tuple[str, str]]]:
    """Map normalized (brand, model) to the canonical spelling used in *data*.

    Built once per generations dict and reused; the source dict is held by
    reference so a patched table in tests rebuilds instead of hitting a
    stale cache. Exact spellings already win in ``_lookup_gens``, so first
    writer wins here and canonical names take precedence over aliases.
    """
    global _norm_index_src, _norm_index
    if data is _norm_index_src:
        return _norm_index

    index: dict[str, dict[str, tuple[str, str]]] = {}
    for brand, models in data.items():
        bucket = index.setdefault(_norm_key(brand), {})
        for model in models:
            bucket.setdefault(_norm_key(model), (brand, model))
    for brand, aliases in _get_model_aliases().items():
        bucket = index.setdefault(_norm_key(brand), {})
        for alias, canonical in aliases.items():
            bucket.setdefault(_norm_key(alias), (brand, canonical))

    _norm_index_src = data
    _norm_index = index
    return index


def _lookup_gens(data: dict, brand: str, model: str) -> list | None:
    """Look up generation list, trying brand aliases and model aliases.

    Falls back to a case- and accent-insensitive match when every exact
    spelling misses. Portuguese listings write the same car as "Série 3",
    "Classe C" or "SEAT Ibiza", and an exact-only lookup dropped the whole
    record — ``get_generation`` returning None means the listing never
    reaches the database at all.
    """
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

    hit = _normalized_index(data).get(_norm_key(brand), {}).get(_norm_key(model))
    if hit:
        canon_brand, canon_model = hit
        gens = data.get(canon_brand, {}).get(canon_model)
        if gens:
            return gens
        alias = model_aliases.get(canon_brand, {}).get(canon_model)
        if alias:
            gens = data.get(canon_brand, {}).get(alias)
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


def infer_model_from_title(
    brand: str, title: str, year: int | None = None
) -> str | None:
    """Return a known *brand* model name found in *title*, or None.

    Word-boundary match so short model codes like ``"320"`` don't fire
    inside ``"2.0"`` or ``"3000"``. Longest-first so ``"Mégane Sport
    Tourer"`` wins over ``"Mégane"``.

    With *year* given, a candidate that has no generation covering that
    year loses to the next one. Numeric model names collide with prices
    and displacements written in the title ("3.100 €" carries "100",
    "BMW 320 D 2000" carries "2000"), and returning the first raw match
    handed the caller a name it could not resolve — the listing was then
    dropped even though a later candidate would have matched.

    The space inside a model name is optional: sellers write "E270",
    "ml320" and "B180d" for what the table calls "E 270", "ML 320" and
    "B 180".
    """
    if not brand or not title:
        return None
    fallback = None
    for m in get_known_models_for_brand(brand):
        pattern = r"\s*".join(re.escape(part) for part in m.split())
        if not re.search(rf"\b{pattern}\b", title, flags=re.IGNORECASE):
            continue
        if year is None:
            return m
        if get_generation(brand, m, year):
            return m
        if fallback is None:
            fallback = m
    return fallback


_model_owner_src: dict | None = None
_model_owner: dict[str, str] = {}


def _model_owners(data: dict) -> dict[str, str]:
    """Map a normalized model name to the single brand that owns it.

    Names shared by two brands ("Corsa" is both an Opel and a Vauxhall) are
    left out — a shared name carries no brand evidence. Rebuilt when the
    generations table changes, same as ``_normalized_index``.
    """
    global _model_owner_src, _model_owner
    if data is _model_owner_src:
        return _model_owner

    from src.parser.brand_normalize import normalize_brand

    canonical = {brand: normalize_brand(brand) for brand in data}
    known = set(canonical.values())
    owners: dict[str, set[str]] = {}
    for brand, models in data.items():
        for model in models:
            owners.setdefault(_norm_key(model), set()).add(canonical[brand])
    for brand, aliases in _get_model_aliases().items():
        canon_brand = normalize_brand(brand)
        if canon_brand not in known:
            continue
        for alias in aliases:
            owners.setdefault(_norm_key(alias), set()).add(canon_brand)

    _model_owner_src = data
    _model_owner = {k: next(iter(v)) for k, v in owners.items() if len(v) == 1}
    return _model_owner


def brand_for_model(model: str) -> str | None:
    """Return the only brand that lists *model*, or None if it is shared.

    The OLX offer JSON carries ``modelo`` but no make at all, so when the
    seller left the make out of the title the model name is the only
    structured brand evidence the listing has.
    """
    if not model:
        return None
    return _model_owners(load_generations()).get(_norm_key(model))
