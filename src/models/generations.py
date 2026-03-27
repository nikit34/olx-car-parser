"""Car generation lookup from reference data."""

import json
from pathlib import Path

_DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "generations.json"
_generations: dict | None = None


def load_generations() -> dict:
    """Load and cache generations reference data."""
    global _generations
    if _generations is None:
        if _DATA_PATH.exists():
            with open(_DATA_PATH, encoding="utf-8") as f:
                _generations = json.load(f)
        else:
            _generations = {}
    return _generations


def get_generation(brand: str, model: str, year: int | None) -> str | None:
    """Return generation name for a given car, or None if unknown."""
    if not year:
        return None
    data = load_generations()
    models = data.get(brand, {})
    gens = models.get(model)
    if not gens:
        return None
    for g in gens:
        if g["year_from"] <= year <= g["year_to"]:
            return g["name"]
    return None
