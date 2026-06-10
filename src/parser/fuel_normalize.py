"""Canonicalise ``fuel_type`` labels that differ only by spelling/case.

OLX and StandVirtual label the same powertrain differently — OLX
``Eléctrico`` vs SV ``Elétrico`` (post-1990 orthography, no first ``c``),
``Híbrido Plug-in`` vs ``Híbrido Plug-In`` (case). LightGBM treats each
distinct string as its own category, so these duplicates fragment the
already-small EV/PHEV classes and dilute their signal; the dashboard's
fuel-group matcher also keys on the OLX spelling and silently mis-buckets
``Elétrico``. Canonicalise at write time (see ``upsert_listing``).

Canonical electric is kept as ``Eléctrico`` — the spelling
``data_loader``'s fuel-group matcher already recognises — so no parallel
change is forced there. We deliberately do NOT merge the hybrid sub-types
``Híbrido (Gasolina)`` / ``Híbrido (Diesel)`` into plain ``Híbrido``: those
carry a real gas-vs-diesel distinction, not just spelling.
"""

from __future__ import annotations

# Lowercased input -> canonical output. Anything not listed is returned
# stripped-but-unchanged (Diesel, Gasolina, GPL, GNC, the Híbrido (…) sub-types).
_FUEL_CANONICAL = {
    "eléctrico": "Eléctrico",
    "elétrico": "Eléctrico",
    "electrico": "Eléctrico",
    "electric": "Eléctrico",
    "híbrido plug-in": "Híbrido Plug-in",
    "hibrido plug-in": "Híbrido Plug-in",
}


def normalize_fuel_type(value: str | None) -> str | None:
    """Map orthographic/case duplicates of a fuel label to one canonical form."""
    if value is None:
        return None
    s = value.strip()
    if not s:
        return s
    return _FUEL_CANONICAL.get(s.lower(), s)
