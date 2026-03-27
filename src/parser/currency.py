"""Currency helpers for OLX.pt (Portugal).

All prices on OLX.pt are in EUR — no conversion needed.
This module is kept for potential future EUR/USD conversion if needed.
"""


def normalize_price(price: float, currency: str = "EUR") -> float:
    """Normalize price. OLX.pt is EUR-only, so this is a passthrough."""
    return price
