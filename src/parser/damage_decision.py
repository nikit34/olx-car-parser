"""Lightweight photo-damage decision rules — no torch/torchvision import.

``photo_damage.py`` carries the inference path (ResNet50 + transforms) and
therefore pulls in torch/torchvision at module load. Consumers that only
need the *decision* — does this listing's stored ``photo_damage_*`` data
mean the listing is flagged? — don't need any of that, and importing the
heavy module at the dashboard / blocker layer is what forced the
torchvision shim in ``tests/conftest.py``. Splitting the rules out lets
the dashboard import ``is_listing_flagged`` without dragging in 800 MB
of GPU runtime.

``photo_damage`` re-exports everything in this module so existing call
sites keep working unchanged.
"""

from __future__ import annotations


DEFAULT_THRESHOLD = 0.20

# Listing-level multi-photo agreement rule (issue #2, audit context #1).
#
# The audit on a 100-listing flagged sample (#1) showed ~87% FP rate when
# flagging on ``max(p_damaged) >= 0.20``. The dominant failure mode is one
# weirdly-lit photo (sun glare on dark glossy panels, harsh reflections,
# OOD interior/engine-bay shots) hitting p≈0.99 on an otherwise pristine
# car. Raising the per-photo threshold alone only fixes ~7/20 audited FPs;
# the principled fix is requiring agreement across multiple photos so that
# a single anomalous shot can't condemn a 50-photo dealer listing.
#
# ``DamageClassifier.predict_listing`` uses these to set ``is_damaged``:
#
#     is_damaged = sum(p.p_damaged >= FLAG_PHOTO_THRESHOLD for p in photos)
#                  >= FLAG_MIN_PHOTOS
FLAG_MIN_PHOTOS = 2
FLAG_PHOTO_THRESHOLD = 0.30


def is_listing_flagged(extras: dict | None) -> bool:
    """Listing-level damage decision with backward-compat fallback.

    Reads ``photo_damage_flagged`` (the multi-photo agreement field added
    by ``verify-photos`` per issue #2 — production-validated in #1 to drop
    flag rate by 70.6 % vs the old max-rule) when present. For listings
    predating that field, falls back to the v2 max-rule
    (``photo_damage_p >= DEFAULT_THRESHOLD``) so the 6 271 legacy rows
    don't all silently drop out of the deal-blocking path.

    Returns ``False`` for listings with no photo_damage data at all
    (``verify-photos`` hasn't classified them yet, or extras is empty).
    """
    if not extras:
        return False
    new = extras.get("photo_damage_flagged")
    if new is not None:
        return bool(new)
    p = extras.get("photo_damage_p") or 0.0
    try:
        return float(p) >= DEFAULT_THRESHOLD
    except (TypeError, ValueError):
        return False
