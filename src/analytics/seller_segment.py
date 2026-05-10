"""Seller-side segmentation — coarse buckets on top of the seller_* fields.

Four buckets + an ``unknown`` fallback, so downstream surfaces (per-segment
residuals on the calibration panel, segment-aware flipper thresholds,
trust priors in decision logic) can branch on a single label instead of
re-deriving the same heuristics. Boundaries follow the design discussion
of 2026-05-10:

    genuine_private    is_business=False AND cars_count<=2 AND
                       listings_count_90d<=2 AND NOT pseudoprivate
                       (one-car-for-sale individual, the canonical case)

    pseudo_private     pseudoprivate=True OR (private account with
                       cars_count>=3 OR listings_count_90d>=3) — the
                       account behaves like a reseller while presenting
                       as Particular

    specialist_dealer  is_business=True AND distinct_brands<=2 — a
                       brand-focused trader (single-marque used-car shop,
                       premium/specialised inventory)

    volume_dealer      is_business=True AND (distinct_brands>=3 OR
                       cars_count>=15) — multi-brand car supermarket

    unknown            seller profile not yet fetched
                       (seller_is_business is None / NaN). Backfill
                       hasn't reached this listing or seller_uuid is
                       still null.

The classifier reads the same flat ``seller_*`` columns that
``Repository.fetch_listings_df`` already produces — see
``src/storage/repository.py`` ``_seller_features``. No SQL changes
needed; no new columns to plumb through the snapshot loader.

NOT a price-model feature yet. The 60-day uuid-history calibration
window only just started accumulating; using ``seller_segment`` as a
categorical feature in the model risks re-introducing the same thin-data
overfit the flipper recalibration was designed to avoid. For now this
is a pure observability/analysis surface — the dashboard reports
per-segment median residual so we can see *which* sellers drive the
global +€414 / +3.7% bias before deciding whether to retrain.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


SEGMENT_GENUINE_PRIVATE = "genuine_private"
SEGMENT_PSEUDO_PRIVATE = "pseudo_private"
SEGMENT_SPECIALIST_DEALER = "specialist_dealer"
SEGMENT_VOLUME_DEALER = "volume_dealer"
SEGMENT_UNKNOWN = "unknown"

SEGMENTS = (
    SEGMENT_GENUINE_PRIVATE,
    SEGMENT_PSEUDO_PRIVATE,
    SEGMENT_SPECIALIST_DEALER,
    SEGMENT_VOLUME_DEALER,
    SEGMENT_UNKNOWN,
)

SEGMENT_LABELS = {
    SEGMENT_GENUINE_PRIVATE: "Genuine private",
    SEGMENT_PSEUDO_PRIVATE: "Pseudo-private flipper",
    SEGMENT_SPECIALIST_DEALER: "Specialist dealer",
    SEGMENT_VOLUME_DEALER: "Volume dealer",
    SEGMENT_UNKNOWN: "Unknown (no profile)",
}


def _coerce_int(v: Any) -> int | None:
    if v is None:
        return None
    if isinstance(v, float) and pd.isna(v):
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _coerce_bool(v: Any) -> bool | None:
    if v is None:
        return None
    if isinstance(v, float) and pd.isna(v):
        return None
    return bool(v)


def classify_seller_segment(row: pd.Series | dict) -> str:
    """Return the seller-segment label for one listing row.

    ``row`` may be a Series or plain dict; missing keys are treated as
    None. The segment label is one of ``SEGMENTS`` — never raises.

    Decision order matters: ``pseudoprivate`` short-circuits before any
    cars_count check, because the JSON-vs-presentation contradiction is
    the single highest-confidence reseller signal we have.
    """
    is_biz = _coerce_bool(row.get("seller_is_business"))
    if is_biz is None:
        return SEGMENT_UNKNOWN

    pseudoprivate = bool(_coerce_bool(row.get("seller_pseudoprivate")))
    cars = _coerce_int(row.get("seller_cars_count"))
    distinct = _coerce_int(row.get("seller_distinct_car_brands"))
    rot90 = _coerce_int(row.get("seller_listings_count_90d"))

    if pseudoprivate:
        return SEGMENT_PSEUDO_PRIVATE

    if not is_biz:
        # Truly private side. NULL counts default to the conservative
        # "looks private" interpretation — pseudo-private requires
        # positive evidence (>=3 concurrent cars or >=3 in 90d), not
        # just "we don't know yet".
        cars_v = cars if cars is not None else 1
        rot_v = rot90 if rot90 is not None else 0
        if cars_v >= 3 or rot_v >= 3:
            return SEGMENT_PSEUDO_PRIVATE
        return SEGMENT_GENUINE_PRIVATE

    # Business side. distinct_car_brands defaults to 1 (a typical
    # single-marque shop) when missing — keeps a fresh business with no
    # facet snapshot yet from getting tagged as multi-brand by accident.
    distinct_v = distinct if distinct is not None else 1
    cars_v = cars if cars is not None else 0
    if distinct_v >= 3 or cars_v >= 15:
        return SEGMENT_VOLUME_DEALER
    return SEGMENT_SPECIALIST_DEALER


def add_seller_segment_column(df: pd.DataFrame, *, col: str = "seller_segment") -> pd.DataFrame:
    """Vectorised wrapper — adds a string column to ``df`` in place and
    returns the frame. No-op when none of the ``seller_*`` columns exist
    (e.g. unmatched listings frame after a release rollback)."""
    expected = {
        "seller_is_business", "seller_pseudoprivate",
        "seller_cars_count", "seller_distinct_car_brands",
        "seller_listings_count_90d",
    }
    if not expected.intersection(df.columns):
        return df
    df[col] = df.apply(classify_seller_segment, axis=1)
    return df
