"""Tests for ``src.analytics.seller_segment``.

Cover the four bucket boundaries + the unknown fallback, plus a couple
of edge cases (pseudoprivate short-circuits, NaN/None handling, default
behaviour when seller_distinct_car_brands is missing).
"""

from __future__ import annotations

import math

import pandas as pd

from src.analytics.seller_segment import (
    SEGMENT_GENUINE_PRIVATE,
    SEGMENT_PSEUDO_PRIVATE,
    SEGMENT_SPECIALIST_DEALER,
    SEGMENT_VOLUME_DEALER,
    SEGMENT_UNKNOWN,
    add_seller_segment_column,
    classify_seller_segment,
)


def _row(**kw) -> dict:
    base = {
        "seller_is_business": None,
        "seller_pseudoprivate": None,
        "seller_cars_count": None,
        "seller_distinct_car_brands": None,
        "seller_listings_count_90d": None,
    }
    base.update(kw)
    return base


def test_unknown_when_business_flag_missing():
    assert classify_seller_segment(_row()) == SEGMENT_UNKNOWN
    assert classify_seller_segment(_row(seller_is_business=float("nan"))) == SEGMENT_UNKNOWN


def test_genuine_private_canonical():
    row = _row(
        seller_is_business=False,
        seller_pseudoprivate=False,
        seller_cars_count=1,
        seller_listings_count_90d=1,
    )
    assert classify_seller_segment(row) == SEGMENT_GENUINE_PRIVATE


def test_genuine_private_with_unknown_counts():
    # NULL cars/rotation on a private account → assume genuine, not pseudo.
    row = _row(seller_is_business=False, seller_pseudoprivate=False)
    assert classify_seller_segment(row) == SEGMENT_GENUINE_PRIVATE


def test_pseudo_private_via_pseudoprivate_flag():
    # The flag short-circuits regardless of cars_count.
    row = _row(
        seller_is_business=True,
        seller_pseudoprivate=True,
        seller_cars_count=1,
    )
    assert classify_seller_segment(row) == SEGMENT_PSEUDO_PRIVATE


def test_pseudo_private_via_concurrent_inventory():
    row = _row(
        seller_is_business=False, seller_pseudoprivate=False,
        seller_cars_count=4,
    )
    assert classify_seller_segment(row) == SEGMENT_PSEUDO_PRIVATE


def test_pseudo_private_via_rotation():
    row = _row(
        seller_is_business=False, seller_pseudoprivate=False,
        seller_cars_count=1, seller_listings_count_90d=5,
    )
    assert classify_seller_segment(row) == SEGMENT_PSEUDO_PRIVATE


def test_specialist_dealer_low_diversity():
    row = _row(
        seller_is_business=True, seller_pseudoprivate=False,
        seller_cars_count=8, seller_distinct_car_brands=2,
    )
    assert classify_seller_segment(row) == SEGMENT_SPECIALIST_DEALER


def test_specialist_dealer_small_inventory():
    # Tiny business, default-distinct=1 → still specialist, not volume.
    row = _row(
        seller_is_business=True, seller_pseudoprivate=False,
        seller_cars_count=3,
    )
    assert classify_seller_segment(row) == SEGMENT_SPECIALIST_DEALER


def test_volume_dealer_via_brand_diversity():
    row = _row(
        seller_is_business=True, seller_pseudoprivate=False,
        seller_cars_count=5, seller_distinct_car_brands=4,
    )
    assert classify_seller_segment(row) == SEGMENT_VOLUME_DEALER


def test_volume_dealer_via_inventory_size():
    row = _row(
        seller_is_business=True, seller_pseudoprivate=False,
        seller_cars_count=20, seller_distinct_car_brands=2,
    )
    assert classify_seller_segment(row) == SEGMENT_VOLUME_DEALER


def test_pseudoprivate_wins_over_volume_dealer():
    # A business JSON contradicting the trader-title is the strongest
    # signal — even if inventory looks like a volume dealer, the
    # presentation mismatch dominates.
    row = _row(
        seller_is_business=True, seller_pseudoprivate=True,
        seller_cars_count=20, seller_distinct_car_brands=5,
    )
    assert classify_seller_segment(row) == SEGMENT_PSEUDO_PRIVATE


def test_accepts_pandas_series():
    s = pd.Series(_row(
        seller_is_business=True, seller_pseudoprivate=False,
        seller_cars_count=2,
    ))
    assert classify_seller_segment(s) == SEGMENT_SPECIALIST_DEALER


def test_add_column_vectorised():
    df = pd.DataFrame([
        _row(seller_is_business=False, seller_cars_count=1),
        _row(seller_is_business=True, seller_cars_count=20, seller_distinct_car_brands=4),
        _row(),  # unknown
    ])
    out = add_seller_segment_column(df)
    assert list(out["seller_segment"]) == [
        SEGMENT_GENUINE_PRIVATE, SEGMENT_VOLUME_DEALER, SEGMENT_UNKNOWN,
    ]


def test_add_column_no_op_when_columns_missing():
    df = pd.DataFrame({"olx_id": ["a", "b"]})
    out = add_seller_segment_column(df)
    assert "seller_segment" not in out.columns


def test_nan_in_counts_handled():
    row = _row(
        seller_is_business=False, seller_pseudoprivate=False,
        seller_cars_count=math.nan, seller_listings_count_90d=math.nan,
    )
    # NaN counts on a private account → genuine (no positive evidence).
    assert classify_seller_segment(row) == SEGMENT_GENUINE_PRIVATE
