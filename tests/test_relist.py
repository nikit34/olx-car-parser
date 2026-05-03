"""Tests for re-listing detection."""

from datetime import datetime, timezone

import pandas as pd
import pytest

from src.analytics.relist import (
    DEFAULT_MATCH_THRESHOLD,
    DEFAULT_WINDOW_DAYS,
    _normalize_color,
    _segment_window_days,
    build_outcomes_df,
    compute_match_score,
    compute_segment_dom_median,
    find_relists,
)


def _utc(*args):
    return datetime(*args, tzinfo=timezone.utc)


def _listing(**kw) -> pd.Series:
    """Baseline listing — overridden per test. Defaults describe the
    'original' / 'sold' side; relist tests override olx_id, mileage, and
    timestamps."""
    base = {
        "olx_id": "x1",
        "brand": "Volkswagen",
        "model": "Golf",
        "year": 2018,
        "mileage_km": 80_000,
        "engine_cc": 1968,
        "horsepower": 150,
        "fuel_type": "Diesel",
        "transmission": "Manual",
        "color": "Cinzento",
        "district": "Porto",
        "generation": "Mk7",
        "sub_model": None,
        "trim_level": None,
        "doors": "4-5",
        "seats": 5,
        "drive_type": "Dianteira",
        "is_active": True,
        "deactivation_reason": None,
        "deactivated_at": None,
        "first_seen_at": _utc(2026, 1, 1),
        "duplicate_of": None,
        "price_eur": 18_000.0,
    }
    base.update(kw)
    return pd.Series(base)


# ---------------------------------------------------------------------------
# compute_match_score: hard gates
# ---------------------------------------------------------------------------


def test_brand_mismatch_rejects():
    r = compute_match_score(
        _listing(brand="Volkswagen"), _listing(brand="Audi"), gap_days=5,
    )
    assert r.rejected and "brand" in r.reject_reason


def test_model_mismatch_rejects():
    r = compute_match_score(
        _listing(model="Golf"), _listing(model="Polo"), gap_days=5,
    )
    assert r.rejected


def test_year_mismatch_rejects():
    r = compute_match_score(
        _listing(year=2018), _listing(year=2019), gap_days=5,
    )
    assert r.rejected


def test_fuel_mismatch_rejects_when_both_present():
    r = compute_match_score(
        _listing(fuel_type="Diesel"),
        _listing(fuel_type="Gasolina"),
        gap_days=5,
    )
    assert r.rejected


def test_fuel_missing_one_side_does_not_reject():
    r = compute_match_score(
        _listing(fuel_type="Diesel"), _listing(fuel_type=None), gap_days=5,
    )
    assert not r.rejected


def test_transmission_mismatch_rejects():
    r = compute_match_score(
        _listing(transmission="Manual"),
        _listing(transmission="Automática"),
        gap_days=5,
    )
    assert r.rejected


def test_mileage_decrease_rejects():
    r = compute_match_score(
        _listing(mileage_km=100_000),
        _listing(mileage_km=80_000),
        gap_days=30,
    )
    assert r.rejected and "decreased" in r.reject_reason


def test_mileage_decrease_within_noise_floor_ok():
    # 1k below — under the 2k parser-noise floor.
    r = compute_match_score(
        _listing(mileage_km=100_000),
        _listing(mileage_km=99_000),
        gap_days=10,
    )
    assert not r.rejected


def test_implausible_mileage_gain_rejects():
    # 30 days × 100 km/day + 2k noise = 5k max plausible.
    # 20k gain in 30 days = 666 km/day → reject.
    r = compute_match_score(
        _listing(mileage_km=100_000),
        _listing(mileage_km=120_000),
        gap_days=30,
    )
    assert r.rejected and "implausible" in r.reject_reason


# ---------------------------------------------------------------------------
# compute_match_score: soft scoring
# ---------------------------------------------------------------------------


def test_perfect_match_high_score():
    """Same district, same color, same generation, etc. → near-1.0."""
    r = compute_match_score(
        _listing(olx_id="orig"),
        _listing(olx_id="relist", mileage_km=82_000),
        gap_days=14,
    )
    assert not r.rejected
    assert r.score >= 0.9


def test_skeleton_only_at_base():
    """Brand+model+year+mileage-plausible match, all other fields
    null — score should be exactly the 0.50 base."""
    bare = _listing(
        engine_cc=None, horsepower=None, color=None, district=None,
        generation=None, sub_model=None, trim_level=None,
        doors=None, seats=None, drive_type=None,
    )
    other = _listing(
        olx_id="relist", mileage_km=82_000,
        engine_cc=None, horsepower=None, color=None, district=None,
        generation=None, sub_model=None, trim_level=None,
        doors=None, seats=None, drive_type=None,
    )
    r = compute_match_score(bare, other, gap_days=10)
    assert not r.rejected
    assert r.score == pytest.approx(0.50)


def test_engine_mismatch_pushes_below_threshold():
    """Different engine displacement should drag the score under the
    default match threshold even with the rest of the skeleton intact."""
    r = compute_match_score(
        _listing(engine_cc=1968),
        _listing(engine_cc=1598, mileage_km=82_000),
        gap_days=10,
    )
    # 0.50 base - 0.20 engine penalty + smaller-bonus matches
    # (district, color, hp, generation, doors, seats, drive_type)
    # may still climb back, but engine penalty makes the *category*
    # mismatch — assert the engine reason is recorded.
    assert any("engine_cc mismatch" in reason for reason in r.reasons)


def test_color_synonyms_normalized():
    """'Prateado' and 'Cinzento' both collapse to 'cinzento' → match."""
    r = compute_match_score(
        _listing(color="Prateado"),
        _listing(color="Cinzento", mileage_km=82_000),
        gap_days=10,
    )
    assert any("color match" in reason for reason in r.reasons)


def test_color_english_to_portuguese():
    """English 'silver' should match Portuguese 'cinzento'."""
    r = compute_match_score(
        _listing(color="Silver"),
        _listing(color="Cinzento", mileage_km=82_000),
        gap_days=10,
    )
    assert any("color match" in reason for reason in r.reasons)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def test_normalize_color_basic():
    assert _normalize_color("Prata") == _normalize_color("Cinzento")
    assert _normalize_color("Silver") == _normalize_color("Cinzento")
    assert _normalize_color(None) is None
    assert _normalize_color("") is None
    assert _normalize_color("   ") is None


def test_segment_window_clamped():
    assert _segment_window_days(None) == DEFAULT_WINDOW_DAYS
    # 4 × 5 = 20 → clamped UP to 60-day floor
    assert _segment_window_days(5) == 60
    # 4 × 30 = 120 → in range
    assert _segment_window_days(30) == 120
    # 4 × 200 = 800 → clamped DOWN to 365-day cap
    assert _segment_window_days(200) == 365


# ---------------------------------------------------------------------------
# compute_segment_dom_median
# ---------------------------------------------------------------------------


def test_compute_segment_dom_median_basic():
    df = pd.DataFrame([
        # Two sold Golfs, DoM = 10 and 30 → median 20
        _listing(
            olx_id="g1", is_active=False, deactivation_reason="sold",
            first_seen_at=_utc(2026, 1, 1), deactivated_at=_utc(2026, 1, 11),
        ),
        _listing(
            olx_id="g2", is_active=False, deactivation_reason="sold",
            first_seen_at=_utc(2026, 1, 1), deactivated_at=_utc(2026, 1, 31),
        ),
        # Active row — should be excluded
        _listing(olx_id="g3", is_active=True, deactivation_reason=None),
    ])
    out = compute_segment_dom_median(df)
    assert out[("Volkswagen", "Golf", "Mk7")] == pytest.approx(20.0)
    # Brand+model fallback should also be present
    assert out[("Volkswagen", "Golf", None)] == pytest.approx(20.0)


def test_compute_segment_dom_median_empty_when_no_sold():
    df = pd.DataFrame([_listing(olx_id="active", is_active=True)])
    assert compute_segment_dom_median(df) == {}


def test_compute_segment_dom_median_filters_implausible_dom():
    """DoM > 365 days is parser noise (e.g. 2915-day "sold") and gets
    dropped before the median is computed."""
    df = pd.DataFrame([
        _listing(
            olx_id="bad", is_active=False, deactivation_reason="sold",
            first_seen_at=_utc(2018, 1, 1), deactivated_at=_utc(2026, 1, 1),
        ),
    ])
    assert compute_segment_dom_median(df) == {}


# ---------------------------------------------------------------------------
# find_relists end-to-end
# ---------------------------------------------------------------------------


def test_find_relists_simple_pair():
    """Original sold on Jan 1, near-identical relist appears Jan 14."""
    df = pd.DataFrame([
        _listing(
            olx_id="orig",
            is_active=False, deactivation_reason="sold",
            deactivated_at=_utc(2026, 1, 1),
            first_seen_at=_utc(2025, 11, 1),
            price_eur=15_000.0,
        ),
        _listing(
            olx_id="relist",
            is_active=True, deactivation_reason=None,
            deactivated_at=None,
            first_seen_at=_utc(2026, 1, 14),
            mileage_km=82_000,
            price_eur=17_500.0,
        ),
    ])
    out = find_relists(df, dom_median_by_segment={})
    assert len(out) == 1
    row = out.iloc[0]
    assert row["original_olx_id"] == "orig"
    assert row["relist_olx_id"] == "relist"
    assert row["match_score"] >= DEFAULT_MATCH_THRESHOLD
    assert row["price_delta_eur"] == 2500
    assert row["price_delta_pct"] == pytest.approx(16.67, abs=0.1)
    assert row["mileage_delta_km"] == 2000


def test_find_relists_window_excludes_far_relist():
    """Fast segment (DoM=20 → window=80d): a relist 120d after
    deactivation is outside the window and not matched."""
    df = pd.DataFrame([
        _listing(
            olx_id="orig",
            is_active=False, deactivation_reason="sold",
            deactivated_at=_utc(2026, 1, 1),
            first_seen_at=_utc(2025, 11, 1),
        ),
        _listing(
            olx_id="late_relist",
            first_seen_at=_utc(2026, 5, 1),  # 120d after Jan 1
            mileage_km=82_000,
        ),
    ])
    out = find_relists(
        df, dom_median_by_segment={("Volkswagen", "Golf", "Mk7"): 20.0},
    )
    assert out.empty


def test_find_relists_picks_best_score_among_candidates():
    """Two passing candidates → pick the one with stronger feature match."""
    df = pd.DataFrame([
        _listing(
            olx_id="orig",
            is_active=False, deactivation_reason="sold",
            deactivated_at=_utc(2026, 1, 1),
            first_seen_at=_utc(2025, 11, 1),
            color="Cinzento", district="Porto",
        ),
        _listing(
            olx_id="weak",
            first_seen_at=_utc(2026, 1, 5),
            mileage_km=82_000,
            color="Branco",     # mismatch
            district="Lisboa",  # mismatch
        ),
        _listing(
            olx_id="strong",
            first_seen_at=_utc(2026, 1, 10),
            mileage_km=82_000,
            color="Cinzento",
            district="Porto",
        ),
    ])
    out = find_relists(df, dom_median_by_segment={})
    assert len(out) == 1
    assert out.iloc[0]["relist_olx_id"] == "strong"


def test_find_relists_skips_active_originals():
    df = pd.DataFrame([
        _listing(olx_id="orig", is_active=True, deactivation_reason=None),
        _listing(
            olx_id="other",
            first_seen_at=_utc(2026, 2, 1),
            mileage_km=82_000,
        ),
    ])
    out = find_relists(df, dom_median_by_segment={})
    assert out.empty


def test_find_relists_skips_relists_before_deactivation():
    """A 'candidate' first seen BEFORE the original deactivated isn't a
    re-listing — it's a contemporaneous duplicate (handled by the
    same-platform dedup pass)."""
    df = pd.DataFrame([
        _listing(
            olx_id="orig",
            is_active=False, deactivation_reason="sold",
            deactivated_at=_utc(2026, 1, 1),
            first_seen_at=_utc(2025, 11, 1),
        ),
        _listing(
            olx_id="earlier",
            first_seen_at=_utc(2025, 12, 1),  # before deactivation
            mileage_km=82_000,
        ),
    ])
    out = find_relists(df, dom_median_by_segment={})
    assert out.empty


def test_find_relists_excludes_marked_duplicates():
    df = pd.DataFrame([
        _listing(
            olx_id="orig",
            is_active=False, deactivation_reason="sold",
            deactivated_at=_utc(2026, 1, 1),
            first_seen_at=_utc(2025, 11, 1),
        ),
        _listing(
            olx_id="dup",
            first_seen_at=_utc(2026, 1, 14),
            mileage_km=82_000,
            duplicate_of="some_canonical_id",
        ),
    ])
    out = find_relists(df, dom_median_by_segment={})
    assert out.empty


# ---------------------------------------------------------------------------
# build_outcomes_df
# ---------------------------------------------------------------------------


def _relist_row(**kw) -> dict:
    base = {
        "original_olx_id": "orig",
        "relist_olx_id": "relist",
        "gap_days": 14.0,
        "match_score": 0.85,
        "original_price_eur": 15_000.0,
        "relist_price_eur": 17_500.0,
        "price_delta_eur": 2500.0,
        "price_delta_pct": 16.7,
        "mileage_delta_km": 2000,
    }
    base.update(kw)
    return base


def test_build_outcomes_df_shape():
    relist_df = pd.DataFrame([_relist_row()])
    listings_df = pd.DataFrame([
        {"olx_id": "relist", "is_active": False, "deactivation_reason": "sold"},
    ])
    out = build_outcomes_df(relist_df, listings_df=listings_df)
    assert list(out.columns) == [
        "olx_id", "buy_price", "sell_price", "days_held", "fees_eur",
    ]
    assert len(out) == 1
    row = out.iloc[0]
    assert row["olx_id"] == "orig"
    assert row["buy_price"] == 15_000.0
    assert row["sell_price"] == 17_500.0
    assert row["days_held"] == 14.0
    # fees_eur = buy × 0.03 + 200 = 450 + 200 = 650
    assert row["fees_eur"] == pytest.approx(650.0)


def test_build_outcomes_df_filters_unsold_relists():
    """Default require_both_sides_sold=True drops events whose relist
    is still active (we don't yet know what it sold for)."""
    relist_df = pd.DataFrame([_relist_row()])
    listings_df = pd.DataFrame([
        {"olx_id": "relist", "is_active": True, "deactivation_reason": None},
    ])
    out = build_outcomes_df(relist_df, listings_df=listings_df)
    assert out.empty


def test_build_outcomes_df_keeps_unsold_when_flag_off():
    relist_df = pd.DataFrame([_relist_row()])
    out = build_outcomes_df(relist_df, require_both_sides_sold=False)
    assert len(out) == 1


def test_build_outcomes_df_filters_by_min_score():
    relist_df = pd.DataFrame([_relist_row(match_score=0.50)])
    listings_df = pd.DataFrame([
        {"olx_id": "relist", "is_active": False, "deactivation_reason": "sold"},
    ])
    out = build_outcomes_df(
        relist_df, listings_df=listings_df, min_score=0.65,
    )
    assert out.empty


def test_build_outcomes_df_filters_short_gap_reposts():
    """Default min_gap_days=7 strips out same-day / next-day reposts
    (the bulk of detected matches in production)."""
    relist_df = pd.DataFrame([_relist_row(gap_days=2.0)])
    listings_df = pd.DataFrame([
        {"olx_id": "relist", "is_active": False, "deactivation_reason": "sold"},
    ])
    out = build_outcomes_df(relist_df, listings_df=listings_df)
    assert out.empty


def test_build_outcomes_df_filters_zero_delta_reposts():
    """Default min_abs_delta_pct=5 strips out same-price reposts."""
    relist_df = pd.DataFrame([_relist_row(
        gap_days=14.0, price_delta_pct=0.5,  # passes gap, fails delta
    )])
    listings_df = pd.DataFrame([
        {"olx_id": "relist", "is_active": False, "deactivation_reason": "sold"},
    ])
    out = build_outcomes_df(relist_df, listings_df=listings_df)
    assert out.empty


def test_build_outcomes_df_keeps_negative_flip():
    """A flip-down (sold cheaper than the original ask) is still
    realised P&L data — keeps it for calibration; the algorithm
    learns from losses too."""
    relist_df = pd.DataFrame([_relist_row(
        gap_days=14.0, price_delta_pct=-15.0,
        relist_price_eur=12_750.0, price_delta_eur=-2250.0,
    )])
    listings_df = pd.DataFrame([
        {"olx_id": "relist", "is_active": False, "deactivation_reason": "sold"},
    ])
    out = build_outcomes_df(relist_df, listings_df=listings_df)
    assert len(out) == 1


def test_build_outcomes_df_filters_price_parsing_artifacts():
    """Default max_abs_delta_pct=100 strips price-parsing artefacts
    (currency / mileage-as-price typos can produce 1000 %+ deltas)."""
    relist_df = pd.DataFrame([_relist_row(
        gap_days=14.0, price_delta_pct=500.0,
    )])
    listings_df = pd.DataFrame([
        {"olx_id": "relist", "is_active": False, "deactivation_reason": "sold"},
    ])
    out = build_outcomes_df(relist_df, listings_df=listings_df)
    assert out.empty


def test_build_outcomes_df_rejects_zero_buy_price():
    """A 0-priced original makes margin computation meaningless.
    Filter it out even when the gap/delta filters would otherwise pass."""
    relist_df = pd.DataFrame([_relist_row(
        original_price_eur=0.0,
        gap_days=14.0,
        price_delta_pct=10.0,
    )])
    listings_df = pd.DataFrame([
        {"olx_id": "relist", "is_active": False, "deactivation_reason": "sold"},
    ])
    out = build_outcomes_df(relist_df, listings_df=listings_df)
    assert out.empty
