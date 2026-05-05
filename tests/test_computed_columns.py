"""Tests for the plate-signal promotion in ``src.analytics.computed_columns``.

The ``photo_damage_*`` signals stay in ``llm_extras`` JSON (consumers
parse on demand), but plate signals are promoted to DataFrame columns
so hazard / decision / anomaly modules can use them directly. Verify
the JSON walk: present-and-true / present-and-false / not-yet-verified
(absent) all map to the right tri-state column values.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.analytics.computed_columns import (
    add_plate_signals,
    enrich_listings,
)


def _df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _readable(out, i):
    """Cell accessor tolerant of pandas dtype quirks (np.bool_, NaN-for-None)."""
    return out.iloc[i]["plate_readable"]


def _is_missing(value) -> bool:
    """True for both Python None and pandas NaN — pandas may coerce object
    columns containing None to float NaN when other rows have ints/strings."""
    return value is None or (isinstance(value, float) and pd.isna(value))


class TestAddPlateSignals:
    def test_promotes_three_columns_when_extras_has_plate_keys(self):
        df = _df([{
            "olx_id": "a",
            "llm_extras": json.dumps({
                "plate_readable": True,
                "plate_n_readable": 2,
                "plate_text_primary": "AB-12-CD",
            }),
        }])
        out = add_plate_signals(df)
        row = out.iloc[0]
        assert bool(row["plate_readable"]) is True
        assert int(row["plate_n_readable"]) == 2
        assert row["plate_text_primary"] == "AB-12-CD"

    def test_distinguishes_verified_no_plate_from_not_verified(self):
        """Tri-state semantics: ``False`` (verified, no plate found) must
        not collapse to the same column value as ``None`` (not yet through
        verify-photos). Downstream features depend on this distinction —
        a row with plate_readable=False is a real "obscured listing"
        signal; a row with plate_readable=None is missing data.
        """
        df = _df([
            {"olx_id": "verified-with-plate", "llm_extras": json.dumps({
                "plate_readable": True, "plate_n_readable": 1,
                "plate_text_primary": "12-AB-34",
            })},
            {"olx_id": "verified-no-plate", "llm_extras": json.dumps({
                "plate_readable": False, "plate_n_readable": 0,
                "plate_text_primary": None,
            })},
            {"olx_id": "not-verified", "llm_extras": json.dumps({
                "damage_severity": 0,  # extras present, but no plate keys
            })},
            {"olx_id": "no-extras", "llm_extras": None},
        ])
        out = add_plate_signals(df)
        # Verified-with-plate row.
        assert bool(out.iloc[0]["plate_readable"]) is True
        assert int(out.iloc[0]["plate_n_readable"]) == 1
        assert out.iloc[0]["plate_text_primary"] == "12-AB-34"
        # Verified-no-plate row — plate_readable False (not None / NaN).
        assert bool(out.iloc[1]["plate_readable"]) is False
        assert not _is_missing(out.iloc[1]["plate_readable"])
        assert int(out.iloc[1]["plate_n_readable"]) == 0
        assert _is_missing(out.iloc[1]["plate_text_primary"])
        # Not-yet-verified rows: tri-state missing on all three.
        for i in (2, 3):
            assert _is_missing(out.iloc[i]["plate_readable"])
            assert _is_missing(out.iloc[i]["plate_n_readable"])
            assert _is_missing(out.iloc[i]["plate_text_primary"])

    def test_handles_malformed_extras_json(self):
        """Garbage in llm_extras (truncated JSON, non-dict root) should
        produce missing columns — never crash and never assert spurious
        plate signals on broken rows."""
        df = _df([
            {"olx_id": "truncated", "llm_extras": '{"plate_readable": tru'},
            {"olx_id": "list-root", "llm_extras": "[1, 2, 3]"},
            {"olx_id": "string-root", "llm_extras": '"just a string"'},
        ])
        out = add_plate_signals(df)
        for i in range(3):
            assert _is_missing(out.iloc[i]["plate_readable"])
            assert _is_missing(out.iloc[i]["plate_n_readable"])
            assert _is_missing(out.iloc[i]["plate_text_primary"])

    def test_accepts_dict_extras_for_in_memory_callers(self):
        """Some callers (test fixtures, in-memory pipeline tests) pass an
        already-parsed dict instead of a JSON string. Don't double-parse
        — accept both shapes."""
        df = pd.DataFrame([{
            "olx_id": "dict-extras",
            "llm_extras": {
                "plate_readable": True,
                "plate_n_readable": 3,
                "plate_text_primary": "AA-00-AA",
            },
        }])
        out = add_plate_signals(df)
        row = out.iloc[0]
        assert bool(row["plate_readable"]) is True
        assert int(row["plate_n_readable"]) == 3
        assert row["plate_text_primary"] == "AA-00-AA"

    def test_no_op_when_llm_extras_column_absent(self):
        """Unmatched-listings DataFrame doesn't carry llm_extras — promotion
        must skip cleanly rather than KeyError. The column simply isn't
        added when the source isn't there."""
        df = _df([{"olx_id": "x", "year": 2020}])
        out = add_plate_signals(df)
        assert "plate_readable" not in out.columns
        assert "plate_n_readable" not in out.columns
        assert "plate_text_primary" not in out.columns

    def test_plate_text_primary_coerces_non_str_to_none(self):
        """Defensive: if a stale row has a non-string primary (legacy bug,
        manual SQL edit), surface it as missing rather than letting the
        bogus type leak into downstream string operations."""
        df = _df([{
            "olx_id": "weird",
            "llm_extras": json.dumps({
                "plate_readable": True, "plate_n_readable": 1,
                "plate_text_primary": 12345,
            }),
        }])
        out = add_plate_signals(df)
        assert _is_missing(out.iloc[0]["plate_text_primary"])


class TestPlateObscuredDerivation:
    """``plate_obscured`` is the gated form of ``plate_readable``: True
    only when verify-photos saw enough exterior photos to make a "no
    plate readable on any of them" call meaningful. Below the threshold,
    the signal is NaN — refuses to fire on small / interior-heavy photo
    sets where false positives dominated the pilot study (2026-05-05).
    """

    def test_obscured_true_when_unreadable_with_enough_exterior(self):
        df = _df([{"olx_id": "a", "llm_extras": json.dumps({
            "plate_readable": False,
            "plate_n_readable": 0,
            "plate_text_primary": None,
            "photo_damage_n_exterior": 7,
        })}])
        out = add_plate_signals(df)
        assert bool(out.iloc[0]["plate_obscured"]) is True

    def test_obscured_false_when_plate_readable(self):
        df = _df([{"olx_id": "a", "llm_extras": json.dumps({
            "plate_readable": True,
            "plate_n_readable": 1,
            "plate_text_primary": "AB-12-CD",
            "photo_damage_n_exterior": 7,
        })}])
        out = add_plate_signals(df)
        assert bool(out.iloc[0]["plate_obscured"]) is False

    def test_obscured_none_when_too_few_exterior(self):
        """4 exterior photos < threshold of 5 → undefined, even though
        plate_readable=False. The pilot showed obscuring rates jump
        from 67% (6-10 photos) to 20% (11-20 photos) — small photo sets
        misclassify "no plate angle in frame" as deliberate hiding."""
        df = _df([{"olx_id": "a", "llm_extras": json.dumps({
            "plate_readable": False,
            "plate_n_readable": 0,
            "plate_text_primary": None,
            "photo_damage_n_exterior": 4,  # below threshold
        })}])
        out = add_plate_signals(df)
        assert _is_missing(out.iloc[0]["plate_obscured"])
        # plate_readable still surfaces (undirected); obscured is the
        # threshold-gated derivation that downstream features should use.
        assert bool(out.iloc[0]["plate_readable"]) is False

    def test_obscured_none_when_plate_keys_absent(self):
        df = _df([{"olx_id": "a", "llm_extras": json.dumps({
            "damage_severity": 0,  # no plate fields written yet
        })}])
        out = add_plate_signals(df)
        assert _is_missing(out.iloc[0]["plate_obscured"])

    def test_obscured_none_when_n_exterior_missing(self):
        """Pre-issue-3 row: plate fields written but n_exterior absent."""
        df = _df([{"olx_id": "a", "llm_extras": json.dumps({
            "plate_readable": False,
            "plate_n_readable": 0,
            "plate_text_primary": None,
        })}])
        out = add_plate_signals(df)
        assert _is_missing(out.iloc[0]["plate_obscured"])


class TestEnrichListingsWiresPlateSignals:
    """``enrich_listings`` is the public entry point — confirm the plate
    columns flow through alongside the existing days_listed / eur_per_km /
    tuning_or_mods_count enrichments."""

    def test_full_enrichment_includes_plate_columns(self):
        df = _df([{
            "olx_id": "a", "first_seen_at": "2026-01-01",
            "price_eur": 10000, "first_price_eur": 10000,
            "mileage_km": 100000,
            "tuning_or_mods": json.dumps(["spoiler"]),
            "llm_extras": json.dumps({
                "plate_readable": True, "plate_n_readable": 2,
                "plate_text_primary": "AB-12-CD",
            }),
        }])
        out = enrich_listings(df)
        # Existing enrichments still present.
        assert "days_listed" in out.columns
        assert "eur_per_km" in out.columns
        assert "tuning_or_mods_count" in out.columns
        # New plate columns wired through.
        assert bool(out.iloc[0]["plate_readable"]) is True
        assert out.iloc[0]["plate_text_primary"] == "AB-12-CD"
