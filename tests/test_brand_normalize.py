"""Tests for src.parser.brand_normalize."""
from __future__ import annotations

import pytest

from src.parser.brand_normalize import normalize_brand


class TestNormalizeBrand:
    def test_canonical_passes_through(self):
        assert normalize_brand("Volkswagen") == "Volkswagen"
        assert normalize_brand("BMW") == "BMW"
        assert normalize_brand("Citroën") == "Citroën"

    def test_vw_alias_canonicalised(self):
        """The 2026-05-03 audit found 1183 'Volkswagen' + 135 'VW' rows
        showing as separate buckets in the segment ranker. Both must
        normalize to 'Volkswagen'."""
        assert normalize_brand("VW") == "Volkswagen"
        assert normalize_brand("vw") == "Volkswagen"
        assert normalize_brand("V.W.") == "Volkswagen"

    def test_citroen_diacritic_canonicalised(self):
        """611 'Citroën' + 186 'Citroen' on the same release."""
        assert normalize_brand("Citroen") == "Citroën"
        assert normalize_brand("citroen") == "Citroën"
        assert normalize_brand("Citroën") == "Citroën"

    def test_mercedes_canonical(self):
        assert normalize_brand("Mercedes") == "Mercedes-Benz"
        assert normalize_brand("MERCEDES") == "Mercedes-Benz"
        assert normalize_brand("Mercedes-Benz") == "Mercedes-Benz"

    def test_whitespace_handling(self):
        assert normalize_brand("  VW  ") == "Volkswagen"
        assert normalize_brand("\tBMW\n") == "BMW"

    def test_unknown_brand_passes_through_trimmed(self):
        """We don't auto-titlecase unknowns — that'd break multi-word
        and acronym brands. Just trim whitespace, leave casing."""
        assert normalize_brand("DS") == "DS"
        assert normalize_brand("  Cupra  ") == "Cupra"

    def test_empty_and_none(self):
        assert normalize_brand("") == ""
        assert normalize_brand(None) == ""
        assert normalize_brand("   ") == ""
