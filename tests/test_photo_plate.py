"""Tests for ``src.parser.photo_plate.normalize_plate``.

The PlateReader itself wraps EasyOCR and isn't unit-tested here (heavy
dep, slow on CPU). The CLI integration test in
``test_cli_verify_photos.py`` stubs PlateReader to assert the
``plate_*`` keys land in ``llm_extras`` correctly.
"""

from __future__ import annotations

import pytest

from src.parser.photo_plate import normalize_plate


class TestNormalizePlateAccepts:
    """All four documented PT plate eras must round-trip through OCR-y noise."""

    @pytest.mark.parametrize("raw,expected", [
        # 1937–1992: digits only
        ("12-34-56", "12-34-56"),
        ("12 34 56", "12-34-56"),
        ("123456", "12-34-56"),
        # 1992–2005: letters at one end
        ("12-34-AB", "12-34-AB"),
        ("AB-12-34", "AB-12-34"),
        # 2005–2020: digit-letter-digit sandwich
        ("12-AB-34", "12-AB-34"),
        # 2020+: letter-digit-letter sandwich
        ("AB-12-CD", "AB-12-CD"),
        # Lowercase tolerated (uppercase'd internally)
        ("ab-12-cd", "AB-12-CD"),
        # Plate buried in OCR fragment (label, sticker text around it)
        ("Modelo: AB-12-CD valido", "AB-12-CD"),
        # OCR sometimes uses interpunct or dot as separator
        ("AB·12·CD", "AB-12-CD"),
        ("AB.12.CD", "AB-12-CD"),
    ])
    def test_known_pt_layouts(self, raw, expected):
        assert normalize_plate(raw) == expected


class TestNormalizePlateRejects:
    """Anything that doesn't form a documented PT layout is None — keeps
    random 6-char OCR strings out of ``plate_text_primary``."""

    @pytest.mark.parametrize("raw", [
        # Blank / None-ish
        "",
        "   ",
        # All letters — no PT format is LLLLLL
        "ABCDEF",
        "AB-CD-EF",
        # Mixed inside a single 2-char group (OCR bleed)
        "1A-23-45",
        "12-A3-45",
        # Wrong length (under 6 alphanumerics)
        "12-34",
        # Non-PT layout (DLL is not documented)
        "12-AB-CD",
        # Pure letter with digit suffix — not a 2-2-2 layout
        "ABC123",
    ])
    def test_invalid_inputs(self, raw):
        assert normalize_plate(raw) is None


class TestNormalizePlateEdgeCases:
    def test_picks_first_match_in_long_text(self):
        # OCR'd dealer card: "Telefone 912 345 678 ... AB-12-CD ..."
        # The phone number is digits-only and would form a DDD-shaped 2-2-2
        # if it weren't for word boundaries. Our regex requires no
        # alphanumeric on either side of the candidate, so '912 345 678'
        # — which has a 9-digit run — does NOT collapse to a plate.
        text = "Telefone 912 345 678 contato AB-12-CD obrigado"
        assert normalize_plate(text) == "AB-12-CD"

    def test_strips_eu_country_band(self):
        # EasyOCR often picks up the EU/P side band as separate text;
        # if it merges it into the plate string, we still want to
        # recover the plate substring.
        assert normalize_plate("P AB-12-CD") == "AB-12-CD"

    def test_handles_none_input(self):
        # Robustness: caller may pass None when guarding on optional text.
        assert normalize_plate(None) is None  # type: ignore[arg-type]
