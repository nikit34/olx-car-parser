"""Text scans precomputed at build time instead of shipped as prose.

The witness dropped the ``description`` column (57% of the file, over
Cloudflare's 25 MiB per-asset cap) and ships four scan-result columns instead.
These tests pin the contract that makes that safe: a frame WITH the columns and
a frame WITH the prose must reach the same verdict, and the columns must win
when both are present — otherwise a stale scan would be silently re-derived.
"""

import pandas as pd
import pytest

from src.analytics.text_signals import (
    TEXT_SIGNAL_COLUMNS,
    add_text_signals,
    hard_block_phrase,
)
from src.analytics.condition_signal import fault_cost_from_flag, minor_fault_cost
from src.dashboard.data_loader import _blocking_deal_reason


class TestAddTextSignals:
    def test_adds_all_four_columns(self):
        df = add_text_signals(pd.DataFrame([
            {"title": "Golf 1.6 TDI", "description": "bom estado, sempre na marca"},
        ]))
        for col in TEXT_SIGNAL_COLUMNS:
            assert col in df.columns
        row = df.iloc[0]
        assert row["text_import_flag"] == 0
        # "no match" is None on pandas 2 and NaN on pandas 3 (PDEP-14 gives
        # the column a str dtype), so every consumer tests the value rather
        # than its identity — and so does every assertion here.
        assert pd.isna(row["text_minor_fault"])
        assert pd.isna(row["text_hard_block_phrase"])

    def test_resolves_each_scan(self):
        df = add_text_signals(pd.DataFrame([
            {"title": "BMW 320d importado da Alemanha",
             "description": "ainda por legalizar", "origin": None},
            {"title": "Passat", "description": "luz da injeção acesa"},
            {"title": "Punto", "description": "Vendo carro. Não pega, só de reboque."},
        ]))
        assert df.loc[0, "text_import_flag"] == 1
        assert df.loc[0, "text_import_legalised"] == 0
        assert not pd.isna(df.loc[1, "text_minor_fault"])
        assert "não pega" in df.loc[2, "text_hard_block_phrase"]

    def test_structured_origin_feeds_the_import_flag(self):
        """``origin`` reinforces the text scan — an imported car with a silent
        description still flags, a national one clears a text false-positive."""
        df = add_text_signals(pd.DataFrame([
            {"title": "Audi A4", "description": "carro impecável", "origin": "imported"},
            {"title": "Audi A4 importado", "description": "", "origin": "national"},
        ]))
        assert df.loc[0, "text_import_flag"] == 1
        assert df.loc[1, "text_import_flag"] == 0

    def test_survives_missing_description_and_nan(self):
        """Frames without the prose (or with pandas NaN in it) must still get
        the columns — absent columns mean "not scanned" to every consumer, so
        silently skipping would turn into a silently lost scan."""
        df = add_text_signals(pd.DataFrame([
            {"title": "Golf para peças"},
            {"title": None},
        ]))
        assert df.loc[0, "text_hard_block_phrase"] == "para peças"
        assert pd.isna(df.loc[1, "text_hard_block_phrase"])

    def test_no_op_without_title_column(self):
        df = add_text_signals(pd.DataFrame([{"olx_id": "x1"}]))
        assert "text_import_flag" not in df.columns

    def test_empty_frame(self):
        df = add_text_signals(pd.DataFrame(columns=["title", "description"]))
        assert list(TEXT_SIGNAL_COLUMNS) == [c for c in TEXT_SIGNAL_COLUMNS if c in df.columns]
        assert len(df) == 0


class TestHardBlockPhrase:
    @pytest.mark.parametrize("text", [
        "Vendo carro. Não pega, só de reboque.",
        "Bom carro mas a junta queimada, vende-se barato.",
        "completo para peças",
    ])
    def test_matches_salvage_phrasings(self, text):
        assert hard_block_phrase("Golf", text) is not None

    def test_clean_text_and_missing_values(self):
        assert hard_block_phrase("Golf 1.6", "bom estado, sempre na marca") is None
        assert hard_block_phrase(None, None) is None
        assert hard_block_phrase(float("nan"), float("nan")) is None


class TestBlockingReasonReadsTheColumn:
    """``_blocking_deal_reason`` is the browser-side consumer."""

    def _row(self, **kw) -> pd.Series:
        base = {"desc_mentions_accident": False, "damage_severity": 0,
                "right_hand_drive": False, "llm_extras": None, "title": "Golf"}
        base.update(kw)
        return pd.Series(base)

    def test_witness_row_blocks_without_prose(self):
        row = self._row(text_hard_block_phrase="não pega")
        assert "não pega" in _blocking_deal_reason(row)

    def test_witness_row_with_empty_column_does_not_block(self):
        """Column present and empty means "scanned, found nothing" — it must
        not fall back to re-scanning, or a witness would block on text the
        build already cleared."""
        row = self._row(text_hard_block_phrase=None,
                        description="Vendo carro. Não pega, só de reboque.")
        assert _blocking_deal_reason(row) is None

    def test_server_row_still_scans_inline(self):
        row = self._row(description="Vendo carro. Não pega, só de reboque.")
        assert "não pega" in _blocking_deal_reason(row)

    def test_precomputed_and_inline_agree(self):
        text = "Bom carro mas a junta queimada, vende-se barato."
        inline = _blocking_deal_reason(self._row(description=text))
        precomputed = _blocking_deal_reason(
            self._row(text_hard_block_phrase=hard_block_phrase("Golf", text)))
        assert inline == precomputed


class TestFaultCostSplit:
    """Detection is precomputed; only the €-sizing needs the asking price."""

    def test_flag_path_matches_text_path(self):
        text = "luz da injeção acesa, catalisador a precisar"
        from_text = minor_fault_cost("Golf", text, 5000)
        from_flag = fault_cost_from_flag(from_text[1], 5000)
        assert from_text == from_flag

    def test_no_flag_is_free(self):
        assert fault_cost_from_flag(None, 5000) == (0.0, None)
        assert fault_cost_from_flag("", 5000) == (0.0, None)
