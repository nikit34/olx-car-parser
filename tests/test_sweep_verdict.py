"""The verdict rule behind the Sensitivity Monitor's exit code.

Load-bearing: `--all` exits non-zero on any `REAL ✓`, so this function alone
decides whether the weekly monitor goes red and a human gets pulled in.
"""
from __future__ import annotations

import pytest

from scripts.sweep_constant import _MIN_EFFECT, _verdict


class TestEffectFloor:
    def test_the_june_retune_would_still_raise_the_alarm(self):
        assert _verdict(-0.49, -0.17) == "REAL ✓"

    def test_an_effect_that_washed_out_on_the_holdout_does_not(self):
        assert _verdict(-0.10, -0.03).startswith("tiny")

    def test_an_interval_covering_zero_is_noise(self):
        assert _verdict(-0.02, 0.07) == "noise (CI∋0)"

    @pytest.mark.parametrize("lo,hi", [(-0.30, -0.11), (-1.0, -0.101)])
    def test_anything_past_the_floor_is_real(self, lo, hi):
        assert _verdict(lo, hi) == "REAL ✓"

    @pytest.mark.parametrize("lo,hi", [(-0.20, -0.10), (-0.11, -0.099)])
    def test_the_floor_itself_is_not_past_it(self, lo, hi):
        assert _verdict(lo, hi).startswith("tiny")

    def test_a_clearly_worse_value_is_never_reported_as_better(self):
        assert _verdict(0.20, 0.40) == "REAL ✗(worse)"
        assert "✓" not in _verdict(0.09, 0.26)

    def test_the_floor_is_symmetric(self):
        assert _verdict(-0.30, -0.11) == "REAL ✓"
        assert _verdict(0.11, 0.30) == "REAL ✗(worse)"

    def test_the_floor_is_the_documented_one(self):
        assert _MIN_EFFECT == pytest.approx(0.10)
