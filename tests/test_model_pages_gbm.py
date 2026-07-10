"""GBM fair-value enrichment of the per-model SEO blob (model_pages).

Covers the display guards (_gbm_passes), the vocab gate (_configs_vocab_ok),
and build_model_pages wiring a stub valuator — WITHOUT loading LightGBM, so it
runs in <1s (the real model is validated separately in CI / build).
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.analytics import model_pages as mp
from src.analytics.price_model import _configs_vocab_ok


# ── _gbm_passes: the cheap-tail / ceiling / agreement guards ─────────────────
def test_gbm_passes_typical_mid_price_kept():
    # asking median 10k (P25 9k, P75 11k), GBM 9k → within all guards.
    assert mp._gbm_passes(9000, 1.0, True, ask=10000, ap25=9000, ap75=11000)


def test_gbm_passes_cheap_tail_suppressed():
    # asking below €5k → model over-predicts the cheap tail → suppress.
    assert not mp._gbm_passes(3800, 1.0, True, ask=4000, ap25=3000, ap75=4800)


def test_gbm_passes_high_end_ceiling_suppressed():
    # asking above €45k → model saturates (ceiling artifact) → suppress.
    assert not mp._gbm_passes(59900, 1.0, True, ask=84000, ap25=55000, ap75=105000)


def test_gbm_passes_wild_disagreement_suppressed():
    # GBM 0.52× asking (e.g. exotic mixed-generation group) → suppress.
    assert not mp._gbm_passes(59900, 1.0, True, ask=114800, ap25=76000, ap75=149000)


def test_gbm_passes_outside_asking_iqr_suppressed():
    # ratio ok (0.8) but the estimate sits below P25×0.85 → inconsistent → drop.
    assert not mp._gbm_passes(6000, 1.0, True, ask=15000, ap25=12000, ap75=18000)


def test_gbm_passes_requires_vocab_and_spec_fill():
    assert not mp._gbm_passes(9000, 1.0, False, ask=10000, ap25=9000, ap75=11000)
    assert not mp._gbm_passes(9000, 0.25, True, ask=10000, ap25=9000, ap75=11000)


# ── _configs_vocab_ok: brand+model must be in the trained vocab ──────────────
def test_configs_vocab_ok_gates_unknown_brand_or_model():
    cat_maps = {"brand": {"Opel": 0, "BMW": 1, "__other__": 2},
                "model": {"Corsa": 0, "Astra": 1, "__other__": 2}}
    df = pd.DataFrame({"brand": ["Opel", "Opel", "Ferrari"],
                       "model": ["Corsa", "Zafira", "Corsa"]})
    ok = _configs_vocab_ok(df, cat_maps)
    assert list(ok) == [True, False, False]  # known / unknown model / unknown brand


def test_cell_year_parses_int_and_band():
    assert mp._cell_year(2018) == 2018
    assert mp._cell_year("2012-2014") == 2014   # band → latest year
    assert mp._cell_year("bogus") is None


# ── build_model_pages with a stub valuator ───────────────────────────────────
def _listings(brand, model, price, n=25, **cols):
    base = dict(is_active=True, brand=brand, model=model, price_eur=price,
                year=2018, mileage_km=90000, engine_cc=1500, horsepower=110,
                seats=5, fuel_type="Gasolina", transmission="Manual",
                generation="A", segment="B", sub_model="X", trim_level="T")
    base.update(cols)
    return pd.DataFrame([base] * n)


def _stub_valuator(per_brand):
    """Return a valuator that predicts a fixed price per brand (no LightGBM)."""
    def _v(cfg):
        pred = cfg["brand"].map(per_brand).astype(float)
        return pd.DataFrame({
            "predicted_price": pred,
            "fair_price_low": pred * 0.85,
            "fair_price_high": pred * 1.2,
            "spec_fill": 1.0,
            "vocab_ok": True,
        }, index=cfg.index)
    return _v


def test_build_model_pages_attaches_gbm_only_when_guards_pass():
    # Mid-price model (asking ~10k) → GBM kept; cheap model (~2k) → suppressed.
    listings = pd.concat([
        _listings("Mid", "Alpha", 10000),
        _listings("Cheapo", "Beta", 2000),
    ], ignore_index=True)
    valuator = _stub_valuator({"Mid": 9000.0, "Cheapo": 1800.0})
    out = mp.build_model_pages(listings, sell_speed=None, valuator=valuator)
    models = out["models"]
    mid = models[mp.slugify("Mid-Alpha")]
    cheap = models[mp.slugify("Cheapo-Beta")]
    assert mid["gm"] == 9000 and mid["gl"] < mid["gm"] < mid["gh"]  # kept
    assert "gm" not in cheap                                        # cheap-tail suppressed


def test_gbm_is_per_listing_median_not_shared_archetype():
    # Two models valued by a MILEAGE-sensitive model must get DIFFERENT gm —
    # proves we value each page's real listings and take the median, rather than
    # collapsing onto one modal archetype (the de-quantization fix).
    v40 = _listings("Volvo", "V40", 16000, mileage_km=60000)
    v60 = _listings("Volvo", "V60", 11500, mileage_km=180000)
    listings = pd.concat([v40, v60], ignore_index=True)

    def _v(cfg):                       # fair value falls €0.05 per km, per row
        km = pd.to_numeric(cfg["mileage_km"], errors="coerce").fillna(1e5)
        pred = (20000 - km * 0.05).clip(lower=1)      # 60k→17000, 180k→11000
        return pd.DataFrame({"predicted_price": pred, "fair_price_low": pred * 0.9,
                             "fair_price_high": pred * 1.1, "spec_fill": 1.0,
                             "vocab_ok": True}, index=cfg.index)

    out = mp.build_model_pages(listings, valuator=_v)
    assert out["models"][mp.slugify("Volvo-V40")]["gm"] == 17000
    assert out["models"][mp.slugify("Volvo-V60")]["gm"] == 11000  # distinct, per-listing


def test_build_model_pages_without_valuator_is_asking_only():
    listings = _listings("Mid", "Alpha", 10000)
    out = mp.build_model_pages(listings, sell_speed=None)  # no valuator
    rec = out["models"][mp.slugify("Mid-Alpha")]
    assert "fm" in rec and "gm" not in rec  # unchanged legacy behaviour


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
