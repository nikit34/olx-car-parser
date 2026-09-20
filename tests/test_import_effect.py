"""The import flag stops being an assumption and becomes a measurement.

Two things are pinned here. First the three-way flag: an advert that says the
car is still foreign-plated is *pending* and keeps every ISV warning, while an
advert that merely says the car was imported is not — on 2026-09-20 that was
289 of 19,964 flagged cars, and treating the other 19,675 as tax-owing is what
these tests exist to prevent.

Second the effect itself: a gap is only reported when it survives matching
inside platform, brand, model and age bucket, and ``publishable`` is False
whenever the bootstrap interval still contains zero.
"""

import numpy as np
import pandas as pd

from src.analytics.import_effect import import_arm, import_effect
from src.analytics.valuations import _import_flags


class TestImportFlags:
    def test_plain_import_is_not_pending(self):
        assert _import_flags("BMW 320d importado da Alemanha", "carro impecável") == (1, 0, 0)

    def test_advert_saying_foreign_plate_is_pending(self):
        assert _import_flags("BMW 320d", "ainda por legalizar")[2] == 1
        assert _import_flags("Golf", "matricula alema, vou legalizar")[2] == 1
        assert _import_flags("Audi A4", "matrícula estrangeira")[2] == 1

    def test_legalised_beats_pending(self):
        imp, leg, pend = _import_flags("Mercedes importado", "já legalizado, ISV pago")
        assert (imp, leg, pend) == (1, 1, 0)

    def test_structured_origin_alone_never_implies_a_tax_owed(self):
        assert _import_flags("Clio 1.5 dCi", "bom estado", "imported") == (1, 0, 0)

    def test_national_origin_clears_a_text_false_positive(self):
        assert _import_flags("Golf importado", "", "national") == (0, 0, 0)


class TestImportArm:
    def test_reads_structured_field_and_text(self):
        arm = import_arm(pd.DataFrame([
            {"title": "A", "description": "", "origin": "imported"},
            {"title": "B", "description": "", "origin": "national"},
            {"title": "C importado", "description": "", "origin": None},
            {"title": "D", "description": "matricula portuguesa", "origin": None},
            {"title": "E", "description": "", "origin": None},
        ]))
        assert list(arm) == ["imported", "national", "imported", "national", None]

    def test_survives_missing_columns_and_nan(self):
        arm = import_arm(pd.DataFrame([{"title": "X", "description": np.nan}]))
        assert list(arm) == [None]


def _corpus(gap_pp: float, cells: int = 40, per_arm: int = 10, seed: int = 7):
    """A corpus where imports sell `gap_pp` slower, everything else equal."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2026-01-01", tz="UTC")
    listings, outcomes = [], []
    for cell in range(cells):
        for arm in ("imported", "national"):
            share = 0.5 + (gap_pp / 100 if arm == "imported" else 0.0)
            for i in range(per_arm):
                oid = f"{cell}-{arm}-{i}"
                listings.append({
                    "olx_id": oid, "source": "olx", "mileage_km": 100_000,
                    "title": "carro importado" if arm == "imported" else "carro",
                    "description": "matricula portuguesa" if arm == "national" else "",
                    "origin": None, "last_seen_at": start + pd.Timedelta(days=200),
                })
                gone = rng.random() < share
                outcomes.append({
                    "car_id": oid, "brand": f"B{cell}", "model": "M", "year": 2015,
                    "first_seen_at": start, "days": 10.0 if gone else 90.0,
                    "still_active": False, "cut": False, "last_ask": 10_000.0,
                })
    return pd.DataFrame(listings), pd.DataFrame(outcomes)


class TestImportEffect:
    def test_finds_a_real_gap_and_calls_it_publishable(self):
        listings, outcomes = _corpus(gap_pp=-25.0)
        got = import_effect(listings, outcomes)
        assert got is not None
        assert got["s30"]["publishable"] is True
        assert got["s30"]["v"] < -10
        assert got["s30"]["hi"] < 0

    def test_refuses_to_publish_a_gap_that_is_not_there(self):
        listings, outcomes = _corpus(gap_pp=0.0)
        got = import_effect(listings, outcomes)
        assert got["s30"]["publishable"] is False
        assert got["s30"]["lo"] < 0 < got["s30"]["hi"]

    def test_none_when_no_cell_holds_both_arms(self):
        listings, outcomes = _corpus(gap_pp=-25.0, cells=2)
        assert import_effect(listings, outcomes) is None

    def test_price_gap_needs_predictions(self):
        listings, outcomes = _corpus(gap_pp=-25.0)
        assert "price" not in import_effect(listings, outcomes)
        preds = {r.car_id: 11_000.0 for r in outcomes.itertuples()}
        got = import_effect(listings, outcomes, preds)
        assert got["price"]["v"] == 0.0
        assert got["price"]["publishable"] is False

    def test_platform_is_part_of_the_cell(self):
        listings, outcomes = _corpus(gap_pp=-25.0)
        listings["source"] = ["olx" if i % 2 == 0 else "standvirtual"
                              for i in range(len(listings))]
        assert import_effect(listings, outcomes) is None
