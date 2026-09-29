import importlib.util
import time
from pathlib import Path

import pandas as pd

_spec = importlib.util.spec_from_file_location(
    "bdd_one_per_car", Path(__file__).resolve().parent.parent / "scripts" / "build_dashboard_data.py")
bdd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bdd)

FRAME = pd.DataFrame({"olx_id": ["a", "b", "c"], "duplicate_of": [None, "a", None]})


def _on(monkeypatch, day):
    monkeypatch.setattr(time, "strftime", lambda fmt, *a: day)


def test_before_the_switch_every_advert_counts(monkeypatch):
    _on(monkeypatch, "2026-10-03")
    assert list(bdd._one_per_car(FRAME)["olx_id"]) == ["a", "b", "c"]


def test_from_the_switch_a_car_on_both_sites_counts_once(monkeypatch):
    _on(monkeypatch, bdd.DEDUP_STATS_FROM)
    assert list(bdd._one_per_car(FRAME)["olx_id"]) == ["a", "c"]


def test_a_frame_without_the_mark_is_left_alone(monkeypatch):
    _on(monkeypatch, "2026-12-01")
    frame = FRAME.drop(columns=["duplicate_of"])
    assert len(bdd._one_per_car(frame)) == 3
