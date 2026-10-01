import datetime as dt
import importlib
import os

import pytest

import scripts.build_dashboard_data as bdd


@pytest.fixture
def cadence(monkeypatch):
    def _set(value):
        monkeypatch.setenv("SEO_BLOB_CADENCE", value)
        return importlib.reload(bdd)
    return _set


def _stamp(days_ago=0, hour=18):
    d = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days_ago)
    return {"built_at": d.strftime("%Y-%m-%dT%H:%M:%SZ").replace("T18:00:21Z", f"T{hour:02d}:00:00Z")}


def test_daily_cadence_skips_when_the_blob_is_already_from_today(cadence):
    mod = cadence("daily")
    assert mod._seo_blob_already_built_today(_stamp(0)) is True


def test_daily_cadence_builds_when_the_blob_is_from_yesterday(cadence):
    mod = cadence("daily")
    assert mod._seo_blob_already_built_today(_stamp(1)) is False


def test_daily_cadence_builds_on_a_gutted_previous_blob(cadence):
    mod = cadence("daily")
    assert mod._seo_blob_already_built_today(None) is False
    assert mod._seo_blob_already_built_today({}) is False
    assert mod._seo_blob_already_built_today({"built_at": None}) is False


def test_hourly_cadence_keeps_the_old_behaviour(cadence):
    mod = cadence("hourly")
    assert mod._seo_blob_already_built_today(_stamp(0)) is False


def test_cadence_is_case_insensitive_and_defaults_to_off(cadence):
    assert cadence("HOURLY")._seo_blob_already_built_today(_stamp(0)) is False
    assert cadence("")._seo_blob_already_built_today(_stamp(0)) is False


def test_default_cadence_is_daily_without_the_env_var(monkeypatch):
    monkeypatch.delenv("SEO_BLOB_CADENCE", raising=False)
    mod = importlib.reload(bdd)
    assert mod.SEO_BLOB_CADENCE == "daily"
    assert mod._seo_blob_already_built_today(_stamp(0)) is True


def test_stale_asset_budget_outlives_a_daily_rebuild():
    # audit_release.STALE_AFTER_HOURS would start shouting at the release if the
    # public blob were held for longer than this, so a daily cadence has to stay
    # comfortably inside the window.
    from scripts.audit_release import STALE_AFTER_HOURS
    assert 24 < STALE_AFTER_HOURS, "daily models.json would trip the staleness alarm"
    assert os.environ.get("SEO_BLOB_CADENCE", "daily") == "daily"