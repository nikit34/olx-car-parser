"""Tests for the dashboard's GitHub-release asset cache.

The cache holds the model/metrics artefacts and is gated by a TTL marker.
The 2026-05-03 bug had the TTL keyed on the DB's own mtime, so a 60 KB
stub written within the TTL window silently shadowed the real release for
two hours. These tests pin the marker-based replacement, plus the rule
that the SQLite DB is no longer a release asset at all.
"""
from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def patched_paths(tmp_path, monkeypatch):
    """Redirect all release-cache paths into a tmp dir so tests can't
    touch the dev machine's real artefacts."""
    from src.dashboard import data_loader as dl

    monkeypatch.setenv("OLX_DB_URL", "postgresql+psycopg://olx@localhost/olx_cars")
    marker = tmp_path / ".last_release_check"
    model = tmp_path / "price_model.joblib"
    metrics = tmp_path / "price_metrics.json"
    importance = tmp_path / "price_importance.json"

    monkeypatch.setattr(dl, "_RELEASE_CHECK_MARKER", marker)
    monkeypatch.setattr(dl, "_MODEL_PATH", model)
    monkeypatch.setattr(dl, "_METRICS_PATH", metrics)
    monkeypatch.setattr(dl, "_IMPORTANCE_PATH", importance)
    monkeypatch.setattr(
        dl, "_RELEASE_ASSETS",
        (
            ("price_model.joblib", model),
            ("price_metrics.json", metrics),
            ("price_importance.json", importance),
        ),
    )
    monkeypatch.setattr(dl, "_LAST_RELEASE_ERROR", None, raising=False)
    return {"dl": dl, "marker": marker}





class TestReleaseCacheTTL:
    def test_marker_within_ttl_skips_api(self, patched_paths):
        """Happy path: marker exists and is fresh → no API call."""
        dl = patched_paths["dl"]
        marker = patched_paths["marker"]
        marker.touch()  # fresh

        with patch.object(dl, "_list_release_assets") as mock_list:
            dl._ensure_release_assets()

        mock_list.assert_not_called()

    def test_stale_marker_triggers_api(self, patched_paths):
        """Marker older than TTL → API gets called even if DB exists."""
        dl = patched_paths["dl"]
        marker = patched_paths["marker"]
        marker.touch()
        # Push marker mtime 3 hours into the past (TTL is 2 h).
        old = time.time() - 3 * 3600
        import os
        os.utime(marker, (old, old))

        with patch.object(dl, "_list_release_assets") as mock_list:
            mock_list.return_value = {}
            dl._ensure_release_assets()

        mock_list.assert_called_once()

    def test_api_failure_does_not_stamp_marker(self, patched_paths):
        """If ``_list_release_assets`` returns None / empty (rate limit,
        network, missing release), the marker must NOT be stamped — we
        want the next call to retry immediately rather than wait out the
        TTL on a transient miss."""
        dl = patched_paths["dl"]
        marker = patched_paths["marker"]

        with patch.object(dl, "_list_release_assets") as mock_list:
            mock_list.return_value = None  # simulated API failure
            dl._ensure_release_assets()

        assert not marker.exists()

    def test_successful_sync_stamps_marker(self, patched_paths):
        """A successful API sync writes the marker so the next call
        within the TTL window short-circuits."""
        dl = patched_paths["dl"]
        marker = patched_paths["marker"]

        with patch.object(dl, "_list_release_assets") as mock_list, \
             patch.object(dl, "_asset_url_if_newer") as mock_newer, \
             patch.object(dl, "_download_asset") as mock_dl:
            mock_list.return_value = {
                "price_model.joblib": {"updated_at": "2026-05-03T00:00:00Z", "url": "x"},
            }
            mock_newer.return_value = None  # nothing actually needs downloading
            dl._ensure_release_assets()

        assert marker.exists()

    def test_force_next_check_removes_marker(self, patched_paths):
        """``_force_next_check`` is the manual override for "ignore the
        TTL, hit the API on the next call"."""
        dl = patched_paths["dl"]
        marker = patched_paths["marker"]
        marker.touch()
        assert marker.exists()

        dl._force_next_check()
        assert not marker.exists()


class TestCDNFallback:
    """Public CDN download path — fires whenever the GitHub API listing
    fails (rate limit, network) or returns no usable assets. It is no
    longer gated on the local DB: the DB is not a release asset, so the
    only thing the fallback can recover is the model/metrics set."""

    def test_cdn_called_when_api_returns_none_and_no_db(self, patched_paths):
        dl = patched_paths["dl"]
        with patch.object(dl, "_list_release_assets", return_value=None), \
             patch.object(dl, "_download_asset") as mock_dl:
            dl._ensure_release_assets()
        # 3 assets, all routed through CDN URL
        assert mock_dl.call_count == 3
        first_url = mock_dl.call_args_list[0].args[0]
        assert "github.com/nikit34/olx-car-parser/releases/download/latest-data" in first_url

    def test_cdn_called_when_api_returns_empty_assets_dict(self, patched_paths):
        """Edge case: release exists but has no assets attached. Old
        ``if assets:`` falsy-empty-dict check skipped both the asset
        loop AND the CDN fallback. Now the CDN path catches it."""
        dl = patched_paths["dl"]
        with patch.object(dl, "_list_release_assets", return_value={}), \
             patch.object(dl, "_download_asset") as mock_dl:
            dl._ensure_release_assets()
        assert mock_dl.call_count == 3

    def test_successful_cdn_clears_prior_api_error(self, patched_paths):
        """User-facing UX: if the API failed but the CDN recovered,
        the empty-state banner shouldn't keep showing the stale API
        error. ``_LAST_RELEASE_ERROR`` is cleared on CDN success."""
        dl = patched_paths["dl"]

        def _fake_dl(url, dest):
            dest.write_bytes(b"artifact")
            return True

        # Pre-populate the error as if a prior API call had failed.
        dl._LAST_RELEASE_ERROR = "GitHub API returned HTTP 403 (rate-limited)"
        with patch.object(dl, "_list_release_assets", return_value=None), \
             patch.object(dl, "_download_asset", side_effect=_fake_dl):
            dl._ensure_release_assets()
        assert dl.get_last_release_error() is None


class TestDBNotPublished:
    """The database is not a release asset (2026-08-28): it was 370 MB,
    republished 6x/day, and nothing in production read it."""

    def test_db_is_not_a_release_asset(self):
        from src.dashboard import data_loader as dl

        assert "olx_cars.db" not in {name for name, _ in dl._RELEASE_ASSETS}

    def test_unconfigured_engine_explains_itself(self, patched_paths, monkeypatch):
        dl = patched_paths["dl"]
        monkeypatch.delenv("OLX_DB_URL", raising=False)
        with patch.object(dl, "_list_release_assets", return_value={}), \
             patch.object(dl, "_download_asset", return_value=True):
            ok = dl._ensure_release_assets()
        assert ok is False
        assert "OLX_DB_URL" in (dl.get_last_release_error() or "")
