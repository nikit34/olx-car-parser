"""Tests for the dashboard's GitHub-release asset cache.

The cache lives at ``data/olx_cars.db`` (plus model/metrics siblings) and
is gated by a TTL marker. The 2026-05-03 bug had the TTL keyed on the
DB's own mtime, so a 60 KB stub written within the TTL window silently
shadowed the real release for two hours. These tests pin the marker-
based replacement.
"""
from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def patched_paths(tmp_path, monkeypatch):
    """Redirect all release-cache paths into a tmp dir so tests can't
    touch the dev machine's real ``data/olx_cars.db``."""
    from src.dashboard import data_loader as dl

    db = tmp_path / "olx_cars.db"
    marker = tmp_path / ".last_release_check"
    model = tmp_path / "price_model.joblib"
    metrics = tmp_path / "price_metrics.json"
    importance = tmp_path / "price_importance.json"

    monkeypatch.setattr(dl, "DB_PATH", db)
    monkeypatch.setattr(dl, "_RELEASE_CHECK_MARKER", marker)
    monkeypatch.setattr(dl, "_MODEL_PATH", model)
    monkeypatch.setattr(dl, "_METRICS_PATH", metrics)
    monkeypatch.setattr(dl, "_IMPORTANCE_PATH", importance)
    monkeypatch.setattr(
        dl, "_RELEASE_ASSETS",
        (
            ("olx_cars.db", db),
            ("price_model.joblib", model),
            ("price_metrics.json", metrics),
            ("price_importance.json", importance),
        ),
    )
    monkeypatch.setattr(dl, "_LAST_RELEASE_ERROR", None, raising=False)
    return {"dl": dl, "db": db, "marker": marker}


def _write_real_db(path):
    """Write a 1.1 MB blob — passes the ``_looks_like_real_db`` gate."""
    path.write_bytes(b"\x00" * 1_100_000)


def _write_stub(path):
    """Write a 60 KB blob — fails ``_looks_like_real_db``, mimicking the
    actual stub that triggered the 2026-05-02 incident."""
    path.write_bytes(b"\x00" * 60_000)


class TestReleaseCacheTTL:
    def test_stub_db_does_not_shadow_real_release(self, patched_paths):
        """The bug we're pinning: a tiny stub DB with a recent mtime must
        NOT block the API check. Pre-fix, this scenario silently returned
        True for two hours and left the dashboard pointing at an empty DB.
        """
        dl = patched_paths["dl"]
        db = patched_paths["db"]
        marker = patched_paths["marker"]

        # Set up: stub DB exists with a current mtime, NO marker file.
        db.write_bytes(b"\x00" * 1024)
        assert not marker.exists()

        with patch.object(dl, "_list_release_assets") as mock_list, \
             patch.object(dl, "_download_asset") as mock_dl:
            mock_list.return_value = {}  # no assets, but the call DID happen
            dl._ensure_release_assets()

        mock_list.assert_called_once()  # TTL did NOT skip the API call

    def test_marker_within_ttl_skips_api(self, patched_paths):
        """Happy path: marker exists and is fresh → no API call."""
        dl = patched_paths["dl"]
        marker = patched_paths["marker"]
        patched_paths["db"].write_bytes(b"\x00" * 100)
        marker.touch()  # fresh

        with patch.object(dl, "_list_release_assets") as mock_list:
            dl._ensure_release_assets()

        mock_list.assert_not_called()

    def test_stale_marker_triggers_api(self, patched_paths):
        """Marker older than TTL → API gets called even if DB exists."""
        dl = patched_paths["dl"]
        marker = patched_paths["marker"]
        patched_paths["db"].write_bytes(b"\x00" * 100)
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
        db = patched_paths["db"]
        marker = patched_paths["marker"]

        with patch.object(dl, "_list_release_assets") as mock_list, \
             patch.object(dl, "_asset_url_if_newer") as mock_newer, \
             patch.object(dl, "_download_asset") as mock_dl:
            mock_list.return_value = {
                "olx_cars.db": {"updated_at": "2026-05-03T00:00:00Z", "url": "x"},
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
    """Public CDN download path — fires when the GitHub API listing
    fails (rate limit, network) or returns no usable assets, AND the
    local DB is missing or a stub. Exercises the 2026-05-03 incident
    where a 60 KB stub fooled the 'DB exists' check and prevented the
    CDN fallback from running."""

    def test_cdn_called_when_api_returns_none_and_no_db(self, patched_paths):
        dl = patched_paths["dl"]
        with patch.object(dl, "_list_release_assets", return_value=None), \
             patch.object(dl, "_download_asset") as mock_dl:
            dl._ensure_release_assets()
        # 4 assets, all routed through CDN URL
        assert mock_dl.call_count == 4
        first_url = mock_dl.call_args_list[0].args[0]
        assert "github.com/nikit34/olx-car-parser/releases/download/latest-data" in first_url

    def test_cdn_called_when_local_db_is_a_stub(self, patched_paths):
        """The 60 KB stub case: API failed, but the dashboard had a
        prior failed-download remnant on disk. Pre-fix this skipped
        CDN download because ``DB_PATH.exists()`` returned True."""
        dl = patched_paths["dl"]
        _write_stub(patched_paths["db"])
        with patch.object(dl, "_list_release_assets", return_value=None), \
             patch.object(dl, "_download_asset") as mock_dl:
            dl._ensure_release_assets()
        assert mock_dl.call_count == 4

    def test_cdn_skipped_when_local_db_is_real(self, patched_paths):
        """If the local DB looks legit (>1 MB), API failure is fine —
        we serve the cache, no need to re-download."""
        dl = patched_paths["dl"]
        _write_real_db(patched_paths["db"])
        with patch.object(dl, "_list_release_assets", return_value=None), \
             patch.object(dl, "_download_asset") as mock_dl:
            ok = dl._ensure_release_assets()
        mock_dl.assert_not_called()
        assert ok is True

    def test_cdn_called_when_api_returns_empty_assets_dict(self, patched_paths):
        """Edge case: release exists but has no assets attached. Old
        ``if assets:`` falsy-empty-dict check skipped both the asset
        loop AND the CDN fallback. Now the CDN path catches it."""
        dl = patched_paths["dl"]
        with patch.object(dl, "_list_release_assets", return_value={}), \
             patch.object(dl, "_download_asset") as mock_dl:
            dl._ensure_release_assets()
        assert mock_dl.call_count == 4

    def test_successful_cdn_clears_prior_api_error(self, patched_paths):
        """User-facing UX: if the API failed but the CDN recovered,
        the empty-state banner shouldn't keep showing the stale API
        error. ``_LAST_RELEASE_ERROR`` is cleared on CDN success."""
        dl = patched_paths["dl"]

        def _fake_dl(url, dest):
            _write_real_db(dest)
            return True

        # Pre-populate the error as if a prior API call had failed.
        dl._LAST_RELEASE_ERROR = "GitHub API returned HTTP 403 (rate-limited)"
        with patch.object(dl, "_list_release_assets", return_value=None), \
             patch.object(dl, "_download_asset", side_effect=_fake_dl):
            dl._ensure_release_assets()
        assert dl.get_last_release_error() is None
