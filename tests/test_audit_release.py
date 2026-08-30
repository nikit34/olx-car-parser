"""Noticing that the release has rotted.

Assets disappear silently: ``--clobber`` deletes before it writes, so an
interrupted upload leaves nothing. Five artefacts had gone missing before
anyone looked, so the check has to be the thing that looks.
"""
from __future__ import annotations

import json

import pytest

from scripts import audit_release


def _fresh(names):
    now = audit_release.datetime.now(audit_release.timezone.utc)
    return {name: now for name in names}


@pytest.fixture
def complete(monkeypatch):
    assets = set(audit_release.expected())
    monkeypatch.setattr(audit_release, "asset_times", lambda: _fresh(assets))
    monkeypatch.setattr(audit_release, "incomplete_chunk_sets", lambda a: {})
    return assets


class TestDetection:
    def test_complete_release_passes(self, complete):
        assert audit_release.main([]) == 0

    def test_a_missing_asset_fails(self, complete, monkeypatch):
        assets = complete - {"models.json"}
        monkeypatch.setattr(audit_release, "asset_times", lambda: _fresh(assets))
        monkeypatch.setattr(audit_release, "local_copy", lambda n: None)
        assert audit_release.main([]) == 1

    def test_a_chunked_asset_counts_as_present(self, monkeypatch):
        """listings.parquet only ever exists as parts plus a manifest."""
        assets = {audit_release.manifest_name("listings.parquet")}
        assert audit_release.satisfied("listings.parquet", assets) is True

    def test_a_gzipped_asset_counts_as_present(self, monkeypatch):
        """valuations.json travels compressed; the plain name is gone from
        the release and healing it back would undo the change."""
        assert audit_release.satisfied("valuations.json", {"valuations.json.gz"}) is True

    def test_parts_missing_under_a_present_manifest_is_caught(self, monkeypatch):
        """A manifest with no parts behind it reads as a whole file to a
        naive check, and would ship a broken witness."""
        meta = json.dumps({"name": "listings.parquet", "digest": "abcd1234",
                           "size": 10, "sha256": "x", "parts": 3}).encode()
        monkeypatch.setattr(audit_release, "_get", lambda url, quiet=False: meta)
        broken = audit_release.incomplete_chunk_sets(
            {"listings.parquet.chunks.json", "listings.parquet.abcd1234.p00"})
        assert broken == {"listings.parquet": "2 of 3 parts missing"}


class TestHealing:
    def test_republishes_from_the_local_copy(self, complete, monkeypatch, tmp_path):
        assets = complete - {"models.json"}
        state = {"assets": assets}
        monkeypatch.setattr(audit_release, "asset_times",
                            lambda: _fresh(state["assets"]))
        local = tmp_path / "models.json"
        local.write_text("{}")
        monkeypatch.setattr(audit_release, "local_copy",
                            lambda n: local if n == "models.json" else None)

        def _publish(paths):
            state["assets"] = state["assets"] | {p.name for p in paths}
            return 0

        monkeypatch.setattr(audit_release, "publish", _publish)
        assert audit_release.main(["--heal"]) == 0

    def test_still_fails_when_there_is_nothing_to_restore_from(self, complete, monkeypatch):
        monkeypatch.setattr(audit_release, "asset_times",
                            lambda: _fresh(complete - {"models.json"}))
        monkeypatch.setattr(audit_release, "local_copy", lambda n: None)
        assert audit_release.main(["--heal"]) == 1


class TestFreshness:
    """Presence is not freshness: a witness the pipeline stopped refreshing
    passes a name check while the dashboard serves last week's market."""

    def _times(self, ages_h: dict[str, int]):
        now = audit_release.datetime.now(audit_release.timezone.utc)
        return {name: now - audit_release.timedelta(hours=h)
                for name, h in ages_h.items()}

    def test_recent_assets_are_not_flagged(self):
        assert audit_release.stale_assets(self._times({"models.json": 2})) == {}

    def test_an_old_asset_is_flagged_with_its_age(self):
        stale = audit_release.stale_assets(self._times({"models.json": 50}))
        assert stale == {"models.json": 50}

    def test_a_chunked_asset_is_judged_by_its_newest_piece(self):
        """Parts carry the timestamps; the name itself never appears."""
        times = self._times({"listings.parquet.abcd1234.p00": 90,
                             "listings.parquet.chunks.json": 3})
        assert "listings.parquet" not in audit_release.stale_assets(times)

    def test_absent_assets_are_left_to_the_missing_check(self):
        assert "models.json" not in audit_release.stale_assets(self._times({}))
