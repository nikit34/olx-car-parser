"""Splitting a witness across release assets.

The scrape host cannot push more than a couple of megabytes to GitHub in
one request, so large witnesses travel in pieces. The risk the tests pin
is a silently corrupt reassembly: a half-finished upload must never be
readable as a whole file.
"""
from __future__ import annotations

import hashlib
import json

import pytest

from scripts import release_chunks


@pytest.fixture
def blob(tmp_path):
    path = tmp_path / "listings.parquet"
    payload = bytes(range(256)) * 20_000
    path.write_bytes(payload)
    return path, payload


class TestUpload:
    def test_writes_parts_then_the_manifest_last(self, blob, monkeypatch):
        """The manifest is the commit point — parts alone must not be
        readable, or a failed run would look complete."""
        path, payload = blob
        order = []
        monkeypatch.setattr(release_chunks, "_upload",
                            lambda p: (order.append(p.name), True)[1])
        monkeypatch.setattr(release_chunks, "_release_asset_names", lambda: set())

        assert release_chunks.upload(path) is True
        assert order[-1] == "listings.parquet.chunks.json"
        assert len(order) - 1 == -(-len(payload) // release_chunks.CHUNK_BYTES)

    def test_a_failed_part_stops_before_the_manifest(self, blob, monkeypatch):
        path, _ = blob
        uploaded = []

        def _fail_second(p):
            uploaded.append(p.name)
            return len(uploaded) < 2

        monkeypatch.setattr(release_chunks, "_upload", _fail_second)
        monkeypatch.setattr(release_chunks, "_release_asset_names", lambda: set())

        assert release_chunks.upload(path) is False
        assert not any(n.endswith("chunks.json") for n in uploaded)

    def test_stale_parts_of_a_previous_digest_are_removed(self, blob, monkeypatch):
        path, _ = blob
        monkeypatch.setattr(release_chunks, "_upload", lambda p: True)
        monkeypatch.setattr(release_chunks, "_release_asset_names",
                            lambda: {"listings.parquet.deadbeef.p00",
                                     "listings.parquet.chunks.json",
                                     "snapshots.parquet.cafe0000.p00"})
        deleted = []

        def _record(*args):
            if args[:2] == ("release", "delete-asset"):
                deleted.append(args[3])
            return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()

        monkeypatch.setattr(release_chunks, "_gh", _record)

        assert release_chunks.upload(path) is True
        assert deleted == ["listings.parquet.deadbeef.p00"]

    def test_removes_the_whole_asset_it_replaces(self, blob, monkeypatch):
        """``fetch`` prefers a whole asset, so a leftover one would shadow
        the parts forever and serve yesterday's data."""
        path, _ = blob
        monkeypatch.setattr(release_chunks, "_upload", lambda p: True)
        monkeypatch.setattr(release_chunks, "_release_asset_names",
                            lambda: {"listings.parquet"})
        deleted = []
        monkeypatch.setattr(release_chunks, "_gh", lambda *a: (
            deleted.append(a[3]) if a[:2] == ("release", "delete-asset") else None,
            type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})())[1])

        assert release_chunks.upload(path) is True
        assert deleted == ["listings.parquet"]

    def test_never_deletes_its_own_manifest(self, blob, monkeypatch):
        """``.p`` also occurs inside ``.parquet``; a loose filter would
        delete the manifest that had just been written."""
        path, _ = blob
        monkeypatch.setattr(release_chunks, "_upload", lambda p: True)
        monkeypatch.setattr(release_chunks, "_release_asset_names",
                            lambda: {"listings.parquet.chunks.json",
                                     "listings.parquet"})
        deleted = []
        monkeypatch.setattr(release_chunks, "_gh", lambda *a: (
            deleted.append(a[3]) if a[:2] == ("release", "delete-asset") else None,
            type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})())[1])

        assert release_chunks.upload(path) is True
        assert "listings.parquet.chunks.json" not in deleted


class TestFetch:
    def _serve(self, monkeypatch, assets: dict[str, bytes]):
        monkeypatch.setattr(release_chunks, "_get",
                            lambda url, timeout=60, quiet=False: assets.get(url.rsplit("/", 1)[-1]))

    def test_whole_asset_wins_when_present(self, monkeypatch):
        self._serve(monkeypatch, {"small.json": b"{}"})
        assert release_chunks.fetch("small.json") == b"{}"

    def test_reassembles_from_parts(self, monkeypatch, blob):
        _, payload = blob
        sha = hashlib.sha256(payload).hexdigest()
        assets = {"listings.parquet.chunks.json": json.dumps({
            "name": "listings.parquet", "sha256": sha, "size": len(payload),
            "parts": 3, "digest": sha[:8]}).encode()}
        step = -(-len(payload) // 3)
        for i in range(3):
            assets[f"listings.parquet.{sha[:8]}.p{i:02d}"] = payload[i * step:(i + 1) * step]
        self._serve(monkeypatch, assets)

        assert release_chunks.fetch("listings.parquet") == payload

    def test_missing_part_yields_nothing(self, monkeypatch, blob):
        _, payload = blob
        sha = hashlib.sha256(payload).hexdigest()
        self._serve(monkeypatch, {"listings.parquet.chunks.json": json.dumps({
            "name": "listings.parquet", "sha256": sha, "size": len(payload),
            "parts": 2, "digest": sha[:8]}).encode()})

        assert release_chunks.fetch("listings.parquet") is None

    def test_digest_mismatch_yields_nothing(self, monkeypatch):
        """Truncated or mixed-generation parts must not reach the dashboard."""
        payload = b"x" * 3000
        assets = {"listings.parquet.chunks.json": json.dumps({
            "name": "listings.parquet", "sha256": hashlib.sha256(b"other").hexdigest(),
            "size": len(payload), "parts": 1, "digest": "aaaaaaaa"}).encode(),
            "listings.parquet.aaaaaaaa.p00": payload}
        self._serve(monkeypatch, assets)

        assert release_chunks.fetch("listings.parquet") is None
