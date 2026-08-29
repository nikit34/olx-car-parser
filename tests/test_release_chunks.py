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


class TestShouldChunk:
    """Only formats a chunk-aware reader consumes may travel split. The
    flipper Worker fetches its JSON from the release on every request, so
    a split valuations.json 404s for it — that took the site down."""

    def _sized(self, tmp_path, name: str, size: int):
        path = tmp_path / name
        path.write_bytes(b"x" * size)
        return path

    def test_large_parquet_is_chunked(self, tmp_path):
        assert release_chunks.should_chunk(
            self._sized(tmp_path, "listings.parquet", 5_000_000)) is True

    def test_large_json_is_not(self, tmp_path):
        assert release_chunks.should_chunk(
            self._sized(tmp_path, "valuations.json", 5_000_000)) is False

    def test_small_parquet_is_not(self, tmp_path):
        assert release_chunks.should_chunk(
            self._sized(tmp_path, "turnover.parquet", 1000)) is False


class FakeGh:
    """A release that answers like ``gh`` does, and remembers the order."""

    def __init__(self, assets=(), fail=()):
        self.assets = {name: f"https://api.github.com/assets/{name}" for name in assets}
        self.fail = tuple(fail)
        self.calls = []

    @staticmethod
    def _result(code: int, out: str = ""):
        return type("R", (), {"returncode": code, "stdout": out, "stderr": ""})()

    def __call__(self, *args):
        self.calls.append(args)
        head = args[:2]
        if head == ("release", "view"):
            return self._result(0, "\n".join(f"{n}\t{u}" for n, u in self.assets.items()))
        if head == ("release", "upload"):
            name = args[3].rsplit("/", 1)[-1]
            if "upload" in self.fail:
                return self._result(1)
            self.assets[name] = f"https://api.github.com/assets/{name}"
            return self._result(0)
        if head == ("release", "delete-asset"):
            self.assets.pop(args[3], None)
            return self._result(0)
        if args[0] == "api":
            if "rename" in self.fail:
                return self._result(1)
            api_url = next(a for a in args if a.startswith("http"))
            new = next(a for a in args if a.startswith("name=")).split("=", 1)[1]
            old = next(n for n, u in self.assets.items() if u == api_url)
            self.assets[new] = self.assets.pop(old)
            return self._result(0)
        return self._result(0)

    def kinds(self):
        return [a[1] if a[0] == "release" else a[0] for a in self.calls]


class TestReplaceInPlace:
    """The Worker fetches these assets on every request, so the window in
    which a name is missing is a window of 503s on the live site."""

    @pytest.fixture(autouse=True)
    def _clean(self, monkeypatch):
        monkeypatch.setattr(release_chunks, "_ASSETS_READ", False)
        monkeypatch.setattr(release_chunks, "_ASSETS", {})
        monkeypatch.setattr(release_chunks.time, "sleep", lambda _s: None)

    @pytest.fixture
    def asset(self, tmp_path):
        path = tmp_path / "models.json"
        path.write_bytes(b'{"models":{}}')
        return path

    def test_a_new_name_goes_straight_up(self, asset, monkeypatch):
        gh = FakeGh()
        monkeypatch.setattr(release_chunks, "_gh", gh)

        assert release_chunks._upload(asset) is True
        assert "models.json" in gh.assets
        assert [k for k in gh.kinds() if k in ("upload", "delete-asset")] == ["upload"]

    def test_the_live_asset_dies_only_after_the_new_bytes_landed(self, asset, monkeypatch):
        gh = FakeGh(assets=["models.json"])
        monkeypatch.setattr(release_chunks, "_gh", gh)

        assert release_chunks._upload(asset) is True
        kinds = gh.kinds()
        assert kinds.index("upload") < kinds.index("delete-asset") < kinds.index("api")
        assert set(gh.assets) == {"models.json"}
        uploaded = [a[3].rsplit("/", 1)[-1] for a in gh.calls if a[:2] == ("release", "upload")]
        assert uploaded == ["models.json" + release_chunks.STAGING_SUFFIX]

    def test_a_failed_upload_leaves_the_live_asset_alone(self, asset, monkeypatch):
        gh = FakeGh(assets=["models.json"], fail=["upload"])
        monkeypatch.setattr(release_chunks, "_gh", gh)

        assert release_chunks._upload(asset) is False
        assert set(gh.assets) == {"models.json"}

    def test_a_failed_rename_still_ends_with_the_asset_in_place(self, asset, monkeypatch):
        gh = FakeGh(assets=["models.json"], fail=["rename"])
        monkeypatch.setattr(release_chunks, "_gh", gh)

        assert release_chunks._upload(asset) is True
        assert set(gh.assets) == {"models.json"}

    def test_the_staging_name_is_not_something_a_reader_asks_for(self, asset, monkeypatch):
        gh = FakeGh(assets=["models.json"])
        monkeypatch.setattr(release_chunks, "_gh", gh)
        release_chunks._upload(asset)

        assert not any(n.endswith(release_chunks.STAGING_SUFFIX) for n in gh.assets)
        assert release_chunks.manifest_name("models.json") != (
            "models.json" + release_chunks.STAGING_SUFFIX)


class TestEnsureRelease:
    """A blipped ``release view`` used to route into ``release create``,
    which answered 422 and killed the whole publish step."""

    @pytest.fixture(autouse=True)
    def _nosleep(self, monkeypatch):
        monkeypatch.setattr(release_chunks.time, "sleep", lambda _s: None)

    def _gh(self, monkeypatch, view_codes, create=(0, "")):
        codes = list(view_codes)
        calls = []

        def _fake(*args):
            calls.append(args)
            if args[:2] == ("release", "view"):
                code = codes.pop(0) if codes else 1
                return type("R", (), {"returncode": code, "stdout": "", "stderr": ""})()
            return type("R", (), {"returncode": create[0], "stdout": "",
                                  "stderr": create[1]})()

        monkeypatch.setattr(release_chunks, "_gh", _fake)
        return calls

    def test_no_create_when_the_release_is_there(self, monkeypatch):
        calls = self._gh(monkeypatch, view_codes=[0])
        release_chunks.ensure_release()
        assert [a[1] for a in calls] == ["view"]

    def test_a_flaky_read_is_retried_before_creating(self, monkeypatch):
        calls = self._gh(monkeypatch, view_codes=[1, 0])
        release_chunks.ensure_release()
        assert not any(a[:2] == ("release", "create") for a in calls)

    def test_a_missing_release_is_created(self, monkeypatch):
        calls = self._gh(monkeypatch, view_codes=[1, 1, 1])
        release_chunks.ensure_release()
        assert any(a[:2] == ("release", "create") for a in calls)

    def test_already_exists_is_not_an_error(self, monkeypatch, capsys):
        self._gh(monkeypatch, view_codes=[1, 1, 1],
                 create=(1, "HTTP 422: Validation Failed\nRelease.tag_name already exists"))
        release_chunks.ensure_release()
        assert "::warning::" not in capsys.readouterr().out

    def test_a_real_creation_failure_warns(self, monkeypatch, capsys):
        self._gh(monkeypatch, view_codes=[1, 1, 1], create=(1, "HTTP 403: forbidden"))
        release_chunks.ensure_release()
        assert "::warning::" in capsys.readouterr().out


class TestPublish:
    def test_counts_failures_without_raising(self, tmp_path, monkeypatch):
        monkeypatch.setattr(release_chunks, "ensure_release", lambda: None)
        a = tmp_path / "a.json"
        a.write_bytes(b"{}")
        b = tmp_path / "b.json"
        b.write_bytes(b"{}")
        monkeypatch.setattr(release_chunks, "_upload", lambda p: p.name != "b.json")

        assert release_chunks.publish([a, b]) == 1

    def test_skips_paths_that_do_not_exist(self, tmp_path, monkeypatch):
        monkeypatch.setattr(release_chunks, "ensure_release", lambda: None)
        monkeypatch.setattr(release_chunks, "_upload", lambda p: True)
        assert release_chunks.publish([tmp_path / "nope.json"]) == 0
