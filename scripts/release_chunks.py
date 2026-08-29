"""Ship a large asset to a GitHub Release in pieces small enough to survive.

The scrape host's link collapses on long uploads: throughput to
``uploads.github.com`` falls to 25-51 KB/s and the connection is cut at
120 s, so anything past a couple of megabytes never lands. Everything the
dashboard needs still has to reach the Cloudflare build, which reads the
release. Splitting keeps each request inside that window.

Part names carry a short digest of the source file, so a half-finished
upload cannot be mixed with the previous generation's parts: the manifest
naming the digest is written last and is the commit point. Stale parts are
deleted only after the new manifest is in place.

Replacing an asset that already exists goes through a staging name, because
``gh release upload --clobber`` deletes the live one before it starts
sending and on this link that leaves the name missing for minutes.

    python -m scripts.release_chunks upload data/dashboard/listings.parquet
    python -m scripts.release_chunks fetch listings.parquet --out /tmp/l.parquet
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.request import Request, urlopen

TAG = "latest-data"
REPO = "nikit34/olx-car-parser"
CHUNK_BYTES = 1_500_000
UPLOAD_RETRIES = 3
WHOLE_LIMIT = 3_000_000
CHUNKABLE_SUFFIXES = (".parquet",)
STAGING_SUFFIX = ".uploading"
BASE_URL = f"https://github.com/{REPO}/releases/download/{TAG}"

_ASSETS: dict[str, str] = {}
_ASSETS_READ = False


def manifest_name(name: str) -> str:
    return f"{name}.chunks.json"


def _digest(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
            size += len(block)
    return h.hexdigest(), size


def _gh(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["gh", *args], capture_output=True, text=True)


def _assets(refresh: bool = False) -> dict[str, str]:
    """Asset name to API URL for everything currently in the release."""
    global _ASSETS_READ
    if _ASSETS_READ and not refresh:
        return _ASSETS
    result = _gh("release", "view", TAG, "--repo", REPO, "--json", "assets",
                 "--jq", '.assets[] | .name + "\t" + .apiUrl')
    _ASSETS.clear()
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            name, _, api_url = line.partition("\t")
            if name.strip() and api_url.strip():
                _ASSETS[name.strip()] = api_url.strip()
    _ASSETS_READ = True
    return _ASSETS


def _put(path: Path) -> bool:
    for attempt in range(UPLOAD_RETRIES):
        result = _gh("release", "upload", TAG, str(path), "--repo", REPO, "--clobber")
        if result.returncode == 0:
            return True
        print(f"  {path.name}: attempt {attempt + 1}/{UPLOAD_RETRIES} failed: "
              f"{result.stderr.strip().splitlines()[-1] if result.stderr.strip() else '?'}",
              file=sys.stderr)
        time.sleep(2 ** attempt)
    return False


def _rename(api_url: str, name: str) -> bool:
    for attempt in range(UPLOAD_RETRIES):
        result = _gh("api", "--method", "PATCH", api_url, "-f", f"name={name}", "--silent")
        if result.returncode == 0:
            return True
        time.sleep(2 ** attempt)
    return False


def _delete(name: str) -> None:
    _gh("release", "delete-asset", TAG, name, "--repo", REPO, "--yes")
    _ASSETS.pop(name, None)


def _upload(path: Path) -> bool:
    """Land *path* under its own name, leaving no gap where it is missing.

    A name nothing serves yet is uploaded straight. Replacing a live one
    sends the new bytes under a staging name first, so the only moment the
    real name is absent is the rename between the two, not the whole
    upload: the Worker reads several of these assets on every request and
    answers 503 while one is gone.
    """
    name = path.name
    if name not in _assets():
        return _put(path)
    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp) / (name + STAGING_SUFFIX)
        shutil.copyfile(path, staged)
        if not _put(staged):
            _delete(staged.name)
            return False
        api_url = _assets(refresh=True).get(staged.name)
        if api_url is None:
            print(f"  {name}: the staged copy never appeared in the release",
                  file=sys.stderr)
            return False
        _delete(name)
        if _rename(api_url, name):
            _ASSETS.pop(staged.name, None)
            _ASSETS[name] = api_url
            return True
        print(f"  {name}: rename failed, uploading under the real name instead",
              file=sys.stderr)
        landed = _put(path)
        _delete(staged.name)
        _assets(refresh=True)
        return landed


def _release_asset_names() -> set[str]:
    return set(_assets(refresh=True))


def upload(path: Path) -> bool:
    """Split *path* into parts, upload them, then commit with a manifest."""
    name = path.name
    sha, size = _digest(path)
    tag = sha[:8]
    parts = (size + CHUNK_BYTES - 1) // CHUNK_BYTES
    print(f"{name}: {size} bytes, {parts} parts, digest {tag}")

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        with path.open("rb") as fh:
            for index in range(parts):
                part = tmpdir / f"{name}.{tag}.p{index:02d}"
                part.write_bytes(fh.read(CHUNK_BYTES))
                if not _upload(part):
                    print(f"{name}: giving up, manifest not written", file=sys.stderr)
                    return False
                part.unlink()

        manifest = tmpdir / manifest_name(name)
        manifest.write_text(json.dumps({
            "name": name, "sha256": sha, "size": size,
            "parts": parts, "digest": tag,
        }))
        if not _upload(manifest):
            print(f"{name}: parts landed but manifest did not", file=sys.stderr)
            return False

    part_pattern = re.compile(rf"^{re.escape(name)}\.([0-9a-f]{{8}})\.p\d{{2}}$")
    assets = _release_asset_names()
    stale = set()
    for asset in assets:
        match = part_pattern.match(asset)
        if match and match.group(1) != tag:
            stale.add(asset)
    if name in assets:
        stale.add(name)
    for asset in sorted(stale):
        _delete(asset)
    if stale:
        print(f"{name}: removed {len(stale)} stale parts")
    return True


def _get(url: str, timeout: int = 60, quiet: bool = False) -> bytes | None:
    try:
        request = Request(url, headers={"User-Agent": "olx-release-chunks"})
        with urlopen(request, timeout=timeout) as response:
            return response.read()
    except Exception as exc:  # noqa: BLE001
        if not quiet:
            print(f"  fetch {url.rsplit('/', 1)[-1]}: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
        return None


def fetch(name: str, base_url: str = BASE_URL) -> bytes | None:
    """Return the asset's bytes, whole or reassembled, or None.

    A digest mismatch returns None rather than a plausible-looking file:
    downstream would bake a corrupt witness into the dashboard bundle.
    """
    raw = _get(f"{base_url}/{name}", quiet=True)
    if raw is not None:
        return raw

    meta_raw = _get(f"{base_url}/{manifest_name(name)}")
    if meta_raw is None:
        return None
    meta = json.loads(meta_raw.decode("utf-8"))

    blob = bytearray()
    for index in range(meta["parts"]):
        part = _get(f"{base_url}/{name}.{meta['digest']}.p{index:02d}")
        if part is None:
            return None
        blob.extend(part)

    if len(blob) != meta["size"] or hashlib.sha256(blob).hexdigest() != meta["sha256"]:
        print(f"  {name}: reassembled copy does not match its manifest", file=sys.stderr)
        return None
    return bytes(blob)


def should_chunk(path: Path) -> bool:
    """Whether *path* travels split.

    Only formats read by chunk-aware code qualify. The flipper Worker
    fetches its JSON straight from the release on every request, so a
    split copy would 404 for it — chunking valuations.json took the site
    down on 2026-08-29.
    """
    return path.suffix in CHUNKABLE_SUFFIXES and path.stat().st_size > WHOLE_LIMIT


def publish(paths: list[Path]) -> int:
    """Upload each path the way it needs, whole or split. Returns failures."""
    failed = 0
    for path in paths:
        if not path.exists():
            continue
        if should_chunk(path):
            ok = upload(path)
        else:
            ok = _upload(path)
        if not ok:
            failed += 1
            print(f"::warning::upload failed: {path.name} "
                  f"({path.stat().st_size} bytes)")
    print(f"uploads: {len(paths) - failed}/{len(paths)} succeeded")
    return failed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    up = sub.add_parser("upload")
    up.add_argument("paths", nargs="+", type=Path)

    pub = sub.add_parser("publish")
    pub.add_argument("paths", nargs="+", type=Path)

    down = sub.add_parser("fetch")
    down.add_argument("name")
    down.add_argument("--out", type=Path, required=True)
    down.add_argument("--base-url", default=BASE_URL)

    args = parser.parse_args(argv)

    if args.cmd == "upload":
        return 0 if all(upload(p) for p in args.paths if p.exists()) else 1

    if args.cmd == "publish":
        return 1 if publish(args.paths) else 0

    blob = fetch(args.name, args.base_url)
    if blob is None:
        return 1
    args.out.write_bytes(blob)
    print(f"{args.name}: {len(blob)} bytes -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
