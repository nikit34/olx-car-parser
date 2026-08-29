"""Check the latest-data release still holds everything that reads from it.

Assets go missing without anyone noticing: ``gh release upload --clobber``
deletes before it writes, so an interrupted upload leaves nothing behind,
and the failure used to be swallowed. Five artefacts had quietly rotted
out of the release by 2026-08-29, including the one the /preco pages read.

With ``--heal`` a missing asset is re-published from the host's copy when
one exists, which is the usual case: the pipeline built the file, only the
upload died. Whatever is still missing afterwards is reported and the exit
code is non-zero, so the run goes red and someone looks.

    python -m scripts.audit_release --heal
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_stlite_bundle import WITNESS_FILES  # noqa: E402
from scripts.release_chunks import REPO, TAG, manifest_name, publish  # noqa: E402
from scripts.release_chunks import _get, BASE_URL  # noqa: E402

WORKER_ASSETS = (
    "hot_deals_all.json",
    "hot_deals_centro.json",
    "hot_deals_norte.json",
    "hot_deals_sul.json",
    "valuations.json",
    "models.json",
    "brands_models.json",
)
MODEL_ARTIFACTS = (
    "price_metrics.json",
    "price_importance.json",
    "price_grouped_importance.json",
    "price_shap_importance.json",
    "price_backtest.json",
)
SEARCH_DIRS = ("data/dashboard", "data/hot_deals", "data")
STALE_AFTER_HOURS = 36


def expected() -> set[str]:
    return set(WITNESS_FILES) | set(WORKER_ASSETS) | set(MODEL_ARTIFACTS)


def asset_times() -> dict[str, datetime]:
    result = subprocess.run(
        ["gh", "api", f"repos/{REPO}/releases/tags/{TAG}", "--jq",
         "[.assets[] | {name, updated_at}]"], capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"cannot read the release: {result.stderr.strip()}")
    return {
        row["name"]: datetime.fromisoformat(row["updated_at"].replace("Z", "+00:00"))
        for row in json.loads(result.stdout)
    }


def release_assets() -> set[str]:
    return set(asset_times())


def stale_assets(times: dict[str, datetime]) -> dict[str, int]:
    """Expected assets whose newest piece is older than the threshold.

    Presence is not freshness: a witness the pipeline stopped refreshing
    still passes a name check while the dashboard quietly serves last
    week's market.
    """
    now = datetime.now(timezone.utc)
    cutoff = timedelta(hours=STALE_AFTER_HOURS)
    stale: dict[str, int] = {}
    for name in expected():
        pieces = [t for asset, t in times.items()
                  if asset == name or asset.startswith(f"{name}.")]
        if not pieces:
            continue
        age = now - max(pieces)
        if age > cutoff:
            stale[name] = int(age.total_seconds() // 3600)
    return stale


def incomplete_chunk_sets(assets: set[str]) -> dict[str, str]:
    """Names whose manifest is present but whose parts are not all there."""
    broken: dict[str, str] = {}
    for asset in assets:
        if not asset.endswith(".chunks.json"):
            continue
        name = asset[: -len(".chunks.json")]
        raw = _get(f"{BASE_URL}/{asset}", quiet=True)
        if raw is None:
            broken[name] = "manifest unreadable"
            continue
        meta = json.loads(raw.decode("utf-8"))
        wanted = {f"{name}.{meta['digest']}.p{i:02d}" for i in range(meta["parts"])}
        gap = wanted - assets
        if gap:
            broken[name] = f"{len(gap)} of {meta['parts']} parts missing"
    return broken


def satisfied(name: str, assets: set[str]) -> bool:
    return name in assets or manifest_name(name) in assets


def local_copy(name: str) -> Path | None:
    for directory in SEARCH_DIRS:
        candidate = ROOT / directory / name
        if candidate.exists():
            return candidate
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--heal", action="store_true",
                        help="Re-publish missing assets from the host's copies.")
    args = parser.parse_args(argv)

    times = asset_times()
    assets = set(times)
    missing = sorted(n for n in expected() if not satisfied(n, assets))
    broken = incomplete_chunk_sets(assets)

    for name, hours in sorted(stale_assets(times).items()):
        print(f"::warning::{name} has not been refreshed for {hours}h")

    if not missing and not broken:
        print(f"release complete: {len(expected())} expected assets present")
        return 0

    for name, reason in sorted(broken.items()):
        print(f"::warning::{name}: {reason}")
    if broken:
        missing = sorted(set(missing) | set(broken))

    if args.heal:
        healable = [p for p in (local_copy(n) for n in missing) if p is not None]
        if healable:
            print(f"re-publishing {len(healable)} missing assets from local copies")
            publish(healable)
            assets = release_assets()
            missing = sorted(n for n in expected() if not satisfied(n, assets))
            missing = sorted(set(missing) | set(incomplete_chunk_sets(assets)))

    if not missing:
        print("release repaired from local copies")
        return 0

    for name in missing:
        where = local_copy(name)
        print(f"::error::missing from the release: {name}"
              f"{'' if where else ' (and no local copy to restore it from)'}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
