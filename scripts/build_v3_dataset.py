"""Assemble the v3 ImageFolder dataset from VLM labels (issue #5).

Reads ``labels.jsonl`` (produced by ``label_damage_vlm.py``) and any
preserved v2 ImageFolder splits, and emits an ImageFolder layout ready
for ``scripts/train_damage_classifier.py``::

    <out>/train/clean/*.jpg
    <out>/train/damaged/*.jpg
    <out>/val/clean/*.jpg
    <out>/val/damaged/*.jpg

If a v2 source ImageFolder is supplied via ``--v2-data`` and exists,
its ``train/`` and ``val/`` (and optionally ``test/``) splits are
mirrored as symlinks under the v3 root — the new VLM-labelled photos
go on top of that. When v2 source isn't around (the host's /tmp gets
pruned periodically — DrBimmer source photos aren't preserved long
term), we still produce a usable dataset purely from VLM labels, with
a deterministic 90/10 train/val split per class.

Stratification: the VLM agrees with the v2 max-rule on most flagged
photos labelled "damaged" (severity ≥ 1). The interesting *new*
gradient is the FP-rich photos that v2 calls damaged at p ≥ 0.30 but
the VLM calls clean — those go to ``train/clean/`` and are the
primary fix for issue #1's blind spots.

Run on the host::

    .venv/bin/python scripts/build_v3_dataset.py \\
        --labels /tmp/v3_data/labels.jsonl \\
        --out    /tmp/v3_data/imagefolder \\
        --v2-data /tmp/yolo_data/combined_v1   # optional, no-op if missing
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def deterministic_split(key: str, val_fraction: float = 0.10) -> str:
    """Return ``"train"`` or ``"val"`` deterministically by content hash.

    Stable across runs so a re-build doesn't reshuffle photos and a
    re-train doesn't accidentally peek at val photos. ``val_fraction``
    is taken from the first 4 bytes of an MD5 — uniform enough at
    sample sizes <10 000.
    """
    h = hashlib.md5(key.encode("utf-8")).digest()
    bucket = int.from_bytes(h[:4], "big") / 0xFFFFFFFF
    return "val" if bucket < val_fraction else "train"


def ensure_symlink(src: Path, dst: Path) -> None:
    """Create ``dst`` as a symlink to ``src`` if absent."""
    if dst.is_symlink() or dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(src.resolve())


def mirror_v2(v2_root: Path, out_root: Path) -> dict[str, int]:
    """Symlink the existing v2 ImageFolder into the v3 root.

    Tolerant: if any expected split or class is missing, just skip it
    (v2 source dirs may have been pruned from /tmp).
    """
    counts: dict[str, int] = {"train/clean": 0, "train/damaged": 0,
                              "val/clean": 0, "val/damaged": 0}
    if not v2_root.exists():
        return counts
    for split in ("train", "val"):
        for cls in ("clean", "damaged"):
            src_dir = v2_root / split / cls
            if not src_dir.is_dir():
                continue
            dst_dir = out_root / split / cls
            dst_dir.mkdir(parents=True, exist_ok=True)
            for src in src_dir.iterdir():
                if not src.is_file() and not src.is_symlink():
                    continue
                ensure_symlink(src, dst_dir / src.name)
                counts[f"{split}/{cls}"] += 1
    return counts


def add_vlm_labels(
    labels_path: Path,
    out_root: Path,
    val_fraction: float,
) -> dict[str, int]:
    """Symlink VLM-labelled photos into the v3 ImageFolder.

    Resumable: ``ensure_symlink`` is idempotent so re-runs after new
    labelling passes simply add the fresh entries.
    """
    counts: dict[str, int] = {"train/clean": 0, "train/damaged": 0,
                              "val/clean": 0, "val/damaged": 0,
                              "skipped_missing_file": 0,
                              "skipped_no_label": 0}
    if not labels_path.exists():
        return counts
    for line in labels_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        damaged = row.get("damaged")
        if damaged is None:
            counts["skipped_no_label"] += 1
            continue
        src = Path(row["photo_path"])
        if not src.exists():
            counts["skipped_missing_file"] += 1
            continue
        cls = "damaged" if damaged else "clean"
        # Split per (olx_id, idx) so all augmentations of the same
        # photo land in the same split.
        split_key = f"{row['olx_id']}_{row['photo_idx']}"
        split = deterministic_split(split_key, val_fraction)
        # Filename includes a ``vlm_`` prefix so we can tell mined
        # photos apart from v2 source photos at a glance during
        # debugging.
        fname = f"vlm_{row['olx_id']}_{row['photo_idx']}.jpg"
        dst = out_root / split / cls / fname
        ensure_symlink(src, dst)
        counts[f"{split}/{cls}"] += 1
    return counts


def count_dir(p: Path) -> int:
    if not p.is_dir():
        return 0
    return sum(1 for _ in p.iterdir())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path,
                    default=Path("/tmp/v3_data/labels.jsonl"))
    ap.add_argument("--v2-data", type=Path,
                    default=Path("/tmp/yolo_data/combined_v1"),
                    help="Existing v2 ImageFolder root (mirrored into v3 if "
                         "present, no-op if missing)")
    ap.add_argument("--out", type=Path,
                    default=Path("/tmp/v3_data/imagefolder"))
    ap.add_argument("--val-fraction", type=float, default=0.10,
                    help="Fraction of mined photos held out for val "
                         "(deterministic per (olx_id, photo_idx))")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Building v3 dataset at {args.out}")
    print(f"  v2 source : {args.v2_data} "
          f"({'present' if args.v2_data.exists() else 'MISSING — VLM-only'})")
    print(f"  labels    : {args.labels} "
          f"({'present' if args.labels.exists() else 'MISSING'})")

    v2_counts = mirror_v2(args.v2_data, args.out)
    vlm_counts = add_vlm_labels(args.labels, args.out, args.val_fraction)

    print("\nv2 mirrored:")
    for k, v in v2_counts.items():
        print(f"  {k:>18s}: {v}")
    print("VLM-labelled added:")
    for k, v in vlm_counts.items():
        print(f"  {k:>18s}: {v}")

    print("\nFinal layout:")
    for split in ("train", "val"):
        for cls in ("clean", "damaged"):
            n = count_dir(args.out / split / cls)
            print(f"  {split}/{cls:>7s}: {n}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
