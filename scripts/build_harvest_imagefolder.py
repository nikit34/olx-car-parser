"""Build ImageFolder dataset combining drbimmer_binary + OLX harvest.

OLX harvest: photos labeled in /tmp/damage_harvest/hand_labels.jsonl with
severity 0 (clean) or >=1 (damaged). Uses cached photos in
/tmp/damage_harvest/{olx_id}_{idx}.jpg.

Output: /tmp/yolo_data/combined_v1/{train,val,test}/{clean,damaged}/*.jpg
- train: drbimmer/train + ALL harvest (since gold is the OLX-domain test)
- val:   drbimmer/val
- test:  drbimmer/test
Symlinks to source files; idempotent.
"""

from __future__ import annotations

import json
from pathlib import Path

import argparse

DRBIMMER = Path("/tmp/yolo_data/drbimmer_binary")
HARVEST_DIR = Path("/tmp/damage_harvest")
LABELS = HARVEST_DIR / "hand_labels.jsonl"


def ensure_symlink(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(src.resolve())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/yolo_data/combined_v1",
                    help="Output ImageFolder root")
    ap.add_argument("--include-harvest-clean", action="store_true",
                    help="Include harvest sev=0 photos in train/clean")
    ap.add_argument("--upsample-damaged", type=int, default=1,
                    help="Replicate harvest damaged N times in train/damaged")
    args = ap.parse_args()
    out_root = Path(args.out)

    # 1. Mirror drbimmer_binary structure
    for split in ("train", "val", "test"):
        for cls in ("clean", "damaged"):
            src_dir = DRBIMMER / split / cls
            dst_dir = out_root / split / cls
            dst_dir.mkdir(parents=True, exist_ok=True)
            for src in src_dir.iterdir():
                if src.is_file():
                    ensure_symlink(src, dst_dir / src.name)

    # 2. Add harvest photos to TRAIN split
    labels = [json.loads(l) for l in LABELS.read_text().splitlines() if l.strip()]
    n_clean = n_damaged = n_missing = 0
    for r in labels:
        olx_id = r["olx_id"]
        idx = r["idx"]
        sev = r["severity"]
        src = HARVEST_DIR / f"{olx_id}_{idx}.jpg"
        if not src.exists():
            n_missing += 1
            continue
        is_damaged = sev >= 1
        if not is_damaged and not args.include_harvest_clean:
            continue
        cls = "damaged" if is_damaged else "clean"
        if is_damaged:
            for k in range(args.upsample_damaged):
                suffix = "" if k == 0 else f"_dup{k}"
                dst = out_root / "train" / cls / f"olx_{olx_id}_{idx}{suffix}.jpg"
                ensure_symlink(src, dst)
            n_damaged += args.upsample_damaged
        else:
            dst = out_root / "train" / cls / f"olx_{olx_id}_{idx}.jpg"
            ensure_symlink(src, dst)
            n_clean += 1

    print(f"Harvest added: {n_clean} clean + {n_damaged} damaged "
          f"(upsample={args.upsample_damaged}, missing={n_missing})")

    # 3. Final counts
    for split in ("train", "val", "test"):
        for cls in ("clean", "damaged"):
            n = sum(1 for _ in (out_root / split / cls).iterdir())
            print(f"  {split}/{cls}: {n}")


if __name__ == "__main__":
    main()
