"""Build a train/val split for the damage dataset — grouped by LISTING.

Splitting by photo would be wrong here and quietly flattering: listings repeat
the same frame under different indices (672895662_1 and _2 are byte-identical)
and always show the same car from several angles. A photo-level split puts
near-duplicates on both sides and reports a score the model has not earned.
So the unit is the olx_id, and every photo of a listing lands on one side.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "data" / "damage_seg"


def build(val_every: int) -> None:
    sub = ROOT / "subset"
    for d in ("images/train", "labels/train", "images/val", "labels/val"):
        p = sub / d
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True, exist_ok=True)

    labels = sorted((ROOT / "labels").glob("*.txt"))
    by_listing: dict[str, list[Path]] = {}
    for lp in labels:
        by_listing.setdefault(lp.stem.rsplit("_", 1)[0], []).append(lp)

    # Interleave listings that HAVE damage and listings that don't, so both
    # sides get instances and hard negatives rather than one side getting all
    # the wrecks.
    pos, neg = [], []
    for lid, lps in sorted(by_listing.items()):
        (pos if any(p.read_text().strip() for p in lps) else neg).append((lid, lps))

    counts = {"train": [0, 0], "val": [0, 0]}
    for group in (pos, neg):
        for i, (lid, lps) in enumerate(group):
            split = "val" if i % val_every == 0 else "train"
            for lp in lps:
                img = ROOT / "images" / f"{lp.stem}.jpg"
                if not img.exists():
                    continue
                shutil.copy(img, sub / f"images/{split}" / img.name)
                shutil.copy(lp, sub / f"labels/{split}" / lp.name)
                counts[split][0] += 1
                counts[split][1] += len([x for x in lp.read_text().splitlines() if x.strip()])

    (sub / "dataset.yaml").write_text(
        f"path: {sub.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: damage\n")
    for s, (n_img, n_inst) in counts.items():
        print(f"{s}: {n_img} images, {n_inst} instances")
    print(f"listings: {len(pos)} with damage, {len(neg)} without")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-every", type=int, default=4,
                    help="every Nth listing goes to val")
    build(ap.parse_args().val_every)
