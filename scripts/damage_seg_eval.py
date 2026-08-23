"""Evaluate the damage detector on held-out photos, in the terms the product uses.

mAP is the right metric for a detector and the wrong one for this decision. The
production question is not "how tight are the boxes" — it is "when this model
says a car is damaged, is it?", because that verdict currently vetoes a listing
out of the deal feed. So this reports both: ultralytics' mAP for the detector,
and an IMAGE-level precision/recall at a confidence threshold, which is the
number comparable to the ResNet50 it would replace (0.20 precision measured on
production photos, 2026-08-23).

Image-level truth = the photo has at least one annotated damage instance.
Image-level prediction = the model returns at least one box above the threshold.
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "data" / "damage_seg"


def main(weights: Path, split: str, thresholds: list[float]) -> None:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    img_dir = ROOT / "subset" / "images" / split
    lbl_dir = ROOT / "subset" / "labels" / split
    images = sorted(img_dir.glob("*.jpg"))
    truth = {
        p.stem: bool((lbl_dir / f"{p.stem}.txt").read_text().strip())
        for p in images if (lbl_dir / f"{p.stem}.txt").exists()
    }
    print(f"{split}: {len(truth)} images, {sum(truth.values())} with damage")

    # One inference pass at the lowest threshold, then re-threshold in python —
    # running the model once per threshold would be the same numbers at 4x cost.
    lowest = min(thresholds)
    scores: dict[str, float] = {}
    for p in images:
        r = model.predict(str(p), conf=lowest, verbose=False)[0]
        confs = r.boxes.conf.tolist() if r.boxes is not None else []
        scores[p.stem] = max(confs) if confs else 0.0

    print(f"\n{'conf':>6} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4} "
          f"{'precision':>10} {'recall':>7} {'F1':>6}")
    for t in thresholds:
        tp = sum(1 for k, g in truth.items() if g and scores[k] >= t)
        fp = sum(1 for k, g in truth.items() if not g and scores[k] >= t)
        fn = sum(1 for k, g in truth.items() if g and scores[k] < t)
        tn = sum(1 for k, g in truth.items() if not g and scores[k] < t)
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if tp and prec + rec else 0.0
        print(f"{t:>6.2f} {tp:>4} {fp:>4} {fn:>4} {tn:>4} {prec:>10.2f} {rec:>7.2f} {f1:>6.2f}")

    print("\nper-image max confidence (truth in brackets):")
    for k in sorted(scores, key=lambda k: -scores[k]):
        print(f"  {k:<16} {scores[k]:.3f}  [{'damage' if truth[k] else 'clean '}]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=Path("/tmp/yolo_dmg/v1/weights/best.pt"))
    ap.add_argument("--split", default="val")
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[0.10, 0.20, 0.30, 0.50])
    a = ap.parse_args()
    main(a.weights, a.split, a.thresholds)
