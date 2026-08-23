"""Turn damage boxes into segmentation masks with SAM, then YOLO-seg labels.

A language model can say WHERE the damage is and WHAT it is; it cannot draw a
pixel-accurate boundary. SAM can do exactly the second half when prompted with
a box. So annotation is split along that line: the boxes carry the judgement,
SAM carries the geometry, and every mask is rendered as an overlay so the
judgement can be re-checked against what SAM actually produced.

Input:  data/damage_seg/ann/*.json — {photo_id: [{"box": [x1,y1,x2,y2],
        "type": "..."}]}. An empty list is a deliberate negative (a photo of a
        damaged car that shows no damage in THIS frame) and is kept: those are
        the hard negatives that stop a detector firing on every silver hatchback.
Output: labels/*.txt (YOLO-seg polygons), overlays/*.jpg (visual check),
        dataset.yaml.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROOT = PROJECT_ROOT / "data" / "damage_seg"

# One class. Subtypes (dent_crease / broken_lamp / panel_missing / …) are kept
# in the annotation JSON so a finer split is possible later without going back
# through the images, but a first detector on a few hundred instances learns
# "damage vs not" far more reliably than five sparse classes.
CLASS_NAMES = ["damage"]

# Which subtypes SAM may segment, and which must keep their box.
#
# Measured on the first probe, not assumed. Given a box, SAM segments the
# salient OBJECT inside it — which is exactly right when the damage IS an
# object or a discontinuity (a smashed lamp lens, a missing panel, a detached
# bumper): the mask came back tight and correct. It is wrong for surface
# deformation: on a dent, a loose box returned the whole bumper (SAM found the
# panel, not the damage) and a tight box returned a 1 695-px shadow fragment,
# 14 % of the box. Training on either would teach "find panels" or "find
# shadows".
#
# So deformation keeps the box as its polygon. A rectangle is a coarser label
# than a mask, but it is an HONEST one, and YOLO-seg reads both from the same
# file. Every label records which path produced it.
# SAM does well on a part you could pick up and carry: a hanging bumper, a
# smashed lamp lens, a panel that is simply gone. It does badly on an absence
# with no clear silhouette — prompted at a missing-headlight cavity it returned
# a thin strip of the shadow line, not the opening. So cavities went to the box
# side too, on the evidence.
SAM_SUBTYPES = {
    "broken_lamp", "missing_part", "detached_part",
    "shattered_glass", "torn_panel", "burnt", "airbag_deployed",
}
BOX_SUBTYPES = {"dent_crease", "scratch", "paint_damage", "misaligned_panel",
                "cracked_glass", "exposed_structure"}


def _polygons_from_mask(mask: np.ndarray, max_points: int = 60) -> list[list[float]]:
    """Outer contour of a binary mask as a normalised polygon."""
    import cv2
    m = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    c = max(contours, key=cv2.contourArea)
    # Douglas-Peucker: a 200-point contour is noise, not signal, and bloats the
    # label file. Tighten epsilon until the polygon fits in max_points.
    eps = 0.002 * cv2.arcLength(c, True)
    for _ in range(8):
        approx = cv2.approxPolyDP(c, eps, True)
        if len(approx) <= max_points:
            break
        eps *= 1.6
    h, w = mask.shape
    return [[float(p[0][0]) / w, float(p[0][1]) / h] for p in approx]


def run(ann_dir: Path, model_name: str, overlays: bool) -> None:
    from ultralytics import SAM
    sam = SAM(model_name)

    img_dir = ROOT / "images"
    lbl_dir = ROOT / "labels"
    ovl_dir = ROOT / "overlays"
    lbl_dir.mkdir(exist_ok=True)
    if overlays:
        ovl_dir.mkdir(exist_ok=True)

    ann: dict[str, list] = {}
    for f in sorted(ann_dir.glob("*.json")):
        ann.update(json.loads(f.read_text()))

    n_pos = n_neg = n_inst = 0
    for photo_id, boxes in ann.items():
        img_path = img_dir / f"{photo_id}.jpg"
        if not img_path.exists():
            print(f"missing image: {photo_id}")
            continue
        label_path = lbl_dir / f"{photo_id}.txt"
        if not boxes:
            label_path.write_text("")          # explicit background image
            n_neg += 1
            continue

        im_w, im_h = Image.open(img_path).size
        sam_idx = [i for i, b in enumerate(boxes)
                   if b.get("type") in SAM_SUBTYPES]
        sam_polys: dict[int, list] = {}
        if sam_idx:
            res = sam(str(img_path), bboxes=[boxes[i]["box"] for i in sam_idx],
                      verbose=False)[0]
            if res.masks is not None:
                for i, mask in zip(sam_idx, res.masks.data.cpu().numpy()):
                    poly = _polygons_from_mask(mask)
                    if len(poly) >= 3:
                        sam_polys[i] = poly

        lines = []
        polys = []
        for i, b in enumerate(boxes):
            poly = sam_polys.get(i)
            if poly is None:
                x1, y1, x2, y2 = b["box"]
                poly = [[x1 / im_w, y1 / im_h], [x2 / im_w, y1 / im_h],
                        [x2 / im_w, y2 / im_h], [x1 / im_w, y2 / im_h]]
                b["mask_source"] = "box"
            else:
                b["mask_source"] = "sam"
            polys.append(poly)
            flat = " ".join(f"{x:.5f} {y:.5f}" for x, y in poly)
            lines.append(f"0 {flat}")
        label_path.write_text("\n".join(lines))
        n_pos += 1
        n_inst += len(lines)

        if overlays and polys:
            im = Image.open(img_path).convert("RGB")
            layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
            d = ImageDraw.Draw(layer)
            # Colour by provenance, and never fill with a colour a car can
            # already be: a red mask over a red Golf is invisible, which is how
            # an unverifiable label sneaks through. Lime = SAM mask, cyan box =
            # the prompt, magenta = box-derived polygon (no mask claimed).
            for b, poly in zip(boxes, polys):
                pts = [(x * im.width, y * im.height) for x, y in poly]
                if b.get("mask_source") == "sam":
                    d.polygon(pts, fill=(0, 255, 90, 70), outline=(0, 255, 90, 255))
                    for k in range(3):
                        d.line(pts + [pts[0]], fill=(0, 0, 0, 200), width=1)
                        d.line(pts + [pts[0]], fill=(0, 255, 90, 255), width=2)
                else:
                    d.polygon(pts, outline=(255, 0, 220, 255), width=4)
                d.rectangle(b["box"], outline=(60, 200, 255, 200), width=2)
            im = Image.alpha_composite(im.convert("RGBA"), layer).convert("RGB")
            im.save(ovl_dir / f"{photo_id}.jpg", quality=88)

    (ROOT / "dataset.yaml").write_text(
        f"path: {ROOT}\ntrain: images\nval: images\nnames:\n  0: damage\n")
    print(f"annotated: {n_pos} images with damage ({n_inst} instances), "
          f"{n_neg} explicit negatives")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", type=Path, default=ROOT / "ann")
    ap.add_argument("--model", default="mobile_sam.pt")
    ap.add_argument("--no-overlays", action="store_true")
    a = ap.parse_args()
    run(a.ann, a.model, overlays=not a.no_overlays)


if __name__ == "__main__":
    main()
