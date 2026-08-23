"""Build 2x2 labelling sheets so four photos can be annotated in one look.

Annotating one photo per look is the bottleneck in this dataset — the queue is
hundreds of images and each one costs a full round trip. A 2x2 sheet with
900-px cells shows four cars at the SAME resolution a single photo was shown
at, so nothing is lost to downscaling, and the boxes drawn on the sheet map
back to per-photo pixels exactly, because the paste geometry (cell origin,
scale, offset) is written next to the sheet.

Sheet coordinates in, per-photo coordinates out — see ``unmap``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent / "data" / "damage_seg"
CELL = 900


def build(start: int, n_sheets: int) -> None:
    order = json.load(open(ROOT / "view_order.json"))
    out = ROOT / "sheets"
    out.mkdir(exist_ok=True)
    geo_path = out / "geometry.json"
    geo = json.loads(geo_path.read_text()) if geo_path.exists() else {}

    for s in range(n_sheets):
        chunk = order[start + s * 4: start + (s + 1) * 4]
        if not chunk:
            break
        sheet = Image.new("RGB", (CELL * 2, CELL * 2), (245, 245, 245))
        d = ImageDraw.Draw(sheet)
        cells = {}
        for i, name in enumerate(chunk):
            p = ROOT / "images" / f"{name}.jpg"
            if not p.exists():
                continue
            im = Image.open(p).convert("RGB")
            scale = min((CELL - 8) / im.width, (CELL - 8) / im.height)
            w, h = int(im.width * scale), int(im.height * scale)
            im = im.resize((w, h))
            cx, cy = (i % 2) * CELL, (i // 2) * CELL
            ox, oy = cx + (CELL - w) // 2, cy + (CELL - h) // 2
            sheet.paste(im, (ox, oy))
            d.rectangle([cx, cy, cx + CELL - 1, cy + CELL - 1], outline=(20, 20, 20), width=3)
            d.rectangle([cx + 4, cy + 4, cx + 40, cy + 34], fill=(20, 20, 20))
            d.text((cx + 16, cy + 12), str(i + 1), fill=(255, 255, 255))
            cells[str(i + 1)] = {"photo_id": name, "ox": ox, "oy": oy, "scale": scale}
        name = f"sheet_{start + s * 4:04d}"
        sheet.save(out / f"{name}.jpg", quality=84)
        geo[name] = cells
        print(f"{name}: " + "  ".join(f"{k}={v['photo_id']}" for k, v in cells.items()))
    geo_path.write_text(json.dumps(geo, indent=1))


def unmap(sheet_name: str, cell: str, box: list[float]) -> tuple[str, list[int]]:
    """Sheet-pixel box → (photo_id, box in that photo's own pixels)."""
    g = json.loads((ROOT / "sheets" / "geometry.json").read_text())[sheet_name][str(cell)]
    x1, y1, x2, y2 = box
    conv = lambda x, o: max(0, int(round((x - o) / g["scale"])))  # noqa: E731
    return g["photo_id"], [conv(x1, g["ox"]), conv(y1, g["oy"]),
                           conv(x2, g["ox"]), conv(y2, g["oy"])]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--sheets", type=int, default=3)
    a = ap.parse_args()
    build(a.start, a.sheets)
