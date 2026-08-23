"""Convert sheet-coordinate annotations into per-photo annotation JSON.

Input JSON: {sheet_name: {cell: [{"box": [...], "type": ..., "note": ...}]}}
where box is in SHEET pixels. Cells with an empty list are explicit negatives.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.damage_seg_sheets import ROOT, unmap  # noqa: E402


def main(src: Path, out: Path) -> None:
    data = json.loads(src.read_text())
    ann: dict[str, list] = {}
    for sheet, cells in data.items():
        for cell, boxes in cells.items():
            photo_id = None
            conv = []
            for b in boxes:
                photo_id, box = unmap(sheet, cell, b["box"])
                conv.append({k: v for k, v in b.items() if k != "box"} | {"box": box})
            if photo_id is None:
                geo = json.loads((ROOT / "sheets" / "geometry.json").read_text())
                photo_id = geo[sheet][str(cell)]["photo_id"]
            ann[photo_id] = conv
    out.write_text(json.dumps(ann, ensure_ascii=False, indent=1))
    n_inst = sum(len(v) for v in ann.values())
    print(f"{out.name}: {len(ann)} photos, {n_inst} instances, "
          f"{sum(1 for v in ann.values() if not v)} negatives")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
