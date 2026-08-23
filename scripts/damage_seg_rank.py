"""Re-rank the unlabelled queue by combining CLIP similarity with the model.

CLIP alone gets you the obvious wrecks and then runs dry: it lifts "car in a
scrapyard" to the top and scores a dented door the same as a clean one. Once a
detector exists, however weak, its response is a different and complementary
signal — it was trained on exactly the thing we are looking for.

So the queue is ordered by the max of two normalised scores rather than either
alone: an image reaches the top if EITHER the scene looks like damage or the
detector fired somewhere on it. Images the detector is confident about but CLIP
is not are exactly the ones worth a human look — they are where the model is
learning something CLIP cannot see.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "data" / "damage_seg"


def main(weights: Path, conf: float, limit: int) -> None:
    from ultralytics import YOLO

    clip = json.loads((ROOT / "clip_damage_rank.json").read_text())
    done = set()
    for f in (ROOT / "ann").glob("*.json"):
        done |= set(json.loads(f.read_text()))
    queue = [n for n in clip if n not in done]
    print(f"unlabelled: {len(queue)}")

    model = YOLO(str(weights))
    scores = {}
    for i, name in enumerate(queue):
        p = ROOT / "images" / f"{name}.jpg"
        if not p.exists():
            continue
        r = model.predict(str(p), conf=conf, verbose=False)[0]
        confs = r.boxes.conf.tolist() if r.boxes is not None else []
        scores[name] = max(confs) if confs else 0.0
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(queue)}", flush=True)

    fired = sum(1 for v in scores.values() if v > 0)
    print(f"detector fired on {fired}/{len(scores)} unlabelled images")

    # Normalise both to [0,1] and take the max: an image qualifies on either
    # signal, so a dent the scene-level embedding cannot see still surfaces.
    mx = max(scores.values()) or 1.0
    combined = {n: max(clip.get(n, 0.0), scores[n] / mx) for n in scores}
    order = [n for n, _ in sorted(combined.items(), key=lambda kv: -kv[1])]
    (ROOT / "view_order.json").write_text(json.dumps(order))
    (ROOT / "model_rank.json").write_text(json.dumps(scores, indent=0))
    print("top of the re-ranked queue:")
    for n in order[:limit]:
        print(f"  {n}  clip={clip.get(n, 0):.2f}  model={scores[n]:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=Path("/tmp/yolo_dmg/v3/weights/best.pt"))
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--limit", type=int, default=8)
    a = ap.parse_args()
    main(a.weights, a.conf, a.limit)
