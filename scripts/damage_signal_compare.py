"""Compare candidate corpus-wide photo signals as RANKING weights, not vetoes.

A veto and a ranking weight are judged by different things. A veto needs
precision at the corpus base rate — at 3% damage that means a false-positive
rate under ~0.5%, which is brutal. A weight only has to be CORRELATED with
damage: it nudges a listing up or down the feed, and a wrong nudge costs a
little ordering quality, not a lost deal.

The right metric for that is ranking quality, so this reports ROC-AUC (and
average precision) of each candidate signal against the hand-labelled photos.
AUC is independent of prevalence, which is exactly what makes it comparable
across our damage-enriched annotation set and the real corpus.

Candidates:
  resnet  — the production damage_classifier_v2 (already scored 65k listings)
  clip    — zero-shot CLIP similarity to damage prompts (no training at all)
  yolo    — the detector trained here on 95 photos
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "data" / "damage_seg"


def _auc(pairs: list[tuple[float, int]]) -> tuple[float, float]:
    """ROC-AUC via rank statistic, plus average precision."""
    pairs = sorted(pairs, key=lambda x: x[0])
    n_pos = sum(y for _, y in pairs)
    n_neg = len(pairs) - n_pos
    if not n_pos or not n_neg:
        return float("nan"), float("nan")
    # ranks with ties averaged
    ranks, i = [0.0] * len(pairs), 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    sum_pos = sum(r for r, (_, y) in zip(ranks, pairs) if y)
    auc = (sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

    ordered = sorted(pairs, key=lambda x: -x[0])
    tp = 0
    ap = 0.0
    for idx, (_, y) in enumerate(ordered, 1):
        if y:
            tp += 1
            ap += tp / idx
    return auc, ap / n_pos


def main(weights_resnet: Path, weights_yolo: Path) -> None:
    ann = {}
    for f in sorted((ROOT / "ann").glob("*.json")):
        ann.update(json.loads(f.read_text()))
    truth = {k: (1 if v else 0) for k, v in ann.items()}
    print(f"labelled photos: {len(truth)} ({sum(truth.values())} with damage)")

    signals: dict[str, dict[str, float]] = {}

    clip = json.loads((ROOT / "clip_damage_rank.json").read_text())
    signals["clip"] = {k: clip[k] for k in truth if k in clip}

    from src.parser.photo_damage import DamageClassifier
    clf = DamageClassifier(weights=weights_resnet)
    res = {}
    for k in truth:
        p = ROOT / "images" / f"{k}.jpg"
        if p.exists():
            res[k] = float(clf.predict_photo(p).p_damaged)
    signals["resnet"] = res

    from ultralytics import YOLO
    y = YOLO(str(weights_yolo))
    yo = {}
    for k in truth:
        p = ROOT / "images" / f"{k}.jpg"
        if not p.exists():
            continue
        r = y.predict(str(p), conf=0.02, verbose=False)[0]
        c = r.boxes.conf.tolist() if r.boxes is not None else []
        yo[k] = max(c) if c else 0.0
    signals["yolo"] = yo

    print(f"\n{'signal':8} {'n':>4} {'ROC-AUC':>8} {'avg-prec':>9}  (0.5 = coin flip)")
    scored = {}
    for name, s in signals.items():
        pairs = [(s[k], truth[k]) for k in s]
        auc, ap = _auc(pairs)
        scored[name] = auc
        print(f"{name:8} {len(pairs):>4} {auc:>8.3f} {ap:>9.3f}")

    # A weight is only worth adding if it ranks better than what we already have.
    both = [k for k in signals["resnet"] if k in signals["yolo"] and k in signals["clip"]]
    if both:
        comb = {k: max(signals["resnet"][k], signals["yolo"][k]) for k in both}
        auc, ap = _auc([(comb[k], truth[k]) for k in both])
        print(f"{'max(r,y)':8} {len(both):>4} {auc:>8.3f} {ap:>9.3f}")


def val_only() -> None:
    """Same comparison, restricted to photos the detector never trained on.

    Without this the YOLO column is meaningless: 95 of the 128 annotated photos
    are in its training set, so scoring it on all of them measures memorisation.
    ResNet and CLIP never saw any of these, so only the val slice compares the
    three on equal terms.
    """
    import json
    from pathlib import Path
    val_dir = ROOT / "subset" / "images" / "val"
    val = {p.stem for p in val_dir.glob("*.jpg")}
    ann = {}
    for f in sorted((ROOT / "ann").glob("*.json")):
        ann.update(json.loads(f.read_text()))
    truth = {k: (1 if v else 0) for k, v in ann.items() if k in val}
    print(f"\n--- held-out only: {len(truth)} photos ({sum(truth.values())} with damage)")

    clip = json.loads((ROOT / "clip_damage_rank.json").read_text())
    from src.parser.photo_damage import DamageClassifier
    from ultralytics import YOLO
    clf = DamageClassifier(weights=Path("damage_classifier_v2.pt"))
    y = YOLO("/tmp/yolo_dmg/v4/weights/best.pt")

    sig = {"clip": {}, "resnet": {}, "yolo": {}}
    for k in truth:
        p = ROOT / "images" / f"{k}.jpg"
        if not p.exists():
            continue
        sig["clip"][k] = clip.get(k, 0.0)
        sig["resnet"][k] = float(clf.predict_photo(p).p_damaged)
        r = y.predict(str(p), conf=0.02, verbose=False)[0]
        c = r.boxes.conf.tolist() if r.boxes is not None else []
        sig["yolo"][k] = max(c) if c else 0.0

    print(f"{'signal':8} {'n':>4} {'ROC-AUC':>8} {'avg-prec':>9}")
    for name, s in sig.items():
        auc, ap = _auc([(s[k], truth[k]) for k in s])
        print(f"{name:8} {len(s):>4} {auc:>8.3f} {ap:>9.3f}")
    ks = list(sig["resnet"])
    comb = {k: max(sig["resnet"][k], sig["yolo"][k]) for k in ks}
    auc, ap = _auc([(comb[k], truth[k]) for k in ks])
    print(f"{'max(r,y)':8} {len(ks):>4} {auc:>8.3f} {ap:>9.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--resnet", type=Path,
                    default=Path(__file__).resolve().parent.parent / "damage_classifier_v2.pt")
    ap.add_argument("--yolo", type=Path, default=Path("/tmp/yolo_dmg/v4/weights/best.pt"))
    a = ap.parse_args()
    main(a.resnet, a.yolo)
    val_only()


