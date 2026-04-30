"""Evaluate the trained binary damage classifier on our gold-standard
OLX photos. Compares against gold's per-photo visible_damage and
listing-level aggregate.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_model(weights_path: Path, device: torch.device):
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    backbone = ckpt["backbone"]
    if backbone == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    else:
        model = models.efficientnet_b3(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)
    return model, ckpt["classes"], ckpt["imgsz"]


def predict_one(model, tf, device, path: Path) -> tuple[float, int]:
    img = Image.open(path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits, dim=1)[0]
    p_damaged = float(prob[1].item())
    return p_damaged, int(prob.argmax().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path,
                    default=PROJECT_ROOT / "damage_classifier_v2.pt")
    ap.add_argument("--gold", type=Path, required=True)
    ap.add_argument("--manifest", type=Path,
                    default=Path("/tmp/photo_label/manifest.jsonl"))
    ap.add_argument("--out", type=Path,
                    default=Path("/tmp/photo_label/compare_classifier.json"))
    ap.add_argument("--threshold", type=float, default=0.20,
                    help="P(damaged) threshold for binary decision "
                         "(production default 0.20 — listing-level F1=0.818, "
                         "recall=100%% on gold)")
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    model, classes, imgsz = load_model(args.weights, device)
    print(f"Classes: {classes}, imgsz: {imgsz}")

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    tf = transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    gold = [json.loads(l) for l in args.gold.read_text().splitlines() if l]
    manifest = {m["olx_id"]: m
                for m in [json.loads(l) for l in args.manifest.read_text().splitlines() if l]}

    rows = []
    for g in gold:
        olx_id = g["olx_id"]
        man = manifest.get(olx_id)
        if not man:
            continue
        path_by_idx = {p["idx"]: p["path"] for p in man["photos"]}
        for gp in g["photos"]:
            idx = gp["idx"]
            path = Path(path_by_idx.get(idx, ""))
            if not path.exists():
                continue
            try:
                p_dam, pred = predict_one(model, tf, device, path)
            except Exception as e:
                print(f"  err {olx_id}_{idx}: {e}")
                continue
            rows.append({
                "olx_id": olx_id,
                "photo_idx": idx,
                "gold_visible_damage": gp["visible_damage"],
                "gold_severity": gp["severity"],
                "p_damaged": round(p_dam, 4),
                "pred": pred,
            })

    args.out.write_text(json.dumps(rows, indent=2))
    print(f"\nSaved {len(rows)} rows to {args.out}\n")

    # Aggregates
    print("=" * 60)
    print(f"AGGREGATES (threshold = {args.threshold})")
    print("=" * 60)

    gold_vd = [r["gold_visible_damage"] for r in rows]
    pred_vd = [r["p_damaged"] >= args.threshold for r in rows]
    tp = sum(1 for g, p in zip(gold_vd, pred_vd) if g and p)
    fp = sum(1 for g, p in zip(gold_vd, pred_vd) if not g and p)
    fn = sum(1 for g, p in zip(gold_vd, pred_vd) if g and not p)
    tn = sum(1 for g, p in zip(gold_vd, pred_vd) if not g and not p)

    print(f"\nPer-photo binary visible_damage:")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  precision = {tp / (tp + fp):.2%}" if tp + fp else "  precision = N/A")
    print(f"  recall    = {tp / (tp + fn):.2%}" if tp + fn else "  recall    = N/A")
    print(f"  accuracy  = {(tp + tn) / len(rows):.2%}")
    if tp + fp and tp + fn:
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
        print(f"  F1        = {f1:.3f}")

    # Threshold sweep
    print("\n--- threshold sweep ---")
    print(f"{'thresh':>8s} {'TP':>4s} {'FP':>4s} {'FN':>4s} {'P':>6s} {'R':>6s} {'F1':>6s}")
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        pred = [r["p_damaged"] >= t for r in rows]
        tp = sum(1 for g, p in zip(gold_vd, pred) if g and p)
        fp = sum(1 for g, p in zip(gold_vd, pred) if not g and p)
        fn = sum(1 for g, p in zip(gold_vd, pred) if g and not p)
        prec = tp / (tp + fp) if tp + fp else 0
        rec = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
        print(f"{t:8.2f} {tp:4d} {fp:4d} {fn:4d} {prec:6.2%} {rec:6.2%} {f1:6.3f}")

    # Per-listing aggregate (max p_damaged across photos)
    print("\n--- per-listing (max p_damaged across photos) ---")
    by_listing: dict[str, dict] = {}
    for r in rows:
        d = by_listing.setdefault(r["olx_id"], {"max_p": 0.0})
        d["max_p"] = max(d["max_p"], r["p_damaged"])
    print(f"{'olx_id':10s} {'gold_sev':>8s} {'max_p':>7s}  flagged@0.5? title")
    for g in gold:
        oid = g["olx_id"]
        if oid not in by_listing:
            continue
        max_p = by_listing[oid]["max_p"]
        gold_sev = g["listing_aggregate"]["max_severity"]
        flagged = max_p >= 0.5
        is_damaged = gold_sev >= 1
        mark = "✓" if (flagged and is_damaged) or (not flagged and not is_damaged) else "✗"
        print(f"{oid:10s}  {gold_sev:>4d}     {max_p:6.3f}     {flagged!s:5s}  {mark}  {g['title'][:40]}")


if __name__ == "__main__":
    main()
