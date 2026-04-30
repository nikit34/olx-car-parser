"""Binary car damage classifier — clean vs damaged.

Fine-tunes ResNet50 / EfficientNet-B0 on the DrBimmer dataset organized as
ImageFolder (drbimmer_binary/{train,val,test}/{clean,damaged}/*.jpg).

Run:
  python scripts/train_damage_classifier.py --epochs 15 --backbone resnet50
"""

import argparse
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path,
                    default=Path("/tmp/yolo_data/drbimmer_binary"))
    ap.add_argument("--backbone", default="resnet50",
                    choices=["resnet50", "efficientnet_b0", "efficientnet_b3"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", type=Path,
                    default=Path("/tmp/yolo_data/damage_classifier.pt"))
    ap.add_argument("--loss", default="ce",
                    choices=["ce", "weighted", "focal"],
                    help="ce = CrossEntropy; weighted = inverse-frequency CE; "
                         "focal = focal loss (gamma=2)")
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms, models

    device = torch.device(args.device if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}, backbone: {args.backbone}, imgsz: {args.imgsz}")

    # Standard ImageNet stats — backbone is pretrained on ImageNet.
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    train_tf = transforms.Compose([
        transforms.Resize((args.imgsz + 32, args.imgsz + 32)),
        transforms.RandomCrop(args.imgsz),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((args.imgsz, args.imgsz)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = datasets.ImageFolder(args.data / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(args.data / "val", transform=eval_tf)
    test_ds = datasets.ImageFolder(args.data / "test", transform=eval_tf) \
        if (args.data / "test").exists() else None

    print(f"Classes: {train_ds.classes}")
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}"
          f"  Test: {len(test_ds) if test_ds else 0}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=4, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=4, pin_memory=False)

    # Backbone
    if args.backbone == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif args.backbone == "efficientnet_b0":
        model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    else:  # b3
        model = models.efficientnet_b3(
            weights=models.EfficientNet_B3_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    counts = [0, 0]
    for _, y in train_ds.samples:
        counts[y] += 1
    print(f"Class counts: clean={counts[0]} damaged={counts[1]}")

    if args.loss == "weighted":
        total = sum(counts)
        weights = torch.tensor(
            [total / (2 * c) if c else 1.0 for c in counts],
            dtype=torch.float, device=device)
        print(f"Class weights: clean={weights[0]:.3f} damaged={weights[1]:.3f}")
        criterion = nn.CrossEntropyLoss(weight=weights)
    elif args.loss == "focal":
        gamma = args.focal_gamma

        def focal_loss(logits, target):
            ce = nn.functional.cross_entropy(logits, target, reduction="none")
            pt = torch.exp(-ce)
            return ((1 - pt) ** gamma * ce).mean()

        criterion = focal_loss
        print(f"Focal loss with gamma={gamma}")
    else:
        criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    t0 = time.monotonic()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            train_correct += (out.argmax(1) == y).sum().item()
            train_total += x.size(0)

        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        # Per-class
        from collections import Counter
        cm = Counter()  # (gold, pred) -> count
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                pred = out.argmax(1)
                val_correct += (pred == y).sum().item()
                val_total += x.size(0)
                for g, p in zip(y.cpu().tolist(), pred.cpu().tolist()):
                    cm[(g, p)] += 1

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        # Class index: 0=clean, 1=damaged (alphabetical in ImageFolder)
        tp = cm[(1, 1)]
        fn = cm[(1, 0)]
        fp = cm[(0, 1)]
        tn = cm[(0, 0)]
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0

        print(f"Epoch {epoch:2d}/{args.epochs}  "
              f"train_loss={train_loss / train_total:.4f}  "
              f"train_acc={train_acc:.3f}  "
              f"val_acc={val_acc:.3f}  "
              f"P={precision:.3f}  R={recall:.3f}  "
              f"(TP={tp} FP={fp} FN={fn} TN={tn})  "
              f"lr={scheduler.get_last_lr()[0]:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "backbone": args.backbone,
                "classes": train_ds.classes,
                "imgsz": args.imgsz,
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
                "val_precision": precision,
                "val_recall": recall,
            }, args.out)
            print(f"  ↳ saved best to {args.out}")

        scheduler.step()

    print(f"\nDone in {(time.monotonic() - t0) / 60:.1f} min")
    print(f"Best val acc: {best_val_acc:.3f}")
    print(f"Weights: {args.out}")


if __name__ == "__main__":
    main()
