"""Binary damage classifier inference (production v2).

Loads the ResNet50 classifier trained on DrBimmer + OLX hand-labeled data and
exposes simple per-photo and per-listing prediction helpers.

Production threshold: ``0.20`` per-photo, aggregated by max across a listing's
photos. On the gold benchmark (51 listings, 9 damaged) this yields F1=0.818
with recall=100% (catches all damaged listings).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Production runners place weights in data/ (persistent symlink survives
# across checkouts); local dev typically keeps them at the repo root.
WEIGHTS_CANDIDATES = (
    PROJECT_ROOT / "damage_classifier_v2.pt",
    PROJECT_ROOT / "data" / "damage_classifier_v2.pt",
)


def _resolve_default_weights() -> Path:
    for p in WEIGHTS_CANDIDATES:
        if p.exists():
            return p
    return WEIGHTS_CANDIDATES[0]


DEFAULT_WEIGHTS = _resolve_default_weights()
DEFAULT_THRESHOLD = 0.20

# Listing-level multi-photo agreement rule (issue #2, audit context #1).
#
# The audit on a 100-listing flagged sample (#1) showed ~87% FP rate when
# flagging on ``max(p_damaged) >= 0.20``. The dominant failure mode is one
# weirdly-lit photo (sun glare on dark glossy panels, harsh reflections,
# OOD interior/engine-bay shots) hitting p≈0.99 on an otherwise pristine
# car. Raising the per-photo threshold alone only fixes ~7/20 audited FPs;
# the principled fix is requiring agreement across multiple photos so that
# a single anomalous shot can't condemn a 50-photo dealer listing.
#
# ``DamageClassifier.predict_listing`` uses these to set ``is_damaged``:
#
#     is_damaged = sum(p.p_damaged >= FLAG_PHOTO_THRESHOLD for p in photos)
#                  >= FLAG_MIN_PHOTOS
#
# ``max_p`` semantics are unchanged — still ``max(p.p_damaged for p in photos)``
# — so downstream consumers that read ``photo_damage_p`` from ``llm_extras``
# (alerts, dashboard) keep working unchanged. ``DEFAULT_THRESHOLD`` is also
# untouched; it still gates single-photo callers and ``PhotoPrediction.is_damaged``.
FLAG_MIN_PHOTOS = 2
FLAG_PHOTO_THRESHOLD = 0.30


def is_listing_flagged(extras: dict | None) -> bool:
    """Listing-level damage decision with backward-compat fallback.

    Reads ``photo_damage_flagged`` (the multi-photo agreement field added
    by ``verify-photos`` per issue #2 — production-validated in #1 to drop
    flag rate by 70.6 % vs the old max-rule) when present. For listings
    predating that field, falls back to the v2 max-rule
    (``photo_damage_p >= DEFAULT_THRESHOLD``) so the 6 271 legacy rows
    don't all silently drop out of the deal-blocking path.

    Returns ``False`` for listings with no photo_damage data at all
    (``verify-photos`` hasn't classified them yet, or extras is empty).
    """
    if not extras:
        return False
    new = extras.get("photo_damage_flagged")
    if new is not None:
        return bool(new)
    # Legacy listing (pre-#2 cron): fall back to the v2 max-rule so the
    # behaviour for old rows is exactly what consumers had before.
    p = extras.get("photo_damage_p") or 0.0
    try:
        return float(p) >= DEFAULT_THRESHOLD
    except (TypeError, ValueError):
        return False


@dataclass
class PhotoPrediction:
    path: Path
    p_damaged: float
    is_damaged: bool


@dataclass
class ListingPrediction:
    olx_id: str
    photos: list[PhotoPrediction]
    max_p: float
    is_damaged: bool


class DamageClassifier:
    """Single-instance classifier wrapper. Reuse one instance across photos."""

    def __init__(
        self,
        weights: Path | str = DEFAULT_WEIGHTS,
        device: str | None = None,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)
        self.threshold = threshold

        ckpt = torch.load(weights, map_location=self.device, weights_only=False)
        backbone = ckpt["backbone"]
        if backbone == "resnet50":
            model = models.resnet50(weights=None)
            model.fc = nn.Linear(model.fc.in_features, 2)
        elif backbone == "efficientnet_b0":
            model = models.efficientnet_b0(weights=None)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
        elif backbone == "efficientnet_b3":
            model = models.efficientnet_b3(weights=None)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        model.load_state_dict(ckpt["state_dict"])
        model.eval().to(self.device)
        self.model = model
        self.classes = ckpt["classes"]
        self.imgsz = ckpt["imgsz"]

        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        self.tf = transforms.Compose([
            transforms.Resize((self.imgsz, self.imgsz)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        # MPS isn't thread-safe — verify-photos runs N worker threads that
        # each call ``predict_listing`` concurrently. Without serialization,
        # overlapping forward passes on a single MPS-resident model crash
        # the runner with ``Trace/BPT trap: 5`` (SIGTRAP, exit 133).
        # Photo I/O still parallelises across threads; only the inference
        # block is locked.
        import threading
        self._inference_lock = threading.Lock()

    def predict_photo(self, path: Path | str) -> PhotoPrediction:
        path = Path(path)
        img = Image.open(path).convert("RGB")
        x = self.tf(img).unsqueeze(0).to(self.device)
        with self._inference_lock, torch.no_grad():
            prob = torch.softmax(self.model(x), dim=1)[0]
        p_damaged = float(prob[1].item())
        return PhotoPrediction(path, p_damaged, p_damaged >= self.threshold)

    def predict_photos_batch(
        self, paths: Iterable[Path | str], batch_size: int = 16
    ) -> list[PhotoPrediction]:
        paths = [Path(p) for p in paths]
        out: list[PhotoPrediction] = []
        for i in range(0, len(paths), batch_size):
            chunk = paths[i:i + batch_size]
            tensors = torch.stack([self.tf(Image.open(p).convert("RGB")) for p in chunk])
            tensors = tensors.to(self.device)
            with self._inference_lock, torch.no_grad():
                probs = torch.softmax(self.model(tensors), dim=1)[:, 1].tolist()
            out.extend(
                PhotoPrediction(p, float(prob), float(prob) >= self.threshold)
                for p, prob in zip(chunk, probs)
            )
        return out

    def predict_listing(
        self, olx_id: str, photo_paths: Iterable[Path | str]
    ) -> ListingPrediction:
        photos = self.predict_photos_batch(photo_paths)
        max_p = max((p.p_damaged for p in photos), default=0.0)
        # Listing-level rule: require multi-photo agreement (issue #2). The
        # per-photo ``self.threshold`` (used by ``PhotoPrediction.is_damaged``
        # and single-photo callers) is intentionally separate from the
        # listing aggregation rule — see ``FLAG_MIN_PHOTOS`` /
        # ``FLAG_PHOTO_THRESHOLD`` docstring above.
        n_above = sum(1 for p in photos if p.p_damaged >= FLAG_PHOTO_THRESHOLD)
        is_damaged = n_above >= FLAG_MIN_PHOTOS
        return ListingPrediction(olx_id, photos, max_p, is_damaged)
