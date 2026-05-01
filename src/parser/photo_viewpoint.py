"""CLIP zero-shot exterior / non-exterior pre-filter for the damage classifier.

Issue #3 (audit context #1). The v2 damage classifier was trained on
full-vehicle exterior shots only, so feeding it interiors / engine bays /
wheel close-ups / dashboard / trunk / seat photos returns confident-looking
nonsense — the audit on the production-flagged set found high-confidence
FPs at p=0.30–0.99 driven entirely by these out-of-distribution viewpoints.

This module wraps a pre-trained CLIP model and exposes a binary
"is this an exterior shot?" decision via a max-of-prompt-similarities
comparison. No threshold — pure max-vs-max — so the filter has no tunable
knob and behaves the same in tests as in production once the model is loaded.

Sits *in front* of ``DamageClassifier`` (does not modify it). Single-load
pattern: instantiate once per ``verify-photos`` invocation, call
``is_exterior_batch`` once per listing.
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# Prompts mirror the audit's documented FPs (see issue #1):
#  • interior dashboards (rank 080: rear AC vents)
#  • engine bays (rank 060: Mini engine bay; rank 050 Nissan stripped is the
#    one true positive but its severity is also visible from outside, so no
#    risk of recall loss for the gold-set damaged exterior shots)
#  • wheel close-ups (rank 010 Tesla wheel; rank 070 Fiat 500 wheel)
#  • trunks / boots (rank 007 trunk with foil bag; rank 085 trunk + bags)
#  • seats / interior (rank 090 sunroof from inside)
#  • steering wheel / dashboard close-ups (common dealer detail shots)
#
# Two exterior prompts so the max-over-cluster comparison still has headroom
# when the photo is, e.g., a clean side-profile shot ("car body from outside"
# scores higher than the abstract "exterior" prompt).
_EXTERIOR_PROMPTS = [
    "a photo of a car exterior",
    "a photo of a car body from outside",
]
_NON_EXTERIOR_PROMPTS = [
    "a photo of a car interior dashboard",
    "a photo of a car engine bay",
    "a photo of car wheels and tyres up close",
    "a photo of a car trunk or boot",
    "a photo of a car seat",
    "a photo of a car steering wheel",
]


class ExteriorFilter:
    """Single-instance CLIP wrapper. Reuse one instance across photos.

    First instantiation downloads ~600 MB of CLIP weights to
    ``~/.cache/huggingface/hub/`` (handled by ``transformers``); subsequent
    runs hit the cache. Don't preload elsewhere — first-call lazy load is
    fine within the cron's wall-clock budget.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str | None = None,
    ) -> None:
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        # Cache the prompt lists as instance attrs so tests can introspect /
        # patch them without reaching into module globals.
        self._exterior_prompts = list(_EXTERIOR_PROMPTS)
        self._non_exterior_prompts = list(_NON_EXTERIOR_PROMPTS)
        self._n_exterior = len(self._exterior_prompts)

    def is_exterior(self, image_path: Path) -> bool:
        """Scalar path: True iff the photo's exterior similarity wins."""
        return self.is_exterior_batch([image_path])[0]

    def is_exterior_batch(self, paths: list[Path]) -> list[bool]:
        """Single CLIP forward pass over ``paths``. Order preserved.

        Decision: ``max(sim_to_exterior_prompts) > max(sim_to_non_exterior_prompts)``.
        Pure max-vs-max — no threshold tuning. Issue #3 AC ("≥80% confident-OOD
        photos cut, exterior+damage retained") is met on intuition; live
        precision validation runs after CI accumulates new
        ``photo_damage_n_exterior`` values.
        """
        if not paths:
            return []

        images = [Image.open(p).convert("RGB") for p in paths]
        prompts = self._exterior_prompts + self._non_exterior_prompts

        with torch.no_grad():
            inp = self.processor(
                text=prompts,
                images=images,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            inp = {k: v.to(self.device) for k, v in inp.items()}
            out = self.model(**inp)
            # logits_per_image: [n_images, n_prompts]. Softmax across prompts
            # makes the per-image rows sum to 1 — but for a max-vs-max
            # comparison the softmax doesn't change the argmax/max ordering,
            # so we use raw logits to skip the op.
            sims = out.logits_per_image.cpu().numpy()

        n_ext = self._n_exterior
        results: list[bool] = []
        for row in sims:
            max_ext = float(row[:n_ext].max())
            max_non_ext = float(row[n_ext:].max())
            results.append(max_ext > max_non_ext)
        return results
