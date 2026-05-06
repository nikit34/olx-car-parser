"""POC: rank listing photos by damage-likelihood with zero-shot CLIP.

Idea: embed every photo, embed a tiny prompt set covering damage vs clean
states, score each photo by ``max(damage prompt sims) - max(clean prompt
sims)``. Photos that the VLM would flag as severity≥2 should bubble to the
top of this ranking; if they do, CLIP can serve as a fast triage stage
before the slow VLM (the 12-photo Passat run took ~140 s on qwen2.5vl:3b
vs. ~2 s here).

Run:
    python scripts/photo_damage_clip_triage.py <listing-url>

The script also accepts ``--ground-truth-from-vlm`` to run the existing
photo_damage_poc VLM in parallel and print correlation stats — useful
when validating CLIP rankings on known cases.
"""

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.parser.photo_fetch import (  # noqa: E402
    download_photos as _download_photos,
    fetch_standvirtual_advert,
)

# Prompt set: damage states first, then "clean" / non-exterior context.
# Two clusters; each photo's "damage score" = max(sim to damage cluster)
# minus max(sim to clean cluster). Positive ⇒ leans damaged.
DAMAGE_PROMPTS = [
    "a photo of a car with major crash damage",
    "a photo of a car with a smashed front bumper",
    "a photo of a car with a broken windshield",
    "a photo of a car with a deployed airbag",
    "a photo of a car body panel with a deep dent",
    "a photo of a car with rust and peeling paint",
    "a photo of a wrecked or salvage car for parts",
]
CLEAN_PROMPTS = [
    "a photo of a clean used car for sale",
    "a photo of an undamaged car exterior",
    "a photo of a car interior dashboard",
    "a photo of a car engine bay",
    "a photo of car wheels and tyres",
]


def fetch_listing(url: str) -> dict:
    ad = fetch_standvirtual_advert(url)
    if not ad:
        raise RuntimeError("__NEXT_DATA__ not found")
    return {
        "olx_id": ad.get("id"),
        "title": ad.get("title", ""),
        "photos": [p["url"] for p in ad["images"]["photos"]],
    }


def download_photos(urls: list[str], olx_id: str, cache_dir: Path) -> list[Path]:
    return _download_photos(urls, olx_id, cache_dir)


def score_with_clip(image_paths: list[Path],
                    model_name: str = "openai/clip-vit-base-patch32"):
    """Return (per-photo damage_scores, raw_sims_matrix) using CLIP."""
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    print(f"Loading {model_name}…")
    t0 = time.monotonic()
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()
    print(f"  loaded in {time.monotonic() - t0:.1f}s")

    images = [Image.open(p).convert("RGB") for p in image_paths]
    prompts = DAMAGE_PROMPTS + CLEAN_PROMPTS
    n_dmg = len(DAMAGE_PROMPTS)

    t0 = time.monotonic()
    with torch.no_grad():
        inp = processor(text=prompts, images=images, return_tensors="pt",
                        padding=True, truncation=True)
        out = model(**inp)
        # logits_per_image: [n_images, n_prompts]; CLIP returns scaled logits,
        # but for ranking we just need the relative magnitudes — softmax across
        # prompts gives a probability that sums to 1 per image.
        sims = out.logits_per_image.softmax(dim=-1).cpu().numpy()
    inf_dt = time.monotonic() - t0
    print(f"  encoded {len(images)} photos × {len(prompts)} prompts in "
          f"{inf_dt:.2f}s ({inf_dt / max(len(images), 1) * 1000:.0f} ms/photo)")

    damage_scores = []
    for row in sims:
        max_dmg = float(row[:n_dmg].max())
        max_clean = float(row[n_dmg:].max())
        damage_scores.append(max_dmg - max_clean)
    return damage_scores, sims, prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--cache-dir", default="/tmp/olx_photo_poc")
    ap.add_argument("--model", default="openai/clip-vit-base-patch32")
    ap.add_argument("--ground-truth", type=Path, default=None,
                    help="Optional path to JSON dict {photo_index: severity} "
                         "for correlation reporting (e.g. dumped from "
                         "photo_damage_poc.py output)")
    args = ap.parse_args()

    listing = fetch_listing(args.url)
    photos = listing["photos"]
    print(f"Title : {listing['title']}")
    print(f"OLX id: {listing['olx_id']}")
    print(f"Photos: {len(photos)}\n")

    paths = download_photos(photos, listing["olx_id"], Path(args.cache_dir))
    damage_scores, sims, prompts = score_with_clip(paths, args.model)

    ranked = sorted(enumerate(damage_scores, 1),
                    key=lambda x: x[1], reverse=True)

    print("\n--- ranking (highest damage-likelihood first) ---")
    n_dmg = len(DAMAGE_PROMPTS)
    for rank, (idx, score) in enumerate(ranked, 1):
        row = sims[idx - 1]
        top_dmg_i = int(row[:n_dmg].argmax())
        top_clean_i = int(row[n_dmg:].argmax()) + n_dmg
        print(f"  rank {rank:2d}  photo #{idx:2d}  score={score:+.3f}"
              f"   top_dmg=\"{prompts[top_dmg_i][:50]}\""
              f"   top_clean=\"{prompts[top_clean_i][:50]}\"")

    if args.ground_truth and args.ground_truth.exists():
        gt = {int(k): v for k, v in
              json.loads(args.ground_truth.read_text()).items()}
        print("\n--- ground-truth correlation ---")
        # Spearman correlation between CLIP score and VLM severity, plus
        # precision@K for "are the top K CLIP photos actually damaged?".
        try:
            from scipy.stats import spearmanr
            pairs = [(damage_scores[i - 1], gt[i]) for i in gt]
            rho, p = spearmanr([x[0] for x in pairs], [x[1] for x in pairs])
            print(f"  Spearman ρ (CLIP score vs VLM severity) = {rho:+.3f} (p={p:.3f})")
        except ImportError:
            pass
        damaged_idx = {i for i, sev in gt.items() if sev >= 2}
        for k in (3, 5, len(damaged_idx)):
            top_k = {idx for idx, _ in ranked[:k]}
            hit = len(top_k & damaged_idx)
            print(f"  precision@{k} (damaged in top-{k}) = "
                  f"{hit}/{k} ({hit / k * 100:.0f}%)")


if __name__ == "__main__":
    main()
