"""POC: detect car damage from listing photos via local VLM.

Compares the photo-based assessment against the text-based damage_severity
that our existing qwen3:4b-instruct extracts. Run on a single listing URL:

    python scripts/photo_damage_poc.py <listing-url>

Pipeline:
  1. Fetch the standvirtual / OLX listing page.
  2. Extract photos and description from __NEXT_DATA__ (standvirtual) or
     equivalent state blob (OLX).
  3. Download up to N photos (default 4) at full size.
  4. Run text-side damage_severity via the existing qwen3 prompt.
  5. Run photo-side damage assessment via qwen2.5-vl:3b — one call per
     photo, emitting JSON with severity (0-3) and a free-text evidence
     string.
  6. Aggregate per-photo results into a single severity (max) and print
     a side-by-side comparison.
"""

import argparse
import base64
import json
import sys
import time
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.parser.llm_enrichment import enrich_from_description
from src.parser.photo_fetch import download_photo, fetch_standvirtual_advert

OLLAMA_URL = "http://localhost:11434"
VLM_MODEL = "qwen2.5vl:3b"

VLM_SYSTEM = """\
You are inspecting one photo of a used car listing in Portugal. Output ONE JSON object only.

Schema:
  visible_damage: bool — true if the photo shows ANY of: dents, scratches deeper than buffing, broken/cracked panels or lights, rust, peeling paint, missing parts, deflated tyres, deployed airbags, bent frame, fluid leaks under car, smashed glass.
  severity: int 0-3
    0 = no visible damage / clean panels
    1 = minor cosmetic (small scratches, light scuffs, minor paint chips)
    2 = significant cosmetic or moderate mechanical (clear dents, rust patches, cracked plastic, mismatched paint)
    3 = major / structural / non-runner (smashed panels, bent frame, deployed airbags, fire damage, salvage)
  damage_areas: list of short labels e.g. ["front bumper","left fender","rear quarter"], [] if none.
  evidence: 1-2 short sentences in English describing what you see.
  is_studio_or_dealer: bool — true if the shot is a clean dealer / studio photo with controlled lighting / showroom floor (suggests minor selection bias toward presentable cars).

If the photo is too blurry / dark / cropped to assess, set visible_damage=false, severity=0, evidence="image not assessable".
"""


def fetch_listing(url: str) -> dict:
    """Return {description, photos: [url,...], title}."""
    ad = fetch_standvirtual_advert(url)
    if not ad:
        raise RuntimeError("__NEXT_DATA__ not found — selector for OLX needs work")
    photos = [p["url"] for p in ad["images"]["photos"]]
    return {
        "title": ad.get("title", ""),
        "description": ad.get("description", "") or "",
        "photos": photos,
        "olx_id": ad.get("id"),
    }


def call_vlm(img_path: Path, model: str = VLM_MODEL) -> dict | None:
    """One vision-LLM call against /api/chat with the image attached."""
    img_b64 = base64.b64encode(img_path.read_bytes()).decode()
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": VLM_SYSTEM},
            {
                "role": "user",
                "content": "Inspect this photo and output the JSON object.",
                "images": [img_b64],
            },
        ],
        "format": "json",
        "stream": False,
        "keep_alive": "10m",
        "options": {"temperature": 0.0, "num_ctx": 4096, "num_predict": 350},
    }
    t0 = time.monotonic()
    r = httpx.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=180)
    dt = time.monotonic() - t0
    if r.status_code != 200:
        return {"_error": f"http {r.status_code}", "_latency_s": dt}
    content = r.json().get("message", {}).get("content", "")
    try:
        out = json.loads(content)
    except json.JSONDecodeError:
        return {"_error": "non-json", "_raw": content[:300], "_latency_s": dt}
    out["_latency_s"] = round(dt, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url", help="Listing URL")
    # Sellers routinely place the cleanest exterior shot first and bury the
    # damage panels mid-gallery. Defaulting to the full set means a single
    # battered fender 8 photos in still triggers the comparison. Cap kept
    # as an override for quick smoke tests.
    ap.add_argument("--max-photos", type=int, default=0,
                    help="Cap photos analysed (0 = all, default)")
    ap.add_argument("--cache-dir", default="/tmp/olx_photo_poc")
    args = ap.parse_args()

    listing = fetch_listing(args.url)
    all_photos = listing["photos"]
    photos = all_photos if args.max_photos <= 0 else all_photos[: args.max_photos]
    print(f"Title    : {listing['title']}")
    print(f"OLX id   : {listing['olx_id']}")
    print(f"Photos   : {len(all_photos)} (analysing all)" if args.max_photos <= 0
          else f"Photos   : {len(all_photos)} (using first {len(photos)})")
    print(f"Desc head: {listing['description'][:150]!r}\n")

    # ------ Text-side damage assessment ------
    print("--- text-side (qwen3:4b-instruct) ---")
    t0 = time.monotonic()
    text_result = enrich_from_description(listing["description"], listing["title"])
    text_dt = time.monotonic() - t0
    if text_result:
        keys = ["damage_severity", "desc_mentions_accident",
                "desc_mentions_repair", "mechanical_condition"]
        for k in keys:
            print(f"  {k:30s} = {text_result.get(k)!r}")
    else:
        print("  (no text result)")
    print(f"  latency = {text_dt:.1f}s\n")

    # ------ Photo-side damage assessment ------
    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    print("--- photo-side (qwen2.5vl:3b) ---")
    photo_results = []
    for i, url in enumerate(photos, 1):
        local = cache / f"{listing['olx_id']}_{i}.jpg"
        if not download_photo(url, local):
            raise RuntimeError(f"Failed to download {url}")
        size_kb = local.stat().st_size // 1024
        print(f"\n  [{i}/{len(photos)}] {local.name} ({size_kb} KB)")
        result = call_vlm(local)
        photo_results.append(result)
        if not result or "_error" in result:
            print(f"    error: {result}")
            continue
        for k in ["visible_damage", "severity", "damage_areas",
                  "evidence", "is_studio_or_dealer"]:
            v = result.get(k)
            print(f"    {k:22s} = {v!r}")
        print(f"    latency               = {result.get('_latency_s')}s")

    # ------ Aggregate ------
    indexed = [(i, r) for i, r in enumerate(photo_results, 1)
               if r and "_error" not in r]
    if indexed:
        worst_idx, worst = max(indexed, key=lambda x: x[1].get("severity", 0))
        photo_severity = worst.get("severity", 0)
        any_damage = any(r.get("visible_damage") for _, r in indexed)
        all_areas = sorted({a for _, r in indexed
                            for a in (r.get("damage_areas") or [])})
        any_studio = any(r.get("is_studio_or_dealer") for _, r in indexed)
        # Per-photo severity histogram makes it obvious whether one photo is
        # an outlier (likely a damage shot) or every photo flags issues
        # (consistent wear, no surprise).
        sev_counts = {s: sum(1 for _, r in indexed if r.get("severity") == s)
                      for s in (0, 1, 2, 3)}
    else:
        photo_severity, any_damage, all_areas, any_studio = None, None, [], None
        worst_idx, worst, sev_counts = None, None, {}

    print("\n--- comparison ---")
    print(f"  text  damage_severity  : {text_result.get('damage_severity') if text_result else None}")
    print(f"  photo damage_severity  : {photo_severity}  (max across {len(indexed)} photos)")
    if worst is not None and photo_severity:
        print(f"  worst photo            : #{worst_idx}  areas={worst.get('damage_areas')}")
        print(f"    evidence             : {worst.get('evidence')!r}")
    if sev_counts:
        hist = " | ".join(f"sev{s}: {sev_counts[s]}" for s in (0, 1, 2, 3))
        print(f"  severity histogram     : {hist}")
    print(f"  photo any_visible_damage: {any_damage}")
    print(f"  photo damage_areas     : {all_areas}")
    print(f"  photo any_studio_shot  : {any_studio}")

    if text_result and photo_severity is not None:
        delta = photo_severity - (text_result.get("damage_severity") or 0)
        if delta >= 1:
            print(f"\n  >>> photo severity is {delta} higher than text — "
                  "possible undisclosed damage <<<")
        elif delta <= -1:
            print(f"\n  >>> photo severity is {-delta} lower than text — "
                  "text may exaggerate or photos selectively chosen <<<")
        else:
            print("\n  agreement.")


if __name__ == "__main__":
    main()
