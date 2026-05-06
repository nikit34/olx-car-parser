"""Compare qwen2.5vl:3b photo-damage outputs against the Claude-labeled
gold standard.

Reads the gold JSONL (one listing per line, with my hand-labels per photo),
runs qwen2.5vl:3b on every photo path referenced, and emits:

  - per-photo agreement table
  - per-listing severity comparison
  - dataset-level Spearman correlation
  - precision / recall for "visible_damage" binary task
  - confusion stats for severity buckets

Usage:
  python scripts/photo_damage_compare.py \
      --gold /tmp/photo_label/gold.jsonl \
      --out  /tmp/photo_label/compare.json
"""

import argparse
import base64
import json
import sys
import time
from collections import Counter
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OLLAMA_URL = "http://localhost:11434"
DEFAULT_VLM_MODEL = "qwen2.5vl:3b"

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
  is_studio_or_dealer: bool — true if the shot is a clean dealer / studio photo with controlled lighting / showroom floor.

If the photo is too blurry / dark / cropped to assess, set visible_damage=false, severity=0, evidence="image not assessable".
"""


def call_vlm(img_path: Path, model: str = DEFAULT_VLM_MODEL) -> dict | None:
    img_b64 = base64.b64encode(img_path.read_bytes()).decode()
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": VLM_SYSTEM},
            {"role": "user",
             "content": "Inspect this photo and output the JSON object.",
             "images": [img_b64]},
        ],
        "format": "json",
        "stream": False,
        "keep_alive": "10m",
        "options": {"temperature": 0.0, "num_ctx": 4096, "num_predict": 350},
    }
    t0 = time.monotonic()
    try:
        r = httpx.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=180)
    except httpx.RequestError as e:
        return {"_error": str(e), "_latency_s": time.monotonic() - t0}
    dt = time.monotonic() - t0
    if r.status_code != 200:
        return {"_error": f"http {r.status_code}", "_latency_s": dt}
    content = r.json().get("message", {}).get("content", "")
    try:
        out = json.loads(content)
    except json.JSONDecodeError:
        return {"_error": "non-json", "_raw": content[:200], "_latency_s": dt}
    out["_latency_s"] = round(dt, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", type=Path, required=True)
    ap.add_argument("--manifest", type=Path,
                    default=Path("/tmp/photo_label/manifest.jsonl"),
                    help="Manifest with photo paths (from sampler)")
    ap.add_argument("--out", type=Path,
                    default=Path("/tmp/photo_label/compare.json"))
    ap.add_argument("--model", default=DEFAULT_VLM_MODEL,
                    help="Ollama model tag, e.g. qwen2.5vl:3b or qwen2.5vl:7b")
    args = ap.parse_args()

    gold_lines = [json.loads(l) for l in args.gold.read_text().splitlines() if l]
    manifest_lines = [json.loads(l) for l in args.manifest.read_text().splitlines() if l]
    manifest_by_id = {m["olx_id"]: m for m in manifest_lines}

    print(f"Gold: {len(gold_lines)} listings, "
          f"{sum(len(g['photos']) for g in gold_lines)} photo labels")

    rows = []
    for g in gold_lines:
        olx_id = g["olx_id"]
        man = manifest_by_id.get(olx_id)
        if not man:
            print(f"  ! no manifest for {olx_id}")
            continue
        path_by_idx = {p["idx"]: p["path"] for p in man["photos"]}

        print(f"\n[{olx_id}] {g['title'][:50]}  ({len(g['photos'])} photos)")
        for gp in g["photos"]:
            idx = gp["idx"]
            path = Path(path_by_idx.get(idx, ""))
            if not path.exists():
                print(f"  photo {idx}: file missing")
                continue
            vlm = call_vlm(path, model=args.model)
            row = {
                "olx_id": olx_id,
                "photo_idx": idx,
                "gold_visible_damage": gp["visible_damage"],
                "gold_severity": gp["severity"],
                "gold_damage_classes": gp["damage_classes"],
                "gold_damage_areas": gp["damage_areas"],
                "vlm": vlm,
            }
            rows.append(row)
            sev_v = vlm.get("severity") if vlm and "_error" not in vlm else None
            vd_v = vlm.get("visible_damage") if vlm and "_error" not in vlm else None
            mark = "✓" if sev_v == gp["severity"] else "✗"
            print(f"  photo {idx:2d}  gold sev={gp['severity']} damage={gp['visible_damage']!s:5s}"
                  f"  | vlm sev={sev_v!s:4s} damage={vd_v!s:5s}  {mark}"
                  f"  ({(vlm or {}).get('_latency_s')}s)")

    args.out.write_text(json.dumps(rows, indent=2))
    print(f"\nSaved {len(rows)} rows to {args.out}")

    # --- Aggregates ---
    print("\n" + "=" * 60)
    print("AGGREGATES")
    print("=" * 60)

    # Photo-level severity agreement
    valid = [r for r in rows if r["vlm"] and "_error" not in r["vlm"]
             and "severity" in r["vlm"]]
    if not valid:
        print("\nNo valid VLM results to aggregate.")
        return

    gold_sev = [r["gold_severity"] for r in valid]
    vlm_sev = [r["vlm"]["severity"] for r in valid]

    exact = sum(1 for g, v in zip(gold_sev, vlm_sev) if g == v)
    within1 = sum(1 for g, v in zip(gold_sev, vlm_sev) if abs(g - v) <= 1)
    print(f"\nPer-photo severity agreement (n={len(valid)}):")
    print(f"  exact match : {exact}/{len(valid)} = {exact / len(valid):.1%}")
    print(f"  within ±1   : {within1}/{len(valid)} = {within1 / len(valid):.1%}")

    try:
        from scipy.stats import spearmanr
        rho, p = spearmanr(gold_sev, vlm_sev)
        print(f"  Spearman ρ  : {rho:+.3f}  (p={p:.3e})")
    except ImportError:
        pass

    # Confusion matrix
    print("\nPer-photo confusion (rows=gold, cols=vlm):")
    print("           vlm=0  vlm=1  vlm=2  vlm=3")
    cm = Counter((g, v) for g, v in zip(gold_sev, vlm_sev))
    for g in (0, 1, 2, 3):
        row = "  gold={:d} ".format(g)
        for v in (0, 1, 2, 3):
            row += f"  {cm.get((g, v), 0):4d} "
        print(row)

    # Binary visible_damage task
    gold_vd = [r["gold_visible_damage"] for r in valid]
    vlm_vd = [bool(r["vlm"].get("visible_damage")) for r in valid]
    tp = sum(1 for g, v in zip(gold_vd, vlm_vd) if g and v)
    fp = sum(1 for g, v in zip(gold_vd, vlm_vd) if not g and v)
    fn = sum(1 for g, v in zip(gold_vd, vlm_vd) if g and not v)
    tn = sum(1 for g, v in zip(gold_vd, vlm_vd) if not g and not v)
    print(f"\nPer-photo visible_damage (binary):")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    if tp + fp > 0:
        print(f"  precision = {tp / (tp + fp):.2%}")
    if tp + fn > 0:
        print(f"  recall    = {tp / (tp + fn):.2%}")
    print(f"  accuracy  = {(tp + tn) / len(valid):.2%}")

    # Per-listing aggregate (max severity)
    print("\nPer-listing max-severity comparison:")
    by_listing: dict[str, dict] = {}
    for r in valid:
        oid = r["olx_id"]
        d = by_listing.setdefault(oid, {"gold": 0, "vlm": 0})
        d["gold"] = max(d["gold"], r["gold_severity"])
        d["vlm"] = max(d["vlm"], r["vlm"]["severity"])
    for g in gold_lines:
        oid = g["olx_id"]
        if oid not in by_listing:
            continue
        d = by_listing[oid]
        text_sev = g["text_severity"]
        delta = d["vlm"] - d["gold"]
        delta_str = f"{delta:+d}" if delta else "ok"
        print(f"  {oid}  text={text_sev}  gold={d['gold']}  vlm={d['vlm']}  Δ(vlm-gold)={delta_str}")


if __name__ == "__main__":
    main()
