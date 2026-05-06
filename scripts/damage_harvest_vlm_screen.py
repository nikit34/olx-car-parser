"""Run qwen2.5vl on each harvested photo and write per-photo VLM predictions.

Reads /tmp/damage_harvest/manifest.jsonl, calls qwen2.5vl:3b for every
photo, and writes /tmp/damage_harvest/vlm_screen.json — a flat list of
{olx_id, idx, path, vlm:{visible_damage, severity, damage_areas, evidence}}.
"""

from __future__ import annotations

import argparse
import base64
import json
import time
from pathlib import Path

import httpx

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5vl:3b"

VLM_SYSTEM = """\
You are inspecting one photo of a used car listing in Portugal. Output ONE JSON object only.

Schema:
  visible_damage: bool
  severity: int 0-3
    0 = no visible damage / clean panels
    1 = minor cosmetic (small scratches, light scuffs, minor paint chips)
    2 = significant cosmetic or moderate mechanical (clear dents, rust patches, cracked plastic, mismatched paint)
    3 = major / structural / non-runner (smashed panels, bent frame, deployed airbags, fire damage, salvage)
  damage_areas: list of short labels e.g. ["front bumper","left fender"], [] if none.
  evidence: 1-2 short sentences in English describing what you see.
  is_studio_or_dealer: bool

If unassessable, set visible_damage=false, severity=0, evidence="image not assessable".
"""


def call_vlm(img_path: Path, model: str) -> dict:
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
        out["_latency_s"] = round(dt, 1)
        return out
    except json.JSONDecodeError:
        return {"_error": "non-json", "_raw": content[:200], "_latency_s": dt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path,
                    default=Path("/tmp/damage_harvest/manifest.jsonl"))
    ap.add_argument("--out", type=Path,
                    default=Path("/tmp/damage_harvest/vlm_screen.json"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    manifest = [json.loads(l) for l in args.manifest.read_text().splitlines() if l]
    rows: list[dict] = []
    n_photos = sum(len(m["photos"]) for m in manifest)
    print(f"Manifest: {len(manifest)} listings, {n_photos} photos. Model: {args.model}")

    done = 0
    for m in manifest:
        print(f"\n[{m['olx_id']}] {m['title'][:60]}  ({len(m['photos'])} photos)")
        for p in m["photos"]:
            done += 1
            res = call_vlm(Path(p["path"]), args.model)
            row = {
                "olx_id": m["olx_id"],
                "idx": p["idx"],
                "path": p["path"],
                "vlm": res,
            }
            sev = res.get("severity", "?")
            damage = res.get("visible_damage", "?")
            areas = ",".join(res.get("damage_areas") or [])[:60]
            print(f"  {done:3d}/{n_photos}  idx={p['idx']:2d}  vlm sev={sev}  dmg={damage}  {areas}  ({res.get('_latency_s')}s)")
            rows.append(row)

    args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=2))

    flagged = sum(1 for r in rows if r["vlm"].get("visible_damage"))
    sev_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for r in rows:
        s = r["vlm"].get("severity")
        if isinstance(s, int) and 0 <= s <= 3:
            sev_counts[s] += 1
    print(f"\nWrote {args.out}: {len(rows)} rows")
    print(f"Flagged visible_damage=True: {flagged}/{len(rows)}")
    print(f"Severity dist: {sev_counts}")


if __name__ == "__main__":
    main()
