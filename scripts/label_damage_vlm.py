"""Auto-label mined photos with qwen2.5vl:3b for v3 retraining (issue #5).

Reads ``mining_manifest.jsonl`` produced by ``mine_damage_fps.py`` and
asks the host's Ollama VLM to score each photo. Maps the VLM JSON
({visible_damage, severity, ...}) to a binary ``damaged`` label::

    damaged = True   iff  severity >= 1  AND  visible_damage == True
    damaged = False  iff  severity == 0  OR   visible_damage == False

Photos where the VLM returns ``_error`` or non-JSON are skipped (their
olx_id+photo_idx is not written, so a later resume retries them).

Output appends to ``labels.jsonl`` next to the manifest::

    {olx_id, photo_idx, photo_path,
     v2_p_damaged, listing_max_p, score_bucket,
     vlm_severity, vlm_visible_damage, vlm_evidence,
     vlm_is_studio_or_dealer, damaged, vlm_latency_s}

Resumable: on each run, photos already present in ``labels.jsonl``
(matched by ``(olx_id, photo_idx)``) are skipped.

The VLM is the slowest stage — qwen2.5vl:3b on the 8 GB Air does
~10-15 s/photo. ~25 photos × 100 listings = ~5-7 hours wall-clock.
Cap with ``--max-listings`` (default 100) to bound the run; the
remaining listings can be picked up on a follow-up pass.

Run on the host (Ollama @ localhost:11434, qwen2.5vl:3b pulled)::

    .venv/bin/python scripts/label_damage_vlm.py \\
        --manifest /tmp/v3_data/mining_manifest.jsonl \\
        --labels   /tmp/v3_data/labels.jsonl \\
        --max-listings 100
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Prompt inlined from scripts/photo_damage_poc.py (an untracked POC).
# Keeping this script self-contained — runner checkouts under _work/ only
# see tracked files, so importing the POC blew up with
# ``No module named 'scripts.photo_damage_poc'``.
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

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5vl:3b"


def call_vlm(
    img_path: Path,
    *,
    model: str = DEFAULT_MODEL,
    ollama_url: str = OLLAMA_URL,
    timeout_s: float = 180.0,
    client: httpx.Client | None = None,
) -> dict:
    """One vision-LLM call — returns the parsed JSON or an ``_error`` dict.

    A separate ``client`` argument keeps the test suite mockable while
    letting production use a long-lived ``httpx.Client`` (saves the TCP
    handshake per request — meaningful at hundreds of calls).
    """
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
        # 30 m keep-alive: with 25 photos/listing × 100 listings the
        # model needs to stay resident across the whole run.
        "keep_alive": "30m",
        "options": {"temperature": 0.0, "num_ctx": 4096, "num_predict": 350},
    }
    t0 = time.monotonic()
    do_post = (client.post if client is not None else httpx.post)
    try:
        r = do_post(f"{ollama_url}/api/chat", json=payload, timeout=timeout_s)
    except httpx.RequestError as e:
        return {"_error": f"request: {e}", "_latency_s": time.monotonic() - t0}
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


def derive_damaged(vlm: dict) -> bool | None:
    """Map the VLM JSON to a binary classifier label.

    Returns ``None`` when the VLM output is unusable (missing fields or
    non-bool ``visible_damage``) so callers can drop the photo.
    """
    if not vlm or "_error" in vlm:
        return None
    sev = vlm.get("severity")
    vd = vlm.get("visible_damage")
    if not isinstance(sev, int) or not isinstance(vd, bool):
        return None
    if sev == 0 or vd is False:
        return False
    if sev >= 1 and vd is True:
        return True
    return None


def already_labelled_keys(labels_path: Path) -> set[tuple[str, int]]:
    """``(olx_id, photo_idx)`` keys already present, for resume safety."""
    if not labels_path.exists():
        return set()
    seen: set[tuple[str, int]] = set()
    for line in labels_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            seen.add((row["olx_id"], int(row["photo_idx"])))
        except (json.JSONDecodeError, KeyError, ValueError):
            continue
    return seen


def group_by_listing(rows: list[dict]) -> list[tuple[str, list[dict]]]:
    """Preserve manifest order while grouping photos under each olx_id."""
    seen: dict[str, list[dict]] = {}
    order: list[str] = []
    for r in rows:
        oid = r["olx_id"]
        if oid not in seen:
            seen[oid] = []
            order.append(oid)
        seen[oid].append(r)
    return [(oid, seen[oid]) for oid in order]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path,
                    default=Path("/tmp/v3_data/mining_manifest.jsonl"))
    ap.add_argument("--labels", type=Path,
                    default=Path("/tmp/v3_data/labels.jsonl"))
    ap.add_argument("--ollama-url", default=OLLAMA_URL)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    # Hard cap. The VLM is ~12 s/photo on an 8 GB M-series Air; 100
    # listings × 25 photos ≈ 5-7 h wall-clock. Going higher risks the
    # cron verify-photos slot.
    ap.add_argument("--max-listings", type=int, default=100,
                    help="Stop after this many listings (default 100, 0=all)")
    args = ap.parse_args()

    if not args.manifest.exists():
        print(f"manifest not found: {args.manifest}", file=sys.stderr)
        return 1

    rows = [json.loads(l) for l in args.manifest.read_text().splitlines() if l.strip()]
    listings = group_by_listing(rows)
    print(f"Manifest: {len(rows)} photos across {len(listings)} listings")

    seen_keys = already_labelled_keys(args.labels)
    if seen_keys:
        print(f"Resuming — {len(seen_keys)} (olx_id, photo_idx) pairs already labelled")

    cap = args.max_listings or len(listings)
    args.labels.parent.mkdir(parents=True, exist_ok=True)

    n_done = 0
    n_listings_done = 0
    t0 = time.monotonic()
    label_counts = {True: 0, False: 0, None: 0}
    with httpx.Client() as client, args.labels.open("a") as out:
        for oid, photos in listings:
            if n_listings_done >= cap:
                break
            # If every photo of this listing is already labelled, treat
            # it as completed for the cap counter and skip the inner loop.
            unlabelled = [p for p in photos
                          if (p["olx_id"], int(p["photo_idx"])) not in seen_keys]
            if not unlabelled:
                n_listings_done += 1
                continue
            print(f"\n[{n_listings_done + 1}/{cap}] {oid} "
                  f"({len(unlabelled)}/{len(photos)} photos to label)")
            for p in unlabelled:
                photo_path = Path(p["photo_path"])
                if not photo_path.exists():
                    print(f"  photo {p['photo_idx']:2d} missing on disk, skip")
                    continue
                vlm = call_vlm(photo_path, model=args.model,
                               ollama_url=args.ollama_url, client=client)
                damaged = derive_damaged(vlm)
                label_counts[damaged] += 1
                if damaged is None:
                    err = vlm.get("_error") if vlm else "no-vlm"
                    print(f"  photo {p['photo_idx']:2d} unusable ({err})")
                    continue
                row = {
                    "olx_id": oid,
                    "photo_idx": p["photo_idx"],
                    "photo_path": p["photo_path"],
                    "v2_p_damaged": p["v2_p_damaged"],
                    "listing_max_p": p["listing_max_p"],
                    "score_bucket": p["score_bucket"],
                    "vlm_severity": vlm.get("severity"),
                    "vlm_visible_damage": vlm.get("visible_damage"),
                    "vlm_evidence": vlm.get("evidence"),
                    "vlm_is_studio_or_dealer": vlm.get("is_studio_or_dealer"),
                    "vlm_latency_s": vlm.get("_latency_s"),
                    "damaged": damaged,
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                n_done += 1
                if n_done % 25 == 0:
                    rate = n_done / (time.monotonic() - t0)
                    print(f"    ...{n_done} photos labelled "
                          f"({rate:.2f} photos/s, "
                          f"clean={label_counts[False]} damaged={label_counts[True]} "
                          f"unusable={label_counts[None]})")
            n_listings_done += 1

    elapsed = (time.monotonic() - t0) / 60
    print(f"\nLabelled {n_done} new photos in {elapsed:.1f} min "
          f"(clean={label_counts[False]}, damaged={label_counts[True]}, "
          f"unusable={label_counts[None]})")
    print(f"Labels: {args.labels}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
