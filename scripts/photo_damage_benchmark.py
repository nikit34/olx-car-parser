"""Benchmark CLIP-triage + VLM photo damage detection across a stratified
sample of listings from the live DB.

For N listings stratified by text-side ``damage_severity`` (sev 0-3, plus
optional sev=null) we:
  1. Pull URL + text-LLM truth from the runner's DB (``--db``).
  2. Fetch ALL photos.
  3. Score every photo via CLIP zero-shot (~30 ms/photo).
  4. Score every photo via qwen2.5vl:3b VLM (~12 s/photo). Optional
     (--no-vlm) — first iteration may want CLIP-only to confirm sample
     diversity before spending the VLM budget.
  5. Aggregate dataset-wide:
        - per-listing P@K (does CLIP rank actual damaged photos at top?)
        - dataset Spearman correlation
        - text vs VLM disagreement counts (does VLM reveal damage that
          text didn't see?).

Output:
  /tmp/photo_bench/<run_id>/results.json  — all per-photo data
  stdout                                  — summary table

Usage:
  python scripts/photo_damage_benchmark.py --n-per-bucket 2,3,3,2 \\
      --db /path/to/olx_cars.db --vlm
"""

import argparse
import json
import re
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Re-use the prompt set from the triage POC so the two scripts stay in
# sync. If we ever fork prompts, fork them here too.
from scripts.photo_damage_clip_triage import (
    DAMAGE_PROMPTS, CLEAN_PROMPTS, score_with_clip,
    download_photos, fetch_listing,
)


def stratified_sample(db_path: Path, counts: dict[int | None, int],
                      min_text_len: int = 50) -> list[dict]:
    """Return N×len(counts) listings, sampled per damage_severity bucket.

    counts maps severity (0-3 or None) to how many to pick. Listings are
    drawn at random so the same script run on the same DB picks different
    ones — set --seed for reproducible runs.
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    out = []
    for sev, n in counts.items():
        if n <= 0:
            continue
        if sev is None:
            where = "json_extract(llm_extras, '$.damage_severity') IS NULL"
        else:
            where = f"json_extract(llm_extras, '$.damage_severity') = {sev}"
        rows = conn.execute(f"""
            SELECT olx_id, url, title, description, llm_extras
            FROM listings
            WHERE {where}
              AND llm_extras IS NOT NULL
              AND url LIKE '%standvirtual%'   -- POC parses standvirtual __NEXT_DATA__
              AND length(coalesce(description, '')) >= {min_text_len}
              AND is_active = 1
            ORDER BY RANDOM()
            LIMIT {n}
        """).fetchall()
        for r in rows:
            r = dict(r)
            r["text_severity"] = sev
            r["llm_extras"] = json.loads(r["llm_extras"]) if r["llm_extras"] else {}
            out.append(r)
    conn.close()
    return out


def call_vlm(img_path: Path, model: str = "qwen2.5vl:3b") -> dict | None:
    """One vision call; mirrors the schema from photo_damage_poc."""
    import base64
    img_b64 = base64.b64encode(img_path.read_bytes()).decode()
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _VLM_SYSTEM},
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
    try:
        r = httpx.post("http://localhost:11434/api/chat", json=payload, timeout=180)
        dt = time.monotonic() - t0
    except httpx.RequestError as e:
        return {"_error": str(e), "_latency_s": time.monotonic() - t0}
    if r.status_code != 200:
        return {"_error": f"http {r.status_code}", "_latency_s": dt}
    content = r.json().get("message", {}).get("content", "")
    try:
        out = json.loads(content)
    except json.JSONDecodeError:
        return {"_error": "non-json", "_raw": content[:200], "_latency_s": dt}
    out["_latency_s"] = round(dt, 1)
    return out


_VLM_SYSTEM = """\
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


def score_listing(listing: dict, cache_dir: Path,
                  run_vlm: bool, clip_model_name: str) -> dict:
    """Run CLIP + (optionally) VLM on every photo of one listing."""
    print(f"\n{'=' * 70}\nListing {listing['olx_id']} | text_sev={listing['text_severity']!r}")
    print(f"  {listing['title']}")
    print(f"  {listing['url']}")

    try:
        scraped = fetch_listing(listing["url"])
    except Exception as e:
        print(f"  ! fetch failed: {e}")
        return {**listing, "error": f"fetch: {e}"}

    photos = scraped["photos"]
    if not photos:
        print("  ! no photos")
        return {**listing, "error": "no photos"}
    print(f"  photos: {len(photos)}")

    paths = download_photos(photos, listing["olx_id"], cache_dir)

    clip_scores, sims, prompts = score_with_clip(paths, model_name=clip_model_name)

    n_dmg = len(DAMAGE_PROMPTS)
    per_photo = []
    for i, (path, score, row) in enumerate(zip(paths, clip_scores, sims), 1):
        rec = {
            "idx": i,
            "url": photos[i - 1],
            "clip_damage_score": round(float(score), 4),
            "clip_top_dmg_prompt": prompts[int(row[:n_dmg].argmax())],
            "clip_top_clean_prompt": prompts[int(row[n_dmg:].argmax()) + n_dmg],
        }
        per_photo.append(rec)

    if run_vlm:
        print("  running VLM on every photo...")
        for i, (path, rec) in enumerate(zip(paths, per_photo), 1):
            t0 = time.monotonic()
            vlm = call_vlm(path)
            rec["vlm"] = vlm
            sev = vlm.get("severity") if vlm and "_error" not in vlm else None
            print(f"    photo #{i:2d}: clip={rec['clip_damage_score']:+.3f}  "
                  f"vlm_sev={sev!r}  (t={time.monotonic() - t0:.1f}s)")

    listing_summary = {
        "olx_id": listing["olx_id"],
        "url": listing["url"],
        "title": listing["title"],
        "text_severity": listing["text_severity"],
        "n_photos": len(photos),
        "max_clip_score": round(max(clip_scores), 4),
        "photos": per_photo,
    }
    if run_vlm:
        sevs = [p["vlm"].get("severity") for p in per_photo
                if p.get("vlm") and "_error" not in p["vlm"]]
        listing_summary["vlm_max_severity"] = max(sevs) if sevs else None
        listing_summary["vlm_severity_counts"] = {
            s: sum(1 for x in sevs if x == s) for s in (0, 1, 2, 3)}
    return listing_summary


def aggregate(results: list[dict], have_vlm: bool):
    print(f"\n{'#' * 70}\n# DATASET SUMMARY ({len(results)} listings)\n{'#' * 70}")

    print("\n--- per-listing ---")
    print(f"{'olx_id':12s}  {'text':>4s}  {'vlm_max':>7s}  "
          f"{'clip_max':>9s}  {'verdict':>8s}  title")
    for r in results:
        if "error" in r:
            print(f"{r['olx_id']:12s}  ERROR: {r['error']}")
            continue
        text = r["text_severity"]
        vmax = r.get("vlm_max_severity")
        cmax = r["max_clip_score"]
        verdict = "—"
        if have_vlm and vmax is not None and text is not None:
            d = vmax - text
            verdict = f"+{d}" if d > 0 else (f"{d}" if d < 0 else "ok")
        print(f"{r['olx_id']:12s}  {str(text):>4s}  {str(vmax):>7s}  "
              f"{cmax:+.3f}  {verdict:>8s}  {r['title'][:40]}")

    if not have_vlm:
        print("\n(VLM disabled — dataset metrics need --vlm to compute)")
        return

    # Dataset-level Spearman: pool every (clip_score, vlm_severity) across
    # all listings into a flat list, then correlate. Per-photo, not
    # per-listing — that's the granularity at which CLIP is acting as a
    # triage filter.
    all_clip = []
    all_vlm = []
    for r in results:
        if "error" in r:
            continue
        for p in r["photos"]:
            v = p.get("vlm")
            if v and "_error" not in v and "severity" in v:
                all_clip.append(p["clip_damage_score"])
                all_vlm.append(v["severity"])

    if all_clip:
        try:
            from scipy.stats import spearmanr
            rho, pval = spearmanr(all_clip, all_vlm)
            print(f"\n--- dataset Spearman (CLIP vs VLM, per photo, n={len(all_clip)}) ---")
            print(f"  ρ = {rho:+.3f}  (p = {pval:.2e})")
        except ImportError:
            pass

    # Per-listing precision@K for "did CLIP put damaged photos at the top?"
    print("\n--- precision@K (CLIP-ranked top-K covers VLM-damaged photos) ---")
    for K in (3, 5):
        hit_total, denom_total = 0, 0
        for r in results:
            if "error" in r:
                continue
            damaged = {p["idx"] for p in r["photos"]
                       if (p.get("vlm") or {}).get("severity", 0) >= 2}
            if not damaged:
                continue
            top_k = [p["idx"] for p in
                     sorted(r["photos"], key=lambda x: x["clip_damage_score"], reverse=True)[:K]]
            hit = len(set(top_k) & damaged)
            hit_total += hit
            denom_total += min(K, len(damaged))
        if denom_total:
            print(f"  P@{K}: {hit_total}/{denom_total} = {hit_total / denom_total:.1%}")

    # The headline finding for the project: how often does VLM disagree
    # with text? That's the value-add this whole feature would deliver.
    print("\n--- text vs VLM disagreement (where photos catch what text missed) ---")
    rows = []
    for r in results:
        if "error" in r or r.get("vlm_max_severity") is None or r["text_severity"] is None:
            continue
        rows.append((r["text_severity"], r["vlm_max_severity"], r["olx_id"]))
    misses = [(t, v, oid) for t, v, oid in rows if v - t >= 1]
    overcalls = [(t, v, oid) for t, v, oid in rows if t - v >= 1]
    print(f"  photo > text by ≥1 (potential undisclosed damage): {len(misses)}/{len(rows)}")
    for t, v, oid in misses:
        print(f"    {oid}: text={t} → vlm={v}  (Δ=+{v - t})")
    print(f"  photo < text by ≥1 (text exaggerates / clean photos selected): "
          f"{len(overcalls)}/{len(rows)}")
    for t, v, oid in overcalls:
        print(f"    {oid}: text={t} → vlm={v}  (Δ={v - t})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, required=True,
                    help="Path to olx_cars.db (typically rsync from runner)")
    ap.add_argument("--n-per-bucket", default="2,3,3,2",
                    help="Comma-separated n for sev=0,1,2,3 (default 2,3,3,2)")
    ap.add_argument("--vlm", action="store_true",
                    help="Run VLM on every photo (slow; ~12s/photo)")
    ap.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    ap.add_argument("--cache-dir", default="/tmp/olx_photo_poc")
    ap.add_argument("--results-dir", default="/tmp/photo_bench")
    ap.add_argument("--seed", type=int, default=None,
                    help="If set, seeds SQLite RANDOM() for reproducible sampling")
    args = ap.parse_args()

    counts_list = [int(x) for x in args.n_per_bucket.split(",")]
    counts = {i: counts_list[i] for i in range(min(len(counts_list), 4))}

    if args.seed is not None:
        # SQLite uses C rand() for RANDOM(); easiest reproducible-sample
        # path is just to seed Python's random and use python-side ORDER BY.
        # Skipping for POC — sampling 10 listings is fast to redo.
        pass

    print(f"Sampling from {args.db}…")
    listings = stratified_sample(args.db, counts)
    print(f"  picked {len(listings)} listings: " + ", ".join(
        f"sev={l['text_severity']}" for l in listings))

    results = []
    for l in listings:
        try:
            r = score_listing(l, Path(args.cache_dir),
                              run_vlm=args.vlm,
                              clip_model_name=args.clip_model)
            results.append(r)
        except Exception as e:
            print(f"  ! failed: {e}")
            results.append({**l, "error": str(e)})

    out_dir = Path(args.results_dir) / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_dir / 'results.json'}")

    aggregate(results, have_vlm=args.vlm)


if __name__ == "__main__":
    main()
