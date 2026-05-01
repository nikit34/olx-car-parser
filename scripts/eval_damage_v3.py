"""Validate damage_classifier_v3 against v2 on a fresh production sample.

This is the "ship/hold" check for issue #5. It draws a random sample
of flagged listings from the host DB, fetches photos, scores each
photo with both v2 and v3, asks the host's qwen2.5vl:3b VLM for a
ground-truth label, and reports listing-level precision / recall /
F1 for both classifiers using the same listing-level rule that
production uses (``FLAG_MIN_PHOTOS`` photos at or above
``FLAG_PHOTO_THRESHOLD``).

The audit (#1) showed v2's flagged precision is ~13%. The
acceptance criterion for shipping v3 is **listing-level precision
≥ 0.50** on a 30-listing fresh audit sample. Recall is monitored on
the same sample (no separate gold benchmark survives — see
issue #5 thread for context); the recall floor is 0.85 vs v2 on
the same sample, so we don't shrink the catch.

Run on the host::

    .venv/bin/python scripts/eval_damage_v3.py \\
        --db /Users/anastasia/olx-car-parser/data/olx_cars.db \\
        --v2 /Users/anastasia/olx-car-parser/data/damage_classifier_v2.pt \\
        --v3 /Users/anastasia/olx-car-parser/data/damage_classifier_v3_alpha.pt \\
        --n-listings 30 \\
        --out /tmp/v3_data/eval_v3_vs_v2.json
"""

from __future__ import annotations

import argparse
import base64
import json
import sqlite3
import sys
import time
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.photo_damage_poc import VLM_SYSTEM  # noqa: E402

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5vl:3b"


def sample_flagged_listings(db_path: Path, n: int,
                            exclude: set[str]) -> list[dict]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    excl_clause = ""
    params: list = []
    if exclude:
        placeholders = ",".join("?" * len(exclude))
        excl_clause = f" AND olx_id NOT IN ({placeholders})"
        params = sorted(exclude)
    rows = conn.execute(f"""
        SELECT olx_id, url,
               CAST(json_extract(llm_extras, '$.photo_damage_p') AS REAL) AS max_p
        FROM listings
        WHERE is_active = 1
          AND llm_extras IS NOT NULL
          AND CAST(json_extract(llm_extras, '$.photo_damage_p') AS REAL) >= 0.20
          AND (url LIKE '%standvirtual%' OR url LIKE '%olx.pt%')
          {excl_clause}
        ORDER BY RANDOM()
        LIMIT ?
    """, [*params, n]).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def vlm_label_listing(photo_paths: list[Path], *, client: httpx.Client,
                      model: str, ollama_url: str) -> tuple[bool, list[dict]]:
    """Return (is_damaged, per-photo VLM rows).

    A listing is "damaged" iff ANY photo gets severity ≥ 1 with
    visible_damage=true (matches the labelling convention in
    ``label_damage_vlm.py``).
    """
    rows: list[dict] = []
    is_damaged = False
    for i, p in enumerate(photo_paths, 1):
        img_b64 = base64.b64encode(p.read_bytes()).decode()
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": VLM_SYSTEM},
                {"role": "user",
                 "content": "Inspect this photo and output the JSON object.",
                 "images": [img_b64]},
            ],
            "format": "json", "stream": False, "keep_alive": "30m",
            "options": {"temperature": 0.0, "num_ctx": 4096, "num_predict": 350},
        }
        try:
            r = client.post(f"{ollama_url}/api/chat", json=payload, timeout=180)
            content = r.json().get("message", {}).get("content", "")
            vlm = json.loads(content)
        except (httpx.RequestError, json.JSONDecodeError, KeyError):
            vlm = {"_error": "skip"}
        sev = vlm.get("severity") if isinstance(vlm.get("severity"), int) else 0
        vd = vlm.get("visible_damage") is True
        if sev >= 1 and vd:
            is_damaged = True
        rows.append({"idx": i, "severity": sev, "visible_damage": vd,
                     "evidence": vlm.get("evidence", ""),
                     "error": vlm.get("_error")})
    return is_damaged, rows


def score(predictions: list[bool], truth: list[bool]) -> dict:
    tp = sum(1 for p, g in zip(predictions, truth) if p and g)
    fp = sum(1 for p, g in zip(predictions, truth) if p and not g)
    fn = sum(1 for p, g in zip(predictions, truth) if not p and g)
    tn = sum(1 for p, g in zip(predictions, truth) if not p and not g)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(prec, 3), "recall": round(rec, 3),
            "f1": round(f1, 3)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--v2", type=Path, required=True)
    ap.add_argument("--v3", type=Path, required=True)
    ap.add_argument("--n-listings", type=int, default=30)
    ap.add_argument("--cache-dir", type=Path, default=Path("/tmp/v3_data/eval_cache"))
    ap.add_argument("--out", type=Path,
                    default=Path("/tmp/v3_data/eval_v3_vs_v2.json"))
    ap.add_argument("--exclude-mined", type=Path,
                    default=Path("/tmp/v3_data/mining_manifest.jsonl"),
                    help="Exclude olx_ids already used for training/labelling")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--ollama-url", default=OLLAMA_URL)
    args = ap.parse_args()

    excl: set[str] = set()
    if args.exclude_mined.exists():
        for line in args.exclude_mined.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    excl.add(json.loads(line)["olx_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    print(f"Excluding {len(excl)} mined ids from eval sample")

    listings = sample_flagged_listings(args.db, args.n_listings, excl)
    print(f"Sampled {len(listings)} fresh flagged listings for eval")

    from src.parser.photo_damage import (DamageClassifier,
                                         FLAG_MIN_PHOTOS,
                                         FLAG_PHOTO_THRESHOLD)
    from src.parser.photo_fetch import download_photo, fetch_photos

    print(f"Loading v2: {args.v2}")
    v2 = DamageClassifier(weights=args.v2)
    print(f"Loading v3: {args.v3}")
    v3 = DamageClassifier(weights=args.v3)

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    truth: list[bool] = []
    pred_v2: list[bool] = []
    pred_v3: list[bool] = []
    t0 = time.monotonic()
    with httpx.Client() as client:
        for i, l in enumerate(listings, 1):
            oid = l["olx_id"]
            d = args.cache_dir / oid
            d.mkdir(parents=True, exist_ok=True)
            urls = fetch_photos(l["url"])
            paths: list[Path] = []
            for j, u in enumerate(urls, 1):
                p = d / f"{j}.jpg"
                if download_photo(u, p):
                    paths.append(p)
            if not paths:
                print(f"  [{i}/{len(listings)}] {oid} no photos, skip")
                continue

            v2_scores = [s.p_damaged for s in v2.predict_photos_batch(paths)]
            v3_scores = [s.p_damaged for s in v3.predict_photos_batch(paths)]

            def listing_flag(scores: list[float]) -> bool:
                return sum(1 for s in scores if s >= FLAG_PHOTO_THRESHOLD) >= FLAG_MIN_PHOTOS

            v2_flag = listing_flag(v2_scores)
            v3_flag = listing_flag(v3_scores)

            gold, vlm_rows = vlm_label_listing(paths, client=client,
                                               model=args.model,
                                               ollama_url=args.ollama_url)
            truth.append(gold)
            pred_v2.append(v2_flag)
            pred_v3.append(v3_flag)
            rows.append({
                "olx_id": oid, "url": l["url"],
                "n_photos": len(paths),
                "v2_max_p": round(max(v2_scores), 4),
                "v3_max_p": round(max(v3_scores), 4),
                "v2_flag": v2_flag, "v3_flag": v3_flag,
                "vlm_damaged": gold,
                "vlm_per_photo": vlm_rows,
            })
            elapsed = time.monotonic() - t0
            print(f"  [{i}/{len(listings)}] {oid:>10s} "
                  f"v2_max={max(v2_scores):.3f} v2_flag={v2_flag!s:5s} "
                  f"v3_max={max(v3_scores):.3f} v3_flag={v3_flag!s:5s} "
                  f"vlm_damaged={gold!s:5s} ({elapsed:.0f}s)")

    summary = {
        "n_listings": len(rows),
        "n_damaged": sum(truth),
        "v2": score(pred_v2, truth),
        "v3": score(pred_v3, truth),
    }
    args.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print("\n" + "=" * 60)
    print(json.dumps(summary, indent=2))
    print(f"\nWritten to {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
