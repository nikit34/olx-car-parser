"""Dry-run: verify text-LLM damage_severity ≥ 2 against photos via the v2 classifier.

Reads candidates from a SQLite DB, fetches their listing photos, runs the
ResNet50 binary damage classifier (``damage_classifier_v2.pt``) and outputs
a JSON report with proposed actions — *without* touching the DB.

Decision logic (production threshold = 0.20 per-photo, max-aggregate at listing):
  text_sev >= 2 AND max_p < 0.20                                  → downgrade to 0 (text overcall)
  text_sev >= 2 AND max_p >= 0.20                                 → keep, mark photo_verified
  text_sev >= 2 AND no photos                                     → no_photos (skip)

Usage:
  python scripts/photo_verify_damage.py \\
      --db data/olx_cars.db \\
      --limit 30   # for first dry run
"""

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.parser.photo_fetch import fetch_photos, download_photo  # noqa: E402, F401


def decide(text_sev: int, max_p: float, n_photos: int, threshold: float) -> tuple[str, int]:
    """Returns (decision, recommended_severity)."""
    if n_photos == 0:
        return "no_photos", text_sev
    if max_p >= threshold:
        return "photo_confirms", text_sev
    return "downgrade", 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--weights", type=Path,
                    default=PROJECT_ROOT / "damage_classifier_v2.pt")
    ap.add_argument("--threshold", type=float, default=0.20,
                    help="P(damaged) listing-level threshold (default 0.20 — "
                         "F1=0.818 with R=100%% on gold).")
    ap.add_argument("--limit", type=int, default=0,
                    help="Cap candidates (0 = all)")
    ap.add_argument("--cache-dir", type=Path,
                    default=Path("/tmp/photo_verify/cache"))
    ap.add_argument("--out", type=Path,
                    default=Path("/tmp/photo_verify/report.json"))
    args = ap.parse_args()

    from src.parser.photo_damage import DamageClassifier
    print(f"Loading {args.weights}…")
    clf = DamageClassifier(args.weights, threshold=args.threshold)
    print(f"Classes: {clf.classes}, imgsz: {clf.imgsz}, device: {clf.device}\n")

    conn = sqlite3.connect(str(args.db))
    conn.row_factory = sqlite3.Row
    limit_clause = f"LIMIT {args.limit}" if args.limit else ""
    rows = conn.execute(f"""
        SELECT olx_id, url, title, llm_extras
        FROM listings
        WHERE is_active = 1
          AND json_extract(llm_extras, '$.damage_severity') >= 2
          AND (url LIKE '%standvirtual%' OR url LIKE '%olx.pt%')
        ORDER BY json_extract(llm_extras, '$.damage_severity') DESC, RANDOM()
        {limit_clause}
    """).fetchall()
    print(f"Candidates: {len(rows)}\n")

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    report = []
    decisions: dict[str, int] = {}
    t0 = time.monotonic()

    for i, r in enumerate(rows, 1):
        olx_id = r["olx_id"]
        text_sev = json.loads(r["llm_extras"]).get("damage_severity")
        photo_urls = fetch_photos(r["url"])
        listing_dir = args.cache_dir / olx_id
        photo_paths = []
        for j, url in enumerate(photo_urls, 1):
            p = listing_dir / f"{olx_id}_{j}.jpg"
            if download_photo(url, p):
                photo_paths.append(p)

        if photo_paths:
            pred = clf.predict_listing(olx_id, photo_paths)
            max_p = pred.max_p
            top = sorted(pred.photos, key=lambda x: -x.p_damaged)[:3]
            hits = [{"photo": p.path.name, "p_damaged": round(p.p_damaged, 3)} for p in top]
        else:
            max_p = 0.0
            hits = []

        decision, rec_sev = decide(text_sev, max_p, len(photo_paths), args.threshold)
        decisions[decision] = decisions.get(decision, 0) + 1

        rec = {
            "olx_id": olx_id,
            "url": r["url"],
            "title": r["title"],
            "text_damage_severity": text_sev,
            "n_photos": len(photo_paths),
            "max_p_damaged": round(max_p, 3),
            "decision": decision,
            "recommended_severity": rec_sev,
            "top_photos": hits,
        }
        report.append(rec)

        print(f"[{i}/{len(rows)}] {olx_id}  text={text_sev}  "
              f"photos={len(photo_paths):2d}  max_p={max_p:.3f}  "
              f"→ {decision}  ({(r['title'] or '')[:40]})")

    args.out.write_text(json.dumps(report, indent=2))
    elapsed = time.monotonic() - t0
    print(f"\nDone in {elapsed / 60:.1f} min ({elapsed / max(len(rows), 1):.1f}s/listing)")
    print(f"Report: {args.out}")
    print(f"\nDecision summary:")
    for d, c in sorted(decisions.items(), key=lambda x: -x[1]):
        print(f"  {d:20s}  {c:4d}  ({c / len(rows):.0%})")


if __name__ == "__main__":
    main()
