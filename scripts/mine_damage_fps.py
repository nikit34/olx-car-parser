"""Mine production photos for v3 damage-classifier retraining (issue #5).

Pulls flagged listings (``photo_damage_p >= 0.20``) from the host DB,
samples ~200 stratified across 5 score buckets, re-fetches every photo
on each listing via ``src.parser.photo_fetch.fetch_photos``, runs the
production v2 ``DamageClassifier`` per-photo, and writes a manifest with
the per-photo scores so the auto-labelling step (``label_damage_vlm.py``)
can pick up from there.

Output layout (default ``--out-dir /tmp/v3_data``)::

    /tmp/v3_data/raw/<olx_id>/<idx>.jpg              — every photo fetched
    /tmp/v3_data/mining_manifest.jsonl               — one line per photo

Manifest fields (one JSON object per line — one row per photo)::

    olx_id          : str   StandVirtual / OLX listing id
    url             : str   listing URL (for spot-check)
    photo_idx       : int   1-based gallery position
    photo_path      : str   absolute path on host
    photo_url       : str   the CDN URL we downloaded from
    v2_p_damaged    : float v2 classifier P(damaged) for this single photo
    listing_max_p   : float max v2 P(damaged) across the listing's photos
    score_bucket    : str   bucket label, derived from ``listing_max_p``

Buckets — the audit (issue #1) showed FPs across the whole flagged
range, so we stratify across the full distribution::

    "0.20-0.35", "0.35-0.50", "0.50-0.70", "0.70-0.90", ">=0.90"

Resumable: if the manifest already exists we skip listings that already
have rows. New listings are appended.

Run on the host (DB locks: don't co-run with verify-photos)::

    /Users/anastasia/actions-runner/_work/olx-car-parser/olx-car-parser/.venv/bin/python \\
        scripts/mine_damage_fps.py \\
        --db /Users/anastasia/olx-car-parser/data/olx_cars.db \\
        --weights /Users/anastasia/olx-car-parser/data/damage_classifier_v2.pt \\
        --out-dir /tmp/v3_data \\
        --max-listings 200
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Bucket edges — left-inclusive, right-exclusive except the final ">=0.90".
BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("0.20-0.35", 0.20, 0.35),
    ("0.35-0.50", 0.35, 0.50),
    ("0.50-0.70", 0.50, 0.70),
    ("0.70-0.90", 0.70, 0.90),
    (">=0.90", 0.90, 1.01),
)


def bucket_for(p: float) -> str:
    """Return the bucket label for a listing's max p_damaged."""
    for label, lo, hi in BUCKETS:
        if lo <= p < hi:
            return label
    return "<0.20"


def stratified_sample(
    db_path: Path,
    per_bucket: int,
    exclude_ids: set[str],
) -> list[dict]:
    """Pull flagged listings stratified by current ``photo_damage_p`` bucket.

    Returns a list of {olx_id, url, listing_max_p} sorted by bucket so
    downstream loops can checkpoint per-bucket if needed.
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    out: list[dict] = []
    excl_clause = ""
    params_excl: list[str] = []
    if exclude_ids:
        placeholders = ",".join("?" * len(exclude_ids))
        excl_clause = f" AND olx_id NOT IN ({placeholders})"
        params_excl = sorted(exclude_ids)
    for label, lo, hi in BUCKETS:
        rows = conn.execute(
            f"""
            SELECT olx_id, url,
                   CAST(json_extract(llm_extras, '$.photo_damage_p') AS REAL) AS max_p
            FROM listings
            WHERE is_active = 1
              AND llm_extras IS NOT NULL
              AND CAST(json_extract(llm_extras, '$.photo_damage_p') AS REAL) >= ?
              AND CAST(json_extract(llm_extras, '$.photo_damage_p') AS REAL) <  ?
              AND (url LIKE '%standvirtual%' OR url LIKE '%olx.pt%')
              {excl_clause}
            ORDER BY RANDOM()
            LIMIT ?
            """,
            [lo, hi, *params_excl, per_bucket],
        ).fetchall()
        for r in rows:
            out.append({
                "olx_id": r["olx_id"],
                "url": r["url"],
                "listing_max_p": float(r["max_p"]),
                "score_bucket": label,
            })
    conn.close()
    return out


def already_mined_ids(manifest_path: Path) -> set[str]:
    if not manifest_path.exists():
        return set()
    ids: set[str] = set()
    for line in manifest_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ids.add(json.loads(line)["olx_id"])
        except (json.JSONDecodeError, KeyError):
            continue
    return ids


def mine_listing(
    listing: dict,
    classifier,
    raw_root: Path,
    fetch_photos_fn,
    download_photo_fn,
) -> list[dict]:
    """Download every photo for one listing and score it with v2.

    Returns rows ready to be appended to the manifest. ``listing_max_p``
    is recomputed from this fresh fetch (DB value is a snapshot — listings
    sometimes mutate their gallery), but the bucket label stays the
    *original* DB-snapshot bucket so the stratification is honest.
    """
    olx_id = listing["olx_id"]
    listing_dir = raw_root / olx_id
    listing_dir.mkdir(parents=True, exist_ok=True)

    photo_urls = fetch_photos_fn(listing["url"])
    if not photo_urls:
        return []

    paths: list[tuple[int, Path, str]] = []
    for idx, purl in enumerate(photo_urls, 1):
        local = listing_dir / f"{idx}.jpg"
        if download_photo_fn(purl, local):
            paths.append((idx, local, purl))

    if not paths:
        return []

    scores = classifier.predict_photos_batch([p for _, p, _ in paths])
    fresh_max = max((s.p_damaged for s in scores), default=0.0)

    rows: list[dict] = []
    for (idx, path, purl), pred in zip(paths, scores):
        rows.append({
            "olx_id": olx_id,
            "url": listing["url"],
            "photo_idx": idx,
            "photo_path": str(path),
            "photo_url": purl,
            "v2_p_damaged": round(pred.p_damaged, 4),
            "listing_max_p": round(fresh_max, 4),
            "score_bucket": listing["score_bucket"],
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, required=True,
                    help="Path to olx_cars.db on the host")
    ap.add_argument("--weights", type=Path, required=True,
                    help="Path to damage_classifier_v2.pt")
    ap.add_argument("--out-dir", type=Path, default=Path("/tmp/v3_data"),
                    help="Root output dir; manifest + raw/ live under here")
    ap.add_argument("--per-bucket", type=int, default=40,
                    help="Listings per score bucket (5 buckets → 200 total)")
    ap.add_argument("--max-listings", type=int, default=0,
                    help="Hard cap across all buckets (0 = use per-bucket)")
    ap.add_argument("--device", default=None,
                    help="torch device override (default: mps if available)")
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    raw_root = out_dir / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "mining_manifest.jsonl"

    excluded = already_mined_ids(manifest_path)
    if excluded:
        print(f"Skipping {len(excluded)} already-mined listings")

    listings = stratified_sample(args.db, args.per_bucket, excluded)
    if args.max_listings:
        listings = listings[: args.max_listings]
    print(f"Sampled {len(listings)} listings across {len(BUCKETS)} buckets")
    bucket_counts: dict[str, int] = {}
    for l in listings:
        bucket_counts[l["score_bucket"]] = bucket_counts.get(l["score_bucket"], 0) + 1
    for label, _, _ in BUCKETS:
        print(f"  {label:>10s}: {bucket_counts.get(label, 0)}")

    # Lazy imports — keeps `--help` cheap and the test suite mock-friendly.
    from src.parser.photo_damage import DamageClassifier
    from src.parser.photo_fetch import download_photo, fetch_photos

    print(f"Loading v2 weights from {args.weights}…")
    clf = DamageClassifier(weights=args.weights, device=args.device)

    t0 = time.monotonic()
    n_photos = 0
    n_failed = 0
    with manifest_path.open("a") as mf:
        for i, listing in enumerate(listings, 1):
            try:
                rows = mine_listing(listing, clf, raw_root, fetch_photos, download_photo)
            except Exception as e:  # pragma: no cover — per-listing best-effort
                print(f"  [{i}/{len(listings)}] {listing['olx_id']} FAILED: {e}")
                n_failed += 1
                continue
            if not rows:
                print(f"  [{i}/{len(listings)}] {listing['olx_id']} no photos")
                n_failed += 1
                continue
            for r in rows:
                mf.write(json.dumps(r, ensure_ascii=False) + "\n")
            mf.flush()
            n_photos += len(rows)
            elapsed = time.monotonic() - t0
            print(f"  [{i}/{len(listings)}] {listing['olx_id']:>10s} "
                  f"bucket={listing['score_bucket']:>10s} "
                  f"photos={len(rows):2d} "
                  f"max_p={rows[0]['listing_max_p']:.3f} "
                  f"elapsed={elapsed:.0f}s")

    print(f"\nDone. {n_photos} photos mined across "
          f"{len(listings) - n_failed} listings ({n_failed} failed). "
          f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
