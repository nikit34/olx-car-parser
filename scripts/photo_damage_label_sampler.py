"""Stratified sampler + photo downloader for the gold-standard labeling pass.

Pulls a balanced batch of listings from the runner's DB, downloads every
photo locally, and writes a manifest. The labeling step (humans or
LLM-with-vision like Claude) reads the manifest, opens photos one by one,
and appends labels to the gold-standard JSONL.

Manifest format (one listing per line):
  {
    "olx_id": "8PZUbo",
    "url": "https://...",
    "title": "...",
    "text_severity": 2,
    "description_head": "first 300 chars...",
    "photos": [
      {"idx": 1, "path": "/tmp/photo_label/8PZUbo_1.jpg", "url": "..."},
      ...
    ]
  }

Usage:
  python scripts/photo_damage_label_sampler.py --db /tmp/olx_cars_runner.db \\
      --n-per-bucket 5,8,8,5,4   # sev=0,1,2,3,null
"""

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.parser.photo_fetch import fetch_standvirtual_advert  # noqa: E402


def fetch_photos(url: str) -> tuple[list[str], dict]:
    """Return (photo_urls, ad_dict) for a standvirtual listing."""
    ad = fetch_standvirtual_advert(url)
    if not ad:
        return [], {}
    try:
        return [p["url"] for p in ad["images"]["photos"]], ad
    except (KeyError, TypeError):
        return [], ad


def stratified_sample(db_path: Path, counts: dict[int | None, int],
                      exclude_olx_ids: set[str] | None = None) -> list[dict]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    out = []
    excl = exclude_olx_ids or set()
    excl_clause = ""
    if excl:
        placeholders = ",".join("?" * len(excl))
        excl_clause = f" AND olx_id NOT IN ({placeholders})"
    for sev, n in counts.items():
        if n <= 0:
            continue
        if sev is None:
            where = "json_extract(llm_extras, '$.damage_severity') IS NULL"
        else:
            where = f"json_extract(llm_extras, '$.damage_severity') = {sev}"
        params = list(excl)
        rows = conn.execute(f"""
            SELECT olx_id, url, title, description, llm_extras
            FROM listings
            WHERE {where}
              AND llm_extras IS NOT NULL
              AND url LIKE '%standvirtual%'
              AND length(coalesce(description, '')) >= 30
              AND is_active = 1
              {excl_clause}
            ORDER BY RANDOM()
            LIMIT {n}
        """, params).fetchall()
        for r in rows:
            d = dict(r)
            d["text_severity"] = sev
            d["llm_extras"] = json.loads(d["llm_extras"]) if d["llm_extras"] else {}
            out.append(d)
    conn.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--n-per-bucket", default="5,8,8,5,4",
                    help="Counts for sev=0,1,2,3,null (default 5,8,8,5,4 = 30)")
    ap.add_argument("--out-dir", type=Path, default=Path("/tmp/photo_label"))
    ap.add_argument("--manifest", type=Path,
                    default=Path("/tmp/photo_label/manifest.jsonl"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    counts_list = [int(x) for x in args.n_per_bucket.split(",")]
    sev_keys: list[int | None] = [0, 1, 2, 3, None]
    counts = {sev_keys[i]: counts_list[i]
              for i in range(min(len(counts_list), 5))}

    # If the manifest already exists, exclude its olx_ids so successive
    # runs append fresh listings instead of re-picking the same ones.
    excl: set[str] = set()
    if args.manifest.exists():
        for line in args.manifest.read_text().splitlines():
            if line.strip():
                excl.add(json.loads(line)["olx_id"])
        print(f"Excluding {len(excl)} already-sampled listings")

    print(f"Sampling from {args.db}…")
    listings = stratified_sample(args.db, counts, exclude_olx_ids=excl)
    print(f"  picked {len(listings)} listings: " +
          ", ".join(f"sev={l['text_severity']}" for l in listings))

    manifest_lines = []
    failures = 0
    for i, l in enumerate(listings, 1):
        print(f"\n[{i}/{len(listings)}] {l['olx_id']} text_sev={l['text_severity']}")
        try:
            photo_urls, ad = fetch_photos(l["url"])
        except Exception as e:
            print(f"  ! fetch failed: {e}")
            failures += 1
            continue

        if not photo_urls:
            print(f"  ! no photos")
            failures += 1
            continue

        photos_meta = []
        for j, purl in enumerate(photo_urls, 1):
            local = args.out_dir / f"{l['olx_id']}_{j}.jpg"
            if not local.exists():
                try:
                    r = httpx.get(purl, follow_redirects=True, timeout=30)
                    r.raise_for_status()
                    local.write_bytes(r.content)
                except Exception as e:
                    print(f"  ! photo {j} download failed: {e}")
                    continue
            photos_meta.append({"idx": j, "path": str(local), "url": purl})

        if not photos_meta:
            print(f"  ! no photos downloaded")
            failures += 1
            continue

        print(f"  {len(photos_meta)} photos cached")

        manifest_lines.append({
            "olx_id": l["olx_id"],
            "url": l["url"],
            "title": l["title"],
            "text_severity": l["text_severity"],
            "description_head": (l["description"] or "")[:300],
            "llm_damage_severity": l["llm_extras"].get("damage_severity"),
            "llm_mechanical_condition": l["llm_extras"].get("mechanical_condition"),
            "photos": photos_meta,
        })

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    # Append: keep already-labeled entries, add new ones.
    with args.manifest.open("a") as f:
        for line in manifest_lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    print(f"\nManifest: {args.manifest}  ({len(manifest_lines)} listings, "
          f"{failures} failures)")
    print(f"Photos:   {args.out_dir}/  ({sum(len(l['photos']) for l in manifest_lines)} total)")


if __name__ == "__main__":
    main()
