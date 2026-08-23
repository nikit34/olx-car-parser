"""Mine damaged-car listings from OLX and stage their photos for annotation.

Feeds the damage segmentation dataset. The pipeline this belongs to:

    harvest (this script)  →  human/Claude draws boxes  →  SAM turns boxes into
    masks  →  YOLO-seg polygons  →  train

Damage is rare in the corpus (a few percent of active listings), so a random
sample is a terrible way to collect it. The OLX JSON API takes a free-text
``query``, and Portuguese damage vocabulary is distinctive enough to raise the
hit rate by an order of magnitude: "sinistrado", "batido", "para peças",
"salvado", "desmancha". Those queries are the whole trick here.

The output manifest is append-only and keyed by (olx_id, photo_idx) so a
re-run tops the pool up without re-downloading or duplicating annotations.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.parser.scraper import (  # noqa: E402
    OlxScraper, ScraperConfig, OLX_API_URL, CARS_CATEGORY_ID,
)
from src.parser.photo_fetch import download_photo, fetch_photos_olx_from_api  # noqa: E402

# Portuguese salvage/damage vocabulary, strongest first. "para peças" and
# "desmancha" pull parts-cars (severity 3); "sinistrado" / "batido" pull the
# mid-range crash damage that is actually the hard case for a classifier.
DAMAGE_QUERIES = [
    "sinistrado",
    "batido",
    "salvado",
    "para peças",
    "acidentado",
    "desmancha",
    "para abate",
    "danificado",
    "capotado",
    "para restauro",
]


def harvest(out_dir: Path, per_query: int, max_photos: int) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "raw"
    img_dir.mkdir(exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"

    seen: set[str] = set()
    if manifest_path.exists():
        for line in manifest_path.read_text().splitlines():
            if line.strip():
                seen.add(json.loads(line)["photo_id"])

    scraper = OlxScraper(ScraperConfig())
    added = 0
    try:
        with manifest_path.open("a") as mf:
            for q in DAMAGE_QUERIES:
                offset = 0
                got = 0
                while got < per_query:
                    url = (f"{OLX_API_URL}?offset={offset}&limit=40"
                           f"&category_id={CARS_CATEGORY_ID}"
                           f"&query={urllib.parse.quote(q)}")
                    payload = scraper._fetch_json(url)
                    rows = (payload or {}).get("data") or []
                    if not rows:
                        break
                    for offer in rows:
                        if got >= per_query:
                            break
                        olx_id = str(offer.get("id"))
                        photos = fetch_photos_olx_from_api(offer)[:max_photos]
                        if not photos:
                            continue
                        got += 1
                        for idx, purl in enumerate(photos):
                            photo_id = f"{olx_id}_{idx}"
                            if photo_id in seen:
                                continue
                            dest = img_dir / f"{photo_id}.jpg"
                            if not dest.exists() and not download_photo(purl, dest):
                                continue
                            mf.write(json.dumps({
                                "photo_id": photo_id,
                                "olx_id": olx_id,
                                "photo_idx": idx,
                                "query": q,
                                "title": offer.get("title", "")[:120],
                                "url": offer.get("url", ""),
                                "file": str(dest.relative_to(out_dir)),
                            }, ensure_ascii=False) + "\n")
                            mf.flush()
                            seen.add(photo_id)
                            added += 1
                    offset += 40
                print(f"{q:14s} listings={got}", flush=True)
    finally:
        scraper.close()
    return added


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "data" / "damage_seg")
    ap.add_argument("--per-query", type=int, default=12,
                    help="listings to take per damage keyword")
    ap.add_argument("--max-photos", type=int, default=4,
                    help="photos per listing")
    args = ap.parse_args()
    n = harvest(args.out, args.per_query, args.max_photos)
    print(f"added {n} photos → {args.out}")


if __name__ == "__main__":
    main()
