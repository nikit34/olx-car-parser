"""OLX-only damage harvest using high-yield keywords.

Targets q-salvado, q-desmancha, q-para-abate, q-sinistrado on OLX-PT.
Extracts photos from olxcdn URLs in the detail HTML.

Output: appends to /tmp/damage_harvest/manifest.jsonl (same file as the
StandVirtual harvest), so labeling/training pipeline picks both up.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/119.0"}

OLX_KEYWORDS = ["salvado", "desmancha", "para-abate", "sinistrado", "para-pecas", "batido"]
OLX_TEMPLATE = "https://www.olx.pt/carros-motos-e-barcos/carros/q-{kw}/?page={page}"

# Photo URL pattern on OLX-PT
OLX_PHOTO_RX = re.compile(
    r'https://ireland\.apollo\.olxcdn\.com:443/v1/files/[^";]+-PT/image(?:;[^"]*)?'
)


def fetch(url: str, retries: int = 2) -> str | None:
    for _ in range(retries + 1):
        try:
            r = httpx.get(url, headers=UA, follow_redirects=True, timeout=20)
            if r.status_code == 200:
                return r.text
        except Exception as e:
            print(f"  err {url}: {e}")
        time.sleep(1.0)
    return None


def extract_olx_listing_urls(html: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    out = []
    for a in soup.find_all("a", href=True):
        href = a.get("href", "")
        if "/d/anuncio/" in href:
            url = href if href.startswith("http") else "https://www.olx.pt" + href
            out.append(url.split("?")[0])
    return list(dict.fromkeys(out))


def fetch_olx_detail(url: str) -> dict | None:
    html = fetch(url)
    if not html:
        return None
    soup = BeautifulSoup(html, "lxml")

    title_el = soup.find("h4") or soup.find("h1")
    title = title_el.get_text(strip=True) if title_el else ""

    desc_el = soup.find(attrs={"data-testid": "ad_description"}) or \
              soup.find(attrs={"data-cy": "ad_description"})
    desc = desc_el.get_text(" ", strip=True) if desc_el else ""

    # ID from URL
    id_m = re.search(r"ID([A-Za-z0-9]+)\.html", url)
    olx_id = id_m.group(1) if id_m else ""

    # Photos: extract olxcdn URLs and dedup by base path (strip ;s=... size variant)
    raw = OLX_PHOTO_RX.findall(html)
    seen = set()
    photos = []
    for u in raw:
        base = u.split(";")[0]
        if base in seen:
            continue
        seen.add(base)
        photos.append(base)

    return {
        "olx_id": olx_id,
        "url": url,
        "title": title,
        "description": (desc or "")[:500],
        "photos": photos,
        "source": "olx",
    }


def harvest(out_dir: Path, manifest_path: Path,
            max_listings: int, max_pages: int, max_photos_per: int):
    out_dir.mkdir(parents=True, exist_ok=True)

    excl: set[str] = set()
    if manifest_path.exists():
        for line in manifest_path.read_text().splitlines():
            if line.strip():
                excl.add(json.loads(line)["olx_id"])
    print(f"Skipping {len(excl)} already-harvested listings")

    listing_urls: list[str] = []
    seen_urls: set[str] = set()

    print("\n--- OLX search collection ---")
    for kw in OLX_KEYWORDS:
        for page in range(1, max_pages + 1):
            url = OLX_TEMPLATE.format(kw=kw, page=page)
            html = fetch(url)
            if not html:
                break
            urls = extract_olx_listing_urls(html)
            new = [u for u in urls if u not in seen_urls]
            seen_urls.update(new)
            listing_urls.extend(new)
            print(f"  kw={kw:14s} page={page}  +{len(new)} unique")
            if not urls:
                break
            time.sleep(0.7)

    print(f"\nTotal candidate OLX listings: {len(listing_urls)}")

    written = failures = 0
    out_records: list[dict] = []
    for i, url in enumerate(listing_urls, 1):
        if written >= max_listings:
            break
        print(f"\n[{i}/{len(listing_urls)}] {url}")
        d = fetch_olx_detail(url)
        if not d or not d.get("photos") or not d.get("olx_id"):
            print(f"  ! skip (id={d.get('olx_id') if d else None}, photos={len(d.get('photos',[])) if d else 0})")
            failures += 1
            continue
        if d["olx_id"] in excl:
            print(f"  ! already harvested {d['olx_id']}")
            continue
        excl.add(d["olx_id"])

        photos_meta = []
        for j, purl in enumerate(d["photos"][:max_photos_per], 1):
            local = out_dir / f"{d['olx_id']}_{j}.jpg"
            if not local.exists():
                try:
                    r = httpx.get(purl, headers=UA, follow_redirects=True, timeout=30)
                    r.raise_for_status()
                    local.write_bytes(r.content)
                except Exception as e:
                    print(f"  ! photo {j} dl failed: {e}")
                    continue
            photos_meta.append({"idx": j, "path": str(local), "url": purl})

        if not photos_meta:
            print("  ! no photos downloaded")
            failures += 1
            continue

        print(f"  {d['olx_id']}: {len(photos_meta)} photos cached  | {d['title'][:60]}")
        out_records.append({
            "olx_id": d["olx_id"],
            "url": d["url"],
            "title": d["title"],
            "source": "olx",
            "harvest_keyword_match": True,
            "description_head": d["description"][:300],
            "photos": photos_meta,
        })
        written += 1
        time.sleep(0.5)

    with manifest_path.open("a") as f:
        for r in out_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nManifest:  {manifest_path}  (+{len(out_records)} listings)")
    print(f"Photos:    {out_dir}/  ({sum(len(r['photos']) for r in out_records)} total)")
    print(f"Failures:  {failures}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("/tmp/damage_harvest"))
    ap.add_argument("--manifest", type=Path,
                    default=Path("/tmp/damage_harvest/manifest.jsonl"))
    ap.add_argument("--max-listings", type=int, default=80)
    ap.add_argument("--max-pages", type=int, default=3)
    ap.add_argument("--max-photos-per", type=int, default=8)
    args = ap.parse_args()

    harvest(args.out_dir, args.manifest, args.max_listings,
            args.max_pages, args.max_photos_per)


if __name__ == "__main__":
    main()
