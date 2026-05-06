"""Harvest damaged-car listings from Standvirtual + OLX-PT search results.

Hits each search URL with damage keywords, paginates a few pages, deduplicates,
downloads photos for each listing, and writes a manifest.jsonl that mirrors
photo_damage_label_sampler.py's format so the same labeling flow works.

Output:
  /tmp/damage_harvest/manifest.jsonl
  /tmp/damage_harvest/<olx_id>_<idx>.jpg
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

UA = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0 Safari/537.36"
    )
}

SV_KEYWORDS = ["sinistrado", "batido", "para+pecas", "desmancha"]
OLX_KEYWORDS = ["sinistrado", "para-pecas", "batido"]

SV_TEMPLATE = "https://www.standvirtual.com/carros?search%5Bfilter_search%5D={kw}&page={page}"
OLX_TEMPLATE = "https://www.olx.pt/carros-motos-e-barcos/carros/q-{kw}/?page={page}"


def fetch(url: str, retries: int = 2) -> str | None:
    for attempt in range(retries + 1):
        try:
            r = httpx.get(url, headers=UA, follow_redirects=True, timeout=20)
            if r.status_code == 200:
                return r.text
            print(f"  http {r.status_code} on {url}")
        except Exception as e:
            print(f"  error on {url}: {e}")
        time.sleep(1.5)
    return None


def extract_sv_listing_urls(html: str) -> list[str]:
    out = []
    soup = BeautifulSoup(html, "lxml")
    for art in soup.find_all("article"):
        a = art.find("a", href=True)
        if not a:
            continue
        url = a.get("href", "")
        if "/anuncio/" in url:
            if not url.startswith("http"):
                url = "https://www.standvirtual.com" + url
            out.append(url)
    return out


def extract_olx_listing_urls(html: str) -> list[str]:
    out = []
    soup = BeautifulSoup(html, "lxml")
    for a in soup.find_all("a", href=True):
        url = a.get("href", "")
        if "/d/anuncio/" in url:
            if not url.startswith("http"):
                url = "https://www.olx.pt" + url
            out.append(url.split("?")[0])
    return out


def fetch_sv_detail(url: str) -> dict | None:
    """Use NEXT_DATA blob like photo_damage_label_sampler does."""
    html = fetch(url)
    if not html:
        return None
    m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.+?)</script>', html, re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(1))
        ad = data["props"]["pageProps"]["advert"]
    except (KeyError, json.JSONDecodeError):
        return None
    photos = [p["url"] for p in ad.get("images", {}).get("photos", [])]
    olx_id_m = re.search(r"ID(\w+)\.html", url)
    return {
        "olx_id": olx_id_m.group(1) if olx_id_m else "",
        "url": url,
        "title": ad.get("title", ""),
        "description": (ad.get("description") or "")[:500],
        "photos": photos,
        "source": "standvirtual",
    }


def fetch_olx_detail(url: str) -> dict | None:
    """Pull og:image and ad-photos imgs from OLX detail HTML."""
    html = fetch(url)
    if not html:
        return None
    soup = BeautifulSoup(html, "lxml")

    # Title
    title_el = soup.find("h4") or soup.find("h1")
    title = title_el.get_text(strip=True) if title_el else ""

    # Description
    desc_el = soup.find(attrs={"data-testid": "ad_description"})
    desc = desc_el.get_text(" ", strip=True) if desc_el else ""

    # olx_id
    id_m = re.search(r"ID(\w+)\.html", url)
    olx_id = id_m.group(1) if id_m else ""
    if not olx_id:
        for el in soup.find_all(string=re.compile(r"ID:\s*\d+")):
            m = re.search(r"ID:\s*(\d+)", el)
            if m:
                olx_id = m.group(1)
                break

    # Photos: prefer gallery imgs; fallback to og:image
    photos: list[str] = []
    gallery = soup.select_one("[data-testid='photo-gallery']") or \
              soup.select_one("[data-cy='ad-photos']")
    if gallery:
        for img in gallery.find_all("img"):
            src = img.get("src") or img.get("data-src") or ""
            if src.startswith("http"):
                photos.append(src)
    # OLX often shows only first image in DOM and lazy-loads others.
    # Try og:image meta tags as a backup.
    if not photos:
        for meta in soup.find_all("meta", property="og:image"):
            c = meta.get("content")
            if c:
                photos.append(c)

    # Dedup while preserving order
    seen = set()
    uniq = []
    for p in photos:
        if p not in seen:
            seen.add(p)
            uniq.append(p)

    return {
        "olx_id": olx_id,
        "url": url,
        "title": title,
        "description": desc[:500],
        "photos": uniq,
        "source": "olx",
    }


def harvest(out_dir: Path, manifest_path: Path,
            max_listings: int, max_pages: int, max_photos_per: int):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Existing manifest IDs to skip
    excl: set[str] = set()
    if manifest_path.exists():
        for line in manifest_path.read_text().splitlines():
            if line.strip():
                excl.add(json.loads(line)["olx_id"])
    print(f"Skipping {len(excl)} already-harvested listings")

    listing_urls: list[tuple[str, str]] = []  # (url, source)

    print("\n--- Standvirtual ---")
    for kw in SV_KEYWORDS:
        for page in range(1, max_pages + 1):
            url = SV_TEMPLATE.format(kw=kw, page=page)
            html = fetch(url)
            if not html:
                break
            urls = extract_sv_listing_urls(html)
            new = [(u, "standvirtual") for u in urls if u not in {x[0] for x in listing_urls}]
            print(f"  kw={kw:14s} page={page}  +{len(new)} unique")
            listing_urls.extend(new)
            if not urls:
                break
            time.sleep(0.8)

    print("\n--- OLX ---")
    for kw in OLX_KEYWORDS:
        for page in range(1, max_pages + 1):
            url = OLX_TEMPLATE.format(kw=kw, page=page)
            html = fetch(url)
            if not html:
                break
            urls = extract_olx_listing_urls(html)
            new = [(u, "olx") for u in urls if u not in {x[0] for x in listing_urls}]
            print(f"  kw={kw:14s} page={page}  +{len(new)} unique")
            listing_urls.extend(new)
            if not urls:
                break
            time.sleep(0.8)

    print(f"\nTotal candidate listings: {len(listing_urls)}")

    written = failures = 0
    out_records: list[dict] = []
    for i, (url, source) in enumerate(listing_urls, 1):
        if written >= max_listings:
            break
        print(f"\n[{i}/{len(listing_urls)}] {source}  {url}")
        if source == "standvirtual":
            d = fetch_sv_detail(url)
        else:
            d = fetch_olx_detail(url)
        if not d or not d.get("photos") or not d.get("olx_id"):
            print("  ! skip (no photos / no id)")
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
            "source": source,
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
    ap.add_argument("--max-listings", type=int, default=40)
    ap.add_argument("--max-pages", type=int, default=2)
    ap.add_argument("--max-photos-per", type=int, default=8)
    args = ap.parse_args()

    harvest(args.out_dir, args.manifest, args.max_listings,
            args.max_pages, args.max_photos_per)


if __name__ == "__main__":
    main()
