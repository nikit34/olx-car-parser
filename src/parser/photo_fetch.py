"""Photo URL extraction + download for OLX and StandVirtual listings.

Promoted from ``scripts/photo_verify_damage.py`` so production code paths
(``verify-photos`` CLI command) and the POC scripts that share the same
fetch logic don't fork copies.

What lives here:
- ``fetch_photos(url)`` — host-dispatched URL list, the canonical helper
  used by the ``verify-photos`` job.
- ``fetch_photos_olx`` / ``fetch_photos_standvirtual`` — per-source
  parsers (separate so callers that already know the host can skip the
  dispatch).
- ``fetch_standvirtual_advert(url)`` — returns the full advert dict from
  ``__NEXT_DATA__``; useful for the labelling sampler that needs photo
  URLs *and* the title/description in one fetch.
- ``download_photo(url, dest)`` — single-photo idempotent download.
- ``download_photos(urls, olx_id, cache_dir)`` — POC convenience that
  wraps download_photo with a per-listing subdir.
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path

import httpx

from src.parser.tls_fingerprint import build_ssl_context


_DEFAULT_HEADERS = {"User-Agent": "Mozilla/5.0"}
_DEFAULT_TIMEOUT = 30


# Module-level pooled client. ``verify-photos`` runs this through a
# ThreadPoolExecutor and fires ~10–20 photo downloads per listing, all
# against the same handful of hosts (apollo.olxcdn.com, standvirtual.com,
# olx.pt). One-shot ``httpx.get`` opens a fresh TCP connection per call,
# burns a fresh ephemeral port per call, and on 2026-05-10 starved the
# scrape host's ephemeral pool — 16 597 sockets stuck in TIME_WAIT
# against a 16 384-port range, so even ``git fetch`` started failing
# with "Can't assign requested address". A persistent Client keeps
# HTTP keep-alive across requests, so 20 sockets serve thousands of
# fetches and the pool never saturates.
_CLIENT = httpx.Client(
    # olx.pt listing pages (the source of the photo URLs) are behind the
    # same CDN rule that 403s httpx's stock handshake — see
    # src/parser/tls_fingerprint. The image CDN itself doesn't care, but
    # this client fetches both.
    verify=build_ssl_context(),
    headers=_DEFAULT_HEADERS,
    timeout=_DEFAULT_TIMEOUT,
    follow_redirects=True,
    limits=httpx.Limits(
        max_keepalive_connections=20,
        max_connections=50,
        keepalive_expiry=60.0,
    ),
)


# OLX serves photos via ``ireland.apollo.olxcdn.com`` with size variants —
# ``;s=4080x3072`` is the original, ``;s=1000x700`` is medium. The 1000x700
# variant is a good balance: ~50–150 KB/photo, classifier resizes to 224
# anyway. Listing-own photos always have ≥1000-px variants; small thumbnails
# of related listings only appear at ≤516-px sizes.
_OLX_PHOTO_RE = re.compile(
    r"apollo\.olxcdn\.com[:\d]*/v1/files/([\w-]+)-PT/image"
    r"(?:;s=(\d+)x(\d+))?"
)
_NEXT_DATA_RE = re.compile(
    r'<script id="__NEXT_DATA__"[^>]*>(.+?)</script>',
    re.DOTALL,
)


def fetch_standvirtual_advert(url: str) -> dict:
    """Return the StandVirtual advert dict from ``__NEXT_DATA__``.

    Empty dict on any HTTP / parsing failure — callers decide whether
    that means "skip this listing" or "raise".
    """
    try:
        r = _CLIENT.get(url)
        r.raise_for_status()
    except httpx.HTTPError:
        return {}
    m = _NEXT_DATA_RE.search(r.text)
    if not m:
        return {}
    try:
        data = json.loads(m.group(1))
        return data["props"]["pageProps"]["advert"]
    except (KeyError, json.JSONDecodeError):
        return {}


def fetch_photos_standvirtual(url: str) -> list[str]:
    """Extract photo URLs from a StandVirtual listing page."""
    ad = fetch_standvirtual_advert(url)
    if not ad:
        return []
    try:
        return [p["url"] for p in ad["images"]["photos"]]
    except (KeyError, TypeError):
        return []


def fetch_photos_olx(url: str) -> list[str]:
    """Extract photo URLs from an OLX listing page.

    Scans the rendered HTML for ``apollo.olxcdn.com`` photo URLs, dedupes
    by photo ID (preserving page order), filters out related-listing
    thumbnails (require at least one 1000-px+ size variant), and returns
    1000x700-sized URLs.
    """
    try:
        r = _CLIENT.get(url)
        r.raise_for_status()
    except httpx.HTTPError:
        return []

    sizes_by_id: dict[str, set[int]] = {}
    order: list[str] = []
    for m in _OLX_PHOTO_RE.finditer(r.text):
        pid = m.group(1)
        if pid not in sizes_by_id:
            sizes_by_id[pid] = set()
            order.append(pid)
        if m.group(2):
            sizes_by_id[pid].add(int(m.group(2)))

    return [
        f"https://ireland.apollo.olxcdn.com:443/v1/files/{pid}-PT/image;s=1000x700"
        for pid in order
        if any(w >= 1000 for w in sizes_by_id[pid])
    ]


def fetch_photos_olx_from_api(offer: dict) -> list[str]:
    """Build OLX photo URLs from a JSON-API offer's ``photos`` list.

    No HTTP request — the API list payload already carries each photo's URL
    template (``...;s={width}x{height}``); we just render it at 1000x700 to
    match :func:`fetch_photos_olx`.
    """
    urls: list[str] = []
    for p in offer.get("photos") or []:
        link = (p or {}).get("link") or ""
        if not link:
            continue
        if "{width}" in link:
            link = link.replace("{width}", "1000").replace("{height}", "700")
        urls.append(link)
    return urls


def fetch_photos(url: str) -> list[str]:
    """Dispatch by URL host: OLX or StandVirtual."""
    if "olx.pt" in url:
        return fetch_photos_olx(url)
    if "standvirtual.com" in url:
        return fetch_photos_standvirtual(url)
    return []


def download_photo(url: str, dest: Path) -> bool:
    """Download a photo to ``dest``, no-op if it already exists."""
    if dest.exists():
        return True
    try:
        r = _CLIENT.get(url)
        r.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(r.content)
        return True
    except httpx.HTTPError:
        return False


def download_photos(
    urls: list[str], olx_id: str, cache_dir: Path,
) -> list[Path]:
    """Bulk-download a listing's photos under ``cache_dir/{olx_id}_{i}.jpg``.

    Skips any photo that fails to download — the returned list contains
    only successfully-fetched paths, in the same order as ``urls``.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i, url in enumerate(urls, 1):
        p = cache_dir / f"{olx_id}_{i}.jpg"
        if download_photo(url, p):
            paths.append(p)
    return paths


_SV_JWT_RE = re.compile(r"/v1/files/([\w-]+)\.([\w-]+)\.([\w-]+)/image")
_PLAIN_FILE_RE = re.compile(r"/v1/files/([\w-]+)/image")


def photo_id_from_url(url: str) -> str | None:
    """The CDN's own file id behind a photo URL, for either platform.

    OLX puts the id straight in the path. StandVirtual signs a JWT whose
    payload names the file in ``fn`` (and the watermark overlay in ``w``,
    which is the same image for every advert and is ignored).
    """
    if not url:
        return None
    m = _SV_JWT_RE.search(url)
    if m:
        payload = m.group(2)
        payload += "=" * (-len(payload) % 4)
        try:
            return json.loads(base64.urlsafe_b64decode(payload)).get("fn")
        except (ValueError, TypeError):
            return None
    m = _PLAIN_FILE_RE.search(url)
    return m.group(1) if m else None


def photo_refs_olx_api(offer: dict) -> list[tuple[str, None]]:
    """``(file id, None)`` per photo of a JSON-API offer, in gallery order.

    No URL is kept: an OLX photo URL is a plain path around the id, so it is
    rebuilt on demand instead of stored.
    """
    refs: list[tuple[str, None]] = []
    for p in offer.get("photos") or []:
        fid = (p or {}).get("filename") or photo_id_from_url((p or {}).get("link") or "")
        if fid:
            refs.append((fid, None))
    return refs


def photo_refs_standvirtual(advert: dict) -> list[tuple[str, str]]:
    """``(file id, signed URL)`` per photo of a StandVirtual advert.

    The URL is carried along because it cannot be rebuilt from the id — the
    JWT is signed by StandVirtual — and the image still has to be fetched once
    to be hashed.
    """
    refs: list[tuple[str, str]] = []
    for p in ((advert.get("images") or {}).get("photos") or []):
        url = (p or {}).get("url") or (p or {}).get("id") or ""
        fid = photo_id_from_url(url)
        if fid:
            refs.append((fid, url))
    return refs
