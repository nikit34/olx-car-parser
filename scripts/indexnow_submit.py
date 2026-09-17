#!/usr/bin/env python3
r"""Submit sitemap URLs to IndexNow (Bing, Yandex, Seznam, Naver).

    python3 scripts/indexnow_submit.py --locales de,fr,it --dry-run
    python3 scripts/indexnow_submit.py --locales de,fr,it
    python3 scripts/indexnow_submit.py --locales pt --match '/preco/[^/]+/\d{4}$'

The key must already answer at https://<host>/<key>.txt; the Worker serves it
from the INDEXNOW_KEY var. The script refuses to submit until it has checked
that, because IndexNow rejects the whole batch on a key it cannot verify.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request

ENDPOINT = "https://api.indexnow.org/indexnow"
BATCH = 10000
UA = "carsbuyer-indexnow/1.0"


def fetch(url: str, timeout: int = 30) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def sitemap_urls(host: str, locale: str) -> list[str]:
    xml = fetch(f"https://{host}/{locale}/sitemap.xml")
    return re.findall(r"<loc>([^<]+)</loc>", xml)


def verify_key(host: str, key: str) -> bool:
    try:
        return fetch(f"https://{host}/{key}.txt", timeout=15).strip() == key
    except urllib.error.URLError:
        return False


def submit(host: str, key: str, urls: list[str]) -> tuple[int, str]:
    payload = json.dumps({
        "host": host,
        "key": key,
        "keyLocation": f"https://{host}/{key}.txt",
        "urlList": urls,
    }).encode()
    req = urllib.request.Request(ENDPOINT, data=payload, method="POST",
                                 headers={"Content-Type": "application/json; charset=utf-8",
                                          "User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, r.read().decode("utf-8", "replace")[:300]
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")[:300]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="carsbuyer.org")
    ap.add_argument("--key", default="0b2d1d9109041a8e753860fe1b6efb6f")
    ap.add_argument("--locales", default="de,fr,it")
    ap.add_argument("--limit", type=int, default=0, help="submit at most N urls per locale")
    ap.add_argument("--match", default="", help="submit only urls matching this regex")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not verify_key(args.host, args.key):
        print(f"key file https://{args.host}/{args.key}.txt does not answer with the key; "
              f"deploy INDEXNOW_KEY first", file=sys.stderr)
        return 1

    try:
        pattern = re.compile(args.match) if args.match else None
    except re.error as err:
        print(f"--match is not a valid regex: {err}", file=sys.stderr)
        return 1

    urls: list[str] = []
    for loc in [x.strip() for x in args.locales.split(",") if x.strip()]:
        got = sitemap_urls(args.host, loc)
        if pattern is not None:
            before = len(got)
            got = [u for u in got if pattern.search(u)]
            print(f"{loc}: {len(got)} of {before} urls match {args.match!r}")
            if not got:
                print(f"{loc}: nothing matched; refusing to submit a guess", file=sys.stderr)
                return 1
        if args.limit:
            got = got[:args.limit]
        if pattern is None:
            print(f"{loc}: {len(got)} urls")
        urls.extend(got)

    seen, ordered = set(), []
    for u in urls:
        if u not in seen:
            seen.add(u)
            ordered.append(u)
    print(f"total {len(ordered)} unique urls")

    if args.dry_run:
        for u in ordered[:5]:
            print(f"  {u}")
        print("dry run, nothing submitted")
        return 0

    for i in range(0, len(ordered), BATCH):
        chunk = ordered[i:i + BATCH]
        status, body = submit(args.host, args.key, chunk)
        print(f"batch {i // BATCH + 1}: {len(chunk)} urls -> HTTP {status} {body.strip()}")
        if status not in (200, 202):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
