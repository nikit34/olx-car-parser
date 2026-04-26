#!/usr/bin/env python3
"""Batch-enrich listings via Claude API and push results to the remote DB.

Pulls un-enriched rows from the remote sqlite over SSH, sends each description
to Claude (Haiku 4.5) with prompt caching + tool-use, and writes back the JSON
in batches.  Designed for high parallelism — 20+ inflight requests against the
Anthropic API are network-bound and don't stress the local box.
"""
from __future__ import annotations

import os
import sys
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Make the package importable so we reuse the exact prompt + schema as the
# inline pipeline — keeps train/eval/inference all on one definition.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.parser.llm_enrichment import (  # noqa: E402
    _SYSTEM_PROMPT,
    _EXTRACT_TOOL,
    _get_token,
    _get_config,
)


DB = "~/olx-car-parser/data/olx_cars.db"
BATCH = 50           # rows per fetch
WORKERS = 20         # parallel Claude calls
PUSH_TIMEOUT = 120   # ssh sqlite write timeout (s)

SSH = ["/opt/homebrew/bin/sshpass", "-p", "1234", "ssh", "-o", "ConnectTimeout=10",
       "anastasia@192.168.1.74"]


def ssh_cmd(cmd: str, timeout: int = 30) -> str:
    r = subprocess.run(SSH + [cmd], capture_output=True, text=True, timeout=timeout)
    return r.stdout.strip()


def fetch_batch() -> list[tuple[str, str]]:
    """Pull a random page of un-enriched (olx_id, description) tuples."""
    rows = ssh_cmd(
        f'sqlite3 -separator "|||" {DB} '
        f'"SELECT olx_id, REPLACE(REPLACE(description, CHAR(10), \\\" \\\"), CHAR(13), \\\" \\\") '
        f"FROM listings WHERE llm_extras IS NULL AND description IS NOT NULL "
        f'AND LENGTH(description) >= 20 ORDER BY RANDOM() LIMIT {BATCH};"'
    )
    if not rows:
        return []
    out = []
    for line in rows.split("\n"):
        if "|||" not in line:
            continue
        olx_id, desc = line.split("|||", 1)
        out.append((olx_id.strip(), desc.strip()))
    return out


def enrich_one(client, model: str, max_tokens: int, max_chars: int, desc: str) -> dict | None:
    """One Claude call. Returns the tool input dict or None."""
    try:
        import anthropic
    except ImportError:
        return None
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=[{
                "type": "text",
                "text": _SYSTEM_PROMPT,
                # Prompt caching: identical system prompt across all workers
                # turns the ~600-token instruction block into a 0.1× cost
                # after the first hit in each 5-minute window.
                "cache_control": {"type": "ephemeral"},
            }],
            tools=[_EXTRACT_TOOL],
            tool_choice={"type": "tool", "name": "record_listing_features"},
            messages=[{"role": "user", "content": desc[:max_chars]}],
        )
        for block in resp.content:
            if getattr(block, "type", None) == "tool_use" and block.name == "record_listing_features":
                return dict(block.input)
        return None
    except anthropic.RateLimitError:
        time.sleep(2)
        return None
    except anthropic.APIStatusError as e:
        if 400 <= e.status_code < 500:
            print(f"  API {e.status_code}: {e.message[:120]}", flush=True)
        return None
    except Exception as e:  # noqa: BLE001
        print(f"  Err: {type(e).__name__}: {str(e)[:120]}", flush=True)
        return None


def push_batch(results: list[tuple[str, dict]]) -> None:
    """Stream all UPDATE statements over a single SSH session."""
    if not results:
        return
    stmts = ["PRAGMA busy_timeout = 30000;", "BEGIN;"]
    for olx_id, data in results:
        j = json.dumps(data, ensure_ascii=False).replace("'", "''")
        stmts.append(
            f"UPDATE listings SET llm_extras='{j}' "
            f"WHERE olx_id='{olx_id}' AND llm_extras IS NULL;"
        )
    stmts.append("COMMIT;")
    sql = "\n".join(stmts)

    for attempt in range(3):
        try:
            r = subprocess.run(
                SSH + [f"sqlite3 {DB}"],
                input=sql, capture_output=True, text=True, timeout=PUSH_TIMEOUT,
            )
            if r.returncode == 0:
                return
            print(f"  Push retry {attempt+1}: {r.stderr[:80]}", flush=True)
        except subprocess.TimeoutExpired:
            print(f"  Push timeout (attempt {attempt+1})", flush=True)
        time.sleep(5)


def main() -> None:
    token = _get_token()
    if not token:
        print("ANTHROPIC_AUTH_TOKEN not set (check .env).", flush=True)
        sys.exit(1)

    try:
        import anthropic
    except ImportError:
        print("`anthropic` not installed. Run: pip install anthropic", flush=True)
        sys.exit(1)

    cfg = _get_config()
    model = cfg["model"]
    max_tokens = cfg.get("max_tokens", 600)
    max_chars = cfg.get("max_chars", 4000)

    # One shared client across the whole pool — anthropic.Anthropic is
    # thread-safe (httpx-based) and reuses the underlying connection pool.
    client = anthropic.Anthropic(auth_token=token, max_retries=3, timeout=60.0)

    print(f"Enriching with Claude {model}, {WORKERS} workers", flush=True)

    total = 0
    failed = 0
    started = time.monotonic()

    while True:
        batch = fetch_batch()
        if not batch:
            elapsed = time.monotonic() - started
            print(f"Done! Enriched {total} ({failed} failed) in {elapsed:.1f}s "
                  f"({total / max(elapsed, 1):.2f}/s).", flush=True)
            break

        print(f"Batch {len(batch)}...", end=" ", flush=True)
        results: list[tuple[str, dict]] = []

        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = {
                pool.submit(enrich_one, client, model, max_tokens, max_chars, desc): olx_id
                for olx_id, desc in batch
            }
            for fut in as_completed(futures):
                olx_id = futures[fut]
                result = fut.result()
                if result:
                    results.append((olx_id, result))
                    total += 1
                else:
                    # Empty {} marks the row "tried but failed" so the next
                    # run doesn't pick it up again immediately.
                    results.append((olx_id, {}))
                    failed += 1

        push_batch(results)
        rate = total / max(time.monotonic() - started, 1)
        print(f"done. Total: {total} enriched, {failed} failed, {rate:.2f}/s.", flush=True)


if __name__ == "__main__":
    main()
