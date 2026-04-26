#!/usr/bin/env python3
"""Batch-enrich listings via local Ollama and push results to the remote DB.

Pulls un-enriched rows from the remote sqlite over SSH, sends each description
to Ollama (`qwen3:4b-instruct` by default — overridable via settings.yaml),
and writes back the JSON in batches.
"""
from __future__ import annotations

import sys
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

# Reuse the exact prompt + config as the inline pipeline so train/eval/inference
# stay on one definition.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.parser.llm_enrichment import _SYSTEM_PROMPT, _get_config  # noqa: E402


DB = "~/olx-car-parser/data/olx_cars.db"
BATCH = 50           # rows per fetch
WORKERS = 6          # parallel Ollama requests; M1 8 GB tops out around here
PUSH_TIMEOUT = 120   # ssh sqlite write timeout (s)

SSH = ["/opt/homebrew/bin/sshpass", "-p", "1234", "ssh", "-o", "ConnectTimeout=10",
       "anastasia@192.168.1.77"]


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


def enrich_one(client: httpx.Client, model: str, max_tokens: int, max_chars: int,
               desc: str) -> dict | None:
    """One Ollama /api/generate call. Returns the parsed JSON dict or None.

    /api/generate (rather than /api/chat) keeps the system prompt byte-stable
    across calls so Ollama reuses the KV-cache slot and skips re-prefilling
    the ~600-token instruction block on every request — by far the biggest
    per-call latency win on an M1 8 GB.
    """
    payload = {
        "model": model,
        "system": _SYSTEM_PROMPT,
        "prompt": desc[:max_chars],
        "format": "json",
        "stream": False,
        "keep_alive": "30m",
        "options": {"temperature": 0.0, "num_predict": max_tokens},
    }
    try:
        resp = client.post("/api/generate", json=payload)
        if resp.status_code != 200:
            return None
        content = resp.json().get("response", "")
        if not content:
            return None
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            stripped = content.strip().strip("`").lstrip("json").strip()
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
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
    cfg = _get_config()
    model = cfg["ollama_model"]
    url = cfg["ollama_url"]
    max_tokens = cfg.get("max_tokens", 600)
    max_chars = cfg.get("max_chars", 4000)

    # Probe early so we fail fast if Ollama is not running.
    try:
        with httpx.Client(base_url=url, timeout=5.0) as probe:
            probe.get("/api/tags").raise_for_status()
    except Exception as e:  # noqa: BLE001
        print(f"Ollama not reachable at {url}: {e}", flush=True)
        sys.exit(1)

    # One shared client across the pool — httpx.Client is thread-safe and
    # reuses the underlying connection pool, which keeps the per-call latency
    # close to model inference time.
    client = httpx.Client(base_url=url, timeout=httpx.Timeout(120.0, connect=10.0))

    print(f"Enriching with Ollama {model} @ {url}, {WORKERS} workers", flush=True)

    total = 0
    failed = 0
    started = time.monotonic()

    try:
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
            print(f"done. Total: {total} enriched, {failed} failed, {rate:.2f}/s.",
                  flush=True)
    finally:
        client.close()


if __name__ == "__main__":
    main()
