#!/usr/bin/env python3
"""Compare local Ollama models on the enrichment task with per-field metrics.

Pipeline:
  1. Pull N random listings from the remote sqlite into `data/eval/sample.jsonl`.
  2. Read hand-verified ground-truth labels from `data/eval/oracle.jsonl`
     (one JSON object per olx_id). Produce that file out-of-band — by hand,
     by Claude in the conversation, or any other careful source. This script
     does NOT call any LLM API for oracle labels.
  3. For every candidate model (`--models gemma3:4b qwen3:4b-instruct …`)
     run extraction against the remote Ollama and cache responses to
     `data/eval/<model-tag>.jsonl`.
  4. Score each candidate against the oracle per-field, with type-aware metrics
     (string=case-insensitive equality, bool=accuracy, list=Jaccard, numeric=
     ±5 % tolerance, ordinal damage_severity = exact + ±1 tolerance).
  5. Print a table; the highest-mean accuracy is the recommendation.

Usage:
    python scripts/eval_model.py --sample-only --n 30   # step 1
    # then populate data/eval/oracle.jsonl by hand / Claude
    python scripts/eval_model.py --models gemma3:4b qwen3:4b-instruct
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

# Reuse the production prompt so the eval matches what the system actually sees.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.parser.llm_enrichment import _SYSTEM_PROMPT, _FIELD_NAMES  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = REPO_ROOT / "data" / "eval"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

REMOTE_HOST = "anastasia@192.168.1.77"
REMOTE_DB = "~/olx-car-parser/data/olx_cars.db"
# Eval picks a backend through the same sticky-per-thread load balancer the
# production enrichment uses, so a multi-host test exercises the real path.
# Set `--ollama` to override and pin to a single host (debug / single-host).
REMOTE_OLLAMA = None

SSH = ["/opt/homebrew/bin/sshpass", "-p", "1234", "ssh",
       "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=no", REMOTE_HOST]


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

SAMPLE_PATH = EVAL_DIR / "sample.jsonl"


def sample_listings(n: int, seed: int = 42) -> list[dict]:
    """Pull N (olx_id, title, description) tuples from the remote DB.

    Persists the exact sample so repeated runs compare identical inputs.
    Re-roll by deleting `data/eval/sample.jsonl`.
    """
    if SAMPLE_PATH.exists():
        rows = [json.loads(l) for l in SAMPLE_PATH.read_text().splitlines() if l.strip()]
        if len(rows) >= n:
            print(f"[sample] reusing {len(rows)} listings from {SAMPLE_PATH}", flush=True)
            return rows[:n]

    print(f"[sample] fetching {n} listings from remote DB…", flush=True)
    # Pin RANDOM() with sqlite's seed param so the sample is reproducible.
    sql = (
        f"SELECT olx_id, COALESCE(title, ''), "
        f"REPLACE(REPLACE(description, CHAR(10), ' '), CHAR(13), ' ') "
        f"FROM listings "
        f"WHERE description IS NOT NULL AND LENGTH(description) >= 50 "
        f"ORDER BY (olx_id || '{seed}') LIMIT {n};"
    )
    raw = subprocess.run(
        SSH + [f'sqlite3 -separator "|||" {REMOTE_DB} "{sql}"'],
        capture_output=True, text=True, timeout=60,
    ).stdout.strip()
    rows = []
    for line in raw.split("\n"):
        if "|||" not in line:
            continue
        parts = line.split("|||", 2)
        if len(parts) != 3:
            continue
        olx_id, title, desc = parts
        rows.append({"olx_id": olx_id.strip(), "title": title.strip(),
                     "description": desc.strip()})
    SAMPLE_PATH.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
    print(f"[sample] saved {len(rows)} listings to {SAMPLE_PATH}", flush=True)
    return rows


# ---------------------------------------------------------------------------
# Oracle (file-based — populated externally with hand-verified labels)
# ---------------------------------------------------------------------------

ORACLE_PATH = EVAL_DIR / "oracle.jsonl"


def _load_cached(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        out[rec["olx_id"]] = rec.get("extracted") or rec.get("oracle") or rec
    return out


def _append_jsonl(path: Path, rec: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_oracle(listings: list[dict]) -> dict[str, dict]:
    """Read ground-truth labels from disk. The file must contain one JSON
    object per olx_id with at least the fields we score on. Listings without
    a label are silently skipped (so a partial oracle still works)."""
    cached = _load_cached(ORACLE_PATH)
    have = sum(1 for l in listings if l["olx_id"] in cached)
    print(f"[oracle] {have}/{len(listings)} listings have ground-truth labels in "
          f"{ORACLE_PATH}", flush=True)
    if have == 0:
        sys.exit(f"{ORACLE_PATH} is empty — populate it before running candidates.")
    return cached


# ---------------------------------------------------------------------------
# Candidate models (remote Ollama)
# ---------------------------------------------------------------------------

def candidate_path(model: str) -> Path:
    return EVAL_DIR / f"{model.replace(':', '_').replace('/', '_')}.jsonl"


def call_ollama(model: str, listing: dict) -> dict | None:
    """Send one extraction request through the production load-balancer.

    Going through `_pick_ollama_url()` means an eval with `workers > 1`
    actually hits both .77 and .69 the same way the production enrichment
    does — so latency/distribution we measure here is what we'd see on a
    real batch.
    """
    from src.parser.llm_enrichment import _pick_ollama_url, _get_client
    text = f"{listing['title']}\n{listing['description']}"[:4000]
    payload = {
        "model": model,
        "system": _SYSTEM_PROMPT,
        "prompt": text,
        "format": "json",
        "stream": False,
        "keep_alive": "30m",
        "options": {
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 1.0,
            "repeat_penalty": 1.0,
            "num_ctx": 4096,
            "num_predict": 300,
            "stop": ["}\n{", "} {"],
        },
    }
    url = REMOTE_OLLAMA or _pick_ollama_url()
    if not url:
        return None
    try:
        r = _get_client(url).post("/api/generate", json=payload)
        if r.status_code != 200:
            return None
        content = r.json().get("response", "")
        if not content:
            return None
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            stripped = content.strip().strip("`").lstrip("json").strip()
            return json.loads(stripped) if stripped else None
    except Exception as e:  # noqa: BLE001
        print(f"  {model} err: {type(e).__name__}: {str(e)[:100]}", flush=True)
        return None


def label_with_candidate(model: str, listings: list[dict],
                         workers: int = 3) -> dict[str, dict]:
    path = candidate_path(model)
    cached = _load_cached(path)
    todo = [l for l in listings if l["olx_id"] not in cached]
    if not todo:
        print(f"[{model}] all {len(listings)} predictions cached", flush=True)
        return cached

    print(f"[{model}] running on {len(todo)} listings (workers={workers})", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(call_ollama, model, l): l for l in todo}
        done = 0
        for fut in as_completed(futures):
            listing = futures[fut]
            result = fut.result()
            if result is not None:
                _append_jsonl(path, {"olx_id": listing["olx_id"], "extracted": result})
                cached[listing["olx_id"]] = result
            done += 1
            if done % 10 == 0:
                print(f"  [{done}/{len(todo)}]", flush=True)
    return cached


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

# Field types: how to compare oracle vs candidate value.
BOOL_FIELDS = {"desc_mentions_accident", "desc_mentions_repair",
               "desc_mentions_customs_cleared", "right_hand_drive",
               "warranty", "taxi_fleet_rental", "first_owner_selling"}
ENUM_FIELDS = {"mechanical_condition", "urgency"}
STR_FIELDS = {"sub_model", "trim_level"}
INT_FIELDS = {"desc_mentions_num_owners"}
NUM_FIELDS = {"mileage_in_description_km"}      # ±5 % tolerance
ORD_FIELDS = {"damage_severity"}                # exact + ±1
LIST_FIELDS = {"tuning_or_mods"}                # Jaccard


def _norm(v):
    if isinstance(v, str):
        return v.strip().lower() or None
    return v


def _score_pair(field: str, oracle_val, cand_val) -> tuple[bool, bool]:
    """Return (counted, correct).  `counted=False` skips items where the
    oracle itself didn't commit (None) — we don't want to reward agreement on
    "I don't know" or punish a model that did commit."""
    ov, cv = _norm(oracle_val), _norm(cand_val)
    if ov is None and cv is None:
        return (False, False)         # both unknown — skip
    if ov is None:
        return (False, False)         # oracle uncertain — skip
    if cv is None:
        return (True, False)          # candidate punted, oracle had answer

    if field in BOOL_FIELDS or field in ENUM_FIELDS:
        return (True, ov == cv)
    if field in STR_FIELDS:
        return (True, ov == cv)
    if field in INT_FIELDS:
        try:
            return (True, int(ov) == int(cv))
        except (TypeError, ValueError):
            return (True, False)
    if field in NUM_FIELDS:
        try:
            o, c = float(ov), float(cv)
            return (True, abs(o - c) / max(o, 1) <= 0.05)
        except (TypeError, ValueError):
            return (True, False)
    if field in ORD_FIELDS:
        try:
            return (True, abs(int(ov) - int(cv)) <= 1)  # ±1 lenient
        except (TypeError, ValueError):
            return (True, False)
    if field in LIST_FIELDS:
        if not isinstance(ov, list) or not isinstance(cv, list):
            return (True, ov == cv)
        os_, cs = set(map(str.lower, map(str, ov))), set(map(str.lower, map(str, cv)))
        if not os_ and not cs:
            return (True, True)
        jacc = len(os_ & cs) / len(os_ | cs)
        return (True, jacc >= 0.5)
    return (True, ov == cv)


def score_candidate(oracle: dict[str, dict], cand: dict[str, dict]) -> dict[str, tuple[int, int]]:
    """Return {field: (n_counted, n_correct)} across all olx_ids."""
    by_field: dict[str, list[int]] = {f: [0, 0] for f in _FIELD_NAMES}
    for olx_id, oracle_val in oracle.items():
        cand_val = cand.get(olx_id)
        if cand_val is None:
            for f in _FIELD_NAMES:
                by_field[f][0] += 1            # candidate missing entirely
            continue
        for f in _FIELD_NAMES:
            counted, correct = _score_pair(f, oracle_val.get(f), cand_val.get(f))
            if counted:
                by_field[f][0] += 1
                by_field[f][1] += int(correct)
    return {f: tuple(v) for f, v in by_field.items()}


def print_table(scores: dict[str, dict[str, tuple[int, int]]]) -> None:
    models = list(scores.keys())
    field_w = max(len(f) for f in _FIELD_NAMES) + 2
    col_w = 14
    header = "field".ljust(field_w) + "".join(m.ljust(col_w) for m in models)
    print()
    print(header)
    print("-" * len(header))
    overall = {m: [0, 0] for m in models}
    for f in _FIELD_NAMES:
        line = f.ljust(field_w)
        for m in models:
            n, c = scores[m][f]
            overall[m][0] += n
            overall[m][1] += c
            cell = f"{c}/{n} ({c/max(n,1)*100:>5.1f}%)" if n else "n/a"
            line += cell.ljust(col_w)
        print(line)
    print("-" * len(header))
    line = "OVERALL".ljust(field_w)
    for m in models:
        n, c = overall[m]
        line += f"{c}/{n} ({c/max(n,1)*100:>5.1f}%)".ljust(col_w)
    print(line)
    print()
    if len(models) >= 2:
        ranked = sorted(models, key=lambda m: overall[m][1] / max(overall[m][0], 1), reverse=True)
        winner = ranked[0]
        runner = ranked[1]
        wn, wc = overall[winner]
        rn, rc = overall[runner]
        delta = wc / max(wn, 1) - rc / max(rn, 1)
        print(f"WINNER: {winner}  (+{delta*100:.1f}pp over {runner})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[],
                    help="Ollama model tags to compare, e.g. gemma3:4b qwen3:4b-instruct")
    ap.add_argument("--n", type=int, default=30, help="sample size")
    ap.add_argument("--workers", type=int, default=3,
                    help="parallel workers per candidate (match remote NUM_PARALLEL)")
    ap.add_argument("--sample-only", action="store_true",
                    help="just produce data/eval/sample.jsonl and exit (step 1)")
    args = ap.parse_args()

    listings = sample_listings(args.n)
    if args.sample_only:
        print(f"[done] sample saved to {SAMPLE_PATH}; populate "
              f"{ORACLE_PATH} with one JSON per olx_id, then re-run with --models.",
              flush=True)
        return
    if not args.models:
        sys.exit("--models is required (or pass --sample-only to just sample)")

    oracle = load_oracle(listings)

    scores = {}
    for model in args.models:
        cand = label_with_candidate(model, listings, workers=args.workers)
        # Score only the listings that have an oracle label.
        labeled_ids = set(oracle)
        cand_subset = {k: v for k, v in cand.items() if k in labeled_ids}
        scores[model] = score_candidate(oracle, cand_subset)

    print_table(scores)


if __name__ == "__main__":
    main()
