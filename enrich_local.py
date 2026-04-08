#!/usr/bin/env python3
"""One-off script: enrich listings locally, push results to remote DB in batches."""
import sys
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import httpx

DB = "~/olx-car-parser/data/olx_cars.db"
OLLAMA_URL = "http://localhost:11434"
MODEL = "qwen3:4b-instruct"
BATCH = 50

PROMPT = """\
Extract structured data from this Portuguese car listing. JSON fields (null if unknown):
desc_mentions_num_owners(int), desc_mentions_accident(bool), accident_details(str), service_history(bool), \
desc_mentions_repair(bool), repair_details(str), \
mileage_in_description_km(int), desc_mentions_customs_cleared(bool), imported(bool), \
right_hand_drive(bool), \
mechanical_condition("excellent"/"good"/"fair"/"poor"), paint_condition(same), \
suspicious_signs(list), extras(list), issues(list), reason_for_sale(str), \
urgency("high"/"medium"/"low"), warranty(bool), tuning_or_mods(list), \
taxi_fleet_rental(bool), recent_maintenance(list), tires_condition("new"/"good"/"fair"/"poor"), \
first_owner_selling(bool).
Rules: mileage_in_description_km=exact km as integer. \
desc_mentions_repair=true if ANY repair/damage mentioned. desc_mentions_accident=true if collision mentioned. \
If "para pecas","vender as pecas","venda de pecas","para desmanchar","so pecas": \
mechanical_condition="poor", suspicious_signs must include "selling for parts", \
desc_mentions_accident=true (likely total loss), reason_for_sale="para pecas (total loss or registration issue)".

"""

SSH = ["/opt/homebrew/bin/sshpass", "-p", "1234", "ssh", "-o", "ConnectTimeout=10",
       "anastasia@192.168.1.74"]


def ssh_cmd(cmd, timeout=30):
    r = subprocess.run(SSH + [cmd], capture_output=True, text=True, timeout=timeout)
    return r.stdout.strip()


def fetch_batch():
    rows = ssh_cmd(
        f'sqlite3 -separator "|||" {DB} '
        f'"SELECT olx_id, REPLACE(REPLACE(description, CHAR(10), \\\" \\\"), CHAR(13), \\\" \\\") '
        f"FROM listings WHERE llm_extras IS NULL AND description IS NOT NULL "
        f'AND LENGTH(description) >= 20 ORDER BY RANDOM() LIMIT {BATCH};"'
    )
    if not rows:
        return []
    result = []
    for line in rows.split("\n"):
        if "|||" not in line:
            continue
        olx_id, desc = line.split("|||", 1)
        result.append((olx_id.strip(), desc.strip()))
    return result


def enrich_one(desc, client):
    try:
        resp = client.post(
            "/api/generate",
            json={
                "model": MODEL,
                "prompt": PROMPT + desc[:1200],
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 512, "stop": ["} {", "}\n{"]},
            },
        )
        if resp.status_code != 200:
            return None
        content = resp.json()["response"]
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            return json.loads(content.strip())
        start = content.find("{")
        if start >= 0:
            depth = 0
            for i in range(start, len(content)):
                if content[i] == "{":
                    depth += 1
                elif content[i] == "}":
                    depth -= 1
                if depth == 0:
                    return json.loads(content[start:i + 1])
        return None
    except Exception as e:
        print(f"  Error: {e}", flush=True)
        return None


def push_batch(results):
    """Push all results in one SSH call using a SQL script."""
    if not results:
        return
    stmts = []
    for olx_id, data in results:
        j = json.dumps(data, ensure_ascii=False).replace("'", "''")
        stmts.append(f"UPDATE listings SET llm_extras='{j}' WHERE olx_id='{olx_id}' AND llm_extras IS NULL;")
    sql = "BEGIN;" + "".join(stmts) + "COMMIT;"
    ssh_cmd(f'sqlite3 {DB} "{sql}"', timeout=60)


def main():
    client = httpx.Client(base_url=OLLAMA_URL, timeout=httpx.Timeout(90, connect=10))
    try:
        r = client.get("/api/tags")
        assert r.status_code == 200
    except Exception:
        print("Ollama not running locally!", flush=True)
        sys.exit(1)

    total = 0
    failed = 0

    while True:
        batch = fetch_batch()
        if not batch:
            print(f"Done! Enriched {total} ({failed} failed).", flush=True)
            break

        print(f"Batch {len(batch)}...", end=" ", flush=True)
        results = []

        def _do(item):
            olx_id, desc = item
            c = httpx.Client(base_url=OLLAMA_URL, timeout=httpx.Timeout(90, connect=10))
            result = enrich_one(desc, c)
            c.close()
            return olx_id, result

        with ThreadPoolExecutor(max_workers=4) as pool:
            for olx_id, result in pool.map(_do, batch):
                if result:
                    results.append((olx_id, result))
                    total += 1
                else:
                    results.append((olx_id, {}))
                    failed += 1

        push_batch(results)
        print(f"done. Total: {total} enriched, {failed} failed.", flush=True)

    client.close()


if __name__ == "__main__":
    main()
