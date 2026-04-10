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
sub_model(str), trim_level(str), \
desc_mentions_num_owners(int), desc_mentions_accident(bool), \
desc_mentions_repair(bool), \
mileage_in_description_km(int), desc_mentions_customs_cleared(bool), \
right_hand_drive(bool), \
mechanical_condition("excellent"/"good"/"fair"/"poor"), \
urgency("high"/"medium"/"low"), warranty(bool), tuning_or_mods(list), \
taxi_fleet_rental(bool), \
first_owner_selling(bool).
Rules: sub_model=engine/body variant from title: "320d","1.6 TDI","2.0 HDi","1.4 TSI","A 200","C 220d". \
trim_level=equipment line from title/description: "AMG Line","M Sport","S-Line","FR","GTI","GT Line","Luxury","Titanium","N-Connecta","Tekna","Avantgarde","Elegance","Comfort","Executive". null if basic/unknown. \
mileage_in_description_km=exact km as integer ("4300 km"→4300, "150 mil km"→150000, \
"89.500km"→89500, "4.300km"→4300). "mil"=thousand ONLY when written as a separate word. \
desc_mentions_repair=true if ANY repair/damage/breakdown mentioned ("avariado","imobilizado" included). \
desc_mentions_accident=true if collision/accident mentioned ("sinistro","acidente","batido"). \
desc_mentions_customs_cleared=look for "desalfandegado","legalizado","por legalizar","documentação em dia","documentos em ordem". \
right_hand_drive=true if right-hand drive/UK/Japan import/"mão inglesa"/"volante à direita"/"condução à direita"/"matrícula inglesa". \
urgency=high if "urgente","preciso vender rápido","emigração","preço para despachar"; medium if "aceito propostas","negociável","oportunidade"; low otherwise. \
warranty=true if "garantia" mentioned (not "sem garantia"). \
tuning_or_mods=list of aftermarket modifications: "reprogramação","stage 1/2","remap","chip tuning","suspensão rebaixada","coilovers","escape desportivo","downpipe","wrap","bodykit". \
taxi_fleet_rental=true if "ex-táxi","TVDE","Uber","Bolt","rent-a-car","frota","carro de empresa". \
first_owner_selling=true if "1 dono desde novo","único dono","comprado novo por mim","vendo o meu". \
If "para peças","vender as peças","venda de peças","para desmanchar","só peças","avariado","imobilizado": \
mechanical_condition="poor", desc_mentions_accident=true, desc_mentions_repair=true.

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
    """Push all results in one SSH call via stdin."""
    if not results:
        return
    stmts = ["PRAGMA busy_timeout = 30000;", "BEGIN;"]
    for olx_id, data in results:
        j = json.dumps(data, ensure_ascii=False).replace("'", "''")
        stmts.append(f"UPDATE listings SET llm_extras='{j}' WHERE olx_id='{olx_id}' AND llm_extras IS NULL;")
    stmts.append("COMMIT;")
    sql = "\n".join(stmts)
    import time
    for attempt in range(3):
        try:
            r = subprocess.run(SSH + [f"sqlite3 {DB}"], input=sql, capture_output=True, text=True, timeout=120)
            if r.returncode == 0:
                return
            print(f"  Push retry {attempt+1}: {r.stderr[:80]}", flush=True)
        except subprocess.TimeoutExpired:
            print(f"  Push timeout (attempt {attempt+1})", flush=True)
        time.sleep(5)


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
