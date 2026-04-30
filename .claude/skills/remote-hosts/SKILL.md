---
name: remote-hosts
description: Map of the LAN machines this project depends on. Two hosts — the scrape host (anastasia@192.168.1.77, macOS, owns the DB and runs the GitHub Actions self-hosted runner + Streamlit dashboard) and the Windows LLM partner (permi@192.168.1.69, Win11 16 GB, runs an Ollama backend on :11434). Use when you need to SSH in, probe Ollama health, debug LAN routing, or understand which host owns what. The dev Mac is *not* part of this pool — it only orchestrates.
---

# remote-hosts

The project runs on two LAN machines, not on this dev Mac. This skill is the map.

## Hosts

### Scrape host — `anastasia@192.168.1.77`

- **OS / hw**: macOS, 8 GB RAM (M1 Air).
- **Role**:
  - Cron scraper (every run writes to `~/olx-car-parser/data/olx_cars.db`).
  - GitHub Actions self-hosted runner — executes `.github/workflows/scrape.yml`.
  - Streamlit dashboard.
  - Local Ollama on `http://localhost:11434` (`NUM_PARALLEL=2`, ctx 1536 — capped by Metal at 8 GB; higher OOMs).
- **Owns**: `olx_cars.db` (the only authoritative copy — see the `release-db` skill).
- **SSH**:
  ```bash
  sshpass -p 1234 ssh -o StrictHostKeyChecking=no anastasia@192.168.1.77
  ```
  Same one-liner pattern is used by `scripts/eval_model.py` (`REMOTE_HOST`, `SSH`).
- **Note**: Ollama on this host is referenced as `http://localhost:11434` in `config/settings.yaml`, *not* as `192.168.1.77` — the runner's DHCP-assigned IP drifts and self-references over LAN started failing with `EHOSTUNREACH`. Don't "fix" it back to the IP.

### Windows LLM partner — `permi@192.168.1.69`

- **OS / hw**: Windows 11, 16 GB RAM, MX230 GPU (4 GB VRAM).
- **Role**: second Ollama backend in the inference pool. Model `qwen3:4b-instruct`. Used by `_pick_ollama_url()` in `src/parser/llm_enrichment.py` via sticky-per-thread round-robin against `config/settings.yaml → llm.ollama_urls`.
- **Endpoint**: `http://192.168.1.69:11434` (HTTP, not SSH — no shell access wired up).
- **Required server config**:
  - `OLLAMA_HOST=0.0.0.0` (otherwise binds loopback only, LAN can't reach it).
  - Windows Firewall inbound rule on TCP 11434.
  - **`NUM_PARALLEL=1`**. `2` forces KvSize=8192 → "failure during GPU discovery" (CUDA timeout). Verified empirically 2026-04-28: 10 consecutive runner-start failures.
- **Pool weight**: 1 (vs 2 for the scrape host's localhost) — see `llm.ollama_weights` in `config/settings.yaml`.
- **Health check** (matches the workflow's "Probe LAN partner Ollama" step):
  ```bash
  curl -sf --max-time 5 http://192.168.1.69:11434/api/tags >/dev/null \
    && echo "✓ reachable" || echo "✗ unreachable"
  ```

## What's *not* in the pool

This dev Mac (M1, 32 GB) **orchestrates but never inferences**. Don't add `localhost` of *this* machine to `ollama_urls` — that's a different "localhost" than the one the runner sees, and the LLM enrichment is supposed to run on the runner, not here.

## When a host doesn't ping / SSH-connect

DHCP on this LAN reshuffles IPs — `anastasiasair2` has bounced .74 ↔ .77 multiple times. **Never declare a host down without an ARP scan first.**

```bash
# 1. Sanity-check own LAN
ipconfig getifaddr en0   # should be 192.168.1.x

# 2. Sweep + ARP
for i in $(seq 1 254); do (ping -c 1 -W 200 192.168.1.$i &>/dev/null && echo "192.168.1.$i alive") & done; wait
arp -a | grep -v incomplete
```

Match by hostname in ARP output:
- `anastasiasair2.home` → the scrape Mac (was .77, may be .74 today)
- `dell.home` → the Windows LLM box (.69)
- `mac.home` → this dev Mac (don't SSH to self)

SSH to the resolved IP and update this skill's "Hosts" section if the address has drifted.

## Quick health check (both hosts)

```bash
# Scrape host: SSH liveness + DB freshness
sshpass -p 1234 ssh anastasia@192.168.1.77 \
  "sqlite3 ~/olx-car-parser/data/olx_cars.db 'SELECT MAX(scraped_at) FROM listings;'"

# LLM partner: HTTP liveness + model loaded
curl -sf --max-time 5 http://192.168.1.69:11434/api/tags | jq -r '.models[].name'
```

## Related

- DB location & release flow → `release-db` skill.
- Inference pool config → `config/settings.yaml` (`llm.ollama_urls`, `llm.ollama_weights`).
- Eval-from-host pattern → `scripts/eval_model.py` (`REMOTE_HOST`, `REMOTE_DB`, `SSH`).
