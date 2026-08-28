---
name: remote-hosts
description: Map of the LAN machines this project depends on. One host that matters — the scrape host (anastasia@192.168.1.77, macOS, owns the DB and runs the GitHub Actions self-hosted runner + Streamlit dashboard) — plus the Windows box (permi@192.168.1.69) which no longer runs anything for this project. Use when you need to SSH in, debug LAN routing, or understand which host owns what. There is NO local inference any more: LLM work is a cloud Gemini → OpenRouter cascade. The dev Mac only orchestrates.
---

# remote-hosts

The project runs on two LAN machines, not on this dev Mac. This skill is the map.

> **Status 2026-08-23**: Ollama is **uninstalled from both machines** — not
> stopped, removed. Scrape host: launchd agents unloaded and deleted,
> `/Applications/Ollama.app`, `/usr/local/bin/ollama` and `~/.ollama` (5.3 GB)
> gone. Windows box: the NSSM service `Ollama` removed, the Inno uninstaller
> run, `%LOCALAPPDATA%\Programs\Ollama` (6.3 GB) and `%USERPROFILE%\.ollama`
> (3.1 GB) gone, the TCP 11434 firewall rule deleted, `OLLAMA_HOST` unset.
> Port 11434 answers on neither host and nothing on the boxes will bring it
> back on reboot.
>
> LLM enrichment is a **cloud provider cascade** — Gemini first, OpenRouter as
> backstop (`llm.providers` in `config/settings.yaml`; keys in the
> `GEMINI_API_KEY` / `OPENROUTER_API_KEY` repo secrets) — and it only ever
> sees the top-K deals the GBM ranks as undervalued. If you find yourself
> reaching for a local model here, the answer is no: an 8 GB Air is the
> scraper, not an inference box.

## Hosts

### Scrape host — `anastasia@192.168.1.77`

- **OS / hw**: macOS, 8 GB RAM (M1 Air).
- **Role**:
  - Cron scraper (every run writes to the local PostgreSQL database `olx_cars`).
  - GitHub Actions self-hosted runner — executes `.github/workflows/scrape.yml`.
  - Streamlit dashboard.
- **No local model server.** Ollama was uninstalled 2026-08-23 (app, symlink,
  launchd agents and the 5.3 GB model store). `curl localhost:11434` fails by
  design — that is not a bug to fix.
- **Owns**: the `olx_cars` PostgreSQL database (the only authoritative copy — see the `release-db` skill).
- **SSH**:
  ```bash
  sshpass -p 1234 ssh -o StrictHostKeyChecking=no anastasia@192.168.1.77
  ```
  Key auth works too, so `ssh anastasia@192.168.1.77` alone is usually enough.
- **Note**: the host's DHCP address drifts between .74 and .77 — resolve it by ARP (below) instead of hard-coding, and don't put a LAN self-reference in config.

### Windows box — `permi@192.168.1.69`

- **OS / hw**: Windows 11, 16 GB RAM, MX230 GPU (4 GB VRAM).
- **Role for this project: none.** It used to be the second Ollama backend;
  the service, the binaries, the 3.1 GB model store and the TCP 11434 firewall
  rule were all removed on 2026-08-23. Nothing here is wired into the scrape
  pipeline any more.
- **SSH**: `ssh permi@192.168.1.69` — key auth works. The default shell is
  `cmd.exe`, and nested quoting mangles anything interesting, so drive it with
  `powershell -NoProfile -EncodedCommand <base64 UTF-16LE>`:
  ```bash
  B64=$(iconv -f UTF-8 -t UTF-16LE < script.ps1 | base64)
  ssh permi@192.168.1.69 "powershell -NoProfile -EncodedCommand $B64"
  ```
- **⚠️ Not ours — don't touch**: this box hosts an **unrelated** project
  (`first-message-builder` / `vacancy_service`) under NSSM services
  `FirstMessageBuilder` / `VacancyService` / `VacancyInbox` / `FirstMessageBot`,
  served by `waitress` on `127.0.0.1:8000`, exposed via a `cloudflared` tunnel
  (`firstmessage`) and reachable over `tailscaled`. Each shows as a **pair** —
  a `venv\Scripts\python.exe` launcher stub (MEM ~0) → `C:\Python311\python.exe`
  worker (holds the port). That is one deployment, **not** duplicates; killing
  the stub breaks the live worker. When removing anything on this machine,
  filter by exact service name and verify these four are still Running
  afterwards.
- **Worth borrowing, not touching**: that project is where the working
  `GEMINI_API_KEY` / `OPENROUTER_API_KEY` come from, and its
  `builder/message_generator.py` is the reference implementation of the
  provider cascade this project copied.

## The dev Mac

This machine (M1, 32 GB) **orchestrates and never runs the pipeline**. Scrapes,
trains and enrichment all happen on the scrape host via the self-hosted runner.
Running them here writes to a database that isn't the authoritative one.

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
- `dell.home` → the Windows box (.69)
- `mac.home` → this dev Mac (don't SSH to self)

SSH to the resolved IP and update this skill's "Hosts" section if the address has drifted.

## Quick health check (both hosts)

```bash
# Scrape host: SSH liveness + DB freshness
# NB: there is NO scraped_at column — freshness = MAX(last_seen_at) or the .db mtime.
sshpass -p 1234 ssh anastasia@192.168.1.77 \
  "psql -d olx_cars -c 'SELECT MAX(last_seen_at) FROM listings;'"

# LLM providers are cloud now — check them from the host, not the LAN:
ssh anastasia@192.168.1.77 \
  "cd ~/olx-car-parser && GEMINI_API_KEY=... .venv/bin/python -m src.cli enrich-cloud --dry-run"
```

## Related

- DB location & release flow → `release-db` skill.
- Provider cascade config → `config/settings.yaml` (`llm.providers`, `gemini:`, `openrouter:`).
- Cascade implementation → `src/parser/cloud_enrichment.py`; the gate that decides
  which listings are worth a call → `src/analytics/value_gate.py`.
