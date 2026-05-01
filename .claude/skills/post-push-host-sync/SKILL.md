---
name: post-push-host-sync
description: Post-push runbook — after `git push` to origin/master, sync the persistent clone on the scrape host (~/olx-car-parser), restart any services whose config changed, and verify HEAD matches what was just pushed. Always run this after a successful push; the persistent clone does NOT auto-pull (only GitHub Actions runs use `actions/checkout` in `_work/`, which doesn't update `~/olx-car-parser`). Skipping this step is how Streamlit / manual-ops drift 24+ commits behind origin.
---

# post-push-host-sync

After every successful `git push origin master`, run this. The scrape host's persistent clone at `~/olx-car-parser` is what backs the Streamlit dashboard, manual `enrich_local.py` runs, ad-hoc SSH operations, and any non-CI execution. It does not auto-pull, so it silently drifts behind master.

The Windows LLM box has no project clone — only an Ollama service. It's only relevant if config touching `llm.ollama_urls`/`llm.ollama_weights` changed, in which case the runner's next checkout already picks it up; nothing to do on .69 itself.

## Steps

### 1. Resolve the scrape host's current IP

DHCP rotates `anastasiasair2` between .74 and .77 (see `remote-hosts` skill, "When a host doesn't ping / SSH-connect"). Don't assume — check:

```bash
arp -a | grep anastasiasair2
# Falls back to a sweep if cache is cold:
# for i in $(seq 1 254); do (ping -c 1 -W 200 192.168.1.$i &>/dev/null && echo "192.168.1.$i alive") & done; wait
```

Use the resolved IP for the rest of the steps. (Examples below assume `.74`.)

### 2. Pull on the host

```bash
sshpass -p 1234 ssh -o ConnectTimeout=5 anastasia@192.168.1.74 \
  "cd ~/olx-car-parser && git fetch origin --quiet && git pull --ff-only origin master"
```

The host's local branch is named `main` but tracks `origin/master` — the explicit `origin master` argument avoids ambiguity. **Use `--ff-only`** — never let a non-FF pull silently merge or rebase someone's WIP on the host.

If `--ff-only` rejects (non-FF), STOP. Don't `--force`, don't `reset --hard`. Inspect with `git log --oneline HEAD..origin/master` and `git status` on the host first; the host may have local commits worth preserving.

### 3. Verify HEAD matches what was just pushed

```bash
LOCAL=$(git rev-parse origin/master)   # on dev Mac (after your push)
REMOTE=$(sshpass -p 1234 ssh anastasia@192.168.1.74 "cd ~/olx-car-parser && git rev-parse HEAD")
[ "$LOCAL" = "$REMOTE" ] && echo "✓ in sync" || echo "✗ MISMATCH: local=$LOCAL remote=$REMOTE"
```

Both should equal the SHA `git push` printed (e.g. `f77e648..1cc1eae`). If they don't, the pull silently no-op'd or pulled the wrong branch — investigate before declaring done.

### 4. Conditionally restart services

Only restart what actually depends on the changed code:

| Pushed change touched… | Action |
|---|---|
| `config/settings.yaml` (Ollama URLs/weights/model name) | `launchctl kickstart -k gui/$(id -u anastasia)/com.olx-car-parser.ollama` on the host (bounces Ollama LaunchAgent so it re-reads env). |
| Modelfile / `scripts/setup-ollama.sh` / new model required | `ssh anastasia@<ip> '~/olx-car-parser/scripts/setup-ollama.sh'` to (re)pull. |
| `src/dashboard/**` or anything Streamlit serves | `pkill -f 'streamlit run'` on the host (it's started manually; the user re-launches as needed — note in your reply so they know). |
| `.github/workflows/scrape.yml` or anything in `src/parser/`, `src/storage/`, `scripts/` (non-dashboard) | Nothing — next scheduled or dispatched workflow run picks it up via `actions/checkout`. |
| Skills / docs / `.claude/**` only | Nothing. |

When in doubt, **don't** restart — restating wrong is worse than skipping. Ask the user which service if the change is ambiguous.

### 5. Report back

One line: pushed SHA, host SHA, restart action taken (if any). Example:
```
✓ host synced to 1cc1eae · no service restart needed (skills-only change)
```

## When to skip this skill

- Push only touched `.github/`, `README.md`, or other CI-only files where the runner's fresh checkout is the only consumer.
- Push was to a non-master branch (feature branches don't go to the host clone).
- Push was rejected / failed — nothing to sync.

## Related

- `remote-hosts` — host map, ARP-scan recipe
- `release-db` — DB lives on this same host; don't touch `data/` during sync
