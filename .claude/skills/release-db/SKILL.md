---
name: release-db
description: Where olx_cars.db lives. The authoritative DB is on the remote scrape host (anastasia@192.168.1.77) and mirrored to the GitHub Release `latest-data`. There is no local copy — for any training, evaluation, dashboards, backtests, date-range queries, or one-off analysis, pull from the release (or query the host over SSH). Never assume `data/olx_cars.db` exists locally; never train/eval against a stale local snapshot.
---

# release-db

This repo runs a 24/7 scraper on a remote Mac. The SQLite DB it writes is the only authoritative state. There is **no local DB on this machine** — keep it that way.

## Where the DB lives

| Location | Path | Role |
|---|---|---|
| Scrape host | `anastasia@192.168.1.77:~/olx-car-parser/data/olx_cars.db` | Live, updated by the cron scrape on every run |
| GitHub Release | tag `latest-data`, asset `olx_cars.db` | Snapshot uploaded after every successful scrape (`scripts/scrape.yml` step "Upload artifacts") |
| Local (this Mac) | — | **Intentionally absent.** Do not commit, do not keep around. `data/*.db` is in `.gitignore` |

The release also carries `price_model.joblib`, `price_metrics.json`, `price_importance.json`, `price_backtest.json`, and `damage_classifier_v2.pt` — same rule, pull from release when needed.

## Rules

1. **Never train, evaluate, backtest, or run date-range queries against a local DB.** The local DB, if one exists, is by definition stale and a partial mirror at best. Use the release snapshot or query the host directly.
2. **Don't recreate `data/olx_cars.db` casually.** Running the scraper locally will create one. If you need to debug a single function, prefer pulling a fresh release copy into a tmp path and pointing the script at it via `--db`.
3. **For "what's the latest …" / "show listings from <date>" questions**, go to the host (it has the most recent rows the release may not yet include).

## Pulling the release DB (one-shot, when you actually need it)

```bash
# Into a tmp path so you remember it's disposable:
mkdir -p /tmp/olx-release && cd /tmp/olx-release
gh release download latest-data --pattern olx_cars.db
# inspect:
sqlite3 olx_cars.db "SELECT MAX(scraped_at), COUNT(*) FROM listings;"
```

Delete `/tmp/olx-release/olx_cars.db` when you're done. Do **not** copy it to `data/`.

## Querying the live host (for the freshest state)

```bash
sshpass -p 1234 ssh anastasia@192.168.1.77 \
  "sqlite3 ~/olx-car-parser/data/olx_cars.db 'SELECT MAX(scraped_at) FROM listings;'"
```

`scripts/eval_model.py` (`REMOTE_HOST`, `REMOTE_DB`) is the canonical example of how to sample listings from the host without ever pulling the file down.

## Release publishing

Owned by `.github/workflows/scrape.yml` — the "Upload artifacts" step runs `gh release upload latest-data data/olx_cars.db --clobber` after every successful run. Don't publish manually; let the workflow do it.

## If you find a `data/olx_cars.db` on this machine

It shouldn't be there. Delete it (`rm data/olx_cars.db data/olx_cars.db-wal data/olx_cars.db-shm 2>/dev/null`). It's gitignored, so nothing is lost. If a script needs one, pull from release into `/tmp` per above.
