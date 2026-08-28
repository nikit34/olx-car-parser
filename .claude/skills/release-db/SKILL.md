---
name: release-db
description: Where the listings database lives. The authoritative copy is on the remote scrape host (anastasia@192.168.1.77) and is NOT published anywhere else — the GitHub Release `latest-data` carries only derived artefacts (models, metrics, dashboard parquets). Engine selection goes through `OLX_DB_URL`; the host runs PostgreSQL, everything else falls back to a local SQLite file. For any training, evaluation, dashboards, backtests, date-range queries, or one-off analysis, point `OLX_DB_URL` at the host or query it over SSH. Never assume `data/olx_cars.db` exists locally; never train/eval against a stale local snapshot.
---

# release-db

This repo runs a 24/7 scraper on a remote Mac. The database it writes is the only authoritative state. There is **no local DB on this machine** — keep it that way.

## Engine selection

Every entry point resolves its engine through `src.storage.database.resolve_db_url`:

1. an explicit non-default path (`--db /tmp/copy.db`) → that SQLite file;
2. else `OLX_DB_URL` → whatever it names (the host's PostgreSQL);
3. else `data/olx_cars.db` → the legacy SQLite file.

So a script keeps working unchanged on either engine, and passing an explicit
path never silently reads production.

## Where the DB lives

| Location | Path | Role |
|---|---|---|
| Scrape host | PostgreSQL on `anastasia@192.168.1.77`, database `olx_cars` | Live, updated by the cron scrape on every run |
| GitHub Release | tag `latest-data` | Derived artefacts only — models, metrics, dashboard parquets. **No DB since 2026-08-28** (370 MB republished 6x/day that nothing in production read) |
| Local (this Mac) | — | **Intentionally absent.** Do not commit, do not keep around. `data/*.db` is in `.gitignore` |

The release carries `price_model.joblib`, `price_metrics.json`, `price_importance.json`, `price_backtest.json`, `damage_classifier_v2.pt` and the dashboard parquets — pull those from the release when needed.

## Rules

1. **Never train, evaluate, backtest, or run date-range queries against a local DB.** The local DB, if one exists, is by definition stale and a partial mirror at best. Query the host directly, or copy a fresh snapshot down.
2. **Don't recreate `data/olx_cars.db` casually.** Running the scraper locally will create one. If you need to debug a single function, prefer copying a fresh host snapshot into a tmp path and pointing the script at it via `--db`.
3. **For "what's the latest …" / "show listings from <date>" questions**, go to the host — it is the only copy.

## Reading the host's database

Point the tooling at it — no copy, no staleness:

```bash
export OLX_DB_URL="postgresql+psycopg://olx@192.168.1.77:5432/olx_cars"
.venv/bin/python -m src.cli stats
```

For ad-hoc SQL, `psql "$OLX_DB_URL"`. When a script genuinely needs a private
snapshot to chew on, dump instead of copying a live file:

```bash
mkdir -p /tmp/olx-snapshot
pg_dump -Fc "postgresql://olx@192.168.1.77:5432/olx_cars" -f /tmp/olx-snapshot/olx.dump
```

Delete the dump when you're done. Do **not** restore it into `data/`.

## Querying the live host (for the freshest state)

```bash
sshpass -p 1234 ssh anastasia@192.168.1.77 \
  "psql -d olx_cars -c 'SELECT MAX(scraped_at) FROM price_snapshots;'"
```

> If the SSH times out, the host's IP has drifted (DHCP rotates `anastasiasair2` between .74 and .77). **ARP-scan before declaring it down** — see `remote-hosts` skill, "When a host doesn't ping / SSH-connect". Quick: `arp -a | grep anastasiasair2`.

Sample listings with `psql` — over SSH, or directly with `OLX_DB_URL` set — instead of pulling data down.

## Release publishing

Owned by `.github/workflows/scrape.yml`, step "Upload model + witnesses to GitHub Releases" — models, metrics, dashboard parquets and hot-deals JSONs, each gated on its build step succeeding. The DB is deliberately not among them. Don't publish manually; let the workflow do it.

## If you find a `data/olx_cars.db` on this machine

It shouldn't be there. Delete it (`rm data/olx_cars.db data/olx_cars.db-wal data/olx_cars.db-shm 2>/dev/null`). It's gitignored, so nothing is lost. If a script needs data, set `OLX_DB_URL` or take a dump per above.

## Migrating a SQLite file into PostgreSQL

`scripts/migrate_sqlite_to_postgres.py` copies through SQLAlchemy's typed
metadata, so SQLite's stringly-typed timestamps and 0/1 booleans land as real
`timestamp` / `boolean` values, then fast-forwards the identity sequences:

```bash
.venv/bin/python -m scripts.migrate_sqlite_to_postgres \
  --sqlite data/olx_cars.db \
  --target "postgresql+psycopg://olx@localhost/olx_cars" --truncate
```

It prints per-table source→target counts and exits non-zero on any mismatch.
Stop the scraper first — rows written mid-copy are not picked up.

## Host cutover (SQLite → PostgreSQL), one time

Nothing below has run yet on `.77`; until `OLX_DB_URL` is set there the host
keeps using the SQLite file and every step of the pipeline behaves as before.

```bash
# on the host
brew install postgresql@17
brew services start postgresql@17
createuser -s olx 2>/dev/null; createdb -O olx olx_cars

# stop the scrape cron first (a run mid-copy loses its rows)
cd ~/olx-car-parser && git pull && .venv/bin/pip install -q -e .
.venv/bin/python -m scripts.migrate_sqlite_to_postgres \
  --sqlite data/olx_cars.db \
  --target "postgresql+psycopg://olx@localhost/olx_cars" --truncate
```

The script refuses to finish quietly: it prints source→target counts per table
and exits 1 on any mismatch. When it matches:

1. Add repository secret `OLX_DB_URL` = `postgresql+psycopg://olx@localhost/olx_cars`
   — the workflows already read it (`scrape.yml`, `retrain-model.yml`,
   `sensitivity.yml`), and the WAL-checkpoint steps switch themselves off once
   it is non-empty.
2. Export the same value in the host's Streamlit / manual-ops shell.
3. Run one scrape and check it wrote: `psql -d olx_cars -c "SELECT MAX(last_scraped_at), COUNT(*) FROM listings;"`.
4. Keep `data/olx_cars.db` untouched for a week as the rollback — clearing
   `OLX_DB_URL` puts everything back on it.

Allow remote reads (dev machine, `psql "$OLX_DB_URL"`) by adding the LAN to
`pg_hba.conf` + `listen_addresses` only if you want them; the pipeline itself
is local to the host.
