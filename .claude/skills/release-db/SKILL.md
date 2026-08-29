---
name: release-db
description: Where the listings database lives. The authoritative copy is on the remote scrape host (anastasia@192.168.1.77) and is NOT published anywhere else — the GitHub Release `latest-data` carries only derived artefacts (models, metrics, dashboard parquets). Engine selection goes through `OLX_DB_URL` and has no fallback — unset raises `DatabaseNotConfigured`. For any training, evaluation, dashboards, backtests, date-range queries, or one-off analysis, point `OLX_DB_URL` at the host or query it over SSH. There is no local database file any more; never train/eval against a stale snapshot.
---

# release-db

This repo runs a 24/7 scraper on a remote Mac. The database it writes is the only authoritative state. There is **no local DB on this machine** — keep it that way.

## Engine selection

Every entry point resolves its engine through `src.storage.database.resolve_db_url`:

1. an explicit URL argument (`--db postgresql+psycopg://…`) → that database;
2. else `OLX_DB_URL` → the host's PostgreSQL.

There is no third option: unset means `DatabaseNotConfigured`, not a quietly
different database. `--db` takes a URL, never a file path.

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

Nothing reads it any more — no code path opens a file. Delete it (`rm data/olx_cars.db data/olx_cars.db-wal data/olx_cars.db-shm 2>/dev/null`); it's gitignored, so nothing is lost. If a script needs data, set `OLX_DB_URL` or take a dump per above.

## Running the tests

They need PostgreSQL too, for the same reason production does — SQLite enforced neither foreign keys nor a parameter ceiling, and both cost a scrape on 2026-08-28. `TEST_DB_URL` points at it (CI runs a `postgres:17` service); with it unset the suite falls back to `postgresql+psycopg://postgres:postgres@localhost:5432/olx_test` and skips if nothing answers. Set explicitly, an unreachable database is an error, never a skip.

## History

The database was SQLite until 2026-08-28, when it moved to PostgreSQL 17 on
the host (`olx_cars`, role `olx`). The one-way migration tool lived at
`scripts/migrate_sqlite_to_postgres.py` and was deleted afterwards — keeping a
script whose `--truncate` targets production is a footgun. `git log` has it if
a rollback ever needs it, along with the retired `data/olx_cars.db`.
