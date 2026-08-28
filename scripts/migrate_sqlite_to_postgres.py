"""Copy the scraper's SQLite database into PostgreSQL.

Reads through SQLAlchemy's typed metadata, so SQLite's stringly-typed
DATETIME columns and 0/1 booleans arrive as real timestamps and booleans
on the target instead of failing the insert.

    python -m scripts.migrate_sqlite_to_postgres \
        --sqlite data/olx_cars.db \
        --target postgresql+psycopg://olx@localhost/olx_cars

Idempotent per run: ``--truncate`` empties the target tables first, so a
failed migration can simply be repeated. Stop the scraper before running
it — rows written to SQLite mid-copy are not picked up.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from sqlalchemy import Integer, create_engine, func, insert, select, text  # noqa: E402

from src.models.listing import Base  # noqa: E402
import src.models.portfolio  # noqa: F401,E402
import src.models.relist  # noqa: F401,E402
import src.models.seller  # noqa: F401,E402

logger = logging.getLogger("migrate_sqlite_to_postgres")

_COPY_ORDER = (
    "listings",
    "price_snapshots",
    "market_stats",
    "unmatched_listings",
    "sellers",
    "portfolio_deals",
    "relist_events",
)


def _check_llm_extras_json(src_engine, sample: int = 5) -> list[str]:
    """Return ids whose ``llm_extras`` is not a JSON object.

    On SQLite a malformed blob makes ``json_extract`` return NULL; on
    PostgreSQL the ``::jsonb`` cast in the same query aborts it. Better to
    find out here than during the first scrape after the cutover.
    """
    bad: list[str] = []
    with src_engine.connect() as src:
        rows = src.execution_options(stream_results=True).execute(text(
            "SELECT olx_id, llm_extras FROM listings WHERE llm_extras IS NOT NULL"
        ))
        for olx_id, raw in rows:
            try:
                if not isinstance(json.loads(raw), dict):
                    raise ValueError
            except (ValueError, TypeError):
                bad.append(olx_id)
                if len(bad) >= sample:
                    break
    return bad


def _copy_table(src_engine, dst_engine, table, batch_size: int) -> int:
    copied = 0
    with src_engine.connect() as src, dst_engine.begin() as dst:
        result = src.execution_options(stream_results=True).execute(select(table))
        while True:
            rows = result.fetchmany(batch_size)
            if not rows:
                break
            dst.execute(insert(table), [dict(r._mapping) for r in rows])
            copied += len(rows)
            logger.info("  %s: %d rows", table.name, copied)
    return copied


def _reset_sequences(dst_engine, table) -> None:
    """Fast-forward the target's identity sequences past the copied ids.

    Only integer primary keys own a sequence — a text key like
    ``sellers.uuid`` has none, and asking for one is an error, not a no-op.
    """
    with dst_engine.begin() as dst:
        for col in table.primary_key.columns:
            if not isinstance(col.type, Integer):
                continue
            sequence = dst.execute(
                text("SELECT pg_get_serial_sequence(:table, :column)"),
                {"table": table.name, "column": col.name},
            ).scalar()
            if not sequence:
                continue
            dst.execute(text(
                "SELECT setval("
                "  :sequence,"
                f"  COALESCE((SELECT MAX({col.name}) FROM {table.name}), 1),"
                f"  (SELECT MAX({col.name}) FROM {table.name}) IS NOT NULL"
                ")"
            ), {"sequence": sequence})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", default=str(_REPO_ROOT / "data" / "olx_cars.db"),
                        help="Source SQLite file (default: data/olx_cars.db).")
    parser.add_argument("--target", default=os.environ.get("OLX_DB_URL", ""),
                        help="Target SQLAlchemy URL (default: $OLX_DB_URL).")
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--truncate", action="store_true",
                        help="Empty the target tables before copying.")
    parser.add_argument("--ignore-invalid-json", action="store_true",
                        help="Migrate even if some llm_extras rows aren't JSON objects.")
    parser.add_argument("--tables", default=",".join(_COPY_ORDER),
                        help="Comma-separated subset of tables to copy.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if not args.target:
        parser.error("no target: pass --target or set OLX_DB_URL")
    if args.target.startswith("sqlite"):
        parser.error("target must not be SQLite")
    source = Path(args.sqlite)
    if not source.exists():
        parser.error(f"source not found: {source}")

    src_engine = create_engine(f"sqlite:///{source}")
    dst_engine = create_engine(args.target)

    bad_json = _check_llm_extras_json(src_engine)
    if bad_json and not args.ignore_invalid_json:
        print("llm_extras is not valid JSON on these listings:", ", ".join(bad_json))
        print("PostgreSQL queries cast that column to jsonb and would fail on them.")
        print("Fix the rows, or re-run with --ignore-invalid-json to migrate anyway.")
        return 1

    logger.info("Creating schema on %s", dst_engine.url.render_as_string(hide_password=True))
    Base.metadata.create_all(dst_engine)

    wanted = [t.strip() for t in args.tables.split(",") if t.strip()]
    tables = [Base.metadata.tables[name] for name in _COPY_ORDER if name in wanted]

    if args.truncate:
        with dst_engine.begin() as dst:
            for table in reversed(tables):
                dst.execute(text(f"TRUNCATE TABLE {table.name} RESTART IDENTITY CASCADE"))
        logger.info("Truncated %d target tables", len(tables))

    totals: dict[str, tuple[int, int]] = {}
    for table in tables:
        logger.info("Copying %s…", table.name)
        copied = _copy_table(src_engine, dst_engine, table, args.batch_size)
        _reset_sequences(dst_engine, table)
        with src_engine.connect() as src, dst_engine.connect() as dst:
            src_n = src.execute(select(func.count()).select_from(table)).scalar_one()
            dst_n = dst.execute(select(func.count()).select_from(table)).scalar_one()
        totals[table.name] = (src_n, dst_n)
        logger.info("  %s: copied %d (source %d, target %d)", table.name, copied, src_n, dst_n)

    mismatched = {k: v for k, v in totals.items() if v[0] != v[1]}
    print("\nRow counts (source → target):")
    for name, (src_n, dst_n) in totals.items():
        flag = "" if src_n == dst_n else "   MISMATCH"
        print(f"  {name:<22} {src_n:>9} → {dst_n:>9}{flag}")
    if mismatched:
        print(f"\n{len(mismatched)} table(s) did not match. Re-run with --truncate.")
        return 1
    print("\nAll tables match. Point OLX_DB_URL at the target and restart the scraper.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
