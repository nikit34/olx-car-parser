"""Copy order for the SQLite → PostgreSQL migration.

SQLite does not enforce foreign keys unless asked, so any copy order works
there. PostgreSQL rejects a child row whose parent has not arrived yet —
the first real-data run died on ``listings`` referencing a ``sellers`` row
that was still two tables away.
"""
from __future__ import annotations

from src.models.listing import Base
from scripts.migrate_sqlite_to_postgres import _ordered_tables


def _order() -> list[str]:
    return [t.name for t in _ordered_tables(set(Base.metadata.tables))]


def test_every_table_is_copied():
    assert set(_order()) == set(Base.metadata.tables)


def test_sellers_precede_listings():
    order = _order()
    assert order.index("sellers") < order.index("listings")


def test_listings_precede_price_snapshots():
    order = _order()
    assert order.index("listings") < order.index("price_snapshots")


def test_subset_keeps_relative_order():
    order = [t.name for t in _ordered_tables({"price_snapshots", "listings"})]
    assert order == ["listings", "price_snapshots"]
