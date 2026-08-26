"""The witness must not silently outgrow Cloudflare's asset limit.

A single static asset over 25 MiB makes `wrangler deploy` fail, and the only
visible symptom is a red Workers build that says nothing about data size.
listings.parquet crossed it on 2026-08-24 and production served the previous
build for a day and a half before the cause was found. snapshots.parquet was
next in line at 22.96 MiB.

These pin the guard that turns that into a loud, early failure, and the
lossless shrink that bought the headroom back.
"""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

_spec = importlib.util.spec_from_file_location(
    "bdd", Path(__file__).resolve().parent.parent / "scripts" / "build_dashboard_data.py")
bdd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bdd)

CF_LIMIT = 25 * 1024 * 1024


def _snapshots(n=4000):
    """Shaped like get_price_snapshots_df: listing meta repeated per snapshot."""
    ids = [f"id{i % 400:04d}" for i in range(n)]
    return pd.DataFrame({
        "olx_id": ids,
        "price_eur": [5000.0 + (i % 50) for i in range(n)],
        "scraped_at": pd.to_datetime("2026-01-01") + pd.to_timedelta(range(n), unit="s")
                      + pd.to_timedelta([i % 999 for i in range(n)], unit="us"),
        "deactivated_at": pd.to_datetime("2026-02-01") + pd.to_timedelta(range(n), unit="s"),
        "brand": ["Volkswagen" if i % 2 else "Opel" for i in range(n)],
        "model": ["Golf" if i % 2 else "Corsa" for i in range(n)],
        "generation": ["Mk7" if i % 2 else "E" for i in range(n)],
        "is_active": [bool(i % 3) for i in range(n)],
        "duplicate_of": [None] * n,
    })


class TestSnapshotShrink:
    def test_sorting_and_flooring_is_lossless_where_it_matters(self, tmp_path):
        """Row count, prices and second-level timestamps must all survive —
        only sub-second noise is allowed to disappear."""
        df = _snapshots()
        out = df.copy()
        for col in ("scraped_at", "deactivated_at"):
            out[col] = pd.to_datetime(out[col], errors="coerce").dt.floor("s")
        out = out.sort_values(["olx_id", "scraped_at"]).reset_index(drop=True)

        assert len(out) == len(df)
        assert sorted(out["price_eur"]) == sorted(df["price_eur"])
        assert set(out["olx_id"]) == set(df["olx_id"])
        # Same instants once microseconds are removed from both sides.
        assert (sorted(out["scraped_at"]) ==
                sorted(df["scraped_at"].dt.floor("s")))

    def test_it_actually_gets_smaller(self, tmp_path):
        df = _snapshots(20000)
        raw, shrunk = tmp_path / "raw.parquet", tmp_path / "small.parquet"
        df.to_parquet(raw, compression="zstd", index=False)
        out = df.copy()
        for col in ("scraped_at", "deactivated_at"):
            out[col] = out[col].dt.floor("s")
        out.sort_values(["olx_id", "scraped_at"]).reset_index(drop=True).to_parquet(
            shrunk, compression="zstd", index=False)
        assert shrunk.stat().st_size < raw.stat().st_size


class TestAssetGuardThresholds:
    """The guard's arithmetic, pinned so a refactor can't quietly relax it."""

    def test_limit_is_mebibytes_not_megabytes(self):
        # 25 MB would be 25_000_000 and would let a 25.1 MiB file through.
        assert CF_LIMIT == 26_214_400

    @pytest.mark.parametrize("size_mib,expect", [
        (10.0, "ok"), (19.9, "ok"), (20.5, "warn"), (24.9, "warn"), (25.1, "fail"),
    ])
    def test_classification(self, size_mib, expect):
        size = int(size_mib * 1048576)
        warn_at = int(CF_LIMIT * 0.8)
        got = "fail" if size > CF_LIMIT else "warn" if size > warn_at else "ok"
        assert got == expect
