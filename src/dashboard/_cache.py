"""Shared @st.cache_data wrappers used by every dashboard page.

Before this module each page defined its own ``_load(_sig)`` /
``_load_snapshots(_sig, n)`` decorated with ``@st.cache_data`` — which
means every page got its OWN cache entry. Navigating between the three
pages re-ran ``load_all()`` up to three times per data refresh, which
on the old Streamlit Cloud free tier blew past the websocket keepalive
interval and manifested as ``ConnectionClosedError 1011``. Putting the
wrappers here means all three pages share one cache entry per signature.

The signature was historically the local DB file's (mtime, size). With
the stlite + CF Pages migration the dashboard no longer reads SQLite —
``data_loader.load_all`` fetches precomputed parquets from the
``latest-data`` release. The signature now reflects the build manifest
(timestamp + total bytes) so the cache invalidates when CI publishes a
fresh build, without depending on a local DB file that doesn't exist
in the browser.
"""
from __future__ import annotations

import streamlit as st

import pandas as pd

from data_loader import (
    load_all as _load_all,
    load_snapshots as _load_snapshots,
    dashboard_data_signature,
)


def release_signature() -> tuple[str, int]:
    """Cache key that changes whenever CI publishes a fresh dashboard build."""
    return dashboard_data_signature()


@st.cache_data(ttl=1800, show_spinner="Loading market data...")
def load_all_cached(_sig: tuple[str, int]):
    return _load_all()


@st.cache_data(ttl=1800)
def load_snapshots_cached(_sig: tuple[str, int], since_days: int):
    return _load_snapshots(since_days)


def render_context_badge(
    listings_df: pd.DataFrame | None,
    signals_df: pd.DataFrame | None,
) -> None:
    """Render a one-line data context caption at the top of a page.

    ``19,308 listings · 10,058 active · 1,492 deals · built 2026-05-12 11:00 UTC``

    The same snapshot drives every page, so this caption lets a viewer
    see the corpus scope on any page without flipping back to the
    Recommendations sidebar (which used to be the only place the count
    was shown).
    """
    total = len(listings_df) if isinstance(listings_df, pd.DataFrame) else 0
    if (
        isinstance(listings_df, pd.DataFrame)
        and not listings_df.empty
        and "is_active" in listings_df.columns
    ):
        active = int(listings_df["is_active"].fillna(False).astype(bool).sum())
    else:
        active = 0
    deals = len(signals_df) if isinstance(signals_df, pd.DataFrame) else 0

    built_at, _ = dashboard_data_signature()
    if built_at:
        built_short = built_at.replace("T", " ").replace("Z", " UTC")
    else:
        built_short = "—"

    st.caption(
        f"{total:,} listings · {active:,} active · {deals:,} deals · built {built_short}"
    )
