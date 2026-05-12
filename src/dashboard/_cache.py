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
