"""Shared @st.cache_data wrappers used by every dashboard page.

Before this module each page defined its own ``_load(_sig)`` /
``_load_snapshots(_sig, n)`` decorated with ``@st.cache_data`` —
which means every page got its OWN cache entry. Navigating between
the three pages re-ran ``load_all()`` (full GBM inference) up to
three times per data refresh, which on Streamlit Cloud's 1 vCPU
free tier blew past the websocket keepalive interval (~30 s) and
manifested as ``ConnectionClosedError 1011 keepalive ping timeout``.

Putting the wrappers here means all three pages share one cache
entry per signature.
"""
from __future__ import annotations

import streamlit as st
import pandas as pd

from data_loader import (
    load_all as _load_all,
    _ensure_release_assets,
    DB_PATH,
)
from src.storage.repository import get_price_snapshots_df
from src.storage.database import init_db, get_session


def release_signature() -> tuple[float, int]:
    """Cheap key that changes whenever the local DB file is replaced.

    ``_ensure_release_assets`` is marker-gated (see data_loader) so
    calling it on every rerun is fast — it only hits GitHub when the
    TTL inside has elapsed.
    """
    _ensure_release_assets()
    if not DB_PATH.exists():
        return (0.0, 0)
    s = DB_PATH.stat()
    return (s.st_mtime, s.st_size)


@st.cache_data(ttl=1800, show_spinner="Loading market data...")
def load_all_cached(_sig: tuple[float, int]):
    return _load_all()


@st.cache_data(ttl=1800)
def load_snapshots_cached(_sig: tuple[float, int], since_days: int):
    init_db()
    s = get_session()
    try:
        return get_price_snapshots_df(s, since_days=since_days)
    finally:
        s.close()
