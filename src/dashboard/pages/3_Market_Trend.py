"""Market trend — global view of how the segment is moving over time.

Two stacked panels (per filter):
  - median ASK (active listings) and median LAST-ASK (sold listings),
    bucketed by week. The gap = how much sellers ask above what
    transactions actually cleared at recently.
  - volume: count of new active listings vs sold deactivations per
    week. Useful for spotting "supply spike" weeks vs "demand spike"
    weeks.

Filters: brand, model, fuel group, year range. The whole page falls
back to "all listings" if no filter is set.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_dashboard_dir = _Path(__file__).resolve().parent.parent
_project_root = _dashboard_dir.parent.parent
_sys.path.insert(0, str(_dashboard_dir))
_sys.path.insert(0, str(_project_root))

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from data_loader import (
    load_all, _ensure_release_assets, DB_PATH, _force_next_check,
    get_last_release_error, _fuel_group,
)


st.set_page_config(page_title="Market Trend", layout="wide")
st.title("Market Trend")


def _release_cache_signature() -> tuple[float, int]:
    _ensure_release_assets()
    if not DB_PATH.exists():
        return (0.0, 0)
    s = DB_PATH.stat()
    return (s.st_mtime, s.st_size)


@st.cache_data(ttl=300)
def _load(_sig: tuple[float, int]):
    return load_all()


listings_df, *_rest = _load(_release_cache_signature())
if listings_df.empty:
    st.warning("No data yet.")
    err = get_last_release_error()
    if err:
        st.error(f"Release fetch failed: {err}")
    if st.button("Force refresh"):
        _force_next_check()
        st.cache_data.clear()
        st.rerun()
    st.stop()

# --- Filters ---
st.sidebar.header("Filters")

brand_options = ["(all)"] + sorted(listings_df["brand"].dropna().unique().tolist())
brand = st.sidebar.selectbox("Brand", brand_options)
model_pool = listings_df if brand == "(all)" else listings_df[listings_df["brand"] == brand]
model_options = ["(all)"] + sorted(model_pool["model"].dropna().unique().tolist())
model = st.sidebar.selectbox("Model", model_options)

if "fuel_type" in listings_df.columns:
    fuel_groups = sorted({_fuel_group(v) for v in listings_df["fuel_type"].tolist()
                          if v and _fuel_group(v) != "Unknown"})
else:
    fuel_groups = []
fuel_pick = st.sidebar.multiselect("Fuel group", fuel_groups)

year_min = int(listings_df["year"].min()) if listings_df["year"].notna().any() else 2000
year_max = int(listings_df["year"].max()) if listings_df["year"].notna().any() else 2026
year_range = st.sidebar.slider("Year", year_min, year_max, (year_min, year_max))

window_days = st.sidebar.slider("Look-back window (days)", 30, 365, 180, step=30)

# --- Apply filters ---
df = listings_df.copy()
if "duplicate_of" in df.columns:
    df = df[df["duplicate_of"].isna()]
if brand != "(all)":
    df = df[df["brand"] == brand]
if model != "(all)":
    df = df[df["model"] == model]
if fuel_pick:
    df = df[df["fuel_type"].map(_fuel_group).isin(fuel_pick)]
df = df[
    (df["year"].between(year_range[0], year_range[1])) | df["year"].isna()
]

if df.empty:
    st.info("No listings match the filters.")
    st.stop()

cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=window_days)

# Active subset: filter by first_seen_at within window so the
# "supply" curve reflects what landed recently, not historical posts
# still active from years ago.
is_active = df["is_active"].astype(bool) if "is_active" in df.columns else pd.Series(True, index=df.index)
is_sold = (~is_active) & (df.get("deactivation_reason", "").astype(str) == "sold")

active_df = df[is_active].copy()
sold_df = df[is_sold].copy()

active_df["first_seen_at"] = pd.to_datetime(active_df.get("first_seen_at"), errors="coerce", utc=True)
sold_df["first_seen_at"] = pd.to_datetime(sold_df.get("first_seen_at"), errors="coerce", utc=True)
sold_df["deactivated_at"] = pd.to_datetime(sold_df.get("deactivated_at"), errors="coerce", utc=True)

active_recent = active_df[active_df["first_seen_at"] >= cutoff]
sold_recent = sold_df[sold_df["deactivated_at"] >= cutoff]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Active in window", len(active_recent))
c2.metric("Sold in window", len(sold_recent))
c3.metric(
    "Median active ask",
    f"€{int(active_df['price_eur'].median()):,}" if not active_df.empty
    else "—",
)
c4.metric(
    "Median sold last-ask",
    f"€{int(sold_df['price_eur'].median()):,}" if not sold_df.empty
    else "—",
)

st.divider()

# --- Weekly aggregation ---
def _weekly_median(d: pd.DataFrame, time_col: str) -> pd.DataFrame:
    if d.empty or time_col not in d.columns:
        return pd.DataFrame(columns=["week", "median_price", "n"])
    work = d[[time_col, "price_eur"]].dropna()
    work = work[work["price_eur"] > 0]
    if work.empty:
        return pd.DataFrame(columns=["week", "median_price", "n"])
    work["week"] = work[time_col].dt.to_period("W").dt.start_time
    return work.groupby("week").agg(
        median_price=("price_eur", "median"),
        n=("price_eur", "size"),
    ).reset_index()


active_weekly = _weekly_median(active_recent, "first_seen_at")
sold_weekly = _weekly_median(sold_recent, "deactivated_at")

# --- Chart 1: median ask vs median sold last-ask ---
fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True,
    row_heights=[0.65, 0.35],
    vertical_spacing=0.06,
    subplot_titles=("Median price (weekly)", "Volume (weekly)"),
)

if not active_weekly.empty:
    fig.add_trace(
        go.Scatter(
            x=active_weekly["week"], y=active_weekly["median_price"],
            mode="lines+markers", name="active ask (median)",
            line=dict(color="#1f77b4", width=2.5),
            marker=dict(size=7),
        ),
        row=1, col=1,
    )
if not sold_weekly.empty:
    fig.add_trace(
        go.Scatter(
            x=sold_weekly["week"], y=sold_weekly["median_price"],
            mode="lines+markers", name="sold last-ask (median)",
            line=dict(color="#ff7f0e", width=2.5),
            marker=dict(size=7),
        ),
        row=1, col=1,
    )

# --- Chart 2: volume bars ---
if not active_weekly.empty:
    fig.add_trace(
        go.Bar(
            x=active_weekly["week"], y=active_weekly["n"],
            name="new actives / week",
            marker_color="rgba(31, 119, 180, 0.55)",
        ),
        row=2, col=1,
    )
if not sold_weekly.empty:
    fig.add_trace(
        go.Bar(
            x=sold_weekly["week"], y=sold_weekly["n"],
            name="sold / week",
            marker_color="rgba(255, 127, 14, 0.55)",
        ),
        row=2, col=1,
    )

fig.update_yaxes(title_text="EUR", tickformat=",", row=1, col=1)
fig.update_yaxes(title_text="count", row=2, col=1)
fig.update_layout(
    height=620, margin=dict(l=10, r=10, t=40, b=10), barmode="group",
    legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
)
st.plotly_chart(fig, use_container_width=True)

# --- Trend summary ---
if len(active_weekly) >= 2:
    early = active_weekly.iloc[:max(1, len(active_weekly) // 3)]["median_price"].median()
    late = active_weekly.iloc[-max(1, len(active_weekly) // 3):]["median_price"].median()
    if early and not pd.isna(early):
        delta_pct = (late - early) / early * 100
        direction = "↑" if delta_pct > 1 else "↓" if delta_pct < -1 else "→"
        st.caption(
            f"Active ask trend (early vs late tercile of window): "
            f"€{early:,.0f} → €{late:,.0f}  ({direction} {delta_pct:+.1f}%)"
        )
