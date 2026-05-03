"""Market trend — global view of how the market is moving over time.

What you see (top to bottom):
  1. Data-range banner: how many days of history actually exist on
     this DB. Nothing on the page can show beyond that — picking a
     "180-day window" doesn't conjure data we never scraped.
  2. Median price over time, two lines:
       - active ask (median): one observation PER LISTING per bucket,
         so a long-lived listing doesn't drown the median by virtue of
         being scraped 30× per week.
       - sold last-ask (median): the last ASK we saw on each
         listing the week it deactivated.
  3. Volume bars: how many DISTINCT active listings vs sold-events
     fell in each bucket.

Bucket frequency (daily / weekly) auto-picks based on look-back, with
a manual override.
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

# --- Data range banner ---
# We can never show market trend beyond what's in price_snapshots, so
# tell the user up-front what the actual coverage is. Without this the
# 180-day default window slider made the page look broken.
from src.storage.repository import get_price_snapshots_df
from src.storage.database import init_db, get_session


@st.cache_data(ttl=600)
def _load_snapshots(_sig, since_days: int):
    init_db()
    s = get_session()
    try:
        return get_price_snapshots_df(s, since_days=since_days)
    finally:
        s.close()


# Pre-load once (1 year cap) just to figure out the actual data span.
_full = _load_snapshots(_release_cache_signature(), 365)
if _full.empty:
    st.warning("No price snapshots yet.")
    st.stop()

_full["scraped_at"] = pd.to_datetime(_full["scraped_at"], errors="coerce", utc=True)
data_start = _full["scraped_at"].min()
data_end = _full["scraped_at"].max()
data_days = max(1, int((data_end - data_start).total_seconds() / 86400))

st.info(
    f"📅 **Data range:** {data_start.strftime('%Y-%m-%d')} → "
    f"{data_end.strftime('%Y-%m-%d')}  ·  **{data_days} days** of price-snapshot history."
)

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

# Cap window at the actual data range — picking 180 days when we only
# have 31 just makes the chart look sparse. User can still expand the
# slider later as the DB grows.
window_max = max(7, min(365, data_days))
window_default = min(window_max, 30)
window_days = st.sidebar.slider(
    "Look-back (days)", 7, window_max, window_default, step=1,
)

# Auto-pick bucket frequency. Daily for short ranges (<35d) gives ~30
# data points; weekly for longer keeps the chart readable.
auto_freq = "D" if window_days <= 35 else "W"
freq_choice = st.sidebar.selectbox(
    "Bucket", ["auto", "daily", "weekly"], index=0,
)
freq = {"auto": auto_freq, "daily": "D", "weekly": "W"}[freq_choice]
freq_label = {"D": "daily", "W": "weekly"}[freq]

# --- Apply filters to snapshots ---
snapshots = _full
if "duplicate_of" in snapshots.columns:
    snapshots = snapshots[snapshots["duplicate_of"].isna()]
if brand != "(all)":
    snapshots = snapshots[snapshots["brand"] == brand]
if model != "(all)":
    snapshots = snapshots[snapshots["model"] == model]
if fuel_pick:
    snapshots = snapshots[snapshots["fuel_type"].map(_fuel_group).isin(fuel_pick)]
snapshots = snapshots[
    (snapshots["year"].between(year_range[0], year_range[1])) | snapshots["year"].isna()
]

cutoff = data_end - pd.Timedelta(days=window_days)
snapshots = snapshots[snapshots["scraped_at"] >= cutoff]

if snapshots.empty:
    st.info("No snapshots match the filters.")
    st.stop()

# --- Apply same filters to listings (for sold and headline metrics) ---
df = listings_df.copy()
if "duplicate_of" in df.columns:
    df = df[df["duplicate_of"].isna()]
if brand != "(all)":
    df = df[df["brand"] == brand]
if model != "(all)":
    df = df[df["model"] == model]
if fuel_pick:
    df = df[df["fuel_type"].map(_fuel_group).isin(fuel_pick)]
df = df[(df["year"].between(year_range[0], year_range[1])) | df["year"].isna()]

is_active = df["is_active"].astype(bool) if "is_active" in df.columns else pd.Series(True, index=df.index)
is_sold = (~is_active) & (df.get("deactivation_reason", "").astype(str) == "sold")
sold_df = df[is_sold].copy()
sold_df["deactivated_at"] = pd.to_datetime(sold_df.get("deactivated_at"), errors="coerce", utc=True)
sold_recent = sold_df[sold_df["deactivated_at"] >= cutoff]

# --- Headline metrics ---
distinct_active = snapshots["olx_id"].nunique()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Distinct listings (window)", f"{distinct_active:,}")
c2.metric("Sold (window)", len(sold_recent))
med_active = float(snapshots["price_eur"].median())
c3.metric("Median active ask", f"€{int(med_active):,}")
c4.metric(
    "Median sold last-ask",
    f"€{int(sold_recent['price_eur'].median()):,}" if not sold_recent.empty
    else "—",
)

st.divider()

# --- Aggregation: ONE observation per listing per bucket ---
# Without this, a listing scraped 30× per week with a stable price
# would dominate the weekly median by sheer count. Per-listing
# aggregation reflects the actual market — each listing votes once
# per bucket, with the median of its observations that bucket.
def _per_listing_bucket(snaps: pd.DataFrame) -> pd.DataFrame:
    if snaps.empty:
        return pd.DataFrame(columns=["bucket", "median_price", "n_listings"])
    s = snaps[snaps["price_eur"] > 0].copy()
    if s.empty:
        return pd.DataFrame(columns=["bucket", "median_price", "n_listings"])
    s["bucket"] = s["scraped_at"].dt.to_period(freq).dt.start_time
    # Step 1: each (listing, bucket) → its median observed price
    per_listing = (
        s.groupby(["olx_id", "bucket"])["price_eur"].median().reset_index()
    )
    # Step 2: bucket → median across distinct listings + count
    return (
        per_listing.groupby("bucket")
        .agg(median_price=("price_eur", "median"),
             n_listings=("olx_id", "nunique"))
        .reset_index()
    )


def _bucket_sold(d: pd.DataFrame) -> pd.DataFrame:
    if d.empty or "deactivated_at" not in d.columns:
        return pd.DataFrame(columns=["bucket", "median_price", "n_listings"])
    work = d[["deactivated_at", "price_eur", "olx_id"]].dropna()
    work = work[work["price_eur"] > 0]
    if work.empty:
        return pd.DataFrame(columns=["bucket", "median_price", "n_listings"])
    work["bucket"] = work["deactivated_at"].dt.to_period(freq).dt.start_time
    return (
        work.groupby("bucket")
        .agg(median_price=("price_eur", "median"),
             n_listings=("olx_id", "nunique"))
        .reset_index()
    )


active_buckets = _per_listing_bucket(snapshots)
sold_buckets = _bucket_sold(sold_recent)

# --- Two-row chart: price + volume ---
fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True,
    row_heights=[0.65, 0.35],
    vertical_spacing=0.07,
    subplot_titles=(
        f"Median price · {freq_label} buckets · per-listing aggregation",
        f"Distinct listings per {freq_label} bucket",
    ),
)

if not active_buckets.empty:
    fig.add_trace(
        go.Scatter(
            x=active_buckets["bucket"], y=active_buckets["median_price"],
            mode="lines+markers", name="active ask (median)",
            line=dict(color="#1f77b4", width=2.5),
            marker=dict(size=7),
            hovertemplate="<b>€%{y:,.0f}</b><br>%{x|%Y-%m-%d}<br>%{customdata} listings<extra></extra>",
            customdata=active_buckets["n_listings"],
        ),
        row=1, col=1,
    )
if not sold_buckets.empty:
    fig.add_trace(
        go.Scatter(
            x=sold_buckets["bucket"], y=sold_buckets["median_price"],
            mode="lines+markers", name="sold last-ask (median)",
            line=dict(color="#ff7f0e", width=2.5),
            marker=dict(size=7),
            hovertemplate="<b>€%{y:,.0f}</b><br>%{x|%Y-%m-%d}<br>%{customdata} sold<extra></extra>",
            customdata=sold_buckets["n_listings"],
        ),
        row=1, col=1,
    )

if not active_buckets.empty:
    fig.add_trace(
        go.Bar(
            x=active_buckets["bucket"], y=active_buckets["n_listings"],
            name="active listings",
            marker_color="rgba(31, 119, 180, 0.55)",
        ),
        row=2, col=1,
    )
if not sold_buckets.empty:
    fig.add_trace(
        go.Bar(
            x=sold_buckets["bucket"], y=sold_buckets["n_listings"],
            name="sold",
            marker_color="rgba(255, 127, 14, 0.75)",
        ),
        row=2, col=1,
    )

fig.update_yaxes(title_text="EUR", tickformat=",", row=1, col=1)
fig.update_yaxes(title_text="count", row=2, col=1)
fig.update_layout(
    height=640, margin=dict(l=10, r=10, t=50, b=10), barmode="group",
    legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="right", x=1),
)
st.plotly_chart(fig, use_container_width=True)

# --- Trend summary ---
if len(active_buckets) >= 3:
    n = len(active_buckets)
    early = active_buckets.iloc[: max(1, n // 3)]["median_price"].median()
    late = active_buckets.iloc[-max(1, n // 3):]["median_price"].median()
    if early and not pd.isna(early):
        delta_pct = (late - early) / early * 100
        direction = "↑" if delta_pct > 1 else "↓" if delta_pct < -1 else "→"
        st.caption(
            f"Trend: median ask moved €{early:,.0f} → €{late:,.0f}  "
            f"({direction} {delta_pct:+.1f}% over the window)"
        )
elif not active_buckets.empty:
    st.caption(
        f"Only {len(active_buckets)} {freq_label} buckets in window — pick a "
        "longer look-back or switch to daily granularity."
    )
