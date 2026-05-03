"""Market Direction — where is the market moving, by segment.

Compare median asking-price trajectories across cohorts that share a
chosen split (fuel group / price tier / year cohort / brand). Answers
questions like "is electric softening while diesel holds?" or "are
2015-2017 cars losing ground while 2020+ stays flat?".

Three blocks:
  1. Multi-line price-over-time chart, one line per group.
  2. Delta table: group / start price / end price / change €/% over
     the look-back window.
  3. Volume bar chart: distinct listings per group in the window.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_dashboard_dir = _Path(__file__).resolve().parent.parent
_project_root = _dashboard_dir.parent.parent
_sys.path.insert(0, str(_dashboard_dir))
_sys.path.insert(0, str(_project_root))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from data_loader import (
    load_all, _ensure_release_assets, DB_PATH, _force_next_check,
    get_last_release_error, _fuel_group,
)
from src.storage.repository import get_price_snapshots_df
from src.storage.database import init_db, get_session


st.set_page_config(page_title="Market Direction", layout="wide")
st.title("Market Direction — by segment")
st.caption(
    "Compare median asking-price trajectories across cohorts sharing a "
    "split (fuel / price tier / year cohort / brand). The question this "
    "page answers: which segments are softening and which are holding?"
)


def _release_cache_signature() -> tuple[float, int]:
    _ensure_release_assets()
    if not DB_PATH.exists():
        return (0.0, 0)
    s = DB_PATH.stat()
    return (s.st_mtime, s.st_size)


@st.cache_data(ttl=300)
def _load(_sig):
    return load_all()


@st.cache_data(ttl=600)
def _load_snapshots(_sig, since_days: int):
    init_db()
    s = get_session()
    try:
        return get_price_snapshots_df(s, since_days=since_days)
    finally:
        s.close()


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

snapshots = _load_snapshots(_release_cache_signature(), 365)
if snapshots.empty:
    st.warning("No snapshot history yet.")
    st.stop()

snapshots["scraped_at"] = pd.to_datetime(
    snapshots["scraped_at"], errors="coerce", utc=True,
)
data_start = snapshots["scraped_at"].min()
data_end = snapshots["scraped_at"].max()
data_days = max(1, int((data_end - data_start).total_seconds() / 86400))

st.info(
    f"📅 **Data range:** {data_start.strftime('%Y-%m-%d')} → "
    f"{data_end.strftime('%Y-%m-%d')}  ·  **{data_days} days** of history."
)

# --- Sidebar ---
st.sidebar.header("Split & window")

split_options = [
    "Fuel group",
    "Year cohort",
    "Price tier",
    "Brand (top 8)",
    "Damage severity",
]
split_dim = st.sidebar.selectbox("Split by", split_options)

_QUICK_LABELS = ("1m", "3m", "6m", "1y", "all")
_QUICK_DAYS = {"1m": 30, "3m": 90, "6m": 180, "1y": 365, "all": data_days}
quick = st.sidebar.segmented_control(
    "Look-back", options=_QUICK_LABELS, default="3m",
    key="md_quick_window",
) or "3m"
window_days_request = _QUICK_DAYS[quick]
window_days = max(7, min(window_days_request, data_days))
if window_days_request > data_days:
    st.sidebar.caption(
        f"{quick} ({window_days_request}d) → clamped to {window_days}d."
    )

if window_days <= 35:
    auto_freq = "D"
elif window_days <= 120:
    auto_freq = "W"
else:
    auto_freq = "M"
freq = st.sidebar.selectbox(
    "Bucket", ["auto", "daily", "weekly", "monthly"], index=0,
)
freq = {"auto": auto_freq, "daily": "D",
        "weekly": "W", "monthly": "M"}[freq]
freq_label = {"D": "daily", "W": "weekly", "M": "monthly"}[freq]

st.sidebar.divider()
brand_options = sorted(listings_df["brand"].dropna().unique().tolist())
brand_filter = st.sidebar.multiselect(
    "Brand filter (optional)", brand_options,
    help="Restrict the comparison to chosen brands. Empty = all.",
)

# --- Apply filters to snapshots ---
df = snapshots
if "duplicate_of" in df.columns:
    df = df[df["duplicate_of"].isna()]
if brand_filter:
    df = df[df["brand"].isin(brand_filter)]
cutoff = data_end - pd.Timedelta(days=window_days)
df = df[df["scraped_at"] >= cutoff]
df = df[df["price_eur"] > 0].copy()

if df.empty:
    st.info("No snapshots after filters.")
    st.stop()

# --- Compute the split column based on chosen dimension ---
def _year_cohort(y):
    if pd.isna(y):
        return "?"
    y = int(y)
    if y < 2010:
        return "≤2009"
    if y < 2015:
        return "2010-2014"
    if y < 2020:
        return "2015-2019"
    return "2020+"


def _price_tier(p):
    if p < 5000:
        return "<€5 k"
    if p < 15000:
        return "€5-15 k"
    if p < 30000:
        return "€15-30 k"
    return "€30 k+"


def _severity_label(s):
    try:
        i = int(s)
    except (TypeError, ValueError):
        return "?"
    return {0: "0 pristine", 1: "1 normal wear", 2: "2 needs repair",
            3: "3 salvage"}.get(i, "?")


if split_dim == "Fuel group":
    df["__group"] = df["fuel_type"].map(_fuel_group)
elif split_dim == "Year cohort":
    df["__group"] = df["year"].map(_year_cohort)
elif split_dim == "Price tier":
    df["__group"] = df["price_eur"].map(_price_tier)
elif split_dim == "Brand (top 8)":
    top_brands = (
        df.groupby("brand")["olx_id"].nunique()
        .sort_values(ascending=False).head(8).index
    )
    df["__group"] = df["brand"].where(df["brand"].isin(top_brands), other="other")
elif split_dim == "Damage severity":
    if "damage_severity" not in listings_df.columns:
        st.info("damage_severity not in current release.")
        st.stop()
    sev_lookup = listings_df.set_index("olx_id")["damage_severity"]
    df["__group"] = df["olx_id"].map(sev_lookup).map(_severity_label)
else:
    df["__group"] = "all"

df = df[df["__group"].notna() & (df["__group"].astype(str) != "")]
df["__group"] = df["__group"].astype(str)

# Drop tiny groups (< 5 distinct listings) — they make noisy lines.
group_sizes = df.groupby("__group")["olx_id"].nunique()
keep_groups = group_sizes[group_sizes >= 5].index.tolist()
df = df[df["__group"].isin(keep_groups)]
if df.empty:
    st.info("All groups had fewer than 5 distinct listings — nothing to compare.")
    st.stop()

st.caption(
    f"Comparing **{len(keep_groups)}** {split_dim.lower()} groups · "
    f"window **{window_days}d** · bucket **{freq_label}**. "
    "Groups with <5 distinct listings hidden."
)

# --- Per-listing-per-bucket aggregation ---
df["bucket"] = df["scraped_at"].dt.to_period(freq).dt.start_time
per_listing = (
    df.groupby(["__group", "olx_id", "bucket"])["price_eur"]
    .median().reset_index()
)
group_buckets = (
    per_listing.groupby(["__group", "bucket"])
    .agg(median_price=("price_eur", "median"),
         n_listings=("olx_id", "nunique"))
    .reset_index()
)

# --- Chart 1: multi-line median price over time ---
fig = px.line(
    group_buckets, x="bucket", y="median_price", color="__group",
    markers=True,
    labels={"__group": split_dim, "median_price": "Median ask (€)",
            "bucket": "Date"},
)
fig.update_traces(line=dict(width=2.5), marker=dict(size=7))
fig.update_layout(
    height=440, margin=dict(l=10, r=10, t=10, b=10),
    yaxis=dict(tickformat=","),
    legend=dict(orientation="h", yanchor="bottom", y=1.02,
                xanchor="right", x=1, title=None),
)
st.plotly_chart(fig, use_container_width=True)

# --- Delta table: how each group moved over the window ---
st.subheader("Direction by group")
rows: list[dict] = []
for grp, sub in group_buckets.groupby("__group"):
    sub = sub.sort_values("bucket")
    if len(sub) < 2:
        early_v = late_v = float(sub["median_price"].iloc[0])
    else:
        n = len(sub)
        early_v = float(sub.iloc[: max(1, n // 3)]["median_price"].median())
        late_v = float(sub.iloc[-max(1, n // 3):]["median_price"].median())
    delta_eur = late_v - early_v
    delta_pct = (delta_eur / early_v * 100) if early_v else 0.0
    rows.append({
        "group": grp,
        "median early (€)": round(early_v),
        "median late (€)": round(late_v),
        "Δ €": round(delta_eur),
        "Δ %": round(delta_pct, 1),
        "distinct listings": int(group_sizes.loc[grp]),
    })
delta_df = pd.DataFrame(rows).sort_values("Δ %", ascending=False)
st.dataframe(delta_df, hide_index=True, use_container_width=True)
st.caption(
    "Early / late = median of the first / last tercile of buckets in "
    "the window. Δ % is the relative move; positive = market firming, "
    "negative = softening for that group."
)

# --- Chart 2: volume comparison (distinct listings per group) ---
st.subheader("Volume by group")
vol = group_sizes.loc[keep_groups].sort_values(ascending=False).reset_index()
vol.columns = ["group", "distinct listings"]
fig_vol = px.bar(vol, x="group", y="distinct listings", text="distinct listings")
fig_vol.update_traces(textposition="outside")
fig_vol.update_layout(
    height=320, margin=dict(l=10, r=10, t=10, b=10),
    xaxis_title=None, yaxis_title="distinct listings",
)
st.plotly_chart(fig_vol, use_container_width=True)
