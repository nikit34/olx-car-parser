"""Segment detail — drill into one (brand, model, generation) combo.

Four panels:
  1. Price-vs-mileage scatter with the model's predicted band overlaid.
     Sold rows are background-grey, active rows are coloured by year,
     and the user's chosen listing (when filtered to one) becomes a star.
  2. Time-series of median active ask + median sold last-ask, weekly,
     so the gap between them shows how much the market currently
     discounts to clear vs what dealers are still hoping to get.
  3. Time-on-market histogram (sold rows only) — is this segment hot
     (median <30d) or stuck (median >90d)?
  4. Calibration scatter: actual last-ask vs model-predicted price for
     sold rows. The 45° line is the no-bias reference; consistent
     above/below is a segment-specific model skew.
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
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_loader import (
    load_all, _ensure_release_assets, DB_PATH, _force_next_check,
    get_last_release_error,
)


st.set_page_config(page_title="Segment Detail", layout="wide")
st.title("Segment Detail")


def _release_cache_signature() -> tuple[float, int]:
    _ensure_release_assets()
    if not DB_PATH.exists():
        return (0.0, 0)
    s = DB_PATH.stat()
    return (s.st_mtime, s.st_size)


@st.cache_data(ttl=300)
def _load(_sig: tuple[float, int]):
    return load_all()


listings_df, history_df, signals_df, brands_models, *_rest = _load(_release_cache_signature())

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

# --- Sidebar: pick segment ---
st.sidebar.header("Segment")
brand = st.sidebar.selectbox("Brand", sorted(brands_models.keys()))
model = st.sidebar.selectbox("Model", sorted(brands_models.get(brand, [])))

seg = listings_df[
    (listings_df["brand"] == brand) & (listings_df["model"] == model)
].copy()
if "duplicate_of" in seg.columns:
    seg = seg[seg["duplicate_of"].isna()]
if seg.empty:
    st.info("No listings in segment.")
    st.stop()

# Generation pick is optional — empty = all. Defensive: ``generation``
# was added to ``get_listings_df`` on 2026-05-03; older DB releases
# pulled before that change won't have the column.
if "generation" not in seg.columns:
    seg["generation"] = ""
gens = sorted({g for g in seg["generation"].fillna("").tolist() if g})
gen_options = ["(all)"] + gens
gen_pick = st.sidebar.selectbox("Generation", gen_options)
if gen_pick != "(all)":
    seg = seg[seg["generation"] == gen_pick]

is_active = seg["is_active"].astype(bool) if "is_active" in seg.columns else pd.Series(True, index=seg.index)
is_sold = ~is_active & (seg.get("deactivation_reason", "").astype(str) == "sold")
active_seg = seg[is_active]
sold_seg = seg[is_sold]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Active", len(active_seg))
c2.metric("Sold (any)", len(sold_seg))
c3.metric(
    "Median active ask",
    f"€{int(active_seg['price_eur'].median()):,}" if not active_seg.empty else "—",
)
c4.metric(
    "Median sold ask",
    f"€{int(sold_seg['price_eur'].median()):,}" if not sold_seg.empty else "—",
)

st.divider()

# Build a per-row predicted price lookup from signals_df once.
pred_lookup: dict[str, float] = {}
fair_low_lookup: dict[str, float] = {}
fair_high_lookup: dict[str, float] = {}
if signals_df is not None and not signals_df.empty and "olx_id" in signals_df.columns:
    for _, sig in signals_df.iterrows():
        oid = sig.get("olx_id")
        if not isinstance(oid, str):
            continue
        p = sig.get("predicted_price")
        if pd.notna(p) and p:
            pred_lookup[oid] = float(p)
        fl = sig.get("fair_price_low")
        fh = sig.get("fair_price_high")
        if pd.notna(fl):
            fair_low_lookup[oid] = float(fl)
        if pd.notna(fh):
            fair_high_lookup[oid] = float(fh)


# --- Panel 1: Price vs mileage with predicted band ---
st.subheader("Price vs mileage (with model band)")

scatter_df = seg[
    seg["price_eur"].notna() & seg["mileage_km"].notna() & (seg["price_eur"] > 0)
].copy()
if scatter_df.empty:
    st.info("No listings with mileage + price.")
else:
    scatter_active = scatter_df[is_active.reindex(scatter_df.index, fill_value=False)]
    scatter_sold = scatter_df[is_sold.reindex(scatter_df.index, fill_value=False)]

    fig = go.Figure()
    if not scatter_sold.empty:
        fig.add_scatter(
            x=scatter_sold["mileage_km"], y=scatter_sold["price_eur"],
            mode="markers",
            marker=dict(size=8, color="#888888", opacity=0.32, symbol="circle-open"),
            name=f"sold ({len(scatter_sold)})",
            hovertemplate="<b>€%{y:,.0f}</b> · %{x:,.0f} km<extra></extra>",
        )
    if not scatter_active.empty:
        has_year = bool(scatter_active["year"].notna().any())
        fig.add_scatter(
            x=scatter_active["mileage_km"], y=scatter_active["price_eur"],
            mode="markers",
            marker=dict(
                size=10,
                color=scatter_active["year"] if has_year else "#1f77b4",
                colorscale="Turbo",
                showscale=has_year,
                colorbar=dict(title="year") if has_year else None,
                opacity=0.85,
                line=dict(width=0.5, color="#222"),
            ),
            name=f"active ({len(scatter_active)})",
            hovertemplate="<b>€%{y:,.0f}</b> · %{x:,.0f} km<extra></extra>",
        )

    # Overlay model band as a shaded ribbon. We don't have a smooth
    # predict-along-mileage curve, but we DO have per-listing fair_low /
    # fair_high. Draw them as scatter ribbons for actives — points where
    # the model's [low, high] sits at each x.
    if pred_lookup and not scatter_active.empty:
        band = scatter_active.copy()
        band["fair_low"] = band["olx_id"].map(fair_low_lookup)
        band["fair_high"] = band["olx_id"].map(fair_high_lookup)
        band = band.dropna(subset=["fair_low", "fair_high"]).sort_values("mileage_km")
        if not band.empty:
            fig.add_scatter(
                x=band["mileage_km"], y=band["fair_low"],
                mode="lines",
                line=dict(color="rgba(50, 200, 50, 0.0)"),
                showlegend=False, hoverinfo="skip",
            )
            fig.add_scatter(
                x=band["mileage_km"], y=band["fair_high"],
                mode="lines",
                line=dict(color="rgba(50, 200, 50, 0.0)"),
                fill="tonexty",
                fillcolor="rgba(50, 200, 50, 0.10)",
                name="model 80% band",
                hoverinfo="skip",
            )

    fig.update_layout(
        xaxis_title="Mileage (km)", yaxis_title="Price (EUR)",
        xaxis=dict(tickformat=","), yaxis=dict(tickformat=","),
        height=460, margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Panel 2: Time-series median price (active asks + sold last-ask) ---
st.subheader("Median price over time")
st.caption(
    "Active line built from per-scrape price snapshots — when a "
    "listing's price drops mid-life, the new value lands in the right "
    "week's median, not stuck at first-seen."
)

from src.analytics.segments import compute_segment_time_series
from src.storage.repository import get_price_snapshots_df
from src.storage.database import init_db, get_session


@st.cache_data(ttl=600)
def _load_snapshots(_sig):
    init_db()
    s = get_session()
    try:
        return get_price_snapshots_df(s, since_days=180)
    finally:
        s.close()


snapshots = _load_snapshots(_release_cache_signature())
if not snapshots.empty:
    seg_snaps = snapshots[
        (snapshots["brand"] == brand) & (snapshots["model"] == model)
    ]
    if gen_pick != "(all)":
        seg_snaps = seg_snaps[seg_snaps["generation"] == gen_pick]
else:
    seg_snaps = pd.DataFrame()

ts = compute_segment_time_series(seg_snaps, sold_listings=sold_seg)
if ts.empty:
    st.info("Not enough snapshot history yet for a time series.")
else:
    label_map = {
        "active_ask_median": "active ask (median)",
        "sold_lastask_median": "sold last-ask (median)",
    }
    ts = ts.assign(label=ts["series"].map(label_map))
    fig_ts = px.line(
        ts, x="bucket", y="value", color="label",
        markers=True,
        color_discrete_map={
            "active ask (median)": "#1f77b4",
            "sold last-ask (median)": "#ff7f0e",
        },
    )
    fig_ts.update_layout(
        xaxis_title=None, yaxis_title="Price (EUR)",
        height=320, margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
    )
    st.plotly_chart(fig_ts, use_container_width=True)
    st.caption(
        "Gap between blue and orange = how much current ASKs sit above "
        "what the market actually cleared at recently."
    )

# --- Panel 3: Time-on-market histogram ---
st.subheader("Time-on-market — sold listings")
if sold_seg.empty:
    st.info("No sold listings in this segment yet.")
else:
    deact = pd.to_datetime(sold_seg["deactivated_at"], errors="coerce", utc=True)
    first = pd.to_datetime(sold_seg["first_seen_at"], errors="coerce", utc=True)
    dom = ((deact - first).dt.total_seconds() / 86400).dropna()
    dom = dom[(dom >= 0) & (dom <= 365)]
    if dom.empty:
        st.info("No sold rows with valid dates.")
    else:
        fig_h = px.histogram(
            dom, nbins=30,
            labels={"value": "days on market"},
        )
        fig_h.update_layout(
            xaxis_title="Days on market", yaxis_title="Count",
            height=280, margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False,
        )
        med = float(dom.median())
        fig_h.add_vline(
            x=med, line_dash="dash", line_color="#ff7f0e",
            annotation_text=f"median {med:.0f}d",
            annotation_position="top right",
        )
        st.plotly_chart(fig_h, use_container_width=True)

# --- Panel 4: Calibration (actual last-ask vs predicted) ---
st.subheader("Model calibration on sold listings")
if not pred_lookup or sold_seg.empty:
    st.info("Need both a saved price model and sold listings to calibrate.")
else:
    cal = sold_seg.copy()
    cal["predicted"] = cal["olx_id"].map(pred_lookup)
    cal = cal.dropna(subset=["predicted", "price_eur"])
    cal = cal[(cal["predicted"] > 0) & (cal["price_eur"] > 0)]
    if cal.empty:
        st.info("No sold listings with both a prediction and an ask.")
    else:
        residual = (cal["price_eur"] - cal["predicted"]).median()
        med_pct = float(((cal["price_eur"] - cal["predicted"]) / cal["predicted"]).median() * 100)

        fig_c = px.scatter(
            cal, x="predicted", y="price_eur",
            hover_data=["olx_id", "year", "mileage_km"],
            opacity=0.7,
        )
        # 45° reference line
        lo = float(min(cal["predicted"].min(), cal["price_eur"].min()))
        hi = float(max(cal["predicted"].max(), cal["price_eur"].max()))
        fig_c.add_scatter(
            x=[lo, hi], y=[lo, hi], mode="lines",
            line=dict(dash="dash", color="#888"),
            name="actual = predicted",
            showlegend=False,
        )
        fig_c.update_layout(
            xaxis_title="Model predicted (€)", yaxis_title="Actual last ask (€)",
            xaxis=dict(tickformat=","), yaxis=dict(tickformat=","),
            height=380, margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_c, use_container_width=True)
        bias_word = "under-predicts" if residual > 0 else "over-predicts"
        st.caption(
            f"Median residual: €{residual:+,.0f} ({med_pct:+.1f}%). "
            f"Model {bias_word} this segment by that amount on sold rows."
        )
