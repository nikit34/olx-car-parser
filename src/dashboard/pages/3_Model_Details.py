"""Model Details — every listing for one (brand, model, generation).

Pick a model. Get a sortable table of every listing — active, sold,
expired — with status, year, mileage, price, days-on-market, the
model's predicted price, deal score, and a direct link to OLX/SV.
Below the table: scatter price-vs-mileage with model band, weekly
median trajectory, time-on-market histogram, and calibration scatter.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_dashboard_dir = _Path(__file__).resolve().parent.parent
_project_root = _dashboard_dir.parent.parent
_sys.path.insert(0, str(_dashboard_dir))
_sys.path.insert(0, str(_project_root))

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_loader import (
    load_all, _ensure_release_assets, DB_PATH, _force_next_check,
    get_last_release_error,
)


st.title("Model Details")
st.caption(
    "Every listing for one (brand, model, generation). Sortable table "
    "of active / sold / expired rows + visual context underneath."
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


# Defensive unpack — Streamlit Cloud can serve a previously-cached
# 8-tuple right after a redeploy that adds a 9th element (predictions
# was added 2026-05-03). Index into the tuple with safe defaults so
# the page boots either way; the next cache miss returns the new shape.
_loaded = _load(_release_cache_signature())
listings_df = _loaded[0] if len(_loaded) > 0 else pd.DataFrame()
history_df = _loaded[1] if len(_loaded) > 1 else pd.DataFrame()
signals_df = _loaded[2] if len(_loaded) > 2 else pd.DataFrame()
brands_models = _loaded[3] if len(_loaded) > 3 else {}
predictions_df = _loaded[8] if len(_loaded) > 8 else pd.DataFrame()
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

# --- Sidebar ---
st.sidebar.header("Pick a model")
brand = st.sidebar.selectbox("Brand", sorted(brands_models.keys()))
model = st.sidebar.selectbox("Model", sorted(brands_models.get(brand, [])))

seg = listings_df[
    (listings_df["brand"] == brand) & (listings_df["model"] == model)
].copy()
if "duplicate_of" in seg.columns:
    seg = seg[seg["duplicate_of"].isna()]
if seg.empty:
    st.info("No listings in this model.")
    st.stop()
if "generation" not in seg.columns:
    seg["generation"] = ""
gens = sorted({g for g in seg["generation"].fillna("").tolist() if g})
gen_pick = st.sidebar.selectbox("Generation", ["(all)"] + gens)
if gen_pick != "(all)":
    seg = seg[seg["generation"] == gen_pick]

# Status pivot
is_active = seg["is_active"].astype(bool) if "is_active" in seg.columns else pd.Series(True, index=seg.index)
reason = seg.get("deactivation_reason", pd.Series("", index=seg.index)).astype(str)
is_sold = ~is_active & (reason == "sold")
is_expired = ~is_active & ~is_sold

status_options = ["Active", "Sold", "Expired/other"]
status_pick = st.sidebar.multiselect(
    "Status", status_options, default=status_options,
)
mask = pd.Series(False, index=seg.index)
if "Active" in status_pick:
    mask |= is_active
if "Sold" in status_pick:
    mask |= is_sold
if "Expired/other" in status_pick:
    mask |= is_expired
seg = seg[mask].copy()

# --- Headline metrics ---
n_active = int(is_active[seg.index.intersection(is_active.index)].sum()) if not seg.empty else 0
sold_subset = is_sold[seg.index.intersection(is_sold.index)] if not seg.empty else pd.Series([], dtype=bool)
n_sold = int(sold_subset.sum())
expired_subset = is_expired[seg.index.intersection(is_expired.index)] if not seg.empty else pd.Series([], dtype=bool)
n_expired = int(expired_subset.sum())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Active", n_active)
c2.metric("Sold", n_sold)
c3.metric("Expired/other", n_expired)
if not seg.empty:
    med_price = seg["price_eur"].median()
    c4.metric("Median price", f"€{int(med_price):,}" if pd.notna(med_price) else "—")
else:
    c4.metric("Median price", "—")

st.divider()

if seg.empty:
    st.info("No listings match the status filter.")
    st.stop()

# --- Build the table ---
seg["__deact_dt"] = pd.to_datetime(seg.get("deactivated_at"), errors="coerce", utc=True)
seg["__first_dt"] = pd.to_datetime(seg.get("first_seen_at"), errors="coerce", utc=True)
seg["__last_dt"] = pd.to_datetime(seg.get("last_seen_at"), errors="coerce", utc=True)
now = pd.Timestamp.now(tz="UTC")

def _row_status(idx):
    if is_active.get(idx, False):
        return "🟢 active"
    if is_sold.get(idx, False):
        return "🔵 sold"
    return "⚫ expired"

seg["__status"] = [_row_status(i) for i in seg.index]
# DoM: for sold/expired = (deactivated - first_seen). For active = (now - first_seen).
seg["__dom"] = (
    (seg["__deact_dt"].fillna(now) - seg["__first_dt"]).dt.total_seconds() / 86400
).round(0)
seg.loc[seg["__dom"] < 0, "__dom"] = np.nan
seg.loc[seg["__dom"] > 730, "__dom"] = np.nan  # parser-noise outliers

# Predicted / deal score from signals (only available for actives the model scored).
sig_lookup = (
    signals_df.set_index("olx_id") if (signals_df is not None and not signals_df.empty)
    else None
)
def _sig_field(oid: str, col: str):
    if sig_lookup is None or oid not in sig_lookup.index:
        return None
    val = sig_lookup.loc[oid, col]
    if isinstance(val, pd.Series):
        val = val.iloc[0]
    return val if pd.notna(val) else None

seg["__predicted"] = seg["olx_id"].map(
    lambda o: _sig_field(o, "predicted_price") if sig_lookup is not None and "predicted_price" in sig_lookup.columns else None
)
seg["__score"] = seg["olx_id"].map(
    lambda o: _sig_field(o, "flip_score") if sig_lookup is not None and "flip_score" in sig_lookup.columns else None
)

# Order rows: active+highest score first, then sold by deactivation desc, then expired.
seg["__sort_bucket"] = seg["__status"].map({"🟢 active": 0, "🔵 sold": 1, "⚫ expired": 2})
seg = seg.sort_values(
    ["__sort_bucket", "__score", "__deact_dt"],
    ascending=[True, False, False],
    na_position="last",
)

table = pd.DataFrame({
    "status": seg["__status"],
    "year": seg["year"].astype("Int64"),
    "km": seg["mileage_km"].astype("Int64"),
    "price €": pd.to_numeric(seg["price_eur"], errors="coerce").round(0).astype("Int64"),
    "predicted €": pd.to_numeric(seg["__predicted"], errors="coerce").round(0).astype("Int64"),
    "score": pd.to_numeric(seg["__score"], errors="coerce").round(0).astype("Int64"),
    "DoM (d)": pd.to_numeric(seg["__dom"], errors="coerce").astype("Int64"),
    "trim": seg.get("trim_level", "").fillna(""),
    "sub_model": seg.get("sub_model", "").fillna(""),
    "fuel": seg.get("fuel_type", "").fillna(""),
    "city": seg.get("city", "").fillna(""),
    "deactivated": seg["__deact_dt"].dt.strftime("%Y-%m-%d").fillna(""),
    "url": seg["url"].fillna(""),
})

st.subheader(f"Listings ({len(table)})")
st.caption(
    "DoM = days-on-market (active rows: now − first_seen; closed rows: "
    "deactivated − first_seen). Predicted / score only populated for "
    "rows the price model scored — usually the active subset."
)

st.dataframe(
    table,
    hide_index=True, use_container_width=True, height=420,
    column_config={
        "url": st.column_config.LinkColumn(
            "link", display_text="↗",
        ),
        "price €": st.column_config.NumberColumn(format="%d"),
        "predicted €": st.column_config.NumberColumn(format="%d"),
        "score": st.column_config.NumberColumn(format="%d"),
        "DoM (d)": st.column_config.NumberColumn(format="%d"),
    },
)

st.divider()

# Build prediction lookups for the rest of the page. Use predictions_df
# (full active set with GB predictions), NOT signals_df (deals only).
# Otherwise the band overlay below misses every active listing whose
# asking price wasn't below the model's predicted — i.e., most of them.
pred_lookup: dict[str, float] = {}
fair_low_lookup: dict[str, float] = {}
fair_high_lookup: dict[str, float] = {}
if predictions_df is not None and not predictions_df.empty:
    for _, row in predictions_df.iterrows():
        oid = row.get("olx_id")
        if not isinstance(oid, str):
            continue
        p = row.get("predicted_price")
        if pd.notna(p) and p:
            pred_lookup[oid] = float(p)
        fl = row.get("fair_price_low")
        fh = row.get("fair_price_high")
        if pd.notna(fl):
            fair_low_lookup[oid] = float(fl)
        if pd.notna(fh):
            fair_high_lookup[oid] = float(fh)

active_seg = seg[is_active.reindex(seg.index, fill_value=False)]
sold_seg = seg[is_sold.reindex(seg.index, fill_value=False)]

# --- Panel: Price-vs-mileage scatter with model band ---
st.subheader("Price vs mileage")

scatter_df = seg[
    seg["price_eur"].notna() & seg["mileage_km"].notna() & (seg["price_eur"] > 0)
].copy()
if scatter_df.empty:
    st.info("No listings with mileage + price.")
else:
    sa = scatter_df[is_active.reindex(scatter_df.index, fill_value=False)]
    ss = scatter_df[is_sold.reindex(scatter_df.index, fill_value=False)]

    fig = go.Figure()
    # Sold rows: warmer colour + higher alpha than the v1 grey-hollow,
    # so they read as part of the data instead of "background noise".
    # User feedback: "более заметные распроданные точки".
    if not ss.empty:
        fig.add_scatter(
            x=ss["mileage_km"], y=ss["price_eur"],
            mode="markers",
            marker=dict(
                size=10, color="rgba(176, 92, 92, 0.55)",
                symbol="x-thin", line=dict(width=2, color="rgba(176, 92, 92, 0.85)"),
            ),
            name=f"sold ({len(ss)})",
            customdata=ss[["olx_id", "url"]].values,
            hovertemplate=(
                "<b>SOLD · €%{y:,.0f}</b> · %{x:,.0f} km<br>"
                "%{customdata[0]} — click to open<extra></extra>"
            ),
        )
    if not sa.empty:
        has_year = bool(sa["year"].notna().any())
        fig.add_scatter(
            x=sa["mileage_km"], y=sa["price_eur"],
            mode="markers",
            marker=dict(
                size=11,
                color=sa["year"] if has_year else "#1f77b4",
                colorscale="Turbo",
                showscale=has_year,
                colorbar=dict(title="year") if has_year else None,
                opacity=0.88,
                line=dict(width=0.5, color="#222"),
            ),
            name=f"active ({len(sa)})",
            customdata=sa[["olx_id", "url"]].values,
            hovertemplate=(
                "<b>€%{y:,.0f}</b> · %{x:,.0f} km<br>"
                "%{customdata[0]} — click to open<extra></extra>"
            ),
        )
    if pred_lookup and not sa.empty:
        band = sa.copy()
        band["fair_low"] = band["olx_id"].map(fair_low_lookup)
        band["fair_high"] = band["olx_id"].map(fair_high_lookup)
        band["predicted"] = band["olx_id"].map(pred_lookup)
        band = band.dropna(
            subset=["fair_low", "fair_high", "predicted"],
        ).sort_values("mileage_km")
        if not band.empty:
            # Smooth via rolling median across mileage so a single
            # quirky listing's prediction doesn't kink the curve. The
            # band itself is the model's 80 % CQR interval (P10–P90);
            # the centre line is the calibrated median (P50).
            win = max(3, min(7, len(band) // 5))
            sm_low = (
                band["fair_low"].rolling(win, min_periods=1, center=True).median()
            )
            sm_high = (
                band["fair_high"].rolling(win, min_periods=1, center=True).median()
            )
            sm_pred = (
                band["predicted"].rolling(win, min_periods=1, center=True).median()
            )
            # Lower edge (P10) — visible dashed line.
            fig.add_scatter(
                x=band["mileage_km"], y=sm_low,
                mode="lines",
                line=dict(color="rgba(46, 160, 67, 0.65)", width=1.5, dash="dash"),
                name="fair low (P10)",
                hovertemplate="<b>P10: €%{y:,.0f}</b> @ %{x:,.0f} km<extra></extra>",
            )
            # Upper edge (P90) — visible dashed line with fill back to P10.
            fig.add_scatter(
                x=band["mileage_km"], y=sm_high,
                mode="lines",
                line=dict(color="rgba(46, 160, 67, 0.65)", width=1.5, dash="dash"),
                fill="tonexty", fillcolor="rgba(46, 160, 67, 0.18)",
                name="fair high (P90)",
                hovertemplate="<b>P90: €%{y:,.0f}</b> @ %{x:,.0f} km<extra></extra>",
            )
            # Center prediction (P50) — solid darker line.
            fig.add_scatter(
                x=band["mileage_km"], y=sm_pred,
                mode="lines",
                line=dict(color="rgba(31, 119, 60, 0.95)", width=2.4),
                name="model predicted (P50)",
                hovertemplate="<b>predicted: €%{y:,.0f}</b> @ %{x:,.0f} km<extra></extra>",
            )
    fig.update_layout(
        xaxis_title="Mileage (km)", yaxis_title="Price (EUR)",
        xaxis=dict(tickformat=","), yaxis=dict(tickformat=","),
        height=440, margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
    )
    # ``on_select="rerun"`` lets the user pick points; we read the
    # selection's customdata (olx_id + url) and offer a one-click
    # link button below the chart so they can jump to the listing.
    chart_event = st.plotly_chart(
        fig, use_container_width=True,
        on_select="rerun", selection_mode="points",
        key="model_details_scatter",
    )
    sel_pts = (
        chart_event.selection.points
        if chart_event and chart_event.selection else []
    )
    if sel_pts:
        # ``customdata`` indexes match the trace ordering above. Plotly
        # returns a single-row list per click; if user selects multiple,
        # we link the first.
        cd = sel_pts[0].get("customdata") or []
        if len(cd) >= 2 and cd[1]:
            st.link_button(
                f"Open listing {cd[0]} ↗", cd[1], use_container_width=True,
            )

# --- Panel: Time-on-market histogram (sold only) ---
st.subheader("Time on market — sold listings")
if sold_seg.empty:
    st.info("No sold listings yet.")
else:
    deact = pd.to_datetime(sold_seg["deactivated_at"], errors="coerce", utc=True)
    first = pd.to_datetime(sold_seg["first_seen_at"], errors="coerce", utc=True)
    dom = ((deact - first).dt.total_seconds() / 86400).dropna()
    dom = dom[(dom >= 0) & (dom <= 365)]
    if dom.empty:
        st.info("No sold rows with valid dates.")
    else:
        fig_h = px.histogram(dom, nbins=30)
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

# --- Panel: Calibration ---
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
        residual = float((cal["price_eur"] - cal["predicted"]).median())
        med_pct = float(((cal["price_eur"] - cal["predicted"]) / cal["predicted"]).median() * 100)
        fig_c = px.scatter(
            cal, x="predicted", y="price_eur",
            hover_data=["olx_id", "year", "mileage_km"],
            opacity=0.7,
        )
        lo = float(min(cal["predicted"].min(), cal["price_eur"].min()))
        hi = float(max(cal["predicted"].max(), cal["price_eur"].max()))
        fig_c.add_scatter(
            x=[lo, hi], y=[lo, hi], mode="lines",
            line=dict(dash="dash", color="#888"),
            name="actual = predicted", showlegend=False,
        )
        fig_c.update_layout(
            xaxis_title="Model predicted (€)", yaxis_title="Actual last ask (€)",
            xaxis=dict(tickformat=","), yaxis=dict(tickformat=","),
            height=380, margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_c, use_container_width=True)
        bias = "under-predicts" if residual > 0 else "over-predicts"
        st.caption(
            f"Median residual: €{residual:+,.0f} ({med_pct:+.1f}%). "
            f"Model {bias} this segment by that amount on sold rows."
        )
