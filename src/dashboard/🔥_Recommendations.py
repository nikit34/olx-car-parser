"""Recommendations — deal-discovery scatter.

stlite + Cloudflare Pages simplification: the deal-cards / TreeSHAP /
decision-context surface lived behind a chunk of text-heavy Streamlit
widgets that made for a 550-line page. The page is now a single plotly
scatter (asking price vs predicted fair price, coloured by flip score)
plus the sidebar brand/model/fuel/year/price filters. Anything that
isn't a chart or a navigation control lives on Market Direction or
Model Details instead.
"""

import sys as _sys
from pathlib import Path as _Path

import streamlit as st
import pandas as pd
import plotly.express as px

_dashboard_dir = _Path(__file__).resolve().parent
_project_root = _dashboard_dir.parent.parent
_sys.path.insert(0, str(_dashboard_dir))
_sys.path.insert(0, str(_project_root))

st.set_page_config(
    page_title="OLX Car Deals",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

from data_loader import reboot_dashboard, get_last_release_error, _fuel_group
from _cache import (
    release_signature as _release_cache_signature,
    load_all_cached,
    render_context_badge,
)


_loaded = load_all_cached(_release_cache_signature())
listings_df: pd.DataFrame = _loaded[0] if len(_loaded) > 0 else pd.DataFrame()
signals_df: pd.DataFrame = _loaded[2] if len(_loaded) > 2 else pd.DataFrame()
brands_models: dict[str, list[str]] = _loaded[3] if len(_loaded) > 3 else {}


# ---------------------------------------------------------------------------
# Empty-state — release missing or fetch failed.
# ---------------------------------------------------------------------------
if listings_df.empty:
    st.sidebar.warning("No data yet. Scraper runs every 4 hours via GitHub Actions.")
    err = get_last_release_error()
    if err:
        st.sidebar.error(f"Release fetch failed: {err}")
    if st.sidebar.button("Force refresh"):
        reboot_dashboard()
    st.title("Car Deals")
    render_context_badge(listings_df, signals_df)
    st.info("Waiting for the first witness build to land in the latest-data release.")
    st.stop()


# ---------------------------------------------------------------------------
# Sidebar — filters
# ---------------------------------------------------------------------------
st.sidebar.title("Filters")

selected_brands = st.sidebar.multiselect(
    "Brand",
    options=sorted(brands_models.keys()),
)

available_models: list[str] = []
if selected_brands:
    for _b in selected_brands:
        available_models.extend(brands_models.get(_b, []))
    available_models = sorted(set(available_models))
selected_models = (
    st.sidebar.multiselect("Model", options=available_models)
    if available_models else []
)

fuel_groups_in_data = (
    sorted({_fuel_group(v) for v in signals_df["fuel_type"].tolist()})
    if not signals_df.empty and "fuel_type" in signals_df.columns
    else []
)
selected_fuels = st.sidebar.multiselect("Fuel type", options=fuel_groups_in_data)

year_min = int(listings_df["year"].min()) if listings_df["year"].notna().any() else 2000
year_max = int(listings_df["year"].max()) if listings_df["year"].notna().any() else 2026
year_range = st.sidebar.slider(
    "Year", min_value=year_min, max_value=year_max, value=(year_min, year_max),
)

price_max_val = (
    int(listings_df["price_eur"].max())
    if listings_df["price_eur"].notna().any() else 50000
)
price_range = st.sidebar.slider(
    "Price (EUR)", min_value=0, max_value=price_max_val,
    value=(0, price_max_val), step=500,
)


# ---------------------------------------------------------------------------
# Apply filters to the deal feed.
# ---------------------------------------------------------------------------
deals = signals_df.copy()
if selected_brands:
    deals = deals[deals["brand"].isin(selected_brands)]
if selected_models:
    deals = deals[deals["model"].isin(selected_models)]
if selected_fuels and "fuel_type" in deals.columns:
    deals = deals[deals["fuel_type"].apply(_fuel_group).isin(selected_fuels)]
if "year" in deals.columns:
    deals = deals[
        deals["year"].between(year_range[0], year_range[1], inclusive="both")
    ]
if "price_eur" in deals.columns:
    deals = deals[
        deals["price_eur"].between(price_range[0], price_range[1], inclusive="both")
    ]


# ---------------------------------------------------------------------------
# Main — context + scatter
# ---------------------------------------------------------------------------
st.title("Car Deals")
render_context_badge(listings_df, signals_df)

if deals.empty:
    st.info("No deals match the current filters.")
    st.stop()

# x = asking price, y = GB-predicted fair price. Points above the y=x line
# are listings the model thinks are under-priced (the deals). Color = flip
# score (composite of undervaluation × 9 opportunity multipliers); size =
# comparable-sample count (more comps → tighter prediction → larger dot).
size_max = max(int(deals["sample_size"].max()), 1) if "sample_size" in deals.columns else 1
fig = px.scatter(
    deals,
    x="price_eur",
    y="predicted_price",
    color="flip_score",
    size="sample_size" if "sample_size" in deals.columns else None,
    size_max=22,
    hover_data={
        "brand": True,
        "model": True,
        "year": ":.0f",
        "mileage_km": ":,",
        "adjusted_undervaluation_pct": ":.1f",
        "flip_score": ":.1f",
        "url": True,
        "price_eur": False,
        "predicted_price": False,
        "sample_size": False,
    },
    color_continuous_scale="RdYlGn",
    labels={
        "price_eur": "Asking price (€)",
        "predicted_price": "Predicted fair price (€)",
        "flip_score": "Flip score",
        "year": "Year",
        "mileage_km": "Mileage (km)",
        "adjusted_undervaluation_pct": "Undervaluation %",
    },
)
# y = x reference line — anything above it is undervalued per the GB model.
_axis_max = max(
    deals["price_eur"].max() if "price_eur" in deals.columns else 0,
    deals["predicted_price"].max() if "predicted_price" in deals.columns else 0,
)
fig.add_shape(
    type="line", x0=0, y0=0, x1=_axis_max, y1=_axis_max,
    line=dict(color="rgba(255,255,255,0.3)", dash="dash", width=1),
)
fig.update_layout(
    height=620,
    margin=dict(l=20, r=20, t=10, b=10),
)
st.plotly_chart(fig, use_container_width=True)
