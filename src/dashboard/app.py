"""OLX Car Parser — Streamlit Dashboard."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from src.dashboard.demo_data import (
    generate_listings,
    generate_price_history,
    generate_buy_signals,
    BRANDS_MODELS,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="OLX Car Analytics",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Load data (cached)
# ---------------------------------------------------------------------------

@st.cache_data
def load_data():
    listings = generate_listings(n=600)
    history = generate_price_history(days=90)
    signals = generate_buy_signals(listings, history)
    return listings, history, signals


listings_df, history_df, signals_df = load_data()

# ---------------------------------------------------------------------------
# Sidebar — filters
# ---------------------------------------------------------------------------
st.sidebar.title("Filters")

selected_brands = st.sidebar.multiselect(
    "Brand",
    options=sorted(BRANDS_MODELS.keys()),
    default=[],
    placeholder="All brands",
)

if selected_brands:
    available_models = []
    for b in selected_brands:
        available_models.extend(BRANDS_MODELS[b])
else:
    available_models = [m for models in BRANDS_MODELS.values() for m in models]

selected_models = st.sidebar.multiselect(
    "Model",
    options=sorted(set(available_models)),
    default=[],
    placeholder="All models",
)

year_range = st.sidebar.slider(
    "Year range",
    min_value=2012,
    max_value=2024,
    value=(2015, 2023),
)

price_range = st.sidebar.slider(
    "Price (USD)",
    min_value=0,
    max_value=40000,
    value=(0, 40000),
    step=500,
)

only_private = st.sidebar.checkbox("Private sellers only", value=False)

# ---------------------------------------------------------------------------
# Apply filters
# ---------------------------------------------------------------------------

def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df.copy()
    if selected_brands:
        filtered = filtered[filtered["brand"].isin(selected_brands)]
    if selected_models:
        filtered = filtered[filtered["model"].isin(selected_models)]
    if "year" in filtered.columns:
        filtered = filtered[
            (filtered["year"] >= year_range[0]) & (filtered["year"] <= year_range[1])
        ]
    if "price_usd" in filtered.columns:
        filtered = filtered[
            (filtered["price_usd"] >= price_range[0])
            & (filtered["price_usd"] <= price_range[1])
        ]
    if only_private and "seller_type" in filtered.columns:
        filtered = filtered[filtered["seller_type"] == "private"]
    return filtered


filtered_listings = apply_filters(listings_df)
filtered_signals = apply_filters(signals_df)


def filter_history(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df.copy()
    if selected_brands:
        filtered = filtered[filtered["brand"].isin(selected_brands)]
    if selected_models:
        filtered = filtered[filtered["model"].isin(selected_models)]
    return filtered


filtered_history = filter_history(history_df)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("OLX Car Market Analytics")
st.caption("Track prices, spot trends, buy smart")

# ---------------------------------------------------------------------------
# KPI cards
# ---------------------------------------------------------------------------
active = filtered_listings[filtered_listings["is_active"]] if "is_active" in filtered_listings.columns else filtered_listings

col1, col2, col3, col4 = st.columns(4)
col1.metric("Active Listings", f"{len(active):,}")
col2.metric("Median Price", f"${int(active['price_usd'].median()):,}" if len(active) else "—")
col3.metric("Buy Signals", f"{len(filtered_signals)}")

if not filtered_history.empty:
    recent = filtered_history[filtered_history["date"] == filtered_history["date"].max()]
    prev = filtered_history[
        filtered_history["date"]
        == (pd.to_datetime(filtered_history["date"]).max() - pd.Timedelta(days=30)).date()
    ]
    if not recent.empty and not prev.empty:
        change = (
            (recent["median_price_usd"].mean() - prev["median_price_usd"].mean())
            / prev["median_price_usd"].mean()
            * 100
        )
        col4.metric("30d Trend", f"{change:+.1f}%", delta=f"{change:+.1f}%")
    else:
        col4.metric("30d Trend", "—")
else:
    col4.metric("30d Trend", "—")

st.divider()

# ---------------------------------------------------------------------------
# Tab layout
# ---------------------------------------------------------------------------
tab_trends, tab_signals, tab_listings, tab_compare = st.tabs(
    ["Price Trends", "Buy Signals", "Listings", "Compare Models"]
)

# ---- TAB 1: Price Trends -------------------------------------------------
with tab_trends:
    st.subheader("Price Trends Over Time")

    if filtered_history.empty:
        st.info("Select brands/models to see trends.")
    else:
        filtered_history["label"] = filtered_history["brand"] + " " + filtered_history["model"]

        fig = px.line(
            filtered_history,
            x="date",
            y="median_price_usd",
            color="label",
            labels={"median_price_usd": "Median Price (USD)", "date": "Date", "label": "Model"},
            height=500,
        )
        fig.update_layout(
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=-0.3),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Min/Max band for selected model
        st.subheader("Price Range Band")
        if selected_models and len(selected_models) == 1:
            model_hist = filtered_history[filtered_history["model"] == selected_models[0]]
            fig_band = go.Figure()
            fig_band.add_trace(go.Scatter(
                x=model_hist["date"], y=model_hist["max_price_usd"],
                mode="lines", line=dict(width=0), name="Max",
            ))
            fig_band.add_trace(go.Scatter(
                x=model_hist["date"], y=model_hist["min_price_usd"],
                mode="lines", line=dict(width=0), fill="tonexty",
                fillcolor="rgba(68, 68, 255, 0.15)", name="Min",
            ))
            fig_band.add_trace(go.Scatter(
                x=model_hist["date"], y=model_hist["median_price_usd"],
                mode="lines", line=dict(color="#4444ff", width=2), name="Median",
            ))
            fig_band.update_layout(
                height=400,
                yaxis_title="Price (USD)",
                hovermode="x unified",
            )
            st.plotly_chart(fig_band, use_container_width=True)
        else:
            st.caption("Select exactly one model in the sidebar to see the price range band.")

# ---- TAB 2: Buy Signals --------------------------------------------------
with tab_signals:
    st.subheader("Buy Opportunities")
    st.caption("Listings priced 15%+ below market median — potential deals")

    if filtered_signals.empty:
        st.info("No buy signals with current filters.")
    else:
        # Highlight top deals
        top = filtered_signals.head(20).copy()
        top["deal_score"] = top["discount_pct"].apply(
            lambda d: "🔥🔥🔥" if d > 25 else ("🔥🔥" if d > 20 else "🔥")
        )
        top = top.rename(columns={
            "brand": "Brand",
            "model": "Model",
            "year": "Year",
            "price_usd": "Price ($)",
            "median_price_usd": "Median ($)",
            "discount_pct": "Discount %",
            "city": "City",
            "mileage_km": "Mileage",
            "deal_score": "Deal",
        })
        st.dataframe(
            top[["Deal", "Brand", "Model", "Year", "Price ($)", "Median ($)",
                 "Discount %", "City", "Mileage"]],
            use_container_width=True,
            hide_index=True,
        )

        # Distribution of discounts
        fig_disc = px.histogram(
            filtered_signals,
            x="discount_pct",
            nbins=20,
            labels={"discount_pct": "Discount below median (%)"},
            title="Discount Distribution",
            height=350,
        )
        st.plotly_chart(fig_disc, use_container_width=True)

# ---- TAB 3: Listings Table -----------------------------------------------
with tab_listings:
    st.subheader("All Listings")

    display_cols = [
        "brand", "model", "year", "price_usd", "mileage_km",
        "engine_type", "transmission", "city", "seller_type", "is_active",
    ]
    st.dataframe(
        filtered_listings[display_cols].sort_values("price_usd"),
        use_container_width=True,
        hide_index=True,
        column_config={
            "price_usd": st.column_config.NumberColumn("Price ($)", format="$%d"),
            "mileage_km": st.column_config.NumberColumn("Mileage", format="%d km"),
            "is_active": st.column_config.CheckboxColumn("Active"),
        },
    )

    st.caption(f"Showing {len(filtered_listings)} listings")

# ---- TAB 4: Compare Models -----------------------------------------------
with tab_compare:
    st.subheader("Model Comparison")

    if filtered_listings.empty:
        st.info("No data for comparison with current filters.")
    else:
        comparison = (
            active.groupby(["brand", "model"])
            .agg(
                count=("price_usd", "size"),
                median_price=("price_usd", "median"),
                min_price=("price_usd", "min"),
                max_price=("price_usd", "max"),
                avg_mileage=("mileage_km", "mean"),
            )
            .reset_index()
            .sort_values("median_price")
        )
        comparison["label"] = comparison["brand"] + " " + comparison["model"]

        col_left, col_right = st.columns(2)

        with col_left:
            fig_box = px.box(
                active,
                x=active["brand"] + " " + active["model"],
                y="price_usd",
                labels={"x": "Model", "price_usd": "Price (USD)"},
                title="Price Distribution by Model",
                height=500,
            )
            fig_box.update_xaxes(tickangle=45)
            st.plotly_chart(fig_box, use_container_width=True)

        with col_right:
            fig_scatter = px.scatter(
                active,
                x="mileage_km",
                y="price_usd",
                color=active["brand"] + " " + active["model"],
                labels={
                    "mileage_km": "Mileage (km)",
                    "price_usd": "Price (USD)",
                    "color": "Model",
                },
                title="Price vs Mileage",
                height=500,
                opacity=0.7,
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        # Stats table
        st.dataframe(
            comparison[["label", "count", "median_price", "min_price", "max_price", "avg_mileage"]]
            .rename(columns={
                "label": "Model",
                "count": "Listings",
                "median_price": "Median ($)",
                "min_price": "Min ($)",
                "max_price": "Max ($)",
                "avg_mileage": "Avg Mileage",
            }),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Median ($)": st.column_config.NumberColumn(format="$%d"),
                "Min ($)": st.column_config.NumberColumn(format="$%d"),
                "Max ($)": st.column_config.NumberColumn(format="$%d"),
                "Avg Mileage": st.column_config.NumberColumn(format="%,.0f km"),
            },
        )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption("OLX Car Parser v0.1 — Demo data. Connect real scraper for live analytics.")
