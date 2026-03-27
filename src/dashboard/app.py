"""OLX.pt Car Parser — Streamlit Dashboard."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from data_loader import load_all

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="OLX.pt Car Analytics",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def load_data():
    return load_all()


listings_df, history_df, signals_df, brands_models = load_data()

# ---------------------------------------------------------------------------
# Sidebar — filters
# ---------------------------------------------------------------------------
st.sidebar.title("Filters")

if listings_df.empty:
    st.sidebar.warning("No data. Run `python -m src.cli scrape` to collect listings.")
else:
    st.sidebar.success(f"{len(listings_df)} listings loaded")
    if st.sidebar.button("Refresh data"):
        st.cache_data.clear()
        st.rerun()

selected_brands = st.sidebar.multiselect(
    "Brand",
    options=sorted(brands_models.keys()),
    default=[],
    placeholder="All brands",
)

if selected_brands:
    available_models = []
    for b in selected_brands:
        available_models.extend(brands_models.get(b, []))
else:
    available_models = [m for models in brands_models.values() for m in models]

selected_models = st.sidebar.multiselect(
    "Model",
    options=sorted(set(available_models)),
    default=[],
    placeholder="All models",
)

# District filter
all_districts = sorted(listings_df["district"].dropna().unique()) if "district" in listings_df.columns else []
selected_districts = st.sidebar.multiselect(
    "District",
    options=all_districts,
    default=[],
    placeholder="All districts",
)

# City filter (filtered by selected districts)
if selected_districts and "city" in listings_df.columns:
    available_cities = sorted(
        listings_df[listings_df["district"].isin(selected_districts)]["city"].dropna().unique()
    )
else:
    available_cities = sorted(listings_df["city"].dropna().unique()) if "city" in listings_df.columns else []

selected_cities = st.sidebar.multiselect(
    "City",
    options=available_cities,
    default=[],
    placeholder="All cities",
)

year_min = int(listings_df["year"].min()) if "year" in listings_df.columns and listings_df["year"].notna().any() else 2010
year_max = int(listings_df["year"].max()) if "year" in listings_df.columns and listings_df["year"].notna().any() else 2024

year_range = st.sidebar.slider("Year range", min_value=year_min, max_value=year_max, value=(year_min, year_max))

price_max_val = int(listings_df["price_eur"].max()) + 1000 if "price_eur" in listings_df.columns and listings_df["price_eur"].notna().any() else 50000
price_range = st.sidebar.slider("Price (EUR)", min_value=0, max_value=price_max_val, value=(0, price_max_val), step=500)

only_private = st.sidebar.checkbox("Particular only", value=False)

# ---------------------------------------------------------------------------
# Apply filters
# ---------------------------------------------------------------------------

def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    f = df.copy()
    if selected_brands:
        f = f[f["brand"].isin(selected_brands)]
    if selected_models:
        f = f[f["model"].isin(selected_models)]
    if selected_districts and "district" in f.columns:
        f = f[f["district"].isin(selected_districts)]
    if selected_cities and "city" in f.columns:
        f = f[f["city"].isin(selected_cities)]
    if "year" in f.columns:
        f = f[f["year"].notna() & (f["year"] >= year_range[0]) & (f["year"] <= year_range[1])]
    if "price_eur" in f.columns:
        f = f[f["price_eur"].notna() & (f["price_eur"] >= price_range[0]) & (f["price_eur"] <= price_range[1])]
    if only_private and "seller_type" in f.columns:
        f = f[f["seller_type"] == "Particular"]
    return f


filtered_listings = apply_filters(listings_df)
filtered_signals = apply_filters(signals_df)


def filter_history(df: pd.DataFrame) -> pd.DataFrame:
    f = df.copy()
    if selected_brands:
        f = f[f["brand"].isin(selected_brands)]
    if selected_models:
        f = f[f["model"].isin(selected_models)]
    return f


filtered_history = filter_history(history_df)

# ---------------------------------------------------------------------------
# Header + KPIs
# ---------------------------------------------------------------------------
st.title("OLX.pt Car Market Analytics")
st.caption("Portugal — track prices, spot trends, buy smart")

active = filtered_listings[filtered_listings["is_active"]] if "is_active" in filtered_listings.columns else filtered_listings

col1, col2, col3, col4 = st.columns(4)
col1.metric("Active Listings", f"{len(active):,}")
col2.metric("Median Price", f"{int(active['price_eur'].median()):,} EUR" if len(active) and active["price_eur"].notna().any() else "—")
col3.metric("Buy Signals", f"{len(filtered_signals)}")

if not filtered_history.empty:
    recent = filtered_history[filtered_history["date"] == filtered_history["date"].max()]
    prev = filtered_history[
        filtered_history["date"] == (pd.to_datetime(filtered_history["date"]).max() - pd.Timedelta(days=30)).date()
    ]
    if not recent.empty and not prev.empty:
        change = (recent["median_price_eur"].mean() - prev["median_price_eur"].mean()) / prev["median_price_eur"].mean() * 100
        col4.metric("30d Trend", f"{change:+.1f}%", delta=f"{change:+.1f}%")
    else:
        col4.metric("30d Trend", "—")
else:
    col4.metric("30d Trend", "—")

st.divider()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_trends, tab_signals, tab_listings, tab_compare, tab_geo = st.tabs(
    ["Price Trends", "Buy Signals", "Listings", "Compare Models", "Geography"]
)

# ---- TAB 1: Price Trends -------------------------------------------------
with tab_trends:
    st.subheader("Price Trends Over Time")
    if filtered_history.empty:
        st.info("No price history yet. Run scraper multiple times to build up trend data.")
    else:
        fh = filtered_history.copy()
        fh["label"] = fh["brand"] + " " + fh["model"]
        fig = px.line(fh, x="date", y="median_price_eur", color="label",
                      labels={"median_price_eur": "Median Price (EUR)", "date": "Date", "label": "Model"},
                      height=500)
        fig.update_layout(hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=-0.3))
        st.plotly_chart(fig, width="stretch")

        st.subheader("Price Range Band")
        if selected_models and len(selected_models) == 1:
            mh = fh[fh["model"] == selected_models[0]]
            fig_band = go.Figure()
            fig_band.add_trace(go.Scatter(x=mh["date"], y=mh["max_price_eur"], mode="lines", line=dict(width=0), name="Max"))
            fig_band.add_trace(go.Scatter(x=mh["date"], y=mh["min_price_eur"], mode="lines", line=dict(width=0),
                                          fill="tonexty", fillcolor="rgba(68,68,255,0.15)", name="Min"))
            fig_band.add_trace(go.Scatter(x=mh["date"], y=mh["median_price_eur"], mode="lines",
                                          line=dict(color="#4444ff", width=2), name="Median"))
            fig_band.update_layout(height=400, yaxis_title="Price (EUR)", hovermode="x unified")
            st.plotly_chart(fig_band, width="stretch")
        else:
            st.caption("Select exactly one model to see the price range band.")

# ---- TAB 2: Buy Signals --------------------------------------------------
with tab_signals:
    st.subheader("Buy Opportunities")
    st.caption("Listings priced 15%+ below market median")
    if filtered_signals.empty:
        st.info("No buy signals with current filters.")
    else:
        top = filtered_signals.head(20).copy()
        top["deal"] = top["discount_pct"].apply(lambda d: "🔥🔥🔥" if d > 25 else ("🔥🔥" if d > 20 else "🔥"))
        signal_cols = ["deal", "brand", "model", "year", "price_eur", "median_price_eur", "discount_pct", "city", "district", "mileage_km", "fuel_type"]
        if "url" in top.columns:
            signal_cols.append("url")
        st.dataframe(
            top[signal_cols]
            .rename(columns={
                "deal": "Deal", "brand": "Brand", "model": "Model", "year": "Year",
                "price_eur": "Price (EUR)", "median_price_eur": "Median (EUR)",
                "discount_pct": "Discount %", "city": "City", "district": "District",
                "mileage_km": "Mileage", "fuel_type": "Fuel", "url": "Link",
            }),
            width="stretch", hide_index=True,
            column_config={
                "Link": st.column_config.LinkColumn("Link", display_text="Open"),
            },
        )
        fig_disc = px.histogram(filtered_signals, x="discount_pct", nbins=20,
                                labels={"discount_pct": "Discount below median (%)"}, title="Discount Distribution", height=350)
        st.plotly_chart(fig_disc, width="stretch")

# ---- TAB 3: Listings Table -----------------------------------------------
with tab_listings:
    st.subheader("All Listings")
    display_cols = [c for c in [
        "brand", "model", "year", "price_eur", "mileage_km", "engine_cc",
        "fuel_type", "horsepower", "transmission", "segment",
        "city", "district", "seller_type", "is_active", "url",
    ] if c in filtered_listings.columns]

    st.dataframe(
        filtered_listings[display_cols].sort_values("price_eur") if "price_eur" in display_cols else filtered_listings[display_cols],
        width="stretch", hide_index=True,
        column_config={
            "price_eur": st.column_config.NumberColumn("Price (EUR)", format="%d EUR"),
            "mileage_km": st.column_config.NumberColumn("Mileage", format="%d km"),
            "engine_cc": st.column_config.NumberColumn("Engine (cc)", format="%d"),
            "horsepower": st.column_config.NumberColumn("HP", format="%d"),
            "is_active": st.column_config.CheckboxColumn("Active"),
            "url": st.column_config.LinkColumn("Link", display_text="Open"),
        },
    )
    st.caption(f"Showing {len(filtered_listings)} listings")

# ---- TAB 4: Compare Models -----------------------------------------------
with tab_compare:
    st.subheader("Model Comparison")
    if filtered_listings.empty or not active["price_eur"].notna().any():
        st.info("No data for comparison with current filters.")
    else:
        comparison = (
            active[active["price_eur"].notna()]
            .groupby(["brand", "model"])
            .agg(count=("price_eur", "size"), median_price=("price_eur", "median"),
                 min_price=("price_eur", "min"), max_price=("price_eur", "max"),
                 avg_mileage=("mileage_km", "mean"))
            .reset_index().sort_values("median_price")
        )
        comparison["label"] = comparison["brand"] + " " + comparison["model"]

        col_left, col_right = st.columns(2)
        with col_left:
            pd_box = active[active["price_eur"].notna()].copy()
            pd_box["label"] = pd_box["brand"] + " " + pd_box["model"]
            fig_box = px.box(pd_box, x="label", y="price_eur",
                             labels={"label": "Model", "price_eur": "Price (EUR)"},
                             title="Price Distribution", height=500)
            fig_box.update_xaxes(tickangle=45)
            st.plotly_chart(fig_box, width="stretch")

        with col_right:
            sd = active[active["price_eur"].notna() & active["mileage_km"].notna()].copy()
            if not sd.empty:
                sd["label"] = sd["brand"] + " " + sd["model"]
                fig_sc = px.scatter(sd, x="mileage_km", y="price_eur", color="label",
                                    labels={"mileage_km": "Mileage (km)", "price_eur": "Price (EUR)", "label": "Model"},
                                    title="Price vs Mileage", height=500, opacity=0.7)
                st.plotly_chart(fig_sc, width="stretch")

        st.dataframe(
            comparison[["label", "count", "median_price", "min_price", "max_price", "avg_mileage"]]
            .rename(columns={"label": "Model", "count": "Listings", "median_price": "Median (EUR)",
                             "min_price": "Min (EUR)", "max_price": "Max (EUR)", "avg_mileage": "Avg Mileage"}),
            width="stretch", hide_index=True,
            column_config={
                "Median (EUR)": st.column_config.NumberColumn(format="%d EUR"),
                "Min (EUR)": st.column_config.NumberColumn(format="%d EUR"),
                "Max (EUR)": st.column_config.NumberColumn(format="%d EUR"),
                "Avg Mileage": st.column_config.NumberColumn(format="%,.0f km"),
            },
        )

# ---- TAB 5: Geography ----------------------------------------------------
with tab_geo:
    st.subheader("Listings by Location")
    if active.empty or "district" not in active.columns:
        st.info("No location data available.")
    else:
        col_g1, col_g2 = st.columns(2)

        with col_g1:
            district_counts = active["district"].value_counts().reset_index()
            district_counts.columns = ["District", "Count"]
            fig_bar = px.bar(district_counts.head(15), x="Count", y="District", orientation="h",
                             title="Listings by District", height=450)
            fig_bar.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_bar, width="stretch")

        with col_g2:
            if active["price_eur"].notna().any():
                district_prices = (
                    active[active["price_eur"].notna()]
                    .groupby("district")["price_eur"]
                    .median().reset_index()
                    .rename(columns={"price_eur": "Median Price (EUR)"})
                    .sort_values("Median Price (EUR)", ascending=False)
                )
                fig_price = px.bar(district_prices.head(15), x="Median Price (EUR)", y="district",
                                   orientation="h", title="Median Price by District", height=450)
                fig_price.update_layout(yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_price, width="stretch")

        # City-level table
        if "city" in active.columns and active["city"].notna().any():
            city_stats = (
                active[active["price_eur"].notna()]
                .groupby(["district", "city"])
                .agg(count=("price_eur", "size"), median=("price_eur", "median"))
                .reset_index()
                .sort_values("count", ascending=False)
            )
            st.dataframe(
                city_stats.head(30).rename(columns={"district": "District", "city": "City",
                                                     "count": "Listings", "median": "Median (EUR)"}),
                width="stretch", hide_index=True,
                column_config={"Median (EUR)": st.column_config.NumberColumn(format="%d EUR")},
            )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(f"OLX.pt Car Parser v0.1 — {len(listings_df)} listings")
