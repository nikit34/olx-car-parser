"""OLX.pt Car Parser — Streamlit Dashboard (Deal Finder)."""

from datetime import date

import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))

from data_loader import load_all, load_portfolio


def plotly_chart_with_click(fig, df, key, **kwargs):
    """Render a Plotly chart; if a point with a URL is clicked, show a link button."""
    event = st.plotly_chart(fig, on_select="rerun", key=key, **kwargs)
    if event and event.selection and event.selection.points:
        pt = event.selection.points[0]
        url = None
        # Try customdata first (px.scatter puts hover_data there as a list)
        cd = pt.get("customdata")
        if cd:
            candidates = cd if isinstance(cd, (list, tuple)) else [cd]
            for v in reversed(candidates):
                if isinstance(v, str) and v.startswith("http"):
                    url = v
                    break
        if url:
            st.link_button("Open listing on OLX", url)


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="OLX Car Deals",
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


listings_df, history_df, signals_df, brands_models, turnover_df, _portfolio_init, unmatched_df = load_data()

# Portfolio loaded separately (not cached — it's mutable)
def get_portfolio():
    return load_portfolio()

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

all_districts = sorted(listings_df["district"].dropna().unique()) if "district" in listings_df.columns else []
selected_districts = st.sidebar.multiselect(
    "District", options=all_districts, default=[], placeholder="All districts",
)

if selected_districts and "city" in listings_df.columns:
    available_cities = sorted(
        listings_df[listings_df["district"].isin(selected_districts)]["city"].dropna().unique()
    )
else:
    available_cities = sorted(listings_df["city"].dropna().unique()) if "city" in listings_df.columns else []

selected_cities = st.sidebar.multiselect(
    "City", options=available_cities, default=[], placeholder="All cities",
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

active = filtered_listings[filtered_listings["is_active"]] if "is_active" in filtered_listings.columns else filtered_listings


def filter_history(df: pd.DataFrame) -> pd.DataFrame:
    f = df.copy()
    if selected_brands:
        f = f[f["brand"].isin(selected_brands)]
    if selected_models:
        f = f[f["model"].isin(selected_models)]
    return f


filtered_history = filter_history(history_df)

# ---------------------------------------------------------------------------
# Header + KPIs (deal-focused)
# ---------------------------------------------------------------------------
st.title("OLX Car Deals")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Active Listings", f"{len(active):,}")
col2.metric("Deals Found", f"{len(filtered_signals)}")

if not filtered_signals.empty:
    avg_discount = filtered_signals["undervaluation_pct"].mean()
    col3.metric("Avg Discount", f"{avg_discount:.0f}%")
    best = filtered_signals.iloc[0]
    best_profit = int(best["predicted_price"] - best["price_eur"])
    col4.metric("Best Deal Profit", f"{best_profit:+,} EUR")
else:
    col3.metric("Avg Discount", "—")
    col4.metric("Best Deal Profit", "—")

st.divider()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_deals, tab_analytics, tab_trends, tab_listings, tab_compare, tab_geo, tab_portfolio, tab_unmatched = st.tabs([
    "Deals", "Analytics", "Price Trends", "Listings", "Compare Models", "Geography", "Portfolio", "Unmatched",
])

# ---- TAB 1: Deals (Buy Signals) --------------------------------------------
with tab_deals:
    st.subheader("Underpriced Cars")
    st.caption("Sorted by flip score = undervaluation % x liquidity x trend. Profit = fair price - asking price.")

    if filtered_signals.empty:
        st.info("No deals found with current filters. Try broadening your search.")
    else:
        deals = filtered_signals.copy()

        # Add estimated profit and ROI columns
        deals["est_profit_eur"] = (deals["predicted_price"] - deals["price_eur"]).round(0).astype(int)
        deals["est_roi_pct"] = ((deals["predicted_price"] - deals["price_eur"]) / deals["price_eur"] * 100).round(1)

        # Profit per day — capital efficiency metric
        if "avg_days_to_sell" in deals.columns:
            deals["profit_per_day"] = (
                deals["est_profit_eur"] / deals["avg_days_to_sell"].replace(0, np.nan)
            ).round(1)
        else:
            deals["profit_per_day"] = np.nan

        # Price drop velocity — how fast the seller is dropping price (EUR/day)
        if "price_change_eur" in deals.columns and "days_listed" in deals.columns:
            deals["price_drop_per_day"] = (
                deals["price_change_eur"] / deals["days_listed"].replace(0, np.nan)
            ).round(1)
        else:
            deals["price_drop_per_day"] = np.nan

        # Deal rating
        deals["deal"] = deals["flip_score"].apply(
            lambda s: "🔥🔥🔥" if s > 30 else ("🔥🔥" if s > 20 else "🔥")
        )
        if "sample_size" in deals.columns:
            deals["confidence"] = deals["sample_size"].apply(
                lambda n: "High" if n >= 10 else ("Medium" if n >= 5 else "Low")
            )

        # --- Top deals cards ---
        top3 = deals.head(3)
        cols = st.columns(3)
        for i, (_, deal) in enumerate(top3.iterrows()):
            with cols[i]:
                profit = int(deal["est_profit_eur"])
                st.markdown(f"### {deal['brand']} {deal['model']} {int(deal['year']) if pd.notna(deal['year']) else '?'}")
                st.markdown(f"**{int(deal['price_eur']):,} EUR** → fair price **{int(deal['predicted_price']):,} EUR**")
                profit_day = deal.get("profit_per_day")
                profit_day_str = f" · **{profit_day:.0f} EUR/day**" if pd.notna(profit_day) else ""
                st.markdown(f"Profit: **{profit:+,} EUR** ({deal['est_roi_pct']:+.0f}% ROI){profit_day_str}")
                details = []
                if pd.notna(deal.get("mileage_km")):
                    details.append(f"{int(deal['mileage_km']):,} km")
                if deal.get("fuel_type"):
                    details.append(deal["fuel_type"])
                if deal.get("district"):
                    details.append(deal["district"])
                drop_day = deal.get("price_drop_per_day")
                if pd.notna(drop_day) and drop_day < 0:
                    details.append(f"seller dropping {drop_day:.0f} EUR/day")
                if details:
                    st.caption(" · ".join(details))
                if deal.get("url"):
                    st.markdown(f"[Open on OLX]({deal['url']})")

        st.divider()

        # --- Full deals table ---
        signal_cols = ["deal", "brand", "model", "generation", "year",
                       "price_eur", "predicted_price", "est_profit_eur", "est_roi_pct",
                       "profit_per_day", "undervaluation_pct", "flip_score"]
        if "avg_days_to_sell" in deals.columns:
            signal_cols.append("avg_days_to_sell")
        if "sample_size" in deals.columns:
            signal_cols += ["sample_size", "confidence"]
        signal_cols += ["mileage_km", "fuel_type", "district", "city"]
        if "days_listed" in deals.columns:
            signal_cols.append("days_listed")
        if "price_change_eur" in deals.columns:
            signal_cols.append("price_change_eur")
        signal_cols.append("price_drop_per_day")
        if "url" in deals.columns:
            signal_cols.append("url")

        avail_cols = [c for c in signal_cols if c in deals.columns]

        st.dataframe(
            deals[avail_cols].rename(columns={
                "deal": "Deal", "brand": "Brand", "model": "Model",
                "generation": "Gen", "year": "Year",
                "price_eur": "Price", "predicted_price": "Fair Price",
                "est_profit_eur": "Profit", "est_roi_pct": "ROI %",
                "profit_per_day": "EUR/day",
                "undervaluation_pct": "Below Market %", "flip_score": "Score",
                "avg_days_to_sell": "Days to Sell", "sample_size": "Sample",
                "confidence": "Conf", "mileage_km": "Mileage",
                "fuel_type": "Fuel", "district": "District", "city": "City",
                "days_listed": "Listed", "price_change_eur": "Price Drop",
                "price_drop_per_day": "Drop/day",
                "url": "Link",
            }),
            width="stretch", hide_index=True,
            column_config={
                "Price": st.column_config.NumberColumn(format="%d EUR"),
                "Fair Price": st.column_config.NumberColumn(format="%d EUR"),
                "Profit": st.column_config.NumberColumn(format="%+d EUR"),
                "ROI %": st.column_config.NumberColumn(format="%+.1f%%"),
                "EUR/day": st.column_config.NumberColumn(format="%.0f"),
                "Below Market %": st.column_config.NumberColumn(format="%.0f%%"),
                "Score": st.column_config.NumberColumn(format="%.0f"),
                "Days to Sell": st.column_config.NumberColumn(format="%.0f"),
                "Mileage": st.column_config.NumberColumn(format="%,d km"),
                "Listed": st.column_config.NumberColumn(format="%d d"),
                "Price Drop": st.column_config.NumberColumn(format="%+d EUR"),
                "Drop/day": st.column_config.NumberColumn(format="%+.0f EUR"),
                "Link": st.column_config.LinkColumn("Link", display_text="Open"),
            },
        )
        st.caption(f"{len(deals)} deals found")

        # --- Profit distribution chart ---
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            fig_profit = px.histogram(
                deals, x="est_profit_eur", nbins=20,
                labels={"est_profit_eur": "Estimated Profit (EUR)"},
                title="Profit Distribution", height=350,
            )
            fig_profit.update_layout(showlegend=False)
            st.plotly_chart(fig_profit, width="stretch")

        with col_chart2:
            # Deals by brand — where are the opportunities?
            brand_deals = deals.groupby("brand").agg(
                count=("brand", "size"),
                avg_profit=("est_profit_eur", "mean"),
            ).reset_index().sort_values("count", ascending=False).head(10)

            fig_brands = px.bar(
                brand_deals, x="count", y="brand", orientation="h",
                color="avg_profit", color_continuous_scale="Greens",
                labels={"count": "Deals", "brand": "Brand", "avg_profit": "Avg Profit"},
                title="Deals by Brand", height=350,
            )
            fig_brands.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_brands, width="stretch")

        # --- Price vs Fair Price scatter ---
        deals["_scatter_size"] = deals["flip_score"].clip(lower=1)
        fig_scatter = px.scatter(
            deals, x="predicted_price", y="price_eur",
            color="est_profit_eur", size="_scatter_size",
            color_continuous_scale="RdYlGn_r",
            hover_data=["brand", "model", "year", "mileage_km", "url"],
            labels={
                "predicted_price": "Fair Price (EUR)",
                "price_eur": "Asking Price (EUR)",
                "est_profit_eur": "Profit (EUR)",
                "flip_score": "Score",
            },
            title="Asking Price vs Fair Price (below diagonal = deal)",
            height=500,
        )
        # Add diagonal line (price = fair price)
        max_price = max(deals["predicted_price"].max(), deals["price_eur"].max())
        fig_scatter.add_trace(go.Scatter(
            x=[0, max_price], y=[0, max_price],
            mode="lines", line=dict(dash="dash", color="gray", width=1),
            name="Break Even", showlegend=False,
        ))
        plotly_chart_with_click(fig_scatter, deals, key="deals_scatter", width="stretch")

        # --- Motivated sellers: biggest price drops ---
        if "price_drop_per_day" in deals.columns and deals["price_drop_per_day"].notna().any():
            st.subheader("Motivated Sellers")
            st.caption("Sellers actively dropping prices — more negotiation room")
            motivated = deals[deals["price_drop_per_day"].notna() & (deals["price_drop_per_day"] < 0)].copy()
            motivated = motivated.sort_values("price_drop_per_day")
            if not motivated.empty:
                mot_cols = ["brand", "model", "year", "price_eur", "predicted_price",
                            "est_profit_eur", "days_listed", "price_change_eur", "price_drop_per_day"]
                if "url" in motivated.columns:
                    mot_cols.append("url")
                avail_mot = [c for c in mot_cols if c in motivated.columns]
                st.dataframe(
                    motivated.head(15)[avail_mot].rename(columns={
                        "brand": "Brand", "model": "Model", "year": "Year",
                        "price_eur": "Price", "predicted_price": "Fair Price",
                        "est_profit_eur": "Profit", "days_listed": "Listed",
                        "price_change_eur": "Total Drop", "price_drop_per_day": "Drop/day",
                        "url": "Link",
                    }),
                    hide_index=True,
                    column_config={
                        "Price": st.column_config.NumberColumn(format="%d EUR"),
                        "Fair Price": st.column_config.NumberColumn(format="%d EUR"),
                        "Profit": st.column_config.NumberColumn(format="%+d EUR"),
                        "Listed": st.column_config.NumberColumn(format="%d d"),
                        "Total Drop": st.column_config.NumberColumn(format="%+d EUR"),
                        "Drop/day": st.column_config.NumberColumn(format="%+.0f EUR"),
                        "Link": st.column_config.LinkColumn("Link", display_text="Open"),
                    },
                )

        # --- Days listed vs price drop curve ---
        drop_data = deals[
            deals["days_listed"].notna() & deals["price_change_pct"].notna()
            & (deals["days_listed"] > 0)
        ].copy() if "price_change_pct" in deals.columns else pd.DataFrame()
        if not drop_data.empty and len(drop_data) >= 5:
            st.subheader("When Do Sellers Drop Prices?")
            st.caption("Avg price change % by days on market — shows when to expect discounts")
            drop_data["days_bucket"] = pd.cut(
                drop_data["days_listed"],
                bins=[0, 7, 14, 21, 30, 45, 60, 90, 999],
                labels=["0-7d", "8-14d", "15-21d", "22-30d", "31-45d", "46-60d", "61-90d", "90d+"],
            )
            bucket_stats = (
                drop_data.groupby("days_bucket", observed=True)
                .agg(avg_drop=("price_change_pct", "mean"), count=("price_change_pct", "size"))
                .reset_index()
            )
            bucket_stats = bucket_stats[bucket_stats["count"] >= 2]
            if not bucket_stats.empty:
                colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in bucket_stats["avg_drop"]]
                fig_drop = go.Figure(go.Bar(
                    x=bucket_stats["days_bucket"].astype(str),
                    y=bucket_stats["avg_drop"],
                    marker_color=colors,
                    text=[f"{v:+.1f}%" for v in bucket_stats["avg_drop"]],
                    textposition="auto",
                ))
                fig_drop.update_layout(
                    xaxis_title="Days on Market", yaxis_title="Avg Price Change %",
                    height=350, title="Price Change by Time on Market",
                )
                st.plotly_chart(fig_drop, width="stretch")


# ---- TAB: Analytics — find optimal cars by value combination -----------------
with tab_analytics:
    st.subheader("Price Factor Analytics")
    st.caption("Find cars with the best combination of value factors at the lowest price")

    ana = active[active["price_eur"].notna()].copy()

    if ana.empty:
        st.info("No active listings with prices.")
    else:
        # Prepare columns
        if "year" in ana.columns:
            ana["age"] = date.today().year - ana["year"]
        if "mileage_km" in ana.columns and "price_eur" in ana.columns:
            ana["eur_per_1k_km"] = (ana["price_eur"] / (ana["mileage_km"] / 1000)).replace([np.inf, -np.inf], np.nan)
        ana["label"] = ana["brand"] + " " + ana["model"]
        if "year" in ana.columns:
            ana["label_year"] = ana["label"] + " " + ana["year"].astype(int).astype(str)

        # ---- 1. Year vs Price colored by Transmission ----
        st.subheader("Year vs Price by Transmission")
        st.caption("Automatic costs 2.4x more — find young manual bargains below the trend")
        yr_data = ana[ana["year"].notna()].copy()
        if not yr_data.empty:
            fig_yr = px.scatter(
                yr_data, x="year", y="price_eur",
                color="transmission",
                symbol="transmission",
                hover_data=["brand", "model", "mileage_km", "fuel_type", "district", "url"],
                labels={"year": "Year", "price_eur": "Price (EUR)", "transmission": "Transmission"},
                height=500, opacity=0.75,
                color_discrete_map={"Automática": "#e63946", "Manual": "#457b9d"},
            )
            # trend lines per transmission
            for trans in yr_data["transmission"].dropna().unique():
                sub = yr_data[yr_data["transmission"] == trans].dropna(subset=["year", "price_eur"])
                if len(sub) >= 5:
                    z = np.polyfit(sub["year"], sub["price_eur"], 1)
                    xs = np.linspace(sub["year"].min(), sub["year"].max(), 50)
                    fig_yr.add_trace(go.Scatter(
                        x=xs, y=np.polyval(z, xs), mode="lines",
                        line=dict(dash="dash", width=2),
                        name=f"{trans} trend", showlegend=True,
                    ))
            fig_yr.update_layout(hovermode="closest")
            plotly_chart_with_click(fig_yr, yr_data, key="yr_scatter", use_container_width=True)

        # ---- 2. Mileage vs Price colored by Fuel ----
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Mileage vs Price by Fuel")
            st.caption("Diesel holds value longer at high mileage — look for low-km gasoline deals")
            mil_data = ana[ana["mileage_km"].notna()].copy()
            if not mil_data.empty:
                fig_mil = px.scatter(
                    mil_data, x="mileage_km", y="price_eur",
                    color="fuel_type",
                    hover_data=["brand", "model", "year", "transmission", "url"],
                    labels={"mileage_km": "Mileage (km)", "price_eur": "Price (EUR)", "fuel_type": "Fuel"},
                    height=450, opacity=0.7,
                )
                fig_mil.update_layout(hovermode="closest")
                plotly_chart_with_click(fig_mil, mil_data, key="mil_scatter", use_container_width=True)

        with col_b:
            st.subheader("Import vs National by Brand")
            st.caption("Imports cost more — national cars of premium brands = hidden value")
            origin_data = ana[ana["origin"].notna() & ana["brand"].notna()].copy()
            if not origin_data.empty:
                brand_origin = (
                    origin_data.groupby(["brand", "origin"])["price_eur"]
                    .median().reset_index()
                )
                # Only brands with both origins
                both = brand_origin.groupby("brand").filter(lambda g: len(g) >= 2)
                if not both.empty:
                    fig_orig = px.bar(
                        both.sort_values(["brand", "origin"]),
                        x="brand", y="price_eur", color="origin", barmode="group",
                        labels={"brand": "Brand", "price_eur": "Median Price (EUR)", "origin": "Origin"},
                        color_discrete_map={"Importado": "#e63946", "Nacional": "#2a9d8f"},
                        height=450,
                    )
                    st.plotly_chart(fig_orig, use_container_width=True)

        # ---- 3. Value Score: composite metric ----
        st.subheader("Value Score — Best Cars for the Money")
        st.caption("Score = year (25) + mileage (30) + price (15) + condition (10) + fuel (10) + HP (5) + transmission (5). Higher = better deal.")

        score_data = ana[
            ana["year"].notna() & ana["mileage_km"].notna() & (ana["mileage_km"] > 0)
        ].copy()

        if len(score_data) >= 5:
            # Normalize each factor 0-1 (higher = better)
            score_data["year_norm"] = (score_data["year"] - score_data["year"].min()) / max(score_data["year"].max() - score_data["year"].min(), 1)
            score_data["mileage_norm"] = 1 - (score_data["mileage_km"] - score_data["mileage_km"].min()) / max(score_data["mileage_km"].max() - score_data["mileage_km"].min(), 1)
            score_data["price_norm"] = 1 - (score_data["price_eur"] - score_data["price_eur"].min()) / max(score_data["price_eur"].max() - score_data["price_eur"].min(), 1)
            hp_ok = score_data["horsepower"].notna() & (score_data["horsepower"] > 0)
            if hp_ok.sum() > 5:
                hp_min = score_data.loc[hp_ok, "horsepower"].min()
                hp_max = score_data.loc[hp_ok, "horsepower"].max()
                score_data["hp_norm"] = np.where(
                    hp_ok,
                    (score_data["horsepower"] - hp_min) / max(hp_max - hp_min, 1),
                    0.5,
                )
            else:
                score_data["hp_norm"] = 0.5

            # Fuel type: hybrids retain value best, diesel good, EV risky
            fuel_rank = {"Híbrido": 1.0, "Híbrido Plug-in": 1.0,
                         "Eléctrico": 0.8, "Diesel": 0.7, "Gasolina": 0.5, "GPL": 0.3}
            score_data["fuel_norm"] = score_data["fuel_type"].map(fuel_rank).fillna(0.5)

            # Transmission: automatic commands higher resale premium
            score_data["trans_norm"] = np.where(
                score_data["transmission"] == "Automática", 1.0,
                np.where(score_data["transmission"] == "Manual", 0.5, 0.5),
            )

            # Condition: no accident + no repair = best; penalty for issues
            score_data["condition_norm"] = 1.0
            if "had_accident" in score_data.columns:
                score_data.loc[score_data["had_accident"] == True, "condition_norm"] = 0.2
            if "needs_repair" in score_data.columns:
                score_data.loc[
                    (score_data["needs_repair"] == True) & (score_data["condition_norm"] > 0.2),
                    "condition_norm",
                ] = 0.5

            score_data["value_score"] = (
                score_data["year_norm"] * 25
                + score_data["mileage_norm"] * 30
                + score_data["price_norm"] * 15
                + score_data["hp_norm"] * 5
                + score_data["fuel_norm"] * 10
                + score_data["trans_norm"] * 5
                + score_data["condition_norm"] * 10
            )

            fig_val = px.scatter(
                score_data, x="price_eur", y="value_score",
                color="brand", size="age",
                hover_data=["model", "year", "mileage_km", "transmission", "fuel_type", "horsepower", "district", "url"],
                labels={"price_eur": "Price (EUR)", "value_score": "Value Score", "brand": "Brand", "age": "Age (years)"},
                title="Value Score vs Price — top right = optimal zone",
                height=550, opacity=0.8,
            )
            # Highlight optimal zone (top-left quadrant)
            med_price = score_data["price_eur"].median()
            med_score = score_data["value_score"].median()
            fig_val.add_shape(
                type="rect", x0=0, x1=med_price, y0=med_score, y1=100,
                line=dict(color="green", width=2, dash="dot"),
                fillcolor="rgba(42,157,143,0.08)",
            )
            fig_val.add_annotation(
                x=med_price * 0.3, y=med_score + (100 - med_score) * 0.85,
                text="OPTIMAL ZONE", showarrow=False,
                font=dict(color="green", size=14, family="Arial Black"),
            )
            fig_val.update_layout(hovermode="closest")
            plotly_chart_with_click(fig_val, score_data, key="val_scatter", use_container_width=True)

            # Top 15 best value table
            top_val = score_data.nlargest(15, "value_score")
            val_cols = ["label", "year", "price_eur", "mileage_km", "transmission",
                        "fuel_type", "horsepower", "origin", "district", "value_score"]
            if "url" in top_val.columns:
                val_cols.append("url")
            avail_val = [c for c in val_cols if c in top_val.columns]
            st.dataframe(
                top_val[avail_val].rename(columns={
                    "label": "Car", "year": "Year", "price_eur": "Price (EUR)",
                    "mileage_km": "Mileage", "transmission": "Trans",
                    "fuel_type": "Fuel", "horsepower": "HP",
                    "origin": "Origin", "district": "District",
                    "value_score": "Score", "url": "Link",
                }),
                hide_index=True,
                column_config={
                    "Price (EUR)": st.column_config.NumberColumn(format="%d EUR"),
                    "Mileage": st.column_config.NumberColumn(format="%,d km"),
                    "Score": st.column_config.NumberColumn(format="%.1f"),
                    "Link": st.column_config.LinkColumn("Link", display_text="Open"),
                },
            )

        # ---- 4. Depreciation Heatmap: Brand x Age ----
        st.subheader("Depreciation Heatmap: Brand x Age")
        st.caption("Median price by brand and age bucket — find brands that hold value")
        dep_heat = ana[ana["year"].notna()].copy()
        if not dep_heat.empty:
            dep_heat["age_bucket"] = pd.cut(
                dep_heat["age"], bins=[0, 5, 8, 11, 14, 17, 20, 50],
                labels=["0-5y", "6-8y", "9-11y", "12-14y", "15-17y", "18-20y", "20y+"],
                right=True,
            )
            pivot_dep = dep_heat.pivot_table(
                values="price_eur", index="brand", columns="age_bucket",
                aggfunc="median",
            )
            # Keep brands with >= 3 listings
            brand_counts = dep_heat["brand"].value_counts()
            keep_brands = brand_counts[brand_counts >= 3].index
            pivot_dep = pivot_dep.loc[pivot_dep.index.isin(keep_brands)]
            if not pivot_dep.empty:
                fig_dep_h = px.imshow(
                    pivot_dep.values,
                    x=[str(c) for c in pivot_dep.columns],
                    y=pivot_dep.index.tolist(),
                    labels=dict(color="Median Price (EUR)"),
                    aspect="auto",
                    color_continuous_scale="YlOrRd",
                    height=max(300, len(pivot_dep) * 50),
                )
                fig_dep_h.update_traces(
                    text=[[f"{v:,.0f}" if pd.notna(v) else "" for v in row] for row in pivot_dep.values],
                    texttemplate="%{text}",
                )
                st.plotly_chart(fig_dep_h, use_container_width=True)

        # ---- 5. Transmission + Fuel Price Matrix ----
        col_c, col_d = st.columns(2)

        with col_c:
            st.subheader("Transmission + Fuel: Median Price")
            tf_data = ana[ana["transmission"].notna() & ana["fuel_type"].notna()].copy()
            if not tf_data.empty:
                pivot_tf = tf_data.pivot_table(
                    values="price_eur", index="fuel_type", columns="transmission",
                    aggfunc="median",
                )
                if not pivot_tf.empty:
                    fig_tf = px.imshow(
                        pivot_tf.values,
                        x=pivot_tf.columns.tolist(),
                        y=pivot_tf.index.tolist(),
                        labels=dict(color="Median Price (EUR)"),
                        color_continuous_scale="Viridis",
                        aspect="auto", height=350,
                    )
                    fig_tf.update_traces(
                        text=[[f"{v:,.0f}" if pd.notna(v) else "" for v in row] for row in pivot_tf.values],
                        texttemplate="%{text}",
                    )
                    st.plotly_chart(fig_tf, use_container_width=True)

        with col_d:
            st.subheader("EUR per 1000 km by Brand")
            st.caption("Cost efficiency — lower = more km for your money")
            eur_km = ana[ana["eur_per_1k_km"].notna() & (ana["eur_per_1k_km"] < 1000)].copy()
            if not eur_km.empty:
                brand_eur_km = (
                    eur_km.groupby("brand")
                    .agg(median_eur_km=("eur_per_1k_km", "median"), count=("eur_per_1k_km", "size"))
                    .reset_index()
                )
                brand_eur_km = brand_eur_km[brand_eur_km["count"] >= 3].sort_values("median_eur_km")
                if not brand_eur_km.empty:
                    fig_ekm = px.bar(
                        brand_eur_km, x="median_eur_km", y="brand", orientation="h",
                        color="median_eur_km", color_continuous_scale="RdYlGn_r",
                        labels={"median_eur_km": "EUR / 1000 km", "brand": "Brand"},
                        height=350,
                    )
                    fig_ekm.update_layout(yaxis=dict(autorange="reversed"), showlegend=False)
                    st.plotly_chart(fig_ekm, use_container_width=True)

        # ---- 6. Sweet Spot Finder: interactive filters ----
        st.subheader("Sweet Spot Finder")
        st.caption("Set your constraints — see what fits best")

        ss1, ss2, ss3, ss4 = st.columns(4)
        with ss1:
            max_budget = st.number_input("Max Budget (EUR)", min_value=0, value=15000, step=1000, key="ss_budget")
        with ss2:
            max_age_ss = st.number_input("Max Age (years)", min_value=1, value=15, step=1, key="ss_age")
        with ss3:
            max_mileage_ss = st.number_input("Max Mileage (km)", min_value=10000, value=250000, step=10000, key="ss_mileage")
        with ss4:
            prefer_trans = st.selectbox("Transmission", ["Any", "Automática", "Manual"], key="ss_trans")

        sweet = ana[
            (ana["price_eur"] <= max_budget)
            & (ana["year"].notna())
            & (ana["age"] <= max_age_ss)
            & (ana["mileage_km"].notna())
            & (ana["mileage_km"] <= max_mileage_ss)
        ].copy()
        if prefer_trans != "Any":
            sweet = sweet[sweet["transmission"] == prefer_trans]

        if sweet.empty:
            st.warning("No cars match these criteria. Try relaxing the constraints.")
        else:
            sweet = sweet.sort_values("price_eur")
            st.success(f"{len(sweet)} cars found")

            fig_sweet = px.scatter(
                sweet, x="mileage_km", y="price_eur",
                color="brand", size="age",
                hover_data=["model", "year", "transmission", "fuel_type", "horsepower", "district", "url"],
                labels={"mileage_km": "Mileage (km)", "price_eur": "Price (EUR)", "brand": "Brand", "age": "Age"},
                title=f"Cars under {max_budget:,} EUR, max {max_age_ss}y, max {max_mileage_ss//1000}k km",
                height=500, opacity=0.8,
            )
            fig_sweet.update_layout(hovermode="closest")
            plotly_chart_with_click(fig_sweet, sweet, key="sweet_scatter", use_container_width=True)

            sweet_cols = ["label", "year", "price_eur", "mileage_km", "transmission",
                          "fuel_type", "horsepower", "origin", "district"]
            if "url" in sweet.columns:
                sweet_cols.append("url")
            avail_sw = [c for c in sweet_cols if c in sweet.columns]
            st.dataframe(
                sweet[avail_sw].rename(columns={
                    "label": "Car", "year": "Year", "price_eur": "Price (EUR)",
                    "mileage_km": "Mileage", "transmission": "Trans",
                    "fuel_type": "Fuel", "horsepower": "HP",
                    "origin": "Origin", "district": "District", "url": "Link",
                }),
                hide_index=True,
                column_config={
                    "Price (EUR)": st.column_config.NumberColumn(format="%d EUR"),
                    "Mileage": st.column_config.NumberColumn(format="%,d km"),
                    "Link": st.column_config.LinkColumn("Link", display_text="Open"),
                },
            )


# ---- TAB 2: Price Trends + Seasonality --------------------------------------
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

    # --- Seasonality ---
    st.subheader("Seasonality")
    if history_df.empty:
        st.info("Need multiple months of data for seasonality analysis.")
    else:
        from src.analytics.seasonality import compute_seasonality, best_buy_sell_months

        season_data = compute_seasonality(history_df, listings_df)
        if season_data is not None and not season_data.empty:
            pivot_season = season_data.pivot_table(
                values="price_index", index="segment", columns="month_name",
            )
            month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            ordered_cols = [m for m in month_order if m in pivot_season.columns]
            if ordered_cols:
                pivot_season = pivot_season[ordered_cols]
                fig_season = px.imshow(
                    pivot_season.values,
                    x=pivot_season.columns.tolist(),
                    y=pivot_season.index.tolist(),
                    labels=dict(color="Price Index"),
                    title="Price Index by Month (100 = yearly average)",
                    aspect="auto",
                    color_continuous_scale="RdYlGn_r",
                )
                fig_season.update_layout(height=max(300, len(pivot_season) * 50))
                st.plotly_chart(fig_season, width="stretch")

                bbs = best_buy_sell_months(season_data)
                if not bbs.empty:
                    st.caption("Best months to buy (low index) and sell (high index)")
                    st.dataframe(bbs, hide_index=True)
        else:
            st.info("Need at least 3 months of scraped data. Keep the scraper running!")

    # --- Fuel Type Premium Trend ---
    st.subheader("Fuel Type Premium Trend")
    st.caption("How median prices change by fuel type over time — spot diesel decline or hybrid growth")
    if not history_df.empty and "fuel_type" in active.columns:
        fuel_ts_data = listings_df[
            listings_df["price_eur"].notna() & listings_df["fuel_type"].notna()
            & listings_df["first_seen_at"].notna()
        ].copy() if "first_seen_at" in listings_df.columns else pd.DataFrame()

        if not fuel_ts_data.empty:
            fuel_ts_data["month"] = pd.to_datetime(fuel_ts_data["first_seen_at"]).dt.to_period("M").dt.to_timestamp()
            fuel_monthly = (
                fuel_ts_data.groupby(["month", "fuel_type"])["price_eur"]
                .median().reset_index()
            )
            if len(fuel_monthly["month"].unique()) >= 2:
                fig_fuel = px.line(
                    fuel_monthly, x="month", y="price_eur", color="fuel_type",
                    labels={"month": "Month", "price_eur": "Median Price (EUR)", "fuel_type": "Fuel"},
                    height=400,
                )
                fig_fuel.update_layout(hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=-0.3))
                st.plotly_chart(fig_fuel, width="stretch")
            else:
                st.info("Need at least 2 months of data for fuel type trends.")
    else:
        st.info("No fuel type data available yet.")

# ---- TAB 3: Listings Table -------------------------------------------------
with tab_listings:
    st.subheader("All Listings")
    display_cols = [c for c in [
        "brand", "model", "year", "price_eur", "days_listed",
        "price_change_eur", "price_change_pct", "eur_per_km",
        "mileage_km", "engine_cc",
        "fuel_type", "horsepower", "transmission", "segment",
        "city", "district", "seller_type", "is_active", "url",
    ] if c in filtered_listings.columns]

    st.dataframe(
        filtered_listings[display_cols].sort_values("price_eur") if "price_eur" in display_cols else filtered_listings[display_cols],
        width="stretch", hide_index=True,
        column_config={
            "price_eur": st.column_config.NumberColumn("Price (EUR)", format="%d EUR"),
            "days_listed": st.column_config.NumberColumn("Days Listed", format="%d"),
            "price_change_eur": st.column_config.NumberColumn("Price Change (EUR)", format="%+d EUR"),
            "price_change_pct": st.column_config.NumberColumn("Price Change %", format="%.1f%%"),
            "eur_per_km": st.column_config.NumberColumn("EUR/km", format="%.3f"),
            "mileage_km": st.column_config.NumberColumn("Mileage", format="%d km"),
            "engine_cc": st.column_config.NumberColumn("Engine (cc)", format="%d"),
            "horsepower": st.column_config.NumberColumn("HP", format="%d"),
            "is_active": st.column_config.CheckboxColumn("Active"),
            "url": st.column_config.LinkColumn("Link", display_text="Open"),
        },
    )
    st.caption(f"Showing {len(filtered_listings)} listings")


# ---- TAB 4: Compare Models ---------------------------------------------------
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
                                    hover_data=["year", "fuel_type", "url"],
                                    labels={"mileage_km": "Mileage (km)", "price_eur": "Price (EUR)", "label": "Model"},
                                    title="Price vs Mileage", height=500, opacity=0.7)
                plotly_chart_with_click(fig_sc, sd, key="compare_scatter", width="stretch")

        if not turnover_df.empty:
            comparison_full = comparison.merge(turnover_df, on=["brand", "model"], how="left")
        else:
            comparison_full = comparison.copy()
            comparison_full["avg_days_to_sell"] = pd.NA
            comparison_full["weekly_turnover"] = pd.NA

        # Capital turnover rate
        if "avg_days_to_sell" in comparison_full.columns:
            comparison_full["capital_turns"] = (365 / comparison_full["avg_days_to_sell"].replace(0, np.nan)).round(1)
        else:
            comparison_full["capital_turns"] = pd.NA

        comp_cols = ["label", "count", "median_price", "min_price", "max_price", "avg_mileage"]
        if "avg_days_to_sell" in comparison_full.columns:
            comp_cols += ["avg_days_to_sell", "weekly_turnover", "capital_turns"]

        st.dataframe(
            comparison_full[comp_cols].rename(columns={
                "label": "Model", "count": "Listings", "median_price": "Median (EUR)",
                "min_price": "Min (EUR)", "max_price": "Max (EUR)", "avg_mileage": "Avg Mileage",
                "avg_days_to_sell": "Avg Days to Sell", "weekly_turnover": "Weekly Turnover %",
                "capital_turns": "Turns/Year",
            }),
            width="stretch", hide_index=True,
            column_config={
                "Median (EUR)": st.column_config.NumberColumn(format="%d EUR"),
                "Min (EUR)": st.column_config.NumberColumn(format="%d EUR"),
                "Max (EUR)": st.column_config.NumberColumn(format="%d EUR"),
                "Avg Mileage": st.column_config.NumberColumn(format="%,.0f km"),
                "Avg Days to Sell": st.column_config.NumberColumn(format="%.0f days"),
                "Weekly Turnover %": st.column_config.NumberColumn(format="%.1f%%"),
                "Turns/Year": st.column_config.NumberColumn(format="%.1f"),
            },
        )

        # --- Depreciation Curve ---
        st.subheader("Depreciation Curve")
        dep_data = active[active["price_eur"].notna() & active["year"].notna()].copy()
        if not dep_data.empty:
            dep_data["label"] = dep_data["brand"] + " " + dep_data["model"]
            model_sizes = dep_data.groupby("label").size()
            dep_available = sorted(model_sizes[model_sizes >= 3].index.tolist())

            if dep_available:
                selected_dep = st.multiselect(
                    "Select models for depreciation curve", dep_available,
                    default=dep_available[:3] if len(dep_available) >= 3 else dep_available,
                    key="dep_models",
                )
                if selected_dep:
                    from src.analytics.regression import depreciation_curve
                    fig_dep = go.Figure()
                    dep_all_subs = []
                    for label in selected_dep:
                        sub = dep_data[dep_data["label"] == label]
                        dep_all_subs.append(sub)
                        fig_dep.add_trace(go.Scatter(
                            x=sub["year"], y=sub["price_eur"],
                            mode="markers", name=label, opacity=0.6,
                            customdata=sub["url"].values if "url" in sub.columns else None,
                            hovertemplate="%{x}, %{y:,.0f} EUR<extra>%{fullData.name}</extra>",
                        ))
                        result = depreciation_curve(active, sub.iloc[0]["brand"], sub.iloc[0]["model"])
                        if result:
                            years = np.linspace(sub["year"].min(), sub["year"].max(), 50)
                            prices = result["slope"] * years + result["intercept"]
                            fig_dep.add_trace(go.Scatter(x=years, y=prices, mode="lines",
                                                          name=f"{label} (R²={result['r_squared']:.2f})",
                                                          line=dict(dash="dash")))
                    dep_combined = pd.concat(dep_all_subs) if dep_all_subs else pd.DataFrame()
                    fig_dep.update_layout(xaxis_title="Year", yaxis_title="Price (EUR)", height=500, hovermode="closest")
                    plotly_chart_with_click(fig_dep, dep_combined, key="dep_scatter", width="stretch")

        # --- Seller Type Spread (Particular vs Profissional) ---
        st.subheader("Seller Type Spread")
        st.caption("Difference between private and dealer prices — this is the flipper's margin")
        if "seller_type" in active.columns:
            spread_data = active[active["price_eur"].notna() & active["seller_type"].notna()].copy()
            spread_data["label"] = spread_data["brand"] + " " + spread_data["model"]
            spread_agg = spread_data.pivot_table(
                values="price_eur", index="label", columns="seller_type",
                aggfunc="median",
            )
            if "Particular" in spread_agg.columns and "Profissional" in spread_agg.columns:
                spread_agg = spread_agg.dropna(subset=["Particular", "Profissional"])
                spread_agg["spread_eur"] = (spread_agg["Profissional"] - spread_agg["Particular"]).round(0)
                spread_agg["spread_pct"] = ((spread_agg["spread_eur"] / spread_agg["Particular"]) * 100).round(1)
                spread_agg = spread_agg.sort_values("spread_eur", ascending=False).reset_index()

                if not spread_agg.empty:
                    col_sp1, col_sp2 = st.columns(2)
                    with col_sp1:
                        top_spread = spread_agg.head(15)
                        colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in top_spread["spread_eur"]]
                        fig_spread = go.Figure(go.Bar(
                            x=top_spread["spread_eur"], y=top_spread["label"],
                            orientation="h", marker_color=colors,
                            text=[f"{v:+,.0f} EUR ({p:+.1f}%)" for v, p in zip(top_spread["spread_eur"], top_spread["spread_pct"])],
                            textposition="auto",
                        ))
                        fig_spread.update_layout(
                            title="Dealer vs Private Price Gap",
                            xaxis_title="Spread (EUR)", yaxis_title="",
                            height=max(300, len(top_spread) * 35 + 80),
                            yaxis=dict(autorange="reversed"),
                        )
                        st.plotly_chart(fig_spread, width="stretch")
                    with col_sp2:
                        st.dataframe(
                            spread_agg[["label", "Particular", "Profissional", "spread_eur", "spread_pct"]].rename(columns={
                                "label": "Model", "Particular": "Private (EUR)", "Profissional": "Dealer (EUR)",
                                "spread_eur": "Spread (EUR)", "spread_pct": "Spread %",
                            }),
                            hide_index=True,
                            column_config={
                                "Private (EUR)": st.column_config.NumberColumn(format="%d EUR"),
                                "Dealer (EUR)": st.column_config.NumberColumn(format="%d EUR"),
                                "Spread (EUR)": st.column_config.NumberColumn(format="%+d EUR"),
                                "Spread %": st.column_config.NumberColumn(format="%+.1f%%"),
                            },
                        )

        # --- Mileage Sensitivity ---
        st.subheader("Mileage Sensitivity")
        st.caption("Price drop per 10,000 km — models with low sensitivity are better for high-mileage flips")
        mil_sens_data = active[
            active["price_eur"].notna() & active["mileage_km"].notna()
            & (active["mileage_km"] > 0)
        ].copy()
        mil_sens_data["label"] = mil_sens_data["brand"] + " " + mil_sens_data["model"]

        sensitivities = []
        for label, group in mil_sens_data.groupby("label"):
            if len(group) < 5:
                continue
            x = group["mileage_km"].values.astype(float)
            y = group["price_eur"].values.astype(float)
            X = np.column_stack([x, np.ones(len(x))])
            try:
                coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                eur_per_10k = coeffs[0] * 10000
                sensitivities.append({"Model": label, "EUR per 10k km": round(eur_per_10k), "Listings": len(group)})
            except Exception:
                pass

        if sensitivities:
            sens_df = pd.DataFrame(sensitivities).sort_values("EUR per 10k km")
            col_ms1, col_ms2 = st.columns(2)
            with col_ms1:
                top_sens = sens_df.head(20)
                colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in top_sens["EUR per 10k km"]]
                fig_sens = go.Figure(go.Bar(
                    x=top_sens["EUR per 10k km"], y=top_sens["Model"],
                    orientation="h", marker_color=colors,
                    text=[f"{v:+,} EUR" for v in top_sens["EUR per 10k km"]],
                    textposition="auto",
                ))
                fig_sens.update_layout(
                    title="Price Change per 10,000 km",
                    xaxis_title="EUR per 10k km", yaxis_title="",
                    height=max(300, len(top_sens) * 30 + 80),
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(fig_sens, width="stretch")
            with col_ms2:
                st.dataframe(
                    sens_df.rename(columns={"EUR per 10k km": "EUR/10k km"}),
                    hide_index=True,
                    column_config={
                        "EUR/10k km": st.column_config.NumberColumn(format="%+d EUR"),
                    },
                )

        # --- Year Sweet Spot ---
        st.subheader("Year Sweet Spot")
        st.caption("For each model — which years have the best flip margin (below-market deals available)")
        sweet_data = active[
            active["price_eur"].notna() & active["year"].notna()
        ].copy()
        sweet_data["label"] = sweet_data["brand"] + " " + sweet_data["model"]

        sweet_spots = []
        for label, group in sweet_data.groupby("label"):
            if len(group) < 5:
                continue
            model_median = group["price_eur"].median()
            for yr, yr_group in group.groupby("year"):
                if len(yr_group) < 2:
                    continue
                yr_median = yr_group["price_eur"].median()
                yr_min = yr_group["price_eur"].min()
                gap_pct = round((model_median - yr_min) / model_median * 100, 1)
                sweet_spots.append({
                    "Model": label, "Year": int(yr), "Listings": len(yr_group),
                    "Year Median": round(yr_median), "Model Median": round(model_median),
                    "Best Price": round(yr_min), "Gap %": gap_pct,
                })

        if sweet_spots:
            sweet_df = pd.DataFrame(sweet_spots)
            sweet_df = sweet_df[sweet_df["Listings"] >= 2].sort_values("Gap %", ascending=False)
            if not sweet_df.empty:
                pivot_sweet = sweet_df.pivot_table(values="Gap %", index="Model", columns="Year", aggfunc="first")
                if not pivot_sweet.empty and len(pivot_sweet) >= 2:
                    fig_sweet = px.imshow(
                        pivot_sweet.values,
                        x=[str(int(c)) for c in pivot_sweet.columns],
                        y=pivot_sweet.index.tolist(),
                        labels=dict(color="Gap % (higher = better deal)"),
                        aspect="auto", color_continuous_scale="YlGn",
                        height=max(300, len(pivot_sweet) * 40 + 80),
                    )
                    fig_sweet.update_traces(
                        text=[[f"{v:.0f}%" if pd.notna(v) else "" for v in row] for row in pivot_sweet.values],
                        texttemplate="%{text}",
                    )
                    fig_sweet.update_layout(title="Deal Gap by Model x Year (higher = bigger discount available)")
                    st.plotly_chart(fig_sweet, width="stretch")

                st.dataframe(
                    sweet_df.head(30),
                    hide_index=True,
                    column_config={
                        "Year Median": st.column_config.NumberColumn(format="%d EUR"),
                        "Model Median": st.column_config.NumberColumn(format="%d EUR"),
                        "Best Price": st.column_config.NumberColumn(format="%d EUR"),
                        "Gap %": st.column_config.NumberColumn(format="%.1f%%"),
                    },
                )

# ---- TAB 5: Geography + Competition Density ---------------------------------
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
                    active[active["price_eur"].notna()].groupby("district")["price_eur"]
                    .median().reset_index().rename(columns={"price_eur": "Median Price (EUR)"})
                    .sort_values("Median Price (EUR)", ascending=False)
                )
                fig_price = px.bar(district_prices.head(15), x="Median Price (EUR)", y="district",
                                   orientation="h", title="Median Price by District", height=450)
                fig_price.update_layout(yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_price, width="stretch")

        if "city" in active.columns and active["city"].notna().any():
            city_stats = (
                active[active["price_eur"].notna()].groupby(["district", "city"])
                .agg(count=("price_eur", "size"), median=("price_eur", "median"))
                .reset_index().sort_values("count", ascending=False)
            )
            st.dataframe(
                city_stats.head(30).rename(columns={"district": "District", "city": "City",
                                                     "count": "Listings", "median": "Median (EUR)"}),
                width="stretch", hide_index=True,
                column_config={"Median (EUR)": st.column_config.NumberColumn(format="%d EUR")},
            )

        # --- Regional Price Differences ---
        st.subheader("Regional Price Differences")
        geo_active = active[active["price_eur"].notna() & active["district"].notna()].copy()
        if not geo_active.empty:
            geo_active["label"] = geo_active["brand"] + " " + geo_active["model"]
            model_district_counts = geo_active.groupby("label")["district"].nunique()
            geo_candidates = sorted(model_district_counts[model_district_counts >= 2].index.tolist())

            if geo_candidates:
                selected_geo = st.multiselect("Models to compare across districts", geo_candidates,
                                               default=geo_candidates[:5] if len(geo_candidates) >= 5 else geo_candidates,
                                               key="geo_models")
                if selected_geo:
                    geo_subset = geo_active[geo_active["label"].isin(selected_geo)]
                    pivot = geo_subset.pivot_table(values="price_eur", index="label", columns="district", aggfunc="median")
                    if not pivot.empty:
                        fig_heat = px.imshow(pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
                                             labels=dict(color="Median Price (EUR)"),
                                             title="Median Price: Model x District", aspect="auto",
                                             color_continuous_scale="RdYlGn_r")
                        fig_heat.update_layout(height=max(300, len(selected_geo) * 60))
                        st.plotly_chart(fig_heat, width="stretch")

                        arbitrage = []
                        for model_label in pivot.index:
                            row = pivot.loc[model_label].dropna()
                            if len(row) >= 2:
                                arbitrage.append({
                                    "Model": model_label, "Cheapest District": row.idxmin(),
                                    "Cheapest Price": int(row.min()), "Most Expensive District": row.idxmax(),
                                    "Most Expensive Price": int(row.max()),
                                    "Arbitrage (EUR)": int(row.max() - row.min()),
                                })
                        if arbitrage:
                            st.subheader("Arbitrage Opportunities")
                            st.dataframe(pd.DataFrame(arbitrage).sort_values("Arbitrage (EUR)", ascending=False),
                                         hide_index=True, column_config={
                                             "Cheapest Price": st.column_config.NumberColumn(format="%d EUR"),
                                             "Most Expensive Price": st.column_config.NumberColumn(format="%d EUR"),
                                             "Arbitrage (EUR)": st.column_config.NumberColumn(format="%d EUR"),
                                         })

        # --- Competition Density ---
        st.subheader("Competition Density")
        from src.analytics.competition import compute_competition_density
        comp_density = compute_competition_density(listings_df, turnover_df)
        if not comp_density.empty:
            comp_models = sorted(comp_density["label"].unique().tolist())
            sel_comp = st.multiselect("Select models for competition analysis", comp_models,
                                       default=comp_models[:5] if len(comp_models) >= 5 else comp_models,
                                       key="comp_models")
            if sel_comp:
                cd = comp_density[comp_density["label"].isin(sel_comp)]
                st.dataframe(
                    cd[["label", "district", "local_count", "national_count", "saturation", "local_median",
                        "avg_days_to_sell"] if "avg_days_to_sell" in cd.columns else
                       ["label", "district", "local_count", "national_count", "saturation", "local_median"]]
                    .rename(columns={
                        "label": "Model", "district": "District", "local_count": "Local",
                        "national_count": "National", "saturation": "Saturation",
                        "local_median": "Local Median (EUR)", "avg_days_to_sell": "Avg Days to Sell",
                    }),
                    hide_index=True,
                    column_config={
                        "Saturation": st.column_config.NumberColumn(format="%.2f"),
                        "Local Median (EUR)": st.column_config.NumberColumn(format="%d EUR"),
                        "Avg Days to Sell": st.column_config.NumberColumn(format="%.0f days"),
                    },
                )
                if "avg_days_to_sell" in cd.columns and cd["avg_days_to_sell"].notna().any():
                    fig_comp = px.scatter(
                        cd[cd["avg_days_to_sell"].notna()],
                        x="saturation", y="avg_days_to_sell", size="local_count",
                        color="label", hover_data=["district"],
                        labels={"saturation": "Saturation Index", "avg_days_to_sell": "Avg Days to Sell",
                                "local_count": "Listings", "label": "Model"},
                        title="Competition vs Sell Speed", height=450,
                    )
                    st.plotly_chart(fig_comp, width="stretch")
        else:
            st.info("Not enough data for competition analysis.")

# ---- TAB 6: Portfolio / Deal Tracker ----------------------------------------
with tab_portfolio:
    st.subheader("Deal Portfolio")
    st.caption("Track your car purchases, repairs, and sales")

    # Add new deal form
    with st.expander("Add New Deal", expanded=False):
        with st.form("add_deal_form"):
            fd1, fd2, fd3 = st.columns(3)
            with fd1:
                d_brand = st.text_input("Brand", key="d_brand")
                d_model = st.text_input("Model", key="d_model")
                d_year = st.number_input("Year", min_value=1990, max_value=2026, value=2018, key="d_year")
            with fd2:
                d_buy_date = st.date_input("Buy Date", value=date.today(), key="d_buy_date")
                d_buy_price = st.number_input("Buy Price (EUR)", min_value=0, value=10000, step=500, key="d_buy_price")
                d_repair = st.number_input("Repair Cost (EUR)", min_value=0, value=0, step=100, key="d_repair")
            with fd3:
                d_reg = st.number_input("IUC + Registration (EUR)", min_value=0, value=0, step=50, key="d_reg")
                d_mileage = st.number_input("Mileage (km)", min_value=0, value=100000, step=5000, key="d_mileage")
                d_notes = st.text_input("Notes", key="d_notes")

            if st.form_submit_button("Add Deal", type="primary"):
                if d_brand and d_model and d_buy_price > 0:
                    from src.storage.database import init_db, get_session
                    from src.storage.repository import add_portfolio_deal
                    init_db(str(__import__("pathlib").Path(__file__).resolve().parent.parent.parent / "data" / "olx_cars.db"))
                    sess = get_session()
                    add_portfolio_deal(sess, {
                        "brand": d_brand, "model": d_model, "year": d_year,
                        "mileage_km": d_mileage, "buy_date": d_buy_date,
                        "buy_price_eur": d_buy_price, "repair_cost_eur": d_repair,
                        "registration_cost_eur": d_reg, "notes": d_notes,
                    })
                    st.success("Deal added!")
                    st.rerun()
                else:
                    st.error("Fill in Brand, Model, and Buy Price.")

    portfolio_df = get_portfolio()

    if portfolio_df.empty:
        st.info("No deals tracked yet. Add your first deal above.")
    else:
        # Summary metrics
        sold = portfolio_df[portfolio_df["sell_price_eur"].notna()]
        unsold = portfolio_df[portfolio_df["sell_price_eur"].isna()]

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Deals", len(portfolio_df))
        m2.metric("Active (unsold)", len(unsold))
        total_invested = portfolio_df["total_cost_eur"].sum()
        m3.metric("Total Invested", f"{total_invested:,.0f} EUR")
        if not sold.empty:
            total_profit = sold["gross_profit_eur"].sum()
            avg_roi = sold["roi_pct"].mean()
            m4.metric("Total Profit", f"{total_profit:+,.0f} EUR",
                      delta=f"{total_profit:+,.0f} EUR",
                      delta_color="normal" if total_profit >= 0 else "inverse")
            m5.metric("Avg ROI", f"{avg_roi:+.1f}%")
        else:
            m4.metric("Total Profit", "—")
            m5.metric("Avg ROI", "—")

        # Portfolio table
        disp = portfolio_df[[c for c in [
            "brand", "model", "year", "buy_date", "buy_price_eur", "repair_cost_eur",
            "registration_cost_eur", "total_cost_eur", "sell_date", "sell_price_eur",
            "gross_profit_eur", "roi_pct", "days_in_inventory", "notes",
        ] if c in portfolio_df.columns]].copy()

        st.dataframe(disp, hide_index=True, column_config={
            "buy_price_eur": st.column_config.NumberColumn("Buy (EUR)", format="%d EUR"),
            "repair_cost_eur": st.column_config.NumberColumn("Repair (EUR)", format="%d EUR"),
            "registration_cost_eur": st.column_config.NumberColumn("Reg (EUR)", format="%d EUR"),
            "total_cost_eur": st.column_config.NumberColumn("Total Cost (EUR)", format="%d EUR"),
            "sell_price_eur": st.column_config.NumberColumn("Sell (EUR)", format="%d EUR"),
            "gross_profit_eur": st.column_config.NumberColumn("Profit (EUR)", format="%+d EUR"),
            "roi_pct": st.column_config.NumberColumn("ROI %", format="%+.1f%%"),
            "days_in_inventory": st.column_config.NumberColumn("Days", format="%d"),
        })

        # Record sale for unsold deals
        if not unsold.empty:
            with st.expander("Record Sale"):
                deal_options = {f"{r['brand']} {r['model']} {r['year']} (bought {r['buy_date']})": r["id"]
                                for _, r in unsold.iterrows()}
                sel_deal = st.selectbox("Select deal", list(deal_options.keys()))
                sell_date = st.date_input("Sell Date", value=date.today(), key="sell_date")
                sell_price = st.number_input("Sell Price (EUR)", min_value=0, value=15000, step=500, key="sell_price")
                if st.button("Record Sale"):
                    from src.storage.database import init_db, get_session
                    from src.storage.repository import update_portfolio_deal
                    init_db(str(__import__("pathlib").Path(__file__).resolve().parent.parent.parent / "data" / "olx_cars.db"))
                    sess = get_session()
                    update_portfolio_deal(sess, deal_options[sel_deal], {
                        "sell_date": sell_date, "sell_price_eur": sell_price,
                    })
                    st.success("Sale recorded!")
                    st.rerun()

# ---- TAB 8: Unmatched Listings -----------------------------------------------
with tab_unmatched:
    st.subheader("Unmatched Listings")
    st.caption("Listings where car generation could not be determined — review and add to generations seed")

    if unmatched_df.empty:
        st.info("No unmatched listings. All scraped cars have a known generation.")
    else:
        st.metric("Unmatched", len(unmatched_df))

        # Summary: missing brand+model combos
        summary = (
            unmatched_df.groupby(["brand", "model", "reason"])
            .agg(count=("olx_id", "size"), years=("year", lambda y: sorted(set(int(v) for v in y.dropna()))))
            .reset_index()
            .sort_values("count", ascending=False)
        )
        st.subheader("Missing Generations")
        st.dataframe(summary.rename(columns={
            "brand": "Brand", "model": "Model", "reason": "Reason",
            "count": "Count", "years": "Years",
        }), hide_index=True)

        # Full table
        st.subheader("All Unmatched")
        um_cols = ["brand", "model", "year", "price_eur", "mileage_km",
                   "fuel_type", "city", "district", "reason"]
        if "url" in unmatched_df.columns:
            um_cols.append("url")
        avail = [c for c in um_cols if c in unmatched_df.columns]
        st.dataframe(
            unmatched_df[avail].rename(columns={
                "brand": "Brand", "model": "Model", "year": "Year",
                "price_eur": "Price (EUR)", "mileage_km": "Mileage",
                "fuel_type": "Fuel", "city": "City", "district": "District",
                "reason": "Reason", "url": "Link",
            }),
            hide_index=True,
            column_config={"Link": st.column_config.LinkColumn("Link", display_text="Open")} if "url" in avail else {},
        )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(f"OLX Car Deals v0.3 — {len(listings_df)} listings")
