"""OLX.pt Car Parser — Streamlit Dashboard."""

from datetime import date

import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from data_loader import load_all, load_portfolio

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


listings_df, history_df, signals_df, brands_models, turnover_df, _portfolio_init = load_data()

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
(tab_trends, tab_signals, tab_listings, tab_compare, tab_geo,
 tab_estimator, tab_profit, tab_premiums, tab_portfolio) = st.tabs([
    "Price Trends", "Buy Signals", "Listings", "Compare Models", "Geography",
    "Price Estimator", "Profit Calculator", "Feature Premiums", "Portfolio",
])

# ---- TAB 1: Price Trends + Seasonality -----------------------------------
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

# ---- TAB 2: Buy Signals --------------------------------------------------
with tab_signals:
    st.subheader("Buy Opportunities")
    st.caption("Listings priced 15%+ below market median")
    if filtered_signals.empty:
        st.info("No buy signals with current filters.")
    else:
        top = filtered_signals.head(20).copy()
        top["deal"] = top["discount_pct"].apply(lambda d: "🔥🔥🔥" if d > 25 else ("🔥🔥" if d > 20 else "🔥"))
        if "sample_size" in top.columns:
            top["confidence"] = top["sample_size"].apply(
                lambda n: "High" if n >= 10 else ("Medium" if n >= 5 else "Low")
            )

        signal_cols = ["deal", "brand", "model", "year", "price_eur", "median_price_eur", "discount_pct"]
        if "sample_size" in top.columns:
            signal_cols += ["sample_size", "confidence"]
        if "days_listed" in top.columns:
            signal_cols.append("days_listed")
        if "price_change_pct" in top.columns:
            signal_cols.append("price_change_pct")
        if "eur_per_km" in top.columns:
            signal_cols.append("eur_per_km")
        signal_cols += ["city", "district", "mileage_km", "fuel_type"]
        if "url" in top.columns:
            signal_cols.append("url")

        st.dataframe(
            top[signal_cols].rename(columns={
                "deal": "Deal", "brand": "Brand", "model": "Model", "year": "Year",
                "price_eur": "Price (EUR)", "median_price_eur": "Median (EUR)",
                "discount_pct": "Discount %", "sample_size": "# Listings",
                "confidence": "Confidence", "days_listed": "Days Listed",
                "price_change_pct": "Price Drop %", "eur_per_km": "EUR/km",
                "city": "City", "district": "District",
                "mileage_km": "Mileage", "fuel_type": "Fuel", "url": "Link",
            }),
            width="stretch", hide_index=True,
            column_config={"Link": st.column_config.LinkColumn("Link", display_text="Open")},
        )
        fig_disc = px.histogram(filtered_signals, x="discount_pct", nbins=20,
                                labels={"discount_pct": "Discount below median (%)"}, title="Discount Distribution", height=350)
        st.plotly_chart(fig_disc, width="stretch")

# ---- TAB 3: Listings Table -----------------------------------------------
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

        if not turnover_df.empty:
            comparison_full = comparison.merge(turnover_df, on=["brand", "model"], how="left")
        else:
            comparison_full = comparison.copy()
            comparison_full["avg_days_to_sell"] = pd.NA
            comparison_full["weekly_turnover"] = pd.NA

        comp_cols = ["label", "count", "median_price", "min_price", "max_price", "avg_mileage"]
        if "avg_days_to_sell" in comparison_full.columns:
            comp_cols += ["avg_days_to_sell", "weekly_turnover"]

        st.dataframe(
            comparison_full[comp_cols].rename(columns={
                "label": "Model", "count": "Listings", "median_price": "Median (EUR)",
                "min_price": "Min (EUR)", "max_price": "Max (EUR)", "avg_mileage": "Avg Mileage",
                "avg_days_to_sell": "Avg Days to Sell", "weekly_turnover": "Weekly Turnover %",
            }),
            width="stretch", hide_index=True,
            column_config={
                "Median (EUR)": st.column_config.NumberColumn(format="%d EUR"),
                "Min (EUR)": st.column_config.NumberColumn(format="%d EUR"),
                "Max (EUR)": st.column_config.NumberColumn(format="%d EUR"),
                "Avg Mileage": st.column_config.NumberColumn(format="%,.0f km"),
                "Avg Days to Sell": st.column_config.NumberColumn(format="%.0f days"),
                "Weekly Turnover %": st.column_config.NumberColumn(format="%.1f%%"),
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
                    for label in selected_dep:
                        sub = dep_data[dep_data["label"] == label]
                        fig_dep.add_trace(go.Scatter(x=sub["year"], y=sub["price_eur"],
                                                      mode="markers", name=label, opacity=0.6))
                        result = depreciation_curve(active, sub.iloc[0]["brand"], sub.iloc[0]["model"])
                        if result:
                            years = np.linspace(sub["year"].min(), sub["year"].max(), 50)
                            prices = result["slope"] * years + result["intercept"]
                            fig_dep.add_trace(go.Scatter(x=years, y=prices, mode="lines",
                                                          name=f"{label} (R²={result['r_squared']:.2f})",
                                                          line=dict(dash="dash")))
                    fig_dep.update_layout(xaxis_title="Year", yaxis_title="Price (EUR)", height=500, hovermode="closest")
                    st.plotly_chart(fig_dep, width="stretch")

# ---- TAB 5: Geography + Competition Density ------------------------------
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

# ---- TAB 6: Price Estimator ----------------------------------------------
with tab_estimator:
    st.subheader("Price Estimator")
    st.caption("Get a recommended price based on regression on existing listings")

    if listings_df.empty:
        st.info("No data yet. Run scraper first.")
    else:
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            est_brand = st.selectbox("Brand", sorted(brands_models.keys()), key="est_brand")
            est_models_list = sorted(brands_models.get(est_brand, []))
            est_model = st.selectbox("Model", est_models_list, key="est_model")
            est_year = st.number_input("Year", min_value=1990, max_value=2026, value=2018, key="est_year")
        with col_e2:
            est_mileage = st.number_input("Mileage (km)", min_value=0, max_value=500000,
                                           value=100000, step=5000, key="est_mileage")
            fuel_opts = sorted(listings_df["fuel_type"].dropna().unique().tolist()) if "fuel_type" in listings_df.columns else []
            est_fuel = st.selectbox("Fuel type", ["Any"] + fuel_opts, key="est_fuel")

        if st.button("Estimate Price", type="primary"):
            from src.analytics.regression import estimate_price
            fuel = est_fuel if est_fuel != "Any" else None
            result = estimate_price(active, est_brand, est_model, est_year, est_mileage, fuel)
            if result:
                st.divider()
                c1, c2, c3 = st.columns(3)
                c1.metric("Quick Sale (P25)", f"{int(result['p25']):,} EUR")
                c2.metric("Fair Price (Median)", f"{int(result['median']):,} EUR")
                c3.metric("Max Price (P75)", f"{int(result['p75']):,} EUR")
                st.caption(f"Based on {result['sample_size']} similar listings")

                comps = active[(active["brand"] == est_brand) & (active["model"] == est_model) & active["price_eur"].notna()].copy()
                if fuel and len(comps[comps["fuel_type"] == fuel]) >= 3:
                    comps = comps[comps["fuel_type"] == fuel]
                comp_cols = [c for c in ["year", "price_eur", "mileage_km", "fuel_type", "city", "district", "url"] if c in comps.columns]
                st.subheader("Comparable Listings")
                st.dataframe(comps[comp_cols].sort_values("price_eur"), hide_index=True,
                             column_config={"price_eur": st.column_config.NumberColumn("Price (EUR)", format="%d EUR"),
                                            "mileage_km": st.column_config.NumberColumn("Mileage", format="%d km"),
                                            "url": st.column_config.LinkColumn("Link", display_text="Open")})
            else:
                st.warning("Not enough data for this combination (need at least 5 listings).")

# ---- TAB 7: Profit Calculator --------------------------------------------
with tab_profit:
    st.subheader("Profit Calculator")
    st.caption("Estimate profit from buying, fixing, and reselling a car")

    if listings_df.empty:
        st.info("No data yet.")
    else:
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            pc_brand = st.selectbox("Brand", sorted(brands_models.keys()), key="pc_brand")
            pc_model = st.selectbox("Model", sorted(brands_models.get(pc_brand, [])), key="pc_model")
            pc_year = st.number_input("Year", min_value=1990, max_value=2026, value=2018, key="pc_year")
            pc_mileage = st.number_input("Mileage (km)", min_value=0, max_value=500000, value=100000, step=5000, key="pc_mileage")
        with col_p2:
            pc_buy_price = st.number_input("Buy Price (EUR)", min_value=0, max_value=200000, value=10000, step=500, key="pc_buy")
            pc_repair = st.number_input("Repair Cost (EUR)", min_value=0, max_value=50000, value=500, step=100, key="pc_repair")
            pc_reg = st.number_input("IUC + Registration (EUR)", min_value=0, max_value=5000, value=200, step=50, key="pc_reg")
            pc_fuel_opts = sorted(listings_df["fuel_type"].dropna().unique().tolist()) if "fuel_type" in listings_df.columns else []
            pc_fuel = st.selectbox("Fuel type", ["Any"] + pc_fuel_opts, key="pc_fuel")

        if st.button("Calculate Profit", type="primary"):
            from src.analytics.regression import estimate_price
            fuel = pc_fuel if pc_fuel != "Any" else None
            result = estimate_price(active, pc_brand, pc_model, pc_year, pc_mileage, fuel)

            if result:
                total_cost = pc_buy_price + pc_repair + pc_reg
                sell_median = int(result["median"])
                sell_p25 = int(result["p25"])
                sell_p75 = int(result["p75"])
                profit_median = sell_median - total_cost
                profit_p25 = sell_p25 - total_cost
                profit_p75 = sell_p75 - total_cost
                roi_median = (profit_median / total_cost * 100) if total_cost > 0 else 0

                st.divider()
                st.subheader("Estimated Results")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Investment", f"{total_cost:,} EUR")
                c2.metric("Sell Price (median)", f"{sell_median:,} EUR")
                c3.metric("Gross Profit", f"{profit_median:+,} EUR",
                          delta=f"{profit_median:+,} EUR",
                          delta_color="normal" if profit_median >= 0 else "inverse")
                c4.metric("ROI", f"{roi_median:+.1f}%",
                          delta=f"{roi_median:+.1f}%",
                          delta_color="normal" if roi_median >= 0 else "inverse")

                st.divider()
                scenarios = pd.DataFrame([
                    {"Scenario": "Quick Sale (P25)", "Sell Price": sell_p25, "Profit": profit_p25,
                     "ROI %": round(profit_p25 / total_cost * 100, 1) if total_cost > 0 else 0},
                    {"Scenario": "Fair Price (Median)", "Sell Price": sell_median, "Profit": profit_median,
                     "ROI %": round(roi_median, 1)},
                    {"Scenario": "Best Case (P75)", "Sell Price": sell_p75, "Profit": profit_p75,
                     "ROI %": round(profit_p75 / total_cost * 100, 1) if total_cost > 0 else 0},
                    {"Scenario": "Break Even", "Sell Price": total_cost, "Profit": 0, "ROI %": 0.0},
                ])
                st.dataframe(scenarios, hide_index=True,
                             column_config={
                                 "Sell Price": st.column_config.NumberColumn(format="%d EUR"),
                                 "Profit": st.column_config.NumberColumn(format="%+d EUR"),
                                 "ROI %": st.column_config.NumberColumn(format="%+.1f%%"),
                             })
                st.caption(f"Based on {result['sample_size']} similar listings. Buy price vs market median: "
                           f"{'below' if pc_buy_price < sell_median else 'above'} by "
                           f"{abs(pc_buy_price - sell_median):,} EUR")
            else:
                st.warning("Not enough data for this combination (need at least 5 listings).")

# ---- TAB 8: Feature Premiums ---------------------------------------------
with tab_premiums:
    st.subheader("Feature Price Premiums")
    st.caption("How much each feature adds or subtracts from the price")

    if listings_df.empty:
        st.info("No data yet.")
    else:
        col_fp1, col_fp2 = st.columns(2)
        with col_fp1:
            fp_brand = st.selectbox("Brand", sorted(brands_models.keys()), key="fp_brand")
        with col_fp2:
            fp_model = st.selectbox("Model", sorted(brands_models.get(fp_brand, [])), key="fp_model")

        from src.analytics.premiums import compute_premiums
        premiums = compute_premiums(active, fp_brand, fp_model)

        if premiums is None:
            st.warning(f"Not enough data for {fp_brand} {fp_model} (need at least 10 listings).")
        else:
            st.metric("Baseline Median Price", f"{int(premiums['baseline_price']):,} EUR")
            st.caption(f"Based on {premiums['sample_size']} listings")

            dimensions = [
                ("transmission", "Transmission"),
                ("fuel_type", "Fuel Type"),
                ("mileage_bracket", "Mileage Range"),
                ("year", "Year"),
                ("color", "Color"),
            ]
            for key, title in dimensions:
                df_dim = premiums.get(key, pd.DataFrame())
                if df_dim is not None and not df_dim.empty:
                    st.subheader(title)
                    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in df_dim["premium_eur"]]
                    fig_prem = go.Figure(go.Bar(
                        x=df_dim["premium_eur"], y=df_dim["category"].astype(str),
                        orientation="h", marker_color=colors,
                        text=[f"{v:+,.0f} EUR ({p:+.1f}%)" for v, p in zip(df_dim["premium_eur"], df_dim["premium_pct"])],
                        textposition="auto",
                    ))
                    fig_prem.update_layout(
                        xaxis_title="Premium (EUR)", yaxis_title="",
                        height=max(200, len(df_dim) * 40 + 80),
                        yaxis=dict(autorange="reversed"),
                    )
                    st.plotly_chart(fig_prem, width="stretch")

# ---- TAB 9: Portfolio / Deal Tracker --------------------------------------
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

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(f"OLX.pt Car Parser v0.2 — {len(listings_df)} listings")
