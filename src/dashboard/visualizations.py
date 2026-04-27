"""OLX Car Parser — Analytics & Model Visualizations Dashboard."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

_dashboard_dir = Path(__file__).resolve().parent
_project_root = _dashboard_dir.parent.parent
sys.path.insert(0, str(_dashboard_dir))
sys.path.insert(0, str(_project_root))

from data_loader import load_all, load_from_db
from src.analytics.computed_columns import enrich_listings
from src.analytics.turnover import compute_turnover_stats
from src.analytics.competition import compute_competition_density
from src.analytics.price_model import (
    train_price_model, predict_prices, load_model, save_model,
    load_metrics_history, compute_permutation_importance, compute_feature_completeness,
    NUMERIC_FEATURES, BOOL_FEATURES, CATEGORICAL_FEATURES,
)
from src.analytics.model_eval import (
    evaluate_oof, worst_residuals, reliability_curve, load_backtest,
)

st.set_page_config(
    page_title="OLX Car Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .block-container { padding-top: 1rem; }
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    div[data-testid="stMetric"] label {
        color: #a8b2d1 !important;
        font-size: 0.85rem !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 24px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Plotly theme
# ---------------------------------------------------------------------------
_COLORS = [
    "#FF6B35", "#00B4D8", "#E63946", "#2EC4B6", "#FFD166",
    "#8338EC", "#06D6A0", "#EF476F", "#118AB2", "#073B4C",
]
_BG = "#0E1117"
_PAPER = "#0E1117"
_GRID = "rgba(255,255,255,0.06)"
_TEXT = "#FAFAFA"
_FONT = dict(family="Inter, system-ui, sans-serif", color=_TEXT)

_LAYOUT = dict(
    paper_bgcolor=_PAPER,
    plot_bgcolor=_BG,
    font=_FONT,
    margin=dict(l=40, r=20, t=50, b=40),
    xaxis=dict(gridcolor=_GRID, zeroline=False),
    yaxis=dict(gridcolor=_GRID, zeroline=False),
    colorway=_COLORS,
    hoverlabel=dict(
        bgcolor="#1a1a2e",
        font_size=13,
        font_family="Inter, system-ui, sans-serif",
    ),
)


def _apply_layout(fig, **kw):
    merged = {**_LAYOUT, **kw}
    fig.update_layout(**merged)
    return fig


# ---------------------------------------------------------------------------
# Data loading — keep module-level work lightweight (no model training)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=600)
def get_data():
    db_data = load_from_db()
    if db_data is None:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    listings, history = db_data
    listings = enrich_listings(listings)
    if "real_mileage_km" in listings.columns:
        listings["mileage_km"] = listings["real_mileage_km"].fillna(listings["mileage_km"])
    turnover = compute_turnover_stats(listings)
    competition = compute_competition_density(listings, turnover)
    return listings, history, turnover, competition


@st.cache_data(ttl=600, show_spinner="Loading model & predictions...")
def get_model_data(_active_df, _listings_df):
    """Load saved model (or train if user requested) and return predictions."""
    active = _active_df.copy()

    saved = load_model()
    if saved is not None:
        models, cat_maps, metrics, oof_preds, calibrator, text_pipeline = saved
    else:
        result = train_price_model(active)
        if result is None:
            return None
        models, cat_maps, metrics, oof_preds, calibrator, text_pipeline = result
        save_model(
            models, cat_maps, metrics,
            oof_preds=oof_preds,
            median_calibrator=calibrator,
            text_pipeline=text_pipeline,
        )

    conformal_q = metrics.get("conformal_q", 0.0)
    per_bucket_q = metrics.get("conformal_q_per_bucket", {})
    _edges_raw = metrics.get("conformal_q_bucket_edges")
    bucket_edges = [tuple(e) for e in _edges_raw] if _edges_raw else None
    price_df = predict_prices(
        models, cat_maps, active,
        conformal_q=conformal_q,
        oof_preds=oof_preds,
        median_calibrator=calibrator,
        text_pipeline=text_pipeline,
        conformal_q_per_bucket=per_bucket_q,
        conformal_q_bucket_edges=bucket_edges,
    )
    importance = compute_permutation_importance(
        models, cat_maps, active, text_pipeline=text_pipeline,
    )
    fill_rate = compute_feature_completeness(active)

    active = active.join(price_df)
    active["fill_rate"] = fill_rate

    return {
        "active": active,
        "models": models,
        "cat_maps": cat_maps,
        "metrics": metrics,
        "importance": importance,
        "oof_preds": oof_preds,
        "calibrator": calibrator,
        "text_pipeline": text_pipeline,
    }


@st.cache_data(ttl=600, show_spinner="Computing deal signals...")
def get_signals(_listings_df, _history_df):
    """Compute flip-score signals (loads saved model, never trains from scratch)."""
    from data_loader import compute_signals
    return compute_signals(_listings_df, _history_df)


def _prepare_active(listings):
    """Prepare active listings with generation + turnover for model input."""
    from src.models.generations import get_generation

    active = listings[listings["is_active"]].copy() if "is_active" in listings.columns else listings.copy()
    if "duplicate_of" in active.columns:
        active = active[active["duplicate_of"].isna()].copy()

    active["generation"] = active.apply(
        lambda r: get_generation(r["brand"], r["model"], r.get("year")), axis=1
    )

    turnover = compute_turnover_stats(listings)
    liq = {}
    if not turnover.empty:
        for _, row in turnover.iterrows():
            days = row.get("avg_days_to_sell")
            if pd.notna(days):
                liq[(row["brand"], row["model"], row.get("generation"))] = float(days)
    if liq:
        active["avg_days_to_sell"] = active.apply(
            lambda r: liq.get((r["brand"], r["model"], r.get("generation"))), axis=1
        )
    return active


listings, history, turnover, competition = get_data()

if listings.empty:
    st.error("No data yet. Scraper runs every 8 hours via GitHub Actions.")
    st.stop()

active = listings[listings["is_active"]].copy() if "is_active" in listings.columns else listings.copy()
inactive = listings[~listings["is_active"]].copy() if "is_active" in listings.columns else pd.DataFrame()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("OLX Car Market Analytics")
st.caption("Data-driven insights from Portuguese car market")

# KPI row
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Active Listings", f"{len(active):,}")
k2.metric("Brands", f"{active['brand'].nunique()}")
median_price = active["price_eur"].median()
k3.metric("Median Price", f"{median_price:,.0f} EUR" if pd.notna(median_price) else "N/A")
avg_year = active["year"].median()
k4.metric("Median Year", f"{int(avg_year)}" if pd.notna(avg_year) else "N/A")
avg_km = active["mileage_km"].median()
k5.metric("Median Mileage", f"{int(avg_km):,} km" if pd.notna(avg_km) else "N/A")

st.markdown("---")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_market, tab_price_model, tab_flip, tab_llm = st.tabs([
    "Market Overview",
    "Price Model (LightGBM)",
    "Flip Score Engine",
    "LLM Enrichment",
])

# =====================================================================
# TAB 1: Market Overview
# =====================================================================
with tab_market:
    st.header("Market Overview")

    # Row 1: Brand distribution + Fuel type
    col1, col2 = st.columns(2)

    with col1:
        brand_counts = active["brand"].value_counts().head(20)
        fig = go.Figure(go.Bar(
            x=brand_counts.values[::-1],
            y=brand_counts.index[::-1],
            orientation="h",
            marker=dict(
                color=brand_counts.values[::-1],
                colorscale=[[0, "#073B4C"], [0.5, "#118AB2"], [1, "#FF6B35"]],
            ),
            text=brand_counts.values[::-1],
            textposition="outside",
            textfont=dict(size=11),
        ))
        _apply_layout(fig, title="Top 20 Brands by Listings", height=550)
        fig.update_xaxes(title_text="Number of Listings")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fuel_counts = active["fuel_type"].value_counts()
        # Group minor hybrids
        fuel_grouped = {}
        for f, c in fuel_counts.items():
            if not f:
                continue
            key = f
            fl = str(f).lower()
            if "híbrido" in fl and "plug" not in fl:
                key = "Hybrid"
            elif "plug" in fl.lower():
                key = "Plug-in Hybrid"
            fuel_grouped[key] = fuel_grouped.get(key, 0) + c
        fuel_df = pd.DataFrame(list(fuel_grouped.items()), columns=["Fuel", "Count"])
        fuel_df = fuel_df.sort_values("Count", ascending=False)

        fig = go.Figure(go.Pie(
            labels=fuel_df["Fuel"],
            values=fuel_df["Count"],
            hole=0.45,
            textinfo="label+percent",
            textposition="outside",
            marker=dict(colors=_COLORS[:len(fuel_df)]),
            pull=[0.05 if i == 0 else 0 for i in range(len(fuel_df))],
        ))
        _apply_layout(fig, title="Fuel Type Distribution", height=550, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Row 2: Price by brand + Year distribution
    col3, col4 = st.columns(2)

    with col3:
        top_brands = active["brand"].value_counts().head(15).index.tolist()
        brand_price = active[active["brand"].isin(top_brands) & active["price_eur"].notna()].copy()
        brand_medians = brand_price.groupby("brand")["price_eur"].median().sort_values()
        brand_price["brand"] = pd.Categorical(brand_price["brand"], categories=brand_medians.index, ordered=True)

        fig = go.Figure()
        for i, brand in enumerate(brand_medians.index):
            data = brand_price[brand_price["brand"] == brand]["price_eur"]
            fig.add_trace(go.Box(
                y=data,
                name=brand,
                marker_color=_COLORS[i % len(_COLORS)],
                boxmean=True,
                line_width=1.5,
            ))
        _apply_layout(fig, title="Price Distribution by Brand", height=500, showlegend=False)
        fig.update_yaxes(title_text="Price (EUR)", tickformat=",")
        fig.update_xaxes(tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        year_data = active[active["year"].notna() & (active["year"] >= 1990)].copy()
        year_counts = year_data["year"].value_counts().sort_index()

        fig = go.Figure(go.Bar(
            x=year_counts.index.astype(int),
            y=year_counts.values,
            marker=dict(
                color=year_counts.values,
                colorscale=[[0, "#073B4C"], [0.5, "#00B4D8"], [1, "#FF6B35"]],
            ),
        ))
        _apply_layout(fig, title="Listings by Year", height=500)
        fig.update_xaxes(title_text="Year", dtick=2)
        fig.update_yaxes(title_text="Count")
        st.plotly_chart(fig, use_container_width=True)

    # Row 3: Price vs Year scatter + Price vs Mileage scatter
    col5, col6 = st.columns(2)

    with col5:
        scatter_data = active[
            active["price_eur"].notna()
            & active["year"].notna()
            & (active["price_eur"] > 500)
            & (active["price_eur"] < 100_000)
            & (active["year"] >= 2000)
        ].copy()

        fuel_map = {}
        for f in scatter_data["fuel_type"].dropna().unique():
            fl = str(f).lower()
            if "diesel" in fl:
                fuel_map[f] = "Diesel"
            elif "eléctrico" in fl or "electr" in fl:
                fuel_map[f] = "Electric"
            elif "híbrido" in fl or "hybrid" in fl:
                fuel_map[f] = "Hybrid"
            else:
                fuel_map[f] = "Petrol"
        scatter_data["fuel_group"] = scatter_data["fuel_type"].map(fuel_map).fillna("Other")

        fig = px.scatter(
            scatter_data,
            x="year", y="price_eur",
            color="fuel_group",
            opacity=0.4,
            color_discrete_sequence=["#FF6B35", "#00B4D8", "#2EC4B6", "#FFD166", "#8338EC"],
            hover_data=["brand", "model"],
        )
        _apply_layout(fig, title="Price vs Year (by Fuel Type)", height=500)
        fig.update_xaxes(title_text="Year")
        fig.update_yaxes(title_text="Price (EUR)", tickformat=",")
        fig.update_traces(marker_size=5)
        st.plotly_chart(fig, use_container_width=True)

    with col6:
        scatter_km = active[
            active["price_eur"].notna()
            & active["mileage_km"].notna()
            & (active["price_eur"] > 500)
            & (active["price_eur"] < 100_000)
            & (active["mileage_km"] > 0)
            & (active["mileage_km"] < 400_000)
        ].copy()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=scatter_km["mileage_km"],
            y=scatter_km["price_eur"],
            mode="markers",
            marker=dict(color="#00B4D8", size=4, opacity=0.35),
            text=scatter_km.apply(
                lambda r: f"{r['brand']} {r['model']} ({int(r['year']) if pd.notna(r.get('year')) else '?'})",
                axis=1,
            ),
            hoverinfo="text+x+y",
            name="Listings",
        ))
        # Rolling median trend line
        km_sorted = scatter_km.sort_values("mileage_km")
        trend_y = km_sorted["price_eur"].rolling(window=max(len(km_sorted) // 20, 10), center=True, min_periods=5).median()
        fig.add_trace(go.Scatter(
            x=km_sorted["mileage_km"], y=trend_y,
            mode="lines", line=dict(color="#FF6B35", width=3),
            name="Trend (rolling median)",
        ))
        _apply_layout(fig, title="Price vs Mileage", height=500)
        fig.update_xaxes(title_text="Mileage (km)", tickformat=",")
        fig.update_yaxes(title_text="Price (EUR)", tickformat=",")
        st.plotly_chart(fig, use_container_width=True)

    # Row 4: Turnover + Segment distribution
    col7, col8 = st.columns(2)

    with col7:
        if not turnover.empty:
            turn = turnover[turnover["avg_days_to_sell"].notna()].copy()
            turn["label"] = turn["brand"] + " " + turn["model"]
            turn = turn.sort_values("avg_days_to_sell").head(25)
            fig = go.Figure(go.Bar(
                x=turn["avg_days_to_sell"],
                y=turn["label"],
                orientation="h",
                marker=dict(
                    color=turn["avg_days_to_sell"],
                    colorscale=[[0, "#06D6A0"], [0.5, "#FFD166"], [1, "#E63946"]],
                ),
                text=turn["avg_days_to_sell"].round(0).astype(int),
                textposition="outside",
            ))
            _apply_layout(fig, title="Fastest Selling Models (avg days)", height=600)
            fig.update_xaxes(title_text="Days to Sell")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No turnover data available.")

    with col8:
        if "segment" in active.columns:
            seg_counts = active["segment"].value_counts().head(10)
            seg_counts = seg_counts[seg_counts.index.notna()]
            fig = go.Figure(go.Bar(
                x=seg_counts.index,
                y=seg_counts.values,
                marker=dict(color=_COLORS[:len(seg_counts)]),
                text=seg_counts.values,
                textposition="outside",
            ))
            _apply_layout(fig, title="Vehicle Segments", height=600)
            fig.update_xaxes(title_text="Segment", tickangle=-30)
            fig.update_yaxes(title_text="Count")
            st.plotly_chart(fig, use_container_width=True)

    # Row 5: Market history
    if not history.empty and "date" in history.columns:
        st.subheader("Market Price Trends")
        hist = history.copy()
        hist["date"] = pd.to_datetime(hist["date"])

        top5_brands = active["brand"].value_counts().head(5).index.tolist()
        hist_top = hist[hist["brand"].isin(top5_brands)]

        daily = (
            hist_top.groupby(["date", "brand"])
            .agg(median_price=("median_price_eur", "mean"), count=("listing_count", "sum"))
            .reset_index()
        )

        fig = px.line(
            daily, x="date", y="median_price", color="brand",
            color_discrete_sequence=_COLORS,
            markers=True,
        )
        _apply_layout(fig, title="Median Price Trend — Top 5 Brands", height=400)
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Median Price (EUR)", tickformat=",")
        fig.update_traces(line_width=2.5, marker_size=5)
        st.plotly_chart(fig, use_container_width=True)


# =====================================================================
# TAB 2: Price Model (LightGBM)
# =====================================================================
with tab_price_model:
    st.header("LightGBM Quantile Regression — Price Model")

    st.markdown("""
    The price model uses **three LightGBM quantile regressors** trained on all active listings:
    - **P10 (Low)** — 10th percentile: floor price estimate
    - **P50 (Median)** — predicted fair market price
    - **P90 (High)** — 90th percentile: ceiling price estimate

    Uses **Conformal Quantile Regression (CQR)** to calibrate the prediction band for guaranteed 80% coverage.
    """)

    # Load model data — prefer saved model; train only on explicit user request
    _has_saved = load_model() is not None
    if not _has_saved:
        st.warning("No trained model found. Train it first (takes ~2 min on 5k listings).")
        if st.button("Train model now", key="train_price"):
            with st.spinner("Training LightGBM quantile models..."):
                _model_active = _prepare_active(listings)
                model_data = get_model_data(_model_active, listings)
        else:
            model_data = None
    else:
        _model_active = _prepare_active(listings)
        model_data = get_model_data(_model_active, listings)

    _show_model = model_data is not None
    if not _show_model:
        st.info("Model metrics and charts will appear here after training.")

    if _show_model:
        m_active = model_data["active"]
        metrics = model_data["metrics"]
        importance = model_data["importance"]

        # Metrics row
        st.subheader("Model Performance (5-fold CV)")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("MAE", f"{metrics['mae']:,.0f} EUR")
        m2.metric("MAPE", f"{metrics['mape']:.1f}%")
        m3.metric("R\u00b2", f"{metrics['r2']:.3f}")
        cov = metrics.get("coverage_80_calibrated") or metrics.get("coverage_80", 0)
        m4.metric("80% Band Coverage", f"{cov:.1%}")
        m5.metric("Trees (early-stop)", f"{metrics.get('best_n_estimators', '?')}")

        # conformal_q is now in log space (multiplicative band widening); show
        # the percent equivalent so the dashboard caption still reads in human
        # units. Older bundles without conformal_q_pct fall back to the raw
        # number with a "(log)" suffix so it isn't mistaken for euros.
        _q_pct = metrics.get("conformal_q_pct")
        if _q_pct is not None:
            _q_str = f"±{_q_pct:.1f}%"
        else:
            _q_str = f"{metrics.get('conformal_q', 0):.2f} (log)"
        # Highlight when CQR is calibrated time-honestly (the trustworthy mode
        # — random-KFold mixes time-adjacent rows and under-estimates the q).
        _q_source = metrics.get("conformal_q_source", "random")
        _q_source_str = (
            "time-aware" if _q_source == "time"
            else "[yellow]random-KFold (no first_seen_at)[/yellow]"
        )
        _calib_str = (
            " | median: isotonic-calibrated"
            if metrics.get("median_calibrated") else ""
        )
        st.caption(
            f"Trained on **{metrics['n_samples']:,}** samples "
            f"| Conformal Q = {_q_str} ({_q_source_str}) "
            f"| {metrics.get('cv_folds', 5)}-fold CV"
            f"{_calib_str}"
        )

        # Row: Predicted vs Actual + Residuals
        col_pva, col_res = st.columns(2)

        pred_data = m_active[
            m_active["predicted_price"].notna()
            & m_active["price_eur"].notna()
            & (m_active["price_eur"] > 0)
        ].copy()
        pred_data["residual"] = pred_data["predicted_price"] - pred_data["price_eur"]
        pred_data["residual_pct"] = (pred_data["residual"] / pred_data["price_eur"] * 100).round(1)

        with col_pva:
            max_price = pred_data[["price_eur", "predicted_price"]].max().max()
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=[0, max_price], y=[0, max_price],
                mode="lines", line=dict(color="#FFD166", width=2, dash="dash"),
                name="Perfect prediction",
                showlegend=True,
            ))

            fig.add_trace(go.Scatter(
                x=pred_data["price_eur"],
                y=pred_data["predicted_price"],
                mode="markers",
                marker=dict(
                    color=pred_data["residual_pct"].clip(-50, 50),
                    colorscale=[[0, "#E63946"], [0.5, "#FAFAFA"], [1, "#06D6A0"]],
                    size=5, opacity=0.6,
                    colorbar=dict(title="Residual %", ticksuffix="%"),
                ),
                text=pred_data.apply(
                    lambda r: f"{r['brand']} {r['model']}<br>Actual: {r['price_eur']:,.0f}<br>Predicted: {r['predicted_price']:,.0f}",
                    axis=1,
                ),
                hoverinfo="text",
                name="Listings",
            ))

            _apply_layout(fig, title="Predicted vs Actual Price", height=500)
            fig.update_xaxes(title_text="Actual Price (EUR)", tickformat=",")
            fig.update_yaxes(title_text="Predicted Price (EUR)", tickformat=",")
            st.plotly_chart(fig, use_container_width=True)

        with col_res:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=pred_data["residual_pct"].clip(-100, 100),
                nbinsx=60,
                marker=dict(
                    color="rgba(0, 180, 216, 0.7)",
                    line=dict(color="#00B4D8", width=0.5),
                ),
                name="Residuals",
            ))
            fig.add_vline(x=0, line_dash="dash", line_color="#FFD166", line_width=2)

            median_res = pred_data["residual_pct"].median()
            fig.add_vline(x=median_res, line_dash="dot", line_color="#FF6B35", line_width=1.5,
                           annotation_text=f"median: {median_res:.1f}%", annotation_position="top")

            _apply_layout(fig, title="Residual Distribution (%)", height=500)
            fig.update_xaxes(title_text="(Predicted - Actual) / Actual (%)")
            fig.update_yaxes(title_text="Count")
            st.plotly_chart(fig, use_container_width=True)

        # Row: Feature Importance + Quantile Bands
        col_imp, col_bands = st.columns(2)

        with col_imp:
            if not importance.empty:
                imp = importance[importance["median_importance"] > 0].head(20).copy()
                imp = imp.sort_values("median_importance")

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=imp["median_importance"],
                    y=imp["feature"],
                    orientation="h",
                    name="Median model",
                    marker=dict(color="#FF6B35"),
                ))
                fig.add_trace(go.Bar(
                    x=imp["low_importance"],
                    y=imp["feature"],
                    orientation="h",
                    name="Low (P10)",
                    marker=dict(color="#00B4D8", opacity=0.7),
                ))
                fig.add_trace(go.Bar(
                    x=imp["high_importance"],
                    y=imp["feature"],
                    orientation="h",
                    name="High (P90)",
                    marker=dict(color="#2EC4B6", opacity=0.7),
                ))

                _apply_layout(fig, title="Permutation Feature Importance", height=600,
                             barmode="group")
                fig.update_xaxes(title_text="Importance (MAE increase on permutation)")
                st.plotly_chart(fig, use_container_width=True)

        with col_bands:
            band_data = pred_data[
                pred_data["fair_price_low"].notna() & pred_data["fair_price_high"].notna()
            ].copy()
            band_data = band_data.sort_values("price_eur").reset_index(drop=True)

            if len(band_data) > 200:
                step = len(band_data) // 200
                band_data = band_data.iloc[::step]

            x_range = list(range(len(band_data)))

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=x_range + x_range[::-1],
                y=list(band_data["fair_price_high"]) + list(band_data["fair_price_low"])[::-1],
                fill="toself",
                fillcolor="rgba(0, 180, 216, 0.15)",
                line=dict(color="rgba(0,0,0,0)"),
                name="P10\u2013P90 band",
                hoverinfo="skip",
            ))

            fig.add_trace(go.Scatter(
                x=x_range, y=band_data["price_eur"],
                mode="markers", marker=dict(color="#E63946", size=4, opacity=0.7),
                name="Actual price",
            ))

            fig.add_trace(go.Scatter(
                x=x_range, y=band_data["predicted_price"],
                mode="markers", marker=dict(color="#FFD166", size=3, opacity=0.5),
                name="Predicted (P50)",
            ))

            _apply_layout(fig, title="Prediction Bands: P10 / P50 / P90", height=600)
            fig.update_xaxes(title_text="Listings (sorted by actual price)", showticklabels=False)
            fig.update_yaxes(title_text="Price (EUR)", tickformat=",")
            st.plotly_chart(fig, use_container_width=True)

    # Row: Metrics history (always show if available, even without fresh model)
    metrics_hist = load_metrics_history()
    if len(metrics_hist) > 1:
        st.subheader("Model Quality Over Time")
        hist_df = pd.DataFrame(metrics_hist)
        hist_df["timestamp"] = pd.to_datetime(hist_df["timestamp"])

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("MAPE (%)", "R\u00b2", "MAE (EUR)", "80% Coverage"),
            vertical_spacing=0.12,
            horizontal_spacing=0.08,
        )

        for metric, row, col, color in [
            ("mape", 1, 1, "#FF6B35"),
            ("r2", 1, 2, "#00B4D8"),
            ("mae", 2, 1, "#2EC4B6"),
            ("coverage_80_calibrated", 2, 2, "#FFD166"),
        ]:
            if metric in hist_df.columns:
                fig.add_trace(go.Scatter(
                    x=hist_df["timestamp"], y=hist_df[metric],
                    mode="lines+markers",
                    marker=dict(color=color, size=8),
                    line=dict(color=color, width=2.5),
                    name=metric.upper(),
                ), row=row, col=col)

        fig.update_layout(
            paper_bgcolor=_PAPER, plot_bgcolor=_BG, font=_FONT,
            height=500, showlegend=False,
            margin=dict(l=40, r=20, t=60, b=40),
        )
        for ax in fig.select_xaxes():
            ax.update(gridcolor=_GRID)
        for ax in fig.select_yaxes():
            ax.update(gridcolor=_GRID)
        st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------------------
    # Model Diagnostics — slice metrics, reliability, worst residuals,
    # time backtest. All read from the bundled OOF predictions, so the
    # numbers here reflect the actual cross-validated quality, not
    # in-sample memorization.
    # ---------------------------------------------------------------
    if _show_model and model_data.get("oof_preds"):
        st.subheader("Model Diagnostics")
        _oof = model_data["oof_preds"]
        _eval = evaluate_oof(m_active, _oof)
        _g = _eval["global"]

        if _g["n"] == 0:
            st.info(
                "No overlap between active listings and bundled OOF predictions — "
                "the DB may have rotated since the last training run."
            )
        else:
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("OOF samples", f"{_g['n']:,}")
            d2.metric("Bias", f"{_g['bias_pct']:+.2f}%",
                      help="Mean of (actual − predicted) / actual. Positive = model under-predicts on average.")
            d3.metric("Coverage(80%)", f"{_g['coverage_80']:.1%}",
                      help="Fraction of true prices inside the [P10, P90] band. Target 80%.")
            d4.metric("Inverted bands", f"{_g['n_inverted_band']}",
                      help="Rows where low > median or median > high. Should be 0 — sort fixes this.")

            # Bucket tables side by side
            tcol1, tcol2 = st.columns(2)
            with tcol1:
                st.markdown("**By price bucket**")
                if not _eval["by_price"].empty:
                    st.dataframe(
                        _eval["by_price"].style.format({
                            "n": "{:,}",
                            "mae": "{:,.0f}",
                            "mape": "{:.1f}",
                            "bias_pct": "{:+.2f}",
                            "coverage_80": "{:.1%}",
                        }),
                        hide_index=True, use_container_width=True,
                    )
            with tcol2:
                st.markdown("**By year bucket**")
                if not _eval["by_year"].empty:
                    st.dataframe(
                        _eval["by_year"].style.format({
                            "n": "{:,}",
                            "mae": "{:,.0f}",
                            "mape": "{:.1f}",
                            "bias_pct": "{:+.2f}",
                            "coverage_80": "{:.1%}",
                        }),
                        hide_index=True, use_container_width=True,
                    )

            st.markdown("**By brand (top 10 by sample count)**")
            if not _eval["by_brand"].empty:
                st.dataframe(
                    _eval["by_brand"].style.format({
                        "n": "{:,}",
                        "mae": "{:,.0f}",
                        "mape": "{:.1f}",
                        "bias_pct": "{:+.2f}",
                        "coverage_80": "{:.1%}",
                    }),
                    hide_index=True, use_container_width=True,
                )

            # Reliability curve
            rel = reliability_curve(m_active, _oof, n_bins=10)
            if not rel.empty:
                rel_col, worst_col = st.columns([3, 2])
                with rel_col:
                    fig = go.Figure()
                    # Nominal target line at 0.80
                    fig.add_hline(
                        y=0.80, line_dash="dash", line_color="#FFD166", line_width=2,
                        annotation_text="target 80%", annotation_position="top right",
                    )
                    fig.add_trace(go.Scatter(
                        x=rel["predicted_mean"],
                        y=rel["empirical_coverage"],
                        mode="lines+markers",
                        line=dict(color="#00B4D8", width=2.5),
                        marker=dict(
                            color="#00B4D8",
                            size=rel["n"].clip(lower=10) ** 0.5,
                            sizemode="area",
                            sizeref=0.05,
                        ),
                        text=rel.apply(
                            lambda r: (
                                f"€{r['predicted_min']:,.0f}–{r['predicted_max']:,.0f}<br>"
                                f"n={int(r['n']):,}<br>"
                                f"empirical: {r['empirical_coverage']:.1%}<br>"
                                f"gap: {r['calibration_gap']:+.3f}"
                            ),
                            axis=1,
                        ),
                        hoverinfo="text",
                        name="Empirical coverage",
                    ))
                    _apply_layout(fig, title="Reliability curve (coverage by predicted-price decile)",
                                 height=400, showlegend=False)
                    fig.update_xaxes(title_text="Mean predicted price (EUR)", tickformat=",")
                    fig.update_yaxes(title_text="Empirical 80% coverage",
                                     tickformat=".0%", range=[0, 1])
                    st.plotly_chart(fig, use_container_width=True)

                with worst_col:
                    st.markdown("**Top 10 worst residuals (|Δ %|)**")
                    worst = worst_residuals(m_active, _oof, n=10)
                    if not worst.empty:
                        display_cols = [c for c in (
                            "olx_id", "brand", "model", "year",
                            "price_eur", "oof_median", "abs_residual_pct", "in_band",
                        ) if c in worst.columns]
                        st.dataframe(
                            worst[display_cols].style.format({
                                "year": "{:.0f}",
                                "price_eur": "{:,.0f}",
                                "oof_median": "{:,.0f}",
                                "abs_residual_pct": "{:.1f}",
                            }),
                            hide_index=True, use_container_width=True,
                        )

        # Time backtest panel — shown only if a backtest run was persisted
        # (CLI: python -m src.cli eval-model --time-backtest).
        _bt = load_backtest()
        if _bt and _bt.get("folds"):
            st.markdown("---")
            st.markdown("**Time-aware backtest** — train on rolling window, test on next slice")
            st.caption(f"Generated {_bt.get('generated_at', '')[:19]} UTC")

            bt_df = pd.DataFrame(_bt["folds"])
            bt_left, bt_right = st.columns([2, 3])
            with bt_left:
                st.dataframe(
                    bt_df[[
                        "fold", "test_from", "test_to",
                        "n_train", "n_test", "mae", "mape", "coverage_80",
                    ]].style.format({
                        "n_train": "{:,}",
                        "n_test": "{:,}",
                        "mae": "{:,.0f}",
                        "mape": "{:.1f}",
                        "coverage_80": "{:.1%}",
                    }),
                    hide_index=True, use_container_width=True,
                )
            with bt_right:
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=("MAPE per fold (%)", "80% coverage per fold"),
                    horizontal_spacing=0.15,
                )
                fig.add_trace(go.Bar(
                    x=bt_df["fold"], y=bt_df["mape"],
                    marker_color="#FF6B35", name="MAPE",
                ), row=1, col=1)
                fig.add_trace(go.Bar(
                    x=bt_df["fold"], y=bt_df["coverage_80"],
                    marker_color="#00B4D8", name="Coverage",
                ), row=1, col=2)
                fig.add_hline(
                    y=0.80, line_dash="dash", line_color="#FFD166",
                    line_width=1.5, row=1, col=2,
                )
                fig.update_layout(
                    paper_bgcolor=_PAPER, plot_bgcolor=_BG, font=_FONT,
                    height=300, showlegend=False,
                    margin=dict(l=40, r=20, t=50, b=40),
                )
                for ax in fig.select_xaxes():
                    ax.update(gridcolor=_GRID, title_text="Fold")
                for ax in fig.select_yaxes():
                    ax.update(gridcolor=_GRID)
                fig.update_yaxes(tickformat=".0%", range=[0, 1], row=1, col=2)
                st.plotly_chart(fig, use_container_width=True)
        elif _show_model:
            st.caption(
                "Time backtest not yet generated. Run "
                "`python -m src.cli eval-model --time-backtest` to populate "
                "`data/price_backtest.json`."
            )

    # Feature category breakdown
    st.subheader("Model Architecture")
    fc1, fc2, fc3 = st.columns(3)
    fc1.markdown(f"**Numeric features** ({len(NUMERIC_FEATURES)})")
    fc1.code("\n".join(NUMERIC_FEATURES), language=None)
    fc2.markdown(f"**Boolean features** ({len(BOOL_FEATURES)})")
    fc2.code("\n".join(BOOL_FEATURES), language=None)
    fc3.markdown(f"**Categorical features** ({len(CATEGORICAL_FEATURES)})")
    fc3.code("\n".join(CATEGORICAL_FEATURES), language=None)


# =====================================================================
# TAB 3: Flip Score Engine
# =====================================================================
with tab_flip:
    st.header("Flip Score — Deal Detection Engine")

    st.markdown("""
    The flip score ranks deals by **profit potential**. It combines ML-predicted
    undervaluation with 7 market opportunity multipliers.

    **Formula:** `flip_score = undervaluation% x liquidity x trend x motivated x urgency x warranty x velocity x confidence`
    """)

    # Load signals — uses saved model, never trains from scratch during page load
    _has_saved_flip = load_model() is not None
    if not _has_saved_flip and not listings.empty:
        st.warning("No trained model found. Go to **Price Model** tab and train first.")
        signals_df = pd.DataFrame()
    elif not listings.empty:
        signals_df, _ = get_signals(listings, history)
    else:
        signals_df = pd.DataFrame()

    if signals_df.empty:
        st.info("No deals found. Need active listings with price data.")
    else:
        deals = signals_df.copy()
        deals["est_profit_eur"] = (deals["predicted_price"] - deals["price_eur"]).round(0)

        # KPI row
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Deals Found", f"{len(deals):,}")
        d2.metric("Max Flip Score", f"{deals['flip_score'].max():.0f}")
        d3.metric("Avg Discount", f"{deals['discount_pct'].mean():.1f}%")
        med_profit = deals["est_profit_eur"].median()
        d4.metric("Median Est. Profit", f"{med_profit:+,.0f} EUR" if pd.notna(med_profit) else "N/A")

        # Row: Flip score distribution + Top deals
        col_fs, col_td = st.columns(2)

        with col_fs:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=deals["flip_score"],
                nbinsx=40,
                marker=dict(
                    color="rgba(255, 107, 53, 0.7)",
                    line=dict(color="#FF6B35", width=0.5),
                ),
            ))
            med_score = deals["flip_score"].median()
            fig.add_vline(x=med_score, line_dash="dash", line_color="#FFD166",
                         annotation_text=f"median: {med_score:.0f}")
            _apply_layout(fig, title="Flip Score Distribution", height=450)
            fig.update_xaxes(title_text="Flip Score")
            fig.update_yaxes(title_text="Count")
            st.plotly_chart(fig, use_container_width=True)

        with col_td:
            top = deals.nlargest(15, "flip_score").copy()
            top["label"] = top["brand"] + " " + top["model"] + " " + top["year"].fillna(0).astype(int).astype(str)
            top = top.sort_values("flip_score")

            fig = go.Figure(go.Bar(
                x=top["flip_score"],
                y=top["label"],
                orientation="h",
                marker=dict(
                    color=top["flip_score"],
                    colorscale=[[0, "#FFD166"], [0.5, "#FF6B35"], [1, "#E63946"]],
                ),
                text=top["flip_score"].round(0).astype(int),
                textposition="outside",
            ))
            _apply_layout(fig, title="Top 15 Deals by Flip Score", height=450)
            fig.update_xaxes(title_text="Flip Score")
            st.plotly_chart(fig, use_container_width=True)

        # Row: Multiplier analysis
        st.subheader("Multiplier Analysis")

        mult_cols = [
            "liquidity_mult", "trend_mult", "motivated_mult",
            "urgency_mult", "warranty_mult", "velocity_mult", "confidence_mult",
        ]
        mult_labels = [
            "Liquidity", "Trend", "Motivated\nSeller", "Urgency",
            "Warranty", "Velocity", "Confidence",
        ]

        col_radar, col_heatmap = st.columns(2)

        with col_radar:
            # Radar chart — average multipliers for top 20 vs all deals
            top20 = deals.nlargest(20, "flip_score")
            avg_all = [deals[c].mean() for c in mult_cols]
            avg_top = [top20[c].mean() for c in mult_cols]

            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=avg_all + [avg_all[0]],
                theta=mult_labels + [mult_labels[0]],
                fill="toself",
                fillcolor="rgba(0, 180, 216, 0.15)",
                line=dict(color="#00B4D8", width=2),
                name="All deals",
            ))
            fig.add_trace(go.Scatterpolar(
                r=avg_top + [avg_top[0]],
                theta=mult_labels + [mult_labels[0]],
                fill="toself",
                fillcolor="rgba(255, 107, 53, 0.15)",
                line=dict(color="#FF6B35", width=2),
                name="Top 20 deals",
            ))

            fig.update_layout(
                polar=dict(
                    bgcolor=_BG,
                    radialaxis=dict(gridcolor=_GRID, color=_TEXT, range=[0, max(max(avg_all), max(avg_top)) * 1.1]),
                    angularaxis=dict(gridcolor=_GRID, color=_TEXT),
                ),
                paper_bgcolor=_PAPER, font=_FONT,
                title="Multiplier Profile: All vs Top 20",
                height=500,
                legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_heatmap:
            # Multiplier correlation with flip_score
            corr_data = deals[mult_cols + ["flip_score", "discount_pct"]].copy()
            corr_data.columns = mult_labels + ["Flip\nScore", "Discount\n%"]
            corr_matrix = corr_data.corr()

            fig = go.Figure(go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale=[[0, "#073B4C"], [0.5, "#0E1117"], [1, "#FF6B35"]],
                zmin=-1, zmax=1,
                text=corr_matrix.values.round(2),
                texttemplate="%{text}",
                textfont=dict(size=11),
            ))
            _apply_layout(fig, title="Multiplier Correlations", height=500)
            st.plotly_chart(fig, use_container_width=True)

        # Row: Undervaluation vs Flip Score + Brand breakdown
        col_uv, col_bb = st.columns(2)

        with col_uv:
            fig = px.scatter(
                deals,
                x="discount_pct", y="flip_score",
                color="confidence_mult",
                size="sample_size",
                color_continuous_scale=[[0, "#E63946"], [0.5, "#FFD166"], [1, "#06D6A0"]],
                hover_data=["brand", "model", "price_eur"],
                opacity=0.7,
            )
            _apply_layout(fig, title="Discount % vs Flip Score", height=450)
            fig.update_xaxes(title_text="Discount vs Median (%)")
            fig.update_yaxes(title_text="Flip Score")
            fig.update_coloraxes(colorbar_title="Confidence")
            st.plotly_chart(fig, use_container_width=True)

        with col_bb:
            brand_deals = deals.groupby("brand").agg(
                deals=("flip_score", "size"),
                avg_score=("flip_score", "mean"),
                avg_discount=("discount_pct", "mean"),
            ).reset_index()
            brand_deals = brand_deals[brand_deals["deals"] >= 3].sort_values("avg_score", ascending=False).head(15)

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=brand_deals["brand"],
                y=brand_deals["avg_score"],
                name="Avg Flip Score",
                marker=dict(color="#FF6B35"),
            ))
            fig.add_trace(go.Bar(
                x=brand_deals["brand"],
                y=brand_deals["avg_discount"],
                name="Avg Discount %",
                marker=dict(color="#00B4D8"),
            ))
            _apply_layout(fig, title="Brand Deal Quality (min 3 deals)", height=450, barmode="group")
            fig.update_xaxes(tickangle=-45)
            fig.update_yaxes(title_text="Score / %")
            st.plotly_chart(fig, use_container_width=True)

        # Waterfall: Flip score decomposition for the top deal
        st.subheader("Flip Score Decomposition — Top Deal")
        top_deal = deals.iloc[0]
        base_pct = top_deal["undervaluation_pct"] if top_deal["undervaluation_pct"] > 0 else top_deal["discount_pct"]

        waterfall_steps = [
            ("Base undervaluation", base_pct, "absolute"),
        ]
        running = base_pct
        for col, label in [
            ("liquidity_mult", "Liquidity"),
            ("trend_mult", "Trend"),
            ("motivated_mult", "Motivated"),
            ("urgency_mult", "Urgency"),
            ("warranty_mult", "Warranty"),
            ("velocity_mult", "Velocity"),
            ("confidence_mult", "Confidence"),
        ]:
            val = top_deal[col]
            effect = running * val - running
            running = running * val
            waterfall_steps.append((label, effect, "relative"))
        waterfall_steps.append(("Flip Score", top_deal["flip_score"], "total"))

        fig = go.Figure(go.Waterfall(
            x=[s[0] for s in waterfall_steps],
            y=[s[1] for s in waterfall_steps],
            measure=[s[2] for s in waterfall_steps],
            increasing=dict(marker_color="#06D6A0"),
            decreasing=dict(marker_color="#E63946"),
            totals=dict(marker_color="#FF6B35"),
            connector=dict(line=dict(color=_GRID, width=1)),
            text=[f"{s[1]:+.1f}" if s[2] == "relative" else f"{s[1]:.1f}" for s in waterfall_steps],
            textposition="outside",
        ))
        _apply_layout(fig,
            title=f"Score Breakdown: {top_deal['brand']} {top_deal['model']} ({int(top_deal['year']) if pd.notna(top_deal.get('year')) else '?'})",
            height=400,
        )
        fig.update_yaxes(title_text="Score contribution")
        st.plotly_chart(fig, use_container_width=True)


# =====================================================================
# TAB 4: LLM Enrichment
# =====================================================================
with tab_llm:
    st.header("LLM Enrichment — Ollama Text Extraction")

    st.markdown("""
    A local **Ollama LLM** (qwen3:4b-instruct) extracts **14 structured fields** from
    Portuguese listing descriptions. These feed both the price model and deal warnings.

    **Pipeline:** Description text → Ollama API → JSON extraction → Data correction → DB storage
    """)

    # Enrichment stats
    has_llm = active["llm_extras"].notna() & (active["llm_extras"] != "")
    enriched_count = has_llm.sum()
    total_count = len(active)

    l1, l2, l3, l4 = st.columns(4)
    l1.metric("LLM Enriched", f"{enriched_count:,}")
    l2.metric("Total Active", f"{total_count:,}")
    l3.metric("Coverage", f"{enriched_count / total_count * 100:.1f}%" if total_count > 0 else "0%")

    # Count listings with meaningful descriptions (length is precomputed at
    # scrape time — cheaper than keeping the raw description text in the df).
    desc_len = pd.to_numeric(active.get("description_length"), errors="coerce")
    has_desc = desc_len.fillna(0) > 20
    l4.metric("With Description", f"{has_desc.sum():,}")

    # LLM fields coverage
    llm_fields = [
        "sub_model", "trim_level", "desc_mentions_accident", "desc_mentions_repair",
        "desc_mentions_num_owners", "desc_mentions_customs_cleared", "right_hand_drive",
        "mechanical_condition", "urgency", "warranty", "tuning_or_mods",
        "taxi_fleet_rental", "first_owner_selling", "real_mileage_km",
    ]

    col_cov, col_cat = st.columns(2)

    with col_cov:
        fill_rates = []
        for field in llm_fields:
            if field in active.columns:
                if active[field].dtype == bool or active[field].dtype == "boolean":
                    rate = active[field].notna().mean()
                else:
                    rate = (active[field].notna() & (active[field] != "") & (active[field] != 0)).mean()
            else:
                rate = 0.0
            fill_rates.append({"Field": field, "Fill Rate": rate})

        fill_df = pd.DataFrame(fill_rates).sort_values("Fill Rate", ascending=True)

        fig = go.Figure(go.Bar(
            x=fill_df["Fill Rate"],
            y=fill_df["Field"],
            orientation="h",
            marker=dict(
                color=fill_df["Fill Rate"],
                colorscale=[[0, "#E63946"], [0.3, "#FFD166"], [1, "#06D6A0"]],
            ),
            text=(fill_df["Fill Rate"] * 100).round(1).astype(str) + "%",
            textposition="outside",
        ))
        _apply_layout(fig, title="LLM Field Fill Rate", height=550)
        fig.update_xaxes(title_text="Fill Rate", tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    with col_cat:
        # Condition distribution
        cond_data = {}

        # Urgency
        if "urgency" in active.columns:
            urg = active["urgency"].value_counts()
            cond_data["Urgency"] = urg.to_dict()

        # Mechanical condition
        if "mechanical_condition" in active.columns:
            mc = active["mechanical_condition"].value_counts()
            cond_data["Condition"] = mc.to_dict()

        # Boolean flags
        bool_flags = {
            "Accident": "desc_mentions_accident",
            "Repair": "desc_mentions_repair",
            "Warranty": "warranty",
            "1st Owner": "first_owner_selling",
            "RHD": "right_hand_drive",
            "Customs": "desc_mentions_customs_cleared",
            "Taxi/Fleet": "taxi_fleet_rental",
        }
        true_counts = {}
        false_counts = {}
        for label, col in bool_flags.items():
            if col in active.columns:
                true_counts[label] = int((active[col] == True).sum())
                false_counts[label] = int((active[col] == False).sum())

        labels = list(true_counts.keys())
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=labels,
            y=[true_counts[l] for l in labels],
            name="True",
            marker=dict(color="#FF6B35"),
        ))
        fig.add_trace(go.Bar(
            x=labels,
            y=[false_counts[l] for l in labels],
            name="False",
            marker=dict(color="#00B4D8"),
        ))
        _apply_layout(fig, title="Boolean Flag Distribution", height=550, barmode="stack")
        fig.update_xaxes(tickangle=-30)
        fig.update_yaxes(title_text="Listings")
        st.plotly_chart(fig, use_container_width=True)

    # Row: Mileage correction + Urgency breakdown
    col_km, col_urg = st.columns(2)

    with col_km:
        # Mileage: attribute vs description (where both exist)
        km_data = active[
            active["mileage_km"].notna()
            & active["real_mileage_km"].notna()
            & (active["mileage_km"] > 0)
            & (active["real_mileage_km"] > 0)
        ].copy()

        if not km_data.empty:
            km_data["mileage_diff"] = km_data["real_mileage_km"] - km_data["mileage_km"]
            km_data["mileage_diff_pct"] = (km_data["mileage_diff"] / km_data["mileage_km"] * 100).round(1)

            fig = go.Figure()

            # Perfect match line
            max_km = km_data[["mileage_km", "real_mileage_km"]].max().max()
            fig.add_trace(go.Scatter(
                x=[0, max_km], y=[0, max_km],
                mode="lines", line=dict(color="#FFD166", width=2, dash="dash"),
                name="Match", showlegend=True,
            ))

            # Color by deviation
            fig.add_trace(go.Scatter(
                x=km_data["mileage_km"],
                y=km_data["real_mileage_km"],
                mode="markers",
                marker=dict(
                    color=km_data["mileage_diff_pct"].clip(-100, 100),
                    colorscale=[[0, "#06D6A0"], [0.5, "#FAFAFA"], [1, "#E63946"]],
                    size=7, opacity=0.7,
                    colorbar=dict(title="Diff %"),
                ),
                text=km_data.apply(
                    lambda r: f"{r['brand']} {r['model']}<br>Attr: {int(r['mileage_km']):,} km<br>Desc: {int(r['real_mileage_km']):,} km<br>Diff: {r['mileage_diff_pct']:+.0f}%",
                    axis=1,
                ),
                hoverinfo="text",
                name="Listings",
            ))

            _apply_layout(fig, title="Mileage: Attribute vs Description (LLM)", height=450)
            fig.update_xaxes(title_text="OLX Attribute (km)", tickformat=",")
            fig.update_yaxes(title_text="Description Mentions (km)", tickformat=",")

            # Stats
            divergent = (km_data["mileage_diff_pct"].abs() > 10).sum()
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"{divergent} listings ({divergent / len(km_data) * 100:.0f}%) have >10% mileage discrepancy")
        else:
            st.info("No listings with both attribute and description mileage.")

    with col_urg:
        if "urgency" in active.columns:
            urg_data = active[active["urgency"].notna()].copy()
            if not urg_data.empty:
                urg_price = urg_data.groupby("urgency").agg(
                    count=("price_eur", "size"),
                    median_price=("price_eur", "median"),
                    avg_discount=("price_change_pct", "mean"),
                ).reset_index()

                order = ["low", "medium", "high"]
                urg_price["urgency"] = pd.Categorical(urg_price["urgency"], categories=order, ordered=True)
                urg_price = urg_price.sort_values("urgency")

                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Bar(
                    x=urg_price["urgency"],
                    y=urg_price["count"],
                    name="Count",
                    marker=dict(color=["#06D6A0", "#FFD166", "#E63946"]),
                ), secondary_y=False)

                fig.add_trace(go.Scatter(
                    x=urg_price["urgency"],
                    y=urg_price["median_price"],
                    mode="lines+markers",
                    name="Median price",
                    line=dict(color="#00B4D8", width=3),
                    marker=dict(size=10),
                ), secondary_y=True)

                fig.update_layout(
                    paper_bgcolor=_PAPER, plot_bgcolor=_BG, font=_FONT,
                    title="Urgency Level Analysis",
                    height=450,
                    margin=dict(l=40, r=40, t=50, b=40),
                    legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
                )
                fig.update_xaxes(title_text="Urgency", gridcolor=_GRID)
                fig.update_yaxes(title_text="Listings", gridcolor=_GRID, secondary_y=False)
                fig.update_yaxes(title_text="Median Price (EUR)", gridcolor=_GRID, secondary_y=True, tickformat=",")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No urgency data available.")

    # Pipeline architecture diagram
    st.subheader("Enrichment Pipeline Architecture")

    fig = go.Figure()

    # Pipeline boxes
    boxes = [
        (0.5, 0.8, "OLX / StandVirtual\nScraper", "#118AB2"),
        (2.5, 0.8, "Raw Listing\n(title + description)", "#073B4C"),
        (4.5, 0.8, "Ollama LLM\n(qwen3:4b)", "#FF6B35"),
        (6.5, 0.8, "JSON Extraction\n(14 fields)", "#8338EC"),
        (8.5, 0.8, "Data Correction\n& Validation", "#2EC4B6"),
        (10.5, 0.8, "SQLite DB\nEnriched Listings", "#06D6A0"),
    ]

    for x, y, text, color in boxes:
        fig.add_shape(type="rect",
            x0=x - 0.9, y0=y - 0.35, x1=x + 0.9, y1=y + 0.35,
            fillcolor=color, opacity=0.85,
            line=dict(color="white", width=1.5),
        )
        fig.add_annotation(x=x, y=y, text=text,
            showarrow=False, font=dict(color="white", size=11, family="Inter"),
        )

    # Arrows
    for i in range(len(boxes) - 1):
        x_start = boxes[i][0] + 0.9
        x_end = boxes[i + 1][0] - 0.9
        fig.add_annotation(
            x=x_end, y=0.8, ax=x_start, ay=0.8,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True,
            arrowhead=3, arrowsize=1.5, arrowwidth=2,
            arrowcolor="#FAFAFA",
        )

    # Downstream arrows
    for x, y, text, color in [
        (4.5, 0.2, "Price Model\n(LightGBM)", "#E63946"),
        (6.5, 0.2, "Flip Score\nEngine", "#FFD166"),
        (8.5, 0.2, "Streamlit\nDashboard", "#00B4D8"),
    ]:
        fig.add_shape(type="rect",
            x0=x - 0.9, y0=y - 0.35, x1=x + 0.9, y1=y + 0.35,
            fillcolor=color, opacity=0.75,
            line=dict(color="white", width=1.5),
        )
        fig.add_annotation(x=x, y=y, text=text,
            showarrow=False, font=dict(color="white", size=11, family="Inter"),
        )

    # Downstream arrows
    fig.add_annotation(x=10.5, y=0.45, ax=10.5, ay=0.57,
        showarrow=True, arrowhead=3, arrowsize=1.5, arrowwidth=2, arrowcolor="#FAFAFA")
    # From DB down to Price Model
    fig.add_annotation(x=4.5, y=0.45, ax=10.5, ay=0.45,
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True, arrowhead=3, arrowsize=1, arrowwidth=1.5, arrowcolor="rgba(255,255,255,0.4)")
    # Price Model to Flip Score
    fig.add_annotation(x=6.5, y=0.2, ax=5.4, ay=0.2,
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True, arrowhead=3, arrowsize=1.5, arrowwidth=2, arrowcolor="#FAFAFA")
    # Flip Score to Dashboard
    fig.add_annotation(x=8.5, y=0.2, ax=7.4, ay=0.2,
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True, arrowhead=3, arrowsize=1.5, arrowwidth=2, arrowcolor="#FAFAFA")

    fig.update_layout(
        paper_bgcolor=_PAPER, plot_bgcolor=_BG,
        xaxis=dict(visible=False, range=[-0.5, 11.5]),
        yaxis=dict(visible=False, range=[-0.3, 1.3], scaleanchor="x"),
        height=300,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Extracted fields table
    st.subheader("Extracted Fields Reference")
    fields_data = [
        ("sub_model", "str", "Engine/body variant", "320d, 1.6 TDI, A 200"),
        ("trim_level", "str", "Equipment line", "AMG Line, M Sport, GTI"),
        ("desc_mentions_accident", "bool", "Collision/accident mentioned", "sinistro, acidente, batido"),
        ("desc_mentions_repair", "bool", "Repair/damage mentioned", "avariado, imobilizado"),
        ("desc_mentions_num_owners", "int", "Number of previous owners", "1 dono, 2 donos"),
        ("desc_mentions_customs_cleared", "bool", "Customs/legalization status", "desalfandegado, legalizado"),
        ("right_hand_drive", "bool", "Right-hand drive / UK import", "mao inglesa, volante a direita"),
        ("mechanical_condition", "enum", "Overall condition assessment", "excellent / good / fair / poor"),
        ("urgency", "enum", "Seller desperation level", "high / medium / low"),
        ("warranty", "bool", "Warranty mentioned", "garantia"),
        ("tuning_or_mods", "list", "Aftermarket modifications", "stage 1, remap, coilovers"),
        ("taxi_fleet_rental", "bool", "Commercial use history", "ex-taxi, TVDE, rent-a-car"),
        ("first_owner_selling", "bool", "Original owner selling", "1 dono desde novo"),
        ("mileage_in_description_km", "int", "Mileage from text", "150 mil km -> 150000"),
    ]
    st.dataframe(
        pd.DataFrame(fields_data, columns=["Field", "Type", "Description", "Examples"]),
        use_container_width=True,
        hide_index=True,
    )
