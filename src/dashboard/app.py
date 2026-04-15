"""OLX.pt Car Parser — Streamlit Dashboard (Deal Cards)."""

import numpy as np
import streamlit as st
import pandas as pd

import sys as _sys
from pathlib import Path as _Path
_dashboard_dir = _Path(__file__).resolve().parent
_project_root = _dashboard_dir.parent.parent
_sys.path.insert(0, str(_dashboard_dir))
_sys.path.insert(0, str(_project_root))

from data_loader import load_all, _force_next_check

st.set_page_config(page_title="Car Deals", layout="wide")


@st.cache_data(ttl=300)
def load_data():
    return load_all()


listings_df, history_df, signals_df, brands_models, turnover_df, _portfolio_init, _unmatched_df, importance_df = load_data()

# ---------------------------------------------------------------------------
# Sidebar — filters
# ---------------------------------------------------------------------------
st.sidebar.title("Filters")

if listings_df.empty:
    st.sidebar.warning("No data yet. Scraper runs every 8 hours via GitHub Actions.")
    st.stop()
else:
    st.sidebar.success(f"{len(listings_df)} listings")
    if st.sidebar.button("Refresh"):
        _force_next_check()
        st.cache_data.clear()
        st.rerun()

selected_brands = st.sidebar.multiselect(
    "Brand",
    options=sorted(brands_models.keys()),
)

available_models = []
if selected_brands:
    for b in selected_brands:
        available_models.extend(brands_models.get(b, []))
    available_models = sorted(set(available_models))

selected_models = st.sidebar.multiselect("Model", options=available_models) if available_models else []

year_min = int(listings_df["year"].min()) if listings_df["year"].notna().any() else 2000
year_max = int(listings_df["year"].max()) if listings_df["year"].notna().any() else 2026
year_range = st.sidebar.slider("Year", min_value=year_min, max_value=year_max, value=(year_min, year_max))

price_max_val = int(listings_df["price_eur"].max()) if listings_df["price_eur"].notna().any() else 50000
price_range = st.sidebar.slider("Price (EUR)", min_value=0, max_value=price_max_val, value=(0, price_max_val), step=500)

only_private = st.sidebar.checkbox("Private sellers only", value=False)
hide_accidents = st.sidebar.checkbox("Hide accidents", value=False)
hide_repair = st.sidebar.checkbox("Hide repair needed", value=False)

if not importance_df.empty:
    with st.sidebar.expander("Feature importance"):
        chart_df = importance_df[["feature", "median_importance"]].copy()
        chart_df = chart_df[chart_df["median_importance"] > 0]
        chart_df = chart_df.set_index("feature").sort_values("median_importance")
        st.bar_chart(chart_df)

from src.analytics.price_model import load_metrics_history

_metrics_history = load_metrics_history()
if _metrics_history:
    with st.sidebar.expander("Model quality"):
        _latest = _metrics_history[-1]
        _mq1, _mq2, _mq3 = st.columns(3)
        _mq1.metric("MAE", f"{_latest['mae']:,.0f} €")
        _mq2.metric("MAPE", f"{_latest['mape']:.1f}%")
        _mq3.metric("R²", f"{_latest['r2']:.2f}")
        _cov = _latest.get("coverage_80_calibrated") or _latest.get("coverage_80")
        if _cov is not None:
            _mq4, _mq5 = st.columns(2)
            _mq4.metric("80% coverage (CQR)", f"{_cov:.1%}", delta=f"{(_cov - 0.80):+.1%}")
            _best_n = _latest.get("best_n_estimators")
            if _best_n is not None:
                _mq5.metric("Trees (CV-tuned)", f"{_best_n}")
        st.caption(f"CV {_latest.get('cv_folds', '?')}-fold · {_latest['n_samples']:,} samples")

        if len(_metrics_history) > 1:
            _hist_df = pd.DataFrame(_metrics_history)
            _hist_df["timestamp"] = pd.to_datetime(_hist_df["timestamp"])
            st.line_chart(_hist_df.set_index("timestamp")[["mape"]])

# ---------------------------------------------------------------------------
# Apply filters
# ---------------------------------------------------------------------------
filtered = signals_df.copy() if not signals_df.empty else pd.DataFrame()

if not filtered.empty:
    if selected_brands:
        filtered = filtered[filtered["brand"].isin(selected_brands)]
    if selected_models:
        filtered = filtered[filtered["model"].isin(selected_models)]
    filtered = filtered[
        (filtered["year"].between(year_range[0], year_range[1]) | filtered["year"].isna()) &
        (filtered["price_eur"].between(price_range[0], price_range[1]) | filtered["price_eur"].isna())
    ]
    if only_private:
        filtered = filtered[filtered["seller_type"] == "Particular"]
    if hide_accidents:
        filtered = filtered[filtered["desc_mentions_accident"] != True]
    if hide_repair:
        filtered = filtered[filtered["desc_mentions_repair"] != True]

# ---------------------------------------------------------------------------
# Main — deal cards
# ---------------------------------------------------------------------------
st.title("Car Deals")

if filtered.empty:
    st.info("No deals found. Adjust filters.")
    st.stop()

deals = filtered.copy()
deals["est_profit_eur"] = (deals["predicted_price"] - deals["price_eur"]).round(0)
deals["est_roi_pct"] = ((deals["predicted_price"] - deals["price_eur"]) / deals["price_eur"] * 100).round(1)

st.caption(f"{len(deals)} deals sorted by flip score")

# --- Cards ---
for _, deal in deals.iterrows():
    with st.container():
        col_main, col_price, col_link = st.columns([4, 3, 1])

        with col_main:
            year = int(deal["year"]) if pd.notna(deal.get("year")) else "?"
            gen = f" ({deal['generation']})" if pd.notna(deal.get("generation")) and deal.get("generation") else ""
            sub = f" {deal['sub_model']}" if pd.notna(deal.get("sub_model")) and deal.get("sub_model") else ""
            trim = f" {deal['trim_level']}" if pd.notna(deal.get("trim_level")) and deal.get("trim_level") else ""
            st.markdown(f"**{deal['brand']} {deal['model']}{sub}{trim}** {year}{gen}")

            details = []
            if pd.notna(deal.get("mileage_km")):
                details.append(f"{int(deal['mileage_km']):,} km")
            if deal.get("fuel_type"):
                details.append(str(deal["fuel_type"]))
            if pd.notna(deal.get("engine_cc")) and deal.get("engine_cc"):
                details.append(f"{int(deal['engine_cc'])/1000:.1f}L")
            if pd.notna(deal.get("horsepower")) and deal.get("horsepower"):
                details.append(f"{int(deal['horsepower'])} cv")
            if deal.get("transmission"):
                details.append(str(deal["transmission"]))
            if deal.get("drive_type"):
                details.append(str(deal["drive_type"]))
            if deal.get("color"):
                details.append(str(deal["color"]))
            if deal.get("district"):
                details.append(str(deal["district"]))
            if details:
                st.caption(" · ".join(details))

            # Warnings & positives on one line
            tags = []
            if deal.get("desc_mentions_accident"):
                tags.append("ДТП")
            if deal.get("desc_mentions_repair"):
                tags.append("ремонт")
            if deal.get("right_hand_drive"):
                tags.append("правый руль")
            if deal.get("taxi_fleet_rental"):
                tags.append("такси/прокат")
            n_own = deal.get("desc_mentions_num_owners")
            if pd.notna(n_own) and n_own and int(n_own) >= 3:
                tags.append(f"{int(n_own)} владельца")
            if deal.get("warranty"):
                tags.append("гарантия")
            if deal.get("first_owner_selling"):
                tags.append("1-й владелец")
            mech = deal.get("mechanical_condition")
            if pd.notna(mech) and mech and mech != "null":
                tags.append(f"мех: {mech}")
            if tags:
                st.caption(" · ".join(tags))

        with col_price:
            price = int(deal["price_eur"]) if pd.notna(deal.get("price_eur")) else 0
            if pd.notna(deal.get("predicted_price")):
                predicted = int(deal["predicted_price"])
                profit = int(deal["est_profit_eur"])
                roi = deal["est_roi_pct"]
                low = int(deal["fair_price_low"]) if pd.notna(deal.get("fair_price_low")) else None
                high = int(deal["fair_price_high"]) if pd.notna(deal.get("fair_price_high")) else None
                if low and high:
                    st.markdown(f"**{price:,} EUR** → {low:,}–{high:,}")
                else:
                    st.markdown(f"**{price:,} EUR** → {predicted:,}")
                st.markdown(f"Profit: **{profit:+,} EUR** ({roi:+.0f}%)")
                flip = deal.get("flip_score", 0)
                sample = int(deal["sample_size"]) if pd.notna(deal.get("sample_size")) else 0
                st.caption(f"Score: {flip:.0f} · based on {sample} listings")
            else:
                st.markdown(f"**{price:,} EUR**")

        with col_link:
            url = deal.get("url")
            if url and isinstance(url, str):
                label = "SV" if "standvirtual" in url else "OLX"
                st.link_button(label, url)

        # --- Top deals from the same segment ---
        if pd.notna(deal.get("predicted_price")):
            deal_gen = deal.get("generation")
            seg_mask = (
                (deals["brand"] == deal["brand"])
                & (deals["model"] == deal["model"])
                & (deals["olx_id"] != deal["olx_id"])
            )
            if deal_gen:
                seg_mask = seg_mask & (deals["generation"] == deal_gen)
            segment = deals[seg_mask].nlargest(5, "flip_score")

            if not segment.empty:
                total_in_seg = int(seg_mask.sum())
                with st.expander(f"Segment — top {len(segment)} of {total_in_seg} deals"):
                    for _, comp in segment.iterrows():
                        c1, c2, c3 = st.columns([5, 2, 1])
                        c_year = int(comp["year"]) if pd.notna(comp.get("year")) else "?"
                        c_km = f"{int(comp['mileage_km']):,} km" if pd.notna(comp.get("mileage_km")) else "—"
                        c_fuel = str(comp["fuel_type"]) if comp.get("fuel_type") else ""
                        c_price = int(comp["price_eur"])
                        c_profit = int(comp["est_profit_eur"]) if pd.notna(comp.get("est_profit_eur")) else 0
                        c_score = comp.get("flip_score", 0)

                        with c1:
                            st.text(f"{c_year} · {c_km} · {c_fuel}")
                        with c2:
                            st.text(f"{c_price:,} EUR ({c_profit:+,}) · Score {c_score:.0f}")
                        with c3:
                            c_url = comp.get("url")
                            if c_url and isinstance(c_url, str):
                                c_label = "SV" if "standvirtual" in c_url else "OLX"
                                st.link_button(c_label, c_url, key=f"seg_{deal['olx_id']}_{comp['olx_id']}")

        st.divider()
