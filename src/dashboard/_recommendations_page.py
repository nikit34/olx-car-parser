"""Recommendations — main deal-cards view.

Loaded by ``app.py`` (the st.navigation router). The router owns
``set_page_config``; this file only renders the page body.
"""

import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import sys as _sys
from pathlib import Path as _Path
_dashboard_dir = _Path(__file__).resolve().parent
_project_root = _dashboard_dir.parent.parent
_sys.path.insert(0, str(_dashboard_dir))
_sys.path.insert(0, str(_project_root))

from data_loader import (
    load_all, _force_next_check, _ensure_release_assets, DB_PATH,
    get_last_release_error, _fuel_group,
)


def _release_cache_signature() -> tuple[float, int]:
    """Cache key that invalidates the moment the local DB cache changes.

    Without this the @st.cache_data(ttl=300) below would serve a stale
    (empty / old) load_all() result for up to 5 minutes after the
    GitHub Release refresh, even though _ensure_release_assets had
    already pulled the new file. Calling _ensure_release_assets here is
    cheap (marker-gated TTL inside) and gives us the up-to-date mtime.
    """
    _ensure_release_assets()
    if not DB_PATH.exists():
        return (0.0, 0)
    s = DB_PATH.stat()
    return (s.st_mtime, s.st_size)


@st.cache_data(ttl=300)
def load_data(_cache_signature: tuple[float, int]):
    return load_all()


(
    listings_df, history_df, signals_df, brands_models, turnover_df,
    _portfolio_init, _unmatched_df, importance_df, predictions_df,
) = load_data(_release_cache_signature())

# ---------------------------------------------------------------------------
# Sidebar — filters
# ---------------------------------------------------------------------------
st.sidebar.title("Filters")

if listings_df.empty:
    st.sidebar.warning("No data yet. Scraper runs every 4 hours via GitHub Actions.")
    err = get_last_release_error()
    if err:
        st.sidebar.error(f"Release fetch failed: {err}")
    if st.sidebar.button("Force refresh"):
        _force_next_check()
        st.cache_data.clear()
        st.rerun()
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


fuel_groups_in_data = (
    sorted(set(_fuel_group(v) for v in signals_df["fuel_type"].tolist()))
    if not signals_df.empty and "fuel_type" in signals_df.columns
    else []
)
selected_fuels = st.sidebar.multiselect("Fuel type", options=fuel_groups_in_data)

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
    if selected_fuels:
        filtered = filtered[filtered["fuel_type"].map(_fuel_group).isin(selected_fuels)]
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

    # Segment filter from the ranker click — applied last so it
    # composes with the sidebar filters rather than replacing them.
    seg_filter = st.session_state.get("segment_filter")
    if seg_filter:
        filtered = filtered[
            (filtered["brand"] == seg_filter["brand"])
            & (filtered["model"] == seg_filter["model"])
        ]
        if seg_filter.get("generation"):
            filtered = filtered[filtered["generation"] == seg_filter["generation"]]

# ---------------------------------------------------------------------------
# Main — deal cards
# ---------------------------------------------------------------------------
st.title("Car Deals")

# --- Segment ranker (composite "is this market worth my time" score) ---
# Surfaces brand/model/generation buckets where the model sees average
# undervaluation, recent sold volume is healthy, and time-on-market is
# short. Click a row to filter the deal feed below to that segment.
from src.analytics.segments import compute_segment_metrics, composite_resale_score

with st.expander("📊 Segment ranker — which models are worth my time", expanded=False):
    seg_metrics = compute_segment_metrics(listings_df, signals=signals_df)
    if seg_metrics.empty:
        st.info("No segment metrics available.")
    else:
        seg_metrics["composite"] = composite_resale_score(seg_metrics)
        # Drop tiny segments (fewer than 2 actives + 0 sold) — they're
        # noise. Keep all otherwise so user can spot quiet markets too.
        seg_metrics = seg_metrics[
            (seg_metrics["n_active"] >= 2) | (seg_metrics["n_sold_60d"] >= 2)
        ]
        seg_metrics = seg_metrics.sort_values("composite", ascending=False).head(40)

        # ``round(None, 0)`` raises TypeError, so coerce all the
        # potentially-None numeric columns to float (None → NaN) before
        # asking pandas to round. NaN survives round() unchanged and
        # renders as "—" in Streamlit's table — which is exactly what
        # the user wants for "we don't have this metric for this row".
        def _num(col):
            return pd.to_numeric(seg_metrics[col], errors="coerce")

        display = pd.DataFrame({
            "segment": seg_metrics["brand"].astype(str).str.cat(
                seg_metrics["model"].astype(str), sep=" "
            ).str.cat(
                seg_metrics["generation"].fillna("").astype(str), sep=" / "
            ).str.replace(r" / $", "", regex=True),
            "active": _num("n_active").astype("Int64"),
            "sold 60d": _num("n_sold_60d").astype("Int64"),
            "median dom (d)": _num("median_dom").round(0),
            "avg uv %": _num("avg_undervaluation_pct").round(1),
            "trend 30d %": _num("trend_30d_pct").round(1),
            "calib residual €": _num("calibration_residual_eur").round(0),
            "score": (_num("composite") * 100).round(0),
        })
        # Click a row → write the segment to session_state so the filter
        # logic below narrows the deal feed to it. Streamlit's
        # ``on_select="rerun"`` reruns the whole script when the user
        # picks a row, so the change takes effect immediately.
        evt = st.dataframe(
            display, hide_index=True, use_container_width=True,
            on_select="rerun", selection_mode="single-row",
            key="segment_ranker_table",
        )
        rows = evt.selection.rows if evt and evt.selection else []
        if rows:
            picked = seg_metrics.iloc[rows[0]]
            new_filter = {
                "brand": picked["brand"],
                "model": picked["model"],
                "generation": picked["generation"] or None,
            }
            if st.session_state.get("segment_filter") != new_filter:
                st.session_state["segment_filter"] = new_filter
                st.rerun()
        st.caption(
            "score = 0.40·undervaluation + 0.25·log(sold 60d) + 0.20·velocity "
            "(14d/dom) + 0.15·trend. Calibration residual: median(actual − predicted) "
            "on sold listings — positive ⇒ model under-prices the segment."
        )

# Active segment-filter chip (from clicking the ranker table above).
seg_filter = st.session_state.get("segment_filter")
if seg_filter:
    chip_col, clear_col = st.columns([6, 1])
    label = f"{seg_filter['brand']} {seg_filter['model']}"
    if seg_filter.get("generation"):
        label += f" / {seg_filter['generation']}"
    chip_col.info(f"📍 Filtered to segment: **{label}**")
    if clear_col.button("Clear", key="clear_segment_filter"):
        st.session_state.pop("segment_filter", None)
        st.rerun()

if filtered.empty:
    st.info("No deals found. Adjust filters.")
    st.stop()

deals = filtered.copy()
deals["est_profit_eur"] = (deals["predicted_price"] - deals["price_eur"]).round(0)
# Guard against price_eur == 0 / NaN — both come up in the active set when a
# listing was scraped before its price snapshot landed. Use NA so the ROI
# column shows blank for those rows instead of crashing the page.
_price = deals["price_eur"].where(deals["price_eur"] > 0, pd.NA)
deals["est_roi_pct"] = ((deals["predicted_price"] - deals["price_eur"]) / _price * 100).round(1)

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
                # Band %: how wide the model's [low, high] band is relative
                # to predicted. Tight = high-confidence flip; wide = model
                # is uncertain (cheap segment / orphan brand).
                band_pct = deal.get("band_pct")
                if pd.notna(band_pct) and band_pct is not None:
                    st.caption(f"Score: {flip:.0f} · {sample} comps · band ±{band_pct/2:.0f}%")
                else:
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

        # --- Market chart: price vs mileage for the same configuration ---
        # Lazy-loaded behind a checkbox: building a plotly figure for every
        # card on every rerun would slow the page noticeably with ~100
        # deals. Compares against listings_df (full active set), not just
        # other rows that scored as flips.
        if st.checkbox("Market — price vs mileage", key=f"chk_{deal['olx_id']}", value=False):
            if listings_df.empty:
                st.info("No listings loaded.")
            else:
                # Build the segment slice on the FULL listings_df — both
                # active and historical (sold/expired). Sold rows give
                # 2-3× the comp density and show how the market priced
                # similar cars that actually moved. Active rows are
                # plotted on top in saturated colour, sold ones underneath
                # at low opacity so the eye still anchors on what's
                # buyable now.
                market = listings_df.copy()
                if "duplicate_of" in market.columns:
                    market = market[market["duplicate_of"].isna()]
                market = market[
                    (market["brand"] == deal["brand"])
                    & (market["model"] == deal["model"])
                ]
                deal_fuel_group = _fuel_group(deal.get("fuel_type"))
                if deal_fuel_group != "Unknown":
                    market = market[market["fuel_type"].map(_fuel_group) == deal_fuel_group]
                deal_cc = deal.get("engine_cc")
                cc_window = 200
                if pd.notna(deal_cc) and deal_cc:
                    cc_int = int(deal_cc)
                    market = market[
                        market["engine_cc"].between(cc_int - cc_window, cc_int + cc_window)
                        | market["engine_cc"].isna()
                    ]
                market = market[
                    market["mileage_km"].notna()
                    & market["price_eur"].notna()
                    & (market["price_eur"] > 0)
                ].copy()

                if len(market) < 2:
                    st.info("Not enough comparable listings (need ≥2).")
                else:
                    market["is_this"] = market["olx_id"] == deal["olx_id"]
                    is_active_col = (
                        market["is_active"]
                        if "is_active" in market.columns
                        else pd.Series(True, index=market.index)
                    )
                    market_active = market[is_active_col & ~market["is_this"]]
                    market_sold = market[~is_active_col & ~market["is_this"]]
                    market_self = market[market["is_this"]]

                    hover_cols = [c for c in ("year", "engine_cc", "horsepower",
                                               "transmission", "city", "url")
                                  if c in market.columns]

                    fig = go.Figure()
                    # Pack olx_id + url into customdata as the first two
                    # columns (followed by the user-facing hover_cols),
                    # so the on_select handler below can extract the
                    # listing URL by fixed index.
                    sold_cd_cols = ["olx_id", "url"] + [
                        c for c in hover_cols if c not in ("url",)
                    ]
                    active_cd_cols = sold_cd_cols
                    if not market_sold.empty:
                        fig.add_scatter(
                            x=market_sold["mileage_km"],
                            y=market_sold["price_eur"],
                            mode="markers",
                            marker=dict(
                                size=10, color="rgba(176, 92, 92, 0.55)",
                                symbol="x-thin",
                                line=dict(width=2, color="rgba(176, 92, 92, 0.85)"),
                            ),
                            name=f"sold / expired ({len(market_sold)})",
                            customdata=market_sold[sold_cd_cols].values,
                            hovertemplate=(
                                "<b>SOLD · €%{y:,.0f}</b> · %{x:,.0f} km<br>"
                                "%{customdata[0]} — click to open<extra></extra>"
                            ),
                        )
                    if not market_active.empty:
                        # Plotly's marker spec is strict about Python types —
                        # pandas .notna().any() returns numpy.bool_, which
                        # the showscale property rejects with a ValueError.
                        # Coerce to plain bool here.
                        has_year = bool(market_active["year"].notna().any())
                        fig.add_scatter(
                            x=market_active["mileage_km"],
                            y=market_active["price_eur"],
                            mode="markers",
                            marker=dict(
                                size=11,
                                color=market_active["year"] if has_year else "#1f77b4",
                                colorscale="Turbo",
                                showscale=has_year,
                                colorbar=dict(title="year") if has_year else None,
                                opacity=0.88,
                                line=dict(width=0.5, color="#222"),
                            ),
                            name=f"active ({len(market_active)})",
                            customdata=market_active[active_cd_cols].values,
                            hovertemplate=(
                                "<b>€%{y:,.0f}</b> · %{x:,.0f} km<br>"
                                "%{customdata[0]} — click to open<extra></extra>"
                            ),
                        )
                    # Model band overlay — make the P10/P90 envelope visible
                    # against the dot cloud so the user can see at a glance
                    # whether the listing sits inside or below the model's
                    # 80 % CQR range. Only drawn for active comps (sold/
                    # historical predictions are stale).
                    band_src = market_active.copy()
                    # Use predictions_df (full active set) — NOT signals_df
                    # (deal-only). Otherwise the band only covers listings
                    # that scored above the undervaluation threshold and
                    # mostly disappears for "not-a-deal" comparables.
                    if not band_src.empty and not predictions_df.empty:
                        pred_idx = predictions_df.set_index("olx_id")
                        for col, target in (
                            ("predicted_price", "predicted"),
                            ("fair_price_low", "fair_low"),
                            ("fair_price_high", "fair_high"),
                        ):
                            band_src[target] = (
                                band_src["olx_id"].map(pred_idx[col])
                                if col in pred_idx.columns else None
                            )
                        band_src = band_src.dropna(
                            subset=["predicted", "fair_low", "fair_high"]
                        ).sort_values("mileage_km")
                    if not band_src.empty:
                        win = max(3, min(7, len(band_src) // 5))
                        sm_low = band_src["fair_low"].rolling(win, min_periods=1, center=True).median()
                        sm_high = band_src["fair_high"].rolling(win, min_periods=1, center=True).median()
                        sm_pred = band_src["predicted"].rolling(win, min_periods=1, center=True).median()
                        fig.add_scatter(
                            x=band_src["mileage_km"], y=sm_low,
                            mode="lines",
                            line=dict(color="rgba(46, 160, 67, 0.65)", width=1.5, dash="dash"),
                            name="fair low (P10)",
                            hovertemplate="<b>P10: €%{y:,.0f}</b> @ %{x:,.0f} km<extra></extra>",
                        )
                        fig.add_scatter(
                            x=band_src["mileage_km"], y=sm_high,
                            mode="lines",
                            line=dict(color="rgba(46, 160, 67, 0.65)", width=1.5, dash="dash"),
                            fill="tonexty", fillcolor="rgba(46, 160, 67, 0.18)",
                            name="fair high (P90)",
                            hovertemplate="<b>P90: €%{y:,.0f}</b> @ %{x:,.0f} km<extra></extra>",
                        )
                        fig.add_scatter(
                            x=band_src["mileage_km"], y=sm_pred,
                            mode="lines",
                            line=dict(color="rgba(31, 119, 60, 0.95)", width=2.4),
                            name="model predicted (P50)",
                            hovertemplate="<b>predicted: €%{y:,.0f}</b> @ %{x:,.0f} km<extra></extra>",
                        )

                    if not market_self.empty:
                        fig.add_scatter(
                            x=market_self["mileage_km"],
                            y=market_self["price_eur"],
                            mode="markers",
                            marker=dict(size=18, symbol="star", color="#FF3B30",
                                        line=dict(width=1.5, color="#000")),
                            name="this listing",
                            hovertext=[f"This listing — {int(deal['price_eur']):,} EUR"],
                            hoverinfo="text",
                        )
                    fig.update_layout(
                        xaxis_title="Mileage (km)",
                        yaxis_title="Price (EUR)",
                        xaxis=dict(tickformat=","),
                        yaxis=dict(tickformat=","),
                        height=420,
                        margin=dict(l=10, r=10, t=10, b=10),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                    xanchor="right", x=1),
                    )
                    chart_event = st.plotly_chart(
                        fig, use_container_width=True,
                        on_select="rerun", selection_mode="points",
                        key=f"chart_{deal['olx_id']}",
                    )
                    sel_pts = (
                        chart_event.selection.points
                        if chart_event and chart_event.selection else []
                    )
                    if sel_pts:
                        cd = sel_pts[0].get("customdata") or []
                        if len(cd) >= 2 and cd[1]:
                            st.link_button(
                                f"Open {cd[0]} ↗", cd[1],
                                use_container_width=True,
                            )

                    _med_p_active = (
                        int(market_active["price_eur"].median())
                        if not market_active.empty else None
                    )
                    _med_p_sold = (
                        int(market_sold["price_eur"].median())
                        if not market_sold.empty else None
                    )
                    _cc_label = (
                        f"{int(deal_cc) - cc_window}–{int(deal_cc) + cc_window} cc"
                        if pd.notna(deal_cc) and deal_cc
                        else "any cc"
                    )
                    parts = [
                        f"{len(market_active)} active",
                        f"{len(market_sold)} sold/expired",
                        deal_fuel_group, _cc_label,
                    ]
                    if _med_p_active is not None:
                        parts.append(f"median active {_med_p_active:,} EUR")
                    if _med_p_sold is not None:
                        parts.append(f"median sold {_med_p_sold:,} EUR")
                    st.caption(" · ".join(parts))

        st.divider()
