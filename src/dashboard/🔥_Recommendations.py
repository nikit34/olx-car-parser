"""Dashboard entry point — Recommendations (deal cards) is the home page.

stlite (the Pyodide build that runs the dashboard on Cloudflare Pages)
does not support ``st.navigation`` / ``st.Page``, so we use the older
filename-based multipage convention: this file is the entry, and
``pages/2_*.py`` / ``pages/3_*.py`` show up in the sidebar automatically.
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

st.set_page_config(
    page_title="OLX Car Deals",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

from data_loader import (
    _force_next_check, reboot_dashboard, get_last_release_error, _fuel_group,
)
from _cache import (
    release_signature as _release_cache_signature,
    load_all_cached,
    load_snapshots_cached,
)
from src.analytics.decision import (
    build_context as _build_decision_context,
    decide as _decide_one,
    VERDICT_ICON,
    VERDICT_BUY, VERDICT_WATCH, VERDICT_REJECT, VERDICT_NO_OPINION,
)


# Defensive unpack — Streamlit Cloud can serve a stale 8-tuple from
# its in-memory cache right after a deploy that grew the return shape
# to 9 (predictions added 2026-05-03). Index with safe defaults; the
# next cache miss returns the new shape and predictions_df becomes
# populated.
_loaded = load_all_cached(_release_cache_signature())
listings_df = _loaded[0] if len(_loaded) > 0 else pd.DataFrame()
history_df = _loaded[1] if len(_loaded) > 1 else pd.DataFrame()
signals_df = _loaded[2] if len(_loaded) > 2 else pd.DataFrame()
brands_models = _loaded[3] if len(_loaded) > 3 else {}
turnover_df = _loaded[4] if len(_loaded) > 4 else pd.DataFrame()
importance_df = _loaded[7] if len(_loaded) > 7 else pd.DataFrame()
predictions_df = _loaded[8] if len(_loaded) > 8 else pd.DataFrame()
grouped_importance_df = _loaded[9] if len(_loaded) > 9 else pd.DataFrame()
# Per-listing TreeSHAP contributions for the "why this price?" expander.
# Slot 10 added 2026-05-10 — safe default keeps older cache entries from
# crashing the page during the rollover.
contributions_lookup = _loaded[10] if len(_loaded) > 10 else {}


# Build a DecisionContext once per data refresh. ``predictions_df`` is
# the right input for the calibration map — it carries predictions for
# the full active set, not just deals. The segment-level maps (DoM,
# trend, calibration) are tiny — the heavy bit is the snapshots load.
# 120-day window keeps memory bounded (~7 MB on prod) while giving
# enough history for an early/late tercile split.
try:
    _snapshots_df = load_snapshots_cached(_release_cache_signature(), 120)
except Exception:  # noqa: BLE001 — repository errors → fall back to no-trend ctx
    _snapshots_df = pd.DataFrame()

# Latest model-quality coverage feeds the global trust signal. Same
# accessor the sidebar 'Model quality' expander uses below.
from src.analytics.price_model import load_metrics_history as _load_mh
_mh = _load_mh()
_latest_coverage = None
if _mh:
    _last = _mh[-1]
    _latest_coverage = _last.get("coverage_80_calibrated") or _last.get("coverage_80")

@st.cache_data(ttl=1800, show_spinner="Building decision context...")
def _cached_decision_context(_sig, _listings, _snapshots, coverage_80):
    # `predicted_lookup` is rebuilt inside from `_listings` to keep the
    # arg list hashable-free (DataFrames + the dict are both skipped
    # via underscore prefix). Cache invalidates by TTL; callers re-run
    # on Refresh which also clears the cache.
    lookup = (
        dict(zip(predictions_df["olx_id"], predictions_df["predicted_price"]))
        if predictions_df is not None and not predictions_df.empty
        else {}
    )
    return _build_decision_context(
        _listings, _snapshots,
        coverage_80=coverage_80,
        predicted_lookup=lookup,
    )


decision_ctx = _cached_decision_context(
    _release_cache_signature(), listings_df, _snapshots_df, _latest_coverage,
)

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
        reboot_dashboard()
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
sort_by_flip = st.sidebar.checkbox(
    "Sort by flip score",
    value=False,
    help=(
        "Re-rank cards by the raw flip_score instead of the "
        "default decision score. The decision score factors in "
        "calibration residual, segment trend, DoM and "
        "seller-side multipliers."
    ),
)
sort_by_decision = not sort_by_flip
hide_non_buy = st.sidebar.checkbox(
    "Hide non-BUY verdicts",
    value=False,
    help="Show only BUY rows from the decision algorithm.",
)

if not importance_df.empty:
    with st.sidebar.expander("Feature importance (CV permutation)"):
        st.caption(
            "Pinball-loss delta when each feature is shuffled on the val "
            "fold (5-fold CV, no in-sample bias). Tail features rank low "
            "because the median model recovers them from siblings."
        )
        chart_df = importance_df[["feature", "median_importance"]].copy()
        chart_df = chart_df[chart_df["median_importance"] > 0]
        chart_df = chart_df.set_index("feature").sort_values("median_importance")
        st.bar_chart(chart_df)

# Slot 11 added 2026-05-11 — safe default keeps older cache entries from
# crashing the page during the rollover.
shap_importance_df = _loaded[11] if len(_loaded) > 11 else pd.DataFrame()
if not shap_importance_df.empty:
    with st.sidebar.expander("Feature importance (mean |SHAP|)"):
        st.caption(
            "Mean |TreeSHAP| contribution in log1p(EUR) units, averaged over "
            "val-fold predictions. SHAP fairly splits credit across "
            "correlated features without needing manual grouping."
        )
        schart_df = shap_importance_df[["feature", "median_importance"]].copy()
        schart_df = schart_df[schart_df["median_importance"] > 0]
        schart_df = schart_df.set_index("feature").sort_values("median_importance")
        st.bar_chart(schart_df)

if not grouped_importance_df.empty:
    with st.sidebar.expander("Feature importance (grouped permutation)"):
        st.caption(
            "Correlated features (vehicle_identity, powertrain, body) are "
            "permuted as a block on val folds, so each bar reflects the "
            "joint contribution instead of being diluted by within-group "
            "correlation."
        )
        gchart_df = grouped_importance_df[["group", "median_importance"]].copy()
        gchart_df = gchart_df[gchart_df["median_importance"] > 0]
        gchart_df = gchart_df.set_index("group").sort_values("median_importance")
        st.bar_chart(gchart_df)

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

# Active segment-filter chip (set by the "View segment" button on a
# deal card — pre-2026-05-04 the segment ranker also wrote it; that
# UI was removed because Pages 2/3 cover the same need better).
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

# Enrich with raw listing fields that compute_signals collapses into
# multipliers but doesn't carry through. ``decide`` reads them again to
# build its own narrative reasons.
_extra_cols = [
    "urgency", "warranty", "first_owner_selling", "taxi_fleet_rental",
    "days_listed", "price_change_eur", "mechanical_condition",
]
_present = [c for c in _extra_cols if c in listings_df.columns]
if _present:
    _extras = listings_df[["olx_id"] + _present].drop_duplicates("olx_id")
    deals = deals.merge(_extras, on="olx_id", how="left", suffixes=("", "_raw"))

# Run the decision algorithm once for every visible deal. Building the
# verdict / score columns up front lets us optionally sort or filter on
# them before the render loop.
_decisions = [_decide_one(row, decision_ctx) for _, row in deals.iterrows()]
deals["verdict"] = [d.verdict for d in _decisions]
deals["decision_score"] = [d.score for d in _decisions]
deals["__decision_obj"] = _decisions

if hide_non_buy:
    deals = deals[deals["verdict"] == VERDICT_BUY]
    if deals.empty:
        st.info("No BUY-verdict deals after current filters.")
        st.stop()

if sort_by_decision:
    deals = deals.sort_values("decision_score", ascending=False)
    _sort_label = "decision score"
else:
    _sort_label = "flip score"

st.caption(f"{len(deals)} deals sorted by {_sort_label}")

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
            # Seller-profile flags. Same presence-only logic as the
            # Telegram alert formatter — fire on definitive disagreements
            # (pseudo-private) or category-mismatches (private seller
            # listing parts), not on threshold-based heuristics.
            # Negative seller flags
            if deal.get("seller_pseudoprivate"):
                tags.append("псевдочастник")
            parts_n = deal.get("seller_parts_count")
            if (pd.notna(parts_n) and parts_n and int(parts_n) > 0
                    and not deal.get("seller_is_business")):
                tags.append(f"продаёт запчасти ({int(parts_n)})")
            # Multiple distinct car brands under a private account is
            # Sergio's pattern (4 brands / 5 cars at backfill snapshot).
            # 3+ brands sits in the top 1.6 % of the corpus (21/1278) so
            # firing definitively here is safe.
            n_brands = deal.get("seller_distinct_car_brands")
            if (pd.notna(n_brands) and n_brands and int(n_brands) >= 3
                    and not deal.get("seller_is_business")):
                tags.append(f"{int(n_brands)} разных марок")
            # Flipper composite — gate on confidence ≥ 0.4 (the lowest
            # any single non-rotation primitive contributes after 2026-05-06
            # recalibration). ≥0.75 is "almost certainly", ≥0.5 is "leans".
            fs = deal.get("flipper_score")
            fc = deal.get("flipper_confidence")
            if (pd.notna(fs) and pd.notna(fc) and fc >= 0.4):
                fs_f = float(fs)
                badge = ("🚨 flipper" if fs_f >= 0.75
                         else "⚠️ likely flipper" if fs_f >= 0.5
                         else None)
                if badge is not None:
                    tags.append(f"{badge} ({fs_f:.2f}, conf={fc:.0%})")
            # Positive-trust signals (green tags). These read as
            # supporting evidence the seller is a real private user
            # rather than a flipper, NOT a guarantee — 24% of the
            # corpus has a Facebook link and 13% have a user-set photo,
            # so the bar is "more identity than the median seller."
            social = deal.get("seller_social_account_type")
            if isinstance(social, str) and social:
                tags.append(f"✓ {social}")
            if deal.get("seller_has_user_photo"):
                tags.append("✓ фото профиля")
            age_days = deal.get("seller_account_age_days")
            if (pd.notna(age_days) and age_days
                    and int(age_days) >= 365 * 7):
                years = int(age_days) // 365
                tags.append(f"✓ акк {years}+ лет")
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

                # Resale-decision verdict — pre-computed above so it can
                # also drive the optional sidebar sort/filter on
                # decision_score.
                _decision = deal.get("__decision_obj")
                if _decision is not None:
                    _icon = VERDICT_ICON.get(_decision.verdict, "")
                    st.markdown(
                        f"{_icon} **{_decision.verdict}** · decision score "
                        f"{_decision.score:.1f}"
                    )
            else:
                st.markdown(f"**{price:,} EUR**")
                _decision = None

        with col_link:
            url = deal.get("url")
            if url and isinstance(url, str):
                label = "SV" if "standvirtual" in url else "OLX"
                st.link_button(label, url)

        # Single "View segment" button replaces:
        #   - the inline "Segment — top X of Y deals" expander
        #   - the lazy "Market — price vs mileage" checkbox + plotly chart
        # Both showed segment-level context inline; Model Details
        # already does this richer (full table + scatter + calibration)
        # and now lives a click away. Click writes the picked
        # (brand, model, generation) into session_state and switches
        # page; Page 3 reads it and pre-selects its sidebar.
        if st.button(
            f"🔍 View {deal['brand']} {deal['model']} in Model Details →",
            key=f"to_details_{deal['olx_id']}",
            use_container_width=True,
        ):
            st.session_state["target_brand"] = deal["brand"]
            st.session_state["target_model"] = deal["model"]
            st.session_state["target_generation"] = (
                deal["generation"] if pd.notna(deal.get("generation")) and deal.get("generation")
                else None
            )
            st.switch_page("pages/3_Model_Details.py")

        if _decision is not None and _decision.reasons:
            with st.expander(f"Why {_decision.verdict}? ({len(_decision.reasons)} reasons)"):
                for _r in _decision.reasons:
                    st.markdown(f"- {_r}")
                _comp = _decision.components
                if _comp:
                    _bits = []
                    if pd.notna(_comp.get("net_margin_pct")):
                        _bits.append(f"net margin {_comp['net_margin_pct']:.1f}%")
                    if pd.notna(_comp.get("expected_profit_eur")):
                        _bits.append(f"expected profit €{int(_comp['expected_profit_eur']):,}")
                    if pd.notna(_comp.get("dom_median")):
                        _bits.append(f"DoM {int(_comp['dom_median'])}d")
                    if pd.notna(_comp.get("trend_90d_pct")):
                        _bits.append(f"trend 90d {_comp['trend_90d_pct']:+.1f}%")
                    if pd.notna(_comp.get("confidence")):
                        _bits.append(f"confidence ×{_comp['confidence']:.2f}")
                    if _bits:
                        st.caption(" · ".join(_bits))

        # Per-listing TreeSHAP breakdown of the model's price. Sums in
        # log1p, displayed as a EUR waterfall: baseline + ordered deltas.
        # The bottom line ("raw model") may differ from the predicted_price
        # shown above the card because predicted_price has isotonic
        # calibration + CQR widening applied on top — the waterfall
        # explains the raw model view, which is what the user asks when
        # clicking "why this price". Difference is usually 2–5%.
        _attr = contributions_lookup.get(str(deal["olx_id"]))
        if _attr and _attr.get("deltas"):
            with st.expander("Почему такая цена? (разбор модели)"):
                base = _attr["baseline_eur"]
                raw = _attr["predicted_eur"]
                st.markdown(f"**Baseline:** {base:,.0f} EUR  *(средняя цена в обучающем наборе)*")
                for _label, _delta in _attr["deltas"]:
                    _sign = "🟢 +" if _delta >= 0 else "🔴 "
                    st.markdown(f"- {_label}: {_sign}{_delta:,.0f} EUR")
                st.markdown(f"**Raw model:** {raw:,.0f} EUR")
                if pd.notna(deal.get("predicted_price")):
                    _calib_diff = float(deal["predicted_price"]) - raw
                    if abs(_calib_diff) >= 1:
                        _sign = "+" if _calib_diff >= 0 else ""
                        st.caption(
                            f"После калибровки: {int(deal['predicted_price']):,} EUR "
                            f"({_sign}{_calib_diff:,.0f} EUR isotonic adjustment)"
                        )

        st.divider()
