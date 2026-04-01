"""OLX.pt Car Parser — Streamlit Dashboard (Deal Finder)."""

from datetime import date

import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

import sys as _sys
from pathlib import Path as _Path
_dashboard_dir = _Path(__file__).resolve().parent
_project_root = _dashboard_dir.parent.parent
_sys.path.insert(0, str(_dashboard_dir))
_sys.path.insert(0, str(_project_root))

from data_loader import load_all, load_listing_feedback, load_portfolio, _force_next_check
from src.analytics.interest_model import score_interest_candidates


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
            label = "Open on StandVirtual" if "standvirtual.com" in url else "Open on OLX"
            st.link_button(label, url)


DISPLAY_LABELS = {
    "segment_group": "Кузов / сегмент",
    "fuel_group": "Топливо",
    "transmission_group": "КПП",
    "brand": "Марка",
    "district_group": "Район",
}

FEEDBACK_LABELS_RU = {
    "interesting": "Интересно",
    "skipped": "Пропустить",
    "bought": "Куплено",
}

MODEL_SOURCE_LABELS = {
    "feedback+portfolio-trained": "Модель подстроена под ваш shortlist по ручной разметке и сделкам из портфеля.",
    "feedback-trained": "Модель подстроена под ваш shortlist по ручной разметке.",
    "portfolio-trained": "Модель подстроена под сделки из портфеля.",
    "prior": "Пока мало подтверждённых решений, поэтому используется базовая эвристика shortlist.",
}


def normalize_category(series: pd.Series, unknown_label: str = "Не указано") -> pd.Series:
    values = series.fillna(unknown_label).astype(str).str.strip()
    return values.replace({"": unknown_label, "nan": unknown_label, "None": unknown_label})


def format_group_label(group_key: str) -> str:
    return DISPLAY_LABELS.get(group_key, group_key)


def prepare_market_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Add cleaner categorical buckets for high-volume visualizations."""
    if df.empty:
        return df

    out = df.copy()

    if "segment" in out.columns:
        out["segment_group"] = normalize_category(out["segment"]).replace({"SUV/TT": "SUV / TT"})
    if "fuel_type" in out.columns:
        out["fuel_group"] = normalize_category(out["fuel_type"]).replace(
            {
                "Híbrido (Gasolina)": "Híbrido",
                "Híbrido (Diesel)": "Híbrido",
                "Híbrido Plug-In": "Híbrido Plug-in",
            }
        )
    if "transmission" in out.columns:
        out["transmission_group"] = normalize_category(out["transmission"])
    if "district" in out.columns:
        out["district_group"] = normalize_category(out["district"])
    if "city" in out.columns:
        out["city_group"] = normalize_category(out["city"])

    if "year" in out.columns:
        out["age"] = (date.today().year - out["year"]).clip(lower=0)
        out["age_bucket"] = pd.cut(
            out["age"],
            bins=[-1, 3, 6, 10, 15, 20, 99],
            labels=["0-3y", "4-6y", "7-10y", "11-15y", "16-20y", "20y+"],
            include_lowest=True,
        )
        out["year_bucket"] = pd.cut(
            out["year"],
            bins=[1980, 2005, 2010, 2015, 2020, 2023, 2100],
            labels=["<=2005", "2006-2010", "2011-2015", "2016-2020", "2021-2023", "2024+"],
            include_lowest=True,
        )

    if "mileage_km" in out.columns:
        out["mileage_bucket"] = pd.cut(
            out["mileage_km"],
            bins=[0, 50_000, 100_000, 150_000, 200_000, 250_000, 300_000, 1_000_000],
            labels=["0-50k", "50-100k", "100-150k", "150-200k", "200-250k", "250-300k", "300k+"],
            include_lowest=True,
        )

    if "price_eur" in out.columns:
        out["price_bucket"] = pd.cut(
            out["price_eur"],
            bins=[0, 5_000, 8_000, 12_000, 18_000, 25_000, 40_000, 1_000_000],
            labels=["<5k", "5-8k", "8-12k", "12-18k", "18-25k", "25-40k", "40k+"],
            include_lowest=True,
        )

    return out


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


def get_listing_feedback():
    return load_listing_feedback()

# ---------------------------------------------------------------------------
# Sidebar — filters
# ---------------------------------------------------------------------------
st.sidebar.title("Filters")

if listings_df.empty:
    st.sidebar.warning("Нет данных. Запустите `python -m src.cli scrape` для сбора объявлений.")
    st.stop()
else:
    st.sidebar.success(f"{len(listings_df)} listings loaded")
    if st.sidebar.button("Refresh data"):
        _force_next_check()
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

if "district" in listings_df.columns:
    district_values = listings_df["district"].dropna().astype(str).str.strip()
    all_districts = sorted(v for v in district_values.unique() if v)
else:
    all_districts = []
selected_districts = st.sidebar.multiselect(
    "District", options=all_districts, default=[], placeholder="All districts",
)

if selected_districts and "city" in listings_df.columns:
    city_values = (
        listings_df[listings_df["district"].isin(selected_districts)]["city"]
        .dropna()
        .astype(str)
        .str.strip()
    )
    available_cities = sorted(v for v in city_values.unique() if v)
else:
    if "city" in listings_df.columns:
        city_values = listings_df["city"].dropna().astype(str).str.strip()
        available_cities = sorted(v for v in city_values.unique() if v)
    else:
        available_cities = []

selected_cities = st.sidebar.multiselect(
    "City", options=available_cities, default=[], placeholder="All cities",
)

year_min = int(listings_df["year"].min()) if "year" in listings_df.columns and listings_df["year"].notna().any() else 2010
year_max = int(listings_df["year"].max()) if "year" in listings_df.columns and listings_df["year"].notna().any() else 2024
year_range = st.sidebar.slider("Year range", min_value=year_min, max_value=year_max, value=(year_min, year_max))

price_max_val = int(listings_df["price_eur"].max()) + 1000 if "price_eur" in listings_df.columns and listings_df["price_eur"].notna().any() else 50000
price_range = st.sidebar.slider("Price (EUR)", min_value=0, max_value=price_max_val, value=(0, price_max_val), step=500)

only_private = st.sidebar.checkbox("Particular only", value=False)
hide_desc_mentions_repair = st.sidebar.checkbox("Hide description mentions repair", value=False)
only_desc_mentions_customs_cleared = st.sidebar.checkbox("Only description mentions customs cleared", value=False)
hide_right_hand_drive = st.sidebar.checkbox("Hide right-hand drive", value=False)
st.sidebar.caption(
    "`repair`, `customs cleared`, `right-hand drive` — извлечённые упоминания из описания объявления."
)
st.sidebar.markdown("### Display")
cohort_min_size = st.sidebar.slider("Минимум объявлений в группе", min_value=3, max_value=20, value=6)
chart_top_n = st.sidebar.slider("Сколько групп показывать", min_value=5, max_value=25, value=12)

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
    if hide_desc_mentions_repair and "desc_mentions_repair" in f.columns:
        f = f[f["desc_mentions_repair"] != True]
    if only_desc_mentions_customs_cleared and "desc_mentions_customs_cleared" in f.columns:
        f = f[f["desc_mentions_customs_cleared"] == True]
    if hide_right_hand_drive and "right_hand_drive" in f.columns:
        f = f[f["right_hand_drive"] != True]
    return f


filtered_listings = prepare_market_segments(apply_filters(listings_df))
filtered_signals = prepare_market_segments(apply_filters(signals_df))

if (
    not filtered_signals.empty
    and "olx_id" in filtered_signals.columns
    and "olx_id" in filtered_listings.columns
    and "segment_group" in filtered_listings.columns
):
    signal_segments = (
        filtered_listings[["olx_id", "segment_group"]]
        .dropna(subset=["olx_id"])
        .drop_duplicates(subset=["olx_id"])
    )
    filtered_signals = filtered_signals.drop(columns=["segment_group"], errors="ignore").merge(
        signal_segments,
        on="olx_id",
        how="left",
    )
    filtered_signals["segment_group"] = filtered_signals["segment_group"].fillna("Не указано")

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
    if pd.notna(best.get("predicted_price")):
        best_profit = int(best["predicted_price"] - best["price_eur"])
        col4.metric("Best Deal Profit", f"{best_profit:+,} EUR")
    else:
        col4.metric("Best Deal Profit", "—")
else:
    col3.metric("Avg Discount", "—")
    col4.metric("Best Deal Profit", "—")

st.divider()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_deals, tab_analytics, tab_trends, tab_listings, tab_compare, tab_geo, tab_lifespan, tab_tos, tab_portfolio, tab_unmatched = st.tabs([
    "Сделки", "Аналитика", "Тренды цен", "Объявления", "Сравнение", "География", "Срок жизни", "Скорость продажи", "Портфель", "Без поколения",
])

# ---- TAB 1: Deals (Buy Signals) --------------------------------------------
with tab_deals:
    st.subheader("Недооценённые автомобили")
    st.caption("Flip-скор = недооценка % × год × пробег × ресурс двигателя × состояние × растаможка × мотивация продавца × владельцы × ликвидность × тренд. "
               "Прибыль = справедливая цена (регрессия по поколению) − запрашиваемая цена.")

    if filtered_signals.empty:
        st.info("Сделок не найдено. Попробуйте расширить фильтры.")
    else:
        deals = filtered_signals.copy()

        # Add estimated profit and ROI columns (only where regression-based predicted_price exists)
        deals["est_profit_eur"] = (deals["predicted_price"] - deals["price_eur"]).round(0)
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

        profit_day_threshold = deals["profit_per_day"].dropna().quantile(0.75)
        roi_threshold = deals["est_roi_pct"].dropna().quantile(0.7)

        def classify_deal(row: pd.Series) -> str:
            text_mentions_issue = (
                bool(row.get("desc_mentions_repair"))
                or bool(row.get("desc_mentions_accident"))
                or row.get("desc_mentions_customs_cleared") is False
                or bool(row.get("right_hand_drive"))
            )
            motivated = pd.notna(row.get("price_drop_per_day")) and row.get("price_drop_per_day") < 0
            sample_size = row.get("sample_size") if pd.notna(row.get("sample_size")) else 0
            if text_mentions_issue:
                return "Description mentions issues"
            if pd.notna(row.get("profit_per_day")) and row.get("profit_per_day") >= profit_day_threshold:
                return "Quick flip"
            if sample_size >= max(cohort_min_size, 8) and row.get("est_roi_pct", 0) >= roi_threshold:
                return "Reliable edge"
            if motivated:
                return "Negotiation play"
            return "Longer hold"

        deals["deal_profile"] = deals.apply(classify_deal, axis=1)

        # --- Top deals cards ---
        top3 = deals.head(3)
        cols = st.columns(3)
        for i, (_, deal) in enumerate(top3.iterrows()):
            with cols[i]:
                st.markdown(f"### {deal['brand']} {deal['model']} {int(deal['year']) if pd.notna(deal['year']) else '?'}")
                if pd.notna(deal.get("predicted_price")):
                    profit = int(deal["est_profit_eur"])
                    st.markdown(f"**{int(deal['price_eur']):,} EUR** → fair price **{int(deal['predicted_price']):,} EUR**")
                    profit_day = deal.get("profit_per_day")
                    profit_day_str = f" · **{profit_day:.0f} EUR/day**" if pd.notna(profit_day) else ""
                    st.markdown(f"Profit: **{profit:+,} EUR** ({deal['est_roi_pct']:+.0f}% ROI){profit_day_str}")
                else:
                    st.markdown(f"**{int(deal['price_eur']):,} EUR** · discount **{deal['discount_pct']:.0f}%** от медианы")
                    st.markdown("_Нет регрессии по поколению — прибыль не рассчитана_")
                details = []
                if pd.notna(deal.get("mileage_km")):
                    details.append(f"{int(deal['mileage_km']):,} km")
                if deal.get("fuel_type"):
                    details.append(deal["fuel_type"])
                if pd.notna(deal.get("engine_cc")) and deal.get("engine_cc"):
                    details.append(f"{int(deal['engine_cc'])/1000:.1f}L")
                if deal.get("district"):
                    details.append(deal["district"])
                drop_day = deal.get("price_drop_per_day")
                if pd.notna(drop_day) and drop_day < 0:
                    details.append(f"seller dropping {drop_day:.0f} EUR/day")
                if details:
                    st.caption(" · ".join(details))
                description_mentions = []
                if deal.get("desc_mentions_accident"):
                    description_mentions.append("упоминание ДТП")
                if deal.get("desc_mentions_repair"):
                    rc = deal.get("desc_estimated_repair_cost_eur")
                    if pd.notna(rc) and rc:
                        description_mentions.append(f"упоминание ремонта ~{int(rc):,} EUR")
                    else:
                        description_mentions.append("упоминание ремонта")
                if deal.get("desc_mentions_customs_cleared") is False:
                    description_mentions.append("нет упоминания о растаможке")
                if deal.get("right_hand_drive"):
                    description_mentions.append("правый руль")
                n_own = deal.get("desc_mentions_num_owners")
                if pd.notna(n_own) and n_own and int(n_own) >= 3:
                    description_mentions.append(f"упоминание: {int(n_own)} владельцев")
                if description_mentions:
                    st.caption("Из описания: " + ", ".join(description_mentions))
                if deal.get("url"):
                    link_label = "Open on StandVirtual" if "standvirtual.com" in deal["url"] else "Open on OLX"
                    st.markdown(f"[{link_label}]({deal['url']})")

        st.divider()

        st.subheader("ML-shortlist для ручной проверки")
        st.caption(
            "Это не классификатор “хорошая машина / плохая машина”. "
            "Он ранжирует рыночные аномалии: недооценку, ожидаемую прибыль, глубину сравнимых объявлений, "
            "ликвидность и поведение продавца. Поля, извлечённые из описания, остаются только обогащением карточки объявления."
        )
        portfolio_for_model = get_portfolio()
        feedback_for_model = get_listing_feedback()
        interesting = score_interest_candidates(
            active,
            deals,
            portfolio_for_model,
            feedback_for_model,
            min_positive_labels=2,
        )
        interesting["feedback_status"] = interesting["feedback_label"].map(FEEDBACK_LABELS_RU)
        shortlist = interesting[
            interesting["interest_class"].ne("Low priority")
            & ~interesting["feedback_label"].isin(["skipped", "bought"])
        ].copy()
        if shortlist.empty:
            st.info("Сейчас нет объявлений, которые модель считает приоритетными для ручной проверки.")
        else:
            shortlist["interest_match_pct"] = (shortlist["interest_probability"] * 100).round(0)
            if "model_source" in shortlist.columns:
                source_label = shortlist["model_source"].iloc[0]
                portfolio_positive = int(shortlist["portfolio_positive_count"].iloc[0]) if "portfolio_positive_count" in shortlist.columns else 0
                feedback_positive = int(shortlist["feedback_positive_count"].iloc[0]) if "feedback_positive_count" in shortlist.columns else 0
                feedback_negative = int(shortlist["feedback_negative_count"].iloc[0]) if "feedback_negative_count" in shortlist.columns else 0
                source_caption = MODEL_SOURCE_LABELS.get(source_label, MODEL_SOURCE_LABELS["prior"])
                st.caption(
                    f"{source_caption} "
                    f"Позитивов из портфеля: {portfolio_positive}, ручных +: {feedback_positive}, ручных -: {feedback_negative}."
                )
            quick_hits = shortlist.head(4)
            hit_cols = st.columns(len(quick_hits))
            for idx, (_, listing) in enumerate(quick_hits.iterrows()):
                with hit_cols[idx]:
                    st.markdown(
                        f"### {listing['brand']} {listing['model']} "
                        f"{int(listing['year']) if pd.notna(listing.get('year')) else '?'}"
                    )
                    st.markdown(
                        f"**{listing['interest_class']}** · "
                        f"{listing['interest_probability'] * 100:.0f}% shortlist match"
                    )
                    if pd.notna(listing.get("price_eur")):
                        st.markdown(f"Цена: **{int(listing['price_eur']):,} EUR**")
                    if pd.notna(listing.get("est_profit_eur")):
                        st.markdown(f"Потенциал: **{int(listing['est_profit_eur']):+,} EUR**")
                    st.caption(listing["interest_reason"])
                    if listing.get("url"):
                        label = "Open on StandVirtual" if "standvirtual.com" in listing["url"] else "Open on OLX"
                        st.link_button(label, listing["url"])

            ml_view = shortlist.head(150).copy()
            ml_view["_size"] = ml_view.get("sample_size", pd.Series(index=ml_view.index, dtype=float)).fillna(2).clip(lower=2)
            fig_ml = px.scatter(
                ml_view,
                x="interest_probability",
                y="est_profit_eur",
                color="interest_class",
                size="_size",
                hover_data=[
                    "brand",
                    "model",
                    "year",
                    "price_eur",
                    "interest_reason",
                    "sample_size",
                    "url",
                ],
                labels={
                    "interest_probability": "Interest probability",
                    "est_profit_eur": "Estimated profit (EUR)",
                    "interest_class": "Class",
                    "_size": "Comparable listings",
                },
                height=460,
                opacity=0.75,
                render_mode="webgl",
            )
            plotly_chart_with_click(fig_ml, ml_view, key="ml_shortlist_scatter", width="stretch")

            ml_cols = [
                "interest_class",
                "feedback_status",
                "interest_match_pct",
                "brand",
                "model",
                "year",
                "price_eur",
                "est_profit_eur",
                "est_roi_pct",
                "sample_size",
                "interest_reason",
                "url",
            ]
            st.dataframe(
                shortlist[[c for c in ml_cols if c in shortlist.columns]].head(40).rename(columns={
                    "interest_class": "Class",
                    "feedback_status": "Your label",
                    "interest_match_pct": "Match",
                    "brand": "Brand",
                    "model": "Model",
                    "year": "Year",
                    "price_eur": "Price",
                    "est_profit_eur": "Profit",
                    "est_roi_pct": "ROI %",
                    "sample_size": "Comps",
                    "interest_reason": "Why shortlisted",
                    "url": "Link",
                }),
                hide_index=True,
                width="stretch",
                column_config={
                    "Match": st.column_config.NumberColumn(format="%.0f%%"),
                    "Price": st.column_config.NumberColumn(format="%d EUR"),
                    "Profit": st.column_config.NumberColumn(format="%+d EUR"),
                    "ROI %": st.column_config.NumberColumn(format="%+.1f%%"),
                    "Comps": st.column_config.NumberColumn(format="%d"),
                    "Link": st.column_config.LinkColumn("Link", display_text="Open"),
                },
            )

        st.subheader("Обучение модели на ваших решениях")
        st.caption(
            "Помечайте shortlist как `Интересно`, `Пропустить` или `Куплено`. "
            "Явные метки сразу влияют на выдачу, а при накоплении примеров модель начинает подстраиваться под ваш стиль отбора. "
            "Она учит именно ваш паттерн shortlist, а не пытается доказать, что машина объективно хорошая."
        )

        feedback_pool = interesting.head(200).copy()
        if feedback_pool.empty:
            st.info("Пока нет кандидатов для ручной разметки.")
        else:
            feedback_pool["interest_match_pct"] = (feedback_pool["interest_probability"] * 100).round(0)
            feedback_pool["feedback_status"] = feedback_pool["feedback_status"].fillna("Без метки")
            feedback_pool["candidate_label"] = feedback_pool.apply(
                lambda row: (
                    f"{row['brand']} {row['model']} "
                    f"{int(row['year']) if pd.notna(row.get('year')) else '?'}"
                    f" · {int(row['price_eur']):,} EUR"
                    f" · {row['interest_class']}"
                    f" · {int(row['interest_match_pct'])}%"
                    f" · {row['feedback_status']}"
                ),
                axis=1,
            )
            candidate_labels = feedback_pool["candidate_label"].tolist()
            default_option = feedback_pool.sort_values(
                ["feedback_label", "interest_probability"],
                ascending=[True, False],
                na_position="first",
            )["candidate_label"].iloc[0]

            with st.form("listing_feedback_form"):
                selected_candidate = st.selectbox(
                    "Объявление для разметки",
                    options=candidate_labels,
                    index=candidate_labels.index(default_option),
                )
                selected_row = feedback_pool.loc[
                    feedback_pool["candidate_label"].eq(selected_candidate)
                ].iloc[0]
                current_label = selected_row.get("feedback_label")
                selected_label = st.radio(
                    "Ваше решение",
                    options=["interesting", "skipped", "bought"],
                    index=["interesting", "skipped", "bought"].index(current_label)
                    if current_label in {"interesting", "skipped", "bought"}
                    else 0,
                    format_func=lambda value: FEEDBACK_LABELS_RU[value],
                    horizontal=True,
                )
                feedback_note = st.text_input(
                    "Короткая заметка",
                    value="" if pd.isna(selected_row.get("feedback_notes")) else str(selected_row.get("feedback_notes")),
                    placeholder="Например: слишком рискованно / хорош для звонка / уже куплен",
                )
                save_col, delete_col = st.columns(2)
                save_feedback = save_col.form_submit_button("Сохранить метку", type="primary")
                delete_feedback = delete_col.form_submit_button("Убрать метку")

            selected_row = feedback_pool.loc[
                feedback_pool["candidate_label"].eq(selected_candidate)
            ].iloc[0]
            st.caption(selected_row["interest_reason"])
            if selected_row.get("url"):
                label = "Open on StandVirtual" if "standvirtual.com" in selected_row["url"] else "Open on OLX"
                st.link_button(label, selected_row["url"])

            if save_feedback:
                from src.storage.database import init_db, get_session
                from src.storage.repository import upsert_listing_feedback

                db_path = _Path(__file__).resolve().parent.parent.parent / "data" / "olx_cars.db"
                init_db(str(db_path))
                sess = get_session()
                upsert_listing_feedback(
                    sess,
                    {
                        "olx_id": selected_row["olx_id"],
                        "url": selected_row.get("url"),
                        "title": selected_row.get("title"),
                        "brand": selected_row.get("brand"),
                        "model": selected_row.get("model"),
                        "year": None if pd.isna(selected_row.get("year")) else int(selected_row["year"]),
                        "price_eur": None if pd.isna(selected_row.get("price_eur")) else float(selected_row["price_eur"]),
                        "label": selected_label,
                        "notes": feedback_note.strip() or None,
                    },
                )
                st.success("Метка сохранена, shortlist обновлён.")
                st.rerun()

            if delete_feedback and pd.notna(selected_row.get("feedback_label")):
                from src.storage.database import init_db, get_session
                from src.storage.repository import delete_listing_feedback

                db_path = _Path(__file__).resolve().parent.parent.parent / "data" / "olx_cars.db"
                init_db(str(db_path))
                sess = get_session()
                delete_listing_feedback(sess, selected_row["olx_id"])
                st.success("Метка удалена.")
                st.rerun()

        if feedback_for_model.empty:
            st.caption("Ручных меток пока нет.")
        else:
            feedback_summary = (
                feedback_for_model["feedback_label"]
                .map(FEEDBACK_LABELS_RU)
                .value_counts()
                .rename_axis("Label")
                .reset_index(name="Count")
            )
            summary_cols = st.columns(max(len(feedback_summary), 1))
            for idx, (_, row) in enumerate(feedback_summary.iterrows()):
                summary_cols[idx].metric(row["Label"], int(row["Count"]))

            recent_feedback = feedback_for_model.copy()
            if "feedback_label" in recent_feedback.columns:
                recent_feedback["feedback_label"] = recent_feedback["feedback_label"].map(FEEDBACK_LABELS_RU)
            recent_cols = [
                "feedback_label",
                "brand",
                "model",
                "year",
                "price_eur",
                "feedback_notes",
                "feedback_updated_at",
                "url",
            ]
            st.dataframe(
                recent_feedback[[c for c in recent_cols if c in recent_feedback.columns]].head(20).rename(columns={
                    "feedback_label": "Label",
                    "brand": "Brand",
                    "model": "Model",
                    "year": "Year",
                    "price_eur": "Price",
                    "feedback_notes": "Notes",
                    "feedback_updated_at": "Updated",
                    "url": "Link",
                }),
                hide_index=True,
                width="stretch",
                column_config={
                    "Price": st.column_config.NumberColumn(format="%d EUR"),
                    "Link": st.column_config.LinkColumn("Link", display_text="Open"),
                },
            )

        st.divider()

        # --- Full deals table ---
        signal_cols = ["deal", "deal_profile", "brand", "model", "generation", "year",
                       "price_eur", "predicted_price", "est_profit_eur", "est_roi_pct",
                       "profit_per_day", "undervaluation_pct", "flip_score"]
        if "avg_days_to_sell" in deals.columns:
            signal_cols.append("avg_days_to_sell")
        if "sample_size" in deals.columns:
            signal_cols += ["sample_size", "confidence"]
        signal_cols += ["mileage_km", "engine_cc", "fuel_type", "segment_group", "district", "city"]
        for extra_col in [
            "desc_mentions_accident",
            "desc_mentions_repair",
            "desc_estimated_repair_cost_eur",
            "desc_mentions_num_owners",
            "desc_mentions_customs_cleared",
            "right_hand_drive",
        ]:
            if extra_col in deals.columns and deals[extra_col].notna().any():
                signal_cols.append(extra_col)
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
                "deal_profile": "Profile",
                "generation": "Gen", "year": "Year",
                "price_eur": "Price", "predicted_price": "Fair Price",
                "est_profit_eur": "Profit", "est_roi_pct": "ROI %",
                "profit_per_day": "EUR/day",
                "undervaluation_pct": "Below Market %", "flip_score": "Score",
                "avg_days_to_sell": "Days to Sell", "sample_size": "Sample",
                "confidence": "Conf", "mileage_km": "Mileage", "engine_cc": "CC",
                "fuel_type": "Fuel", "segment_group": "Segment", "district": "District", "city": "City",
                "desc_mentions_accident": "Desc: accident",
                "desc_mentions_repair": "Desc: repair",
                "desc_estimated_repair_cost_eur": "Desc repair est.",
                "desc_mentions_num_owners": "Desc owners",
                "desc_mentions_customs_cleared": "Desc customs",
                "right_hand_drive": "RHD",
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
                "CC": st.column_config.NumberColumn(format="%d"),
                "Accident": st.column_config.CheckboxColumn(),
                "Repair": st.column_config.CheckboxColumn(),
                "Repair Cost": st.column_config.NumberColumn(format="%d EUR"),
                "Owners": st.column_config.NumberColumn(format="%d"),
                "Customs": st.column_config.CheckboxColumn(),
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
                title="Распределение прибыли", height=350,
            )
            fig_profit.update_layout(showlegend=False)
            st.plotly_chart(fig_profit, width="stretch")

        with col_chart2:
            profile_stats = (
                deals.groupby("deal_profile")
                .agg(
                    count=("deal_profile", "size"),
                    median_profit=("est_profit_eur", "median"),
                    median_roi=("est_roi_pct", "median"),
                )
                .reset_index()
                .sort_values(["median_profit", "count"], ascending=[False, False])
            )
            fig_profiles = px.bar(
                profile_stats,
                x="count",
                y="deal_profile",
                orientation="h",
                color="median_roi",
                color_continuous_scale="YlGn",
                labels={"count": "Deals", "deal_profile": "Profile", "median_roi": "Median ROI %"},
                title="Архетипы сделок",
                height=350,
            )
            fig_profiles.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_profiles, width="stretch")

        st.subheader("Где рынок отдаёт лучшие сделки")
        st.caption(
            "Тепловая карта агрегирует ROI по сегменту и ценовому диапазону. "
            "Внутри ячейки — число найденных сделок, так что легко видеть не только доходность, но и глубину рынка."
        )
        hotspot_data = (
            deals[deals["segment_group"].notna() & deals["price_bucket"].notna()]
            .groupby(["segment_group", "price_bucket"], observed=True)
            .agg(median_roi=("est_roi_pct", "median"), count=("est_roi_pct", "size"))
            .reset_index()
        )
        hotspot_data = hotspot_data[hotspot_data["count"] >= cohort_min_size]
        if not hotspot_data.empty:
            segment_order = (
                hotspot_data.groupby("segment_group")["count"]
                .sum()
                .sort_values(ascending=False)
                .head(chart_top_n)
                .index
            )
            hotspot_data = hotspot_data[hotspot_data["segment_group"].isin(segment_order)]
            roi_pivot = hotspot_data.pivot(index="segment_group", columns="price_bucket", values="median_roi")
            count_pivot = hotspot_data.pivot(index="segment_group", columns="price_bucket", values="count")
            roi_pivot = roi_pivot.loc[segment_order]
            count_pivot = count_pivot.reindex(index=roi_pivot.index, columns=roi_pivot.columns)
            fig_hot = px.imshow(
                roi_pivot.values,
                x=[str(c) for c in roi_pivot.columns],
                y=roi_pivot.index.tolist(),
                labels=dict(color="Median ROI %"),
                aspect="auto",
                color_continuous_scale="YlGn",
                height=max(320, len(roi_pivot) * 40 + 80),
            )
            fig_hot.update_traces(
                text=[
                    [
                        f"{int(count_pivot.iloc[r, c])}" if pd.notna(count_pivot.iloc[r, c]) else ""
                        for c in range(len(count_pivot.columns))
                    ]
                    for r in range(len(count_pivot.index))
                ],
                texttemplate="%{text}",
            )
            st.plotly_chart(fig_hot, width="stretch")

        priority_data = deals[deals["avg_days_to_sell"].notna()].copy()
        if not priority_data.empty:
            st.subheader("Карта приоритета: прибыль vs ликвидность")
            st.caption(
                "Каждая точка — конкретное объявление. Верхний левый квадрат — сделки, "
                "где рынок обычно продаётся быстрее, а потенциальная прибыль выше медианы."
            )
            priority_data["_bubble_size"] = priority_data["sample_size"].fillna(1).clip(lower=1)
            fig_priority = px.scatter(
                priority_data,
                x="avg_days_to_sell",
                y="est_profit_eur",
                color="deal_profile",
                size="_bubble_size",
                hover_data=[
                    "brand",
                    "model",
                    "year",
                    "price_eur",
                    "predicted_price",
                    "est_roi_pct",
                    "sample_size",
                    "url",
                ],
                labels={
                    "avg_days_to_sell": "Typical days to sell",
                    "est_profit_eur": "Estimated profit (EUR)",
                    "deal_profile": "Profile",
                    "_bubble_size": "Sample size",
                },
                height=500,
                opacity=0.75,
                render_mode="webgl",
            )
            fig_priority.add_vline(
                x=priority_data["avg_days_to_sell"].median(),
                line_dash="dot",
                line_color="gray",
                opacity=0.5,
            )
            fig_priority.add_hline(
                y=priority_data["est_profit_eur"].median(),
                line_dash="dot",
                line_color="gray",
                opacity=0.5,
            )
            plotly_chart_with_click(fig_priority, priority_data, key="deals_priority", width="stretch")

        # --- Motivated sellers: biggest price drops ---
        if "price_drop_per_day" in deals.columns and deals["price_drop_per_day"].notna().any():
            st.subheader("Мотивированные продавцы")
            st.caption("Продавцы, активно снижающие цену — больше пространства для торга. "
                       "Чем быстрее падает цена, тем больше продавец хочет продать.")
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
            st.subheader("Когда продавцы снижают цену?")
            st.caption("Средний % изменения цены по периодам на рынке. "
                       "Показывает через сколько дней продавцы начинают давать скидки.")
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
    st.subheader("Аналитика ценовых факторов")
    st.caption("Поиск автомобилей с лучшим сочетанием характеристик по минимальной цене. "
               "Графики показывают как год, пробег, топливо и КПП влияют на стоимость.")

    ana = active[active["price_eur"].notna()].copy() if "price_eur" in active.columns else pd.DataFrame()

    if ana.empty:
        st.info("Нет активных объявлений с ценами.")
    else:
        # Prepare columns
        if "year" in ana.columns:
            ana["age"] = date.today().year - ana["year"]
        if "mileage_km" in ana.columns and "price_eur" in ana.columns:
            ana["eur_per_1k_km"] = (ana["price_eur"] / (ana["mileage_km"] / 1000)).replace([np.inf, -np.inf], np.nan)
        ana["label"] = ana["brand"] + " " + ana["model"]
        if "year" in ana.columns:
            ana["label_year"] = ana["label"] + " " + ana["year"].astype(int).astype(str)

        # ---- 1. Year vs Price aggregated by Transmission ----
        st.subheader("Годовые корзины vs цена")
        st.caption(
            "Вместо сырого scatter здесь показана медианная цена по годовым корзинам. "
            "На больших выборках такой вид быстрее показывает, где автомат или механика реально расходятся по рынку."
        )
        yr_data = ana[ana["year_bucket"].notna() & ana["transmission_group"].notna()].copy()
        if not yr_data.empty:
            year_summary = (
                yr_data.groupby(["year_bucket", "transmission_group"], observed=True)
                .agg(median_price=("price_eur", "median"), count=("price_eur", "size"))
                .reset_index()
            )
            year_summary = year_summary[year_summary["count"] >= cohort_min_size]
            if not year_summary.empty:
                fig_yr = px.line(
                    year_summary,
                    x="year_bucket",
                    y="median_price",
                    color="transmission_group",
                    markers=True,
                    hover_data=["count"],
                    labels={
                        "year_bucket": "Year bucket",
                        "median_price": "Median price (EUR)",
                        "transmission_group": "Transmission",
                    },
                    height=460,
                )
                fig_yr.update_layout(hovermode="x unified")
                st.plotly_chart(fig_yr, use_container_width=True)

        # ---- 2. Mileage vs Price colored by Fuel ----
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Пробег vs цена по типу топлива")
            st.caption(
                "График агрегирован по корзинам пробега. "
                "Так проще увидеть, в каких диапазонах дизель, бензин и гибриды действительно расходятся по цене."
            )
            mil_data = ana[ana["mileage_bucket"].notna() & ana["fuel_group"].notna()].copy()
            if not mil_data.empty:
                mileage_summary = (
                    mil_data.groupby(["mileage_bucket", "fuel_group"], observed=True)
                    .agg(median_price=("price_eur", "median"), count=("price_eur", "size"))
                    .reset_index()
                )
                mileage_summary = mileage_summary[mileage_summary["count"] >= cohort_min_size]
                if not mileage_summary.empty:
                    fig_mil = px.line(
                        mileage_summary,
                        x="mileage_bucket",
                        y="median_price",
                        color="fuel_group",
                        markers=True,
                        hover_data=["count"],
                        labels={
                            "mileage_bucket": "Mileage bucket",
                            "median_price": "Median price (EUR)",
                            "fuel_group": "Fuel",
                        },
                        height=450,
                    )
                    fig_mil.update_layout(hovermode="x unified")
                    st.plotly_chart(fig_mil, use_container_width=True)

        with col_b:
            st.subheader("Импорт vs Национальный по маркам")
            st.caption(
                "Показывает медиану цены по происхождению. Это хороший срез для поиска брендов, "
                "где локальный рынок заметно дешевле импортного."
            )
            origin_data = ana[ana["origin"].notna() & ana["brand"].notna()].copy()
            if not origin_data.empty:
                brand_origin = (
                    origin_data.groupby(["brand", "origin"])["price_eur"]
                    .median()
                    .reset_index()
                )
                # Only brands with both origins
                both = brand_origin.groupby("brand").filter(lambda g: len(g) >= 2)
                if not both.empty:
                    brand_order = origin_data["brand"].value_counts().head(chart_top_n).index
                    both = both[both["brand"].isin(brand_order)]
                    fig_orig = px.bar(
                        both.sort_values(["brand", "origin"]),
                        x="brand", y="price_eur", color="origin", barmode="group",
                        labels={"brand": "Brand", "price_eur": "Median Price (EUR)", "origin": "Origin"},
                        color_discrete_map={"Importado": "#e63946", "Nacional": "#2a9d8f"},
                        height=450,
                    )
                    st.plotly_chart(fig_orig, use_container_width=True)

        # ---- 3. Value Score: composite metric ----
        st.subheader("Value Score — лучшие авто за свои деньги")
        st.caption("Скор = год (25) + ресурс двигателя (30) + цена (15) + состояние (10) + топливо (10) + л.с. (5) + КПП (5). "
                   "Пробег нормализуется по объёму двигателя: малолитражки изнашиваются быстрее. "
                   "Чем выше скор — тем лучше соотношение цена/качество. Зелёная зона = оптимальные варианты.")

        score_data = ana[
            ana["year"].notna() & ana["mileage_km"].notna() & (ana["mileage_km"] > 0)
        ].copy()

        if len(score_data) >= 5:
            # Normalize each factor 0-1 (higher = better)
            score_data["year_norm"] = (score_data["year"] - score_data["year"].min()) / max(score_data["year"].max() - score_data["year"].min(), 1)
            # Mileage normalized by engine lifespan: 200k on 1.0L ≠ 200k on 2.0L
            from src.dashboard.data_loader import _expected_lifespan_km
            score_data["_lifespan"] = score_data.apply(
                lambda r: _expected_lifespan_km(
                    int(r["engine_cc"]) if pd.notna(r.get("engine_cc")) and r.get("engine_cc") else None,
                    r.get("fuel_type") if pd.notna(r.get("fuel_type")) else None,
                ), axis=1,
            )
            score_data["_wear_pct"] = (score_data["mileage_km"] / score_data["_lifespan"]).clip(0, 1)
            score_data["mileage_norm"] = 1 - score_data["_wear_pct"]
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

            # Description mention penalty: textual mentions of issues lower the score
            score_data["condition_norm"] = 1.0
            if "desc_mentions_accident" in score_data.columns:
                score_data.loc[score_data["desc_mentions_accident"] == True, "condition_norm"] = 0.2
            if "desc_mentions_repair" in score_data.columns:
                score_data.loc[
                    (score_data["desc_mentions_repair"] == True) & (score_data["condition_norm"] > 0.2),
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

            st.subheader("Карта сегментов рынка")
            split_field = st.selectbox(
                "Сегментировать рынок по",
                ["segment_group", "fuel_group", "transmission_group", "brand"],
                format_func=format_group_label,
                key="analytics_market_split",
            )
            market_summary = (
                score_data.groupby(split_field)
                .agg(
                    listings=("price_eur", "size"),
                    median_price=("price_eur", "median"),
                    median_mileage=("mileage_km", "median"),
                    median_age=("age", "median"),
                    median_value_score=("value_score", "median"),
                )
                .reset_index()
            )
            market_summary = market_summary[
                (market_summary["listings"] >= cohort_min_size)
                & (market_summary[split_field] != "Не указано")
            ]
            if split_field == "brand":
                market_summary = market_summary.sort_values("listings", ascending=False).head(chart_top_n)
            if not market_summary.empty:
                fig_market = px.scatter(
                    market_summary,
                    x="median_age",
                    y="median_price",
                    size="listings",
                    color="median_value_score",
                    text=split_field,
                    color_continuous_scale="YlGn",
                    hover_data=["median_mileage"],
                    labels={
                        "median_age": "Median age (years)",
                        "median_price": "Median price (EUR)",
                        "median_value_score": "Median value score",
                        "listings": "Listings",
                    },
                    height=500,
                )
                fig_market.update_traces(textposition="top center")
                st.plotly_chart(fig_market, use_container_width=True)

            fig_val = px.scatter(
                score_data, x="price_eur", y="value_score",
                color="segment_group",
                hover_data=["model", "year", "mileage_km", "transmission", "fuel_type", "horsepower", "district", "url"],
                labels={"price_eur": "Price (EUR)", "value_score": "Value Score", "segment_group": "Segment"},
                title="Value Score vs Price — левый верхний сектор = лучшие сочетания",
                height=550,
                opacity=0.55,
                render_mode="webgl",
            )
            # Highlight optimal zone (top-left quadrant)
            med_price = score_data["price_eur"].median()
            med_score = score_data["value_score"].median()
            fig_val.add_shape(
                type="rect",
                x0=score_data["price_eur"].min(),
                x1=med_price,
                y0=med_score,
                y1=score_data["value_score"].max(),
                line=dict(color="green", width=2, dash="dot"),
                fillcolor="rgba(42,157,143,0.08)",
            )
            fig_val.add_annotation(
                x=med_price * 0.35,
                y=med_score + (score_data["value_score"].max() - med_score) * 0.8,
                text="Sweet spot",
                showarrow=False,
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

        # ---- 4. Market heatmap: Age x Mileage ----
        st.subheader("Карта рынка: возраст × пробег")
        st.caption(
            "В ячейках медианная цена, подпись показывает число объявлений. "
            "Это даёт базовую «норму рынка» по состоянию машины и помогает сразу увидеть аномально дорогие или дешёвые зоны."
        )
        dep_heat = ana[ana["age_bucket"].notna() & ana["mileage_bucket"].notna()].copy()
        if not dep_heat.empty:
            heat_data = (
                dep_heat.groupby(["age_bucket", "mileage_bucket"], observed=True)
                .agg(median_price=("price_eur", "median"), count=("price_eur", "size"))
                .reset_index()
            )
            heat_data = heat_data[heat_data["count"] >= cohort_min_size]
            if not heat_data.empty:
                pivot_dep = heat_data.pivot(index="age_bucket", columns="mileage_bucket", values="median_price")
                pivot_counts = heat_data.pivot(index="age_bucket", columns="mileage_bucket", values="count")
                pivot_counts = pivot_counts.reindex(index=pivot_dep.index, columns=pivot_dep.columns)
                fig_dep_h = px.imshow(
                    pivot_dep.values,
                    x=[str(c) for c in pivot_dep.columns],
                    y=[str(i) for i in pivot_dep.index],
                    labels=dict(color="Median Price (EUR)"),
                    aspect="auto",
                    color_continuous_scale="YlOrRd",
                    height=max(320, len(pivot_dep) * 55),
                )
                fig_dep_h.update_traces(
                    text=[
                        [
                            f"{int(pivot_counts.iloc[r, c])}" if pd.notna(pivot_counts.iloc[r, c]) else ""
                            for c in range(len(pivot_counts.columns))
                        ]
                        for r in range(len(pivot_counts.index))
                    ],
                    texttemplate="%{text}",
                )
                st.plotly_chart(fig_dep_h, use_container_width=True)

        # ---- 5. Transmission + Fuel Price Matrix ----
        col_c, col_d = st.columns(2)

        with col_c:
            st.subheader("КПП + Топливо: медиана цены")
            # Caption added below after data check
            tf_data = ana[ana["transmission"].notna() & ana["fuel_type"].notna()].copy()
            if not tf_data.empty:
                pivot_tf = tf_data.pivot_table(
                    values="price_eur", index="fuel_type", columns="transmission",
                    aggfunc="median", observed=True,
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
            st.subheader("EUR за 1000 км по маркам")
            st.caption("Стоимость за единицу пробега. Чем ниже показатель — тем больше километров за ваши деньги.")
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
        st.subheader("Поиск оптимального авто")
        st.caption("Задайте ваши ограничения (бюджет, возраст, пробег, КПП) и смотрите что подходит лучше всего.")

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
            st.warning("Нет машин по этим критериям. Попробуйте ослабить ограничения.")
        else:
            sweet = sweet.sort_values("price_eur")
            st.success(f"{len(sweet)} cars found")

            fig_sweet = px.scatter(
                sweet, x="mileage_km", y="price_eur",
                color="segment_group",
                hover_data=["model", "year", "transmission", "fuel_type", "horsepower", "district", "url"],
                labels={"mileage_km": "Mileage (km)", "price_eur": "Price (EUR)", "segment_group": "Segment"},
                title=f"Cars under {max_budget:,} EUR, max {max_age_ss}y, max {max_mileage_ss//1000}k km",
                height=500,
                opacity=0.7,
                render_mode="webgl",
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
    st.subheader("Динамика цен")
    st.caption("Как менялись медианные цены моделей со временем. "
               "Растущий тренд = рынок дорожает, падающий = снижение спроса.")
    if filtered_history.empty:
        st.info("Нет истории цен. Запустите скрапер несколько раз для сбора данных.")
    else:
        fh = filtered_history.copy()
        fh["label"] = fh["brand"] + " " + fh["model"]
        fig = px.line(fh, x="date", y="median_price_eur", color="label",
                      labels={"median_price_eur": "Median Price (EUR)", "date": "Date", "label": "Model"},
                      height=500)
        fig.update_layout(hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=-0.3))
        st.plotly_chart(fig, width="stretch")

        st.subheader("Ценовой коридор")
        st.caption("Полоса между минимальной и максимальной ценой модели. "
                   "Узкий коридор = стабильный рынок, широкий = разброс в ценах.")
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
            st.caption("Выберите ровно одну модель для просмотра ценового коридора.")

    # --- Seasonality ---
    st.subheader("Сезонность")
    st.caption("Индекс цен по месяцам (100 = среднегодовая). "
               "Ниже 100 = дешевле, выше = дороже. Помогает выбрать лучший момент для покупки/продажи.")
    if history_df.empty:
        st.info("Нужны данные за несколько месяцев для анализа сезонности.")
    else:
        from src.analytics.seasonality import compute_seasonality, best_buy_sell_months

        season_data = compute_seasonality(history_df, listings_df)
        if season_data is not None and not season_data.empty:
            pivot_season = season_data.pivot_table(
                values="price_index", index="segment", columns="month_name", observed=True,
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
                    st.caption("Лучшие месяцы для покупки (низкий индекс) и продажи (высокий индекс)")
                    st.dataframe(bbs, hide_index=True)
        else:
            st.info("Нужно минимум 3 месяца данных. Продолжайте запускать скрапер!")

    # --- Fuel Type Premium Trend ---
    st.subheader("Тренд цен по типу топлива")
    st.caption("Как менялись медианные цены разных типов топлива со временем. "
               "Показывает тренды: снижение дизеля, рост гибридов и т.д.")
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
    st.subheader("Все объявления")
    st.caption("Полная таблица всех объявлений с фильтрами. Кликните по заголовку столбца для сортировки.")
    display_cols = [c for c in [
        "brand", "model", "year", "price_eur", "days_listed",
        "price_change_eur", "price_change_pct", "eur_per_km",
        "mileage_km", "engine_cc",
        "fuel_type", "horsepower", "transmission", "segment",
        "desc_mentions_accident", "desc_mentions_repair", "desc_estimated_repair_cost_eur",
        "desc_mentions_num_owners", "desc_mentions_customs_cleared", "right_hand_drive",
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
            "desc_mentions_accident": st.column_config.CheckboxColumn("Desc: accident"),
            "desc_mentions_repair": st.column_config.CheckboxColumn("Desc: repair"),
            "desc_estimated_repair_cost_eur": st.column_config.NumberColumn("Desc repair est.", format="%d EUR"),
            "desc_mentions_num_owners": st.column_config.NumberColumn("Desc owners", format="%d"),
            "desc_mentions_customs_cleared": st.column_config.CheckboxColumn("Desc customs"),
            "right_hand_drive": st.column_config.CheckboxColumn("RHD"),
            "is_active": st.column_config.CheckboxColumn("Active"),
            "url": st.column_config.LinkColumn("Link", display_text="Open"),
        },
    )
    st.caption(f"Showing {len(filtered_listings)} listings")


# ---- TAB 4: Compare Models ---------------------------------------------------
with tab_compare:
    st.subheader("Сравнение моделей")
    st.caption(
        "На больших выборках важнее агрегированные метрики модели: глубина рынка, ликвидность и ширина ценового коридора. "
        "Поэтому здесь фокус смещён с boxplot по сотням моделей на cohort-view."
    )
    if filtered_listings.empty or not active["price_eur"].notna().any():
        st.info("Нет данных для сравнения с текущими фильтрами.")
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
        comparison_full["price_spread_pct"] = (
            (comparison_full["max_price"] - comparison_full["min_price"])
            / comparison_full["median_price"].replace(0, np.nan)
            * 100
        ).round(1)

        comparison_view = comparison_full[comparison_full["count"] >= max(3, cohort_min_size)].copy()

        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("#### Карта моделей: цена vs ликвидность")
            st.caption(
                "Каждый пузырь — модель. Левее = ниже порог входа, ниже = быстрее обычно продаётся. "
                "Цвет показывает, насколько широкий у модели ценовой коридор."
            )
            liquid_view = comparison_view[comparison_view["avg_days_to_sell"].notna()].copy()
            if not liquid_view.empty:
                fig_market = px.scatter(
                    liquid_view,
                    x="median_price",
                    y="avg_days_to_sell",
                    size="count",
                    color="price_spread_pct",
                    color_continuous_scale="Turbo",
                    hover_data=["label", "min_price", "max_price", "capital_turns", "avg_mileage"],
                    labels={
                        "median_price": "Median price (EUR)",
                        "avg_days_to_sell": "Avg days to sell",
                        "count": "Listings",
                        "price_spread_pct": "Price spread %",
                    },
                    height=500,
                )
                fig_market.add_vline(
                    x=liquid_view["median_price"].median(),
                    line_dash="dot",
                    line_color="gray",
                    opacity=0.5,
                )
                fig_market.add_hline(
                    y=liquid_view["avg_days_to_sell"].median(),
                    line_dash="dot",
                    line_color="gray",
                    opacity=0.5,
                )
                st.plotly_chart(fig_market, width="stretch")

        with col_right:
            st.markdown("#### Модели с самым широким ценовым коридором")
            st.caption(
                "Широкий коридор означает большой разброс цен внутри одной модели. "
                "Это полезный сигнал для торга и поиска неэффективностей рынка."
            )
            spread_view = comparison_view.sort_values(
                ["price_spread_pct", "count"], ascending=[False, False]
            ).head(chart_top_n)
            if not spread_view.empty:
                fig_gap = px.bar(
                    spread_view,
                    x="price_spread_pct",
                    y="label",
                    orientation="h",
                    color="capital_turns" if spread_view["capital_turns"].notna().any() else "count",
                    labels={
                        "price_spread_pct": "Price spread %",
                        "label": "Model",
                        "capital_turns": "Turns / year",
                        "count": "Listings",
                    },
                    height=max(320, len(spread_view) * 30 + 80),
                )
                fig_gap.update_layout(yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_gap, width="stretch")

        comp_cols = ["label", "count", "median_price", "min_price", "max_price", "avg_mileage"]
        if "avg_days_to_sell" in comparison_full.columns:
            comp_cols += ["avg_days_to_sell", "weekly_turnover", "capital_turns", "price_spread_pct"]

        st.dataframe(
            comparison_full[comp_cols].rename(columns={
                "label": "Model", "count": "Listings", "median_price": "Median (EUR)",
                "min_price": "Min (EUR)", "max_price": "Max (EUR)", "avg_mileage": "Avg Mileage",
                "avg_days_to_sell": "Avg Days to Sell", "weekly_turnover": "Weekly Turnover %",
                "capital_turns": "Turns/Year", "price_spread_pct": "Price Spread %",
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
                "Price Spread %": st.column_config.NumberColumn(format="%.1f%%"),
            },
        )

        # --- Depreciation Curve ---
        st.subheader("Кривая амортизации")
        st.caption("Как цена падает по годам для каждой модели. R² показывает надёжность тренда (ближе к 1 = точнее).")
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
        st.subheader("Разница «частник vs дилер»")
        st.caption("Разница в медиане цены между частниками и дилерами. "
                   "Это и есть маржа перекупа: купить у частника, продать по дилерской цене.")
        if "seller_type" in active.columns:
            spread_data = active[active["price_eur"].notna() & active["seller_type"].notna()].copy()
            spread_data["label"] = spread_data["brand"] + " " + spread_data["model"]
            spread_agg = spread_data.pivot_table(
                values="price_eur", index="label", columns="seller_type",
                aggfunc="median", observed=True,
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
        st.subheader("Чувствительность цены к пробегу")
        st.caption("Падение цены за каждые 10 000 км. "
                   "Модели с низкой чувствительностью (близко к 0) — лучше для перепродажи с большим пробегом.")
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
        st.subheader("Оптимальные года для покупки")
        st.caption("Для каждой модели — какие года выпуска дают наибольшую маржу при перепродаже. "
                   "Высокий Gap % = большая разница между минимальной и медианной ценой.")
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
    st.subheader("Объявления по регионам")
    st.caption("Распределение объявлений и цен по районам Португалии. "
               "Помогает найти регионы с меньшей конкуренцией и лучшими ценами.")
    if active.empty or "district" not in active.columns:
        st.info("Нет данных о расположении.")
    else:
        geo_base = active[active["district_group"] != "Не указано"].copy()
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            district_counts = geo_base["district_group"].value_counts().reset_index()
            district_counts.columns = ["District", "Count"]
            fig_bar = px.bar(district_counts.head(15), x="Count", y="District", orientation="h",
                             title="Listings by District", height=450)
            fig_bar.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_bar, width="stretch")
        with col_g2:
            if geo_base["price_eur"].notna().any():
                district_prices = (
                    geo_base[geo_base["price_eur"].notna()].groupby("district_group")["price_eur"]
                    .median().reset_index().rename(columns={"price_eur": "Median Price (EUR)"})
                    .sort_values("Median Price (EUR)", ascending=False)
                )
                fig_price = px.bar(district_prices.head(15), x="Median Price (EUR)", y="district_group",
                                   orientation="h", title="Median Price by District", height=450)
                fig_price.update_layout(yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_price, width="stretch")

        if "city" in active.columns and active["city"].notna().any():
            city_stats = (
                geo_base[(geo_base["price_eur"].notna()) & (geo_base["city_group"] != "Не указано")]
                .groupby(["district_group", "city_group"])
                .agg(count=("price_eur", "size"), median=("price_eur", "median"))
                .reset_index().sort_values("count", ascending=False)
            )
            st.dataframe(
                city_stats.head(30).rename(columns={"district_group": "District", "city_group": "City",
                                                     "count": "Listings", "median": "Median (EUR)"}),
                width="stretch", hide_index=True,
                column_config={"Median (EUR)": st.column_config.NumberColumn(format="%d EUR")},
            )

        # --- Regional Price Differences ---
        st.subheader("Региональные различия в ценах")
        st.caption("Цены на одну и ту же модель в разных районах. Арбитраж = купить дёшево в одном районе, продать дороже в другом.")
        geo_active = geo_base[geo_base["price_eur"].notna()].copy()
        if not geo_active.empty:
            geo_active["label"] = geo_active["brand"] + " " + geo_active["model"]
            model_district_counts = geo_active.groupby("label")["district_group"].nunique()
            geo_candidates = sorted(model_district_counts[model_district_counts >= 2].index.tolist())

            if geo_candidates:
                selected_geo = st.multiselect("Models to compare across districts", geo_candidates,
                                               default=geo_candidates[:5] if len(geo_candidates) >= 5 else geo_candidates,
                                               key="geo_models")
                if selected_geo:
                    geo_subset = geo_active[geo_active["label"].isin(selected_geo)]
                    pivot = geo_subset.pivot_table(
                        values="price_eur", index="label", columns="district_group", aggfunc="median", observed=True
                    )
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
        st.subheader("Плотность конкуренции")
        st.caption("Сколько аналогичных авто продаётся в районе. "
                   "Высокая насыщенность = больше конкурентов, дольше продажа.")
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
            st.info("Недостаточно данных для анализа конкуренции.")

# ---- TAB: Срок жизни (Listing Lifespan Analysis) ----------------------------
with tab_lifespan:
    st.subheader("Срок жизни объявлений")
    st.caption("Анализ того, как долго объявления остаются на OLX. "
               "Неактивные = проданные или снятые. Короткий срок жизни = высокий спрос или выгодная цена.")

    life = filtered_listings.copy()

    if life.empty or "first_seen_at" not in life.columns:
        st.info("Нет данных для анализа срока жизни.")
    else:
        life["first_seen"] = pd.to_datetime(life["first_seen_at"]).dt.tz_localize(None)
        life["last_seen"] = pd.to_datetime(life["last_seen_at"]).dt.tz_localize(None)
        now = pd.Timestamp.now()

        life["lifespan_days"] = np.where(
            life["is_active"],
            (now - life["first_seen"]).dt.days,
            (life["last_seen"] - life["first_seen"]).dt.days,
        )
        life["status"] = np.where(life["is_active"], "На OLX", "Продано / снято")

        inactive_life = life[~life["is_active"]].copy()
        active_life = life[life["is_active"]].copy()

        # --- KPIs ---
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Всего объявлений", f"{len(life):,}")
        k2.metric("Активных", f"{len(active_life):,}")
        k3.metric("Продано / снято", f"{len(inactive_life):,}")
        if not inactive_life.empty:
            median_life = inactive_life["lifespan_days"].median()
            k4.metric("Медиана срока (снятые)", f"{median_life:.0f} дн.")
        else:
            k4.metric("Медиана срока (снятые)", "—")

        st.divider()

        # --- 1. Distribution + Survival curve ---
        col_l1, col_l2 = st.columns(2)

        with col_l1:
            st.markdown("#### Распределение срока жизни")
            st.caption("Сколько объявлений продержались определённое количество дней. "
                       "Пик слева = быстро ушедшие с рынка (высокий спрос или заниженная цена).")
            if not inactive_life.empty:
                fig_life_hist = px.histogram(
                    inactive_life, x="lifespan_days", nbins=30,
                    labels={"lifespan_days": "Дней на OLX"},
                    color_discrete_sequence=["#3498db"],
                    height=400,
                )
                fig_life_hist.update_layout(
                    xaxis_title="Дней на OLX",
                    yaxis_title="Количество объявлений",
                    showlegend=False,
                )
                st.plotly_chart(fig_life_hist, use_container_width=True)
            else:
                st.info("Нет снятых объявлений для анализа.")

        with col_l2:
            st.markdown("#### Кривая выживаемости")
            st.caption("Какой % объявлений ещё на рынке через N дней. "
                       "Резкое падение = массовые продажи. Плоский участок = «зависшие» объявления.")
            if not inactive_life.empty and len(inactive_life) >= 5:
                max_days = int(inactive_life["lifespan_days"].quantile(0.95))
                days_range = range(0, max(max_days + 1, 2))
                total = len(inactive_life)
                survival = [{"Дней": d, "На рынке %": (inactive_life["lifespan_days"] > d).sum() / total * 100}
                            for d in days_range]
                surv_df = pd.DataFrame(survival)

                fig_surv = px.line(
                    surv_df, x="Дней", y="На рынке %",
                    labels={"Дней": "Дней на OLX", "На рынке %": "% ещё на рынке"},
                    height=400,
                )
                fig_surv.update_traces(fill="tozeroy", fillcolor="rgba(52, 152, 219, 0.15)")
                fig_surv.update_layout(yaxis_range=[0, 105])
                for pct_label, pct in [("25% ушли", 75), ("50% ушли", 50), ("75% ушли", 25)]:
                    day_at = surv_df[surv_df["На рынке %"] <= pct]
                    if not day_at.empty:
                        d = int(day_at.iloc[0]["Дней"])
                        fig_surv.add_vline(x=d, line_dash="dot", line_color="gray", opacity=0.5)
                        fig_surv.add_annotation(x=d, y=pct + 5, text=f"{pct_label}: {d}д",
                                                showarrow=False, font=dict(size=10, color="gray"))
                st.plotly_chart(fig_surv, use_container_width=True)

        # --- 2. Median lifespan by brand ---
        st.markdown("#### Медианный срок жизни по маркам")
        st.caption("Какие марки продаются быстрее всего? Зелёные = быстрее медианы, красные = медленнее. "
                   "Короткий срок = высокий спрос, длинный = возможно завышена цена.")
        if not inactive_life.empty:
            brand_life = (
                inactive_life.groupby("brand")
                .agg(median_days=("lifespan_days", "median"), count=("lifespan_days", "size"))
                .reset_index()
            )
            brand_life = brand_life[brand_life["count"] >= 3].sort_values("median_days")
            if not brand_life.empty:
                overall_med = brand_life["median_days"].median()
                colors = ["#2ecc71" if d <= overall_med else "#e74c3c" for d in brand_life["median_days"]]
                fig_brand_life = go.Figure(go.Bar(
                    x=brand_life["median_days"], y=brand_life["brand"],
                    orientation="h", marker_color=colors,
                    text=[f"{d:.0f} дн. ({c} авто)" for d, c in zip(brand_life["median_days"], brand_life["count"])],
                    textposition="auto",
                ))
                fig_brand_life.update_layout(
                    xaxis_title="Медиана дней на OLX", yaxis_title="",
                    height=max(300, len(brand_life) * 30 + 80),
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(fig_brand_life, use_container_width=True)

        # --- 3. Price vs Lifespan + Lifespan by price range ---
        col_l3, col_l4 = st.columns(2)

        with col_l3:
            st.markdown("#### Цена vs Срок жизни")
            st.caption("Зависимость цены от времени на OLX. "
                       "Дешёвые авто обычно уходят быстрее. Точки внизу слева = горячие лоты.")
            price_inactive = inactive_life[inactive_life["price_eur"].notna()]
            if not price_inactive.empty and len(price_inactive) >= 5:
                fig_price_life = px.scatter(
                    price_inactive, x="lifespan_days", y="price_eur",
                    color="brand",
                    hover_data=["model", "year", "mileage_km", "url"],
                    labels={"lifespan_days": "Дней на OLX", "price_eur": "Цена (EUR)", "brand": "Марка"},
                    height=450, opacity=0.7,
                )
                fig_price_life.update_layout(hovermode="closest")
                plotly_chart_with_click(fig_price_life, price_inactive, key="price_life_scatter", use_container_width=True)

        with col_l4:
            st.markdown("#### Срок жизни по ценовым диапазонам")
            st.caption("В каком ценовом сегменте объявления уходят быстрее. "
                       "Самые ликвидные ценовые точки видны сразу.")
            price_life = inactive_life[inactive_life["price_eur"].notna()].copy()
            if not price_life.empty and len(price_life) >= 5:
                price_life["price_bucket"] = pd.cut(
                    price_life["price_eur"],
                    bins=[0, 3000, 5000, 8000, 12000, 18000, 25000, 40000, 999999],
                    labels=["<3k", "3-5k", "5-8k", "8-12k", "12-18k", "18-25k", "25-40k", "40k+"],
                )
                bucket_life = (
                    price_life.groupby("price_bucket", observed=True)
                    .agg(median_days=("lifespan_days", "median"), count=("lifespan_days", "size"))
                    .reset_index()
                )
                bucket_life = bucket_life[bucket_life["count"] >= 2]
                if not bucket_life.empty:
                    fig_bucket = px.bar(
                        bucket_life, x="price_bucket", y="median_days",
                        text=[f"{d:.0f}д ({c})" for d, c in zip(bucket_life["median_days"], bucket_life["count"])],
                        labels={"price_bucket": "Ценовой диапазон (EUR)", "median_days": "Медиана дней"},
                        color="median_days", color_continuous_scale="RdYlGn_r",
                        height=400,
                    )
                    fig_bucket.update_traces(textposition="auto")
                    fig_bucket.update_layout(showlegend=False)
                    st.plotly_chart(fig_bucket, use_container_width=True)

        # --- 4. Quick sellers ---
        st.divider()
        st.markdown("#### Быстрые продажи (до 7 дней)")
        st.caption("Объявления, которые были на OLX менее недели — самые горячие лоты. "
                   "Анализ характеристик помогает понять, что пользуется максимальным спросом.")

        quick = inactive_life[inactive_life["lifespan_days"] <= 7].copy()
        if quick.empty:
            st.info("Нет объявлений, снятых менее чем за 7 дней.")
        else:
            st.success(f"Найдено {len(quick)} быстрых продаж из {len(inactive_life)} снятых "
                       f"({len(quick)/max(len(inactive_life),1)*100:.0f}%)")

            col_q1, col_q2 = st.columns(2)

            with col_q1:
                st.markdown("**Сравнение: быстрые vs долгие продажи**")
                st.caption("Что отличает быстро проданные авто от зависших.")
                slow = inactive_life[inactive_life["lifespan_days"] > 30].copy()

                comparison_rows = []
                for metric, label, fmt in [
                    ("price_eur", "Медиана цены (EUR)", "{:,.0f}"),
                    ("year", "Медиана года", "{:.0f}"),
                    ("mileage_km", "Медиана пробега (км)", "{:,.0f}"),
                    ("engine_cc", "Медиана объёма (cc)", "{:,.0f}"),
                ]:
                    if metric in quick.columns:
                        q_val = quick[metric].median()
                        s_val = slow[metric].median() if not slow.empty else None
                        comparison_rows.append({
                            "Показатель": label,
                            "Быстрые (< 7д)": fmt.format(q_val) if pd.notna(q_val) else "—",
                            "Долгие (> 30д)": fmt.format(s_val) if s_val is not None and pd.notna(s_val) else "—",
                        })

                if "fuel_type" in quick.columns:
                    q_fuel = quick["fuel_type"].value_counts(normalize=True).head(3)
                    s_fuel = slow["fuel_type"].value_counts(normalize=True).head(3) if not slow.empty else pd.Series(dtype=float)
                    comparison_rows.append({
                        "Показатель": "Топ топливо",
                        "Быстрые (< 7д)": ", ".join(f"{f} ({p:.0%})" for f, p in q_fuel.items()),
                        "Долгие (> 30д)": ", ".join(f"{f} ({p:.0%})" for f, p in s_fuel.items()) if not s_fuel.empty else "—",
                    })

                if "transmission" in quick.columns:
                    q_auto = (quick["transmission"] == "Automática").mean()
                    s_auto = (slow["transmission"] == "Automática").mean() if not slow.empty else None
                    comparison_rows.append({
                        "Показатель": "Доля АКПП",
                        "Быстрые (< 7д)": f"{q_auto:.0%}",
                        "Долгие (> 30д)": f"{s_auto:.0%}" if s_auto is not None else "—",
                    })

                # Condition comparison
                for col, label in [
                    ("desc_mentions_accident", "Доля с упоминанием ДТП в описании"),
                    ("desc_mentions_repair", "Доля с упоминанием ремонта в описании"),
                ]:
                    if col in quick.columns and quick[col].notna().any():
                        q_pct = (quick[col] == True).mean()
                        s_pct = (slow[col] == True).mean() if not slow.empty and col in slow.columns else None
                        comparison_rows.append({
                            "Показатель": label,
                            "Быстрые (< 7д)": f"{q_pct:.0%}",
                            "Долгие (> 30д)": f"{s_pct:.0%}" if s_pct is not None else "—",
                        })

                if comparison_rows:
                    st.dataframe(pd.DataFrame(comparison_rows), hide_index=True)

            with col_q2:
                st.markdown("**Топ марки по быстрым продажам**")
                st.caption("Процент показывает долю быстрых продаж от всех снятых объявлений марки.")
                q_brands = quick["brand"].value_counts().reset_index()
                q_brands.columns = ["brand", "quick_count"]
                total_brands = inactive_life["brand"].value_counts().reset_index()
                total_brands.columns = ["brand", "total_count"]
                q_brands = q_brands.merge(total_brands, on="brand")
                q_brands["quick_pct"] = (q_brands["quick_count"] / q_brands["total_count"] * 100).round(1)
                q_brands = q_brands.sort_values("quick_count", ascending=False).head(10)

                fig_q_brands = px.bar(
                    q_brands, x="quick_count", y="brand", orientation="h",
                    text=[f"{c} ({p:.0f}%)" for c, p in zip(q_brands["quick_count"], q_brands["quick_pct"])],
                    labels={"quick_count": "Быстрых продаж", "brand": "Марка"},
                    color="quick_pct", color_continuous_scale="Greens",
                    height=350,
                )
                fig_q_brands.update_traces(textposition="auto")
                fig_q_brands.update_layout(yaxis=dict(autorange="reversed"), showlegend=False)
                st.plotly_chart(fig_q_brands, use_container_width=True)

            # Quick sellers table
            quick_sorted = quick.sort_values("lifespan_days")
            q_cols = ["brand", "model", "year", "price_eur", "mileage_km",
                      "fuel_type", "transmission", "district", "lifespan_days"]
            if "url" in quick_sorted.columns:
                q_cols.append("url")
            avail_q = [c for c in q_cols if c in quick_sorted.columns]
            st.dataframe(
                quick_sorted.head(30)[avail_q].rename(columns={
                    "brand": "Марка", "model": "Модель", "year": "Год",
                    "price_eur": "Цена (EUR)", "mileage_km": "Пробег",
                    "fuel_type": "Топливо", "transmission": "КПП",
                    "district": "Район", "lifespan_days": "Дней на OLX",
                    "url": "Ссылка",
                }),
                hide_index=True,
                column_config={
                    "Цена (EUR)": st.column_config.NumberColumn(format="%d EUR"),
                    "Пробег": st.column_config.NumberColumn(format="%,d км"),
                    "Ссылка": st.column_config.LinkColumn("Ссылка", display_text="Открыть"),
                },
            )

        # --- 5. Lifespan by model ---
        st.divider()
        st.markdown("#### Срок жизни по моделям")
        st.caption("Медианное время на OLX для каждой модели. "
                   "Зелёные = продаются быстрее среднего, красные = медленнее.")

        if not inactive_life.empty:
            inactive_life["label"] = inactive_life["brand"] + " " + inactive_life["model"]
            model_life = (
                inactive_life.groupby("label")
                .agg(median_days=("lifespan_days", "median"), count=("lifespan_days", "size"))
                .reset_index()
            )
            model_life = model_life[model_life["count"] >= 3].sort_values("median_days")

            if not model_life.empty:
                overall_median = inactive_life["lifespan_days"].median()
                colors_m = ["#2ecc71" if d <= overall_median else "#e74c3c" for d in model_life["median_days"]]
                fig_model_life = go.Figure(go.Bar(
                    x=model_life["median_days"], y=model_life["label"],
                    orientation="h", marker_color=colors_m,
                    text=[f"{d:.0f}д ({c})" for d, c in zip(model_life["median_days"], model_life["count"])],
                    textposition="auto",
                ))
                fig_model_life.add_vline(x=overall_median, line_dash="dash", line_color="gray",
                                          annotation_text=f"Медиана: {overall_median:.0f}д")
                fig_model_life.update_layout(
                    xaxis_title="Медиана дней на OLX", yaxis_title="",
                    height=max(400, len(model_life) * 28 + 80),
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(fig_model_life, use_container_width=True)

        # --- 6. Lifespan by district ---
        st.markdown("#### Срок жизни по районам")
        st.caption("Где машины продаются быстрее? Короткий срок в районе = высокий спрос или мало предложений.")
        if not inactive_life.empty and "district" in inactive_life.columns:
            dist_life = (
                inactive_life[inactive_life["district"].notna()]
                .groupby("district")
                .agg(median_days=("lifespan_days", "median"), count=("lifespan_days", "size"))
                .reset_index()
            )
            dist_life = dist_life[dist_life["count"] >= 3].sort_values("median_days")
            if not dist_life.empty:
                fig_dist_life = px.bar(
                    dist_life, x="median_days", y="district", orientation="h",
                    text=[f"{d:.0f}д ({c})" for d, c in zip(dist_life["median_days"], dist_life["count"])],
                    labels={"median_days": "Медиана дней", "district": "Район"},
                    color="median_days", color_continuous_scale="RdYlGn_r",
                    height=max(300, len(dist_life) * 30 + 80),
                )
                fig_dist_life.update_traces(textposition="auto")
                fig_dist_life.update_layout(yaxis=dict(autorange="reversed"), showlegend=False)
                st.plotly_chart(fig_dist_life, use_container_width=True)

        # --- 7. Active listings age distribution ---
        st.divider()
        st.markdown("#### Возраст активных объявлений")
        st.caption("Сколько дней текущие объявления уже на OLX. "
                   "Объявления-долгожители (> 30д) часто имеют завышенную цену или проблемы.")
        if not active_life.empty:
            fig_active_age = px.histogram(
                active_life, x="lifespan_days", nbins=25,
                labels={"lifespan_days": "Дней на OLX"},
                color_discrete_sequence=["#e67e22"],
                height=350,
            )
            fig_active_age.update_layout(
                xaxis_title="Дней на OLX", yaxis_title="Количество объявлений",
                showlegend=False,
            )
            # Mark 30-day threshold
            fig_active_age.add_vline(x=30, line_dash="dash", line_color="red", opacity=0.5,
                                      annotation_text="30 дней", annotation_position="top right")
            st.plotly_chart(fig_active_age, use_container_width=True)

            stale = active_life[active_life["lifespan_days"] > 30]
            if not stale.empty:
                st.caption(f"**{len(stale)}** объявлений висят более 30 дней "
                           f"({len(stale)/max(len(active_life),1)*100:.0f}% от активных). "
                           "Они могут быть готовы к снижению цены — потенциал для торга.")


# ---- TAB: Time-to-Sale / Price Optimization --------------------------------
with tab_tos:
    st.subheader("Скорость продажи и оптимизация цены")
    st.caption("Анализ времени продажи по моделям и оптимизация цены для максимизации P(продажа) × маржа.")

    from src.analytics.time_to_sale import (
        build_tos_dataset, fit_tos_model, predict_sale_probability,
        optimize_price, compute_tos_stats, DEFAULT_TOS_WEIGHTS, TOS_FEATURE_COLUMNS,
    )

    tos_stats = compute_tos_stats(listings_df)

    if tos_stats.empty:
        st.warning("Нет данных о проданных объявлениях. Нужно больше данных для анализа.")
    else:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        total_sold = int(tos_stats["sold_count"].sum())
        overall_median = float(tos_stats.loc[tos_stats["sold_count"] >= 3, "median_days"].median()) if len(tos_stats[tos_stats["sold_count"] >= 3]) > 0 else 0
        fast_pct = float(tos_stats.loc[tos_stats["sold_count"] >= 3, "pct_under_30d"].mean()) if len(tos_stats[tos_stats["sold_count"] >= 3]) > 0 else 0

        col1.metric("Продано всего", total_sold)
        col2.metric("Медиана дней", f"{overall_median:.0f}")
        col3.metric("% за 30 дней", f"{fast_pct:.0f}%")
        col4.metric("Моделей с данными", len(tos_stats[tos_stats["sold_count"] >= 3]))

        # Per-model table
        st.subheader("Время продажи по моделям")
        show_stats = tos_stats[tos_stats["sold_count"] >= 2].copy()
        if not show_stats.empty:
            show_stats = show_stats.rename(columns={
                "brand": "Марка", "model": "Модель",
                "sold_count": "Продано", "median_days": "Медиана дн.",
                "mean_days": "Среднее дн.", "pct_under_30d": "% < 30д",
                "sale_rate_pct": "% продаж", "total_count": "Всего",
            })
            st.dataframe(show_stats, hide_index=True, use_container_width=True)

        # Histogram of days to sale
        sold_listings = listings_df[listings_df["is_active"] == False].copy()
        if not sold_listings.empty:
            first_s = pd.to_datetime(sold_listings["first_seen_at"]).dt.tz_localize(None)
            if "deactivated_at" in sold_listings.columns:
                end_s = pd.to_datetime(sold_listings["deactivated_at"]).dt.tz_localize(None).fillna(
                    pd.to_datetime(sold_listings["last_seen_at"]).dt.tz_localize(None))
            else:
                end_s = pd.to_datetime(sold_listings["last_seen_at"]).dt.tz_localize(None)
            sold_listings["days_to_sale"] = (end_s - first_s).dt.total_seconds() / 86400
            sold_valid = sold_listings[sold_listings["days_to_sale"].between(0.5, 180)]

            if not sold_valid.empty:
                fig_hist = px.histogram(
                    sold_valid, x="days_to_sale", nbins=30,
                    title="Распределение дней до продажи",
                    labels={"days_to_sale": "Дней до продажи", "count": "Кол-во"},
                )
                fig_hist.add_vline(x=30, line_dash="dash", line_color="red",
                                  annotation_text="30 дней")
                st.plotly_chart(fig_hist, use_container_width=True)

        # Price optimization tool
        st.divider()
        st.subheader("Калькулятор оптимальной цены")
        st.caption("Найти цену, максимизирующую P(продажа за N дней) × маржа")

        # Load trained weights if available
        import yaml as _yaml
        _weights_path = _project_root / "data" / "tos_weights.yaml"
        if _weights_path.exists():
            with open(_weights_path) as _f:
                _saved = _yaml.safe_load(_f) or {}
            _tos_weights = _saved.get("weights", DEFAULT_TOS_WEIGHTS)
            st.caption("Используются обученные веса модели")
        else:
            _tos_weights = DEFAULT_TOS_WEIGHTS
            st.caption("Используются веса по умолчанию (запустите `train-tos` для обучения)")

        available_brands = sorted(listings_df["brand"].dropna().unique())

        opt_col1, opt_col2 = st.columns(2)
        with opt_col1:
            opt_brand = st.selectbox("Марка", available_brands, key="tos_brand")
            brand_models = sorted(
                listings_df[listings_df["brand"] == opt_brand]["model"].dropna().unique()
            ) if opt_brand else []
            opt_model = st.selectbox("Модель", brand_models, key="tos_model")
            opt_price = st.number_input("Текущая цена (EUR)", min_value=500, max_value=200000,
                                        value=10000, step=500, key="tos_price")
        with opt_col2:
            opt_year = st.number_input("Год", min_value=1990, max_value=2026, value=2015, key="tos_year")
            opt_mileage = st.number_input("Пробег (км)", min_value=0, max_value=500000,
                                          value=150000, step=10000, key="tos_mileage")
            opt_target = st.slider("Целевой срок продажи (дней)", 7, 90, 30, key="tos_target")
            opt_min_price = st.number_input("Мин. приемлемая цена (EUR)", min_value=0,
                                            max_value=200000, value=0, step=500, key="tos_min")

        if opt_brand and opt_model and st.button("Оптимизировать цену", key="tos_run"):
            matching = listings_df[
                (listings_df["brand"] == opt_brand) &
                (listings_df["model"] == opt_model) &
                listings_df["price_eur"].notna()
            ]
            if matching.empty:
                st.error(f"Нет данных для {opt_brand} {opt_model}")
            else:
                market_median = float(matching["price_eur"].median())

                features = {
                    "price_ratio": opt_price / market_median,
                    "mileage_norm": min(opt_mileage, 500000) / 500000,
                    "year_norm": (opt_year - 2000) / 25,
                    "is_diesel": 0.0,
                    "is_professional": 0.0,
                    "price_drop_rate": 0.0,
                    "has_accident": 0.0,
                    "has_repair": 0.0,
                }

                result = optimize_price(
                    features, market_median, _tos_weights,
                    target_days=opt_target,
                    min_acceptable_price=opt_min_price if opt_min_price > 0 else None,
                )

                r1, r2, r3 = st.columns(3)
                r1.metric("Оптимальная цена", f"{result['optimal_price']:,} EUR")
                r2.metric("P(продажа)", f"{result['sale_probability']:.0%}")
                r3.metric("Медиана рынка", f"{market_median:,.0f} EUR")

                # Price curve chart
                curve = pd.DataFrame(result["price_curve"])
                fig_curve = go.Figure()
                fig_curve.add_trace(go.Scatter(
                    x=curve["price"], y=curve["probability"],
                    mode="lines", name="P(продажа)",
                    yaxis="y1",
                ))
                fig_curve.add_trace(go.Scatter(
                    x=curve["price"], y=curve["expected_value"],
                    mode="lines", name="Ожидаемая маржа (EUR)",
                    yaxis="y2", line=dict(dash="dot"),
                ))
                fig_curve.add_vline(x=result["optimal_price"], line_dash="dash",
                                   line_color="green", annotation_text="Оптимум")
                fig_curve.add_vline(x=opt_price, line_dash="dash",
                                   line_color="red", annotation_text="Текущая")
                fig_curve.update_layout(
                    title="Цена vs Вероятность продажи",
                    xaxis_title="Цена (EUR)",
                    yaxis=dict(title="P(продажа)", side="left"),
                    yaxis2=dict(title="Ожидаемая маржа (EUR)", side="right", overlaying="y"),
                    legend=dict(x=0.01, y=0.99),
                )
                st.plotly_chart(fig_curve, use_container_width=True)


# ---- TAB: Portfolio / Deal Tracker ----------------------------------------
with tab_portfolio:
    st.subheader("Портфель сделок")
    st.caption("Учёт ваших покупок, ремонтов и продаж автомобилей. "
               "Добавляйте сделки и отслеживайте ROI.")

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
        st.info("Нет отслеживаемых сделок. Добавьте первую сделку выше.")
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
    st.subheader("Объявления без поколения")
    st.caption("Объявления, для которых не удалось определить поколение автомобиля. "
               "Проверьте и при необходимости добавьте в конфигурацию поколений.")

    if unmatched_df.empty:
        st.info("Нет объявлений без поколения. Все машины распознаны.")
    else:
        st.metric("Unmatched", len(unmatched_df))

        # Summary: missing brand+model combos
        summary = (
            unmatched_df.groupby(["brand", "model", "reason"])
            .agg(count=("olx_id", "size"), years=("year", lambda y: sorted(set(int(v) for v in y.dropna()))))
            .reset_index()
            .sort_values("count", ascending=False)
        )
        st.subheader("Недостающие поколения")
        st.dataframe(summary.rename(columns={
            "brand": "Brand", "model": "Model", "reason": "Reason",
            "count": "Count", "years": "Years",
        }), hide_index=True)

        # Full table
        st.subheader("Все нераспознанные")
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
