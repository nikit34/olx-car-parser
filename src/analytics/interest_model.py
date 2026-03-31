"""Ranking model for shortlist candidates."""

from __future__ import annotations

import numpy as np
import pandas as pd


FEEDBACK_POSITIVE_LABELS = {"interesting", "bought"}
FEEDBACK_NEGATIVE_LABELS = {"skipped"}

INTEREST_FEATURE_COLUMNS = [
    "signal_strength",
    "profit_score",
    "roi_score",
    "confidence_score",
    "liquidity_score",
    "seller_pressure_score",
    "staleness_score",
    "risk_penalty",
]

DEFAULT_INTEREST_WEIGHTS = {
    "bias": -1.1,
    "signal_strength": 3.2,
    "profit_score": 1.6,
    "roi_score": 1.1,
    "confidence_score": 1.1,
    "liquidity_score": 0.9,
    "seller_pressure_score": 0.8,
    "staleness_score": 0.3,
    "risk_penalty": -2.1,
}


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30, 30)))


def build_interest_reason(row: pd.Series) -> str:
    """Return short, human-readable explanation for shortlist ranking."""
    reasons = []
    undervaluation = row.get("undervaluation_pct")
    est_profit = row.get("est_profit_eur")
    sample_size = row.get("sample_size")
    avg_days = row.get("avg_days_to_sell")
    drop_per_day = row.get("price_drop_per_day")

    if pd.notna(undervaluation) and undervaluation >= 12:
        reasons.append(f"below model by {undervaluation:.0f}%")
    if pd.notna(est_profit) and est_profit >= 1000:
        reasons.append(f"profit ~{int(est_profit):,} EUR")
    if pd.notna(sample_size) and sample_size >= 8:
        reasons.append(f"{int(sample_size)} comparable listings")
    if pd.notna(avg_days) and avg_days <= 30:
        reasons.append("liquid market")
    if pd.notna(drop_per_day) and drop_per_day < -20:
        reasons.append("seller is reducing price")

    return ", ".join(reasons[:3]) if reasons else "market anomaly, manual review needed"


def build_interest_features(active_df: pd.DataFrame, deals_df: pd.DataFrame) -> pd.DataFrame:
    """Build normalized feature matrix for candidate ranking."""
    if active_df.empty:
        return pd.DataFrame()

    base_cols = [
        "olx_id",
        "predicted_price",
        "undervaluation_pct",
        "flip_score",
        "sample_size",
        "avg_days_to_sell",
        "price_drop_per_day",
        "est_profit_eur",
        "est_roi_pct",
        "deal_profile",
        "confidence",
    ]
    deal_features = (
        deals_df[[c for c in base_cols if c in deals_df.columns]].copy()
        if not deals_df.empty
        else pd.DataFrame()
    )

    merged = active_df.copy()
    if not deal_features.empty and "olx_id" in merged.columns and "olx_id" in deal_features.columns:
        merged = merged.merge(deal_features, on="olx_id", how="left")

    merged = merged[merged["price_eur"].notna()].copy()
    if "url" in merged.columns:
        merged = merged[merged["url"].notna() & merged["url"].astype(str).str.startswith("http")].copy()

    if merged.empty:
        return pd.DataFrame()

    merged["signal_strength"] = (
        merged.get("undervaluation_pct", pd.Series(index=merged.index, dtype=float))
        .fillna(0)
        .clip(lower=0, upper=35)
        / 35
    )
    merged["profit_score"] = (
        merged.get("est_profit_eur", pd.Series(index=merged.index, dtype=float))
        .fillna(0)
        .clip(lower=0, upper=4000)
        / 4000
    )
    merged["roi_score"] = (
        merged.get("est_roi_pct", pd.Series(index=merged.index, dtype=float))
        .fillna(0)
        .clip(lower=0, upper=35)
        / 35
    )
    merged["confidence_score"] = (
        merged.get("sample_size", pd.Series(index=merged.index, dtype=float))
        .fillna(0)
        .clip(lower=0, upper=15)
        / 15
    )

    if "avg_days_to_sell" in merged.columns:
        merged["liquidity_score"] = (
            (45 - merged["avg_days_to_sell"].fillna(45).clip(lower=7, upper=90)) / 38
        ).clip(lower=0, upper=1)
    else:
        merged["liquidity_score"] = 0.25

    if "price_drop_per_day" in merged.columns:
        merged["seller_pressure_score"] = (
            (-merged["price_drop_per_day"].fillna(0)).clip(lower=0, upper=150) / 150
        )
    else:
        merged["seller_pressure_score"] = 0.0

    if "days_listed" in merged.columns:
        merged["staleness_score"] = merged["days_listed"].fillna(0).clip(lower=0, upper=60) / 60
    else:
        merged["staleness_score"] = 0.0

    # Do not let LLM-interpreted condition fields decide whether a car is "good".
    # Quality/condition flags stay in the UI as review hints, but shortlist scoring
    # is driven only by market behaviour and explicit user feedback.
    merged["risk_penalty"] = 0.0

    prior_logit = np.full(len(merged), DEFAULT_INTEREST_WEIGHTS["bias"], dtype=float)
    for feature in INTEREST_FEATURE_COLUMNS:
        prior_logit += merged[feature].fillna(0).to_numpy(dtype=float) * DEFAULT_INTEREST_WEIGHTS[feature]
    merged["prior_probability"] = _sigmoid(prior_logit)

    return merged


def _fit_logistic_weights(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    initial_weights: dict[str, float] | None = None,
    epochs: int = 250,
    learning_rate: float = 0.35,
    l2_penalty: float = 0.01,
) -> dict[str, float]:
    """Fit a tiny logistic model using gradient descent."""
    if train_df.empty:
        return dict(initial_weights or DEFAULT_INTEREST_WEIGHTS)

    base = dict(initial_weights or DEFAULT_INTEREST_WEIGHTS)
    weights = np.array([base.get(col, 0.0) for col in feature_cols], dtype=float)
    bias = float(base.get("bias", 0.0))

    x_values = train_df[feature_cols].fillna(0).to_numpy(dtype=float)
    y_values = train_df["training_label"].to_numpy(dtype=float)

    for _ in range(epochs):
        logits = x_values @ weights + bias
        predictions = _sigmoid(logits)
        error = predictions - y_values
        grad_w = (x_values.T @ error) / len(train_df) + l2_penalty * weights
        grad_b = error.mean()
        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b

    fitted = {"bias": float(bias)}
    fitted.update({col: float(weight) for col, weight in zip(feature_cols, weights)})
    return fitted


def _prepare_feedback_frame(feedback_df: pd.DataFrame | None) -> pd.DataFrame:
    """Normalize explicit user feedback to a stable schema."""
    if feedback_df is None or feedback_df.empty or "olx_id" not in feedback_df.columns:
        return pd.DataFrame()

    label_col = "feedback_label" if "feedback_label" in feedback_df.columns else "label"
    if label_col not in feedback_df.columns:
        return pd.DataFrame()

    out = feedback_df.copy()
    out["olx_id"] = out["olx_id"].astype(str).str.strip()
    out["feedback_label"] = out[label_col].astype(str).str.strip().str.lower()
    out = out[
        out["olx_id"].ne("")
        & out["feedback_label"].isin(FEEDBACK_POSITIVE_LABELS | FEEDBACK_NEGATIVE_LABELS)
    ].copy()
    if out.empty:
        return pd.DataFrame()

    rename_map = {}
    if "notes" in out.columns and "feedback_notes" not in out.columns:
        rename_map["notes"] = "feedback_notes"
    if "updated_at" in out.columns and "feedback_updated_at" not in out.columns:
        rename_map["updated_at"] = "feedback_updated_at"
    if rename_map:
        out = out.rename(columns=rename_map)

    keep_cols = ["olx_id", "feedback_label"]
    for column in [
        "feedback_notes",
        "feedback_updated_at",
        "url",
        "title",
        "brand",
        "model",
        "year",
        "price_eur",
    ]:
        if column in out.columns:
            keep_cols.append(column)

    sort_columns = ["feedback_updated_at"] if "feedback_updated_at" in out.columns else None
    if sort_columns:
        out = out.sort_values(sort_columns, ascending=False, na_position="last")

    return out[keep_cols].drop_duplicates(subset=["olx_id"], keep="first")


def _attach_training_sources(
    feature_df: pd.DataFrame,
    portfolio_df: pd.DataFrame | None,
    feedback_df: pd.DataFrame | None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Attach portfolio and explicit feedback labels to scored candidates."""
    labeled = feature_df.copy()
    labeled["portfolio_positive"] = False

    if portfolio_df is not None and not portfolio_df.empty and "olx_listing_id" in portfolio_df.columns:
        positive_ids = {
            str(value)
            for value in portfolio_df["olx_listing_id"].dropna().astype(str)
            if value.strip()
        }
        if positive_ids:
            labeled["portfolio_positive"] = labeled["olx_id"].astype(str).isin(positive_ids)

    normalized_feedback = _prepare_feedback_frame(feedback_df)
    if normalized_feedback.empty:
        labeled["feedback_label"] = pd.NA
        labeled["feedback_notes"] = pd.NA
        labeled["feedback_updated_at"] = pd.NaT
    else:
        labeled = labeled.merge(normalized_feedback, on="olx_id", how="left", suffixes=("", "_feedback"))

    labeled["feedback_positive"] = labeled["feedback_label"].isin(FEEDBACK_POSITIVE_LABELS)
    labeled["feedback_negative"] = labeled["feedback_label"].isin(FEEDBACK_NEGATIVE_LABELS)

    stats = {
        "portfolio_positive_count": int(labeled["portfolio_positive"].sum()),
        "feedback_positive_count": int(labeled["feedback_positive"].sum()),
        "feedback_negative_count": int(labeled["feedback_negative"].sum()),
    }
    return labeled, stats


def _build_training_frame(
    labeled_df: pd.DataFrame,
    min_positive_labels: int = 3,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Prepare pseudo-labeled training data from portfolio and explicit feedback."""
    if labeled_df.empty:
        return pd.DataFrame(), {
            "training_positive_count": 0,
            "training_negative_count": 0,
            "feedback_label_count": 0,
        }

    labeled = labeled_df.copy()
    labeled["training_label"] = np.nan
    labeled.loc[labeled["portfolio_positive"], "training_label"] = 1.0
    labeled.loc[labeled["feedback_positive"], "training_label"] = 1.0
    labeled.loc[labeled["feedback_negative"], "training_label"] = 0.0

    positive_count = int((labeled["training_label"] == 1.0).sum())
    negative_count = int((labeled["training_label"] == 0.0).sum())
    feedback_label_count = int(labeled["feedback_label"].notna().sum())

    min_required_positives = (
        1 if feedback_label_count >= 2 and positive_count >= 1 and negative_count >= 1 else min_positive_labels
    )
    if positive_count < min_required_positives:
        return pd.DataFrame(), {
            "training_positive_count": positive_count,
            "training_negative_count": negative_count,
            "feedback_label_count": feedback_label_count,
        }

    train_df = labeled[labeled["training_label"].notna()].copy()

    negatives_pool = labeled[labeled["training_label"].isna()].copy()
    target_negative_count = max(positive_count * 3, 12)
    if not negatives_pool.empty and negative_count < target_negative_count:
        negative_limit = min(target_negative_count - negative_count, len(negatives_pool))
        pseudo_negatives = negatives_pool.sort_values(
            ["prior_probability", "risk_penalty", "signal_strength"],
            ascending=[True, False, True],
        ).head(negative_limit)
        pseudo_negatives["training_label"] = 0.0
        train_df = pd.concat([train_df, pseudo_negatives], ignore_index=True)

    negative_count = int((train_df["training_label"] == 0.0).sum())
    if negative_count == 0:
        return pd.DataFrame(), {
            "training_positive_count": positive_count,
            "training_negative_count": negative_count,
            "feedback_label_count": feedback_label_count,
        }

    return train_df, {
        "training_positive_count": positive_count,
        "training_negative_count": negative_count,
        "feedback_label_count": feedback_label_count,
    }


def _apply_feedback_overrides(scored: pd.DataFrame) -> pd.DataFrame:
    """Make explicit user feedback immediately affect ranking output."""
    if "feedback_label" not in scored.columns:
        return scored

    overridden = scored.copy()
    label_series = overridden["feedback_label"].fillna("")

    interesting_mask = label_series.eq("interesting")
    bought_mask = label_series.eq("bought")
    skipped_mask = label_series.eq("skipped")

    overridden.loc[interesting_mask, "interest_probability"] = np.maximum(
        overridden.loc[interesting_mask, "interest_probability"],
        0.72,
    )
    overridden.loc[bought_mask, "interest_probability"] = np.maximum(
        overridden.loc[bought_mask, "interest_probability"],
        0.95,
    )
    overridden.loc[skipped_mask, "interest_probability"] = np.minimum(
        overridden.loc[skipped_mask, "interest_probability"],
        0.08,
    )
    return overridden


def score_interest_candidates(
    active_df: pd.DataFrame,
    deals_df: pd.DataFrame,
    portfolio_df: pd.DataFrame | None = None,
    feedback_df: pd.DataFrame | None = None,
    min_positive_labels: int = 3,
) -> pd.DataFrame:
    """Score listings and optionally fine-tune from portfolio examples."""
    feature_df = build_interest_features(active_df, deals_df)
    if feature_df.empty:
        return pd.DataFrame()

    scored, source_stats = _attach_training_sources(feature_df, portfolio_df, feedback_df)
    train_df, training_stats = _build_training_frame(scored, min_positive_labels=min_positive_labels)

    if not train_df.empty:
        weights = _fit_logistic_weights(
            train_df,
            INTEREST_FEATURE_COLUMNS,
            initial_weights=DEFAULT_INTEREST_WEIGHTS,
        )
        has_feedback = source_stats["feedback_positive_count"] + source_stats["feedback_negative_count"] > 0
        has_portfolio = source_stats["portfolio_positive_count"] > 0
        if has_feedback and has_portfolio:
            model_source = "feedback+portfolio-trained"
        elif has_feedback:
            model_source = "feedback-trained"
        else:
            model_source = "portfolio-trained"
    else:
        weights = dict(DEFAULT_INTEREST_WEIGHTS)
        model_source = "prior"

    logits = np.full(len(scored), weights["bias"], dtype=float)
    for feature in INTEREST_FEATURE_COLUMNS:
        logits += scored[feature].fillna(0).to_numpy(dtype=float) * weights[feature]

    scored["interest_probability"] = _sigmoid(logits)
    scored = _apply_feedback_overrides(scored)
    scored["model_source"] = model_source
    for key, value in source_stats.items():
        scored[key] = value
    for key, value in training_stats.items():
        scored[key] = value

    scored["interest_class"] = np.select(
        [
            (scored["interest_probability"] >= 0.80)
            & (scored["risk_penalty"] < 0.45)
            & (scored["signal_strength"] > 0.15),
            scored["interest_probability"] >= 0.62,
            scored["interest_probability"] >= 0.45,
        ],
        ["Hot now", "Review", "Watchlist"],
        default="Low priority",
    )
    scored.loc[scored["feedback_label"].eq("bought"), "interest_class"] = "Bought"
    scored.loc[scored["feedback_label"].eq("skipped"), "interest_class"] = "Skipped"
    scored.loc[
        scored["feedback_label"].eq("interesting") & scored["interest_class"].eq("Low priority"),
        "interest_class",
    ] = "Review"
    scored["interest_reason"] = scored.apply(build_interest_reason, axis=1)
    reason_prefix = np.select(
        [
            scored["feedback_label"].eq("interesting"),
            scored["feedback_label"].eq("bought"),
            scored["feedback_label"].eq("skipped"),
        ],
        [
            "you marked it as interesting",
            "already bought",
            "you skipped this listing",
        ],
        default="",
    )
    scored["interest_reason"] = np.where(
        reason_prefix != "",
        np.where(
            scored["interest_reason"].ne(""),
            reason_prefix + "; " + scored["interest_reason"],
            reason_prefix,
        ),
        scored["interest_reason"],
    )
    scored = scored.sort_values(
        by=["interest_probability", "est_profit_eur", "sample_size", "price_eur"],
        ascending=[False, False, False, True],
        na_position="last",
    )
    return scored
