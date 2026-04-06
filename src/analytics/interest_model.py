"""Ranking model for shortlist candidates.

Uses market sale velocity (how fast similar cars sell) as the primary
automatic signal instead of manual user feedback.  Portfolio entries
still serve as explicit positive examples when available.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


INTEREST_FEATURE_COLUMNS = [
    "signal_strength",
    "profit_score",
    "roi_score",
    "confidence_score",
    "liquidity_score",
    "seller_pressure_score",
    "staleness_score",
    "sale_velocity_score",
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
    "sale_velocity_score": 1.2,
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
    velocity = row.get("sale_velocity_score")

    if pd.notna(undervaluation) and undervaluation >= 12:
        reasons.append(f"below model by {undervaluation:.0f}%")
    if pd.notna(est_profit) and est_profit >= 1000:
        reasons.append(f"profit ~{int(est_profit):,} EUR")
    if pd.notna(sample_size) and sample_size >= 8:
        reasons.append(f"{int(sample_size)} comparable listings")
    if pd.notna(avg_days) and avg_days <= 30:
        reasons.append("liquid market")
    if pd.notna(velocity) and velocity >= 0.4:
        reasons.append("segment sells fast")
    if pd.notna(drop_per_day) and drop_per_day < -20:
        reasons.append("seller is reducing price")

    return ", ".join(reasons[:3]) if reasons else "market anomaly, manual review needed"


def _compute_segment_velocity(inactive_df: pd.DataFrame | None) -> dict[tuple[str, str], float]:
    """Compute fraction of quickly-sold listings per brand+model segment.

    Returns a dict mapping (brand, model) → float in [0, 1], where 1 means
    all recently deactivated listings in that segment sold within 21 days.
    """
    if inactive_df is None or inactive_df.empty:
        return {}

    needed = {"deactivated_at", "first_seen_at", "brand", "model"}
    if not needed.issubset(inactive_df.columns):
        return {}

    sold = inactive_df[
        inactive_df["deactivated_at"].notna() & inactive_df["first_seen_at"].notna()
    ].copy()
    if sold.empty:
        return {}

    sold["_lifespan"] = (
        pd.to_datetime(sold["deactivated_at"]) - pd.to_datetime(sold["first_seen_at"])
    ).dt.days
    sold = sold[sold["_lifespan"] > 0]
    if sold.empty:
        return {}

    velocity: dict[tuple[str, str], float] = {}
    for (brand, model), group in sold.groupby(["brand", "model"]):
        if len(group) < 3:
            continue
        fast_fraction = float((group["_lifespan"] <= 21).mean())
        velocity[(brand, model)] = fast_fraction

    return velocity


def build_interest_features(
    active_df: pd.DataFrame,
    deals_df: pd.DataFrame,
    inactive_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
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

    # Sale velocity: fraction of similar deactivated listings that sold within 21 days
    segment_velocity = _compute_segment_velocity(inactive_df)
    if segment_velocity:
        merged["sale_velocity_score"] = merged.apply(
            lambda r: segment_velocity.get((r.get("brand"), r.get("model")), 0.0),
            axis=1,
        )
    else:
        merged["sale_velocity_score"] = 0.0

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


def _attach_training_sources(
    feature_df: pd.DataFrame,
    portfolio_df: pd.DataFrame | None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Attach portfolio labels to scored candidates."""
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

    stats = {
        "portfolio_positive_count": int(labeled["portfolio_positive"].sum()),
    }
    return labeled, stats


def _build_training_frame(
    labeled_df: pd.DataFrame,
    min_positive_labels: int = 3,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Prepare pseudo-labeled training data from portfolio."""
    if labeled_df.empty:
        return pd.DataFrame(), {
            "training_positive_count": 0,
            "training_negative_count": 0,
        }

    labeled = labeled_df.copy()
    labeled["training_label"] = np.nan
    labeled.loc[labeled["portfolio_positive"], "training_label"] = 1.0

    positive_count = int((labeled["training_label"] == 1.0).sum())

    if positive_count < min_positive_labels:
        return pd.DataFrame(), {
            "training_positive_count": positive_count,
            "training_negative_count": 0,
        }

    train_df = labeled[labeled["training_label"].notna()].copy()

    negatives_pool = labeled[labeled["training_label"].isna()].copy()
    target_negative_count = max(positive_count * 3, 12)
    negative_count = 0
    if not negatives_pool.empty:
        negative_limit = min(target_negative_count, len(negatives_pool))
        pseudo_negatives = negatives_pool.sort_values(
            ["prior_probability", "risk_penalty", "signal_strength"],
            ascending=[True, False, True],
        ).head(negative_limit)
        pseudo_negatives["training_label"] = 0.0
        train_df = pd.concat([train_df, pseudo_negatives], ignore_index=True)
        negative_count = negative_limit

    if negative_count == 0:
        return pd.DataFrame(), {
            "training_positive_count": positive_count,
            "training_negative_count": 0,
        }

    return train_df, {
        "training_positive_count": positive_count,
        "training_negative_count": negative_count,
    }


def score_interest_candidates(
    active_df: pd.DataFrame,
    deals_df: pd.DataFrame,
    inactive_df: pd.DataFrame | None = None,
    portfolio_df: pd.DataFrame | None = None,
    min_positive_labels: int = 3,
) -> pd.DataFrame:
    """Score listings using market signals and optionally fine-tune from portfolio."""
    feature_df = build_interest_features(active_df, deals_df, inactive_df)
    if feature_df.empty:
        return pd.DataFrame()

    scored, source_stats = _attach_training_sources(feature_df, portfolio_df)
    train_df, training_stats = _build_training_frame(scored, min_positive_labels=min_positive_labels)

    if not train_df.empty:
        weights = _fit_logistic_weights(
            train_df,
            INTEREST_FEATURE_COLUMNS,
            initial_weights=DEFAULT_INTEREST_WEIGHTS,
        )
        model_source = "portfolio-trained"
    else:
        weights = dict(DEFAULT_INTEREST_WEIGHTS)
        model_source = "sale-velocity"

    logits = np.full(len(scored), weights["bias"], dtype=float)
    for feature in INTEREST_FEATURE_COLUMNS:
        logits += scored[feature].fillna(0).to_numpy(dtype=float) * weights[feature]

    scored["interest_probability"] = _sigmoid(logits)
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
    scored["interest_reason"] = scored.apply(build_interest_reason, axis=1)
    scored = scored.sort_values(
        by=["interest_probability", "est_profit_eur", "sample_size", "price_eur"],
        ascending=[False, False, False, True],
        na_position="last",
    )
    return scored
