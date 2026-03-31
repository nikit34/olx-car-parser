"""Ranking model for interesting listings."""

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
    """Return short, human-readable explanation for ranking."""
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

    if row.get("needs_repair"):
        reasons.append("needs repair check")
    elif row.get("had_accident"):
        reasons.append("accident history")

    return ", ".join(reasons[:3]) if reasons else "worth a manual look"


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

    risk_penalty = pd.Series(0.0, index=merged.index)
    if "needs_repair" in merged.columns:
        risk_penalty += np.where(merged["needs_repair"] == True, 0.45, 0.0)
    if "had_accident" in merged.columns:
        risk_penalty += np.where(merged["had_accident"] == True, 0.35, 0.0)
    if "customs_cleared" in merged.columns:
        risk_penalty += np.where(merged["customs_cleared"] == False, 0.25, 0.0)
    if "num_owners" in merged.columns:
        risk_penalty += np.where(merged["num_owners"].fillna(0) >= 4, 0.15, 0.0)
    merged["risk_penalty"] = risk_penalty.clip(upper=1.0)

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


def _build_training_frame(
    feature_df: pd.DataFrame,
    portfolio_df: pd.DataFrame | None,
    min_positive_labels: int = 3,
) -> tuple[pd.DataFrame, int]:
    """Prepare pseudo-labeled training data from portfolio outcomes."""
    if portfolio_df is None or portfolio_df.empty or "olx_listing_id" not in portfolio_df.columns:
        return pd.DataFrame(), 0

    positive_ids = {
        str(value)
        for value in portfolio_df["olx_listing_id"].dropna().astype(str)
        if value.strip()
    }
    if not positive_ids:
        return pd.DataFrame(), 0

    labeled = feature_df.copy()
    labeled["portfolio_positive"] = labeled["olx_id"].astype(str).isin(positive_ids)
    positive_count = int(labeled["portfolio_positive"].sum())
    if positive_count < min_positive_labels:
        return pd.DataFrame(), positive_count

    positives = labeled[labeled["portfolio_positive"]].copy()
    negatives_pool = labeled[~labeled["portfolio_positive"]].copy()
    if negatives_pool.empty:
        return pd.DataFrame(), positive_count

    negative_limit = min(max(positive_count * 3, 12), len(negatives_pool))
    negatives = negatives_pool.sort_values(
        ["prior_probability", "risk_penalty", "signal_strength"],
        ascending=[True, False, True],
    ).head(negative_limit)

    train_df = pd.concat([positives, negatives], ignore_index=True)
    train_df["training_label"] = np.where(train_df["portfolio_positive"], 1.0, 0.0)
    return train_df, positive_count


def score_interest_candidates(
    active_df: pd.DataFrame,
    deals_df: pd.DataFrame,
    portfolio_df: pd.DataFrame | None = None,
    min_positive_labels: int = 3,
) -> pd.DataFrame:
    """Score listings and optionally fine-tune from portfolio examples."""
    feature_df = build_interest_features(active_df, deals_df)
    if feature_df.empty:
        return pd.DataFrame()

    train_df, positive_count = _build_training_frame(
        feature_df, portfolio_df, min_positive_labels=min_positive_labels
    )
    if not train_df.empty:
        weights = _fit_logistic_weights(
            train_df,
            INTEREST_FEATURE_COLUMNS,
            initial_weights=DEFAULT_INTEREST_WEIGHTS,
        )
        model_source = "portfolio-trained"
    else:
        weights = dict(DEFAULT_INTEREST_WEIGHTS)
        model_source = "prior"

    logits = np.full(len(feature_df), weights["bias"], dtype=float)
    for feature in INTEREST_FEATURE_COLUMNS:
        logits += feature_df[feature].fillna(0).to_numpy(dtype=float) * weights[feature]

    scored = feature_df.copy()
    scored["interest_probability"] = _sigmoid(logits)
    scored["model_source"] = model_source
    scored["portfolio_positive_count"] = positive_count
    scored["portfolio_positive"] = False

    if portfolio_df is not None and not portfolio_df.empty and "olx_listing_id" in portfolio_df.columns:
        positive_ids = {
            str(value)
            for value in portfolio_df["olx_listing_id"].dropna().astype(str)
            if value.strip()
        }
        scored["portfolio_positive"] = scored["olx_id"].astype(str).isin(positive_ids)

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
