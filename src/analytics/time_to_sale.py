"""Time-to-sale survival model and price optimization."""

from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Feature columns used by the model
TOS_FEATURE_COLUMNS = [
    "price_ratio",
    "mileage_norm",
    "year_norm",
    "is_diesel",
    "is_professional",
    "price_drop_rate",
    "has_accident",
    "has_repair",
]

DEFAULT_TOS_WEIGHTS = {
    "bias": 0.5,
    "price_ratio": -2.5,       # higher price ratio → longer to sell
    "mileage_norm": 0.3,       # higher mileage → slightly longer
    "year_norm": -0.8,         # newer → sells faster
    "is_diesel": -0.2,         # diesel slightly faster in PT
    "is_professional": -0.3,   # dealers price more aggressively
    "price_drop_rate": -0.5,   # already dropping → sells sooner
    "has_accident": 0.8,       # accident history → slower
    "has_repair": 0.4,         # repair needed → slower
}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def build_tos_dataset(listings_df: pd.DataFrame) -> pd.DataFrame:
    """Build training dataset from listings with known outcomes.

    Returns DataFrame with features, days_to_sale, and censored flag.
    Censored=True means listing is still active (we don't know final sale time).
    """
    if listings_df.empty:
        return pd.DataFrame()

    df = listings_df.copy()

    # Compute days on market
    first = pd.to_datetime(df["first_seen_at"]).dt.tz_localize(None)

    if "deactivated_at" in df.columns:
        deact = pd.to_datetime(df["deactivated_at"]).dt.tz_localize(None)
    else:
        deact = pd.Series(pd.NaT, index=df.index)

    last = pd.to_datetime(df["last_seen_at"]).dt.tz_localize(None)
    now = pd.Timestamp.now()

    # For inactive listings with deactivated_at, use that; else last_seen_at
    end_time = deact.fillna(last)
    # For active listings, use now (censored observation)
    is_active = df["is_active"].fillna(True)
    end_time = end_time.where(~is_active, now)

    df["days_to_sale"] = (end_time - first).dt.total_seconds() / 86400
    df["days_to_sale"] = df["days_to_sale"].clip(lower=0.5)  # min half a day
    df["censored"] = is_active

    # Filter: need price and brand/model for market median
    df = df[df["price_eur"].notna() & df["brand"].notna()].copy()
    if df.empty:
        return pd.DataFrame()

    # Compute market median per brand+model for price_ratio
    medians = (
        df[df["price_eur"].notna()]
        .groupby(["brand", "model"])["price_eur"]
        .median()
        .rename("market_median")
    )
    df = df.join(medians, on=["brand", "model"])
    df["price_ratio"] = (df["price_eur"] / df["market_median"].replace(0, np.nan)).fillna(1.0)

    # Normalize features
    df["mileage_norm"] = (df.get("mileage_km", pd.Series(0, index=df.index))
                          .fillna(150000).clip(0, 500000) / 500000)
    df["year_norm"] = ((df.get("year", pd.Series(2015, index=df.index))
                        .fillna(2015) - 2000) / 25).clip(0, 1)
    df["is_diesel"] = (df.get("fuel_type", pd.Series("", index=df.index))
                       .fillna("").str.lower().str.contains("diesel")).astype(float)
    df["is_professional"] = (df.get("seller_type", pd.Series("", index=df.index))
                             .fillna("").str.lower().str.contains("profissional")).astype(float)

    # Price drop rate: (first_price - current_price) / days_listed
    if "first_price_eur" in df.columns:
        price_drop = (df["first_price_eur"].fillna(df["price_eur"]) - df["price_eur"])
        days_listed = df["days_to_sale"].clip(lower=1)
        df["price_drop_rate"] = (price_drop / days_listed).clip(0, 200) / 200
    else:
        df["price_drop_rate"] = 0.0

    df["has_accident"] = (df.get("desc_mentions_accident", pd.Series(False, index=df.index))
                          .fillna(False).astype(float))
    df["has_repair"] = (df.get("desc_mentions_repair", pd.Series(False, index=df.index))
                        .fillna(False).astype(float))

    # Keep only rows with valid days_to_sale
    df = df[df["days_to_sale"].notna() & (df["days_to_sale"] > 0)].copy()

    return df


def fit_tos_model(
    df: pd.DataFrame,
    target_days: int = 30,
    epochs: int = 300,
    learning_rate: float = 0.3,
    l2_penalty: float = 0.01,
) -> dict[str, float]:
    """Train logistic model: P(sold within target_days | features).

    Uses only non-censored (sold) observations for positive labels,
    and censored observations still active after target_days as negatives.
    """
    if df.empty:
        log.warning("No data to train time-to-sale model")
        return dict(DEFAULT_TOS_WEIGHTS)

    # Label: 1 = sold within target_days, 0 = not sold within target_days
    sold = df[~df["censored"]].copy()
    sold["label"] = (sold["days_to_sale"] <= target_days).astype(float)

    # Also use censored (active) listings that have been listed > target_days as negative
    still_active = df[df["censored"] & (df["days_to_sale"] > target_days)].copy()
    still_active["label"] = 0.0

    train = pd.concat([sold, still_active], ignore_index=True)

    if len(train) < 10:
        log.warning("Too few training examples (%d), using default weights", len(train))
        return dict(DEFAULT_TOS_WEIGHTS)

    positive = train["label"].sum()
    negative = len(train) - positive
    if positive < 3 or negative < 3:
        log.warning("Insufficient class balance (pos=%d neg=%d), using defaults",
                    int(positive), int(negative))
        return dict(DEFAULT_TOS_WEIGHTS)

    log.info("Training TOS model: %d samples (%d sold within %dd, %d not)",
             len(train), int(positive), target_days, int(negative))

    # Gradient descent
    weights = np.array([DEFAULT_TOS_WEIGHTS.get(col, 0.0)
                        for col in TOS_FEATURE_COLUMNS], dtype=float)
    bias = float(DEFAULT_TOS_WEIGHTS.get("bias", 0.0))

    X = train[TOS_FEATURE_COLUMNS].fillna(0).to_numpy(dtype=float)
    y = train["label"].to_numpy(dtype=float)

    for _ in range(epochs):
        logits = X @ weights + bias
        preds = _sigmoid(logits)
        error = preds - y
        grad_w = (X.T @ error) / len(train) + l2_penalty * weights
        grad_b = error.mean()
        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b

    fitted = {"bias": float(bias)}
    fitted.update({col: float(w) for col, w in zip(TOS_FEATURE_COLUMNS, weights)})

    log.info("TOS weights: %s", {k: round(v, 3) for k, v in fitted.items()})
    return fitted


def predict_sale_probability(
    df: pd.DataFrame,
    weights: dict[str, float] | None = None,
) -> pd.Series:
    """Predict P(sold within target_days) for each row."""
    w = weights or DEFAULT_TOS_WEIGHTS
    if df.empty:
        return pd.Series(dtype=float)

    logits = np.full(len(df), w.get("bias", 0.0), dtype=float)
    for col in TOS_FEATURE_COLUMNS:
        if col in df.columns:
            logits += df[col].fillna(0).to_numpy(dtype=float) * w.get(col, 0.0)

    return pd.Series(_sigmoid(logits), index=df.index, name="sale_probability")


def optimize_price(
    listing_features: dict,
    market_median: float,
    weights: dict[str, float] | None = None,
    min_price_ratio: float = 0.70,
    max_price_ratio: float = 1.15,
    target_days: int = 30,
    min_acceptable_price: float | None = None,
    steps: int = 50,
) -> dict:
    """Find optimal price that maximizes P(sale) * margin.

    Returns dict with optimal_price, sale_probability, and price_curve.
    """
    w = weights or DEFAULT_TOS_WEIGHTS

    # Build feature row template
    base_row = pd.DataFrame([listing_features])
    for col in TOS_FEATURE_COLUMNS:
        if col not in base_row.columns:
            base_row[col] = 0.0

    ratios = np.linspace(min_price_ratio, max_price_ratio, steps)
    prices = ratios * market_median
    probabilities = []
    scores = []

    floor = min_acceptable_price or (market_median * min_price_ratio)

    for ratio, price in zip(ratios, prices):
        row = base_row.copy()
        row["price_ratio"] = ratio
        prob = float(predict_sale_probability(row, w).iloc[0])
        margin = max(price - floor, 0)
        score = prob * margin
        probabilities.append(prob)
        scores.append(score)

    best_idx = int(np.argmax(scores))

    return {
        "optimal_price": round(float(prices[best_idx])),
        "optimal_ratio": round(float(ratios[best_idx]), 3),
        "sale_probability": round(float(probabilities[best_idx]), 3),
        "expected_value": round(float(scores[best_idx])),
        "target_days": target_days,
        "market_median": round(market_median),
        "price_curve": [
            {
                "price": round(float(p)),
                "ratio": round(float(r), 3),
                "probability": round(float(prob), 3),
                "expected_value": round(float(s)),
            }
            for p, r, prob, s in zip(prices, ratios, probabilities, scores)
        ],
    }


def compute_tos_stats(listings_df: pd.DataFrame) -> pd.DataFrame:
    """Per brand+model stats: median days to sale, sale rate, count."""
    if listings_df.empty:
        return pd.DataFrame()

    sold = listings_df[
        (listings_df["is_active"] == False) &
        listings_df["first_seen_at"].notna()
    ].copy()

    if sold.empty:
        return pd.DataFrame()

    first = pd.to_datetime(sold["first_seen_at"]).dt.tz_localize(None)
    if "deactivated_at" in sold.columns:
        end = pd.to_datetime(sold["deactivated_at"]).dt.tz_localize(None).fillna(
            pd.to_datetime(sold["last_seen_at"]).dt.tz_localize(None)
        )
    else:
        end = pd.to_datetime(sold["last_seen_at"]).dt.tz_localize(None)

    sold["days_to_sale"] = (end - first).dt.total_seconds() / 86400

    # Per brand+model
    stats = (
        sold.groupby(["brand", "model"])
        .agg(
            median_days=("days_to_sale", "median"),
            mean_days=("days_to_sale", "mean"),
            sold_count=("days_to_sale", "size"),
            pct_under_30d=("days_to_sale", lambda x: (x <= 30).mean() * 100),
        )
        .round(1)
        .reset_index()
    )

    # Total listings per brand+model
    totals = (
        listings_df.groupby(["brand", "model"])
        .size()
        .reset_index(name="total_count")
    )
    stats = stats.merge(totals, on=["brand", "model"], how="left")
    stats["sale_rate_pct"] = (stats["sold_count"] / stats["total_count"] * 100).round(1)

    return stats.sort_values("sold_count", ascending=False)
