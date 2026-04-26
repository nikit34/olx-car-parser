"""Quality diagnostics for the price model.

Reads the OOF predictions shipped with the bundle (no retraining required) and
slices error metrics by price bucket, year, and brand so you can see *where*
the model is wrong, not just the headline MAPE.

Static diagnostics (cheap, run from the bundle alone):
- ``evaluate_oof``      — global + bucket MAE/MAPE/coverage tables
- ``worst_residuals``   — top-N listings the model misses by largest %
- ``reliability_curve`` — empirical 80% band coverage by predicted-price decile

Time-aware diagnostic (slow, retrains per fold):
- ``time_backtest``     — rolling-window train/test using ``first_seen_at``;
                          honest measure of how the model performs on
                          tomorrow's listings versus today's CV (which mixes
                          time-adjacent rows across folds).

Backtest results are persisted to ``data/price_backtest.json`` so the dashboard
can render them without re-running the 30-minute training loop.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from src.analytics.price_model import (
    _ALL_FEATURES,
    _LGB_PARAMS,
    _MIN_N_ESTIMATORS,
    _QUANTILES,
    CATEGORICAL_FEATURES,
    _filter_training_data,
    _prepare_X,
)


_BACKTEST_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "price_backtest.json"


# Buckets are open on the right (low ≤ x < high) so each row lands in exactly
# one. Labels are picked to read naturally in dashboard tables; ordering is
# preserved by using a Categorical with these labels at table time.
_PRICE_BUCKETS: list[tuple[float, float, str]] = [
    (0, 3000, "<€3k"),
    (3000, 7000, "€3–7k"),
    (7000, 15000, "€7–15k"),
    (15000, 30000, "€15–30k"),
    (30000, float("inf"), "€30k+"),
]
_PRICE_BUCKET_ORDER = [b[2] for b in _PRICE_BUCKETS]

_YEAR_BUCKETS: list[tuple[float, float, str]] = [
    (0, 2010, "≤2009"),
    (2010, 2015, "2010–2014"),
    (2015, 2019, "2015–2018"),
    (2019, 2022, "2019–2021"),
    (2022, 9999, "2022+"),
]
_YEAR_BUCKET_ORDER = [b[2] for b in _YEAR_BUCKETS]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _join_oof(
    listings_df: pd.DataFrame,
    oof_preds: Mapping[str, tuple[float, float, float]],
) -> pd.DataFrame:
    """Inner-join listings with bundled OOF predictions on olx_id.

    Drops rows missing olx_id, price, or an OOF entry. Adds derived columns
    used by every downstream diagnostic so each function doesn't recompute
    them: ``oof_low / oof_median / oof_high / residual / abs_residual_pct /
    in_band``.
    """
    if "olx_id" not in listings_df.columns or "price_eur" not in listings_df.columns:
        return listings_df.iloc[0:0].copy()
    if not oof_preds:
        return listings_df.iloc[0:0].copy()

    df = listings_df[
        listings_df["price_eur"].notna() & listings_df["olx_id"].notna()
    ].copy()
    if df.empty:
        return df

    keys = df["olx_id"].astype(str)
    bands = keys.map(lambda k: oof_preds.get(k))
    mask = bands.notna()
    df = df.loc[mask].copy()
    bands = bands.loc[mask]

    df["oof_low"] = bands.map(lambda b: float(b[0])).values
    df["oof_median"] = bands.map(lambda b: float(b[1])).values
    df["oof_high"] = bands.map(lambda b: float(b[2])).values
    price = df["price_eur"].astype(float)
    df["residual"] = price - df["oof_median"]
    df["abs_residual_pct"] = (df["residual"].abs() / price.clip(lower=1)) * 100
    df["in_band"] = (price >= df["oof_low"]) & (price <= df["oof_high"])
    return df


def _bucketize(values: pd.Series, buckets: list[tuple[float, float, str]]) -> pd.Series:
    """Map numeric values to bucket labels per (low, high, label) ranges."""
    result = pd.Series([None] * len(values), index=values.index, dtype=object)
    for low, high, label in buckets:
        mask = (values >= low) & (values < high)
        result.loc[mask] = label
    return result


def _bucket_stats(df: pd.DataFrame, group_col: str, bucket_order: list[str] | None = None) -> pd.DataFrame:
    """Per-group n, mae, mape, bias_pct, coverage_80.

    bias_pct is signed (positive = model under-predicts, since residual =
    actual − predicted). Useful for spotting systematic over/underpricing in a
    segment, which a symmetric MAPE column hides.
    """
    cols = ["bucket", "n", "mae", "mape", "bias_pct", "coverage_80"]
    if df.empty or group_col not in df.columns:
        return pd.DataFrame(columns=cols)

    rows = []
    for bucket, group in df.groupby(group_col, sort=False, dropna=True):
        if group.empty or bucket is None:
            continue
        price = group["price_eur"].astype(float)
        rows.append({
            "bucket": bucket,
            "n": int(len(group)),
            "mae": round(float(group["residual"].abs().mean()), 0),
            "mape": round(float(group["abs_residual_pct"].mean()), 1),
            "bias_pct": round(float((group["residual"] / price.clip(lower=1) * 100).mean()), 2),
            "coverage_80": round(float(group["in_band"].mean()), 3),
        })

    out = pd.DataFrame(rows, columns=cols)
    if bucket_order:
        # Preserve the natural ordering (cheap → expensive, old → new) instead of
        # whatever pandas groupby happens to emit.
        out["bucket"] = pd.Categorical(out["bucket"], categories=bucket_order, ordered=True)
        out = out.sort_values("bucket").reset_index(drop=True)
        out["bucket"] = out["bucket"].astype(str)
    return out


# ---------------------------------------------------------------------------
# Static diagnostics
# ---------------------------------------------------------------------------

def evaluate_oof(
    listings_df: pd.DataFrame,
    oof_preds: Mapping[str, tuple[float, float, float]],
    top_n_brands: int = 10,
) -> dict:
    """Full OOF diagnostic report.

    Returns:
        {
          "global": {
              "n", "mae", "mape", "r2", "coverage_80",
              "bias_pct", "n_inverted_band",
          },
          "by_price":  DataFrame[bucket, n, mae, mape, bias_pct, coverage_80],
          "by_year":   DataFrame[same],
          "by_brand":  DataFrame[same],
        }
    """
    df = _join_oof(listings_df, oof_preds)
    empty = pd.DataFrame(columns=["bucket", "n", "mae", "mape", "bias_pct", "coverage_80"])
    if df.empty:
        return {
            "global": {"n": 0},
            "by_price": empty,
            "by_year": empty,
            "by_brand": empty,
        }

    actual = df["price_eur"].astype(float).values
    median = df["oof_median"].values

    ss_res = float(np.sum((actual - median) ** 2))
    ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Should be 0 — predict_prices sorts every row before returning, and
    # train_price_model writes already-sorted bands into oof_preds. A non-zero
    # count means the bundle is corrupt or this code is reading a pre-fix
    # artifact, both of which warrant a loud signal in the dashboard.
    n_inverted = int(
        ((df["oof_low"] > df["oof_median"]) | (df["oof_median"] > df["oof_high"])).sum()
    )

    g = {
        "n": int(len(df)),
        "mae": round(float(np.mean(np.abs(actual - median))), 0),
        "mape": round(
            float(np.mean(np.abs((actual - median) / np.maximum(actual, 1)) * 100)), 1
        ),
        "r2": round(r2, 3),
        "coverage_80": round(float(df["in_band"].mean()), 3),
        "bias_pct": round(
            float(np.mean((actual - median) / np.maximum(actual, 1) * 100)), 2
        ),
        "n_inverted_band": n_inverted,
    }

    df = df.copy()
    df["_price_bucket"] = _bucketize(
        df["price_eur"].astype(float), _PRICE_BUCKETS,
    )
    if "year" in df.columns:
        df["_year_bucket"] = _bucketize(
            pd.to_numeric(df["year"], errors="coerce").fillna(-1),
            _YEAR_BUCKETS,
        )
    else:
        df["_year_bucket"] = None

    if "brand" in df.columns:
        top_brands = df["brand"].value_counts().head(top_n_brands).index
        df["_brand_bucket"] = df["brand"].where(df["brand"].isin(top_brands))
        brand_stats = _bucket_stats(df, "_brand_bucket")
        # Sort brand table by sample count desc (most common first)
        if not brand_stats.empty:
            brand_stats = brand_stats.sort_values("n", ascending=False).reset_index(drop=True)
    else:
        brand_stats = empty

    return {
        "global": g,
        "by_price": _bucket_stats(df, "_price_bucket", _PRICE_BUCKET_ORDER),
        "by_year": _bucket_stats(df, "_year_bucket", _YEAR_BUCKET_ORDER),
        "by_brand": brand_stats,
    }


def worst_residuals(
    listings_df: pd.DataFrame,
    oof_preds: Mapping[str, tuple[float, float, float]],
    n: int = 20,
) -> pd.DataFrame:
    """Top-N listings by absolute residual percentage.

    Useful for hand-inspection — the rows the model misses by the largest %
    are usually where some signal is missing (rare trim, damage flag the LLM
    didn't catch, currency mistake, etc.) rather than noise.
    """
    df = _join_oof(listings_df, oof_preds)
    if df.empty:
        return pd.DataFrame()

    cols = [
        c for c in (
            "olx_id", "brand", "model", "year", "mileage_km", "price_eur",
            "oof_low", "oof_median", "oof_high",
            "residual", "abs_residual_pct", "in_band",
        )
        if c in df.columns
    ]
    return df.nlargest(n, "abs_residual_pct")[cols].reset_index(drop=True)


def reliability_curve(
    listings_df: pd.DataFrame,
    oof_preds: Mapping[str, tuple[float, float, float]],
    n_bins: int = 10,
) -> pd.DataFrame:
    """Empirical 80% band coverage by predicted-price decile.

    A well-calibrated band has empirical_coverage ≈ 0.80 in every bin. Bins
    that fall below (under-confident) or above (over-confident) the line on
    the dashboard's reliability diagram show where CQR's symmetric widening
    isn't quite right — typically the cheapest decile, where pinball loss is
    dominated by absolute € error and the band ends up too wide.
    """
    df = _join_oof(listings_df, oof_preds)
    if df.empty:
        return pd.DataFrame()

    df = df.sort_values("oof_median").reset_index(drop=True)
    try:
        df["_bin"] = pd.qcut(df["oof_median"], q=n_bins, duplicates="drop")
    except ValueError:
        # Too few unique predicted prices (degenerate fixture or tiny dataset)
        return pd.DataFrame()

    rows = []
    for _, group in df.groupby("_bin", observed=True):
        if group.empty:
            continue
        rows.append({
            "predicted_min": round(float(group["oof_median"].min()), 0),
            "predicted_max": round(float(group["oof_median"].max()), 0),
            "predicted_mean": round(float(group["oof_median"].mean()), 0),
            "actual_mean": round(float(group["price_eur"].astype(float).mean()), 0),
            "n": int(len(group)),
            "empirical_coverage": round(float(group["in_band"].mean()), 3),
            "nominal_coverage": 0.80,
            "calibration_gap": round(float(group["in_band"].mean() - 0.80), 3),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Time backtest (slow — retrains per fold)
# ---------------------------------------------------------------------------

def time_backtest(
    listings_df: pd.DataFrame,
    *,
    n_splits: int = 5,
    n_estimators_per_q: dict[str, int] | None = None,
    conformal_q: float = 0.0,
) -> pd.DataFrame:
    """Rolling-window time-aware backtest using ``first_seen_at``.

    Each fold trains on listings seen before t_split and tests on the next
    1/n_splits chunk of the date range. Reports MAE/MAPE/coverage on the test
    slice. Honest measure of how the model performs on *tomorrow's* listings
    versus today's CV (which mixes time-adjacent rows across folds and so
    over-states quality on a non-stationary market).

    ``conformal_q`` (in log space) is the band-widening exponent applied to
    each fold's predictions — should be the production bundle's active q so
    the reported coverage reflects what the deployed model delivers, not the
    raw uncalibrated bands. Default 0.0 means raw bands (kept for backward
    compatibility and as a "how miscalibrated is the bare model" baseline).

    The fit uses fixed ``n_estimators_per_q`` (defaulting to the bundle's
    early-stop counts) instead of nested CV per fold — that would multiply the
    cost by 5×.
    """
    import lightgbm as lgb

    if "first_seen_at" not in listings_df.columns:
        raise ValueError("time_backtest requires 'first_seen_at' on listings")

    df = listings_df[
        listings_df["price_eur"].notna()
        & listings_df["year"].notna()
        & listings_df["mileage_km"].notna()
        & listings_df["first_seen_at"].notna()
    ].copy()
    df["first_seen_at"] = pd.to_datetime(df["first_seen_at"])
    df = df.sort_values("first_seen_at").reset_index(drop=True)
    df, _ = _filter_training_data(df)

    if len(df) < 200:
        return pd.DataFrame()

    n_estimators_per_q = n_estimators_per_q or {
        name: max(_LGB_PARAMS["n_estimators"] // 4, _MIN_N_ESTIMATORS)
        for name in _QUANTILES
    }
    cat_indices = [_ALL_FEATURES.index(c) for c in CATEGORICAL_FEATURES]

    fold_size = len(df) // n_splits
    rows: list[dict] = []
    for fold in range(1, n_splits):
        split_at = fold * fold_size
        train = df.iloc[:split_at]
        test = df.iloc[split_at: split_at + fold_size]
        if len(train) < 100 or len(test) < 50:
            continue

        X_train, cat_maps = _prepare_X(train)
        y_train_log = np.log1p(
            np.maximum(train["price_eur"].astype(float).values, 0)
        )
        X_test, _ = _prepare_X(test, cat_maps)
        y_test = test["price_eur"].astype(float).values

        log_preds: dict[str, np.ndarray] = {}
        for name, alpha in _QUANTILES.items():
            params = {**_LGB_PARAMS, "n_estimators": n_estimators_per_q[name]}
            model = lgb.LGBMRegressor(objective="quantile", alpha=alpha, **params)
            model.fit(X_train, y_train_log, categorical_feature=cat_indices)
            log_preds[name] = model.predict(X_test)

        # Apply CQR widening in log space (asymmetric in price space — exactly
        # what we want for heavy-tailed targets), then back-transform and
        # repair any quantile crossing.
        stacked = np.sort(
            np.stack([
                np.expm1(log_preds["low"] - conformal_q),
                np.expm1(log_preds["median"]),
                np.expm1(log_preds["high"] + conformal_q),
            ]),
            axis=0,
        )
        low = np.maximum(stacked[0], 0)
        median = np.maximum(stacked[1], 0)
        high = np.maximum(stacked[2], 0)

        mae = float(np.mean(np.abs(y_test - median)))
        mape = float(np.mean(np.abs((y_test - median) / np.maximum(y_test, 1)) * 100))
        coverage = float(np.mean((y_test >= low) & (y_test <= high)))
        bias_pct = float(np.mean((y_test - median) / np.maximum(y_test, 1) * 100))

        rows.append({
            "fold": fold,
            "train_until": train["first_seen_at"].max().isoformat(),
            "test_from": test["first_seen_at"].min().isoformat(),
            "test_to": test["first_seen_at"].max().isoformat(),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "mae": round(mae, 0),
            "mape": round(mape, 1),
            "bias_pct": round(bias_pct, 2),
            "coverage_80": round(coverage, 3),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_backtest(backtest_df: pd.DataFrame) -> None:
    """Persist time-backtest results next to the model bundle for the dashboard.

    Includes a generated_at timestamp so the dashboard can warn if the file is
    older than the current model bundle (i.e. backtest is from a previous
    training run and may be stale).
    """
    _BACKTEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "folds": backtest_df.to_dict(orient="records") if not backtest_df.empty else [],
    }
    _BACKTEST_PATH.write_text(json.dumps(payload, indent=2))


def load_backtest() -> dict | None:
    """Load persisted backtest, or None if missing/corrupt."""
    if not _BACKTEST_PATH.exists():
        return None
    try:
        return json.loads(_BACKTEST_PATH.read_text())
    except (json.JSONDecodeError, ValueError):
        return None
