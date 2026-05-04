"""Per-listing hazard model — P(sold within horizon_days from listing).

Binary LightGBM classifier on top of the same active+sold corpus the
price model uses. Target: did this listing sell within horizon_days
(default 30) of first_seen_at?

Refines decision.py's segment-level liquidity signals (dom_median,
dom_fast_share) to per-listing: instead of "VW Golf Mk7 sells fast in
Porto on average", it answers "this specific Golf, with this price /
photos / damage profile, has 65 % chance of selling in 30 days."

Censoring rules — what gets dropped from training:
  - Active listings younger than horizon_days (target unobservable)
  - Listings without first_seen_at (can't compute age)
  - Listings deactivated for non-sold reasons within horizon
    (ambiguous: parser noise vs genuine removal)

Negatives are observably not-sold-within-horizon:
  - Active and age >= horizon (still listed past the deadline)
  - Deactivated past horizon (any reason — visibly survived past it)

Features are intentionally numeric-only for v1: brand/model patterns
get captured indirectly through residual_pct when predictions_df is
supplied (the price model has already learned them). Categorical
encoding adds 30 brands × dozens of models worth of high-cardinality
splits with small per-class sample sizes — bumps complexity without
proportional payoff at this corpus size.

Notably absent: num_price_drops, max_drop_pct, days_since_last_drop.
They're informative on training rows (which by construction are old
enough to have observable drops) but on a fresh listing they're
identically zero — the train/predict distribution shift would push
the model to under-predict P(sold) on new listings.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score


_MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_MODEL_PATH = _MODEL_DIR / "hazard_model.joblib"
_METRICS_PATH = _MODEL_DIR / "hazard_metrics.json"
_MODEL_MAX_AGE_HOURS = 24

_SCHEMA_VERSION = 1

DEFAULT_HORIZON_DAYS = 30
MIN_SAMPLES = 200
VAL_FRAC = 0.2

NUMERIC_FEATURES = [
    "year",
    "mileage_km",
    "engine_cc",
    "horsepower",
    "log_price",
    "photo_count",
    "description_length",
    "damage_severity",
]

PREDICTION_FEATURES = ["residual_pct", "band_pct"]


# ---------------------------------------------------------------------------
# Target labelling (with censoring)
# ---------------------------------------------------------------------------


def _build_target(
    listings_df: pd.DataFrame,
    horizon_days: int,
    *,
    now: pd.Timestamp | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (y, labeled_mask) aligned to ``listings_df.index``.

    ``y[i] = 1``  → listing i sold within horizon_days of first_seen_at
    ``y[i] = 0``  → listing i is observably not sold by horizon
    ``labeled[i] = False`` → row is censored / unlabelable; drop from training

    *now* — override "today" for deterministic tests; defaults to current UTC.
    """
    n = len(listings_df)
    y = np.zeros(n, dtype=int)
    labeled = np.zeros(n, dtype=bool)
    if n == 0 or "first_seen_at" not in listings_df.columns:
        return y, labeled

    first_seen = pd.to_datetime(
        listings_df["first_seen_at"], errors="coerce", utc=True,
    )
    deactivated = pd.to_datetime(
        listings_df.get("deactivated_at"), errors="coerce", utc=True,
    )
    is_active = (
        listings_df["is_active"].astype(bool)
        if "is_active" in listings_df.columns
        else pd.Series(True, index=listings_df.index)
    )
    reason = (
        listings_df["deactivation_reason"].astype(str)
        if "deactivation_reason" in listings_df.columns
        else pd.Series("", index=listings_df.index)
    )

    if now is None:
        now = pd.Timestamp.now(tz="UTC")
    days_active = (now - first_seen).dt.total_seconds() / 86400
    days_to_deact = (deactivated - first_seen).dt.total_seconds() / 86400

    pos = (
        (~is_active)
        & (reason == "sold")
        & first_seen.notna()
        & deactivated.notna()
        & (days_to_deact >= 0)
        & (days_to_deact <= horizon_days)
    )
    neg_active = is_active & first_seen.notna() & (days_active >= horizon_days)
    neg_deact_past = (
        (~is_active)
        & first_seen.notna()
        & deactivated.notna()
        & (days_to_deact > horizon_days)
    )
    # Censored / dropped:
    #   - active and age < horizon (don't know yet)
    #   - first_seen_at missing
    #   - deactivated within horizon for non-sold reason (ambiguous)

    y[pos.values] = 1
    y[(neg_active | neg_deact_past).values] = 0
    labeled[(pos | neg_active | neg_deact_past).values] = True
    return y, labeled


# ---------------------------------------------------------------------------
# Feature build
# ---------------------------------------------------------------------------


def _build_features(
    listings_df: pd.DataFrame,
    predictions_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Build the numeric feature matrix (no categoricals — see module
    docstring for why).

    *predictions_df* must be aligned to ``listings_df.index`` — pass
    the direct output of ``price_model.predict_prices``.
    """
    df = listings_df
    out = pd.DataFrame(index=df.index)

    out["year"] = pd.to_numeric(df.get("year"), errors="coerce")
    out["mileage_km"] = pd.to_numeric(df.get("mileage_km"), errors="coerce")
    out["engine_cc"] = pd.to_numeric(df.get("engine_cc"), errors="coerce")
    out["horsepower"] = pd.to_numeric(df.get("horsepower"), errors="coerce")
    out["log_price"] = np.log1p(
        pd.to_numeric(df.get("price_eur"), errors="coerce")
        .fillna(0).clip(lower=0)
    )
    out["photo_count"] = pd.to_numeric(df.get("photo_count"), errors="coerce")
    out["description_length"] = pd.to_numeric(
        df.get("description_length"), errors="coerce",
    )
    out["damage_severity"] = pd.to_numeric(
        df.get("damage_severity"), errors="coerce",
    )

    feature_names = list(NUMERIC_FEATURES)

    if predictions_df is not None and not predictions_df.empty:
        pred = predictions_df.reindex(df.index)
        price = pd.to_numeric(df.get("price_eur"), errors="coerce")
        pred_price = pd.to_numeric(pred.get("predicted_price"), errors="coerce")
        fair_low = pd.to_numeric(pred.get("fair_price_low"), errors="coerce")
        fair_high = pd.to_numeric(pred.get("fair_price_high"), errors="coerce")
        safe_pred = pred_price.where(pred_price > 0)
        out["residual_pct"] = (price - safe_pred) / safe_pred * 100
        out["band_pct"] = (fair_high - fair_low) / safe_pred * 100
        feature_names = NUMERIC_FEATURES + PREDICTION_FEATURES

    return out[feature_names], feature_names


# ---------------------------------------------------------------------------
# Time-aware split
# ---------------------------------------------------------------------------


def _split_train_val(
    X: pd.DataFrame,
    y: np.ndarray,
    first_seen: pd.Series,
    val_frac: float = VAL_FRAC,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, str] | None:
    """Sort labeled rows by first_seen_at; oldest (1-val_frac) train,
    newest val_frac validate — the honest answer to 'how does the model
    perform on tomorrow's listings'.

    Returns ``None`` when either fold ends up below 50 rows.

    On the production corpus the time-tail tends to be heavily skewed
    toward the positive class: fresh intake includes listings that
    came in and sold within their first 30 days (label=1), while the
    negatives (active and ≥horizon old) are pulled from the older
    backlog. When the val fold ends up single-class, time-aware
    early-stopping is meaningless — we fall back to a stratified
    random split so the model still has a usable val set, at the cost
    of temporal honesty in the AUC. The chosen mode is returned so the
    caller can record it on the bundle and downstream consumers can
    decide whether to trust the metric for production drift checks.
    """
    n = len(X)
    if n < 100:
        return None

    order = first_seen.fillna(pd.Timestamp.min).argsort().values
    cutoff = int(n * (1 - val_frac))
    train_idx = order[:cutoff]
    val_idx = order[cutoff:]
    if len(train_idx) >= 50 and len(val_idx) >= 20:
        y_train, y_val = y[train_idx], y[val_idx]
        if len(np.unique(y_train)) >= 2 and len(np.unique(y_val)) >= 2:
            return (
                X.iloc[train_idx], y_train,
                X.iloc[val_idx], y_val,
                "time_aware",
            )

    # Stratified random fallback — only path that survives a
    # single-class time-tail.
    from sklearn.model_selection import train_test_split

    if len(np.unique(y)) < 2:
        return None
    try:
        idx_train, idx_val = train_test_split(
            np.arange(n), test_size=val_frac, random_state=42, stratify=y,
        )
    except ValueError:
        return None
    if len(idx_train) < 50 or len(idx_val) < 20:
        return None
    return (
        X.iloc[idx_train], y[idx_train],
        X.iloc[idx_val], y[idx_val],
        "stratified_random",
    )


# ---------------------------------------------------------------------------
# Train / predict
# ---------------------------------------------------------------------------


_LGB_PARAMS = dict(
    objective="binary",
    n_estimators=1000,
    max_depth=6,
    num_leaves=31,
    learning_rate=0.05,
    min_child_samples=20,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)
_EARLY_STOPPING_ROUNDS = 40
_MIN_N_ESTIMATORS = 50


def train_hazard_model(
    listings_df: pd.DataFrame,
    predictions_df: pd.DataFrame | None = None,
    *,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    min_samples: int = MIN_SAMPLES,
) -> dict | None:
    """Fit the binary classifier; return bundle dict or None on
    insufficient data.

    The pipeline:
      1. Compute target with censoring (drop active-and-young, etc.)
      2. Build feature matrix; drop rows with NaN in either feature
         set (LightGBM handles NaN, but val-set ROC-AUC needs clean
         labels so we filter at the row level).
      3. Time-aware 80/20 split by first_seen_at.
      4. Fit on train fold with early stopping on val; record
         best_iteration.
      5. Compute val-set AUC, log-loss, base-rate.
      6. Refit on full labeled corpus with best_iteration so the
         shipped model uses the entire signal, not just 80 %.
    """
    if listings_df is None or listings_df.empty:
        return None

    y_all, labeled = _build_target(listings_df, horizon_days)
    if not labeled.any():
        return None

    X_all, feature_names = _build_features(listings_df, predictions_df)
    # Drop rows with NaN in any required feature — LightGBM tolerates
    # NaN at fit time, but we want clean ROC-AUC, and the predict path
    # will independently surface NaN as "unscored" anyway.
    feature_valid = X_all.notna().all(axis=1).values
    keep = labeled & feature_valid

    if keep.sum() < min_samples:
        return None

    X_kept = X_all[keep].copy()
    y_kept = y_all[keep]
    first_seen_kept = pd.to_datetime(
        listings_df.loc[keep, "first_seen_at"], errors="coerce", utc=True,
    )

    split = _split_train_val(X_kept, y_kept, first_seen_kept)
    if split is None:
        return None
    X_train, y_train, X_val, y_val, split_mode = split

    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    fit_params = {**_LGB_PARAMS, "scale_pos_weight": float(pos_weight)}
    model = lgb.LGBMClassifier(**fit_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(_EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    best_iter = max(int(model.best_iteration_ or fit_params["n_estimators"]),
                    _MIN_N_ESTIMATORS)

    # Validation metrics on the held-out tail
    val_proba = model.predict_proba(X_val)[:, 1]
    metrics = {
        "auc": float(roc_auc_score(y_val, val_proba)),
        "logloss": float(log_loss(y_val, val_proba, labels=[0, 1])),
        "base_rate_train": float(y_train.mean()),
        "base_rate_val": float(y_val.mean()),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_positive_train": int(y_train.sum()),
        "n_positive_val": int(y_val.sum()),
        "best_iteration": best_iter,
        "horizon_days": horizon_days,
        "n_dropped_censored": int((~labeled).sum()),
        "n_dropped_nan_features": int(labeled.sum() - keep.sum()),
        "split_mode": split_mode,
    }

    # Refit on full labeled corpus with the tuned best_iteration so the
    # shipped model uses every available row, not just the 80 % train fold.
    final_params = {**fit_params, "n_estimators": best_iter}
    final_model = lgb.LGBMClassifier(**final_params)
    final_model.fit(X_kept, y_kept)

    return {
        "schema_version": _SCHEMA_VERSION,
        "model": final_model,
        "feature_names": feature_names,
        "horizon_days": horizon_days,
        "uses_predictions": predictions_df is not None and not predictions_df.empty,
        "metrics": metrics,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }


def predict_sold_probability(
    bundle: dict,
    listings_df: pd.DataFrame,
    predictions_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Per-listing P(sold within horizon_days from first_seen_at).

    Returns a DataFrame with columns ``olx_id`` and
    ``prob_sold_within_horizon`` (NaN for rows whose feature row has
    any NaN — those can't be scored). Index aligned to listings_df.
    """
    cols = ["olx_id", "prob_sold_within_horizon"]
    if bundle is None or listings_df is None or listings_df.empty:
        return pd.DataFrame(columns=cols)

    X, features = _build_features(listings_df, predictions_df)
    if features != bundle.get("feature_names"):
        raise ValueError(
            f"feature mismatch: bundle expects {bundle['feature_names']}, "
            f"input produced {features}",
        )

    out = pd.DataFrame(index=listings_df.index)
    out["olx_id"] = listings_df.get("olx_id")
    out["prob_sold_within_horizon"] = np.nan

    valid = X.notna().all(axis=1)
    if not valid.any():
        return out[cols]

    proba = bundle["model"].predict_proba(X[valid].values)[:, 1]
    out.loc[valid, "prob_sold_within_horizon"] = proba
    return out[cols]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_model(bundle: dict) -> None:
    if bundle is None:
        return
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, _MODEL_PATH)
    summary = {
        "schema_version": bundle["schema_version"],
        "horizon_days": bundle["horizon_days"],
        "uses_predictions": bundle["uses_predictions"],
        "n_features": len(bundle["feature_names"]),
        "trained_at": bundle["trained_at"],
        **bundle["metrics"],
    }
    _append_metrics(summary)


def load_model(max_age_hours: float = _MODEL_MAX_AGE_HOURS) -> dict | None:
    """Load saved bundle if fresh + schema-compatible. Otherwise None
    so the caller can decide to retrain or skip hazard scoring."""
    if not _MODEL_PATH.exists():
        return None
    age_hours = (time.time() - _MODEL_PATH.stat().st_mtime) / 3600
    if age_hours > max_age_hours:
        return None
    try:
        bundle = joblib.load(_MODEL_PATH)
    except Exception:
        return None
    if bundle.get("schema_version") != _SCHEMA_VERSION:
        return None
    expected = {"model", "feature_names", "horizon_days", "metrics"}
    if not expected.issubset(bundle.keys()):
        return None
    return bundle


def _append_metrics(metrics: dict) -> None:
    history: list[dict] = []
    if _METRICS_PATH.exists():
        try:
            history = json.loads(_METRICS_PATH.read_text())
        except (json.JSONDecodeError, ValueError):
            pass
    history.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **metrics,
    })
    history = history[-100:]
    _METRICS_PATH.write_text(json.dumps(history, indent=2))


def load_metrics_history() -> list[dict]:
    if not _METRICS_PATH.exists():
        return []
    try:
        return json.loads(_METRICS_PATH.read_text())
    except (json.JSONDecodeError, ValueError):
        return []
