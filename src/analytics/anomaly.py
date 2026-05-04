"""IsolationForest anomaly detection — negative filter for the deal scorer.

Catches listings whose feature combination falls outside the typical
distribution: parser artefacts (engine_cc=10000, mileage 9_999_999),
suspiciously cheap rare configurations that weren't flagged as salvage
elsewhere, and other patterns the explicit hard gates in
``decision.py`` don't cover.

Designed as a soft signal — emit ``anomaly_score`` ∈ [0, 1] per
listing (1.0 = extreme outlier), let the consumer decide whether to
gate-reject (e.g. ``decide()`` rejecting ``score >= 0.9``) or
down-weight confidence.

Trains on the enriched active+sold corpus (no labels needed). Default
contamination 0.05 = top 5 % flagged. Lower for cleaner data, higher
for noisier datasets where you want more aggressive flagging.

When ``predictions_df`` is supplied (the same shape ``predict_prices``
returns — index aligned with ``listings_df``), residual_pct and
band_pct are added to the feature set. That's what catches the
"suspiciously underpriced for its specs" pattern — the most
useful anomaly type for deal-scoring.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


_MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_MODEL_PATH = _MODEL_DIR / "anomaly_model.joblib"
_METRICS_PATH = _MODEL_DIR / "anomaly_metrics.json"
_MODEL_MAX_AGE_HOURS = 24

# Bumped when the bundle on-disk shape changes (feature list,
# transforms, contamination semantics). load_model rejects mismatches
# so consumers can't accidentally apply an artifact trained against a
# different feature set.
_SCHEMA_VERSION = 1

# Base features available without external predictions
BASE_FEATURES = [
    "log_price",
    "year",
    "log_mileage",
    "log_photo_count",
    "log_description_length",
    "engine_cc",
    "horsepower",
    "seats",
]

# Optional features that depend on price_model output
PREDICTION_FEATURES = [
    "residual_pct",   # (price - predicted) / predicted × 100
    "band_pct",       # (high - low) / predicted × 100
]

_MIN_SAMPLES = 100


def _build_features(
    listings_df: pd.DataFrame,
    predictions_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Build the numeric feature matrix.

    Numeric only — IsolationForest doesn't natively handle categoricals
    and one-hot encoding ~30 brands × dozens of models would dilute
    the anomaly signal across thousands of rare combinations. Brand /
    model patterns are captured indirectly via residual_pct when
    ``predictions_df`` is supplied (the price model has already
    learned them).

    ``predictions_df`` must be aligned to ``listings_df.index`` — pass
    the direct output of ``price_model.predict_prices``.
    """
    df = listings_df
    out = pd.DataFrame(index=df.index)

    out["log_price"] = np.log1p(
        pd.to_numeric(df.get("price_eur"), errors="coerce")
        .fillna(0).clip(lower=0)
    )
    out["year"] = pd.to_numeric(df.get("year"), errors="coerce")
    out["log_mileage"] = np.log1p(
        pd.to_numeric(df.get("mileage_km"), errors="coerce")
        .fillna(0).clip(lower=0)
    )
    out["log_photo_count"] = np.log1p(
        pd.to_numeric(df.get("photo_count"), errors="coerce")
        .fillna(0).clip(lower=0)
    )
    out["log_description_length"] = np.log1p(
        pd.to_numeric(df.get("description_length"), errors="coerce")
        .fillna(0).clip(lower=0)
    )
    out["engine_cc"] = pd.to_numeric(df.get("engine_cc"), errors="coerce")
    out["horsepower"] = pd.to_numeric(df.get("horsepower"), errors="coerce")
    out["seats"] = pd.to_numeric(df.get("seats"), errors="coerce")

    features = list(BASE_FEATURES)

    if predictions_df is not None and not predictions_df.empty:
        pred = predictions_df.reindex(df.index)
        price = pd.to_numeric(df.get("price_eur"), errors="coerce")
        pred_price = pd.to_numeric(pred.get("predicted_price"), errors="coerce")
        fair_low = pd.to_numeric(pred.get("fair_price_low"), errors="coerce")
        fair_high = pd.to_numeric(pred.get("fair_price_high"), errors="coerce")
        # Guard against /0 — leave those rows with NaN, dropped later.
        safe_pred = pred_price.where(pred_price > 0)
        out["residual_pct"] = (price - safe_pred) / safe_pred * 100
        out["band_pct"] = (fair_high - fair_low) / safe_pred * 100
        features.extend(PREDICTION_FEATURES)

    return out[features], features


# ---------------------------------------------------------------------------
# Train / score
# ---------------------------------------------------------------------------


def train_anomaly_detector(
    listings_df: pd.DataFrame,
    predictions_df: pd.DataFrame | None = None,
    *,
    contamination: float = 0.05,
    n_estimators: int = 200,
    min_samples: int = _MIN_SAMPLES,
) -> dict | None:
    """Fit IsolationForest on the corpus.

    Returns a bundle dict ready for ``save_model`` / ``score_anomalies``,
    or None when there's not enough data after dropping NaN-feature rows.

    *contamination* — expected fraction of anomalies in the training
    data (also sets the per-row decision threshold). Default 0.05 = 5 %.
    Lower for cleaner data (1 %), higher for noisier datasets (10 %).
    """
    if listings_df is None or listings_df.empty:
        return None

    X, features = _build_features(listings_df, predictions_df)
    valid = X.notna().all(axis=1)
    X_valid = X[valid].copy()
    n_dropped = int((~valid).sum())

    if len(X_valid) < min_samples:
        return None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_valid.values)

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    # Capture training-set raw score range so score_anomalies can map to
    # a stable [0, 1] scale across runs without re-scaling per call.
    raw_scores = model.score_samples(X_scaled)
    raw_min = float(raw_scores.min())
    raw_max = float(raw_scores.max())
    threshold_raw = float(np.quantile(raw_scores, contamination))

    return {
        "schema_version": _SCHEMA_VERSION,
        "model": model,
        "scaler": scaler,
        "feature_names": features,
        "contamination": contamination,
        "threshold_raw": threshold_raw,
        "raw_min": raw_min,
        "raw_max": raw_max,
        "n_samples": len(X_valid),
        "n_dropped_nan": n_dropped,
        "uses_predictions": predictions_df is not None and not predictions_df.empty,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }


def score_anomalies(
    bundle: dict,
    listings_df: pd.DataFrame,
    predictions_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return per-listing ``anomaly_score`` and ``is_anomaly`` flag.

    ``anomaly_score`` ∈ [0, 1] mapped from the IsolationForest raw
    decision_function output using the training-set min/max captured
    in the bundle. Higher = more anomalous (1.0 = at-or-beyond the
    most-anomalous training row). Comparable across runs against the
    same bundle.

    ``is_anomaly`` = True when the raw score falls below the
    contamination-derived threshold from training.

    Listings whose feature row has any NaN can't be scored — they
    pass through with anomaly_score=NaN and is_anomaly=False.
    """
    cols = ["olx_id", "anomaly_score", "is_anomaly"]
    if bundle is None or listings_df is None or listings_df.empty:
        return pd.DataFrame(columns=cols)

    X, features = _build_features(listings_df, predictions_df)
    if features != bundle.get("feature_names"):
        # Bundle expects features that the input can't supply (e.g.
        # bundle was trained with predictions, caller didn't pass
        # predictions_df). Refuse to score rather than silently
        # returning bogus values.
        raise ValueError(
            f"feature mismatch: bundle expects {bundle['feature_names']}, "
            f"input produced {features}",
        )

    valid = X.notna().all(axis=1)
    out = pd.DataFrame(index=listings_df.index)
    out["olx_id"] = listings_df.get("olx_id")
    out["anomaly_score"] = np.nan
    out["is_anomaly"] = False

    if not valid.any():
        return out[cols]

    X_scaled = bundle["scaler"].transform(X[valid].values)
    raw = bundle["model"].score_samples(X_scaled)

    raw_min, raw_max = bundle["raw_min"], bundle["raw_max"]
    span = raw_max - raw_min
    if span > 1e-9:
        # Lower raw = more anomalous; invert so higher final = more anomalous.
        # Clip so test-time outliers worse than the training extreme don't
        # produce >1.0 scores.
        scaled = 1.0 - (raw - raw_min) / span
        scaled = np.clip(scaled, 0.0, 1.0)
    else:
        scaled = np.zeros_like(raw)

    out.loc[valid, "anomaly_score"] = scaled
    out.loc[valid, "is_anomaly"] = raw < bundle["threshold_raw"]
    return out[cols]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_model(bundle: dict) -> None:
    """Persist the bundle to disk and append metrics to history."""
    if bundle is None:
        return
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, _MODEL_PATH)
    metrics = {
        "schema_version": bundle["schema_version"],
        "contamination": bundle["contamination"],
        "n_samples": bundle["n_samples"],
        "n_dropped_nan": bundle["n_dropped_nan"],
        "uses_predictions": bundle["uses_predictions"],
        "n_features": len(bundle["feature_names"]),
        "trained_at": bundle["trained_at"],
    }
    _append_metrics(metrics)


def load_model(max_age_hours: float = _MODEL_MAX_AGE_HOURS) -> dict | None:
    """Load the saved bundle if it exists, is fresh, and matches schema.

    Mirrors ``price_model.load_model``: returns None on staleness or
    schema mismatch so the dashboard can decide whether to retrain
    inline or skip anomaly scoring entirely until the next CI run.
    """
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
    expected_keys = {
        "model", "scaler", "feature_names", "contamination",
        "threshold_raw", "raw_min", "raw_max",
    }
    if not expected_keys.issubset(bundle.keys()):
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
