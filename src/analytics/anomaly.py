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

# Schema v2 (2026-05-04):
#   - per-segment median imputation for engine_cc/hp/seats/photo_count/
#     description_length so a missing field doesn't drop the row (was
#     ~35 % unscored on the production corpus).
#   - within-segment normalisation of log_price so a BMW M3 at €43.5k
#     doesn't read as a global outlier just because most cars are €5–15 k.
#   - split into TWO IsolationForests: feature_model (numeric specs) and
#     residual_model (price-vs-prediction). Caller gets both scores back
#     so downstream can react differently to a "weird specs" anomaly vs
#     a "suspiciously priced for the segment" anomaly.
_SCHEMA_VERSION = 2

# Base features available without external predictions
BASE_FEATURES = [
    "log_price_seg_resid",   # log_price minus segment median(log_price)
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

# Features we attempt to impute when missing. log_price_seg_resid is
# excluded — if price_eur is missing the listing isn't a candidate at all.
_IMPUTE_FEATURES = (
    "engine_cc", "horsepower", "seats",
    "log_photo_count", "log_description_length", "year", "log_mileage",
)

_MIN_SAMPLES = 100
# Segments need at least this many priced listings to anchor a median;
# below that we fall back to the next coarser key in the chain.
_MIN_SEGMENT_SAMPLES = 5


def _build_segment_lookups(
    df: pd.DataFrame, X: pd.DataFrame,
) -> dict[str, dict]:
    """Per-segment median lookups for imputation + log_price normalisation.

    Layered fallback (most-specific → least-specific):
      (brand, model, generation) → (brand, model) → (brand,) → global

    A segment qualifies only if it has at least ``_MIN_SEGMENT_SAMPLES``
    priced rows for the feature in question — otherwise we fall through
    to the next coarser key. That stops us from "imputing" on the basis
    of one quirky outlier in a 2-row brand bucket.

    Stored in the bundle so inference can apply the same imputation
    without the caller re-providing the full corpus.
    """
    cols = list(_IMPUTE_FEATURES) + ["log_price"]
    work = X[cols].copy()
    work["__brand"] = df.get("brand", pd.Series("", index=df.index)).fillna("")
    work["__model"] = df.get("model", pd.Series("", index=df.index)).fillna("")
    work["__generation"] = (
        df.get("generation", pd.Series("", index=df.index)).fillna("").astype(str)
    )

    out: dict[str, dict] = {
        "global": {},
        "brand": {},          # {(brand,) : {feature: median}}
        "brand_model": {},    # {(brand, model) : {...}}
        "segment": {},        # {(brand, model, generation) : {...}}
    }

    for col in cols:
        global_med = work[col].median()
        out["global"][col] = (
            float(global_med) if pd.notna(global_med) else 0.0
        )

    for keys, grp in work.groupby(["__brand"], dropna=False):
        if len(grp) < _MIN_SEGMENT_SAMPLES:
            continue
        out["brand"][keys] = {
            col: float(grp[col].median())
            for col in cols if pd.notna(grp[col].median())
        }
    for keys, grp in work.groupby(["__brand", "__model"], dropna=False):
        if len(grp) < _MIN_SEGMENT_SAMPLES:
            continue
        out["brand_model"][keys] = {
            col: float(grp[col].median())
            for col in cols if pd.notna(grp[col].median())
        }
    for keys, grp in work.groupby(
        ["__brand", "__model", "__generation"], dropna=False,
    ):
        if len(grp) < _MIN_SEGMENT_SAMPLES:
            continue
        out["segment"][keys] = {
            col: float(grp[col].median())
            for col in cols if pd.notna(grp[col].median())
        }
    return out


def _segment_lookup(lookups: dict, brand, model, generation, feature):
    """Walk the fallback chain for a single (segment, feature) pair."""
    seg_key = (str(brand or ""), str(model or ""), str(generation or ""))
    bm_key = seg_key[:2]
    b_key = seg_key[:1]
    for layer, key in (
        ("segment", seg_key),
        ("brand_model", bm_key),
        ("brand", b_key),
    ):
        v = lookups.get(layer, {}).get(key, {}).get(feature)
        if v is not None:
            return v
    return lookups.get("global", {}).get(feature, 0.0)


def _build_features(
    listings_df: pd.DataFrame,
    predictions_df: pd.DataFrame | None = None,
    lookups: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Build numeric feature matrices for the two IsolationForests.

    Returns ``(features_df, info)`` where ``features_df`` carries every
    column referenced by either model and ``info`` contains:
      - ``base_features``: column names for the spec-axis IF.
      - ``residual_features``: column names for the price-vs-prediction
        IF (empty when ``predictions_df`` is None / empty).
      - ``lookups``: the per-segment median table (built when not
        supplied — train path; reused when supplied — score path).

    Numeric only — IsolationForest doesn't natively handle categoricals
    and one-hot encoding ~30 brands × dozens of models would dilute
    the anomaly signal across thousands of rare combinations. Brand /
    model patterns are captured via per-segment normalisation of
    log_price (``log_price_seg_resid``) plus the residual_pct feature
    when predictions are supplied.

    ``predictions_df`` must be aligned to ``listings_df.index``.
    """
    df = listings_df
    out = pd.DataFrame(index=df.index)

    log_price = np.log1p(
        pd.to_numeric(df.get("price_eur"), errors="coerce")
        .fillna(0).clip(lower=0)
    )
    out["log_price"] = log_price  # raw; segment-residualised below
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

    # Lookups: build at train time, reuse at score time.
    if lookups is None:
        lookups = _build_segment_lookups(df, out)

    # Per-segment median for log_price → residual feature. Captures
    # "this row is unusually cheap/expensive for its segment" without
    # the global "BMW M3 is unusually expensive vs the corpus" noise.
    brands = df.get("brand", pd.Series("", index=df.index)).fillna("")
    models = df.get("model", pd.Series("", index=df.index)).fillna("")
    generations = (
        df.get("generation", pd.Series("", index=df.index)).fillna("").astype(str)
    )
    seg_log_price_med = pd.Series(
        [
            _segment_lookup(lookups, b, m, g, "log_price")
            for b, m, g in zip(brands, models, generations)
        ],
        index=df.index,
        dtype=float,
    )
    out["log_price_seg_resid"] = log_price - seg_log_price_med

    # Imputation for engine_cc / hp / seats / photo / desc_len / year /
    # mileage. The 2026-05-04 audit showed 35 % of listings dropping out
    # of scoring because at least one of these was NaN; per-segment
    # median fill brings coverage to ~95 %.
    for col in _IMPUTE_FEATURES:
        if col not in out.columns:
            continue
        missing = out[col].isna()
        if not missing.any():
            continue
        fill_vals = pd.Series(
            [
                _segment_lookup(lookups, b, m, g, col)
                for b, m, g in zip(
                    brands[missing], models[missing], generations[missing],
                )
            ],
            index=out.index[missing],
            dtype=float,
        )
        out.loc[missing, col] = fill_vals

    base_features = list(BASE_FEATURES)
    residual_features: list[str] = []

    if predictions_df is not None and not predictions_df.empty:
        pred = predictions_df.reindex(df.index)
        price = pd.to_numeric(df.get("price_eur"), errors="coerce")
        pred_price = pd.to_numeric(pred.get("predicted_price"), errors="coerce")
        fair_low = pd.to_numeric(pred.get("fair_price_low"), errors="coerce")
        fair_high = pd.to_numeric(pred.get("fair_price_high"), errors="coerce")
        safe_pred = pred_price.where(pred_price > 0)
        out["residual_pct"] = (price - safe_pred) / safe_pred * 100
        out["band_pct"] = (fair_high - fair_low) / safe_pred * 100
        residual_features = list(PREDICTION_FEATURES)

    return out, {
        "base_features": base_features,
        "residual_features": residual_features,
        "lookups": lookups,
    }


# ---------------------------------------------------------------------------
# Train / score
# ---------------------------------------------------------------------------


def _fit_one_if(
    X: pd.DataFrame, features: list[str], *,
    contamination: float, n_estimators: int,
) -> dict | None:
    """Fit a single IsolationForest on the given feature subset.

    Returns ``{model, scaler, raw_min, raw_max, threshold_raw, n_samples}``
    or None when fewer than ``_MIN_SAMPLES`` rows have all features
    populated. Captures the training-set raw-score range so inference
    can produce a stable [0, 1] mapping.
    """
    valid = X[features].notna().all(axis=1)
    Xv = X.loc[valid, features].copy()
    if len(Xv) < _MIN_SAMPLES:
        return None
    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xv.values)
    model = IsolationForest(
        n_estimators=n_estimators, contamination=contamination,
        random_state=42, n_jobs=-1,
    )
    model.fit(Xs)
    raw = model.score_samples(Xs)
    return {
        "model": model,
        "scaler": scaler,
        "raw_min": float(raw.min()),
        "raw_max": float(raw.max()),
        "threshold_raw": float(np.quantile(raw, contamination)),
        "n_samples": int(len(Xv)),
    }


def train_anomaly_detector(
    listings_df: pd.DataFrame,
    predictions_df: pd.DataFrame | None = None,
    *,
    contamination: float = 0.05,
    n_estimators: int = 200,
    min_samples: int = _MIN_SAMPLES,
) -> dict | None:
    """Fit two IsolationForests on the corpus and bundle them together.

    The 2026-05-04 split:
      - ``feature_model`` — trained on ``BASE_FEATURES`` (specs + price-
        relative-to-segment). Catches "engine_cc=10000 / mileage 9_999_999 /
        suspicious-spec" patterns.
      - ``residual_model`` — trained on ``PREDICTION_FEATURES``. Catches
        "ask drifted off the model's predicted band" patterns.

    Returns a bundle dict ready for ``save_model`` / ``score_anomalies``,
    or None when there's not enough data to fit even the feature model
    (the spec axis is the strict requirement; the residual axis is
    optional and falls back to None when predictions_df is missing).

    Per-segment median imputation (added v2) brings the coverage of the
    feature model from ~65 % to ~95 % on the production corpus —
    listings that were dropped by the v1 ``X.notna().all()`` filter for
    a single missing engine_cc / horsepower / seats now contribute.
    """
    if listings_df is None or listings_df.empty:
        return None

    X, info = _build_features(listings_df, predictions_df)
    base_features = info["base_features"]
    residual_features = info["residual_features"]
    lookups = info["lookups"]

    feature_bundle = _fit_one_if(
        X, base_features,
        contamination=contamination, n_estimators=n_estimators,
    )
    if feature_bundle is None:
        return None

    residual_bundle = None
    if residual_features:
        residual_bundle = _fit_one_if(
            X, residual_features,
            contamination=contamination, n_estimators=n_estimators,
        )

    n_total = len(X)
    n_feat_valid = X[base_features].notna().all(axis=1).sum()
    n_res_valid = (
        int(X[residual_features].notna().all(axis=1).sum())
        if residual_features else 0
    )

    return {
        "schema_version": _SCHEMA_VERSION,
        "feature_bundle": feature_bundle,
        "residual_bundle": residual_bundle,
        "base_features": base_features,
        "residual_features": residual_features,
        "segment_lookups": lookups,
        "contamination": contamination,
        "n_samples": n_total,
        "n_feature_valid": int(n_feat_valid),
        "n_residual_valid": n_res_valid,
        "uses_predictions": bool(residual_features),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }


def _score_one_axis(
    sub: dict, X: pd.DataFrame, features: list[str], target_index,
) -> tuple[pd.Series, pd.Series]:
    """Map raw IsolationForest output → [0, 1] score + is_anomaly flag,
    aligned to ``target_index``. Rows missing any feature get NaN /
    False. Used twice — once for feature_bundle, once for residual."""
    score = pd.Series(np.nan, index=target_index, dtype=float)
    flag = pd.Series(False, index=target_index, dtype=bool)
    if sub is None or not features:
        return score, flag
    valid = X[features].notna().all(axis=1)
    if not valid.any():
        return score, flag
    Xs = sub["scaler"].transform(X.loc[valid, features].values)
    raw = sub["model"].score_samples(Xs)
    span = sub["raw_max"] - sub["raw_min"]
    if span > 1e-9:
        scaled = 1.0 - (raw - sub["raw_min"]) / span
        scaled = np.clip(scaled, 0.0, 1.0)
    else:
        scaled = np.zeros_like(raw)
    score.loc[valid] = scaled
    flag.loc[valid] = raw < sub["threshold_raw"]
    return score, flag


def score_anomalies(
    bundle: dict,
    listings_df: pd.DataFrame,
    predictions_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return per-listing anomaly scores from both axes.

    Output columns:
      - ``feature_anomaly_score`` ∈ [0, 1] from the spec-axis IF.
      - ``feature_is_anomaly`` boolean (raw score below training threshold).
      - ``residual_anomaly_score`` ∈ [0, 1] or NaN — populated only when
        the bundle was trained with predictions AND the caller supplied
        ``predictions_df``.
      - ``residual_is_anomaly`` boolean.
      - ``anomaly_score`` (legacy) — element-wise max of the two axes,
        for back-compat with consumers reading the v1 column name.
      - ``is_anomaly`` (legacy) — OR of the two axes.

    Per-segment median imputation runs inside ``_build_features`` using
    the lookup table the bundle captured at train time, so a listing
    with NaN engine_cc/hp/seats no longer drops out of scoring.
    """
    cols = [
        "olx_id",
        "feature_anomaly_score", "feature_is_anomaly",
        "residual_anomaly_score", "residual_is_anomaly",
        "anomaly_score", "is_anomaly",
    ]
    if bundle is None or listings_df is None or listings_df.empty:
        return pd.DataFrame(columns=cols)

    base_features = bundle.get("base_features") or []
    residual_features = bundle.get("residual_features") or []
    lookups = bundle.get("segment_lookups")

    X, info = _build_features(listings_df, predictions_df, lookups=lookups)

    if base_features != info["base_features"]:
        raise ValueError(
            f"feature mismatch: bundle expects base={base_features}, "
            f"input produced {info['base_features']}",
        )
    if (residual_features or info["residual_features"]) and \
            residual_features != info["residual_features"]:
        # Bundle was trained with predictions, caller didn't pass them
        # (or vice versa). Tolerate the asymmetric case where bundle
        # has residual features but caller can't supply them — just
        # leave residual_anomaly_score = NaN.
        if not info["residual_features"]:
            residual_features = []

    feat_score, feat_flag = _score_one_axis(
        bundle.get("feature_bundle"), X, base_features, listings_df.index,
    )
    res_score, res_flag = _score_one_axis(
        bundle.get("residual_bundle"), X, residual_features, listings_df.index,
    )

    out = pd.DataFrame(index=listings_df.index)
    out["olx_id"] = listings_df.get("olx_id")
    out["feature_anomaly_score"] = feat_score
    out["feature_is_anomaly"] = feat_flag
    out["residual_anomaly_score"] = res_score
    out["residual_is_anomaly"] = res_flag

    # Legacy aggregate so v1 consumers (dashboard cards, decision.py
    # gating on ``anomaly_score >= 0.9``) keep working unchanged.
    legacy = pd.concat([feat_score, res_score], axis=1).max(axis=1, skipna=True)
    out["anomaly_score"] = legacy
    out["is_anomaly"] = (feat_flag | res_flag).fillna(False)
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
        "n_feature_valid": bundle["n_feature_valid"],
        "n_residual_valid": bundle["n_residual_valid"],
        "uses_predictions": bundle["uses_predictions"],
        "n_base_features": len(bundle["base_features"]),
        "n_residual_features": len(bundle["residual_features"]),
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
        "feature_bundle", "base_features", "residual_features",
        "segment_lookups", "contamination",
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
