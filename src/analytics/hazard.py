"""Per-listing hazard model — P(sold within horizon_days from listing).

Binary LightGBM classifier on top of the same active+sold corpus the
price model uses. Target: did this listing sell within horizon_days
(default 30) of first_seen_at?

Refines decision.py's segment-level liquidity signals (dom_median,
dom_fast_share) to per-listing: instead of "VW Golf Mk7 sells fast in
Porto on average", answers "this specific Golf, with this price /
photos / damage profile, has 65 % chance of selling in 30 days".

Censoring rules — what gets dropped from training:
  - Active listings younger than horizon_days (target unobservable)
  - Listings without first_seen_at (can't compute age)
  - Listings deactivated for non-sold reasons within horizon
    (ambiguous: parser noise vs genuine removal)

Negatives are observably-not-sold-within-horizon:
  - Active and age >= horizon (still listed past the deadline)
  - Deactivated past horizon (any reason — visibly survived past it)

Schema v2 (2026-05-04):
  - Per-segment median imputation for photo_count / engine_cc /
    horsepower / mileage_km / description_length / damage_severity.
    v1 dropped any row with one NaN feature; on the production corpus
    that was 49 % unscored (mostly photo_count). Layered fallback
    (brand+model+gen → brand+model → brand → global) brings coverage
    above ~90 %.
  - Categorical features added: brand, model, fuel_type, transmission,
    segment, district. LightGBM handles them natively via pandas
    Categorical dtype; train-time category levels persist in the
    bundle so predict-time encodes deterministically.
  - num_price_drops / max_drop_pct still excluded — they're zero on a
    fresh listing, so including them creates a train/predict
    distribution shift that pushes the model to under-predict P(sold)
    on new entries.

Time-aware split with stratified-random fallback when the time-tail
collapses to a single class (typical when fresh intake skews positive
because recently-sold listings dominate the newest first_seen_at).
``metrics.split_mode`` records which path was taken so consumers
can decide whether to trust the AUC for production drift checks.
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

_SCHEMA_VERSION = 2

DEFAULT_HORIZON_DAYS = 30
MIN_SAMPLES = 200
VAL_FRAC = 0.2
# A segment must hold at least this many priced rows for its median
# to anchor an imputation; below that we fall through to the next
# coarser key.
_MIN_SEGMENT_SAMPLES = 5

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

# Numeric columns whose NaNs we fill from per-segment medians. log_price
# is excluded — if price_eur is missing the listing isn't a hazard
# candidate at all.
_IMPUTE_FEATURES = (
    "photo_count",
    "description_length",
    "engine_cc",
    "horsepower",
    "mileage_km",
    "damage_severity",
    "year",
)

CATEGORICAL_FEATURES = [
    "brand",
    "model",
    "fuel_type",
    "transmission",
    "segment",
    "district",
]

PREDICTION_FEATURES = ["residual_pct", "band_pct"]


# ---------------------------------------------------------------------------
# Per-segment imputation
# ---------------------------------------------------------------------------


def _build_segment_lookups(
    df: pd.DataFrame, X: pd.DataFrame,
) -> dict[str, dict]:
    """Per-segment median table for the layered imputation fallback:
    (brand, model, generation) → (brand, model) → (brand,) → global.

    A segment qualifies only if it carries at least
    ``_MIN_SEGMENT_SAMPLES`` non-NaN priced rows for the feature in
    question — that stops us from "imputing" on the basis of one
    outlier in a 2-row brand bucket. Stored in the bundle so inference
    applies the same imputation without re-providing the corpus.
    """
    cols = list(_IMPUTE_FEATURES)
    work = X[cols].copy()
    work["__brand"] = df.get("brand", pd.Series("", index=df.index)).fillna("").astype(str)
    work["__model"] = df.get("model", pd.Series("", index=df.index)).fillna("").astype(str)
    work["__generation"] = (
        df.get("generation", pd.Series("", index=df.index)).fillna("").astype(str)
    )

    out: dict[str, dict] = {
        "global": {},
        "brand": {},        # {(brand,) : {feature: median}}
        "brand_model": {},  # {(brand, model) : {...}}
        "segment": {},      # {(brand, model, generation) : {...}}
    }
    for col in cols:
        med = work[col].median()
        out["global"][col] = float(med) if pd.notna(med) else 0.0

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


def _segment_lookup(lookups: dict, brand, model, generation, feature: str) -> float:
    """Walk the (segment → brand_model → brand → global) chain."""
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


def _apply_segment_imputation(
    df: pd.DataFrame,
    X: pd.DataFrame,
    lookups: dict,
) -> pd.DataFrame:
    """Fill NaN in _IMPUTE_FEATURES from the lookup table. Returns a
    new DataFrame; original X untouched."""
    out = X.copy()
    brands = df.get("brand", pd.Series("", index=df.index)).fillna("").astype(str)
    models = df.get("model", pd.Series("", index=df.index)).fillna("").astype(str)
    generations = (
        df.get("generation", pd.Series("", index=df.index)).fillna("").astype(str)
    )
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
    return out


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

    ``y[i] = 1``  → sold within horizon_days of first_seen_at
    ``y[i] = 0``  → observably not sold by horizon
    ``labeled[i] = False`` → row is censored / unlabelable; drop from training
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
        & first_seen.notna() & deactivated.notna()
        & (days_to_deact >= 0) & (days_to_deact <= horizon_days)
    )
    neg_active = is_active & first_seen.notna() & (days_active >= horizon_days)
    neg_deact_past = (
        (~is_active)
        & first_seen.notna() & deactivated.notna()
        & (days_to_deact > horizon_days)
    )

    y[pos.values] = 1
    y[(neg_active | neg_deact_past).values] = 0
    labeled[(pos | neg_active | neg_deact_past).values] = True
    return y, labeled


# ---------------------------------------------------------------------------
# Feature build (numeric + categoricals)
# ---------------------------------------------------------------------------


def _build_features(
    listings_df: pd.DataFrame,
    predictions_df: pd.DataFrame | None = None,
    *,
    lookups: dict | None = None,
    categorical_levels: dict[str, list[str]] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Build the LightGBM-ready feature DataFrame.

    Returns ``(X, info)`` where ``X`` carries every feature column and
    ``info`` contains:
      - ``feature_names``: full ordered column list
      - ``categorical_features``: subset that's pandas Categorical
      - ``categorical_levels``: per-cat sorted level list (for predict-time
        encoding consistency; built when not supplied)
      - ``lookups``: imputation table (built when not supplied)

    *predictions_df* must be aligned to ``listings_df.index`` — pass
    the direct output of ``price_model.predict_prices``. When supplied,
    residual_pct and band_pct join the feature list.
    """
    df = listings_df
    out = pd.DataFrame(index=df.index)

    out["year"] = pd.to_numeric(df.get("year"), errors="coerce")
    out["mileage_km"] = pd.to_numeric(df.get("mileage_km"), errors="coerce")
    out["engine_cc"] = pd.to_numeric(df.get("engine_cc"), errors="coerce")
    out["horsepower"] = pd.to_numeric(df.get("horsepower"), errors="coerce")
    # No fillna here — a missing price_eur must propagate to log_price=NaN
    # so train + predict can drop the row (we can't impute price; the
    # listing isn't a hazard candidate without it).
    raw_price = pd.to_numeric(df.get("price_eur"), errors="coerce").clip(lower=0)
    out["log_price"] = np.log1p(raw_price)
    out["photo_count"] = pd.to_numeric(df.get("photo_count"), errors="coerce")
    out["description_length"] = pd.to_numeric(
        df.get("description_length"), errors="coerce",
    )
    out["damage_severity"] = pd.to_numeric(
        df.get("damage_severity"), errors="coerce",
    )

    # Per-segment median imputation. Build at train time, reuse at score time.
    if lookups is None:
        lookups = _build_segment_lookups(df, out)
    out = _apply_segment_imputation(df, out, lookups)

    # Categoricals — pandas Categorical so LightGBM picks them up natively.
    saved_levels: dict[str, list[str]] = {}
    for col in CATEGORICAL_FEATURES:
        raw = df.get(col)
        if raw is None:
            values = pd.Series([pd.NA] * len(df), index=df.index, dtype="object")
        else:
            values = raw.astype("string")
        if categorical_levels is not None and col in categorical_levels:
            levels = categorical_levels[col]
            cat = pd.Categorical(values, categories=levels)
        else:
            cat = pd.Categorical(values)
            saved_levels[col] = list(cat.categories)
        out[col] = cat

    feature_names = list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)

    if predictions_df is not None and not predictions_df.empty:
        pred = predictions_df.reindex(df.index)
        price = pd.to_numeric(df.get("price_eur"), errors="coerce")
        pred_price = pd.to_numeric(pred.get("predicted_price"), errors="coerce")
        fair_low = pd.to_numeric(pred.get("fair_price_low"), errors="coerce")
        fair_high = pd.to_numeric(pred.get("fair_price_high"), errors="coerce")
        safe_pred = pred_price.where(pred_price > 0)
        out["residual_pct"] = (price - safe_pred) / safe_pred * 100
        out["band_pct"] = (fair_high - fair_low) / safe_pred * 100
        feature_names = (
            NUMERIC_FEATURES + CATEGORICAL_FEATURES + PREDICTION_FEATURES
        )

    info = {
        "feature_names": feature_names,
        "categorical_features": list(CATEGORICAL_FEATURES),
        "categorical_levels": (
            saved_levels if categorical_levels is None else categorical_levels
        ),
        "lookups": lookups,
    }
    return out[feature_names], info


# ---------------------------------------------------------------------------
# Train / val split
# ---------------------------------------------------------------------------


def _split_train_val(
    X: pd.DataFrame,
    y: np.ndarray,
    first_seen: pd.Series,
    val_frac: float = VAL_FRAC,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, str] | None:
    """Time-aware first; stratified-random fallback when the time-tail
    collapses to a single class. Returns
    ``(X_train, y_train, X_val, y_val, mode)`` or None.

    On the production corpus the time-tail tends to be heavily skewed
    toward the positive class (fresh intake includes listings that came
    in and sold within their first 30 days; negatives are pulled from
    the older backlog). When the val fold ends up single-class,
    time-aware early-stopping is meaningless — we fall back to a
    stratified random split so the model still has a usable val set.
    Less temporally honest, but the only path that lets the model fit
    at all on this corpus shape. ``mode`` is recorded on the bundle so
    drift checks can discount the AUC accordingly.
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

    Pipeline:
      1. Compute target with censoring
      2. Build features (per-segment imputed numerics + categoricals);
         drop only rows where log_price is NaN (price_eur missing)
         since those can't be hazard candidates regardless of segment
      3. Time-aware 80/20 split with stratified-random fallback
      4. Fit on train fold with early stopping on val
      5. Compute val-set AUC, log-loss
      6. Refit on full labeled corpus with the tuned best_iteration
    """
    if listings_df is None or listings_df.empty:
        return None

    y_all, labeled = _build_target(listings_df, horizon_days)
    if not labeled.any():
        return None

    X_all, info = _build_features(listings_df, predictions_df)
    feature_names = info["feature_names"]

    # Drop only rows where the target is missing OR log_price is NaN
    # (price_eur was missing; can't impute price). Other numerics get
    # filled by per-segment median; LightGBM tolerates NaN categoricals.
    log_price_valid = X_all["log_price"].notna().values
    keep = labeled & log_price_valid

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
        categorical_feature=info["categorical_features"],
    )
    best_iter = max(int(model.best_iteration_ or fit_params["n_estimators"]),
                    _MIN_N_ESTIMATORS)

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
        "n_dropped_no_price": int(labeled.sum() - keep.sum()),
        "split_mode": split_mode,
    }

    final_params = {**fit_params, "n_estimators": best_iter}
    final_model = lgb.LGBMClassifier(**final_params)
    final_model.fit(
        X_kept, y_kept,
        categorical_feature=info["categorical_features"],
    )

    return {
        "schema_version": _SCHEMA_VERSION,
        "model": final_model,
        "feature_names": feature_names,
        "categorical_features": info["categorical_features"],
        "categorical_levels": info["categorical_levels"],
        "lookups": info["lookups"],
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

    Returns a DataFrame with ``olx_id`` and
    ``prob_sold_within_horizon``. Listings whose ``log_price`` is NaN
    (price_eur missing) get NaN probability — every other column is
    imputed from the bundle's per-segment lookup table.
    """
    cols = ["olx_id", "prob_sold_within_horizon"]
    if bundle is None or listings_df is None or listings_df.empty:
        return pd.DataFrame(columns=cols)

    X, info = _build_features(
        listings_df, predictions_df,
        lookups=bundle.get("lookups"),
        categorical_levels=bundle.get("categorical_levels"),
    )
    if info["feature_names"] != bundle.get("feature_names"):
        raise ValueError(
            f"feature mismatch: bundle expects {bundle['feature_names']}, "
            f"input produced {info['feature_names']}",
        )

    out = pd.DataFrame(index=listings_df.index)
    out["olx_id"] = listings_df.get("olx_id")
    out["prob_sold_within_horizon"] = np.nan

    valid = X["log_price"].notna()
    if not valid.any():
        return out[cols]

    proba = bundle["model"].predict_proba(X[valid])[:, 1]
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
    expected = {
        "model", "feature_names", "categorical_features",
        "categorical_levels", "lookups", "horizon_days", "metrics",
    }
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
