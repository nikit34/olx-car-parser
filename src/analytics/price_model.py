"""LightGBM price model with quantile regression, cross-validation, and persistence.

Uses all available features including LLM-extracted fields (accident,
condition, RHD, etc.) and market data (avg_days_to_sell).

Three models: median (predicted price), 10th percentile (low), 90th percentile (high).
LightGBM handles NaN natively and supports native categorical features.

Cross-validation provides MAE/MAPE/R² quality metrics.  Model can be
saved/loaded to avoid retraining on every dashboard visit.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import make_scorer, mean_pinball_loss
from sklearn.model_selection import KFold


_MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_MODEL_PATH = _MODEL_DIR / "price_model.joblib"
_METRICS_PATH = _MODEL_DIR / "price_metrics.json"
_IMPORTANCE_PATH = _MODEL_DIR / "price_importance.json"
_MODEL_MAX_AGE_HOURS = 24
_MIN_CATEGORY_COUNT = 3
_OTHER_CATEGORY = "__other__"
# Bumped whenever the bundle on-disk shape changes (feature list, log target,
# CQR semantics, OOF preds, etc.). load_model rejects mismatches so the
# dashboard can't accidentally consume an artifact trained against a different
# feature set or target transform.
# v4: rule-based damage signals + TF-IDF text PCs + per-bucket conformal_q +
# monotonic constraints + inverse-log sample weights
# v5: LLM-extracted damage_severity (0-3) — replaces the rule-based score's
# role as the primary damage feature. Keyword rules stay as a cheap backup.
# v7: drop near-zero LLM/text features (ablation 2026-04). The columns
# stopped feeding the model, but the TF-IDF→SVD pipeline and damage-keyword
# scan kept running on every train/predict for nothing.
# v8: drop the dead pipeline + helpers entirely. Bundle no longer carries
# `text_pipeline`; `_prepare_X` stops fitting a TF-IDF/SVD on every call.
_SCHEMA_VERSION = 10  # v10: uncertainty model uses 3 extra meta-features
# Fraction of the dataset (newest rows by first_seen_at) used as the
# time-honest conformal calibration window. Random-KFold CQR mixes
# time-adjacent rows across folds and over-estimates coverage on real future
# listings — backtest showed 80% nominal coverage delivering 61–69% on
# tomorrow's data. Calibrating on a held-out time-tail closes that gap.
_TIME_CALIBRATION_FRAC = 0.2


NUMERIC_FEATURES = [
    "year", "mileage_km", "engine_cc", "horsepower",
    "avg_days_to_sell",
    "photo_count", "description_length", "seats",
    # NOTE: price-history features (num_price_drops, max_drop_pct,
    # price_drop_velocity, days_since_last_drop) are intentionally excluded.
    # They are post-hoc — known only after observing the listing for days —
    # and create circular logic: model learns "listings that dropped are
    # cheaper" instead of explaining price from the car's own attributes.
    # Dashboard still displays them as separate indicators.
    #
    # NOTE: LLM-extracted damage_severity and rule-based damage_score were
    # dropped in schema v7 — the 2026-04 ablation showed median permutation
    # importance ≤ 0.002 and removing them gave +0.7 % MAPE drift inside
    # CV noise while improving tail pinball. text_pc_0..7 (TF-IDF→SVD on
    # title+description) dropped for the same reason — none of the 8
    # components ranked above 0.003. The TF-IDF pipeline + damage-keyword
    # scan are gone in v8 (no callers, no consumers).
]

BOOL_FEATURES: list[str] = []
# All boolean LLM/rule signals were dropped in schema v7 — see ablation
# note above. desc_mentions_*, right_hand_drive, taxi_fleet_rental,
# warranty, first_owner_selling, title_has_parts_only,
# title_has_severe_damage all sat at near-zero permutation importance.

CATEGORICAL_FEATURES = [
    "brand", "model", "fuel_type", "transmission", "segment",
    "generation",
    "color", "district", "drive_type", "sub_model", "trim_level",
    "doors",
    # urgency and mechanical_condition (LLM-inferred) dropped in v7.
]

_ALL_FEATURES = NUMERIC_FEATURES + BOOL_FEATURES + CATEGORICAL_FEATURES


# --- Monotonic constraints -------------------------------------------------
# Domain priors: more mileage / damage / RHD / taxi-rental → cheaper; newer /
# warranty / first-owner → more expensive. LightGBM enforces these per-tree
# splits, which dramatically reduces variance in sparse subgroups (e.g. a
# 2003 Honda Civic with 50k km can otherwise produce nonsense). Categoricals
# get 0 (no constraint — ordering is arbitrary). horsepower-vs-price isn't
# strictly monotone (small electric cars are pricey) so leave 0.
_MONOTONE_BY_FEATURE: dict[str, int] = {
    "year": 1,
    "mileage_km": -1,
}


def _monotone_constraints() -> list[int]:
    return [_MONOTONE_BY_FEATURE.get(f, 0) for f in _ALL_FEATURES]


def _build_categorical_mapping(vals: pd.Series) -> dict[str, int]:
    """Build a stable mapping, grouping rare categories into __other__."""
    non_missing = vals.dropna().astype(str)
    if non_missing.empty:
        return {}

    counts = non_missing.value_counts()
    frequent = sorted(counts[counts >= _MIN_CATEGORY_COUNT].index.tolist())

    if len(frequent) < len(counts):
        values = [*frequent, _OTHER_CATEGORY]
    else:
        values = frequent

    return {value: i for i, value in enumerate(values)}


def _encode_categoricals(
    df: pd.DataFrame,
    cat_maps: dict[str, dict[str, int]] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    """Ordinal-encode categoricals for LightGBM."""
    out = df.copy()
    maps: dict[str, dict[str, int]] = cat_maps or {}

    for col in CATEGORICAL_FEATURES:
        if col not in out.columns:
            out[col] = np.nan
            continue

        if col not in maps:
            maps[col] = _build_categorical_mapping(out[col])

        mapping = maps[col]
        encoded = pd.Series(np.nan, index=out.index, dtype=float)
        non_missing = out[col].notna()

        if non_missing.any():
            vals = out.loc[non_missing, col].astype(str)
            if _OTHER_CATEGORY in mapping:
                known = set(mapping)
                known.discard(_OTHER_CATEGORY)
                vals = vals.where(vals.isin(known), _OTHER_CATEGORY)
            encoded.loc[non_missing] = vals.map(mapping).astype(float)

        out[col] = encoded

    return out, maps


def _prepare_X(
    df: pd.DataFrame,
    cat_maps: dict[str, dict[str, int]] | None = None,
) -> tuple[np.ndarray, dict[str, dict[str, int]]]:
    """Prepare feature matrix from listings DataFrame.

    Returns (X_arr, cat_maps). cat_maps is the ordinal-encoding mapping
    fitted on this dataframe; pass it back in to reuse the encoding for
    new rows (CV folds, calibration slice, ``predict_prices``).
    """
    X = df.reindex(columns=_ALL_FEATURES).copy()

    for col in BOOL_FEATURES:
        if col in X.columns:
            X[col] = X[col].astype(float)

    for col in NUMERIC_FEATURES:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    X, maps = _encode_categoricals(X, cat_maps)
    X_arr = X[_ALL_FEATURES].values.astype(float)
    return X_arr, maps


_LGB_PARAMS = dict(
    # Ceiling only — the actual tree count comes from early stopping in CV
    # (`best_n_estimators`), so this just has to be high enough not to cap
    # the search.  On ~8k rows early stopping was pegged at 699/700 before
    # this bump, meaning the old cap was the bottleneck, not overfitting.
    n_estimators=2000,
    # max_depth and num_leaves must be consistent: a tree of depth d can have
    # at most 2^d leaves. With max_depth=4, num_leaves was silently capped at
    # 16 — depth=6 lets num_leaves=31 actually be reached.
    max_depth=6,
    learning_rate=0.05,
    min_child_samples=10,
    reg_lambda=1.5,
    num_leaves=31,
    random_state=42,
    verbose=-1,
    n_jobs=-1,
    # NOTE: monotone_constraints intentionally NOT set here. LightGBM
    # rejects them with the ``quantile`` objective the tail models use, so
    # we apply them only to the median model (which uses ``regression_l1``,
    # whose loss minimum equals the α=0.5 pinball loss). See
    # _model_for_quantile() for the construction.
)


def _model_for_quantile(
    name: str, alpha: float, n_estimators: int,
) -> lgb.LGBMRegressor:
    """Build a quantile regressor with monotone constraints applied where
    LGBM allows them.

    Median uses the standard L2 ``regression`` objective so we can enforce
    domain priors (year↑ → price↑, mileage↑ → price↓, etc.). LGBM 4.6
    rejects monotone constraints with ``quantile`` AND with
    ``regression_l1``, so plain L2 is the only path. The optimum of L2 is
    the conditional mean rather than the median — but in log1p(price)
    space (which we train on) the conditional distribution is close to
    symmetric, so mean ≈ median. The tails keep the standard ``quantile``
    objective; LGBM doesn't let us apply monotone there, but the median's
    monotonic shape indirectly stabilizes them via the post-band-assembly
    sort anyway.
    """
    params = {**_LGB_PARAMS, "n_estimators": n_estimators}
    if name == "median":
        return lgb.LGBMRegressor(
            objective="regression",
            monotone_constraints=_monotone_constraints(),
            monotone_constraints_method="advanced",
            **params,
        )
    return lgb.LGBMRegressor(objective="quantile", alpha=alpha, **params)


# Days-on-market tiers used to estimate the gap between last ask and the
# actual (unknown) sold price for inactive ``deactivation_reason == "sold"``
# listings. Cars that cleared quickly were likely priced near the market;
# slow movers were over-priced and the closing transaction probably came
# in lower. The discounts approximate typical PT used-car negotiation
# margins (3-7 %) padded for over-asking on the long-tail. Sample weights
# step down in parallel so a 200-day-old sold listing contributes less
# signal than a 5-day-old one — both the target estimate and our trust
# in it weaken with time-on-market.
#
# These numbers are heuristic; CI's time-aware backtest is the place to
# tune them. Default tiers were picked to keep median sold-target ≈ 95 %
# of last ask, matching aggregate price-drop telemetry on the 2026-05-03
# corpus (median price_change_pct on sold listings = -4.8 %).
_SOLD_TIERS = (
    # (max_days_inclusive, price_multiplier, sample_weight)
    (14,  0.96, 0.90),
    (60,  0.94, 0.70),
    (180, 0.91, 0.50),
    (365, 0.88, 0.30),
)
_SOLD_MAX_DAYS = 365  # rows beyond this are parser noise (e.g. 2915-day "sold")


def _build_sold_target_adjustment(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-row (target_multiplier, sample_weight) for sold-listing inclusion.

    Active rows pass through unchanged — multiplier=1.0, weight=1.0. Sold
    rows (``deactivation_reason == "sold"``) are scaled by their
    days-on-market bucket. Sold rows with bogus dates (negative or
    > _SOLD_MAX_DAYS) get weight=0 so LightGBM ignores them entirely
    rather than the row being silently dropped — keeps the index aligned
    with the rest of the training arrays.
    """
    n = len(df)
    multiplier = np.ones(n, dtype=float)
    weight = np.ones(n, dtype=float)

    if "is_active" not in df.columns or "deactivation_reason" not in df.columns:
        return multiplier, weight

    is_active = df["is_active"].astype(bool).values
    reason = df["deactivation_reason"].astype(str).values
    sold_mask = (~is_active) & (reason == "sold")
    if not sold_mask.any():
        return multiplier, weight

    deactivated = pd.to_datetime(
        df.get("deactivated_at"), errors="coerce", utc=True,
    )
    first_seen = pd.to_datetime(
        df.get("first_seen_at"), errors="coerce", utc=True,
    )
    days = (deactivated - first_seen).dt.days.fillna(-1).values

    for i in np.where(sold_mask)[0]:
        d = days[i]
        if d < 0 or d > _SOLD_MAX_DAYS:
            weight[i] = 0.0
            continue
        for max_d, mult, w in _SOLD_TIERS:
            if d <= max_d:
                multiplier[i] = mult
                weight[i] = w
                break

    return multiplier, weight


def _compute_sample_weights(y_price: np.ndarray) -> np.ndarray:
    """Inverse-log sample weights so pinball loss is balanced across price
    tiers.

    Without weights, a €30k car contributes ~10× the absolute pinball loss
    of a €3k car for the same relative error — the optimizer over-fits the
    expensive end and under-fits the cheap segment. weight ∝ 1/log1p(price)
    re-scales contributions so a 10% miss costs roughly the same in loss
    regardless of price tier. Normalized so mean weight = 1, which keeps
    the absolute loss magnitude comparable to the unweighted baseline (so
    early-stopping thresholds don't have to be retuned).
    """
    y = np.maximum(y_price, 0).astype(float)
    raw = 1.0 / np.log1p(y + 1.0)  # +1 avoids 1/log1p(0) = inf
    if raw.sum() == 0:
        return np.ones_like(raw)
    return raw * (len(raw) / raw.sum())

_QUANTILES = {"low": 0.1, "median": 0.5, "high": 0.9}
_EARLY_STOPPING_ROUNDS = 40
_MIN_N_ESTIMATORS = 50


def _pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    """Quantile (pinball) loss — the native metric for quantile regression."""
    diff = y_true - y_pred
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1) * diff)))


def _filter_training_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop outliers that would poison the quantile targets.

    Returns (filtered_df, drop_stats) for logging.  Filters:
      1. Price that moved 10× in either direction (data-entry noise, wrong currency)
      2. Prices outside [1st, 99th] percentile (heavy-tail clip)
      3. km / max(age, 1) outside [500, 100 000] (implausible mileage)
    """
    start = len(df)
    out = df.copy()
    stats = {"start": start}

    # 1. 10× price swings from first known price
    if "first_price_eur" in out.columns:
        first = pd.to_numeric(out["first_price_eur"], errors="coerce")
        last = pd.to_numeric(out["price_eur"], errors="coerce")
        ratio = last / first.where(first > 0)
        extreme = (ratio >= 10) | (ratio <= 0.1)
        dropped = int(extreme.fillna(False).sum())
        if dropped:
            out = out[~extreme.fillna(False)]
        stats["dropped_10x_swing"] = dropped

    # 2. Price percentile clip
    prices = pd.to_numeric(out["price_eur"], errors="coerce")
    if len(prices) > 0:
        low_p, high_p = prices.quantile([0.01, 0.99])
        price_mask = (prices >= low_p) & (prices <= high_p)
        stats["dropped_price_percentile"] = int((~price_mask.fillna(False)).sum())
        out = out[price_mask.fillna(False)]

    # 3. Mileage sanity (km per year of age)
    current_year = datetime.now(timezone.utc).year
    years = pd.to_numeric(out["year"], errors="coerce")
    km = pd.to_numeric(out["mileage_km"], errors="coerce")
    age = (current_year - years).clip(lower=1)
    km_per_year = km / age
    implausible = (km_per_year < 500) | (km_per_year > 100_000)
    stats["dropped_mileage_sanity"] = int(implausible.fillna(False).sum())
    out = out[~implausible.fillna(False)]

    stats["kept"] = len(out)
    return out, stats


def _conformal_q_from_scores(
    scores: np.ndarray,
    target_coverage: float = 0.80,
) -> float:
    """The standard CQR quantile-of-scores formula, separated so both the
    random-KFold and time-aware paths use exactly the same statistic."""
    n = len(scores)
    if n == 0:
        return 0.0
    q_level = min(np.ceil(target_coverage * (n + 1)) / n, 1.0)
    return float(np.quantile(scores, q_level))


# Meta-features specific to the uncertainty model. They describe "how
# trustworthy is this listing's data" — orthogonal to the price-relevant
# fields used by the main quantile models. Listings with low completeness
# / few LLM-filled flags / no photos / bare-bones descriptions are exactly
# the rows where huge residuals come from in the cheap-price decile, so
# making the uncertainty model see them explicitly is the targeted fix.
_LLM_FLAG_COLS = (
    "desc_mentions_accident", "desc_mentions_repair",
    "desc_mentions_num_owners", "desc_mentions_customs_cleared",
    "right_hand_drive", "taxi_fleet_rental",
    "warranty", "first_owner_selling",
)


def _uncertainty_extra_features(df: pd.DataFrame) -> np.ndarray:
    """Per-row meta-features that signal listing-data trustworthiness.

    Returns an ``(n_rows, 3)`` float array with these columns:

      0. ``data_completeness`` — fraction of ``_ALL_FEATURES`` non-null.
         Mirrors the runtime helper used elsewhere; computed locally
         here to avoid circular dependence with ``compute_feature_completeness``.
      1. ``llm_flags_filled`` — count (0..8) of LLM-extracted flags
         that are non-null on this row. A serious seller fills these
         out; a salvage / call-only listing leaves them all blank.
      2. ``desc_quality`` — ``log1p(description_length) × log1p(photo_count)``.
         Captures the interaction "long description AND many photos →
         well-documented listing"; either alone is weaker.
    """
    n = len(df)
    if n == 0:
        return np.empty((0, 3))

    # 1. Data completeness — fraction of main-model features non-null.
    cols_present = [c for c in _ALL_FEATURES if c in df.columns]
    if cols_present:
        completeness = df[cols_present].notna().mean(axis=1).values
    else:
        completeness = np.zeros(n)

    # 2. LLM flag fill count.
    flag_count = np.zeros(n)
    for col in _LLM_FLAG_COLS:
        if col in df.columns:
            flag_count = flag_count + df[col].notna().astype(int).values

    # 3. Description × photo interaction.
    desc_len = df.get("description_length", pd.Series(0, index=df.index))
    photos = df.get("photo_count", pd.Series(0, index=df.index))
    desc_log = np.log1p(np.maximum(desc_len.fillna(0).values.astype(float), 0))
    photo_log = np.log1p(np.maximum(photos.fillna(0).values.astype(float), 0))
    desc_quality = desc_log * photo_log

    return np.column_stack([completeness, flag_count, desc_quality])


def _train_uncertainty_model(
    df_train: pd.DataFrame,
    X_train_main: np.ndarray,
    scores_train: np.ndarray,
    df_cal: pd.DataFrame,
    X_cal_main: np.ndarray,
    scores_cal: np.ndarray,
    cat_indices: list[int],
    target_coverage: float = 0.80,
) -> tuple[lgb.LGBMRegressor, float] | None:
    """Per-row CQR-score model + additive conformal correction.

    Trains a regressor ``s ≈ f(features)`` on one half of the cal-set
    where the target is the *signed* CQR score
    (``s_i = max(low_i − y_i, y_i − high_i)`` — negative when ``y`` lies
    safely inside the band, positive when outside). MAE objective keeps
    the fit robust to extreme outliers from salvage rows.

    On the held-out half computes the additive correction ``q_offset``
    such that ``coverage_at(g(x) + q_offset) ≈ target_coverage``. This
    is the classic locally-adaptive conformal recipe — the same
    statistical guarantee as the per-bucket scheme but with the bucket
    replaced by a learned per-row prediction.

    The final inference-time widening (built in ``_per_row_uncertainty_q``)
    is ``max(0, g(x) + q_offset)`` (floored against the global conformal q
    so we never shrink below the marginal-coverage value).

    Feature set is the main model's ``_ALL_FEATURES`` ⊕ three extra
    meta-features built by ``_uncertainty_extra_features`` (data
    completeness, LLM-flag fill count, description × photo interaction)
    that explicitly signal listing-data trustworthiness.

    Returns ``None`` when either half is too small to fit reliably —
    callers fall back to the per-bucket conformal q.
    """
    if len(X_train_main) < 30 or len(X_cal_main) < 20:
        return None

    extras_train = _uncertainty_extra_features(df_train)
    extras_cal = _uncertainty_extra_features(df_cal)
    X_train_unc = np.hstack([X_train_main, extras_train])
    X_cal_unc = np.hstack([X_cal_main, extras_cal])

    model = lgb.LGBMRegressor(
        objective="regression_l1",     # MAE — robust to score outliers
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=15,
        min_data_in_leaf=20,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        verbose=-1,
        random_state=42,
    )
    # Train on signed scores so rows safely inside the band (negative s)
    # contribute to lowering predicted q, not just zero them out — that's
    # what lets per-row q get *narrower* than per-bucket on easy rows.
    # Categorical indices are unchanged: the 3 extras are appended *after*
    # _ALL_FEATURES, all numeric, so cat_indices still points to the same
    # columns by position.
    model.fit(
        X_train_unc, scores_train,
        categorical_feature=cat_indices,
    )

    # Additive conformal correction: pick q_offset so that
    # Pr(s_i <= g(x_i) + q_offset) = target_coverage on the held-out half.
    # Equivalent to running the marginal conformal recipe on the residuals
    # (s_i - g(x_i)) instead of raw scores.
    pred_cal = model.predict(X_cal_unc)
    residuals = scores_cal - pred_cal
    q_offset = _conformal_q_from_scores(residuals, target_coverage=target_coverage)
    return model, float(q_offset)


def _per_row_uncertainty_q(
    df: pd.DataFrame,
    X_main: np.ndarray,
    uncertainty_model: lgb.LGBMRegressor,
    q_offset: float,
    floor_q: float,
) -> np.ndarray:
    """Predict per-row CQR widening: ``max(g(x) + q_offset, floor_q, 0)``.

    Builds the extended uncertainty-feature matrix on the fly by
    concatenating ``X_main`` (the prepared main-model features) with
    the 3 extras from ``_uncertainty_extra_features``. Caller passes
    the raw DataFrame so we can read ``description_length``,
    ``photo_count`` and the LLM-flag columns directly.

    The floor at ``floor_q`` (typically the global conformal q) keeps
    bands from collapsing to zero on rows where the regressor predicts
    very low scores. The floor at 0 prevents a *negative* widening,
    which would tighten the band — for rows where the model predicts
    "this row is well inside the band" we keep the original quantile
    edges, not narrower."""
    extras = _uncertainty_extra_features(df)
    X_unc = np.hstack([X_main, extras])
    raw = uncertainty_model.predict(X_unc) + q_offset
    return np.maximum(np.maximum(raw, 0.0), floor_q)


def _time_aware_conformal_q(
    df: pd.DataFrame,
    best_iters_per_q: dict[str, int],
    calibration_frac: float = _TIME_CALIBRATION_FRAC,
) -> tuple[
    float,
    dict[str, float],
    list[tuple[float, float, str]],
    tuple[lgb.LGBMRegressor, float] | None,
] | None:
    """Compute conformal_q from a time-honest holdout, globally, per
    price bucket, and as a per-row uncertainty model.

    Trains the three quantile models on the oldest ``(1 - calibration_frac)``
    of the data (sorted by ``first_seen_at``), predicts the newest fraction,
    and computes:
      - global q from all band-miss scores
      - bucket edges from the empirical decile distribution of the cal-set
        predicted prices (10 bins by default; falls back to the static
        5-bucket scheme on small data)
      - per-bucket q from scores grouped by those edges
      - per-row uncertainty model + q_scale (option C): a regressor that
        predicts the CQR score from features, trained on half the cal
        set and calibrated on the other half. Replaces the bucket-q
        lookup at inference time with a per-row width — addresses the
        bottleneck where 77% of deals end up NO_OPINION because the
        cheap-decile bucket-q is dominated by salvage outliers.

    Per-bucket q on dynamic deciles closes the within-bucket coverage gap
    the static 5-bucket scheme leaves — reliability curve showed deciles
    1-3 spanning 14pp of coverage even though they all sit inside the
    static ``<€3k`` bucket.

    Returns (global_q, per_bucket_q, bucket_edges, uncertainty_bundle)
    or None if the data lacks ``first_seen_at`` or is too small to split.
    ``uncertainty_bundle`` is ``(model, q_scale)`` when training
    succeeded, else ``None`` (the caller falls back to bucket-q).
    """
    if "first_seen_at" not in df.columns:
        return None
    sorted_df = df.copy()
    sorted_df["_fsa"] = pd.to_datetime(sorted_df["first_seen_at"], errors="coerce")
    sorted_df = sorted_df.dropna(subset=["_fsa"]).sort_values("_fsa").reset_index(drop=True)
    if len(sorted_df) < 200:
        return None

    cutoff = int(len(sorted_df) * (1 - calibration_frac))
    train = sorted_df.iloc[:cutoff]
    cal = sorted_df.iloc[cutoff:]
    if len(train) < 100 or len(cal) < 50:
        return None

    X_train, cat_maps = _prepare_X(train)
    X_cal, _ = _prepare_X(cal, cat_maps)
    y_train_price = train["price_eur"].astype(float).values
    y_train_log = np.log1p(np.maximum(y_train_price, 0))
    y_cal_log = np.log1p(np.maximum(cal["price_eur"].astype(float).values, 0))
    cat_indices = [_ALL_FEATURES.index(c) for c in CATEGORICAL_FEATURES]
    sample_weight = _compute_sample_weights(y_train_price)

    # Need median preds too so we can bucketize the calibration rows by
    # predicted price (matches what predict_prices does at inference time).
    log_preds: dict[str, np.ndarray] = {}
    for name, alpha in _QUANTILES.items():
        model = _model_for_quantile(name, alpha, best_iters_per_q[name])
        model.fit(
            X_train, y_train_log,
            sample_weight=sample_weight,
            categorical_feature=cat_indices,
        )
        log_preds[name] = model.predict(X_cal)

    scores = np.maximum(
        log_preds["low"] - y_cal_log, y_cal_log - log_preds["high"],
    )
    global_q = _conformal_q_from_scores(scores)

    # Per-bucket q. Bucketize by predicted median in price space (what the
    # dashboard sees). Edges are dynamic deciles of the calibration-set
    # predicted prices — closes the within-bucket coverage gap the static
    # 5-bucket scheme leaves. Falls back to the static scheme on tiny sets.
    cal_pred_price = np.expm1(log_preds["median"])
    bucket_edges = _compute_decile_edges(cal_pred_price)
    bucket_labels = _bucketize_price(cal_pred_price, edges=bucket_edges)
    per_bucket: dict[str, float] = {}
    for label in set(bucket_labels):
        if label is None:
            continue
        mask = np.array([b == label for b in bucket_labels])
        if mask.sum() < 30:
            continue  # too few rows for a reliable per-bucket q
        per_bucket[label] = _conformal_q_from_scores(scores[mask])

    # Per-row uncertainty model (option C). Split the cal-set in half:
    # first half trains ``score = f(features)``, second half calibrates
    # the global q_scale so empirical 80% coverage holds. Falls back to
    # None if either half is too small — caller then keeps bucket-q.
    uncertainty_bundle: tuple[lgb.LGBMRegressor, float] | None = None
    n_cal = len(X_cal)
    if n_cal >= 50:
        half = n_cal // 2
        cal_df = cal.reset_index(drop=True)
        uncertainty_bundle = _train_uncertainty_model(
            cal_df.iloc[:half], X_cal[:half], scores[:half],
            cal_df.iloc[half:], X_cal[half:], scores[half:],
            cat_indices,
        )

    return global_q, per_bucket, bucket_edges, uncertainty_bundle


# Default bucket boundaries used when no decile edges are stored in the
# bundle (e.g. a tiny synthetic test set, or a v5 bundle predating decile
# persistence). 5 fixed price ranges, kept in sync with
# src.analytics.model_eval._PRICE_BUCKETS for the diagnostic side.
_DEFAULT_BUCKET_EDGES: list[tuple[float, float, str]] = [
    (0, 3000, "<€3k"),
    (3000, 7000, "€3–7k"),
    (7000, 15000, "€7–15k"),
    (15000, 30000, "€15–30k"),
    (30000, float("inf"), "€30k+"),
]
# How many dynamic deciles to compute when training data is large enough.
# 10 is fine-grained enough to close the within-bucket coverage gap shown by
# the reliability curve, while keeping ~480 samples per bin on the local DB
# (enough for a stable 80th-percentile q estimate).
_DECILE_BUCKETS = 10
# Below this row count we fall back to the 5-bucket default — a fixed edge
# set is more reliable than 10 deciles each holding 5–20 samples.
_MIN_ROWS_FOR_DECILES = 200


def _compute_decile_edges(
    predicted_prices: np.ndarray,
    n_bins: int = _DECILE_BUCKETS,
) -> list[tuple[float, float, str]]:
    """Build dynamic bucket edges from the empirical distribution of predicted
    prices on the calibration set. Edges are emitted as (low, high, label)
    triples matching the static-edge format so downstream lookup logic stays
    identical.

    Falls back to the static 5-bucket scheme when there are too few rows for
    a stable decile estimate, or when many predictions tie at the same value
    (which would collapse multiple deciles to identical edges).
    """
    valid = predicted_prices[~np.isnan(predicted_prices)]
    if len(valid) < _MIN_ROWS_FOR_DECILES:
        return list(_DEFAULT_BUCKET_EDGES)

    # n_bins-1 internal edges → n_bins buckets. Skip the first/last quantiles
    # because we want bucket extents to span (-inf, +inf) on the ends.
    quantiles = np.linspace(1.0 / n_bins, 1.0 - 1.0 / n_bins, n_bins - 1)
    edges = list(np.quantile(valid, quantiles))

    # Collapse consecutive identical edges (happens when a price is hugely
    # over-represented, e.g. dealers listing many cars at €9999). If we lose
    # too many bins this way, fall back to the static scheme.
    deduped: list[float] = []
    for e in edges:
        if not deduped or e > deduped[-1]:
            deduped.append(float(e))
    if len(deduped) < n_bins - 2:
        return list(_DEFAULT_BUCKET_EDGES)

    bounds = [-float("inf"), *deduped, float("inf")]
    out: list[tuple[float, float, str]] = []
    for i in range(len(bounds) - 1):
        low, high = bounds[i], bounds[i + 1]
        # Human-readable label using the actual € range. For the leftmost
        # bin we render "<€X" and for the rightmost ">=€Y" so users
        # reading metrics don't see "-inf" garbage.
        if low == -float("inf"):
            label = f"<€{high:,.0f}"
        elif high == float("inf"):
            label = f"≥€{low:,.0f}"
        else:
            label = f"€{low:,.0f}–{high:,.0f}"
        out.append((low, high, label))
    return out


def _bucketize_price(
    values: np.ndarray,
    edges: list[tuple[float, float, str]] | None = None,
) -> list[str | None]:
    """Map an array of prices to bucket labels for class-conditional CQR.
    Uses *edges* when supplied (e.g. dynamic deciles persisted in the model
    bundle), or the static 5-bucket fallback otherwise."""
    bucket_edges = edges if edges is not None else _DEFAULT_BUCKET_EDGES
    out: list[str | None] = []
    for v in values:
        label: str | None = None
        if not np.isnan(v):
            for low, high, name in bucket_edges:
                if low <= v < high:
                    label = name
                    break
        out.append(label)
    return out


def _per_row_conformal_q(
    predicted_price: np.ndarray,
    global_q: float,
    per_bucket_q: dict[str, float],
    edges: list[tuple[float, float, str]] | None = None,
) -> np.ndarray:
    """Look up the per-row conformal_q based on the predicted-price bucket.

    Rows whose bucket has no calibrated q (small bucket, or predicted price
    out of range) fall back to the global q. Pass *edges* to use dynamic
    decile boundaries persisted in the model bundle; without it the static
    5-bucket scheme is used.
    """
    labels = _bucketize_price(predicted_price, edges=edges)
    out = np.full(len(predicted_price), float(global_q))
    for i, label in enumerate(labels):
        if label is not None and label in per_bucket_q:
            out[i] = per_bucket_q[label]
    return out


def _cv_metrics(
    df: pd.DataFrame,
    n_splits: int = 5,
) -> tuple[
    dict, dict[str, int], np.ndarray, IsotonicRegression | None,
    tuple[lgb.LGBMRegressor, float] | None,
]:
    """K-fold CV for all three quantile models with per-fold categorical
    encoding (no leakage) and early stopping to tune n_estimators.

    Models are trained in log1p(price) space (heavy-tailed targets bias raw-EUR
    pinball toward expensive listings; log-target evens the influence and
    typically improves MAPE on the cheap end). User-facing metrics (MAE, MAPE,
    R², pinball) are reported on the price scale by back-transforming OOF
    predictions with expm1.

    Returns:
      metrics: dict of CV metrics for the dashboard
      suggested: per-quantile early-stop iteration counts (log-pinball converges
                 at different rates for α=0.1/0.5/0.9, so each tail tunes its own)
      oof_band: (3, n) array of [low, median, high] OOF predictions in price
                space — already calibrated with conformal_q, sorted to repair
                crossings, and clamped at 0. Indexed by df row position.
    """
    raw_price = df["price_eur"].values.astype(float)
    # Apply the same sold-listing target discount + sample-weight haircut
    # the final fit uses, so CV metrics reflect the actual training
    # objective. Active rows pass through unchanged.
    sold_mult, sold_w = _build_sold_target_adjustment(df)
    y_all = raw_price * sold_mult
    # Train on log1p — clamp at 0 first so log1p never sees a negative input
    # (filter step normally already enforces this, but be defensive)
    y_log = np.log1p(np.maximum(y_all, 0))
    sample_weight_all = _compute_sample_weights(y_all) * sold_w
    n_splits = min(n_splits, max(2, len(df) // 20))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cat_indices = [_ALL_FEATURES.index(c) for c in CATEGORICAL_FEATURES]

    # OOF in log space (raw model output, before back-transform)
    oof_log = {name: np.full(len(y_all), np.nan) for name in _QUANTILES}
    best_iters: dict[str, list[int]] = {name: [] for name in _QUANTILES}

    for train_idx, val_idx in kf.split(df):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        # Encode categoricals on train only — no leakage into the val fold.
        X_train, cat_maps_fold = _prepare_X(train_df)
        X_val, _ = _prepare_X(val_df, cat_maps_fold)
        y_train_log = y_log[train_idx]
        y_val_log = y_log[val_idx]
        sw_train = sample_weight_all[train_idx]
        sw_val = sample_weight_all[val_idx]

        for name, alpha in _QUANTILES.items():
            model = _model_for_quantile(
                name, alpha, _LGB_PARAMS["n_estimators"],
            )
            model.fit(
                X_train, y_train_log,
                sample_weight=sw_train,
                eval_set=[(X_val, y_val_log)],
                eval_sample_weight=[sw_val],
                categorical_feature=cat_indices,
                callbacks=[lgb.early_stopping(_EARLY_STOPPING_ROUNDS, verbose=False)],
            )
            oof_log[name][val_idx] = model.predict(X_val)
            best_iters[name].append(
                int(model.best_iteration_ or _LGB_PARAMS["n_estimators"])
            )

    # Per-quantile early-stop suggestion (median across folds). Tails converge
    # at different rates than median, so each gets its own count. Compute now
    # because the time-aware CQR helper needs them.
    suggested: dict[str, int] = {}
    for name in _QUANTILES:
        iters = best_iters[name]
        median_iters = int(np.median(iters)) if iters else _LGB_PARAMS["n_estimators"]
        suggested[name] = max(median_iters, _MIN_N_ESTIMATORS)

    # Random-KFold CQR — score = max(log_low − log_y, log_y − log_high). The
    # widening factor stays in log space and is applied to log predictions at
    # predict time, then expm1 back. This is asymmetric in price space (which
    # is what we want for heavy-tailed targets: a € band that grows with
    # price). Kept for diagnostic comparison vs the time-aware value below.
    scores = np.maximum(oof_log["low"] - y_log, y_log - oof_log["high"])
    conformal_q_log_random = _conformal_q_from_scores(scores)

    # Time-aware CQR — re-train on oldest 80% by first_seen_at, score the
    # newest 20%. Honest answer to "how wide must the band be to cover 80%
    # of *tomorrow's* listings". Returns (global_q, per_bucket_q,
    # bucket_edges) where bucket_edges are dynamic deciles of cal-set
    # predicted prices. Falls back to random + static buckets when the
    # dataset has no first_seen_at column or is too small to split.
    time_result = _time_aware_conformal_q(df, suggested)
    per_bucket_q: dict[str, float] = {}
    bucket_edges: list[tuple[float, float, str]] = list(_DEFAULT_BUCKET_EDGES)
    uncertainty_bundle: tuple[lgb.LGBMRegressor, float] | None = None
    if time_result is not None:
        (
            conformal_q_log_time, per_bucket_q, bucket_edges,
            uncertainty_bundle,
        ) = time_result
        conformal_q_log = conformal_q_log_time
    else:
        conformal_q_log_time = None
        conformal_q_log = conformal_q_log_random

    # Isotonic post-calibration of the median quantile. Fit
    # ``predicted → actual`` on out-of-fold pairs to remove systematic shifts
    # in the global pred-vs-actual relationship (locally we observed +€1.3k
    # under-prediction in the €30k+ tier). Only the median is calibrated this
    # way — for low/high the isotonic target (=actual) is the wrong reference
    # (those predict the 10th and 90th percentile, not the actual price).
    # Caveat: 1D isotonic cannot fix subgroup mispricing where the model
    # mis-reads features (e.g. the −27% bias on ``<€3k`` cars whose actual
    # price is depressed by undetected damage/salvage flags). That needs
    # feature-level work, not post-hoc calibration.
    median_oof_price_raw = np.maximum(np.expm1(oof_log["median"]), 0)
    median_calibrator = IsotonicRegression(
        out_of_bounds="clip", increasing=True,
    )
    median_calibrator.fit(median_oof_price_raw, y_all)
    calibrated_median_price = median_calibrator.predict(median_oof_price_raw)

    # Back-transform OOF band edges to price scale and assemble the band
    # around the isotonic-calibrated median. Per-row q from class-conditional
    # CQR (each row uses its bucket's q where available, falling back to the
    # global q). We can't naive-sort all three because isotonic can move the
    # median *outside* [raw_low, raw_high] (e.g. for the cheap-segment
    # over-prediction case where the model says €4500 and isotonic
    # compresses it to €1500). A blind sort would then put raw_low into the
    # median slot and the calibrated value into low — silently dropping the
    # calibration. Instead:
    #   1. Repair true low/high crossing (rare independent-quantile artifact)
    #   2. Bracket the band around the calibrated median by min/max
    per_row_q = _per_row_conformal_q(
        calibrated_median_price, conformal_q_log, per_bucket_q,
        edges=bucket_edges,
    )
    raw_low_price = np.expm1(oof_log["low"] - per_row_q)
    raw_high_price = np.expm1(oof_log["high"] + per_row_q)
    fixed_low = np.minimum(raw_low_price, raw_high_price)
    fixed_high = np.maximum(raw_low_price, raw_high_price)
    band_low = np.minimum(fixed_low, calibrated_median_price)
    band_high = np.maximum(fixed_high, calibrated_median_price)
    oof_band = np.maximum(
        np.stack([band_low, calibrated_median_price, band_high]),
        0,
    )

    # User-facing metrics on price scale (back-transformed + calibrated)
    median_oof = oof_band[1]
    low_oof_calibrated = oof_band[0]
    high_oof_calibrated = oof_band[2]

    mae = float(np.mean(np.abs(y_all - median_oof)))
    mape = float(np.mean(np.abs((y_all - median_oof) / np.maximum(y_all, 1))) * 100)
    ss_res = float(np.sum((y_all - median_oof) ** 2))
    ss_tot = float(np.sum((y_all - np.mean(y_all)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Pinball on price scale uses the raw (uncalibrated) OOF quantile
    # predictions — that's the thing the model directly optimizes (after
    # expm1 the relative ordering is preserved for monotonic transforms).
    raw_low = np.maximum(np.expm1(oof_log["low"]), 0)
    raw_high = np.maximum(np.expm1(oof_log["high"]), 0)
    pinball = {
        "low": _pinball_loss(y_all, raw_low, _QUANTILES["low"]),
        "median": _pinball_loss(y_all, median_oof, _QUANTILES["median"]),
        "high": _pinball_loss(y_all, raw_high, _QUANTILES["high"]),
    }

    # Coverage: invariant under monotonic transforms, so log/price agree.
    coverage_raw = float(np.mean((y_all >= raw_low) & (y_all <= raw_high)))
    coverage_calibrated = float(np.mean(
        (y_all >= low_oof_calibrated) & (y_all <= high_oof_calibrated)
    ))

    metrics = {
        "mae": round(mae, 0),
        "mape": round(mape, 1),
        "r2": round(r2, 3),
        "pinball_low": round(pinball["low"], 1),
        "pinball_median": round(pinball["median"], 1),
        "pinball_high": round(pinball["high"], 1),
        "coverage_80": round(coverage_raw, 3),
        "coverage_80_calibrated": round(coverage_calibrated, 3),
        # conformal_q is in LOG space (the band-widening exponent). Apply it
        # to log predictions in predict_prices, then expm1 to price scale.
        # The active value is time-aware when first_seen_at is available,
        # otherwise random-KFold (older fallback path).
        "conformal_q": round(conformal_q_log, 4),
        "conformal_q_random": round(conformal_q_log_random, 4),
        "conformal_q_time": (
            round(conformal_q_log_time, 4)
            if conformal_q_log_time is not None else None
        ),
        "conformal_q_source": "time" if conformal_q_log_time is not None else "random",
        # Class-conditional q — one entry per price bucket where the cal slice
        # had ≥30 samples. predict_prices looks up each row's bucket and
        # widens the band by that q (vs the marginal q above). Buckets are
        # dynamic deciles of the cal-set predicted prices, persisted as
        # ``conformal_q_bucket_edges`` so predict_prices uses the same
        # boundaries at inference.
        "conformal_q_per_bucket": {
            k: round(v, 4) for k, v in per_bucket_q.items()
        },
        "conformal_q_bucket_edges": [
            (low, high, label) for low, high, label in bucket_edges
        ],
        # Approximate ± multiplicative band widening for human display:
        # a row's price band stretches by roughly (exp(q) − 1) × 100 % on
        # each side relative to the raw model band.
        "conformal_q_pct": round(float(np.expm1(conformal_q_log) * 100), 1),
        # Back-compat single number for the dashboard's "Trees (early-stop)"
        # KPI; the actual fit uses the per-quantile dict below.
        "best_n_estimators": suggested["median"],
        "best_n_estimators_per_q": suggested,
        "n_samples": int(len(y_all)),
        "cv_folds": n_splits,
        "log_target": True,
        "median_calibrated": True,
        "sample_weighted": True,
        "monotone_constraints": True,
        "n_features": len(_ALL_FEATURES),
    }
    return metrics, suggested, oof_band, median_calibrator, uncertainty_bundle


def train_price_model(
    listings_df: pd.DataFrame,
    min_samples: int = 50,
) -> tuple[
    dict[str, lgb.LGBMRegressor],
    dict[str, dict[str, int]],
    dict,
    dict[str, tuple[float, float, float]],
    IsotonicRegression | None,
    tuple[lgb.LGBMRegressor, float] | None,
] | None:
    """Train quantile regression models: median, low (10th), high (90th).

    Returns (models, category_maps, metrics, oof_preds, median_calibrator,
    uncertainty_bundle) or None if insufficient data.

    - ``oof_preds``: dict olx_id → (low, median, high) of cross-validated,
      calibrated, crossing-repaired predictions in price space.
    - ``median_calibrator``: IsotonicRegression for median post-calibration
      on new rows; OOF preds already have it baked in.
    - ``uncertainty_bundle``: ``(LGBMRegressor, q_scale)`` for per-row
      CQR widening, or ``None`` if the time-aware split was too small
      to fit. ``predict_prices`` prefers this over the per-bucket q
      lookup when present.
    """
    needed = {"price_eur", "year", "mileage_km"}
    if not needed.issubset(listings_df.columns):
        return None

    df = listings_df[
        listings_df["price_eur"].notna()
        & listings_df["year"].notna()
        & listings_df["mileage_km"].notna()
    ].copy()

    df, filter_stats = _filter_training_data(df)

    if len(df) < min_samples:
        return None

    raw_price = df["price_eur"].values.astype(float)
    sold_mult, sold_w = _build_sold_target_adjustment(df)
    y_price = raw_price * sold_mult
    y_log = np.log1p(np.maximum(y_price, 0))
    sample_weight = _compute_sample_weights(y_price) * sold_w

    # CV with per-fold cat encoding + per-quantile early stopping → tuned
    # n_estimators dict + OOF predictions in price space + isotonic calibrator
    # for the median quantile.
    (
        metrics, best_iters_per_q, oof_band, median_calibrator,
        uncertainty_bundle,
    ) = _cv_metrics(df)
    metrics["filter_stats"] = filter_stats
    n_sold = int((sold_w != 1.0).sum())
    n_dropped_sold = int((sold_w == 0.0).sum())
    metrics["sold_inclusion"] = {
        "total_rows": len(df),
        "sold_rows_used": n_sold - n_dropped_sold,
        "sold_rows_dropped_bad_dates": n_dropped_sold,
        "active_rows": len(df) - n_sold,
    }

    # Final models: fit on full filtered data with CV-tuned per-quantile iters
    X_arr, cat_maps = _prepare_X(df)
    cat_indices = [_ALL_FEATURES.index(c) for c in CATEGORICAL_FEATURES]

    models = {}
    for name, quantile in _QUANTILES.items():
        model = _model_for_quantile(name, quantile, best_iters_per_q[name])
        model.fit(
            X_arr, y_log,
            sample_weight=sample_weight,
            categorical_feature=cat_indices,
        )
        models[name] = model

    # Build OOF map keyed by olx_id (the stable join key the dashboard uses).
    # Listings without an olx_id fall back to model.predict at score time.
    oof_preds: dict[str, tuple[float, float, float]] = {}
    if "olx_id" in df.columns:
        for i, oid in enumerate(df["olx_id"].values):
            if oid is None or (isinstance(oid, float) and np.isnan(oid)):
                continue
            oof_preds[str(oid)] = (
                float(oof_band[0, i]),
                float(oof_band[1, i]),
                float(oof_band[2, i]),
            )

    return models, cat_maps, metrics, oof_preds, median_calibrator, uncertainty_bundle


def predict_prices(
    models: dict[str, lgb.LGBMRegressor],
    cat_maps: dict[str, dict[str, int]],
    listings_df: pd.DataFrame,
    conformal_q: float = 0.0,
    oof_preds: dict[str, tuple[float, float, float]] | None = None,
    median_calibrator: IsotonicRegression | None = None,
    conformal_q_per_bucket: dict[str, float] | None = None,
    conformal_q_bucket_edges: list[tuple[float, float, str]] | None = None,
    uncertainty_bundle: tuple[lgb.LGBMRegressor, float] | None = None,
) -> pd.DataFrame:
    """Predict fair price range for each listing.

    Models predict in log1p space; this function back-transforms with expm1.

    Per-row band widening (CQR) is selected in this priority order:
      1. ``uncertainty_bundle`` — a regressor predicting score directly
         from features, scaled by ``q_scale`` (option C). Replaces the
         bucket-q lookup at inference time and lets each listing carry
         its own band width. Falls through to (2) when None.
      2. ``conformal_q_per_bucket`` + ``conformal_q_bucket_edges`` —
         class-conditional CQR keyed on predicted-price decile. Falls
         through to (3) when None.
      3. ``conformal_q`` — single global widening factor, always used
         as a floor under (1) so the band never shrinks below the
         marginal-coverage value.

    *oof_preds* — dict olx_id → (low, median, high) of out-of-fold CV
    predictions, shipped with the bundle. Listings whose olx_id is in this
    dict get OOF preds instead of in-sample model.predict (prevents the
    deal-scoring loop from comparing against a memorized fair price).

    *median_calibrator* — IsotonicRegression mapping raw_predicted → actual,
    applied to the median for new rows only (OOF preds are already
    calibrated). The band assembly below brackets the calibrated median by
    min/max so isotonic-induced crossings don't leak into the [low, high].
    """
    X_arr, _ = _prepare_X(listings_df, cat_maps)

    # Model output is in log1p(price) space.
    log_median = models["median"].predict(X_arr)
    raw_median = np.expm1(log_median)

    # Apply isotonic calibration to the median for new rows. OOF preds (set
    # below) skip this since they were already calibrated at training time.
    # We track whether calibration ran so the band-assembly step below can
    # bracket the median by min/max instead of naive-sorting (which would
    # silently swap a calibrated median out of position when isotonic moves
    # it outside the raw [low, high] range).
    median_was_calibrated = median_calibrator is not None
    if median_was_calibrated:
        median = median_calibrator.predict(np.maximum(raw_median, 0))
    else:
        median = raw_median

    # Per-row CQR widening — priority order documented in the docstring:
    #   1. uncertainty_bundle (option C, per-row from features)
    #   2. conformal_q_per_bucket (class-conditional, by price decile)
    #   3. conformal_q (single global value)
    if uncertainty_bundle is not None:
        unc_model, q_scale = uncertainty_bundle
        per_row_q = _per_row_uncertainty_q(
            listings_df, X_arr, unc_model, q_scale, floor_q=conformal_q,
        )
    elif conformal_q_per_bucket:
        per_row_q = _per_row_conformal_q(
            median, conformal_q, conformal_q_per_bucket,
            edges=conformal_q_bucket_edges,
        )
    else:
        per_row_q = np.full(len(median), float(conformal_q))

    log_low = models["low"].predict(X_arr) - per_row_q
    log_high = models["high"].predict(X_arr) + per_row_q

    low = np.expm1(log_low)
    high = np.expm1(log_high)

    # Override with OOF predictions for listings the model was trained on.
    if oof_preds and "olx_id" in listings_df.columns:
        olx_ids = listings_df["olx_id"].values
        for i, oid in enumerate(olx_ids):
            if oid is None or (isinstance(oid, float) and np.isnan(oid)):
                continue
            entry = oof_preds.get(str(oid))
            if entry is None:
                continue
            low[i], median[i], high[i] = entry

    # Repair quantile crossing without losing the calibrated median.
    # Two crossing sources: (1) independent quantile fits can produce raw
    # low > raw high; (2) isotonic calibration can move the median outside
    # [raw_low, raw_high]. A blind sort fixes (1) but breaks (2) — it would
    # bury the calibrated median in the low or high slot. So we fix the band
    # edges first, then bracket the median by min/max.
    if median_was_calibrated:
        fixed_low = np.minimum(low, high)
        fixed_high = np.maximum(low, high)
        low = np.minimum(fixed_low, median)
        high = np.maximum(fixed_high, median)
    else:
        # Pre-isotonic behaviour — naive sort is fine when median sits
        # naturally inside [low, high].
        stacked = np.sort(np.stack([low, median, high]), axis=0)
        low, median, high = stacked[0], stacked[1], stacked[2]

    return pd.DataFrame({
        "predicted_price": np.maximum(median, 0),
        "fair_price_low": np.maximum(low, 0),
        "fair_price_high": np.maximum(high, 0),
    }, index=listings_df.index).round(0)


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def save_model(
    models: dict[str, lgb.LGBMRegressor],
    cat_maps: dict[str, dict[str, int]],
    metrics: dict,
    oof_preds: dict[str, tuple[float, float, float]] | None = None,
    median_calibrator: IsotonicRegression | None = None,
    uncertainty_bundle: tuple[lgb.LGBMRegressor, float] | None = None,
) -> None:
    """Save trained model bundle to disk and append metrics to history."""
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "schema_version": _SCHEMA_VERSION,
        "feature_names": list(_ALL_FEATURES),
        "models": models,
        "cat_maps": cat_maps,
        "metrics": metrics,
        "oof_preds": oof_preds or {},
        "median_calibrator": median_calibrator,
        "uncertainty_bundle": uncertainty_bundle,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    joblib.dump(bundle, _MODEL_PATH)
    _append_metrics(metrics)


def load_model(
    max_age_hours: float = _MODEL_MAX_AGE_HOURS,
) -> tuple[
    dict, dict, dict, dict, IsotonicRegression | None,
    tuple[lgb.LGBMRegressor, float] | None,
] | None:
    """Load saved model if it exists, is fresh, and matches current schema.

    Returns (models, cat_maps, metrics, oof_preds, median_calibrator) or
    None.

    Mismatches that cause rejection:
      - file age > max_age_hours
      - schema_version != _SCHEMA_VERSION (e.g. text features were added,
        isotonic calibrator was added)
      - feature_names != _ALL_FEATURES (training features were renamed/
        added/removed)

    Returning None falls through to "no fresh model" — the dashboard then logs
    a warning and skips price predictions until the next CI training run.
    """
    if not _MODEL_PATH.exists():
        return None
    age_hours = (time.time() - _MODEL_PATH.stat().st_mtime) / 3600
    if age_hours > max_age_hours:
        return None
    try:
        bundle = joblib.load(_MODEL_PATH)
        if bundle.get("schema_version") != _SCHEMA_VERSION:
            return None
        if bundle.get("feature_names") != list(_ALL_FEATURES):
            return None
        return (
            bundle["models"],
            bundle["cat_maps"],
            bundle.get("metrics", {}),
            bundle.get("oof_preds", {}),
            bundle.get("median_calibrator"),
            bundle.get("uncertainty_bundle"),
        )
    except Exception:
        return None


def _append_metrics(metrics: dict) -> None:
    """Append metrics entry to the JSON history file."""
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
    """Load full metrics history for dashboard display."""
    if not _METRICS_PATH.exists():
        return []
    try:
        return json.loads(_METRICS_PATH.read_text())
    except (json.JSONDecodeError, ValueError):
        return []


def save_importance(importance_df: pd.DataFrame) -> None:
    """Persist permutation importance next to the model.

    Computed once at training time and shipped in the data release so the
    dashboard doesn't rerun a 690-predict permutation loop on every load.
    """
    if importance_df is None or importance_df.empty:
        return
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    _IMPORTANCE_PATH.write_text(importance_df.to_json(orient="records"))


def load_importance() -> pd.DataFrame:
    """Return the shipped importance frame, or empty if missing."""
    if not _IMPORTANCE_PATH.exists():
        return pd.DataFrame()
    try:
        return pd.read_json(_IMPORTANCE_PATH, orient="records")
    except (ValueError, json.JSONDecodeError):
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Feature analysis
# ---------------------------------------------------------------------------

def compute_feature_completeness(listings_df: pd.DataFrame) -> pd.Series:
    """Fraction of model features that are non-null per listing (0–1)."""
    cols = [c for c in _ALL_FEATURES if c in listings_df.columns]
    if not cols:
        return pd.Series(0.0, index=listings_df.index, name="feature_fill_rate")
    present = listings_df[cols].notna().sum(axis=1)
    return (present / len(_ALL_FEATURES)).rename("feature_fill_rate")


def compute_data_completeness(
    feature_fill_rate: pd.Series,
    sample_sizes: pd.Series,
) -> pd.Series:
    """Combined data completeness: feature fill (60%) + sample confidence (40%)."""
    sample_conf = (sample_sizes.fillna(0) / 20).clip(upper=1.0)
    return (0.6 * feature_fill_rate + 0.4 * sample_conf).rename("data_completeness")


def compute_permutation_importance(
    models: dict[str, lgb.LGBMRegressor],
    cat_maps: dict[str, dict[str, int]],
    listings_df: pd.DataFrame,
    n_repeats: int = 10,
) -> pd.DataFrame:
    """Permutation importance for each feature across all three quantile models.

    Scorer: alpha-aligned negative pinball loss (the metric each quantile model
    is actually optimizing) instead of the sklearn default R², which makes no
    sense for the tail quantiles. Targets are in log space because the models
    predict log1p(price).

    Note: importance is computed on the same data the model was trained on,
    which mildly overstates feature contributions. A cleaner version would
    score on a held-out fold, but that costs a re-fit per quantile and the
    current ranking is already stable enough for the dashboard's purpose.
    """
    df = listings_df[
        listings_df["price_eur"].notna()
        & listings_df["year"].notna()
        & listings_df["mileage_km"].notna()
    ].copy()

    y_log = np.log1p(np.maximum(df["price_eur"].values.astype(float), 0))
    X_arr, _ = _prepare_X(df, cat_maps)

    rows = []
    for name, alpha in _QUANTILES.items():
        scorer = make_scorer(
            mean_pinball_loss, alpha=alpha, greater_is_better=False,
        )
        result = permutation_importance(
            models[name], X_arr, y_log,
            n_repeats=n_repeats, random_state=42, n_jobs=-1,
            scoring=scorer,
        )
        rows.append(pd.Series(
            result.importances_mean, index=_ALL_FEATURES, name=f"{name}_importance",
        ))

    imp = pd.concat(rows, axis=1).reset_index()
    imp.columns = ["feature", "low_importance", "median_importance", "high_importance"]
    return imp.sort_values("median_importance", ascending=False).reset_index(drop=True)
