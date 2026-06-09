"""EXPERIMENT (2026-06-09): does a config-similarity neighbour estimate add
anything OVER the shipped v13 dropout model — as an input feature (A) or as an
output blend with the segment median (B)?

Context: enc_plat (coarse brand|generation target-encoding) was removed in v13 —
it washed on the time-aware forward split because the GBM is already an adaptive
weighted-neighbour estimator. This re-tests two *different* shapes:

  A. INPUT FEATURE — a finer, coherent config-kNN: within the same
     (brand,model,generation,fuel) segment, a Gaussian-kernel-weighted mean of
     neighbour log-prices over standardised [year, mileage], credibility-shrunk
     toward the segment mean. Leakage-safe (OOF on FIT, from-FIT for HOLDOUT).
     Prior: likely WASH (the tree already does this), but enc_plat's key was
     broken (Renault|Mk4 pooled 11 models) so a coherent kNN is a fair re-test.

  B. OUTPUT BLEND — pred' = w·GBM + (1−w)·segment_median, a robustness anchor the
     tree CAN'T override (would have damped the Mégane phantom). Swept over fixed
     w and a confidence-weighted w = n/(n+k). Value is the over-prediction TAIL,
     not mean MAPE. Caveat: the bug that motivates it is already fixed by
     dropout, so this measures RESIDUAL benefit on top of the shipped model.

Protocol mirrors scripts/exp_feature_dropout.py: time-aware FIT/CALIB/HOLDOUT,
spec-dropout on FIT (mirror shipped), conformal q on full-feature CALIB,
bootstrap CIs on every delta vs the baseline GBM.

Usage:  python -m scripts.exp_segment_neighbor --n-boot 2000 --seed 42
"""
from __future__ import annotations

import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.analytics import price_model as pm
from scripts.tune_lgb_params import load_sold

QUANTS = {"median": 0.5, "low": 0.1, "high": 0.9}
COV_ALPHA = 0.80
PHANTOM_MULT = 1.3
SEG = ["brand", "model", "generation"]
_CAT = None


def _cat_idx():
    global _CAT
    if _CAT is None:
        _CAT = [pm._ALL_FEATURES.index(c) for c in pm.CATEGORICAL_FEATURES]
    return _CAT


def _make(name, alpha, params, n_extra=0):
    p = dict(params, random_state=42, verbose=-1, n_jobs=-1)
    if name == "median":
        mono = pm._monotone_constraints() + [0] * n_extra
        return lgb.LGBMRegressor(objective="regression", monotone_constraints=mono,
                                 monotone_constraints_method="advanced", **p)
    # quantile objective forbids monotone constraints
    return lgb.LGBMRegressor(objective="quantile", alpha=alpha, **p)


def _seg_key(df):
    return (df["brand"].astype(str) + "|" + df["model"].astype(str)
            + "|" + df["generation"].astype(str)).values


# ---- config-kNN (variant A) ------------------------------------------------
def _knn_logprice(fit_df, y_fit_log, target_df):
    """Gaussian-kernel-weighted neighbour log-price for each target row, using
    FIT rows as the neighbour pool. Neighbourhood = same brand|model|generation
    (+ fuel when present); distance over standardised [year, mileage]; shrunk
    toward the segment mean (k0=10) and the global mean when the segment is thin."""
    gmean = float(np.mean(y_fit_log))
    fk = _seg_key(fit_df); tk = _seg_key(target_df)
    yr_f = pd.to_numeric(fit_df["year"], errors="coerce").values.astype(float)
    km_f = pd.to_numeric(fit_df["mileage_km"], errors="coerce").values.astype(float)
    fuel_f = fit_df["fuel_type"].astype(str).values
    yr_t = pd.to_numeric(target_df["year"], errors="coerce").values.astype(float)
    km_t = pd.to_numeric(target_df["mileage_km"], errors="coerce").values.astype(float)
    fuel_t = target_df["fuel_type"].astype(str).values
    yr_sd = np.nanstd(yr_f) or 1.0
    km_sd = np.nanstd(km_f) or 1.0
    # index FIT rows by segment
    from collections import defaultdict
    seg_rows = defaultdict(list)
    for i, k in enumerate(fk):
        seg_rows[k].append(i)
    seg_mean = {k: float(np.mean(y_fit_log[idx])) for k, idx in seg_rows.items()}
    out = np.empty(len(target_df))
    K0 = 10.0
    for t in range(len(target_df)):
        idx = np.array(seg_rows.get(tk[t], []), dtype=int)
        base = seg_mean.get(tk[t], gmean)
        if len(idx) == 0:
            out[t] = gmean
            continue
        # prefer same-fuel neighbours when the target fuel is known
        if fuel_t[t] not in ("nan", "None", ""):
            same = idx[fuel_f[idx] == fuel_t[t]]
            if len(same) >= 3:
                idx = same
        d2 = np.zeros(len(idx))
        if not np.isnan(yr_t[t]):
            d2 += ((yr_f[idx] - yr_t[t]) / yr_sd) ** 2
        if not np.isnan(km_t[t]):
            d2 += np.nan_to_num(((km_f[idx] - km_t[t]) / km_sd) ** 2, nan=0.0)
        w = np.exp(-0.5 * d2)
        sw = w.sum()
        knn = (w * y_fit_log[idx]).sum() / sw if sw > 0 else base
        # credibility shrink toward the segment mean (effective n = sw)
        out[t] = (sw * knn + K0 * base) / (sw + K0)
    return out


def _knn_oof(fit_df, y_fit_log, n_splits=5):
    from sklearn.model_selection import KFold
    out = np.empty(len(fit_df))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    idx = np.arange(len(fit_df))
    for tr, te in kf.split(idx):
        out[te] = _knn_logprice(fit_df.iloc[tr], y_fit_log[tr], fit_df.iloc[te])
    return out


def _fit_quants(X, y, params, n_extra=0):
    out = {}
    for name, alpha in QUANTS.items():
        m = _make(name, alpha, params, n_extra)
        m.fit(X, y, categorical_feature=_cat_idx())
        out[name] = m
    return out


def _pred_log(models, X):
    lo, hi = models["low"].predict(X), models["high"].predict(X)
    return {"med": models["median"].predict(X), "lo": np.minimum(lo, hi), "hi": np.maximum(lo, hi)}


def _conformal_q(p, y_log, alpha=COV_ALPHA):
    e = np.maximum(p["lo"] - y_log, y_log - p["hi"])
    n = len(e)
    return float(np.quantile(e, min(1.0, np.ceil((n + 1) * alpha) / n)))


def _metrics_from_price(med, lo, hi, actual):
    y_log = np.log1p(actual)
    return {
        "mape": np.mean(np.abs(med - actual) / actual) * 100,
        "ratio": np.median(med / actual),
        "phantom": np.mean(med > PHANTOM_MULT * actual) * 100,
        "cover": np.mean((np.log1p(np.maximum(lo, 0)) <= y_log) & (y_log <= np.log1p(np.maximum(hi, 0)))) * 100,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-estimators", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data", default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    df = load_sold(args.data)
    h = int(len(df) * 0.70); f = int(len(df) * 0.70 * 0.80)
    fit, calib, hold = df.iloc[:f].reset_index(drop=True), df.iloc[f:h].reset_index(drop=True), df.iloc[h:].reset_index(drop=True)
    y_fit = np.log1p(np.maximum(fit["price_eur"].values.astype(float), 0))
    y_cal = np.log1p(np.maximum(calib["price_eur"].values.astype(float), 0))
    actual = hold["price_eur"].values.astype(float)
    params = {k: pm._LGB_PARAMS[k] for k in pm._LGB_PARAMS}; params["n_estimators"] = args.n_estimators
    print(f"{len(df)} rows → FIT {len(fit)} / CALIB {len(calib)} / HOLDOUT {len(hold)} (dropout {pm._SPEC_DROPOUT_FRAC})\n")

    # --- baseline dropout-aware GBM (mirror shipped v13) ---
    _, cm = pm._prepare_X(fit)
    fit_aug = pm._apply_spec_dropout(fit, pm._SPEC_DROPOUT_FRAC, np.random.default_rng(pm._SPEC_DROPOUT_SEED))
    Xf, _ = pm._prepare_X(fit_aug, cm)
    Xc, _ = pm._prepare_X(calib, cm)
    Xh, _ = pm._prepare_X(hold, cm)
    base_m = _fit_quants(Xf, y_fit, params)
    qb = _conformal_q(_pred_log(base_m, Xc), y_cal)
    pb = _pred_log(base_m, Xh)
    gbm_med = np.expm1(pb["med"]); gbm_lo = np.expm1(pb["lo"] - qb); gbm_hi = np.expm1(pb["hi"] + qb)
    base_mx = _metrics_from_price(gbm_med, gbm_lo, gbm_hi, actual)

    idx = np.arange(len(hold)); boots = [rng.choice(idx, len(idx), replace=True) for _ in range(args.n_boot)]
    def ci(a): return tuple(np.percentile(a, [2.5, 97.5]))
    def boot_delta(metric_fn, variant_vals, base_vals):
        d = np.array([metric_fn(variant_vals, b) - metric_fn(base_vals, b) for b in boots])
        return ci(d)

    print(f"=== BASELINE (v13 dropout GBM) ===")
    print(f"  MAPE={base_mx['mape']:.2f}  P50/act={base_mx['ratio']:.3f}  "
          f"phantom={base_mx['phantom']:.1f}%  cover={base_mx['cover']:.1f}%\n")

    # ---------- Variant A: config-kNN as a feature ----------
    knn_fit = _knn_oof(fit_aug, y_fit)          # OOF on the (augmented) fit rows
    knn_hold = _knn_logprice(fit, y_fit, hold)  # holdout from full FIT
    XfA = np.hstack([Xf, knn_fit.reshape(-1, 1)])
    XhA = np.hstack([Xh, knn_hold.reshape(-1, 1)])
    a_m = _fit_quants(XfA, y_fit, params, n_extra=1)
    pa = _pred_log(a_m, XhA)
    a_med = np.expm1(pa["med"])
    a_mape = lambda vals, b: np.mean(np.abs(vals[b] - actual[b]) / actual[b]) * 100
    lo_a, hi_a = boot_delta(a_mape, a_med, gbm_med)
    am = _metrics_from_price(a_med, np.expm1(pa["lo"] - qb), np.expm1(pa["hi"] + qb), actual)
    print("=== A: config-kNN feature ===")
    print(f"  MAPE={am['mape']:.2f} (Δ{am['mape']-base_mx['mape']:+.2f} [{lo_a:+.2f},{hi_a:+.2f}])  "
          f"P50/act={am['ratio']:.3f}  phantom={am['phantom']:.1f}%  cover={am['cover']:.1f}%")
    print(f"  verdict: {'REAL ✓' if hi_a<0 else 'worse ✗' if lo_a>0 else 'noise (CI∋0)'}\n")

    # ---------- Variant B: output blend with segment median ----------
    seg_med_log = fit.assign(_k=_seg_key(fit), _y=y_fit).groupby("_k")["_y"].median().to_dict()
    seg_n = pd.Series(_seg_key(fit)).value_counts().to_dict()
    hk = _seg_key(hold)
    seg_med_h = np.array([np.expm1(seg_med_log[k]) if k in seg_med_log else np.nan for k in hk])
    seg_n_h = np.array([seg_n.get(k, 0) for k in hk])
    has = ~np.isnan(seg_med_h)
    b_mape = lambda vals, b: np.mean(np.abs(vals[b] - actual[b]) / actual[b]) * 100
    print("=== B: pred' = w·GBM + (1−w)·segment_median  (rows w/ a segment median: "
          f"{has.sum()}/{len(hold)}) ===")
    print(f"  {'w':>14} {'MAPE':>7} {'Δvs base':>16} {'P50/act':>8} {'phantom%':>9} {'cover%':>7}")
    def blend_eval(w_arr, label):
        blended = gbm_med.copy()
        blended[has] = w_arr[has] * gbm_med[has] + (1 - w_arr[has]) * seg_med_h[has]
        delta = blended - gbm_med            # translate band to recentre
        lo2, hi2 = gbm_lo + delta, gbm_hi + delta
        mx = _metrics_from_price(blended, lo2, hi2, actual)
        lo_d, hi_d = boot_delta(b_mape, blended, gbm_med)
        sig = "↓" if hi_d < 0 else "↑" if lo_d > 0 else ""
        print(f"  {label:>14} {mx['mape']:>7.2f} {mx['mape']-base_mx['mape']:>+7.2f}[{lo_d:+.2f},{hi_d:+.2f}] "
              f"{mx['ratio']:>8.3f} {mx['phantom']:>9.1f} {mx['cover']:>7.1f} {sig}")
    for w in (1.0, 0.85, 0.70, 0.50):
        blend_eval(np.full(len(hold), w), f"{w:.2f}")
    for k in (5.0, 20.0):
        w_arr = seg_n_h / (seg_n_h + k)      # confidence-weighted: thin segment → shrink to median
        blend_eval(w_arr, f"n/(n+{k:.0f})")
    print("\nphantom% = pred > 1.3×actual (over-prediction tail). B's value is the tail, not mean MAPE.")


if __name__ == "__main__":
    main()
