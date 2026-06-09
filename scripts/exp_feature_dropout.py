"""EXPERIMENT (2026-06-09): should missingness-aware (spec-dropout) training
ship? Promotes scripts/proto_feature_dropout.py to a decision-grade test.

Root cause (audit 2026-06-08, see project_missing_feature_overprediction): the
price model leans ~70% on coarse baseline features; when the per-car specs
{mileage, horsepower, engine_cc, fuel_type} are absent it falls back to that
baseline and over-predicts behind a deceptively tight band (its NaN branches
are under-trained — ~97% of training rows are full-feature).

Hypothesis: randomly NaN-ing a RANDOM SUBSET of specs on a fraction of TRAIN
rows teaches (a) the median head an honest lower marginal and (b) the low/high
CQR heads to WIDEN when specs are absent — restoring coverage there — at ~no
cost on full-feature rows.

Three things the prototype lacked, added here for a ship decision:
  1. REALISTIC dropout — per touched row, drop k∈{1,2,3,4} random specs (not
     always all 4), covering the real combinatorial missing patterns.
  2. BOOTSTRAP CIs on every delta (dropout − baseline), like tune_lgb_params.
  3. CQR CHECK — a real conformal q is calibrated on a held-back CALIB slice
     (full-feature, as in prod) and we verify 80% coverage is preserved on the
     FULL holdout (must not break — project_cqr_calibration) while improving on
     the stripped holdout.

Protocol (trust the HOLDOUT, not the training objective):
  sold rows sorted by deactivated_at →
    HOLDOUT  = newest 30%   (never seen)
    TRAIN    = oldest 70%, split into
      FIT    = oldest 80% of TRAIN  (model fit)
      CALIB  = newest 20% of TRAIN  (conformal q, full-feature — time-honest)
  baseline arm: FIT as-is.  dropout arm: FIT with random spec-dropout.
  enc_plat + cat_maps fit once on FIT (dropout doesn't touch brand/generation),
  shared by both arms so the ONLY difference is the augmentation.

Usage:  python -m scripts.exp_feature_dropout --drop-frac 0.40 --n-boot 2000 --seed 42
"""
from __future__ import annotations

import argparse

import lightgbm as lgb
import numpy as np

from src.analytics import price_model as pm
from scripts.tune_lgb_params import load_sold

SPECS = ["mileage_km", "horsepower", "engine_cc", "fuel_type"]
QUANTS = {"median": 0.5, "low": 0.1, "high": 0.9}
COV_ALPHA = 0.80          # target CQR coverage
PHANTOM_MULT = 1.3        # "looks like a 30%+ deal": predicted > 1.3 × actual
_CAT = None


def _cat_idx():
    global _CAT
    if _CAT is None:
        _CAT = [pm._ALL_FEATURES.index(c) for c in pm.CATEGORICAL_FEATURES]
    return _CAT


def _make(name, alpha, params):
    p = dict(params, random_state=42, verbose=-1, n_jobs=-1)
    if name == "median":
        return lgb.LGBMRegressor(objective="regression",
                                 monotone_constraints=pm._monotone_constraints(),
                                 monotone_constraints_method="advanced", **p)
    return lgb.LGBMRegressor(objective="quantile", alpha=alpha, **p)


def _random_dropout(df, frac, rng):
    """Copy of df with a random subset of specs NaN'd on `frac` of rows.
    Per touched row: k ~ Uniform{1..4} specs dropped (covers all patterns,
    including the all-4 'fully stripped' case prod sees)."""
    d = df.copy()
    n = len(d)
    touched = rng.random(n) < frac
    idx = np.flatnonzero(touched)
    for i in idx:
        k = rng.integers(1, len(SPECS) + 1)
        for c in rng.choice(SPECS, size=k, replace=False):
            d.iat[i, d.columns.get_loc(c)] = np.nan
    return d, touched.sum()


def _strip(df, cols):
    d = df.copy()
    for c in cols:
        d[c] = np.nan
    return d


def _fit(fit_df, y, params, plat, cm):
    out = {}
    x, _ = pm._prepare_X(fit_df, cm, plat_enc=plat)
    for name, alpha in QUANTS.items():
        m = _make(name, alpha, params)
        m.fit(x, y, categorical_feature=_cat_idx())
        out[name] = m
    return out


def _pred_log(models, df, plat, cm):
    """Raw quantile predictions in LOG space (CQR widening applied later)."""
    x, _ = pm._prepare_X(df, cm, plat_enc=plat)
    lo, hi = models["low"].predict(x), models["high"].predict(x)
    return {"med": models["median"].predict(x),
            "lo": np.minimum(lo, hi), "hi": np.maximum(lo, hi)}


def _conformal_q(pred_log, y_log, alpha=COV_ALPHA):
    """Symmetric additive CQR conformity score on a calibration slice:
    q = the alpha-quantile (finite-sample-corrected) of
    E_i = max(lo_i − y_i, y_i − hi_i)."""
    e = np.maximum(pred_log["lo"] - y_log, y_log - pred_log["hi"])
    n = len(e)
    level = min(1.0, np.ceil((n + 1) * alpha) / n)
    return float(np.quantile(e, level))


def _metrics(pred_log, actual, q):
    """All evaluation metrics for one arm/regime as a dict (no refit)."""
    med = np.expm1(pred_log["med"])
    lo = np.expm1(pred_log["lo"] - q)
    hi = np.expm1(pred_log["hi"] + q)
    y_log = np.log1p(actual)
    return {
        "mape": np.mean(np.abs(med - actual) / actual) * 100,
        "ratio": np.median(med / actual),          # over-prediction (1.0 = unbiased)
        "phantom": np.mean(med > PHANTOM_MULT * actual) * 100,
        "cover": np.mean((y_log >= pred_log["lo"] - q) & (y_log <= pred_log["hi"] + q)) * 100,
        "mpiw": np.median((hi - lo) / np.maximum(med, 1)) * 100,
    }


def _metric_on(pred_log, actual, q, sel):
    sub = {k: v[sel] for k, v in pred_log.items()}
    return _metrics(sub, actual[sel], q)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drop-frac", type=float, default=0.40, help="fraction of FIT rows touched by dropout")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-estimators", type=int, default=1000, help="fixed (no early stop); equal for both arms")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data", default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    df = load_sold(args.data)
    h_cut = int(len(df) * 0.70)
    train, hold = df.iloc[:h_cut].reset_index(drop=True), df.iloc[h_cut:].reset_index(drop=True)
    f_cut = int(len(train) * 0.80)
    fit, calib = train.iloc[:f_cut].reset_index(drop=True), train.iloc[f_cut:].reset_index(drop=True)
    y_fit = np.log1p(np.maximum(fit["price_eur"].values.astype(float), 0))
    y_cal = np.log1p(np.maximum(calib["price_eur"].values.astype(float), 0))
    actual = hold["price_eur"].values.astype(float)

    params = {k: pm._LGB_PARAMS[k] for k in pm._LGB_PARAMS}
    params["n_estimators"] = args.n_estimators
    print(f"{len(df)} sold rows → FIT {len(fit)} / CALIB {len(calib)} / HOLDOUT {len(hold)}")
    print(f"dropout: {args.drop_frac:.0%} of FIT rows, k∈{{1..4}} random specs each; "
          f"n_estimators={args.n_estimators}, bootstrap={args.n_boot}\n")

    # enc_plat + cat_maps from FIT (unaffected by dropout — keys are brand|gen).
    plat = pm._fit_platform_encoding(fit, y_fit)
    _, cm = pm._prepare_X(fit, plat_enc=plat)

    base = _fit(fit, y_fit, params, plat, cm)
    fit_drop, n_touched = _random_dropout(fit, args.drop_frac, rng)
    drop = _fit(fit_drop, y_fit, params, plat, cm)
    print(f"dropout touched {n_touched}/{len(fit)} FIT rows\n")

    # conformal q calibrated on full-feature CALIB (as in prod).
    qb = _conformal_q(_pred_log(base, calib, plat, cm), y_cal)
    qd = _conformal_q(_pred_log(drop, calib, plat, cm), y_cal)
    print(f"conformal q (log-space, 80% target): baseline={qb:.4f}  dropout={qd:.4f}\n")

    # Regimes on the holdout: which specs are NaN'd before predicting.
    regimes = {
        "FULL (specs present)": [],
        "mileage only NaN": ["mileage_km"],
        "mileage+hp NaN": ["mileage_km", "horsepower"],
        "all 4 specs NaN": SPECS,
    }
    idx = np.arange(len(hold))
    boots = [rng.choice(idx, len(idx), replace=True) for _ in range(args.n_boot)]

    def ci(arr):
        lo, hi = np.percentile(arr, [2.5, 97.5])
        return lo, hi

    METS = [("ratio", "P50/act"), ("mape", "MAPE%"), ("phantom", "phantom%"),
            ("cover", "cover%"), ("mpiw", "band%")]
    for label, drop_cols in regimes.items():
        hb = _strip(hold, drop_cols) if drop_cols else hold
        pb = _pred_log(base, hb, plat, cm)
        pd_ = _pred_log(drop, hb, plat, cm)
        mb, md = _metrics(pb, actual, qb), _metrics(pd_, actual, qd)
        # bootstrap: one metric dict per arm per resample, then deltas per key.
        acc = {key: np.empty(len(boots)) for key, _ in METS}
        for j, b in enumerate(boots):
            bb, bd = _metric_on(pb, actual, qb, b), _metric_on(pd_, actual, qd, b)
            for key, _ in METS:
                acc[key][j] = bd[key] - bb[key]
        print(f"=== {label} ===")
        print(f"  {'metric':9} {'baseline':>9} {'dropout':>9} {'Δ(drop−base)':>14} {'95% CI':>18}")
        for key, name in METS:
            lo, hi = ci(acc[key])
            d = md[key] - mb[key]
            sig = "" if (lo < 0 < hi) else (" ↓" if hi < 0 else " ↑")
            print(f"  {name:9} {mb[key]:>9.2f} {md[key]:>9.2f} {d:>+14.2f} "
                  f"[{lo:+.2f}, {hi:+.2f}]{sig}")
        print()

    print("Read: FULL must show no harm (MAPE Δ≈0, cover stays ~80). Stripped regimes")
    print("should show phantom% ↓ and cover% ↑ with CIs excluding 0. ↓/↑ = CI excludes 0.")


if __name__ == "__main__":
    main()
