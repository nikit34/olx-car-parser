"""Joint, time-aware tuning of the price-model LGB hyperparameters.

Why: the 2026-06-08 watchlist scan (scripts/sweep_constant.py --all) flipped
num_leaves / max_depth / min_child_samples / reg_lambda all to REAL → the model
is under-capacity for the grown corpus. Per-knob bests DON'T compose (capacity↑
wants reg↑ to control variance), so they must be tuned JOINTLY.

Honest protocol (no peeking at the test):
  1. sort sold rows by deactivated_at; oldest 70% = TUNE, newest 30% = HOLDOUT.
  2. search: random over 6 LGB params; objective = time-aware 3-fold CV MAPE on TUNE.
  3. VERDICT: refit current-params vs best-params on ALL of TUNE, evaluate on the
     untouched HOLDOUT (MAPE + pinball + coverage, bootstrap CI). Only the holdout
     number is trustworthy — the CV objective is what we optimized against.

Reuses prod feature prep (price_model._prepare_X / _fit_platform_encoding /
_monotone_constraints) so a tuned param set drops straight into _LGB_PARAMS.
Writes the recommendation to data/price_lgb_tuned.json for HUMAN REVIEW — does
NOT modify _LGB_PARAMS or retrain. No optuna dependency (random search).

Usage:  python -m scripts.tune_lgb_params --trials 40
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.analytics import price_model as pm

_CACHE = Path("/tmp/olx-release/listings.parquet")
_OUT = Path(__file__).resolve().parent.parent / "data" / "price_lgb_tuned.json"
_TUNABLE = ["num_leaves", "max_depth", "min_child_samples",
            "learning_rate", "reg_lambda", "n_estimators"]


def _norm_fuel(s: str) -> str:
    s = str(s).lower()
    if "plug" in s: return "PHEV"
    if "íbrid" in s or "ibrid" in s: return "Hybrid"
    if "elétr" in s or "eléctr" in s or "electr" in s: return "EV"
    if "diesel" in s or "asóleo" in s or "asoleo" in s: return "Diesel"
    if "asolina" in s: return "Petrol"
    return "Other"


def load_sold(data_path: str | None) -> pd.DataFrame:
    path = Path(data_path) if data_path else _CACHE
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        repo = subprocess.run(["gh", "repo", "view", "--json", "nameWithOwner",
                               "--jq", ".nameWithOwner"], capture_output=True, text=True,
                              check=True).stdout.strip()
        subprocess.run(["gh", "release", "download", "latest-data", "--repo", repo,
                        "--pattern", "listings.parquet", "--dir", str(path.parent),
                        "--clobber"], check=True)
    df = pd.read_parquet(path)
    df = df[df["deactivation_reason"].astype(str).str.lower() == "sold"].copy()
    df = df.dropna(subset=["price_eur", "year", "mileage_km"])
    df["age"] = (2026 - df["year"]).clip(lower=1)
    df = df[df["price_eur"].between(800, 150000)
            & df["mileage_km"].between(1000, 500000) & (df["age"] < 25)]
    df["fuel_norm"] = df["fuel_type"].map(_norm_fuel)
    df["dt"] = pd.to_datetime(df["deactivated_at"], errors="coerce")
    return df.dropna(subset=["dt"]).sort_values("dt").reset_index(drop=True)


def _make(name, alpha, params):
    p = dict(params, random_state=42, verbose=-1, n_jobs=-1)
    if name == "median":
        return lgb.LGBMRegressor(objective="regression",
                                 monotone_constraints=pm._monotone_constraints(),
                                 monotone_constraints_method="advanced", **p)
    return lgb.LGBMRegressor(objective="quantile", alpha=alpha, **p)


_CAT_IDX = None
def _cat_idx():
    global _CAT_IDX
    if _CAT_IDX is None:
        _CAT_IDX = [pm._ALL_FEATURES.index(c) for c in pm.CATEGORICAL_FEATURES]
    return _CAT_IDX


def _oof(df, y, folds, params, quants):
    """Leakage-safe OOF preds (platform map fit per fold-train)."""
    out = {q: np.full(len(df), np.nan) for q in quants}
    tested = np.zeros(len(df), bool)
    for tr, te in folds:
        tr_df, te_df = df.iloc[tr], df.iloc[te]
        plat = pm._fit_platform_encoding(tr_df, y[tr])
        x_tr, cm = pm._prepare_X(tr_df, plat_enc=plat)
        x_te, _ = pm._prepare_X(te_df, cm, plat_enc=plat)
        for name, alpha in quants.items():
            m = _make(name, alpha, params)
            m.fit(x_tr, y[tr], categorical_feature=_cat_idx())
            out[name][te] = m.predict(x_te)
        tested[te] = True
    return out, tested


def _fit_predict(tune_df, y_tune, hold_df, params, quants):
    """Fit on ALL of TUNE, predict HOLDOUT (the untouched test)."""
    plat = pm._fit_platform_encoding(tune_df, y_tune)
    x_tr, cm = pm._prepare_X(tune_df, plat_enc=plat)
    x_te, _ = pm._prepare_X(hold_df, cm, plat_enc=plat)
    out = {}
    for name, alpha in quants.items():
        m = _make(name, alpha, params)
        m.fit(x_tr, y_tune, categorical_feature=_cat_idx())
        out[name] = m.predict(x_te)
    return out


def _mape(oof_med, price, mask):
    p = np.expm1(oof_med)
    return float(np.mean(np.abs(p[mask] - price[mask]) / price[mask]) * 100)


def _pinball(oof_q, price, alpha):
    p = np.expm1(oof_q); d = price - p
    return float(np.mean(np.maximum(alpha * d, (alpha - 1) * d)))


def _sample(rng):
    return {
        "num_leaves": int(rng.integers(15, 201)),
        "max_depth": int(rng.integers(4, 13)),
        "min_child_samples": int(rng.integers(3, 61)),
        "learning_rate": float(np.exp(rng.uniform(np.log(0.02), np.log(0.12)))),
        "reg_lambda": float(np.exp(rng.uniform(np.log(0.1), np.log(12.0)))),
        "n_estimators": int(rng.integers(250, 901)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--top", type=int, default=6, help="how many top-CV configs to confirm on the holdout")
    ap.add_argument("--data", default=None)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    df = load_sold(args.data)
    cut = int(len(df) * 0.70)
    tune, hold = df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True)
    y_tune = np.log1p(np.maximum(tune["price_eur"].values.astype(float), 0))
    price_h = hold["price_eur"].values.astype(float)
    fuel_h = hold["fuel_norm"].values
    print(f"{len(df)} sold rows → TUNE {len(tune)} (oldest 70%) / HOLDOUT {len(hold)} (newest 30%)")
    print(f"TUNE dt: {tune['dt'].min().date()}..{tune['dt'].max().date()}  "
          f"HOLDOUT dt: {hold['dt'].min().date()}..{hold['dt'].max().date()}\n")

    tune_folds = list(TimeSeriesSplit(n_splits=3).split(np.arange(len(tune))))
    MED = {"median": 0.5}
    ALL = pm._QUANTILES

    current = {k: pm._LGB_PARAMS[k] for k in _TUNABLE if k in pm._LGB_PARAMS}
    current["n_estimators"] = 400   # match the sweep harness baseline (prod early-stops near here)

    def cv_mape(params):
        oof, tested = _oof(tune, y_tune, tune_folds, params, MED)
        return _mape(oof["median"], tune["price_eur"].values.astype(float), tested)

    t0 = time.time()
    base_cv = cv_mape(current)
    print(f"current params CV-MAPE (tune, time-aware) = {base_cv:.3f}\ncurrent = {current}\n")
    print(f"random search, {args.trials} trials (seed {args.seed}) …")
    trials_log: list[tuple[float, dict]] = []
    best_cv = base_cv
    for i in range(args.trials):
        p = _sample(rng)
        cv = cv_mape(p)
        trials_log.append((cv, p))
        flag = ""
        if cv < best_cv:
            best_cv = cv; flag = "  <-- best"
        print(f"  [{i+1:>3}/{args.trials}] cv-mape={cv:.3f}  nl={p['num_leaves']:>3} "
              f"md={p['max_depth']:>2} mcs={p['min_child_samples']:>2} "
              f"lr={p['learning_rate']:.3f} l2={p['reg_lambda']:.2f} ne={p['n_estimators']:>3}{flag}")
    trials_log.sort(key=lambda t: t[0])
    print(f"\nsearch done in {time.time()-t0:.0f}s. best CV-MAPE={trials_log[0][0]:.3f} "
          f"(current {base_cv:.3f}, Δ={trials_log[0][0]-base_cv:+.3f})")

    # ---- CONFIRMATION: do the TOP-K CV configs ALL survive the untouched HOLDOUT? ----
    # One lucky config clearing the holdout is weak; the whole top-CV region clearing
    # it is robust. Each top-K config is evaluated on the holdout it never saw.
    full = np.ones(len(hold), bool)
    cur_pred = _fit_predict(tune, y_tune, hold, current, ALL)
    cur_m = _mape(cur_pred["median"], price_h, full)
    cb = np.expm1(cur_pred["median"])
    idx = np.arange(len(hold))

    def holdout_delta_ci(pred):
        bb = np.expm1(pred["median"]); m = _mape(pred["median"], price_h, full)
        boot = [(np.mean(np.abs(bb[s]-price_h[s])/price_h[s])
                 - np.mean(np.abs(cb[s]-price_h[s])/price_h[s])) * 100
                for s in (rng.choice(idx, len(idx), replace=True) for _ in range(args.n_boot))]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        v = "SHIP ✓" if hi < 0 else "worse ✗" if lo > 0 else "wash"
        return m, m - cur_m, lo, hi, v

    print(f"\n=== HOLDOUT confirmation: top-{args.top} CV configs vs current "
          f"(current holdout MAPE={cur_m:.2f}) ===")
    print(f"{'rank':>4} {'cvMAPE':>7} {'hMAPE':>6} {'Δ':>6} {'95% CI':>16}  verdict   params")
    topk_rows = []
    for rank, (cv, p) in enumerate(trials_log[:args.top], 1):
        pred = _fit_predict(tune, y_tune, hold, p, ALL)
        m, d, lo, hi, v = holdout_delta_ci(pred)
        topk_rows.append({"cv": round(cv, 3), "holdout_mape": round(m, 2),
                          "delta": round(d, 2), "ci": [round(lo, 2), round(hi, 2)],
                          "verdict": v, "params": p})
        ps = f"nl{p['num_leaves']} md{p['max_depth']} mcs{p['min_child_samples']} lr{p['learning_rate']:.3f} l2{p['reg_lambda']:.2f} ne{p['n_estimators']}"
        print(f"{rank:>4} {cv:>7.3f} {m:>6.2f} {d:>+6.2f} [{lo:+.2f},{hi:+.2f}]  {v:<8} {ps}")

    n_ship = sum(r["verdict"] == "SHIP ✓" for r in topk_rows)
    n_worse = sum(r["verdict"] == "worse ✗" for r in topk_rows)
    robust = (topk_rows[0]["verdict"] == "SHIP ✓") and (n_ship >= (args.top + 1) // 2) and n_worse == 0
    summary = (f"ROBUST ✓ — {n_ship}/{args.top} top configs clear the holdout, 0 worse"
               if robust else
               f"FRAGILE — only {n_ship}/{args.top} clear holdout, {n_worse} worse; "
               f"the win may be a lucky single config")
    print(f"\nCONFIRMATION: {summary}")
    # consistency of the top region (is the winning direction stable?)
    tp = [r["params"] for r in topk_rows]
    rng_str = lambda k: f"{min(x[k] for x in tp)}–{max(x[k] for x in tp)}"
    print(f"top-{args.top} param ranges: num_leaves {rng_str('num_leaves')}, "
          f"max_depth {rng_str('max_depth')}, min_child {rng_str('min_child_samples')}, "
          f"lr {min(x['learning_rate'] for x in tp):.3f}–{max(x['learning_rate'] for x in tp):.3f}, "
          f"reg_lambda {min(x['reg_lambda'] for x in tp):.2f}–{max(x['reg_lambda'] for x in tp):.2f}")

    best = topk_rows[0]
    _OUT.write_text(json.dumps({
        "tuned_at_utc": None, "seed": args.seed, "trials": args.trials,
        "tune_rows": len(tune), "holdout_rows": len(hold),
        "current_params": current, "current_holdout_mape": round(cur_m, 2),
        "best_params": best["params"], "best_holdout_mape": best["holdout_mape"],
        "best_holdout_delta_ci": best["ci"], "confirmation": summary,
        "top_k": topk_rows,
        "note": "HUMAN REVIEW before editing _LGB_PARAMS. Ship only if ROBUST. "
                "n_estimators is early-stopped in prod — treat as ceiling hint, not pinned.",
    }, indent=2))
    print(f"\nWrote {_OUT}. Ship only if ROBUST; trust holdout, not CV.")


if __name__ == "__main__":
    main()
