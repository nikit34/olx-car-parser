"""Sensitivity harness — sweep any hardcoded price-model constant and measure
its effect on out-of-fold quality, under BOTH random-KFold and a TIME-AWARE
forward split, with bootstrap CIs. Turns "why is this number 20?" debates into
a 5-minute evidence check.

Why it exists: most of the model's magic numbers (credibility K, fold guards,
category mins, …) live in the noise floor — sweeping proves it before anyone
hand-tunes. And random KFold flatters (it lets the model peek at contemporaneous
sales); the time-aware split is the honest read. This tool always shows both.

Data: pulls sold listings from the `latest-data` GitHub Release snapshot
(see the release-db skill — no local DB on the dev Mac). It reuses the prod
feature pipeline (price_model._prepare_X / _model_for_quantile /
_fit_platform_encoding) so a swept constant flows through exactly as in prod.
It is a RELATIVE-delta tool: absolute MAPE differs from prod CV (no turnover
features, fixed n_estimators), but ΔMAPE vs the current value is what matters.

Usage:
  python -m scripts.sweep_constant --const _PLAT_CRED_K --values 0.3,5,20,40,80
  python -m scripts.sweep_constant --const _LGB_PARAMS.num_leaves --values 15,31,63
  python -m scripts.sweep_constant --const _MONOTONE_BY_FEATURE.enc_plat --values 0,1
  python -m scripts.sweep_constant --const _LGB_PARAMS.learning_rate --values 0.03,0.05,0.1 --full
Flags:
  --full     also fit low/high quantiles -> report pinball + [P10,P90] coverage
  --data P   use a local listings.parquet instead of downloading the release
  --segments d,p,h,e,phev  restrict reported fuel segments (default: all)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, TimeSeriesSplit

from src.analytics import price_model as pm

_CACHE = Path("/tmp/olx-release/listings.parquet")
_N_EST = 400          # fixed (no early stopping) so every swept value is comparable
_N_BOOT = 2000
_RNG = np.random.RandomState(42)


def _norm_fuel(s: str) -> str:
    s = str(s).lower()
    if "plug" in s:
        return "PHEV"
    if "íbrid" in s or "ibrid" in s:
        return "Hybrid"
    if "elétr" in s or "eléctr" in s or "electr" in s:
        return "EV"
    if "diesel" in s or "asóleo" in s or "asoleo" in s:
        return "Diesel"
    if "asolina" in s:
        return "Petrol"
    return "Other"


def load_sold(data_path: str | None) -> pd.DataFrame:
    """Sold listings from the release snapshot (the only labelled price signal)."""
    path = Path(data_path) if data_path else _CACHE
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        repo = subprocess.run(
            ["gh", "repo", "view", "--json", "nameWithOwner", "--jq", ".nameWithOwner"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        print(f"Downloading listings.parquet from {repo}:latest-data …", file=sys.stderr)
        subprocess.run(
            ["gh", "release", "download", "latest-data", "--repo", repo,
             "--pattern", "listings.parquet", "--dir", str(path.parent), "--clobber"],
            check=True,
        )
    df = pd.read_parquet(path)
    df = df[df["deactivation_reason"].astype(str).str.lower() == "sold"].copy()
    df = df.dropna(subset=["price_eur", "year", "mileage_km"])
    df["age"] = (2026 - df["year"]).clip(lower=1)
    df = df[
        df["price_eur"].between(800, 150000)
        & df["mileage_km"].between(1000, 500000)
        & (df["age"] < 25)
    ]
    df["fuel_norm"] = df["fuel_type"].map(_norm_fuel)
    df["dt"] = pd.to_datetime(df["deactivated_at"], errors="coerce")
    df = df.dropna(subset=["dt"]).reset_index(drop=True)
    return df


def _folds(df: pd.DataFrame, mode: str):
    if mode == "random":
        return list(KFold(5, shuffle=True, random_state=42).split(df))
    order = np.argsort(df["dt"].values, kind="stable")
    return [(order[tr], order[te]) for tr, te in TimeSeriesSplit(n_splits=4).split(order)]


def evaluate(df: pd.DataFrame, folds, full: bool) -> dict:
    """OOF predictions reusing prod feature prep + model construction so the
    patched constant takes effect. Platform encoding is leakage-safe per fold."""
    y = np.log1p(np.maximum(df["price_eur"].values.astype(float), 0))
    n = len(df)
    quants = pm._QUANTILES if full else {"median": 0.5}
    oof = {q: np.full(n, np.nan) for q in quants}
    tested = np.zeros(n, bool)
    cat_idx = [pm._ALL_FEATURES.index(c) for c in pm.CATEGORICAL_FEATURES]
    for tr, te in folds:
        tr_df, te_df = df.iloc[tr], df.iloc[te]
        plat = pm._fit_platform_encoding(tr_df, y[tr])      # uses pm._PLAT_CRED_K
        x_tr, cmaps = pm._prepare_X(tr_df, plat_enc=plat)
        x_te, _ = pm._prepare_X(te_df, cmaps, plat_enc=plat)
        for name, alpha in quants.items():
            model = pm._model_for_quantile(name, alpha, _N_EST)   # uses pm._LGB_PARAMS + monotone
            model.fit(x_tr, y[tr], categorical_feature=cat_idx)
            oof[name][te] = model.predict(x_te)
        tested[te] = True
    return {"oof": oof, "tested": tested, "y": y, "price": df["price_eur"].values.astype(float)}


def _mape(res, mask):
    m = mask & res["tested"]            # TimeSeriesSplit never predicts the first block
    p = np.expm1(res["oof"]["median"]); pr = res["price"]
    return float(np.mean(np.abs(p[m] - pr[m]) / pr[m]) * 100)


def _coverage(res, mask):
    m = mask & res["tested"]
    lo = np.expm1(res["oof"]["low"]); hi = np.expm1(res["oof"]["high"]); pr = res["price"]
    return float(np.mean((pr[m] >= lo[m]) & (pr[m] <= hi[m])) * 100)


def _set_const(path: str, raw: str):
    """Patch pm.<const> (scalar) or pm.<DICT>.<key>; return (restore_fn, parsed)."""
    val: object
    for cast in (int, float):
        try:
            val = cast(raw); break
        except ValueError:
            val = raw
    if "." in path:
        name, key = path.split(".", 1)
        d = getattr(pm, name)
        old = d.get(key, KeyError)
        d[key] = val
        def restore():
            if old is KeyError:
                d.pop(key, None)
            else:
                d[key] = old
        return restore, val
    old = getattr(pm, path)
    setattr(pm, path, val)
    return (lambda: setattr(pm, path, old)), val


def _current(path: str):
    if "." in path:
        name, key = path.split(".", 1)
        return getattr(pm, name).get(key)
    return getattr(pm, path)


# Constants worth periodically re-checking as data drifts. The LGB params are
# the ones with real effect size (bucket C); the rest are guards we expect to
# stay noise — the point is to be ALERTED if any flips noise→REAL.
_WATCHLIST = [
    ("_PLAT_CRED_K", "5,20,80"),
    ("_LGB_PARAMS.num_leaves", "15,31,63"),
    ("_LGB_PARAMS.max_depth", "4,6,8"),
    ("_LGB_PARAMS.learning_rate", "0.03,0.05,0.1"),
    ("_LGB_PARAMS.min_child_samples", "5,10,20"),
    ("_LGB_PARAMS.reg_lambda", "0.5,1.5,5"),
    ("_MONOTONE_BY_FEATURE.enc_plat", "0,1"),
]


def _verdict(lo: float, hi: float) -> str:
    return "REAL ✓" if hi < 0 else "REAL ✗(worse)" if lo > 0 else "noise (CI∋0)"


def sweep_one(df, folds, segs, mask_for, const, values, full, compact) -> bool:
    """Sweep one constant; print results. Returns True if any value is a real
    (CI-clears-0) IMPROVEMENT over the current value — i.e. worth a human look."""
    baseline_val = _current(const)
    base = {m: evaluate(df, folds[m], full) for m in ("random", "time")}
    base_mape = {m: {s: _mape(base[m], mask_for(s)) for s in segs} for m in ("random", "time")}
    cache, rows = {}, []
    for raw in values:
        restore, _ = _set_const(const, raw)
        try:
            cache[raw] = {m: evaluate(df, folds[m], full) for m in ("random", "time")}
        finally:
            restore()
        res = cache[raw]
        d_rnd = _mape(res["random"], mask_for("all")) - base_mape["random"]["all"]
        d_time = _mape(res["time"], mask_for("all")) - base_mape["time"]["all"]
        mk = np.where(res["time"]["tested"])[0]
        b_oof, t_oof, pr = base["time"]["oof"]["median"], res["time"]["oof"]["median"], res["time"]["price"]
        boot = []
        for _ in range(_N_BOOT):
            s = _RNG.choice(mk, len(mk), replace=True)
            e_b = np.abs(np.expm1(b_oof[s]) - pr[s]) / pr[s]
            e_t = np.abs(np.expm1(t_oof[s]) - pr[s]) / pr[s]
            boot.append((e_t.mean() - e_b.mean()) * 100)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append((raw, d_rnd, d_time, lo, hi, _verdict(lo, hi)))
    any_real = any(r[5] == "REAL ✓" for r in rows)

    if compact:
        best = min(rows, key=lambda r: r[2])   # most-negative time ΔMAPE
        tag = "  <-- WORTH A LOOK" if any_real else ""
        print(f"{const:<34} cur={str(baseline_val):>6}  best={str(best[0]):>6} "
              f"Δt={best[2]:+.2f} CI[{best[3]:+.2f},{best[4]:+.2f}]  "
              f"{'REAL' if any_real else 'noise'}{tag}")
        return any_real

    print(f"Constant: {const}  (current prod value = {baseline_val!r})\n")
    hdr = f"{'value':>10} | {'RND ΔMAPE':>10} | {'TIME ΔMAPE':>11} {'time CI(ALL)':>16}  verdict"
    print(hdr); print("-" * len(hdr))
    for raw, d_rnd, d_time, lo, hi, v in rows:
        star = "  *≈current*" if str(raw) == str(baseline_val) else ""
        print(f"{str(raw):>10} | {d_rnd:>+10.2f} | {d_time:>+11.2f} [{lo:+.2f},{hi:+.2f}]  {v}{star}")
    print(f"\nTIME-AWARE ΔMAPE by segment (neg=better; baseline = current {baseline_val!r}):")
    print(f"{'value':>10} | " + "".join(f"{s:>9}" for s in segs))
    for raw in values:
        cells = "".join(f"{_mape(cache[raw]['time'], mask_for(s)) - base_mape['time'][s]:>+9.2f}" for s in segs)
        print(f"{str(raw):>10} | {cells}")
    if full:
        print(f"\n[P10,P90] coverage (target 80) by value, time-aware ALL:")
        for raw in values:
            print(f"  {raw:>10}: {_coverage(cache[raw]['time'], mask_for('all')):.1f}%")
    return any_real


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--const", help="e.g. _PLAT_CRED_K or _LGB_PARAMS.num_leaves")
    ap.add_argument("--values", help="comma-separated values to sweep")
    ap.add_argument("--all", action="store_true", help="scan the whole watchlist; exit 1 if any flips noise→REAL")
    ap.add_argument("--full", action="store_true", help="fit low/high too -> pinball + coverage")
    ap.add_argument("--data", default=None, help="local listings.parquet (else download release)")
    ap.add_argument("--segments", default="all,Diesel,Petrol,Hybrid,PHEV,EV")
    args = ap.parse_args()
    if not args.all and not (args.const and args.values):
        ap.error("give either --all, or both --const and --values")

    df = load_sold(args.data)
    fuel = df["fuel_norm"].values
    segs = [s.strip() for s in args.segments.split(",")]
    def mask_for(s):
        return np.ones(len(df), bool) if s == "all" else (fuel == s)
    folds = {m: _folds(df, m) for m in ("random", "time")}
    print(f"Loaded {len(df)} sold rows from the release snapshot.\n")

    if args.all:
        print("WATCHLIST sensitivity scan (time-aware; 'REAL' = data drifted, worth a human look):\n")
        any_real = False
        for const, values in _WATCHLIST:
            try:
                r = sweep_one(df, folds, segs, mask_for, const,
                              [v.strip() for v in values.split(",")], args.full, compact=True)
                any_real = any_real or r
            except Exception as e:  # noqa: BLE001 — one bad const shouldn't abort the scan
                print(f"{const:<34} ERROR: {e}")
        print("\nAll noise → constants are still well-set; nothing to tune."
              if not any_real else
              "\nSomething flipped REAL → re-sweep it with --const for detail before changing.")
        sys.exit(1 if any_real else 0)

    sweep_one(df, folds, segs, mask_for, args.const,
              [v.strip() for v in args.values.split(",")], args.full, compact=False)
    print("\nRandom KFold flatters (peeks at contemporaneous sales); trust the "
          "TIME column. 'noise (CI∋0)' = not worth hand-tuning.")


if __name__ == "__main__":
    main()
