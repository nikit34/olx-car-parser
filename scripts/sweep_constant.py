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
feature pipeline (price_model._prepare_X / _model_for_quantile) so a swept
constant flows through exactly as in prod - including the sold-target
adjustment and the sample weights, so a constant that only bites through those
is visible here. Constants that move the target scale itself (_SOLD_TIERS) are
refused: every metric shifts with them, so the sweep cannot rank values - derive
those from observed relist price deltas instead. It is a RELATIVE-delta tool:
absolute MAPE differs from prod CV (no turnover features, fixed n_estimators),
but ΔMAPE vs the current value is what matters.

Usage:
  python -m scripts.sweep_constant --const _LGB_PARAMS.num_leaves --values 15,31,63
  python -m scripts.sweep_constant --const _LGB_PARAMS.min_child_samples --values 5,8,20
  python -m scripts.sweep_constant --const _LGB_PARAMS.learning_rate --values 0.03,0.05,0.1 --full
  python -m scripts.sweep_constant --drift
Flags:
  --full          also fit low/high quantiles -> report pinball + [P10,P90] coverage
  --data P        use a local listings.parquet instead of downloading the release
  --segments d,p,h,e,phev  restrict reported fuel segments (default: all)
  --spec-dropout F  mirror the shipped spec-dropout regime (default = prod fraction)
  --drift         run the price-level drift probe alone (it also runs under --all)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, TimeSeriesSplit

from scripts import release_chunks
from src.analytics import price_model as pm

_CACHE = Path("/tmp/olx-release/listings.parquet")
_N_EST = 1100         # fixed (no early stopping) so every swept value is comparable
_N_BOOT = 2000
_RNG = np.random.RandomState(42)
_DRIFT_FLAG_PCT = 3.0
_DRIFT_RNG_SEED = 4242


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
        print(f"Downloading listings.parquet from {release_chunks.REPO}:"
              f"{release_chunks.TAG} …", file=sys.stderr)
        blob = release_chunks.fetch("listings.parquet")
        if blob is None:
            raise SystemExit("listings.parquet is not readable from the "
                             f"{release_chunks.TAG} release")
        path.write_bytes(blob)
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


def evaluate(df: pd.DataFrame, folds, full: bool, spec_dropout: float = 0.0) -> dict:
    """OOF predictions reusing prod feature prep + model construction so the
    patched constant takes effect. Platform encoding is leakage-safe per fold.

    ``spec_dropout`` mirrors the shipped missingness-aware regime
    (pm._apply_spec_dropout): on that fraction of each fold's FITTING rows a
    random subset of the discriminative specs is NaN'd, so the model learns the
    same missing-feature behaviour it ships with. Validation rows keep real
    features. 0.0 → legacy behaviour (no augmentation). cat_maps are fit on the
    FULL train fold (no category loss) then the fitting matrix is built from the
    augmented copy — exactly as train_price_model does."""
    sold_mult, sold_w = pm._build_sold_target_adjustment(df)
    y_price = df["price_eur"].values.astype(float) * sold_mult
    y = np.log1p(np.maximum(y_price, 0))
    weights = pm._compute_sample_weights(y_price) * sold_w
    n = len(df)
    quants = pm._QUANTILES if full else {"median": 0.5}
    oof = {q: np.full(n, np.nan) for q in quants}
    tested = np.zeros(n, bool)
    cat_idx = [pm._ALL_FEATURES.index(c) for c in pm.CATEGORICAL_FEATURES]
    for fi, (tr, te) in enumerate(folds):
        tr_df, te_df = df.iloc[tr], df.iloc[te]
        if spec_dropout > 0:
            _, cmaps = pm._prepare_X(tr_df)                  # maps on full fold
            tr_fit = pm._apply_spec_dropout(
                tr_df, spec_dropout, np.random.default_rng(1234 + fi),
            )
            x_tr, _ = pm._prepare_X(tr_fit, cmaps)
        else:
            x_tr, cmaps = pm._prepare_X(tr_df)
        x_te, _ = pm._prepare_X(te_df, cmaps)
        for name, alpha in quants.items():
            model = pm._model_for_quantile(name, alpha, _N_EST)   # uses pm._LGB_PARAMS + monotone
            model.fit(
                x_tr, y[tr], sample_weight=weights[tr],
                categorical_feature=cat_idx,
            )
            oof[name][te] = model.predict(x_te)
        tested[te] = True
    return {"oof": oof, "tested": tested, "y": y,
            "price": df["price_eur"].values.astype(float),
            "target": y_price}


def _mape(res, mask):
    m = mask & res["tested"]            # TimeSeriesSplit never predicts the first block
    p = np.expm1(res["oof"]["median"]); pr = res["price"]
    return float(np.mean(np.abs(p[m] - pr[m]) / pr[m]) * 100)


def _coverage(res, mask):
    m = mask & res["tested"]
    lo = np.expm1(res["oof"]["low"]); hi = np.expm1(res["oof"]["high"]); pr = res["price"]
    return float(np.mean((pr[m] >= lo[m]) & (pr[m] <= hi[m])) * 100)


_TARGET_SCALE_CONSTS = ("_SOLD_TIERS", "_SOLD_MAX_DAYS")


def _set_const(path: str, raw: str):
    """Patch pm.<const> (scalar) or pm.<DICT>.<key>; return (restore_fn, parsed)."""
    if path.split(".", 1)[0] in _TARGET_SCALE_CONSTS:
        sys.exit(
            f"{path} moves the training target itself, so every metric here "
            "shifts with it and the sweep cannot rank values. Derive it from "
            "observed relist price deltas instead (see relist.find_relists)."
        )
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
    ("_LGB_PARAMS.num_leaves", "15,31,63"),
    ("_LGB_PARAMS.max_depth", "4,6,8"),
    ("_LGB_PARAMS.learning_rate", "0.03,0.05,0.1"),
    ("_LGB_PARAMS.min_child_samples", "5,10,20"),
    ("_LGB_PARAMS.reg_lambda", "0.5,1.5,5"),
    # enc_plat removed in v13 (project_missing_feature_overprediction); its
    # _PLAT_CRED_K / enc_plat-monotone knobs no longer affect the model.
]


_MIN_EFFECT = 0.10


def _summary_label(any_real: bool, lo: float, hi: float) -> str:
    if any_real:
        return "REAL"
    if hi < 0 or lo > 0:
        return f"under the {_MIN_EFFECT:.2f} gate"
    return "noise"


def _verdict(lo: float, hi: float) -> str:
    if hi < -_MIN_EFFECT:
        return "REAL ✓"
    if lo > _MIN_EFFECT:
        return "REAL ✗(worse)"
    if hi < 0 or lo > 0:
        return f"tiny (|Δ|<{_MIN_EFFECT:.2f})"
    return "noise (CI∋0)"


def _month_index(dt: pd.Series) -> tuple[np.ndarray, list[str]]:
    """Sale month per row as 0-based consecutive ints, plus the month labels."""
    per = dt.dt.to_period("M")
    labels = sorted(per.unique())
    lookup = {p: i for i, p in enumerate(labels)}
    return per.map(lookup).to_numpy(dtype=int), [str(p) for p in labels]


def _boot_month_means(
    resid: np.ndarray, midx: np.ndarray, n_months: int, n_boot: int, rng, block: int = 200,
) -> np.ndarray:
    """(n_boot, n_months) matrix of resampled per-month mean residuals.

    Resampling happens WITHIN each month: month sizes are a property of the
    scrape, not of the quantity being estimated, so the thing with sampling
    error is each month's level, not how many cars sold that month. Drawn in
    blocks because one (n_boot, n_rows) index matrix for the fattest month is
    hundreds of MB.
    """
    out = np.empty((n_boot, n_months))
    for m in range(n_months):
        r = resid[midx == m]
        done = 0
        while done < n_boot:
            k = min(block, n_boot - done)
            out[done:done + k, m] = r[rng.randint(0, len(r), size=(k, len(r)))].mean(axis=1)
            done += k
    return out


def drift_probe(df, folds, spec_dropout: float = 0.0, n_boot: int = _N_BOOT) -> bool:
    """Mix-adjusted price-level index by sale month, and whether it trends.

    The price model carries no date feature and trains on the full history
    with no recency weight, so it cannot represent a moving market: a drift
    in the PT level has to surface as a trend in the random-KFold OOF
    residual grouped by sale month. Measured 2026-09-18 on 82.5k sold rows
    that index was flat (~1pp of wobble, no direction) and exponential
    recency weighting came out noise at every half-life from 180d to 30d,
    which is why the full history still trains unweighted. This probe is
    what keeps that answer honest without anyone re-running the
    investigation: the weekly job goes red when the level starts moving.

    Flags only when the bootstrap CI of the window-long drift clears zero
    AND the drift exceeds _DRIFT_FLAG_PCT. The CI alone is not enough - on
    ~80k rows a clean but economically pointless slope would page a human
    every Sunday. That magnitude gate is a judgement call (twice the wobble
    seen at calibration time), not a measured constant.
    """
    midx, labels = _month_index(df["dt"])
    n_m = len(labels)
    if n_m < 3:
        print(f"PRICE-LEVEL DRIFT probe: needs >=3 sale months, got {n_m}. Skipped.")
        return False

    res = evaluate(df, folds["random"], False, spec_dropout)
    resid = res["y"] - res["oof"]["median"]

    means = np.array([resid[midx == m].mean() for m in range(n_m)])
    counts = np.array([int((midx == m).sum()) for m in range(n_m)])
    boot = _boot_month_means(
        resid, midx, n_m, n_boot, np.random.RandomState(_DRIFT_RNG_SEED),
    )

    x = np.arange(n_m, dtype=float)
    xc = x - x.mean()
    span = float(n_m - 1)
    drift = float(np.expm1(means @ xc / (xc @ xc) * span) * 100)
    boot_drift = np.expm1(boot @ xc / (xc @ xc) * span) * 100
    lo, hi = (float(v) for v in np.percentile(boot_drift, [2.5, 97.5]))
    real = (lo > 0 or hi < 0) and abs(drift) >= _DRIFT_FLAG_PCT

    print("\nPRICE-LEVEL DRIFT probe (mix-adjusted OOF residual by sale month;")
    print(" positive = market above what the pooled model expects):\n")
    print(f"{'month':>9} {'n':>8} {'level %':>9}")
    for m in range(n_m):
        print(f"{labels[m]:>9} {counts[m]:>8} {np.expm1(means[m]) * 100:>+9.2f}")
    print(
        f"\ndrift over {labels[0]}..{labels[-1]}: {drift:+.2f}% "
        f"CI [{lo:+.2f},{hi:+.2f}]  "
        + (
            f"REAL (|drift| >= {_DRIFT_FLAG_PCT}% and CI clears 0)  <-- WORTH A LOOK"
            if real
            else "flat (CI∋0)" if not (lo > 0 or hi < 0)
            else f"trending but under the {_DRIFT_FLAG_PCT}% gate"
        )
    )
    return real


def sweep_one(df, folds, segs, mask_for, const, values, full, compact, spec_dropout=0.0) -> bool:
    """Sweep one constant; print results. Returns True if any value is a real
    (CI-clears-0) IMPROVEMENT over the current value — i.e. worth a human look."""
    baseline_val = _current(const)
    base = {m: evaluate(df, folds[m], full, spec_dropout) for m in ("random", "time")}
    base_mape = {m: {s: _mape(base[m], mask_for(s)) for s in segs} for m in ("random", "time")}
    cache, rows = {}, []
    for raw in values:
        restore, _ = _set_const(const, raw)
        try:
            cache[raw] = {m: evaluate(df, folds[m], full, spec_dropout) for m in ("random", "time")}
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
        label = _summary_label(any_real, best[3], best[4])
        print(f"{const:<34} cur={str(baseline_val):>6}  best={str(best[0]):>6} "
              f"Δt={best[2]:+.2f} CI[{best[3]:+.2f},{best[4]:+.2f}]  "
              f"{label}{tag}")
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
    global _N_EST
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--const", help="e.g. _LGB_PARAMS.num_leaves or _LGB_PARAMS.reg_lambda")
    ap.add_argument("--values", help="comma-separated values to sweep")
    ap.add_argument("--all", action="store_true", help="scan the whole watchlist; exit 1 if any flips noise→REAL")
    ap.add_argument("--full", action="store_true", help="fit low/high too -> pinball + coverage")
    ap.add_argument("--data", default=None, help="local listings.parquet (else download release)")
    ap.add_argument("--n-est", type=int, default=_N_EST,
                    help=f"trees per fit, fixed for every swept value (default {_N_EST}). "
                         "Raise it to tell a real effect apart from one the tree budget creates.")
    ap.add_argument("--segments", default="all,Diesel,Petrol,Hybrid,PHEV,EV")
    ap.add_argument("--spec-dropout", type=float, default=pm._SPEC_DROPOUT_FRAC,
                    help="mirror the shipped spec-dropout regime (default = prod "
                         f"{pm._SPEC_DROPOUT_FRAC}); 0 = legacy non-dropout model")
    ap.add_argument("--drift", action="store_true",
                    help="price-level drift probe only; exit 1 if the level moved")
    args = ap.parse_args()
    if not args.all and not args.drift and not (args.const and args.values):
        ap.error("give either --all, or --drift, or both --const and --values")
    if args.const and args.const.split(".", 1)[0] in _TARGET_SCALE_CONSTS:
        ap.error(
            f"{args.const} moves the training target itself, so every metric "
            "here shifts with it and the sweep cannot rank values. Derive it "
            "from observed relist price deltas instead (relist.find_relists)."
        )

    _N_EST = args.n_est

    df = load_sold(args.data)
    fuel = df["fuel_norm"].values
    segs = [s.strip() for s in args.segments.split(",")]
    def mask_for(s):
        return np.ones(len(df), bool) if s == "all" else (fuel == s)
    folds = {m: _folds(df, m) for m in ("random", "time")}
    dz = args.spec_dropout
    print(f"Loaded {len(df)} sold rows from the release snapshot.")
    print(f"spec-dropout regime: {dz:.2f}" + (" (mirrors shipped model)" if dz > 0 else " (legacy non-dropout)"))
    print(f"trees per fit: {_N_EST} (fixed, no early stopping)\n")

    if args.drift and not args.all:
        sys.exit(1 if drift_probe(df, folds, spec_dropout=dz) else 0)

    if args.all:
        print("WATCHLIST sensitivity scan (time-aware; 'REAL' = data drifted, worth a human look;\n"
          f"          an effect smaller than {_MIN_EFFECT:.2f} MAPE does not survive a holdout, so it is not one):\n")
        any_real = False
        for const, values in _WATCHLIST:
            try:
                r = sweep_one(df, folds, segs, mask_for, const,
                              [v.strip() for v in values.split(",")], args.full, compact=True,
                              spec_dropout=dz)
                any_real = any_real or r
            except Exception as e:  # noqa: BLE001 — one bad const shouldn't abort the scan
                print(f"{const:<34} ERROR: {e}")
        print("\nAll noise → constants are still well-set; nothing to tune."
              if not any_real else
              "\nSomething flipped REAL → re-sweep it with --const for detail before changing.")
        try:
            drifted = drift_probe(df, folds, spec_dropout=dz)
        except Exception as e:  # noqa: BLE001 — a broken probe shouldn't hide the sweep verdict
            print(f"\nPRICE-LEVEL DRIFT probe ERROR: {e}")
            drifted = False
        if drifted:
            print("\nThe market level moved → recency weighting is worth re-testing "
                  "(half-lives 180/120/90/60d) before trusting the current model's level.")
        sys.exit(1 if (any_real or drifted) else 0)

    sweep_one(df, folds, segs, mask_for, args.const,
              [v.strip() for v in args.values.split(",")], args.full, compact=False,
              spec_dropout=dz)
    print("\nRandom KFold flatters (peeks at contemporaneous sales); trust the "
          "TIME column. 'noise (CI∋0)' = not worth hand-tuning.")


if __name__ == "__main__":
    main()
