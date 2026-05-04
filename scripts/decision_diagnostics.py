"""Diagnostics for ``decision.decide()``.

Runs the resale-decision algorithm against the current active set and
prints two things:

  1. **Verdict histogram** — how many listings end up in each bucket
     (BUY / WATCH / SKIP / REJECT / NO_OPINION) under the current
     thresholds. Useful for sanity-checking after editing the tunables
     in ``src/analytics/decision.py``.

  2. **Context summary** — distribution of segment-level inputs
     (DoM median, fast-sale share, 90-day trend, calibration residual).
     Lets you see whether the context looks sane before trusting any
     verdict it produces.

Run it like this:

    .venv/bin/python -m scripts.decision_diagnostics

The script reads the same DB / release artefacts the Streamlit dashboard
uses — no extra setup.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src" / "dashboard"))

from src.analytics.decision import (  # noqa: E402
    DecisionContext,
    build_context,
    decide,
    VERDICT_BUY,
    VERDICT_WATCH,
    VERDICT_SKIP,
    VERDICT_REJECT,
    VERDICT_NO_OPINION,
)
from src.analytics.price_model import load_metrics_history  # noqa: E402


_VERDICT_ORDER = (VERDICT_BUY, VERDICT_WATCH, VERDICT_SKIP, VERDICT_REJECT, VERDICT_NO_OPINION)


def _load_dashboard_artifacts():
    """Pull the same dataframes the Streamlit Recommendations page uses."""
    from data_loader import load_all  # type: ignore  # local-path import

    loaded = load_all()
    listings_df = loaded[0] if len(loaded) > 0 else pd.DataFrame()
    signals_df = loaded[2] if len(loaded) > 2 else pd.DataFrame()
    predictions_df = loaded[8] if len(loaded) > 8 else pd.DataFrame()
    return listings_df, signals_df, predictions_df


def _load_snapshots(since_days: int = 120) -> pd.DataFrame:
    from src.storage.database import init_db, get_session
    from src.storage.repository import get_price_snapshots_df

    init_db()
    s = get_session()
    try:
        return get_price_snapshots_df(s, since_days=since_days)
    finally:
        s.close()


def _enrich_signals_with_listing_flags(
    signals_df: pd.DataFrame, listings_df: pd.DataFrame,
) -> pd.DataFrame:
    """``compute_signals`` collapses urgency / warranty / days_listed into
    multipliers; ``decide`` reads them again to build narrative reasons.
    Re-attach them so verdicts on the diagnostic match what the dashboard
    would show."""
    if signals_df.empty:
        return signals_df
    extra_cols = [
        "urgency", "warranty", "first_owner_selling", "taxi_fleet_rental",
        "days_listed", "price_change_eur", "mechanical_condition",
    ]
    present = [c for c in extra_cols if c in listings_df.columns]
    if not present:
        return signals_df
    extras = listings_df[["olx_id"] + present].drop_duplicates("olx_id")
    return signals_df.merge(extras, on="olx_id", how="left", suffixes=("", "_raw"))


def _print_verdict_histogram(decisions: list) -> None:
    counts = Counter(d.verdict for d in decisions)
    total = sum(counts.values()) or 1
    print("\n=== Verdict distribution (current actives) ===")
    print(f"{'verdict':<12} {'count':>7} {'share':>7}  median-score")
    for v in _VERDICT_ORDER:
        n = counts.get(v, 0)
        share = n / total * 100
        scores = [d.score for d in decisions if d.verdict == v and not pd.isna(d.score)]
        med = float(np.median(scores)) if scores else float("nan")
        print(f"{v:<12} {n:>7} {share:>6.1f}%  {med:>7.1f}")
    print(f"{'TOTAL':<12} {total:>7}")


def _print_verdict_reasons(decisions: list) -> None:
    """Group decisions by (verdict, first_reason) — surfaces *why* the
    algorithm rejects most rows. Critical for decoding NO_OPINION:
    sample-too-small vs band-too-wide are different problems with
    different fixes."""
    from collections import defaultdict

    buckets: dict[tuple[str, str], int] = defaultdict(int)
    for d in decisions:
        head = d.reasons[0] if d.reasons else "(no reason)"
        # Strip varying numerics so equivalent rejects bucket together.
        import re as _re
        norm = _re.sub(r"\d+(?:\.\d+)?%?", "N", head)
        norm = _re.sub(r"€\s*[\d,]+", "€N", norm)
        buckets[(d.verdict, norm)] += 1

    print("\n=== Top reasons per verdict ===")
    by_verdict: dict[str, list[tuple[str, int]]] = {}
    for (v, reason), n in buckets.items():
        by_verdict.setdefault(v, []).append((reason, n))
    for v in _VERDICT_ORDER:
        rows = sorted(by_verdict.get(v, []), key=lambda r: -r[1])[:5]
        if not rows:
            continue
        print(f"\n  {v}:")
        for reason, n in rows:
            print(f"    {n:>4}× {reason}")


def _summarise_context(ctx: DecisionContext) -> None:
    print("\n=== DecisionContext summary ===")
    if ctx.coverage_80 is not None:
        delta = ctx.coverage_80 - 0.80
        print(f"coverage_80         : {ctx.coverage_80:.1%}  (Δ vs target {delta:+.1%})")
    else:
        print("coverage_80         : —  (no metrics history)")

    def _stats(label: str, values: Iterable[float], unit: str = "") -> None:
        arr = np.fromiter((v for v in values if not pd.isna(v)), dtype=float)
        if arr.size == 0:
            print(f"{label:<20}: empty")
            return
        q = np.quantile(arr, [0.1, 0.5, 0.9])
        print(
            f"{label:<20}: n={arr.size:>4}  "
            f"P10={q[0]:.1f}{unit}  P50={q[1]:.1f}{unit}  P90={q[2]:.1f}{unit}"
        )

    _stats("DoM median (d)", ctx.dom_median.values(), "d")
    _stats("DoM fast share", ctx.dom_fast_share.values())
    _stats("Trend 90d (%)", ctx.trend_90d_pct.values(), "%")
    _stats("Calib residual (%)", ctx.calibration_resid_pct.values(), "%")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshots-window-days", type=int, default=120,
        help="History window used to compute the 90d-trend signal. "
             "Smaller = faster, less coverage on slow segments.",
    )
    args = parser.parse_args(argv)

    print("Loading dashboard artefacts …")
    listings_df, signals_df, predictions_df = _load_dashboard_artifacts()
    if listings_df.empty:
        print("No data available — pull the latest release first.")
        return 1
    print(f"  listings: {len(listings_df):,}")
    print(f"  signals : {len(signals_df):,}")
    print(f"  preds   : {len(predictions_df):,}")

    print("Loading price snapshots …")
    snapshots_df = _load_snapshots(args.snapshots_window_days)
    print(f"  snapshots: {len(snapshots_df):,}")

    metrics_history = load_metrics_history()
    coverage_80 = None
    if metrics_history:
        latest = metrics_history[-1]
        coverage_80 = latest.get("coverage_80_calibrated") or latest.get("coverage_80")

    pred_lookup = (
        dict(zip(predictions_df["olx_id"], predictions_df["predicted_price"]))
        if not predictions_df.empty else {}
    )

    print("Building DecisionContext …")
    ctx = build_context(
        listings_df, snapshots_df,
        coverage_80=coverage_80, predicted_lookup=pred_lookup,
    )

    print("Running decide() over current signals …")
    enriched = _enrich_signals_with_listing_flags(signals_df, listings_df)
    decisions = [decide(row, ctx) for _, row in enriched.iterrows()]

    _print_verdict_histogram(decisions)
    _print_verdict_reasons(decisions)
    _summarise_context(ctx)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
