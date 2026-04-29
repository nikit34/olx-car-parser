"""Ablation: drop low-importance LLM/text features, compare CV metrics.

Trains the price model twice on the same data:
  baseline — current _ALL_FEATURES (43 features)
  ablation — drops near-zero permutation-importance features
             (desc_mentions_*, damage_*, text_pc_*, urgency, warranty,
              mechanical_condition, first_owner_selling, tuning_or_mods,
              taxi_fleet_rental, right_hand_drive, title_has_*)

Prints side-by-side MAE / MAPE / R² / pinball / coverage. Does NOT save
models — pure measurement.
"""
from __future__ import annotations

import json
import time

import pandas as pd

from src.storage.database import init_db, get_session
from src.storage.repository import get_listings_df
from src.analytics.computed_columns import enrich_listings
from src.analytics.turnover import compute_turnover_stats
from src.dashboard.data_loader import prepare_active_for_model
from src.analytics import price_model as pm


# Features to drop in the ablation run. Picked by median_importance < 0.005
# in data/price_importance.json, plus the rule-based damage flags that
# permutation importance already shows as exactly 0.
DROP = {
    # numeric
    "desc_mentions_num_owners", "tuning_or_mods_count",
    "damage_severity", "damage_score",
    "text_pc_0", "text_pc_1", "text_pc_2", "text_pc_3",
    "text_pc_4", "text_pc_5", "text_pc_6", "text_pc_7",
    # bool
    "desc_mentions_accident", "desc_mentions_repair",
    "desc_mentions_customs_cleared", "right_hand_drive",
    "taxi_fleet_rental", "warranty", "first_owner_selling",
    "title_has_parts_only", "title_has_severe_damage",
    # categorical
    "urgency", "mechanical_condition",
}


def load_active() -> pd.DataFrame:
    init_db()
    s = get_session()
    listings = get_listings_df(s)
    s.close()
    listings = enrich_listings(listings)
    if "real_mileage_km" in listings.columns:
        listings["mileage_km"] = listings["real_mileage_km"].fillna(
            listings["mileage_km"]
        )
    turnover = compute_turnover_stats(listings)
    return prepare_active_for_model(listings, turnover=turnover)


def run(active: pd.DataFrame, label: str, drop: set[str] | None = None) -> dict:
    # Snapshot module-level lists so we can monkey-patch and restore.
    saved = {
        "NUMERIC": list(pm.NUMERIC_FEATURES),
        "BOOL": list(pm.BOOL_FEATURES),
        "CAT": list(pm.CATEGORICAL_FEATURES),
        "ALL": list(pm._ALL_FEATURES),
    }
    try:
        if drop:
            pm.NUMERIC_FEATURES = [f for f in saved["NUMERIC"] if f not in drop]
            pm.BOOL_FEATURES = [f for f in saved["BOOL"] if f not in drop]
            pm.CATEGORICAL_FEATURES = [f for f in saved["CAT"] if f not in drop]
            pm._ALL_FEATURES = (
                pm.NUMERIC_FEATURES + pm.BOOL_FEATURES + pm.CATEGORICAL_FEATURES
            )

        t0 = time.time()
        result = pm.train_price_model(active)
        elapsed = time.time() - t0
        if result is None:
            raise RuntimeError("train_price_model returned None")
        models, cat_maps, metrics, _, _, text_pipeline = result
        metrics = dict(metrics)
        metrics["_label"] = label
        metrics["_elapsed_s"] = round(elapsed, 1)
        metrics["_n_features"] = len(pm._ALL_FEATURES)
        return metrics
    finally:
        pm.NUMERIC_FEATURES = saved["NUMERIC"]
        pm.BOOL_FEATURES = saved["BOOL"]
        pm.CATEGORICAL_FEATURES = saved["CAT"]
        pm._ALL_FEATURES = saved["ALL"]


def fmt(v):
    if isinstance(v, float):
        return f"{v:.4f}" if abs(v) < 10 else f"{v:.1f}"
    return str(v)


def main() -> None:
    active = load_active()
    print(f"Loaded {len(active)} active listings.\n")

    print("[1/2] Baseline (all features)...")
    base = run(active, "baseline")
    print(
        f"  done in {base['_elapsed_s']}s · "
        f"n_features={base['_n_features']}"
    )

    print("\n[2/2] Ablation (drop near-zero LLM/text features)...")
    abl = run(active, "ablation", drop=DROP)
    print(
        f"  done in {abl['_elapsed_s']}s · "
        f"n_features={abl['_n_features']}"
    )

    out = {"baseline": base, "ablation": abl, "dropped": sorted(DROP)}
    p = "/Users/permi/olx-car-parser/data/feature_ablation.json"
    with open(p, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nResults saved to {p}")

    keys = [
        "mae", "mape", "r2",
        "pinball_low", "pinball_median", "pinball_high",
        "coverage_80", "coverage_80_calibrated",
        "conformal_q", "conformal_q_pct",
        "best_n_estimators",
    ]
    print(
        f"\n{'metric':<25} {'baseline':>12} {'ablation':>12} {'delta':>12}"
    )
    print("-" * 65)
    for k in keys:
        b, a = base.get(k), abl.get(k)
        if isinstance(b, (int, float)) and isinstance(a, (int, float)):
            d = a - b
            sign = "+" if d > 0 else ""
            print(f"{k:<25} {fmt(b):>12} {fmt(a):>12} {sign + fmt(d):>12}")
        else:
            print(f"{k:<25} {str(b):>12} {str(a):>12}")

    dt = abl["_elapsed_s"] - base["_elapsed_s"]
    sign = "+" if dt > 0 else ""
    print(
        f"\nBaseline trained {base['_elapsed_s']}s · "
        f"Ablation trained {abl['_elapsed_s']}s "
        f"(delta={sign}{dt:.1f}s)"
    )


if __name__ == "__main__":
    main()
