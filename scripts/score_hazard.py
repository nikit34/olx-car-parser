"""Train (if needed) the hazard model and inspect P(sold within horizon).

Mirrors scripts/score_anomalies.py for the per-listing hazard model.
Reads the production listings, runs the same enrichment + active-set
preparation that price_model uses, optionally pulls price predictions
to enable residual_pct / band_pct features (the dominant signals on a
real corpus), trains a binary LightGBM, and prints distribution stats
+ a sample of the top fastest- and slowest-selling active listings.

Run it like this:

    .venv/bin/python -m scripts.score_hazard [--retrain]
                                             [--horizon 30]
                                             [--top-n 10]
                                             [--no-predictions]

Without ``--retrain`` it loads the existing bundle if fresh
(``data/hazard_model.joblib``, max age 24 h).
"""
from __future__ import annotations

import argparse
import logging

import pandas as pd

from src.analytics.hazard import (
    DEFAULT_HORIZON_DAYS,
    load_model,
    predict_sold_probability,
    save_model,
    train_hazard_model,
)
from src.storage.database import get_session, init_db
from src.storage.repository import get_listings_df


def _enrich_and_prepare(listings_df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same enrichment + active-set prep used everywhere
    else so hazard scores are computed on the same universe price_model
    and anomaly scoring see."""
    from src.analytics.computed_columns import enrich_listings
    from src.analytics.turnover import compute_turnover_stats
    from src.dashboard.data_loader import prepare_active_for_model

    listings_df = enrich_listings(listings_df)
    if "real_mileage_km" in listings_df.columns:
        listings_df["mileage_km"] = (
            listings_df["real_mileage_km"].fillna(listings_df["mileage_km"])
        )
    turnover = compute_turnover_stats(listings_df)
    return prepare_active_for_model(listings_df, turnover=turnover, include_sold=True)


def _maybe_get_predictions(active_df: pd.DataFrame) -> pd.DataFrame | None:
    """Load price model and return per-row predictions, or None when no
    fresh price model exists. Mirror of the helper in score_anomalies.py."""
    from src.analytics.price_model import load_model as load_price_model
    from src.analytics.price_model import predict_prices

    saved = load_price_model(max_age_hours=14 * 24)
    if saved is None:
        return None
    models, cat_maps, metrics, oof_preds, calibrator = saved
    return predict_prices(
        models, cat_maps, active_df,
        conformal_q=metrics.get("conformal_q", 0.0),
        oof_preds=oof_preds,
        median_calibrator=calibrator,
        conformal_q_per_bucket=metrics.get("conformal_q_per_bucket"),
        conformal_q_bucket_edges=metrics.get("conformal_q_bucket_edges"),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train + inspect the hazard model.",
    )
    parser.add_argument(
        "--retrain", action="store_true",
        help="Force retrain even if a fresh bundle exists on disk.",
    )
    parser.add_argument(
        "--horizon", type=int, default=DEFAULT_HORIZON_DAYS,
        help=f"Sale-window horizon in days (default {DEFAULT_HORIZON_DAYS}).",
    )
    parser.add_argument(
        "--no-predictions", action="store_true",
        help="Skip residual_pct / band_pct features.",
    )
    parser.add_argument(
        "--top-n", type=int, default=10,
        help="Print top-N fastest and slowest predicted listings.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    log = logging.getLogger("score_hazard")

    init_db()
    session = get_session()
    log.info("Loading listings...")
    listings_df = get_listings_df(session)
    session.close()
    log.info("  %d raw listings", len(listings_df))
    if listings_df.empty:
        log.warning("Empty listings table — nothing to do")
        return 0

    log.info("Enriching + preparing active set...")
    active = _enrich_and_prepare(listings_df)
    log.info("  %d rows in active set", len(active))

    predictions_df = None
    if not args.no_predictions:
        log.info("Loading price model for residual features...")
        predictions_df = _maybe_get_predictions(active)
        if predictions_df is None:
            log.warning(
                "  No fresh price model — falling back to base features only "
                "(run `python -m src.cli train-model` first to enable "
                "residual_pct / band_pct).",
            )

    bundle = None if args.retrain else load_model()
    if bundle is None or bundle.get("horizon_days") != args.horizon:
        if bundle is not None:
            log.info(
                "Existing bundle horizon=%dd != requested %dd — retraining",
                bundle.get("horizon_days"), args.horizon,
            )
        log.info("Training fresh hazard model (horizon=%dd)...", args.horizon)
        bundle = train_hazard_model(
            active, predictions_df, horizon_days=args.horizon,
        )
        if bundle is None:
            log.error(
                "Training failed: insufficient data after censoring + NaN drop",
            )
            return 1
        save_model(bundle)
        m = bundle["metrics"]
        log.info(
            "  saved bundle: AUC=%.3f · logloss=%.3f · base_rate_train=%.2f · "
            "n_train=%d · n_val=%d · best_iter=%d · split=%s · "
            "censored=%d · nan_dropped=%d",
            m["auc"], m["logloss"], m["base_rate_train"],
            m["n_train"], m["n_val"], m["best_iteration"],
            m.get("split_mode", "?"),
            m["n_dropped_censored"], m["n_dropped_nan_features"],
        )
    else:
        m = bundle["metrics"]
        log.info(
            "Loaded existing bundle: horizon=%dd · AUC=%.3f · features=%d",
            bundle["horizon_days"], m["auc"], len(bundle["feature_names"]),
        )

    log.info("Scoring active set...")
    # Restrict to genuinely active listings — that's where the per-listing
    # liquidity score is actionable. Sold rows in the corpus already have
    # an outcome.
    active_only = active[active["is_active"].astype(bool)] if "is_active" in active.columns else active
    log.info("  scoring %d active listings", len(active_only))

    if predictions_df is not None:
        active_preds = predictions_df.reindex(active_only.index)
    else:
        active_preds = None

    try:
        scores = predict_sold_probability(bundle, active_only, active_preds)
    except ValueError as exc:
        log.warning("Feature mismatch (%s) — retraining matching the input...", exc)
        bundle = train_hazard_model(
            active, predictions_df, horizon_days=args.horizon,
        )
        if bundle is None:
            log.error("Retraining failed")
            return 1
        save_model(bundle)
        scores = predict_sold_probability(bundle, active_only, active_preds)

    valid = scores["prob_sold_within_horizon"].notna()
    log.info(
        "Scored %d / %d active listings (others have NaN features)",
        int(valid.sum()), len(scores),
    )
    if valid.any():
        probs = scores.loc[valid, "prob_sold_within_horizon"]
        log.info(
            "  P(sold within %dd) distribution: median=%.2f · "
            "p25=%.2f · p75=%.2f · p10=%.2f · p90=%.2f",
            bundle["horizon_days"],
            probs.median(), probs.quantile(0.25), probs.quantile(0.75),
            probs.quantile(0.10), probs.quantile(0.90),
        )

    # Join meta for human-readable output.
    meta_cols = [
        "olx_id", "brand", "model", "year", "mileage_km",
        "price_eur", "url",
    ]
    meta = active_only.reindex(columns=[c for c in meta_cols if c in active_only.columns])
    joined = meta.join(scores[["prob_sold_within_horizon"]])
    joined = joined.dropna(subset=["prob_sold_within_horizon"])

    n = min(args.top_n, len(joined))
    if n:
        log.info("Top %d most likely to sell within %dd:", n, bundle["horizon_days"])
        for _, row in joined.nlargest(n, "prob_sold_within_horizon").iterrows():
            log.info(
                "  P=%.3f | %s %s %s · %s km · €%s · %s",
                row["prob_sold_within_horizon"],
                row.get("brand"), row.get("model"), row.get("year"),
                row.get("mileage_km"), row.get("price_eur"),
                row.get("olx_id"),
            )
        log.info("Top %d least likely to sell within %dd:", n, bundle["horizon_days"])
        for _, row in joined.nsmallest(n, "prob_sold_within_horizon").iterrows():
            log.info(
                "  P=%.3f | %s %s %s · %s km · €%s · %s",
                row["prob_sold_within_horizon"],
                row.get("brand"), row.get("model"), row.get("year"),
                row.get("mileage_km"), row.get("price_eur"),
                row.get("olx_id"),
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
