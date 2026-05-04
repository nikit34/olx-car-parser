"""Train (if needed) the IsolationForest anomaly model and inspect results.

Mirrors the price-model train flow in src.cli but for anomaly detection.
Reads the production listings, runs the same enrichment + active-set
preparation that price_model uses, optionally pulls price predictions
to enable residual_pct / band_pct features, trains IsolationForest, and
prints the top-N most anomalous listings for manual inspection.

Run it like this:

    .venv/bin/python -m scripts.score_anomalies [--retrain] [--top-n 20]
                                                [--no-predictions]
                                                [--contamination 0.05]

Without ``--retrain`` it loads the existing bundle if fresh
(``data/anomaly_model.joblib``, max age 24 h) — useful for repeated
inspection runs without re-fitting on every invocation.
"""
from __future__ import annotations

import argparse
import logging

import pandas as pd

from src.analytics.anomaly import (
    load_model,
    save_model,
    score_anomalies,
    train_anomaly_detector,
)
from src.storage.database import get_session, init_db
from src.storage.repository import get_listings_df


def _enrich_and_prepare(listings_df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same enrichment + active-set prep that price_model
    training uses, so anomaly scores are computed on the same universe
    the price model sees."""
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
    """Load price model and return per-row predictions, or None if no
    fresh price model exists."""
    from src.analytics.price_model import load_model as load_price_model
    from src.analytics.price_model import predict_prices

    saved = load_price_model(max_age_hours=14 * 24)
    if saved is None:
        return None
    models, cat_maps, metrics, oof_preds, calibrator, uncertainty = saved
    edges_raw = metrics.get("conformal_q_bucket_edges")
    bucket_edges = [tuple(e) for e in edges_raw] if edges_raw else None
    return predict_prices(
        models, cat_maps, active_df,
        conformal_q=metrics.get("conformal_q", 0.0),
        oof_preds=oof_preds,
        median_calibrator=calibrator,
        conformal_q_per_bucket=metrics.get("conformal_q_per_bucket"),
        conformal_q_bucket_edges=bucket_edges,
        uncertainty_bundle=uncertainty,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train + inspect the IsolationForest anomaly model.",
    )
    parser.add_argument(
        "--retrain", action="store_true",
        help="Force retrain even if a fresh bundle exists on disk.",
    )
    parser.add_argument(
        "--no-predictions", action="store_true",
        help="Skip residual_pct / band_pct features (uses base features only).",
    )
    parser.add_argument(
        "--contamination", type=float, default=0.05,
        help="Expected anomaly fraction (default 0.05).",
    )
    parser.add_argument(
        "--top-n", type=int, default=20,
        help="Number of most-anomalous listings to print.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    log = logging.getLogger("score_anomalies")

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
                "residual_pct / band_pct)",
            )
        else:
            log.info("  %d predictions joined", predictions_df.notna().any(axis=1).sum())

    bundle = None if args.retrain else load_model()
    if bundle is None:
        log.info("Training fresh IsolationForest (contamination=%.2f)...",
                 args.contamination)
        bundle = train_anomaly_detector(
            active, predictions_df, contamination=args.contamination,
        )
        if bundle is None:
            log.error("Training failed: insufficient data after NaN filter")
            return 1
        save_model(bundle)
        log.info(
            "  saved bundle: n_samples=%d, n_dropped_nan=%d, "
            "uses_predictions=%s, features=%d",
            bundle["n_samples"], bundle["n_dropped_nan"],
            bundle["uses_predictions"], len(bundle["feature_names"]),
        )
    else:
        log.info(
            "Loaded existing bundle (uses_predictions=%s, features=%d)",
            bundle["uses_predictions"], len(bundle["feature_names"]),
        )

    log.info("Scoring active listings...")
    # If the saved bundle was trained with predictions but caller passed
    # --no-predictions, fall back to retraining without predictions to
    # avoid the feature-mismatch error.
    try:
        scores = score_anomalies(bundle, active, predictions_df)
    except ValueError as exc:
        log.warning("Feature mismatch (%s) — retraining matching the input...", exc)
        bundle = train_anomaly_detector(
            active, predictions_df, contamination=args.contamination,
        )
        if bundle is None:
            log.error("Retraining failed")
            return 1
        save_model(bundle)
        scores = score_anomalies(bundle, active, predictions_df)

    n_flagged = int(scores["is_anomaly"].sum())
    n_scored = int(scores["anomaly_score"].notna().sum())
    log.info(
        "Scored %d listings (%d unscored due to NaN features); "
        "%d flagged as anomalies (%.1f%%)",
        n_scored, len(scores) - n_scored,
        n_flagged, 100 * n_flagged / max(n_scored, 1),
    )

    # Join back with listing meta for human-readable output.
    meta_cols = [
        "olx_id", "brand", "model", "year", "mileage_km",
        "price_eur", "url",
    ]
    meta = active.reindex(columns=[c for c in meta_cols if c in active.columns])
    joined = meta.join(scores[["anomaly_score", "is_anomaly"]])
    top = joined.dropna(subset=["anomaly_score"]).nlargest(
        args.top_n, "anomaly_score",
    )
    log.info("Top %d most anomalous:", len(top))
    for _, row in top.iterrows():
        log.info(
            "  score=%.3f flag=%s | %s %s %s · %s km · €%s · %s",
            row["anomaly_score"], row["is_anomaly"],
            row.get("brand"), row.get("model"), row.get("year"),
            row.get("mileage_km"), row.get("price_eur"),
            row.get("olx_id"),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
