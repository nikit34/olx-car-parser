"""Batch re-listing detection: scan deactivated listings and persist matches.

Reads the production listings table, builds the segment DoM context the
matcher uses for its dynamic search window, scans every
``deactivation_reason='sold'`` row for a later listing of the same
physical car, and stores accepted matches in ``relist_events``.

Idempotent — re-runs upsert into the (original_olx_id, relist_olx_id)
unique constraint, so re-running after a threshold tweak refreshes
scores without producing duplicates.

Run it like this:

    .venv/bin/python -m scripts.detect_relists [--threshold 0.65] [--dry-run]

The output of this script is what feeds ``decision.calibrate_thresholds``
via ``relist.build_outcomes_df`` — until this script has run there's no
realised-P&L ground truth for the threshold grid-search to chew on.
"""
from __future__ import annotations

import argparse
import logging

from src.analytics.decision import build_context
from src.analytics.relist import DEFAULT_MATCH_THRESHOLD, find_relists
from src.storage.database import get_session, init_db
from src.storage.repository import get_listings_df, record_relist_events


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Detect re-listings of sold cars and store events.",
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_MATCH_THRESHOLD,
        help=f"Minimum match score (default {DEFAULT_MATCH_THRESHOLD}).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print results without writing to DB.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    log = logging.getLogger("detect_relists")

    init_db()
    session = get_session()

    log.info("Loading listings...")
    listings_df = get_listings_df(session)
    log.info("  %d listings loaded", len(listings_df))
    if listings_df.empty:
        log.warning("Empty listings table — nothing to do")
        return 0

    log.info("Building segment DoM context (for dynamic window)...")
    ctx = build_context(listings_df, snapshots_df=None)
    log.info("  %d segments with DoM data", len(ctx.dom_median))

    log.info("Scanning for re-listings (threshold=%.2f)...", args.threshold)
    relist_df = find_relists(
        listings_df, ctx.dom_median, threshold=args.threshold,
    )

    if relist_df.empty:
        log.info("No re-listings detected at threshold %.2f", args.threshold)
        return 0

    median_pct = (
        relist_df["price_delta_pct"].dropna().median()
        if relist_df["price_delta_pct"].notna().any() else float("nan")
    )
    log.info(
        "Detected %d re-listing events: median score=%.3f, gap=%.0fd, "
        "Δprice=%+.1f%%",
        len(relist_df),
        relist_df["match_score"].median(),
        relist_df["gap_days"].median(),
        median_pct,
    )

    top = relist_df.nlargest(min(10, len(relist_df)), "match_score")
    log.info("Top matches:")
    for _, row in top.iterrows():
        log.info(
            "  %s -> %s: score=%.2f, gap=%.0fd, Δ=%+.0f€ (%+.1f%%), window=%dd",
            row["original_olx_id"], row["relist_olx_id"],
            row["match_score"], row["gap_days"],
            row.get("price_delta_eur") or 0,
            row.get("price_delta_pct") or 0,
            row.get("window_days_used") or 0,
        )

    if args.dry_run:
        log.info("(dry-run: not writing to DB)")
        return 0

    log.info("Writing to DB...")
    inserted = record_relist_events(session, relist_df)
    log.info("  %d new events inserted (others updated in place)", inserted)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
