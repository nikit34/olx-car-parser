"""Pre-LLM value gate — pick the genuinely interesting listings to enrich.

Free-tier OpenRouter can only afford ~40 listings/day, so we must spend those
calls on the deals that matter. The GBM price model needs NO LLM input to
estimate fair value (mileage/hp/cc/fuel/brand/model/year are structured scrape
fields), so we can price every fresh listing *before* spending a cloud call
and rank by undervaluation.

Rather than re-implement the deal funnel, this reuses the exact production
path — ``compute_signals`` — which already materialises the deal feed with the
band-width, GBM-discount and net-profit gates applied. We then take the top-K
of that feed by GBM discount, excluding the cheap tier (where the model is
condition-blind and over-predicts) and low-spec rows, and excluding anything
already enriched. See the cheap-tail audit in project memory for why the
€4k / spec_fill / band filters matter.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def rank_deal_olx_ids(
    session,
    *,
    gate: dict,
    limit: int,
    exclude_ids: set[str] | frozenset[str] = frozenset(),
) -> list[str]:
    """Return up to ``limit`` olx_ids of the most-undervalued current deals.

    Ranking = GBM ``undervaluation_pct`` descending over the production
    ``compute_signals`` deal feed (which already enforces band_pct ≤ 0.40,
    net-profit ≥ €500 and 0 < discount ≤ 60% internally). This adds:
      * ``price_eur >= gate['min_price_eur']`` — skip the condition-blind cheap
        tail where the model over-predicts phantom bargains.
      * ``spec_fill >= gate['min_spec_fill']`` — need ≥2 of 4 discriminative
        specs, else the fair value is a coarse baseline guess.
      * exclude ``exclude_ids`` — listings already OpenRouter-enriched.

    Returns [] (never raises) when there is no fresh model / no signals.
    """
    if limit <= 0:
        return []

    # Imported lazily — loading the analytics stack (LightGBM/sklearn) is
    # expensive and pointless if the caller bailed on budget/secret first.
    from src.storage.repository import get_listings_df, get_price_history_df
    from src.analytics.computed_columns import enrich_listings
    from src.analytics.turnover import compute_turnover_stats
    from src.parser.llm_enrichment import merge_real_mileage
    from src.dashboard.data_loader import compute_signals

    listings = get_listings_df(session)
    if listings is None or listings.empty:
        logger.info("value_gate: no listings")
        return []
    listings = enrich_listings(listings)
    listings = merge_real_mileage(listings)
    history = get_price_history_df(session)
    turnover = compute_turnover_stats(listings)

    try:
        signals, *_ = compute_signals(listings, history, turnover=turnover)
    except Exception as e:  # noqa: BLE001 — a missing/stale model must not crash the run
        logger.warning("value_gate: compute_signals failed (%s) — no candidates", e)
        return []

    if signals is None or signals.empty:
        logger.info("value_gate: compute_signals produced no deals")
        return []

    required = {"olx_id", "price_eur", "undervaluation_pct", "spec_fill"}
    missing = required - set(signals.columns)
    if missing:
        logger.warning("value_gate: signals missing columns %s — cannot rank", missing)
        return []

    df = signals.copy()
    df = df[
        (df["price_eur"] >= gate["min_price_eur"])
        & (df["spec_fill"] >= gate["min_spec_fill"])
        & (df["undervaluation_pct"] > gate["min_discount_pct"])
        & (df["undervaluation_pct"] <= gate["max_discount_pct"])
    ]
    if exclude_ids:
        df = df[~df["olx_id"].astype(str).isin(exclude_ids)]
    if df.empty:
        logger.info("value_gate: no candidates after filters")
        return []

    df = df.sort_values("undervaluation_pct", ascending=False)
    ids = [str(x) for x in df["olx_id"].head(limit).tolist()]
    logger.info(
        "value_gate: %d candidate deals (top discount %.1f%%), selecting %d",
        len(df), float(df["undervaluation_pct"].iloc[0]), len(ids),
    )
    return ids
