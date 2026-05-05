"""Flipper-detection composite — human-readable score for the dashboard.

Identifies listings posted by resellers who systematically pad asking
prices with their margin. Distinct from ``seller_type=Profissional``:
captures private-account flippers (gray-area sellers who avoid the
business label) AND surfaces a per-listing decomposition so the
dashboard can show *why* a listing scored as flipper-likely.

NOT a price_model feature. The price model gets the raw primitives
(``seller_listings_count_90d``, ``plate_obscured``, ``seller_pseudoprivate``)
and learns interactions itself; pre-composing throws away signal. This
module is for human-facing surfaces — dashboard tooltips, decision
explanations, alert templates — where a single 0–1 number is more
legible than four separate features.

Methodology — soft weighted vote across the four signals available to us:
    seller_listings_count_90d   weight 0.40   ← strongest, time-axis signal
    seller_cars_count snapshot  weight 0.20   ← concurrent inventory
    seller_pseudoprivate        weight 0.25   ← business posing as private
    plate_obscured              weight 0.15   ← seller-behavior signal

Each primitive maps a value to [0, 1]. The composite is a weighted mean
over the *available* primitives (NaN-safe denominator), so listings
with partial data still score, just with lower confidence — exposed
via ``flipper_confidence`` (= sum of weights of available primitives).

Pilot study (37 hand-labeled listings, 2026-05-05): plate alone produced
~60% false-positive rate as a hazard flag, dominated by privacy-conscious
private sellers. The composite avoids that floor by requiring agreement
across multiple independent surfaces — a single privacy-blurred plate
on a one-listing private seller scores ~0.15, well below the 0.5
threshold a downstream consumer would naturally use.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# Primitive weights — sum to 1.0. Tuned on pilot intuition; revisit
# after backfill_sellers populates seller_uuid for the live corpus and
# we can compute correlation between flipper_score and observed
# margin (asking - sold price) on closed listings.
_W_LISTINGS_90D = 0.40
_W_CARS_COUNT = 0.20
_W_PSEUDOPRIVATE = 0.25
_W_PLATE_OBSCURED = 0.15


def _score_listings_90d(n: float | None) -> float | None:
    """Map 90-day rotation count to [0, 1]. None when missing.

    Buckets reflect the prior: a normal private seller has 1-2 listings
    over 90 days (sold the old car, listed a new one); 3-5 is suspect;
    6+ is almost certainly a flipper or undeclared dealer."""
    if n is None or (isinstance(n, float) and pd.isna(n)):
        return None
    n = int(n)
    if n <= 2:
        return 0.0
    if n <= 5:
        return 0.5
    return 1.0


def _score_cars_count(n: float | None) -> float | None:
    """Map snapshot ``seller_cars_count`` to [0, 1]. None when missing.

    A genuinely private seller has cars_count == 1 (their one car for
    sale). Anything 2+ is concurrent inventory — suspicious for a
    Particular-labeled account."""
    if n is None or (isinstance(n, float) and pd.isna(n)):
        return None
    n = int(n)
    if n <= 1:
        return 0.0
    if n <= 3:
        return 0.5
    return 1.0


def _score_pseudoprivate(flag: bool | None) -> float | None:
    """``seller_pseudoprivate`` is already binary (business JSON contradicts
    Particular trader-title) — pass through."""
    if flag is None or (isinstance(flag, float) and pd.isna(flag)):
        return None
    return 1.0 if bool(flag) else 0.0


def _score_plate_obscured(flag: bool | None) -> float | None:
    """``plate_obscured`` is tri-state (computed in ``computed_columns``);
    None means below-threshold photo coverage, not signal-absent."""
    if flag is None or (isinstance(flag, float) and pd.isna(flag)):
        return None
    return 1.0 if bool(flag) else 0.0


@dataclass(frozen=True)
class FlipperBreakdown:
    """Per-listing decomposition of the flipper score.

    Used by the dashboard to render a tooltip explaining *why* a listing
    scored as flipper-likely (or didn't). ``contributions`` maps each
    primitive name to its (raw_value, mapped_score, weight) triple, so
    a UI can show e.g. "seller_listings_count_90d: 8 → 1.0 × 0.40".
    Primitives with missing data are omitted from ``contributions``.
    """
    score: float | None         # None when no primitive had data
    confidence: float           # 0.0–1.0, fraction of total weight available
    contributions: dict[str, tuple[float, float, float]]


def score_listing(row: pd.Series | dict) -> FlipperBreakdown:
    """Score a single listing. Accepts a Series row or plain dict."""
    primitives = (
        ("seller_listings_count_90d", row.get("seller_listings_count_90d"),
         _score_listings_90d, _W_LISTINGS_90D),
        ("seller_cars_count", row.get("seller_cars_count"),
         _score_cars_count, _W_CARS_COUNT),
        ("seller_pseudoprivate", row.get("seller_pseudoprivate"),
         _score_pseudoprivate, _W_PSEUDOPRIVATE),
        ("plate_obscured", row.get("plate_obscured"),
         _score_plate_obscured, _W_PLATE_OBSCURED),
    )
    weighted_sum = 0.0
    weight_total = 0.0
    contributions: dict[str, tuple[float, float, float]] = {}
    for name, raw, mapper, weight in primitives:
        mapped = mapper(raw)
        if mapped is None:
            continue
        weighted_sum += mapped * weight
        weight_total += weight
        # Coerce raw to a float for the breakdown — None / NaN skipped above.
        try:
            raw_f = float(raw) if not isinstance(raw, bool) else float(int(raw))
        except (TypeError, ValueError):
            raw_f = float("nan")
        contributions[name] = (raw_f, mapped, weight)
    if weight_total == 0.0:
        return FlipperBreakdown(score=None, confidence=0.0, contributions={})
    score = weighted_sum / weight_total
    confidence = weight_total  # already in [0, 1] since weights sum to 1.0
    return FlipperBreakdown(
        score=score, confidence=confidence, contributions=contributions,
    )


def compute_flipper_score(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``flipper_score`` and ``flipper_confidence`` columns.

    Vectorised over rows. ``flipper_score`` is in [0, 1] or NaN when no
    primitive had data (every seller_* and plate_* field None — typically
    a freshly scraped row before backfill_sellers and verify-photos run).
    ``flipper_confidence`` is the fraction of total primitive weight
    that contributed to the score, useful for the dashboard to decide
    whether to display the score or hide it as too-thin-to-trust.

    No-op when none of the input columns are present (e.g. unmatched
    listings DataFrame).
    """
    expected = {
        "seller_listings_count_90d", "seller_cars_count",
        "seller_pseudoprivate", "plate_obscured",
    }
    if not expected.intersection(df.columns):
        return df

    scores: list[float] = []
    confs: list[float] = []
    # Fill missing columns with None so ``score_listing`` doesn't KeyError
    # on partial DataFrames (e.g. tests that provide just plate_obscured).
    for col in expected:
        if col not in df.columns:
            df[col] = None

    for _, row in df.iterrows():
        bd = score_listing(row)
        scores.append(np.nan if bd.score is None else float(bd.score))
        confs.append(float(bd.confidence))
    df["flipper_score"] = scores
    df["flipper_confidence"] = confs
    return df
