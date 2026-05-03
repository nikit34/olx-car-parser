"""Re-listing detection: match a deactivated listing to a later listing
of the same physical car.

Used to extract ground-truth resale outcomes — the gap between the
original ask and the re-listed ask is the closest proxy we have to a
realised flip margin without manual portfolio entry. Output is
consumable by ``decision.calibrate_thresholds`` to grid-search
``DEFAULT_TUNABLES`` against actual realised P&L.

The matcher is text-fingerprint only (no VIN field on OLX/SV; no photo
embeddings yet) with tolerances on the noisy attributes:

  - exact: brand, model, year, fuel_type, transmission
  - tolerant: engine_cc (±50 cc), horsepower (±10 hp), color (after
              Portuguese-synonym normalization), mileage (must increase,
              plausible km/day rate)
  - bonus: same district, generation/sub_model/trim_level/doors/seats/
           drive_type when both present

Search window is dynamic per segment — derived from the segment's
median days-on-market (4× DoM, clamped to [60, 365]) so fast-moving
segments don't waste budget on stale candidates and slow-moving ones
don't get truncated. Beyond 365 days the match-score distribution is
dominated by random-coincidence pairs in the 2025 cohort backtest, so
that's the hard outer cap.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# Any pair scoring at or above this is recorded as a re-listing event.
# 0.65 = the brand/model/year skeleton (0.50 base) plus at least three
# corroborating signals (e.g. district, color match, engine_cc match).
# Tighter than 0.5 to suppress same-make-model-year-but-different-unit
# candidates that share only the skeleton.
DEFAULT_MATCH_THRESHOLD = 0.65

# Default segment window when DoM is unavailable. Six months is the
# point at which the empirical false-positive rate from "different
# units of the same model" starts dominating.
DEFAULT_WINDOW_DAYS = 180
# Hard outer cap regardless of segment liquidity.
MAX_WINDOW_DAYS = 365
MIN_WINDOW_DAYS = 60

# Mileage tolerances. Parser noise floor: ±2000 km (rounding /
# free-text km vs odometer attribute). Plausible delta rate: 100 km/day
# average across the gap (a frequently-driven car does ~50 km/day; 2×
# headroom for storage-then-sale or commercial use).
MILEAGE_NOISE_FLOOR_KM = 2000
MAX_KM_PER_DAY = 100.0


# ---------------------------------------------------------------------------
# Color normalization
# ---------------------------------------------------------------------------

# OLX/SV color strings are free-text Portuguese; the underlying physical
# colors collapse to ~10 buckets. Map common variants so e.g.
# "Cinzento" and "Prateado" don't false-mismatch when the same car is
# re-listed with a slightly different label by another seller.
_COLOR_SYNONYMS: dict[str, str] = {
    # Silvery / gray family — the noisiest bucket on free-text PT
    "cinza": "cinzento",
    "cinzento": "cinzento",
    "cinzento escuro": "cinzento",
    "cinzento claro": "cinzento",
    "prata": "cinzento",
    "prateado": "cinzento",
    "silver": "cinzento",
    "gray": "cinzento",
    "grey": "cinzento",
    # Common solid colors — allow English fall-throughs
    "white": "branco", "branco": "branco",
    "black": "preto", "preto": "preto",
    "red": "vermelho", "vermelho": "vermelho",
    "blue": "azul", "azul": "azul",
    "green": "verde", "verde": "verde",
    "yellow": "amarelo", "amarelo": "amarelo",
    "orange": "laranja", "laranja": "laranja",
    "brown": "castanho", "castanho": "castanho",
    "beige": "bege", "bege": "bege",
}


def _normalize_color(c) -> str | None:
    if c is None or (isinstance(c, float) and pd.isna(c)):
        return None
    s = str(c).strip().lower()
    if not s:
        return None
    return _COLOR_SYNONYMS.get(s, s)


# ---------------------------------------------------------------------------
# Match scoring
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MatchResult:
    score: float
    reasons: tuple[str, ...]
    rejected: bool
    reject_reason: str | None = None


def _segment_window_days(dom_median) -> int:
    """Adaptive search window: 4× typical DoM, clamped to [60, 365]."""
    if dom_median is None or pd.isna(dom_median):
        return DEFAULT_WINDOW_DAYS
    return int(min(max(4 * float(dom_median), MIN_WINDOW_DAYS), MAX_WINDOW_DAYS))


def _val(row, key: str):
    if hasattr(row, "get"):
        return row.get(key)
    return row[key] if key in row else None


def _present(v) -> bool:
    if v is None:
        return False
    if isinstance(v, float) and pd.isna(v):
        return False
    if isinstance(v, str) and not v.strip():
        return False
    return True


def compute_match_score(
    original,
    candidate,
    gap_days: float,
) -> MatchResult:
    """Score a (deactivated original, later candidate) pair on [0, 1].

    Returns ``MatchResult`` with ``rejected=True`` and a one-line
    ``reject_reason`` when any hard gate fails (saves the caller from
    inspecting a 0.0 score). Otherwise ``score`` is the sum of feature
    contributions starting from a 0.50 base for the brand/model/year
    skeleton, capped at 1.0.

    *gap_days* is the calendar gap between the original's
    ``deactivated_at`` and the candidate's ``first_seen_at``; used
    only by the mileage-plausibility gate.
    """
    reasons: list[str] = []

    # ---- Hard gates -------------------------------------------------------
    o_brand, c_brand = _val(original, "brand"), _val(candidate, "brand")
    if not _present(o_brand) or not _present(c_brand) or o_brand != c_brand:
        return MatchResult(0.0, (), True, "brand mismatch")

    o_model, c_model = _val(original, "model"), _val(candidate, "model")
    if not _present(o_model) or not _present(c_model) or o_model != c_model:
        return MatchResult(0.0, (), True, "model mismatch")

    o_year, c_year = _val(original, "year"), _val(candidate, "year")
    if not _present(o_year) or not _present(c_year) or int(o_year) != int(c_year):
        return MatchResult(0.0, (), True, "year mismatch")

    # Fuel + transmission: hard reject when both present and differ.
    o_fuel, c_fuel = _val(original, "fuel_type"), _val(candidate, "fuel_type")
    if _present(o_fuel) and _present(c_fuel) and o_fuel != c_fuel:
        return MatchResult(0.0, (), True, "fuel_type mismatch")

    o_trans, c_trans = _val(original, "transmission"), _val(candidate, "transmission")
    if _present(o_trans) and _present(c_trans) and o_trans != c_trans:
        return MatchResult(0.0, (), True, "transmission mismatch")

    # Mileage: must not decrease (with parser-noise floor); plausible
    # per-day gain rate.
    o_km, c_km = _val(original, "mileage_km"), _val(candidate, "mileage_km")
    if _present(o_km) and _present(c_km):
        delta = float(c_km) - float(o_km)
        if delta < -MILEAGE_NOISE_FLOOR_KM:
            return MatchResult(
                0.0, (), True, f"mileage decreased by {-delta:.0f} km",
            )
        max_plausible = MAX_KM_PER_DAY * max(gap_days, 1) + MILEAGE_NOISE_FLOOR_KM
        if delta > max_plausible:
            return MatchResult(
                0.0, (), True,
                f"implausible mileage gain {delta:.0f} km in {gap_days:.1f} days",
            )

    # ---- Soft scoring: 0.50 base for the skeleton -------------------------
    score = 0.50
    reasons.append("brand+model+year skeleton")

    # Engine displacement
    o_cc, c_cc = _val(original, "engine_cc"), _val(candidate, "engine_cc")
    if _present(o_cc) and _present(c_cc):
        if abs(float(o_cc) - float(c_cc)) <= 50:
            score += 0.10
            reasons.append(f"engine_cc match (±50): {int(o_cc)}≈{int(c_cc)}")
        else:
            score -= 0.20
            reasons.append(
                f"engine_cc mismatch: {int(o_cc)} vs {int(c_cc)} (penalty)"
            )

    # Horsepower
    o_hp, c_hp = _val(original, "horsepower"), _val(candidate, "horsepower")
    if _present(o_hp) and _present(c_hp):
        if abs(float(o_hp) - float(c_hp)) <= 10:
            score += 0.05
            reasons.append(f"hp match (±10): {int(o_hp)}≈{int(c_hp)}")
        else:
            score -= 0.15
            reasons.append("hp mismatch (penalty)")

    # District proximity
    o_d, c_d = _val(original, "district"), _val(candidate, "district")
    if _present(o_d) and _present(c_d):
        if str(o_d).strip().lower() == str(c_d).strip().lower():
            score += 0.10
            reasons.append(f"same district: {o_d}")

    # Color (after Portuguese-synonym normalization)
    o_color = _normalize_color(_val(original, "color"))
    c_color = _normalize_color(_val(candidate, "color"))
    if o_color and c_color and o_color == c_color:
        score += 0.10
        reasons.append(f"color match: {o_color}")

    # Generation / sub_model / trim_level / body — only when BOTH present
    for field, weight in (
        ("generation", 0.05),
        ("sub_model", 0.05),
        ("trim_level", 0.05),
        ("doors", 0.05),
        ("seats", 0.03),
        ("drive_type", 0.03),
    ):
        o_v, c_v = _val(original, field), _val(candidate, field)
        if _present(o_v) and _present(c_v) and str(o_v) == str(c_v):
            score += weight
            reasons.append(f"{field} match: {o_v}")

    score = min(score, 1.0)
    return MatchResult(score, tuple(reasons), False, None)


# ---------------------------------------------------------------------------
# Segment DoM helper (used to size the dynamic search window)
# ---------------------------------------------------------------------------


def compute_segment_dom_median(
    listings_df: pd.DataFrame,
) -> dict[tuple, float]:
    """Median days-on-market for sold listings per segment.

    Returns a map keyed on (brand, model, generation_or_None) with a
    (brand, model, None) fallback entry — the same shape
    ``decision.build_context.dom_median`` produces, but without that
    function's other machinery (trend, calibration, fast-share). Used
    by ``find_relists`` to size its dynamic search window per segment;
    self-contained so re-listing detection has no dependency on the
    decision algorithm module.

    Empty map when the input lacks the columns needed for sold
    detection or has no sold rows — ``find_relists`` then uses
    DEFAULT_WINDOW_DAYS for all segments.
    """
    if listings_df is None or listings_df.empty:
        return {}
    needed = {
        "is_active", "deactivation_reason", "deactivated_at",
        "first_seen_at", "brand", "model",
    }
    if not needed.issubset(listings_df.columns):
        return {}

    df = listings_df.copy()
    if "duplicate_of" in df.columns:
        df = df[df["duplicate_of"].isna()]
    is_active = df["is_active"].astype(bool)
    reason = df["deactivation_reason"].astype(str)
    sold = df[(~is_active) & (reason == "sold")].copy()
    if sold.empty:
        return {}

    first = pd.to_datetime(sold["first_seen_at"], errors="coerce", utc=True)
    last = pd.to_datetime(sold["deactivated_at"], errors="coerce", utc=True)
    sold["__dom"] = (last - first).dt.total_seconds() / 86400
    sold = sold[
        (sold["__dom"].notna())
        & (sold["__dom"] >= 0)
        & (sold["__dom"] <= 365)
    ]
    if sold.empty:
        return {}

    if "generation" not in sold.columns:
        sold["generation"] = pd.NA

    out: dict[tuple, float] = {}
    grouped = sold.groupby(["brand", "model", "generation"], dropna=False)
    for (b, m, g), grp in grouped:
        gen_key = g if (g is not None and pd.notna(g) and str(g) != "") else None
        out[(b, m, gen_key)] = float(grp["__dom"].median())
    # (brand, model, None) fallback for callers that didn't capture generation
    grouped_bm = sold.groupby(["brand", "model"], dropna=False)
    for (b, m), grp in grouped_bm:
        key = (b, m, None)
        if key not in out:
            out[key] = float(grp["__dom"].median())
    return out


# ---------------------------------------------------------------------------
# Detection driver
# ---------------------------------------------------------------------------


def find_relists(
    listings_df: pd.DataFrame,
    dom_median_by_segment: Mapping[tuple, float] | None = None,
    *,
    threshold: float = DEFAULT_MATCH_THRESHOLD,
) -> pd.DataFrame:
    """Scan a listings dataframe for re-listing pairs.

    For each row with ``deactivation_reason == "sold"``, search for
    candidate listings of the same brand+model+year whose
    ``first_seen_at`` falls within the segment's adaptive window
    (4 × DoM, clamped) after the original's ``deactivated_at``. The
    highest-scoring candidate above ``threshold`` is recorded as a
    re-listing — ties are broken by smaller gap.

    Returns columns:
        original_olx_id, relist_olx_id, gap_days, match_score,
        original_price_eur, relist_price_eur, price_delta_eur,
        price_delta_pct, mileage_delta_km, match_reasons (list[str]),
        window_days_used.

    *dom_median_by_segment* is the same map produced by
    ``decision.build_context`` — keyed on (brand, model, generation_or_None).
    Missing entries fall back to (brand, model, None) and then to
    DEFAULT_WINDOW_DAYS.
    """
    if listings_df is None or listings_df.empty:
        return _empty_relist_df()

    df = listings_df.copy()
    if "duplicate_of" in df.columns:
        df = df[df["duplicate_of"].isna()].copy()

    # Normalise timestamp columns so date math is consistent across
    # mixed tz-naive / tz-aware inputs (production DB stores naive UTC).
    df["__deact"] = pd.to_datetime(df.get("deactivated_at"), errors="coerce", utc=True)
    df["__first_seen"] = pd.to_datetime(df.get("first_seen_at"), errors="coerce", utc=True)

    is_active = (
        df["is_active"].astype(bool)
        if "is_active" in df.columns
        else pd.Series(True, index=df.index)
    )
    reason = (
        df["deactivation_reason"].astype(str)
        if "deactivation_reason" in df.columns
        else pd.Series("", index=df.index)
    )
    originals = df[(~is_active) & (reason == "sold") & df["__deact"].notna()].copy()
    if originals.empty:
        return _empty_relist_df()

    # Index every row with a known first_seen by (brand, model, year)
    # for O(1) per-segment candidate lookup.
    df_with_dates = df[df["__first_seen"].notna()].copy()
    by_key: dict[tuple, list] = {}
    for idx in df_with_dates.index:
        key = (
            df_with_dates.at[idx, "brand"],
            df_with_dates.at[idx, "model"],
            df_with_dates.at[idx, "year"],
        )
        by_key.setdefault(key, []).append(idx)

    rows: list[dict] = []
    dom_map = dom_median_by_segment or {}

    for orig_idx, orig in originals.iterrows():
        key = (orig["brand"], orig["model"], orig["year"])
        candidate_indices = by_key.get(key, [])
        if not candidate_indices:
            continue

        gen = orig.get("generation") if "generation" in orig else None
        gen_key = gen if (gen is not None and pd.notna(gen) and str(gen) != "") else None
        dom_median = dom_map.get((orig["brand"], orig["model"], gen_key))
        if dom_median is None:
            dom_median = dom_map.get((orig["brand"], orig["model"], None))
        window_days = _segment_window_days(dom_median)

        deact = orig["__deact"]
        window_end = deact + pd.Timedelta(days=window_days)

        # (score, cand_idx, MatchResult, gap_days)
        best: tuple[float, object, MatchResult, float] | None = None

        for cand_idx in candidate_indices:
            if cand_idx == orig_idx:
                continue
            cand = df_with_dates.loc[cand_idx]
            first_seen = cand["__first_seen"]
            if pd.isna(first_seen) or first_seen <= deact or first_seen > window_end:
                continue

            gap_days = (first_seen - deact).total_seconds() / 86400
            result = compute_match_score(orig, cand, gap_days)
            if result.rejected or result.score < threshold:
                continue
            if best is None or result.score > best[0] or (
                result.score == best[0] and gap_days < best[3]
            ):
                best = (result.score, cand_idx, result, gap_days)

        if best is None:
            continue

        score, cand_idx, result, gap_days = best
        cand = df_with_dates.loc[cand_idx]
        orig_price = orig.get("price_eur")
        cand_price = cand.get("price_eur")
        delta_eur: float | None = None
        delta_pct: float | None = None
        if _present(orig_price) and _present(cand_price):
            delta_eur = float(cand_price) - float(orig_price)
            if float(orig_price) > 0:
                delta_pct = delta_eur / float(orig_price) * 100

        mileage_delta = None
        if _present(orig.get("mileage_km")) and _present(cand.get("mileage_km")):
            mileage_delta = int(float(cand["mileage_km"]) - float(orig["mileage_km"]))

        rows.append({
            "original_olx_id": orig.get("olx_id"),
            "relist_olx_id": cand.get("olx_id"),
            "gap_days": round(gap_days, 1),
            "match_score": round(score, 3),
            "original_price_eur": (
                float(orig_price) if _present(orig_price) else None
            ),
            "relist_price_eur": (
                float(cand_price) if _present(cand_price) else None
            ),
            "price_delta_eur": (
                round(delta_eur, 0) if delta_eur is not None else None
            ),
            "price_delta_pct": (
                round(delta_pct, 1) if delta_pct is not None else None
            ),
            "mileage_delta_km": mileage_delta,
            "match_reasons": list(result.reasons),
            "window_days_used": window_days,
        })

    if not rows:
        return _empty_relist_df()
    return pd.DataFrame(rows)


def _empty_relist_df() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "original_olx_id", "relist_olx_id", "gap_days", "match_score",
        "original_price_eur", "relist_price_eur",
        "price_delta_eur", "price_delta_pct",
        "mileage_delta_km", "match_reasons", "window_days_used",
    ])


# ---------------------------------------------------------------------------
# Outcomes adapter for calibrate_thresholds
# ---------------------------------------------------------------------------


def build_outcomes_df(
    relist_df: pd.DataFrame,
    *,
    fees_pct: float = 0.03,
    flat_fees_eur: float = 200.0,
    min_score: float = DEFAULT_MATCH_THRESHOLD,
    min_gap_days: float = 7.0,
    min_abs_delta_pct: float = 5.0,
    max_abs_delta_pct: float = 100.0,
    require_both_sides_sold: bool = True,
    listings_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Convert detected re-listings into the shape ``calibrate_thresholds``
    expects.

    Maps:
      - ``buy_price``  ← original_price_eur
      - ``sell_price`` ← relist_price_eur
      - ``days_held``  ← gap_days
      - ``fees_eur``   ← buy_price × fees_pct + flat_fees_eur
      - ``olx_id``     ← original_olx_id  (the listing a hypothetical
                         flipper would have purchased)

    Filtering is tighter than the underlying matcher because most
    detected re-listings are dealer reposts of the same unit at the
    same price, not actual resales. The 2026-05 production cohort
    showed:

      - 67 % of matches have gap ≤ 7 days (mostly same-day or
        next-day reposts triggered by ``mark_inactive`` false-sold)
      - median price_delta_pct ≈ 0 % (no real margin)
      - long tail of price_delta_pct > 100 % is price-parsing
        artefacts (currency / mileage-as-price typos), not flips

    Defaults strip those out: ``min_gap_days=7``, ``min_abs_delta_pct=5``,
    ``max_abs_delta_pct=100``. With the 2026-05 cohort the strict
    defaults yield ~65 events vs the 459 raw matches — a much cleaner
    signal for ``calibrate_thresholds``.

    *require_both_sides_sold* (default True) keeps only events where
    the re-listed candidate has itself become inactive with reason
    ``sold``, i.e. the proxy resale actually closed. Active re-lists
    give a ceiling estimate (still listed, may not transact at ask)
    and are less reliable as ground truth — flip to False only when
    the strict-sold dataset is too small.
    """
    cols = ["olx_id", "buy_price", "sell_price", "days_held", "fees_eur"]
    if relist_df is None or relist_df.empty:
        return pd.DataFrame(columns=cols)

    abs_delta = relist_df["price_delta_pct"].abs()
    mask = (
        (relist_df["match_score"].fillna(0) >= min_score)
        & relist_df["original_price_eur"].notna()
        & relist_df["relist_price_eur"].notna()
        & (relist_df["original_price_eur"].fillna(0) > 0)
        & (relist_df["gap_days"].fillna(0) >= min_gap_days)
        & relist_df["price_delta_pct"].notna()
        & (abs_delta >= min_abs_delta_pct)
        & (abs_delta <= max_abs_delta_pct)
    )
    df = relist_df[mask].copy()

    if require_both_sides_sold:
        if listings_df is None or listings_df.empty:
            return pd.DataFrame(columns=cols)
        if "olx_id" not in listings_df.columns:
            return pd.DataFrame(columns=cols)
        active = (
            listings_df["is_active"].astype(bool)
            if "is_active" in listings_df.columns
            else pd.Series(True, index=listings_df.index)
        )
        reason = (
            listings_df["deactivation_reason"].astype(str)
            if "deactivation_reason" in listings_df.columns
            else pd.Series("", index=listings_df.index)
        )
        sold_ids = set(
            listings_df.loc[(~active) & (reason == "sold"), "olx_id"].astype(str)
        )
        df = df[df["relist_olx_id"].astype(str).isin(sold_ids)]

    if df.empty:
        return pd.DataFrame(columns=cols)

    out = pd.DataFrame({
        "olx_id": df["original_olx_id"].astype(str),
        "buy_price": df["original_price_eur"].astype(float),
        "sell_price": df["relist_price_eur"].astype(float),
        "days_held": df["gap_days"].astype(float),
    })
    out["fees_eur"] = (out["buy_price"] * fees_pct + flat_fees_eur).round(0)
    return out
