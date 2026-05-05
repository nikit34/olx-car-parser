"""Computed columns added to listings DataFrame."""

import json

import pandas as pd


def add_days_on_market(df: pd.DataFrame) -> pd.DataFrame:
    if "first_seen_at" not in df.columns:
        return df
    now = pd.Timestamp.now()
    first = pd.to_datetime(df["first_seen_at"]).dt.tz_localize(None)
    df["days_listed"] = (now - first).dt.days
    return df


def add_price_changes(df: pd.DataFrame) -> pd.DataFrame:
    if "first_price_eur" not in df.columns or "price_eur" not in df.columns:
        return df
    df["price_change_eur"] = df["price_eur"] - df["first_price_eur"]
    df["price_change_pct"] = pd.to_numeric(
        df["price_change_eur"] / df["first_price_eur"].replace(0, pd.NA) * 100,
        errors="coerce",
    ).round(1)
    return df


def add_eur_per_km(df: pd.DataFrame) -> pd.DataFrame:
    if "price_eur" not in df.columns or "mileage_km" not in df.columns:
        return df
    df["eur_per_km"] = pd.to_numeric(
        df["price_eur"] / df["mileage_km"].replace(0, pd.NA),
        errors="coerce",
    ).round(3)
    return df


def _json_list_len(val) -> int:
    """Count items in a JSON list string, return 0 for null/empty/invalid."""
    if not val or (isinstance(val, float) and pd.isna(val)):
        return 0
    if isinstance(val, list):
        return len(val)
    try:
        parsed = json.loads(val)
        return len(parsed) if isinstance(parsed, list) else 0
    except (json.JSONDecodeError, TypeError):
        return 0


def add_list_counts(df: pd.DataFrame) -> pd.DataFrame:
    if "tuning_or_mods" in df.columns:
        df["tuning_or_mods_count"] = df["tuning_or_mods"].apply(_json_list_len)
    return df


_PLATE_KEYS = ("plate_readable", "plate_n_readable", "plate_text_primary")

# Minimum exterior photos before "no plate readable on any of them" is a
# meaningful signal. Below this, ``plate_readable=False`` is mostly an
# artefact of small / interior-heavy photo sets — the human-labelled pilot
# (37 listings, 2026-05-05) showed obscuring rates ranging 20% on
# 11-20-photo listings up to 67% on 6-10-photo listings, with the lower-
# photo bucket dominated by "no plate angle in frame" noise rather than
# deliberate obscuring. Threshold of 5 is a conservative balance between
# coverage (enough listings make the cut) and signal-to-noise (low photo
# counts can't reliably express a "deliberately hidden" intent).
_PLATE_OBSCURED_MIN_EXTERIOR = 5


def _extract_plate_fields(
    raw,
) -> tuple[bool | None, int | None, str | None, bool | None]:
    """Pull listing-level plate signals from an ``llm_extras`` cell.

    Returns ``(plate_readable, plate_n_readable, plate_text_primary,
    plate_obscured)`` — each ``None`` when the listing hasn't been through
    ``verify-photos`` yet OR when the JSON is malformed. Distinguishing
    "verified, no plate found" from "not yet verified" matters for
    feature use: the former is a real signal, the latter is missing data.

    ``plate_obscured`` is the tri-state derived signal:
        True  — verified, ≥``_PLATE_OBSCURED_MIN_EXTERIOR`` exterior photos
                were scored, AND none surfaced a readable PT plate. The
                exterior-photo threshold is the FPR guard from the pilot
                study: under it, "no plate readable" mostly reflects a
                small or interior-biased photo set rather than a
                deliberately hidden plate.
        False — verified, plate was readable on at least one photo.
        None  — not yet verified, OR exterior photo count below threshold
                (signal undefined; downstream code treats as missing).
    """
    if not raw or (isinstance(raw, float) and pd.isna(raw)):
        return None, None, None, None
    if isinstance(raw, dict):
        data = raw
    else:
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None, None, None, None
        if not isinstance(data, dict):
            return None, None, None, None
    # ``plate_readable`` is the canonical "has the plate reader run yet?"
    # marker — verify-photos always writes it (True or False) when it
    # processes a row, so its absence == not yet processed.
    if "plate_readable" not in data:
        return None, None, None, None
    readable = bool(data.get("plate_readable"))
    n_readable = data.get("plate_n_readable")
    if isinstance(n_readable, bool) or not isinstance(n_readable, int):
        n_readable = int(bool(readable))  # backstop for malformed legacy rows
    primary = data.get("plate_text_primary")
    if primary is not None and not isinstance(primary, str):
        primary = None
    # plate_obscured: only meaningful when the listing had enough exterior
    # photos to plausibly contain a plate. Pulled from the same JSON since
    # verify-photos writes ``photo_damage_n_exterior`` alongside the plate
    # fields. Fall through to None when the count isn't there (legacy rows
    # written before issue #3 added the field).
    n_exterior = data.get("photo_damage_n_exterior")
    if isinstance(n_exterior, bool) or not isinstance(n_exterior, int):
        obscured: bool | None = None
    elif n_exterior < _PLATE_OBSCURED_MIN_EXTERIOR:
        obscured = None  # too few exterior photos to make a call
    else:
        obscured = not readable
    return readable, n_readable, primary, obscured


def add_plate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Promote plate signals from ``llm_extras`` to columns.

    Photo-damage signals stay in ``llm_extras`` (consumers parse on
    demand) — but the plate signal is binary, lightweight, and explicitly
    intended as a downstream feature ("listing where no photo shows a
    readable plate" is a soft suspicion hint). Promoting it to columns
    means hazard / decision / anomaly / price_model modules can use
    ``df["plate_readable"]`` directly instead of every consumer
    reimplementing the JSON walk.

    Adds four columns:
      • ``plate_readable`` — nullable bool. ``None`` for not-yet-verified.
      • ``plate_n_readable`` — nullable int.
      • ``plate_text_primary`` — nullable str.
      • ``plate_obscured`` — nullable bool. Derived: ``plate_readable=False``
        AND ``photo_damage_n_exterior >= _PLATE_OBSCURED_MIN_EXTERIOR``.
        See ``_PLATE_OBSCURED_MIN_EXTERIOR`` for the threshold rationale.
    No-ops when ``llm_extras`` isn't present (e.g. unmatched listings df).
    """
    if "llm_extras" not in df.columns:
        return df
    quads = df["llm_extras"].apply(_extract_plate_fields)
    df["plate_readable"] = [t[0] for t in quads]
    df["plate_n_readable"] = [t[1] for t in quads]
    df["plate_text_primary"] = [t[2] for t in quads]
    df["plate_obscured"] = [t[3] for t in quads]
    return df


def enrich_listings(df: pd.DataFrame) -> pd.DataFrame:
    df = add_days_on_market(df)
    df = add_price_changes(df)
    df = add_eur_per_km(df)
    df = add_list_counts(df)
    df = add_plate_signals(df)
    return df
