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
    for field in ("suspicious_signs", "extras", "issues", "tuning_or_mods", "recent_maintenance"):
        if field in df.columns:
            df[f"{field}_count"] = df[field].apply(_json_list_len)
    if "reason_for_sale" in df.columns:
        df["has_reason_for_sale"] = df["reason_for_sale"].notna() & (df["reason_for_sale"] != "")
    return df


def enrich_listings(df: pd.DataFrame) -> pd.DataFrame:
    df = add_days_on_market(df)
    df = add_price_changes(df)
    df = add_eur_per_km(df)
    df = add_list_counts(df)
    return df
