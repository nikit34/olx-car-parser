"""Computed columns added to listings DataFrame."""

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


def enrich_listings(df: pd.DataFrame) -> pd.DataFrame:
    df = add_days_on_market(df)
    df = add_price_changes(df)
    df = add_eur_per_km(df)
    return df
