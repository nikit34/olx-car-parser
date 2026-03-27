"""Generate realistic demo data for dashboard development."""

import random
from datetime import datetime, timedelta

import pandas as pd

BRANDS_MODELS = {
    "Toyota": ["Camry", "RAV4", "Corolla"],
    "Volkswagen": ["Passat", "Golf", "Tiguan"],
    "Skoda": ["Octavia", "Superb", "Kodiaq"],
    "Hyundai": ["Tucson", "Sonata", "Elantra"],
    "Kia": ["Sportage", "Optima", "Cerato"],
    "Honda": ["Civic", "CR-V", "Accord"],
    "Mazda": ["CX-5", "6", "3"],
    "Nissan": ["Qashqai", "X-Trail", "Leaf"],
}

CITIES = [
    "Kyiv", "Kharkiv", "Odesa", "Dnipro", "Lviv",
    "Zaporizhzhia", "Vinnytsia", "Poltava", "Chernihiv", "Mykolaiv",
]

ENGINE_TYPES = ["petrol", "diesel", "gas", "hybrid", "electric"]
TRANSMISSIONS = ["manual", "automatic"]
BODY_TYPES = ["sedan", "suv", "hatchback", "wagon", "crossover"]

# Base prices USD by segment
BASE_PRICES = {
    "Camry": 14000, "RAV4": 18000, "Corolla": 10000,
    "Passat": 12000, "Golf": 9000, "Tiguan": 16000,
    "Octavia": 11000, "Superb": 14000, "Kodiaq": 17000,
    "Tucson": 15000, "Sonata": 12000, "Elantra": 9000,
    "Sportage": 14000, "Optima": 11000, "Cerato": 8500,
    "Civic": 10000, "CR-V": 15000, "Accord": 12000,
    "CX-5": 16000, "6": 11000, "3": 9000,
    "Qashqai": 13000, "X-Trail": 14000, "Leaf": 11000,
}


def _price_for_year(base: float, year: int) -> float:
    age = 2026 - year
    depreciation = max(0.3, 1.0 - age * 0.08)
    return base * depreciation


def _seasonal_factor(date: datetime) -> float:
    """Spring/summer prices slightly higher."""
    month = date.month
    if month in (3, 4, 5):
        return 1.04
    elif month in (6, 7, 8):
        return 1.02
    elif month in (11, 12, 1):
        return 0.95
    return 1.0


def generate_listings(n: int = 500, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    rows = []
    for i in range(n):
        brand = random.choice(list(BRANDS_MODELS.keys()))
        model = random.choice(BRANDS_MODELS[brand])
        year = random.randint(2012, 2024)
        base = BASE_PRICES[model]
        price_usd = _price_for_year(base, year)
        price_usd *= random.uniform(0.8, 1.25)
        price_usd = round(price_usd / 100) * 100

        rows.append({
            "olx_id": f"olx-{100000 + i}",
            "brand": brand,
            "model": model,
            "year": year,
            "price_usd": price_usd,
            "mileage_km": random.randint(20, 350) * 1000,
            "engine_type": random.choice(ENGINE_TYPES),
            "engine_volume": round(random.choice([1.4, 1.6, 1.8, 2.0, 2.2, 2.5, 3.0]), 1),
            "transmission": random.choice(TRANSMISSIONS),
            "body_type": random.choice(BODY_TYPES),
            "city": random.choice(CITIES),
            "seller_type": random.choices(["private", "dealer"], weights=[0.7, 0.3])[0],
            "is_active": random.random() > 0.15,
        })
    return pd.DataFrame(rows)


def generate_price_history(days: int = 90, seed: int = 42) -> pd.DataFrame:
    """Generate daily market stats for each model over N days."""
    random.seed(seed)
    rows = []
    end_date = datetime(2026, 3, 27)
    start_date = end_date - timedelta(days=days)

    for brand, models in BRANDS_MODELS.items():
        for model in models:
            base = BASE_PRICES[model]
            # Random walk for market trend
            trend = 0.0
            for day_offset in range(days + 1):
                date = start_date + timedelta(days=day_offset)
                trend += random.uniform(-0.005, 0.004)  # slight downward bias
                trend = max(-0.2, min(0.15, trend))
                seasonal = _seasonal_factor(date)
                median = base * (1 + trend) * seasonal
                noise = random.uniform(-0.03, 0.03)

                rows.append({
                    "brand": brand,
                    "model": model,
                    "date": date.date(),
                    "median_price_usd": round(median * (1 + noise)),
                    "avg_price_usd": round(median * (1 + noise + 0.02)),
                    "min_price_usd": round(median * 0.65),
                    "max_price_usd": round(median * 1.4),
                    "listing_count": random.randint(15, 120),
                })
    return pd.DataFrame(rows)


def generate_buy_signals(listings_df: pd.DataFrame, history_df: pd.DataFrame) -> pd.DataFrame:
    """Find listings priced significantly below rolling average."""
    signals = []
    latest_stats = history_df.sort_values("date").groupby(["brand", "model"]).last().reset_index()

    for _, listing in listings_df[listings_df["is_active"]].iterrows():
        stat = latest_stats[
            (latest_stats["brand"] == listing["brand"])
            & (latest_stats["model"] == listing["model"])
        ]
        if stat.empty:
            continue
        median = stat.iloc[0]["median_price_usd"]
        if listing["price_usd"] < median * 0.85:
            discount = round((1 - listing["price_usd"] / median) * 100, 1)
            signals.append({
                "olx_id": listing["olx_id"],
                "brand": listing["brand"],
                "model": listing["model"],
                "year": listing["year"],
                "price_usd": listing["price_usd"],
                "median_price_usd": round(median),
                "discount_pct": discount,
                "city": listing["city"],
                "mileage_km": listing["mileage_km"],
                "engine_type": listing["engine_type"],
            })

    df = pd.DataFrame(signals)
    if not df.empty:
        df = df.sort_values("discount_pct", ascending=False)
    return df
