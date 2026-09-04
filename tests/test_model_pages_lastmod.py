import pandas as pd

from src.analytics.model_pages import MIN_MODEL_N, MIN_YEAR_PAGE_N, build_model_pages


def _rows(year, n, price=7000, start=0):
    return [{
        "brand": "Volkswagen", "model": "Golf", "year": year, "is_active": True,
        "price_eur": price + (i % 7) * 100, "mileage_km": 150000 + (i % 5) * 1000,
        "fuel_type": "Diesel", "district": "Porto", "transmission": "Manual",
        "olx_id": f"g{year}{start + i}",
    } for i in range(n)]


def _frame(price_2014=7000):
    return pd.DataFrame(_rows(2016, MIN_YEAR_PAGE_N + 2, start=100) + _rows(2014, MIN_MODEL_N, price=price_2014, start=200))


def test_first_build_stamps_model_and_year_pages_with_the_build_date():
    doc = build_model_pages(_frame(), today="2026-09-04")
    rec = doc["models"]["volkswagen-golf"]
    assert rec["u"] == "2026-09-04"
    assert all(c["u"] == "2026-09-04" for c in rec["yr"] if c.get("pg"))


def test_unchanged_numbers_keep_the_previous_stamp():
    first = build_model_pages(_frame(), today="2026-09-01")
    second = build_model_pages(_frame(), published=first, today="2026-09-04")
    rec = second["models"]["volkswagen-golf"]
    assert rec["u"] == "2026-09-01"
    assert all(c["u"] == "2026-09-01" for c in rec["yr"] if c.get("pg"))


def test_a_changed_year_median_refreshes_that_cell_and_the_model():
    first = build_model_pages(_frame(), today="2026-09-01")
    second = build_model_pages(_frame(price_2014=9000), published=first, today="2026-09-04")
    rec = second["models"]["volkswagen-golf"]
    by_year = {c["y"]: c for c in rec["yr"]}
    assert rec["u"] == "2026-09-04"
    assert by_year[2014]["u"] == "2026-09-04"
    assert by_year[2016]["u"] == "2026-09-01"


def test_previous_blob_without_stamps_degrades_to_the_build_date():
    first = build_model_pages(_frame(), today="2026-09-01")
    for rec in first["models"].values():
        rec.pop("u", None)
        for c in rec.get("yr", []):
            c.pop("u", None)
    second = build_model_pages(_frame(), published=first, today="2026-09-04")
    assert second["models"]["volkswagen-golf"]["u"] == "2026-09-04"
