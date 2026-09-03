import datetime as dt

import pandas as pd

from src.analytics.model_pages import (
    MIN_MODEL_N,
    MIN_WINDOW_PAGE_N,
    MIN_YEAR_PAGE_N,
    RETIRE_WINDOW_PAGE_N,
    WINDOW_DAYS,
    build_model_pages,
)

NOW = pd.Timestamp.now(tz="UTC")


def _rows(year, n, active, days_ago=None, start=0, price=7000):
    seen = None if days_ago is None else (NOW - pd.Timedelta(days=days_ago)).isoformat()
    return [{
        "brand": "Volkswagen", "model": "Golf", "year": year,
        "price_eur": price + (i % 7) * 100, "mileage_km": 150000 + (i % 5) * 1000,
        "fuel_type": "Diesel", "district": "Porto", "transmission": "Manual",
        "olx_id": f"g{year}{start + i}", "is_active": active,
        "last_seen_at": seen if not active else NOW.isoformat(),
        "deactivated_at": seen if not active else None,
    } for i in range(n)]


def _golf():
    return build_model_pages(pd.DataFrame(
        _rows(2016, MIN_YEAR_PAGE_N + 2, True)
        + _rows(2014, MIN_MODEL_N, True, start=100)
        + _rows(2010, 3, True, start=200)
        + _rows(2010, MIN_WINDOW_PAGE_N, False, days_ago=40, start=300, price=4000)
        + _rows(2008, 2, True, start=400)
        + _rows(2008, MIN_WINDOW_PAGE_N + 5, False, days_ago=WINDOW_DAYS + 30, start=500, price=3000)
    ))["models"]["volkswagen-golf"]


def _cell(rec, year):
    return next((c for c in rec["yr"] if c["y"] == year), None)


def test_window_cell_gives_a_page_to_a_year_thin_in_active_but_deep_in_six_months():
    rec = _golf()
    c = _cell(rec, 2010)
    assert c and c["pg"] == 1 and c["w"] == WINDOW_DAYS
    assert c["n"] == MIN_WINDOW_PAGE_N + 3 and c["na"] == 3
    assert c["fm"] < 5000


def test_active_years_keep_their_active_cell_and_no_window_flag():
    rec = _golf()
    for y in (2016, 2014):
        c = _cell(rec, y)
        assert c and c.get("pg") == 1 and "w" not in c


def test_listings_closed_before_the_window_do_not_make_a_page():
    rec = _golf()
    assert _cell(rec, 2008) is None
    assert not any(isinstance(c["y"], str) and "2008" in c["y"] and c.get("pg") for c in rec["yr"])


def test_no_year_appears_twice_after_merging_window_cells():
    rec = _golf()
    years = [c["y"] for c in rec["yr"]]
    assert len(years) == len(set(years))
    assert not any(isinstance(y, str) and "2010" in y for y in years)


def test_window_page_survives_on_the_retirement_floor_only_when_previously_published():
    thin = (_rows(2014, MIN_MODEL_N, True, start=100)
            + _rows(2010, RETIRE_WINDOW_PAGE_N, False, days_ago=20, start=300))
    fresh = build_model_pages(pd.DataFrame(thin))["models"]["volkswagen-golf"]
    assert _cell(fresh, 2010) is None or not _cell(fresh, 2010).get("pg")
    published = {"models": {"volkswagen-golf": {"yr": [{"y": 2010, "n": 25, "pg": 1, "w": WINDOW_DAYS}]}}}
    kept = build_model_pages(pd.DataFrame(thin), published=published)["models"]["volkswagen-golf"]
    assert _cell(kept, 2010) and _cell(kept, 2010)["pg"] == 1


def test_frames_without_time_columns_still_build():
    rows = [{k: v for k, v in r.items() if k not in ("last_seen_at", "deactivated_at")}
            for r in _rows(2014, MIN_MODEL_N, True)]
    rec = build_model_pages(pd.DataFrame(rows))["models"]["volkswagen-golf"]
    assert _cell(rec, 2014)["pg"] == 1
