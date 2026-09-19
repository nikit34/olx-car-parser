"""Turnover: the days-to-sell figure the deal scorer multiplies by.

``avg_days_to_sell`` feeds the liquidity multiplier in ``data_loader``, and it
used to be the mean over listings that had merely disappeared — which reads
fast twice over: the listings still on sale were dropped, and a listing that
came back a fortnight later as a new advert counted as a sale. ``weekly_turnover``
rides along in the same table and nothing reads it today; it is pinned here so
that stays a deliberate choice rather than a silent regression.
"""

from __future__ import annotations

import pandas as pd

from src.analytics.turnover import compute_turnover_stats

NOW = pd.Timestamp.now().normalize()


def _rows(n, days, active=False, start=0, brand="Volkswagen", model="Golf",
          generation="Mk7"):
    out = []
    for i in range(n):
        first = NOW - pd.Timedelta(days=days + 1)
        seen = first + pd.Timedelta(days=days)
        out.append({
            "olx_id": f"t{start + i}", "brand": brand, "model": model,
            "generation": generation, "is_active": active,
            "price_eur": 9000.0, "year": 2015, "district": "Porto",
            "first_seen_at": first,
            "last_seen_at": seen,
            "last_scraped_at": seen,
            "deactivated_at": pd.NaT if active else seen,
        })
    return out


def _df(*groups):
    return pd.DataFrame([r for g in groups for r in g])


def test_days_to_sell_counts_the_listings_still_on_sale():
    ended = _rows(30, days=10) + _rows(20, days=50, start=100)
    live = _rows(25, days=100, active=True, start=500)
    only_ended = compute_turnover_stats(_df(ended))
    with_live = compute_turnover_stats(_df(ended, live))
    assert only_ended["avg_days_to_sell"].iloc[0] == 10
    assert with_live["avg_days_to_sell"].iloc[0] == 50


def test_a_listing_that_came_back_is_not_a_sale():
    ended = _rows(40, days=10) + _rows(30, days=60, start=100)
    live = _rows(10, days=120, active=True, start=500)
    back = {r["olx_id"] for r in ended[:20]}
    plain = compute_turnover_stats(_df(ended, live))
    fixed = compute_turnover_stats(_df(ended, live), relisted=back)
    assert plain["avg_days_to_sell"].iloc[0] == 10
    assert fixed["avg_days_to_sell"].iloc[0] == 60
    assert fixed["weekly_turnover"].iloc[0] < plain["weekly_turnover"].iloc[0]


def test_a_thin_segment_gets_no_number_rather_than_a_noisy_one():
    out = compute_turnover_stats(_df(_rows(3, days=10), _rows(5, days=90, active=True, start=500)))
    assert out["avg_days_to_sell"].isna().all()


def test_an_empty_frame_keeps_the_shape():
    out = compute_turnover_stats(pd.DataFrame())
    assert list(out.columns) == ["brand", "model", "generation",
                                 "avg_days_to_sell", "weekly_turnover"]
    assert out.empty
