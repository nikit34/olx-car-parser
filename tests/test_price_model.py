"""Tests for LightGBM price model."""

import json
import numpy as np
import pandas as pd
import pytest
from sklearn.isotonic import IsotonicRegression

from src.analytics.price_model import (
    train_price_model, predict_prices, save_model, load_model,
    load_metrics_history, _METRICS_PATH, _MODEL_PATH, _SCHEMA_VERSION,
)


def _sample_listings(n: int = 200, with_first_seen_at: bool = False) -> pd.DataFrame:
    """Generate synthetic listings with realistic price patterns."""
    rng = np.random.RandomState(42)
    years = rng.randint(2008, 2024, size=n)
    mileage = rng.randint(10000, 300000, size=n)
    engine_cc = rng.choice([1000, 1200, 1400, 1600, 2000], size=n)
    brands = rng.choice(["Volkswagen", "Renault", "BMW"], size=n)
    models = []
    for b in brands:
        if b == "Volkswagen":
            models.append(rng.choice(["Golf", "Polo"]))
        elif b == "Renault":
            models.append(rng.choice(["Clio", "Megane"]))
        else:
            models.append("320d")

    # Synthetic Portuguese-ish title and description so the TF-IDF pipeline
    # has actual tokens to fit on. Real fixture would be much more varied;
    # this just needs enough vocabulary diversity (and min_df=3 satisfied)
    # for the pipeline to train and persist.
    _CONDITIONS = ["bom estado", "excelente", "muito bom", "carro de família",
                   "primeiro dono", "pouco usado", "diesel económico"]
    titles = [f"{b} {m} {y}" for b, m, y in zip(brands, models, years)]
    descriptions = [
        f"Vendo {b} {m} de {y}, {rng.choice(_CONDITIONS)}, "
        f"{int(rng.randint(50, 200))} mil km, ITV em dia."
        for b, m, y in zip(brands, models, years)
    ]

    accident = rng.choice([True, False, None], size=n, p=[0.05, 0.75, 0.20])
    repair = rng.choice([True, False, None], size=n, p=[0.10, 0.70, 0.20])
    rhd = rng.choice([True, False, None], size=n, p=[0.02, 0.78, 0.20])

    # Price = f(year, mileage, engine, condition) + noise
    price = (
        (years - 2000) * 600
        - mileage * 0.02
        + engine_cc * 2
        + rng.normal(0, 1000, size=n)
    ).clip(min=500)
    # Accident cars are cheaper
    for i in range(n):
        if accident[i] is True:
            price[i] *= 0.7
        if rhd[i] is True:
            price[i] *= 0.8

    out = pd.DataFrame({
        "olx_id": [f"t{i}" for i in range(n)],
        "brand": brands,
        "model": models,
        "year": years,
        "mileage_km": mileage,
        "engine_cc": engine_cc,
        "price_eur": price.round(0),
        "fuel_type": rng.choice(["Diesel", "Gasoline"], size=n),
        "transmission": rng.choice(["Manual", "Automatic"], size=n),
        "segment": "Citadino",
        "horsepower": (engine_cc * 0.07 + rng.normal(0, 10, size=n)).clip(min=50).round(0),
        "is_active": True,
        "title": titles,
        "description": descriptions,
        # LLM fields
        "desc_mentions_accident": accident,
        "desc_mentions_repair": repair,
        "right_hand_drive": rhd,
        "desc_mentions_customs_cleared": rng.choice([True, False, None], size=n, p=[0.15, 0.65, 0.20]),
        "desc_mentions_num_owners": rng.choice([1, 2, 3, None], size=n, p=[0.3, 0.2, 0.1, 0.4]),
        "taxi_fleet_rental": rng.choice([True, False, None], size=n, p=[0.03, 0.77, 0.20]),
        "warranty": rng.choice([True, False, None], size=n, p=[0.15, 0.65, 0.20]),
        "first_owner_selling": rng.choice([True, False, None], size=n, p=[0.20, 0.60, 0.20]),
        "urgency": rng.choice(["high", "medium", "low", None], size=n, p=[0.05, 0.15, 0.50, 0.30]),
    })
    if with_first_seen_at:
        base = pd.Timestamp("2026-01-01", tz="UTC")
        out["first_seen_at"] = [
            base + pd.Timedelta(days=int(d)) for d in rng.randint(0, 60, size=n)
        ]
    return out


def test_train_returns_model_and_metrics():
    # 400 rows so the time-aware CQR helper has enough data after the 80/20
    # calibration split (it requires ≥100 train rows and ≥50 calibration rows
    # post-filter, and bails early below 200 total).
    df = _sample_listings(400, with_first_seen_at=True)
    result = train_price_model(df)
    assert result is not None
    models, cat_maps, metrics, oof_preds, calibrator, _unc = result
    assert "brand" in cat_maps
    assert "model" in cat_maps
    assert "low" in models and "median" in models and "high" in models
    # Metrics from cross-validation
    assert "mae" in metrics and metrics["mae"] > 0
    assert "mape" in metrics and metrics["mape"] > 0
    assert "r2" in metrics
    assert "n_samples" in metrics
    assert "cv_folds" in metrics and metrics["cv_folds"] == 5
    # Per-quantile early-stop iters are tracked separately
    assert "best_n_estimators_per_q" in metrics
    assert set(metrics["best_n_estimators_per_q"]) == {"low", "median", "high"}
    # Log target on by default; conformal_q is in log space and a
    # human-readable percent is also exposed.
    assert metrics.get("log_target") is True
    assert "conformal_q_pct" in metrics
    # Time-aware CQR fields
    assert "conformal_q_random" in metrics
    assert metrics.get("conformal_q_source") == "time"  # first_seen_at present
    assert metrics.get("conformal_q_time") is not None
    assert metrics.get("median_calibrated") is True
    # v4 fields
    assert metrics.get("sample_weighted") is True
    assert metrics.get("monotone_constraints") is True
    assert "conformal_q_per_bucket" in metrics
    assert isinstance(metrics["conformal_q_per_bucket"], dict)
    assert "n_features" in metrics
    # OOF preds keyed by olx_id, one entry per training listing
    assert isinstance(oof_preds, dict)
    assert len(oof_preds) > 0
    sample_id = next(iter(oof_preds))
    lo, med, hi = oof_preds[sample_id]
    # OOF entries are already crossing-repaired and clamped
    assert lo <= med <= hi
    assert lo >= 0
    # Median calibrator returned
    assert isinstance(calibrator, IsotonicRegression)


def test_per_bucket_conformal_q_lookup():
    """_per_row_conformal_q uses bucket-specific q where available, falls
    back to global for buckets without one. Default static-edge fallback."""
    from src.analytics.price_model import _per_row_conformal_q
    predicted = np.array([1500.0, 5000.0, 25000.0, 100000.0])
    per_bucket = {"<€3k": 0.10, "€3–7k": 0.08}  # cheap buckets only
    out = _per_row_conformal_q(predicted, global_q=0.05, per_bucket_q=per_bucket)
    assert out[0] == 0.10  # <€3k → bucket-specific
    assert out[1] == 0.08  # €3–7k → bucket-specific
    assert out[2] == 0.05  # €15–30k → no bucket entry, falls back to global
    assert out[3] == 0.05  # €30k+ → no bucket entry, falls back to global


def test_per_bucket_conformal_q_with_dynamic_edges():
    """Dynamic edges from training preds override the static 5-bucket
    fallback. Each row must look up the bucket in the persisted edge list."""
    from src.analytics.price_model import _per_row_conformal_q
    edges = [
        (-float("inf"), 4000.0, "low"),
        (4000.0, 12000.0, "mid"),
        (12000.0, float("inf"), "high"),
    ]
    predicted = np.array([1500.0, 8000.0, 30000.0])
    per_bucket = {"low": 0.20, "mid": 0.05}  # high has no entry
    out = _per_row_conformal_q(
        predicted, global_q=0.10, per_bucket_q=per_bucket, edges=edges,
    )
    assert out[0] == 0.20  # in "low" bucket
    assert out[1] == 0.05  # in "mid" bucket
    assert out[2] == 0.10  # "high" has no per-bucket q → global


def test_uncertainty_bundle_trains_and_predicts_positive_q():
    """Option C: per-row uncertainty model trains on the cal-set residuals
    and yields strictly-positive widening factors at inference time."""
    df = _sample_listings(400, with_first_seen_at=True)
    result = train_price_model(df)
    assert result is not None
    _models, _maps, _metrics, _oof, _calib, uncertainty = result
    assert uncertainty is not None, (
        "400 rows × ≥80 cal split should be enough to fit the uncertainty model"
    )
    unc_model, q_scale = uncertainty
    assert q_scale > 0
    # Predict on the same training distribution and check non-negative output.
    from src.analytics.price_model import _prepare_X
    X, _ = _prepare_X(df.iloc[:30])
    raw_q = unc_model.predict(X)
    # Raw predictions can be slightly negative (regression target is unbounded);
    # the inference helper clamps at 0 and applies q_scale.
    from src.analytics.price_model import _per_row_uncertainty_q
    per_row_q = _per_row_uncertainty_q(X, unc_model, q_scale, floor_q=0.05)
    assert (per_row_q >= 0.05).all()  # never below floor
    assert per_row_q.std() > 0         # genuinely per-row, not a constant


def test_predict_prices_uses_uncertainty_when_present(monkeypatch):
    """When uncertainty_bundle is supplied, it overrides per-bucket q —
    different rows get different band widths from the same model."""
    df = _sample_listings(400, with_first_seen_at=True)
    result = train_price_model(df)
    assert result is not None
    models, cat_maps, metrics, oof_preds, calibrator, uncertainty = result
    assert uncertainty is not None

    # Predict with vs without the uncertainty bundle. The bands should
    # differ — uncertainty path is per-row, bucket path is per-decile.
    _conf_q = metrics.get("conformal_q", 0.0)
    _bucket_q = metrics.get("conformal_q_per_bucket", {})
    _edges = [tuple(e) for e in metrics.get("conformal_q_bucket_edges") or []]

    bucket_path = predict_prices(
        models, cat_maps, df.head(50),
        conformal_q=_conf_q,
        oof_preds={},  # disable OOF override so the band path is exercised
        median_calibrator=calibrator,
        conformal_q_per_bucket=_bucket_q,
        conformal_q_bucket_edges=_edges,
    )
    unc_path = predict_prices(
        models, cat_maps, df.head(50),
        conformal_q=_conf_q,
        oof_preds={},
        median_calibrator=calibrator,
        conformal_q_per_bucket=_bucket_q,
        conformal_q_bucket_edges=_edges,
        uncertainty_bundle=uncertainty,
    )
    # Predicted median is unaffected by widening choice.
    np.testing.assert_allclose(
        bucket_path["predicted_price"], unc_path["predicted_price"],
    )
    # But bands should differ on at least some rows.
    bucket_widths = (bucket_path["fair_price_high"] - bucket_path["fair_price_low"]).values
    unc_widths = (unc_path["fair_price_high"] - unc_path["fair_price_low"]).values
    assert not np.allclose(bucket_widths, unc_widths)


def test_predict_prices_falls_back_to_per_bucket_q_when_uncertainty_none():
    """Backward compat — old bundles without uncertainty_bundle keep
    using the per-bucket lookup. A frame with uncertainty_bundle=None
    must produce identical output to one not passing the param at all."""
    df = _sample_listings(400, with_first_seen_at=True)
    result = train_price_model(df)
    assert result is not None
    models, cat_maps, metrics, _oof, calibrator, _unc = result
    _conf_q = metrics.get("conformal_q", 0.0)
    _bucket_q = metrics.get("conformal_q_per_bucket", {})
    _edges = [tuple(e) for e in metrics.get("conformal_q_bucket_edges") or []]

    a = predict_prices(
        models, cat_maps, df.head(20),
        conformal_q=_conf_q,
        median_calibrator=calibrator,
        conformal_q_per_bucket=_bucket_q,
        conformal_q_bucket_edges=_edges,
    )
    b = predict_prices(
        models, cat_maps, df.head(20),
        conformal_q=_conf_q,
        median_calibrator=calibrator,
        conformal_q_per_bucket=_bucket_q,
        conformal_q_bucket_edges=_edges,
        uncertainty_bundle=None,
    )
    np.testing.assert_allclose(a["fair_price_low"], b["fair_price_low"])
    np.testing.assert_allclose(a["fair_price_high"], b["fair_price_high"])


def test_compute_decile_edges_falls_back_on_small_data():
    """Below the row threshold, decile computation returns the static 5-bucket
    scheme — small synthetic test fixtures shouldn't get noisy 10-bin edges."""
    from src.analytics.price_model import (
        _compute_decile_edges, _DEFAULT_BUCKET_EDGES,
    )
    tiny = np.array([1000.0, 2000.0, 3000.0])
    edges = _compute_decile_edges(tiny)
    assert edges == list(_DEFAULT_BUCKET_EDGES)


def test_compute_decile_edges_produces_n_bins_on_real_data():
    """On enough rows, decile edges yield n_bins buckets covering (-inf, inf)."""
    import numpy as np
    from src.analytics.price_model import _compute_decile_edges
    rng = np.random.RandomState(42)
    prices = rng.lognormal(mean=9.0, sigma=0.6, size=2000)  # ~€8k median
    edges = _compute_decile_edges(prices, n_bins=10)
    assert len(edges) == 10
    # Edges must tile the real line without gaps.
    assert edges[0][0] == -float("inf")
    assert edges[-1][1] == float("inf")
    # And each bucket's high boundary equals the next bucket's low.
    for i in range(len(edges) - 1):
        assert edges[i][1] == edges[i + 1][0]


def test_train_falls_back_to_random_cqr_without_first_seen_at():
    """No first_seen_at column → time-aware q is None, source falls back to random."""
    df = _sample_listings(with_first_seen_at=False)
    assert "first_seen_at" not in df.columns
    _models, _maps, metrics, _oof, _calib, _unc = train_price_model(df)
    assert metrics.get("conformal_q_source") == "random"
    assert metrics.get("conformal_q_time") is None
    # Active conformal_q equals the random one in this fallback path
    assert metrics["conformal_q"] == metrics["conformal_q_random"]


def test_train_returns_none_for_small_data():
    df = _sample_listings(10)
    result = train_price_model(df, min_samples=50)
    assert result is None


def test_predictions_are_positive():
    df = _sample_listings()
    models, cat_maps, _metrics, _oof, _calib, _unc = train_price_model(df)
    preds = predict_prices(models, cat_maps, df)
    assert (preds["predicted_price"] >= 0).all()
    assert (preds["fair_price_low"] >= 0).all()
    assert (preds["fair_price_high"] >= 0).all()
    assert len(preds) == len(df)


def test_predictions_never_cross():
    """low ≤ median ≤ high must hold for every row, with or without OOF override."""
    df = _sample_listings(300)
    models, cat_maps, metrics, oof_preds, calibrator, _unc = train_price_model(df)

    # In-sample (uses OOF preds for every row)
    in_sample = predict_prices(
        models, cat_maps, df,
        conformal_q=metrics.get("conformal_q", 0.0),
        oof_preds=oof_preds,
        median_calibrator=calibrator,
    )
    assert (in_sample["fair_price_low"] <= in_sample["predicted_price"]).all()
    assert (in_sample["predicted_price"] <= in_sample["fair_price_high"]).all()

    # Out-of-sample (synthetic rows the model has never seen, no OOF override).
    # Build new olx_ids so nothing collides with the OOF dict.
    fresh = df.copy()
    fresh["olx_id"] = [f"new_{i}" for i in range(len(fresh))]
    oos = predict_prices(
        models, cat_maps, fresh,
        conformal_q=metrics.get("conformal_q", 0.0),
        oof_preds=oof_preds,
        median_calibrator=calibrator,
    )
    assert (oos["fair_price_low"] <= oos["predicted_price"]).all()
    assert (oos["predicted_price"] <= oos["fair_price_high"]).all()


def test_oof_preds_used_for_known_olx_ids():
    """Listings present in oof_preds get exactly the OOF values, not model.predict."""
    df = _sample_listings(150)
    models, cat_maps, _metrics, oof_preds, calibrator, _unc = train_price_model(df)

    # Pick an olx_id that's in the OOF dict and predict on a single-row df.
    target_id = next(iter(oof_preds))
    row = df[df["olx_id"] == target_id].head(1).copy()
    if row.empty:  # all training rows survived filtering
        target_id = df["olx_id"].iloc[0]
        row = df.head(1).copy()

    expected_low, expected_median, expected_high = oof_preds[str(target_id)]
    preds = predict_prices(
        models, cat_maps, row,
        conformal_q=0.0,
        oof_preds=oof_preds,
        median_calibrator=calibrator,
    )
    # Round-trip through DataFrame.round(0) — compare with rounded expected too
    assert preds["predicted_price"].iloc[0] == round(expected_median)
    assert preds["fair_price_low"].iloc[0] == round(expected_low)
    assert preds["fair_price_high"].iloc[0] == round(expected_high)


def test_calibrator_changes_predictions_for_new_rows():
    """Isotonic calibrator should actually shift the median for new rows.

    Builds a row with no olx_id match (so it bypasses the OOF override) and
    verifies the calibrated prediction differs from the raw model output for
    at least some rows. Both can match if isotonic ends up identity, but
    given heterogeneous training data that's vanishingly rare.
    """
    df = _sample_listings(400)
    models, cat_maps, _metrics, _oof, calibrator, _unc = train_price_model(df)

    fresh = df.head(50).copy()
    fresh["olx_id"] = [f"new_{i}" for i in range(len(fresh))]
    raw = predict_prices(models, cat_maps, fresh, median_calibrator=None)
    cal = predict_prices(models, cat_maps, fresh, median_calibrator=calibrator)
    # At least one row got nudged by the calibration map
    assert not np.allclose(raw["predicted_price"], cal["predicted_price"])


def test_newer_cars_predicted_higher():
    df = _sample_listings(500)
    models, cat_maps, _metrics, _oof, _calib, _unc = train_price_model(df)

    old_car = pd.DataFrame([{
        "year": 2010, "mileage_km": 150000, "engine_cc": 1600,
        "brand": "Volkswagen", "model": "Golf",
        "fuel_type": "Diesel", "transmission": "Manual",
        "segment": "Citadino", "horsepower": 110,
    }])
    new_car = pd.DataFrame([{
        "year": 2022, "mileage_km": 30000, "engine_cc": 1600,
        "brand": "Volkswagen", "model": "Golf",
        "fuel_type": "Diesel", "transmission": "Manual",
        "segment": "Citadino", "horsepower": 110,
    }])

    pred_old = predict_prices(models, cat_maps, old_car)["predicted_price"].iloc[0]
    pred_new = predict_prices(models, cat_maps, new_car)["predicted_price"].iloc[0]
    assert pred_new > pred_old


def test_handles_missing_features():
    """Model should handle NaN in optional features."""
    df = _sample_listings()
    # Drop some optional columns
    df.loc[df.index[:50], "engine_cc"] = np.nan
    df.loc[df.index[:30], "horsepower"] = np.nan

    result = train_price_model(df)
    assert result is not None
    models, cat_maps, _metrics, _oof, _calib, _unc = result

    # Predict on row with missing features
    sparse = pd.DataFrame([{
        "year": 2018, "mileage_km": 100000,
        "brand": "Volkswagen", "model": "Golf",
    }])
    preds = predict_prices(models, cat_maps, sparse)
    assert len(preds) == 1
    assert preds["predicted_price"].iloc[0] > 0


def test_rare_categories_grouped():
    """Rare categories should be grouped into __other__."""
    df = _sample_listings(600)
    # Each model appears exactly once → all below _MIN_CATEGORY_COUNT
    df["model"] = [f"Model_{i}" for i in range(len(df))]

    result = train_price_model(df)
    assert result is not None
    models, cat_maps, _metrics, _oof, _calib, _unc = result
    assert "__other__" in cat_maps["model"]

    unseen = pd.DataFrame([{
        "year": 2018, "mileage_km": 100000, "engine_cc": 1600,
        "brand": "Volkswagen", "model": "Future_Model",
        "fuel_type": "Diesel", "transmission": "Manual",
        "segment": "Citadino", "horsepower": 110,
    }])
    pred = predict_prices(models, cat_maps, unseen)["predicted_price"].iloc[0]
    assert pred > 0


def test_save_and_load_model(tmp_path, monkeypatch):
    """Save/load round-trip produces identical predictions, oof_preds,
    and calibrator."""
    monkeypatch.setattr("src.analytics.price_model._MODEL_PATH", tmp_path / "model.joblib")
    monkeypatch.setattr("src.analytics.price_model._METRICS_PATH", tmp_path / "metrics.json")

    df = _sample_listings()
    models, cat_maps, metrics, oof_preds, calibrator, _unc = train_price_model(df)
    save_model(
        models, cat_maps, metrics,
        oof_preds=oof_preds,
        median_calibrator=calibrator,
    )

    loaded = load_model(max_age_hours=1)
    assert loaded is not None
    l_models, l_cat_maps, l_metrics, l_oof, l_calib, l_unc = loaded

    preds_orig = predict_prices(
        models, cat_maps, df,
        oof_preds=oof_preds,
        median_calibrator=calibrator,
    )
    preds_loaded = predict_prices(
        l_models, l_cat_maps, df,
        oof_preds=l_oof,
        median_calibrator=l_calib,
    )
    pd.testing.assert_frame_equal(preds_orig, preds_loaded)
    assert l_metrics["mae"] == metrics["mae"]
    assert l_oof == oof_preds
    assert isinstance(l_calib, IsotonicRegression)


def test_load_model_rejects_schema_mismatch(tmp_path, monkeypatch):
    """A bundle with a stale schema_version is rejected, not silently used."""
    import joblib
    monkeypatch.setattr("src.analytics.price_model._MODEL_PATH", tmp_path / "model.joblib")
    monkeypatch.setattr("src.analytics.price_model._METRICS_PATH", tmp_path / "metrics.json")

    df = _sample_listings()
    models, cat_maps, metrics, oof_preds, calibrator, _unc = train_price_model(df)
    save_model(
        models, cat_maps, metrics,
        oof_preds=oof_preds,
        median_calibrator=calibrator,
    )

    # Hand-corrupt the schema_version field, simulating an artifact trained
    # against an older feature list.
    bundle = joblib.load(tmp_path / "model.joblib")
    bundle["schema_version"] = _SCHEMA_VERSION - 1
    joblib.dump(bundle, tmp_path / "model.joblib")

    assert load_model(max_age_hours=1) is None


def test_load_model_rejects_feature_mismatch(tmp_path, monkeypatch):
    """A bundle whose feature_names list differs from current is rejected."""
    import joblib
    monkeypatch.setattr("src.analytics.price_model._MODEL_PATH", tmp_path / "model.joblib")
    monkeypatch.setattr("src.analytics.price_model._METRICS_PATH", tmp_path / "metrics.json")

    df = _sample_listings()
    models, cat_maps, metrics, oof_preds, calibrator, _unc = train_price_model(df)
    save_model(
        models, cat_maps, metrics,
        oof_preds=oof_preds,
        median_calibrator=calibrator,
    )

    bundle = joblib.load(tmp_path / "model.joblib")
    bundle["feature_names"] = bundle["feature_names"] + ["fictional_feature"]
    joblib.dump(bundle, tmp_path / "model.joblib")

    assert load_model(max_age_hours=1) is None


def test_load_model_rejects_stale_bundle(tmp_path, monkeypatch):
    """A bundle older than max_age_hours is rejected so consumers fall through
    to "no fresh model" instead of serving predictions from an obsolete fit.

    Regression: the 2026-04 incident where a v4-schema bundle survived on the
    self-hosted runner because CI hid train-model failures behind
    `continue-on-error: true` and re-uploaded the stale on-disk file every
    cron. Freshness is the last barrier when schema/feature checks happen to
    match an old format.
    """
    import os
    import time
    monkeypatch.setattr("src.analytics.price_model._MODEL_PATH", tmp_path / "model.joblib")
    monkeypatch.setattr("src.analytics.price_model._METRICS_PATH", tmp_path / "metrics.json")

    df = _sample_listings()
    models, cat_maps, metrics, oof_preds, calibrator, _unc = train_price_model(df)
    save_model(
        models, cat_maps, metrics,
        oof_preds=oof_preds,
        median_calibrator=calibrator,
    )

    # Backdate mtime by 2 hours; load_model with max_age_hours=1 must reject.
    two_hours_ago = time.time() - 2 * 3600
    os.utime(tmp_path / "model.joblib", (two_hours_ago, two_hours_ago))

    assert load_model(max_age_hours=1) is None
    # Generous window still accepts the same file — proves the rejection
    # was about freshness, not corruption from utime.
    assert load_model(max_age_hours=24) is not None


def test_metrics_history(tmp_path, monkeypatch):
    """Each save appends to metrics history."""
    monkeypatch.setattr("src.analytics.price_model._MODEL_PATH", tmp_path / "model.joblib")
    monkeypatch.setattr("src.analytics.price_model._METRICS_PATH", tmp_path / "metrics.json")

    df = _sample_listings()
    models, cat_maps, metrics, oof_preds, calibrator, _unc = train_price_model(df)

    save_model(
        models, cat_maps, metrics,
        oof_preds=oof_preds, median_calibrator=calibrator,
    )
    save_model(
        models, cat_maps, metrics,
        oof_preds=oof_preds, median_calibrator=calibrator,
    )

    history = load_metrics_history()
    assert len(history) == 2
    assert "timestamp" in history[0]
    assert history[0]["mae"] == metrics["mae"]


class TestSoldTargetAdjustment:
    """``_build_sold_target_adjustment`` lets the trainer reuse sold
    listings as ~3× extra training data without trusting last-ask as
    the actual sold price. The 2026-05-03 audit found 12 450 sold rows
    sitting unused; this is the helper that brings them into the fit."""

    def _row(self, **overrides):
        from datetime import datetime, timezone, timedelta
        defaults = {
            "is_active": True,
            "deactivation_reason": None,
            "first_seen_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
            "deactivated_at": None,
        }
        defaults.update(overrides)
        return defaults

    def test_active_rows_pass_through_unchanged(self):
        from src.analytics.price_model import _build_sold_target_adjustment
        df = pd.DataFrame([self._row(), self._row(), self._row()])
        mult, w = _build_sold_target_adjustment(df)
        assert (mult == 1.0).all()
        assert (w == 1.0).all()

    def test_quick_sold_gets_small_discount_high_weight(self):
        from datetime import datetime, timezone, timedelta
        from src.analytics.price_model import _build_sold_target_adjustment
        first = datetime(2026, 1, 1, tzinfo=timezone.utc)
        df = pd.DataFrame([self._row(
            is_active=False, deactivation_reason="sold",
            first_seen_at=first, deactivated_at=first + timedelta(days=5),
        )])
        mult, w = _build_sold_target_adjustment(df)
        # 5 days → first tier (≤14): 0.96 multiplier, 0.90 weight
        assert mult[0] == pytest.approx(0.96)
        assert w[0] == pytest.approx(0.90)

    def test_slow_sold_gets_bigger_discount_lower_weight(self):
        from datetime import datetime, timezone, timedelta
        from src.analytics.price_model import _build_sold_target_adjustment
        first = datetime(2026, 1, 1, tzinfo=timezone.utc)
        df = pd.DataFrame([self._row(
            is_active=False, deactivation_reason="sold",
            first_seen_at=first, deactivated_at=first + timedelta(days=200),
        )])
        mult, w = _build_sold_target_adjustment(df)
        # 200 days → 4th tier (≤365): 0.88 multiplier, 0.30 weight
        assert mult[0] == pytest.approx(0.88)
        assert w[0] == pytest.approx(0.30)

    def test_bogus_dates_get_zero_weight(self):
        """Pre-fix DB had rows with 2915-day "sold" lifespans (parser
        noise from a since-fixed date-extraction bug). Those must not
        contribute to training — weight=0 makes LightGBM ignore them
        without dropping the row from the index."""
        from datetime import datetime, timezone, timedelta
        from src.analytics.price_model import _build_sold_target_adjustment
        first = datetime(2026, 1, 1, tzinfo=timezone.utc)
        df = pd.DataFrame([
            self._row(  # 2915 days = bogus
                is_active=False, deactivation_reason="sold",
                first_seen_at=first,
                deactivated_at=first + timedelta(days=2915),
            ),
            self._row(  # negative days = clock-skew bug
                is_active=False, deactivation_reason="sold",
                first_seen_at=first,
                deactivated_at=first - timedelta(days=5),
            ),
        ])
        mult, w = _build_sold_target_adjustment(df)
        assert (w == 0.0).all()

    def test_expired_not_treated_as_sold(self):
        """Only ``deactivation_reason == "sold"`` triggers the
        adjustment. "Expired" / "removed" / unknown rows pass through
        unchanged so they don't pollute the target."""
        from datetime import datetime, timezone, timedelta
        from src.analytics.price_model import _build_sold_target_adjustment
        first = datetime(2026, 1, 1, tzinfo=timezone.utc)
        df = pd.DataFrame([
            self._row(
                is_active=False, deactivation_reason="expired",
                first_seen_at=first, deactivated_at=first + timedelta(days=20),
            ),
        ])
        mult, w = _build_sold_target_adjustment(df)
        assert mult[0] == 1.0 and w[0] == 1.0

    def test_train_logs_sold_inclusion_metrics(self):
        """End-to-end: the train_price_model output records how many
        sold rows were used vs dropped, so CI history can track whether
        the inclusion is helping."""
        from datetime import datetime, timezone, timedelta
        df = _sample_listings(n=200)
        # Half the rows become "sold" with varied lifespans.
        df["is_active"] = [True] * 100 + [False] * 100
        df["deactivation_reason"] = [None] * 100 + ["sold"] * 100
        first_seen = pd.to_datetime(
            [datetime(2026, 1, 1, tzinfo=timezone.utc)] * len(df), utc=True,
        )
        df["first_seen_at"] = first_seen
        df["deactivated_at"] = first_seen + pd.to_timedelta(
            [None] * 100 + list(np.tile([5, 30, 100, 250], 25)),
            unit="D",
        )
        result = train_price_model(df)
        assert result is not None
        _, _, metrics, _, _, _ = result
        assert "sold_inclusion" in metrics
        si = metrics["sold_inclusion"]
        assert si["sold_rows_used"] > 0
        # ``_filter_training_data`` clips the 1st/99th percentiles, so a
        # handful of the 100 actives may be filtered out — check the
        # majority survived rather than the exact count.
        assert si["active_rows"] >= 90
