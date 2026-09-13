"""``scripts/build_country_blobs.py`` — the Portuguese blob shapes, per country.

The frame is built by hand in the exact ``get_country_listings_df`` shape
(the ``get_listings_df`` columns plus the country extras), so the test does
not need the database or the AutoScout24 crawl.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

from scripts import build_country_blobs as bcb
from src.storage import repository
from src.storage.repository import _seller_columns_for

REPO_ROOT = Path(__file__).resolve().parent.parent
REGION = "Bayern"
MODELS = (
    ("Skoda", "Citigo", 11000.0),
    ("Opel", "Adam", 12000.0),
    ("Ford", "B-MAX", 13000.0),
)
YEARS = tuple(range(2012, 2020))
PER_CELL = 5


def _row(i: int, brand: str, model: str, year: int, price: float, now: pd.Timestamp) -> dict:
    external_id = f"{700000 + i}"
    age = max(1, now.year - year)
    row = {
        "olx_id": f"as24_de:{external_id}",
        "url": f"https://www.autoscout24.de/angebote/{brand.lower()}-{model.lower()}-{external_id}",
        "title": f"{brand} {model} 1.0 TSI",
        "brand": brand, "model": model, "year": year,
        "price_eur": float(round(price)),
        "first_price_eur": float(round(price)),
        "num_price_drops": 0, "max_drop_pct": 0.0,
        "price_drop_velocity": None, "days_since_last_drop": None,
        "mileage_km": 12000 * age + (i % 7) * 1500, "engine_cc": 999,
        "fuel_type": "Gasolina", "horsepower": 75,
        "transmission": "Manual", "segment": "Citadino",
        "doors": 5, "seats": 4, "color": None, "condition": None, "drive_type": None,
        "photo_count": 12, "description_length": None,
        "city": "München", "district": REGION,
        "seller_type": "Profissional", "is_active": True,
        "generation": None, "description": None, "llm_extras": None,
        "first_seen_at": now - pd.Timedelta(days=3 + (i % 50)),
        "last_seen_at": now - pd.Timedelta(days=1),
        "last_scraped_at": now - pd.Timedelta(days=1),
        "sub_model": "1.0 TSI", "trim_level": None,
        "desc_mentions_repair": None, "desc_mentions_accident": None,
        "real_mileage_km": None, "desc_mentions_num_owners": None,
        "desc_mentions_customs_cleared": None, "right_hand_drive": None,
        "mechanical_condition": None, "damage_severity": None, "urgency": None,
        "warranty": None, "tuning_or_mods": None, "taxi_fleet_rental": None,
        "first_owner_selling": None,
        "source": "as24_de", "duplicate_of": None,
        "deactivated_at": None, "deactivation_reason": None,
        "seller_uuid": None, "seller_displayed_as": None, "seller_profile_url": None,
        "seller_listings_count_90d": None,
        **_seller_columns_for(None, None),
        "country": "DE", "external_id": external_id,
        "image_url": ("https://prod.pictures.autoscout24.net/listing-images/"
                      f"{external_id}/720x540.webp"),
        "price_label": "Guter Preis", "co2_g_km": 108, "registration_month": 6,
        "vat_reclaimable": True, "is_damaged": False, "origin": None,
    }
    return row


def _country_frame() -> pd.DataFrame:
    """3 models × 8 years × 5 listings, all in one region, one bargain per model.

    Prices depreciate 7 %/year with a ±10 % jitter; the newest cell of each
    model carries one listing at half price so the deal path has something to
    surface once a (stubbed) model says it is undervalued.
    """
    now = pd.Timestamp.now("UTC").tz_localize(None).floor("s")
    rows: list[dict] = []
    i = 0
    for brand, model, base in MODELS:
        for year in YEARS:
            for k in range(PER_CELL):
                i += 1
                price = base * (0.93 ** (YEARS[-1] - year)) * (0.90 + 0.05 * k)
                if year == YEARS[-1] and k == 0:
                    price = base * 0.5
                rows.append(_row(i, brand, model, year, price, now))
    return pd.DataFrame(rows)


@pytest.fixture
def country_frame(monkeypatch):
    frame = _country_frame()
    assert len(frame) == len(MODELS) * len(YEARS) * PER_CELL == 120

    def _load(session, cc):
        assert cc == "DE"
        return frame.copy()

    monkeypatch.setattr(repository, "get_country_listings_df", _load, raising=False)
    monkeypatch.setattr(bcb, "_published_models", lambda out_dir, cc: None)
    return frame


@pytest.fixture
def no_model(monkeypatch):
    monkeypatch.setattr("src.analytics.price_model.load_model", lambda *a, **k: None)


def _read(path):
    return json.loads(path.read_text())


class TestModelPages:
    def test_models_blob_has_the_pt_shape(self, tmp_path, country_frame, no_model):
        manifest = bcb.build_country("DE", None, tmp_path, min_models=3)

        path = tmp_path / "models_de.json"
        assert path.exists()
        doc = _read(path)
        assert doc["v"] == 1
        assert doc["built_at"] == manifest["built_at"]
        assert set(doc["models"]) == {"skoda-citigo", "opel-adam", "ford-b-max"}
        for rec in doc["models"].values():
            assert rec["n"] == len(YEARS) * PER_CELL
            assert rec["fl"] <= rec["fm"] <= rec["fh"]
            assert rec["yr"], "per-year cells expected"
            assert {c["y"] for c in rec["yr"]} & set(YEARS)
            assert all(c["n"] >= PER_CELL for c in rec["yr"])
            assert "gm" not in rec, "no model → asking-only"
            assert rec["u"]
        assert "mq" not in doc

    def test_regions_roll_up_as_districts(self, tmp_path, country_frame, no_model):
        bcb.build_country("DE", None, tmp_path, min_models=3)
        doc = _read(tmp_path / "models_de.json")
        assert "bayern" in doc["districts"]
        bayern = doc["districts"]["bayern"]
        assert bayern["lbl"] == REGION and bayern["n"] == 120
        assert {slug for slug, _n, _fm in bayern["top"]} == set(doc["models"])

    def test_collapse_guard_keeps_the_previous_blob(self, tmp_path, country_frame, no_model):
        stale = tmp_path / "models_de.json"
        stale.write_text('{"v":1,"models":{}}')
        manifest = bcb.build_country("DE", None, tmp_path, min_models=10)
        assert _read(stale) == {"v": 1, "models": {}}
        assert manifest["rows"]["model_pages"] == 3
        assert "models_de.json" not in manifest["files_bytes"]
        assert (tmp_path / "brands_models_de.json").exists()

    def test_no_model_means_no_deal_artifacts_but_a_finished_build(
            self, tmp_path, country_frame, no_model):
        manifest = bcb.build_country("DE", None, tmp_path, min_models=3)
        assert not (tmp_path / "hot_deals_de_all.json").exists()
        assert not (tmp_path / "valuations_de.json").exists()
        assert (tmp_path / "manifest_de.json").exists()
        assert manifest["rows"]["listings"] == 120
        assert manifest["rows"]["active"] == 120
        assert manifest["rows"]["hot_deals"] == 0
        assert manifest["rows"]["valuations"] == 0
        brands = _read(tmp_path / "brands_models_de.json")
        assert brands == {"Ford": ["B-MAX"], "Opel": ["Adam"], "Skoda": ["Citigo"]}

    def test_no_model_flag_via_main(self, tmp_path, country_frame, monkeypatch):
        monkeypatch.setattr("src.storage.database.init_db", lambda *a, **k: None)
        closed: dict = {}

        class _Session:
            def close(self):
                closed["yes"] = True

        monkeypatch.setattr("src.storage.database.get_session", lambda: _Session())
        rc = bcb.main(["--country", "DE", "--out", str(tmp_path),
                       "--no-model", "--min-models", "3"])
        assert rc == 0 and closed["yes"]
        doc = _read(tmp_path / "models_de.json")
        assert doc["models"] and all("gm" not in r for r in doc["models"].values())
        assert not (tmp_path / "hot_deals_de_all.json").exists()


class TestDealFeed:
    def test_hot_deals_carry_the_card_fields(self, tmp_path, country_frame, patched_gb_model):
        with patched_gb_model():
            manifest = bcb.build_country("DE", None, tmp_path, min_models=3)

        path = tmp_path / "hot_deals_de_all.json"
        assert path.exists()
        payload = _read(path)
        assert payload["zone"] == "all"
        assert payload["built_at"] == manifest["built_at"]
        deals = payload["deals"]
        assert deals, "the stubbed model reads every listing as undervalued"
        assert manifest["rows"]["hot_deals"] == len(deals)
        by_id = {r["olx_id"]: r for r in country_frame.to_dict("records")}
        for deal in deals:
            src = by_id[deal["olx_id"]]
            assert deal["url"] == src["url"]
            assert deal["url"].startswith("https://www.autoscout24.de/")
            assert isinstance(deal["price_eur"], int) and deal["price_eur"] > 0
            assert isinstance(deal["fair_median"], int) and deal["fair_median"] > deal["price_eur"]
            assert deal["photo_urls"] == [src["image_url"]]
            assert deal["verdict"] in ("BUY", "WATCH")
            assert deal["district"] == REGION
            assert deal["days_on_market"] is not None
        assert {d["brand"] for d in deals} == {b for b, _m, _p in MODELS}

    def test_valuations_and_asking_only_pages_with_a_stub_model(
            self, tmp_path, country_frame, patched_gb_model):
        with patched_gb_model():
            manifest = bcb.build_country("DE", None, tmp_path, min_models=3)

        valuations = _read(tmp_path / "valuations_de.json")
        assert valuations["v"] == 1
        assert len(valuations["cars"]) == 120 == manifest["rows"]["valuations"]
        car = valuations["cars"]["as24_de:700001"]
        assert car["p"] > 0 and car["fl"] <= car["fm"] <= car["fh"]
        assert car["ms"] == "skoda-citigo"

        doc = _read(tmp_path / "models_de.json")
        assert all("gm" not in r for r in doc["models"].values())
        assert "mq" not in doc or isinstance(doc["mq"], dict)


class TestGbmBand:
    """A valuator that can actually price a configs frame puts the band on the
    pages and the model-quality card on the blob — the same wiring the
    Portuguese build uses, proven here without LightGBM."""

    _BASE = {brand: base for brand, _model, base in MODELS}
    _METRICS = {"mae": 780.0, "mape": 9.4, "r2": 0.91,
                "coverage_80_calibrated": 0.81, "n_samples": 4200,
                "conformal_q": 0.0, "conformal_q_per_bucket": {},
                "conformal_q_bucket_edges": None}

    @pytest.fixture
    def stub_valuation(self, monkeypatch):
        def _value_configs(cfg, bundle=None):
            assert "price_eur" not in cfg.columns, "the valuator prices specs, not asks"
            base = cfg["brand"].map(self._BASE).astype(float)
            age = YEARS[-1] - pd.to_numeric(cfg["year"], errors="coerce").fillna(YEARS[-1])
            pred = base * (0.93 ** age) * 0.97
            return pd.DataFrame({
                "predicted_price": pred,
                "fair_price_low": pred * 0.88,
                "fair_price_high": pred * 1.12,
                "spec_fill": 1.0,
                "vocab_ok": True,
            }, index=cfg.index)

        monkeypatch.setattr("src.analytics.price_model.value_configs", _value_configs)
        monkeypatch.setattr(bcb, "_load_bundle", lambda cc: {
            "models": {}, "cat_maps": {}, "metrics": dict(self._METRICS),
            "median_calibrator": None, "uncertainty_bundle": None,
        })

    def test_band_and_quality_card_reach_the_blob(self, tmp_path, country_frame,
                                                  stub_valuation, patched_gb_model, capsys):
        with patched_gb_model():
            bcb.build_country("DE", None, tmp_path, min_models=3)
        assert "GBM band failed" not in capsys.readouterr().out

        doc = _read(tmp_path / "models_de.json")
        assert doc["mq"] == {"mae": 780, "mape": 9.4, "r2": 0.91, "cov": 0.81, "n": 4200}
        for rec in doc["models"].values():
            assert rec["gl"] <= rec["gm"] <= rec["gh"]
            assert 0.8 < rec["gm"] / rec["fm"] < 1.2
            assert any("gm" in cell for cell in rec["yr"]), "per-year band expected"


class TestCountryModelScope:
    """``compute_signals`` asks for the loaders by name at call time, so the
    scope is what makes it read the German bundle and ignore the Portuguese
    anomaly/hazard fits."""

    def test_loaders_are_bound_then_restored(self, monkeypatch):
        from src.analytics import anomaly, hazard, price_model

        seen: list[str | None] = []
        monkeypatch.setattr(price_model, "load_model",
                            lambda *a, country=None, **k: seen.append(country) or "bundle")
        monkeypatch.setattr(price_model, "load_importance",
                            lambda *a, country=None, **k: country)
        monkeypatch.setattr(anomaly, "load_model", lambda *a, **k: "pt-anomaly")
        monkeypatch.setattr(hazard, "load_model", lambda *a, **k: "pt-hazard")
        before = (price_model.load_model, price_model.load_importance,
                  anomaly.load_model, hazard.load_model)

        with bcb._country_model_scope("DE"):
            assert price_model.load_model(max_age_hours=1) == "bundle"
            assert seen == ["DE"]
            assert price_model.load_importance() == "DE"
            assert anomaly.load_model() is None
            assert hazard.load_model() is None

        assert (price_model.load_model, price_model.load_importance,
                anomaly.load_model, hazard.load_model) == before
        assert anomaly.load_model() == "pt-anomaly"

    def test_restoration_survives_a_failure_inside(self, monkeypatch):
        from src.analytics import price_model

        before = price_model.load_model
        with pytest.raises(RuntimeError):
            with bcb._country_model_scope("FR"):
                raise RuntimeError("synthetic")
        assert price_model.load_model is before


class TestUnshippableValues:
    """One blob the Worker could not parse must cost that blob, not the country."""

    def test_nan_and_timestamp_are_refused_by_name(self, tmp_path, capsys):
        assert bcb._dump_json(tmp_path / "a.json", {"x": float("nan")}, "DE") is None
        assert bcb._dump_json(tmp_path / "b.json", {"x": pd.Timestamp("2026-01-01")},
                              "DE") is None
        out = capsys.readouterr().out
        assert "a.json SKIPPED" in out and "b.json SKIPPED" in out
        assert not (tmp_path / "a.json").exists() and not (tmp_path / "b.json").exists()

    def test_a_refused_blob_still_leaves_a_manifest(self, tmp_path, country_frame,
                                                    no_model, monkeypatch):
        real = bcb._dump_json

        def _refuse_models(path, doc, cc):
            return None if path.name.startswith("models_") else real(path, doc, cc)

        monkeypatch.setattr(bcb, "_dump_json", _refuse_models)
        manifest = bcb.build_country("DE", None, tmp_path, min_models=3)
        assert "models_de.json" not in manifest["files_bytes"]
        assert (tmp_path / "manifest_de.json").exists()
        assert (tmp_path / "brands_models_de.json").exists()


class TestCoverageFromTheBundle:
    """The band-confidence step reads this country's own coverage, not None."""

    def test_calibrated_coverage_wins_and_absence_is_neutral(self):
        assert bcb._coverage_80({"metrics": {"coverage_80_calibrated": 0.81,
                                             "coverage_80": 0.77}}) == 0.81
        assert bcb._coverage_80({"metrics": {"coverage_80": 0.77}}) == 0.77
        assert bcb._coverage_80({"metrics": {}}) is None
        assert bcb._coverage_80(None) is None

    def test_the_build_hands_it_to_the_decision_context(self, tmp_path, country_frame,
                                                        patched_gb_model, monkeypatch):
        from src.analytics import decision

        seen: dict = {}
        real = decision.build_context

        def _spy(listings, snapshots=None, **kw):
            seen["coverage_80"] = kw.get("coverage_80")
            return real(listings, snapshots, **kw)

        monkeypatch.setattr(decision, "build_context", _spy)
        monkeypatch.setattr(bcb, "_load_bundle", lambda cc: {
            "models": {}, "cat_maps": {},
            "metrics": {"coverage_80_calibrated": 0.83},
            "median_calibrator": None, "uncertainty_bundle": None,
        })
        with patched_gb_model():
            bcb.build_country("DE", None, tmp_path, min_models=3)
        assert seen["coverage_80"] == 0.83


class TestSignalsFailure:
    def test_a_crashing_compute_signals_still_ships_the_pages(
            self, tmp_path, country_frame, patched_gb_model, monkeypatch, capsys):
        def _boom(*a, **k):
            raise RuntimeError("synthetic failure")

        monkeypatch.setattr("src.dashboard.data_loader.compute_signals", _boom)
        with patched_gb_model():
            manifest = bcb.build_country("DE", None, tmp_path, min_models=3)
        assert (tmp_path / "models_de.json").exists()
        assert not (tmp_path / "hot_deals_de_all.json").exists()
        assert manifest["rows"]["signals"] == 0
        assert "compute_signals FAILED" in capsys.readouterr().out


class TestEmptyCountry:
    def test_no_rows_writes_only_the_manifest(self, tmp_path, monkeypatch, no_model):
        monkeypatch.setattr(repository, "get_country_listings_df",
                            lambda session, cc: pd.DataFrame(), raising=False)
        manifest = bcb.build_country("IT", None, tmp_path)
        assert manifest["rows"] == {"listings": 0}
        assert sorted(p.name for p in tmp_path.iterdir()) == ["manifest_it.json"]


class TestWorkerFieldContract:
    """The blob keys this script writes against the ones ``pages-intl.js`` reads.

    Four agents built the two sides of this seam independently, and nothing
    else crosses it: the Worker's own smoke test renders hand-written fixture
    objects, so a rename on the Python side would leave both suites green and
    only show up as an empty card in production. These tests read the real
    JavaScript and the real blobs, so neither side can be renamed alone.
    """

    PAGES_INTL = REPO_ROOT / "flipper-club" / "src" / "pages-intl.js"

    DEAL_FALLBACKS = {"image_url"}

    @staticmethod
    def _js() -> str:
        return TestWorkerFieldContract.PAGES_INTL.read_text(encoding="utf-8")

    @staticmethod
    def _verdict_block(js: str) -> str:
        """The body of ``verdictBlock``, which renders one valuation record."""
        start = js.index("function verdictBlock(")
        rest = js[start + 1:]
        end = min(i for i in (rest.find("\nfunction "), rest.find("\nexport function "))
                  if i != -1)
        return rest[:end]

    def test_every_deal_field_the_feed_reads_is_written(self, tmp_path, country_frame,
                                                        patched_gb_model):
        with patched_gb_model():
            bcb.build_country("DE", None, tmp_path, min_models=3)

        deals = _read(tmp_path / "hot_deals_de_all.json")["deals"]
        assert deals
        written = set(deals[0])
        assert all(set(d) == written for d in deals)

        read_by_worker = set(re.findall(r"\bd\.([A-Za-z_][A-Za-z_0-9]*)", self._js()))
        assert read_by_worker, "the feed card stopped reading any deal field"
        missing = read_by_worker - written - self.DEAL_FALLBACKS
        assert not missing, f"pages-intl.js reads deal fields nobody writes: {sorted(missing)}"
        assert self.DEAL_FALLBACKS.isdisjoint(written), (
            "a tolerated fallback is now written for real; drop it from DEAL_FALLBACKS"
        )

    @staticmethod
    def _writable_valuation_keys() -> set[str]:
        """Every key ``build_valuations`` can put in a record.

        Read off the writer's own source rather than off a built blob: the
        writer drops ``None`` values, so an optional key such as the sell-speed
        is simply absent on a corpus too thin to have one, and a blob is a
        floor on the contract rather than the contract itself.
        """
        src = (REPO_ROOT / "src" / "analytics" / "valuations.py").read_text(encoding="utf-8")
        body = src[src.index("def build_valuations("):]
        body = body[:body.rindex('return {"v": 1, "cars": cars}')]
        literal = re.search(r"rec = \{(.*?)\n        \}", body, re.S)
        assert literal, "build_valuations no longer builds its record as a dict literal"
        return (set(re.findall(r'"([a-z_]+)":', literal.group(1)))
                | set(re.findall(r'rec\["([a-z_]+)"\]\s*=', body)))

    def test_every_valuation_field_the_tool_reads_is_written(self, tmp_path, country_frame,
                                                             patched_gb_model):
        with patched_gb_model():
            bcb.build_country("DE", None, tmp_path, min_models=3)

        cars = _read(tmp_path / "valuations_de.json")["cars"]
        assert cars
        writable = self._writable_valuation_keys()
        assert {"p", "fm", "fl", "fh", "sd", "ms"} <= writable

        block = self._verdict_block(self._js())
        read_by_worker = set(re.findall(r"\brec\.([A-Za-z_][A-Za-z_0-9]*)", block))
        assert read_by_worker, "verdictBlock stopped reading any valuation field"
        missing = read_by_worker - writable
        assert not missing, f"verdictBlock reads valuation keys nobody writes: {sorted(missing)}"

        built = set().union(*(set(rec) for rec in cars.values()))
        assert built <= writable, f"the blob grew keys the writer cannot name: {built - writable}"
        assert {"p", "fm", "fl", "fh"} <= built, "a country valuation lost its price band"

    def test_the_worker_builds_the_olx_id_the_frame_produces(self):
        """``valuationKey`` in i18n.js must rebuild ``get_country_listings_df``'s id."""
        from src.countries import source_for

        i18n = (REPO_ROOT / "flipper-club" / "src" / "i18n.js").read_text(encoding="utf-8")
        template = re.search(r"export function valuationKey\(loc, id\) \{.*?return `([^`]+)`",
                             i18n, re.S)
        assert template, "valuationKey no longer returns a template literal"
        for cc in ("de", "fr", "it"):
            built = template.group(1).replace("${l.code}", cc).replace("${id}", "u-1")
            assert built == f"{source_for(cc.upper())}:u-1"


class TestWhatTheCrawlCadenceCannotSupport:
    """Numbers a fortnightly walk cannot honestly produce are not produced.

    A country corpus is walked cell by cell inside a politeness budget, so a
    given listing is re-confirmed weeks apart. Days-to-sell is measured from
    when a listing stops being seen, and a live deal is a claim about right
    now; both would be reporting the request budget rather than the market.
    """

    def test_days_to_sell_is_absent_unless_the_market_is_asked_for_it(
            self, tmp_path, country_frame, no_model):
        bcb.build_country("DE", None, tmp_path, min_models=3)

        doc = _read(tmp_path / "models_de.json")
        assert "lqm" not in doc
        for rec in doc["models"].values():
            assert "lq" not in rec
            assert "sd" not in rec and "sn" not in rec

    def test_asking_for_it_builds_the_curve_that_is_otherwise_never_built(
            self, tmp_path, country_frame, no_model, monkeypatch):
        seen = []

        def _fake_curve(listings, *args, **kwargs):
            seen.append(len(listings))
            return {"models": {}, "market": {"n": 40, "md": 29, "s30": 0.6}}

        monkeypatch.setattr("src.analytics.liquidity.build_liquidity", _fake_curve)

        bcb.build_country("DE", None, tmp_path, min_models=3)
        assert seen == [], "the curve was computed for a market that cannot support it"

        bcb.build_country("DE", None, tmp_path, min_models=3, with_liquidity=True)
        assert seen, "the flag did not reach the curve"
        assert _read(tmp_path / "models_de.json")["lqm"]["md"] == 29

    def test_a_listing_nobody_confirmed_lately_does_not_reach_the_feed(self):
        now = pd.Timestamp.now(tz="UTC")
        merged = pd.DataFrame([
            {"olx_id": "as24_de:1",
             "last_scraped_at": now - pd.Timedelta(days=2)},
            {"olx_id": "as24_de:2",
             "last_scraped_at": now - pd.Timedelta(days=bcb.MAX_UNSEEN_DAYS + 5)},
        ])
        kept = bcb._seen_recently(merged, "DE")
        assert list(kept["olx_id"]) == ["as24_de:1"]

    def test_a_frame_without_the_column_is_left_alone(self):
        merged = pd.DataFrame([{"olx_id": "as24_de:1"}])
        assert len(bcb._seen_recently(merged, "DE")) == 1
