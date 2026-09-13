"""Country-suffixed price-model artefacts next to the Portuguese ones.

``train-model --country DE`` must be able to save a German bundle on the same
runner without ever touching ``data/price_model.joblib``, and every loader
must find it again under the same ``country`` argument.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.analytics import price_model as pm

_KEYS = {"model", "metrics", "importance", "grouped_importance", "shap_importance"}


class TestArtifactPaths:
    def test_home_market_keeps_todays_names(self):
        for country in (None, "PT", "pt"):
            paths = pm.artifact_paths(country)
            assert set(paths) == _KEYS
            assert paths["model"] == pm._MODEL_PATH
            assert paths["metrics"] == pm._METRICS_PATH
            assert paths["importance"] == pm._IMPORTANCE_PATH
            assert paths["grouped_importance"] == pm._GROUPED_IMPORTANCE_PATH
            assert paths["shap_importance"] == pm._SHAP_IMPORTANCE_PATH

    def test_home_market_is_literally_the_data_directory_bundle(self):
        """Pinned against the file names CI uploads and the release audit
        expects, so a refactor of the constants cannot quietly move them."""
        repo_data = Path(pm.__file__).resolve().parent.parent.parent / "data"
        for country in (None, "PT"):
            paths = pm.artifact_paths(country)
            assert paths["model"] == repo_data / "price_model.joblib"
            assert paths["metrics"] == repo_data / "price_metrics.json"
            assert paths["importance"] == repo_data / "price_importance.json"
            assert paths["grouped_importance"] == repo_data / "price_grouped_importance.json"
            assert paths["shap_importance"] == repo_data / "price_shap_importance.json"

    def test_country_suffix_sits_before_the_extension(self):
        paths = pm.artifact_paths("DE")
        assert set(paths) == _KEYS
        assert paths["model"].name == "price_model_de.joblib"
        assert paths["metrics"].name == "price_metrics_de.json"
        assert paths["importance"].name == "price_importance_de.json"
        assert paths["grouped_importance"].name == "price_grouped_importance_de.json"
        assert paths["shap_importance"].name == "price_shap_importance_de.json"
        assert all(p.parent == pm._MODEL_DIR for p in paths.values())

    def test_country_code_is_case_insensitive(self):
        assert pm.artifact_paths("de") == pm.artifact_paths("DE")
        assert pm.artifact_paths("fr")["model"].name == "price_model_fr.joblib"

    def test_paths_follow_model_dir_at_call_time(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pm, "_MODEL_DIR", tmp_path)
        assert pm.artifact_paths("IT")["model"] == tmp_path / "price_model_it.joblib"
        assert pm.artifact_paths(None)["model"] == tmp_path / "price_model.joblib"

    def test_a_reassigned_home_constant_still_wins(self, tmp_path, monkeypatch):
        """Existing tests point ``_MODEL_PATH`` at a temp file; that must keep
        steering the home market even though the paths are now derived."""
        monkeypatch.setattr(pm, "_MODEL_PATH", tmp_path / "model.joblib")
        assert pm.artifact_paths(None)["model"] == tmp_path / "model.joblib"
        assert pm.artifact_paths("PT")["model"] == tmp_path / "model.joblib"

    def test_rejects_anything_but_a_two_letter_code(self):
        with pytest.raises(ValueError):
            pm.artifact_paths("Deutschland")


def _save_stub(country: str | None, tag: str) -> None:
    pm.save_model(
        {"median": f"model-{tag}"}, {"brand": {"Opel": 0}},
        {"mae": 1000.0, "mape": 20.0, "r2": 0.9, "n_samples": 10},
        oof_preds={"x1": (1.0, 2.0, 3.0)},
        country=country,
    )


@pytest.fixture
def isolated_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "_MODEL_DIR", tmp_path)
    monkeypatch.setattr(pm, "_MODEL_PATH", tmp_path / "price_model.joblib")
    monkeypatch.setattr(pm, "_METRICS_PATH", tmp_path / "price_metrics.json")
    monkeypatch.setattr(pm, "_IMPORTANCE_PATH", tmp_path / "price_importance.json")
    return tmp_path


class TestCountryBundles:
    def test_de_bundle_round_trips_and_leaves_pt_alone(self, isolated_dir):
        _save_stub("DE", "de")
        assert (isolated_dir / "price_model_de.joblib").exists()
        assert not (isolated_dir / "price_model.joblib").exists()

        loaded = pm.load_model(max_age_hours=1, country="DE")
        assert loaded is not None
        models, cat_maps, metrics, oof, calibrator, uncertainty = loaded
        assert models == {"median": "model-de"}
        assert cat_maps == {"brand": {"Opel": 0}}
        assert metrics["mae"] == 1000.0
        assert oof == {"x1": (1.0, 2.0, 3.0)}
        assert calibrator is None and uncertainty is None

        assert pm.load_model(max_age_hours=1) is None
        assert pm.load_model(max_age_hours=1, country="FR") is None

    def test_two_markets_do_not_share_a_file(self, isolated_dir):
        _save_stub("DE", "de")
        _save_stub("FR", "fr")
        _save_stub(None, "pt")
        assert pm.load_model(max_age_hours=1, country="de")[0] == {"median": "model-de"}
        assert pm.load_model(max_age_hours=1, country="FR")[0] == {"median": "model-fr"}
        assert pm.load_model(max_age_hours=1)[0] == {"median": "model-pt"}

    def test_metrics_history_is_per_country(self, isolated_dir):
        _save_stub("DE", "de")
        _save_stub("DE", "de")
        history = pm.load_metrics_history(country="DE")
        assert len(history) == 2
        assert history[0]["mae"] == 1000.0 and "timestamp" in history[0]
        assert pm.load_metrics_history() == []
        assert (isolated_dir / "price_metrics_de.json").exists()
        assert not (isolated_dir / "price_metrics.json").exists()

    def test_importance_frames_are_per_country(self, isolated_dir):
        frame = pd.DataFrame({"feature": ["year", "mileage_km"],
                              "median_importance": [0.5, 0.3]})
        pm.save_importance(frame, country="DE")
        pm.save_grouped_importance(frame.assign(group=["a", "b"]), country="DE")
        pm.save_shap_importance(frame, country="DE")
        for name in ("price_importance_de.json", "price_grouped_importance_de.json",
                     "price_shap_importance_de.json"):
            assert (isolated_dir / name).exists()
        assert not (isolated_dir / "price_importance.json").exists()

        pd.testing.assert_frame_equal(pm.load_importance(country="DE"), frame)
        assert list(pm.load_grouped_importance(country="de")["group"]) == ["a", "b"]
        pd.testing.assert_frame_equal(pm.load_shap_importance(country="DE"), frame)
        assert pm.load_importance().empty
        assert pm.load_grouped_importance().empty
        assert pm.load_shap_importance().empty

    def test_home_market_default_is_unchanged(self, isolated_dir):
        """No ``country`` → today's file names, today's history file."""
        _save_stub(None, "pt")
        assert (isolated_dir / "price_model.joblib").exists()
        assert (isolated_dir / "price_metrics.json").exists()
        assert pm.load_model(max_age_hours=1)[0] == {"median": "model-pt"}
        assert len(pm.load_metrics_history()) == 1


class TestTrainModelCountryCli:
    """``train-model --country`` reads the country frame and never the OLX one."""

    @pytest.fixture
    def cli(self, monkeypatch):
        import src.cli as cli_mod
        from src.storage import repository

        monkeypatch.setattr(cli_mod, "init_db", lambda *a, **k: None)
        monkeypatch.setattr(cli_mod, "get_session", lambda: MagicMock())

        def _pt_must_not_run(session):
            raise AssertionError("the PT listings query must not run for --country DE")

        monkeypatch.setattr(repository, "get_listings_df", _pt_must_not_run)
        return cli_mod

    def test_thin_country_corpus_exits_one_with_the_floor(self, cli, monkeypatch):
        from typer.testing import CliRunner
        from src.storage import repository

        calls: dict = {}
        thin = pd.DataFrame({
            "olx_id": [f"as24_de:{i}" for i in range(12)],
            "price_eur": [9000 + i for i in range(12)],
            "is_active": [True] * 10 + [False] * 2,
        })

        def _country(session, cc):
            calls["cc"] = cc
            return thin

        monkeypatch.setattr(repository, "get_country_listings_df", _country, raising=False)
        result = CliRunner().invoke(cli.app, ["train-model", "--country", "de"])
        assert result.exit_code == 1, result.output
        assert calls["cc"] == "DE"
        assert "1500" in result.output
        assert "10 priced active" in result.output

    def test_empty_country_frame_exits_one(self, cli, monkeypatch):
        from typer.testing import CliRunner
        from src.storage import repository

        monkeypatch.setattr(repository, "get_country_listings_df",
                            lambda session, cc: pd.DataFrame(), raising=False)
        result = CliRunner().invoke(cli.app, ["train-model", "--country", "FR"])
        assert result.exit_code == 1, result.output
        assert "FR" in result.output

    def test_default_country_is_the_pt_path(self, monkeypatch):
        from typer.testing import CliRunner
        import src.cli as cli_mod
        from src.storage import repository

        monkeypatch.setattr(cli_mod, "init_db", lambda *a, **k: None)
        monkeypatch.setattr(cli_mod, "get_session", lambda: MagicMock())
        monkeypatch.setattr(repository, "get_listings_df", lambda session: pd.DataFrame())
        monkeypatch.setattr(repository, "get_relist_events_df", lambda session: pd.DataFrame())

        def _country_must_not_run(session, cc):
            raise AssertionError("the country query must not run without --country")

        monkeypatch.setattr(repository, "get_country_listings_df", _country_must_not_run,
                            raising=False)
        result = CliRunner().invoke(cli_mod.app, ["train-model"])
        assert result.exit_code == 1, result.output
        assert "No listings" in result.output
