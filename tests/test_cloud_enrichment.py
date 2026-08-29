"""Tests for the value-gated cloud enrichment path.

Three things are pinned here, in the order they matter:

1. The GATE — only the top-ranked undervalued deals ever reach a model, and
   the cheap tail / spec-poor / already-enriched rows never do.
2. The CASCADE — Gemini first, OpenRouter when Gemini can't answer, and a
   provider that has no key or no daily budget left costs zero requests.
3. The LEDGER — spend is charged to the provider that made the request and
   survives a crash mid-run, because a lost counter means overrunning a free
   tier the next time.
"""
import json
import types

import pandas as pd
import pytest

from src.parser import cloud_enrichment as ce


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def test_parse_json_object_plain():
    assert ce._parse_json_object('{"a": 1}') == {"a": 1}


def test_parse_json_object_fenced():
    assert ce._parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert ce._parse_json_object('```\n{"a": 2}\n```') == {"a": 2}


def test_parse_json_object_with_prose_around():
    assert ce._parse_json_object('Here you go: {"a": 1, "b": true} done') == {"a": 1, "b": True}


def test_parse_json_object_garbage():
    assert ce._parse_json_object("not json at all") is None
    assert ce._parse_json_object("") is None
    assert ce._parse_json_object("[1,2,3]") is None  # array is not an object


# ---------------------------------------------------------------------------
# Corrections — vocab validation + column mapping
# ---------------------------------------------------------------------------

class _Listing:
    def __init__(self, extras, brand="BMW", mileage_km=180000, title="", description=""):
        self._llm_extras = extras
        self.brand = brand
        self.mileage_km = mileage_km
        self.title = title
        self.description = description
        self.url = "http://x"
        self.olx_id = "ABC"


def test_cloud_corrections_full_valid():
    extras = {
        "sub_model": "320d", "trim_level": "M Sport",
        "mileage_in_description_km": 180000,
        "mechanical_condition": "good",
        "desc_mentions_accident": False, "desc_mentions_repair": True,
        "desc_mentions_num_owners": 2, "desc_mentions_customs_cleared": False,
        "right_hand_drive": False, "warranty": True, "urgency": "high",
        "negotiable": True, "red_flags": ["km alta"], "deal_note_pt": "bom",
    }
    c = ce.cloud_corrections(_Listing(extras))
    assert c["sub_model"] == "320d"
    assert c["trim_level"] == "M Sport"
    assert c["mechanical_condition"] == "good"
    assert c["desc_mentions_repair"] is True
    assert c["desc_mentions_accident"] is False
    assert c["desc_mentions_num_owners"] == 2
    assert c["warranty"] is True
    assert c["urgency"] == "high"
    # damage_severity is derived by correct_listing_data (repair → 2)
    assert c["damage_severity"] == 2
    # display-only keys never become corrections/columns
    assert "negotiable" not in c
    assert "red_flags" not in c
    assert "deal_note_pt" not in c


def test_cloud_corrections_rejects_bad_enum():
    extras = {"mechanical_condition": "razoavel", "urgency": "baixa"}  # PT, not the English vocab
    c = ce.cloud_corrections(_Listing(extras))
    assert "mechanical_condition" not in c
    assert "urgency" not in c


def test_cloud_corrections_enum_case_normalized():
    c = ce.cloud_corrections(_Listing({"urgency": "HIGH", "mechanical_condition": "Excellent"}))
    assert c["urgency"] == "high"
    assert c["mechanical_condition"] == "excellent"


def test_cloud_corrections_bool_type_strict():
    # a truthy non-bool must NOT be written to a bool column
    c = ce.cloud_corrections(_Listing({"warranty": "yes", "desc_mentions_accident": 1}))
    assert "warranty" not in c
    assert "desc_mentions_accident" not in c


def test_cloud_corrections_num_owners_rejects_bool():
    c = ce.cloud_corrections(_Listing({"desc_mentions_num_owners": True}))
    assert "desc_mentions_num_owners" not in c


def test_cloud_corrections_empty_extras():
    assert ce.cloud_corrections(_Listing(None)) == {}
    assert ce.cloud_corrections(_Listing("not a dict")) == {}


# ---------------------------------------------------------------------------
# call_openrouter — model chain within the provider + request accounting
# ---------------------------------------------------------------------------

class _Resp:
    def __init__(self, status, payload=None, text=""):
        self.status_code = status
        self._payload = payload
        self.text = text

    def json(self):
        return self._payload


class _FakeClient:
    """Minimal httpx.Client stand-in returning scripted responses."""
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def post(self, url, json=None, headers=None):
        self.calls.append(json)
        return self._responses.pop(0)


def _content(obj):
    return {"choices": [{"message": {"content": json.dumps(obj)}}]}


def _make_cfg(providers=("openrouter",), **overrides):
    """Config in the shape get_llm_config() returns."""
    cfg = {
        "providers": list(providers),
        "providers_cfg": {
            "gemini": {
                "base_url": "https://generativelanguage.googleapis.com/v1beta",
                "models": ["gemini-flash-latest", "gemini-flash-lite-latest"],
                "timeout_seconds": 60, "max_attempts": 2, "retry_backoff_seconds": 0,
                "max_rpm": 1000, "slot_wait_seconds": 0, "daily_request_budget": 150,
            },
            "openrouter": {
                "base_url": "https://openrouter.ai/api/v1",
                "models": ["google/gemma-4-26b-a4b-it:free", "google/gemma-4-31b-it:free"],
                "timeout_seconds": 90, "max_attempts": 2, "retry_backoff_seconds": 0,
                "referer": "r", "title": "t", "daily_request_budget": 40,
            },
        },
        "temperature": 0.1, "max_tokens": 500, "max_chars": 2500,
        "per_run_request_cap": 20,
        "budget_state_file": "unused-in-unit-tests.json",
        "gate": {},
    }
    cfg.update(overrides)
    return cfg


@pytest.fixture
def _cfg():
    return _make_cfg()


@pytest.fixture(autouse=True)
def _clear_gemini_pacing():
    """The RPM window is module state — reset it between tests."""
    ce._GEMINI_CALLS.clear()
    yield
    ce._GEMINI_CALLS.clear()


def test_call_openrouter_success_first_model(monkeypatch, _cfg):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    fake = _FakeClient([_Resp(200, _content({"sub_model": "320d"}))])
    monkeypatch.setattr(ce.httpx, "Client", lambda **kw: fake)
    out, n = ce.call_openrouter("BMW 320d 180000km nacional impecável", _cfg)
    assert out == {"sub_model": "320d"}
    assert n == 1


def test_call_openrouter_429_falls_through_to_next_model(monkeypatch, _cfg):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    # model1: 429 twice (initial + 1 retry), model2: 200
    fake = _FakeClient([
        _Resp(429, text="rate limited"),
        _Resp(429, text="rate limited"),
        _Resp(200, _content({"urgency": "low"})),
    ])
    monkeypatch.setattr(ce.httpx, "Client", lambda **kw: fake)
    out, n = ce.call_openrouter("some long enough description text here", _cfg)
    assert out == {"urgency": "low"}
    assert n == 3


def test_call_openrouter_all_fail(monkeypatch, _cfg):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    fake = _FakeClient([_Resp(429), _Resp(429), _Resp(429), _Resp(429)])
    monkeypatch.setattr(ce.httpx, "Client", lambda **kw: fake)
    out, n = ce.call_openrouter("some long enough description text here", _cfg)
    assert out is None
    assert n == 4


def test_call_openrouter_401_aborts_immediately(monkeypatch, _cfg):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-bad")
    fake = _FakeClient([_Resp(401, text="unauthorized")])
    monkeypatch.setattr(ce.httpx, "Client", lambda **kw: fake)
    out, n = ce.call_openrouter("some long enough description text here", _cfg)
    assert out is None
    assert n == 1  # no fallback attempts on a hard 4xx


def test_call_openrouter_404_advances_to_next_model(monkeypatch, _cfg):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    # model1 delisted (404) → should try model2, which succeeds
    fake = _FakeClient([
        _Resp(404, text="No endpoints found"),
        _Resp(200, _content({"sub_model": "1.6 TDI"})),
    ])
    monkeypatch.setattr(ce.httpx, "Client", lambda **kw: fake)
    out, n = ce.call_openrouter("some long enough description text here", _cfg)
    assert out == {"sub_model": "1.6 TDI"}
    assert n == 2


def test_call_openrouter_402_no_credits_aborts(monkeypatch, _cfg):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    fake = _FakeClient([_Resp(402, text="insufficient credits")])
    monkeypatch.setattr(ce.httpx, "Client", lambda **kw: fake)
    out, n = ce.call_openrouter("some long enough description text here", _cfg)
    assert out is None and n == 1  # account-level → no fallback


def test_call_openrouter_no_key(monkeypatch, _cfg):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    out, n = ce.call_openrouter("text", _cfg)
    assert out is None and n == 0


def test_call_openrouter_respects_request_cap(monkeypatch, _cfg):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    fake = _FakeClient([_Resp(429), _Resp(429), _Resp(429), _Resp(429)])
    monkeypatch.setattr(ce.httpx, "Client", lambda **kw: fake)
    out, n = ce.call_openrouter("some long enough description text here", _cfg, request_cap=2)
    assert out is None
    assert n == 2


def test_enrich_from_description_short_desc_zero_requests(tmp_path, _cfg):
    ledger = ce.BudgetLedger(tmp_path / "b.json")
    out, n = ce.enrich_from_description("short", "", _cfg, ledger=ledger)
    assert out is None and n == 0
    assert ledger.total() == 0


def test_call_openrouter_malformed_200_falls_through(monkeypatch, _cfg):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    # model1: 200 but body has no 'choices'; model2: valid
    fake = _FakeClient([_Resp(200, {"unexpected": 1}), _Resp(200, _content({"warranty": True}))])
    monkeypatch.setattr(ce.httpx, "Client", lambda **kw: fake)
    out, n = ce.call_openrouter("some long enough description text here", _cfg)
    assert out == {"warranty": True}
    assert n == 2


def test_call_openrouter_non_json_content_falls_through(monkeypatch, _cfg):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    fake = _FakeClient([
        _Resp(200, {"choices": [{"message": {"content": "sorry, I cannot help"}}]}),
        _Resp(200, _content({"urgency": "low"})),
    ])
    monkeypatch.setattr(ce.httpx, "Client", lambda **kw: fake)
    out, n = ce.call_openrouter("some long enough description text here", _cfg)
    assert out == {"urgency": "low"}
    assert n == 2


def test_call_openrouter_response_format_dropped_only_for_gemma(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    cfg = _make_cfg()
    cfg["providers_cfg"]["openrouter"]["models"] = [
        "google/gemma-4-26b-a4b-it:free", "openai/gpt-oss-20b:free",
    ]
    cfg["providers_cfg"]["openrouter"]["max_attempts"] = 1
    # gemma (model1) 404 → advance; gpt-oss (model2) succeeds
    fake = _FakeClient([_Resp(404, text="x"), _Resp(200, _content({"sub_model": "A 200"}))])
    monkeypatch.setattr(ce.httpx, "Client", lambda **kw: fake)
    out, n = ce.call_openrouter("some long enough description text here", cfg)
    assert out == {"sub_model": "A 200"}
    assert "response_format" not in fake.calls[0]      # gemma body: dropped
    assert fake.calls[1]["response_format"] == {"type": "json_object"}  # non-gemma: kept


# ---------------------------------------------------------------------------
# Value gate — failure-mode robustness
# ---------------------------------------------------------------------------

def _patch_gate_loaders(monkeypatch, compute_signals_impl):
    monkeypatch.setattr("src.storage.repository.get_listings_df", lambda s: pd.DataFrame({"x": [1]}))
    monkeypatch.setattr("src.storage.repository.get_price_history_df", lambda s: pd.DataFrame())
    monkeypatch.setattr("src.analytics.computed_columns.enrich_listings", lambda df: df)
    monkeypatch.setattr("src.analytics.turnover.compute_turnover_stats", lambda df: pd.DataFrame())
    monkeypatch.setattr("src.parser.llm_enrichment.merge_real_mileage", lambda df: df)
    monkeypatch.setattr("src.dashboard.data_loader.compute_signals", compute_signals_impl)


_GATE = {"min_price_eur": 4000, "min_spec_fill": 0.5, "max_band_pct": 0.40,
         "min_discount_pct": 0.0, "max_discount_pct": 60.0}


def test_rank_deal_olx_ids_compute_signals_raises(monkeypatch):
    from src.analytics import value_gate

    def boom(l, h, turnover=None):
        raise RuntimeError("no fresh model")
    _patch_gate_loaders(monkeypatch, boom)
    assert value_gate.rank_deal_olx_ids(None, gate=_GATE, limit=5) == []


def test_rank_deal_olx_ids_missing_column(monkeypatch):
    from src.analytics import value_gate
    # signals lacking 'spec_fill' → gate cannot rank → []
    sig = pd.DataFrame([{"olx_id": "A", "price_eur": 9000, "undervaluation_pct": 30.0}])
    _patch_gate_loaders(monkeypatch, lambda l, h, turnover=None: (sig, None, None, None, None, None))
    assert value_gate.rank_deal_olx_ids(None, gate=_GATE, limit=5) == []


# ---------------------------------------------------------------------------
# CLI end-to-end — budget accrual + exclude-marker across two runs
# ---------------------------------------------------------------------------

def test_enrich_cloud_cli_e2e(tmp_path, monkeypatch, fresh_schema):
    """Two runs: the first enriches and spends, the second finds nothing to do."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from typer.testing import CliRunner
    from src.cli import app
    from src.models.listing import Listing

    engine = create_engine(fresh_schema)
    Listing.metadata.create_all(engine)
    TS = sessionmaker(bind=engine)
    seed = TS()
    seed.add_all([
        Listing(olx_id="A", url="http://a", brand="BMW", model="Serie 3",
                title="BMW 320d", description="BMW 320d 2015 nacional impecavel, 2 donos, negociavel"),
        Listing(olx_id="B", url="http://b", brand="Audi", model="A4",
                title="Audi A4", description="Audi A4 2.0 TDI 2016 nacional, garantia, sem acidentes"),
    ])
    seed.commit()
    seed.close()

    budget_file = tmp_path / "budget.json"
    cfg = _make_cfg(providers=("gemini", "openrouter"),
                    budget_state_file=str(budget_file), per_run_request_cap=5)

    all_ids = ["A", "B"]

    monkeypatch.setenv("GEMINI_API_KEY", "g-test")
    monkeypatch.setattr("src.cli.init_db", lambda *a, **k: None)
    monkeypatch.setattr("src.cli.get_session", lambda: TS())
    monkeypatch.setattr("src.parser.cloud_enrichment.get_llm_config", lambda: dict(cfg))
    monkeypatch.setattr(
        "src.analytics.value_gate.rank_deal_olx_ids",
        lambda session, *, gate, limit, exclude_ids=frozenset():
            [i for i in all_ids if i not in exclude_ids][:limit],
    )
    monkeypatch.setattr(
        "src.parser.cloud_enrichment.call_gemini",
        lambda text, cfg, request_cap=None: (
            {"mechanical_condition": "good", "desc_mentions_accident": False,
             "warranty": True, "urgency": "low"}, 1),
    )

    runner = CliRunner()
    r1 = runner.invoke(app, ["enrich-cloud"])
    assert r1.exit_code == 0, r1.output

    # both listings enriched by the first provider; 2 requests charged to it
    assert json.loads(budget_file.read_text())["providers"] == {"gemini": 2}
    chk = TS()
    rows = {l.olx_id: l for l in chk.query(Listing).all()}
    assert rows["A"].mechanical_condition == "good"
    assert rows["A"].warranty is True
    assert "_or_enriched" in json.loads(rows["A"].llm_extras)
    chk.close()

    # second run: both already marked → excluded → no new spend
    r2 = runner.invoke(app, ["enrich-cloud"])
    assert r2.exit_code == 0, r2.output
    assert json.loads(budget_file.read_text())["providers"] == {"gemini": 2}

    # The CLI does not close the session it is handed, and an
    # idle-in-transaction backend would block the schema teardown.
    TS.close_all()
    engine.dispose()


def test_enrich_cloud_cli_noop_without_any_key(tmp_path, monkeypatch):
    """No key anywhere → exit 0, no DB touched, no budget file written."""
    from typer.testing import CliRunner
    from src.cli import app

    budget_file = tmp_path / "budget.json"
    cfg = _make_cfg(providers=("gemini", "openrouter"), budget_state_file=str(budget_file))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr("src.parser.cloud_enrichment.get_llm_config", lambda: dict(cfg))

    def _boom():
        raise AssertionError("must not open the DB when no provider has a key")

    monkeypatch.setattr("src.cli.init_db", _boom)

    r = CliRunner().invoke(app, ["enrich-cloud"])
    assert r.exit_code == 0, r.output
    assert not budget_file.exists()


def test_enrich_cloud_cli_dry_run_spends_nothing(tmp_path, monkeypatch):
    """--dry-run prints the ranked candidates and calls no provider."""
    from typer.testing import CliRunner
    from src.cli import app

    budget_file = tmp_path / "budget.json"
    cfg = _make_cfg(providers=("gemini",), budget_state_file=str(budget_file))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr("src.cli.init_db", lambda *a, **k: None)
    monkeypatch.setattr("src.cli.get_session", lambda: None)
    monkeypatch.setattr("src.parser.cloud_enrichment.get_llm_config", lambda: dict(cfg))
    monkeypatch.setattr(
        "src.analytics.value_gate.rank_deal_olx_ids",
        lambda session, *, gate, limit, exclude_ids=frozenset(): ["Z1", "Z2"],
    )

    def _no_calls(*a, **k):
        raise AssertionError("dry run must not call a provider")

    monkeypatch.setattr("src.parser.cloud_enrichment.call_gemini", _no_calls)
    # the exclude-marker query needs a session; dry run still runs it, so give
    # it one that answers "nothing enriched yet"
    monkeypatch.setattr("src.cli.get_session", lambda: _FakeSession())

    r = CliRunner().invoke(app, ["enrich-cloud", "--dry-run"])
    assert r.exit_code == 0, r.output
    assert not budget_file.exists()


class _FakeSession:
    """Answers the 'already enriched' query with an empty set."""

    def query(self, *a, **k):
        return self

    def filter(self, *a, **k):
        return self

    def all(self):
        return []


# ---------------------------------------------------------------------------
# BudgetLedger — per-provider daily spend
# ---------------------------------------------------------------------------

def test_ledger_roundtrip(tmp_path, _cfg):
    state = tmp_path / "budget.json"
    ledger = ce.BudgetLedger(state)
    ledger.charge("gemini", 3)
    ledger.charge("openrouter", 2)
    assert ce.BudgetLedger(state).used("gemini") == 3
    assert ce.BudgetLedger(state).used("openrouter") == 2
    assert ce.BudgetLedger(state).total() == 5


def test_ledger_resets_on_a_new_day(tmp_path):
    state = tmp_path / "budget.json"
    state.write_text(json.dumps({"date": "2000-01-01",
                                 "providers": {"gemini": 99, "openrouter": 40}}))
    ledger = ce.BudgetLedger(state)
    assert ledger.used("gemini") == 0
    assert ledger.used("openrouter") == 0


def test_ledger_survives_corrupt_state(tmp_path):
    state = tmp_path / "budget.json"
    state.write_text("{not json")
    assert ce.BudgetLedger(state).total() == 0


def test_ledger_remaining_is_per_provider(tmp_path, _cfg):
    ledger = ce.BudgetLedger(tmp_path / "b.json")
    ledger.charge("openrouter", 38)
    assert ledger.remaining("openrouter", _cfg) == 2     # ceiling 40
    assert ledger.remaining("gemini", _cfg) == 150       # untouched


def test_ledger_persists_on_every_charge(tmp_path):
    """A killed run must not lose the spend it already made."""
    state = tmp_path / "b.json"
    ledger = ce.BudgetLedger(state)
    ledger.charge("gemini", 1)
    # no explicit save() here — simulate the process dying right after
    assert json.loads(state.read_text())["providers"]["gemini"] == 1


# ---------------------------------------------------------------------------
# Value gate — filter + rank + exclude + limit
# ---------------------------------------------------------------------------

def test_rank_deal_olx_ids(monkeypatch):
    from src.analytics import value_gate

    signals = pd.DataFrame([
        {"olx_id": "keep_top", "price_eur": 12000, "undervaluation_pct": 30.0, "spec_fill": 1.0},
        {"olx_id": "keep_mid", "price_eur": 9000, "undervaluation_pct": 20.0, "spec_fill": 0.75},
        {"olx_id": "cheap_tail", "price_eur": 3000, "undervaluation_pct": 50.0, "spec_fill": 1.0},   # < €4k
        {"olx_id": "low_spec", "price_eur": 8000, "undervaluation_pct": 40.0, "spec_fill": 0.25},    # spec<0.5
        {"olx_id": "already", "price_eur": 15000, "undervaluation_pct": 35.0, "spec_fill": 1.0},     # excluded
        {"olx_id": "not_under", "price_eur": 8000, "undervaluation_pct": 0.0, "spec_fill": 1.0},     # disc<=0
    ])

    def fake_compute_signals(listings, history, turnover=None):
        return signals, None, None, None, None, None

    monkeypatch.setattr("src.storage.repository.get_listings_df", lambda s: pd.DataFrame({"x": [1]}))
    monkeypatch.setattr("src.storage.repository.get_price_history_df", lambda s: pd.DataFrame())
    monkeypatch.setattr("src.analytics.computed_columns.enrich_listings", lambda df: df)
    monkeypatch.setattr("src.analytics.turnover.compute_turnover_stats", lambda df: pd.DataFrame())
    monkeypatch.setattr("src.parser.llm_enrichment.merge_real_mileage", lambda df: df)
    monkeypatch.setattr("src.dashboard.data_loader.compute_signals", fake_compute_signals)

    gate = {"min_price_eur": 4000, "min_spec_fill": 0.5, "max_band_pct": 0.40,
            "min_discount_pct": 0.0, "max_discount_pct": 60.0}
    ids = value_gate.rank_deal_olx_ids(None, gate=gate, limit=10, exclude_ids={"already"})
    # cheap_tail, low_spec, already, not_under all filtered → ranked by discount desc
    assert ids == ["keep_top", "keep_mid"]


def test_rank_deal_olx_ids_respects_limit(monkeypatch):
    from src.analytics import value_gate
    signals = pd.DataFrame([
        {"olx_id": f"d{i}", "price_eur": 10000, "undervaluation_pct": float(50 - i), "spec_fill": 1.0}
        for i in range(10)
    ])
    monkeypatch.setattr("src.storage.repository.get_listings_df", lambda s: pd.DataFrame({"x": [1]}))
    monkeypatch.setattr("src.storage.repository.get_price_history_df", lambda s: pd.DataFrame())
    monkeypatch.setattr("src.analytics.computed_columns.enrich_listings", lambda df: df)
    monkeypatch.setattr("src.analytics.turnover.compute_turnover_stats", lambda df: pd.DataFrame())
    monkeypatch.setattr("src.parser.llm_enrichment.merge_real_mileage", lambda df: df)
    monkeypatch.setattr("src.dashboard.data_loader.compute_signals",
                        lambda l, h, turnover=None: (signals, None, None, None, None, None))
    gate = {"min_price_eur": 4000, "min_spec_fill": 0.5, "max_band_pct": 0.40,
            "min_discount_pct": 0.0, "max_discount_pct": 60.0}
    ids = value_gate.rank_deal_olx_ids(None, gate=gate, limit=3)
    assert ids == ["d0", "d1", "d2"]


def test_rank_deal_olx_ids_empty_signals(monkeypatch):
    from src.analytics import value_gate
    monkeypatch.setattr("src.storage.repository.get_listings_df", lambda s: pd.DataFrame({"x": [1]}))
    monkeypatch.setattr("src.storage.repository.get_price_history_df", lambda s: pd.DataFrame())
    monkeypatch.setattr("src.analytics.computed_columns.enrich_listings", lambda df: df)
    monkeypatch.setattr("src.analytics.turnover.compute_turnover_stats", lambda df: pd.DataFrame())
    monkeypatch.setattr("src.parser.llm_enrichment.merge_real_mileage", lambda df: df)
    monkeypatch.setattr("src.dashboard.data_loader.compute_signals",
                        lambda l, h, turnover=None: (pd.DataFrame(), None, None, None, None, None))
    gate = {"min_price_eur": 4000, "min_spec_fill": 0.5, "max_band_pct": 0.40,
            "min_discount_pct": 0.0, "max_discount_pct": 60.0}
    assert value_gate.rank_deal_olx_ids(None, gate=gate, limit=5) == []


# ---------------------------------------------------------------------------
# call_gemini — model chain, retry-vs-abandon, JSON mode
# ---------------------------------------------------------------------------

def _gem(obj):
    return {"candidates": [{"content": {"parts": [{"text": json.dumps(obj)}]}}]}


def test_call_gemini_success(monkeypatch, _cfg):
    monkeypatch.setenv("GEMINI_API_KEY", "g-test")
    fake = _FakeClient([_Resp(200, _gem({"sub_model": "320d"}))])
    monkeypatch.setattr(ce.httpx, "Client", lambda **kw: fake)
    out, n = ce.call_gemini("BMW 320d 180000 km nacional impecável", _cfg)
    assert out == {"sub_model": "320d"} and n == 1


def test_call_gemini_asks_for_json_and_no_thinking(monkeypatch, _cfg):
    """Both matter: thinking eats the token budget, JSON mode removes fences."""
    monkeypatch.setenv("GEMINI_API_KEY", "g-test")
    fake = _FakeClient([_Resp(200, _gem({"warranty": True}))])
    monkeypatch.setattr(ce.httpx, "Client", lambda **kw: fake)
    ce.call_gemini("some long enough description text here", _cfg)
    gen = fake.calls[0]["generationConfig"]
    assert gen["responseMimeType"] == "application/json"
    assert gen["thinkingConfig"]["thinkingBudget"] == 0


def test_call_gemini_503_retries_then_succeeds(monkeypatch, _cfg):
    """Free-tier Gemini serves 'overloaded' in bursts; the next try lands."""
    monkeypatch.setenv("GEMINI_API_KEY", "g-test")
    fake = _FakeClient([_Resp(503, text="overloaded"), _Resp(200, _gem({"urgency": "low"}))])
    monkeypatch.setattr(ce.httpx, "Client", lambda **kw: fake)
    out, n = ce.call_gemini("some long enough description text here", _cfg)
    assert out == {"urgency": "low"} and n == 2


def test_call_gemini_429_advances_to_the_next_model(monkeypatch, _cfg):
    """Gemini quotas are per MODEL, so an exhausted model must not kill the provider.

    Measured 2026-08-23: the free tier allows ~20 requests/day for EACH model
    and the buckets are separate — flash was answering 429 while flash-lite
    served the identical request. Abandoning the provider on the first 429
    would throw away the second model's untouched daily allowance.
    """
    monkeypatch.setenv("GEMINI_API_KEY", "g-test")
    fake = _FakeClient([_Resp(429, text="quota"), _Resp(200, _gem({"a": 1}))])
    monkeypatch.setattr(ce.httpx, "Client", lambda **kw: fake)
    out, n = ce.call_gemini("some long enough description text here", _cfg)
    assert out == {"a": 1} and n == 2


def test_call_gemini_429_does_not_retry_the_same_model(monkeypatch, _cfg):
    """The window is a day wide — an immediate retry is a guaranteed second 429."""
    monkeypatch.setenv("GEMINI_API_KEY", "g-test")
    cfg = _make_cfg()
    cfg["providers_cfg"]["gemini"]["models"] = ["only-model"]
    fake = _FakeClient([_Resp(429, text="quota"), _Resp(200, _gem({"a": 1}))])
    monkeypatch.setattr(ce.httpx, "Client", lambda **kw: fake)
    out, n = ce.call_gemini("some long enough description text here", cfg)
    assert out is None and n == 1


def test_call_gemini_403_abandons_provider(monkeypatch, _cfg):
    monkeypatch.setenv("GEMINI_API_KEY", "g-bad")
    fake = _FakeClient([_Resp(403, text="key revoked")])
    monkeypatch.setattr(ce.httpx, "Client", lambda **kw: fake)
    out, n = ce.call_gemini("some long enough description text here", _cfg)
    assert out is None and n == 1


def test_call_gemini_empty_candidate_advances_model(monkeypatch, _cfg):
    monkeypatch.setenv("GEMINI_API_KEY", "g-test")
    fake = _FakeClient([
        _Resp(200, {"candidates": [{"finishReason": "MAX_TOKENS", "content": {"parts": []}}]}),
        _Resp(200, _gem({"sub_model": "1.6 TDI"})),
    ])
    monkeypatch.setattr(ce.httpx, "Client", lambda **kw: fake)
    out, n = ce.call_gemini("some long enough description text here", _cfg)
    assert out == {"sub_model": "1.6 TDI"} and n == 2


def test_call_gemini_no_key(monkeypatch, _cfg):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    out, n = ce.call_gemini("some long enough description text here", _cfg)
    assert out is None and n == 0


def test_gemini_slot_limiter_gives_up_instead_of_waiting(monkeypatch, _cfg):
    """Out of pace → no request at all, so the cascade can move on fast."""
    monkeypatch.setenv("GEMINI_API_KEY", "g-test")
    pcfg = dict(_cfg["providers_cfg"]["gemini"], max_rpm=1, slot_wait_seconds=0)
    _cfg["providers_cfg"]["gemini"] = pcfg
    fake = _FakeClient([_Resp(200, _gem({"a": 1})), _Resp(200, _gem({"b": 2}))])
    monkeypatch.setattr(ce.httpx, "Client", lambda **kw: fake)
    first, n1 = ce.call_gemini("some long enough description text here", _cfg)
    second, n2 = ce.call_gemini("another long enough description text here", _cfg)
    assert first == {"a": 1} and n1 == 1
    assert second is None and n2 == 0      # second never left the process


# ---------------------------------------------------------------------------
# The cascade — order, fallback, budget skipping, charging
# ---------------------------------------------------------------------------

def _stub_provider(monkeypatch, name, result, n_req):
    monkeypatch.setattr(f"src.parser.cloud_enrichment.call_{name}",
                        lambda text, cfg, request_cap=None: (result, n_req))


def test_cascade_uses_first_provider_when_it_answers(tmp_path, monkeypatch):
    cfg = _make_cfg(providers=("gemini", "openrouter"))
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setenv("OPENROUTER_API_KEY", "o")
    _stub_provider(monkeypatch, "gemini", {"sub_model": "320d"}, 1)
    _stub_provider(monkeypatch, "openrouter", {"sub_model": "WRONG"}, 1)
    ledger = ce.BudgetLedger(tmp_path / "b.json")
    out, n = ce.call_llm("text", cfg, ledger=ledger)
    assert out == {"sub_model": "320d"} and n == 1
    assert ledger.used("gemini") == 1
    assert ledger.used("openrouter") == 0


def test_cascade_falls_back_to_second_provider(tmp_path, monkeypatch):
    cfg = _make_cfg(providers=("gemini", "openrouter"))
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setenv("OPENROUTER_API_KEY", "o")
    _stub_provider(monkeypatch, "gemini", None, 2)          # tried and failed
    _stub_provider(monkeypatch, "openrouter", {"urgency": "low"}, 1)
    ledger = ce.BudgetLedger(tmp_path / "b.json")
    out, n = ce.call_llm("text", cfg, ledger=ledger)
    assert out == {"urgency": "low"}
    assert n == 3                                            # 2 wasted + 1 good
    assert ledger.used("gemini") == 2                        # failures still cost
    assert ledger.used("openrouter") == 1


def test_cascade_skips_provider_without_key_for_free(tmp_path, monkeypatch):
    cfg = _make_cfg(providers=("gemini", "openrouter"))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "o")

    def _boom(*a, **k):
        raise AssertionError("keyless provider must not be called")

    monkeypatch.setattr(ce, "call_gemini", _boom)
    _stub_provider(monkeypatch, "openrouter", {"warranty": True}, 1)
    ledger = ce.BudgetLedger(tmp_path / "b.json")
    out, n = ce.call_llm("text", cfg, ledger=ledger)
    assert out == {"warranty": True} and n == 1
    assert ledger.used("gemini") == 0


def test_cascade_skips_provider_whose_daily_budget_is_spent(tmp_path, monkeypatch):
    cfg = _make_cfg(providers=("gemini", "openrouter"))
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setenv("OPENROUTER_API_KEY", "o")

    def _boom(*a, **k):
        raise AssertionError("exhausted provider must not be called")

    monkeypatch.setattr(ce, "call_gemini", _boom)
    _stub_provider(monkeypatch, "openrouter", {"urgency": "high"}, 1)
    ledger = ce.BudgetLedger(tmp_path / "b.json")
    ledger.charge("gemini", 150)                             # ceiling reached
    out, _ = ce.call_llm("text", cfg, ledger=ledger)
    assert out == {"urgency": "high"}


def test_cascade_returns_none_when_every_provider_refuses(tmp_path, monkeypatch):
    cfg = _make_cfg(providers=("gemini", "openrouter"))
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setenv("OPENROUTER_API_KEY", "o")
    _stub_provider(monkeypatch, "gemini", None, 1)
    _stub_provider(monkeypatch, "openrouter", None, 1)
    ledger = ce.BudgetLedger(tmp_path / "b.json")
    out, n = ce.call_llm("text", cfg, ledger=ledger)
    assert out is None and n == 2


def test_cascade_provider_exception_does_not_abort_the_run(tmp_path, monkeypatch):
    cfg = _make_cfg(providers=("gemini", "openrouter"))
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setenv("OPENROUTER_API_KEY", "o")

    def _raise(*a, **k):
        raise RuntimeError("provider SDK blew up")

    monkeypatch.setattr(ce, "call_gemini", _raise)
    _stub_provider(monkeypatch, "openrouter", {"urgency": "low"}, 1)
    ledger = ce.BudgetLedger(tmp_path / "b.json")
    out, _ = ce.call_llm("text", cfg, ledger=ledger)
    assert out == {"urgency": "low"}


def test_cascade_request_cap_bounds_the_whole_chain(tmp_path, monkeypatch):
    """One wedged listing must not drain the run across providers."""
    cfg = _make_cfg(providers=("gemini", "openrouter"))
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setenv("OPENROUTER_API_KEY", "o")
    _stub_provider(monkeypatch, "gemini", None, 2)
    _stub_provider(monkeypatch, "openrouter", None, 2)
    ledger = ce.BudgetLedger(tmp_path / "b.json")
    out, n = ce.call_llm("text", cfg, ledger=ledger, request_cap=2)
    assert out is None
    assert n == 2                                            # stopped after gemini


def test_unknown_provider_name_is_dropped_not_fatal(monkeypatch, tmp_path):
    cfg_yaml = tmp_path / "settings.yaml"
    cfg_yaml.write_text(
        "llm:\n  providers:\n    - typo-provider\n    - openrouter\n"
    )
    monkeypatch.setattr(ce, "CONFIG_PATH", cfg_yaml)
    assert ce.get_llm_config()["providers"] == ["openrouter"]


def test_config_falls_back_when_every_provider_name_is_bogus(monkeypatch, tmp_path):
    cfg_yaml = tmp_path / "settings.yaml"
    cfg_yaml.write_text("llm:\n  providers:\n    - nonsense\n")
    monkeypatch.setattr(ce, "CONFIG_PATH", cfg_yaml)
    assert ce.get_llm_config()["providers"] == ["gemini"]


def test_real_settings_yaml_declares_the_cascade():
    """The shipped config must actually wire Gemini → OpenRouter."""
    cfg = ce.get_llm_config()
    assert cfg["providers"] == ["gemini", "openrouter"]
    assert cfg["providers_cfg"]["gemini"]["models"]
    assert cfg["providers_cfg"]["openrouter"]["models"]
    assert cfg["gate"]["min_price_eur"] == 4000


def test_call_gemini_invalid_key_400_abandons_provider(monkeypatch, _cfg):
    """Gemini answers a bad key with 400 INVALID_ARGUMENT, not 401.

    Without special-casing it, a dead key looks like a per-model problem and
    burns one request on every model in the chain.
    """
    monkeypatch.setenv("GEMINI_API_KEY", "not-a-real-key")
    fake = _FakeClient([
        _Resp(400, text='{"error":{"code":400,"message":"API key not valid. Please pass a valid API key."}}'),
        _Resp(200, _gem({"a": 1})),
    ])
    monkeypatch.setattr(ce.httpx, "Client", lambda **kw: fake)
    out, n = ce.call_gemini("some long enough description text here", _cfg)
    assert out is None
    assert n == 1


def test_openrouter_disables_reasoning(monkeypatch, _cfg):
    """A reasoning model would spend the token budget thinking and return no content."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    fake = _FakeClient([_Resp(200, _content({"urgency": "low"}))])
    monkeypatch.setattr(ce.httpx, "Client", lambda **kw: fake)
    ce.call_openrouter("some long enough description text here", _cfg)
    assert fake.calls[0]["reasoning"] == {"enabled": False}


# ---------------------------------------------------------------------------
# Value gate — a deal with no description is not a candidate
# ---------------------------------------------------------------------------

def _gate_with_listings(monkeypatch, listings, signals):
    monkeypatch.setattr("src.storage.repository.get_listings_df", lambda s: listings)
    monkeypatch.setattr("src.storage.repository.get_price_history_df", lambda s: pd.DataFrame())
    monkeypatch.setattr("src.analytics.computed_columns.enrich_listings", lambda df: df)
    monkeypatch.setattr("src.analytics.turnover.compute_turnover_stats", lambda df: pd.DataFrame())
    monkeypatch.setattr("src.parser.llm_enrichment.merge_real_mileage", lambda df: df)
    monkeypatch.setattr("src.dashboard.data_loader.compute_signals",
                        lambda l, h, turnover=None: (signals, None, None, None, None, None))


_GATE = {"min_price_eur": 4000, "min_spec_fill": 0.5, "max_band_pct": 0.40,
         "min_discount_pct": 0.0, "max_discount_pct": 60.0}


def test_gate_skips_deals_with_no_description(monkeypatch):
    """The exact production failure: StandVirtual rows rank top and carry no text.

    Undervaluation comes from structured fields, so a description-less listing
    can outrank everything and then produce nothing — the run spends zero
    requests and reports success while doing no work.
    """
    from src.analytics import value_gate

    listings = pd.DataFrame([
        {"olx_id": "sv_top", "description": None},
        {"olx_id": "sv_blank", "description": "   "},
        {"olx_id": "olx_readable", "description": "Vendo BMW 320d nacional, 2 donos, com garantia."},
    ])
    signals = pd.DataFrame([
        {"olx_id": "sv_top", "price_eur": 15000, "undervaluation_pct": 40.0, "spec_fill": 1.0},
        {"olx_id": "sv_blank", "price_eur": 14000, "undervaluation_pct": 35.0, "spec_fill": 1.0},
        {"olx_id": "olx_readable", "price_eur": 12000, "undervaluation_pct": 20.0, "spec_fill": 1.0},
    ])
    _gate_with_listings(monkeypatch, listings, signals)
    ids = value_gate.rank_deal_olx_ids(None, gate=_GATE, limit=10)
    assert ids == ["olx_readable"]


def test_gate_keeps_ranking_among_readable_deals(monkeypatch):
    from src.analytics import value_gate

    listings = pd.DataFrame([
        {"olx_id": f"d{i}", "description": "Descrição suficientemente longa para ler."}
        for i in range(3)
    ])
    signals = pd.DataFrame([
        {"olx_id": "d0", "price_eur": 9000, "undervaluation_pct": 10.0, "spec_fill": 1.0},
        {"olx_id": "d1", "price_eur": 9000, "undervaluation_pct": 30.0, "spec_fill": 1.0},
        {"olx_id": "d2", "price_eur": 9000, "undervaluation_pct": 20.0, "spec_fill": 1.0},
    ])
    _gate_with_listings(monkeypatch, listings, signals)
    assert value_gate.rank_deal_olx_ids(None, gate=_GATE, limit=10) == ["d1", "d2", "d0"]


def test_gate_survives_listings_without_a_description_column(monkeypatch):
    """Older frames may not carry it — warn, don't crash a scheduled run."""
    from src.analytics import value_gate

    listings = pd.DataFrame([{"olx_id": "x"}])
    signals = pd.DataFrame([
        {"olx_id": "x", "price_eur": 9000, "undervaluation_pct": 20.0, "spec_fill": 1.0},
    ])
    _gate_with_listings(monkeypatch, listings, signals)
    assert value_gate.rank_deal_olx_ids(None, gate=_GATE, limit=5) == ["x"]


# ---------------------------------------------------------------------------
# thinkingConfig — needed by flash, rejected by flash-lite
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_thinking_memo():
    ce._NO_THINKING_MODELS.clear()
    yield
    ce._NO_THINKING_MODELS.clear()


def test_gemini_strips_thinking_knob_when_the_model_rejects_it(monkeypatch, _cfg):
    """flash-lite answers 400 to thinkingConfig — that's a payload fault, not a model fault."""
    monkeypatch.setenv("GEMINI_API_KEY", "g-test")
    fake = _FakeClient([
        _Resp(400, text='{"error":{"code":400,"message":"Request contains an invalid argument."}}'),
        _Resp(200, _gem({"sub_model": "320d"})),
    ])
    monkeypatch.setattr(ce.httpx, "Client", lambda **kw: fake)
    out, n = ce.call_gemini("some long enough description text here", _cfg)
    assert out == {"sub_model": "320d"} and n == 2
    assert "thinkingConfig" in fake.calls[0]["generationConfig"]
    assert "thinkingConfig" not in fake.calls[1]["generationConfig"]


def test_thinking_rejection_is_remembered_across_calls(monkeypatch, _cfg):
    """Otherwise the discovery costs a wasted request on every single listing."""
    monkeypatch.setenv("GEMINI_API_KEY", "g-test")
    first = _FakeClient([
        _Resp(400, text="invalid argument"),
        _Resp(200, _gem({"a": 1})),
    ])
    monkeypatch.setattr(ce.httpx, "Client", lambda **kw: first)
    ce.call_gemini("some long enough description text here", _cfg)

    second = _FakeClient([_Resp(200, _gem({"b": 2}))])
    monkeypatch.setattr(ce.httpx, "Client", lambda **kw: second)
    out, n = ce.call_gemini("another long enough description text here", _cfg)
    assert out == {"b": 2}
    assert n == 1                                        # no wasted probe this time
    assert "thinkingConfig" not in second.calls[0]["generationConfig"]


def test_bad_key_400_is_not_mistaken_for_a_thinking_rejection(monkeypatch, _cfg):
    monkeypatch.setenv("GEMINI_API_KEY", "bad")
    fake = _FakeClient([
        _Resp(400, text='{"error":{"message":"API key not valid. Please pass a valid API key."}}'),
        _Resp(200, _gem({"a": 1})),
    ])
    monkeypatch.setattr(ce.httpx, "Client", lambda **kw: fake)
    out, n = ce.call_gemini("some long enough description text here", _cfg)
    assert out is None and n == 1
    assert not ce._NO_THINKING_MODELS
