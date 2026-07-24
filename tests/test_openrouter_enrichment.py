"""Tests for the value-gated OpenRouter enrichment path."""
import json
import types

import pandas as pd
import pytest

from src.parser import openrouter_enrichment as ore


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def test_parse_json_object_plain():
    assert ore._parse_json_object('{"a": 1}') == {"a": 1}


def test_parse_json_object_fenced():
    assert ore._parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert ore._parse_json_object('```\n{"a": 2}\n```') == {"a": 2}


def test_parse_json_object_with_prose_around():
    assert ore._parse_json_object('Here you go: {"a": 1, "b": true} done') == {"a": 1, "b": True}


def test_parse_json_object_garbage():
    assert ore._parse_json_object("not json at all") is None
    assert ore._parse_json_object("") is None
    assert ore._parse_json_object("[1,2,3]") is None  # array is not an object


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


def test_openrouter_corrections_full_valid():
    extras = {
        "sub_model": "320d", "trim_level": "M Sport",
        "mileage_in_description_km": 180000,
        "mechanical_condition": "good",
        "desc_mentions_accident": False, "desc_mentions_repair": True,
        "desc_mentions_num_owners": 2, "desc_mentions_customs_cleared": False,
        "right_hand_drive": False, "warranty": True, "urgency": "high",
        "negotiable": True, "red_flags": ["km alta"], "deal_note_pt": "bom",
    }
    c = ore.openrouter_corrections(_Listing(extras))
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


def test_openrouter_corrections_rejects_bad_enum():
    extras = {"mechanical_condition": "razoavel", "urgency": "baixa"}  # PT, not the English vocab
    c = ore.openrouter_corrections(_Listing(extras))
    assert "mechanical_condition" not in c
    assert "urgency" not in c


def test_openrouter_corrections_enum_case_normalized():
    c = ore.openrouter_corrections(_Listing({"urgency": "HIGH", "mechanical_condition": "Excellent"}))
    assert c["urgency"] == "high"
    assert c["mechanical_condition"] == "excellent"


def test_openrouter_corrections_bool_type_strict():
    # a truthy non-bool must NOT be written to a bool column
    c = ore.openrouter_corrections(_Listing({"warranty": "yes", "desc_mentions_accident": 1}))
    assert "warranty" not in c
    assert "desc_mentions_accident" not in c


def test_openrouter_corrections_num_owners_rejects_bool():
    c = ore.openrouter_corrections(_Listing({"desc_mentions_num_owners": True}))
    assert "desc_mentions_num_owners" not in c


def test_openrouter_corrections_empty_extras():
    assert ore.openrouter_corrections(_Listing(None)) == {}
    assert ore.openrouter_corrections(_Listing("not a dict")) == {}


# ---------------------------------------------------------------------------
# call_openrouter — model fallback chain + request accounting
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


@pytest.fixture
def _cfg():
    return {
        "base_url": "https://openrouter.ai/api/v1",
        "models": ["google/gemma-4-26b-a4b-it:free", "google/gemma-4-31b-it:free"],
        "temperature": 0.1, "max_tokens": 500, "max_chars": 2500,
        "timeout_seconds": 90, "max_retries": 1,
        "referer": "r", "title": "t",
    }


def test_call_openrouter_success_first_model(monkeypatch, _cfg):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    fake = _FakeClient([_Resp(200, _content({"sub_model": "320d"}))])
    monkeypatch.setattr(ore.httpx, "Client", lambda **kw: fake)
    out, n = ore.call_openrouter("BMW 320d 180000km nacional impecável", _cfg)
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
    monkeypatch.setattr(ore.httpx, "Client", lambda **kw: fake)
    out, n = ore.call_openrouter("some long enough description text here", _cfg)
    assert out == {"urgency": "low"}
    assert n == 3


def test_call_openrouter_all_fail(monkeypatch, _cfg):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    fake = _FakeClient([_Resp(429), _Resp(429), _Resp(429), _Resp(429)])
    monkeypatch.setattr(ore.httpx, "Client", lambda **kw: fake)
    out, n = ore.call_openrouter("some long enough description text here", _cfg)
    assert out is None
    assert n == 4


def test_call_openrouter_401_aborts_immediately(monkeypatch, _cfg):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-bad")
    fake = _FakeClient([_Resp(401, text="unauthorized")])
    monkeypatch.setattr(ore.httpx, "Client", lambda **kw: fake)
    out, n = ore.call_openrouter("some long enough description text here", _cfg)
    assert out is None
    assert n == 1  # no fallback attempts on a hard 4xx


def test_call_openrouter_no_key(monkeypatch, _cfg):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    out, n = ore.call_openrouter("text", _cfg)
    assert out is None and n == 0


def test_call_openrouter_respects_request_cap(monkeypatch, _cfg):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    fake = _FakeClient([_Resp(429), _Resp(429), _Resp(429), _Resp(429)])
    monkeypatch.setattr(ore.httpx, "Client", lambda **kw: fake)
    out, n = ore.call_openrouter("some long enough description text here", _cfg, request_cap=2)
    assert out is None
    assert n == 2


def test_enrich_from_description_short_desc_zero_requests(_cfg):
    out, n = ore.enrich_from_description("short", "", _cfg)
    assert out is None and n == 0


# ---------------------------------------------------------------------------
# Budget state file
# ---------------------------------------------------------------------------

def test_budget_roundtrip_and_reset(tmp_path, monkeypatch):
    state = tmp_path / "budget.json"
    ore.save_budget(state, 7)
    assert ore.load_budget(state)["requests"] == 7
    # stale date → reset to 0
    state.write_text(json.dumps({"date": "2000-01-01", "requests": 99}))
    assert ore.load_budget(state)["requests"] == 0


def test_remaining_daily(tmp_path):
    state = tmp_path / "b.json"
    ore.save_budget(state, 30)
    cfg = {"budget_state_file": str(state), "daily_request_budget": 40}
    assert ore.remaining_daily(cfg) == 10


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
