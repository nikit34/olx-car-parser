"""The vision photo veto: what it blocks, what it must not block, what it costs.

This replaced a classifier that vetoed at precision 0.20 — four good listings
removed for every bad one. The tests here pin the properties that made the
replacement worth making: it only fires on damage that needs repair, it never
fires on a car it could not see, and one deal costs one request.
"""

import json

import pytest

from src.parser import photo_verdict as pv
from src.dashboard.data_loader import _blocking_deal_reason


class _Resp:
    def __init__(self, status, payload=None, text=""):
        self.status_code = status
        self._payload = payload
        self.text = text

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def post(self, url, json=None, headers=None):
        self.calls.append((url, json))
        return self._responses.pop(0)


def _gem(obj):
    return {"candidates": [{"content": {"parts": [{"text": json.dumps(obj)}]}}]}


@pytest.fixture
def cfg():
    c = pv.get_vision_config()
    c["models"] = ["m1", "m2"]
    c["max_rpm"] = 1000
    c["slot_wait_seconds"] = 0
    c["retry_backoff_seconds"] = 0
    return c


@pytest.fixture(autouse=True)
def _clean():
    from src.parser.cloud_enrichment import _GEMINI_CALLS, _NO_THINKING_MODELS
    _GEMINI_CALLS.clear()
    _NO_THINKING_MODELS.clear()
    yield
    _GEMINI_CALLS.clear()
    _NO_THINKING_MODELS.clear()


# ---------------------------------------------------------------------------
# The veto rule
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sev,blocks", [(0, False), (1, False), (2, True), (3, True)])
def test_only_repair_worthy_damage_blocks(sev, blocks, cfg):
    """Severity 1 is wear for age. Vetoing on it is the old mistake."""
    assert pv.verdict_blocks({"severity": sev}, cfg) is blocks


def test_garbage_verdict_does_not_block(cfg):
    for bad in (None, {}, {"severity": None}, {"severity": "bad"}, "nope"):
        assert pv.verdict_blocks(bad, cfg) is False


def test_blocking_reason_reads_the_verdict():
    row = {"llm_extras": json.dumps(
        {"vlm_damage": {"severity": 3, "evidence": "front end destroyed"}})}
    reason = _blocking_deal_reason(row)
    assert reason and "front end destroyed" in reason


def test_blocking_reason_ignores_mild_verdicts():
    row = {"llm_extras": json.dumps(
        {"vlm_damage": {"severity": 1, "evidence": "faded lamps"}})}
    assert _blocking_deal_reason(row) is None


def test_the_old_classifier_no_longer_vetoes():
    """photo_damage_flagged used to remove the listing; now it only weighs."""
    row = {"llm_extras": json.dumps(
        {"photo_damage_p": 0.99, "photo_damage_flagged": True})}
    assert _blocking_deal_reason(row) is None


# ---------------------------------------------------------------------------
# The call
# ---------------------------------------------------------------------------

def test_one_deal_costs_one_request(monkeypatch, cfg):
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    fake = _FakeClient([_Resp(200, _gem({"severity": 2, "visible_damage": True}))])
    monkeypatch.setattr(pv.httpx, "Client", lambda **kw: fake)
    out, n = pv.judge_sheet(b"jpegbytes", cfg)
    assert out["severity"] == 2 and n == 1


def test_photos_go_up_as_one_image(monkeypatch, cfg):
    """Three frames in one request, not three requests."""
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    fake = _FakeClient([_Resp(200, _gem({"severity": 0}))])
    monkeypatch.setattr(pv.httpx, "Client", lambda **kw: fake)
    pv.judge_sheet(b"jpegbytes", cfg)
    parts = fake.calls[0][1]["contents"][0]["parts"]
    assert len(parts) == 1 and "inlineData" in parts[0]


def test_no_key_is_a_no_op_not_a_crash(monkeypatch, cfg):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    out, n = pv.judge_sheet(b"x", cfg)
    assert out is None and n == 0


def test_rate_limited_model_hands_over_to_the_next(monkeypatch, cfg):
    """Quotas are per model, so a 429 must not end the check."""
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    fake = _FakeClient([_Resp(429, text="quota"), _Resp(200, _gem({"severity": 3}))])
    monkeypatch.setattr(pv.httpx, "Client", lambda **kw: fake)
    out, n = pv.judge_sheet(b"x", cfg)
    assert out["severity"] == 3 and n == 2


def test_request_cap_is_respected(monkeypatch, cfg):
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    fake = _FakeClient([_Resp(500), _Resp(500), _Resp(500), _Resp(500)])
    monkeypatch.setattr(pv.httpx, "Client", lambda **kw: fake)
    out, n = pv.judge_sheet(b"x", cfg, request_cap=2)
    assert out is None and n == 2


def test_vision_budget_is_separate_from_the_text_cascade(tmp_path):
    """Different model, different free-tier bucket — so a different counter."""
    from src.parser.cloud_enrichment import BudgetLedger, get_llm_config
    cfg = get_llm_config()
    cfg["providers_cfg"][pv.PROVIDER_KEY] = pv.get_vision_config()
    led = BudgetLedger(tmp_path / "b.json")
    led.charge("gemini", 30)
    assert led.remaining(pv.PROVIDER_KEY, cfg) == pv.get_vision_config()["daily_request_budget"]


# ---------------------------------------------------------------------------
# Sheet building
# ---------------------------------------------------------------------------

def test_sheet_is_one_wide_image(tmp_path):
    from PIL import Image
    paths = []
    for i, size in enumerate([(800, 600), (1000, 500)]):
        p = tmp_path / f"{i}.jpg"
        Image.new("RGB", size, "grey").save(p)
        paths.append(p)
    data = pv.build_sheet(paths, 460)
    assert data
    im = Image.open(__import__("io").BytesIO(data))
    assert im.height == 460 and im.width > 460      # side by side, not stacked


def test_sheet_survives_a_corrupt_download(tmp_path):
    from PIL import Image
    good = tmp_path / "g.jpg"
    Image.new("RGB", (600, 400), "grey").save(good)
    bad = tmp_path / "b.jpg"
    bad.write_bytes(b"not an image")
    assert pv.build_sheet([bad, good], 460) is not None


def test_no_photos_means_no_sheet_and_no_verdict(tmp_path):
    assert pv.build_sheet([], 460) is None
