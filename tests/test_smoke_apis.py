"""Smoke tests for external API health (LIVE network calls)."""

import json
import os

import pytest

pytestmark = pytest.mark.smoke


def _get_key_from_config() -> str:
    try:
        import yaml
        from pathlib import Path
        cfg_path = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"
        if cfg_path.exists():
            with open(cfg_path) as f:
                return yaml.safe_load(f).get("llm", {}).get("openrouter_api_key", "")
    except Exception:
        pass
    return ""


def _get_api_key() -> str:
    return os.environ.get("OPENROUTER_API_KEY") or _get_key_from_config()


_has_api_key = bool(_get_api_key())


@pytest.mark.skipif(not _has_api_key, reason="No OPENROUTER_API_KEY")
class TestOpenRouterSmokeTest:
    """Verify OpenRouter LLM endpoint returns valid JSON for generation queries."""

    def test_llm_returns_json(self):
        import httpx

        resp = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {_get_api_key()}",
                "Content-Type": "application/json",
            },
            json={
                "model": "google/gemma-3-12b-it:free",
                "messages": [{"role": "user", "content":
                    'Return ONLY this JSON array: [{"name":"test","year_from":2000,"year_to":2010}]'}],
                "max_tokens": 100,
                "temperature": 0.0,
            },
            timeout=30,
        )
        if resp.status_code == 401:
            pytest.skip("OpenRouter API key invalid or expired")
        assert resp.status_code == 200, f"OpenRouter returned {resp.status_code}"

        content = resp.json()["choices"][0]["message"].get("content")
        assert content is not None, "LLM returned null content"

        # Should be parseable as JSON
        text = content.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        start, end = text.find("["), text.rfind("]")
        if start >= 0 and end > start:
            text = text[start:end + 1]
        data = json.loads(text)
        assert isinstance(data, list)
        assert len(data) > 0
