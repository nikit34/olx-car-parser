"""Smoke tests for external API health (LIVE network calls)."""

import json
import os
import urllib.parse
import urllib.request

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


class TestDBpediaSmokeTest:
    """Verify DBpedia SPARQL endpoint is reachable and returns automobile data."""

    def test_dbpedia_returns_results(self):
        query = """\
PREFIX dbo: <http://dbpedia.org/ontology/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT DISTINCT ?genLabel ?sy WHERE {
  ?gen a dbo:Automobile .
  ?gen dbo:manufacturer ?mfg .
  ?gen dbo:productionStartYear ?sy .
  ?gen rdfs:label ?genLabel . FILTER(LANG(?genLabel) = "en")
  ?mfg rdfs:label "Volkswagen"@en .
} LIMIT 5"""

        body = urllib.parse.urlencode({"query": query}).encode()
        req = urllib.request.Request(
            "https://dbpedia.org/sparql",
            data=body,
            headers={
                "Accept": "application/sparql-results+json",
                "User-Agent": "olx-car-parser-test/1.0",
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())

        results = data["results"]["bindings"]
        assert len(results) > 0, "DBpedia returned no Volkswagen automobile data"
        assert "genLabel" in results[0]
        assert "sy" in results[0]


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
