"""Routing olx.pt requests through the Worker relay.

The scrape host's address is blocked by OLX: every direct request answers
403 with a CloudFront page. The liveness probe read that as "unknown" and
deferred, so no OLX listing was marked sold for five days while 6985 dead
ones stayed in the active corpus — and nothing complained.
"""
from __future__ import annotations

import pytest

from src.parser.relay import RELAY_PATH_PREFIXES, relay_rewrite


@pytest.fixture(autouse=True)
def _relay_env(monkeypatch):
    monkeypatch.setenv("OLX_RELAY_URL", "https://carsbuyer.org/_olx")
    monkeypatch.setenv("OLX_RELAY_TOKEN", "secret")


class TestRouting:
    def test_listing_page_is_relayed(self):
        url, headers = relay_rewrite(
            "https://www.olx.pt/d/anuncio/golf-IDJxY4M.html", "UA/1.0")
        assert url.startswith("https://carsbuyer.org/_olx?path=")
        assert "%2Fd%2Fanuncio%2F" in url
        assert headers["X-Relay-Token"] == "secret"
        assert headers["X-Relay-UA"] == "UA/1.0"

    def test_offers_api_is_relayed_with_its_query(self):
        url, _ = relay_rewrite("https://www.olx.pt/api/v1/offers/?limit=40")
        assert "%3Flimit%3D40" in url

    def test_standvirtual_goes_direct(self):
        """The Worker refuses it, so relaying would earn the same 403 by a
        longer route — and StandVirtual answers us fine."""
        target = "https://www.standvirtual.com/carros/anuncio/x.html"
        assert relay_rewrite(target) == (target, {})

    def test_an_olx_path_the_worker_refuses_goes_direct(self):
        target = "https://www.olx.pt/some/other/page"
        assert relay_rewrite(target) == (target, {})

    def test_prefixes_match_the_worker(self):
        """These mirror RELAY_PATH_PREFIXES in flipper-club/src/index.js;
        drift means requests the Worker rejects."""
        assert RELAY_PATH_PREFIXES == ("/api/v1/offers", "/d/anuncio/")


class TestConfiguration:
    def test_no_relay_configured_passes_through(self, monkeypatch):
        monkeypatch.delenv("OLX_RELAY_URL", raising=False)
        monkeypatch.delenv("OLX_RELAY_TOKEN", raising=False)
        target = "https://www.olx.pt/d/anuncio/x.html"
        assert relay_rewrite(target) == (target, {})

    def test_url_without_token_is_not_used(self, monkeypatch):
        """A tokenless relay answers 403; going direct at least has a chance."""
        monkeypatch.delenv("OLX_RELAY_TOKEN", raising=False)
        target = "https://www.olx.pt/d/anuncio/x.html"
        assert relay_rewrite(target) == (target, {})


class TestProbeUsesRelay:
    def test_probe_requests_the_relay_url(self, monkeypatch):
        from src.storage import repository

        seen = {}

        class _Resp:
            status_code = 404
            text = ""

        def _get(url, timeout=None, headers=None):
            seen["url"] = url
            seen["headers"] = headers
            return _Resp()

        monkeypatch.setattr(repository._PROBE_CLIENT, "get", _get)
        assert repository._verify_listing_alive(
            "https://www.olx.pt/d/anuncio/golf-IDJxY4M.html") is False
        assert seen["url"].startswith("https://carsbuyer.org/_olx?path=")
        assert seen["headers"]["X-Relay-Token"] == "secret"


class TestProbePacing:
    """A bounded pool caps requests in flight, not their rate: eight threads
    fire together, drain, fire together. Routing that burst through the relay
    would get the Worker's addresses blocked instead of this host's."""

    def test_probes_beyond_the_burst_are_spaced(self, monkeypatch):
        """The bucket starts full, so the first few are free by design; what
        matters is that the ones after it are paced rather than fired."""
        import time

        from src.storage import repository

        class _Resp:
            status_code = 404
            text = ""

        monkeypatch.setattr(repository._PROBE_CLIENT, "get",
                            lambda *a, **k: _Resp())
        monkeypatch.setattr("src.parser.scraper._OLX_MAX_RPS", 5.0)
        repository._PROBE_LIMITER = None

        url = "https://www.olx.pt/d/anuncio/x.html"
        for _ in range(5):
            repository._verify_listing_alive(url)

        start = time.monotonic()
        for _ in range(4):
            repository._verify_listing_alive(url)
        elapsed = time.monotonic() - start
        repository._PROBE_LIMITER = None

        assert elapsed >= 0.5, f"four probes past the burst took {elapsed:.2f}s"
