"""The TLS handshake every OLX-facing client must use.

OLX's CDN blocks httpx's stock ClientHello fingerprint. On 2026-08-10 that
silently killed the scrape — every request 403'd before reaching OLX, the
scrape step failed on every cron for thirteen days, and because the rest of
the workflow runs regardless, the dashboard kept republishing the same stale
snapshot with a fresh build timestamp.

These tests pin the two context flags that move us off the blocked
fingerprint, and pin that all three olx.pt clients actually use them. They
are offline: the live check lives behind the ``smoke`` marker elsewhere.
"""

import ssl

import pytest

from src.parser.tls_fingerprint import build_ssl_context


def _pool_ssl_context(client):
    """The context httpx will actually hand to the socket."""
    return client._transport._pool._ssl_context


class TestBuildSslContext:
    def test_disables_session_tickets(self):
        assert build_ssl_context().options & ssl.OP_NO_TICKET

    def test_enables_post_handshake_auth(self):
        assert build_ssl_context().post_handshake_auth is True

    def test_differs_from_stock_context(self):
        """The whole point: not the fingerprint OLX bans."""
        stock = ssl.create_default_context()
        ours = build_ssl_context()
        assert (stock.options & ssl.OP_NO_TICKET, stock.post_handshake_auth) != (
            ours.options & ssl.OP_NO_TICKET, ours.post_handshake_auth
        )

    def test_still_verifies_certificates(self):
        ctx = build_ssl_context()
        assert ctx.verify_mode == ssl.CERT_REQUIRED
        assert ctx.check_hostname is True


class TestClientsUseIt:
    @pytest.mark.parametrize("name", ["scraper", "probe", "photo"])
    def test_olx_facing_client_carries_the_fingerprint(self, name):
        if name == "scraper":
            from src.parser.scraper import OlxScraper, ScraperConfig
            scraper = OlxScraper(ScraperConfig())
            try:
                ctx = _pool_ssl_context(scraper.client)
            finally:
                scraper.close()
        elif name == "probe":
            from src.storage.repository import _PROBE_CLIENT
            ctx = _pool_ssl_context(_PROBE_CLIENT)
        else:
            from src.parser.photo_fetch import _CLIENT
            ctx = _pool_ssl_context(_CLIENT)
        assert ctx.options & ssl.OP_NO_TICKET
        assert ctx.post_handshake_auth is True

    def test_scraper_never_advertises_httpx_as_user_agent(self):
        """``python-httpx/x.y`` is 403'd on its own, independently of TLS."""
        from src.parser.scraper import OlxScraper, ScraperConfig
        scraper = OlxScraper(ScraperConfig())
        try:
            assert "httpx" not in scraper.client.headers["user-agent"].lower()
            assert "httpx" not in scraper._random_headers()["User-Agent"].lower()
        finally:
            scraper.close()


@pytest.mark.smoke
def test_live_olx_api_accepts_our_handshake():
    """Live check against the real CDN — the one that would have caught 2026-08-10.

    Run with ``pytest -m smoke`` from the scrape host; CI skips it.
    """
    from src.parser.scraper import OlxScraper, ScraperConfig, OLX_API_URL, CARS_CATEGORY_ID
    scraper = OlxScraper(ScraperConfig())
    try:
        payload = scraper._fetch_json(
            f"{OLX_API_URL}?offset=0&limit=1&category_id={CARS_CATEGORY_ID}"
        )
    finally:
        scraper.close()
    assert payload is not None, "OLX refused the handshake — CDN rules moved again"
    assert payload.get("data"), "OLX answered but returned no offers"
