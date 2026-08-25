"""A blocked scraper must not look like an empty market.

``build_hot_deals`` drops a deal whose OLX page carries no usable photo — the
listing is 410/redirected and its link would be dead anyway. That rule is right
only when the page was actually fetched. On 2026-08-25 OLX started 403ing our
address, every photo fetch raised, and all 21 BUY/WATCH deals were discarded as
"dead": the feed shipped 72-byte files and the site showed "Sem negócios" while
the decision engine was working perfectly.

So the fetch now reports three outcomes and these tests pin them apart.
"""

import importlib.util
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest

_spec = importlib.util.spec_from_file_location(
    "build_hot_deals", Path(__file__).resolve().parent.parent / "scripts" / "build_hot_deals.py")
bhd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bhd)


@pytest.fixture(autouse=True)
def _clear_cache():
    bhd._PHOTO_CACHE.clear()
    yield
    bhd._PHOTO_CACHE.clear()


def _raise(exc):
    def _fake(*a, **kw):
        raise exc
    return _fake


class TestFetchPhotoUrls:
    @pytest.mark.parametrize("code", [404, 410])
    def test_gone_listing_returns_empty_so_the_deal_is_dropped(self, code, monkeypatch):
        monkeypatch.setattr(bhd, "urlopen", _raise(
            HTTPError("u", code, "gone", {}, None)))
        assert bhd.fetch_photo_urls("https://www.olx.pt/d/anuncio/x-IDa.html") == []

    @pytest.mark.parametrize("code", [403, 429, 500, 503])
    def test_block_returns_none_so_the_deal_survives(self, code, monkeypatch):
        """The distinction that matters: 403 says something about us, not
        about the listing."""
        monkeypatch.setattr(bhd, "urlopen", _raise(
            HTTPError("u", code, "blocked", {}, None)))
        assert bhd.fetch_photo_urls("https://www.olx.pt/d/anuncio/x-IDb.html") is None

    def test_network_error_returns_none(self, monkeypatch):
        monkeypatch.setattr(bhd, "urlopen", _raise(URLError("dns")))
        assert bhd.fetch_photo_urls("https://www.olx.pt/d/anuncio/x-IDc.html") is None

    def test_photos_parsed_in_page_order(self, monkeypatch):
        html = (
            '<img src="https://ireland.apollo.olxcdn.com:443/v1/files/aaa-PT/image;s=1000x700">'
            '<img src="https://ireland.apollo.olxcdn.com:443/v1/files/bbb-PT/image;s=1000x700">'
            # A related-listing thumbnail: no >=1000px variant, must be filtered.
            '<img src="https://ireland.apollo.olxcdn.com:443/v1/files/ccc-PT/image;s=200x150">'
        )

        class _Resp:
            def read(self): return html.encode()
            def __enter__(self): return self
            def __exit__(self, *a): return False

        monkeypatch.setattr(bhd, "urlopen", lambda *a, **kw: _Resp())
        got = bhd.fetch_photo_urls("https://www.olx.pt/d/anuncio/x-IDd.html")
        assert [u.split("/files/")[1].split("-PT")[0] for u in got] == ["aaa", "bbb"]

    def test_result_is_cached_per_url(self, monkeypatch):
        calls = []
        monkeypatch.setattr(bhd, "urlopen", _raise(HTTPError("u", 403, "b", {}, None)))
        url = "https://www.olx.pt/d/anuncio/x-IDe.html"
        assert bhd.fetch_photo_urls(url) is None
        monkeypatch.setattr(bhd, "urlopen", lambda *a, **kw: calls.append(1))
        assert bhd.fetch_photo_urls(url) is None
        assert calls == [], "second call must come from the cache"
