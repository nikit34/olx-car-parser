"""``fetch_photos`` must route through the Worker relay, and
``photo_gallery_reachable`` must be able to say "we could not look" apart
from "there is nothing there".

Both matter for the rating's first-photo gate. OLX 403s the scrape host's
address outright, so without the relay every gallery reads as empty — and an
empty gallery is what ``_blocking_deal_reason`` treats as "proven no car
photo". The relay turns that from a silent mass-false-positive into a real
fetch; the reachability probe covers the case where even the relay is down,
so a block can never be persisted as a verdict about the car.

No network here: the module-level ``_CLIENT`` is stubbed.
"""

from __future__ import annotations

import httpx

from src.parser import photo_fetch as pf


class _Resp:
    def __init__(self, status_code: int, text: str = ""):
        self.status_code = status_code
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("boom", request=None, response=None)


class _FakeClient:
    def __init__(self, resp: _Resp):
        self.resp = resp
        self.calls: list[tuple[str, dict | None]] = []

    def get(self, url, headers=None, **_kw):
        self.calls.append((url, headers))
        return self.resp


_GALLERY_HTML = (
    '<img src="https://ireland.apollo.olxcdn.com:443/v1/files/aaa-PT/image;s=1000x700">'
    '<img src="https://ireland.apollo.olxcdn.com:443/v1/files/bbb-PT/image;s=1000x700">'
    # related-listing thumbnail: no >=1000px variant, must be filtered out
    '<img src="https://ireland.apollo.olxcdn.com:443/v1/files/ccc-PT/image;s=200x150">'
)
_URL = "https://www.olx.pt/d/anuncio/vw-golf-IDabc123.html"


class TestRelayRouting:
    def test_olx_gallery_fetch_goes_through_the_relay(self, monkeypatch):
        monkeypatch.setattr(pf, "relay_rewrite",
                            lambda url, ua=None, **kw: ("https://relay.test/?path=x",
                                                        {"X-Relay-Token": "tok"}))
        fake = _FakeClient(_Resp(200, _GALLERY_HTML))
        monkeypatch.setattr(pf, "_CLIENT", fake)

        photos = pf.fetch_photos(_URL)
        assert len(photos) == 2
        assert fake.calls[0][0] == "https://relay.test/?path=x"
        assert fake.calls[0][1] == {"X-Relay-Token": "tok"}

    def test_without_relay_configured_it_fetches_directly(self, monkeypatch):
        """No token in the environment must not break local runs — the relay
        rewrite falls through to the original URL."""
        monkeypatch.setattr(pf, "relay_rewrite", lambda url, ua=None, **kw: (url, {}))
        fake = _FakeClient(_Resp(200, _GALLERY_HTML))
        monkeypatch.setattr(pf, "_CLIENT", fake)

        assert len(pf.fetch_photos(_URL)) == 2
        assert fake.calls[0][0] == _URL

    def test_standvirtual_is_not_rewritten(self, monkeypatch):
        """The Worker only serves OLX prefixes; routing SV through it would
        earn the same 403 by a longer route."""
        calls = []

        def _rewrite(url, ua=None, **kw):
            calls.append(url)
            return url, {}

        monkeypatch.setattr(pf, "relay_rewrite", _rewrite)
        monkeypatch.setattr(pf, "fetch_standvirtual_advert",
                            lambda url: {"images": {"photos": [{"url": "https://cdn/a.jpg"}]}})
        photos = pf.fetch_photos_standvirtual("https://www.standvirtual.com/carros/x.html")
        assert photos == ["https://cdn/a.jpg"]
        assert calls == []


class TestGalleryReachability:
    def test_blocked_page_reports_false(self, monkeypatch):
        """403 is the block signature — the exact shape that must not be
        persisted as an empty gallery."""
        monkeypatch.setattr(pf, "relay_rewrite", lambda url, ua=None, **kw: (url, {}))
        monkeypatch.setattr(pf, "_CLIENT", _FakeClient(_Resp(403)))
        assert pf.photo_gallery_reachable(_URL) is False

    def test_readable_page_reports_true(self, monkeypatch):
        monkeypatch.setattr(pf, "relay_rewrite", lambda url, ua=None, **kw: (url, {}))
        monkeypatch.setattr(pf, "_CLIENT", _FakeClient(_Resp(200, _GALLERY_HTML)))
        assert pf.photo_gallery_reachable(_URL) is True

    def test_dead_listing_reports_false(self, monkeypatch):
        """404/410 is a definitive answer, same as a block: the page cannot be
        read, so the caller writes nothing rather than guessing."""
        monkeypatch.setattr(pf, "relay_rewrite", lambda url, ua=None, **kw: (url, {}))
        monkeypatch.setattr(pf, "_CLIENT", _FakeClient(_Resp(410)))
        assert pf.photo_gallery_reachable(_URL) is False

    def test_connection_error_reports_false(self, monkeypatch):
        class _Boom:
            def get(self, *_a, **_kw):
                raise httpx.ConnectError("dns")

        monkeypatch.setattr(pf, "relay_rewrite", lambda url, ua=None, **kw: (url, {}))
        monkeypatch.setattr(pf, "_CLIENT", _Boom())
        assert pf.photo_gallery_reachable(_URL) is False

    def test_standvirtual_needs_no_probe(self, monkeypatch):
        """SV's gallery comes from a __NEXT_DATA__ fetch that reports its own
        failures, so a separate probe would just double the requests."""
        monkeypatch.setattr(pf, "relay_rewrite",
                            lambda *a, **kw: (_ for _ in ()).throw(AssertionError("no probe")))
        assert pf.photo_gallery_reachable("https://www.standvirtual.com/carros/x.html") is None


class TestEmptyGalleryIsNotAnError:
    def test_readable_page_with_no_photos_yields_empty_list(self, monkeypatch):
        """The genuine case: page answered, gallery empty. Still [] — the
        caller separates it via photo_gallery_reachable."""
        monkeypatch.setattr(pf, "relay_rewrite", lambda url, ua=None, **kw: (url, {}))
        monkeypatch.setattr(pf, "_CLIENT", _FakeClient(_Resp(200, "<html>nothing</html>")))
        assert pf.fetch_photos(_URL) == []
        assert pf.photo_gallery_reachable(_URL) is True