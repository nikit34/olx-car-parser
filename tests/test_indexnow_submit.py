"""Tests for the IndexNow submitter (scripts/indexnow_submit.py)."""
from __future__ import annotations

import pytest

from scripts import indexnow_submit as ins


SITEMAP = [
    "https://carsbuyer.org/pt/preco/audi-a3",
    "https://carsbuyer.org/pt/preco/audi-a3/2012",
    "https://carsbuyer.org/pt/preco/opel-corsa/2006",
    "https://carsbuyer.org/pt/preco/opel-corsa/diesel",
    "https://carsbuyer.org/pt/depreciacao/audi-a3",
    "https://carsbuyer.org/pt/guias/vender-carro-com-credito",
]

YEAR_PAGES = r"/preco/[^/]+/\d{4}$"


@pytest.fixture
def wired(monkeypatch):
    sent: list[list[str]] = []
    monkeypatch.setattr(ins, "verify_key", lambda host, key: True)
    monkeypatch.setattr(ins, "sitemap_urls", lambda host, locale: list(SITEMAP))
    monkeypatch.setattr(ins, "submit",
                        lambda host, key, urls: (sent.append(list(urls)), (200, "ok"))[1])
    return sent


def _main(argv):
    import sys
    from unittest.mock import patch
    with patch.object(sys, "argv", ["indexnow_submit.py", *argv]):
        return ins.main()


class TestMatchFilter:
    def test_only_the_year_pages_travel(self, wired):
        assert _main(["--locales", "pt", "--match", YEAR_PAGES]) == 0
        assert wired == [[
            "https://carsbuyer.org/pt/preco/audi-a3/2012",
            "https://carsbuyer.org/pt/preco/opel-corsa/2006",
        ]]

    def test_without_a_filter_the_whole_sitemap_travels(self, wired):
        assert _main(["--locales", "pt"]) == 0
        assert wired == [SITEMAP]

    def test_a_filter_that_matches_nothing_submits_nothing(self, wired):
        assert _main(["--locales", "pt", "--match", "/nao-existe/"]) == 1
        assert wired == []

    def test_a_broken_regex_is_refused_before_any_request(self, wired):
        assert _main(["--locales", "pt", "--match", "[unclosed"]) == 1
        assert wired == []

    def test_a_dry_run_sends_nothing(self, wired):
        assert _main(["--locales", "pt", "--match", YEAR_PAGES, "--dry-run"]) == 0
        assert wired == []
