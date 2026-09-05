import datetime as dt
import json

from scripts import monitor_digest as md


NOW = dt.datetime(2026, 9, 3, 8, 0, tzinfo=dt.timezone.utc)


def fake_fetch(responses):
    def fetch(url, headers=None, timeout=30):
        for key, value in responses.items():
            if key in url:
                return value
        return 404, b""
    return fetch


def test_site_check_flags_bad_status_and_thin_sitemap():
    fetch = fake_fetch({
        "/sitemap.xml": (200, b"<loc>a</loc><loc>b</loc>"),
        "/vender": (500, b""),
        "carsbuyer.org": (200, b"ok"),
    })
    lines, warnings = md.check_site(fetch)
    assert "sitemap 2" in lines[0]
    assert any("sitemap" in w for w in warnings)
    assert any("/vender" in w for w in warnings)


def test_site_check_is_quiet_when_healthy():
    fetch = fake_fetch({
        "/sitemap.xml": (200, b"<loc>x</loc>" * 1200),
        "carsbuyer.org": (200, b"ok"),
    })
    _, warnings = md.check_site(fetch)
    assert warnings == []


def test_release_age_warns_when_stale():
    payload = {"assets": [
        {"name": "models.json", "updated_at": "2026-09-02T20:00:00Z"},
        {"name": "hot_deals_all.json", "updated_at": "2026-09-02T22:30:00Z"},
    ]}
    fetch = fake_fetch({"releases/tags/latest-data": (200, json.dumps(payload).encode())})
    lines, warnings = md.check_release(fetch, None, NOW)
    assert "9.5 ч" in lines[0]
    assert warnings


def test_leads_summary_counts_fresh_only():
    leads = {"leads": [
        {"ts": "2026-09-03T07:30:00Z", "name": "Renault Clio", "ano": 2014, "distrito": "Braga"},
        {"ts": "2026-08-30T07:30:00Z", "name": "Opel Corsa", "ano": 2016, "distrito": "Porto"},
    ]}
    fetch = fake_fetch({"leads.json": (200, json.dumps(leads).encode())})
    lines, warnings, fresh = md.leads_summary(fetch, "u", "p", NOW)
    assert fresh == 1
    assert "всего 2" in lines[0] and "Renault Clio 2014 Braga" in lines[0]
    assert warnings == []


def test_leads_summary_without_credentials_does_not_fetch():
    lines, warnings, fresh = md.leads_summary(fake_fetch({}), None, None, NOW)
    assert fresh == 0 and warnings == [] and "нет доступа" in lines[0]


def test_watched_senders_and_forwarded_mailbox():
    assert md.is_watched("Flexicar Porto <porto@flexicar.pt>", "permikov134@yandex.ru")
    assert md.is_watched("Someone <x@example.com>", "Ola <ola@carsbuyer.org>")
    assert not md.is_watched("LinkedIn <news@linkedin.com>", "permikov134@yandex.ru")


def test_gsc_page_buckets():
    rows = [
        {"keys": ["https://carsbuyer.org/preco/opel-corsa/2016"], "impressions": 100, "clicks": 3},
        {"keys": ["https://carsbuyer.org/preco/opel-corsa"], "impressions": 50, "clicks": 0},
        {"keys": ["https://carsbuyer.org/vender/opel-corsa"], "impressions": 20, "clicks": 1},
    ]
    total, year, vender = md.summarise_pages(rows)
    assert total["impr"] == 170 and year["impr"] == 100 and vender["clicks"] == 1
    assert round(md.ctr(year), 1) == 3.0


def test_digest_puts_warnings_first_and_stays_under_telegram_limit():
    text = md.build_digest(NOW, [["Сайт: ok"], ["x" * 5000]], ["sitemap мал"])
    assert text.startswith("⚠️")
    assert "sitemap мал" in text.splitlines()[1]
    assert len(text) <= 4000


def test_press_reminder_only_inside_its_window():
    assert md.press_reminder(dt.date(2026, 9, 27)) == []
    assert md.press_reminder(dt.date(2026, 9, 28))
    assert md.press_reminder(dt.date(2026, 10, 4))
    assert md.press_reminder(dt.date(2026, 10, 5)) == []


def test_clicks_summary_reports_yesterday_and_the_week():
    days = {"days": {"2026-09-02": {"ano": 3, "avaliar": 1}, "2026-08-20": {"ano": 9}, "2026-09-03": {"ano": 1}}}
    fetch = fake_fetch({"clicks.json": (200, json.dumps(days).encode())})
    lines, fresh = md.clicks_summary(fetch, "u", "p", dt.date(2026, 9, 3))
    assert fresh == 4
    assert "вчера 4 (ano 3, avaliar 1)" in lines[0] and "за 7 дней 4" in lines[0]
    lines, fresh = md.clicks_summary(fetch, "u", "p", dt.date(2026, 9, 10))
    assert fresh == 0 and "вчера 0" in lines[0] and "за 7 дней 1" in lines[0]
