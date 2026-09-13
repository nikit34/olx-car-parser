import base64
import datetime as dt
import json

import pytest

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


def test_clicks_summary_separates_dropped_hits_from_live_ones():
    payload = {
        "days": {"2026-09-02": {"ano": 2}},
        "drops": {"2026-09-02": {"prefetch": 9, "bot": 3}, "2026-08-01": {"bot": 99}},
        "hits": [
            {"t": "2026-09-02T10:00:00Z", "drop": None, "net": "Vodafone Portugal"},
            {"t": "2026-09-02T11:00:00Z", "drop": None, "net": "Vodafone Portugal"},
            {"t": "2026-09-02T12:00:00Z", "drop": "prefetch", "net": "Google LLC"},
            {"t": "2026-07-01T12:00:00Z", "drop": None, "net": "MEO"},
        ],
    }
    fetch = fake_fetch({"clicks.json": (200, json.dumps(payload).encode())})
    lines, fresh = md.clicks_summary(fetch, "u", "p", dt.date(2026, 9, 3))
    assert fresh == 2
    assert "отсеяно 12 (prefetch 9, bot 3)" in lines[0]
    assert "99" not in lines[0]
    assert lines[1] == "Откуда шли живые клики: Vodafone Portugal 2"


def test_gsc_failure_says_what_google_actually_answered():
    adc = json.dumps({"client_id": "c", "client_secret": "s", "refresh_token": "r"})

    def refused(url, payload, headers=None, timeout=30):
        return 400, json.dumps({
            "error": "invalid_grant",
            "error_description": "Token has been expired or revoked.",
        }).encode()

    lines, warns = md.gsc_summary(refused, adc, dt.date(2026, 9, 13))
    line = lines[0]
    assert warns, "отказ Search Console не попал в предупреждения"
    assert "invalid_grant" in line and "revoked" in line
    assert "не удалось получить токен" not in line

    def offline(url, payload, headers=None, timeout=30):
        return 0, b"<urlopen error timed out>"

    assert "сеть" in md.gsc_summary(offline, adc, dt.date(2026, 9, 13))[0][0]

    half = json.dumps({"client_id": "c", "refresh_token": "r"})
    line = md.gsc_summary(refused, half, dt.date(2026, 9, 13))[0][0]
    assert "client_secret" in line, "a half-pasted secret must name the missing field"


def test_gsc_signs_a_service_account_assertion_google_would_accept():
    crypto = pytest.importorskip("cryptography", reason="подпись JWT требует cryptography")
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding, rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()
    adc = json.dumps({
        "type": "service_account", "client_email": "bot@proj.iam.gserviceaccount.com",
        "private_key": pem, "token_uri": "https://oauth2.googleapis.com/token",
    })

    seen = {}

    def post(url, payload, headers=None, timeout=30):
        if url.endswith("/token"):
            seen.update(payload)
            return 200, json.dumps({"access_token": "t"}).encode()
        seen["headers"] = headers or {}
        return 200, json.dumps({"rows": []}).encode()

    lines, warns = md.gsc_summary(post, adc, dt.date(2026, 9, 13))
    assert warns == [] and "Search Console 04.09" in lines[0]
    assert seen["grant_type"] == "urn:ietf:params:oauth:grant-type:jwt-bearer"

    head, body, sig = seen["assertion"].split(".")

    def unpad(chunk):
        return json.loads(base64.urlsafe_b64decode(chunk + "=" * (-len(chunk) % 4)))

    assert unpad(head) == {"alg": "RS256", "typ": "JWT"}
    claims = unpad(body)
    assert claims["iss"] == "bot@proj.iam.gserviceaccount.com"
    assert claims["scope"] == md.GSC_SCOPE
    assert claims["exp"] - claims["iat"] == 3600
    key.public_key().verify(
        base64.urlsafe_b64decode(sig + "=" * (-len(sig) % 4)),
        f"{head}.{body}".encode(), padding.PKCS1v15(), hashes.SHA256(),
    )
    assert "x-goog-user-project" not in seen["headers"], \
        "квота-проект от ADC не должна уезжать с сервисным аккаунтом"


def test_gsc_service_account_without_a_key_says_which_field_is_missing():
    adc = json.dumps({"type": "service_account", "client_email": "bot@proj.iam.gserviceaccount.com"})

    def unused(url, payload, headers=None, timeout=30):
        raise AssertionError("до сети дойти не должно")

    lines, warns = md.gsc_summary(unused, adc, dt.date(2026, 9, 13))
    assert "private_key" in lines[0] and warns


def test_gsc_reports_a_rejected_query_separately_from_a_rejected_token():
    adc = json.dumps({"client_id": "c", "client_secret": "s", "refresh_token": "r"})

    def post(url, payload, headers=None, timeout=30):
        if url.endswith("/token"):
            return 200, json.dumps({"access_token": "t"}).encode()
        return 403, json.dumps({"error": {"status": "PERMISSION_DENIED", "message": "User does not have permission"}}).encode()

    line = md.gsc_summary(post, adc, dt.date(2026, 9, 13))[0][0]
    assert "запрос отклонён" in line and "PERMISSION_DENIED" in line


def test_ai_summary_counts_the_week_and_names_the_cited_pages():
    payload = {
        "days": {
            "2026-09-02": {"chatgpt-user": 4, "gptbot": 11},
            "2026-09-01": {"chatgpt-user": 1},
            "2026-07-01": {"chatgpt-user": 999},
        },
        "hits": [
            {"t": "2026-09-02T09:00:00Z", "agent": "chatgpt-user", "path": "/preco/citroen-c3/2018"},
            {"t": "2026-09-02T10:00:00Z", "agent": "chatgpt-user", "path": "/preco/citroen-c3/2018"},
            {"t": "2026-09-01T10:00:00Z", "agent": "chatgpt-user", "path": "/avaliar"},
            {"t": "2026-07-01T10:00:00Z", "agent": "chatgpt-user", "path": "/mercado"},
        ],
    }
    fetch = fake_fetch({"ai.json": (200, json.dumps(payload).encode())})
    lines = md.ai_summary(fetch, "u", "p", dt.date(2026, 9, 3))
    assert "за 7 дней: 16" in lines[0]
    assert "chatgpt-user 5" in lines[0] and "gptbot 11" in lines[0]
    assert "999" not in lines[0]
    assert lines[1].startswith("В живых ответах (3):")
    assert "/preco/citroen-c3/2018 2" in lines[1] and "/mercado" not in lines[1]


def test_ai_summary_says_so_when_no_agent_came():
    fetch = fake_fetch({"ai.json": (200, json.dumps({"days": {}, "hits": []}).encode())})
    lines = md.ai_summary(fetch, "u", "p", dt.date(2026, 9, 3))
    assert len(lines) == 1 and "ни одного захода" in lines[0]


def test_clicks_summary_stays_quiet_when_nothing_was_dropped():
    payload = {"days": {"2026-09-02": {"ano": 1}}, "drops": {}, "hits": []}
    fetch = fake_fetch({"clicks.json": (200, json.dumps(payload).encode())})
    lines, _ = md.clicks_summary(fetch, "u", "p", dt.date(2026, 9, 3))
    assert len(lines) == 1 and "отсеяно" not in lines[0]
