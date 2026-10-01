import argparse
import base64
import datetime as dt
import email
import email.header
import email.utils
import imaplib
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

SITE = "https://carsbuyer.org"
PAGES = ["/pt", "/pt/avaliar", "/pt/vender", "/pt/preco/opel-corsa/2016", "/pt/mercado", "/pt/sitemap.xml"]
SITEMAP_FLOOR = 1000
BLOB_MAX_AGE_H = 8.0
RELEASE_API = "https://api.github.com/repos/nikit34/olx-car-parser/releases/tags/latest-data"
GSC_SITE = "sc-domain:carsbuyer.org"
GSC_PROJECT = "valiant-striker-417414"
WATCH_SENDERS = (
    "venderomeucarro.pt", "flexicar.pt", "autohub.pt", "idrivemobile.pt", "standcapelo.com",
    "caautomoveis.com", "auto1.com", "via-everflow.io", "everflow", "carvertical",
    "danielautosite@gmail.com", "serieoriginalvendas@gmail.com", "notify.cloudflare.com",
    "dekra", "verificar.pt",
)
YEAR_PAGE = re.compile(r"^https://carsbuyer\.org/pt/preco/[^/]+/\d{4}$")


UA = "Mozilla/5.0 (compatible; carsbuyer-monitor/1.0; +https://carsbuyer.org)"


def http_get(url, headers=None, timeout=30):
    hdrs = {"User-Agent": UA}
    hdrs.update(headers or {})
    req = urllib.request.Request(url, headers=hdrs)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except Exception as e:
        return 0, str(e).encode()


def http_post_json(url, payload, headers=None, timeout=30):
    body = json.dumps(payload).encode()
    hdrs = {"Content-Type": "application/json", "User-Agent": UA}
    hdrs.update(headers or {})
    req = urllib.request.Request(url, data=body, headers=hdrs, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except Exception as e:
        return 0, str(e).encode()


def check_site(fetch):
    lines, warnings = [], []
    parts = []
    for path in PAGES:
        status, body = fetch(SITE + path)
        if path == "/pt/sitemap.xml":
            n = body.count(b"<loc>") if status == 200 else 0
            parts.append(f"sitemap {n}")
            if status != 200 or n < SITEMAP_FLOOR:
                warnings.append(f"sitemap: {status}, {n} URL (мин. {SITEMAP_FLOOR})")
            continue
        parts.append(f"{path} {status}")
        if status != 200:
            warnings.append(f"{path} отвечает {status}")
    lines.append("Сайт: " + " · ".join(parts))
    return lines, warnings


def release_age_hours(fetch, token, now):
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    status, body = fetch(RELEASE_API, headers)
    if status != 200:
        return None
    try:
        assets = json.loads(body).get("assets", [])
    except Exception:
        return None
    stamps = [a.get("updated_at") for a in assets if a.get("name") in ("hot_deals_all.json", "models.json")]
    stamps = [s for s in stamps if s]
    if not stamps:
        return None
    latest = max(dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc) for s in stamps)
    return (now - latest).total_seconds() / 3600


def check_release(fetch, token, now):
    age = release_age_hours(fetch, token, now)
    if age is None:
        return ["Данные: релиз недоступен"], ["не удалось прочитать релиз latest-data"]
    line = f"Данные: обновлены {age:.1f} ч назад"
    warnings = [f"данные не обновлялись {age:.1f} ч"] if age > BLOB_MAX_AGE_H else []
    return [line], warnings


def parse_ts(value):
    try:
        return dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


def leads_summary(fetch, user, password, now, window_h=24):
    if not user or not password:
        return ["Заявки: нет доступа (ANALYTICS_USER/PASS)"], [], 0
    auth = base64.b64encode(f"{user}:{password}".encode()).decode()
    status, body = fetch(SITE + "/analytics/leads.json", {"Authorization": f"Basic {auth}"})
    if status != 200:
        return [f"Заявки: leads.json отвечает {status}"], [f"leads.json {status}"], 0
    try:
        leads = json.loads(body).get("leads", [])
    except Exception:
        return ["Заявки: leads.json не читается"], ["leads.json не JSON"], 0
    fresh = []
    for lead in leads:
        ts = parse_ts(lead.get("ts"))
        if ts and (now - ts).total_seconds() <= window_h * 3600:
            fresh.append(lead)
    desc = [
        " ".join(str(x) for x in (lead.get("name") or lead.get("modelo") or "?", lead.get("ano") or "", lead.get("distrito") or "") if x)
        for lead in fresh
    ]
    line = f"Заявки: всего {len(leads)}, за {window_h} ч {len(fresh)}"
    if desc:
        line += ": " + "; ".join(desc[:5])
    return [line], [], len(fresh)


def admin_json(fetch, user, password, name):
    auth = base64.b64encode(f"{user}:{password}".encode()).decode()
    status, body = fetch(f"{SITE}/analytics/{name}", {"Authorization": f"Basic {auth}"})
    if status != 200:
        return None, f"{name} отвечает {status}"
    try:
        return json.loads(body), None
    except Exception:
        return None, f"{name} не читается"


def clicks_summary(fetch, user, password, today):
    if not user or not password:
        return ["Клики на историю: нет доступа"], 0
    data, err = admin_json(fetch, user, password, "clicks.json")
    if err:
        return [f"Клики на историю: {err}"], 0
    days = data.get("days", {})
    drops = data.get("drops", {})
    hits = data.get("hits", [])
    yesterday = (today - dt.timedelta(days=1)).isoformat()
    week = {(today - dt.timedelta(days=i)).isoformat() for i in range(1, 8)}
    y = days.get(yesterday, {})
    y_total = sum(y.values())
    w_total = sum(sum(v.values()) for d, v in days.items() if d in week)
    detail = ", ".join(f"{k} {v}" for k, v in sorted(y.items(), key=lambda kv: -kv[1]) if v)
    line = f"Клики на историю: вчера {y_total}" + (f" ({detail})" if detail else "") + f", за 7 дней {w_total}"
    dropped = tally(
        (k, n) for d, v in drops.items() if d in week for k, n in v.items()
    )
    if dropped:
        line += f"; отсеяно {sum(dropped.values())} ({fmt_tally(dropped)})"
    lines = [line]
    nets = tally(
        (h.get("net") or h.get("country") or "сеть неизвестна", 1)
        for h in hits
        if not h.get("drop") and (h.get("t") or "")[:10] in week
    )
    if nets:
        lines.append(f"Откуда шли живые клики: {fmt_tally(nets, top=3)}")
    return lines, y_total


def sell_summary(fetch, user, password, today):
    if not user or not password:
        return ["Хотят продать стенду: нет доступа"], 0
    data, err = admin_json(fetch, user, password, "sell.json")
    if err:
        return [f"Хотят продать стенду: {err}"], 0
    days = data.get("days", {})
    drops = data.get("drops", {})
    hits = data.get("hits", [])
    yesterday = (today - dt.timedelta(days=1)).isoformat()
    week = {(today - dt.timedelta(days=i)).isoformat() for i in range(1, 8)}
    y = days.get(yesterday, {})
    y_total = sum(y.values())
    w_total = sum(sum(v.values()) for d, v in days.items() if d in week)
    total = sum(sum(v.values()) for v in days.values())
    detail = ", ".join(f"{k} {v}" for k, v in sorted(y.items(), key=lambda kv: -kv[1]) if v)
    line = f"Хотят продать стенду: вчера {y_total}" + (f" ({detail})" if detail else "") + f", за 7 дней {w_total}, всего {total}"
    dropped = tally((k, n) for d, v in drops.items() if d in week for k, n in v.items())
    if dropped:
        line += f"; отсеяно {sum(dropped.values())} ({fmt_tally(dropped)})"
    lines = [line]
    live = [h for h in hits if not h.get("drop")]
    if live:
        browsers = len({(h.get("net"), h.get("ua")) for h in live})
        cars = tally((f"{h['m']}/{h['y']}" if h.get("y") else h["m"], 1) for h in live if h.get("m"))
        places = tally((h.get("region") or h.get("country") or "?", 1) for h in live)
        parts = [f"≈{browsers} разных браузеров"]
        if cars:
            parts.append(f"машины: {fmt_tally(cars, top=3)}")
        parts.append(f"откуда: {fmt_tally(places, top=3)}")
        lines.append(f"Живые клики за 30 дней ({len(live)}): " + "; ".join(parts))
    return lines, y_total


def ai_summary(fetch, user, password, today):
    if not user or not password:
        return ["ИИ-цитирование: нет доступа"]
    data, err = admin_json(fetch, user, password, "ai.json")
    if err:
        return [f"ИИ-цитирование: {err}"]
    days = data.get("days", {})
    hits = data.get("hits", [])
    week = {(today - dt.timedelta(days=i)).isoformat() for i in range(1, 8)}
    by_agent = tally((a, n) for d, v in days.items() if d in week for a, n in v.items())
    asked = [h for h in hits if (h.get("t") or "")[:10] in week]
    total = sum(by_agent.values())
    if not total:
        return ["ИИ-цитирование: за 7 дней ни одного захода ИИ-агента"]
    lines = [f"ИИ-цитирование за 7 дней: {total} ({fmt_tally(by_agent, top=4)})"]
    if asked:
        pages = fmt_tally(tally((h.get("path") or "?", 1) for h in asked), top=3)
        lines.append(f"В живых ответах ({len(asked)}): {pages}")
    return lines


def tally(pairs):
    out = {}
    for key, n in pairs:
        out[key] = out.get(key, 0) + n
    return out


def fmt_tally(counts, top=None):
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    if top:
        items = items[:top]
    return ", ".join(f"{k} {v}" for k, v in items)


def decode_header(value):
    if not value:
        return ""
    out = []
    for chunk, charset in email.header.decode_header(value):
        if isinstance(chunk, bytes):
            out.append(chunk.decode(charset or "utf-8", errors="replace"))
        else:
            out.append(chunk)
    return "".join(out)


def is_watched(sender, to):
    s = (sender or "").lower()
    t = (to or "").lower()
    return "ola@carsbuyer.org" in t or any(w in s for w in WATCH_SENDERS)


def mail_summary(imap_factory, user, password, since):
    if not user or not password:
        return ["Почта: нет доступа (MAIL_IMAP_USER/PASSWORD)"], 0
    try:
        box = imap_factory()
        box.login(user, password)
        box.select("INBOX", readonly=True)
        _, data = box.search(None, "SINCE", since.strftime("%d-%b-%Y"))
        ids = data[0].split() if data and data[0] else []
        found = []
        for mid in ids[-60:]:
            _, parts = box.fetch(mid, "(BODY.PEEK[HEADER.FIELDS (FROM TO SUBJECT DATE)])")
            raw = b""
            for part in parts:
                if isinstance(part, tuple):
                    raw += part[1]
            msg = email.message_from_bytes(raw)
            sender = decode_header(msg.get("From"))
            to = decode_header(msg.get("To"))
            if is_watched(sender, to):
                found.append(f"{sender} — {decode_header(msg.get('Subject'))}")
        box.logout()
    except Exception as e:
        return [f"Почта: ошибка IMAP ({type(e).__name__})"], 0
    if not found:
        return ["Почта: ответов от стендов и партнёров нет"], 0
    return ["Почта, новое:"] + [f"• {f}" for f in found[-10:]], len(found)


def google_error(status, body):
    if status == 0:
        return f"сеть: {(body or b'').decode('utf-8', 'replace')[:80]}"
    try:
        err = json.loads(body or b"{}")
    except Exception:
        return f"HTTP {status}"
    detail = err.get("error")
    if isinstance(detail, dict):
        detail = detail.get("status") or detail.get("message")
    note = err.get("error_description") or ""
    return f"HTTP {status} {detail or '?'}" + (f" — {note[:90]}" if note else "")


GSC_SCOPE = "https://www.googleapis.com/auth/webmasters.readonly"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"


def jwt_assertion(adc, issued_at):
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding

    def seg(obj):
        raw = json.dumps(obj, separators=(",", ":")).encode()
        return base64.urlsafe_b64encode(raw).rstrip(b"=")

    aud = adc.get("token_uri") or GOOGLE_TOKEN_URL
    signing_input = seg({"alg": "RS256", "typ": "JWT"}) + b"." + seg({
        "iss": adc["client_email"], "scope": GSC_SCOPE, "aud": aud,
        "iat": issued_at, "exp": issued_at + 3600,
    })
    key = serialization.load_pem_private_key(adc["private_key"].encode(), password=None)
    sig = key.sign(signing_input, padding.PKCS1v15(), hashes.SHA256())
    return aud, (signing_input + b"." + base64.urlsafe_b64encode(sig).rstrip(b"=")).decode()


def gsc_auth(post, adc, now=None):
    if adc.get("type") == "service_account":
        missing = [k for k in ("client_email", "private_key") if not adc.get(k)]
        if missing:
            return None, "в GSC_ADC_JSON нет полей: " + ", ".join(missing)
        stamp = int((now or dt.datetime.now(dt.timezone.utc)).timestamp())
        try:
            url, assertion = jwt_assertion(adc, stamp)
        except ImportError:
            return None, "для сервисного аккаунта нужен пакет cryptography"
        except Exception as exc:
            return None, f"не удалось подписать JWT: {type(exc).__name__}"
        payload = {"grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer", "assertion": assertion}
        extra = {}
    else:
        missing = [k for k in ("client_id", "client_secret", "refresh_token") if not adc.get(k)]
        if missing:
            return None, "в GSC_ADC_JSON нет полей: " + ", ".join(missing)
        url = GOOGLE_TOKEN_URL
        payload = {
            "client_id": adc["client_id"], "client_secret": adc["client_secret"],
            "refresh_token": adc["refresh_token"], "grant_type": "refresh_token",
        }
        extra = {"x-goog-user-project": GSC_PROJECT}
    status, body = post(url, payload)
    if status != 200:
        return None, google_error(status, body)
    try:
        token = json.loads(body).get("access_token")
    except Exception:
        return None, f"HTTP {status}, ответ не JSON"
    if not token:
        return None, f"HTTP {status}, в ответе нет access_token"
    return {"Authorization": f"Bearer {token}", **extra}, None


def gsc_query(post, headers, start, end, dimensions):
    url = "https://www.googleapis.com/webmasters/v3/sites/" + urllib.parse.quote(GSC_SITE, safe="") + "/searchAnalytics/query"
    status, body = post(url, {"startDate": start, "endDate": end, "dimensions": dimensions, "rowLimit": 5000, "dataState": "all"},
                        headers)
    if status != 200:
        return None, google_error(status, body)
    try:
        return json.loads(body).get("rows", []), None
    except Exception:
        return None, f"HTTP {status}, ответ не JSON"


def summarise_pages(rows):
    year = {"impr": 0, "clicks": 0}
    vender = {"impr": 0, "clicks": 0}
    total = {"impr": 0, "clicks": 0}
    for row in rows or []:
        page = row["keys"][0]
        impr, clicks = row.get("impressions", 0), row.get("clicks", 0)
        total["impr"] += impr
        total["clicks"] += clicks
        if YEAR_PAGE.match(page):
            year["impr"] += impr
            year["clicks"] += clicks
        if "/pt/vender" in page:
            vender["impr"] += impr
            vender["clicks"] += clicks
    return total, year, vender


def ctr(bucket):
    return (bucket["clicks"] / bucket["impr"] * 100) if bucket["impr"] else 0.0


def gsc_summary(post, adc_json, today):
    if not adc_json:
        return ["Search Console: нет доступа (GSC_ADC_JSON)"], ["Search Console: секрет GSC_ADC_JSON не задан"]
    try:
        adc = json.loads(adc_json)
    except Exception:
        return ["Search Console: GSC_ADC_JSON не JSON"], ["Search Console: GSC_ADC_JSON не разбирается как JSON"]
    headers, why = gsc_auth(post, adc)
    if not headers:
        return [f"Search Console: токен не выдан — {why}"], [f"Search Console: токен не выдан — {why}"]
    end = today - dt.timedelta(days=3)
    start = end - dt.timedelta(days=6)
    rows, why = gsc_query(post, headers, start.isoformat(), end.isoformat(), ["page"])
    if rows is None:
        return [f"Search Console: запрос отклонён — {why}"], [f"Search Console: запрос отклонён — {why}"]
    total, year, vender = summarise_pages(rows)
    return [
        f"Search Console {start.strftime('%d.%m')}–{end.strftime('%d.%m')}: {total['impr']} показов, {total['clicks']} кликов, CTR {ctr(total):.1f}%",
        f"• страницы года: {year['impr']} показов, CTR {ctr(year):.1f}% (цель > 1.5%)",
        f"• /vender: {vender['impr']} показов, {vender['clicks']} кликов",
    ], []


CF_GRAPHQL = "https://api.cloudflare.com/client/v4/graphql"
CF_ACCOUNT = "6545f93bd664df2dfebb147bda85a191"
CF_ZONE = "8e62dd00726a3150eb7fb278cbb1865b"
CF_WORKER = "olx-car-parser"
CPU_LIMIT_WARN = 50
WORKER_QUERY = """query($acc:String!,$script:String!,$from:Time!,$to:Time!){viewer{accounts(filter:{accountTag:$acc}){
workersInvocationsAdaptive(limit:50,filter:{scriptName:$script,datetime_geq:$from,datetime_lt:$to}){sum{requests} dimensions{status}}}}}"""
GOOGLEBOT_QUERY = """query($zone:String!,$day:Date!){viewer{zones(filter:{zoneTag:$zone}){
httpRequestsAdaptiveGroups(limit:100,filter:{date:$day,userAgent_like:"%Googlebot%",verifiedBotCategory:"Search Engine Crawler"}){count dimensions{edgeResponseStatus}}}}}"""


def cf_graphql(post, token, query, variables):
    status, body = post(CF_GRAPHQL, {"query": query, "variables": variables}, {"Authorization": f"Bearer {token}"})
    try:
        data = json.loads(body or b"{}")
    except Exception:
        data = {}
    if status != 200 or data.get("errors") or not data.get("data"):
        errors = data.get("errors") or []
        return None, (errors[0].get("message") if errors else None) or f"HTTP {status}"
    return data["data"], None


def worker_health(post, token, day):
    if not token:
        return ["Воркер: нет доступа к аналитике Cloudflare (CF_API_TOKEN)"], []
    label = day.strftime("%d.%m")
    data, why = cf_graphql(post, token, WORKER_QUERY, {
        "acc": CF_ACCOUNT, "script": CF_WORKER,
        "from": f"{day.isoformat()}T00:00:00Z", "to": f"{(day + dt.timedelta(days=1)).isoformat()}T00:00:00Z"})
    if data is None:
        return [f"Воркер: аналитика Cloudflare не ответила — {why}"], [f"Cloudflare: запрос аналитики отклонён — {why}"]
    by_status = {}
    for row in data["viewer"]["accounts"][0]["workersInvocationsAdaptive"]:
        key = row["dimensions"]["status"]
        by_status[key] = by_status.get(key, 0) + row["sum"]["requests"]
    cpu = by_status.get("exceededResources", 0)
    lines = [f"Воркер {label}: {sum(by_status.values())} запросов, превышений CPU {cpu}"]
    warnings = []
    if cpu >= CPU_LIMIT_WARN:
        warnings.append(f"Воркер: {cpu} превышений CPU за {label} (Error 1102, посетитель видит 503)")
    gb, why = cf_graphql(post, token, GOOGLEBOT_QUERY, {"zone": CF_ZONE, "day": day.isoformat()})
    if gb is None:
        lines.append(f"• Googlebot: аналитика зоны не ответила — {why}")
        return lines, warnings
    codes = {}
    for row in gb["viewer"]["zones"][0]["httpRequestsAdaptiveGroups"]:
        code = row["dimensions"]["edgeResponseStatus"]
        codes[code] = codes.get(code, 0) + row["count"]
    errors = sum(n for code, n in codes.items() if code >= 500)
    lines.append(f"• Googlebot: {sum(codes.values())} запросов, 5xx {errors}, 404 {codes.get(404, 0)}")
    if errors:
        warnings.append(f"Googlebot получил {errors} ответов 5xx за {label}")
    return lines, warnings


REMINDERS = (
    (dt.date(2026, 10, 5), dt.date(2026, 10, 8),
     "📰 Пресс-рассылка индекса ушла 28.09 в 9 редакций: проверить ответы в ящике carsbuyer, напомнить только тем, кто не ответил."),
    (dt.date(2026, 10, 19), dt.date(2026, 10, 22),
     "🔎 Перемерить /comparar в Search Console: те же 10 адресов плюс контроль из /pt/preco и /pt/depreciacao. Расширять пул, только если /comparar индексируется лучше контроля."),
    (dt.date(2026, 10, 26), dt.date(2026, 10, 29),
     "🔎 Конец заморозки SEO после обвала 15.09: замер Search Console и решение про noindex тонких страниц лет; заодно перевести build_model_pages на _one_per_car (цены и число объявлений без дублей OLX/SV, выпадет ~28 страниц моделей и ~81 страница лет)."),
    (dt.date(2026, 10, 26), dt.date(2026, 10, 30),
     "🔁 После замера Search Console вернуть на /pt/preco блок «PARTILHAR ESTE PREÇO» и код виджета (сняты 29.09 на время заморозки, см. коммит 587310e)."),
    (dt.date(2026, 10, 7), dt.date(2026, 10, 12),
     "📉 Замер после сжатия SEO-поверхности (30.09 URL-инспекция: 65% «Google не знает», 9% crawled-not-indexed, индексировано ~9 страниц; показов 1-3 в сутки). Сайт ужат 4864 -> 2246 URL тремя коммитами: суточная сборка models.json (ced003f), гейт страницы модели в локалях + INTL_WAVE_MODELS 15 (0fe7197), волны PT 60/60/80 -> 20 (73f9ba3). Проверить URL Inspection по той же выборке 120 URL: доля unknown должна упасть, crawled-not-indexed — вырасти. Не сдвинулось — причина не в объёме, смотреть контент страниц года."),
    (dt.date(2026, 10, 27), dt.date(2026, 10, 31),
     "🏷 Кнопка «Pede uma proposta a um stand» (/pt/ir/vender) висит с 29.09: 8-10 и больше живых кликов в строке «Хотят продать стенду» — звонить стендам, меньше — снять кнопку."),
)


def dated_reminders(today):
    return [text for start, end, text in REMINDERS if start <= today < end]


def build_digest(now, sections, warnings):
    head = f"📊 Carsbuyer · {now.strftime('%d.%m %H:%M')} UTC"
    if warnings:
        head = "⚠️ " + head + "\n" + "\n".join(f"⚠️ {w}" for w in warnings)
    body = "\n".join(line for section in sections for line in section)
    return (head + "\n" + body)[:4000]


def send_telegram(post, token, chat_id, text):
    status, body = post(f"https://api.telegram.org/bot{token}/sendMessage",
                        {"chat_id": chat_id, "text": text, "disable_web_page_preview": True})
    return status == 200


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    now = dt.datetime.now(dt.timezone.utc)
    env = os.environ.get

    site_lines, site_warn = check_site(http_get)
    rel_lines, rel_warn = check_release(http_get, env("GITHUB_TOKEN"), now)
    health_lines, health_warn = worker_health(http_post_json, env("CF_API_TOKEN"), now.date() - dt.timedelta(days=1))
    lead_lines, lead_warn, fresh_leads = leads_summary(http_get, env("ANALYTICS_USER"), env("ANALYTICS_PASS"), now)
    click_lines, fresh_clicks = clicks_summary(http_get, env("ANALYTICS_USER"), env("ANALYTICS_PASS"), now.date())
    sell_lines, fresh_sell = sell_summary(http_get, env("ANALYTICS_USER"), env("ANALYTICS_PASS"), now.date())
    ai_lines = ai_summary(http_get, env("ANALYTICS_USER"), env("ANALYTICS_PASS"), now.date())
    mail_lines, mail_new = mail_summary(lambda: imaplib.IMAP4_SSL("imap.yandex.com", 993),
                                        env("MAIL_IMAP_USER"), env("MAIL_IMAP_PASSWORD"), now - dt.timedelta(days=1))
    weekly = args.force or now.weekday() == 0
    reminders = dated_reminders(now.date())
    sections = [site_lines, health_lines, rel_lines, lead_lines, click_lines, sell_lines, ai_lines, mail_lines, reminders]
    gsc_warn = []
    if weekly:
        gsc_lines, gsc_warn = gsc_summary(http_post_json, env("GSC_ADC_JSON"), now.date())
        sections.append(gsc_lines)
    warnings = site_warn + health_warn + rel_warn + lead_warn + gsc_warn
    text = build_digest(now, sections, warnings)
    print(text)
    quiet = (not warnings and not fresh_leads and not fresh_clicks and not fresh_sell
             and not mail_new and not weekly and not reminders)
    if quiet:
        print("\nnothing new, digest not sent")
        return 0
    if args.dry_run:
        return 0
    token, chat = env("TELEGRAM_BOT_TOKEN"), env("MONITOR_TELEGRAM_CHAT_ID")
    if not token or not chat:
        print("\ntelegram not configured", file=sys.stderr)
        return 1
    return 0 if send_telegram(http_post_json, token, chat, text) else 1


if __name__ == "__main__":
    sys.exit(main())
