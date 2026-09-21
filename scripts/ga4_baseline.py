import json
import os
import subprocess
import sys
import urllib.request

PROPERTY = os.environ.get("GA4_PROPERTY", "551644004")
DAYS = os.environ.get("DAYS", "14")
EVENTS = ("valuation_result", "olx_open", "lead_email", "fork_vender", "fork_comprar", "history_check")


def token():
    out = subprocess.run(
        ["gcloud", "auth", "application-default", "print-access-token"],
        capture_output=True, text=True,
    )
    return out.stdout.strip()


def report(tok, body):
    req = urllib.request.Request(
        f"https://analyticsdata.googleapis.com/v1beta/properties/{PROPERTY}:runReport",
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")
        if "SCOPE_INSUFFICIENT" in detail or e.code == 403:
            print("нет доступа к GA4 Data API. Выполни один раз:", file=sys.stderr)
            print("  gcloud auth application-default login --scopes="
                  "https://www.googleapis.com/auth/cloud-platform,"
                  "https://www.googleapis.com/auth/analytics.readonly", file=sys.stderr)
        else:
            print(detail[:400], file=sys.stderr)
        return None


def rows(res):
    return res.get("rows", []) if res else []


def main():
    tok = token()
    if not tok:
        print("gcloud не дал токен", file=sys.stderr)
        return 1

    period = [{"startDate": f"{DAYS}daysAgo", "endDate": "yesterday"}]

    totals = report(tok, {
        "dateRanges": period,
        "dimensions": [{"name": "eventName"}],
        "metrics": [{"name": "eventCount"}],
        "limit": 50,
    })
    if totals is None:
        return 2

    traffic = report(tok, {
        "dateRanges": period,
        "dimensions": [{"name": "sessionSource"}],
        "metrics": [{"name": "sessions"}],
        "limit": 15,
    })
    daily = report(tok, {
        "dateRanges": period,
        "dimensions": [{"name": "date"}, {"name": "eventName"}],
        "metrics": [{"name": "eventCount"}],
        "dimensionFilter": {"filter": {"fieldName": "eventName", "inListFilter": {"values": list(EVENTS)}}},
        "limit": 500,
    })

    days = int(DAYS)
    print(f"ФОН ЗА {days} ДНЕЙ, property {PROPERTY} (только посетители, давшие согласие)\n")
    print("события:")
    counts = {}
    for r in rows(totals):
        name = r["dimensionValues"][0]["value"]
        n = int(r["metricValues"][0]["value"])
        counts[name] = n
        mark = "  <-- воронка" if name in EVENTS else ""
        print(f"  {name:24} {n:7}   {n / days:6.2f} в день{mark}")

    print("\nисточники сессий:")
    for r in rows(traffic):
        print(f"  {r['dimensionValues'][0]['value']:24} {r['metricValues'][0]['value']:>7}")

    print("\nключевые числа для эксперимента:")
    for e in ("valuation_result", "lead_email", "olx_open"):
        n = counts.get(e, 0)
        print(f"  {e:18} всего {n:5}, в день {n / days:.2f}")

    if daily and rows(daily):
        print("\nпо дням:")
        for r in rows(daily):
            d, ev = r["dimensionValues"][0]["value"], r["dimensionValues"][1]["value"]
            print(f"  {d}  {ev:20} {r['metricValues'][0]['value']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
