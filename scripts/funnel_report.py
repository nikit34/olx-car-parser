import json
import os
import sys
import urllib.request

ACCOUNT = os.environ.get("CF_ACCOUNT_ID", "6545f93bd664df2dfebb147bda85a191")
TOKEN = os.environ.get("CF_API_TOKEN", "")
DAYS = int(os.environ.get("DAYS", "14"))
DATASET = os.environ.get("FUNNEL_DATASET", "carsbuyer_funnel")


def query(sql):
    req = urllib.request.Request(
        f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT}/analytics_engine/sql",
        data=sql.encode("utf-8"),
        headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "text/plain"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def main():
    if not TOKEN:
        print("нужен CF_API_TOKEN с правом Account Analytics Read", file=sys.stderr)
        return 1

    by_event = query(f"""
        SELECT blob1 AS event, blob2 AS src, sum(_sample_interval) AS n
        FROM {DATASET}
        WHERE timestamp > NOW() - INTERVAL '{DAYS}' DAY
        GROUP BY event, src ORDER BY n DESC
    """)
    by_day = query(f"""
        SELECT toDate(timestamp) AS day, blob1 AS event, sum(_sample_interval) AS n
        FROM {DATASET}
        WHERE timestamp > NOW() - INTERVAL '{DAYS}' DAY
        GROUP BY day, event ORDER BY day
    """)

    by_path = query(f"""
        SELECT blob1 AS event, blob4 = '/avaliar' AS pt, sum(_sample_interval) AS n
        FROM {DATASET}
        WHERE timestamp > NOW() - INTERVAL '{DAYS}' DAY AND blob1 = 'valuation'
        GROUP BY event, pt
    """)
    dropped = query(f"""
        SELECT blob1 AS event, blob2 AS src, blob6 AS why, sum(_sample_interval) AS n
        FROM {DATASET}
        WHERE timestamp > NOW() - INTERVAL '{DAYS}' DAY
          AND blob1 IN ('visit', 'visit_bot', 'valuation_bot')
        GROUP BY event, src, why ORDER BY n DESC
    """)

    rows = by_event.get("data", [])
    visits_fb = sum(int(r["n"]) for r in rows if r["event"] == "visit" and r["src"] == "fb")
    bots_fb = sum(int(r["n"]) for r in rows if r["event"] == "visit_bot" and r["src"] == "fb")
    visits_all = sum(int(r["n"]) for r in rows if r["event"] == "visit")
    visits_gpt = sum(int(r["n"]) for r in rows if r["event"] == "visit" and r["src"] == "chatgpt.com")
    split = {str(r["pt"]): int(r["n"]) for r in by_path.get("data", [])}
    valuations = split.get("1", 0)
    val_intl = split.get("0", 0)
    val_fb = sum(int(r["n"]) for r in rows if r["event"] == "valuation" and r["src"] == "fb")

    print(f"период: {DAYS} дней")
    print(f"переходов с меткой fb:        {visits_fb}   (порог 300, стоп по каналу ниже 150)")
    print(f"из них отсеяно как боты:      {bots_fb}   (скрейперы соцсетей, в порог не идут)")
    print(f"переходов из chatgpt.com:     {visits_gpt}")
    print(f"переходов со всеми метками:   {visits_all}")
    print(f"оценок на /pt/avaliar:        {valuations}")
    print(f"оценок на de/fr/it:           {val_intl}   (в порог fb не идут)")
    print(f"оценок с сохранённой меткой:  {val_fb}")
    if visits_fb:
        print(f"оценок на переход из fb:      {100 * valuations / visits_fb:.1f}%  (норма 20, граница жизни 10)")
    print()
    print("человек или нет (why пусто = человек; sem-ua/prefetch/bot/lang/nav/dc = отсеяно):")
    for r in dropped.get("data", []):
        print(f"  {r['event']:13} {r['src'] or '-':14} {r['why'] or '-':9} {r['n']}")
    print()
    print("по дням:")
    for r in by_day.get("data", []):
        print(f"  {r['day']}  {r['event']:12} {r['n']}")
    print()
    print("метка внутри сайта не переносится: оценка считается по общему числу за период")
    print("против фона до эксперимента, а не по атрибуции на посетителя")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
