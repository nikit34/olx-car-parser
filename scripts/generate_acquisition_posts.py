import argparse
import json
from pathlib import Path


def eur(v):
    if v is None:
        return "n/d"
    try:
        n = int(round(float(v)))
    except (TypeError, ValueError):
        return "n/d"
    return f"{n:,}".replace(",", " ") + " €"


def pick_deals(payload):
    if isinstance(payload, dict) and isinstance(payload.get("deals"), list):
        return payload["deals"]
    if isinstance(payload, list):
        return payload
    return []


def score(d):
    ds = d.get("decision_score")
    if ds is None:
        ds = d.get("flip_score")
    if ds is None:
        ds = 0
    try:
        ds = float(ds)
    except (TypeError, ValueError):
        ds = 0
    profit = d.get("est_profit_eur") or 0
    try:
        profit = float(profit)
    except (TypeError, ValueError):
        profit = 0
    return (ds, profit)


def name_of(d):
    title = (d.get("title") or "").strip()
    if title:
        return title
    brand = (d.get("brand") or "").strip()
    model = (d.get("model") or "").strip()
    return (brand + " " + model).strip() or "Viatura"


def spec_of(d):
    bits = []
    if d.get("year"):
        bits.append(str(d.get("year")))
    if d.get("mileage_km"):
        try:
            bits.append(f"{int(d.get('mileage_km')):,}".replace(",", " ") + " km")
        except (TypeError, ValueError):
            pass
    if d.get("fuel_type"):
        bits.append(str(d.get("fuel_type")))
    loc = city_short(d)
    if loc:
        bits.append(loc)
    return " · ".join(bits)


def city_short(d):
    raw = str(d.get("city") or d.get("district") or "").strip()
    if not raw:
        return ""
    cut = raw.split(" (")[0].split(",")[0].strip()
    if cut and cut == cut.upper():
        cut = cut.title()
    return cut


def short_name(d):
    brand = (d.get("brand") or "").strip()
    model = (d.get("model") or "").strip()
    base = (brand + " " + model).strip()
    if base:
        return base
    return name_of(d)


def car_url(host, olx_id, src):
    return f"https://{host}/pt/car?olx_id={olx_id}&utm_source={src}&utm_medium=social"


def deal_block(d, host, src):
    name = name_of(d)
    price = eur(d.get("price_eur"))
    fair = eur(d.get("fair_median"))
    profit = d.get("est_profit_eur")
    disc = d.get("discount_pct")
    try:
        disc_s = f"{round(float(disc) * 100)}%" if disc is not None else "n/d"
    except (TypeError, ValueError):
        disc_s = "n/d"
    profit_s = eur(profit) if profit is not None else "n/d"
    url = car_url(host, d.get("olx_id"), src)
    spec = spec_of(d)
    lines = [
        f"{name}",
        spec,
        f"Pedido {price} · Justo {fair} · Poupas {profit_s} ({disc_s} abaixo)",
        f"Ver análise completa: {url}",
    ]
    return "\n".join([x for x in lines if x])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deals", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--host", default="carsbuyer.org")
    ap.add_argument("--limit", type=int, default=7)
    args = ap.parse_args()
    payload = json.loads(Path(args.deals).read_text(encoding="utf-8"))
    deals = [d for d in pick_deals(payload) if d.get("olx_id")]
    deals = sorted(deals, key=score, reverse=True)[: max(args.limit, 1)]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mercado_telegram = f"https://{args.host}/pt/mercado?utm_source=telegram&utm_medium=social"
    mercado_facebook = f"https://{args.host}/pt/mercado?utm_source=facebook&utm_medium=social"
    mercado_tiktok = f"https://{args.host}/pt/mercado?utm_source=tiktok&utm_medium=social"
    avaliar_base = f"https://{args.host}/pt/avaliar"
    tg_parts = []
    for i, d in enumerate(deals, 1):
        tg_parts.append(f"Negócio {i}/{len(deals)}\n" + deal_block(d, args.host, "telegram"))
    (out / "telegram.txt").write_text(
        "\n\n---\n\n".join(tg_parts)
        + f"\n\nTodos os negócios de hoje: {mercado_telegram}\nAvaliar o teu anúncio: {avaliar_base}?utm_source=telegram&utm_medium=social\n",
        encoding="utf-8",
    )
    fb_parts = []
    for i, d in enumerate(deals, 1):
        fb_parts.append(
            f"Pediram-me para avaliar este: {name_of(d)} ({spec_of(d)}).\n"
            + f"Está pedido {eur(d.get('price_eur'))}, a nossa mediana para este modelo é {eur(d.get('fair_median'))}.\n"
            + deal_block(d, args.host, "facebook")
            + "\nPagavas este valor? Porquê?"
        )
    (out / "facebook.txt").write_text(
        "\n\n---\n\n".join(fb_parts)
        + f"\n\nVer todos: {mercado_facebook}\nAvaliação gratuita: {avaliar_base}?utm_source=facebook&utm_medium=social\n",
        encoding="utf-8",
    )
    tt_parts = []
    for i, d in enumerate(deals, 1):
        tt_parts.append(
            f"HOOK (0-3s): Este {short_name(d)} está {eur(d.get('price_eur'))} e vale {eur(d.get('fair_median'))}.\n"
            + f"MEIO (3-15s): {spec_of(d)}. Desconto {eur(d.get('est_profit_eur'))}. Vê o intervalo justo e os sinais de risco na página.\n"
            + f"CTA (15-25s): Link na bio e na descrição: {car_url(args.host, d.get('olx_id'), 'tiktok')}.\n"
            + f"Legenda: {short_name(d)} por {eur(d.get('price_eur'))} — justo {eur(d.get('fair_median'))}? Avaliação independente, sem registo. {mercado_tiktok}"
        )
    (out / "tiktok.txt").write_text("\n\n---\n\n".join(tt_parts) + "\n", encoding="utf-8")
    rows = ["day;channel;link"]
    for i, d in enumerate(deals, 1):
        rows.append(f"{i};telegram;{car_url(args.host, d.get('olx_id'), 'telegram')}")
        rows.append(f"{i};tiktok;{car_url(args.host, d.get('olx_id'), 'tiktok')}")
        rows.append(f"{i};facebook;{car_url(args.host, d.get('olx_id'), 'facebook')}")
    (out / "schedule.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(str(out))


if __name__ == "__main__":
    raise SystemExit(main())
