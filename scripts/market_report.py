import argparse
import hashlib
import html
import json
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

UTC = timezone.utc
CODE_SNAPSHOT_LINES = "L137-L161"


def esc(value):
    return html.escape("" if value is None else str(value))


def thousands(value):
    return f"{int(round(value)):,}".replace(",", ".")


def eur(value):
    return f"{thousands(value)} €"


def km(value):
    return "não indicada" if value is None or pd.isna(value) else f"{thousands(value)} km"


def signed(value, unit):
    if value is None or pd.isna(value):
        return "-"
    value = int(round(value))
    if value == 0:
        return f"0 {unit}".strip()
    sign = "+" if value > 0 else "−"
    return f"{sign}{thousands(abs(value))} {unit}".strip()


def ts_utc(value):
    return value.strftime("%d/%m/%Y %H:%M UTC")


def day(value):
    return value.strftime("%d/%m/%Y")


def to_utc(series):
    return pd.to_datetime(series, format="mixed", utc=True, errors="coerce")


def read_table(path):
    return pd.read_csv(path) if str(path).endswith(".csv") else pd.read_parquet(path)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collection_gap(steps, ref):
    ok = [s for s in steps if s.get("scrape_step") == "success" and s.get("step_started") and s.get("step_completed")]
    for s in ok:
        s["_start"] = pd.Timestamp(s["step_started"])
        s["_end"] = pd.Timestamp(s["step_completed"])
    before = max((s for s in ok if s["_end"] <= ref), key=lambda s: s["_end"])
    after = min((s for s in ok if s["_start"] >= ref), key=lambda s: s["_start"])
    return before, after


def select(listings, params, ref_end, gap_start):
    f = params["filters"]
    m = listings
    m = m[(m.brand == f["brand"]) & (m.model == f["model"])]
    m = m[m.generation == f["generation"]]
    m = m[m.fuel_type.fillna("").str.startswith(f["fuel_prefix"])]
    m = m[m.year.between(f["year_min"], f["year_max"])]
    m = m[m.horsepower.between(f["hp_min"], f["hp_max"])]
    m = m[m.engine_cc.between(f["cc_min"], f["cc_max"])]
    m = m[m.seller_type == f["seller_type"]]
    m = m[m.first_seen_at <= ref_end]
    m = m[m.last_scraped_at >= pd.Timestamp(f["seen_since"], tz=UTC)]
    m = m[m.deactivated_at.isna() | (m.deactivated_at >= gap_start)]
    return m.copy()


def car_key(row):
    if pd.isna(row.mileage_km):
        return ("id", row.olx_id)
    return (int(row.year), int(row.mileage_km), row.district)


def build_cars(sel, snaps, gap_start, gap_end, first_after_end):
    sel = sel.assign(_key=sel.apply(car_key, axis=1))
    cars = []
    for _, group in sel.groupby("_key", sort=False):
        group = group.sort_values(["source", "first_seen_at"], key=lambda c: c.map({"olx": 0, "standvirtual": 1}) if c.name == "source" else c)
        rep = group.iloc[0]
        hist = snaps[snaps.olx_id.isin(group.olx_id)].sort_values("scraped_at")
        pre = hist[hist.scraped_at <= gap_start]
        win = hist[(hist.scraped_at >= gap_end) & (hist.scraped_at <= first_after_end)]
        after = hist[hist.scraped_at >= gap_end]
        pre_row = pre.iloc[-1] if len(pre) else None
        win_row = win.iloc[0] if len(win) else None
        after_row = after.iloc[0] if len(after) else None
        if pre_row is not None:
            price_a = float(pre_row.price_eur)
            changed = win_row is not None and float(win_row.price_eur) != price_a
            price_b = float(win_row.price_eur) if changed else price_a
            kind = "changed" if changed else "exact"
        else:
            price_a = price_b = float(after_row.price_eur)
            kind = "after"
        removed = group.deactivated_at.max() if group.deactivated_at.notna().all() else None
        origin = next((o for o in group.origin if isinstance(o, str) and o), None)
        records = []
        for _, r in group.iterrows():
            records.append({
                "olx_id": r.olx_id,
                "source": "OLX" if r.source == "olx" else "Standvirtual",
                "url": r.url,
                "title": r.title,
                "published": r.first_seen_at,
                "last_seen": r.last_scraped_at,
                "removed": r.deactivated_at if pd.notna(r.deactivated_at) else None,
                "history": [(h.scraped_at, float(h.price_eur)) for h in hist[hist.olx_id == r.olx_id].itertuples()],
            })
        cars.append({
            "key": group._key.iloc[0],
            "year": int(rep.year),
            "hp": int(rep.horsepower),
            "cc": int(rep.engine_cc),
            "gearbox": next((g for g in group.transmission if isinstance(g, str) and g), "não indicada"),
            "km": None if pd.isna(rep.mileage_km) else int(rep.mileage_km),
            "city": rep.city,
            "district": rep.district,
            "origin": {"national": "nacional", "imported": "importado"}.get(origin, "não indicada"),
            "published": group.first_seen_at.min(),
            "last_seen": group.last_scraped_at.max(),
            "removed": removed,
            "confirmed": bool((group.last_scraped_at >= gap_end).any()),
            "pre": pre_row,
            "win": win_row,
            "after": after_row,
            "price_a": price_a,
            "price_b": price_b,
            "kind": kind,
            "records": records,
        })
    cars.sort(key=lambda c: (c["km"] is None, c["km"] or 0))
    for i, c in enumerate(cars, 1):
        c["n"] = i
    return cars


def stats(values):
    return {"n": len(values), "min": min(values), "max": max(values), "median": statistics.median(values)}


CSS = """
:root{--ink:#14181d;--muted:#5b6470;--line:#d9dde3;--soft:#f4f5f7;--accent:#0b5cad;--warn:#8a4b00}
*{box-sizing:border-box}
html{background:#fff}
body{margin:0;color:var(--ink);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;background:#fff}
main{max-width:1060px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:26px;line-height:1.2;margin:0 0 6px}
h2{font-size:18px;margin:32px 0 10px;padding-top:8px;border-top:1px solid var(--line)}
h3{font-size:15px;margin:18px 0 6px}
p{margin:6px 0}
ul{margin:6px 0;padding-left:20px}
li{margin:3px 0}
.sub{color:var(--muted);font-size:15px;margin:0 0 16px}
.box{border:1px solid var(--line);background:var(--soft);border-radius:8px;padding:12px 14px;margin:12px 0}
.box.warn{border-color:#e8c48f;background:#fff8ec}
.meta{display:grid;grid-template-columns:minmax(150px,220px) 1fr;gap:4px 14px;margin:10px 0}
.meta div:nth-child(odd){color:var(--muted)}
.table-wrap{overflow-x:auto;margin:8px 0}
table{border-collapse:collapse;width:100%;font-size:12.5px}
th,td{border:1px solid var(--line);padding:5px 6px;text-align:left;vertical-align:top}
th{background:var(--soft);font-weight:600}
td.num,th.num{text-align:right;white-space:nowrap}
.tag{display:inline-block;padding:0 6px;border-radius:4px;font-size:11.5px;background:#e9eef5}
.tag.no{background:#fde8e8;color:#8b1c1c}
.tag.chg{background:#fff1d6;color:var(--warn)}
.card{border:1px solid var(--line);border-radius:8px;padding:10px 12px;margin:10px 0;break-inside:avoid}
.card h3{margin:0 0 4px}
.small{font-size:12px;color:var(--muted)}
a{color:var(--accent);word-break:break-all}
.toolbar{position:sticky;top:0;z-index:2;background:#fff;border-bottom:1px solid var(--line);margin:-24px -16px 16px;padding:10px 16px;display:flex;gap:12px;align-items:center;justify-content:space-between;flex-wrap:wrap}
.btn{display:inline-block;background:var(--accent);color:#fff;text-decoration:none;font-weight:600;padding:9px 16px;border-radius:6px;word-break:normal}
.sign{margin-top:24px}
@page{size:A4;margin:14mm 12mm 16mm}
@page wide{size:A4 landscape;margin:12mm}
@media print{
  .no-print{display:none!important}
  body{font-size:10.5px}
  main{max-width:none;padding:0}
  h1{font-size:20px}
  h2{font-size:14px;margin:18px 0 8px;break-after:avoid}
  h3{font-size:12px}
  table{font-size:8.5px}
  th,td{padding:3px 4px}
  .small{font-size:8.5px}
  .tag{font-size:8px;padding:0 4px}
  td .small{font-size:7.5px}
  .table-wrap{overflow:visible}
  .wide{page:wide}
  tr{break-inside:avoid}
  a{color:var(--ink)}
}
"""


def price_cell(row):
    if row is None:
        return "-"
    return f"{eur(row.price_eur)}<br><span class=small>{esc(ts_utc(row.scraped_at))}</span>"


def considered_cell(c):
    if c["kind"] == "exact":
        return eur(c["price_a"])
    if c["kind"] == "changed":
        return f"<span class='tag chg'>alterado</span><br>A: {eur(c['price_a'])}<br>B: {eur(c['price_b'])}"
    return f"{eur(c['price_a'])}<br><span class=small>primeiro preço registado após publicação</span>"


def render(params, cars, gap_before, gap_after, steps, files, token, generator_commit):
    ref = pd.Timestamp(params["ref_date"])
    ref_txt = day(ref)
    v = params["vehicle"]
    a = params["author"]
    issued = day(pd.Timestamp(params["issued"]))
    gap_start, gap_end = gap_before["_end"], gap_after["_start"]
    confirmed = [c for c in cars if c["confirmed"]]
    unconfirmed = [c for c in cars if not c["confirmed"]]
    changed = [c for c in cars if c["kind"] == "changed"]
    after_only = [c for c in cars if c["kind"] == "after"]
    n_records = sum(len(c["records"]) for c in cars)
    sa = stats([c["price_a"] for c in confirmed])
    sb = stats([c["price_b"] for c in confirmed])
    ta = stats([c["price_a"] for c in cars])
    tb = stats([c["price_b"] for c in cars])
    known_km = [c for c in cars if c["km"] is not None]
    lower_km = [c for c in known_km if c["km"] < v["km"]]
    no_km = [c for c in cars if c["km"] is None]
    origins = {}
    for c in cars:
        origins[c["origin"]] = origins.get(c["origin"], 0) + 1
    dup_groups = [c for c in cars if len(c["records"]) > 1]
    repo = params["data"]["repo"]
    pipe = params["data"]["pipeline_commit"]
    code_url = f"{repo}/blob/{pipe}/src/storage/repository.py#{CODE_SNAPSHOT_LINES}"
    runs_url = f"{repo}/actions/workflows/scrape.yml"
    gen_url = f"{repo}/blob/{generator_commit}/scripts/market_report.py"
    pdf_href = f"/relatorio/{token}/relatorio.pdf"

    out = []
    w = out.append
    w("<!doctype html><html lang=pt-PT><head><meta charset=utf-8>")
    w("<meta name=viewport content='width=device-width,initial-scale=1'>")
    w("<meta name=robots content='noindex,nofollow'>")
    w(f"<title>Relatório de dados de mercado - {esc(v['label'])} - {ref_txt}</title>")
    w(f"<style>{CSS}</style></head><body><main>")
    w(f"<div class='toolbar no-print'><span class=small>Relatório disponível para consulta e descarga</span><a class=btn href='{pdf_href}'>Descarregar PDF</a></div>")
    w("<h1>Relatório de dados de mercado</h1>")
    w(f"<p class=sub>Anúncios de {esc(v['label'])} ({params['filters']['hp_min']}-{params['filters']['hp_max']} cv, {params['filters']['year_min']}-{params['filters']['year_max']}) disponíveis em Portugal à data de {ref_txt}</p>")
    w("<div class=meta>")
    for label, value in [
        ("Data de emissão", issued),
        ("Data de referência", ref_txt),
        ("Requerente", params["requester"]),
        ("Referência indicada pelo requerente", params["reference"]),
        ("Elaborado por", f"{a['name']}, {a['site']} ({a['email']})"),
        ("Identificador do relatório", params["report_id"]),
    ]:
        w(f"<div>{esc(label)}</div><div>{esc(value)}</div>")
    w("</div>")
    w("<div class='box warn'><strong>Natureza do documento.</strong> Este relatório apresenta dados históricos de anúncios de venda, isto é, preços pedidos pelos vendedores, recolhidos automaticamente pelo carsbuyer.org. Não é uma peritagem nem uma avaliação da viatura de referência, não apresenta preços de transações concluídas e não propõe qualquer valor para a viatura.</div>")

    w("<h2>1. Viatura de referência</h2>")
    w("<p>Dados indicados pelo requerente, não verificados pelo autor:</p><div class=meta>")
    for label, value in [
        ("Modelo", v["label"]),
        ("Ano", v["year"]),
        ("Potência", f"{v['hp']} cv"),
        ("Quilometragem", km(v["km"])),
        ("Matrícula", f"{v['plate']} ({v['registration_country']})"),
    ]:
        w(f"<div>{esc(label)}</div><div>{esc(value)}</div>")
    w("</div>")

    w("<h2>2. Resumo</h2><ul>")
    w(f"<li>Foram encontrados {n_records} anúncios que cumprem os critérios da secção 4, correspondentes a <strong>{len(cars)} viaturas distintas</strong> depois de excluídos os duplicados (secção 5).</li>")
    w(f"<li><strong>{len(confirmed)} viaturas</strong> estavam disponíveis a {ref_txt}: foram publicadas até essa data e continuavam anunciadas depois do intervalo sem recolhas. {len(unconfirmed)} viaturas foram retiradas entre {day(gap_start)} e {day(gap_end)}, pelo que não é possível confirmar se estavam disponíveis a {ref_txt}; são apresentadas, mas não entram no cálculo principal.</li>")
    w(f"<li>Preços pedidos das {len(confirmed)} viaturas confirmadas: mínimo {eur(sa['min'])}, máximo {eur(sa['max'])}, mediana {eur(sa['median'])} no cenário A e {eur(sb['median'])} no cenário B (secção 6).</li>")
    w(f"<li>Em {len(changed)} viaturas o preço mudou entre a última recolha antes do intervalo sem recolhas e a primeira recolha depois; o preço em vigor a {ref_txt} é um dos dois valores, e por isso são apresentados dois cenários.</li>")
    lk = "; ".join(f"n.º {c['n']}: {km(c['km'])}, {c['year']}, {eur(c['price_a'])}" for c in lower_km) or "nenhuma"
    w(f"<li>Quilometragem dos comparáveis: de {km(min(c['km'] for c in known_km))} a {km(max(c['km'] for c in known_km))}. Com quilometragem inferior à da viatura de referência ({km(v['km'])}): {esc(lk)}. Sem quilometragem no anúncio: {len(no_km)}.</li>")
    w(f"<li>Não houve recolhas entre {ts_utc(gap_start)} e {ts_utc(gap_end)} (secção 3 e anexo B).</li>")
    w("</ul>")

    w("<h2>3. Fonte e recolha dos dados</h2>")
    w("<p>O carsbuyer.org recolhe automaticamente anúncios de automóveis usados publicados no OLX.pt e no Standvirtual. As recolhas estão agendadas até 12 vezes por dia (de duas em duas horas, mais uma recolha alargada diária); o número de recolhas efetivamente concluídas em cada dia consta do anexo B.</p>")
    w("<p>Para cada anúncio são guardados o título, o endereço (URL), a marca, o modelo, a geração, o ano, a quilometragem, a potência, a cilindrada, o combustível, a caixa, a localidade, o distrito, o tipo de vendedor, a origem indicada no anúncio (nacional ou importado) e a data de publicação indicada pela plataforma.</p>")
    w(f"<p>O preço é guardado como registo com data e hora (UTC) quando o anúncio é encontrado pela primeira vez e sempre que o preço muda. Uma recolha em que o preço não muda não cria novo registo (<a href='{esc(code_url)}'>código-fonte da regra</a>). São ainda guardadas a data e hora da última recolha em que o anúncio foi encontrado e a data da recolha em que deixou de o ser. Por isso, o preço em vigor numa data é o último registo anterior a essa data, enquanto o anúncio continuar publicado.</p>")
    w(f"<p><strong>Intervalo sem recolhas.</strong> A última recolha concluída antes de {ref_txt} terminou em {ts_utc(gap_start)} (<a href='{esc(gap_before['url'])}'>execução</a>). Em todas as execuções agendadas entre essa hora e {ts_utc(gap_end)} o passo de recolha falhou e nenhum anúncio foi lido; a recolha seguinte começou em {ts_utc(gap_end)} (<a href='{esc(gap_after['url'])}'>execução</a>). O histórico de execuções é público em <a href='{esc(runs_url)}'>{esc(runs_url)}</a>.</p>")
    if params["data"].get("source_text"):
        w(f"<p><strong>Extração.</strong> {esc(params['data']['source_text'])} Ficheiros de extração, conservados pelo autor, e respetivas somas SHA-256:</p><ul>")
    else:
        w(f"<p><strong>Extração.</strong> Os dados deste relatório foram extraídos em {esc(params['data']['downloaded'])} da publicação «{esc(params['data']['release'])}» do repositório público <a href='{esc(repo)}'>{esc(repo)}</a>. Ficheiros usados e respetivas somas SHA-256:</p><ul>")
    for name, digest in files.items():
        w(f"<li><code>{esc(name)}</code>: <code>{esc(digest)}</code></li>")
    w(f"</ul><p>O cálculo foi feito pelo programa <a href='{esc(gen_url)}'>scripts/market_report.py</a>, publicado no mesmo repositório.</p>")

    f = params["filters"]
    w("<h2>4. Critérios de seleção</h2><ul>")
    w(f"<li>Marca e modelo: {esc(f['brand'])} {esc(f['model'])}, geração {esc(f['generation'])}.</li>")
    w(f"<li>Motor: gasóleo, cilindrada entre {thousands(f['cc_min'])} e {thousands(f['cc_max'])} cm³ (2.0 TDI), potência entre {f['hp_min']} e {f['hp_max']} cv.</li>")
    w(f"<li>Ano: {f['year_min']} a {f['year_max']} (ano da viatura de referência ± 1).</li>")
    w("<li>Vendedor particular.</li>")
    w(f"<li>Anúncio publicado até {ref_txt}, segundo a data indicada pela plataforma.</li>")
    w(f"<li>Anúncio ainda publicado na última recolha antes do intervalo sem recolhas ({day(gap_start)}) ou publicado depois dela, e encontrado pelo sistema em setembro de 2026.</li>")
    w(f"<li>Disponibilidade a {ref_txt} considerada confirmada quando o anúncio continuava publicado em recolhas posteriores a {ts_utc(gap_end)}.</li>")
    w("<li>Não foram aplicados critérios de preço, caixa, estado, equipamento ou cor; a caixa de velocidades é indicada para cada viatura.</li></ul>")

    w("<h2>5. Duplicados</h2>")
    w(f"<p>O mesmo carro pode estar publicado no OLX e no Standvirtual, ou duas vezes na mesma plataforma. Anúncios com o mesmo ano, a mesma quilometragem e o mesmo distrito são tratados como uma única viatura; o preço não é usado neste critério, porque pode diferir entre plataformas e ao longo do tempo. Foram agrupados {n_records - len(cars)} anúncios duplicados em {len(dup_groups)} viaturas; o anexo A indica todos os anúncios de cada viatura.</p>")
    w("<div class=table-wrap><table><tr><th>N.º</th><th>Anúncios agrupados</th></tr>")
    for c in dup_groups:
        items = "<br>".join(f"{esc(r['source'])} {esc(r['olx_id'])}, publicado em {day(r['published'])}" for r in c["records"])
        w(f"<tr><td>{c['n']}</td><td>{items}</td></tr>")
    w("</table></div>")

    w("<h2>6. Preço à data de referência e estatística</h2><ul>")
    w(f"<li><strong>Preço em vigor a {ref_txt}.</strong> É o último registo de preço feito até {ts_utc(gap_start)}. Se a primeira recolha depois do intervalo ({ts_utc(gap_end)} a {ts_utc(gap_after['_end'])}) registou outro preço, o preço mudou algures entre essas datas e o valor em vigor a {ref_txt} não é conhecido: o cenário A usa o preço anterior e o cenário B o preço registado depois.</li>")
    w(f"<li><strong>Anúncios publicados durante o intervalo sem recolhas</strong> ({len(after_only)}): é usado o primeiro preço registado, nos dois cenários.</li>")
    w("<li><strong>Intervalo de preços</strong>: o menor e o maior preço considerados. <strong>Mediana</strong>: o valor central depois de ordenar os preços; com um número par de valores, a média dos dois centrais.</li></ul>")
    w("<div class=table-wrap><table><tr><th>Conjunto</th><th class=num>Viaturas</th><th class=num>Mínimo</th><th class=num>Máximo</th><th class=num>Mediana A</th><th class=num>Mediana B</th></tr>")
    w(f"<tr><td>Disponibilidade confirmada a {ref_txt}</td><td class=num>{sa['n']}</td><td class=num>{eur(min(sa['min'], sb['min']))}</td><td class=num>{eur(max(sa['max'], sb['max']))}</td><td class=num>{eur(sa['median'])}</td><td class=num>{eur(sb['median'])}</td></tr>")
    w(f"<tr><td>Todas, incluindo as não confirmadas</td><td class=num>{ta['n']}</td><td class=num>{eur(min(ta['min'], tb['min']))}</td><td class=num>{eur(max(ta['max'], tb['max']))}</td><td class=num>{eur(ta['median'])}</td><td class=num>{eur(tb['median'])}</td></tr>")
    w("</table></div>")

    w("<section class=wide><h2>7. Viaturas comparáveis</h2>")
    w(f"<p>Ordenadas por quilometragem. Datas e horas dos registos em UTC. «Antes do intervalo»: último registo de preço até {day(gap_start)}. «Primeira recolha depois»: preço registado na recolha de {day(gap_end)}, apenas quando mudou.</p>")
    w("<div class=table-wrap><table><tr><th>N.º</th><th class=num>Ano</th><th class=num>Potência</th><th>Caixa</th><th class=num>Quilometragem</th><th>Localização</th><th>Publicado em</th><th>Preço antes do intervalo</th><th>Primeira recolha depois</th><th>Preço considerado</th><th>Disponível a " + ref_txt + "</th></tr>")
    for c in cars:
        status = "<span class=tag>confirmado</span>" if c["confirmed"] else f"<span class='tag no'>não confirmado</span><br><span class=small>retirado na recolha de {day(c['removed'])}</span>"
        win = price_cell(c["win"]) if c["kind"] == "changed" else ("sem alteração" if c["pre"] is not None else "-")
        w(f"<tr><td>{c['n']}</td><td class=num>{c['year']}</td><td class=num>{c['hp']} cv</td><td>{esc(c['gearbox'])}</td><td class=num>{km(c['km'])}</td><td>{esc(c['city'])} ({esc(c['district'])})</td><td>{day(c['published'])}</td><td>{price_cell(c['pre'])}</td><td>{win}</td><td>{considered_cell(c)}</td><td>{status}</td></tr>")
    w("</table></div></section>")

    w("<h2>8. Diferenças face à viatura de referência</h2>")
    w(f"<p>Diferença = comparável menos viatura de referência ({v['year']}, {v['hp']} cv, {km(v['km'])}). «Origem indicada» é a indicação do próprio anúncio (nacional ou importado); todos os comparáveis estão anunciados no mercado português, enquanto a viatura de referência tem matrícula {esc(v['registration_adj'])}. O relatório não quantifica o efeito destas diferenças no preço.</p>")
    w("<div class=table-wrap><table><tr><th>N.º</th><th class=num>Quilometragem</th><th class=num>Diferença de km</th><th class=num>Diferença de potência</th><th class=num>Diferença de ano</th><th>Origem indicada</th></tr>")
    for c in cars:
        dk = signed(c["km"] - v["km"], "km") if c["km"] is not None else "-"
        w(f"<tr><td>{c['n']}</td><td class=num>{km(c['km'])}</td><td class=num>{dk}</td><td class=num>{signed(c['hp'] - v['hp'], 'cv')}</td><td class=num>{signed(c['year'] - v['year'], '')}</td><td>{esc(c['origin'])}</td></tr>")
    w("</table></div>")
    origin_txt = ", ".join(f"{n} {o}" for o, n in sorted(origins.items(), key=lambda x: -x[1]))
    w(f"<p class=small>Origem indicada: {esc(origin_txt)}.</p>")

    w("<h2>9. Limitações</h2><ul>")
    w("<li>Os valores são preços pedidos em anúncios, não preços de transações concluídas.</li>")
    w("<li>Não existem capturas de ecrã nem cópias arquivadas das páginas dos anúncios. Os dados apresentados são os que o sistema registou em cada recolha. Vários anúncios já foram retirados e os respetivos endereços podem já não funcionar.</li>")
    w(f"<li>Não houve recolhas entre {ts_utc(gap_start)} e {ts_utc(gap_end)}. Por isso, o preço em vigor a {ref_txt} é desconhecido em {len(changed)} viaturas (dois cenários) e a disponibilidade nessa data não pode ser confirmada em {len(unconfirmed)} viaturas.</li>")
    w("<li>O sistema não guarda um registo de cada recolha por anúncio: guarda os registos de preço (primeira observação e cada alteração), a última recolha em que o anúncio foi encontrado e a recolha em que deixou de o ser. Assume-se, como o sistema foi desenhado, que cada recolha percorre todos os anúncios publicados; não é possível prová-lo anúncio a anúncio.</li>")
    w("<li>Ano, quilometragem, potência, caixa e origem são os declarados nos anúncios e não foram verificados. Um anúncio não indica a quilometragem.</li>")
    w("<li>A data de publicação é a indicada pela plataforma e pode refletir uma renovação do anúncio.</li>")
    w("<li>Alguns anúncios registam várias alterações de preço em poucos dias, incluindo subidas (ver anexo A). Os valores são apresentados tal como foram registados, sem correção.</li>")
    w("<li>A retirada de um anúncio não significa que a viatura tenha sido vendida.</li>")
    w("<li>Não foram considerados estado de conservação, equipamento, histórico de manutenção ou de sinistros, nem a diferença de mercado de matrícula.</li>")
    w("</ul>")

    w("<h2>10. Declaração</h2>")
    w(f"<p>Declaro que os dados apresentados foram extraídos do arquivo do carsbuyer.org e tratados exclusivamente segundo os critérios aqui descritos, definidos sem consideração do resultado, e que o relatório não contém qualquer valor proposto para a viatura de referência.</p>")
    w(f"<p class=sign>{esc(a['name'])}<br>{esc(a['site'])}, {esc(a['email'])}<br>{issued}</p>")

    w("<h2>Anexo A. Anúncios e histórico de preços</h2>")
    w("<p class=small>Todos os registos de preço guardados para cada anúncio, em UTC. A linha a negrito é o último registo antes do intervalo sem recolhas.</p>")
    for c in cars:
        km_txt = km(c["km"])
        w(f"<div class=card><h3>N.º {c['n']}: {c['year']}, {c['hp']} cv, {esc(km_txt)}, {esc(c['city'])} ({esc(c['district'])})</h3>")
        for r in c["records"]:
            tail = f"retirado na recolha de {day(r['removed'])}" if r["removed"] is not None else f"ainda publicado na última recolha, {ts_utc(r['last_seen'])}"
            w(f"<p><strong>{esc(r['source'])} {esc(r['olx_id'])}</strong>: {esc(r['title'])}<br><a href='{esc(r['url'])}'>{esc(r['url'])}</a><br><span class=small>Publicado em {day(r['published'])}; {tail}.</span></p>")
            pre_ts = max((t for t, _ in r["history"] if t <= gap_start), default=None)
            rows = []
            for t, p in r["history"]:
                line = f"{ts_utc(t)}: {eur(p)}"
                rows.append(f"<li><strong>{line}</strong></li>" if t == pre_ts else f"<li>{line}</li>")
            w("<ul class=small>" + "".join(rows) + "</ul>")
        w("</div>")

    w("<h2>Anexo B. Execuções do sistema de recolha</h2>")
    w(f"<p class=small>Execuções agendadas do processo «Scrape OLX» e resultado do passo de recolha, por dia (UTC). Fonte: <a href='{esc(runs_url)}'>{esc(runs_url)}</a>.</p>")
    per_day = {}
    for s in steps:
        d = s["created_at"][:10]
        k = "ok" if s.get("scrape_step") == "success" else "fail"
        per_day.setdefault(d, {"ok": 0, "fail": 0})[k] += 1
    w("<div class=table-wrap><table><tr><th>Dia</th><th class=num>Recolha concluída</th><th class=num>Recolha falhada ou não iniciada</th></tr>")
    for d in sorted(per_day):
        dd = datetime.strptime(d, "%Y-%m-%d").strftime("%d/%m/%Y")
        w(f"<tr><td>{dd}</td><td class=num>{per_day[d]['ok']}</td><td class=num>{per_day[d]['fail']}</td></tr>")
    w("</table></div>")
    w("</main></body></html>")
    summary = {
        "records": n_records, "cars": len(cars), "confirmed": len(confirmed), "unconfirmed": len(unconfirmed),
        "changed": len(changed), "after_only": len(after_only), "confirmed_A": sa, "confirmed_B": sb, "all_A": ta, "all_B": tb,
        "lower_km": [(c["n"], c["km"], c["price_a"]) for c in lower_km], "no_km": len(no_km),
        "gap_start": str(gap_start), "gap_end": str(gap_end), "origins": origins,
    }
    return "".join(out), summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--listings", required=True)
    ap.add_argument("--snapshots", required=True)
    ap.add_argument("--steps", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chrome", default="")
    args = ap.parse_args()

    params = json.loads(Path(args.params).read_text())
    token = params["token"]
    generator_commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                      cwd=Path(__file__).parent).stdout.strip()
    listings = read_table(args.listings)
    for col in ("first_seen_at", "last_seen_at", "last_scraped_at", "deactivated_at"):
        listings[col] = to_utc(listings[col])
    snaps = read_table(args.snapshots)[["olx_id", "price_eur", "scraped_at"]].copy()
    snaps["scraped_at"] = to_utc(snaps["scraped_at"])
    steps = json.loads(Path(args.steps).read_text())

    ref = pd.Timestamp(params["ref_date"], tz=UTC)
    ref_end = ref + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    gap_before, gap_after = collection_gap(steps, ref)
    sel = select(listings, params, ref_end, gap_before["_end"])
    cars = build_cars(sel, snaps, gap_before["_end"], gap_after["_start"], gap_after["_end"])
    files = {Path(args.listings).name: sha256(args.listings), Path(args.snapshots).name: sha256(args.snapshots)}
    doc, summary = render(params, cars, gap_before, gap_after, steps, files, token, generator_commit or "master")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.html").write_text(doc, encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str, ensure_ascii=False), encoding="utf-8")
    if args.chrome:
        subprocess.run([args.chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={out / 'report.pdf'}", (out / "report.html").resolve().as_uri()],
                       check=True, capture_output=True)
    print(json.dumps(summary, indent=1, default=str, ensure_ascii=False))


if __name__ == "__main__":
    main()
