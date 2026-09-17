import { escapeHtml, layout, analyticsClick } from "./templates.js";
import {
  crumbs, breadcrumbLd, faqLd, yearCells, yearCell, yearPageYears,
  depreciationFit,
} from "./seo-pages.js";
import {
  t, href, labelL, fmtEurL, fmtKmL, fmtNumL, fmtDateL, monthTagL,
} from "./i18n.js";

const YEAR_WINDOW_MONTHS = 6;
const GBM_THIRD = 1 / 3;
const MAX_INSIGHTS = 5;
const FEED_MODEL_CHIPS = 8;

const pctInt = x => Math.round(Math.abs(x) * 100);
const signedPct = x => `${x > 0 ? "+" : x < 0 ? "-" : ""}${pctInt(x)}`;

function modelName(rec) {
  return `${escapeHtml(rec.b)} ${escapeHtml(rec.m)}`;
}

function day(builtAt) {
  return (builtAt || "").slice(0, 10);
}

function vars(loc, rec, extra = {}) {
  return {
    brand: escapeHtml(rec.b), model: escapeHtml(rec.m),
    country: loc.countryName, source: loc.source.name,
    ...extra,
  };
}

export function intlProvenance(loc, { n, builtAt, measure = null, unit = null,
                                      measureId = "asking-price-median", source = null }) {
  const d = day(builtAt);
  const src = source || loc.source.name;
  const un = unit || t(loc, "common.unit_active");
  const sample = n != null ? `${fmtNumL(loc, n)} ${un}` : t(loc, "common.na");
  const line = t(loc, "common.provenance", {
    n: sample, date: d || t(loc, "common.na"),
    measure: measure || t(loc, "common.measure"), source: src,
  });
  return `<p class="mono fc-prov" data-sample="${n != null ? n : ""}" data-updated="${escapeHtml(d)}"`
    + ` data-measure="${escapeHtml(measureId)}" data-source="${escapeHtml(src)}">${line}</p>`;
}

export function intlInsights(loc, rec, stats) {
  const out = [];
  const fit = depreciationFit(rec);
  if (fit && fit.cells.length >= 5 && stats.depMed && fit.rate > 0 && fit.rate < 0.30) {
    const mine = pctInt(fit.rate), mkt = pctInt(stats.depMed);
    const rel = (fit.rate - stats.depMed) / stats.depMed;
    const key = rel > 0.18 ? "ins.dep_fast" : rel < -0.18 ? "ins.dep_slow" : "ins.dep_mid";
    out.push(t(loc, key, { mine, mkt }));
  }
  if (rec.fm > 0 && rec.fl != null && rec.fh != null && stats.spreadMed) {
    const mine = (rec.fh - rec.fl) / rec.fm;
    const sp = pctInt(mine), mkt = pctInt(stats.spreadMed);
    if (mine >= stats.spreadMed * 1.25) out.push(t(loc, "ins.spread_wide", { sp, mkt }));
    else if (mine <= stats.spreadMed * 0.75) out.push(t(loc, "ins.spread_tight", { sp, mkt }));
  }
  if (rec.kmm != null && stats.kmMed) {
    const km = fmtKmL(loc, rec.kmm), mkt = fmtKmL(loc, stats.kmMed);
    if (rec.kmm >= stats.kmMed * 1.2) out.push(t(loc, "ins.km_high", { km, mkt }));
    else if (rec.kmm <= stats.kmMed * 0.8) out.push(t(loc, "ins.km_low", { km, mkt }));
  }
  if (Array.isArray(rec.fu) && rec.fu.length) {
    const [fuel, share] = rec.fu[0];
    if (share >= 0.85) {
      out.push(t(loc, "ins.fuel_mono", { fuel: escapeHtml(labelL(loc, fuel)), share: pctInt(share) }));
    }
  }
  const step = biggestStep(rec);
  if (step) {
    out.push(t(loc, "ins.step", {
      lo: step.lo, hi: step.hi, pct: pctInt(step.gap),
      plo: fmtEurL(loc, step.plo), phi: fmtEurL(loc, step.phi),
    }));
  }
  if (rec.gm > 0 && rec.fm > 0 && stats.gapMed != null) {
    const gap = rec.fm / rec.gm - 1;
    const fair = fmtEurL(loc, rec.gm);
    if (gap > stats.gapMed + 0.08) out.push(t(loc, "ins.gap_over", { fair }));
    else if (gap < stats.gapMed - 0.08) out.push(t(loc, "ins.gap_under", { fair }));
  }
  return out.slice(0, MAX_INSIGHTS);
}

function biggestStep(rec) {
  const cs = yearCells(rec, 5).slice().sort((a, b) => a.y - b.y);
  let best = null;
  for (let i = 1; i < cs.length; i++) {
    const a = cs[i - 1], b = cs[i];
    if (b.y - a.y !== 1 || !(a.fm > 0) || !(b.fm > 0)) continue;
    const gap = (b.fm - a.fm) / a.fm;
    if (gap <= 0.06) continue;
    if (!best || gap > best.gap) best = { lo: a.y, hi: b.y, gap, plo: a.fm, phi: b.fm };
  }
  return best;
}

function intlPriceChart(loc, cells, { w = 640, h = 220, color = "#177A47" } = {}) {
  const pts = cells.slice().sort((a, b) => a.y - b.y);
  if (pts.length < 2) return "";
  const padL = 16, padR = 14, padT = 20, padB = 30;
  const xs = pts.map(p => p.y);
  const ys = pts.map(p => p.fm);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const y1 = Math.max(...ys);
  const X = y => padL + ((y - x0) / Math.max(1, x1 - x0)) * (w - padL - padR);
  const Y = v => padT + (1 - v / Math.max(1, y1)) * (h - padT - padB);
  const line = pts.map((p, i) => `${i ? "L" : "M"}${X(p.y).toFixed(1)},${Y(p.fm).toFixed(1)}`).join("");
  const area = `${line}L${X(x1).toFixed(1)},${Y(0).toFixed(1)}L${X(x0).toFixed(1)},${Y(0).toFixed(1)}Z`;
  const unit = t(loc, "common.listings");
  const dots = pts.map(p =>
    `<circle cx="${X(p.y).toFixed(1)}" cy="${Y(p.fm).toFixed(1)}" r="3" fill="${color}">`
    + `<title>${p.y}: ${escapeHtml(fmtEurL(loc, p.fm))} (${fmtNumL(loc, p.n)} ${escapeHtml(unit)})</title></circle>`).join("");
  const step = Math.max(1, Math.ceil(pts.length / 6));
  const xlab = pts.filter((_, i) => i % step === 0 || i === pts.length - 1).map(p => {
    const anchor = p.y === x0 ? "start" : p.y === x1 ? "end" : "middle";
    return `<text x="${X(p.y).toFixed(1)}" y="${h - 9}" text-anchor="${anchor}" class="c-ax">${p.y}</text>`;
  }).join("");
  const ticks = [0, 0.5, 1].map(f => {
    const v = y1 * f;
    return `<line x1="${padL}" x2="${w - padR}" y1="${Y(v).toFixed(1)}" y2="${Y(v).toFixed(1)}" class="c-grid"/>`
      + `<text x="${padL + 2}" y="${(Y(v) - 5).toFixed(1)}" text-anchor="start" class="c-ax">${escapeHtml(fmtEurL(loc, Math.round(v)))}</text>`;
  }).join("");
  return `<svg class="fc-chart" viewBox="0 0 ${w} ${h}" role="img"`
    + ` aria-label="${escapeHtml(t(loc, "model.cap_median"))}">${ticks}`
    + `<path d="${area}" fill="${color}" opacity="0.10"/>`
    + `<path d="${line}" fill="none" stroke="${color}" stroke-width="2.2" stroke-linejoin="round"/>`
    + `${dots}${xlab}</svg>`;
}

function gauge(loc, { lo, hi, at, label }) {
  if (!(lo > 0) || !(hi > 0) || !(at > 0)) return "";
  const span = Math.max(1, hi - lo);
  const pos = Math.min(100, Math.max(0, ((at - lo) / span) * 100));
  return `<div style="margin-top:14px;">
    <div class="gauge-head"><span>${escapeHtml(fmtEurL(loc, lo))}</span><span>${escapeHtml(label)}</span><span>${escapeHtml(fmtEurL(loc, hi))}</span></div>
    <div class="gauge-track"><span class="gauge-pin" style="left:${pos.toFixed(1)}%;"></span></div>
  </div>`;
}

function statBlock(items) {
  return `<div class="fc-stat-row">${items.map(it =>
    `<div class="fc-stat"><div class="k">${escapeHtml(it.k)}</div><div class="v">${it.v}</div>`
    + `${it.s ? `<div class="s">${it.s}</div>` : ""}</div>`).join("")}</div>`;
}

function eyebrow(text) {
  return `<div class="eyebrow"><span class="e-dot"></span><span class="mono">${escapeHtml(text)}</span></div>`;
}

function homeCrumb(loc) {
  return { name: t(loc, "common.crumb_home"), href: href(loc, "landing") };
}

function hubCrumb(loc) {
  return { name: t(loc, "common.crumb_hub"), href: href(loc, "hub") };
}

function orgLd(loc, host, description) {
  const origin = `https://${host}`;
  return {
    "@type": "Organization",
    "@id": `${origin}${href(loc, "landing")}#org`,
    "name": "Carsbuyer",
    "url": `${origin}${href(loc, "landing")}`,
    "description": description,
    "areaServed": loc.country,
  };
}

function graph(nodes) {
  return { "@context": "https://schema.org", "@graph": nodes.filter(Boolean) };
}

export function renderIntlInfo({ loc, host, title, message, cta = true }) {
  const body = `<div class="info">
      <div class="ic">🚗</div>
      <h1>${escapeHtml(title)}</h1>
      <p>${escapeHtml(message)}</p>
      ${cta ? `<a class="btn-dark" href="${href(loc, "hub")}">${t(loc, "info.button")}</a>` : ""}
    </div>`;
  return layout({ title, body, zone: "all", nav: null, depositCount: null, locale: loc, host });
}

export function renderIntlNotFound({ loc, host, path = "", suggestions = [] }) {
  const chips = suggestions.slice(0, 12).map(s =>
    `<a class="mchip" href="${href(loc, "model", s.slug)}">${escapeHtml(s.m)} <span class="mut">${escapeHtml(fmtEurL(loc, s.fm))}</span></a>`).join("");
  const text = path
    ? t(loc, "nf.text", { path: escapeHtml(path) })
    : t(loc, "nf.text_nopath");
  const body = `<div class="fc-404">
      <div class="eyebrow" style="justify-content:center;margin-bottom:16px;"><span class="e-dot"></span><span class="mono">${t(loc, "nf.eyebrow")}</span></div>
      <h1 class="fc-h1">${t(loc, "nf.title")}</h1>
      <p class="fc-p">${text}</p>
      <div class="hero-actions" style="justify-content:center;margin:22px 0 30px;">
        <a class="btn-dark" href="${href(loc, "hub")}">${t(loc, "nf.hub")}</a>
        <a class="chip" href="${href(loc, "avaliar")}">${t(loc, "nf.avaliar")}</a>
        <a class="chip" href="${href(loc, "mercado")}">${t(loc, "nf.mercado")}</a>
      </div>
      ${chips ? `<div class="sec-label" style="text-align:left;">${t(loc, "nf.popular")}</div><div class="mchips" style="justify-content:center;">${chips}</div>` : ""}
    </div>`;
  return layout({
    title: t(loc, "nf.title_meta"),
    description: t(loc, "nf.desc", { country: loc.countryName }),
    body, zone: "all", nav: null, depositCount: null, index: false, host, locale: loc,
  });
}

export function renderIntlLanding({ loc, host, stats, builtAt }) {
  const s = stats || { models: 0, listings: 0, priceMed: null, kmMed: null };
  const steps = [
    { n: "01", t: t(loc, "landing.step1_t", { source: loc.source.name }),
      d: t(loc, "landing.step1_d", { source: loc.source.name, country: loc.countryName }) },
    { n: "02", t: t(loc, "landing.step2_t"), d: t(loc, "landing.step2_d") },
    { n: "03", t: t(loc, "landing.step3_t"), d: t(loc, "landing.step3_d") },
  ];
  const body = `
    <section class="hero">
      <div class="hero-grid">
        <div class="hero-copy">
          ${eyebrow(t(loc, "landing.eyebrow", { source: loc.source.name, models: fmtNumL(loc, s.models) }))}
          <h1 class="hero-title">${t(loc, "landing.h1")}</h1>
          <p class="lede">${t(loc, "landing.lede", { source: loc.source.name })}</p>
          <div class="hero-actions">
            <a class="btn-dark" href="${href(loc, "hub")}">${t(loc, "landing.cta_hub")}</a>
            <a class="btn-outline" href="${href(loc, "avaliar")}" style="font-size:15px;padding:14px 22px;">${t(loc, "landing.cta_avaliar")}</a>
          </div>
          <div class="note" style="margin-top:10px;">${t(loc, "landing.note")}</div>
          <div class="hero-stats">
            <div><div class="stat-num">${fmtNumL(loc, s.models)}</div><div class="stat-cap">${t(loc, "landing.stat_models")}</div></div>
            <div class="stat-div"></div>
            <div><div class="stat-num green">${fmtNumL(loc, s.listings)}</div><div class="stat-cap">${t(loc, "landing.stat_listings")}</div></div>
            <div class="stat-div"></div>
            <div><div class="stat-num">${escapeHtml(fmtEurL(loc, s.priceMed))}</div><div class="stat-cap">${t(loc, "landing.stat_price")}</div></div>
          </div>
        </div>
      </div>
    </section>

    <section class="section" style="padding:40px 22px 30px;">
      <div class="sec-label">${t(loc, "landing.steps_label")}</div>
      <div class="steps">
        ${steps.map(x => `<div class="step-card"><div class="step-n">${x.n}</div><div class="step-t">${x.t}</div><div class="step-d">${x.d}</div></div>`).join("")}
      </div>
    </section>

    <section class="section" style="padding:8px 22px 0;">
      <div class="cta-banner" style="background:#fff;border:1px solid #E8E6E1;">
        <div style="flex:1 1 360px;">
          <h2 style="color:#16181D;">${t(loc, "landing.cta2_h")}</h2>
          <p style="color:#5B606B;">${t(loc, "landing.cta2_p")}</p>
        </div>
        <a class="btn-dark" href="${href(loc, "avaliar")}" style="font-size:15px;padding:14px 26px;">${t(loc, "landing.cta2_btn")}</a>
      </div>
    </section>

    <section class="section" style="padding:24px 22px 70px;">
      <div class="cta-banner">
        <div style="flex:1 1 360px;">
          <h2>${t(loc, "landing.indep_h")}</h2>
          <p>${t(loc, "landing.indep_p")}</p>
        </div>
        <a class="btn-bright" href="${href(loc, "mercado")}">${t(loc, "landing.indep_btn")}</a>
      </div>
      ${intlProvenance(loc, { n: s.listings || null, builtAt })}
    </section>`;
  const desc = t(loc, "landing.desc", { country: loc.countryName, source: loc.source.name });
  return layout({
    title: t(loc, "landing.title", { country: loc.countryName }),
    description: desc,
    body, zone: "all", nav: "landing", depositCount: null, index: true, host, locale: loc,
    canonical: `https://${host}${href(loc, "landing")}`,
    jsonLd: graph([
      orgLd(loc, host, t(loc, "landing.org_desc", { country: loc.countryName, source: loc.source.name })),
      {
        "@type": "WebSite",
        "url": `https://${host}${href(loc, "landing")}`,
        "name": "Carsbuyer",
        "inLanguage": loc.lang,
      },
    ]),
  });
}

export function renderIntlHub({ loc, host, models, builtAt, stats }) {
  const list = Object.entries(models)
    .map(([slug, r]) => ({ slug, b: r.b, m: r.m, fm: r.fm, n: r.n }))
    .sort((a, b) => (b.n || 0) - (a.n || 0));
  const chips = list.map(x =>
    `<a class="mchip" href="${href(loc, "model", x.slug)}">${escapeHtml(x.b)} ${escapeHtml(x.m)}`
    + ` <span class="mut">${escapeHtml(t(loc, "hub.chip", { price: fmtEurL(loc, x.fm), n: fmtNumL(loc, x.n) }))}</span></a>`).join("");
  const body = crumbs([homeCrumb(loc), { name: t(loc, "common.crumb_hub") }]) + `
    <section class="section fc-wrap" style="padding-top:16px;">
      ${eyebrow(t(loc, "hub.eyebrow", { n: fmtNumL(loc, list.length), source: loc.source.name }))}
      <h1 class="fc-h1">${t(loc, "hub.h1", { country: loc.countryName })}</h1>
      <p class="fc-p">${t(loc, "hub.lede", { source: loc.source.name })}</p>
      <div class="hero-actions" style="margin-bottom:18px;">
        <a class="btn-dark" href="${href(loc, "avaliar")}">${t(loc, "hub.cta_avaliar")}</a>
        <a class="chip" href="${href(loc, "mercado")}">${t(loc, "hub.cta_market")}</a>
      </div>
    </section>
    <section class="section fc-wide" style="padding-top:0;padding-bottom:50px;">
      <div class="mchips">${chips}</div>
      <p class="fc-p" style="margin-top:22px;">${t(loc, "hub.also", { method: href(loc, "metodologia"), market: href(loc, "mercado") })}</p>
      ${intlProvenance(loc, { n: stats ? stats.listings : null, builtAt })}
    </section>`;
  return layout({
    title: t(loc, "hub.title", { country: loc.countryName }),
    description: t(loc, "hub.desc", { country: loc.countryName, source: loc.source.name }),
    body, zone: "all", nav: "precos", depositCount: null, index: true, host, locale: loc,
    canonical: `https://${host}${href(loc, "hub")}`,
    jsonLd: graph([breadcrumbLd(host, [homeCrumb(loc), { name: t(loc, "common.crumb_hub"), href: href(loc, "hub") }])]),
  });
}

function yearRows(loc, rec, pageYears) {
  const cells = (rec.yr || []).filter(c => c.fm != null);
  if (!cells.length) return "";
  const published = new Set(pageYears);
  const rows = cells.map(c => {
    const label = typeof c.y === "number" ? String(c.y) : escapeHtml(String(c.y));
    const link = published.has(c.y)
      ? `<a href="${hrefYear(loc, rec.__slug, c.y)}">${label}</a>`
      : label;
    return `<tr>
      <td>${link}</td>
      <td class="mut">${fmtNumL(loc, c.n)}</td>
      <td>${escapeHtml(fmtEurL(loc, c.fm))}</td>
      <td class="mut">${c.gm != null ? escapeHtml(fmtEurL(loc, c.gm)) : "—"}</td>
      <td class="mut">${escapeHtml(fmtEurL(loc, c.fl))} – ${escapeHtml(fmtEurL(loc, c.fh))}</td>
      <td class="mut">${c.km != null ? escapeHtml(fmtKmL(loc, c.km)) : "—"}</td>
    </tr>`;
  }).join("");
  return `<div class="fc-scroll"><table class="fc-tbl">
    <thead><tr>
      <th>${t(loc, "model.th_year")}</th><th>${t(loc, "model.th_n")}</th>
      <th>${t(loc, "model.th_median")}</th><th>${t(loc, "model.th_fair")}</th>
      <th>${t(loc, "model.th_range")}</th><th>${t(loc, "model.th_km")}</th>
    </tr></thead><tbody>${rows}</tbody></table></div>`;
}

let INTL_WAVE = 0;

export function setIntlWave(n) {
  const v = parseInt(n, 10);
  INTL_WAVE = Number.isFinite(v) && v > 0 ? v : 0;
}

const INTL_WAVE_CACHE = new Map();

function intlWaveSlugs(loc, models, builtAt) {
  if (!INTL_WAVE || !models) return null;
  const key = `${loc.code}:${builtAt || ""}:${Object.keys(models).length}:${INTL_WAVE}`;
  const hit = INTL_WAVE_CACHE.get(key);
  if (hit) return hit;
  const set = new Set(Object.entries(models)
    .sort((a, b) => (b[1].n || 0) - (a[1].n || 0) || (a[0] < b[0] ? -1 : 1))
    .slice(0, INTL_WAVE)
    .map(([slug]) => slug));
  if (INTL_WAVE_CACHE.size > 8) INTL_WAVE_CACHE.clear();
  INTL_WAVE_CACHE.set(key, set);
  return set;
}

export function intlInWave(loc, models, slug, builtAt) {
  const wave = intlWaveSlugs(loc, models, builtAt);
  return !wave || wave.has(slug);
}

export function intlPublishedYears(loc, models, slug, rec, builtAt) {
  if (!intlInWave(loc, models, slug, builtAt)) return [];
  return yearPageYears(rec);
}

function hrefYear(loc, slug, year) {
  return `${href(loc, "model", slug)}/${year}`;
}

export function renderIntlModelPage({ loc, host, models, rec, slug, builtAt, stats, siblings = [], extras = "" }) {
  const pageYears = intlPublishedYears(loc, models, slug, rec, builtAt);
  const withSlug = Object.assign({}, rec, { __slug: slug });
  const canonical = `https://${host}${href(loc, "model", slug)}`;
  const altJson = `${canonical}.json`;
  const month = monthTagL(loc, builtAt);
  const price = fmtEurL(loc, rec.fm);
  const v = vars(loc, rec, {
    n: fmtNumL(loc, rec.n), price, lo: fmtEurL(loc, rec.fl), hi: fmtEurL(loc, rec.fh),
  });
  const insights = intlInsights(loc, rec, stats);
  const gbm = rec.gm > 0 ? gbmBlock(loc, rec, slug) : "";
  const chart = intlPriceChart(loc, yearCells(rec, 5));
  const range = (rec.y0 && rec.y1) ? t(loc, "model.range_from", { range: `${rec.y0}–${rec.y1}` }) : "";
  const sibs = siblings.length
    ? `<div class="sec-label" style="margin-top:34px;">${t(loc, "model.siblings_h", { brand: escapeHtml(rec.b) })}</div>
       <div class="mchips">${siblings.map(s =>
         `<a class="mchip" href="${href(loc, "model", s.slug)}">${escapeHtml(s.m)} <span class="mut">${escapeHtml(fmtEurL(loc, s.fm))}</span></a>`).join("")}
       <a class="mchip" href="${href(loc, "hub")}">${t(loc, "model.all_models")}</a></div>`
    : "";
  const body = crumbs([homeCrumb(loc), hubCrumb(loc), { name: `${rec.b} ${rec.m}` }]) + `
    <section class="section fc-wrap" style="padding-top:16px;">
      ${eyebrow(t(loc, "common.eyebrow", { source: loc.source.name }))}
      <h1 class="fc-h1">${t(loc, "model.h1", v)}</h1>
      <p class="fc-p">${t(loc, "model.lede", v)}</p>
      ${statBlock([
        { k: t(loc, "model.cap_median"), v: escapeHtml(price), s: escapeHtml(t(loc, "model.cap_n", { n: fmtNumL(loc, rec.n) })) },
        { k: t(loc, "model.th_range"), v: `${escapeHtml(fmtEurL(loc, rec.fl))} – ${escapeHtml(fmtEurL(loc, rec.fh))}`, s: "" },
        { k: t(loc, "model.th_km"), v: escapeHtml(fmtKmL(loc, rec.kmm)), s: (rec.y0 && rec.y1) ? `${rec.y0}–${rec.y1}` : "" },
      ])}
      ${gauge(loc, { lo: rec.fl, hi: rec.fh, at: rec.fm, label: t(loc, "model.gauge") })}
      ${gbm}
      ${insights.length ? `<h2 class="fc-h2">${t(loc, "model.insights_h", v)}</h2>
      <ul class="fc-insights">${insights.map(i => `<li>${i}</li>`).join("")}</ul>` : ""}
      <h2 class="fc-h2">${t(loc, "model.table_h", v)}</h2>
      ${chart}
      ${yearRows(loc, withSlug, pageYears)}
      ${rec.yt ? `<p class="fc-p mono" style="font-size:12px;">${t(loc, "model.table_thin", { n: fmtNumL(loc, rec.yt) })}</p>` : ""}
      <p class="fc-p">${t(loc, "model.table_cta", { brand: escapeHtml(rec.b), model: escapeHtml(rec.m), range, avaliar: href(loc, "avaliar") })}</p>
      <h2 class="fc-h2">${t(loc, "model.bridge_h")}</h2>
      <p class="fc-p">${t(loc, "model.bridge_p", v)}</p>
      <div class="hero-actions"><a class="btn-dark" href="${href(loc, "avaliar")}">${t(loc, "model.bridge_btn")}</a></div>
      <p class="fc-p" style="margin-top:26px;">${t(loc, "model.trust", { n: fmtNumL(loc, rec.n), source: loc.source.name, method: href(loc, "metodologia"), avaliar: href(loc, "avaliar") })}</p>
      ${sibs}
      ${intlProvenance(loc, { n: rec.n, builtAt })}
    </section>` + extras;
  const faq = modelFaq(loc, rec, stats, range);
  const title = month
    ? t(loc, "model.title_m", { brand: escapeHtml(rec.b), model: escapeHtml(rec.m), price, month, n: fmtNumL(loc, rec.n) })
    : t(loc, "model.title", { brand: escapeHtml(rec.b), model: escapeHtml(rec.m), price, n: fmtNumL(loc, rec.n) });
  return layout({
    title,
    description: t(loc, "model.desc", v),
    body, zone: "all", nav: "precos", depositCount: null, index: true, host, locale: loc,
    canonical, altJson,
    jsonLd: graph([
      breadcrumbLd(host, [homeCrumb(loc), hubCrumb(loc), { name: `${rec.b} ${rec.m}`, href: href(loc, "model", slug) }]),
      faqLd(faq),
      {
        "@type": "Dataset",
        "name": t(loc, "model.ds_name", v),
        "description": t(loc, "model.ds_desc", v),
        "url": canonical,
        "inLanguage": loc.lang,
        "license": "https://creativecommons.org/licenses/by/4.0/",
        "variableMeasured": [
          { "@type": "PropertyValue", "name": t(loc, "model.var_asking"), "value": rec.fm },
          ...(rec.gm > 0 ? [{ "@type": "PropertyValue", "name": t(loc, "model.var_fair"), "value": rec.gm }] : []),
        ],
      },
    ]),
  });
}

function gbmBlock(loc, rec, slug) {
  const band = `${fmtEurL(loc, rec.gl)} – ${fmtEurL(loc, rec.gh)}`;
  const span = Math.max(1, rec.gh - rec.gl);
  const at = (rec.fm - rec.gl) / span;
  const key = at > 1 - GBM_THIRD ? "model.gbm_hi" : at < GBM_THIRD ? "model.gbm_lo" : "model.gbm_mid";
  return `<div class="fc-out" style="margin-top:18px;">
    ${eyebrow(t(loc, "model.gbm_eyebrow"))}
    ${statBlock([
      { k: t(loc, "model.gbm_cap"), v: escapeHtml(fmtEurL(loc, rec.gm)), s: "" },
      { k: t(loc, "model.gbm_range"), v: escapeHtml(band), s: "" },
    ])}
    <p class="fc-p" style="margin-top:12px;">${t(loc, key, { price: fmtEurL(loc, rec.fm), band: escapeHtml(band) })}</p>
    <p class="fc-p mono" style="font-size:12px;">${t(loc, "model.gbm_note", { brand: escapeHtml(rec.b), model: escapeHtml(rec.m), avaliar: href(loc, "avaliar") })}</p>
  </div>`;
}

function modelFaq(loc, rec, stats, range) {
  const v = vars(loc, rec, {
    n: fmtNumL(loc, rec.n), price: fmtEurL(loc, rec.fm),
    lo: fmtEurL(loc, rec.fl), hi: fmtEurL(loc, rec.fh),
  });
  const pairs = [
    [t(loc, "model.faq1_q", v), t(loc, "model.faq1_a", v)],
    [t(loc, "model.faq2_q", v), t(loc, "model.faq2_a", v)],
    [t(loc, "model.faq3_q", v), t(loc, "model.faq3_a", Object.assign({}, v, {
      range: (rec.y0 && rec.y1) ? t(loc, "model.faq3_range", { range: `${rec.y0}–${rec.y1}` }) : "",
    }))],
  ];
  if (rec.kmm != null) {
    pairs.push([t(loc, "model.faq4_q", v), t(loc, "model.faq4_a", Object.assign({}, v, {
      km: fmtKmL(loc, rec.kmm), range,
    }))]);
  }
  if (rec.gm > 0) {
    pairs.push([t(loc, "model.faq5_q", v), t(loc, "model.faq5_a", Object.assign({}, v, {
      fair: fmtEurL(loc, rec.gm), lo: fmtEurL(loc, rec.gl), hi: fmtEurL(loc, rec.gh),
    }))]);
  }
  return pairs.map(([q, a]) => [stripTags(q), stripTags(a)]);
}

function stripTags(s) {
  return String(s).replace(/<[^>]+>/g, "").replace(/&nbsp;/g, " ");
}

export function intlModelJson(loc, rec, slug, { host, builtAt, models = null }) {
  const base = `https://${host}`;
  const published = new Set(intlPublishedYears(loc, models, slug, rec, builtAt));
  return {
    source: "Carsbuyer",
    source_url: `${base}${href(loc, "model", slug)}`,
    licence: t(loc, "json.licence"),
    measured: "asking_price",
    measured_note: t(loc, "json.measured_note", { source: loc.source.name }),
    collected_until: day(builtAt) || null,
    updated_at: builtAt || null,
    market: loc.country, currency: "EUR", language: loc.code,
    brand: rec.b, model: rec.m, slug,
    sample_size: rec.n,
    asking_price: { median: rec.fm, p25: rec.fl, p75: rec.fh },
    fair_value_estimate: rec.gm != null
      ? { median: rec.gm, low: rec.gl, high: rec.gh, note: t(loc, "json.gbm_note") }
      : null,
    mileage_km_median: rec.kmm != null ? rec.kmm : null,
    model_years: (rec.y0 && rec.y1) ? { from: rec.y0, to: rec.y1 } : null,
    fuel_mix: Array.isArray(rec.fu)
      ? rec.fu.map(([f, share]) => ({ fuel: labelL(loc, f), share })) : null,
    days_to_sell: rec.sd != null ? { median_days: rec.sd, sample_size: rec.sn } : null,
    by_year: (rec.yr || []).map(c => ({
      year: c.y, sample_size: c.n,
      asking_price: { median: c.fm, p25: c.fl, p75: c.fh },
      fair_value_estimate: c.gm != null ? { median: c.gm, low: c.gl, high: c.gh } : null,
      mileage_km_median: c.km != null ? c.km : null,
      page: published.has(c.y) ? `${base}${hrefYear(loc, slug, c.y)}` : null,
      page_absent_because: published.has(c.y) ? null
        : typeof c.y !== "number" ? "merged_band"
        : (c.n || 0) < 10 ? "below_year_floor"
        : "outside_publication_wave",
    })),
    years_omitted_thin_sample: rec.yt || 0,
    page_coverage_note: t(loc, "json.coverage_note", { min: 10 }),
    related: {
      models_index: `${base}${href(loc, "hub")}`,
      methodology: `${base}${href(loc, "metodologia")}`,
      valuation_tool: `${base}${href(loc, "avaliar")}`,
    },
    models_in_market: models ? Object.keys(models).length : null,
  };
}

export function intlYearJson(loc, rec, slug, year, cell, { host, builtAt }) {
  const base = `https://${host}`;
  const windowed = cell.w === 1;
  return {
    source: "Carsbuyer",
    source_url: `${base}${hrefYear(loc, slug, year)}`,
    licence: t(loc, "json.licence"),
    measured: "asking_price",
    measured_note: windowed
      ? t(loc, "json.measured_window", { source: loc.source.name, days: YEAR_WINDOW_MONTHS * 30 })
      : t(loc, "json.measured_note", { source: loc.source.name }),
    collected_until: day(builtAt) || null,
    updated_at: builtAt || null,
    market: loc.country, currency: "EUR", language: loc.code,
    brand: rec.b, model: rec.m, slug, model_year: year,
    sample_size: cell.n,
    asking_price: { median: cell.fm, p25: cell.fl, p75: cell.fh },
    fair_value_estimate: cell.gm != null
      ? { median: cell.gm, low: cell.gl, high: cell.gh, note: t(loc, "json.gbm_note") }
      : null,
    mileage_km_median: cell.km != null ? cell.km : null,
    related: {
      model: `${base}${href(loc, "model", slug)}`,
      models_index: `${base}${href(loc, "hub")}`,
      methodology: `${base}${href(loc, "metodologia")}`,
    },
  };
}

export function renderIntlYearPage({ loc, host, models, rec, slug, year, cell, stats, builtAt }) {
  const all = yearCells(rec, 1).slice().sort((a, b) => a.y - b.y);
  const idx = all.findIndex(c => c.y === year);
  const older = idx > 0 ? all[idx - 1] : null;
  const newer = (idx >= 0 && idx < all.length - 1) ? all[idx + 1] : null;
  const win = all.slice(Math.max(0, idx - 3), idx + 4).slice().sort((a, b) => b.y - a.y);
  const pageYears = new Set(intlPublishedYears(loc, models, slug, rec, builtAt));
  const canonical = `https://${host}${hrefYear(loc, slug, year)}`;
  const altJson = `${canonical}.json`;
  const windowed = cell.w === 1;
  const sample = windowed
    ? t(loc, "year.sample_window", { months: YEAR_WINDOW_MONTHS })
    : t(loc, "year.sample_active");
  const price = fmtEurL(loc, cell.fm);
  const v = vars(loc, rec, {
    year, n: fmtNumL(loc, cell.n), sample, price,
    lo: fmtEurL(loc, cell.fl), hi: fmtEurL(loc, cell.fh),
  });
  const km = cell.km != null ? t(loc, "year.lede_km", { km: fmtKmL(loc, cell.km) }) : "";
  const stepBlock = yearStep(loc, rec, slug, year, cell, older, newer, pageYears);
  const share = rec.n > 0
    ? t(loc, "year.step_share", {
        year, share: pctInt(cell.n / rec.n), brand: escapeHtml(rec.b), model: escapeHtml(rec.m),
        n: fmtNumL(loc, cell.n), total: fmtNumL(loc, rec.n),
        tail: cell.n / rec.n >= 0.15 ? t(loc, "year.share_many")
             : cell.n / rec.n <= 0.04 ? t(loc, "year.share_few") : "",
      })
    : "";
  const rows = win.map(c => `<tr>
      <td>${pageYears.has(c.y) ? `<a href="${hrefYear(loc, slug, c.y)}">${c.y}</a>` : c.y}${c.y === year ? " ←" : ""}</td>
      <td class="mut">${fmtNumL(loc, c.n)}</td>
      <td>${escapeHtml(fmtEurL(loc, c.fm))}</td>
      <td class="mut">${escapeHtml(fmtEurL(loc, c.fl))} – ${escapeHtml(fmtEurL(loc, c.fh))}</td>
      <td class="mut">${c.km != null ? escapeHtml(fmtKmL(loc, c.km)) : "—"}</td>
    </tr>`).join("");
  const navYears = [...pageYears].sort((a, b) => b - a).map(y =>
    `<a href="${hrefYear(loc, slug, y)}"${y === year ? ' class="on"' : ""}>${y}</a>`).join("");
  const body = crumbs([
    homeCrumb(loc), hubCrumb(loc),
    { name: `${rec.b} ${rec.m}`, href: href(loc, "model", slug) },
    { name: String(year) },
  ]) + `
    <section class="section fc-wrap" style="padding-top:16px;">
      ${eyebrow(t(loc, "year.eyebrow", { brand: rec.b, model: rec.m, year, source: loc.source.name }))}
      <h1 class="fc-h1">${t(loc, "year.h1", v)}</h1>
      <p class="fc-p">${t(loc, "year.lede", Object.assign({}, v, { active: "", km }))}</p>
      ${statBlock([
        { k: t(loc, "year.cap_median", { year }) + (windowed ? t(loc, "year.cap_window", { months: YEAR_WINDOW_MONTHS }) : ""),
          v: escapeHtml(price), s: escapeHtml(t(loc, "year.cap_n", { n: fmtNumL(loc, cell.n) })) },
        { k: t(loc, "year.th_range"), v: `${escapeHtml(fmtEurL(loc, cell.fl))} – ${escapeHtml(fmtEurL(loc, cell.fh))}`, s: "" },
        ...(cell.gm != null ? [{ k: t(loc, "year.gbm_cap", { year }), v: escapeHtml(fmtEurL(loc, cell.gm)),
          s: escapeHtml(t(loc, "year.gbm_range", { lo: fmtEurL(loc, cell.gl), hi: fmtEurL(loc, cell.gh) })) }] : []),
      ])}
      ${gauge(loc, { lo: cell.fl, hi: cell.fh, at: cell.fm, label: t(loc, "year.gauge", { year }) })}
      ${windowed ? `<p class="fc-p mono" style="font-size:12px;">${t(loc, "year.sample_note")}</p>` : ""}
      ${stepBlock ? `<h2 class="fc-h2">${t(loc, "year.step_h")}</h2>${stepBlock}` : ""}
      ${share ? `<p class="fc-p">${share}</p>` : ""}
      <h2 class="fc-h2">${t(loc, "year.table_h", { brand: escapeHtml(rec.b), model: escapeHtml(rec.m) })}</h2>
      <div class="fc-scroll"><table class="fc-tbl">
        <thead><tr><th>${t(loc, "model.th_year")}</th><th>${t(loc, "model.th_n")}</th><th>${t(loc, "model.th_median")}</th><th>${t(loc, "year.th_range")}</th><th>${t(loc, "model.th_km")}</th></tr></thead>
        <tbody>${rows}</tbody></table></div>
      <p class="fc-p"><a href="${href(loc, "model", slug)}">${t(loc, "year.all_years", { brand: escapeHtml(rec.b), model: escapeHtml(rec.m) })}</a></p>
      ${navYears ? `<div class="sec-label" style="margin-top:26px;">${t(loc, "year.nav_label")}</div><div class="fc-yearlinks">${navYears}</div>` : ""}
      <div class="cta-banner" style="margin-top:34px;background:#fff;border:1px solid #E8E6E1;">
        <div style="flex:1 1 360px;">
          <h2 style="color:#16181D;">${t(loc, "year.cta_h", { brand: escapeHtml(rec.b), model: escapeHtml(rec.m), year })}</h2>
          <p style="color:#5B606B;">${t(loc, "year.cta_p", v)}</p>
        </div>
        <a class="btn-dark" href="${href(loc, "avaliar")}">${t(loc, "year.cta_btn", { year })}</a>
      </div>
      <p class="fc-p" style="margin-top:22px;">${t(loc, "year.links", { model_url: href(loc, "model", slug), method: href(loc, "metodologia"), hub: href(loc, "hub"), brand: escapeHtml(rec.b), model: escapeHtml(rec.m) })}</p>
      ${intlProvenance(loc, {
        n: cell.n, builtAt,
        measure: t(loc, "year.measure", { brand: rec.b, model: rec.m, year }),
        unit: windowed ? t(loc, "year.unit_window") : t(loc, "common.unit_active"),
      })}
    </section>`;
  const month = monthTagL(loc, builtAt);
  const title = month
    ? t(loc, "year.title_m", { brand: escapeHtml(rec.b), model: escapeHtml(rec.m), year, price, month, n: fmtNumL(loc, cell.n) })
    : t(loc, "year.title", { brand: escapeHtml(rec.b), model: escapeHtml(rec.m), year, price, n: fmtNumL(loc, cell.n) });
  return layout({
    title,
    description: t(loc, "year.desc", Object.assign({}, v, {
      km: cell.km != null ? t(loc, "year.desc_km", { km: fmtKmL(loc, cell.km) }) : "",
      date: fmtDateL(loc, builtAt) || day(builtAt),
    })),
    body, zone: "all", nav: "precos", depositCount: null, index: true, host, locale: loc,
    canonical, altJson,
    jsonLd: graph([
      breadcrumbLd(host, [
        homeCrumb(loc), hubCrumb(loc),
        { name: `${rec.b} ${rec.m}`, href: href(loc, "model", slug) },
        { name: String(year), href: hrefYear(loc, slug, year) },
      ]),
      faqLd(yearFaq(loc, rec, year, cell, newer, windowed)),
      {
        "@type": "Dataset",
        "name": t(loc, "year.ds_name", v),
        "description": t(loc, "year.ds_desc", v),
        "url": canonical,
        "inLanguage": loc.lang,
        "license": "https://creativecommons.org/licenses/by/4.0/",
      },
    ]),
  });
}

function yearStep(loc, rec, slug, year, cell, older, newer, pageYears) {
  const out = [];
  const linkFor = y => pageYears.has(y) ? hrefYear(loc, slug, y) : href(loc, "model", slug);
  if (newer && newer.fm > 0 && cell.fm > 0) {
    const d = (newer.fm - cell.fm) / cell.fm;
    const kmTail = (newer.km != null && cell.km != null)
      ? (newer.km < cell.km
          ? t(loc, "year.step_km_less", { km: fmtKmL(loc, cell.km - newer.km) })
          : t(loc, "year.step_km_more", { km: fmtKmL(loc, newer.km - cell.km) }))
      : "";
    const overlap = newer.fl != null && cell.fh != null && newer.fl < cell.fh;
    if (overlap) {
      out.push(t(loc, "year.step_newer_overlap", {
        href: linkFor(newer.y), brand: escapeHtml(rec.b), model: escapeHtml(rec.m), ny: newer.y,
        nprice: fmtEurL(loc, newer.fm), price: fmtEurL(loc, cell.fm), year,
        rng: `${fmtEurL(loc, cell.fl)}–${fmtEurL(loc, cell.fh)}`,
        nrng: `${fmtEurL(loc, newer.fl)}–${fmtEurL(loc, newer.fh)}`,
        km: kmTail, nn: fmtNumL(loc, newer.n),
      }));
    } else {
      out.push(t(loc, "year.step_newer_sep", {
        href: linkFor(newer.y), brand: escapeHtml(rec.b), model: escapeHtml(rec.m), ny: newer.y,
        nprice: fmtEurL(loc, newer.fm), pct: signedPct(d), year, km: kmTail,
        flat: d <= 0 ? t(loc, "year.step_newer_flat") : "",
      }));
    }
  }
  if (older && older.fm > 0 && cell.fm > 0) {
    const save = (cell.fm - older.fm) / cell.fm;
    const kmTail = (older.km != null && cell.km != null)
      ? (older.km < cell.km
          ? t(loc, "year.step_km_less", { km: fmtKmL(loc, cell.km - older.km) })
          : t(loc, "year.step_km_more", { km: fmtKmL(loc, older.km - cell.km) }))
      : "";
    if (save <= 0.01) {
      out.push(t(loc, "year.step_older_none", {
        href: linkFor(older.y), oy: older.y, oprice: fmtEurL(loc, older.fm), km: kmTail,
        same: save >= 0 ? t(loc, "year.step_older_same", { year }) : t(loc, "year.step_older_above", { year }),
      }));
    } else if (older.fh != null && cell.fl != null && older.fh > cell.fl) {
      out.push(t(loc, "year.step_older_overlap", {
        href: linkFor(older.y), oy: older.y, save: pctInt(save), oprice: fmtEurL(loc, older.fm),
        km: kmTail, year,
        orng: `${fmtEurL(loc, older.fl)}–${fmtEurL(loc, older.fh)}`,
        rng: `${fmtEurL(loc, cell.fl)}–${fmtEurL(loc, cell.fh)}`,
      }));
    } else {
      out.push(t(loc, "year.step_older_sep", {
        href: linkFor(older.y), oy: older.y, save: pctInt(save),
        oprice: fmtEurL(loc, older.fm), km: kmTail,
      }));
    }
  }
  const fit = depreciationFit(rec);
  if (fit && fit.cells.length >= 5 && fit.rate > 0 && fit.rate < 0.30) {
    out.push(t(loc, "year.step_fit", {
      brand: escapeHtml(rec.b), model: escapeHtml(rec.m), rate: pctInt(fit.rate),
      series: t(loc, "year.step_series", { n: fmtNumL(loc, fit.cells.length), from: fit.oldest.y, to: fit.newest.y }),
      vs: newer && newer.fm > 0 && cell.fm > 0
        ? t(loc, "year.step_fit_vs", { pct: signedPct((newer.fm - cell.fm) / cell.fm), year, ny: newer.y })
        : "",
    }));
  }
  return out.map(p => `<p class="fc-p">${p}</p>`).join("");
}

function yearFaq(loc, rec, year, cell, newer, windowed) {
  const sample = windowed
    ? t(loc, "year.sample_window", { months: YEAR_WINDOW_MONTHS })
    : t(loc, "year.sample_active");
  const v = vars(loc, rec, {
    year, n: fmtNumL(loc, cell.n), sample, price: fmtEurL(loc, cell.fm),
    lo: fmtEurL(loc, cell.fl), hi: fmtEurL(loc, cell.fh),
  });
  const pairs = [[
    t(loc, "year.faq1_q", v),
    t(loc, "year.faq1_a", Object.assign({}, v, { note: windowed ? t(loc, "year.sample_note") : "" })),
  ]];
  if (cell.km != null) {
    pairs.push([t(loc, "year.faq2_q", v),
                t(loc, "year.faq2_a", Object.assign({}, v, { km: fmtKmL(loc, cell.km) }))]);
  }
  if (newer && newer.fm > 0 && cell.fm > 0) {
    const overlap = newer.fl != null && cell.fh != null && newer.fl < cell.fh;
    const q = t(loc, "year.faq3_q", Object.assign({}, v, { ny: newer.y }));
    const a = overlap
      ? t(loc, "year.faq3_a_overlap", Object.assign({}, v, {
          ny: newer.y, nprice: fmtEurL(loc, newer.fm), nn: fmtNumL(loc, newer.n),
          nlo: fmtEurL(loc, newer.fl), nhi: fmtEurL(loc, newer.fh), fit: "",
        }))
      : t(loc, "year.faq3_a_sep", Object.assign({}, v, {
          ny: newer.y, nprice: fmtEurL(loc, newer.fm),
          pct: signedPct((newer.fm - cell.fm) / cell.fm),
        }));
    pairs.push([q, a]);
  }
  if (cell.gm != null) {
    pairs.push([t(loc, "year.faq4_q", v), t(loc, "year.faq4_a", Object.assign({}, v, {
      fair: fmtEurL(loc, cell.gm), lo: fmtEurL(loc, cell.gl), hi: fmtEurL(loc, cell.gh),
    }))]);
  }
  return pairs.map(([q, a]) => [stripTags(q), stripTags(a)]);
}

export function renderIntlAvaliar({ loc, host, models, builtAt, stats, rec = null,
                                    carId = null, sourceUrl = null, query = "", spec = null }) {
  const canonical = `https://${host}${href(loc, "avaliar")}`;
  const notFound = Boolean(query) && !rec;
  const form = `<form class="fc-form" method="get" action="${href(loc, "avaliar")}">
      <div class="fc-field" style="flex:1 1 320px;">
        <label for="q">${escapeHtml(loc.source.name)}</label>
        <input id="q" name="q" type="text" value="${escapeHtml(query)}" placeholder="${escapeHtml(t(loc, "av.placeholder", { source: loc.source.name }))}"
          style="font-family:var(--mono);font-size:14px;padding:10px 12px;border:1px solid #E0DDD6;border-radius:10px;background:#fff;color:#16181D;width:100%;">
      </div>
      <button type="submit" class="btn-dark" style="font-size:14px;padding:12px 20px;">${t(loc, "av.btn")}</button>
    </form>`;
  const byBrand = {};
  for (const [slug, r] of Object.entries(models || {})) {
    (byBrand[r.b] = byBrand[r.b] || []).push([slug, r.m]);
  }
  const options = Object.keys(byBrand).sort((a, b) => a.localeCompare(b, loc.collate))
    .map(b => `<optgroup label="${escapeHtml(b)}">`
      + byBrand[b].sort((a, c) => a[1].localeCompare(c[1], loc.collate))
          .map(([slug, m]) => `<option value="${escapeHtml(slug)}"${spec && spec.slug === slug ? " selected" : ""}>${escapeHtml(m)}</option>`).join("")
      + `</optgroup>`).join("");
  const thisYear = new Date().getUTCFullYear();
  const years = [`<option value="">${escapeHtml(t(loc, "av.spec_year"))}</option>`];
  for (let y = thisYear; y >= thisYear - 35; y--) {
    years.push(`<option value="${y}"${spec && spec.year === y ? " selected" : ""}>${y}</option>`);
  }
  const specForm = `<h2 class="fc-h2" id="escolher">${t(loc, "av.spec_h")}</h2>
    <form class="fc-form" method="get" action="${href(loc, "avaliar")}">
      <div class="fc-field">
        <label for="modelo">${t(loc, "av.spec_model")}</label>
        <select id="modelo" name="modelo" required>
          <option value="">${escapeHtml(t(loc, "av.spec_model"))}</option>${options}
        </select>
      </div>
      <div class="fc-field">
        <label for="ano">${t(loc, "av.spec_year")}</label>
        <select id="ano" name="ano">${years.join("")}</select>
      </div>
      <button type="submit" class="btn-dark" style="font-size:14px;padding:12px 20px;">${t(loc, "av.spec_btn")}</button>
    </form>
    <p class="fc-p mono" style="font-size:12px;">${t(loc, "av.spec_note")}</p>`;
  const verdict = rec ? verdictBlock(loc, rec, sourceUrl, carId) : "";
  const specOut = (!rec && spec) ? specBlock(loc, spec) : "";
  const body = crumbs([homeCrumb(loc), { name: t(loc, "av.crumb") }]) + `
    <section class="section fc-wrap" style="padding-top:16px;">
      ${eyebrow(t(loc, "av.eyebrow", { source: loc.source.name }))}
      <h1 class="fc-h1">${t(loc, "av.h1")}</h1>
      <p class="fc-p">${t(loc, "av.lede", { source: loc.source.name })}</p>
      ${form}
      <p class="fc-p mono" style="font-size:12px;">${t(loc, "av.note")}</p>
      ${notFound ? `<div class="fc-out"><p class="fc-p" style="margin:0;">${t(loc, "av.notfound", { source: loc.source.name })}</p></div>` : ""}
      ${verdict}
      ${specOut}
      ${specForm}
      <div class="cta-banner" style="margin-top:30px;background:#fff;border:1px solid #E8E6E1;">
        <div style="flex:1 1 360px;">
          <h2 style="color:#16181D;">${t(loc, "av.cta_h")}</h2>
          <p style="color:#5B606B;">${t(loc, "av.cta_p")}</p>
        </div>
        <a class="btn-dark" href="#escolher">${t(loc, "av.cta_btn")}</a>
      </div>
      ${intlProvenance(loc, { n: stats ? stats.listings : null, builtAt })}
    </section>`;
  return layout({
    title: t(loc, "av.title"),
    description: t(loc, "av.desc", { source: loc.source.name }),
    body, zone: "all", nav: "avaliar", depositCount: null, index: true, host, locale: loc,
    canonical,
    jsonLd: graph([
      breadcrumbLd(host, [homeCrumb(loc), { name: t(loc, "av.crumb"), href: href(loc, "avaliar") }]),
      {
        "@type": "WebApplication",
        "name": t(loc, "av.app_name"),
        "url": canonical,
        "applicationCategory": "FinanceApplication",
        "operatingSystem": "Web",
        "inLanguage": loc.lang,
        "description": t(loc, "av.app_desc", { source: loc.source.name }),
        "offers": { "@type": "Offer", "price": "0", "priceCurrency": "EUR" },
        "featureList": [t(loc, "av.feat1"), t(loc, "av.feat2"), t(loc, "av.feat3")],
      },
      faqLd([
        [stripTags(t(loc, "av.faq1_q", { country: loc.countryName })), stripTags(t(loc, "av.faq1_a", { source: loc.source.name }))],
        [stripTags(t(loc, "av.faq2_q")), stripTags(t(loc, "av.faq2_a"))],
        [stripTags(t(loc, "av.faq3_q")), stripTags(t(loc, "av.faq3_a", { source: loc.source.name }))],
      ]),
    ]),
  });
}

function verdictBlock(loc, rec, sourceUrl, carId, from = "avaliar") {
  const price = rec.p, fair = rec.fm, lo = rec.fl, hi = rec.fh;
  const above = hi != null && price > hi;
  const below = lo != null && price < lo;
  const tag = above ? t(loc, "av.v_above") : below ? t(loc, "av.v_below") : t(loc, "av.v_within");
  const delta = fair != null ? fair - price : null;
  const line = above
    ? t(loc, "av.v_line_above", { amount: fmtEurL(loc, price - (hi != null ? hi : fair)) })
    : below
      ? t(loc, "av.v_line_below", { amount: fmtEurL(loc, (lo != null ? lo : fair) - price) })
      : (delta != null && delta > 0
          ? t(loc, "av.v_line_within_pos", { amount: fmtEurL(loc, delta) })
          : t(loc, "av.v_line_within"));
  const title = rec.t ? escapeHtml(rec.t) : t(loc, "av.v_fallback");
  const meta = [
    rec.y != null ? String(rec.y) : null,
    rec.km != null ? escapeHtml(fmtKmL(loc, rec.km)) : null,
    rec.fu ? escapeHtml(labelL(loc, rec.fu)) : null,
    rec.ct ? escapeHtml(rec.ct) : null,
  ].filter(Boolean).join(" · ");
  const open = sourceUrl || (carId ? `${loc.source.url}/angebote/${encodeURIComponent(carId)}` : null);
  return `<div class="fc-out" style="margin-top:18px;">
    <div class="mono" style="font-size:12px;font-weight:700;color:#177A47;">${tag}</div>
    <h2 class="fc-h2" style="margin-top:8px;">${title}</h2>
    ${meta ? `<div class="mono" style="font-size:12px;color:#8A8F98;">${meta}</div>` : ""}
    ${statBlock([
      { k: t(loc, "av.v_cap_price"), v: escapeHtml(fmtEurL(loc, price)), s: "" },
      { k: t(loc, "av.v_cap_fair"), v: escapeHtml(fmtEurL(loc, fair)), s: escapeHtml(line) },
    ])}
    ${gauge(loc, { lo, hi, at: price, label: t(loc, "av.v_gauge") })}
    ${rec.sd != null ? `<p class="fc-p" style="margin-top:12px;">${t(loc, "av.v_sell", { days: fmtNumL(loc, rec.sd) })}</p>` : ""}
    <div class="hero-actions" style="margin-top:12px;">
      ${open ? `<a class="btn-dark" href="${escapeHtml(open)}" target="_blank" rel="noopener nofollow"${analyticsClick("olx_open", { source: from, market: loc.code, site: loc.source.host })}>${t(loc, "av.v_open")}</a>` : ""}
      ${rec.ms ? `<a class="chip" href="${href(loc, "model", rec.ms)}">${t(loc, "av.v_model")}</a>` : ""}
      <a class="chip" href="${href(loc, "avaliar")}">${t(loc, "av.v_another")}</a>
      <a class="chip" href="${href(loc, "mercado")}">${t(loc, "av.v_market")}</a>
    </div>
    <p class="fc-p mono" style="font-size:12px;margin-top:12px;">${t(loc, "av.v_foot", { country: loc.countryName })}</p>
  </div>`;
}

function specBlock(loc, spec) {
  const { rec, slug, year, cell } = spec;
  const use = cell || rec;
  const caveat = !cell && year
    ? t(loc, "av.res_caveat_year", { year })
    : (cell && cell.w === 1 ? t(loc, "av.res_caveat_window", { months: YEAR_WINDOW_MONTHS }) : "");
  return `<div class="fc-out" style="margin-top:18px;">
    <h2 class="fc-h2" style="margin-top:0;">${escapeHtml(rec.b)} ${escapeHtml(rec.m)}${year ? ` · ${year}` : ""}</h2>
    ${statBlock([
      { k: t(loc, "av.res_cap_median"), v: escapeHtml(fmtEurL(loc, use.fm)), s: escapeHtml(t(loc, "model.cap_n", { n: fmtNumL(loc, use.n) })) },
      { k: t(loc, "av.res_cap_range"), v: `${escapeHtml(fmtEurL(loc, use.fl))} – ${escapeHtml(fmtEurL(loc, use.fh))}`, s: "" },
    ])}
    ${gauge(loc, { lo: use.fl, hi: use.fh, at: use.fm, label: t(loc, "av.res_gauge") })}
    ${caveat ? `<p class="fc-p mono" style="font-size:12px;margin-top:10px;">${caveat}</p>` : ""}
    ${rec.sd != null ? `<p class="fc-p">${t(loc, "av.res_sell", { days: fmtNumL(loc, rec.sd), source: loc.source.name })}</p>` : ""}
    <div class="hero-actions">
      <a class="chip" href="${href(loc, "model", slug)}">${t(loc, "av.res_btn_year")}</a>
      <a class="chip" href="${href(loc, "avaliar")}">${t(loc, "av.res_btn_paste")}</a>
    </div>
    <p class="fc-p mono" style="font-size:12px;">${t(loc, "av.res_foot", { source: loc.source.name })}</p>
  </div>`;
}

export function feedRegions(deals, districts) {
  const counts = new Map();
  for (const d of deals || []) {
    const label = String(d.district || "").trim();
    if (label) counts.set(label, (counts.get(label) || 0) + 1);
  }
  const out = [];
  for (const [slug, rec] of Object.entries(districts || {})) {
    const label = (rec && rec.lbl) || slug;
    const n = counts.get(label) || 0;
    if (n) out.push({ slug, label, n });
  }
  return out.sort((a, b) => b.n - a.n || a.label.localeCompare(b.label));
}

export function regionLabel(districts, slug) {
  const rec = (districts || {})[String(slug || "").toLowerCase()];
  return rec ? ((rec.lbl || slug)) : null;
}

function feedRegionChips(loc, regions, active) {
  if (regions.length < 2) return "";
  const all = `<a class="mchip${active ? "" : " on"}" href="${href(loc, "mercado")}">`
    + `${escapeHtml(t(loc, "feed.region_all"))}</a>`;
  const chips = regions.map(r =>
    `<a class="mchip${active === r.slug ? " on" : ""}" href="${href(loc, "mercado")}?region=${encodeURIComponent(r.slug)}">`
    + `${escapeHtml(r.label)} <span class="mut">${fmtNumL(loc, r.n)}</span></a>`).join("");
  return `<div class="sec-label" style="margin-top:28px;">${t(loc, "feed.region_h")}</div>`
    + `<div class="mchips">${all}${chips}</div>`;
}

export function renderIntlFeed({ loc, host, deals, builtAt, models = null,
                                districts = null, region = null }) {
  const all = (deals || []).slice();
  const label = region ? regionLabel(districts, region) : null;
  const list = label ? all.filter(d => String(d.district || "") === label) : all;
  const tiles = list.map(d => dealTile(loc, d)).join("");
  const chips = feedModelChips(loc, list, models);
  const regionChips = feedRegionChips(loc, feedRegions(all, districts), label ? region : null);
  const sub = list.length === 1
    ? t(loc, "feed.sub_one", { country: loc.countryName })
    : t(loc, "feed.sub", { country: loc.countryName, n: fmtNumL(loc, list.length) });
  const body = crumbs([homeCrumb(loc), { name: t(loc, "feed.crumb") }]) + `
    <section class="feed">
      <div class="feed-head">
        <h1>${t(loc, "feed.h1")}${label ? ` · ${escapeHtml(label)}` : ""}</h1>
        <p>${sub}</p>
      </div>
      ${regionChips}
      ${list.length ? `<div class="grid">${tiles}</div>` : `<div class="info">
        <div class="ic">🚗</div>
        <h1>${t(loc, "feed.empty_title")}</h1>
        <p>${t(loc, "feed.empty_p")}</p>
        <a class="btn-dark" href="${href(loc, "avaliar")}">${t(loc, "nf.avaliar")}</a>
      </div>`}
      ${chips}
      <p class="mono fc-prov">${t(loc, "feed.prov", { date: day(builtAt) || t(loc, "common.na"), source: loc.source.name })}</p>
    </section>`;
  return layout({
    title: t(loc, "feed.title", { country: loc.countryName, source: loc.source.name }),
    description: t(loc, "feed.desc", { source: loc.source.name }),
    body, zone: "all", nav: "feed", depositCount: null,
    index: !label && list.length > 0, host, locale: loc,
    canonical: `https://${host}${href(loc, "mercado")}`,
    jsonLd: graph([breadcrumbLd(host, [homeCrumb(loc), { name: t(loc, "feed.crumb"), href: href(loc, "mercado") }])]),
  });
}

export function renderIntlArchive({ loc, host, weeks = [] }) {
  const path = href(loc, "arquivo");
  const permalink = `https://${host}${path}`;
  const shown = weeks.slice().sort().reverse();
  const latest = shown[0] || null;
  const tok = w => escapeHtml(String(w).toLowerCase());
  const rows = shown.map(w => `<tr>
      <td class="mono">${escapeHtml(w)}</td>
      <td><a href="${path}/${tok(w)}.json" style="color:#177A47;font-weight:600;">${t(loc, "arch.col_full")}</a></td>
      <td class="mut mono">${path}/${tok(w)}/{slug}.json</td></tr>`).join("");
  const crumbItems = [homeCrumb(loc), { name: t(loc, "arch.crumb") }];
  const body = crumbs(crumbItems) + `
    <section class="fc-sec">
      ${eyebrow(t(loc, "arch.eyebrow", { n: fmtNumL(loc, shown.length) }))}
      <h1 class="fc-h1">${t(loc, "arch.h1")}</h1>
      <p class="fc-p">${t(loc, "arch.p1")}</p>
      <p class="fc-p">${t(loc, "arch.p2")}</p>
      <p class="fc-p">${latest
        ? t(loc, "arch.latest", { url: `<span class="mono fc-url">${escapeHtml(permalink)}/${tok(latest)}.json</span>` })
        : t(loc, "arch.empty")}</p>
    </section>
    ${shown.length ? `<section class="fc-sec">
      <h2 class="fc-h2">${t(loc, "arch.weeks_h")}</h2>
      <div class="fc-scroll"><table class="fc-tbl">
        <thead><tr><th>${t(loc, "arch.col_week")}</th><th>${t(loc, "arch.col_full")}</th><th>${t(loc, "arch.col_model")}</th></tr></thead>
        <tbody>${rows}</tbody>
      </table></div>
    </section>` : ""}
    <section class="fc-sec">
      <h2 class="fc-h2">${t(loc, "arch.cut_h")}</h2>
      <p class="fc-p">${t(loc, "arch.cut_p", { source: loc.source.name, method: href(loc, "metodologia") })}</p>
      <p class="fc-p">${t(loc, "arch.cut_p2")}</p>
      <p class="fc-p"><a href="${href(loc, "hub")}">${t(loc, "common.all_models")}</a> · <a href="${href(loc, "indice")}">${t(loc, "idx.crumb")}</a></p>
    </section>`;
  return layout({
    title: t(loc, "arch.title", { country: loc.countryName }),
    description: t(loc, "arch.desc", { country: loc.countryName }),
    body, zone: "all", nav: "feed", depositCount: null, index: true, host, locale: loc,
    canonical: permalink,
    jsonLd: graph([
      {
        "@type": "Dataset", "url": permalink, "inLanguage": loc.lang,
        "name": t(loc, "arch.ds_name", { country: loc.countryName }),
        "description": t(loc, "arch.desc", { country: loc.countryName }),
        "creator": { "@type": "Organization", "name": "Carsbuyer", "url": `https://${host}/` },
        "isAccessibleForFree": true,
        "distribution": shown.slice(0, 20).map(w => ({
          "@type": "DataDownload", "encodingFormat": "application/json",
          "contentUrl": `${permalink}/${String(w).toLowerCase()}.json`,
        })),
      },
      breadcrumbLd(host, crumbItems),
    ]),
  });
}

function indexStats(loc, snap) {
  const items = [];
  if (snap.models != null) items.push({ k: t(loc, "idx.stat_models"), v: fmtNumL(loc, snap.models), s: "" });
  if (snap.listings != null) items.push({ k: t(loc, "idx.stat_listings"), v: fmtNumL(loc, snap.listings), s: "" });
  if (snap.priceMed != null) items.push({ k: t(loc, "idx.stat_price"), v: escapeHtml(fmtEurL(loc, snap.priceMed)), s: "" });
  if (snap.kmMed != null) items.push({ k: t(loc, "idx.stat_km"), v: escapeHtml(fmtKmL(loc, snap.kmMed)), s: "" });
  return items.length ? statBlock(items) : "";
}

function weekRows(loc, rows) {
  return rows.map(r => `<tr>
      <td class="mono">${escapeHtml(r.week)}</td>
      <td class="mono">${r.priceMed != null ? escapeHtml(fmtEurL(loc, r.priceMed)) : t(loc, "common.na")}</td>
      <td class="mono">${r.listings != null ? fmtNumL(loc, r.listings) : t(loc, "common.na")}</td></tr>`).join("");
}

export function renderIntlMarketIndex({ loc, host, snapshot, history = [], gaps = [],
                                        months = [], month = null, pinned = null }) {
  const path = href(loc, "indice");
  const tail = month ? `/${month.month}` : (pinned ? `/${String(pinned).toLowerCase()}` : "");
  const permalink = `https://${host}${path}${tail}`;
  const snap = snapshot || history[history.length - 1] || {};
  const rows = month
    ? (month.rows || [])
    : history.slice().reverse().slice(0, 26);
  const crumbItems = [homeCrumb(loc), ...(month
    ? [{ name: t(loc, "idx.crumb"), href: path }, { name: month.month }]
    : [{ name: t(loc, "idx.crumb") }])];
  const monthList = months.length && !month
    ? `<section class="fc-sec">
        <h2 class="fc-h2">${t(loc, "idx.months_h")}</h2>
        <div class="mchips">${months.slice().reverse().map(c =>
          `<a class="mchip" href="${path}/${escapeHtml(c.month)}">${escapeHtml(c.month)}`
          + `${c.priceMed != null ? ` <span class="mut">${escapeHtml(fmtEurL(loc, c.priceMed))}</span>` : ""}</a>`).join("")}</div>
      </section>`
    : "";
  const body = crumbs(crumbItems) + `
    <section class="fc-sec">
      ${eyebrow(t(loc, "idx.eyebrow", { week: escapeHtml(month ? month.month : (pinned || snap.week || "")) }))}
      <h1 class="fc-h1">${month ? t(loc, "idx.month_h1", { month: escapeHtml(month.month) }) : t(loc, "idx.h1")}</h1>
      <p class="fc-p">${t(loc, "idx.lede")}</p>
      ${indexStats(loc, month || snap)}
      ${gaps.length && !month ? `<p class="fc-p mono" style="font-size:12px;">${t(loc, "idx.gap", { weeks: escapeHtml(gaps.join(", ")) })}</p>` : ""}
      ${month || pinned ? `<p class="fc-p"><a href="${path}">${t(loc, "idx.back")}</a></p>` : ""}
    </section>
    ${rows.length ? `<section class="fc-sec">
      <h2 class="fc-h2">${t(loc, "idx.weeks_h")}</h2>
      <div class="fc-scroll"><table class="fc-tbl">
        <thead><tr><th>${t(loc, "idx.col_week")}</th><th>${t(loc, "idx.col_price")}</th><th>${t(loc, "idx.col_listings")}</th></tr></thead>
        <tbody>${weekRows(loc, rows)}</tbody>
      </table></div>
    </section>` : ""}
    ${monthList}
    <section class="fc-sec">
      <p class="fc-p mono" style="font-size:12px;">${t(loc, "idx.foot", { source: loc.source.name })}</p>
      <p class="fc-p"><a href="${href(loc, "arquivo")}">${t(loc, "arch.crumb")}</a> · <a href="${href(loc, "hub")}">${t(loc, "common.all_models")}</a></p>
    </section>`;
  return layout({
    title: month
      ? `${t(loc, "idx.month_h1", { month: month.month })} · ${loc.countryName}`
      : t(loc, "idx.title", { country: loc.countryName }),
    description: month
      ? t(loc, "idx.month_desc", { country: loc.countryName, month: month.month })
      : t(loc, "idx.desc", { country: loc.countryName, source: loc.source.name }),
    body, zone: "all", nav: "feed", depositCount: null,
    index: rows.length > 0, host, locale: loc, canonical: permalink,
    jsonLd: graph([breadcrumbLd(host, crumbItems)]),
  });
}

export function carPath(loc, olxId) {
  return `${href(loc, "car")}?olx_id=${encodeURIComponent(String(olxId || ""))}`;
}

export function renderIntlCar({ loc, host, deal, rec = null, builtAt = null }) {
  const name = (rec && rec.t) || deal.title
    || [deal.brand, deal.model, deal.year].filter(v => v != null && v !== "").join(" ");
  const card = rec || {
    t: name, y: deal.year, km: deal.mileage_km, fu: deal.fuel_type, ct: deal.city,
    p: deal.price_eur, fl: deal.fair_low, fm: deal.fair_median, fh: deal.fair_high,
    sd: deal.sell_days != null ? deal.sell_days : null, ms: slugOf(deal),
  };
  const photos = Array.isArray(deal.photo_urls) && deal.photo_urls.length
    ? deal.photo_urls.slice(0, 5)
    : (deal.image_url ? [deal.image_url] : []);
  const shots = photos.length
    ? `<div class="fc-shots">${photos.map((u, i) =>
        `<img src="${escapeHtml(u)}" alt="${escapeHtml(name)}" ${i ? `loading="lazy"` : `fetchpriority="high"`}>`).join("")}</div>`
    : "";
  const sig = [];
  const push = (k, v) => { if (v != null && v !== "") sig.push({ k, v, s: "" }); };
  push(t(loc, "car.sig_km"), deal.mileage_km != null ? escapeHtml(fmtKmL(loc, deal.mileage_km)) : null);
  push(t(loc, "car.sig_seller"), deal.seller_type ? escapeHtml(labelL(loc, deal.seller_type)) : null);
  push(t(loc, "car.sig_days"), deal.days_on_market != null ? escapeHtml(fmtNumL(loc, deal.days_on_market)) : null);
  push(t(loc, "car.sig_discount"), deal.discount_pct != null ? `${pctInt(deal.discount_pct)}%` : null);
  push(t(loc, "car.sig_damage"), deal.damage_severity != null ? `${deal.damage_severity} / 3` : null);
  push(t(loc, "car.sig_photo"), deal.photo_damage_p != null ? `${pctInt(deal.photo_damage_p)}%` : null);
  push(t(loc, "car.sig_sample"), deal.sample_size != null ? escapeHtml(fmtNumL(loc, deal.sample_size)) : null);

  const body = `
    <section class="fc-sec">
      <a class="chip" href="${href(loc, "mercado")}">${t(loc, "car.back")}</a>
      ${shots}
      ${verdictBlock(loc, card, deal.url || null, null, "car")}
      ${sig.length ? `<h2 class="fc-h2">${t(loc, "car.signals_h")}</h2>${statBlock(sig)}` : ""}
      <p class="mono fc-prov">${t(loc, "feed.prov", {
        date: day(builtAt) || t(loc, "common.na"), source: loc.source.name,
      })}</p>
    </section>`;
  return layout({
    title: t(loc, "av.v_title", { title: name }),
    description: t(loc, "car.desc", { title: name, source: loc.source.name }),
    body, zone: "all", nav: "feed", depositCount: null, index: false, host, locale: loc,
    canonical: `https://${host}${carPath(loc, deal.olx_id)}`,
    jsonLd: graph([breadcrumbLd(host, [homeCrumb(loc),
      { name: t(loc, "feed.crumb"), href: href(loc, "mercado") }])]),
  });
}

function dealTile(loc, d) {
  const name = [d.brand, d.model, d.year].filter(v => v != null && v !== "").join(" ");
  const photo = d.image_url || (Array.isArray(d.photo_urls) ? d.photo_urls[0] : null);
  const saving = (d.fair_median != null && d.price_eur != null && d.fair_median > d.price_eur)
    ? d.fair_median - d.price_eur : null;
  const seller = d.seller_type ? labelL(loc, d.seller_type) : t(loc, "feed.seller_unknown");
  const sub = [
    d.mileage_km != null ? fmtKmL(loc, d.mileage_km) : null,
    d.fuel_type ? labelL(loc, d.fuel_type) : null,
  ].filter(Boolean).join(" · ");
  const days = d.days_on_market != null
    ? (d.days_on_market === 1 ? t(loc, "feed.day_one") : t(loc, "feed.days", { d: fmtNumL(loc, d.days_on_market) }))
    : "";
  const detail = carPath(loc, d.olx_id);
  return `<article class="tile">
    <a href="${detail}" class="thumb">${photo
      ? `<img src="${escapeHtml(photo)}" alt="${escapeHtml(name)}" loading="lazy">`
      : `<div class="striped" style="height:100%;"><span class="striped-label">Carsbuyer</span></div>`}</a>
    <div class="tbody">
      <div class="tile-title"><a href="${detail}">${escapeHtml(name)}</a></div>
      <div class="tile-sub">${escapeHtml(sub)}</div>
      <div class="price-row">
        <div><div style="font-size:11px;color:#8A8F98;">${t(loc, "av.v_cap_price")}</div>
        <div class="mono" style="font-weight:700;font-size:22px;letter-spacing:-0.02em;">${escapeHtml(fmtEurL(loc, d.price_eur))}</div></div>
        ${d.fair_median != null ? `<div style="margin-bottom:3px;"><span class="fair-strike">${escapeHtml(t(loc, "feed.fair", { price: fmtEurL(loc, d.fair_median) }))}</span></div>` : ""}
        ${saving != null ? `<span class="profit-pill">${escapeHtml(t(loc, "feed.saving", { amount: fmtEurL(loc, saving) }))}</span>` : ""}
      </div>
      <div class="tile-foot">
        ${d.district ? `<span>${escapeHtml(d.district)}</span><span class="sep">·</span>` : ""}
        ${days ? `<span>${escapeHtml(days)}</span>` : ""}
        <span class="seller">${escapeHtml(seller)}</span>
      </div>
      ${d.url ? `<a class="btn-outline" href="${escapeHtml(d.url)}" target="_blank" rel="noopener nofollow"${analyticsClick("olx_open", { source: "feed", market: loc.code, site: loc.source.host })}
        style="width:100%;margin-top:14px;font-size:14px;padding:11px;background:#FAFAF8;text-align:center;">${t(loc, "feed.open")}</a>` : ""}
    </div>
  </article>`;
}

function feedModelChips(loc, deals, models) {
  if (!models) return "";
  const seen = [];
  for (const d of deals) {
    const slug = slugOf(d);
    if (slug && models[slug] && !seen.includes(slug)) seen.push(slug);
    if (seen.length >= FEED_MODEL_CHIPS) break;
  }
  if (!seen.length) return "";
  const chips = seen.map(slug => {
    const r = models[slug];
    return `<a class="mchip" href="${href(loc, "model", slug)}">${escapeHtml(r.b)} ${escapeHtml(r.m)}`
      + ` <span class="mut">${escapeHtml(t(loc, "feed.chip", { price: fmtEurL(loc, r.fm) }))}`
      + `${escapeHtml(t(loc, "feed.chip_count", { n: fmtNumL(loc, r.n) }))}</span></a>`;
  }).join("");
  return `<div class="sec-label" style="margin-top:34px;">${t(loc, "feed.models_h")}</div><div class="mchips">${chips}</div>`;
}

function slugOf(d) {
  if (!d.brand || !d.model) return null;
  return `${d.brand} ${d.model}`.toLowerCase()
    .normalize("NFD").replace(/[̀-ͯ]/g, "")
    .replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");
}

export function renderIntlMethodology({ loc, host, stats, builtAt }) {
  const today = (stats && stats.listings)
    ? t(loc, "method.s1_today", { listings: fmtNumL(loc, stats.listings), models: fmtNumL(loc, stats.models) })
    : "";
  const body = crumbs([homeCrumb(loc), { name: t(loc, "method.crumb") }]) + `
    <section class="section fc-wrap" style="padding-top:16px;">
      <h1 class="fc-h1">${t(loc, "method.h1")}</h1>
      <p class="fc-p">${t(loc, "method.intro")}</p>

      <h2 class="fc-h2">${t(loc, "method.s1_h")}</h2>
      <p class="fc-p">${t(loc, "method.s1_p", { source: loc.source.name, today })}</p>

      <h2 class="fc-h2">${t(loc, "method.s2_h")}</h2>
      <p class="fc-p">${t(loc, "method.s2_p")}</p>

      <h2 class="fc-h2">${t(loc, "method.s3_h")}</h2>
      <p class="fc-p">${t(loc, "method.s3_p")}</p>

      <h2 class="fc-h2">${t(loc, "method.s4_h")}</h2>
      <p class="fc-p">${t(loc, "method.s4_p")}</p>
      <ul class="fc-ul">
        <li class="fc-li">${t(loc, "method.s4_li1")}</li>
        <li class="fc-li">${t(loc, "method.s4_li2")}</li>
        <li class="fc-li">${t(loc, "method.s4_li3", { min: 10 })}</li>
      </ul>

      <h2 class="fc-h2">${t(loc, "method.s5_h")}</h2>
      <p class="fc-p">${t(loc, "method.s5_p1")}</p>
      <p class="fc-p">${t(loc, "method.s5_p2")}</p>
      <p class="fc-p">${t(loc, "method.s5_p3")}</p>
      <ul class="fc-ul">
        <li class="fc-li">${t(loc, "method.s5_li1")}</li>
        <li class="fc-li">${t(loc, "method.s5_li2")}</li>
        <li class="fc-li">${t(loc, "method.s5_li3")}</li>
        <li class="fc-li">${t(loc, "method.s5_li4")}</li>
      </ul>
      <p class="fc-p">${t(loc, "method.s5_p4")}</p>

      <h2 class="fc-h2">${t(loc, "method.s6_h")}</h2>
      <ul class="fc-ul">
        <li class="fc-li">${t(loc, "method.s6_li1", { avaliar: href(loc, "avaliar") })}</li>
        <li class="fc-li">${t(loc, "method.s6_li2", { source: loc.source.name })}</li>
        <li class="fc-li">${t(loc, "method.s6_li3")}</li>
      </ul>

      <h2 class="fc-h2">${t(loc, "method.lic_h")}</h2>
      <p class="fc-p">${t(loc, "method.lic_p", { source: loc.source.name })}</p>
      <p class="fc-p">${t(loc, "method.links", { about: href(loc, "sobre"), hub: href(loc, "hub") })}</p>
      ${intlProvenance(loc, { n: stats ? stats.listings : null, builtAt })}
    </section>`;
  return layout({
    title: t(loc, "method.title"),
    description: t(loc, "method.desc", { country: loc.countryName }),
    body, zone: "all", nav: null, depositCount: null, index: true, host, locale: loc,
    canonical: `https://${host}${href(loc, "metodologia")}`,
    jsonLd: graph([
      breadcrumbLd(host, [homeCrumb(loc), { name: t(loc, "method.crumb"), href: href(loc, "metodologia") }]),
      {
        "@type": "TechArticle",
        "headline": t(loc, "method.headline", { country: loc.countryName }),
        "inLanguage": loc.lang,
        "url": `https://${host}${href(loc, "metodologia")}`,
        "dateModified": builtAt || null,
      },
    ]),
  });
}

export function renderIntlAbout({ loc, host, stats, builtAt }) {
  const models = stats ? fmtNumL(loc, stats.models) : "0";
  const listings = (stats && stats.listings)
    ? t(loc, "about.li1_listings", { listings: fmtNumL(loc, stats.listings) }) : "";
  const body = crumbs([homeCrumb(loc), { name: t(loc, "about.crumb") }]) + `
    <section class="section fc-wrap" style="padding-top:16px;">
      <h1 class="fc-h1">${t(loc, "about.h1")}</h1>
      <p class="fc-p">${t(loc, "about.p1", { country: loc.countryName })}</p>
      <h2 class="fc-h2">${t(loc, "about.h2a")}</h2>
      <p class="fc-p">${t(loc, "about.p2")}</p>
      <h2 class="fc-h2">${t(loc, "about.h2b")}</h2>
      <p class="fc-p">${t(loc, "about.p3")}</p>
      <h2 class="fc-h2">${t(loc, "about.h2c")}</h2>
      <ul class="fc-ul">
        <li class="fc-li">${t(loc, "about.li1", { models, listings })}</li>
        <li class="fc-li">${t(loc, "about.li2")}</li>
        <li class="fc-li">${t(loc, "about.li3")}</li>
      </ul>
      <h2 class="fc-h2">${t(loc, "about.h2d")}</h2>
      <p class="fc-p">${t(loc, "about.p4", { method: href(loc, "metodologia") })}</p>
      <p class="fc-p">${t(loc, "about.links", { method: href(loc, "metodologia"), hub: href(loc, "hub"), avaliar: href(loc, "avaliar") })}</p>
      ${intlProvenance(loc, { n: stats ? stats.listings : null, builtAt })}
    </section>`;
  return layout({
    title: t(loc, "about.title"),
    description: t(loc, "about.desc", { country: loc.countryName, source: loc.source.name, models }),
    body, zone: "all", nav: null, depositCount: null, index: true, host, locale: loc,
    canonical: `https://${host}${href(loc, "sobre")}`,
    jsonLd: graph([
      breadcrumbLd(host, [homeCrumb(loc), { name: t(loc, "about.crumb"), href: href(loc, "sobre") }]),
      orgLd(loc, host, t(loc, "about.org_desc", { country: loc.countryName, source: loc.source.name })),
    ]),
  });
}

export function renderIntlPrivacy({ loc, host }) {
  const body = crumbs([homeCrumb(loc), { name: t(loc, "priv.crumb") }]) + `
    <section class="section fc-wrap" style="padding-top:16px;padding-bottom:50px;">
      <h1 class="fc-h1">${t(loc, "priv.h1")}</h1>
      <p class="fc-p">${t(loc, "priv.intro")}</p>
      <h2 class="fc-h2">${t(loc, "priv.h_always")}</h2>
      <p class="fc-p">${t(loc, "priv.p_always")}</p>
      <h2 class="fc-h2">${t(loc, "priv.h_consent")}</h2>
      <p class="fc-p">${t(loc, "priv.p_consent")}</p>
      <h2 class="fc-h2">${t(loc, "priv.h_listings")}</h2>
      <p class="fc-p">${t(loc, "priv.p_listings", { source: loc.source.name })}</p>
      <h2 class="fc-h2">${t(loc, "priv.h_rights")}</h2>
      <p class="fc-p">${t(loc, "priv.p_rights", { about: href(loc, "sobre") })}</p>
    </section>`;
  return layout({
    title: t(loc, "priv.title"),
    description: t(loc, "priv.desc", { source: loc.source.name }),
    body, zone: "all", nav: null, depositCount: null, index: true, host, locale: loc,
    canonical: `https://${host}${href(loc, "privacidade")}`,
    jsonLd: graph([breadcrumbLd(host, [homeCrumb(loc), { name: t(loc, "priv.crumb"), href: href(loc, "privacidade") }])]),
  });
}

export function intlSitemapPaths(loc, models, hasDeals = true, builtAt = null) {
  const out = [
    { path: href(loc, "landing"), freq: "daily", prio: "0.9" },
    { path: href(loc, "hub"), freq: "weekly", prio: "0.7" },
    { path: href(loc, "avaliar"), freq: "weekly", prio: "0.8" },
    ...(hasDeals ? [{ path: href(loc, "mercado"), freq: "daily", prio: "0.8" }] : []),
    { path: href(loc, "metodologia"), freq: "monthly", prio: "0.6" },
    { path: href(loc, "sobre"), freq: "monthly", prio: "0.6" },
    { path: href(loc, "privacidade"), freq: "yearly", prio: "0.2" },
  ];
  for (const [slug, rec] of Object.entries(models || {})) {
    out.push({ path: href(loc, "model", slug), freq: "daily", prio: "0.6" });
    for (const y of intlPublishedYears(loc, models, slug, rec, builtAt)) {
      out.push({ path: hrefYear(loc, slug, y), freq: "daily", prio: "0.5" });
    }
  }
  return out;
}

export function intlYearCell(rec, year) {
  return yearCell(rec, year);
}

export function intlSiblings(models, slug, rec, limit = 8) {
  return Object.entries(models)
    .filter(([sl, r]) => r.b === rec.b && sl !== slug)
    .sort((a, b) => (b[1].n || 0) - (a[1].n || 0))
    .slice(0, limit)
    .map(([sl, r]) => ({ slug: sl, m: `${r.b} ${r.m}`, fm: r.fm, n: r.n }));
}
