// Second-layer SEO pages — everything narrower or wider than "one model".
//
// Why a second file: templates.js owns the product (feed, car, checkout) and the
// design system. This owns the pages that exist to answer a search query, all of
// which are derived from the SAME models.json blob the /preco pages already read.
// No new data source, no new pipeline — the numbers are re-cut, not re-collected.
//
// What lives here:
//   /preco/{slug}/{ano}      one model in one year  — the query is "quanto vale
//                            um Golf de 2012", and the model page answers it for
//                            every year at once, which wins none of them.
//   /depreciacao/{slug}      how fast this model loses value (+ the hub)
//   /comparar/{a}-vs-{b}     two models side by side (+ the hub)
//   /liquidez                how long each model takes to sell
//   /sobrevalorizados        where asking price and estimated fair value diverge
//   /mercado/indice          market-wide medians, with a permanent weekly archive
//   /metodologia /sobre      how the numbers are made, and by whom
//   /isv                     ISV estimator for an imported car
//   404                      a real not-found page (see index.js: unknown paths
//                            used to fall into the analytics Basic-Auth gate and
//                            answer 401, which Googlebot reads as "forbidden",
//                            not "gone")
//
// THE PUBLISHING RULE, applied everywhere below: a page exists only where the
// sample behind it is big enough for its number to mean something. A model-year
// with 4 listings is a row in the parent table, never a URL of its own. Minting
// URLs for thin cells is how a 250-page site becomes a 1300-page site that ranks
// for less than it did before.

import {
  layout, escapeHtml, fmtEur, fmtKm, fmtNum, fmtBuilt, slugify,
  present, thumbBlock, gradeChip,
} from "./templates.js";

// ── Publishing thresholds ────────────────────────────────────────────────────
// A model-year gets its own URL at 10+ active listings. Rationale: at n=5 (the
// blob's own floor for showing a row) a single outlier moves the median by more
// than the year-over-year step we are claiming to measure, so the page would be
// asserting a difference it cannot see. 10 is where the per-year medians stop
// crossing each other out of order in the corpus.
export const MIN_YEAR_PAGE_N = 10;

// Условия повторного использования опубликованных цифр. Полный текст лежит на
// /metodologia#licenca и продублирован в поле licence каждого .json-эндпоинта,
// чтобы машиночитаемая и человекочитаемая формулировки не разъехались.
const licenseUrl = (host) => `https://${host}/metodologia#licenca`;

// A depreciation curve needs enough points, over enough time, that a straight
// line through them is a description rather than an interpolation.
const DEP_MIN_CELLS = 8;      // year cells (n>=5) feeding the fit
const DEP_MIN_SPAN = 8;       // years between the oldest and newest cell
const DEP_MIN_R2 = 0.55;      // fit must actually explain the points
const DEP_MIN_RATE = 0.03;    // <3%/yr is noise or a collector car, not depreciation
const DEP_MAX_RATE = 0.22;    // >22%/yr means the year mix, not age, is driving it

// Comparison pages are generated, so they need a hard cap or they become the
// site: 243 models is 29 403 unordered pairs. These bounds keep only pairs a
// buyer would actually put against each other.
const CMP_POOL = 60;          // deepest-sampled models only
const CMP_PRICE_TOL = 0.25;   // medians within 25% of each other
const CMP_PER_MODEL = 3;      // so one popular model can't monopolise the set
const CMP_MAX = 90;

// ── Page-set selection ───────────────────────────────────────────────────────
// These are pure functions of the blob, and BOTH the router and the sitemap call
// them. That is deliberate: if the two disagreed, we would either serve pages no
// sitemap advertises or advertise pages that 404 — and the second is the failure
// mode Search Console reports as a site-wide error.

// ── Staged rollout ───────────────────────────────────────────────────────────
//
// 565 model-year pages appearing on one crawl is a bad way to find out whether
// the layer works: if indexation comes back at 40% there is no way to tell an
// unconvincing page template from a crawl-budget wall.
//
// SEO_WAVE_MODELS (wrangler.toml [vars]) caps the layer to the N deepest-sampled
// models. The gate is applied in ONE place and everything reads it — router,
// sitemap, and the year links on the model page — so a page outside the current
// wave is not merely unlisted, it is unreachable and unlinked. Widening the
// number is the whole release step; no deploy of code, no data rebuild.
//
// Empty ⇒ no cap, every qualifying page is live.
let WAVE_MODELS = 0;
export function setWave(n) {
  const v = parseInt(n, 10);
  WAVE_MODELS = Number.isFinite(v) && v > 0 ? v : 0;
}

let _waveKey = null, _waveVal = null;
/** Slugs inside the current wave. Everything, when no cap is set. */
export function waveSlugs(models, builtAt) {
  if (!WAVE_MODELS) return null;                 // null = "no gate", not "empty"
  const key = `${builtAt || ""}:${Object.keys(models).length}:${WAVE_MODELS}`;
  if (_waveKey === key && _waveVal) return _waveVal;
  _waveVal = new Set(Object.entries(models)
    .sort((a, b) => (b[1].n || 0) - (a[1].n || 0) || (a[0] < b[0] ? -1 : 1))
    .slice(0, WAVE_MODELS)
    .map(([slug]) => slug));
  _waveKey = key;
  return _waveVal;
}

/**
 * Year pages actually published for this model right now.
 *
 * The single source the router, the sitemap and the model page's year links all
 * call. Splitting them is how a sitemap ends up advertising 404s.
 */
export function publishedYearPages(models, slug, rec, builtAt) {
  const wave = waveSlugs(models, builtAt);
  if (wave && !wave.has(slug)) return [];
  return yearPageYears(rec);
}

/** Whether this model's depreciation page is published in the current wave. */
export function publishedDepreciation(models, slug, rec, builtAt) {
  const wave = waveSlugs(models, builtAt);
  if (wave && !wave.has(slug)) return false;
  return depreciationOk(rec);
}

/** Comparison pairs published in the current wave (both sides must be in). */
export function publishedPairs(models, builtAt) {
  const wave = waveSlugs(models, builtAt);
  const all = comparePairs(models);
  return wave ? all.filter(([a, b]) => wave.has(a) && wave.has(b)) : all;
}

/** Facet URLs published for this model in the current wave. */
export function publishedFacets(models, slug, rec, builtAt) {
  const wave = waveSlugs(models, builtAt);
  if (wave && !wave.has(slug)) return [];
  return facetKeys(rec);
}

/** Integer-year cells (bands excluded), newest first. */
export function yearCells(rec, minN = 1) {
  return (rec && Array.isArray(rec.yr) ? rec.yr : [])
    .filter(c => typeof c.y === "number" && c.fm != null && (c.n || 0) >= minN)
    .sort((a, b) => b.y - a.y);
}

/** Years of this model that clear the year-page floor. */
export function yearPageYears(rec) {
  return yearCells(rec, MIN_YEAR_PAGE_N).map(c => c.y);
}

/** The one cell for {slug}/{year}, or null when that year has no page. */
export function yearCell(rec, year) {
  return yearCells(rec, MIN_YEAR_PAGE_N).find(c => c.y === year) || null;
}

/**
 * Log-linear fit of median asking price on year → annual depreciation rate.
 *
 * Log-linear, not straight-line, because depreciation is a percentage of what
 * is left: a car that loses 10%/yr loses far more euros in year two than in
 * year nine, and a straight line through euros would report the average of two
 * different things. Returns { rate, r2, cells, span, newest, oldest } or null.
 */
export function depreciationFit(rec) {
  const cs = yearCells(rec, 5).slice().sort((a, b) => a.y - b.y);
  if (cs.length < 2) return null;
  const xs = cs.map(c => c.y);
  const ys = cs.map(c => Math.log(c.fm));
  const n = xs.length;
  const mx = xs.reduce((a, b) => a + b, 0) / n;
  const my = ys.reduce((a, b) => a + b, 0) / n;
  let sxy = 0, sxx = 0;
  for (let i = 0; i < n; i++) { sxy += (xs[i] - mx) * (ys[i] - my); sxx += (xs[i] - mx) ** 2; }
  if (sxx === 0) return null;
  const b = sxy / sxx;                 // d(log price)/d(year), positive = newer costs more
  const a = my - b * mx;
  let ssRes = 0, ssTot = 0;
  for (let i = 0; i < n; i++) {
    ssRes += (ys[i] - (a + b * xs[i])) ** 2;
    ssTot += (ys[i] - my) ** 2;
  }
  const r2 = ssTot > 0 ? 1 - ssRes / ssTot : 0;
  return {
    rate: 1 - Math.exp(-b),            // fraction of value lost per year of age
    r2, cells: cs, span: xs[n - 1] - xs[0],
    oldest: cs[0], newest: cs[n - 1],
    predict: y => Math.exp(a + b * y),
  };
}

/** Whether this model earns a /depreciacao page. */
export function depreciationOk(rec) {
  const f = depreciationFit(rec);
  return !!(f
    && f.cells.length >= DEP_MIN_CELLS
    && f.span >= DEP_MIN_SPAN
    && f.r2 >= DEP_MIN_R2
    && f.rate >= DEP_MIN_RATE
    && f.rate <= DEP_MAX_RATE);
}

/** Slugs with a depreciation page, deepest sample first. */
export function depreciationSlugs(models) {
  return Object.entries(models)
    .filter(([, r]) => depreciationOk(r))
    .sort((a, b) => (b[1].n || 0) - (a[1].n || 0))
    .map(([s]) => s);
}

/**
 * The comparison set: deterministic, so routing and the sitemap agree.
 *
 * Cross-brand only (a Golf against a Polo is the same brand's own ladder, which
 * the model page's sibling chips already cover) and price-adjacent, because
 * "{A} ou {B}" is a question people ask about cars that are actual substitutes.
 * Ordering inside a pair is alphabetical so /comparar/a-vs-b and /comparar/b-vs-a
 * can never both exist.
 */
export function comparePairs(models) {
  const pool = Object.entries(models)
    .filter(([, r]) => r.fm > 0)
    .sort((a, b) => (b[1].n || 0) - (a[1].n || 0))
    .slice(0, CMP_POOL);
  const cand = [];
  for (let i = 0; i < pool.length; i++) {
    for (let j = i + 1; j < pool.length; j++) {
      const [sa, ra] = pool[i], [sb, rb] = pool[j];
      if (ra.b === rb.b) continue;
      const dist = Math.abs(ra.fm - rb.fm) / Math.max(ra.fm, rb.fm);
      if (dist > CMP_PRICE_TOL) continue;
      const [x, y] = sa < sb ? [sa, sb] : [sb, sa];
      cand.push({ a: x, b: y, dist, depth: Math.min(ra.n || 0, rb.n || 0) });
    }
  }
  // Deepest samples first; ties broken by how close the two prices are, then by
  // slug so the set is stable across builds with identical inputs.
  cand.sort((p, q) => (q.depth - p.depth) || (p.dist - q.dist)
    || (p.a + p.b).localeCompare(q.a + q.b));
  const used = new Map();
  const out = [];
  for (const c of cand) {
    if ((used.get(c.a) || 0) >= CMP_PER_MODEL || (used.get(c.b) || 0) >= CMP_PER_MODEL) continue;
    used.set(c.a, (used.get(c.a) || 0) + 1);
    used.set(c.b, (used.get(c.b) || 0) + 1);
    out.push([c.a, c.b]);
    if (out.length >= CMP_MAX) break;
  }
  return out;
}

/**
 * Resolve "{a}-vs-{b}" back to two slugs.
 *
 * Slugs contain hyphens ("alfa-romeo-giulietta"), so a split on the first "-vs-"
 * is not enough — try every "-vs-" boundary and accept the one where BOTH halves
 * are real models. Anything not in the generated pair set is a 404: a URL a
 * scraper invented must not become a page.
 */
export function parseComparePath(rest, models, pairSet) {
  const parts = rest.split("-vs-");
  for (let i = 1; i < parts.length; i++) {
    const a = parts.slice(0, i).join("-vs-");
    const b = parts.slice(i).join("-vs-");
    if (!models[a] || !models[b]) continue;
    const [x, y] = a < b ? [a, b] : [b, a];
    if (pairSet && !pairSet.has(`${x}-vs-${y}`)) continue;
    return { a: x, b: y };
  }
  return null;
}

export function comparePairKey(a, b) {
  return a < b ? `${a}-vs-${b}` : `${b}-vs-${a}`;
}

// ── Corpus statistics ────────────────────────────────────────────────────────
// Every comparative sentence on a page ("faster than the market") needs a
// market to compare against. That is this: medians over all 243 models, from
// the same blob, computed once per build and memoised — 243 records is nothing,
// but recomputing it on every request of every page is still waste.
let _statsKey = null, _statsVal = null;

export function corpusStats(models, builtAt) {
  const key = `${builtAt || ""}:${models ? Object.keys(models).length : 0}`;
  if (_statsKey === key && _statsVal) return _statsVal;
  const med = arr => {
    const v = arr.filter(x => x != null && isFinite(x)).sort((a, b) => a - b);
    return v.length ? v[Math.floor(v.length / 2)] : null;
  };
  const recs = Object.values(models || {});
  const rates = [];
  for (const r of recs) {
    const f = depreciationFit(r);
    if (f && f.cells.length >= 5 && f.rate > 0 && f.rate < 0.30) rates.push(f.rate);
  }
  const stats = {
    models: recs.length,
    priceMed: med(recs.map(r => r.fm)),
    kmMed: med(recs.map(r => r.kmm)),
    sellMed: med(recs.map(r => r.sd)),
    // Spread = interquartile width as a share of the median. A wide spread means
    // condition/spec/history decide the price more than the badge does.
    spreadMed: med(recs.map(r => (r.fm > 0 && r.fl != null && r.fh != null)
      ? (r.fh - r.fl) / r.fm : null)),
    depMed: med(rates),
    // Asking vs estimated fair value, where the model's estimate was publishable.
    gapMed: med(recs.map(r => (r.gm > 0 && r.fm > 0) ? r.fm / r.gm - 1 : null)),
    listings: recs.reduce((s, r) => s + (r.n || 0), 0),
  };
  _statsKey = key; _statsVal = stats;
  return stats;
}

// ── Prose generated from THIS page's numbers ─────────────────────────────────
//
// The problem this solves: with one hand-written template behind 243 pages, the
// median 7-gram overlap between any two model pages was 49% — only the figures
// differed. At 1 000+ pages that is the reason Google keeps a fraction of them.
//
// The fix is not a rewrite (nobody rewrites 565 pages, and a rewrite drifts away
// from the data the moment the data moves). It is to derive the sentences FROM
// the figures: each rule below reads this model's own numbers, compares them to
// the corpus, and only speaks when it has something true and specific to say.
// A model that depreciates at the market rate simply does not get that sentence.
//
// Every rule states a consequence for the reader, because a number without a
// consequence is still boilerplate — just boilerplate with a variable in it.
export function modelInsights(rec, stats) {
  const out = [];
  const pct = x => Math.round(Math.abs(x) * 100);
  const B = escapeHtml(rec.b), M = escapeHtml(rec.m);

  // 1. Depreciation against the market.
  const fit = depreciationFit(rec);
  if (fit && fit.cells.length >= 5 && stats.depMed && fit.rate > 0 && fit.rate < 0.30) {
    const mine = fit.rate, mkt = stats.depMed;
    const rel = (mine - mkt) / mkt;
    if (rel > 0.18) {
      out.push(`Perde valor depressa: cerca de <b>${pct(mine)}% por cada ano de idade</b>, contra ${pct(mkt)}% na mediana do mercado. Comprar um exemplar dois ou três anos mais velho poupa mais aqui do que na maioria dos modelos.`);
    } else if (rel < -0.18) {
      out.push(`Segura bem o valor: cerca de <b>${pct(mine)}% por ano de idade</b>, contra ${pct(mkt)}% na mediana do mercado. Pagar por um ano mais recente custa menos do que noutros modelos — e revender também dói menos.`);
    } else {
      out.push(`Desvaloriza ao ritmo do mercado: cerca de <b>${pct(mine)}% por ano de idade</b> (mediana do mercado: ${pct(mkt)}%).`);
    }
  }

  // 2. Time to sell — the seller's question, and the buyer's leverage.
  if (rec.sd != null && rec.sn != null && stats.sellMed) {
    const d = rec.sd, mkt = stats.sellMed;
    if (d <= mkt * 0.75) {
      out.push(`Vende rápido: mediana de <b>${d} dias</b> no OLX (mercado: ${mkt} dias), em ${rec.sn} vendas observadas. A anunciar, tens pouca pressão para descer o preço; a comprar, os bons exemplares desaparecem em dias.`);
    } else if (d >= mkt * 1.25) {
      out.push(`Demora a sair: mediana de <b>${d} dias</b> no OLX contra ${mkt} do mercado, em ${rec.sn} vendas observadas. Quem vende costuma ter de ceder no preço, e quem compra tem margem para negociar.`);
    } else {
      out.push(`Tempo até vender em linha com o mercado: mediana de <b>${d} dias</b> (mercado: ${mkt}), em ${rec.sn} vendas observadas.`);
    }
  }

  // 3. Price dispersion — how much the individual car matters versus the badge.
  if (rec.fm > 0 && rec.fl != null && rec.fh != null && stats.spreadMed) {
    const sp = (rec.fh - rec.fl) / rec.fm, mkt = stats.spreadMed;
    if (sp > mkt * 1.25) {
      out.push(`Os preços estão muito dispersos: metade dos anúncios ocupa uma faixa de ${pct(sp)}% em torno da mediana, contra ${pct(mkt)}% no mercado. Aqui o estado, os quilómetros e a versão pesam mais do que o modelo — a mediana sozinha diz pouco sobre o carro à tua frente.`);
    } else if (sp < mkt * 0.75) {
      out.push(`Os preços são apertados: metade dos anúncios cabe em ${pct(sp)}% em torno da mediana, contra ${pct(mkt)}% no mercado. Um anúncio muito fora desta faixa tem quase sempre uma razão — versão rara, quilometragem extrema ou algo por dizer.`);
    }
  }

  // 4. Mileage on offer.
  if (rec.kmm != null && stats.kmMed) {
    const rel = (rec.kmm - stats.kmMed) / stats.kmMed;
    if (rel > 0.20) {
      out.push(`A quilometragem mediana à venda é alta: <b>${fmtKm(rec.kmm)}</b> contra ${fmtKm(stats.kmMed)} no mercado. Um exemplar abaixo disso justifica pedir mais do que a mediana.`);
    } else if (rel < -0.20) {
      out.push(`A quilometragem mediana à venda é baixa: <b>${fmtKm(rec.kmm)}</b> contra ${fmtKm(stats.kmMed)} no mercado, por isso a mediana de preço aqui é a de carros pouco rodados.`);
    }
  }

  // 5. Fuel mix — only when it is genuinely lopsided, i.e. when it constrains
  //    what you can actually find.
  if (Array.isArray(rec.fu) && rec.fu.length && rec.fu[0][1] >= 0.80) {
    const [f, share] = rec.fu[0];
    out.push(`O mercado deste modelo é praticamente todo ${escapeHtml(String(f).toLowerCase())}: <b>${pct(share)}% dos anúncios</b>. As outras motorizações aparecem pouco, e quando aparecem o preço não segue esta mediana.`);
  }

  // 6. The steepest year-over-year step in the table — where the money is.
  const cs = yearCells(rec, MIN_YEAR_PAGE_N).slice().sort((a, b) => a.y - b.y);
  let best = null;
  for (let i = 1; i < cs.length; i++) {
    if (cs[i].y !== cs[i - 1].y + 1) continue;
    const step = (cs[i].fm - cs[i - 1].fm) / cs[i].fm;
    if (step > 0.12 && (!best || step > best.step)) best = { step, lo: cs[i - 1], hi: cs[i] };
  }
  if (best) {
    out.push(`O maior degrau está entre <b>${best.lo.y} e ${best.hi.y}</b>: ${pct(best.step)}% de diferença na mediana (${fmtEur(best.lo.fm)} contra ${fmtEur(best.hi.fm)}). Se o ano não é decisivo para ti, é aqui que se poupa mais de uma vez só.`);
  }

  // 7. Where the market sits against our own fair-value estimate.
  if (rec.gm > 0 && rec.fm > 0 && stats.gapMed != null) {
    const gap = rec.fm / rec.gm - 1;
    if (gap > (stats.gapMed + 0.10)) {
      out.push(`O que se pede está acima do valor justo que estimamos (${fmtEur(rec.gm)}) mais do que é habitual neste mercado. Trata a mediana como preço de partida, não como preço final.`);
    } else if (gap < (stats.gapMed - 0.10)) {
      out.push(`O que se pede está abaixo do valor justo que estimamos (${fmtEur(rec.gm)}) mais do que é habitual — sinal de oferta a mais ou de procura fraca neste momento.`);
    }
  }

  return out;
}

// ── Small shared blocks ──────────────────────────────────────────────────────

/**
 * The provenance line, identical in shape on every data page.
 *
 * Same words, same order, same data-* attributes everywhere, so a model reading
 * the page can lift "how many, how fresh, of what" without parsing prose — and a
 * human gets the caveat in the same place every time.
 */
export function provenance({ n, builtAt, measure = "Preço pedido em anúncios ativos (mediana e P25-P75)", extra = "" }) {
  const day = (builtAt || "").slice(0, 10);
  return `<p class="mono fc-prov" data-sample="${n != null ? n : ""}" data-updated="${escapeHtml(day)}" data-measure="asking-price-median" data-source="OLX Portugal">`
    + `Amostra: ${n != null ? fmtNum(n) + " anúncios ativos" : "n/d"} · Recolhido até: ${day || "n/d"} · Medida: ${escapeHtml(measure)} · Fonte: OLX Portugal${extra ? " · " + extra : ""}`
    + `</p>`;
}

/** Visible breadcrumb; mirrors whatever BreadcrumbList the page emits. */
export function crumbs(items) {
  const inner = items.map((it, i) => {
    const last = i === items.length - 1;
    return last
      ? `<span style="color:#16181D;">${escapeHtml(it.name)}</span>`
      : `<a href="${escapeHtml(it.href)}" style="color:#8A8F98;">${escapeHtml(it.name)}</a>`;
  }).join(" › ");
  return `<nav class="section fc-crumbs" aria-label="Breadcrumb">${inner}</nav>`;
}

export function breadcrumbLd(host, items) {
  return {
    "@type": "BreadcrumbList",
    "itemListElement": items.map((it, i) => ({
      "@type": "ListItem", "position": i + 1, "name": it.name,
      ...(it.href ? { "item": `https://${host}${it.href}` } : {}),
    })),
  };
}

export function faqLd(pairs) {
  return {
    "@type": "FAQPage",
    "mainEntity": pairs.map(([q, a]) => ({
      "@type": "Question", "name": q,
      "acceptedAnswer": { "@type": "Answer", "text": a },
    })),
  };
}

/**
 * Median-price-by-year line chart, inline SVG.
 *
 * Inline because it must render with the HTML (no JS, no request) — this chart
 * IS the page's argument, and a chart that paints late is a chart nobody sees.
 * viewBox + preserveAspectRatio make it fluid without a media query.
 */
export function priceChart(cells, { w = 640, h = 220, color = "#177A47" } = {}) {
  const pts = cells.slice().sort((a, b) => a.y - b.y);
  if (pts.length < 2) return "";
  const padL = 16, padR = 14, padT = 20, padB = 30;
  const xs = pts.map(p => p.y);
  const ys = pts.map(p => p.fm);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const y1 = Math.max(...ys), y0 = 0;
  const X = y => padL + ((y - x0) / Math.max(1, x1 - x0)) * (w - padL - padR);
  const Y = v => padT + (1 - (v - y0) / Math.max(1, y1 - y0)) * (h - padT - padB);
  const line = pts.map((p, i) => `${i ? "L" : "M"}${X(p.y).toFixed(1)},${Y(p.fm).toFixed(1)}`).join("");
  const area = `${line}L${X(x1).toFixed(1)},${Y(0).toFixed(1)}L${X(x0).toFixed(1)},${Y(0).toFixed(1)}Z`;
  const dots = pts.map(p =>
    `<circle cx="${X(p.y).toFixed(1)}" cy="${Y(p.fm).toFixed(1)}" r="3" fill="${color}"><title>${p.y}: ${fmtEur(p.fm)} (${p.n} anúncios)</title></circle>`).join("");
  // At most six year labels, otherwise they collide on a phone.
  const step = Math.max(1, Math.ceil(pts.length / 6));
  const xlab = pts.filter((_, i) => i % step === 0 || i === pts.length - 1).map(p => {
    // The end labels are anchored inwards or half of them falls outside the viewBox.
    const anchor = p.y === x0 ? "start" : p.y === x1 ? "end" : "middle";
    return `<text x="${X(p.y).toFixed(1)}" y="${h - 9}" text-anchor="${anchor}" class="c-ax">${p.y}</text>`;
  }).join("");
  const ticks = [0, 0.5, 1].map(f => {
    const v = y1 * f;
    return `<line x1="${padL}" x2="${w - padR}" y1="${Y(v).toFixed(1)}" y2="${Y(v).toFixed(1)}" class="c-grid"/>`
      + `<text x="${padL + 2}" y="${(Y(v) - 5).toFixed(1)}" text-anchor="start" class="c-ax">${fmtEur(Math.round(v))}</text>`;
  }).join("");
  return `<svg class="fc-chart" viewBox="0 0 ${w} ${h}" role="img"
    aria-label="Preço mediano pedido por ano de fabrico">${ticks}
    <path d="${area}" fill="${color}" opacity="0.10"/>
    <path d="${line}" fill="none" stroke="${color}" stroke-width="2.2" stroke-linejoin="round"/>
    ${dots}${xlab}</svg>`;
}

// ═══ /preco/{slug}/{ano} ═════════════════════════════════════════════════════
//
// The model page answers "quanto vale um Golf" for eighteen model years at once,
// which means it competes with itself on every one of them and wins none. This
// page answers exactly one: "quanto vale um Golf de 2012".
//
// It only exists where the year has 10+ active listings (MIN_YEAR_PAGE_N).
// Thinner years stay as a row in the parent table — visible, linked, honest, and
// not a URL asking to be indexed on four data points.
export function renderYearPage({ rec, slug, year, cell, neighbours, liveDeals, pageYears,
                                 stats, host, depositCount, builtAt }) {
  const B = escapeHtml(rec.b), M = escapeHtml(rec.m);
  const FM = fmtEur(cell.fm), FL = fmtEur(cell.fl), FH = fmtEur(cell.fh);
  const canonical = `https://${host}/preco/${slug}/${year}`;
  const hasG = cell.gm != null && cell.gl != null && cell.gh != null;
  const refYear = parseInt((builtAt || "").slice(0, 4), 10) || null;
  const age = refYear ? refYear - year : null;
  const share = rec.n ? Math.round((cell.n / rec.n) * 100) : null;

  let pin = 50;
  if (cell.fh > cell.fl) pin = Math.max(6, Math.min(94, Math.round((cell.fm - cell.fl) / (cell.fh - cell.fl) * 100)));

  const hero = `
    <div class="side-card" style="max-width:680px;margin:0 auto;">
      <div class="eyebrow" style="margin-bottom:14px;"><span class="e-dot"></span><span class="mono">${B.toUpperCase()} ${M.toUpperCase()} · ${year} · OLX PORTUGAL</span></div>
      <h1 class="fc-h1">Quanto vale um ${B} ${M} de ${year}?</h1>
      <p class="lede" style="font-size:16px;margin:0 0 20px;">Nos <b>${cell.n} anúncios ativos</b> de ${B} ${M} do ano ${year} no OLX, o preço pedido mediano é <b>${FM}</b>${cell.km != null ? `, com ${fmtKm(cell.km)} de quilometragem mediana` : ""}. É o que o mercado pede hoje por este ano concreto, não uma avaliação da tua viatura.</p>
      <div class="side-prices">
        <div><div class="cap">Preço mediano (pedido) · ${year}</div><div class="big">${FM}</div></div>
        <div class="side-fair"><div class="cap">${cell.n} anúncios${age != null ? ` · ${age} ano${age === 1 ? "" : "s"}` : ""}</div><div class="v">${cell.km != null ? fmtKm(cell.km) : "—"}</div></div>
      </div>
      <div style="margin-top:16px;">
        <div class="gauge-head"><span>${FL}</span><span>intervalo típico (50% dos anúncios de ${year})</span><span>${FH}</span></div>
        <div class="gauge-track"><div class="gauge-pin" style="left:${pin}%;"></div></div>
      </div>
      ${hasG ? `<div style="margin-top:16px;padding-top:14px;border-top:1px solid #EFECE6;"><div class="cap">Valor justo estimado para ${year}</div><div class="mono" style="font-weight:700;font-size:20px;color:#177A47;">${fmtEur(cell.gm)}</div><div class="mono" style="font-size:12px;color:#5B606B;">intervalo ${fmtEur(cell.gl)} – ${fmtEur(cell.gh)}</div></div>` : ""}
      ${provenance({ n: cell.n, builtAt, measure: `Preço pedido, ${B} ${M} do ano ${year} (mediana e P25-P75)` })}
    </div>`;

  // The one comparison a buyer on this page is actually making: is the next year
  // up worth its premium, and how much does the year below save?
  const stepBlock = (() => {
    const bits = [];
    const older = neighbours.older, newer = neighbours.newer;
    if (newer) {
      const d = (newer.fm - cell.fm) / cell.fm;
      const href = pageYears.includes(newer.y) ? `/preco/${slug}/${newer.y}` : `/preco/${slug}`;
      bits.push(`<li>Um <a href="${href}">${B} ${M} de ${newer.y}</a> pede em mediana ${fmtEur(newer.fm)}, <b>${d >= 0 ? "+" : ""}${Math.round(d * 100)}%</b> face a ${year}${d > 0.10 ? " — um degrau grande para um ano de diferença" : d > 0 ? "" : " — mais recente e ainda assim não mais caro, sinal de amostra desigual"}.</li>`);
    }
    if (older) {
      const d = (cell.fm - older.fm) / cell.fm;
      const href = pageYears.includes(older.y) ? `/preco/${slug}/${older.y}` : `/preco/${slug}`;
      bits.push(`<li>Descer para <a href="${href}">${older.y}</a> poupa cerca de <b>${Math.round(d * 100)}%</b> (mediana ${fmtEur(older.fm)})${older.km != null && cell.km != null ? `, com ${fmtKm(Math.abs(older.km - cell.km))} ${older.km > cell.km ? "a mais" : "a menos"} no conta-quilómetros` : ""}.</li>`);
    }
    if (share != null) {
      bits.push(`<li>O ano ${year} representa <b>${share}%</b> de todos os ${B} ${M} à venda agora (${cell.n} de ${rec.n} anúncios)${share >= 15 ? " — é dos anos com mais escolha" : share <= 5 ? " — há pouca oferta, por isso conta com menos margem para escolher" : ""}.</li>`);
    }
    return bits.length ? `<section class="section fc-wrap"><h2 class="fc-h2">Vale a pena pagar por um ano mais recente?</h2><ul class="fc-insights">${bits.join("")}</ul></section>` : "";
  })();

  // Neighbouring years in a table, each linked where it has its own page.
  const rows = neighbours.window.map(c => {
    const isSelf = c.y === year;
    const label = pageYears.includes(c.y) && !isSelf
      ? `<a href="/preco/${slug}/${c.y}" style="color:#177A47;font-weight:600;">${c.y}</a>`
      : `${isSelf ? `<b>${c.y}</b>` : c.y}`;
    return `<tr${isSelf ? ' style="background:#F6FBF8;"' : ""}>
      <td>${label}</td><td>${c.n}</td><td>${fmtEur(c.fm)}</td>
      <td class="mut">${fmtEur(c.fl)} – ${fmtEur(c.fh)}</td>
      <td class="mut">${c.km != null ? fmtKm(c.km) : "—"}</td></tr>`;
  }).join("");
  const table = rows ? `
    <section class="section fc-wrap">
      <h2 class="fc-h2">${B} ${M}: anos vizinhos</h2>
      <div class="fc-scroll"><table class="fc-tbl">
        <thead><tr><th>Ano</th><th>Anúncios</th><th>Mediano (pedido)</th><th>P25–P75</th><th>Km mediano</th></tr></thead>
        <tbody>${rows}</tbody></table></div>
      <p class="fc-p" style="margin-top:12px;"><a href="/preco/${slug}">Ver todos os anos de ${B} ${M}&nbsp;→</a></p>
    </section>` : "";

  const yearNav = pageYears.length > 1 ? `
    <section class="section fc-wrap">
      <div class="sec-label" style="margin-bottom:10px;">OUTROS ANOS COM DADOS SUFICIENTES</div>
      <div class="fc-yearlinks">${pageYears.map(y =>
        y === year ? `<a class="on" href="/preco/${slug}/${y}">${y}</a>` : `<a href="/preco/${slug}/${y}">${y}</a>`).join("")}</div>
    </section>` : "";

  const deals = (liveDeals || []).length ? `
    <section class="section fc-wide">
      <div class="sec-label">${B} ${M} DE ${year} ABAIXO DO PREÇO JUSTO AGORA</div>
      <div class="grid">${liveDeals.slice(0, 3).map(d => {
        const p = present(d);
        return `<a class="tile" href="/car?olx_id=${encodeURIComponent(d.olx_id)}" style="max-width:none;">
          <div class="thumb">${thumbBlock(p, 168, 28)}${gradeChip(p)}</div>
          <div class="tbody"><div class="tile-title">${escapeHtml(p.name)}</div>
          <div class="tile-sub">${p.subHtml}</div>
          <div class="price-row"><div class="price">${p.priceStr}</div><div class="fair-strike">${p.fairStr}</div></div>
          </div></a>`;
      }).join("")}</div>
    </section>` : `
    <section class="section fc-wrap">
      <p class="fc-p">Sem ${B} ${M} de ${year} abaixo do preço justo neste momento. <a href="/avaliar">Avalia um anúncio concreto</a> ou <a href="/mercado">vê o mercado completo</a>.</p>
    </section>`;

  const cta = `
    <section class="section fc-wide">
      <div class="cta-banner">
        <div style="flex:1 1 360px;">
          <h2>Tens um ${B} ${M} de ${year}?</h2>
          <p>Esta é a mediana do ano. Cola o link do teu anúncio e dizemos o preço justo desse carro em concreto — quilómetros, versão e estado incluídos.</p>
        </div>
        <a class="btn-bright" href="/avaliar?modelo=${encodeURIComponent(slug)}&ano=${year}">Avaliar o meu ${year}&nbsp;&nbsp;→</a>
      </div>
    </section>`;

  const links = `
    <section class="section fc-wrap" style="padding-bottom:70px;">
      <p class="fc-p"><a href="/preco/${slug}">Preços de ${B} ${M} por ano</a>${depreciationOk(rec) ? ` · <a href="/depreciacao/${slug}">Curva de desvalorização</a>` : ""} · <a href="/precos">Todos os modelos</a></p>
    </section>`;

  const body = crumbs([
    { name: "Início", href: "/" }, { name: "Preços", href: "/precos" },
    { name: `${rec.b} ${rec.m}`, href: `/preco/${slug}` }, { name: String(year) },
  ]) + `<div style="padding-top:14px;">${hero}</div>${stepBlock}${table}${yearNav}${deals}${cta}${links}`;

  const faqs = [[
    `Quanto vale um ${rec.b} ${rec.m} de ${year} em Portugal?`,
    `Nos ${cell.n} anúncios ativos de ${rec.b} ${rec.m} do ano ${year} no OLX Portugal, o preço pedido mediano é ${FM}, com metade dos anúncios entre ${FL} e ${FH}. São preços pedidos em anúncios ativos, não preços de venda fechados.`,
  ]];
  if (cell.km != null) faqs.push([
    `Qual é a quilometragem típica de um ${rec.b} ${rec.m} de ${year}?`,
    `A quilometragem mediana dos ${rec.b} ${rec.m} de ${year} à venda é ${fmtKm(cell.km)}. Um exemplar bastante abaixo desse valor justifica um preço acima da mediana do ano, e vice-versa.`,
  ]);
  if (neighbours.newer) {
    const d = Math.round((neighbours.newer.fm - cell.fm) / cell.fm * 100);
    faqs.push([
      `Compensa comprar um ${rec.b} ${rec.m} de ${neighbours.newer.y} em vez de ${year}?`,
      `Um ${rec.b} ${rec.m} de ${neighbours.newer.y} pede em mediana ${fmtEur(neighbours.newer.fm)}, ou seja ${d >= 0 ? "+" : ""}${d}% face aos ${FM} de ${year}. A diferença compensa se a quilometragem e o estado acompanharem; caso contrário estás a pagar pelo ano na matrícula.`,
    ]);
  }
  if (hasG) faqs.push([
    `Um ${rec.b} ${rec.m} de ${year} a ${FM} é caro?`,
    `Para o ano ${year} estimamos um valor justo de ${fmtEur(cell.gm)} (intervalo ${fmtEur(cell.gl)} a ${fmtEur(cell.gh)}) com base nas características típicas dos exemplares deste ano. O preço pedido mediano é ${FM}. Um carro concreto pode estar acima ou abaixo consoante quilómetros, versão e histórico.`,
  ]);

  const jsonLd = {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "Dataset",
        "license": licenseUrl(host),
        "name": `Preços de ${rec.b} ${rec.m} de ${year} em Portugal`,
        "description": `Mediana e intervalo interquartil dos preços pedidos em ${cell.n} anúncios ativos de ${rec.b} ${rec.m} do ano ${year} no OLX Portugal.`,
        "creator": { "@type": "Organization", "name": "Flipper Club", "url": `https://${host}/` },
        "isAccessibleForFree": true,
        "temporalCoverage": String(year),
        "variableMeasured": hasG ? ["Preço pedido (EUR)", "Valor justo estimado (EUR)"] : "Preço pedido (EUR)",
        "dateModified": builtAt || undefined,
        "url": canonical,
        "distribution": {
          "@type": "DataDownload",
          "encodingFormat": "application/json",
          "contentUrl": `${canonical}.json`,
        },
      },
      // AggregateOffer, not Product: we are describing the ASK across a market of
      // listings we do not own or sell. itemOffered names the car; no seller is
      // claimed, no availability, no single price — which is exactly what an
      // aggregate of third-party asking prices is.
      {
        "@type": "AggregateOffer",
        "priceCurrency": "EUR",
        "lowPrice": cell.fl, "highPrice": cell.fh, "offerCount": cell.n,
        "url": canonical,
        "itemOffered": {
          "@type": "Car", "name": `${rec.b} ${rec.m} ${year}`,
          "brand": { "@type": "Brand", "name": rec.b },
          "model": rec.m, "vehicleModelDate": String(year),
          ...(cell.km != null ? {
            "mileageFromOdometer": { "@type": "QuantitativeValue", "value": cell.km, "unitCode": "KMT" },
          } : {}),
        },
      },
      breadcrumbLd(host, [
        { name: "Início", href: "/" }, { name: "Preços", href: "/precos" },
        { name: `${rec.b} ${rec.m}`, href: `/preco/${slug}` }, { name: String(year) },
      ]),
      faqLd(faqs),
    ],
  };

  return layout({
    title: `Quanto vale um ${rec.b} ${rec.m} de ${year}? Preço em Portugal`,
    description: `${rec.b} ${rec.m} de ${year} usado: preço mediano ${FM} (${FL}–${FH}) em ${cell.n} anúncios ativos do OLX Portugal${cell.km != null ? `, ${fmtKm(cell.km)} medianos` : ""}. Avaliação independente.`,
    canonical, jsonLd, body, zone: "all", nav: "precos", depositCount, index: true, host,
    altJson: `${canonical}.json`,
  });
}

// ═══ 404 ═════════════════════════════════════════════════════════════════════
//
// Unknown paths used to fall through to the analytics Basic-Auth gate and answer
// 401. To a crawler 401 means "there is something here and you may not have it",
// so every typo'd link, every old URL, every stray trailing slash was spending
// crawl budget and logging a site-wide access error instead of resolving.
//
// This is a real 404: correct status, useful body, noindex,follow so the links
// on it still carry signal back into the site.
export function renderNotFound({ suggestions = [], depositCount = 0, host = null, path = "" } = {}) {
  const chips = suggestions.slice(0, 12).map(s =>
    `<a class="mchip" href="/preco/${encodeURIComponent(s.slug)}">${escapeHtml(s.m)} <span class="mut">${fmtEur(s.fm)}</span></a>`).join("");
  const body = `
    <div class="fc-404">
      <div class="eyebrow" style="justify-content:center;margin-bottom:16px;"><span class="e-dot"></span><span class="mono">ERRO 404</span></div>
      <h1 class="fc-h1">Esta página não existe</h1>
      <p class="fc-p">${path ? `Não temos nada em <span class="mono">${escapeHtml(path)}</span>. ` : ""}Pode ter sido um link antigo ou um endereço mal escrito. O que existe está tudo a partir daqui:</p>
      <div class="hero-actions" style="justify-content:center;margin:22px 0 30px;">
        <a class="btn-dark" href="/precos">Preços por modelo</a>
        <a class="chip" href="/avaliar">Avaliar o meu carro</a>
        <a class="chip" href="/mercado">Mercado</a>
      </div>
      ${chips ? `<div class="sec-label" style="text-align:left;">MODELOS MAIS PROCURADOS</div><div class="mchips" style="justify-content:center;">${chips}</div>` : ""}
    </div>`;
  return layout({
    title: "Página não encontrada",
    description: "A página que procuras não existe. Vê os preços de carros usados por modelo em Portugal.",
    body, zone: "all", nav: null, depositCount, index: false, host,
  });
}

// ═══ /depreciacao/{slug} ═════════════════════════════════════════════════════
//
// "{modelo} desvalorização" is a query nobody on this market answers with real
// numbers, and it is the single most linkable thing this dataset can produce:
// "a Golf loses X% a year, and the fall flattens after N" is a sentence forums
// and press quote, which is where the links come from.
//
// Published only where the curve is a measurement (DEP_* guards above) — a fit
// through six scattered points would be a drawing, not a finding.
export function renderDepreciationPage({ rec, slug, fit, stats, pageYears, host, depositCount, builtAt }) {
  const B = escapeHtml(rec.b), M = escapeHtml(rec.m);
  const canonical = `https://${host}/depreciacao/${slug}`;
  const cs = fit.cells;                       // oldest → newest, n>=5
  const newest = fit.newest, oldest = fit.oldest;
  const ratePct = Math.round(fit.rate * 100);
  const keep = yrs => Math.round(Math.pow(1 - fit.rate, yrs) * 100);
  const loseEur = yrs => Math.round(newest.fm * (1 - Math.pow(1 - fit.rate, yrs)));

  // Euros per year of age, early life versus late life. The percentage rate is
  // constant by construction (that is what a log-linear fit means); the EUROS
  // are not, and euros are what the reader is deciding with.
  const mid = Math.floor(cs.length / 2);
  const segCost = seg => {
    if (seg.length < 2) return null;
    const span = seg[seg.length - 1].y - seg[0].y;
    if (span <= 0) return null;
    return Math.round((seg[seg.length - 1].fm - seg[0].fm) / span);
  };
  const lateSeg = cs.slice(mid);              // newer years
  const earlySeg = cs.slice(0, mid + 1);      // older years
  const costNew = segCost(lateSeg), costOld = segCost(earlySeg);

  const mkt = stats.depMed;
  const vsMarket = mkt
    ? (fit.rate > mkt * 1.15
        ? `Mais rápido do que o mercado (mediana ${Math.round(mkt * 100)}% ao ano).`
        : fit.rate < mkt * 0.85
          ? `Mais devagar do que o mercado (mediana ${Math.round(mkt * 100)}% ao ano) — segura melhor o valor do que a média.`
          : `Em linha com o mercado (mediana ${Math.round(mkt * 100)}% ao ano).`)
    : "";

  const rows = cs.slice().sort((a, b) => b.y - a.y).map(c => {
    const vs = Math.round((c.fm / newest.fm - 1) * 100);
    const ageGap = newest.y - c.y;
    const link = pageYears.includes(c.y)
      ? `<a href="/preco/${slug}/${c.y}" style="color:#177A47;font-weight:600;">${c.y}</a>` : c.y;
    return `<tr><td>${link}</td><td>${ageGap}</td><td>${c.n}</td><td>${fmtEur(c.fm)}</td>
      <td class="mut">${vs === 0 ? "—" : `${vs}%`}</td>
      <td class="mut">${c.km != null ? fmtKm(c.km) : "—"}</td></tr>`;
  }).join("");

  const body = crumbs([
    { name: "Início", href: "/" }, { name: "Desvalorização", href: "/depreciacao" },
    { name: `${rec.b} ${rec.m}` },
  ]) + `
    <div style="padding-top:14px;">
      <div class="side-card" style="max-width:680px;margin:0 auto;">
        <div class="eyebrow" style="margin-bottom:14px;"><span class="e-dot"></span><span class="mono">CURVA DE DESVALORIZAÇÃO · OLX PORTUGAL</span></div>
        <h1 class="fc-h1">Quanto se desvaloriza um ${B} ${M}?</h1>
        <p class="lede" style="font-size:16px;margin:0 0 18px;">Medido nos preços pedidos de <b>${rec.n} anúncios ativos</b> entre ${oldest.y} e ${newest.y}, um ${B} ${M} perde cerca de <b>${ratePct}% por cada ano de idade</b>. ${vsMarket}</p>
        <div class="fc-stat-row">
          <div class="fc-stat"><div class="k">POR ANO</div><div class="v">${ratePct}%</div><div class="s">do valor restante</div></div>
          <div class="fc-stat"><div class="k">AOS 3 ANOS</div><div class="v">${keep(3)}%</div><div class="s">-${fmtEur(loseEur(3))} sobre ${fmtEur(newest.fm)}</div></div>
          <div class="fc-stat"><div class="k">AOS 5 ANOS</div><div class="v">${keep(5)}%</div><div class="s">-${fmtEur(loseEur(5))}</div></div>
          <div class="fc-stat"><div class="k">AOS 8 ANOS</div><div class="v">${keep(8)}%</div><div class="s">-${fmtEur(loseEur(8))}</div></div>
        </div>
        ${provenance({ n: rec.n, builtAt, measure: `Preço pedido mediano por ano de fabrico, ${oldest.y}-${newest.y}`, extra: `Ajuste log-linear, R²=${fit.r2.toFixed(2)}` })}
      </div>
    </div>
    <section class="section fc-wrap">
      <h2 class="fc-h2">A curva</h2>
      ${priceChart(cs)}
      <p class="fc-p" style="margin-top:10px;">Cada ponto é a mediana dos preços pedidos nesse ano de fabrico. A queda é uma percentagem do que resta, por isso é grande em euros nos primeiros anos e pequena no fim — mesmo quando a percentagem não muda.</p>
    </section>
    ${(costNew && costOld && costNew > 0) ? `
    <section class="section fc-wrap">
      <h2 class="fc-h2">Onde está o dinheiro</h2>
      <ul class="fc-insights">
        <li>Entre ${lateSeg[0].y} e ${newest.y}, cada ano de idade custa em mediana <b>${fmtEur(costNew)}</b>.</li>
        <li>Entre ${oldest.y} e ${earlySeg[earlySeg.length - 1].y}, cada ano custa <b>${fmtEur(Math.max(0, costOld))}</b>${costOld < costNew * 0.6 ? " — a queda já abrandou, e é aqui que um ano a mais na matrícula sai barato" : ""}.</li>
        ${costNew > 0 ? `<li>Comprar ${Math.min(3, Math.max(1, newest.y - lateSeg[0].y))} ano${(newest.y - lateSeg[0].y) > 1 ? "s" : ""} abaixo do mais recente com dados poupa cerca de <b>${fmtEur(costNew * Math.min(3, Math.max(1, newest.y - lateSeg[0].y)))}</b> à partida.</li>` : ""}
      </ul>
    </section>` : ""}
    <section class="section fc-wrap">
      <h2 class="fc-h2">Preço mediano por ano</h2>
      <div class="fc-scroll"><table class="fc-tbl">
        <thead><tr><th>Ano</th><th>Anos vs. ${newest.y}</th><th>Anúncios</th><th>Mediano (pedido)</th><th>vs. ${newest.y}</th><th>Km mediano</th></tr></thead>
        <tbody>${rows}</tbody></table></div>
    </section>
    <section class="section fc-wide">
      <div class="cta-banner">
        <div style="flex:1 1 360px;">
          <h2>Quanto vale o TEU ${B} ${M} hoje?</h2>
          <p>Esta curva é do modelo. Cola o link do teu anúncio e dizemos o valor justo do teu carro concreto, com os teus quilómetros e a tua versão.</p>
        </div>
        <a class="btn-bright" href="/avaliar?modelo=${encodeURIComponent(slug)}">Avaliar o meu carro&nbsp;&nbsp;→</a>
      </div>
    </section>
    <section class="section fc-wrap" style="padding-bottom:70px;">
      <p class="fc-p"><a href="/preco/${slug}">Preço de ${B} ${M} por ano</a> · <a href="/depreciacao">Desvalorização de outros modelos</a> · <a href="/precos">Todos os modelos</a></p>
    </section>`;

  const faqs = [
    [`Quanto se desvaloriza um ${rec.b} ${rec.m} por ano?`,
     `Cerca de ${ratePct}% do valor restante por cada ano de idade, medido nos preços pedidos de ${rec.n} anúncios ativos de ${rec.b} ${rec.m} entre ${oldest.y} e ${newest.y} no OLX Portugal. A percentagem é constante, mas em euros a perda é muito maior nos primeiros anos.`],
    [`Quanto vale um ${rec.b} ${rec.m} ao fim de 5 anos?`,
     `Ao ritmo medido, um ${rec.b} ${rec.m} mantém cerca de ${keep(5)}% do valor ao fim de 5 anos. Sobre a mediana de ${fmtEur(newest.fm)} pedida pelos exemplares de ${newest.y}, isso são cerca de ${fmtEur(loseEur(5))} perdidos.`],
  ];
  if (mkt) faqs.push([
    `O ${rec.b} ${rec.m} desvaloriza mais do que a média?`,
    `A mediana do mercado português de usados que medimos é ${Math.round(mkt * 100)}% por ano de idade. O ${rec.b} ${rec.m} está nos ${ratePct}%, ou seja ${fit.rate > mkt ? "acima" : fit.rate < mkt ? "abaixo" : "em linha com"} a média.`]);

  const jsonLd = {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "Dataset",
        "license": licenseUrl(host),
        "name": `Desvalorização de ${rec.b} ${rec.m} em Portugal`,
        "description": `Preço pedido mediano de ${rec.b} ${rec.m} por ano de fabrico (${oldest.y}-${newest.y}) e taxa de desvalorização anual, a partir de ${rec.n} anúncios ativos do OLX Portugal.`,
        "creator": { "@type": "Organization", "name": "Flipper Club", "url": `https://${host}/` },
        "isAccessibleForFree": true,
        "temporalCoverage": `${oldest.y}/${newest.y}`,
        "variableMeasured": ["Preço pedido (EUR)", "Desvalorização anual (%)"],
        "dateModified": builtAt || undefined,
        "url": canonical,
      },
      breadcrumbLd(host, [
        { name: "Início", href: "/" }, { name: "Desvalorização", href: "/depreciacao" },
        { name: `${rec.b} ${rec.m}` },
      ]),
      faqLd(faqs),
    ],
  };
  return layout({
    title: `${rec.b} ${rec.m}: desvalorização por ano`,
    description: `Um ${rec.b} ${rec.m} perde cerca de ${ratePct}% por ano de idade e mantém ${keep(5)}% ao fim de 5 anos, medido em ${rec.n} anúncios ativos do OLX Portugal. Curva completa por ano.`,
    canonical, jsonLd, body, zone: "all", nav: "precos", depositCount, index: true, host,
  });
}

// ── /depreciacao — hub, ranked ───────────────────────────────────────────────
export function renderDepreciationHub({ rows, stats, host, depositCount, builtAt }) {
  const canonical = `https://${host}/depreciacao`;
  const tr = rows.map(r => `<tr>
      <td><a href="/depreciacao/${r.slug}" style="color:#177A47;font-weight:600;">${escapeHtml(r.b)} ${escapeHtml(r.m)}</a></td>
      <td>${Math.round(r.rate * 100)}%</td>
      <td class="mut">${Math.round(Math.pow(1 - r.rate, 5) * 100)}%</td>
      <td class="mut">${r.n}</td>
      <td class="mut">${r.span} anos</td></tr>`).join("");
  const body = crumbs([{ name: "Início", href: "/" }, { name: "Desvalorização" }]) + `
    <section class="section fc-wrap" style="padding-top:16px;">
      <h1 class="fc-h1">Que carros se desvalorizam mais em Portugal</h1>
      <p class="fc-p">Taxa de desvalorização por ano de idade, medida nos preços pedidos de anúncios ativos do OLX. Só entram modelos com histórico suficiente para a curva significar alguma coisa: pelo menos ${DEP_MIN_CELLS} anos com amostra e ${DEP_MIN_SPAN} anos de intervalo.</p>
      ${stats.depMed ? `<p class="fc-p">A mediana do mercado é <b>${Math.round(stats.depMed * 100)}% por ano</b>. Acima disso, o carro custa-te mais a ter; abaixo, revendes com menos perda.</p>` : ""}
      <div class="fc-scroll"><table class="fc-tbl">
        <thead><tr><th>Modelo</th><th>Por ano</th><th>Valor aos 5 anos</th><th>Anúncios</th><th>Histórico</th></tr></thead>
        <tbody>${tr}</tbody></table></div>
      ${provenance({ n: stats.listings, builtAt, measure: "Preço pedido mediano por ano de fabrico, ajuste log-linear" })}
      <p class="fc-p" style="margin-top:18px;"><a href="/precos">Todos os modelos</a> · <a href="/liquidez">Quanto tempo demoram a vender</a> · <a href="/metodologia">Como calculamos</a></p>
    </section>
    <div style="height:60px;"></div>`;
  return layout({
    title: "Desvalorização de carros usados em Portugal",
    description: `Que modelos perdem mais valor por ano em Portugal, medido em anúncios ativos do OLX. ${rows.length} modelos com curva completa e valor retido aos 5 anos.`,
    canonical, body, zone: "all", nav: "precos", depositCount, index: true, host,
    jsonLd: {
      "@context": "https://schema.org",
      "@graph": [
        {
          "@type": "CollectionPage", "url": canonical, "inLanguage": "pt-PT",
          "name": "Desvalorização de carros usados em Portugal",
        },
        breadcrumbLd(host, [{ name: "Início", href: "/" }, { name: "Desvalorização" }]),
      ],
    },
  });
}

// ═══ /comparar/{a}-vs-{b} ════════════════════════════════════════════════════
//
// "{A} ou {B} qual comprar usado" is how people actually shop, and no page on
// this market answers it with both cars' real numbers side by side.
//
// The page does NOT pick a winner overall — we measure price, dispersion, time
// to sell and depreciation, and none of those is "the better car". It names the
// winner PER DIMENSION and says what each one costs you, which is the honest
// version and also the more useful one.
export function renderComparePage({ a, b, ra, rb, stats, host, depositCount, builtAt }) {
  const canonical = `https://${host}/comparar/${a}-vs-${b}`;
  const nameA = `${ra.b} ${ra.m}`, nameB = `${rb.b} ${rb.m}`;
  const A = escapeHtml(nameA), Bn = escapeHtml(nameB);
  const fitA = depreciationFit(ra), fitB = depreciationFit(rb);
  const depA = (fitA && fitA.cells.length >= 5 && fitA.rate > 0 && fitA.rate < 0.30) ? fitA.rate : null;
  const depB = (fitB && fitB.cells.length >= 5 && fitB.rate > 0 && fitB.rate < 0.30) ? fitB.rate : null;
  const sprA = ra.fm > 0 ? (ra.fh - ra.fl) / ra.fm : null;
  const sprB = rb.fm > 0 ? (rb.fh - rb.fl) / rb.fm : null;

  const card = (slug, r, name, wins) => `
    <div class="fc-vs-card${wins ? " win" : ""}">
      <div class="mono" style="font-size:11px;color:#8A8F98;letter-spacing:.04em;">${escapeHtml(r.b).toUpperCase()}</div>
      <div style="font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:19px;margin:3px 0 12px;">${escapeHtml(r.m)}</div>
      <div class="cap">Preço mediano (pedido)</div>
      <div class="mono" style="font-weight:700;font-size:27px;letter-spacing:-.02em;">${fmtEur(r.fm)}</div>
      <div class="mono" style="font-size:12px;color:#5B606B;margin-top:2px;">${fmtEur(r.fl)} – ${fmtEur(r.fh)} · ${r.n} anúncios</div>
      <div style="margin-top:12px;font-size:13.5px;line-height:1.8;color:#3A3F47;">
        ${r.kmm != null ? `Km mediano · <b>${fmtKm(r.kmm)}</b><br>` : ""}
        ${r.sd != null ? `Vende em · <b>${r.sd} dias</b><br>` : ""}
        ${r.y0 && r.y1 ? `Anos à venda · <b>${r.y0}–${r.y1}</b><br>` : ""}
        ${Array.isArray(r.fu) && r.fu.length ? `Combustível · <b>${escapeHtml(String(r.fu[0][0]))} ${Math.round(r.fu[0][1] * 100)}%</b>` : ""}
      </div>
      <a class="btn-dark" href="/preco/${slug}" style="display:block;text-align:center;margin-top:14px;font-size:13.5px;padding:11px;">Ver ${escapeHtml(r.m)}&nbsp;→</a>
    </div>`;

  // Per-dimension verdicts. Each row states which side wins AND why that matters,
  // because "cheaper" is only an advantage once you know what it costs elsewhere.
  const verdicts = [];
  const cheaper = ra.fm <= rb.fm ? "a" : "b";
  const dPct = Math.round(Math.abs(ra.fm - rb.fm) / Math.max(ra.fm, rb.fm) * 100);
  verdicts.push({
    k: "Preço de entrada",
    w: cheaper,
    t: dPct === 0
      ? `As medianas são praticamente iguais (${fmtEur(ra.fm)} contra ${fmtEur(rb.fm)}).`
      : `${cheaper === "a" ? A : Bn} entra ${dPct}% mais barato: ${fmtEur(Math.min(ra.fm, rb.fm))} contra ${fmtEur(Math.max(ra.fm, rb.fm))}.`,
  });
  if (ra.sd != null && rb.sd != null) {
    const w = ra.sd <= rb.sd ? "a" : "b";
    verdicts.push({
      k: "Tempo até vender",
      w,
      t: `${w === "a" ? A : Bn} sai mais depressa (${Math.min(ra.sd, rb.sd)} contra ${Math.max(ra.sd, rb.sd)} dias medianos). Quem vende tem menos pressão no preço; quem compra tem menos tempo para decidir.`,
    });
  }
  if (depA != null && depB != null) {
    const w = depA <= depB ? "a" : "b";
    verdicts.push({
      k: "Desvalorização",
      w,
      t: `${w === "a" ? A : Bn} segura melhor o valor: ${Math.round(Math.min(depA, depB) * 100)}% por ano contra ${Math.round(Math.max(depA, depB) * 100)}%. Em cinco anos a diferença vale mais do que o desconto na compra.`,
    });
  }
  if (sprA != null && sprB != null) {
    const w = sprA <= sprB ? "a" : "b";
    verdicts.push({
      k: "Previsibilidade do preço",
      w,
      t: `${w === "a" ? A : Bn} tem preços mais concentrados (${Math.round(Math.min(sprA, sprB) * 100)}% de dispersão contra ${Math.round(Math.max(sprA, sprB) * 100)}%). Onde a dispersão é maior, o estado e a versão decidem mais do que o modelo — dá para fazer melhores negócios, e piores.`,
    });
  }
  if (ra.kmm != null && rb.kmm != null) {
    const w = ra.kmm <= rb.kmm ? "a" : "b";
    verdicts.push({
      k: "Quilometragem à venda",
      w,
      t: `Os ${w === "a" ? A : Bn} à venda estão menos rodados (${fmtKm(Math.min(ra.kmm, rb.kmm))} contra ${fmtKm(Math.max(ra.kmm, rb.kmm))} medianos), o que também explica parte da diferença de preço.`,
    });
  }
  const scoreA = verdicts.filter(v => v.w === "a").length;
  const scoreB = verdicts.filter(v => v.w === "b").length;

  const vrows = verdicts.map(v => `<tr>
      <td>${escapeHtml(v.k)}</td>
      <td class="nm">${v.w === "a" ? `<b>${escapeHtml(ra.m)}</b>` : escapeHtml(ra.m)}${v.w === "a" ? '<span class="fc-win">+</span>' : ""}</td>
      <td class="nm">${v.w === "b" ? `<b>${escapeHtml(rb.m)}</b>` : escapeHtml(rb.m)}${v.w === "b" ? '<span class="fc-win">+</span>' : ""}</td>
    </tr>`).join("");

  const body = crumbs([
    { name: "Início", href: "/" }, { name: "Comparar", href: "/comparar" },
    { name: `${nameA} vs ${nameB}` },
  ]) + `
    <section class="section fc-wrap" style="padding-top:16px;">
      <div class="eyebrow" style="margin-bottom:14px;"><span class="e-dot"></span><span class="mono">COMPARAÇÃO A PARTIR DE ANÚNCIOS ATIVOS · OLX PORTUGAL</span></div>
      <h1 class="fc-h1">${A} ou ${Bn}: qual comprar usado?</h1>
      <p class="fc-p">Comparação com números do mercado português de hoje: ${ra.n} anúncios ativos de ${A} e ${rb.n} de ${Bn}. Não dizemos qual é o melhor carro — dizemos o que cada um custa a comprar, a ter e a revender, e deixamos a escolha contigo.</p>
      <div class="fc-vs">
        ${card(a, ra, nameA, scoreA > scoreB)}
        ${card(b, rb, nameB, scoreB > scoreA)}
      </div>
      ${provenance({ n: ra.n + rb.n, builtAt, measure: "Preço pedido em anúncios ativos dos dois modelos (mediana e P25-P75)" })}
    </section>
    <section class="section fc-wrap">
      <h2 class="fc-h2">Quem ganha em quê</h2>
      <div class="fc-scroll"><table class="fc-tbl">
        <thead><tr><th>Critério</th><th>${escapeHtml(ra.m)}</th><th>${escapeHtml(rb.m)}</th></tr></thead>
        <tbody>${vrows}</tbody></table></div>
      <ul class="fc-insights" style="margin-top:16px;">${verdicts.map(v => `<li>${v.t}</li>`).join("")}</ul>
    </section>
    <section class="section fc-wide">
      <div class="cta-banner">
        <div style="flex:1 1 360px;">
          <h2>Já tens um anúncio em vista?</h2>
          <p>A mediana compara modelos. Para saber se AQUELE carro está bem de preço, cola o link do anúncio.</p>
        </div>
        <a class="btn-bright" href="/avaliar">Avaliar um anúncio&nbsp;&nbsp;→</a>
      </div>
    </section>
    <section class="section fc-wrap" style="padding-bottom:70px;">
      <p class="fc-p"><a href="/preco/${a}">Preços de ${A}</a> · <a href="/preco/${b}">Preços de ${Bn}</a> · <a href="/comparar">Outras comparações</a></p>
    </section>`;

  const faqs = [
    [`${nameA} ou ${nameB}: qual é mais barato em Portugal?`,
     dPct === 0
       ? `As medianas estão praticamente empatadas: ${fmtEur(ra.fm)} para o ${nameA} e ${fmtEur(rb.fm)} para o ${nameB}, em anúncios ativos do OLX Portugal.`
       : `O ${cheaper === "a" ? nameA : nameB} pede em mediana ${fmtEur(Math.min(ra.fm, rb.fm))} contra ${fmtEur(Math.max(ra.fm, rb.fm))} do ${cheaper === "a" ? nameB : nameA}, uma diferença de ${dPct}% nos anúncios ativos do OLX Portugal.`],
  ];
  if (depA != null && depB != null) faqs.push([
    `${nameA} ou ${nameB}: qual perde menos valor?`,
    `O ${depA <= depB ? nameA : nameB} desvaloriza cerca de ${Math.round(Math.min(depA, depB) * 100)}% por ano de idade, contra ${Math.round(Math.max(depA, depB) * 100)}% do outro. Sobre cinco anos, essa diferença costuma pesar mais do que o desconto na compra.`]);
  if (ra.sd != null && rb.sd != null) faqs.push([
    `Qual se vende mais depressa, ${nameA} ou ${nameB}?`,
    `O ${ra.sd <= rb.sd ? nameA : nameB} vende em mediana em ${Math.min(ra.sd, rb.sd)} dias no OLX, contra ${Math.max(ra.sd, rb.sd)} dias do outro. Um modelo que sai depressa dá menos margem de negociação a quem compra.`]);

  return layout({
    title: `${nameA} ou ${nameB}? Comparação de preços usados`,
    description: `${nameA} (${fmtEur(ra.fm)}) contra ${nameB} (${fmtEur(rb.fm)}) no mercado português de usados: preço, quilometragem, tempo até vender e desvalorização, em anúncios ativos do OLX.`,
    canonical, body, zone: "all", nav: "precos", depositCount, index: true, host,
    jsonLd: {
      "@context": "https://schema.org",
      "@graph": [
        breadcrumbLd(host, [
          { name: "Início", href: "/" }, { name: "Comparar", href: "/comparar" },
          { name: `${nameA} vs ${nameB}` },
        ]),
        faqLd(faqs),
      ],
    },
  });
}

// ── /comparar — hub ──────────────────────────────────────────────────────────
export function renderCompareHub({ pairs, models, host, depositCount, builtAt }) {
  const canonical = `https://${host}/comparar`;
  const items = pairs.map(([a, b]) => {
    const ra = models[a], rb = models[b];
    return `<a class="mchip" href="/comparar/${a}-vs-${b}">${escapeHtml(ra.b)} ${escapeHtml(ra.m)} <span class="mut">vs</span> ${escapeHtml(rb.b)} ${escapeHtml(rb.m)}</a>`;
  }).join("");
  const body = crumbs([{ name: "Início", href: "/" }, { name: "Comparar" }]) + `
    <section class="section fc-wrap" style="padding-top:16px;">
      <h1 class="fc-h1">Comparar carros usados em Portugal</h1>
      <p class="fc-p">Cada comparação usa os anúncios ativos dos dois modelos no OLX: preço mediano, dispersão, quilometragem, tempo até vender e desvalorização. Comparamos apenas modelos de marcas diferentes com preços próximos, porque são esses que se disputam de facto — um carro contra outro três vezes mais caro não é uma decisão que alguém tome.</p>
      <div class="mchips">${items}</div>
      ${provenance({ n: null, builtAt, measure: "Preço pedido mediano dos dois modelos comparados" })}
      <p class="fc-p" style="margin-top:18px;"><a href="/precos">Todos os modelos</a> · <a href="/depreciacao">Desvalorização</a> · <a href="/metodologia">Como calculamos</a></p>
    </section>
    <div style="height:60px;"></div>`;
  return layout({
    title: "Comparar carros usados em Portugal",
    description: `${pairs.length} comparações de carros usados em Portugal com preço, quilometragem, tempo até vender e desvalorização, a partir de anúncios ativos do OLX.`,
    canonical, body, zone: "all", nav: "precos", depositCount, index: true, host,
    jsonLd: {
      "@context": "https://schema.org",
      "@graph": [
        { "@type": "CollectionPage", "url": canonical, "inLanguage": "pt-PT", "name": "Comparar carros usados em Portugal" },
        breadcrumbLd(host, [{ name: "Início", href: "/" }, { name: "Comparar" }]),
      ],
    },
  });
}

// ═══ /liquidez — how long each model takes to sell ═══════════════════════════
//
// Days-to-sell is the one number here that neither Standvirtual nor a valuation
// book publishes, because it needs listings watched over time rather than a
// snapshot. It answers the seller's real question ("how long will this take")
// and hands the buyer their leverage ("this one sits, so offer less").
//
// One ranked hub, not 243 per-model pages: per model this is a single figure,
// and a page built on a single figure is a thin page. It lives on the model page
// too, where it has context.
export function renderLiquidityHub({ rows, stats, host, depositCount, builtAt }) {
  const canonical = `https://${host}/liquidez`;
  const mkt = stats.sellMed;
  const tr = rows.map(r => `<tr>
      <td><a href="/preco/${r.slug}" style="color:#177A47;font-weight:600;">${escapeHtml(r.b)} ${escapeHtml(r.m)}</a></td>
      <td>${r.sd} dias</td>
      <td class="mut">${mkt ? (r.sd < mkt ? `${Math.round((1 - r.sd / mkt) * 100)}% mais rápido` : r.sd > mkt ? `${Math.round((r.sd / mkt - 1) * 100)}% mais lento` : "na mediana") : "—"}</td>
      <td class="mut">${r.sn}</td>
      <td class="mut">${fmtEur(r.fm)}</td></tr>`).join("");
  const fastest = rows[0], slowest = rows[rows.length - 1];
  const body = crumbs([{ name: "Início", href: "/" }, { name: "Tempo de venda" }]) + `
    <section class="section fc-wrap" style="padding-top:16px;">
      <h1 class="fc-h1">Quanto tempo demora a vender cada carro em Portugal</h1>
      <p class="fc-p">Mediana de dias entre a publicação e o desaparecimento do anúncio no OLX, por modelo. Medimos os anúncios ao longo do tempo, por isso isto não é uma estimativa a partir do preço — é o que aconteceu.</p>
      ${(fastest && slowest) ? `<p class="fc-p">Do mais rápido ao mais lento vão <b>${fastest.sd}</b> e <b>${slowest.sd}</b> dias: um ${escapeHtml(fastest.b)} ${escapeHtml(fastest.m)} sai em cerca de ${fastest.sd} dias, um ${escapeHtml(slowest.b)} ${escapeHtml(slowest.m)} demora ${slowest.sd}. Se vais anunciar, é a diferença entre pedir o preço todo e ter de ceder.</p>` : ""}
      ${mkt ? `<p class="fc-p">Mediana do mercado: <b>${mkt} dias</b>.</p>` : ""}
      <div class="fc-scroll"><table class="fc-tbl">
        <thead><tr><th>Modelo</th><th>Mediana até vender</th><th>vs. mercado</th><th>Vendas observadas</th><th>Preço mediano</th></tr></thead>
        <tbody>${tr}</tbody></table></div>
      ${provenance({ n: rows.reduce((s, r) => s + (r.sn || 0), 0), builtAt, measure: "Dias medianos entre publicação e remoção do anúncio" })}
      <p class="fc-p" style="margin-top:18px;"><a href="/precos">Preços por modelo</a> · <a href="/depreciacao">Desvalorização</a> · <a href="/metodologia">Como medimos</a></p>
    </section>
    <div style="height:60px;"></div>`;
  return layout({
    title: "Quanto tempo demora a vender um carro em Portugal",
    description: `Dias medianos até vender por modelo no OLX Portugal${mkt ? ` (mediana do mercado: ${mkt} dias)` : ""}. Medido em anúncios reais acompanhados ao longo do tempo.`,
    canonical, body, zone: "all", nav: "precos", depositCount, index: true, host,
    jsonLd: {
      "@context": "https://schema.org",
      "@graph": [
        {
          "@type": "Dataset", "license": licenseUrl(host), "url": canonical, "inLanguage": "pt-PT",
          "name": "Tempo mediano até vender, por modelo (Portugal)",
          "description": "Dias medianos entre publicação e remoção de anúncios de carros usados no OLX Portugal, por modelo.",
          "creator": { "@type": "Organization", "name": "Flipper Club", "url": `https://${host}/` },
          "isAccessibleForFree": true, "dateModified": builtAt || undefined,
          "variableMeasured": "Dias até vender (mediana)",
        },
        breadcrumbLd(host, [{ name: "Início", href: "/" }, { name: "Tempo de venda" }]),
      ],
    },
  });
}

// ═══ /sobrevalorizados — asking price against our own estimate ═══════════════
//
// The mirror of /mercado: that page finds individual listings below fair value,
// this one finds where a whole MODEL is systematically asked above (or below)
// what we estimate it is worth. Both directions on one page on purpose — a list
// of only "overpriced" reads as an accusation, and the underpriced half is the
// actionable one for a buyer.
export function renderValuationGap({ over, under, stats, host, depositCount, builtAt }) {
  const canonical = `https://${host}/sobrevalorizados`;
  const row = r => `<tr>
      <td><a href="/preco/${r.slug}" style="color:#177A47;font-weight:600;">${escapeHtml(r.b)} ${escapeHtml(r.m)}</a></td>
      <td>${fmtEur(r.fm)}</td>
      <td class="mut">${fmtEur(r.gm)}</td>
      <td>${r.gap > 0 ? "+" : ""}${Math.round(r.gap * 100)}%</td>
      <td class="mut">${r.n}</td></tr>`;
  const body = crumbs([{ name: "Início", href: "/" }, { name: "Preço pedido vs. valor justo" }]) + `
    <section class="section fc-wrap" style="padding-top:16px;">
      <h1 class="fc-h1">Onde o preço pedido se afasta do valor justo</h1>
      <p class="fc-p">Para cada modelo comparamos o que o mercado <b>pede</b> com o que o nosso modelo <b>estima</b> que vale, para quilometragem e versões típicas desse modelo. Um desvio grande não significa que alguém esteja a enganar ninguém: significa que a oferta e a procura desse modelo estão desalinhadas neste momento, e é aí que se negoceia.</p>
      ${stats.gapMed != null ? `<p class="fc-p">No conjunto do mercado, o preço pedido está ${stats.gapMed >= 0 ? "acima" : "abaixo"} da estimativa em <b>${Math.abs(Math.round(stats.gapMed * 100))}%</b> na mediana — é o normal, e é a referência contra a qual ler a tabela.</p>` : ""}
      <h2 class="fc-h2">Pedem mais do que estimamos</h2>
      <p class="fc-p">Se vais comprar, entra a negociar. Se vais vender, o mercado está a teu favor.</p>
      <div class="fc-scroll"><table class="fc-tbl">
        <thead><tr><th>Modelo</th><th>Pedido (mediana)</th><th>Valor justo estimado</th><th>Desvio</th><th>Anúncios</th></tr></thead>
        <tbody>${over.map(row).join("")}</tbody></table></div>
      <h2 class="fc-h2">Pedem menos do que estimamos</h2>
      <p class="fc-p">Oferta a mais ou procura fraca. Bom momento para comprar, mau para anunciar.</p>
      <div class="fc-scroll"><table class="fc-tbl">
        <thead><tr><th>Modelo</th><th>Pedido (mediana)</th><th>Valor justo estimado</th><th>Desvio</th><th>Anúncios</th></tr></thead>
        <tbody>${under.map(row).join("")}</tbody></table></div>
      ${provenance({ n: null, builtAt, measure: "Preço pedido mediano vs. valor justo estimado pelo modelo" })}
      <p class="fc-p" style="margin-top:18px;">A estimativa só é publicada onde passa os nossos limites de fiabilidade — ver <a href="/metodologia">metodologia</a>. Modelos onde não passa não aparecem aqui.</p>
      <p class="fc-p"><a href="/mercado">Anúncios concretos abaixo do valor justo</a> · <a href="/precos">Todos os modelos</a></p>
    </section>
    <div style="height:60px;"></div>`;
  return layout({
    title: "Preço pedido vs. valor justo por modelo",
    description: "Que modelos de carros usados em Portugal são pedidos acima ou abaixo do valor justo estimado, a partir de anúncios ativos do OLX.",
    canonical, body, zone: "all", nav: "precos", depositCount, index: true, host,
    jsonLd: {
      "@context": "https://schema.org",
      "@graph": [
        { "@type": "CollectionPage", "url": canonical, "inLanguage": "pt-PT", "name": "Preço pedido vs. valor justo por modelo" },
        breadcrumbLd(host, [{ name: "Início", href: "/" }, { name: "Preço pedido vs. valor justo" }]),
      ],
    },
  });
}

// ═══ /mercado/indice — the market index, with a permanent weekly archive ═════
//
// Journalists and forums link to a number they can cite with a date. A page whose
// figures change under the link is not citable, so every week gets its OWN
// permanent URL (/mercado/indice/2026-W35) that never changes again, and the
// bare /mercado/indice always shows the latest plus the trend.
export function renderMarketIndex({ snapshot, history, host, depositCount, isArchive = false }) {
  const wk = snapshot.week;                 // display form, ISO: "2026-W35"
  // URL form is lower-case, because the router normalises every public path to
  // lower case and a canonical that disagreed with its own URL would 301 to
  // itself. The page still SHOWS the ISO spelling.
  const wkSlug = wk.toLowerCase();
  const canonical = isArchive
    ? `https://${host}/mercado/indice/${wkSlug}`
    : `https://${host}/mercado/indice`;
  const prev = history.filter(h => h.week < wk).sort((a, b) => a.week < b.week ? 1 : -1)[0] || null;
  const delta = (now, then, fmt) => {
    if (then == null || now == null || !then) return "";
    const d = (now - then) / then;
    if (Math.abs(d) < 0.005) return `<div class="s">estável vs. semana anterior</div>`;
    return `<div class="s" style="color:${d > 0 ? "#B4551F" : "#177A47"};">${d > 0 ? "+" : ""}${(d * 100).toFixed(1)}% vs. semana anterior</div>`;
  };
  const rows = history.slice().sort((a, b) => a.week < b.week ? 1 : -1).slice(0, 26).map(h => `<tr>
      <td>${h.week === wk && !isArchive ? `<b>${escapeHtml(h.week)}</b>` : `<a href="/mercado/indice/${escapeHtml(h.week.toLowerCase())}" style="color:#177A47;font-weight:600;">${escapeHtml(h.week)}</a>`}</td>
      <td class="mut">${escapeHtml(h.date || "")}</td>
      <td>${fmtEur(h.priceMed)}</td>
      <td class="mut">${fmtNum(h.listings)}</td>
      <td class="mut">${h.models}</td>
      <td class="mut">${h.sellMed != null ? h.sellMed + " dias" : "—"}</td></tr>`).join("");

  const body = crumbs(isArchive
    ? [{ name: "Início", href: "/" }, { name: "Índice de mercado", href: "/mercado/indice" }, { name: wk }]
    : [{ name: "Início", href: "/" }, { name: "Índice de mercado" }]) + `
    <section class="section fc-wrap" style="padding-top:16px;">
      <div class="eyebrow" style="margin-bottom:14px;"><span class="e-dot"></span><span class="mono">SEMANA ${escapeHtml(wk)} · ${escapeHtml(snapshot.date || "")}</span></div>
      <h1 class="fc-h1">Índice do mercado de usados em Portugal${isArchive ? ` — ${escapeHtml(wk)}` : ""}</h1>
      <p class="fc-p">Retrato semanal do que está à venda no OLX Portugal: quanto se pede, quanto há e quanto demora a sair.${isArchive ? " Este é o registo permanente desta semana — os números desta página não voltam a mudar." : " Cada semana fica guardada num endereço próprio, para poderes citar um número com data."}</p>
      <div class="fc-stat-row" style="margin:20px 0 8px;">
        <div class="fc-stat"><div class="k">PREÇO MEDIANO</div><div class="v">${fmtEur(snapshot.priceMed)}</div>${delta(snapshot.priceMed, prev && prev.priceMed)}</div>
        <div class="fc-stat"><div class="k">ANÚNCIOS ATIVOS</div><div class="v">${fmtNum(snapshot.listings)}</div>${delta(snapshot.listings, prev && prev.listings)}</div>
        <div class="fc-stat"><div class="k">MODELOS COBERTOS</div><div class="v">${snapshot.models}</div><div class="s">com amostra suficiente</div></div>
        <div class="fc-stat"><div class="k">DIAS ATÉ VENDER</div><div class="v">${snapshot.sellMed != null ? snapshot.sellMed : "—"}</div><div class="s">mediana do mercado</div></div>
        <div class="fc-stat"><div class="k">KM MEDIANO</div><div class="v">${snapshot.kmMed != null ? fmtNum(snapshot.kmMed) : "—"}</div><div class="s">à venda</div></div>
        <div class="fc-stat"><div class="k">DESVALORIZAÇÃO</div><div class="v">${snapshot.depMed != null ? Math.round(snapshot.depMed * 100) + "%" : "—"}</div><div class="s">por ano de idade</div></div>
      </div>
      ${provenance({ n: snapshot.listings, builtAt: snapshot.builtAt, measure: "Mediana das medianas de preço pedido por modelo" })}
    </section>
    ${rows ? `<section class="section fc-wrap">
      <h2 class="fc-h2">Histórico semanal</h2>
      <div class="fc-scroll"><table class="fc-tbl">
        <thead><tr><th>Semana</th><th>Data</th><th>Preço mediano</th><th>Anúncios</th><th>Modelos</th><th>Dias até vender</th></tr></thead>
        <tbody>${rows}</tbody></table></div>
      <p class="fc-p" style="margin-top:12px;">O histórico começa na semana em que passámos a guardar os cortes. Cresce uma linha por semana, e nenhuma linha antiga é reescrita.</p>
    </section>` : ""}
    <section class="section fc-wrap" style="padding-bottom:70px;">
      <h2 class="fc-h2">Podes citar isto</h2>
      <p class="fc-p">Estes números podem ser usados com atribuição a Carsbuyer e indicação da data — mudam todas as semanas, por isso a data faz parte do número. ${isArchive ? `Endereço permanente desta semana: <span class="mono fc-url">${escapeHtml(canonical)}</span>.` : ""}</p>
      <p class="fc-p"><a href="/precos">Preços por modelo</a> · <a href="/liquidez">Tempo de venda</a> · <a href="/sobrevalorizados">Pedido vs. valor justo</a> · <a href="/metodologia">Metodologia</a></p>
    </section>`;

  return layout({
    title: isArchive
      ? `Índice do mercado de usados em Portugal — ${wk}`
      : "Índice do mercado de carros usados em Portugal",
    description: `Semana ${wk}: preço mediano ${fmtEur(snapshot.priceMed)} em ${fmtNum(snapshot.listings)} anúncios ativos de ${snapshot.models} modelos no OLX Portugal${snapshot.sellMed != null ? `, ${snapshot.sellMed} dias medianos até vender` : ""}.`,
    canonical, body, zone: "all", nav: "feed", depositCount, index: true, host,
    jsonLd: {
      "@context": "https://schema.org",
      "@graph": [
        {
          "@type": "Dataset", "license": licenseUrl(host), "url": canonical, "inLanguage": "pt-PT",
          "name": `Índice do mercado de carros usados em Portugal — ${wk}`,
          "description": "Preço pedido mediano, número de anúncios ativos, quilometragem mediana e dias medianos até vender no mercado português de carros usados.",
          "creator": { "@type": "Organization", "name": "Flipper Club", "url": `https://${host}/` },
          "isAccessibleForFree": true,
          "temporalCoverage": snapshot.date || undefined,
          "dateModified": snapshot.builtAt || undefined,
          "variableMeasured": ["Preço pedido mediano (EUR)", "Anúncios ativos", "Dias até vender (mediana)"],
        },
        breadcrumbLd(host, isArchive
          ? [{ name: "Início", href: "/" }, { name: "Índice de mercado", href: "/mercado/indice" }, { name: wk }]
          : [{ name: "Início", href: "/" }, { name: "Índice de mercado" }]),
      ],
    },
  });
}

// ═══ Identity ════════════════════════════════════════════════════════════════
//
// Who stands behind the numbers is an E-E-A-T signal, and for a money topic its
// absence is a real deduction. But a name and a contact address are the site
// operator's to publish, not ours to invent: a made-up address is worse than
// none, because it is a promise the site cannot keep.
//
// So both come from env (SITE_AUTHOR / SITE_CONTACT_EMAIL in wrangler.toml
// [vars]); unset, the pages render the brand-level version and simply omit the
// blocks that would otherwise state something untrue.
let SITE_AUTHOR = "", SITE_CONTACT = "";
export function setSiteIdentity({ author, contact } = {}) {
  SITE_AUTHOR = (author || "").trim();
  SITE_CONTACT = (contact || "").trim();
}
function authorBlock() {
  if (!SITE_AUTHOR && !SITE_CONTACT) return "";
  return `<div class="exclusive" style="background:#FAFAF8;border:1px solid #EFECE6;align-items:flex-start;margin-top:8px;">
      <span style="font-size:15px;">✍️</span>
      <span class="x" style="color:#5B606B;">${SITE_AUTHOR ? `<b style="color:#16181D;">${escapeHtml(SITE_AUTHOR)}</b> mantém o Carsbuyer: recolhe os dados, escreve o código que os trata e responde por eles. ` : ""}${SITE_CONTACT ? `Erros, dúvidas de método e pedidos de dados: <a href="mailto:${escapeHtml(SITE_CONTACT)}" style="color:#177A47;font-weight:600;">${escapeHtml(SITE_CONTACT)}</a>.` : ""}</span>
    </div>`;
}

// ═══ /metodologia ════════════════════════════════════════════════════════════
//
// Every number on this site is an estimate, and the honest move is to publish
// where it comes from and where it stops working — including the thresholds that
// make us DROP a figure rather than show a weak one.
export function renderMethodology({ stats, host, depositCount, builtAt }) {
  const canonical = `https://${host}/metodologia`;
  const body = crumbs([{ name: "Início", href: "/" }, { name: "Metodologia" }]) + `
    <section class="section fc-wrap" style="padding-top:16px;">
      <h1 class="fc-h1">Como calculamos os preços</h1>
      <p class="fc-p">Sem caixa preta: aqui está de onde vêm os números, o que cada um mede, e em que casos preferimos não mostrar nada a mostrar um valor fraco.</p>

      <h2 class="fc-h2">1. De onde vêm os dados</h2>
      <p class="fc-p">Recolhemos diariamente os anúncios de automóveis do <b>OLX Portugal</b> e guardamos o histórico de cada um: preço, alterações de preço, e o dia em que o anúncio desaparece. Usamos apenas anúncios <b>ativos</b> no momento do cálculo${stats.listings ? `; hoje são ${fmtNum(stats.listings)} anúncios em ${stats.models} modelos` : ""}. Não compramos nem vendemos carros, não somos stand e não recebemos de nenhum vendedor.</p>

      <h2 class="fc-h2">2. O que é um "preço" aqui</h2>
      <p class="fc-p">É o <b>preço pedido</b> num anúncio ativo — não o preço a que o carro foi vendido. Ninguém em Portugal publica preços de transação, e inventá-los seria pior do que dizer o que temos. Preços pedidos e preços de venda não são a mesma coisa: a diferença costuma ser a margem de negociação, e é maior nos modelos que demoram a sair (ver <a href="/liquidez">tempo de venda</a>).</p>

      <h2 class="fc-h2">3. Mediana e intervalo, nunca um número sozinho</h2>
      <p class="fc-p">Para cada modelo mostramos a <b>mediana</b> (o valor que divide os anúncios ao meio) e o <b>intervalo interquartil P25-P75</b>, onde cabe metade dos anúncios. Usamos a mediana e não a média porque um único carro de coleção ou um anúncio com preço simbólico destrói uma média e não mexe numa mediana. O intervalo vai sempre junto: uma mediana sem dispersão parece uma precisão que não existe.</p>

      <h2 class="fc-h2">4. Quando é que uma página existe</h2>
      <p class="fc-p">Um número só é publicado quando a amostra por trás dele significa alguma coisa:</p>
      <ul class="fc-ul">
        <li class="fc-li"><b>20 anúncios ativos</b> — mínimo para um modelo ter página.</li>
        <li class="fc-li"><b>5 anúncios</b> — mínimo para uma linha por ano na tabela. Anos mais finos são juntados em intervalos de dois ou mais anos, ou omitidos e contados no rodapé da tabela.</li>
        <li class="fc-li"><b>${MIN_YEAR_PAGE_N} anúncios</b> — mínimo para um ano ter <b>página própria</b>. Abaixo disso, um único anúncio fora do normal move a mediana mais do que a diferença entre anos que estaríamos a afirmar.</li>
        <li class="fc-li"><b>${DEP_MIN_CELLS} anos com amostra e ${DEP_MIN_SPAN} anos de intervalo</b> — mínimo para uma <a href="/depreciacao">curva de desvalorização</a>, mais um ajuste que explique de facto os pontos (R² ≥ ${DEP_MIN_R2}).</li>
      </ul>

      <h2 class="fc-h2">5. O "valor justo estimado"</h2>
      <p class="fc-p">Além dos preços pedidos, treinamos um modelo estatístico (gradient boosting) que estima o valor de um carro a partir de ano, quilometragem, motorização, versão, distrito e outras características. Nas páginas de modelo, esse valor é a <b>mediana das estimativas dos anúncios reais</b> daquele modelo — não a estimativa de um carro-tipo inventado.</p>
      <p class="fc-p">Não o publicamos sempre. A estimativa é <b>retirada</b> quando:</p>
      <ul class="fc-ul">
        <li class="fc-li">o preço fica abaixo de <b>5 000 €</b> — no fundo do mercado o modelo sobrestima sistematicamente;</li>
        <li class="fc-li">o preço fica acima de <b>45 000 €</b> — no topo o modelo satura e carros muito diferentes colapsam no mesmo valor;</li>
        <li class="fc-li">a estimativa discorda demasiado dos anúncios reais (fora de 0,70x a 1,40x da mediana pedida, ou fora do contexto do intervalo P25-P75);</li>
        <li class="fc-li">faltam características suficientes nos anúncios para o modelo distinguir versões.</li>
      </ul>
      <p class="fc-p">Nesses casos a página mostra só os preços pedidos. Preferimos uma página com menos números a uma página com um número errado.</p>

      <h2 class="fc-h2">6. Dias até vender</h2>
      <p class="fc-p">Acompanhamos cada anúncio até desaparecer e medimos a <b>mediana de dias entre publicação e remoção</b>. Um anúncio pode desaparecer por venda ou por desistência, e não conseguimos distinguir os dois — por isso a leitura correta é "tempo até sair do mercado", que na prática é o que interessa a quem anuncia.</p>

      <h2 class="fc-h2">7. Desvalorização</h2>
      <p class="fc-p">Ajustamos uma reta ao <b>logaritmo</b> da mediana de preço contra o ano de fabrico, e reportamos a taxa anual daí resultante. Logaritmo porque a desvalorização é uma percentagem do que resta: em euros perde-se muito mais no primeiro ano do que no nono, e uma reta sobre euros misturaria as duas coisas. Como é um corte transversal de anos diferentes hoje (e não o mesmo carro seguido ao longo do tempo), a leitura correta é "quanto custa a mais um ano mais recente", não "quanto vou perder no próximo ano".</p>

      <h2 class="fc-h2">8. O que isto não faz</h2>
      <ul class="fc-ul">
        <li class="fc-li">Não avalia a <b>tua</b> viatura. A mediana de um modelo não sabe do teu histórico, dos teus extras nem do estado da tua embraiagem. Para o carro concreto, <a href="/avaliar">avalia o anúncio</a>.</li>
        <li class="fc-li">Não distingue carros <b>importados por legalizar</b> na mediana do modelo. Um preço muito abaixo do normal costuma ter ISV por pagar — e o <a href="/isv">ISV</a> pode valer milhares.</li>
        <li class="fc-li">Não cobre carros vendidos fora do OLX (stands com stock próprio, particulares em grupos fechados, leilões).</li>
        <li class="fc-li">Não é aconselhamento financeiro nem uma avaliação para efeitos legais ou de seguro.</li>
      </ul>

      <h2 class="fc-h2" id="licenca">9. Reutilização e atribuição</h2>
      <p class="fc-p">Os números destas páginas são estatísticas nossas, calculadas a partir de anúncios públicos do OLX Portugal. <b>Podes citá-los e reutilizá-los</b>, desde que a fonte seja atribuída ao Carsbuyer e seja indicada a data de recolha: a mediana de um modelo muda ao longo do tempo, e uma citação sem data deixa de ser verificável. Se precisares dos valores sem a marcação da página, cada modelo publica o mesmo conteúdo em JSON, bastando acrescentar <b>.json</b> ao endereço, como em <a href="/preco/opel-corsa.json">/preco/opel-corsa.json</a>.</p>

      <h2 class="fc-h2">10. Correções</h2>
      <p class="fc-p">Se um número parece errado, provavelmente vale a pena olhar: a amostra pode estar contaminada por anúncios repetidos ou por uma versão mal classificada. Todas as páginas indicam o tamanho da amostra e a data de recolha, para que qualquer afirmação nossa seja verificável.</p>
      ${authorBlock()}
      ${provenance({ n: stats.listings, builtAt, measure: "Preço pedido em anúncios ativos (mediana e P25-P75)" })}
      <p class="fc-p" style="margin-top:18px;"><a href="/sobre">Quem faz isto</a> · <a href="/precos">Preços por modelo</a> · <a href="/mercado/indice">Índice de mercado</a></p>
    </section>
    <div style="height:60px;"></div>`;
  return layout({
    title: "Metodologia: como calculamos os preços",
    description: "De onde vêm os dados, o que é a mediana e o intervalo P25-P75, quando publicamos um valor justo estimado e quando o retiramos. Método completo do Carsbuyer.",
    canonical, body, zone: "all", nav: null, depositCount, index: true, host,
    jsonLd: {
      "@context": "https://schema.org",
      "@graph": [
        {
          "@type": "TechArticle", "headline": "Como calculamos os preços de carros usados em Portugal",
          "url": canonical, "inLanguage": "pt-PT", "dateModified": builtAt || undefined,
          "publisher": { "@type": "Organization", "name": "Flipper Club", "url": `https://${host}/` },
          ...(SITE_AUTHOR ? { "author": { "@type": "Person", "name": SITE_AUTHOR } } : {}),
        },
        breadcrumbLd(host, [{ name: "Início", href: "/" }, { name: "Metodologia" }]),
      ],
    },
  });
}

// ═══ /sobre ══════════════════════════════════════════════════════════════════
export function renderAbout({ stats, host, depositCount, builtAt }) {
  const canonical = `https://${host}/sobre`;
  const body = crumbs([{ name: "Início", href: "/" }, { name: "Quem somos" }]) + `
    <section class="section fc-wrap" style="padding-top:16px;">
      <h1 class="fc-h1">Quem faz isto, e porquê</h1>
      <p class="fc-p">O Carsbuyer é um projeto independente que mede o mercado português de carros usados a partir dos anúncios que estão de facto à venda. Nasceu de uma pergunta simples que ninguém em Portugal respondia com números: <i>quanto vale mesmo este carro?</i></p>

      <h2 class="fc-h2">Independentes de quem?</h2>
      <p class="fc-p">Não somos stand, não somos intermediário e não representamos nenhum vendedor. Não temos carros para colocar, por isso não temos motivo para inflacionar nem para desvalorizar nenhum modelo. Os números que publicamos são os mesmos que usamos para as nossas próprias decisões — se estivessem enviesados, seríamos os primeiros prejudicados.</p>

      <h2 class="fc-h2">Como nos pagamos</h2>
      <p class="fc-p">As avaliações e os preços por modelo são gratuitos e ficam gratuitos. O que se paga é outra coisa: no <a href="/mercado">mercado</a> listamos anúncios que estão abaixo do valor justo, e um depósito reembolsável de 5 € desbloqueia o contacto do vendedor de um desses carros e reserva-o durante 24 horas. O depósito volta para ti. Não vendemos os teus dados, não temos publicidade paga por marcas e não aceitamos pagamento para mexer numa avaliação.</p>

      <h2 class="fc-h2">O que temos hoje</h2>
      <ul class="fc-ul">
        <li class="fc-li"><b>${stats.models}</b> modelos com amostra suficiente para publicar preços${stats.listings ? `, sobre ${fmtNum(stats.listings)} anúncios ativos` : ""}.</li>
        <li class="fc-li">Preço mediano por modelo e <b>por ano de fabrico</b>, sempre com o intervalo onde cabe metade dos anúncios.</li>
        <li class="fc-li"><a href="/liquidez">Tempo mediano até vender</a> — medido em anúncios reais, não estimado.</li>
        <li class="fc-li"><a href="/depreciacao">Curvas de desvalorização</a> para os modelos com histórico suficiente.</li>
        <li class="fc-li">Um <a href="/mercado/indice">índice semanal do mercado</a>, com registo permanente de cada semana.</li>
      </ul>

      <h2 class="fc-h2">Erramos?</h2>
      <p class="fc-p">Sim, e por isso publicamos o <a href="/metodologia">método completo</a>, o tamanho de cada amostra e a data de recolha em todas as páginas. Quando um número não é fiável, retiramo-lo em vez de o disfarçar — há modelos onde verás preços pedidos e nenhuma estimativa de valor justo, e isso é intencional.</p>

      <h2 class="fc-h2">Podes usar os nossos números</h2>
      <p class="fc-p">Com atribuição ao Carsbuyer e a data — os valores mudam todos os dias. Cada página de modelo tem uma <b>versão em JSON</b> ligada no cabeçalho, e há um <a href="/llms.txt">llms.txt</a> com a estrutura completa para quem lê o site com ferramentas automáticas.</p>
      ${authorBlock()}
      ${provenance({ n: stats.listings, builtAt, measure: "Preço pedido em anúncios ativos (mediana e P25-P75)" })}
      <p class="fc-p" style="margin-top:18px;"><a href="/metodologia">Metodologia</a> · <a href="/precos">Preços por modelo</a> · <a href="/avaliar">Avaliar um carro</a></p>
    </section>
    <div style="height:60px;"></div>`;
  return layout({
    title: "Quem somos — avaliação independente de usados",
    description: `Projeto independente que mede o mercado português de carros usados a partir de anúncios ativos do OLX: ${stats.models} modelos, método publicado, sem ligação a stands.`,
    canonical, body, zone: "all", nav: null, depositCount, index: true, host,
    jsonLd: {
      "@context": "https://schema.org",
      "@graph": [
        {
          "@type": "AboutPage", "url": canonical, "inLanguage": "pt-PT", "name": "Quem somos",
          "mainEntity": {
            "@type": "Organization", "name": "Flipper Club", "alternateName": "Carsbuyer",
            "url": `https://${host}/`, "areaServed": "PT",
            "description": "Avaliação independente de carros usados em Portugal a partir de anúncios ativos do OLX.",
            ...(SITE_CONTACT ? { "email": SITE_CONTACT } : {}),
            ...(SITE_AUTHOR ? { "founder": { "@type": "Person", "name": SITE_AUTHOR } } : {}),
          },
        },
        breadcrumbLd(host, [{ name: "Início", href: "/" }, { name: "Quem somos" }]),
      ],
    },
  });
}

// ═══ /isv — imported-car tax estimator ═══════════════════════════════════════
//
// "simulador ISV" is a large query and the reason importing looks cheap until it
// isn't: a €9 000 German car with €4 000 of ISV is not a €9 000 car. We already
// compute this server-side for flagged imports on /mercado; this exposes the
// same tables as a calculator, next to the one thing the ISV simulators do not
// have — what that model actually costs in Portugal today.
//
// TABLES ARE LOCK-STEP WITH src/analytics/isv.py. Same brackets, same order,
// same age-reduction table. If one changes, change both (paired-comment pact,
// like slugify ↔ model_pages.slugify). Verified against Autoridade Tributária
// 2026 rates, which are unchanged from 2024/2025.
const ISV_TABLES = {
  cc: [[1000, 1.09, 849.03], [1250, 1.18, 850.69], [null, 5.61, 6194.88]],
  co2: {
    "WLTP:petrol": [[110, 0.44, 43.02], [115, 1.10, 115.80], [120, 1.38, 147.79],
      [130, 5.27, 619.17], [145, 6.38, 762.73], [175, 41.54, 5819.56],
      [195, 51.38, 7247.39], [235, 193.01, 34190.52], [null, 233.81, 41910.96]],
    "WLTP:diesel": [[110, 1.72, 11.50], [120, 18.96, 1906.19], [140, 65.04, 7360.85],
      [150, 127.40, 16080.57], [160, 160.81, 21176.06], [170, 221.69, 29227.38],
      [190, 274.08, 36987.98], [null, 282.35, 38271.32]],
    "NEDC:petrol": [[99, 4.62, 427.00], [115, 8.09, 750.99], [145, 52.56, 5903.94],
      [175, 61.24, 7140.17], [195, 155.97, 23627.27], [null, 205.65, 33390.12]],
    "NEDC:diesel": [[79, 5.78, 439.04], [95, 23.45, 1848.58], [120, 79.22, 7195.63],
      [140, 175.73, 18924.92], [160, 195.43, 21720.92], [null, 268.42, 33447.90]],
  },
  particulas: 500,
  reducao: [[1, 0.10], [2, 0.20], [3, 0.28], [4, 0.35], [5, 0.43], [6, 0.52],
    [7, 0.60], [8, 0.65], [9, 0.70], [10, 0.75], [null, 0.80]],
};

/**
 * ISV estimate. Pure, exported, and SERIALISED INTO THE PAGE via toString()
 * below — so the number the browser shows and the number Node tests are
 * produced by the same function, not by two copies that drift.
 *
 * Mirrors src/analytics/isv.py::compute_isv, including its refusals: a BEV is
 * exempt (0), a PHEV returns null because its reduced regime needs per-car
 * eligibility we do not have, and missing cilindrada/CO2 returns null rather
 * than a guess.
 *
 * Self-contained on purpose: it receives the tables as an argument and calls
 * nothing from module scope, because its source is what ships to the browser.
 */
export function estimateIsv(T, { cc, co2, fuel, regYear, asOfYear, isEu }) {
  if (fuel === "bev") return { exempt: true, isv: 0 };
  if (fuel === "phev") return null;
  if (!(cc > 0) || !(co2 > 0) || !(regYear > 0)) return null;
  var bracket = function (v, tbl) {
    for (var i = 0; i < tbl.length; i++) if (tbl[i][0] === null || v <= tbl[i][0]) return tbl[i];
    return tbl[tbl.length - 1];
  };
  var fclass = (fuel === "diesel") ? "diesel" : "petrol";
  var cycle = (regYear <= 2019) ? "NEDC" : "WLTP";
  var bcc = bracket(cc, T.cc);
  var compCc = Math.max(0, cc * bcc[1] - bcc[2]);
  var bco2 = bracket(co2, T.co2[cycle + ":" + fclass]);
  var compCo2 = Math.max(0, co2 * bco2[1] - bco2[2]);
  var part = (fclass === "diesel") ? T.particulas : 0;
  var gross = compCc + compCo2 + part;
  var age = Math.max(0, asOfYear - regYear);
  var red = 0;
  if (isEu) {
    for (var j = 0; j < T.reducao.length; j++) {
      if (T.reducao[j][0] === null || age <= T.reducao[j][0]) { red = T.reducao[j][1]; break; }
    }
  }
  return {
    exempt: false, cycle: cycle, age: age, reduction: red,
    cilindrada: compCc, co2: compCo2, particulas: part,
    gross: gross, isv: gross * (1 - red),
    ambiguousCycle: regYear >= 2018 && regYear <= 2019,
  };
}

// Exported only so the tests can drive estimateIsv() with the shipped tables.
export const ISV_TABLES_FOR_TEST = ISV_TABLES;

export function renderIsv({ topModels, host, depositCount, builtAt, refYear }) {
  const canonical = `https://${host}/isv`;
  const year = refYear || new Date().getUTCFullYear();
  const years = [];
  for (let y = year; y >= year - 25; y--) years.push(y);
  const modelLinks = (topModels || []).slice(0, 12).map(m =>
    `<a class="mchip" href="/preco/${encodeURIComponent(m.slug)}">${escapeHtml(m.b)} ${escapeHtml(m.m)} <span class="mut">${fmtEur(m.fm)}</span></a>`).join("");

  const body = crumbs([{ name: "Início", href: "/" }, { name: "Simulador ISV" }]) + `
    <section class="section fc-wrap" style="padding-top:16px;">
      <h1 class="fc-h1">Simulador de ISV: quanto custa legalizar um carro importado</h1>
      <p class="fc-p">O ISV é a fatura que transforma um bom negócio na Alemanha num mau negócio em Portugal. Calcula-se a partir da <b>cilindrada</b>, das <b>emissões de CO2</b> e da <b>idade</b> do carro — e para um usado com alguns anos a redução por idade corta uma boa parte.</p>
      <div class="fc-form">
        <div class="fc-field"><label for="isv-cc">Cilindrada (cm³)</label><input id="isv-cc" type="number" inputmode="numeric" min="1" max="9000" step="1" placeholder="1598"></div>
        <div class="fc-field"><label for="isv-co2">CO2 (g/km)</label><input id="isv-co2" type="number" inputmode="numeric" min="1" max="500" step="1" placeholder="110"></div>
        <div class="fc-field"><label for="isv-fuel">Combustível</label><select id="isv-fuel">
          <option value="petrol">Gasolina</option>
          <option value="diesel">Gasóleo (diesel)</option>
          <option value="petrol-hybrid">Híbrido (não plug-in)</option>
          <option value="petrol-lpg">GPL</option>
          <option value="bev">Elétrico</option>
          <option value="phev">Híbrido plug-in</option>
        </select></div>
        <div class="fc-field"><label for="isv-year">1.ª matrícula</label><select id="isv-year">${years.map(y => `<option value="${y}">${y}</option>`).join("")}</select></div>
        <div class="fc-field"><label for="isv-eu">Matriculado na UE</label><select id="isv-eu"><option value="1">Sim</option><option value="0">Não (sem redução)</option></select></div>
      </div>
      <div class="fc-out" id="isv-out" aria-live="polite">
        <div class="cap">Preenche a cilindrada e o CO2</div>
        <div class="big">—</div>
      </div>
      <p class="fc-p" style="margin-top:14px;">A cilindrada e o CO2 estão no certificado de conformidade e no documento único do carro. Para carros de 2018 e 2019 o ciclo de medição (NEDC ou WLTP) é ambíguo pelo ano — assumimos NEDC, o que pode subestimar ou sobrestimar; confirma no documento.</p>

      <h2 class="fc-h2">O que este número inclui e o que não inclui</h2>
      <ul class="fc-ul">
        <li class="fc-li"><b>Inclui:</b> componente de cilindrada, componente ambiental (CO2), taxa de partículas de 500 € nos gasóleos, e a redução por anos de uso da Tabela D (só para carros matriculados na UE).</li>
        <li class="fc-li"><b>Não inclui:</b> transporte, inspeção, legalização e chapas, IUC do ano, eventual IVA em carros novos ou quase-novos, nem a margem de quem vende.</li>
        <li class="fc-li"><b>Híbridos plug-in</b> têm um regime reduzido que depende da autonomia elétrica e do CO2 homologado; não o calculamos porque exige dados por viatura que não temos. <b>Elétricos</b> estão isentos.</li>
      </ul>

      <h2 class="fc-h2">A pergunta que interessa: compensa importar?</h2>
      <p class="fc-p">O ISV sozinho não responde. O que responde é: <b>preço lá fora + ISV + transporte + legalização</b> contra <b>o que esse modelo custa em Portugal hoje</b>. A segunda metade dessa conta é o que medimos todos os dias:</p>
      <div class="mchips">${modelLinks}</div>
      <p class="fc-p" style="margin-top:14px;"><a href="/precos">Ver preço de qualquer modelo em Portugal&nbsp;→</a></p>
      ${provenance({ n: null, builtAt, measure: "Tabelas de ISV 2026 (Código do ISV, art.º 7.º e 11.º)", extra: "Estimativa, não vinculativa" })}
      <p class="fc-p" style="margin-top:18px;">Estimativa indicativa a partir das tabelas em vigor. O valor liquidado pela Autoridade Tributária no processo de admissão é o que conta.</p>
    </section>
    <script>
    (function(){
      var T = ${JSON.stringify(ISV_TABLES)};
      var el = function(id){ return document.getElementById(id); };
      var out = el('isv-out');
      if(!out) return;
      var estimateIsv = ${estimateIsv.toString()};
      function eur(n){ return '€' + Math.round(n).toLocaleString('pt-PT'); }
      function calc(){
        var fuel = el('isv-fuel').value;
        var r = estimateIsv(T, {
          cc: parseFloat(el('isv-cc').value),
          co2: parseFloat(el('isv-co2').value),
          fuel: fuel,
          regYear: parseInt(el('isv-year').value, 10),
          asOfYear: ${year},
          isEu: el('isv-eu').value === '1',
        });
        if (r && r.exempt){
          out.innerHTML = '<div class="cap">Veículo elétrico</div><div class="big">€0</div>'
            + '<div class="mono" style="font-size:12px;color:#5B606B;margin-top:6px;">Isento de ISV.</div>';
          return;
        }
        if (!r && fuel === 'phev'){
          out.innerHTML = '<div class="cap">Híbrido plug-in</div><div class="big">—</div>'
            + '<div class="mono" style="font-size:12px;color:#5B606B;margin-top:6px;">Regime reduzido dependente da autonomia elétrica e do CO2 homologado. Não estimamos para não dar um número errado.</div>';
          return;
        }
        if (!r){
          out.innerHTML = '<div class="cap">Preenche a cilindrada e o CO2</div><div class="big">—</div>';
          return;
        }
        out.innerHTML =
          '<div class="cap">ISV estimado · ' + r.cycle + ' · ' + r.age + ' ano' + (r.age===1?'':'s') + ' de uso</div>' +
          '<div class="big">' + eur(r.isv) + '</div>' +
          '<div class="mono" style="font-size:12.5px;color:#5B606B;margin-top:10px;line-height:1.7;">' +
          'Cilindrada ' + eur(r.cilindrada) + ' · CO2 ' + eur(r.co2) +
          (r.particulas ? ' · Partículas ' + eur(r.particulas) : '') +
          '<br>Bruto ' + eur(r.gross) + (r.reduction ? ' − ' + Math.round(r.reduction*100) + '% por idade' : ' · sem redução (fora da UE)') +
          (r.ambiguousCycle ? '<br>Atenção: 2018-2019 é ambíguo entre NEDC e WLTP — confirma no documento.' : '') +
          '</div>';
      }
      ['isv-cc','isv-co2','isv-fuel','isv-year','isv-eu'].forEach(function(id){
        var n = el(id); if(!n) return;
        n.addEventListener('input', calc); n.addEventListener('change', calc);
      });
      calc();
    })();
    </script>
    <div style="height:60px;"></div>`;

  const faqs = [
    ["Como se calcula o ISV de um carro importado?",
     "O ISV soma uma componente de cilindrada e uma componente ambiental de CO2, mais 500 € de taxa de partículas nos gasóleos. Ao total aplica-se a redução por anos de uso da Tabela D, que vai de 10% no primeiro ano até 80% a partir dos dez anos, e que só se aplica a carros já matriculados na União Europeia."],
    ["Um carro elétrico paga ISV em Portugal?",
     "Não. Os veículos exclusivamente elétricos estão isentos de ISV. Os híbridos plug-in têm um regime reduzido que depende da autonomia elétrica e do CO2 homologado de cada viatura."],
    ["Compensa importar um carro da Alemanha para Portugal?",
     "Depende da conta completa: preço lá fora mais ISV, transporte e legalização, contra o que esse modelo custa em Portugal hoje. O ISV é a parcela que costuma ser subestimada, e num gasóleo com CO2 alto pode ultrapassar metade do preço de compra."],
  ];

  return layout({
    title: "Simulador de ISV 2026 para carro importado",
    description: "Calcula o ISV de um carro importado a partir da cilindrada, do CO2 e da idade, com as tabelas em vigor. Depois compara com o que esse modelo custa em Portugal hoje.",
    canonical, body, zone: "all", nav: null, depositCount, index: true, host,
    jsonLd: {
      "@context": "https://schema.org",
      "@graph": [
        {
          "@type": "WebApplication", "name": "Simulador de ISV", "url": canonical,
          "applicationCategory": "FinanceApplication", "operatingSystem": "Web",
          "inLanguage": "pt-PT", "isAccessibleForFree": true,
          "offers": { "@type": "Offer", "price": "0", "priceCurrency": "EUR" },
          "description": "Estimativa do Imposto Sobre Veículos para admissão de um carro usado importado em Portugal.",
        },
        breadcrumbLd(host, [{ name: "Início", href: "/" }, { name: "Simulador ISV" }]),
        faqLd(faqs),
      ],
    },
  });
}

// ═══ Machine-readable twins ══════════════════════════════════════════════════
//
// /preco/{slug}.json and /preco/{slug}/{ano}.json — the same figures the HTML
// shows, without the markup.
//
// This is the cheapest thing on the list and probably the highest-leverage for
// the one channel where being the ORIGINAL source of a number is the whole
// advantage: an answer engine that can take the figures in one parse, with the
// sample size and the date attached, cites them with attribution far more often
// than one that has to infer them from a table. Every field is named in full —
// the blob's two-letter keys are a size optimisation for our own transport, not
// an interface anyone else should have to decode.
export function modelJson(rec, slug, { host, builtAt }) {
  const base = `https://${host}`;
  return {
    source: "Carsbuyer",
    source_url: `${base}/preco/${slug}`,
    licence: "Citação permitida com atribuição a Carsbuyer e indicação da data.",
    measured: "asking_price",
    measured_note: "Preços PEDIDOS em anúncios ativos do OLX Portugal, não preços de venda fechados.",
    collected_until: (builtAt || "").slice(0, 10) || null,
    updated_at: builtAt || null,
    market: "PT", currency: "EUR",
    brand: rec.b, model: rec.m, slug,
    sample_size: rec.n,
    asking_price: { median: rec.fm, p25: rec.fl, p75: rec.fh },
    fair_value_estimate: rec.gm != null
      ? { median: rec.gm, low: rec.gl, high: rec.gh,
          note: "Estimativa do nosso modelo para specs típicas deste modelo; publicada apenas quando passa os limites de fiabilidade descritos em /metodologia." }
      : null,
    mileage_km_median: rec.kmm != null ? rec.kmm : null,
    model_years: (rec.y0 && rec.y1) ? { from: rec.y0, to: rec.y1 } : null,
    fuel_mix: Array.isArray(rec.fu) ? rec.fu.map(([f, share]) => ({ fuel: f, share })) : null,
    days_to_sell: rec.sd != null ? { median_days: rec.sd, sample_size: rec.sn } : null,
    by_year: (rec.yr || []).map(c => ({
      year: c.y, sample_size: c.n,
      asking_price: { median: c.fm, p25: c.fl, p75: c.fh },
      fair_value_estimate: c.gm != null ? { median: c.gm, low: c.gl, high: c.gh } : null,
      mileage_km_median: c.km != null ? c.km : null,
      page: (typeof c.y === "number" && c.n >= MIN_YEAR_PAGE_N) ? `${base}/preco/${slug}/${c.y}` : null,
    })),
    years_omitted_thin_sample: rec.yt || 0,
    related: {
      depreciation: depreciationOk(rec) ? `${base}/depreciacao/${slug}` : null,
      methodology: `${base}/metodologia`,
    },
  };
}

export function yearJson(rec, slug, year, cell, { host, builtAt }) {
  const base = `https://${host}`;
  return {
    source: "Carsbuyer",
    source_url: `${base}/preco/${slug}/${year}`,
    licence: "Citação permitida com atribuição a Carsbuyer e indicação da data.",
    measured: "asking_price",
    measured_note: "Preços PEDIDOS em anúncios ativos do OLX Portugal, não preços de venda fechados.",
    collected_until: (builtAt || "").slice(0, 10) || null,
    updated_at: builtAt || null,
    market: "PT", currency: "EUR",
    brand: rec.b, model: rec.m, slug, year,
    sample_size: cell.n,
    asking_price: { median: cell.fm, p25: cell.fl, p75: cell.fh },
    fair_value_estimate: cell.gm != null ? { median: cell.gm, low: cell.gl, high: cell.gh } : null,
    mileage_km_median: cell.km != null ? cell.km : null,
    share_of_model_listings: rec.n ? Math.round((cell.n / rec.n) * 1000) / 1000 : null,
    related: { model: `${base}/preco/${slug}`, methodology: `${base}/metodologia` },
  };
}

// ═══ Facets: /preco/{slug}/{combustivel} and /preco/{slug}/{distrito} ════════
//
// Two query clusters the model page cannot win because it answers them mixed
// together: "Golf diesel usado preço" and "carros usados Porto preços".
//
// Both are gated on the blob carrying `fx` / `dt` cells (added to
// model_pages.py alongside this). Until the pipeline publishes them these
// functions return nothing and the routes 404 — the pages appear by themselves
// on the next build, with no deploy.

/** Facet cells of one kind: "fuel" → rec.fx, "district" → rec.dt. */
export function facetCells(rec, kind) {
  const arr = kind === "fuel" ? rec.fx : rec.dt;
  return Array.isArray(arr) ? arr : [];
}

/** Find a facet cell by key, or null. */
export function facetCell(rec, kind, key) {
  return facetCells(rec, kind).find(c => c.k === key) || null;
}

/** Which facet kind a path segment belongs to, if any. */
export function facetKind(rec, key) {
  if (facetCell(rec, "fuel", key)) return "fuel";
  if (facetCell(rec, "district", key)) return "district";
  return null;
}

/** Every facet URL segment this model publishes (for the sitemap). */
export function facetKeys(rec) {
  return [...facetCells(rec, "fuel"), ...facetCells(rec, "district")].map(c => c.k);
}

export function renderFacetPage({ rec, slug, kind, cell, siblingsCells, stats, host, depositCount, builtAt }) {
  const B = escapeHtml(rec.b), M = escapeHtml(rec.m);
  const label = escapeHtml(cell.lbl);
  const canonical = `https://${host}/preco/${slug}/${cell.k}`;
  const isFuel = kind === "fuel";
  // "um Golf diesel" vs "um Golf no Porto" — the preposition is the difference
  // between a sentence and a slot-filled template.
  const phrase = isFuel ? `${B} ${M} ${label.toLowerCase()}` : `${B} ${M} no distrito de ${label}`;
  const titlePhrase = isFuel ? `${rec.b} ${rec.m} ${cell.lbl.toLowerCase()}` : `${rec.b} ${rec.m} em ${cell.lbl}`;
  const share = rec.n ? Math.round(cell.n / rec.n * 100) : null;
  const vsAll = rec.fm > 0 ? (cell.fm - rec.fm) / rec.fm : null;

  let pin = 50;
  if (cell.fh > cell.fl) pin = Math.max(6, Math.min(94, Math.round((cell.fm - cell.fl) / (cell.fh - cell.fl) * 100)));

  // The comparison that makes this page worth existing: this facet against the
  // model's other facets of the same kind. "Diesel or petrol, which holds its
  // price" is answered everywhere with opinion and nowhere with a number.
  const others = siblingsCells.filter(c => c.k !== cell.k);
  const compare = others.map(o => {
    const d = (cell.fm - o.fm) / o.fm;
    const href = `/preco/${slug}/${o.k}`;
    return `<li>Contra <a href="${href}">${escapeHtml(o.lbl)}</a> (${o.n} anúncios, mediana ${fmtEur(o.fm)}): ${Math.abs(Math.round(d * 100))}% ${d >= 0 ? "mais caro" : "mais barato"}${o.km != null && cell.km != null ? `, com ${fmtKm(Math.abs(cell.km - o.km))} ${cell.km > o.km ? "a mais" : "a menos"} de quilometragem mediana` : ""}.</li>`;
  }).join("");

  const body = crumbs([
    { name: "Início", href: "/" }, { name: "Preços", href: "/precos" },
    { name: `${rec.b} ${rec.m}`, href: `/preco/${slug}` }, { name: cell.lbl },
  ]) + `
    <div style="padding-top:14px;">
      <div class="side-card" style="max-width:680px;margin:0 auto;">
        <div class="eyebrow" style="margin-bottom:14px;"><span class="e-dot"></span><span class="mono">${escapeHtml(cell.lbl).toUpperCase()} · OLX PORTUGAL</span></div>
        <h1 class="fc-h1">Quanto vale um ${phrase} usado?</h1>
        <p class="lede" style="font-size:16px;margin:0 0 20px;">Em <b>${cell.n} anúncios ativos</b>${isFuel ? "" : " no distrito"}, um ${phrase} pede em mediana <b>${fmtEur(cell.fm)}</b>${cell.km != null ? `, com ${fmtKm(cell.km)} medianos` : ""}${cell.y0 && cell.y1 ? `, para anos ${cell.y0}-${cell.y1}` : ""}.</p>
        <div class="side-prices">
          <div><div class="cap">Preço mediano (pedido)</div><div class="big">${fmtEur(cell.fm)}</div></div>
          <div class="side-fair"><div class="cap">${cell.n} anúncios${share != null ? ` · ${share}% do modelo` : ""}</div><div class="v">${cell.km != null ? fmtKm(cell.km) : "—"}</div></div>
        </div>
        <div style="margin-top:16px;">
          <div class="gauge-head"><span>${fmtEur(cell.fl)}</span><span>intervalo típico (50% dos anúncios)</span><span>${fmtEur(cell.fh)}</span></div>
          <div class="gauge-track"><div class="gauge-pin" style="left:${pin}%;"></div></div>
        </div>
        ${provenance({ n: cell.n, builtAt, measure: `Preço pedido, ${titlePhrase} (mediana e P25-P75)` })}
      </div>
    </div>
    ${(compare || vsAll != null) ? `
    <section class="section fc-wrap">
      <h2 class="fc-h2">${isFuel ? "Contra as outras motorizações" : "Contra o resto do país"}</h2>
      <ul class="fc-insights">
        ${vsAll != null ? `<li>Face a todos os ${B} ${M} do país (mediana ${fmtEur(rec.fm)}), este corte pede <b>${Math.abs(Math.round(vsAll * 100))}% ${vsAll >= 0 ? "mais" : "menos"}</b>.</li>` : ""}
        ${compare}
      </ul>
    </section>` : ""}
    ${siblingsCells.length > 1 ? `
    <section class="section fc-wrap">
      <div class="sec-label" style="margin-bottom:10px;">${isFuel ? "OUTRAS MOTORIZAÇÕES" : "OUTROS DISTRITOS"}</div>
      <div class="fc-yearlinks">${siblingsCells.map(c =>
        c.k === cell.k ? `<a class="on" href="/preco/${slug}/${c.k}">${escapeHtml(c.lbl)}</a>`
                       : `<a href="/preco/${slug}/${c.k}">${escapeHtml(c.lbl)}</a>`).join("")}</div>
    </section>` : ""}
    <section class="section fc-wide">
      <div class="cta-banner">
        <div style="flex:1 1 360px;">
          <h2>Tens um ${phrase}?</h2>
          <p>Esta é a mediana do corte. Cola o link do teu anúncio e dizemos o valor justo desse carro em concreto.</p>
        </div>
        <a class="btn-bright" href="/avaliar?modelo=${encodeURIComponent(slug)}">Avaliar o meu carro&nbsp;&nbsp;→</a>
      </div>
    </section>
    <section class="section fc-wrap" style="padding-bottom:70px;">
      <p class="fc-p"><a href="/preco/${slug}">Todos os ${B} ${M}</a>${isFuel ? "" : ` · <a href="/precos/${cell.k}">Carros usados ${emDistrito(cell.k, escapeHtml(cell.lbl))}</a>`} · <a href="/precos">Todos os modelos</a></p>
    </section>`;

  const faqs = [[
    `Quanto vale um ${titlePhrase} usado?`,
    `Em ${cell.n} anúncios ativos no OLX Portugal, um ${titlePhrase} pede em mediana ${fmtEur(cell.fm)}, com metade dos anúncios entre ${fmtEur(cell.fl)} e ${fmtEur(cell.fh)}. São preços pedidos, não preços de venda fechados.`,
  ]];
  if (others.length) {
    const o = others[0];
    const d = Math.round((cell.fm - o.fm) / o.fm * 100);
    faqs.push([
      isFuel ? `${rec.b} ${rec.m}: ${cell.lbl.toLowerCase()} ou ${o.lbl.toLowerCase()}?`
             : `Um ${rec.b} ${rec.m} é mais caro em ${cell.lbl} ou em ${o.lbl}?`,
      `A mediana pedida é ${fmtEur(cell.fm)} para ${cell.lbl.toLowerCase()} e ${fmtEur(o.fm)} para ${o.lbl.toLowerCase()}, uma diferença de ${Math.abs(d)}%. A comparação é entre anúncios ativos do mesmo modelo, por isso a diferença é do corte e não do modelo.`,
    ]);
  }

  return layout({
    title: `Quanto vale um ${titlePhrase} usado?`,
    description: `${titlePhrase}: preço mediano ${fmtEur(cell.fm)} (${fmtEur(cell.fl)}–${fmtEur(cell.fh)}) em ${cell.n} anúncios ativos do OLX Portugal. Comparação com os outros cortes do mesmo modelo.`,
    canonical, body, zone: "all", nav: "precos", depositCount, index: true, host,
    jsonLd: {
      "@context": "https://schema.org",
      "@graph": [
        {
          "@type": "Dataset",
        "license": licenseUrl(host),
          "name": `Preços de ${titlePhrase} em Portugal`,
          "description": `Mediana e intervalo interquartil dos preços pedidos em ${cell.n} anúncios ativos de ${titlePhrase} no OLX Portugal.`,
          "creator": { "@type": "Organization", "name": "Flipper Club", "url": `https://${host}/` },
          "isAccessibleForFree": true, "dateModified": builtAt || undefined,
          "variableMeasured": "Preço pedido (EUR)", "url": canonical,
          ...(cell.y0 && cell.y1 ? { "temporalCoverage": `${cell.y0}/${cell.y1}` } : {}),
        },
        {
          "@type": "AggregateOffer", "priceCurrency": "EUR",
          "lowPrice": cell.fl, "highPrice": cell.fh, "offerCount": cell.n, "url": canonical,
          "itemOffered": {
            "@type": "Car", "name": titlePhrase,
            "brand": { "@type": "Brand", "name": rec.b }, "model": rec.m,
            ...(isFuel ? { "fuelType": cell.lbl } : {}),
          },
        },
        breadcrumbLd(host, [
          { name: "Início", href: "/" }, { name: "Preços", href: "/precos" },
          { name: `${rec.b} ${rec.m}`, href: `/preco/${slug}` }, { name: cell.lbl },
        ]),
        faqLd(faqs),
      ],
    },
  });
}

// ═══ /precos/{distrito} — the market in one district ═════════════════════════
// Portuguese district names and the definite article.
//
// Most districts take a bare "em Lisboa" / "em Braga" / "em Faro". Porto takes
// the article — "no Porto" — and "em Porto" reads wrong to any Portuguese
// speaker, which on a site whose whole claim is local credibility is a worse
// look than it is a grammar slip. An exception list rather than a rule, because
// that is honestly what it is: the article is lexical, not derivable.
const DISTRICT_ARTICLE = { porto: "no" };
function emDistrito(key, label) {
  return `${DISTRICT_ARTICLE[key] || "em"} ${label}`;
}

export function renderDistrictPage({ key, rec, models, stats, host, depositCount, builtAt }) {
  const canonical = `https://${host}/precos/${key}`;
  const L = escapeHtml(rec.lbl);
  const vsNational = stats.priceMed ? (rec.fm - stats.priceMed) / stats.priceMed : null;
  const rows = (rec.top || []).map(([slug, n, fm]) => {
    const m = models[slug];
    if (!m) return "";
    const d = m.fm > 0 ? (fm - m.fm) / m.fm : null;
    return `<tr>
      <td><a href="/preco/${slug}" style="color:#177A47;font-weight:600;">${escapeHtml(m.b)} ${escapeHtml(m.m)}</a></td>
      <td>${n}</td><td>${fmtEur(fm)}</td>
      <td class="mut">${fmtEur(m.fm)}</td>
      <td>${d == null ? "—" : `${d >= 0 ? "+" : ""}${Math.round(d * 100)}%`}</td></tr>`;
  }).join("");

  const body = crumbs([
    { name: "Início", href: "/" }, { name: "Preços", href: "/precos" }, { name: rec.lbl },
  ]) + `
    <section class="section fc-wrap" style="padding-top:16px;">
      <div class="eyebrow" style="margin-bottom:14px;"><span class="e-dot"></span><span class="mono">${L.toUpperCase()} · OLX PORTUGAL</span></div>
      <h1 class="fc-h1">Preços de carros usados ${emDistrito(key, L)}</h1>
      <p class="fc-p">Nos <b>${fmtNum(rec.n)} anúncios ativos</b> com localização ${emDistrito(key, L)}, o preço pedido mediano é <b>${fmtEur(rec.fm)}</b>${rec.kmm != null ? `, com ${fmtKm(rec.kmm)} de quilometragem mediana` : ""}.${vsNational != null ? ` Isso é <b>${Math.abs(Math.round(vsNational * 100))}% ${vsNational >= 0 ? "acima" : "abaixo"}</b> da mediana nacional (${fmtEur(stats.priceMed)}).` : ""}</p>
      <div class="fc-stat-row" style="margin:18px 0 6px;">
        <div class="fc-stat"><div class="k">MEDIANO AQUI</div><div class="v">${fmtEur(rec.fm)}</div><div class="s">${fmtEur(rec.fl)} – ${fmtEur(rec.fh)}</div></div>
        <div class="fc-stat"><div class="k">ANÚNCIOS</div><div class="v">${fmtNum(rec.n)}</div><div class="s">ativos agora</div></div>
        ${rec.kmm != null ? `<div class="fc-stat"><div class="k">KM MEDIANO</div><div class="v">${fmtNum(rec.kmm)}</div><div class="s">à venda aqui</div></div>` : ""}
      </div>
      ${provenance({ n: rec.n, builtAt, measure: `Preço pedido em anúncios com localização em ${rec.lbl}` })}
    </section>
    ${rows ? `<section class="section fc-wrap">
      <h2 class="fc-h2">Modelos mais anunciados ${emDistrito(key, L)}</h2>
      <p class="fc-p">A última coluna é o que interessa: onde o preço local se afasta do nacional, ou há mais oferta aqui, ou o mesmo carro custa mais por causa da procura.</p>
      <div class="fc-scroll"><table class="fc-tbl">
        <thead><tr><th>Modelo</th><th>Anúncios aqui</th><th>Mediano aqui</th><th>Mediano nacional</th><th>Diferença</th></tr></thead>
        <tbody>${rows}</tbody></table></div>
    </section>` : ""}
    <section class="section fc-wrap" style="padding-bottom:70px;">
      <p class="fc-p"><a href="/precos">Todos os modelos</a> · <a href="/mercado">Carros abaixo do valor justo</a> · <a href="/avaliar">Avaliar um anúncio</a></p>
    </section>`;

  return layout({
    title: `Preços de carros usados ${emDistrito(key, rec.lbl)}`,
    description: `Carros usados ${emDistrito(key, rec.lbl)}: preço mediano ${fmtEur(rec.fm)} em ${fmtNum(rec.n)} anúncios ativos do OLX, e como cada modelo se compara com a mediana nacional.`,
    canonical, body, zone: "all", nav: "precos", depositCount, index: true, host,
    jsonLd: {
      "@context": "https://schema.org",
      "@graph": [
        {
          "@type": "Dataset", "license": licenseUrl(host), "url": canonical, "inLanguage": "pt-PT",
          "name": `Preços de carros usados ${emDistrito(key, rec.lbl)}`,
          "description": `Preço pedido mediano e intervalo interquartil em ${rec.n} anúncios ativos de carros usados com localização ${emDistrito(key, rec.lbl)}, OLX Portugal.`,
          "creator": { "@type": "Organization", "name": "Flipper Club", "url": `https://${host}/` },
          "isAccessibleForFree": true, "dateModified": builtAt || undefined,
          "spatialCoverage": { "@type": "Place", "name": rec.lbl, "containedInPlace": { "@type": "Country", "name": "Portugal" } },
          "variableMeasured": "Preço pedido (EUR)",
        },
        {
          "@type": "AggregateOffer", "priceCurrency": "EUR",
          "lowPrice": rec.fl, "highPrice": rec.fh, "offerCount": rec.n, "url": canonical,
          "areaServed": { "@type": "Place", "name": rec.lbl },
        },
        breadcrumbLd(host, [
          { name: "Início", href: "/" }, { name: "Preços", href: "/precos" }, { name: rec.lbl },
        ]),
      ],
    },
  });
}
