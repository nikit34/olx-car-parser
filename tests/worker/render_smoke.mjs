// Render every public page template against a REAL models.json and assert the
// things that silently break.
//
// Why this exists as a file instead of a heredoc in the workflow: `node --check`
// only parses. It accepted a file with raw CSS pasted at the top level, and it
// accepts a const used above its declaration — both of which throw at request
// time, in production, on the first visitor. Only rendering catches them.
//
// Run:  node tests/worker/render_smoke.mjs [path/to/models.json]
// With no argument it fetches the live latest-data blob; offline, pass a file.

import { readFileSync } from "node:fs";
import {
  renderLanding, renderGrid, renderAvaliar, renderModelPage, renderModelsHub,
  renderModelWidget, renderInfo, slugify, setAnalyticsId,
} from "../../flipper-club/src/templates.js";
import {
  renderYearPage, renderNotFound, renderDepreciationPage, renderDepreciationHub,
  renderComparePage, renderCompareHub, renderLiquidityHub, renderValuationGap,
  renderMarketIndex, renderMethodology, renderAbout, renderIsv,
  setSiteIdentity, corpusStats, modelInsights, provenance,
  yearCells, yearCell, yearPageYears, depreciationOk, depreciationFit, depreciationSlugs,
  comparePairs, parseComparePath, comparePairKey, modelJson, yearJson, MIN_YEAR_PAGE_N,
  estimateIsv, ISV_TABLES_FOR_TEST,
} from "../../flipper-club/src/seo-pages.js";

const HOST = "carsbuyer.org";
const RELEASE = "https://github.com/nikit34/olx-car-parser/releases/download/latest-data/models.json";

let failures = 0;
function check(name, fn) {
  try { fn(); console.log(`  ok   ${name}`); }
  catch (err) { failures++; console.error(`  FAIL ${name}\n       ${err && err.message}`); }
}
function assert(cond, msg) { if (!cond) throw new Error(msg); }

// Every rendered page has to satisfy these, or the page is broken in a way that
// costs traffic rather than throwing.
function assertPage(html, { indexable, canonical = null, label }) {
  assert(typeof html === "string" && html.length > 500, `${label}: no output`);
  assert(html.startsWith("<!doctype html>"), `${label}: not a document`);
  assert(html.includes('<html lang="pt-PT">'), `${label}: lang is not pt-PT`);
  assert(!/undefined|\[object Object\]|NaN(?![a-zA-Z])/.test(stripJsonLd(html)),
    `${label}: rendered "undefined"/"NaN"/"[object Object]" into the page`);
  assert(!html.includes("fonts.googleapis.com"), `${label}: still linking Google Fonts`);
  assert(html.includes('rel="preload" href="/fonts/'), `${label}: font preload missing`);
  const robots = html.match(/<meta name="robots" content="([^"]+)">/);
  assert(robots, `${label}: no robots meta`);
  if (indexable) {
    assert(robots[1] === "index,follow", `${label}: expected index,follow, got ${robots[1]}`);
    assert(html.includes('<link rel="canonical"'), `${label}: indexable page with no canonical`);
    if (canonical) assert(html.includes(`<link rel="canonical" href="${canonical}">`),
      `${label}: canonical is not ${canonical}`);
    assert(/<meta name="description" content="[^"]{60,}">/.test(html), `${label}: description missing or too short`);
    assert(/<h1[^>]*>/.test(html), `${label}: no h1`);
  } else {
    assert(robots[1] === "noindex,follow", `${label}: expected noindex,follow, got ${robots[1]}`);
  }
  // Any JSON-LD present must parse — an unparseable block is markup Google drops
  // silently, which is the worst possible failure mode for structured data.
  for (const m of html.matchAll(/<script type="application\/ld\+json">([\s\S]*?)<\/script>/g)) {
    try { JSON.parse(m[1].replace(/\\u003c/g, "<")); }
    catch (e) { throw new Error(`${label}: JSON-LD does not parse — ${e.message}`); }
  }
}
// JSON-LD legitimately contains the token "undefined" nowhere, but `NaN` can
// appear inside a base64 image or a hash; strip the LD blocks before the scan so
// the check stays about visible copy.
function stripJsonLd(html) {
  return html.replace(/<script type="application\/ld\+json">[\s\S]*?<\/script>/g, "")
             .replace(/<script>[\s\S]*?<\/script>/g, "");
}

function ldTypes(html) {
  const out = new Set();
  for (const m of html.matchAll(/<script type="application\/ld\+json">([\s\S]*?)<\/script>/g)) {
    const doc = JSON.parse(m[1].replace(/\\u003c/g, "<"));
    for (const node of (doc["@graph"] || [doc])) if (node["@type"]) out.add(node["@type"]);
  }
  return out;
}

async function loadModels() {
  const arg = process.argv[2];
  if (arg) return JSON.parse(readFileSync(arg, "utf8"));
  const r = await fetch(RELEASE);
  if (!r.ok) throw new Error(`models.json fetch → ${r.status}`);
  return r.json();
}

const mdoc = await loadModels();
const models = mdoc.models;
const builtAt = mdoc.built_at;
const stats = corpusStats(models, builtAt);
const slugs = Object.keys(models);
console.log(`models.json: ${slugs.length} models, built ${builtAt}`);

// ── page-set selection ──────────────────────────────────────────────────────
const yearPages = slugs.flatMap(s => yearPageYears(models[s]).map(y => [s, y]));
const depSlugs = depreciationSlugs(models);
const pairs = comparePairs(models);
console.log(`derived: ${yearPages.length} model-year pages, ${depSlugs.length} depreciation pages, ${pairs.length} comparisons`);

check("year pages all clear the publishing floor", () => {
  assert(yearPages.length > 0, "no year pages at all");
  for (const [s, y] of yearPages) {
    const c = yearCell(models[s], y);
    assert(c && c.n >= MIN_YEAR_PAGE_N, `${s}/${y} below the floor`);
    assert(typeof c.y === "number", `${s}/${y} is a band, not a year`);
  }
});

check("depreciation pages have a plausible, well-fitting curve", () => {
  assert(depSlugs.length > 0, "no depreciation pages");
  for (const s of depSlugs) {
    const f = depreciationFit(models[s]);
    assert(f.rate > 0 && f.rate < 0.30, `${s}: implausible rate ${f.rate}`);
    assert(f.r2 >= 0.55, `${s}: weak fit ${f.r2}`);
  }
});

check("comparison pairs are canonical, cross-brand and resolvable", () => {
  assert(pairs.length > 0, "no comparison pairs");
  const set = new Set(pairs.map(([a, b]) => comparePairKey(a, b)));
  assert(set.size === pairs.length, "duplicate pairs in the set");
  for (const [a, b] of pairs) {
    assert(a < b, `${a}-vs-${b} is not alphabetically ordered`);
    assert(models[a] && models[b], `${a}-vs-${b} references a missing model`);
    assert(models[a].b !== models[b].b, `${a}-vs-${b} is same-brand`);
    const parsed = parseComparePath(`${a}-vs-${b}`, models, set);
    assert(parsed && parsed.a === a && parsed.b === b, `${a}-vs-${b} does not round-trip`);
  }
  // A hyphen-heavy slug pair must still resolve (this is the reason
  // parseComparePath tries every "-vs-" boundary instead of the first).
  const hyphenated = pairs.find(([a, b]) => a.split("-").length > 2 || b.split("-").length > 2);
  if (hyphenated) {
    const r = parseComparePath(`${hyphenated[0]}-vs-${hyphenated[1]}`, models, set);
    assert(r, `hyphen-heavy pair ${hyphenated.join("-vs-")} failed to parse`);
  }
});

check("comparison set is stable across calls", () => {
  const again = comparePairs(models).map(([a, b]) => `${a}-vs-${b}`).join("|");
  assert(again === pairs.map(([a, b]) => `${a}-vs-${b}`).join("|"),
    "comparePairs is not deterministic — router and sitemap would disagree");
});

check("an invented comparison is refused", () => {
  const set = new Set(pairs.map(([a, b]) => comparePairKey(a, b)));
  assert(parseComparePath("nao-existe-vs-tambem-nao", models, set) === null, "accepted a made-up pair");
  const [a] = pairs[0];
  const far = slugs.find(s => models[s].b !== models[a].b && !set.has(comparePairKey(a, s)));
  if (far) assert(parseComparePath(`${a}-vs-${far}`, models, set) === null,
    "accepted a pair outside the generated set");
});

// ── rendering ───────────────────────────────────────────────────────────────
const deep = slugs.slice().sort((x, y) => models[y].n - models[x].n)[0];

check("model page renders with its new blocks", () => {
  const rec = models[deep];
  const html = renderModelPage({
    rec, slug: deep, liveDeals: [], siblings: [], host: HOST, depositCount: 0, builtAt,
    insights: modelInsights(rec, stats),
    yearPages: yearPageYears(rec),
    competitors: [{ slug: "opel-astra", b: "Opel", m: "Astra", fm: 7000 }],
    comparisons: [{ href: `${deep}-vs-opel-astra`, m: "Opel Astra" }],
    hasDepreciation: depreciationOk(rec),
    provenanceHtml: provenance({ n: rec.n, builtAt }),
    altJson: `https://${HOST}/preco/${deep}.json`,
  });
  assertPage(html, { indexable: true, canonical: `https://${HOST}/preco/${deep}`, label: "model" });
  const types = ldTypes(html);
  for (const t of ["Dataset", "AggregateOffer", "BreadcrumbList", "FAQPage"]) {
    assert(types.has(t), `model page lost its ${t} schema`);
  }
  assert(!types.has("Product"), "model page emitted Product markup for cars we do not sell");
  assert(html.includes('rel="alternate" type="application/json"'), "model page has no JSON twin link");
  assert(html.includes(`/preco/${deep}/`), "model page does not link any year page");
  assert(html.includes("/comparar/"), "model page does not link a comparison");
  assert(html.includes("fc-prov"), "model page has no provenance line");
});

check("model insights differ between models", () => {
  const top = slugs.slice().sort((x, y) => models[y].n - models[x].n).slice(0, 30);
  const texts = top.map(s => modelInsights(models[s], stats).join(" "));
  const unique = new Set(texts);
  assert(unique.size >= top.length * 0.8,
    `insight text repeats: ${unique.size} distinct of ${top.length} — the block is boilerplate again`);
  assert(texts.every(t => t.length > 40), "some models produced no insights at all");
});

check("year page renders", () => {
  const [s, y] = yearPages[0];
  const rec = models[s];
  const all = yearCells(rec, 1).slice().sort((a, b) => a.y - b.y);
  const i = all.findIndex(c => c.y === y);
  const html = renderYearPage({
    rec, slug: s, year: y, cell: yearCell(rec, y),
    neighbours: {
      older: i > 0 ? all[i - 1] : null,
      newer: i < all.length - 1 ? all[i + 1] : null,
      window: all.slice(Math.max(0, i - 3), i + 4).sort((a, b) => b.y - a.y),
    },
    liveDeals: [], pageYears: yearPageYears(rec), stats, host: HOST, depositCount: 0, builtAt,
  });
  assertPage(html, { indexable: true, canonical: `https://${HOST}/preco/${s}/${y}`, label: "year" });
  const types = ldTypes(html);
  for (const t of ["Dataset", "AggregateOffer", "BreadcrumbList", "FAQPage"]) {
    assert(types.has(t), `year page lost its ${t} schema`);
  }
  assert(html.includes(`/preco/${s}"`) || html.includes(`/preco/${s}<`) || html.includes(`href="/preco/${s}`),
    "year page does not link back to its model");
});

check("every year page renders without throwing", () => {
  for (const [s, y] of yearPages) {
    const rec = models[s];
    const all = yearCells(rec, 1).slice().sort((a, b) => a.y - b.y);
    const i = all.findIndex(c => c.y === y);
    const html = renderYearPage({
      rec, slug: s, year: y, cell: yearCell(rec, y),
      neighbours: {
        older: i > 0 ? all[i - 1] : null,
        newer: i < all.length - 1 ? all[i + 1] : null,
        window: all.slice(Math.max(0, i - 3), i + 4).sort((a, b) => b.y - a.y),
      },
      liveDeals: [], pageYears: yearPageYears(rec), stats, host: HOST, depositCount: 0, builtAt,
    });
    assert(html.includes("<h1"), `${s}/${y}: no h1`);
    assert(!stripJsonLd(html).includes("undefined"), `${s}/${y}: rendered "undefined"`);
  }
});

check("every depreciation page renders without throwing", () => {
  for (const s of depSlugs) {
    const rec = models[s];
    const html = renderDepreciationPage({
      rec, slug: s, fit: depreciationFit(rec), stats,
      pageYears: yearPageYears(rec), host: HOST, depositCount: 0, builtAt,
    });
    assertPage(html, { indexable: true, canonical: `https://${HOST}/depreciacao/${s}`, label: `depreciacao/${s}` });
    assert(html.includes("<svg"), `${s}: depreciation page has no chart`);
  }
});

check("every comparison page renders without throwing", () => {
  for (const [a, b] of pairs) {
    const html = renderComparePage({
      a, b, ra: models[a], rb: models[b], stats, host: HOST, depositCount: 0, builtAt,
    });
    assertPage(html, { indexable: true, canonical: `https://${HOST}/comparar/${a}-vs-${b}`, label: `comparar/${a}-vs-${b}` });
    assert(html.includes(`/preco/${a}`) && html.includes(`/preco/${b}`),
      `${a}-vs-${b}: does not link both model pages`);
  }
});

check("hubs render", () => {
  const depRows = depSlugs.map(s => {
    const r = models[s], f = depreciationFit(r);
    return { slug: s, b: r.b, m: r.m, n: r.n, rate: f.rate, span: f.span };
  }).sort((x, y) => y.rate - x.rate);
  assertPage(renderDepreciationHub({ rows: depRows, stats, host: HOST, depositCount: 0, builtAt }),
    { indexable: true, canonical: `https://${HOST}/depreciacao`, label: "depreciacao hub" });

  assertPage(renderCompareHub({ pairs, models, host: HOST, depositCount: 0, builtAt }),
    { indexable: true, canonical: `https://${HOST}/comparar`, label: "comparar hub" });

  const liq = Object.entries(models).filter(([, r]) => r.sd != null)
    .map(([slug, r]) => ({ slug, b: r.b, m: r.m, sd: r.sd, sn: r.sn, fm: r.fm }))
    .sort((x, y) => x.sd - y.sd);
  assertPage(renderLiquidityHub({ rows: liq, stats, host: HOST, depositCount: 0, builtAt }),
    { indexable: true, canonical: `https://${HOST}/liquidez`, label: "liquidez" });

  const gap = Object.entries(models).filter(([, r]) => r.gm > 0 && r.fm > 0)
    .map(([slug, r]) => ({ slug, b: r.b, m: r.m, fm: r.fm, gm: r.gm, n: r.n, gap: r.fm / r.gm - 1 }))
    .sort((x, y) => y.gap - x.gap);
  assertPage(renderValuationGap({ over: gap.slice(0, 25), under: gap.slice(-25).reverse(), stats, host: HOST, depositCount: 0, builtAt }),
    { indexable: true, canonical: `https://${HOST}/sobrevalorizados`, label: "sobrevalorizados" });
});

check("trust pages render, and stay honest with no identity configured", () => {
  setSiteIdentity({});
  const meth = renderMethodology({ stats, host: HOST, depositCount: 0, builtAt });
  assertPage(meth, { indexable: true, canonical: `https://${HOST}/metodologia`, label: "metodologia" });
  assert(!meth.includes("mailto:"), "methodology invented a contact address");
  const about = renderAbout({ stats, host: HOST, depositCount: 0, builtAt });
  assertPage(about, { indexable: true, canonical: `https://${HOST}/sobre`, label: "sobre" });
  assert(!about.includes("mailto:"), "about invented a contact address");
  assert(!/"author"|"founder"/.test(about), "about invented an author");

  setSiteIdentity({ author: "Nome Teste", contact: "teste@example.org" });
  const about2 = renderAbout({ stats, host: HOST, depositCount: 0, builtAt });
  assert(about2.includes("mailto:teste@example.org"), "configured contact not rendered");
  assert(about2.includes("Nome Teste"), "configured author not rendered");
  setSiteIdentity({});
});

check("the ISV estimate matches src/analytics/isv.py", () => {
  // Fixed cases computed with compute_isv(as_of_year=2026). If a bracket in
  // either implementation drifts, one of these moves.
  const cases = [
    [{ cc: 1968, co2: 120, fuel: "diesel", regYear: 2016 }, 1914, "NEDC"],
    [{ cc: 1598, co2: 110, fuel: "petrol", regYear: 2021 }, 1582, "WLTP"],
    [{ cc: 2993, co2: 190, fuel: "diesel", regYear: 2012 }, 5730, "NEDC"],
    [{ cc: 999, co2: 95, fuel: "petrol", regYear: 2026 }, 216, "WLTP"],
  ];
  for (const [input, expected, cycle] of cases) {
    const r = estimateIsv(ISV_TABLES_FOR_TEST, { ...input, asOfYear: 2026, isEu: true });
    assert(Math.round(r.isv) === expected,
      `ISV(${JSON.stringify(input)}) = ${Math.round(r.isv)}, isv.py says ${expected}`);
    assert(r.cycle === cycle, `wrong CO2 cycle for ${input.regYear}`);
  }
  // The refusals are part of the contract: an electric car is exempt, a plug-in
  // hybrid returns nothing rather than a wrong number, and missing inputs do
  // not produce a guess.
  assert(estimateIsv(ISV_TABLES_FOR_TEST, { fuel: "bev", asOfYear: 2026 }).isv === 0, "BEV is not exempt");
  assert(estimateIsv(ISV_TABLES_FOR_TEST, { fuel: "phev", asOfYear: 2026 }) === null, "PHEV produced a number");
  assert(estimateIsv(ISV_TABLES_FOR_TEST, { cc: 0, co2: 120, fuel: "diesel", regYear: 2016, asOfYear: 2026 }) === null,
    "estimated with no cilindrada");
  // Outside the EU there is no age reduction at all.
  const eu = estimateIsv(ISV_TABLES_FOR_TEST, { cc: 1968, co2: 120, fuel: "diesel", regYear: 2016, asOfYear: 2026, isEu: true });
  const non = estimateIsv(ISV_TABLES_FOR_TEST, { cc: 1968, co2: 120, fuel: "diesel", regYear: 2016, asOfYear: 2026, isEu: false });
  assert(non.isv > eu.isv && Math.round(non.isv) === Math.round(non.gross), "non-EU car got an age reduction");
});

check("ISV page renders and ships the one implementation", () => {
  const top = slugs.slice(0, 12).map(s => ({ slug: s, b: models[s].b, m: models[s].m, fm: models[s].fm }));
  const html = renderIsv({ topModels: top, host: HOST, depositCount: 0, builtAt, refYear: 2026 });
  assertPage(html, { indexable: true, canonical: `https://${HOST}/isv`, label: "isv" });
  assert(ldTypes(html).has("WebApplication"), "ISV page lost its WebApplication schema");
  // Spot-check one bracket from each table against src/analytics/isv.py.
  for (const needle of ["849.03", "6194.88", "41910.96", "38271.32", "33390.12", "33447.9", '"particulas":500']) {
    assert(html.replace(/\s/g, "").includes(needle.replace(/\s/g, "")),
      `ISV tables drifted from isv.py (missing ${needle})`);
  }
  // The browser must run the SAME function Node just checked, serialised in —
  // not a second copy of the arithmetic.
  assert(html.includes("var estimateIsv = function estimateIsv("),
    "the page carries its own copy of the ISV math instead of the shared one");
});

check("market index renders, current and archived", () => {
  const snap = { week: "2026-W35", date: "2026-08-25", builtAt, models: stats.models,
                 listings: stats.listings, priceMed: stats.priceMed, kmMed: stats.kmMed,
                 sellMed: stats.sellMed, depMed: stats.depMed };
  const prev = { ...snap, week: "2026-W34", date: "2026-08-18", priceMed: Math.round(stats.priceMed * 1.02) };
  assertPage(renderMarketIndex({ snapshot: snap, history: [prev, snap], host: HOST, depositCount: 0 }),
    { indexable: true, canonical: `https://${HOST}/mercado/indice`, label: "indice" });
  // The URL form of the week is lower-case (the router normalises every public
  // path); the page still SHOWS the ISO spelling.
  const archive = renderMarketIndex({ snapshot: snap, history: [prev, snap], host: HOST, depositCount: 0, isArchive: true });
  assertPage(archive, { indexable: true, canonical: `https://${HOST}/mercado/indice/2026-w35`, label: "indice archive" });
  assert(archive.includes("2026-W35"), "archive page does not show the ISO week");
});

check("404 page is noindex, links back, and never 401s in spirit", () => {
  const sugg = slugs.slice(0, 12).map(s => ({ slug: s, m: `${models[s].b} ${models[s].m}`, fm: models[s].fm }));
  const html = renderNotFound({ suggestions: sugg, depositCount: 0, host: HOST, path: "/pagina-que-nao-existe" });
  assertPage(html, { indexable: false, label: "404" });
  assert(html.includes("/precos") && html.includes("/avaliar"), "404 page has no way back");
});

check("existing product pages still render", () => {
  const hub = renderModelsHub({
    models: slugs.map(s => ({ slug: s, b: models[s].b, m: models[s].m, fm: models[s].fm, n: models[s].n })),
    depositCount: 0, builtAt, host: HOST,
  });
  assertPage(hub, { indexable: true, canonical: `https://${HOST}/precos`, label: "precos" });
  assert(hub.includes("/depreciacao") && hub.includes("/liquidez"), "hub does not link the new sections");

  const av = renderAvaliar({ rec: null, olxId: null, sourceUrl: null, query: "", models, spec: null,
                             depositCount: 0, host: HOST, builtAt });
  assertPage(av, { indexable: true, canonical: `https://${HOST}/avaliar`, label: "avaliar" });
  const t = ldTypes(av);
  for (const want of ["WebApplication", "FAQPage", "BreadcrumbList"]) {
    assert(t.has(want), `/avaliar is missing ${want} schema`);
  }

  const deal = {
    olx_id: "ID1", brand: models[deep].b, model: models[deep].m, title: "Carro de teste",
    price_eur: 7000, fair_median: 8500, fair_low: 7600, fair_high: 9400, discount_pct: 0.17,
    est_profit_eur: 1500, year: 2014, mileage_km: 180000, fuel_type: "Diesel",
    district: "Porto", photo_urls: [], days_on_market: 12, first_seen_at: "2026-08-01T00:00:00Z",
    seller_type: "Particular",
  };
  const grid = renderGrid({
    deals: [deal], zone: "all", sort: "score", view: "comprar", unlockedSet: new Set(),
    depositEur: 5, depositCount: 0, zoneCounts: { all: 1, norte: 1, centro: 0, sul: 0 },
    host: HOST, builtAt,
    modelLinks: [{ slug: deep, b: models[deep].b, m: models[deep].m, fm: models[deep].fm, count: 1 }],
  });
  assertPage(grid, { indexable: true, canonical: `https://${HOST}/mercado`, label: "mercado" });
  assert(grid.includes(`/preco/${deep}`), "/mercado does not link any model page");
  assert(grid.includes("/mercado/indice"), "/mercado does not link the index");

  assertPage(renderLanding({
    stats: { deals: 1, avgDisc: "17%", totalProfit: "€1 500" }, featured: deal,
    depositEur: 5, depositCount: 0, host: HOST,
  }), { indexable: true, label: "landing" });

  const w = renderModelWidget({ rec: models[deep], slug: deep, host: HOST });
  assert(w.includes("noindex,follow"), "widget is no longer noindex");
  assert(w.includes(`/preco/${deep}`), "widget lost its attribution link");

  assertPage(renderInfo({ zone: "all", depositCount: 0, title: "T", message: "M" }),
    { indexable: false, label: "info" });
});

check("JSON twins carry the sample size and the date", () => {
  const rec = models[deep];
  const j = modelJson(rec, deep, { host: HOST, builtAt });
  assert(j.sample_size === rec.n, "model JSON lost the sample size");
  assert(j.collected_until === builtAt.slice(0, 10), "model JSON lost the collection date");
  assert(j.asking_price.median === rec.fm, "model JSON median disagrees with the blob");
  assert(j.measured === "asking_price", "model JSON does not say what it measures");
  assert(JSON.parse(JSON.stringify(j)), "model JSON is not serialisable");
  const linked = j.by_year.filter(y => y.page).map(y => y.year).sort((x, y) => x - y);
  const expect = yearPageYears(rec).slice().sort((x, y) => x - y);
  assert(JSON.stringify(linked) === JSON.stringify(expect),
    "model JSON links a different year set than the router serves");

  const [s, y] = yearPages[0];
  const yj = yearJson(models[s], s, y, yearCell(models[s], y), { host: HOST, builtAt });
  assert(yj.year === y && yj.sample_size >= MIN_YEAR_PAGE_N, "year JSON is wrong");
});

check("the valuation event still fires on both /avaliar paths", () => {
  // Merged with the parallel GA4 work: the event has to survive on the
  // paste-a-link path AND the pick-a-model path, next to the new JSON-LD.
  setAnalyticsId("G-TESTONLY");
  const slug = Object.keys(models)[0];
  const withSpec = renderAvaliar({
    rec: null, olxId: null, sourceUrl: null, query: "", models,
    spec: { rec: { ...models[slug], t: `${models[slug].b} ${models[slug].m}` }, slug, year: 2014, cell: null },
    depositCount: 0, host: HOST, builtAt,
  });
  assert(withSpec.includes("valuation_result"), "valuation event lost on the model-pick path");
  assert(withSpec.includes('"source":"model"'), "valuation event lost its source on the model-pick path");
  setAnalyticsId("");
  const bare = renderAvaliar({ rec: null, olxId: null, sourceUrl: null, query: "", models,
                               spec: null, depositCount: 0, host: HOST, builtAt });
  assert(!bare.includes("valuation_result"), "valuation event fired with no result and no measurement id");
});

check("analytics stays off unless a measurement id is set", () => {
  setAnalyticsId("");
  const off = renderMethodology({ stats, host: HOST, depositCount: 0, builtAt });
  assert(!off.includes("gtag"), "analytics leaked with no measurement id");
  setAnalyticsId("G-TESTONLY");
  const on = renderMethodology({ stats, host: HOST, depositCount: 0, builtAt });
  assert(on.includes("G-TESTONLY"), "analytics tag missing");
  assert(on.includes("analytics_storage:'denied'"), "consent mode default is not denied");
  assert(on.indexOf("consent','default'") < on.indexOf("googletagmanager"),
    "consent declared after the gtag loader");
  setAnalyticsId("");
});

console.log(failures ? `\n${failures} check(s) FAILED` : "\nall render checks passed");
process.exit(failures ? 1 : 0);
