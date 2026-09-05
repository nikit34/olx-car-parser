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
  renderLanding, renderGrid, renderCarPage, renderAvaliar, renderModelPage, renderModelsHub,
  renderModelWidget, renderInfo, slugify, setAnalyticsId, renderPrivacy, ageTable,
} from "../../flipper-club/src/templates.js";
import {
  renderYearPage, renderNotFound, renderDepreciationPage, renderDepreciationHub,
  renderComparePage, renderCompareHub, renderLiquidityHub, renderValuationGap,
  renderMarketIndex, renderMarketMonth, renderMethodology, renderAbout, renderIsv,
  monthlyCuts, isoWeekMonth, weeksOfMonth, monthLabel, IDX_MIN_MONTH_WEEKS,
  setSiteIdentity, corpusStats, modelInsights, provenance,
  yearCells, yearCell, yearPageYears, depreciationOk, depreciationFit, depreciationSlugs,
  comparePairs, parseComparePath, comparePairKey, comparePriceGap, modelClass, comparePool,
  modelJson, yearJson, MIN_YEAR_PAGE_N,
  yearGap,
  depreciationAge, depreciationJson,
  estimateIsv, ISV_TABLES_FOR_TEST, renderDistrictPage,
  renderFacetPage, renderDuelPage, renderDuelHub, duel, duelJson, DUELS, withPrep,
  renderLiquidityPage, liquidityJson, liquidityOk, districtRanking,
  renderImportPage, renderImportHub, importJson, importOk, importSlugs,
  renderVenderPage, renderVenderHub, venderJson, venderOk,
} from "../../flipper-club/src/seo-pages.js";
import { GUIDES, renderGuide, renderGuidesHub } from "../../flipper-club/src/guides.js";

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
const market = mdoc.lqm || null;
const idoc = JSON.parse(readFileSync(new URL("./fixtures/import.json", import.meta.url), "utf8"));
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
    assert(c, `${s}/${y} has no cell`);
    assert(c.pg !== undefined ? !!c.pg : c.n >= MIN_YEAR_PAGE_N,
      `${s}/${y} is served but clears neither its pg flag nor the floor (n=${c.n}, pg=${c.pg})`);
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

check("every model in the comparison pool has a segment", () => {
  const missing = comparePool(models).map(([s]) => s).filter(s => !modelClass(s));
  assert(missing.length === 0,
    `unclassified in the top-${comparePool(models).length} pool: ${missing.join(", ")}`
    + " — add each to MODEL_CLASS in seo-pages.js");
});

check("every comparison is same-segment and price-adjacent at equal model year", () => {
  for (const [a, b] of pairs) {
    const cls = modelClass(a);
    assert(cls, `${a} has no segment but is being compared`);
    assert(cls === modelClass(b), `${a}-vs-${b} crosses segments (${cls} vs ${modelClass(b)})`);
    const gap = comparePriceGap(models[a], models[b]);
    assert(gap, `${a}-vs-${b} has no overlapping model year`);
    assert(gap.years >= 6, `${a}-vs-${b} rests on only ${gap.years} common years`);
    assert(gap.dist <= 0.5, `${a}-vs-${b} is ${gap.ratio.toFixed(2)}x apart at equal year`);
  }
});

check("the age-matched gap is narrower than the raw-median gap where age differs", () => {
  const ra = models["volkswagen-golf"], rb = models["opel-astra"];
  if (!ra || !rb) return;
  const gap = comparePriceGap(ra, rb);
  assert(gap, "no Golf/Astra overlap in this blob");
  const raw = Math.max(ra.fm / rb.fm, rb.fm / ra.fm);
  const matched = Math.max(gap.ratio, 1 / gap.ratio);
  assert(matched < raw, `age-matched ${matched.toFixed(2)}x is not below raw ${raw.toFixed(2)}x`);
  assert(gap.dist <= 0.5, "Golf vs Astra fell outside the price bound");
});

check("comparePriceGap ignores year bands and unmatched years", () => {
  const A = { yr: [{ y: 2015, fm: 10000, n: 20 }, { y: 2014, fm: 9000, n: 10 },
                   { y: "2010-2012", fm: 5000, n: 30 }, { y: 2001, fm: 1000, n: 8 }] };
  const B = { yr: [{ y: 2015, fm: 5000, n: 20 }, { y: 2014, fm: 4500, n: 10 },
                   { y: "2010-2012", fm: 2500, n: 30 }] };
  const g = comparePriceGap(A, B);
  assert(g.years === 2, `expected 2 matched years, got ${g.years}`);
  assert(Math.abs(g.ratio - 2) < 1e-9, `expected a 2x ratio, got ${g.ratio}`);
  assert(g.cells[0].y === 2015, "cells are not newest-first");
  assert(comparePriceGap(A, { yr: [{ y: 1990, fm: 900, n: 9 }] }) === null,
    "returned a gap for models that never overlap");
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
  const trustBox = (html.match(/Como lemos estes números\.[\s\S]*?<\/span>/) || [])[0] || "";
  assert(trustBox, "the model page lost the 'Como lemos estes números' disclosure entirely");
  assert(/não preços de venda fechados/.test(trustBox),
    "the disclosure no longer says these are not closed-sale prices");
  assert(/preços <b>pedidos<\/b>/.test(trustBox),
    "the disclosure no longer says the prices are asking prices");
  assert(/ISV/.test(trustBox),
    "the disclosure lost the one caveat the page cannot restate elsewhere: an import's unpaid ISV");
  assert(trustBox.includes('href="/metodologia"'),
    "the disclosure cites a method it does not link");
  const sellerCta = (html.match(/Vais vender o teu[\s\S]{0,600}?<\/section>/) || [])[0] || "";
  assert(sellerCta, "the model page lost its seller CTA");
  assert(/anunciados no OLX/.test(sellerCta),
    "the seller CTA claims a population wider than the corpus it measures");
  assert(!/à venda pede entre/.test(sellerCta),
    "the seller CTA is back to claiming every car for sale in the country");
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

check("alternatives are labelled for what they actually are", () => {
  const rec = models[deep];
  const base = {
    rec, slug: deep, liveDeals: [], siblings: [], host: HOST, depositCount: 0, builtAt,
    insights: modelInsights(rec, stats), yearPages: yearPageYears(rec), comparisons: [],
  };
  const seg = renderModelPage({
    ...base, competitorKind: "segment",
    competitors: [{ slug: "opel-astra", b: "Opel", m: "Astra", fm: 3000, ratio: 1.7 }],
  });
  assert(seg.includes("ALTERNATIVAS NO MESMO SEGMENTO"), "segment list is not labelled as one");
  assert(seg.includes("PREÇO AO MESMO ANO"), "segment list does not say what its number is");
  assert(seg.includes(">+70%<"), "segment chip does not carry the same-year difference");
  assert(!seg.includes("MESMA FAIXA DE PREÇO"), "segment list kept the price-band label");

  const price = renderModelPage({
    ...base, competitorKind: "price",
    competitors: [{ slug: "opel-astra", b: "Opel", m: "Astra", fm: 3000 }],
  });
  assert(price.includes("ALTERNATIVAS NA MESMA FAIXA DE PREÇO"), "price list lost its label");
  assert(!price.includes("PREÇO AO MESMO ANO"),
    "price-band list claims a same-year comparison it did not make");
  assert(!/[+−]\d+%<\/span>/.test(price.split("MESMA FAIXA DE PREÇO")[1].slice(0, 600)),
    "price-band chip shows a percentage instead of the median it measured");
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
  const yearMain = html.slice(html.indexOf("<main"), html.indexOf("</main>") + 7)
    .replace(/<script[\s\S]*?<\/script>/g, "").replace(/<footer[\s\S]*?<\/footer>/g, "");
  assert(yearMain.includes('href="/metodologia"'),
    "the year page body cites a method it does not link");
  assert(/preços? pedidos?/i.test(yearMain),
    "the year page body no longer says what it measures");
  assert(/Metade dos \d+ anúncios/.test(yearMain),
    "the year page states the interquartile range as if it were the whole range");
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

check("a one-year step is only claimed where the two price ranges separate", () => {
  let claimed = 0, withheld = 0;
  for (const [s2, y] of yearPages) {
    const rec = models[s2];
    const all = yearCells(rec, 1).slice().sort((a, b) => a.y - b.y);
    const i = all.findIndex(c => c.y === y);
    const cell = yearCell(rec, y), newer = i < all.length - 1 ? all[i + 1] : null;
    if (!newer) continue;
    const html = renderYearPage({
      rec, slug: s2, year: y, cell,
      neighbours: {
        older: i > 0 ? all[i - 1] : null, newer,
        window: all.slice(Math.max(0, i - 3), i + 4).sort((a, b) => b.y - a.y),
      },
      liveDeals: [], pageYears: yearPageYears(rec), stats, host: HOST, depositCount: 0, builtAt,
    });
    const g = yearGap(cell, newer);
    const pct = Math.round(g.pct * 100);
    const step = new RegExp(`>${pct >= 0 ? "\\+" : ""}${pct}%</b> face a ${y}`);
    if (g.separated) {
      assert(step.test(html), `${s2}/${y}: separated ranges but no step stated`);
      claimed++;
    } else {
      assert(!step.test(html),
        `${s2}/${y}: states a ${pct}% one-year step while P25-P75 overlap ${Math.round(g.overlap * 100)}%`);
      assert(html.includes("sobrepõem-se"), `${s2}/${y}: overlap withheld the step but never says so`);
      // Withholding the two-cell claim must not leave the heading unanswered:
      // where the model's own curve is publishable, the page cites its rate.
      if (depreciationOk(rec)) {
        assert(html.includes("ritmo medido em toda a série"),
          `${s2}/${y}: step withheld and the model's measured rate never offered`);
      }
      withheld++;
    }
  }
  assert(claimed > 0 && withheld > 0,
    `the gate never went both ways (${claimed} claimed, ${withheld} withheld) — it is not being exercised`);
});

check("deals from neighbouring years are labelled as such, never as this year", () => {
  const [s2, y] = yearPages[0];
  const rec = models[s2];
  const all = yearCells(rec, 1).slice().sort((a, b) => a.y - b.y);
  const i = all.findIndex(c => c.y === y);
  const near = {
    olx_id: "IDN", brand: rec.b, model: rec.m, title: "Carro de teste",
    price_eur: 7000, fair_median: 8500, fair_low: 7600, fair_high: 9400, discount_pct: 0.17,
    est_profit_eur: 1500, year: y - 1, mileage_km: 180000, fuel_type: "Diesel",
    district: "Porto", photo_urls: [], days_on_market: 12, first_seen_at: "2026-08-01T00:00:00Z",
    seller_type: "Particular",
  };
  const args = {
    rec, slug: s2, year: y, cell: yearCell(rec, y),
    neighbours: {
      older: i > 0 ? all[i - 1] : null, newer: i < all.length - 1 ? all[i + 1] : null,
      window: all.slice(Math.max(0, i - 3), i + 4).sort((a, b) => b.y - a.y),
    },
    pageYears: yearPageYears(rec), stats, host: HOST, depositCount: 0, builtAt,
  };
  const nearHtml = renderYearPage({ ...args, liveDeals: [near], dealsNear: true });
  assert(nearHtml.includes("ANOS PRÓXIMOS"), "neighbouring-year deals are not labelled");
  assert(!nearHtml.includes(`${String(rec.m).toUpperCase()} DE ${y} ABAIXO`),
    "neighbouring-year deals are presented as deals of this year");

  const ownHtml = renderYearPage({ ...args, liveDeals: [{ ...near, year: y }], dealsNear: false });
  assert(ownHtml.includes(`DE ${y} ABAIXO`), "a deal of this very year lost its label");
  assert(!ownHtml.includes("ANOS PRÓXIMOS"), "a deal of this very year is labelled as a neighbour");
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
    assert(html.includes("Quanto custa um ano de idade"), `${s}: no euro ladder`);
    assert(html.includes("Há um ponto de inflexão?"), `${s}: does not answer the inflection question`);
    assert(html.includes("mudou de geração"), `${s}: does not say the series crosses generations`);
  }
});

check("every liquidity page renders without throwing", () => {
  const liqSlugs = Object.keys(models).filter(s2 => liquidityOk(models[s2]));
  assert(liqSlugs.length, "fixture carries no liquidity records");
  for (const s2 of liqSlugs) {
    const rec = models[s2];
    const html = renderLiquidityPage({
      rec, slug: s2, market, hasDepreciation: false,
      host: HOST, depositCount: 0, builtAt,
    });
    assertPage(html, { indexable: true, canonical: `https://${HOST}/liquidez/${s2}`, label: `liquidez/${s2}` });
    assert(html.includes("<svg"), `${s2}: liquidity page has no curve`);
    assert(html.includes("Sair do OLX não é o mesmo que vender"), `${s2}: drops the expiry caveat`);
    assert(/\d+ em cada 100 desaparecem no primeiro mês/.test(html), `${s2}: no headline share`);
    assert(html.includes('"@type": "FAQPage"') || html.includes('"@type":"FAQPage"'), `${s2}: no FAQ block`);
  }
});

check("the liquidity page never claims a sale it cannot see", () => {
  const liqSlugs = Object.keys(models).filter(s2 => liquidityOk(models[s2]));
  for (const s2 of liqSlugs) {
    const rec = models[s2], lq = rec.lq;
    const html = renderLiquidityPage({ rec, slug: s2, market, host: HOST, depositCount: 0, builtAt });
    const json = liquidityJson(rec, s2, { host: HOST, builtAt });
    assert(json.gone_in_30d === lq.s30, `${s2}: JSON twin disagrees with the record`);
    assert(json.caveat.includes("not proof of a sale"), `${s2}: JSON twin drops the caveat`);
    if (lq.rb != null) {
      assert(html.includes("É um mínimo"), `${s2}: relist share is stated as a rate, not a floor`);
    }
    if (lq.cd != null && lq.hd != null) {
      assert(html.includes("não fica parado por se ter baixado o preço"),
        `${s2}: the price-cut medians are left to read as causal`);
    }
    for (const cells of [lq.pb, lq.ab, lq.dt]) {
      for (const c of cells || []) {
        assert(c.n >= 40, `${s2}: a cut of ${c.n} listings reached the page`);
      }
    }
  }
});

check("every import page renders and adds up on the page", () => {
  const slugs = importSlugs(idoc);
  assert(slugs.length, "import fixture carries no models");
  for (const slug of slugs) {
    const rec = idoc.models[slug];
    const html = renderImportPage({
      rec, slug, costs: idoc.costs, hasModelPage: true,
      host: HOST, depositCount: 0, builtAt: idoc.built_at,
    });
    assertPage(html, { indexable: true, canonical: `https://${HOST}/importar/${slug}`, label: `importar/${slug}` });
    assert(html.includes("Total à porta"), `${slug}: no landed-cost column`);
    assert(html.includes("O que esta conta não sabe"), `${slug}: drops the caveats`);
    assert(html.includes("IVA"), `${slug}: never explains the VAT side`);
    for (const c of rec.yr) {
      assert(c.ll === c.dm + c.isv + Math.round(idoc.costs.lo) ||
             c.ll === Math.round(c.dm + c.isv + idoc.costs.lo),
        `${slug} ${c.y}: the landed cost does not equal the printed parts`);
      assert(c.gl === c.ptm - c.lh, `${slug} ${c.y}: the saving is not the difference shown`);
    }
    const json = importJson(rec, slug, idoc.costs, { host: HOST, builtAt: idoc.built_at });
    assert(json.years.length === rec.yr.length, `${slug}: JSON twin lost a year`);
    assert(json.caveat.includes("asking prices"), `${slug}: JSON twin drops the caveat`);
  }
});

check("a model that loses money says so instead of hiding the row", () => {
  const losing = importSlugs(idoc).map(s2 => idoc.models[s2]).find(r => (r.yr || []).some(c => c.gl < 0));
  if (!losing) return;
  const slug = importSlugs(idoc).find(s2 => idoc.models[s2] === losing);
  const html = renderImportPage({ rec: losing, slug, costs: idoc.costs, host: HOST, depositCount: 0, builtAt: idoc.built_at });
  assert(html.includes("mais €") || html.includes("mais&nbsp;€"), `${slug}: a losing year is not shown as a loss`);
});

check("the import hub ranks models and states both sides", () => {
  const rows = importSlugs(idoc).map(slug => {
    const r = idoc.models[slug];
    return { slug, b: r.b, m: r.m, med_gap: r.med_gap, wins: r.wins,
             cells: (r.yr || []).length, nde: r.nde, npt: r.npt };
  });
  const html = renderImportHub({ rows, costs: idoc.costs, host: HOST, depositCount: 0, builtAt: idoc.built_at });
  assertPage(html, { indexable: true, canonical: `https://${HOST}/importar`, label: "importar hub" });
  assert(html.includes("AutoScout24"), "the hub never says where the German prices come from");
  const gaps = rows.map(r => r.med_gap);
  assert(gaps.slice(1).every((g, i) => g <= gaps[i]), "the hub is not ranked by the difference");
});

check("the euro figures rest on the fit, not on the thinnest cell", () => {
  for (const s of depSlugs) {
    const rec = models[s], fit = depreciationFit(rec);
    const av = depreciationAge(rec, fit, builtAt);
    assert(av.base.age === Math.ceil(av.minAge), `${s}: base is not the youngest measured age`);
    const cells = yearCells(rec, 5);
    const newest = cells[0];
    const spread = Math.abs(av.base.price - newest.fm) / newest.fm;
    assert(spread < 0.6, `${s}: the fitted base is ${Math.round(spread * 100)}% away from the newest median`);
    assert(av.base.price > 0 && isFinite(av.base.price), `${s}: no base price`);
  }
});

check("a knee nobody would shop at is not published as advice", () => {
  for (const s of depSlugs) {
    const rec = models[s], fit = depreciationFit(rec);
    const av = depreciationAge(rec, fit, builtAt);
    if (!av.cheapFrom) {
      assert(av.capCost > 0, `${s}: no cost quoted at the cap instead`);
      continue;
    }
    assert(av.cheapFrom.age <= 15, `${s}: quotes a sub-500 age of ${av.cheapFrom.age}`);
  }
});

check("the age view is inside its own measured range and monotone", () => {
  for (const s of depSlugs) {
    const rec = models[s], fit = depreciationFit(rec);
    const av = depreciationAge(rec, fit, builtAt);
    assert(av, `${s}: no age view`);
    assert(av.minAge >= 0 && av.maxAge > av.minAge, `${s}: bad age range ${av.minAge}-${av.maxAge}`);
    assert(av.at(av.minAge - 1) === null && av.at(av.maxAge + 1) === null,
      `${s}: quotes a cost outside the measured ages`);
    let prev = Infinity;
    for (let a = av.minAge; a <= av.maxAge; a++) {
      const c = av.at(a);
      assert(c >= 0 && c <= prev, `${s}: cost of a year of age is not falling at ${a}`);
      prev = c;
    }
    assert(av.halfLife > 1 && av.halfLife < 40, `${s}: implausible half-life ${av.halfLife}`);
    if (av.cheapFrom) {
      assert(av.cheapFrom.age >= av.minAge && av.cheapFrom.age <= av.capAge,
        `${s}: the sub-500 age is outside the range a buyer shops in`);
      assert(av.cheapFrom.cost <= 500, `${s}: sub-500 age costs ${av.cheapFrom.cost}`);
      assert(av.at(av.cheapFrom.age - 1) === null || av.at(av.cheapFrom.age - 1) > 500,
        `${s}: the sub-500 age is not the first one`);
    }
    if (av.bend) {
      assert(av.bend.published, `${s}: published a bend that failed its own guard`);
      assert(av.bend.F >= 10, `${s}: bend published at F=${av.bend.F}`);
      assert(Math.abs(av.bend.early - av.bend.late) >= 0.03, `${s}: bend published on a 0pp difference`);
      assert(av.bend.age >= 4 && av.bend.age <= 15, `${s}: bend outside the tested window`);
    }
  }
  const bends = depSlugs.filter(s => {
    const rec = models[s], f = depreciationFit(rec);
    return (depreciationAge(rec, f, builtAt) || {}).bend;
  }).length;
  assert(bends <= depSlugs.length * 0.5,
    `${bends}/${depSlugs.length} models publish a bend — the guard is not holding`);
});

check("the depreciation JSON twin carries the citable numbers", () => {
  for (const s of depSlugs.slice(0, 12)) {
    const rec = models[s], fit = depreciationFit(rec);
    const j = depreciationJson(rec, s, fit, depreciationAge(rec, fit, builtAt), { host: HOST, builtAt });
    const round = JSON.parse(JSON.stringify(j));
    assert(!/null,\s*"annual_depreciation_rate"/.test(JSON.stringify(round)), `${s}: rate missing`);
    assert(round.annual_depreciation_rate > 0 && round.annual_depreciation_rate < 0.3, `${s}: bad rate`);
    assert(round.sample_size > 0 && round.collected_until, `${s}: no provenance in the JSON`);
    assert(round.half_life_years > 1, `${s}: no half-life`);
    assert(round.cost_of_one_year_of_age.length >= 8, `${s}: euro ladder too short`);
    assert(round.rate_bend || round.rate_bend_note, `${s}: silent about the bend test`);
    assert(!JSON.stringify(round).includes("NaN"), `${s}: NaN leaked into the JSON`);
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
    assert(html.includes("ao mesmo ano") || html.includes("Ao mesmo ano"),
      `${a}-vs-${b}: no same-model-year price comparison on the page`);
    assert(html.includes("Preço lado a lado, ano a ano"),
      `${a}-vs-${b}: the year-by-year table is missing`);
  }
});

check("hubs render", () => {
  const depRows = depSlugs.map(s => {
    const r = models[s], f = depreciationFit(r), av = depreciationAge(r, f, builtAt);
    return { slug: s, b: r.b, m: r.m, n: r.n, rate: f.rate, span: f.span,
             half: av && av.halfLife, cheapAge: av && av.cheapFrom ? av.cheapFrom.age : null };
  }).sort((x, y) => y.rate - x.rate);
  assertPage(renderDepreciationHub({ rows: depRows, stats, host: HOST, depositCount: 0, builtAt }),
    { indexable: true, canonical: `https://${HOST}/depreciacao`, label: "depreciacao hub" });

  assertPage(renderCompareHub({ pairs, models, host: HOST, depositCount: 0, builtAt }),
    { indexable: true, canonical: `https://${HOST}/comparar`, label: "comparar hub" });

  const liq = Object.entries(models).filter(([, r]) => (r.lq && r.lq.s30 != null) || r.sd != null)
    .map(([slug, r]) => ({ slug, b: r.b, m: r.m, sd: r.sd, sn: r.sn, fm: r.fm,
                           lq: (r.lq && r.lq.s30 != null) ? r.lq : null, page: liquidityOk(r) }))
    .sort((x, y) => (y.lq ? y.lq.s30 : 0) - (x.lq ? x.lq.s30 : 0));
  const liqHub = renderLiquidityHub({ rows: liq, market, host: HOST, depositCount: 0, builtAt });
  assertPage(liqHub, { indexable: true, canonical: `https://${HOST}/liquidez`, label: "liquidez" });
  if (liq.some(r => r.page)) {
    assert(liqHub.includes(`href="/liquidez/${liq.find(r => r.page).slug}"`),
      "liquidity hub does not link the per-model pages");
  }
  const noCurve = { slug: "x-y", b: "X", m: "Y", sd: 21, sn: 30, fm: 6000, lq: null, page: false };
  assertPage(renderLiquidityHub({ rows: [noCurve], market: null, host: HOST, depositCount: 0, builtAt }),
    { indexable: true, canonical: `https://${HOST}/liquidez`, label: "liquidez (blob sem curva)" });

  const gap = Object.entries(models).filter(([, r]) => r.gm > 0 && r.fm > 0)
    .map(([slug, r]) => ({ slug, b: r.b, m: r.m, fm: r.fm, gm: r.gm, n: r.n, gap: r.fm / r.gm - 1,
                           s30: (r.lq && r.lq.s30 != null) ? r.lq.s30 : null, page: liquidityOk(r) }))
    .sort((x, y) => y.gap - x.gap);
  assertPage(renderValuationGap({ over: gap.slice(0, 25), under: gap.slice(-25).reverse(), market, stats, host: HOST, depositCount: 0, builtAt }),
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

check("model quality is rendered when measured, and absent when not", () => {
  const mq = { mae: 1665, mape: 25.7, r2: 0.915, cov: 0.809, n: 79532, folds: 5, ts: "2026-08-30" };
  const meth = renderMethodology({ stats, mq, host: HOST, depositCount: 0, builtAt });
  assertPage(meth, { indexable: true, canonical: `https://${HOST}/metodologia`, label: "metodologia+mq" });
  assert(meth.includes("25,7%"), "measured MAPE not rendered on /metodologia");
  assert(meth.includes("81%"), "measured band coverage not rendered on /metodologia");
  assert(/79\D?532/.test(meth), "sample size behind the measurement not rendered");
  const about = renderAbout({ stats, mq, host: HOST, depositCount: 0, builtAt });
  assert(about.includes("25,7%") && about.includes("81%"), "measured error not carried to /sobre");

  for (const empty of [null, undefined, {}, { mae: 1665 }]) {
    for (const [label, page] of [["metodologia", renderMethodology({ stats, mq: empty, host: HOST, depositCount: 0, builtAt })],
                                 ["sobre", renderAbout({ stats, mq: empty, host: HOST, depositCount: 0, builtAt })]]) {
      assertPage(page, { indexable: true, canonical: `https://${HOST}/${label}`, label: `${label}-no-mq` });
      assert(!/undefined|NaN/.test(page), `${label} leaked a placeholder with no measurement`);
      assert(!/erra em média <b>/.test(page), `${label} claimed an error rate it was not given`);
    }
  }
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

check("months are cut from the weekly rows, and only when closed", () => {
  assert(isoWeekMonth("2026-W35") === "2026-08", `W35 landed in ${isoWeekMonth("2026-W35")}`);
  assert(isoWeekMonth("2026-W36") === "2026-09", `W36 landed in ${isoWeekMonth("2026-W36")}`);
  assert(weeksOfMonth("2026-08").join() === "2026-W32,2026-W33,2026-W34,2026-W35",
    `August's weeks came out as ${weeksOfMonth("2026-08").join()}`);
  assert(monthLabel("2026-08") === "agosto de 2026", "month label is not Portuguese");

  const row = (week, priceMed, listings) => ({ week, date: "2026-08-01", models: 10, listings,
                                               priceMed, kmMed: 170000, sellMed: 30, depMed: 0.1 });
  const hist = [row("2026-W32", 8000, 100), row("2026-W33", 8200, 110),
                row("2026-W34", 8400, 120), row("2026-W35", 8600, 130),
                row("2026-W36", 9000, 200)];
  const cuts = monthlyCuts(hist, "2026-W37");
  assert(cuts.length === 1 && cuts[0].month === "2026-08",
    `expected only closed August, got ${cuts.map(c => c.month).join()}`);
  assert(cuts[0].priceMed === 8300, `August median price came out ${cuts[0].priceMed}`);
  assert(cuts[0].listings === 115, `August median listings came out ${cuts[0].listings}`);
  assert(cuts[0].n === 4 && cuts[0].monthWeeks === 4 && cuts[0].missing.length === 0,
    "August coverage is misreported");
  assert(cuts[0].from === "2026-08-03" && cuts[0].to === "2026-08-30",
    `August period came out ${cuts[0].from}—${cuts[0].to}`);
  assert(monthlyCuts(hist, "2026-W36").every(c => c.month !== "2026-09"),
    "published a month that has not closed");
  assert(monthlyCuts([row("2026-W35", 8600, 130)], "2026-W40").length === 0,
    `published a month off fewer than ${IDX_MIN_MONTH_WEEKS} weeks`);
  const holed = monthlyCuts([row("2026-W32", 8000, 100), row("2026-W35", 8600, 130)], "2026-W40");
  assert(holed.length === 1 && holed[0].n === 2 && holed[0].missing.length === 2,
    "a month with a gap did not report it");
});

check("the monthly page renders and stays on its own address", () => {
  const row = (week, priceMed) => ({ week, date: "2026-08-01", models: stats.models, listings: stats.listings,
                                     priceMed, kmMed: stats.kmMed, sellMed: stats.sellMed, depMed: stats.depMed, builtAt });
  const cuts = monthlyCuts([
    row("2026-W27", 7600), row("2026-W28", 7700), row("2026-W29", 7800), row("2026-W30", 7900),
    row("2026-W32", 8000), row("2026-W33", 8200), row("2026-W34", 8400), row("2026-W35", 8600),
  ], "2026-W40");
  assert(cuts.length === 2, `expected July and August, got ${cuts.map(c => c.month).join()}`);
  const aug = cuts[1];
  const html = renderMarketMonth({ cut: aug, months: cuts, host: HOST, depositCount: 0 });
  assertPage(html, { indexable: true, canonical: `https://${HOST}/mercado/indice/2026-08`, label: "indice month" });
  assert(html.includes("agosto de 2026"), "the month page never names its month");
  assert(html.includes("/mercado/indice/2026-w35"), "the month page does not link the weeks behind it");
  assert(html.includes("vs. julho de 2026"), "the month page does not compare with the previous published month");
  assert(html.includes(`https://${HOST}/mercado/indice/2026-08`), "the month page never states its permanent address");

  const snap = { week: "2026-W36", date: "2026-09-01", builtAt, models: stats.models, listings: stats.listings,
                 priceMed: stats.priceMed, kmMed: stats.kmMed, sellMed: stats.sellMed, depMed: stats.depMed };
  const hub = renderMarketIndex({ snapshot: snap, history: [snap], host: HOST, depositCount: 0, months: cuts });
  assert(hub.includes("/mercado/indice/2026-08") && hub.includes("Arquivo mensal"),
    "the hub does not link its monthly archive");
  const bare = renderMarketIndex({ snapshot: snap, history: [snap], host: HOST, depositCount: 0, months: [] });
  assert(!bare.includes("/mercado/indice/2026-08"), "the hub links a month it has no cut for");
  assert(bare.includes("Ainda não há nenhum mês fechado"), "the empty monthly archive says nothing");
});

check("district pages take the right Portuguese article", () => {
  // "em Porto" reads wrong to a Portuguese speaker; Porto takes the article.
  // Everything else is bare "em". The article is lexical, so this is a list, and
  // a list needs a test or it silently loses an entry.
  const mk = (key, lbl) => renderDistrictPage({
    key, rec: { lbl, n: 4000, fl: 4000, fm: 8000, fh: 14000, kmm: 175000, top: [] },
    models, stats, host: HOST, depositCount: 0, builtAt,
  });
  const porto = mk("porto", "Porto");
  assert(porto.includes("carros usados no Porto"), 'Porto lost its article ("em Porto")');
  assert(!porto.includes("carros usados em Porto"), 'Porto still says "em Porto" somewhere');
  assert(porto.includes("<title>Preços de carros usados no Porto"), "the title is what people see in the SERP");
  for (const [key, lbl] of [["lisboa", "Lisboa"], ["braga", "Braga"], ["faro", "Faro"], ["setubal", "Setúbal"]]) {
    const h = mk(key, lbl);
    assert(h.includes(`carros usados em ${lbl}`), `${lbl} should be bare "em"`);
  }
});

check("a district page stands on the country when it cannot stand on models", () => {
  const thin = renderDistrictPage({
    key: "braganca",
    rec: { lbl: "Bragança", n: 107, fl: 3500, fm: 6000, fh: 11000, kmm: 195000, top: [] },
    models, districts: mdoc.districts || {}, stats, host: HOST, depositCount: 0, builtAt,
  });
  assertPage(thin, { indexable: true, canonical: `https://${HOST}/precos/braganca`, label: "precos/braganca" });
  assert(thin.includes("Modelo a modelo, aqui não dá"), "a thin district hides why the table is missing");
  assert(!thin.includes("<th>Mediano nacional</th>"), "a thin district renders an empty model table");

  const ds = mdoc.districts || {};
  const keys = Object.keys(ds);
  if (keys.length >= 3) {
    const k = keys[0];
    const page = renderDistrictPage({
      key: k, rec: ds[k], models, districts: ds, stats, host: HOST, depositCount: 0, builtAt,
    });
    assertPage(page, { indexable: true, canonical: `https://${HOST}/precos/${k}`, label: `precos/${k}` });
    assert(page.includes("mais caro</b>"), `${k}: no place in the national ranking`);
    const other = keys.find(x => x !== k);
    assert(page.includes(`href="/precos/${other}"`), `${k}: the ranking does not link the other districts`);
    const rank = districtRanking(ds, k);
    assert(rank.pos >= 1 && rank.pos <= rank.total, `${k}: ranking position out of range`);
  }
});

const DUEL = {
  a: { n: 243, r: 0.063, km: 260000, fm: 5500 },
  b: { n: 238, r: 0.085, km: 152768, fm: 11950 },
  ci: 0.0094, t: 4.57, r2: 0.77, y0: 1981, y1: 2023,
  gap: [[11, -0.236, 0.089], [25, 0.062, 0.118]],
};

check("Portuguese contracts the preposition with the article", () => {
  assert(withPrep("de", "a caixa automática") === "da caixa automática", "de + a");
  assert(withPrep("de", "o diesel") === "do diesel", "de + o");
  assert(withPrep("a", "a caixa manual") === "à caixa manual", "a + a");
  assert(withPrep("a", "o diesel") === "ao diesel", "a + o");
  assert(withPrep("de", "gasolina") === "de gasolina", "no article, no contraction");
  for (const S of Object.values(DUELS)) {
    for (const side of [S.a, S.b]) {
      assert(/^(o|a) /.test(side.subj), `${side.lbl}: subject carries no article`);
    }
  }
});

check("the gearbox facet reads as a gearbox, not as a fuel", () => {
  const rec = models[deep];
  const cells = [
    { k: "manual", lbl: "Manual", n: 60, fl: 2800, fm: 4999, fh: 7500, km: 232000, y0: 2004, y1: 2016,
      vs: { automatica: [0.63, 9] }, vsm: [0.98, 27] },
    { k: "automatica", lbl: "Automática", n: 20, fl: 14500, fm: 21500, fh: 27000, km: 127000, y0: 2017, y1: 2024,
      vs: { manual: [1.59, 9] }, vsm: [1.08, 13] },
  ];
  const html = renderFacetPage({
    rec, slug: deep, kind: "transmission", cell: cells[1], siblingsCells: cells,
    stats, host: HOST, depositCount: 0, builtAt,
  });
  assertPage(html, { indexable: true, canonical: `https://${HOST}/preco/${deep}/automatica`, label: "gearbox facet" });
  assert(html.includes("com caixa automática"), "gearbox facet does not name the gearbox");
  assert(!html.includes("no distrito"), "gearbox facet reuses the district preposition");
  assert(html.includes("59% mais caro"), "gearbox facet lost the year-matched gap");
  assert(html.includes("vehicleTransmission"), "gearbox facet does not mark up the gearbox");

  const bare = cells.map(c => ({ ...c, vs: undefined, vsm: undefined }));
  const naked = renderFacetPage({
    rec, slug: deep, kind: "transmission", cell: bare[1], siblingsCells: bare,
    stats, host: HOST, depositCount: 0, builtAt,
  });
  assert(!/% (mais caro|mais barato)/.test(naked), "claimed a gap it could not measure year-matched");
  assert(naked.includes("inclui a diferença de anos"), "silently dropped the comparison instead of explaining it");
});

for (const kind of Object.keys(DUELS)) {
  const S = DUELS[kind];
  check(`the ${kind} duel page states both curves and its own margin`, () => {
    const rec = { ...models[deep], [S.key]: DUEL };
    const av = duel(rec, kind, builtAt);
    assert(av && av.decisive && av.winner === "a", "the fit was read wrong");
    const html = renderDuelPage({ rec, slug: deep, av, stats, host: HOST, depositCount: 0, builtAt });
    assertPage(html, { indexable: true, canonical: `https://${HOST}/${S.path}/${deep}`, label: `${kind} duel` });
    assert(html.includes("6,3%") && html.includes("8,5%"), "one of the two rates is missing");
    assert(html.includes("±0,9 pp"), "the page hides the margin on its own claim");
    assert(html.includes("quilometragem igualada"), "the page never says the mileage is controlled");
    assert(html.includes("preços pedidos"), "the page passes asking prices off as sales");
    assert(html.includes(`/preco/${deep}/${S.a.facet}`) && html.includes(`/preco/${deep}/${S.b.facet}`),
      "the duel page does not link the two facet cuts it is built from");

    const flat = { ...rec, [S.key]: { ...DUEL, b: { ...DUEL.b, r: 0.064 }, t: 0.4 } };
    const fav = duel(flat, kind, builtAt);
    assert(!fav.decisive, "a 0.1pp difference was called decisive");
    const nullPage = renderDuelPage({ rec: flat, slug: deep, av: fav, stats, host: HOST, depositCount: 0, builtAt });
    assert(nullPage.includes("não decide a desvalorização"), "the null result is not stated as a result");
    assert(!nullPage.includes("segura melhor o preço.</b>"), "a null result claims a winner");

    const j = duelJson(rec, deep, av, { host: HOST, builtAt });
    assert(j.holds_value_better === S.a.json && j.rate_difference_pp_per_year > 0,
      "JSON twin disagrees with the page");
    assert(j[S.a.json] && j[S.b.json], "JSON twin does not name its two sides");
    assert(JSON.parse(JSON.stringify(j)), "duel JSON is not serialisable");

    const hub = renderDuelHub({
      spec: S, rows: [{ slug: deep, b: rec.b, m: rec.m, av }], other: null,
      stats, host: HOST, depositCount: 0, builtAt,
    });
    assertPage(hub, { indexable: true, canonical: `https://${HOST}/${S.path}`, label: `${kind} duel hub` });
    assert(hub.includes(`/${S.path}/${deep}`), "the hub does not link its own page");
  });
}

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
    deals: [deal], zone: "all", sort: "score", view: "comprar",
    depositCount: 0, zoneCounts: { all: 1, norte: 1, centro: 0, sul: 0 },
    host: HOST, builtAt,
    modelLinks: [{ slug: deep, b: models[deep].b, m: models[deep].m, fm: models[deep].fm, count: 1 }],
  });
  assertPage(grid, { indexable: true, canonical: `https://${HOST}/mercado`, label: "mercado" });
  assert(grid.includes(`/preco/${deep}`), "/mercado does not link any model page");
  assert(grid.includes("/mercado/indice"), "/mercado does not link the index");

  assertPage(renderLanding({
    stats: { deals: 1, avgDisc: "17%", totalProfit: "€1 500" }, featured: deal,
    depositCount: 0, host: HOST,
  }), { indexable: true, label: "landing" });

  const w = renderModelWidget({ rec: models[deep], slug: deep, host: HOST });
  assert(w.includes("noindex,follow"), "widget is no longer noindex");
  assert(w.includes(`/preco/${deep}`), "widget lost its attribution link");

  assertPage(renderInfo({ zone: "all", depositCount: 0, title: "T", message: "M" }),
    { indexable: false, label: "info" });
});

check("a cut whose gap is entirely age says so instead of printing a percentage", () => {
  const rec = { b: "Renault", m: "Clio", n: 458, fl: 2000, fm: 3900, fh: 7000 };
  const base = { k: "faro", lbl: "Faro", n: 18, fl: 2387, fm: 8595, fh: 9000, y0: 1995, y1: 2019 };
  const args = c => ({
    rec, slug: "renault-clio", kind: "district", cell: c, siblingsCells: [c],
    stats, host: HOST, depositCount: 0, builtAt,
  });

  const flat = renderFacetPage(args({ ...base, dr: [1.0, 17] }));
  assert(!/0% (mais|menos)/.test(flat), "printed a 0% difference as if it were one");
  assert(flat.includes("pede o mesmo que o mesmo modelo fora deste distrito"),
    "a cut in line with its comparison base does not say so");
  assert(!/comparando ano a ano/.test(flat),
    "described a per-listing ratio as a year-by-year comparison");
  assert(flat.includes("pela mediana do resto do modelo no seu próprio ano"),
    "did not say how the age-controlled number was computed");
  const flatDesc = (flat.match(/<meta name="description" content="([^"]*)"/) || [])[1] || "";
  assert(!/(mais|menos) \d+%/.test(flatDesc),
    "the meta description claims a gap the data does not show");

  const real = renderFacetPage(args({ ...base, dr: [1.22, 17] }));
  assert(/mais 22%/.test(real), "a gap big enough to state was not stated");
  const realDesc = (real.match(/<meta name="description" content="([^"]*)"/) || [])[1] || "";
  assert(/mais 22%/.test(realDesc), "the meta description dropped a gap the page states");
  assert(real.includes("misturas de idades diferentes"),
    "did not explain why the raw medians are further apart");

  const noise = renderFacetPage(args({ ...base, vsm: [1.02, 6] }));
  assert(!/(mais|menos) 2%/.test(noise), "printed a 2% year-matched gap as a finding");
  assert(/comparando ano a ano/.test(noise), "a year-matched cut does not say which method it used");
  assert(noise.includes("pede <b>o mesmo</b>"), "a null result is not reported as one");
});

check("JSON twins carry the sample size and the date", () => {
  const rec = models[deep];
  const j = modelJson(rec, deep, { host: HOST, builtAt, models });
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

check("the negotiation facts show up on a pasted listing", () => {
  const rec = { t: "VW Golf 1.6 TDI", y: 2015, km: 150000, fu: "Diesel", p: 9000,
                fl: 9500, fm: 11000, fh: 12500, ct: "Porto", sd: 29, dom: 68,
                ph: [[68, 10500], [40, 9900], [6, 9000]], mf: "fuga de óleo" };
  const out = renderAvaliar({ rec, olxId: "JqGTZ", sourceUrl: null, query: "", models,
                              spec: null, depositCount: 0, host: HOST, builtAt });
  assert(out.includes("baixou o preço"), "the price track is not rendered");
  assert(out.includes("2 vezes"), "the number of cuts is wrong");
  assert(out.includes("−14%"), "the size of the cut is wrong");
  assert(out.includes("68 dias"), "days on market are not rendered");
  assert(out.includes("fuga de óleo"), "the seller's own words are not quoted");
});

check("a listing sold for parts says so instead of quoting a fair price", () => {
  const rec = { t: "VW Golf", y: 2015, p: 900, fl: 9500, fm: 11000, fh: 12500,
                hb: "para peças" };
  const out = renderAvaliar({ rec, olxId: "JqGTZ", sourceUrl: null, query: "", models,
                              spec: null, depositCount: 0, host: HOST, builtAt });
  assert(out.includes("para peças"), "the blocking phrase is not quoted");
  assert(out.includes("não se aplica"), "the fair band is not disclaimed");
});

check("a listing with no history at all still renders", () => {
  const rec = { t: "VW Golf", y: 2015, p: 9000, fl: 9500, fm: 11000, fh: 12500 };
  const out = renderAvaliar({ rec, olxId: "JqGTZ", sourceUrl: null, query: "", models,
                              spec: null, depositCount: 0, host: HOST, builtAt });
  assert(out.includes("Preço pedido"), "the card broke without the optional facts");
  assert(!out.includes("baixou o preço"), "invented a price cut out of nothing");
  assert(!out.includes("Anúncio online"), "invented an age out of nothing");
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

check("brand and revenue copy: Carsbuyer everywhere, no single-revenue claim", () => {
  const priv = renderPrivacy({ depositCount: 0, host: HOST, contact: "x@y.pt" });
  assert(priv.includes('og:site_name" content="Carsbuyer"'), "og:site_name is not Carsbuyer");
  assert(!priv.includes("Flipper Club"), "Flipper Club still leaks into a public page");
  assert(priv.includes("mailto:x@y.pt"), "privacy page does not use the configured contact");
  assert(priv.includes("90 dias"), "privacy page does not state the lead retention");
  const deal = {
    olx_id: "ID1", brand: models[deep].b, model: models[deep].m, title: "Carro de teste",
    price_eur: 7000, fair_median: 8500, fair_low: 7600, fair_high: 9400, discount_pct: 0.17,
    est_profit_eur: 1500, year: 2014, mileage_km: 180000, fuel_type: "Diesel",
    district: "Porto", photo_urls: [], days_on_market: 12, first_seen_at: "2026-08-01T00:00:00Z",
    seller_type: "Particular",
  };
  const landing = renderLanding({ stats: { deals: 1, avgDisc: "17%", totalProfit: "€1 500" }, featured: deal,
                                  depositCount: 0, host: HOST });
  assert(!landing.includes("única receita"), "landing still says the deposit is the only revenue");
  assert(landing.includes('href="/vender"'), "landing seller chip does not lead to the seller hub");
  assert(!landing.includes("em breve"), "seller path is still a placeholder");
});

check("no public page still sells a deposit", () => {
  const deal = {
    olx_id: "ID1", brand: models[deep].b, model: models[deep].m, title: "Carro de teste",
    price_eur: 7000, fair_median: 8500, fair_low: 7600, fair_high: 9400, discount_pct: 0.17,
    est_profit_eur: 1500, year: 2014, mileage_km: 180000, fuel_type: "Diesel",
    district: "Porto", photo_urls: [], days_on_market: 12, first_seen_at: "2026-08-01T00:00:00Z",
    seller_type: "Particular", url: "https://www.olx.pt/d/anuncio/teste",
  };
  const car = renderCarPage({ deal, zone: "all", view: "comprar", depositCount: 0,
                              modelHref: `/preco/${deep}`, host: HOST });
  assertPage(car, { indexable: false, label: "car" });
  assert(car.includes(`href="${deal.url}"`), "car page does not link the seller's OLX ad");
  assert(car.includes('rel="noopener nofollow"'), "the OLX link lost rel=noopener nofollow");
  assert(car.includes(deal.seller_type), "car page no longer names the seller type");

  const pages = [
    ["landing", renderLanding({ stats: { deals: 1, avgDisc: "17%", totalProfit: "€1 500" },
                                featured: deal, depositCount: 0, host: HOST })],
    ["grid", renderGrid({ deals: [deal], zone: "all", sort: "score", view: "comprar",
                          depositCount: 0, zoneCounts: { all: 1, norte: 1, centro: 0, sul: 0 },
                          host: HOST, builtAt })],
    ["car", car],
    ["avaliar", renderAvaliar({ rec: null, olxId: null, sourceUrl: null, query: "", models,
                                spec: null, depositCount: 0, host: HOST, builtAt })],
    ["privacidade", renderPrivacy({ depositCount: 0, host: HOST, contact: "x@y.pt" })],
    ["sobre", renderAbout({ stats, host: HOST, depositCount: 0, builtAt })],
  ];
  for (const [label, html] of pages) {
    for (const word of [/dep[óo]sito/i, /\bStripe\b/, /\/reservas/, /\/claim/, /desbloquear/i]) {
      assert(!word.test(html), `${label} still says ${word}`);
    }
  }
});

check("titles lead with the number", () => {
  const rec = models[deep];
  const page = renderModelPage({ rec, slug: deep, liveDeals: [], siblings: [], host: HOST, depositCount: 0, builtAt });
  const t = page.match(/<title>([^<]*)<\/title>/)[1];
  assert(t.includes("€") && t.includes("anúncios"), `model title has no number: ${t}`);
  const [s, y] = yearPages[0];
  const yrec = models[s];
  const all = yearCells(yrec, 1).slice().sort((a, b) => a.y - b.y);
  const i = all.findIndex(c => c.y === y);
  const yp = renderYearPage({
    rec: yrec, slug: s, year: y, cell: yearCell(yrec, y),
    neighbours: { older: i > 0 ? all[i - 1] : null, newer: i < all.length - 1 ? all[i + 1] : null,
                  window: all.slice(Math.max(0, i - 3), i + 4).sort((a, b) => b.y - a.y) },
    liveDeals: [], pageYears: yearPageYears(yrec), stats, host: HOST, depositCount: 0, builtAt,
    historyUrl: "https://example.test/h",
  });
  const yt = yp.match(/<title>([^<]*)<\/title>/)[1];
  assert(yt.includes("€") && yt.includes(String(y)), `year title has no number: ${yt}`);
  assert(yp.includes('rel="nofollow sponsored noopener"'), "year page history link is not marked sponsored");
  assert(yp.includes('href="/ir/historico?from=ano"') && !yp.includes("https://example.test/h"), "year page history link must go through the counted redirect");
});

check("the seller lead form and the history block render on /avaliar", () => {
  const slug = deep;
  const withSpec = renderAvaliar({ rec: null, olxId: null, sourceUrl: null, query: "", models,
    spec: { rec: models[slug], slug, year: 2016, cell: null }, depositCount: 0, host: HOST, builtAt,
    historyUrl: "https://example.test/h" });
  assert(withSpec.includes('action="/lead"'), "no lead form on the spec estimate");
  assert(withSpec.includes('name="consent"'), "lead form has no consent box");
  assert(withSpec.includes('id="vender"'), "lead form lost its anchor");
  assert(withSpec.includes('id="escolher"'), "model picker lost its anchor");
  const rec = { t: "VW Golf", y: 2015, km: 40000, fu: "Diesel", p: 9000, fl: 9500, fm: 11000, fh: 12500,
                imp: 1, ms: slug, sd: 20, dom: 70, ph: [[60, 11000], [30, 10000], [3, 9000]] };
  const pasted = renderAvaliar({ rec, olxId: "JqGTZ", sourceUrl: null, query: "", models, spec: null,
                                 depositCount: 0, host: HOST, builtAt, historyUrl: "https://example.test/h" });
  assert(pasted.includes('rel="nofollow sponsored noopener"'), "partner link is not marked sponsored");
  assert(pasted.includes('href="/ir/historico?from=avaliar"') && !pasted.includes("https://example.test/h"), "history link must go through the counted redirect");
  assert(pasted.includes("importação"), "import reason not listed");
  assert(pasted.includes("baixou 2 vezes"), "price-cut reason not listed");
  assert(pasted.includes("#vender"), "no path from a pasted listing to the seller form");
  const noUrl = renderAvaliar({ rec, olxId: "JqGTZ", sourceUrl: null, query: "", models, spec: null,
                                depositCount: 0, host: HOST, builtAt });
  assert(!noUrl.includes("sponsored"), "history block rendered with no partner url configured");
});

check("seller pages render with the numbers, the form and the JSON twin", () => {
  const eligible = slugs.filter(s => venderOk(models[s]));
  assert(eligible.length > 0, "no model is eligible for a seller page");
  const s = eligible.includes(deep) ? deep : eligible[0];
  const rec = models[s];
  const page = renderVenderPage({ rec, slug: s, market: mdoc.lqm || null, pageYears: yearPageYears(rec),
                                  hasLiquidity: liquidityOk(rec), hasDepreciation: false,
                                  host: HOST, depositCount: 0, builtAt });
  assertPage(page, { indexable: true, canonical: `https://${HOST}/vender/${s}`, label: "vender" });
  const t = page.match(/<title>([^<]*)<\/title>/)[1];
  assert(t.includes("€") && /^Vender /.test(t), `seller title is off: ${t}`);
  assert(page.includes('action="/lead"') && page.includes('id="vender"'), "seller page has no lead form");
  assert(page.includes(`/preco/${s}`), "seller page does not link the price page");
  assert(page.includes("Quanto pedir"), "seller page lost its main section");
  const types = ldTypes(page);
  for (const want of ["Dataset", "BreadcrumbList", "FAQPage"]) assert(types.has(want), `seller page is missing ${want}`);
  const j = venderJson(rec, s, { host: HOST, builtAt });
  assert(j.asking.median === rec.fm && Array.isArray(j.years), "seller JSON twin is malformed");
  const hub = renderVenderHub({ rows: eligible.slice(0, 5).map(x => ({ slug: x, b: models[x].b, m: models[x].m, n: models[x].n,
    fm: models[x].fm, fl: models[x].fl, fh: models[x].fh, sd: models[x].sd, s30: null, cu: null, cp: null })),
    market: mdoc.lqm || null, host: HOST, depositCount: 0, builtAt });
  assertPage(hub, { indexable: true, canonical: `https://${HOST}/vender`, label: "vender hub" });
  assert(hub.includes(`/vender/${eligible[0]}`), "hub does not link the seller pages");
});

check("a six-month window cell is labelled honestly on the year page and the JSON twin", () => {
  const [s] = yearPages[0];
  const rec = models[s];
  const cell = { y: 2009, n: 23, fl: 2500, fm: 3200, fh: 4100, km: 190000, pg: 1, w: 180, na: 4 };
  const page = renderYearPage({
    rec, slug: s, year: 2009, cell, neighbours: { older: null, newer: null, window: [cell] },
    liveDeals: [], pageYears: [2009], stats, host: HOST, depositCount: 0, builtAt,
  });
  assert(page.includes("últimos 6 meses"), "window label missing on the year page");
  assert(page.includes("4 ainda ativos"), "active share of the window sample missing");
  assert(!page.includes("<b>23 anúncios ativos</b>"), "window cell still called active listings");
  const twin = yearJson(rec, s, 2009, cell, { host: HOST, builtAt });
  assert(twin.window_days === 180 && twin.active_in_sample === 4, "JSON twin does not carry the window");
  assert(/fechados/.test(twin.measured_note), "JSON twin note does not mention closed listings");
  const active = renderYearPage({
    rec, slug: s, year: 2009, cell: { ...cell, w: undefined, na: undefined }, neighbours: { older: null, newer: null, window: [] },
    liveDeals: [], pageYears: [2009], stats, host: HOST, depositCount: 0, builtAt,
  });
  assert(active.includes("anúncios ativos</b>"), "active cell lost its wording");
});

check("the bare /avaliar page carries the valuation guide with real numbers", () => {
  const bare = renderAvaliar({ rec: null, olxId: null, sourceUrl: null, query: "", models, spec: null,
                               depositCount: 0, host: HOST, builtAt, market: mdoc.lqm || { s30: 0.6, md: 29, cu: 0.35, cp: 0.08 },
                               stats: corpusStats(models, builtAt) });
  assert(bare.includes("Como se calcula o valor de um carro usado"), "guide section missing");
  assert(bare.includes("Quanto vale um carro usado por idade"), "age table missing");
  assert(bare.includes("Perguntas frequentes"), "visible FAQ missing");
  assert((bare.match(/<details class="indep-note"/g) || []).length >= 4, "FAQ entries are not rendered");
  assert(ldTypes(bare).has("FAQPage"), "FAQ schema lost");
  assert(bare.includes("/vender") && bare.includes("/depreciacao"), "guide does not link the seller and depreciation layers");
  const rows = ageTable(models, builtAt);
  assert(rows.length >= 3 && rows.every(r => r.med > 0 && r.n >= 50), "age table rows are malformed");
  const withQuery = renderAvaliar({ rec: null, olxId: null, sourceUrl: null, query: "abc", models, spec: null,
                                    depositCount: 0, host: HOST, builtAt, market: null, stats: null });
  assert(!withQuery.includes("Quanto vale um carro usado por idade"), "guide leaked onto a noindex result page");
});

check("year pages carry the seller path and a prefilled lead form", () => {
  const [s, y] = yearPages[0];
  const rec = models[s];
  const cell = yearCell(rec, y);
  const base = { rec, slug: s, year: y, cell, neighbours: { older: null, newer: null, window: [cell] },
                 liveDeals: [], pageYears: [y], stats, host: HOST, depositCount: 0, builtAt };
  const withVender = renderYearPage({ ...base, hasVender: true });
  assert(withVender.includes(`/vender/${s}#vender`), "year page does not link the seller page");
  assert(withVender.includes('action="/lead"') && withVender.includes(`name="ano" min="1980" max="2027" required value="${y}"`),
    "year page lead form is missing or not prefilled with the year");
  const without = renderYearPage({ ...base, hasVender: false });
  assert(without.includes(`/avaliar?modelo=${encodeURIComponent(s)}&ano=${y}#vender`), "fallback seller path missing");
});

check("valuation results offer a WhatsApp share link back to the page", () => {
  const rec = { t: "VW Golf 1.6 TDI", y: 2015, km: 150000, fu: "Diesel", p: 9000, fl: 9500, fm: 11000, fh: 12500 };
  const pasted = renderAvaliar({ rec, olxId: "JqGTZ", sourceUrl: null, query: "", models, spec: null,
                                 depositCount: 0, host: HOST, builtAt });
  assert(pasted.includes("https://wa.me/?text="), "no WhatsApp link on the pasted result");
  assert(pasted.includes(encodeURIComponent(`https://${HOST}/avaliar?q=JqGTZ`)), "share link does not point back to the result");
  const slug = deep;
  const spec = renderAvaliar({ rec: null, olxId: null, sourceUrl: null, query: "", models,
                               spec: { rec: models[slug], slug, year: 2016, cell: null }, depositCount: 0, host: HOST, builtAt });
  assert(spec.includes(encodeURIComponent(`https://${HOST}/avaliar?modelo=${slug}&ano=2016`)), "spec result share link is wrong");
  const noHost = renderAvaliar({ rec, olxId: "JqGTZ", sourceUrl: null, query: "", models, spec: null, depositCount: 0, host: null, builtAt });
  assert(!noHost.includes("wa.me"), "share link rendered without a host to point at");
});

check("every seller guide renders as an indexable article with FAQ, sources and the lead form", () => {
  assert(GUIDES.length >= 6, "guide registry is too small");
  const st = corpusStats(models, builtAt);
  for (const guide of GUIDES) {
    const page = renderGuide({ guide, models, market: mdoc.lqm || { s30: 0.6, md: 29, cu: 0.35, cp: 0.08 }, stats: st,
                               host: HOST, depositCount: 0, builtAt });
    assertPage(page, { indexable: true, canonical: `https://${HOST}/guias/${guide.slug}`, label: `guia ${guide.slug}` });
    const t = ldTypes(page);
    for (const want of ["Article", "FAQPage", "BreadcrumbList"]) assert(t.has(want), `${guide.slug} is missing ${want}`);
    assert(page.includes('action="/lead"') && page.includes('name="nome_modelo" required'), `${guide.slug} has no free-text lead form`);
    assert(!page.includes("undefined") && !page.includes("NaN"), `${guide.slug} leaks undefined/NaN`);
    const words = page.replace(/<script[\s\S]*?<\/script>/g, "").replace(/<[^>]+>/g, " ").split(/\s+/).filter(Boolean).length;
    assert(words >= 450, `${guide.slug} is thin: ${words} words`);
  }
  const hub = renderGuidesHub({ market: mdoc.lqm || null, stats: st, host: HOST, depositCount: 0, builtAt });
  assertPage(hub, { indexable: true, canonical: `https://${HOST}/guias`, label: "guias hub" });
  for (const guide of GUIDES) assert(hub.includes(`/guias/${guide.slug}`), `hub does not link ${guide.slug}`);
});

console.log(failures ? `\n${failures} check(s) FAILED` : "\nall render checks passed");
process.exit(failures ? 1 : 0);
