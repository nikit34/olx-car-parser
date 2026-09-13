import { readFileSync } from "node:fs";
import worker from "../../flipper-club/src/index.js";
import { comparePairs, comparePairKey, comparePriceGap } from "../../flipper-club/src/seo-pages.js";
import { LOCALES, setIntlLocales } from "../../flipper-club/src/i18n.js";
import {
  gapRows, intlCompareSitemap,
  renderIntlComparePage, renderIntlCompareHub, renderIntlValuationGap, renderIntlModelWidget,
} from "../../flipper-club/src/intl-compare.js";

const HOST = "carsbuyer.org";
const NNBSP = " ";

let failures = 0;
function assert(cond, msg) { if (!cond) throw new Error(msg); }
async function check(name, fn) {
  try { await fn(); console.log(`  ok   ${name}`); }
  catch (err) { failures++; console.error(`  FAIL ${name}\n       ${err && err.message}`); }
}

const dePath = process.argv[2] || "tests/worker/fixtures/models_de.json";
const ddoc = JSON.parse(readFileSync(dePath, "utf8"));
const deModels = ddoc.models;

let served = ddoc;

const kv = new Map();
function makeEnv(intlLocales) {
  return {
    CANONICAL_HOST: HOST,
    INTL_LOCALES: intlLocales,
    KV: {
      async get(k, type) {
        const v = kv.get(k);
        if (v === undefined) return null;
        return type === "json" ? JSON.parse(v) : v;
      },
      async put(k, v) { kv.set(k, v); },
      async list() { return { keys: [] }; },
      async delete(k) { kv.delete(k); },
    },
    ASSETS: { async fetch() { return new Response("asset", { status: 200 }); } },
  };
}
const env = makeEnv("de,fr,it");
const envOff = makeEnv("");

const realFetch = globalThis.fetch;
const json = (obj, status = 200) => new Response(JSON.stringify(obj), { status });
globalThis.fetch = async (input, init) => {
  const u = typeof input === "string" ? input : input.url;
  if (u.includes("models_de.json")) return json(served);
  if (u.includes("models_fr.json") || u.includes("models_it.json")) {
    return new Response("not found", { status: 404 });
  }
  if (u.includes("models.json")) return new Response("not found", { status: 404 });
  if (u.includes("hot_deals_")) return json({ deals: [] });
  if (u.includes("valuations")) return json({ cars: {} });
  if (u.includes("import.json")) return json({ built_at: ddoc.built_at, models: {} });
  return realFetch(input, init);
};

const get = (path, e = env) =>
  worker.fetch(new Request(`https://${HOST}${path}`, { method: "GET" }), e);
const body = async (path, e = env) => (await get(path, e)).text();

setIntlLocales("de,fr,it");

const PAIRS = comparePairs(deModels);
const GAPS = gapRows(deModels);
const R = LOCALES.de.routes;
const CMP = `/de/${R.compare}`;
const GAP = `/de/${R.overvalued}`;

assert(PAIRS.length > 0, "the fixture generates no comparison pairs — the suite would prove nothing");
assert(GAPS.length >= 4, "the fixture has too few GBM bands to exercise the valuation-gap page");

const [pairA, pairB] = PAIRS[0];
const PAIR_PATH = `${CMP}/${pairA}-vs-${pairB}`;

const flipped = PAIRS.find(([a, b]) => {
  const g = comparePriceGap(deModels[a], deModels[b]);
  if (!g) return false;
  const byMedian = Math.sign(deModels[a].fm - deModels[b].fm);
  const byYear = Math.sign(g.ratio - 1);
  return byMedian !== 0 && byYear !== 0 && byMedian !== byYear;
});

function visibleText(html) {
  return html
    .replace(/<script[\s\S]*?<\/script>/g, " ")
    .replace(/<style[\s\S]*?<\/style>/g, " ")
    .replace(/<[^>]+>/g, " ")
    .replace(/&nbsp;/g, " ");
}

const FOREIGN = {
  de: ["anúncio", "anúncios", "preço", "preços", "usados", "desvalorização", "quilometragem",
       "comparação", "annonces", "kilométrage", "décote", "occasion", "médiane",
       "annunci", "chilometraggio", "svalutazione", "prezzo", "prezzi", "mediana"],
  fr: ["anúncio", "anúncios", "preço", "preços", "usados", "desvalorização",
       "Angebote", "Kilometerstand", "Wertverlust", "Erstzulassung", "Gebrauchtwagen",
       "annunci", "chilometraggio", "svalutazione", "prezzi", "mediana"],
  it: ["anúncio", "anúncios", "preço", "preços", "usados", "desvalorização",
       "Angebote", "Kilometerstand", "Wertverlust", "Erstzulassung", "Gebrauchtwagen",
       "annonces", "kilométrage", "décote", "médiane"],
};

function assertClean(code, html, where) {
  const seen = visibleText(html);
  for (const w of FOREIGN[code]) {
    assert(!new RegExp(w, "i").test(seen), `${where} leaks the foreign word "${w}"`);
  }
  assert(!/undefined|\[object Object\]|NaN(?![a-zA-Z])/.test(seen),
    `${where} rendered undefined/NaN into the page`);
  assert(!/\{[a-z_]+\}/.test(seen), `${where} left an unfilled {placeholder}`);
}

function metaDescription(html) {
  const m = /<meta name="description" content="([^"]*)">/.exec(html);
  return m ? m[1] : null;
}

function ldGraph(html) {
  const blocks = [...html.matchAll(/<script type="application\/ld\+json">([\s\S]*?)<\/script>/g)];
  assert(blocks.length === 1, `expected one JSON-LD block, got ${blocks.length}`);
  return JSON.parse(blocks[0][1].replace(/\\u003c/g, "<"))["@graph"];
}

function h1Of(html) {
  const m = /<h1[^>]*>([\s\S]*?)<\/h1>/.exec(html);
  return m ? m[1].replace(/<[^>]+>/g, "").trim() : null;
}

await check("the German comparison hub is a real page, not a translated Portuguese one", async () => {
  const r = await get(CMP);
  assert(r.status === 200, `${CMP} → ${r.status}`);
  const html = await r.text();
  assert(html.includes('<html lang="de-DE">'), "the hub does not declare lang de-DE");
  assert(html.includes("index,follow"), "the hub is not indexable");
  assert(html.includes(`<link rel="canonical" href="https://${HOST}${CMP}">`), "the hub canonical is wrong");
  assert(html.includes(`href="https://${HOST}${CMP}.json"`), "the hub does not advertise its JSON twin");
  const desc = metaDescription(html);
  assert(desc && desc.length > 60, `the hub description is ${desc ? desc.length : 0} chars, expected over 60`);
  assert(h1Of(html), "the hub has no h1");
  assert(html.includes(`href="${PAIR_PATH}"`), "the hub does not link its first pair");
  const chips = [...html.matchAll(new RegExp(`href="${CMP}/([a-z0-9-]+)"`, "g"))].map(m => m[1]);
  assert(chips.length === PAIRS.length,
    `the hub links ${chips.length} pairs, the generator publishes ${PAIRS.length}`);
  assert(html.includes("Modelljahr"), "the hub does not state the same-model-year rule in German");
  assert(html.includes("AutoScout24"), "the hub does not name its own source");
  assertClean("de", html, "the German hub");
  const g = ldGraph(html);
  const types = g.map(n => n["@type"]);
  for (const want of ["CollectionPage", "BreadcrumbList"]) {
    assert(types.includes(want), `the hub JSON-LD has no ${want}`);
  }
  assert(g.find(n => n["@type"] === "CollectionPage").inLanguage === "de-DE",
    "the hub CollectionPage is not in German");
});

await check("the hub JSON twin lists exactly the pairs the router serves", async () => {
  const r = await get(`${CMP}.json`);
  assert(r.status === 200, `${CMP}.json → ${r.status}`);
  assert(r.headers.get("content-type").startsWith("application/json"), "the hub twin is not JSON");
  const doc = await r.json();
  assert(doc.market === "DE" && doc.language === "de", "the hub twin does not declare the German market");
  assert(doc.source_url === `https://${HOST}${CMP}`, "the hub twin points elsewhere");
  assert(doc.method === "same_model_year", "the hub twin does not declare the comparison method");
  assert(doc.comparisons.length === PAIRS.length,
    `the hub twin lists ${doc.comparisons.length} pairs, the generator publishes ${PAIRS.length}`);
  const advertised = new Set(doc.comparisons.map(c => new URL(c.page).pathname));
  for (const [a, b] of PAIRS) {
    assert(advertised.has(`${CMP}/${a}-vs-${b}`), `the hub twin omits ${a}-vs-${b}`);
  }
  assert(!JSON.stringify(doc).includes("OLX"), "the hub twin still names the Portuguese source");
});

await check("a comparison page compares year by year, not median against median", async () => {
  assert(flipped, "the fixture has no pair where the two rules disagree — the check would be empty");
  const [a, b] = flipped;
  const ra = deModels[a], rb = deModels[b];
  const g = comparePriceGap(ra, rb);
  const dearerByYear = g.ratio > 1 ? ra : rb;
  const cheaperByMedian = ra.fm <= rb.fm ? ra : rb;
  assert(dearerByYear === cheaperByMedian,
    "the chosen pair no longer flips between the two rules");
  const html = await body(`${CMP}/${a}-vs-${b}`);
  const seen = visibleText(html);
  const dearerName = `${dearerByYear.b} ${dearerByYear.m}`;
  const pct = Math.round((Math.max(g.ratio, 1 / g.ratio) - 1) * 100);
  const claim = new RegExp(`${dearerName.replace(/[.*+?^${}()|[\\]\\\\]/g, "\\\\$&")}[^.]{0,120}${pct}`);
  assert(claim.test(seen),
    `the page does not name ${dearerName} as the dearer side at the same model year`);
  assert(/Alter|Modelljahr/.test(seen),
    "the page does not warn that the two headline medians are of different ages");
  const rows = [...html.matchAll(/<td class="mono">(\d{4})<\/td>/g)].map(m => +m[1]);
  assert(rows.length === Math.min(g.cells.length, 12),
    `the year table shows ${rows.length} rows, expected ${Math.min(g.cells.length, 12)}`);
  for (const y of rows) {
    assert(g.cells.some(c => c.y === y), `the year table shows ${y}, which is not a common year`);
  }
});

await check("a comparison page carries canonical, description, h1 and parseable JSON-LD", async () => {
  const r = await get(PAIR_PATH);
  assert(r.status === 200, `${PAIR_PATH} → ${r.status}`);
  const html = await r.text();
  assert(html.includes("index,follow"), "the comparison page is not indexable");
  assert(html.includes(`<link rel="canonical" href="https://${HOST}${PAIR_PATH}">`),
    "the comparison canonical is not its own URL");
  assert(html.includes(`href="https://${HOST}${PAIR_PATH}.json"`),
    "the comparison page does not advertise its JSON twin");
  const desc = metaDescription(html);
  assert(desc && desc.length > 60, `the comparison description is ${desc ? desc.length : 0} chars`);
  const h1 = h1Of(html);
  assert(h1 && h1.includes(deModels[pairA].m) && h1.includes(deModels[pairB].m),
    "the comparison h1 does not name both models");
  assert(html.includes(`href="/de/${R.model}/${pairA}"`) && html.includes(`href="/de/${R.model}/${pairB}"`),
    "the comparison page does not link both model pages");
  assert(new RegExp(`\\d${NNBSP}€`).test(html), "no German money formatting on the comparison page");
  assert(!/€\d/.test(html), "the comparison page prints Portuguese-style money");
  assert(!/Tage|Verkaufsdauer/.test(visibleText(html)),
    "the comparison page claims a time-to-sell figure, which is withheld for this market");
  assertClean("de", html, "the German comparison page");
  const g = ldGraph(html);
  const types = g.map(n => n["@type"]);
  for (const want of ["BreadcrumbList", "FAQPage"]) {
    assert(types.includes(want), `the comparison JSON-LD has no ${want}`);
  }
  const faq = g.find(n => n["@type"] === "FAQPage");
  assert(faq.mainEntity.length >= 1, "the comparison FAQPage is empty");
  for (const q of faq.mainEntity) {
    assert(q.name && q.acceptedAnswer.text, "an FAQ entry is empty");
    assert(!/<[a-z]/i.test(q.acceptedAnswer.text), "FAQ answers still carry markup");
  }
  const crumb = g.find(n => n["@type"] === "BreadcrumbList");
  assert(crumb.itemListElement.at(-1).item === `https://${HOST}${PAIR_PATH}`,
    "the comparison breadcrumb does not end at this page");
});

await check("the comparison JSON twin is year-by-year and matches the page", async () => {
  const r = await get(`${PAIR_PATH}.json`);
  assert(r.status === 200, `${PAIR_PATH}.json → ${r.status}`);
  assert(r.headers.get("content-type").startsWith("application/json"), "the comparison twin is not JSON");
  const doc = await r.json();
  const g = comparePriceGap(deModels[pairA], deModels[pairB]);
  assert(doc.source_url === `https://${HOST}${PAIR_PATH}`, "the comparison twin points elsewhere");
  assert(doc.market === "DE" && doc.language === "de", "the comparison twin is not the German market");
  assert(doc.comparison.method === "same_model_year", "the comparison twin does not declare its method");
  assert(doc.comparison.common_years === g.years,
    `the twin reports ${doc.comparison.common_years} common years, the generator finds ${g.years}`);
  assert(doc.by_year.length === g.cells.length, "the twin drops common years from by_year");
  assert(doc.models.length === 2 && doc.models.every(m => m.page.startsWith(`https://${HOST}/de/${R.model}/`)),
    "the twin does not link both locale model pages");
  assert(doc.models.every(m => m.sample_size > 0), "a compared model has no sample");
  const dearer = g.ratio > 1 ? pairA : pairB;
  assert(doc.comparison.dearer === dearer, "the twin names the wrong dearer side");
  assert(!JSON.stringify(doc).includes("OLX"), "the comparison twin names the Portuguese source");
});

await check("an invented comparison URL is a 404, not an invented page", async () => {
  const unpublished = (() => {
    const set = new Set(PAIRS.map(([a, b]) => comparePairKey(a, b)));
    const slugs = Object.keys(deModels);
    for (const a of slugs) for (const b of slugs) {
      if (a < b && !set.has(`${a}-vs-${b}`)) return `${a}-vs-${b}`;
    }
    return null;
  })();
  assert(unpublished, "every possible pair is published — the check would be empty");
  for (const bad of [`${CMP}/${unpublished}`, `${CMP}/gibt-es-nicht-vs-auch-nicht`,
                     `${CMP}/${pairA}`, `${CMP}/${pairA}-vs-${pairB}/extra`]) {
    const r = await get(bad);
    assert(r.status === 404, `${bad} → ${r.status}, expected 404`);
    assert((await r.text()).includes('<html lang="de-DE">'), `${bad} does not 404 in German`);
  }
  assert((await get(`${CMP}/${unpublished}.json`)).status === 404,
    "the JSON twin of an unpublished pair is not a 404");
});

await check("the valuation-gap page splits both directions without repeating a model", async () => {
  const r = await get(GAP);
  assert(r.status === 200, `${GAP} → ${r.status}`);
  const html = await r.text();
  assert(html.includes("index,follow"), "the valuation-gap page is not indexable");
  assert(html.includes(`<link rel="canonical" href="https://${HOST}${GAP}">`), "its canonical is wrong");
  assert(html.includes(`href="https://${HOST}${GAP}.json"`), "it does not advertise its JSON twin");
  const desc = metaDescription(html);
  assert(desc && desc.length > 60, `the valuation-gap description is ${desc ? desc.length : 0} chars`);
  assert(h1Of(html), "the valuation-gap page has no h1");
  const linked = [...html.matchAll(new RegExp(`href="/de/${R.model}/([a-z0-9-]+)"`, "g"))].map(m => m[1]);
  assert(new Set(linked).size === linked.length,
    "a model appears in both directions of the valuation-gap table");
  for (const slug of linked) {
    assert(deModels[slug] && deModels[slug].gm > 0,
      `${slug} is listed without a published fair-value band`);
  }
  assert(linked.length === GAPS.filter(x => x.gap !== 0).length,
    `the page lists ${linked.length} models, ${GAPS.length} carry a band`);
  assert(html.includes("Schätzwert") || html.includes("Geschätzter Wert"),
    "the page does not use the German term for the estimate");
  assertClean("de", html, "the German valuation-gap page");
  const g = ldGraph(html);
  const types = g.map(n => n["@type"]);
  for (const want of ["CollectionPage", "BreadcrumbList", "FAQPage"]) {
    assert(types.includes(want), `the valuation-gap JSON-LD has no ${want}`);
  }
});

await check("the valuation-gap JSON twin carries both directions and the bands", async () => {
  const r = await get(`${GAP}.json`);
  assert(r.status === 200, `${GAP}.json → ${r.status}`);
  assert(r.headers.get("content-type").startsWith("application/json"), "the twin is not JSON");
  const doc = await r.json();
  assert(doc.source_url === `https://${HOST}${GAP}`, "the twin points elsewhere");
  assert(doc.models_with_estimate === GAPS.length, "the twin miscounts the models with an estimate");
  assert(doc.models_in_market === Object.keys(deModels).length, "the twin miscounts the market");
  const over = doc.asked_above_estimate, under = doc.asked_below_estimate;
  assert(over.length && under.length, "the twin has an empty direction on a fixture that has both");
  assert(over.every(x => x.deviation > 0), "a model below the estimate is listed as above");
  assert(under.every(x => x.deviation < 0), "a model above the estimate is listed as below");
  const seen = new Set([...over, ...under].map(x => x.slug));
  assert(seen.size === over.length + under.length, "the twin repeats a model across both directions");
  for (const x of [...over, ...under]) {
    assert(x.fair_value_estimate && x.fair_value_estimate.median > 0, `${x.slug} has no band in the twin`);
    assert(x.page.startsWith(`https://${HOST}/de/${R.model}/`), `${x.slug} does not link its locale page`);
  }
});

await check("a market with no fair-value band 404s instead of rendering an empty table", async () => {
  const stripped = JSON.parse(JSON.stringify(ddoc));
  for (const rec of Object.values(stripped.models)) {
    delete rec.gm; delete rec.gl; delete rec.gh;
    for (const c of rec.yr || []) { delete c.gm; delete c.gl; delete c.gh; }
  }
  served = stripped;
  try {
    const r = await get(GAP);
    assert(r.status === 404, `${GAP} with no bands → ${r.status}, expected 404`);
    const html = await r.text();
    assert(html.includes('<html lang="de-DE">'), "the empty-market 404 is not in German");
    assert((await get(`${GAP}.json`)).status === 404, "the JSON twin still answers with no bands");
    const xml = await body("/de/sitemap.xml");
    assert(!xml.includes(`<loc>https://${HOST}${GAP}</loc>`),
      "the sitemap still advertises the valuation-gap page after it stopped existing");
    const hub = await body(CMP);
    assert(!hub.includes(`href="${GAP}"`),
      "the comparison hub still links the valuation-gap page after it stopped existing");
  } finally {
    served = ddoc;
  }
  assert((await get(GAP)).status === 200, "the valuation-gap page did not come back with the real blob");
});

await check("the embeddable card is self-contained, noindex and canonical to the model page", async () => {
  const slug = pairA;
  const path = `/de/widget/${slug}`;
  const r = await get(path);
  assert(r.status === 200, `${path} → ${r.status}`);
  assert(r.headers.get("content-security-policy") === "frame-ancestors *",
    "the card cannot be embedded — no permissive frame-ancestors");
  const html = await r.text();
  assert(html.includes('<html lang="de-DE">'), "the card is not in German");
  assert(html.includes('content="noindex,follow"'), "the card is not noindex");
  assert(html.includes(`<link rel="canonical" href="https://${HOST}/de/${R.model}/${slug}">`),
    "the card is not canonical to its model page");
  const desc = metaDescription(html);
  assert(desc && desc.length > 60, `the card description is ${desc ? desc.length : 0} chars`);
  assert(h1Of(html), "the card has no h1");
  assert(!html.includes("fc-header") && !html.includes("class=\"footer\""),
    "the card drags in the shared header or footer");
  assert(!html.includes("fc-consent"), "the card carries the consent banner");
  assert((html.match(new RegExp(`https://${HOST}/de/${R.model}/${slug}`, "g")) || []).length >= 2,
    "the card does not link back to the canonical model page");
  assert(html.includes("Carsbuyer"), "the card drops the attribution");
  assert(new RegExp(`\\d${NNBSP}€`).test(html), "the card does not format money in German");
  assertClean("de", html, "the German card");
  assert((await get("/de/widget/gibt-es-nicht")).status === 404, "an unknown slug renders a card anyway");
  assert((await get("/de/widget")).status === 404, "the bare widget path is not a 404");
});

await check("the German sitemap advertises exactly the pages this module serves", async () => {
  const xml = await body("/de/sitemap.xml");
  const locs = [...xml.matchAll(/<loc>([^<]+)<\/loc>/g)].map(m => new URL(m[1]).pathname);
  const mine = intlCompareSitemap(LOCALES.de, deModels).map(u => u.path);
  assert(mine.length === PAIRS.length + 2,
    `sitemap() emits ${mine.length} paths, expected ${PAIRS.length + 2}`);
  for (const p of mine) {
    assert(locs.includes(p), `the German sitemap does not list ${p}`);
    const r = await get(p);
    assert(r.status === 200, `the sitemap advertises ${p} but the router answers ${r.status}`);
  }
  const ours = locs.filter(p => p === CMP || p.startsWith(`${CMP}/`) || p === GAP || p.startsWith(`${GAP}/`));
  assert(ours.length === mine.length,
    `the sitemap carries ${ours.length} of our URLs, sitemap() emits ${mine.length}`);
  assert(new Set(locs).size === locs.length, "the German sitemap has duplicates");
  assert(!locs.some(p => p.endsWith(".json")), "the sitemap advertises a JSON twin");
  assert(!locs.some(p => p.startsWith("/de/widget")), "the sitemap advertises the noindex card");
});

await check("the localised segments exist in all three markets and nowhere else", async () => {
  for (const [cc, keys] of [["fr", LOCALES.fr.routes], ["it", LOCALES.it.routes]]) {
    for (const path of [`/${cc}/${keys.compare}`, `/${cc}/${keys.overvalued}`, `/${cc}/widget/${pairA}`]) {
      const r = await get(path);
      assert(r.status === 503, `${path} with no blob → ${r.status}, expected 503`);
      assert((await r.text()).includes(`<html lang="${LOCALES[cc].lang}">`),
        `${path} does not degrade in its own language`);
    }
    const bare = await get(`/${cc}/widget`);
    assert(bare.status === 404,
      `/${cc}/widget → ${bare.status}; the card has no hub, so it must 404`);
    const slashed = await get(`/${cc}/widget/`);
    assert(slashed.status === 301,
      `/${cc}/widget/ → ${slashed.status}, expected the canonical-spelling redirect`);
    assert(new URL(slashed.headers.get("location"), `https://${HOST}`).pathname === `/${cc}/widget`,
      `/${cc}/widget/ redirects to ${slashed.headers.get("location")}, not to the bare path that 404s`);
  }
  for (const p of [CMP, `${CMP}.json`, PAIR_PATH, GAP, `/de/widget/${pairA}`]) {
    const r = await get(p, envOff);
    assert(r.status === 404, `${p} with no locales live → ${r.status}, expected 404`);
    assert((await r.text()).includes('<html lang="pt-PT">'),
      `${p} does not fall back to the Portuguese 404`);
  }
  const strayed = [];
  for (const cc of ["de", "fr", "it"]) {
    for (const key of ["compare", "overvalued"]) strayed.push(`/${LOCALES[cc].routes[key]}`);
  }
  for (const p of strayed.concat(["/de/comparar", "/de/sobrevalorizados"])) {
    const r = await get(p);
    assert(r.status === 404,
      `${p} → ${r.status}; a localised segment must not answer outside its own locale`);
  }
});

await check("the footer of every locale page reaches the comparison hub", async () => {
  const html = await body("/de");
  assert(html.includes(`href="${CMP}"`), "the German footer does not link the comparison hub");
  assert(html.includes("Modellvergleich"), "the German footer link has no German label");
  assertClean("de", html, "the German landing page with the extra footer link");
});

await check("the French pages read as French, not as translated German", async () => {
  const loc = LOCALES.fr;
  const pages = [
    ["hub", renderIntlCompareHub({ loc, host: HOST, pairs: PAIRS, models: deModels, builtAt: ddoc.built_at, hasGap: true })],
    ["comparison", renderIntlComparePage({ loc, host: HOST, a: pairA, b: pairB, ra: deModels[pairA], rb: deModels[pairB], builtAt: ddoc.built_at })],
    ["valuation gap", renderIntlValuationGap({ loc, host: HOST, rows: GAPS, stats: { gapMed: -0.02 }, models: deModels, builtAt: ddoc.built_at, hasPairs: true })],
    ["card", renderIntlModelWidget({ loc, host: HOST, rec: deModels[pairA], slug: pairA })],
  ];
  for (const [what, html] of pages) {
    assertClean("fr", html, `the French ${what}`);
    assert(h1Of(html), `the French ${what} has no h1`);
    const desc = metaDescription(html);
    assert(desc && desc.length > 60, `the French ${what} description is ${desc ? desc.length : 0} chars`);
    assert(html.includes(`<link rel="canonical" href="https://${HOST}/fr/`), `the French ${what} canonical is not a /fr URL`);
  }
  const hubHtml = pages[0][1];
  assert(/millésime/.test(hubHtml), "the French hub does not use the French term for model year");
  assert(/annonces/.test(hubHtml), "the French hub does not use the French word for listings");
  assert(new RegExp(`\\d${NNBSP}€`).test(hubHtml) || /€/.test(hubHtml), "no euro sign on the French hub");
  const cardHtml = pages[3][1];
  assert(cardHtml.includes(`https://${HOST}/fr/${LOCALES.fr.routes.model}/${pairA}`),
    "the French card does not link the French model page");
});

await check("the Italian pages read as Italian, not as translated German", async () => {
  const loc = LOCALES.it;
  const pages = [
    ["hub", renderIntlCompareHub({ loc, host: HOST, pairs: PAIRS, models: deModels, builtAt: ddoc.built_at, hasGap: true })],
    ["comparison", renderIntlComparePage({ loc, host: HOST, a: pairA, b: pairB, ra: deModels[pairA], rb: deModels[pairB], builtAt: ddoc.built_at })],
    ["valuation gap", renderIntlValuationGap({ loc, host: HOST, rows: GAPS, stats: { gapMed: 0.03 }, models: deModels, builtAt: ddoc.built_at, hasPairs: true })],
    ["card", renderIntlModelWidget({ loc, host: HOST, rec: deModels[pairA], slug: pairA })],
  ];
  for (const [what, html] of pages) {
    assertClean("it", html, `the Italian ${what}`);
    assert(h1Of(html), `the Italian ${what} has no h1`);
    const desc = metaDescription(html);
    assert(desc && desc.length > 60, `the Italian ${what} description is ${desc ? desc.length : 0} chars`);
    assert(html.includes(`<link rel="canonical" href="https://${HOST}/it/`), `the Italian ${what} canonical is not an /it URL`);
  }
  const gapHtml = pages[2][1];
  assert(/stima/.test(gapHtml), "the Italian valuation-gap page does not use the Italian word for estimate");
  assert(/annunci/.test(gapHtml), "the Italian valuation-gap page does not use the Italian word for listings");
});

await check("the four locales carry four different sentences, not one copied four times", async () => {
  const sample = ["cmp.hub_h1", "cmp.hub_lede", "cmp.h1", "cmp.v_dep_t", "gap.h1", "gap.lede", "wid.h", "wid.cta"];
  for (const key of sample) {
    const values = ["pt", "de", "fr", "it"].map(c => LOCALES[c].strings[key]);
    assert(new Set(values).size === 4, `"${key}" is not written separately for all four locales`);
    for (const v of values) assert(v.trim().length > 0, `"${key}" is empty somewhere`);
  }
});

console.log(failures ? `\n${failures} check(s) FAILED` : "\nall intl compare checks passed");
process.exit(failures ? 1 : 0);
