import { readFileSync } from "node:fs";
import worker from "../../flipper-club/src/index.js";
import "../../flipper-club/src/intl-facets.js";
import { publishedCells, yearPageYears } from "../../flipper-club/src/seo-pages.js";
import { LOCALES } from "../../flipper-club/src/i18n.js";
import { intlFacetKeys, regionKeys } from "../../flipper-club/src/intl-facets.js";

const HOST = "carsbuyer.org";

let failures = 0;
function assert(cond, msg) { if (!cond) throw new Error(msg); }
async function check(name, fn) {
  try { await fn(); console.log(`  ok   ${name}`); }
  catch (err) { failures++; console.error(`  FAIL ${name}\n       ${err && err.message}`); }
}

const dePath = process.argv[2] || "tests/worker/fixtures/models_de.json";
const ddoc = JSON.parse(readFileSync(dePath, "utf8"));
const deModels = ddoc.models;
const deDistricts = ddoc.districts || {};

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
  if (/models_(de|fr|it)\.json/.test(u)) return json(ddoc);
  if (u.includes("models.json")) return json(ddoc);
  if (u.includes("hot_deals_")) return json({ deals: [] });
  if (u.includes("valuations")) return json({ cars: {} });
  if (u.includes("import.json")) return json({ built_at: ddoc.built_at, models: {} });
  return realFetch(input, init);
};

const get = (path, e = env) => worker.fetch(new Request(`https://${HOST}${path}`), e);
const body = async (path, e = env) => (await get(path, e)).text();

function visibleText(html) {
  return html
    .replace(/<script[\s\S]*?<\/script>/g, " ")
    .replace(/<style[\s\S]*?<\/style>/g, " ")
    .replace(/<[^>]+>/g, " ")
    .replace(/&nbsp;/g, " ");
}

function metaContent(html, name) {
  const re = new RegExp(`<meta name="${name}" content="([^"]*)"`);
  const m = re.exec(html);
  return m ? m[1] : null;
}

function canonicalOf(html) {
  const m = /<link rel="canonical" href="([^"]+)"/.exec(html);
  return m ? m[1] : null;
}

function ldBlocks(html) {
  return [...html.matchAll(/<script type="application\/ld\+json">([\s\S]*?)<\/script>/g)]
    .map(m => JSON.parse(m[1].replace(/\\u003c/g, "<")));
}

const LEAKS = {
  de: ["anúncio", "preço", "usados", "quilometragem", "OLX", "mediana",
       "annonces", "occasion", "kilométrage", "médiane", "prix",
       "annunci", "prezzo", "usate", "chilometraggio"],
  fr: ["anúncio", "preço", "usados", "quilometragem", "OLX",
       "Angebote", "Kilometerstand", "Erstzulassung", "gebraucht",
       "annunci", "prezzo", "chilometraggio"],
  it: ["anúncio", "preço", "usados", "quilometragem", "OLX",
       "Angebote", "Kilometerstand", "Erstzulassung", "gebraucht",
       "annonces", "occasion", "kilométrage", "médiane"],
};

function assertNoLeak(code, html, where) {
  const seen = visibleText(html);
  for (const w of LEAKS[code]) {
    assert(!seen.toLowerCase().includes(w.toLowerCase()),
      `${where} leaks the foreign word "${w}"`);
  }
  assert(!/undefined|\[object Object\]|NaN(?![a-zA-Z])/.test(
    html.replace(/<script[\s\S]*?<\/script>/g, "")), `${where} rendered undefined/NaN`);
  assert(!/\{[a-z_]+\}/.test(visibleText(html)), `${where} rendered an unsubstituted placeholder`);
}

const facetTargets = [];
for (const [slug, rec] of Object.entries(deModels)) {
  for (const kind of ["fuel", "transmission", "district"]) {
    for (const cell of publishedCells(rec, kind)) {
      facetTargets.push({ slug, kind, key: cell.k, cell, rec });
    }
  }
}
const suppressed = [];
for (const [slug, rec] of Object.entries(deModels)) {
  for (const kind of ["fuel", "transmission"]) {
    const all = Array.isArray(rec[kind === "fuel" ? "fx" : "tx"]) ? rec[kind === "fuel" ? "fx" : "tx"] : [];
    if (all.length && !publishedCells(rec, kind).length) suppressed.push({ slug, key: all[0].k });
  }
}
const withPair = facetTargets.find(f => f.cell.vs && Object.keys(f.cell.vs).length);
const withMatched = facetTargets.find(f => Array.isArray(f.cell.vsm));
const fuelTarget = facetTargets.find(f => f.kind === "fuel");
const gearTarget = facetTargets.find(f => f.kind === "transmission");
const districtTarget = facetTargets.find(f => f.kind === "district");
const regions = regionKeys(deDistricts);
const deepModel = Object.entries(deModels)
  .filter(([, r]) => yearPageYears(r).length >= 2)
  .sort((a, b) => (b[1].n || 0) - (a[1].n || 0))[0][0];

assert(facetTargets.length > 20, `fixture has only ${facetTargets.length} published cuts`);
assert(regions.length >= 3, `fixture has only ${regions.length} regions`);
assert(fuelTarget && gearTarget && districtTarget, "fixture is missing one of the three cut kinds");

const P = {
  facet: (code, slug, key) => `${LOCALES[code].prefix}/${LOCALES[code].routes.model}/${slug}/${key}`,
  region: (code, key) => `${LOCALES[code].prefix}/${LOCALES[code].routes.regions}/${key}`,
  regionHub: code => `${LOCALES[code].prefix}/${LOCALES[code].routes.regions}`,
};

await check("a fuel cut renders a German page with canonical, h1 and a real description", async () => {
  const path = P.facet("de", fuelTarget.slug, fuelTarget.key);
  const r = await get(path);
  assert(r.status === 200, `${path} → ${r.status}`);
  const html = await r.text();
  assert(html.includes('<html lang="de-DE">'), `${path} does not declare lang de-DE`);
  assert(canonicalOf(html) === `https://${HOST}${path}`,
    `${path} canonical is ${canonicalOf(html)}`);
  const desc = metaContent(html, "description");
  assert(desc && desc.length > 60, `${path} description is ${desc && desc.length} chars`);
  assert(/<h1 class="fc-h1">[^<]/.test(html), `${path} has no h1 with text`);
  assert(html.includes('<meta name="robots" content="index,follow">'), `${path} is not indexable`);
  assertNoLeak("de", html, path);
});

await check("every published cut of every model answers 200", async () => {
  const bad = [];
  for (const f of facetTargets) {
    const path = P.facet("de", f.slug, f.key);
    const r = await get(path);
    if (r.status !== 200) bad.push(`${path} → ${r.status}`);
  }
  assert(!bad.length, `${bad.length} cut pages did not answer 200: ${bad.slice(0, 4).join("; ")}`);
});

await check("the three cut kinds each say what they are comparing against", async () => {
  const headKey = { fuel: "facet.h2_fuel", transmission: "facet.h2_gear", district: "facet.h2_region" };
  const heads = new Set();
  for (const target of [fuelTarget, gearTarget, districtTarget]) {
    const path = P.facet("de", target.slug, target.key);
    const seen = visibleText(await body(path));
    const want = LOCALES.de.strings[headKey[target.kind]];
    assert(typeof want === "string" && want.length > 5, `no heading string for ${target.kind}`);
    assert(seen.includes(want), `${path} (${target.kind}) does not carry "${want}"`);
    for (const [kind, key] of Object.entries(headKey)) {
      if (kind === target.kind) continue;
      assert(!seen.includes(LOCALES.de.strings[key]),
        `${path} (${target.kind}) also shows the ${kind} heading`);
    }
    heads.add(want);
  }
  assert(heads.size === 3, "the three cut kinds share a heading");
});

await check("a German page really is written in German", async () => {
  const seen = visibleText(await body(P.facet("de", fuelTarget.slug, fuelTarget.key)));
  for (const w of ["Angebote", "Median", "Kilometerstand", "Erstzulassung"]) {
    assert(seen.includes(w), `the German cut page never says "${w}"`);
  }
  const fr = visibleText(await body(P.facet("fr", fuelTarget.slug, fuelTarget.key)));
  for (const w of ["annonces", "médiane", "kilométrage", "immatriculation"]) {
    assert(fr.toLowerCase().includes(w), `the French cut page never says "${w}"`);
  }
  const it = visibleText(await body(P.facet("it", fuelTarget.slug, fuelTarget.key)));
  for (const w of ["annunci", "mediana", "chilometraggio", "immatricolazione"]) {
    assert(it.toLowerCase().includes(w), `the Italian cut page never says "${w}"`);
  }
});

await check("an age-controlled ratio always names the method that produced it", async () => {
  assert(withMatched, "fixture has no year-matched cut to check");
  const path = P.facet("de", withMatched.slug, withMatched.key);
  const seen = visibleText(await body(path));
  assert(seen.includes("Jahr für Jahr verglichen"),
    `${path} states a ratio without naming the year-by-year method`);
  assert(/Stichprobe auf beiden Seiten/.test(seen),
    `${path} does not say the comparison needs sample on both sides`);
});

await check("a sibling duel reports the ratio and disowns the raw medians", async () => {
  assert(withPair, "fixture has no cut with a sibling ratio");
  const path = P.facet("de", withPair.slug, withPair.key);
  const seen = visibleText(await body(path));
  assert(/Altersmischung/.test(seen),
    `${path} compares two cuts without warning that raw medians mix ages`);
});

await check("a cut below its sample floor has no page", async () => {
  assert(suppressed.length, "fixture has no suppressed solo cut to check");
  for (const s of suppressed.slice(0, 3)) {
    const path = P.facet("de", s.slug, s.key);
    const r = await get(path);
    assert(r.status === 404, `suppressed cut ${path} → ${r.status}, expected 404`);
  }
});

await check("an unknown cut key and an unknown model both 404, not an empty shell", async () => {
  const a = await get(P.facet("de", fuelTarget.slug, "kernfusion"));
  assert(a.status === 404, `unknown cut key → ${a.status}`);
  const b = await get(P.facet("de", "marke-die-es-nicht-gibt", fuelTarget.key));
  assert(b.status === 404, `unknown model → ${b.status}`);
  const html = await b.text();
  assert(visibleText(html).includes(LOCALES.de.strings["nf.title"]),
    "the 404 is not the localised one");
});

await check("the cut route does not shadow the model, year or .json pages", async () => {
  const years = yearPageYears(deModels[deepModel]);
  assert(years.length >= 1, "fixture model has no year page");
  const model = await get(`/de/preis/${deepModel}`);
  assert(model.status === 200, `model page → ${model.status}`);
  const year = await get(`/de/preis/${deepModel}/${years[0]}`);
  assert(year.status === 200, `year page → ${year.status}`);
  const modelJson = await get(`/de/preis/${deepModel}.json`);
  assert(modelJson.status === 200, `model .json → ${modelJson.status}`);
  const yearJson = await get(`/de/preis/${deepModel}/${years[0]}.json`);
  assert(yearJson.status === 200, `year .json → ${yearJson.status}`);
  const yd = await yearJson.json();
  assert(yd.model_year === years[0], "the year .json was served by the wrong route");
});

await check("a cut page has a JSON twin at the same path plus .json", async () => {
  const path = P.facet("de", fuelTarget.slug, fuelTarget.key);
  const r = await get(`${path}.json`);
  assert(r.status === 200, `${path}.json → ${r.status}`);
  assert(r.headers.get("content-type").startsWith("application/json"), "the twin is not JSON");
  const doc = await r.json();
  assert(doc.source_url === `https://${HOST}${path}`, `twin source_url is ${doc.source_url}`);
  assert(doc.market === "DE" && doc.language === "de", "twin does not declare the German market");
  assert(doc.sample_size === fuelTarget.cell.n, "twin sample size disagrees with the blob");
  assert(doc.asking_price.median === fuelTarget.cell.fm, "twin median disagrees with the blob");
  assert(doc.facet.key === fuelTarget.key, "twin names the wrong cut");
  assert(!JSON.stringify(doc).includes("OLX"), "twin still names the Portuguese source");
  assert(doc.related.model === `https://${HOST}/de/preis/${fuelTarget.slug}`,
    "twin does not link back to its model");
});

await check("a year-matched twin carries the ratio and the method note", async () => {
  const path = P.facet("de", withMatched.slug, withMatched.key);
  const doc = await (await get(`${path}.json`)).json();
  assert(doc.vs_model_year_matched, "twin dropped the year-matched ratio");
  assert(doc.vs_model_year_matched.ratio === withMatched.cell.vsm[0], "twin ratio is wrong");
  assert(doc.vs_model_year_matched.shared_years === withMatched.cell.vsm[1], "twin year count is wrong");
  assert(typeof doc.vs_model_year_matched.note === "string"
    && doc.vs_model_year_matched.note.length > 40, "twin ratio carries no method note");
});

await check("cut JSON-LD parses and describes this cut", async () => {
  const path = P.facet("de", districtTarget.slug, districtTarget.key);
  const blocks = ldBlocks(await body(path));
  assert(blocks.length >= 1, "no JSON-LD block");
  const nodes = blocks[0]["@graph"];
  assert(Array.isArray(nodes), "JSON-LD is not a @graph");
  const types = nodes.map(n => n["@type"]);
  for (const want of ["BreadcrumbList", "FAQPage", "Dataset", "AggregateOffer"]) {
    assert(types.includes(want), `JSON-LD has no ${want} (${types.join(",")})`);
  }
  const ds = nodes.find(n => n["@type"] === "Dataset");
  assert(ds.url === `https://${HOST}${path}`, "Dataset url is not the canonical");
  assert(ds.inLanguage === "de-DE", "Dataset is not in German");
  const offer = nodes.find(n => n["@type"] === "AggregateOffer");
  assert(offer.offerCount === districtTarget.cell.n, "AggregateOffer count disagrees with the blob");
});

await check("the region hub lists every region that has a page", async () => {
  const path = P.regionHub("de");
  const r = await get(path);
  assert(r.status === 200, `${path} → ${r.status}`);
  const html = await r.text();
  assert(canonicalOf(html) === `https://${HOST}${path}`, `${path} canonical is wrong`);
  const desc = metaContent(html, "description");
  assert(desc && desc.length > 60, `${path} description is ${desc && desc.length} chars`);
  assert(/<h1 class="fc-h1">[^<]/.test(html), `${path} has no h1`);
  for (const key of regions) {
    assert(html.includes(`href="${P.region("de", key)}"`), `the hub does not link ${key}`);
  }
  assertNoLeak("de", html, path);
  const blocks = ldBlocks(html);
  const list = blocks[0]["@graph"].find(n => n["@type"] === "ItemList");
  assert(list && list.numberOfItems === regions.length, "the hub ItemList count is wrong");
});

await check("the region hub has a JSON twin", async () => {
  const r = await get(`${P.regionHub("de")}.json`);
  assert(r.status === 200, `region hub .json → ${r.status}`);
  const doc = await r.json();
  assert(doc.regions.length === regions.length, "hub twin lists a different region count");
  assert(doc.regions.every(x => x.asking_price.median > 0), "hub twin has a region with no median");
  assert(doc.source_url === `https://${HOST}${P.regionHub("de")}`, "hub twin source_url is wrong");
});

await check("every region page renders with its median, range and sample", async () => {
  const bad = [];
  for (const key of regions) {
    const path = P.region("de", key);
    const r = await get(path);
    if (r.status !== 200) { bad.push(`${path} → ${r.status}`); continue; }
    const html = await r.text();
    if (canonicalOf(html) !== `https://${HOST}${path}`) bad.push(`${path} canonical`);
    const desc = metaContent(html, "description");
    if (!desc || desc.length <= 60) bad.push(`${path} description`);
    if (!/<h1 class="fc-h1">[^<]/.test(html)) bad.push(`${path} h1`);
  }
  assert(!bad.length, `${bad.length} region pages are wrong: ${bad.slice(0, 4).join("; ")}`);
});

await check("a region page shows the deepest models and the national comparison", async () => {
  const key = regions[0];
  const rec = deDistricts[key];
  const path = P.region("de", key);
  const html = await body(path);
  const seen = visibleText(html);
  const deep = (rec.top || []).filter(([slug]) => deModels[slug]).slice(0, 3);
  assert(deep.length, "fixture region has no known deep model");
  for (const [slug] of deep) {
    assert(html.includes(`href="/de/preis/${slug}"`), `${path} does not link its deep model ${slug}`);
  }
  assert(seen.includes(LOCALES.de.strings["region.th_median_nat"]),
    `${path} does not put the national median next to the local one`);
  assertNoLeak("de", html, path);
  const nodes = ldBlocks(html)[0]["@graph"];
  const ds = nodes.find(n => n["@type"] === "Dataset");
  assert(ds && ds.spatialCoverage && ds.spatialCoverage.name === rec.lbl,
    "the region Dataset does not name the region");
});

await check("a region page has a JSON twin and an unknown region 404s", async () => {
  const key = regions[0];
  const r = await get(`${P.region("de", key)}.json`);
  assert(r.status === 200, `region .json → ${r.status}`);
  const doc = await r.json();
  assert(doc.region.key === key, "region twin names the wrong region");
  assert(doc.sample_size === deDistricts[key].n, "region twin sample disagrees with the blob");
  assert(doc.asking_price.p25 === deDistricts[key].fl
    && doc.asking_price.p75 === deDistricts[key].fh, "region twin range disagrees with the blob");
  assert(doc.rank_by_median >= 1, "region twin has no rank");
  const miss = await get(P.region("de", "atlantis"));
  assert(miss.status === 404, `unknown region → ${miss.status}`);
  const missJson = await get(`${P.region("de", "atlantis")}.json`);
  assert(missJson.status === 404, `unknown region .json → ${missJson.status}`);
});

await check("the footer of a locale page carries the region hub, the Portuguese one does not", async () => {
  const html = await body("/de");
  assert(html.includes(`href="${P.regionHub("de")}"`), "the German footer has no region hub link");
  const pt = await body("/", envOff);
  assert(!pt.includes("/de/regionen"), "the Portuguese footer leaked a locale link");
  assert(!pt.includes(">Preços por região<"), "the Portuguese footer grew a region link");
});

await check("the sitemap advertises exactly the cut and region pages the router serves", async () => {
  await get(P.regionHub("de"));
  const xml = await body("/de/sitemap.xml");
  const locs = [...xml.matchAll(/<loc>([^<]+)<\/loc>/g)].map(m => new URL(m[1]).pathname);
  const set = new Set(locs);
  assert(set.size === locs.length, "the German sitemap has duplicates");
  for (const f of facetTargets) {
    assert(set.has(P.facet("de", f.slug, f.key)),
      `the sitemap does not list the cut ${f.slug}/${f.key}`);
  }
  assert(set.has(P.regionHub("de")), "the sitemap does not list the region hub");
  for (const key of regions) {
    assert(set.has(P.region("de", key)), `the sitemap does not list the region ${key}`);
  }
  for (const s of suppressed) {
    assert(!set.has(P.facet("de", s.slug, s.key)),
      `the sitemap advertises the suppressed cut ${s.slug}/${s.key}`);
  }
  const mine = locs.filter(p =>
    /^\/de\/regionen(\/|$)/.test(p) || /^\/de\/preis\/[a-z0-9-]+\/[a-z][a-z0-9-]*$/.test(p));
  assert(mine.length === facetTargets.length + regions.length + 1,
    `the sitemap lists ${mine.length} of my URLs, the router serves ${facetTargets.length + regions.length + 1}`);
  const bad = [];
  for (const p of mine) {
    const r = await get(p);
    if (r.status !== 200) bad.push(`${p} → ${r.status}`);
  }
  assert(!bad.length, `the sitemap advertises URLs the router refuses: ${bad.slice(0, 4).join("; ")}`);
});

await check("the sitemap still carries the pages the core module owns", async () => {
  const xml = await body("/de/sitemap.xml");
  const locs = new Set([...xml.matchAll(/<loc>([^<]+)<\/loc>/g)].map(m => new URL(m[1]).pathname));
  for (const p of ["/de", "/de/preise", "/de/bewerten", "/de/methodik", "/de/ueber-uns"]) {
    assert(locs.has(p), `the sitemap lost the core page ${p}`);
  }
  assert(locs.has(`/de/preis/${deepModel}`), "the sitemap lost a model page");
});

await check("French and Italian serve the same layers in their own words", async () => {
  for (const code of ["fr", "it"]) {
    const facet = P.facet(code, fuelTarget.slug, fuelTarget.key);
    const r = await get(facet);
    assert(r.status === 200, `${facet} → ${r.status}`);
    const html = await r.text();
    assert(canonicalOf(html) === `https://${HOST}${facet}`, `${facet} canonical is wrong`);
    assert(html.includes(`<html lang="${LOCALES[code].lang}">`), `${facet} declares the wrong lang`);
    const desc = metaContent(html, "description");
    assert(desc && desc.length > 60, `${facet} description is ${desc && desc.length} chars`);
    assertNoLeak(code, html, facet);
    ldBlocks(html);

    const hub = P.regionHub(code);
    const hr = await get(hub);
    assert(hr.status === 200, `${hub} → ${hr.status}`);
    assertNoLeak(code, await hr.text(), hub);

    const region = P.region(code, regions[0]);
    const rr = await get(region);
    assert(rr.status === 200, `${region} → ${rr.status}`);
    const rhtml = await rr.text();
    assert(canonicalOf(rhtml) === `https://${HOST}${region}`, `${region} canonical is wrong`);
    assertNoLeak(code, rhtml, region);

    const twin = await get(`${region}.json`);
    assert(twin.status === 200, `${region}.json → ${twin.status}`);
    assert((await twin.json()).language === code, `${region}.json declares the wrong language`);
  }
});

await check("the segments are the localised ones and nothing answers on the wrong prefix", async () => {
  assert(LOCALES.de.routes.regions === "regionen", "the German region segment is wrong");
  assert(LOCALES.fr.routes.regions === "regions", "the French region segment is wrong");
  assert(LOCALES.it.routes.regions === "regioni", "the Italian region segment is wrong");
  assert(LOCALES.pt.routes.regions === undefined, "a Portuguese region route was registered");
  const wrong = await get(`/de/regions/${regions[0]}`);
  assert(wrong.status === 404, `/de/regions/... → ${wrong.status}, expected 404`);
  const alsoWrong = await get(`/fr/regionen/${regions[0]}`);
  assert(alsoWrong.status === 404, `/fr/regionen/... → ${alsoWrong.status}, expected 404`);
});

await check("with INTL_LOCALES unset every page of this layer is gone", async () => {
  for (const p of [P.facet("de", fuelTarget.slug, fuelTarget.key), P.regionHub("de"),
                   P.region("de", regions[0]), `${P.regionHub("de")}.json`]) {
    const r = await get(p, envOff);
    assert(r.status === 404, `${p} with no locales → ${r.status}`);
  }
});

await check("the Portuguese root is untouched by this layer", async () => {
  const r = await get("/precos", envOff);
  assert(r.status === 200, `/precos → ${r.status}`);
  const html = await r.text();
  assert(!html.includes("regionen") && !html.includes("regioni"),
    "a locale segment leaked into the Portuguese hub");
  const hubKeys = Object.keys(LOCALES.pt.routes);
  assert(!hubKeys.includes("regions") && !hubKeys.includes("regionsJson"),
    "the Portuguese route table grew a region route");
});

await check("the model page links block is offered to the integrator and points at real pages", async () => {
  const { intlModelCutLinks } = await import("../../flipper-club/src/intl-facets.js");
  const rec = deModels[fuelTarget.slug];
  const html = intlModelCutLinks(LOCALES.de, rec, fuelTarget.slug);
  assert(html && html.includes("<h2"), "the links block rendered nothing");
  const hrefs = [...html.matchAll(/href="([^"]+)"/g)].map(m => m[1]);
  assert(hrefs.length >= 2, "the links block has no links");
  for (const key of intlFacetKeys(rec)) {
    assert(hrefs.includes(P.facet("de", fuelTarget.slug, key)),
      `the links block omits the cut ${key}`);
  }
  const bad = [];
  for (const h of hrefs) {
    const r = await get(h);
    if (r.status !== 200) bad.push(`${h} → ${r.status}`);
  }
  assert(!bad.length, `the links block points at ${bad.join("; ")}`);
  const empty = intlModelCutLinks(LOCALES.de, { b: "X", m: "Y", n: 1 }, "x-y");
  assert(empty === "", "the links block rendered a shell for a model with no cuts");
});

console.log(failures ? `\n${failures} check(s) FAILED` : "\nall intl facet checks passed");
process.exit(failures ? 1 : 0);
