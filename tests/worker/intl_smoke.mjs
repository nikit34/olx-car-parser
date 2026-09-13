import { readFileSync, readdirSync } from "node:fs";
import worker from "../../flipper-club/src/index.js";
import { yearPageYears } from "../../flipper-club/src/seo-pages.js";
import { LOCALES, KEYS, missingKeys } from "../../flipper-club/src/i18n.js";

const HOST = "carsbuyer.org";
const NNBSP = " ";
const UUID = "bb920d7a-1111-4222-8333-444455556666";
const MISS_UUID = "ffffffff-1111-4222-8333-444455556666";

let failures = 0;
function assert(cond, msg) { if (!cond) throw new Error(msg); }
async function check(name, fn) {
  try { await fn(); console.log(`  ok   ${name}`); }
  catch (err) { failures++; console.error(`  FAIL ${name}\n       ${err && err.message}`); }
}

const ptPath = process.argv[2] || "tests/worker/fixtures/models.json";
const dePath = process.argv[3] || "tests/worker/fixtures/models_de.json";
const mdoc = JSON.parse(readFileSync(ptPath, "utf8"));
const ddoc = JSON.parse(readFileSync(dePath, "utf8"));
const deModels = ddoc.models;

const DEALS = [
  {
    olx_id: "as24_de:aaa", url: "https://www.autoscout24.de/angebote/aaa",
    image_url: "https://prod.pictures.autoscout24.net/aaa/720x540.webp",
    brand: "Volkswagen", model: "Golf", year: 2016, mileage_km: 128000,
    fuel_type: "Diesel", seller_type: "Profissional", district: "Bayern",
    price_eur: 11900, fair_median: 14200, fair_low: 12800, fair_high: 15600,
    discount_pct: 0.16, decision_score: 0.81, verdict: "BUY", days_on_market: 6,
  },
  {
    olx_id: "as24_de:bbb", url: "https://www.autoscout24.de/angebote/bbb",
    image_url: "https://prod.pictures.autoscout24.net/bbb/720x540.webp",
    brand: "BMW", model: "320", year: 2014, mileage_km: 191000,
    fuel_type: "Gasolina", seller_type: "Particular", district: "Hessen",
    price_eur: 8400, fair_median: 9900, fair_low: 8900, fair_high: 11200,
    discount_pct: 0.15, decision_score: 0.64, verdict: "WATCH", days_on_market: 1,
  },
];

const VALUATION = {
  t: "Volkswagen Golf 1.6 TDI Comfortline", y: 2016, km: 128000, fu: "Diesel",
  p: 11900, fl: 12800, fm: 14200, fh: 15600, ct: "München", sd: 31, ms: "volkswagen-golf",
};
const CARS = { [`as24_de:${UUID}`]: VALUATION, "as24_de:abc-uuid": VALUATION };

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
  if (u.includes("models_de.json")) return json(ddoc);
  if (u.includes("models_fr.json") || u.includes("models_it.json")) {
    return new Response("not found", { status: 404 });
  }
  if (u.includes("models.json")) return json(mdoc);
  if (u.includes("hot_deals_de_all.json")) return json({ deals: DEALS, built_at: ddoc.built_at });
  if (u.includes("hot_deals_")) return json({ deals: [] });
  if (u.includes("valuations_de.json")) return json({ v: 1, cars: CARS });
  if (u.includes("valuations")) return json({ cars: {} });
  if (u.includes("import.json")) return json({ built_at: ddoc.built_at, models: {} });
  return realFetch(input, init);
};

const get = (path, e = env, method = "GET") =>
  worker.fetch(new Request(`https://${HOST}${path}`, { method }), e);
const body = async (path, e = env) => (await get(path, e)).text();

const deep = Object.entries(deModels)
  .filter(([, r]) => yearPageYears(r).length >= 3)
  .sort((a, b) => (b[1].n || 0) - (a[1].n || 0))[0][0];
const deepYears = yearPageYears(deModels[deep]);

function visibleText(html) {
  return html
    .replace(/<script[\s\S]*?<\/script>/g, " ")
    .replace(/<style[\s\S]*?<\/style>/g, " ")
    .replace(/<[^>]+>/g, " ")
    .replace(/&nbsp;/g, " ");
}

const PT_GIVEAWAYS = [
  "anúncio", "anúncios", "preço", "preços", "usados", "grátis", "vendedor",
  "Quem somos", "Avaliação", "carros", "OLX",
];

await check("/de is a German page, not a translated Portuguese one", async () => {
  const r = await get("/de");
  assert(r.status === 200, `/de → ${r.status}`);
  const html = await r.text();
  assert(html.includes('<html lang="de-DE">'), "/de does not declare lang de-DE");
  assert(html.includes('<meta property="og:locale" content="de_DE">'), "/de has the wrong og:locale");
  assert(!/undefined|\[object Object\]|NaN(?![a-zA-Z])/.test(html.replace(/<script[\s\S]*?<\/script>/g, "")),
    "/de rendered undefined/NaN into the page");
  const seen = visibleText(html);
  for (const w of PT_GIVEAWAYS) {
    assert(!new RegExp(w, "i").test(seen), `/de leaks the Portuguese word "${w}"`);
  }
  assert(seen.includes("AutoScout24"), "/de does not name its own data source");
  assert(html.includes('href="/de/preise"') && html.includes('href="/de/bewerten"'),
    "/de does not link its own localised routes");
});

await check("the German hub lists models under the localised prefix", async () => {
  const r = await get("/de/preise");
  assert(r.status === 200, `/de/preise → ${r.status}`);
  const html = await r.text();
  assert(html.includes("index,follow"), "/de/preise is not indexable");
  assert(html.includes(`<link rel="canonical" href="https://${HOST}/de/preise">`),
    "/de/preise has the wrong canonical");
  assert(html.includes(`href="/de/preis/${deep}"`), "the hub does not link the deepest model");
  assert(!/preços|anúncios/i.test(visibleText(html)), "the German hub leaks Portuguese");
});

await check("a German model page carries canonical, FAQ JSON-LD and German money", async () => {
  const r = await get(`/de/preis/${deep}`);
  assert(r.status === 200, `/de/preis/${deep} → ${r.status}`);
  const html = await r.text();
  assert(html.includes("index,follow"), "model page is not indexable");
  assert(html.includes(`<link rel="canonical" href="https://${HOST}/de/preis/${deep}">`),
    "model page canonical is not the locale URL");
  assert(html.includes(`href="https://${HOST}/de/preis/${deep}.json"`),
    "model page does not advertise its JSON twin");
  const blocks = [...html.matchAll(/<script type="application\/ld\+json">([\s\S]*?)<\/script>/g)];
  assert(blocks.length === 1, `expected one JSON-LD block, got ${blocks.length}`);
  const ld = JSON.parse(blocks[0][1].replace(/\\u003c/g, "<"));
  const types = ld["@graph"].map(n => n["@type"]);
  for (const want of ["BreadcrumbList", "FAQPage", "Dataset"]) {
    assert(types.includes(want), `model page JSON-LD has no ${want}`);
  }
  const faq = ld["@graph"].find(n => n["@type"] === "FAQPage");
  assert(faq.mainEntity.length >= 3, "FAQPage carries fewer than three questions");
  for (const q of faq.mainEntity) {
    assert(q.name && q.acceptedAnswer.text, "an FAQ entry is empty");
    assert(!/<[a-z]/i.test(q.acceptedAnswer.text), "FAQ answers still carry markup");
  }
  assert(new RegExp(`\\d${NNBSP}€`).test(html), "no German money formatting on the page");
  assert(!/€\d/.test(html), "the page still prints Portuguese-style €13.990");
  assert(html.includes("Erstzulassung"), "the model page does not use the German term for first registration");
  for (const y of deepYears.slice(0, 3)) {
    assert(html.includes(`href="/de/preis/${deep}/${y}"`), `model page does not link its ${y} page`);
  }
});

await check("the JSON twins answer under the locale prefix", async () => {
  const m = await get(`/de/preis/${deep}.json`);
  assert(m.status === 200, `model .json → ${m.status}`);
  assert(m.headers.get("content-type").startsWith("application/json"), "model .json is not JSON");
  const doc = await m.json();
  assert(doc.market === "DE" && doc.language === "de", "model .json does not declare the German market");
  assert(doc.source_url === `https://${HOST}/de/preis/${deep}`, "model .json points elsewhere");
  assert(doc.asking_price.median > 0, "model .json has no median");
  assert(!JSON.stringify(doc).includes("OLX"), "model .json still names the Portuguese source");
  const withPage = doc.by_year.filter(y => y.page);
  assert(withPage.length === deepYears.length,
    `model .json advertises ${withPage.length} year pages, router serves ${deepYears.length}`);
  assert(withPage.every(y => y.page.startsWith(`https://${HOST}/de/preis/${deep}/`)),
    "model .json year pages are not locale URLs");

  const y = await get(`/de/preis/${deep}/${deepYears[0]}.json`);
  assert(y.status === 200, `year .json → ${y.status}`);
  const ydoc = await y.json();
  assert(ydoc.model_year === deepYears[0] && ydoc.market === "DE", "year .json is not this year in DE");
  assert(ydoc.related.model === `https://${HOST}/de/preis/${deep}`, "year .json does not link its model");
});

await check("a German year page renders and a thin year 301s to its model", async () => {
  const r = await get(`/de/preis/${deep}/${deepYears[0]}`);
  assert(r.status === 200, `year page → ${r.status}`);
  const html = await r.text();
  assert(html.includes("index,follow"), "year page is not indexable");
  assert(html.includes(`<link rel="canonical" href="https://${HOST}/de/preis/${deep}/${deepYears[0]}">`),
    "year page canonical is wrong");
  assert(!/anúncios|preços/i.test(visibleText(html)), "year page leaks Portuguese");

  const published = new Set(deepYears);
  const thin = (deModels[deep].yr || []).find(c => typeof c.y === "number" && !published.has(c.y));
  assert(thin, "the fixture has no thin year to test the redirect with");
  const red = await get(`/de/preis/${deep}/${thin.y}`);
  assert(red.status === 301, `thin year → ${red.status}, expected 301`);
  assert(new URL(red.headers.get("location")).pathname === `/de/preis/${deep}`,
    `thin year redirects to ${red.headers.get("location")}`);
  const redJson = await get(`/de/preis/${deep}/${thin.y}.json`);
  assert(redJson.status === 301
    && new URL(redJson.headers.get("location")).pathname === `/de/preis/${deep}.json`,
    "the JSON twin of a thin year does not point at the model's JSON twin");
  assert((await get(`/de/preis/${deep}/1900`)).status === 404, "a year we never had data for is not a 404");
  assert((await get("/de/preis/gibt-es-nicht")).status === 404, "an unknown model is not a 404");
});

await check("/de/bewerten works both as a form and as a link lookup", async () => {
  const bare = await get("/de/bewerten");
  assert(bare.status === 200, `/de/bewerten → ${bare.status}`);
  const bareHtml = await bare.text();
  assert(bareHtml.includes('name="q"') && bareHtml.includes('name="modelo"')
    && bareHtml.includes('name="ano"'),
    "the valuation page is missing one of its two entry paths");
  assert(bareHtml.includes(`<option value="${deep}"`), "the model picker does not offer the deepest model");
  assert(!/anúncio|preço/i.test(visibleText(bareHtml)), "the valuation page leaks Portuguese");

  const url = `https://www.autoscout24.de/angebote/volkswagen-golf-1-6-tdi-${UUID}`;
  const hit = await get(`/de/bewerten?q=${encodeURIComponent(url)}`);
  assert(hit.status === 200, `?q=<autoscout url> → ${hit.status}`);
  const hitHtml = await hit.text();
  assert(hitHtml.includes("Volkswagen Golf 1.6 TDI Comfortline"), "the pasted listing was not resolved");
  assert(hitHtml.includes(`href="${url}"`) && hitHtml.includes('rel="noopener nofollow"'),
    "the verdict does not link back to the listing safely");
  assert(hitHtml.includes('href="/de/preis/volkswagen-golf"'), "the verdict does not link the model page");
  assert(hitHtml.includes("11.900"), "the asking price is not rendered in German format");

  const bareId = await get("/de/bewerten?q=abc-uuid");
  assert((await bareId.text()).includes("Volkswagen Golf 1.6 TDI Comfortline"),
    "a bare listing id does not resolve against valuations_de.json");

  const missUrl = `https://www.autoscout24.de/angebote/x-${MISS_UUID}`;
  const miss = await get(`/de/bewerten?q=${encodeURIComponent(missUrl)}`);
  assert(miss.status === 200, "a miss is not a 200");
  assert((await miss.text()).includes("noch nicht"), "a miss does not say the listing is unknown");

  const spec = await get(`/de/bewerten?modelo=${deep}&ano=${deepYears[0]}`);
  assert(spec.status === 200, `spec lookup → ${spec.status}`);
  assert((await spec.text()).includes(String(deepYears[0])), "the spec lookup does not echo the year");
});

await check("/de/markt lists the German deals with safe outbound links", async () => {
  const r = await get("/de/markt");
  assert(r.status === 200, `/de/markt → ${r.status}`);
  const html = await r.text();
  for (const d of DEALS) {
    assert(html.includes(`href="${d.url}"`), `the feed does not link ${d.olx_id}`);
    assert(html.includes(d.image_url), `the feed does not show the photo of ${d.olx_id}`);
  }
  const links = [...html.matchAll(/<a [^>]*href="https:\/\/www\.autoscout24\.de[^"]*"[^>]*>/g)].map(m => m[0]);
  assert(links.length >= DEALS.length, `expected ${DEALS.length} source links, got ${links.length}`);
  for (const a of links) {
    assert(a.includes('target="_blank"') && a.includes("noopener"),
      `a source link opens unsafely: ${a}`);
  }
  assert(html.includes("Händler") && html.includes("Privat"), "seller types are not shown in German");
  assert(html.includes("Bayern") && html.includes("Hessen"), "regions are not shown");
  assert(html.includes("14.200"), "the fair median is not shown next to the asking price");
  assert(!/anúncio|poupas/i.test(visibleText(html)), "the feed leaks Portuguese");
});

await check("the German trust pages are indexable", async () => {
  const pages = [
    ["/de/methodik", "AutoScout24"],
    ["/de/ueber-uns", "Carsbuyer"],
    ["/de/datenschutz", "Google Analytics"],
  ];
  for (const [path, needle] of pages) {
    const r = await get(path);
    assert(r.status === 200, `${path} → ${r.status}`);
    const html = await r.text();
    assert(html.includes("index,follow"), `${path} is not indexable`);
    assert(html.includes(`<link rel="canonical" href="https://${HOST}${path}">`), `${path} canonical is wrong`);
    assert(html.includes(needle), `${path} lost its content`);
    assert(!/anúncios|preços|Quem somos/i.test(visibleText(html)), `${path} leaks Portuguese`);
  }
  const method = await body("/de/methodik");
  assert(/Händler und Privatverkäufer/.test(method),
    "the methodology does not say both dealers and private sellers are in the sample");
  assert(/verlangte|Verlangter|verlangten/.test(method),
    "the methodology does not say the prices are asking prices");
});

await check("/de/sitemap.xml advertises exactly what the router serves", async () => {
  const xml = await body("/de/sitemap.xml");
  const locs = [...xml.matchAll(/<loc>([^<]+)<\/loc>/g)].map(m => new URL(m[1]).pathname);
  assert(locs.length > 0, "the German sitemap is empty");
  assert(new Set(locs).size === locs.length, "the German sitemap has duplicates");
  assert(locs.every(p => p === "/de" || p.startsWith("/de/")), "the German sitemap lists foreign URLs");
  for (const p of ["/de", "/de/preise", "/de/bewerten", "/de/markt", "/de/methodik",
                   "/de/ueber-uns", "/de/datenschutz"]) {
    assert(locs.includes(p), `the German sitemap does not list ${p}`);
  }
  const expectedYears = Object.values(deModels).reduce((n, r) => n + yearPageYears(r).length, 0);
  const gotYears = locs.filter(p => /^\/de\/preis\/[^/]+\/\d{4}$/.test(p)).length;
  assert(gotYears === expectedYears,
    `sitemap has ${gotYears} year URLs, the generator publishes ${expectedYears}`);
  assert(locs.filter(p => /^\/de\/preis\/[^/]+$/.test(p)).length === Object.keys(deModels).length,
    "the sitemap model count disagrees with the blob");
  for (const p of locs) {
    const r = await get(p);
    assert(r.status === 200, `the sitemap advertises ${p} but the router answers ${r.status}`);
  }
  console.log(`       (resolved all ${locs.length} German sitemap URLs)`);
});

await check("a live locale with no blob degrades instead of lying", async () => {
  const hub = await get("/fr/prix");
  assert(hub.status === 503, `/fr/prix → ${hub.status}, expected 503`);
  const frHtml = await hub.text();
  assert(frHtml.includes('<html lang="fr-FR">'), "the French 503 is not in French");
  assert(frHtml.includes("préparation"), "the French 503 does not say the data is being prepared");
  assert((await get("/fr/cote/x")).status === 404, "a model page with no blob is not a 404");
  const itHub = await get("/it/prezzi");
  assert(itHub.status === 503, `/it/prezzi → ${itHub.status}, expected 503`);
  assert((await itHub.text()).includes('<html lang="it-IT">'), "the Italian 503 is not in Italian");
  const it404 = await get("/it/prezzo/x");
  assert(it404.status === 404, "an Italian model page with no blob is not a 404");
  assert((await it404.text()).includes('<html lang="it-IT">'), "the Italian 404 is not in Italian");
});

await check("only the configured prefixes exist", async () => {
  for (const p of ["/es", "/es/precios", "/nl", "/de/preise/extra", "/de/gibt-es-nicht"]) {
    const r = await get(p);
    assert(r.status === 404, `${p} → ${r.status}, expected 404`);
  }
  const cased = await get("/DE/Preise");
  assert(cased.status === 301, `/DE/Preise → ${cased.status}`);
  assert(new URL(cased.headers.get("location")).pathname === "/de/preise",
    `/DE/Preise redirects to ${cased.headers.get("location")}`);
  const slash = await get("/de/preise/");
  assert(slash.status === 301 && new URL(slash.headers.get("location")).pathname === "/de/preise",
    "a trailing slash under a locale does not normalise");
  const post = await worker.fetch(new Request(`https://${HOST}/de/preise`, { method: "POST" }), env);
  assert(post.status === 404, `POST /de/preise → ${post.status}, expected 404`);
  const head = await get("/de/preise", env, "HEAD");
  assert(head.status === 200 && !(await head.text()), "HEAD under a locale does not mirror GET");
});

await check("robots and llms.txt publish the live locales", async () => {
  const robots = await body("/robots.txt");
  const sitemaps = robots.split("\n").filter(l => l.startsWith("Sitemap:"));
  assert(sitemaps.includes(`Sitemap: https://${HOST}/sitemap.xml`), "the Portuguese sitemap line is gone");
  for (const cc of ["de", "fr", "it"]) {
    assert(sitemaps.includes(`Sitemap: https://${HOST}/${cc}/sitemap.xml`),
      `robots.txt does not point at the ${cc} sitemap`);
  }
  assert(sitemaps.length === 4, `robots has ${sitemaps.length} Sitemap lines, expected 4`);

  const llms = await body("/llms.txt");
  for (const cc of ["de", "fr", "it"]) {
    assert(llms.includes(`https://${HOST}/${cc}/${LOCALES[cc].routes.hub}`),
      `llms.txt does not list the ${cc} hub`);
  }
  assert(llms.includes("Deutschland") && llms.includes("France") && llms.includes("Italia"),
    "llms.txt does not name the live markets");
});

await check("the Portuguese root is untouched by the expansion", async () => {
  const r = await get("/");
  assert(r.status === 200, `/ → ${r.status}`);
  const html = await r.text();
  assert(html.includes('<html lang="pt-PT">'), "the Portuguese root changed language");
  assert(html.includes("Português") && html.includes("Deutsch"),
    "the language switcher is missing while locales are live");
  const hub = await body("/precos");
  assert(hub.includes('<html lang="pt-PT">'), "/precos changed language");
  assert(hub.includes('<meta property="og:locale" content="pt_PT">'), "/precos changed og:locale");
  assert(hub.includes('href="/preco/'), "/precos lost its Portuguese model links");
});

await check("with INTL_LOCALES unset the locale prefixes do not exist", async () => {
  kv.clear();
  for (const p of ["/de", "/de/preise", "/fr/prix", "/it/prezzi", "/de/sitemap.xml"]) {
    const r = await get(p, envOff);
    assert(r.status === 404, `${p} with no locales → ${r.status}, expected 404`);
    assert((await r.text()).includes('<html lang="pt-PT">'),
      `${p} does not fall back to the Portuguese 404`);
  }
  const robots = await body("/robots.txt", envOff);
  const sitemaps = robots.split("\n").filter(l => l.startsWith("Sitemap:"));
  assert(sitemaps.length === 1 && sitemaps[0] === `Sitemap: https://${HOST}/sitemap.xml`,
    `robots has ${sitemaps.length} Sitemap lines with no locales live`);
  const llms = await body("/llms.txt", envOff);
  assert(!llms.includes("Deutschland"), "llms.txt advertises a market that is not live");
  const home = await body("/", envOff);
  assert(!home.includes("Deutsch"), "the language switcher shows with no locales live");
  kv.clear();
});

const ptDeep = Object.entries(mdoc.models).sort((a, b) => (b[1].n || 0) - (a[1].n || 0))[0][0];
const alternates = html =>
  [...html.matchAll(/<link rel="alternate" hreflang="([^"]+)" href="([^"]+)">/g)]
    .map(m => ({ lang: m[1], href: m[2] }));
const LANGS = ["pt-PT", "de-DE", "fr-FR", "it-IT"];

await check("a locale page lists every live language, itself included", async () => {
  const html = await body(`/de/preis/${deep}`);
  const alts = alternates(html);
  const langs = alts.filter(a => a.lang !== "x-default").map(a => a.lang);
  assert(langs.length === LANGS.length,
    `the German model page lists ${langs.length} alternates, expected ${LANGS.length}`);
  assert(new Set(langs).size === langs.length, "the alternate set repeats a language");
  for (const want of LANGS) assert(langs.includes(want), `no hreflang for ${want}`);
  const by = Object.fromEntries(alts.map(a => [a.lang, a.href]));
  assert(by["de-DE"] === `https://${HOST}/de/preis/${deep}`,
    "the page does not point hreflang at itself");
  assert(by["pt-PT"] === `https://${HOST}/preco/${deep}`, `pt alternate is ${by["pt-PT"]}`);
  assert(by["fr-FR"] === `https://${HOST}/fr/cote/${deep}`, `fr alternate is ${by["fr-FR"]}`);
  assert(by["it-IT"] === `https://${HOST}/it/prezzo/${deep}`, `it alternate is ${by["it-IT"]}`);
  assert(by["x-default"] === by["pt-PT"], "x-default is not the Portuguese page");

  const year = alternates(await body(`/de/preis/${deep}/${deepYears[0]}`));
  const yby = Object.fromEntries(year.map(a => [a.lang, a.href]));
  assert(yby["pt-PT"] === `https://${HOST}/preco/${deep}/${deepYears[0]}`,
    "the year page alternate is not the same year in Portuguese");
  assert(yby["it-IT"] === `https://${HOST}/it/prezzo/${deep}/${deepYears[0]}`,
    "the year page alternate is not the same year in Italian");
});

await check("the Portuguese root carries the same reciprocal set", async () => {
  for (const [ptPath, dePath] of [["/precos", "/de/preise"], [`/preco/${ptDeep}`, `/de/preis/${ptDeep}`],
                                  ["/metodologia", "/de/methodik"], ["/sobre", "/de/ueber-uns"],
                                  ["/privacidade", "/de/datenschutz"], ["/avaliar", "/de/bewerten"]]) {
    const alts = alternates(await body(ptPath));
    const langs = alts.filter(a => a.lang !== "x-default").map(a => a.lang);
    for (const want of LANGS) assert(langs.includes(want), `${ptPath} has no hreflang for ${want}`);
    const by = Object.fromEntries(alts.map(a => [a.lang, a.href]));
    assert(by["pt-PT"] === `https://${HOST}${ptPath}`, `${ptPath} does not list itself`);
    assert(by["de-DE"] === `https://${HOST}${dePath}`,
      `${ptPath} points German at ${by["de-DE"]}, expected ${dePath}`);
    assert(by["x-default"] === `https://${HOST}${ptPath}`, `${ptPath} x-default is ${by["x-default"]}`);
    const mirror = Object.fromEntries(alternates(await body(dePath)).map(a => [a.lang, a.href]));
    for (const lang of [...LANGS, "x-default"]) {
      assert(mirror[lang] === by[lang],
        `${dePath} and ${ptPath} disagree on ${lang}: ${mirror[lang]} vs ${by[lang]}`);
    }
  }
});

await check("x-default points at the Portuguese root on the home family", async () => {
  const by = Object.fromEntries(alternates(await body("/de")).map(a => [a.lang, a.href]));
  assert(by["x-default"] === `https://${HOST}/`, `x-default is ${by["x-default"]}, expected the root`);
  assert(by["pt-PT"] === `https://${HOST}/`, "the Portuguese alternate of /de is not the root");
  assert(by["de-DE"] === `https://${HOST}/de`, "/de does not list itself");
  assert(by["fr-FR"] === `https://${HOST}/fr` && by["it-IT"] === `https://${HOST}/it`,
    "the home family is missing a live locale");
});

await check("a page with no counterpart claims no alternate at all", async () => {
  const district = Object.keys(mdoc.districts || {})[0];
  assert(district, "the fixture has no district to test with");
  const html = await body(`/precos/${district}`);
  assert(html.includes(`<link rel="canonical" href="https://${HOST}/precos/${district}">`),
    "the district page lost its canonical, so the test proves nothing");
  assert(alternates(html).length === 0,
    "a Portugal-only page advertises a translation that does not exist");
});

await check("with INTL_LOCALES unset no hreflang is emitted anywhere", async () => {
  kv.clear();
  for (const p of ["/", "/precos", `/preco/${ptDeep}`, "/metodologia", "/sobre", "/privacidade"]) {
    const html = await body(p, envOff);
    assert(alternates(html).length === 0, `${p} emits hreflang with no locales live`);
    assert(!html.includes('rel="alternate" hreflang='), `${p} emits a stray hreflang link`);
  }
  kv.clear();
});

await check("every locale defines every key", async () => {
  const gaps = missingKeys();
  assert(Object.keys(gaps).length === 0, `key sets disagree: ${JSON.stringify(gaps)}`);
  assert(KEYS.length > 200, `only ${KEYS.length} keys — did a dictionary shrink?`);
});

await check("every key the pages ask for exists in all four locales", async () => {
  const dir = "flipper-club/src";
  const sources = readdirSync(dir).filter(f => f.endsWith(".js")).sort().map(f => `${dir}/${f}`);
  assert(sources.length >= 6, `only ${sources.length} source files — did the scan break?`);
  const patterns = [/(?<![\w$.])t\(\s*\w+\s*,\s*"([^"]+)"/g, /(?<![\w$.])it\(\s*\w+\s*,\s*"([^"]+)"/g];
  const used = new Set();
  for (const file of sources) {
    const src = readFileSync(file, "utf8");
    for (const re of patterns) for (const m of src.matchAll(re)) used.add(m[1]);
  }
  assert(used.size > 100, `only found ${used.size} t() calls — did the scan break?`);
  for (const key of used) {
    for (const code of Object.keys(LOCALES)) {
      assert(typeof LOCALES[code].strings[key] === "string",
        `locale ${code} is missing the key "${key}" that the pages render`);
    }
  }
});

await check("no locale string is broken or carries a stray placeholder", async () => {
  const refPh = k => (LOCALES.de.strings[k].match(/\{[a-z_]+\}/g) || []).sort().join(",");
  for (const code of Object.keys(LOCALES)) {
    for (const key of KEYS) {
      const s = LOCALES[code].strings[key];
      assert(!s.includes("undefined"), `${code}/${key} contains "undefined"`);
      assert(s.trim().length > 0, `${code}/${key} is empty`);
      const ph = (s.match(/\{[a-z_]+\}/g) || []).sort().join(",");
      assert(ph === refPh(key), `${code}/${key} has placeholders ${ph}, de has ${refPh(key)}`);
    }
  }
});

console.log(failures ? `\n${failures} check(s) FAILED` : "\nall intl checks passed");
process.exit(failures ? 1 : 0);
