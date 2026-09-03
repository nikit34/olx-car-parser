// Exercise the Worker's routing through worker.fetch() with a stubbed env.
//
// The bug this file exists for: unknown paths fell through to the Basic-Auth
// asset gate and answered 401. Nothing in the template tests could see that —
// it was a routing decision, and only calling fetch() reaches it.
//
// It also proves the thing that makes a 1 000-page expansion safe: every URL in
// sitemap.xml resolves to a 200, and the pages the router refuses are exactly
// the ones the sitemap never advertised.
//
// Run:  node tests/worker/route_smoke.mjs [path/to/models.json]

import { existsSync, readFileSync } from "node:fs";
import worker from "../../flipper-club/src/index.js";
import { yearPageYears, liquidityOk, depreciationSlugs, comparePairs, isoWeek, isoWeekStart, missingWeeks, DUELS, isoWeekMonth, monthlyCuts, importSlugs, venderOk } from "../../flipper-club/src/seo-pages.js";
import { GUIDES } from "../../flipper-club/src/guides.js";

const HOST = "carsbuyer.org";
const RELEASE = "https://github.com/nikit34/olx-car-parser/releases/download/latest-data/models.json";

let failures = 0;
function assert(cond, msg) { if (!cond) throw new Error(msg); }
async function check(name, fn) {
  try { await fn(); console.log(`  ok   ${name}`); }
  catch (err) { failures++; console.error(`  FAIL ${name}\n       ${err && err.message}`); }
}

const arg = process.argv[2];
const mdoc = arg ? JSON.parse(readFileSync(arg, "utf8")) : await (await fetch(RELEASE)).json();
const models = mdoc.models;
const idoc = JSON.parse(readFileSync(new URL("./fixtures/import.json", import.meta.url), "utf8"));

// ── stub env ────────────────────────────────────────────────────────────────
const kv = new Map();
const env = {
  CANONICAL_HOST: HOST,
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
  // The asset bucket is configured single-page-application, so a miss returns
  // index.html with a 200 — modelled here, because that is precisely why the
  // router cannot "probe ASSETS and 404 on a miss".
  ASSETS: {
    async fetch(req) {
      const path = new URL(req.url).pathname;
      if (path.startsWith("/fonts/")) return new Response("woff2-bytes", { status: 200, headers: { "content-type": "font/woff2" } });
      if (path.startsWith("/files/") || path.startsWith("/data/")) return new Response("asset", { status: 200 });
      const onDisk = new URL(`../../dashboard-static${path}`, import.meta.url);
      if (existsSync(onDisk)) {
        return new Response(readFileSync(onDisk),
          { status: 200, headers: { "content-type": path.endsWith(".ico") ? "image/vnd.microsoft.icon" : "image/png" } });
      }
      return new Response("<html>spa fallback</html>", { status: 200, headers: { "content-type": "text/html" } });
    },
  },
};

// getModels/getDeals go through global fetch; serve the blob from memory and
// give the deals feed an empty-but-valid answer so the bridges are exercised.
const realFetch = globalThis.fetch;
globalThis.fetch = async (input, init) => {
  const u = typeof input === "string" ? input : input.url;
  if (u.includes("import.json")) return new Response(JSON.stringify(idoc), { status: 200 });
  if (u.includes("models.json")) return new Response(JSON.stringify(mdoc), { status: 200 });
  if (u.includes("hot_deals_")) return new Response(JSON.stringify({ deals: [] }), { status: 200 });
  if (u.includes("valuations.json")) return new Response(JSON.stringify({}), { status: 200 });
  return realFetch(input, init);
};

const get = (path, method = "GET") =>
  worker.fetch(new Request(`https://${HOST}${path}`, { method }), env);

// ── the 401 regression ──────────────────────────────────────────────────────
await check("unknown paths are 404, not 401", async () => {
  for (const path of ["/sobre-nos", "/faq", "/blog", "/pagina-que-nao-existe", "/en", "/wp-admin", "/x/y/z"]) {
    const r = await get(path);
    assert(r.status === 404, `${path} → ${r.status} (was the Basic-Auth gate reached?)`);
    const body = await r.text();
    assert(body.includes("<!doctype html>"), `${path}: 404 is not the real page`);
    assert(body.includes("noindex,follow"), `${path}: 404 is not noindex`);
  }
});

await check("trailing slash and case 301 to the canonical spelling", async () => {
  const cases = [
    ["/precos/", "/precos"],
    ["/PRECO/VOLKSWAGEN-GOLF", "/preco/volkswagen-golf"],
    ["/Mercado", "/mercado"],
    ["/avaliar//", "/avaliar"],
    ["/preco/volkswagen-golf/", "/preco/volkswagen-golf"],
  ];
  for (const [from, to] of cases) {
    const r = await get(from);
    assert(r.status === 301, `${from} → ${r.status}, expected 301`);
    assert(new URL(r.headers.get("location")).pathname === to,
      `${from} → ${r.headers.get("location")}, expected ${to}`);
  }
  const root = await get("/");
  assert(root.status === 200, `/ → ${root.status} (root must not redirect to "")`);
});

await check("the query string survives normalisation", async () => {
  const r = await get("/mercado/?zone=norte&sort=profit");
  assert(r.status === 301, `→ ${r.status}`);
  const loc = new URL(r.headers.get("location"));
  assert(loc.pathname === "/mercado" && loc.search === "?zone=norte&sort=profit",
    `lost the query: ${loc}`);
});

await check("analytics paths keep their case", async () => {
  // Streamlit's multipage nav generates /analytics/Market_Direction; lower-casing
  // it would 404 the dashboard for the only people who use it.
  const r = await get("/analytics/Market_Direction");
  assert(r.status !== 301, `analytics path was normalised (→ ${r.status})`);
  assert(r.status === 401, `expected the auth gate, got ${r.status}`);
});

await check("internal assets still require auth", async () => {
  for (const p of ["/files/app.py", "/data/dashboard/listings.parquet", "/index.html"]) {
    const r = await get(p);
    assert(r.status === 401, `${p} → ${r.status}, internal asset must stay gated`);
  }
});

await check("fonts are public and long-cached", async () => {
  const r = await get("/fonts/space-grotesk-var.woff2");
  assert(r.status === 200, `→ ${r.status}: a 401 on a preloaded font means fallback type`);
  assert(/max-age=31536000/.test(r.headers.get("cache-control") || ""), "font is not long-cached");
});

// ── the new pages ───────────────────────────────────────────────────────────
const slugs = Object.keys(models);
const deep = slugs.slice().sort((a, b) => models[b].n - models[a].n)[0];
const deepYear = yearPageYears(models[deep])[0];

await check("model, year and JSON routes answer", async () => {
  for (const [path, ctype] of [
    [`/preco/${deep}`, "text/html"],
    [`/preco/${deep}/${deepYear}`, "text/html"],
    [`/preco/${deep}.json`, "application/json"],
    [`/preco/${deep}/${deepYear}.json`, "application/json"],
  ]) {
    const r = await get(path);
    assert(r.status === 200, `${path} → ${r.status}`);
    assert((r.headers.get("content-type") || "").includes(ctype), `${path}: wrong content-type`);
  }
  const j = await (await get(`/preco/${deep}.json`)).json();
  assert(j.sample_size === models[deep].n, "JSON sample size disagrees with the blob");
  assert(j.collected_until, "JSON has no collection date");
});

await check("a year we hold data for but do not publish points at the model", async () => {
  const rec = models[deep];
  const published = new Set(yearPageYears(rec));
  const thin = (rec.yr || []).find(c => typeof c.y === "number" && !published.has(c.y));
  if (!thin) return;                       // this model has no thin years — fine
  const r = await get(`/preco/${deep}/${thin.y}`);
  assert(r.status === 301, `thin year ${thin.y} → ${r.status}, should redirect to the model`);
  assert(new URL(r.headers.get("location")).pathname === `/preco/${deep}`,
    `thin year ${thin.y} redirects to ${r.headers.get("location")}, not to its model`);
  const j = await get(`/preco/${deep}/${thin.y}.json`);
  assert(j.status === 301 && new URL(j.headers.get("location")).pathname === `/preco/${deep}.json`,
    "the JSON twin of a thin year does not point at the model's JSON twin");
  const never = await get(`/preco/${deep}/1900`);
  assert(never.status === 404, `a year we never had data for → ${never.status}, should be 404`);
});

await check("HEAD answers whatever GET answers", async () => {
  for (const p of ["/", `/preco/${deep}`, "/sitemap.xml", "/robots.txt", "/mercado", "/precos"]) {
    const g = await get(p);
    const h = await get(p, "HEAD");
    assert(h.status === g.status, `HEAD ${p} → ${h.status}, GET → ${g.status}`);
    assert(h.headers.get("content-type") === g.headers.get("content-type"),
      `HEAD ${p} answers a different content-type than GET`);
    assert(!(await h.text()), `HEAD ${p} returned a body`);
  }
  const missing = await get("/pagina-que-nao-existe", "HEAD");
  assert(missing.status === 404, `HEAD on an unknown path → ${missing.status}`);
});

await check("plain http upgrades to https instead of answering", async () => {
  const r = await worker.fetch(new Request(`http://${HOST}/preco/${deep}`), env);
  assert(r.status === 301, `http:// → ${r.status}, should redirect`);
  assert(r.headers.get("location") === `https://${HOST}/preco/${deep}`,
    `http:// redirects to ${r.headers.get("location")}`);
  const viaHeader = await worker.fetch(new Request(`https://${HOST}/preco/${deep}`,
    { headers: { "cf-visitor": '{"scheme":"http"}' } }), env);
  assert(viaHeader.status === 301, "a proxied http request was served instead of upgraded");
  const secure = await get(`/preco/${deep}`);
  assert(secure.status === 200, "https stopped working");
});

await check("nonsense under /preco 404s instead of 500ing", async () => {
  for (const p of [`/preco/nao-existe`, `/preco/${deep}/abc`, `/preco/${deep}/2014/extra`, `/preco/%`, `/preco/${deep}/9999`]) {
    const r = await get(p);
    assert(r.status === 404, `${p} → ${r.status}`);
  }
});

await check("the privacy page survives the new routing", async () => {
  // Merged with the parallel consent-banner work. It is an exact product path,
  // so it has to clear normalisation, the 404 fallthrough and the asset gate —
  // and it is what the consent banner links to, so a 401 or 404 here breaks
  // consent, not just SEO.
  const r = await get("/privacidade");
  assert(r.status === 200, `/privacidade → ${r.status}`);
  const body = await r.text();
  assert(body.includes("index,follow"), "/privacidade is not indexable");
  assert(body.includes("fc-consent"), "consent banner CSS was lost in the merge");
  const slash = await get("/privacidade/");
  assert(slash.status === 301, `/privacidade/ → ${slash.status}`);
  const xml = await (await get("/sitemap.xml")).text();
  assert(xml.includes(`<loc>https://${HOST}/privacidade</loc>`), "/privacidade missing from sitemap");
});

await check("the second-layer hubs and pages answer", async () => {
  const depSlug = depreciationSlugs(models)[0];
  const [pa, pb] = comparePairs(models)[0];
  for (const p of ["/depreciacao", `/depreciacao/${depSlug}`, "/comparar", `/comparar/${pa}-vs-${pb}`,
                   "/liquidez", "/sobrevalorizados", "/metodologia", "/sobre", "/isv", "/mercado/indice"]) {
    const r = await get(p);
    assert(r.status === 200, `${p} → ${r.status}`);
  }
});

await check("a liquidity page exists exactly where the curve does", async () => {
  const withLq = slugs.filter(s2 => models[s2].lq && models[s2].lq.s30 != null);
  assert(withLq.length, "fixture carries no liquidity records");
  const slug = withLq[0];
  const r = await get(`/liquidez/${slug}`);
  assert(r.status === 200, `/liquidez/${slug} → ${r.status}`);
  const body = await r.text();
  assert(body.includes("primeiro mês"), "liquidity page never states the 30-day share");
  assert(body.includes("ciclos de 30 dias"), "liquidity page hides the expiry caveat");
  const without = slugs.find(s2 => !(models[s2].lq && models[s2].lq.s30 != null));
  if (without) {
    assert((await get(`/liquidez/${without}`)).status === 404,
      "a liquidity page exists for a model with no curve");
  }
  assert((await get("/liquidez/nao-existe")).status === 404, "unknown slug is not a 404");
});

await check("the liquidity page has a JSON twin and is linked, not orphaned", async () => {
  const slug = slugs.find(s2 => models[s2].lq && models[s2].lq.s30 != null);
  const r = await get(`/liquidez/${slug}.json`);
  assert(r.status === 200, `/liquidez/${slug}.json → ${r.status}`);
  assert((r.headers.get("content-type") || "").includes("application/json"), "JSON twin is not served as JSON");
  const j = JSON.parse(await r.text());
  assert(j.slug === slug, "JSON twin is about another model");
  assert(j.gone_in_30d > 0 && j.gone_in_30d <= 1, "JSON twin has no 30-day share");
  assert(j.sample_ended > 0, "JSON twin has no sample size");
  assert(typeof j.caveat === "string" && j.caveat.length > 20, "JSON twin drops the caveat");

  const modelPage = await (await get(`/preco/${slug}`)).text();
  assert(modelPage.includes(`/liquidez/${slug}`), "model page does not link its liquidity page");
  const hub = await (await get("/liquidez")).text();
  assert(hub.includes(`href="/liquidez/${slug}"`), "/liquidez does not link the per-model pages");
  const xml = await (await get("/sitemap.xml")).text();
  assert(xml.includes(`<loc>https://${HOST}/liquidez/${slug}</loc>`), "sitemap missing the liquidity page");
});

await check("the import pages exist only where both markets have the same year", async () => {
  const slugs = importSlugs(idoc);
  assert(slugs.length, "import fixture carries no models");
  const hub = await get("/importar");
  assert(hub.status === 200, `/importar → ${hub.status}`);
  const hubBody = await hub.text();
  assert(hubBody.includes(`href="/importar/${slugs[0]}"`), "hub does not link its models");

  const r = await get(`/importar/${slugs[0]}`);
  assert(r.status === 200, `/importar/${slugs[0]} → ${r.status}`);
  const body = await r.text();
  assert(body.includes("Total à porta"), "the landed-cost column is missing");
  assert(body.includes("ISV"), "the page never mentions the tax");
  assert((await get("/importar/nao-existe")).status === 404, "unknown import slug is not a 404");

  const j = await get(`/importar/${slugs[0]}.json`);
  assert(j.status === 200, "import JSON twin missing");
  const doc = JSON.parse(await j.text());
  assert(doc.slug === slugs[0], "JSON twin is about another model");
  assert(doc.years.length >= 2, "JSON twin published a model with one year");
  assert(doc.fixed_costs_eur.low > 0 && doc.fixed_costs_eur.high > doc.fixed_costs_eur.low,
    "JSON twin quotes a single legalisation number");
  for (const y of doc.years) {
    assert(y.landed_low_eur === Math.round(y.de_asking_median + y.isv_median_eur + doc.fixed_costs_eur.low),
      `${y.year}: landed cost is not price + tax + fees`);
    assert(y.de_listings >= 10 && y.pt_listings >= 5, `${y.year}: a cell below its floor shipped`);
  }

  const xml = await (await get("/sitemap.xml")).text();
  assert(xml.includes(`<loc>https://${HOST}/importar</loc>`), "sitemap missing the import hub");
  assert(xml.includes(`<loc>https://${HOST}/importar/${slugs[0]}</loc>`), "sitemap missing the import page");
});

await check("no German data means no import layer at all, not an empty page", async () => {
  const prevFetch = globalThis.fetch;
  globalThis.fetch = async (input, init) => {
    const u = typeof input === "string" ? input : input.url;
    if (u.includes("import.json")) return new Response("nope", { status: 404 });
    return prevFetch(input, init);
  };
  try {
    assert((await get("/importar")).status === 404, "/importar answered without data behind it");
    assert((await get(`/importar/${importSlugs(idoc)[0]}`)).status === 404,
      "an import page answered without data behind it");
    const xml = await (await get("/sitemap.xml")).text();
    assert(!xml.includes("/importar"), "sitemap advertises the import layer with no data");
  } finally { globalThis.fetch = prevFetch; }
});

await check("the depreciation curve has a JSON twin", async () => {
  const depSlug = depreciationSlugs(models)[0];
  const r = await get(`/depreciacao/${depSlug}.json`);
  assert(r.status === 200, `/depreciacao/${depSlug}.json → ${r.status}`);
  assert((r.headers.get("content-type") || "").includes("application/json"), "JSON twin is not served as JSON");
  const j = JSON.parse(await r.text());
  assert(j.slug === depSlug, "JSON twin is about another model");
  assert(j.annual_depreciation_rate > 0, "JSON twin has no rate");
  assert(Array.isArray(j.cost_of_one_year_of_age) && j.cost_of_one_year_of_age.length > 0,
    "JSON twin has no euro ladder");
  assert(j.rate_bend || j.rate_bend_note, "JSON twin is silent about the bend test");
  const html = await (await get(`/depreciacao/${depSlug}`)).text();
  assert(html.includes(`/depreciacao/${depSlug}.json`), "the page never points at its own JSON");
});

await check("generated pages outside the published set 404", async () => {
  const notDep = slugs.find(s => !depreciationSlugs(models).includes(s));
  assert((await get(`/depreciacao/${notDep}`)).status === 404, "served a depreciation page with no curve");
  assert((await get(`/depreciacao/${notDep}.json`)).status === 404, "served JSON for a curve we do not publish");
  assert((await get("/comparar/volkswagen-golf-vs-volkswagen-golf")).status === 404, "served a self-comparison");
  assert((await get("/mercado/indice/1999-w03")).status === 404, "served an index week we never recorded");
  assert((await get("/mercado/indice/lixo")).status === 404, "served a malformed index week");
  assert((await get("/mercado/indice/1999-03")).status === 404, "served an index month we never recorded");
  assert((await get("/mercado/indice/2026-13")).status === 404, "served a month that does not exist");
  // The ISO spelling is what a human copies out of the page text; it must land
  // on the lower-case URL rather than 404.
  const upper = await get("/mercado/indice/1999-W03");
  assert(upper.status === 301 && new URL(upper.headers.get("location")).pathname === "/mercado/indice/1999-w03",
    `ISO-cased week did not normalise (${upper.status})`);
});

await check("the weekly index writes exactly one snapshot per week", async () => {
  kv.clear();
  await get("/mercado/indice");
  const afterFirst = [...kv.keys()].filter(k => k.startsWith("idx:week:"));
  assert(afterFirst.length === 1, `expected 1 week key, got ${afterFirst.length}`);
  const written = kv.get(afterFirst[0]);
  await get("/mercado/indice");
  await get("/mercado/indice");
  assert([...kv.keys()].filter(k => k.startsWith("idx:week:")).length === 1, "wrote a second key for the same week");
  assert(kv.get(afterFirst[0]) === written, "rewrote an archived week — the URL is no longer citable");
  // …and that week is now reachable at its permanent address.
  const wk = afterFirst[0].replace("idx:week:", "");
  assert((await get(`/mercado/indice/${wk.toLowerCase()}`)).status === 200, "archived week is not reachable");
});

// ── sitemap ↔ router agreement ──────────────────────────────────────────────
await check("sitemap lists the model-year pages", async () => {
  const xml = await (await get("/sitemap.xml")).text();
  const locs = [...xml.matchAll(/<loc>([^<]+)<\/loc>/g)].map(m => new URL(m[1]).pathname);
  const expectedYears = slugs.reduce((n, s) => n + yearPageYears(models[s]).length, 0);
  const gotYears = locs.filter(p => /^\/preco\/[^/]+\/\d{4}$/.test(p)).length;
  assert(gotYears === expectedYears, `sitemap has ${gotYears} year URLs, router serves ${expectedYears}`);
  assert(locs.includes("/metodologia") && locs.includes("/sobre"), "trust pages missing from sitemap");
  assert(locs.filter(p => p.startsWith("/comparar/")).length === comparePairs(models).length,
    "sitemap comparison count disagrees with the generator");
  assert(new Set(locs).size === locs.length, "sitemap contains duplicate URLs");
});

await check("every sitemap URL resolves to a 200", async () => {
  const xml = await (await get("/sitemap.xml")).text();
  const locs = [...xml.matchAll(/<loc>([^<]+)<\/loc>/g)].map(m => new URL(m[1]).pathname);
  // Sample rather than fetch ~1 200 pages: one of every shape, plus a spread.
  const shapes = new Map();
  for (const p of locs) {
    const shape = p.replace(/\/preco\/[^/]+\/\d{4}/, "/preco/*/YYYY")
                   .replace(/\/preco\/[^/]+/, "/preco/*")
                   .replace(/\/depreciacao\/[^/]+/, "/depreciacao/*")
                   .replace(/\/liquidez\/[^/]+/, "/liquidez/*")
                   .replace(/\/comparar\/[^/]+/, "/comparar/*")
                   .replace(/\/mercado\/indice\/[^/]+/, "/mercado/indice/*");
    if (!shapes.has(shape)) shapes.set(shape, []);
    if (shapes.get(shape).length < 8) shapes.get(shape).push(p);
  }
  const sample = [...shapes.values()].flat();
  for (const p of sample) {
    const r = await get(p);
    assert(r.status === 200, `sitemap advertises ${p} but the router answers ${r.status}`);
  }
  console.log(`       (sampled ${sample.length} of ${locs.length} sitemap URLs across ${shapes.size} shapes)`);
});

await check("robots and llms.txt describe the new surface", async () => {
  const robots = await (await get("/robots.txt")).text();
  assert(robots.includes("Sitemap: https://carsbuyer.org/sitemap.xml"), "sitemap line missing");
  const groups = robots.split(/\n(?=User-agent:)/).filter(g => g.startsWith("User-agent:"));
  assert(groups.length >= 8, `robots has ${groups.length} groups, expected the wildcard plus the answer engines`);
  for (const g of groups) {
    const who = g.split("\n")[0];
    for (const path of ["/reservas", "/claim", "/reserve", "/unlocked", "/analytics", "/_olx"]) {
      assert(g.includes(`Disallow: ${path}`), `${who} is not told to skip ${path}`);
    }
    assert(g.includes("Allow: /"), `${who} lost its Allow`);
  }
  const llms = await (await get("/llms.txt")).text();
  assert(!/vaga/i.test(llms), "llms.txt promises a wave section with no wave set");
  const gatedLlms = await (await worker.fetch(new Request(`https://${HOST}/llms.txt`),
    { ...env, SEO_WAVE_MODELS: "5" })).text();
  assert(/## Publicação por vagas/.test(gatedLlms),
    "llms.txt hides the wave that decides which of its address templates resolve");
  assert(/page: null/.test(gatedLlms), "llms.txt does not say where the withheld numbers are");
  for (const needle of ["/preco/{slug}/{ano}", "/depreciacao/{slug}", "/comparar/{slug-a}-vs-{slug-b}",
                        "/preco/{slug}.json", "/mercado/indice"]) {
    assert(llms.includes(needle), `llms.txt does not document ${needle}`);
  }
});

await check("wrong method on a known path is a 404, not a 401", async () => {
  const r = await get("/reserve");       // GET on a POST-only route
  assert(r.status === 404, `→ ${r.status}`);
});

await check("the canonical-host redirect still bypasses the webhook", async () => {
  const wh = await worker.fetch(
    new Request("https://olx-car-parser.permikov134.workers.dev/webhook/stripe", { method: "POST", body: "{}" }), env);
  assert(wh.status !== 301 && wh.status !== 308, `webhook was redirected (${wh.status})`);
  const hz = await worker.fetch(new Request("https://olx-car-parser.permikov134.workers.dev/healthz"), env);
  assert(hz.status === 200, `healthz was redirected (${hz.status})`);
  const page = await worker.fetch(new Request("https://olx-car-parser.permikov134.workers.dev/precos"), env);
  assert(page.status === 301, `off-host page not redirected (${page.status})`);
});

// ── facets (fuel / district) ────────────────────────────────────────────────
//
// The live blob does not carry `fx`/`dt` yet — model_pages.py emits them from
// the next data build. That is the designed order: the routes must 404 today
// and light up on their own when the data lands, with no deploy. Both halves
// are checked here, against the real blob and against an augmented copy.

await check("facet URLs 404 while the blob has no facet cells", async () => {
  if (models[deep].fx || models[deep].dt || models[deep].tx) return;   // data has landed; skip
  for (const p of [`/preco/${deep}/diesel`, `/preco/${deep}/porto`, "/precos/porto",
                   `/preco/${deep}/automatica`,
                   ...Object.values(DUELS).flatMap(d => [`/${d.path}`, `/${d.path}/${deep}`])]) {
    const r = await get(p);
    assert(r.status === 404, `${p} → ${r.status}, must 404 before the data exists`);
  }
  for (const p of ["/", "/precos", `/preco/${deep}`, "/depreciacao", "/metodologia"]) {
    const body = await (await get(p)).text();
    const links = Object.values(DUELS).flatMap(d =>
      [...body.matchAll(new RegExp(`href="(/${d.path}[^"]*)"`, "g"))].map(m => m[1]));
    assert(links.length === 0, `${p} links ${links[0]} while no model carries the fit`);
  }
});

await check("facet pages appear when the blob carries the cells", async () => {
  const augmented = JSON.parse(JSON.stringify(mdoc));
  augmented.built_at = "2026-08-26T00:00:00Z";        // bust the corpus-stats memo
  const rec = augmented.models[deep];
  rec.fx = [
    { k: "diesel", lbl: "Diesel", n: 40, fl: 5000, fm: 7000, fh: 9000, km: 190000, y0: 2010, y1: 2018,
      vsm: [1.05, 7], vs: { gasolina: [1.2, 6] } },
    { k: "gasolina", lbl: "Gasolina", n: 18, fl: 4200, fm: 5800, fh: 7600, km: 150000, y0: 2008, y1: 2016,
      vsm: [0.9, 5], vs: { diesel: [0.833, 6] } },
  ];
  rec.tx = [
    { k: "manual", lbl: "Manual", n: 60, fl: 2800, fm: 4999, fh: 7500, km: 232000, y0: 2004, y1: 2016 },
    { k: "automatica", lbl: "Automática", n: 20, fl: 14500, fm: 21500, fh: 27000, km: 127000, y0: 2017, y1: 2024 },
  ];
  const FIT = {
    a: { n: 40, r: 0.066, km: 190000, fm: 7000 },
    b: { n: 30, r: 0.086, km: 150000, fm: 5800 },
    ci: 0.0097, t: 4.07, r2: 0.76, y0: 2006, y1: 2022,
    gap: [[6, 0.11, 0.05], [12, -0.02, 0.06]],
  };
  for (const d of Object.values(DUELS)) rec[d.key] = FIT;
  rec.dt = [{ k: "porto", lbl: "Porto", n: 22, fl: 5200, fm: 7300, fh: 9100, km: 185000 }];
  augmented.districts = {
    porto: { lbl: "Porto", n: 4200, fl: 4000, fm: 8000, fh: 14000, kmm: 175000,
             top: [[deep, 22, 7300]] },
  };

  const prevFetch = globalThis.fetch;
  globalThis.fetch = async (input, init) => {
    const u = typeof input === "string" ? input : input.url;
    if (u.includes("models.json")) return new Response(JSON.stringify(augmented), { status: 200 });
    return prevFetch(input, init);
  };
  try {
    for (const p of [`/preco/${deep}/diesel`, `/preco/${deep}/gasolina`, `/preco/${deep}/porto`, "/precos/porto",
                     `/preco/${deep}/manual`, `/preco/${deep}/automatica`,
                     ...Object.values(DUELS).flatMap(d => [`/${d.path}`, `/${d.path}/${deep}`])]) {
      const r = await get(p);
      assert(r.status === 200, `${p} → ${r.status}`);
      const body = await r.text();
      assert(body.includes('<link rel="canonical" href="https://carsbuyer.org' + p + '">'),
        `${p}: canonical is not self`);
      assert(body.includes("index,follow"), `${p}: not indexable`);
      assert(!body.replace(/<script[\s\S]*?<\/script>/g, "").includes("undefined"),
        `${p}: rendered "undefined"`);
    }
    const diesel = await (await get(`/preco/${deep}/diesel`)).text();
    assert(diesel.includes(`/preco/${deep}/gasolina`), "fuel facet does not link its sibling");
    assert(diesel.includes("20% mais caro") && diesel.includes("comparando ano a ano"),
      "fuel facet states no year-matched gap");
    assert(!diesel.includes("21% mais caro"), "fuel facet fell back to the raw medians");

    const auto = await (await get(`/preco/${deep}/automatica`)).text();
    assert(!/% (mais caro|mais barato)/.test(auto),
      "gearbox facet claims a price gap it cannot measure year-matched");
    assert(auto.includes("inclui a diferença de anos"), "gearbox facet hides why there is no percentage");
    assert(auto.includes(`/preco/${deep}/manual`), "gearbox facet does not link its sibling");

    for (const d of Object.values(DUELS)) {
      const page = await (await get(`/${d.path}/${deep}`)).text();
      for (const href of [...page.matchAll(/href="(\/preco\/[^"]+)"/g)].map(m => m[1])) {
        assert((await get(href)).status === 200,
          `${d.path}/${deep} links ${href}, which does not resolve`);
      }
      assert(/6,6%/.test(page) && /8,6%/.test(page), `${d.path}: lost one of the two rates`);
      assert(page.includes("±1,0 pp"), `${d.path}: hides the interval`);
      assert(page.includes(`/preco/${deep}/${d.a.facet}`), `${d.path}: does not link its facet cuts`);
      const dj = await (await get(`/${d.path}/${deep}.json`)).json();
      assert(dj.distinguishable_at_95 === true && dj.holds_value_better === d.a.json,
        `${d.path}: JSON twin disagrees with the page`);
      assert(dj.measured === "asking_price", `${d.path}: JSON does not say what it measures`);
      const hub = await (await get(`/${d.path}`)).text();
      assert(hub.includes(`/${d.path}/${deep}`), `${d.path}: hub does not link its own page`);
    }

    for (const k of ["diesel", "manual", "porto"]) {
      const r = await get(`/preco/${deep}/${k}.json`);
      assert(r.status === 200, `/preco/${deep}/${k}.json → ${r.status}`);
      assert((r.headers.get("content-type") || "").includes("application/json"),
        `/preco/${deep}/${k}.json still answers with the HTML page`);
      const fj = await r.json();
      assert(fj.facet && fj.facet.key === k, `${k}.json does not name its own cut`);
      assert(fj.measured === "asking_price", `${k}.json does not say what it measures`);
      assert(fj.sample_size > 0, `${k}.json lost the sample size`);
      for (const sib of fj.siblings) {
        assert((await get(new URL(sib.page).pathname)).status === 200,
          `${k}.json advertises ${sib.page} but it answers otherwise`);
      }
    }
    const facetHtml = await (await get(`/preco/${deep}/diesel`)).text();
    assert(facetHtml.includes(`/preco/${deep}/diesel.json`),
      "the facet page hides its own JSON twin");

    const solo = slugs.find(s2 => {
      const r = models[s2];
      for (const kind of ["fx", "tx"]) {
        const cells = r[kind] || [];
        if (cells.length === 1 && r.n && cells[0].n / r.n >= 0.85) return true;
      }
      return false;
    });
    if (solo) {
      const r = models[solo];
      const kind = (r.fx || []).length === 1 && r.fx[0].n / r.n >= 0.85 ? "fx" : "tx";
      const key = r[kind][0].k;
      const red = await get(`/preco/${solo}/${key}`);
      assert(red.status === 301,
        `a retired near-duplicate facet answered ${red.status}, throwing away an indexed URL`);
      assert(new URL(red.headers.get("location")).pathname === `/preco/${solo}`,
        "a retired facet does not fold into its model page");
      const xml2 = await (await get("/sitemap.xml")).text();
      assert(!xml2.includes(`<loc>https://${HOST}/preco/${solo}/${key}</loc>`),
        "sitemap still advertises a retired facet");
    }

    const unknown = await get(`/preco/${deep}/nao-existe`);
    assert(unknown.status === 404, `unknown facet → ${unknown.status}`);
    for (const d of Object.values(DUELS)) {
      const noDuel = slugs.find(s2 => s2 !== deep && !augmented.models[s2][d.key]);
      if (!noDuel) continue;
      assert((await get(`/${d.path}/${noDuel}`)).status === 404,
        `${d.path}: a page exists for a model with no fit`);
    }
    assert((await get("/precos/nao-existe")).status === 404, "served a district we have no data for");

    const xml = await (await get("/sitemap.xml")).text();
    for (const want of [`/preco/${deep}/diesel`, `/preco/${deep}/porto`, "/precos/porto",
                        `/preco/${deep}/automatica`,
                        ...Object.values(DUELS).flatMap(d => [`/${d.path}`, `/${d.path}/${deep}`])]) {
      assert(xml.includes(`<loc>https://carsbuyer.org${want}</loc>`), `sitemap missing ${want}`);
    }

    // In the sitemap is not enough. A page nothing on the site links to is an
    // orphan: a crawler reaches it once and a reader never does.
    const modelPage = await (await get(`/preco/${deep}`)).text();
    for (const want of [`/preco/${deep}/diesel`, `/preco/${deep}/gasolina`, `/preco/${deep}/porto`,
                        `/preco/${deep}/manual`, `/preco/${deep}/automatica`,
                        ...Object.values(DUELS).map(d => `/${d.path}/${deep}`)]) {
      assert(modelPage.includes(want), `model page does not link ${want}`);
    }
    const hub = await (await get("/precos")).text();
    assert(hub.includes('href="/precos/porto"'), "/precos does not link the district pages");
  } finally {
    globalThis.fetch = prevFetch;
  }
});

// ── staged rollout ──────────────────────────────────────────────────────────
await check("a wave gates the router, the sitemap and the on-page links together", async () => {
  const wave = 5;
  const gated = { ...env, SEO_WAVE_MODELS: String(wave) };
  const g = (p) => worker.fetch(new Request(`https://${HOST}${p}`), gated);
  const inWave = slugs.slice().sort((a, b) => (models[b].n || 0) - (models[a].n || 0)
    || (a < b ? -1 : 1)).slice(0, wave);
  const outside = slugs.find(s => !inWave.includes(s) && yearPageYears(models[s]).length);
  assert(outside, "fixture has no model outside a 5-model wave");

  // The model page itself stays live — only the second layer is staged.
  assert((await g(`/preco/${outside}`)).status === 200, "wave hid a model page");
  const yr = yearPageYears(models[outside])[0];
  assert((await g(`/preco/${outside}/${yr}`)).status === 404,
    "a year page outside the wave is still reachable");
  const insideYear = yearPageYears(models[inWave[0]])[0];
  assert((await g(`/preco/${inWave[0]}/${insideYear}`)).status === 200,
    "a year page inside the wave is not reachable");

  // Unreachable AND unlinked: an in-page link to a 404 is worse than no page.
  const page = await (await g(`/preco/${outside}`)).text();
  assert(!page.includes(`/preco/${outside}/${yr}`), "model page links a year page the wave hides");

  const feed = await (await g(`/preco/${outside}.json`)).json();
  const advertised = feed.by_year.filter(c => c.page);
  assert(advertised.length === 0,
    `JSON feed advertises ${advertised.length} year pages the wave hides`);
  assert(feed.by_year.some(c => c.year === yr),
    "JSON feed dropped the year cell along with its page");
  assert(feed.related.depreciation === null && feed.related.facets.length === 0,
    "JSON feed advertises related pages the wave hides");
  assert(feed.by_fuel.every(c => c.page === null) && feed.by_transmission.every(c => c.page === null),
    "JSON feed advertises facet pages the wave hides");
  const insideFeed = await (await g(`/preco/${inWave[0]}.json`)).json();
  const advertisedInside = insideFeed.by_year.filter(x => x.page);
  assert(advertisedInside.length > 0,
    "the deepest model in the wave advertises no year page at all — the feed lost its link graph");
  assert(insideFeed.related.depreciation || insideFeed.related.liquidity,
    "the feed advertises no related page for the deepest model in the wave");
  assert(insideFeed.related.facets.length > 0,
    "the feed advertises no facet page for the deepest model in the wave");
  assert(insideFeed.by_fuel.some(c => c.page) || insideFeed.by_transmission.some(c => c.page),
    "the feed carries facet breakdowns but no addresses for any of them");
  for (const c of advertisedInside) {
    assert((await g(new URL(c.page).pathname)).status === 200,
      `JSON feed advertises ${c.page} but it answers otherwise`);
  }
  for (const u of [insideFeed.related.depreciation, insideFeed.related.liquidity,
                   ...insideFeed.related.facets, ...insideFeed.related.duels].filter(Boolean)) {
    assert((await g(new URL(u).pathname)).status === 200,
      `JSON feed advertises ${u} but it answers otherwise`);
  }

  const duelOutside = slugs.find(s2 => !inWave.includes(s2)
    && Object.values(DUELS).some(d => models[s2][d.key]));
  if (duelOutside) {
    const d = Object.values(DUELS).find(x => models[duelOutside][x.key]);
    const page = await (await g(`/${d.path}/${duelOutside}`)).text();
    for (const href of [...page.matchAll(/href="(\/preco\/[^"]+)"/g)].map(m => m[1])) {
      assert((await g(href)).status === 200,
        `a duel page outside the wave links ${href}, which the wave hides`);
    }
  }

  const xml = await (await g("/sitemap.xml")).text();
  assert(!xml.includes(`<loc>https://${HOST}/preco/${outside}/${yr}</loc>`),
    "sitemap advertises a page outside the wave");
  const gatedYears = [...xml.matchAll(/<loc>[^<]*\/preco\/[^/<]+\/\d{4}<\/loc>/g)].length;
  const allYears = slugs.reduce((n, s2) => n + yearPageYears(models[s2]).length, 0);
  assert(gatedYears > 0 && gatedYears < allYears,
    `wave did not narrow the sitemap (${gatedYears} of ${allYears})`);

  // Everything the gated sitemap still advertises must resolve.
  for (const loc of [...xml.matchAll(/<loc>([^<]+)<\/loc>/g)].slice(0, 60)) {
    const path = new URL(loc[1]).pathname;
    assert((await g(path)).status === 200, `gated sitemap advertises ${path} but it answers otherwise`);
  }
});

await check("the deal feed carries its own markup, and only on the canonical view", async () => {
  const yr = yearPageYears(models[deep])[0];
  const feedDeal = {
    olx_id: "MKT1", brand: models[deep].b, model: models[deep].m, title: "Carro de teste",
    price_eur: 7000, fair_median: 8500, fair_low: 7600, fair_high: 9400, discount_pct: 0.17,
    est_profit_eur: 1500, year: yr, mileage_km: 180000, fuel_type: "Diesel",
    district: "Porto", photo_urls: [], days_on_market: 12,
    first_seen_at: "2026-08-01T00:00:00Z", seller_type: "Particular",
  };
  const feedStamp = "2026-09-01T06:39:25Z";
  const prevFetch = globalThis.fetch;
  const fresh = { ...env, KV: { ...env.KV, async get() { return null; }, async put() {} } };
  globalThis.fetch = async (input, init) => {
    const u = typeof input === "string" ? input : input.url;
    if (u.includes("hot_deals_")) {
      return new Response(JSON.stringify({ built_at: feedStamp, deals: [feedDeal] }), { status: 200 });
    }
    return prevFetch(input, init);
  };
  try {
    const page = await (await worker.fetch(new Request(`https://${HOST}/mercado`), fresh)).text();
    const m = page.match(/<script type="application\/ld\+json">([\s\S]*?)<\/script>/);
    assert(m, "/mercado still ships no JSON-LD");
    const graph = JSON.parse(m[1])["@graph"];
    const types = graph.map(n => n["@type"]);
    for (const t of ["CollectionPage", "ItemList", "BreadcrumbList"]) {
      assert(types.includes(t), `/mercado JSON-LD is missing ${t}`);
    }
    const coll = graph.find(n => n["@type"] === "CollectionPage");
    assert(coll.dateModified === feedStamp,
      `dateModified is ${coll.dateModified}, not the feed's own stamp`);
    const list = graph.find(n => n["@type"] === "ItemList");
    assert(list.numberOfItems === list.itemListElement.length, "ItemList miscounts itself");
    list.itemListElement.forEach((it, i) => {
      assert(it.position === i + 1, `ItemList position ${it.position} at index ${i}`);
    });
    assert(!coll.mainEntity,
      "CollectionPage claims a list of reference links is the page's main content");
    for (const it of list.itemListElement) {
      const path = new URL(it.url).pathname;
      assert(page.includes(`href="${path}"`), `ItemList advertises ${path}, the page never links it`);
      assert((await worker.fetch(new Request(`https://${HOST}${path}`), fresh)).status === 200,
        `ItemList advertises ${path} but it answers otherwise`);
    }
    assert(page.includes(`/preco/${deep}/${yr}`), "the feed does not link the year of the car it shows");

    const zoned = await (await worker.fetch(new Request(`https://${HOST}/mercado?zone=norte`), fresh)).text();
    assert(!zoned.includes("application/ld+json"),
      "a filtered view ships an ItemList the canonical URL does not serve");
  } finally {
    globalThis.fetch = prevFetch;
  }
});


await check("lastmod tells a frozen archive cut apart from a page rebuilt daily", async () => {
  const kv2 = new Map();
  const env2 = { ...env, KV: {
    async get(k, type) { const v = kv2.get(k); return v === undefined ? null : (type === "json" ? JSON.parse(v) : v); },
    async put(k, v) { kv2.set(k, v); },
    async list() { return { keys: [] }; },
    async delete(k) { kv2.delete(k); },
  } };
  const frozen = { week: "2026-W10", date: "2026-03-09", builtAt: "2026-03-09T06:00:00Z" };
  await env2.KV.put("idx:weeks", JSON.stringify([frozen]));
  const xml = await (await worker.fetch(new Request(`https://${HOST}/sitemap.xml`), env2)).text();
  const row = re => (xml.match(re) || [""])[0];
  const archive = row(new RegExp(`<url><loc>https://${HOST}/mercado/indice/2026-w10</loc>[^]*?</url>`));
  assert(archive.includes("<lastmod>2026-03-09</lastmod>"),
    `a permanent weekly cut claims it changed today: ${archive}`);
  const priv = row(new RegExp(`<url><loc>https://${HOST}/privacidade</loc>[^]*?</url>`));
  assert(!priv.includes("<lastmod>"), `the one static page still dates itself to the build: ${priv}`);
  const model = row(new RegExp(`<url><loc>https://${HOST}/preco/${deep}</loc>[^]*?</url>`));
  assert(model.includes(`<lastmod>${String(mdoc.built_at).slice(0, 10)}</lastmod>`),
    "a page rebuilt every few hours lost its build stamp");
  assert(model.includes("<changefreq>daily</changefreq>"),
    "a page rebuilt every few hours still advertises weekly");
});

await check("the liquidity layer is staged on its own knob, not the price wave", async () => {
  const priceWaved = { ...env, SEO_WAVE_MODELS: "5" };
  const outside = slugs.slice().sort((a, b) => (models[b].n || 0) - (models[a].n || 0)
    || (a < b ? -1 : 1)).slice(5).find(s2 => liquidityOk(models[s2]));
  assert(outside, "fixture has no liquidity model outside a 5-model price wave");
  const g = (p, e) => worker.fetch(new Request(`https://${HOST}${p}`), e);
  assert((await g(`/liquidez/${outside}`, priceWaved)).status === 200,
    "the price wave still hides a liquidity page it does not stage");

  const liqCapped = { ...priceWaved, LIQ_WAVE_MODELS: "1" };
  const deepest = Object.entries(models)
    .filter(([, r]) => liquidityOk(r))
    .sort((a, b) => ((b[1].lq && b[1].lq.n) || 0) - ((a[1].lq && a[1].lq.n) || 0)
                 || (a[0] < b[0] ? -1 : 1))[0][0];
  assert((await g(`/liquidez/${deepest}`, liqCapped)).status === 200,
    "the liquidity wave hid its own deepest model");
  const hidden = Object.keys(models).find(s2 => s2 !== deepest && liquidityOk(models[s2]));
  if (hidden) {
    assert((await g(`/liquidez/${hidden}`, liqCapped)).status === 404,
      "a liquidity page outside its own wave is still reachable");
    const xml = await (await g("/sitemap.xml", liqCapped)).text();
    assert(!xml.includes(`<loc>https://${HOST}/liquidez/${hidden}</loc>`),
      "sitemap advertises a liquidity page outside its own wave");
  }
});

await check("a cacheable page carries nothing that belongs to one visitor", async () => {
  const PUBLIC = ["/precos", `/preco/${deep}`, `/preco/${deep}/${deepYear}`, "/liquidez",
    "/depreciacao", "/comparar", "/metodologia", "/sobre", "/isv", "/privacidade",
    "/sobrevalorizados", "/mercado/indice"];
  const UID = "deadbeefdeadbeefdeadbeefdeadbeef";
  const stateful = { ...env, KV: { ...env.KV,
    async list(arg) {
      const prefix = (arg && arg.prefix) || "";
      if (prefix === `unlock:${UID}:`) {
        return { keys: [{ name: `${prefix}A1` }, { name: `${prefix}A2` }], list_complete: true };
      }
      return env.KV.list(arg);
    } } };
  const withCookie = p => worker.fetch(new Request(`https://${HOST}${p}`,
    { headers: { cookie: `fc_uid=${UID}` } }), stateful);
  const sanity = await (await withCookie("/reservas")).text();
  assert(/€10 em depósito/.test(sanity),
    "the stub gives this visitor no deposits, so the comparison below would be vacuous");
  for (const p of PUBLIC) {
    const anon = await get(p);
    if (anon.status !== 200) continue;
    const cc = anon.headers.get("cache-control") || "";
    assert(/^public\b/.test(cc), `${p} is not cacheable (${cc})`);
    assert(!anon.headers.get("set-cookie"),
      `${p} sets a visitor cookie on a response a shared cache may keep`);
    const mine = await withCookie(p);
    assert((await anon.text()) === (await mine.text()),
      `${p} renders differently for a visitor with a cookie — a shared cache would leak it`);
  }
  for (const p of ["/", "/mercado", "/avaliar", "/reservas"]) {
    const r = await get(p);
    const cc = r.headers.get("cache-control") || "";
    assert(/private/.test(cc), `${p} carries per-visitor state but is cacheable (${cc})`);
  }
});

// ── KV must not be load-bearing for a render ────────────────────────────────
// The outage these exist for: KV refused ops, every rendered page answered
// 1101, and /healthz plus the sitemap stayed green the whole time.
const PAGES = ["/", "/mercado", "/precos", "/privacidade", "/sobre", "/metodologia", "/isv",
  "/liquidez", "/depreciacao", "/comparar", "/mercado/indice", "/sobrevalorizados", "/avaliar",
  "/reservas", "/car?olx_id=x", `/preco/${deep}`, `/preco/${deep}/${deepYear}`];

await check("a KV that fails every op degrades the page, never 500s", async () => {
  const bang = op => { throw new Error(`KV ${op} failed: 429 Too Many Requests`); };
  const hostile = { ...env, KV: {
    async get() { bang("GET"); }, async put() { bang("PUT"); },
    async delete() { bang("DELETE"); }, async list() { bang("LIST"); },
  } };
  for (const p of PAGES) {
    const r = await worker.fetch(new Request(`https://${HOST}${p}`), hostile);
    assert(r.status < 500, `${p} → ${r.status} with KV refusing every op`);
  }
});

await check("a cookie-less render spends no KV list op", async () => {
  // Every crawler hit arrives without the cookie, and one scan per render is
  // what drained the daily list quota.
  let lists = 0;
  const counted = { ...env, KV: { ...env.KV, async list(arg) { lists++; return env.KV.list(arg); } } };
  for (const p of PAGES) await worker.fetch(new Request(`https://${HOST}${p}`), counted);
  assert(lists === 0, `${lists} list op(s) on cookie-less renders`);

  // A returning visitor is still scanned: that scan is what badges the tiles.
  const r = await worker.fetch(new Request(`https://${HOST}/mercado`,
    { headers: { cookie: "fc_uid=deadbeefdeadbeefdeadbeefdeadbeef" } }), counted);
  assert(r.status === 200, `returning visitor → ${r.status}`);
  assert(lists === 1, `returning visitor spent ${lists} list op(s), expected 1`);
});

// ── weekly index: the cron, and the honesty about gaps ──────────────────────

await check("ISO week labels and their Mondays agree, including the year seam", async () => {
  const wk = (s2) => isoWeek(new Date(s2 + "T12:00:00Z"));
  assert(wk("2026-08-30") === "2026-W35", "Sunday still belongs to the week that started Monday");
  assert(wk("2026-08-31") === "2026-W36", "Monday must start a new week");
  // The week containing 1 January is the classic off-by-one; both of these are W53.
  assert(wk("2026-12-28") === "2026-W53" && wk("2027-01-03") === "2026-W53",
    "year seam mislabelled");
  // Round-trip: the Monday of a label must map back to that label.
  for (const label of ["2026-W01", "2026-W35", "2026-W53", "2027-W01"]) {
    const mon = isoWeekStart(label);
    assert(mon && mon.getUTCDay() === 1, `${label}: week start is not a Monday`);
    assert(isoWeek(mon) === label, `${label} does not round-trip (got ${isoWeek(mon)})`);
  }
});

await check("gaps are reported, never invented", async () => {
  const hist = [{ week: "2026-W35" }, { week: "2026-W38" }];
  assert(JSON.stringify(missingWeeks(hist, "2026-W39")) === JSON.stringify(["2026-W36", "2026-W37"]),
    "gap detection is wrong");
  // The current week is not a gap — it is simply not written yet.
  assert(missingWeeks([{ week: "2026-W35" }], "2026-W36").length === 0, "current week counted as a gap");
  assert(missingWeeks([], "2026-W36").length === 0, "an empty archive is not a hole");
});

await check("the cron writes one row per week and never rewrites one", async () => {
  kv.clear();
  const monday = new Date("2026-08-31T00:05:00Z");
  const wk = isoWeek(monday);

  const r1 = await worker.scheduled({ scheduledTime: monday.getTime() }, env, {});
  const keys = () => [...kv.keys()].filter(k => k.startsWith("idx:week:"));
  assert(keys().length === 1, `cron wrote ${keys().length} keys, expected 1`);
  const written = kv.get(`idx:week:${wk}`);
  assert(written, "cron did not write the current week");

  // The rest of the week must be no-ops — that is what makes a daily cron safe.
  for (let d = 1; d < 7; d++) {
    const t = new Date(monday.getTime() + d * 86400000);
    await worker.scheduled({ scheduledTime: t.getTime() }, env, {});
  }
  assert(keys().length === 1, "a later day in the same week wrote a second row");
  assert(kv.get(`idx:week:${wk}`) === written, "the archived week was rewritten");

  // The row records which path wrote it. This is the whole point: on Monday the
  // question "did the cron fire, or did a crawler happen by?" is answered by the
  // data instead of by an argument from timing.
  assert(JSON.parse(written).src === "cron", "a cron-written row is not marked as such");

  // Next Monday is a new week and must be recorded.
  const nextMon = new Date(monday.getTime() + 7 * 86400000);
  await worker.scheduled({ scheduledTime: nextMon.getTime() }, env, {});
  assert(keys().length === 2, "a new week was not recorded");
  assert(kv.get(`idx:week:${isoWeek(nextMon)}`), "next week's key is missing");
});

await check("a week the cron missed shows on the page as a gap", async () => {
  kv.clear();
  // Build the history relative to the REAL current week, because that is what
  // the handler compares against — a fixture pinned to fixed dates would stop
  // meaning anything the moment the calendar moved past it.
  const nowWk = isoWeek(new Date());
  const back = (n) => isoWeek(new Date(isoWeekStart(nowWk).getTime() - n * 7 * 86400000));
  const twoAgo = back(2), oneAgo = back(1);
  const row = (week) => ({ week, date: "2026-01-01", models: 10, listings: 100,
                           priceMed: 8000, kmMed: 170000, sellMed: 29, depMed: 0.1 });
  // twoAgo recorded, oneAgo missed, and the current week will be written on load.
  kv.set("idx:weeks", JSON.stringify([row(twoAgo)]));
  kv.set(`idx:week:${twoAgo}`, JSON.stringify(row(twoAgo)));

  const page = await (await get("/mercado/indice")).text();
  assert(JSON.parse(kv.get(`idx:week:${nowWk}`)).src === "web",
    "a row written by a page request is not marked as such");
  assert(page.includes(oneAgo), `the missing week ${oneAgo} is not named on the page`);
  assert(/Faltam 1 semana no hist/.test(page), "the page does not admit the gap");
  // And the hole must not have been quietly filled with today's numbers.
  assert(!kv.has(`idx:week:${oneAgo}`), "a past week was backfilled with current data");
  // The current week, by contrast, is written on this very request.
  assert(kv.has(`idx:week:${nowWk}`), "the current week was not recorded");
});

await check("a closed month gets a permanent address, the open one does not", async () => {
  kv.clear();
  const nowWk = isoWeek(new Date());
  const at = (n) => isoWeek(new Date(isoWeekStart(nowWk).getTime() - n * 7 * 86400000));
  const hist = [];
  for (let i = 8; i >= 1; i--) {
    hist.push({ week: at(i), date: "2026-01-01", models: 10, listings: 100 + i,
                priceMed: 8000 + i, kmMed: 170000, sellMed: 29, depMed: 0.1 });
  }
  kv.set("idx:weeks", JSON.stringify(hist));
  const cuts = monthlyCuts(hist, nowWk);
  assert(cuts.length >= 1, "eight weeks back did not close a single month");

  for (const c of cuts) {
    const r = await get(`/mercado/indice/${c.month}`);
    assert(r.status === 200, `/mercado/indice/${c.month} → ${r.status}`);
    const body = await r.text();
    assert(body.includes(`https://${HOST}/mercado/indice/${c.month}`), `${c.month}: no permanent address on the page`);
    assert(body.includes(`<link rel="canonical" href="https://${HOST}/mercado/indice/${c.month}">`),
      `${c.month}: the month page is not its own canonical`);
  }
  const open = isoWeekMonth(nowWk);
  assert((await get(`/mercado/indice/${open}`)).status === 404, "served the month still in progress");

  const before = kv.get("idx:weeks");
  await get(`/mercado/indice/${cuts[0].month}`);
  await get("/mercado/indice");
  const after = JSON.parse(kv.get("idx:weeks"));
  for (const h of JSON.parse(before)) {
    const same = after.find(x => x.week === h.week);
    assert(same && JSON.stringify(same) === JSON.stringify(h), `week ${h.week} changed under its own URL`);
  }

  const xml = await (await get("/sitemap.xml")).text();
  const locs = [...xml.matchAll(/<loc>([^<]+)<\/loc>/g)].map(m => new URL(m[1]).pathname);
  const listed = locs.filter(p => /^\/mercado\/indice\/\d{4}-\d{2}$/.test(p)).sort();
  const expected = cuts.map(c => `/mercado/indice/${c.month}`).sort();
  assert(listed.join() === expected.join(), `sitemap months ${listed.join()} vs router ${expected.join()}`);
});

await check("no data means no row, not a row of nulls", async () => {
  kv.clear();
  const prevFetch = globalThis.fetch;
  globalThis.fetch = async (input, init) => {
    const u = typeof input === "string" ? input : input.url;
    if (u.includes("models.json")) return new Response("nope", { status: 500 });
    return prevFetch(input, init);
  };
  try {
    await worker.scheduled({ scheduledTime: Date.now() }, env, {});
    assert([...kv.keys()].filter(k => k.startsWith("idx:week:")).length === 0,
      "wrote a snapshot with no market data behind it");
  } finally { globalThis.fetch = prevFetch; }
});

await check("every icon the pages link to is served without the auth gate", async () => {
  const head = await (await worker.fetch(new Request(`https://${HOST}/`), env, {})).text();
  const linked = [...head.matchAll(/<link rel="(?:icon|apple-touch-icon)"[^>]*href="([^"]+)"/g)]
    .map(m => m[1]);
  assert(linked.length >= 2, "the pages stopped linking any icon at all");
  for (const href of linked) {
    const res = await worker.fetch(new Request(`https://${HOST}${href}`), env, {});
    assert(res.status === 200, `${href} answered ${res.status} — a linked icon must never 401 or 404`);
    const ct = res.headers.get("content-type") || "";
    assert(!ct.includes("text/html"),
      `${href} fell through to the SPA fallback and returned HTML instead of an image`);
  }
});

await check("seller lead POST stores the lead and answers with the thanks page", async () => {
  const stored = () => [...kv.keys()].filter(k => k.startsWith("lead:")).length;
  const before = stored();
  const post = (params, origin = `https://${HOST}`) => worker.fetch(new Request(`https://${HOST}/lead`, {
    method: "POST", body: params,
    headers: { Origin: origin, "Content-Type": "application/x-www-form-urlencoded" },
  }), env);
  const full = () => new URLSearchParams({ modelo: deep, nome_modelo: "Carro Teste", ano: "2016", km: "120000",
                                           distrito: "Porto", contacto: "912345678", nome: "Ana", consent: "1" });
  const r = await post(full());
  assert(r.status === 200, `POST /lead → ${r.status}`);
  const t = await r.text();
  assert(t.includes("Pedido recebido"), "thanks page missing");
  assert(t.includes("noindex"), "thanks page is indexable");
  assert(stored() === before + 1, "lead was not stored");
  const noConsent = await post(new URLSearchParams({ ano: "2016", contacto: "912345678" }));
  assert(noConsent.status === 400, `missing consent → ${noConsent.status}`);
  const badContact = await post(new URLSearchParams({ ano: "2016", contacto: "ola", consent: "1" }));
  assert(badContact.status === 400, `bad contact → ${badContact.status}`);
  const trap = await post(new URLSearchParams({ ano: "2016", contacto: "912345678", consent: "1", website: "http://spam" }));
  assert(trap.status === 200, `honeypot → ${trap.status}`);
  assert(stored() === before + 1, "honeypot submission was stored");
  const foreign = await post(full(), "https://evil.example");
  assert(foreign.status === 403, `cross-origin POST → ${foreign.status}`);
  const g = await get("/lead");
  assert(g.status === 302, `GET /lead → ${g.status}`);
  const admin = await get("/analytics/leads.json");
  assert(admin.status === 401, `leads.json without auth → ${admin.status}`);
  const robots = await (await get("/robots.txt")).text();
  assert(robots.includes("Disallow: /lead"), "robots does not block /lead");
});

await check("seller pages resolve, the hub links them and the sitemap lists them", async () => {
  const hub = await get("/vender");
  assert(hub.status === 200, `/vender → ${hub.status}`);
  const hubBody = await hub.text();
  const eligible = slugs.filter(s => venderOk(models[s]));
  assert(eligible.length > 0, "no eligible seller model in the fixture");
  const s = eligible[0];
  assert(hubBody.includes(`/vender/${s}`), "hub does not link an eligible model");
  const page = await get(`/vender/${s}`);
  assert(page.status === 200, `/vender/${s} → ${page.status}`);
  const j = await get(`/vender/${s}.json`);
  assert(j.status === 200 && (j.headers.get("content-type") || "").includes("json"), "seller JSON twin is not served");
  const miss = await get("/vender/modelo-que-nao-existe");
  assert(miss.status === 404, `unknown seller page → ${miss.status}`);
  const sm = await (await get("/sitemap.xml")).text();
  assert(sm.includes(`/vender/${s}<`) || sm.includes(`/vender/${s}</loc>`), "sitemap does not list the seller page");
  assert(sm.includes(`https://${HOST}/vender</loc>`), "sitemap does not list the seller hub");
});

await check("seller guides resolve and are listed in the sitemap", async () => {
  const hub = await get("/guias");
  assert(hub.status === 200, `/guias → ${hub.status}`);
  const first = GUIDES[0].slug;
  const page = await get(`/guias/${first}`);
  assert(page.status === 200, `/guias/${first} → ${page.status}`);
  const miss = await get("/guias/guia-que-nao-existe");
  assert(miss.status === 404, `unknown guide → ${miss.status}`);
  const sm = await (await get("/sitemap.xml")).text();
  for (const g of GUIDES) assert(sm.includes(`/guias/${g.slug}</loc>`), `sitemap does not list ${g.slug}`);
});

console.log(failures ? `\n${failures} check(s) FAILED` : "\nall route checks passed");
process.exit(failures ? 1 : 0);
