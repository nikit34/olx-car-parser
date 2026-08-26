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

import { readFileSync } from "node:fs";
import worker from "../../flipper-club/src/index.js";
import { yearPageYears, depreciationSlugs, comparePairs } from "../../flipper-club/src/seo-pages.js";

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
      return new Response("<html>spa fallback</html>", { status: 200, headers: { "content-type": "text/html" } });
    },
  },
};

// getModels/getDeals go through global fetch; serve the blob from memory and
// give the deals feed an empty-but-valid answer so the bridges are exercised.
const realFetch = globalThis.fetch;
globalThis.fetch = async (input, init) => {
  const u = typeof input === "string" ? input : input.url;
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

await check("a year below the publishing floor 404s", async () => {
  const rec = models[deep];
  const published = new Set(yearPageYears(rec));
  const thin = (rec.yr || []).find(c => typeof c.y === "number" && !published.has(c.y));
  if (!thin) return;                       // this model has no thin years — fine
  const r = await get(`/preco/${deep}/${thin.y}`);
  assert(r.status === 404, `thin year ${thin.y} → ${r.status}, should be 404`);
});

await check("nonsense under /preco 404s instead of 500ing", async () => {
  for (const p of [`/preco/nao-existe`, `/preco/${deep}/abc`, `/preco/${deep}/2014/extra`, `/preco/%`, `/preco/${deep}/9999`]) {
    const r = await get(p);
    assert(r.status === 404, `${p} → ${r.status}`);
  }
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

await check("generated pages outside the published set 404", async () => {
  const notDep = slugs.find(s => !depreciationSlugs(models).includes(s));
  assert((await get(`/depreciacao/${notDep}`)).status === 404, "served a depreciation page with no curve");
  assert((await get("/comparar/volkswagen-golf-vs-volkswagen-golf")).status === 404, "served a self-comparison");
  assert((await get("/mercado/indice/1999-w03")).status === 404, "served an index week we never recorded");
  assert((await get("/mercado/indice/lixo")).status === 404, "served a malformed index week");
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
  const llms = await (await get("/llms.txt")).text();
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
  if (models[deep].fx || models[deep].dt) return;   // data has landed; skip
  for (const p of [`/preco/${deep}/diesel`, `/preco/${deep}/porto`, "/precos/porto"]) {
    const r = await get(p);
    assert(r.status === 404, `${p} → ${r.status}, must 404 before the data exists`);
  }
});

await check("facet pages appear when the blob carries the cells", async () => {
  const augmented = JSON.parse(JSON.stringify(mdoc));
  augmented.built_at = "2026-08-26T00:00:00Z";        // bust the corpus-stats memo
  const rec = augmented.models[deep];
  rec.fx = [
    { k: "diesel", lbl: "Diesel", n: 40, fl: 5000, fm: 7000, fh: 9000, km: 190000, y0: 2010, y1: 2018 },
    { k: "gasolina", lbl: "Gasolina", n: 18, fl: 4200, fm: 5800, fh: 7600, km: 150000, y0: 2008, y1: 2016 },
  ];
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
    for (const p of [`/preco/${deep}/diesel`, `/preco/${deep}/gasolina`, `/preco/${deep}/porto`, "/precos/porto"]) {
      const r = await get(p);
      assert(r.status === 200, `${p} → ${r.status}`);
      const body = await r.text();
      assert(body.includes('<link rel="canonical" href="https://carsbuyer.org' + p + '">'),
        `${p}: canonical is not self`);
      assert(body.includes("index,follow"), `${p}: not indexable`);
      assert(!body.replace(/<script[\s\S]*?<\/script>/g, "").includes("undefined"),
        `${p}: rendered "undefined"`);
    }
    // The comparison that justifies the page existing at all.
    const diesel = await (await get(`/preco/${deep}/diesel`)).text();
    assert(diesel.includes(`/preco/${deep}/gasolina`), "fuel facet does not link its sibling");
    assert(/2[01]% (mais caro|mais barato)/.test(diesel), "fuel facet states no price gap");

    const unknown = await get(`/preco/${deep}/nao-existe`);
    assert(unknown.status === 404, `unknown facet → ${unknown.status}`);
    assert((await get("/precos/nao-existe")).status === 404, "served a district we have no data for");

    const xml = await (await get("/sitemap.xml")).text();
    for (const want of [`/preco/${deep}/diesel`, `/preco/${deep}/porto`, "/precos/porto"]) {
      assert(xml.includes(`<loc>https://carsbuyer.org${want}</loc>`), `sitemap missing ${want}`);
    }
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

console.log(failures ? `\n${failures} check(s) FAILED` : "\nall route checks passed");
process.exit(failures ? 1 : 0);
