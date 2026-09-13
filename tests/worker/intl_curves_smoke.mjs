import { readFileSync } from "node:fs";
import worker from "../../flipper-club/src/index.js";
import "../../flipper-club/src/intl-curves.js";
import {
  depreciationSlugs, depreciationOk, depreciationFit, depreciationAge,
  duel, duelSlugs, duelOk,
} from "../../flipper-club/src/seo-pages.js";
import { LOCALES, href, t } from "../../flipper-club/src/i18n.js";
import { intlCurvesModule, duelCopy, depreciationPath, duelPath } from "../../flipper-club/src/intl-curves.js";

const HOST = "carsbuyer.org";

let failures = 0;
function assert(cond, msg) { if (!cond) throw new Error(msg); }
async function check(name, fn) {
  try { await fn(); console.log(`  ok   ${name}`); }
  catch (err) { failures++; console.error(`  FAIL ${name}\n       ${err && err.message}`); }
}

const dePath = process.argv[2] || "tests/worker/fixtures/models_de.json";
const ddoc = JSON.parse(readFileSync(dePath, "utf8"));
const models = ddoc.models;
const builtAt = ddoc.built_at;

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

function h1Of(html) {
  const m = /<h1[^>]*>([\s\S]*?)<\/h1>/.exec(html);
  return m ? visibleText(m[1]).trim() : null;
}

function ldBlocks(html) {
  return [...html.matchAll(/<script type="application\/ld\+json">([\s\S]*?)<\/script>/g)]
    .map(m => JSON.parse(m[1].replace(/\\u003c/g, "<")));
}

const LEAKS = {
  de: ["anúncio", "preço", "usados", "quilometragem", "OLX", "desvalorização",
       "annonces", "occasion", "kilométrage", "médiane", "décote",
       "annunci", "prezzo", "chilometraggio", "svalutazione"],
  fr: ["anúncio", "preço", "usados", "quilometragem", "OLX", "desvalorização",
       "Angebote", "Kilometerstand", "Erstzulassung", "Wertverlust",
       "annunci", "prezzo", "chilometraggio", "svalutazione"],
  it: ["anúncio", "preço", "usados", "quilometragem", "OLX", "desvalorização",
       "Angebote", "Kilometerstand", "Erstzulassung", "Wertverlust",
       "annonces", "occasion", "kilométrage", "décote"],
};

function assertNoLeak(code, html, where) {
  const seen = visibleText(html);
  for (const w of LEAKS[code]) {
    assert(!seen.toLowerCase().includes(w.toLowerCase()),
      `${where} leaks the foreign word "${w}"`);
  }
  assert(!/undefined|\[object Object\]|NaN(?![a-zA-Z])/.test(
    html.replace(/<script[\s\S]*?<\/script>/g, "")), `${where} rendered undefined/NaN`);
  const stray = /\{[a-z][a-z0-9_]*\}/.exec(visibleText(html));
  assert(!stray, `${where} rendered the unsubstituted placeholder ${stray && stray[0]}`);
}

async function assertPage(code, path, where) {
  const res = await get(path);
  assert(res.status === 200, `${where} is ${res.status}, want 200`);
  const html = await res.text();
  const canonical = canonicalOf(html);
  assert(canonical === `https://${HOST}${path}`, `${where} canonical is ${canonical}`);
  const desc = metaContent(html, "description");
  assert(desc && desc.length > 60, `${where} description is ${desc && desc.length} chars`);
  const h1 = h1Of(html);
  assert(h1 && h1.length > 10, `${where} has no usable h1`);
  assert(!/<meta name="robots" content="[^"]*noindex/.test(html), `${where} is noindex`);
  const lds = ldBlocks(html);
  assert(lds.length >= 1, `${where} has no JSON-LD`);
  const nodes = lds.flatMap(b => b["@graph"] || [b]);
  assert(nodes.some(n => n["@type"] === "BreadcrumbList"), `${where} has no BreadcrumbList`);
  assertNoLeak(code, html, where);
  return html;
}

async function assertTwin(path, where, fields) {
  const res = await get(`${path}.json`);
  assert(res.status === 200, `${where}.json is ${res.status}`);
  assert((res.headers.get("content-type") || "").includes("json"),
    `${where}.json is not served as json`);
  const doc = await res.json();
  assert(doc.source_url === `https://${HOST}${path}`,
    `${where}.json source_url is ${doc.source_url}`);
  for (const f of fields) {
    assert(doc[f] !== undefined, `${where}.json has no ${f}`);
  }
  return doc;
}

const depSlugs = depreciationSlugs(models);
const fuelSlugs = duelSlugs(models, "fuel", builtAt);
const gearSlugs = duelSlugs(models, "gear", builtAt);
const de = LOCALES.de, fr = LOCALES.fr, it = LOCALES.it;

console.log(`intl curves: ${depSlugs.length} depreciation, ${fuelSlugs.length} fuel duels, ${gearSlugs.length} gear duels`);

await check("the depreciation hub renders, in German", async () => {
  const html = await assertPage("de", href(de, "depreciation"), "de depreciation hub");
  assert(html.includes("Wertverlust"), "hub does not say Wertverlust");
  assert(/Modell/.test(visibleText(html)), "hub table has no Modell column");
});

await check("every published depreciation page renders", async () => {
  for (const slug of depSlugs) {
    await assertPage("de", depreciationPath(de, slug), `de depreciation ${slug}`);
  }
});

await check("a depreciation page claims only what its curve measured", async () => {
  for (const slug of depSlugs.slice(0, 8)) {
    const rec = models[slug];
    const fit = depreciationFit(rec);
    const av = depreciationAge(rec, fit, builtAt);
    const html = await body(depreciationPath(de, slug));
    const seen = visibleText(html);
    if (av.bend) {
      assert(seen.includes(`${av.bend.age} Jahren`) || seen.includes(`${av.bend.age} Jahre`),
        `${slug} measured a bend at ${av.bend.age} and does not say so`);
    } else {
      assert(/Nein, nicht in Prozent/.test(seen),
        `${slug} has no measured bend but does not refuse the claim`);
    }
    if (!av.halfLife) {
      assert(!/Hälfte seines Werts alle/.test(seen), `${slug} claims a half-life it has none of`);
    }
  }
});

await check("the depreciation JSON twin carries the fit", async () => {
  const slug = depSlugs[0];
  const doc = await assertTwin(depreciationPath(de, slug), `de depreciation ${slug}`,
    ["annual_depreciation_rate", "fit_r2", "by_model_year", "year_of_age_cost", "bend"]);
  const fit = depreciationFit(models[slug]);
  assert(Math.abs(doc.annual_depreciation_rate - fit.rate) < 0.002,
    `twin rate ${doc.annual_depreciation_rate} != fit ${fit.rate}`);
  assert(doc.language === "de" && doc.market === "DE", "twin is not the German market");
  assert(doc.by_model_year.length === fit.cells.length, "twin drops year cells");
});

await check("the depreciation hub has a JSON twin of its own", async () => {
  const doc = await assertTwin(href(de, "depreciation"), "de depreciation hub", ["models"]);
  assert(doc.models.length === depSlugs.length, "hub twin lists a different model set");
  assert(doc.models.every(m => m.url.startsWith(`https://${HOST}/de/wertverlust/`)),
    "hub twin links somewhere else");
});

await check("both duel hubs render, in German", async () => {
  for (const kind of ["fuel", "gear"]) {
    const html = await assertPage("de", href(de, kind === "fuel" ? "duel_fuel" : "duel_gear"),
      `de ${kind} duel hub`);
    const C = duelCopy(de, kind);
    assert(html.includes(C.a.lbl) && html.includes(C.b.lbl), `${kind} hub misses a side`);
  }
});

await check("every published duel page renders", async () => {
  for (const [kind, slugs] of [["fuel", fuelSlugs], ["gear", gearSlugs]]) {
    for (const slug of slugs) {
      await assertPage("de", duelPath(de, kind, slug), `de ${kind} duel ${slug}`);
    }
  }
});

await check("a wide interval refuses to name a winner", async () => {
  let seenDraw = 0;
  for (const [kind, slugs] of [["fuel", fuelSlugs], ["gear", gearSlugs]]) {
    const C = duelCopy(de, kind);
    for (const slug of slugs) {
      const av = duel(models[slug], kind, builtAt);
      const seen = visibleText(await body(duelPath(de, kind, slug)));
      if (av.decisive) {
        const fav = C[av.winner].fav;
        assert(seen.includes(fav), `${slug} (${kind}) is decisive but never says "${fav}"`);
      } else {
        seenDraw++;
        assert(!seen.includes(C.a.fav) && !seen.includes(C.b.fav),
          `${slug} (${kind}) picks a side the interval cannot support`);
        assert(seen.includes("nicht unterscheidbar") || seen.includes("Messgenauigkeit"),
          `${slug} (${kind}) does not say the gap is inside the measurement`);
        assert(!/wir wissen es nicht/i.test(seen.replace("„wir wissen es nicht“", "")),
          `${slug} (${kind}) turns a draw into "we could not see one"`);
      }
    }
  }
  assert(seenDraw > 0, "fixture has no draw to test");
});

await check("the duel JSON twin carries the interval, not just the winner", async () => {
  const slug = fuelSlugs[0];
  const doc = await assertTwin(duelPath(de, "fuel", slug), `de fuel duel ${slug}`,
    ["rate_difference_pp_per_year", "rate_difference_ci95_half_width_pp",
     "distinguishable_at_95", "diesel", "gasoline"]);
  const av = duel(models[slug], "fuel", builtAt);
  assert(doc.distinguishable_at_95 === av.decisive, "twin disagrees with the significance test");
  assert(doc.holds_value_better === (av.decisive ? (av.winner === "a" ? "diesel" : "gasoline") : null),
    "twin names a winner the page does not");
  assert(doc.diesel.sample_size === av.a.n, "twin sample size is wrong");
});

await check("both duel hubs have a JSON twin", async () => {
  for (const [kind, routeKey, aJson] of [["fuel", "duel_fuel", "diesel"], ["gear", "duel_gear", "manual"]]) {
    const doc = await assertTwin(href(de, routeKey), `de ${kind} hub`, ["models", "holds_value_better_count"]);
    assert(doc.sides.includes(aJson), `${kind} hub twin has no ${aJson} side`);
    const rows = kind === "fuel" ? fuelSlugs : gearSlugs;
    assert(doc.models.length === rows.length, `${kind} hub twin lists a different model set`);
  }
});

await check("a model without the data 404s instead of rendering an empty shell", async () => {
  const noGear = Object.keys(models).find(s => !duelOk(models[s], "gear"));
  assert(noGear, "fixture has no model without a gearbox duel");
  const res = await get(duelPath(de, "gear", noGear));
  assert(res.status === 404, `gear duel for ${noGear} is ${res.status}, want 404`);
  const html = await res.text();
  assertNoLeak("de", html, "de gear-duel 404");
  const missing = await get(depreciationPath(de, "kein-solches-modell"));
  assert(missing.status === 404, `unknown depreciation slug is ${missing.status}`);
  const noFuel = Object.keys(models).find(s => !duelOk(models[s], "fuel"));
  if (noFuel) {
    const r = await get(duelPath(de, "fuel", noFuel));
    assert(r.status === 404, `fuel duel for ${noFuel} is ${r.status}, want 404`);
  }
  const deep = await get(`${depreciationPath(de, depSlugs[0])}/2018`);
  assert(deep.status === 404, `a deeper depreciation path is ${deep.status}, want 404`);
});

await check("a JSON twin 404s where its page does", async () => {
  const res = await get(`${depreciationPath(de, "kein-solches-modell")}.json`);
  assert(res.status === 404, `unknown depreciation twin is ${res.status}`);
  const noGear = Object.keys(models).find(s => !duelOk(models[s], "gear"));
  const r2 = await get(`${duelPath(de, "gear", noGear)}.json`);
  assert(r2.status === 404, `gear twin for ${noGear} is ${r2.status}`);
});

await check("the sitemap advertises exactly what the router serves", async () => {
  const res = await get("/de/sitemap.xml");
  assert(res.status === 200, `de sitemap is ${res.status}`);
  const xml = await res.text();
  const advertised = intlCurvesModule.sitemap(de, models, builtAt).map(e => e.path);
  assert(advertised.length === depSlugs.length + fuelSlugs.length + gearSlugs.length + 3,
    `module advertises ${advertised.length} paths`);
  for (const p of advertised) {
    assert(xml.includes(`<loc>https://${HOST}${p}</loc>`), `sitemap.xml is missing ${p}`);
    const r = await get(p);
    assert(r.status === 200, `advertised ${p} answers ${r.status}`);
  }
  const mine = [...xml.matchAll(/<loc>https:\/\/[^<]*?(\/de\/(?:wertverlust|diesel-oder-benziner|schaltung-oder-automatik)[^<]*)<\/loc>/g)]
    .map(m => m[1]);
  assert(mine.length === advertised.length,
    `sitemap carries ${mine.length} of my paths, module advertises ${advertised.length}`);
});

await check("French and Italian speak their own language", async () => {
  for (const [code, loc] of [["fr", fr], ["it", it]]) {
    await assertPage(code, href(loc, "depreciation"), `${code} depreciation hub`);
    await assertPage(code, depreciationPath(loc, depSlugs[0]), `${code} depreciation page`);
    await assertPage(code, href(loc, "duel_fuel"), `${code} fuel duel hub`);
    await assertPage(code, duelPath(loc, "fuel", fuelSlugs[0]), `${code} fuel duel page`);
    await assertPage(code, duelPath(loc, "gear", gearSlugs[0]), `${code} gear duel page`);
    await assertTwin(depreciationPath(loc, depSlugs[0]), `${code} depreciation twin`,
      ["annual_depreciation_rate"]);
  }
  const frHtml = await body(depreciationPath(fr, depSlugs[0]));
  assert(frHtml.includes("décote") || frHtml.includes("Décote"), "fr page never says décote");
  const itHtml = await body(depreciationPath(it, depSlugs[0]));
  assert(itHtml.includes("svalutazione") || itHtml.includes("Svalutazione"),
    "it page never says svalutazione");
});

await check("localised paths differ per locale and never collide", async () => {
  const paths = new Set();
  for (const loc of [de, fr, it]) {
    for (const key of ["depreciation", "duel_fuel", "duel_gear"]) {
      const p = href(loc, key);
      assert(!paths.has(p), `path ${p} is claimed twice`);
      paths.add(p);
    }
  }
  assert(href(de, "depreciation") === "/de/wertverlust", "de segment changed");
  assert(href(fr, "depreciation") === "/fr/decote", "fr segment changed");
  assert(href(it, "depreciation") === "/it/svalutazione", "it segment changed");
  assert(href(de, "duel_gear") === "/de/schaltung-oder-automatik", "de gear segment changed");
  assert(href(fr, "duel_fuel") === "/fr/diesel-ou-essence", "fr fuel segment changed");
  assert(href(it, "duel_gear") === "/it/manuale-o-automatico", "it gear segment changed");
});

await check("with INTL_LOCALES empty none of it exists", async () => {
  for (const p of [href(de, "depreciation"), depreciationPath(de, depSlugs[0]),
                   href(de, "duel_fuel"), duelPath(de, "fuel", fuelSlugs[0])]) {
    const res = await get(p, envOff);
    assert(res.status === 404, `${p} answers ${res.status} with no locales live`);
  }
});

await check("the Portuguese root is untouched", async () => {
  const res = await get("/depreciacao", envOff);
  assert(res.status === 200, `/depreciacao is ${res.status}`);
  const html = await res.text();
  assert(html.includes("Desvalorização"), "/depreciacao stopped speaking Portuguese");
  assert(!html.includes("Wertverlust"), "/depreciacao leaks German");
  const one = await get(`/depreciacao/${depSlugs[0]}`, envOff);
  assert(one.status === 200, `/depreciacao/${depSlugs[0]} is ${one.status}`);
  const off = await get("/wertverlust", envOff);
  assert(off.status === 404, "a locale segment answers at the root");
});

await check("the footer of a locale page links the new family", async () => {
  const html = await body(href(de, "hub"));
  assert(html.includes(`href="/de/wertverlust"`), "German footer has no depreciation link");
  const pt = await body("/precos", envOff);
  assert(!pt.includes("/de/wertverlust"), "the Portuguese footer picked up a locale link");
});

if (failures) {
  console.error(`\n${failures} check(s) failed`);
  process.exit(1);
}
console.log("\nall intl curves checks passed");
