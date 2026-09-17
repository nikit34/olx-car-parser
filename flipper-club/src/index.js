// Cloudflare Worker entry — public car-valuation site over live OLX listings.
//
// Product model (no auth, no accounts, nothing to pay for):
//   GET  /pt                   → one car at a time (top-ranked by decision_score).
//                                Photos, specs, signals and the seller's own OLX
//                                link are all public.
//   POST /pt/lead              → a seller asks professional buyers for offers.
//   GET  /healthz              → unauthenticated liveness.
//
// Public valuation surface (indexable; seo-pages.js renders it):
//   /pt/precos                    every model
//   /pt/preco/{slug}              one model, by year          (+ .json)
//   /pt/preco/{slug}/{ano}        one model in one year       (+ .json)
//   /pt/preco/{slug}/{facet}      one model, one fuel/district
//   /pt/precos/{distrito}         the market in one district
//   /pt/depreciacao[/{slug}]      how fast a model loses value
//   /pt/comparar[/{a}-vs-{b}]     two models side by side
//   /pt/liquidez                  how long each model takes to sell
//   /pt/sobrevalorizados          asking price vs. our estimate
//   /pt/mercado/indice[/{semana|mes}]  weekly market index + permanent weekly
//                                  and monthly archive
//   /pt/metodologia /pt/sobre /pt/isv  how the numbers are made, by whom, and ISV
//
// Every one of those exists only where its sample clears a floor, and both the
// router and sitemap.xml decide that with the SAME functions in seo-pages.js.
// Anything else is a real 404 — never the Basic-Auth 401 the asset gate used to
// answer for every unknown path.
//
// Internal-only (NOT the product):
//   GET  /analytics/*  and  /files/* /data/* …  → the stlite analytics bundle
//   and its raw data assets. Gated by HTTP Basic Auth against the Worker
//   secrets ANALYTICS_USER / ANALYTICS_PASS. Fail-closed: if those secrets are
//   unset, access is denied — raw parquets/model internals never go public.
//
// Data source: getDeals() fetches hot_deals_{zone}.json from the latest-data
// GitHub Release and caches it in KV for 15 min. A missing/broken feed renders a
// degraded banner (no fake data).

import {
  renderGrid, renderCarPage, renderInfo,
  renderLanding,
  renderAvaliar, renderModelPage, renderModelsHub, renderModelWidget, slugify,
  setAnalyticsId,
  renderPrivacy, renderLeadThanks,
} from "./templates.js";
import {
  renderYearPage, renderNotFound, renderDepreciationPage, renderDepreciationHub,
  renderComparePage, renderCompareHub, renderLiquidityHub, renderValuationGap,
  renderMarketIndex, renderMethodology, renderAbout, renderIsv,
  setSiteIdentity, corpusStats, modelInsights, provenance,
  yearCells, yearCell, yearPageYears, depreciationOk, depreciationFit, depreciationSlugs,
  comparePairs, parseComparePath, comparePairKey, comparePriceGap, modelClass,
  modelJson, yearJson,
  depreciationAge, depreciationJson,
  renderFacetPage, renderDistrictPage, facetCell, facetKind, facetJson, publishedCells, retiredFacetKind,
  renderLiquidityPage, liquidityJson, publishedLiquidity, setLiqWave, liqWaveSlugs,
  renderVenderPage, renderVenderHub, venderJson, publishedVender, setVenderWave,
  renderImportPage, renderImportHub, importJson, importOk, importSlugs,
  isoWeek, missingWeeks, monthlyCuts, renderMarketMonth, renderArchiveHub, breadcrumbLd,
  setWave, waveSlugs, publishedYearPages, publishedDepreciation, publishedPairs, publishedFacets,
  DUELS, duel, duelByPath, duelJson, duelSlugs, duelsFor, publishedDuel,
  renderDuelPage, renderDuelHub,
} from "./seo-pages.js";
import { GUIDES, GUIDES_UPDATED, guideBySlug, guideBlock, renderGuide, renderGuidesHub } from "./guides.js";
import {
  setIntlLocales, intlLocales, liveLocales, localeForPath, href as ihref,
  parseAs24Id, valuationKey, t as it, setSlugMarkets,
  registerIntlPageModule, intlPageModules, setNavLive,
} from "./i18n.js";
import {
  renderIntlLanding, renderIntlHub, renderIntlModelPage, renderIntlYearPage,
  renderIntlAvaliar, renderIntlFeed, renderIntlMethodology, renderIntlAbout,
  renderIntlPrivacy, renderIntlNotFound, renderIntlInfo, renderIntlCar,
  renderIntlArchive, renderIntlMarketIndex,
  intlModelJson, intlYearJson, intlSitemapPaths, intlYearCell, intlSiblings,
  setIntlWave, intlPublishedYears, intlInWave,
} from "./pages-intl.js";
import { intlModelCutLinks } from "./intl-facets.js";
import { intlModelCurveLinks } from "./intl-curves.js";
import "./intl-compare.js";
import "./intl-guides.js";

const ZONES = ["norte", "centro", "sul", "all"];

const ICON_PATHS = new Set([
  "/favicon.ico", "/icon-96.png", "/icon-192.png", "/apple-touch-icon.png",
]);

const PRODUCT_PATHS = new Set([
  "/", "/mercado", "/car", "/avaliar", "/lead", "/ir/historico",
  "/precos", "/historico",
  // Приватность обязана быть здесь: гейт стоит ВЫШЕ её обработчика, и без
  // записи в этом списке Basic-Auth отдавал бы 401 и Googlebot, и человеку,
  // пришедшему по ссылке из баннера согласия. (Неизвестные пути теперь отдают
  // настоящую 404, но известный роут всё равно обязан быть в этом списке.)
  "/privacidade",
  // Second-layer SEO pages (seo-pages.js). Hubs are exact paths; their per-item
  // children (/pt/depreciacao/{slug}, /pt/comparar/{a}-vs-{b}, /pt/mercado/indice/{week|month})
  // are prefix-routed above the asset gate.
  "/depreciacao", "/comparar", "/liquidez", "/sobrevalorizados", "/importar", "/vender", "/guias",
  "/metodologia", "/sobre", "/isv",
  ...Object.values(DUELS).map(d => `/${d.path}`),
]);

// Paths that belong to the internal analytics bundle rather than the product.
//
// This list is what makes a real 404 possible. Before, ANY unknown path fell
// into the Basic-Auth asset gate and answered 401 — /pt/sobre, /faq, /blog, a
// mistyped link, a stray trailing slash, all of them. Googlebot reads 401 as
// "there is something here you may not have", so those URLs sat in Search
// Console as site-wide access errors and kept getting re-crawled instead of
// being dropped.
//
// It cannot be replaced by "ask ASSETS and 404 if it misses": the bucket is
// configured single-page-application, so a miss returns index.html with a 200.
// The set has to be explicit.
const INTERNAL_ASSET_PREFIXES = ["/files/", "/data/"];
const INTERNAL_ASSET_EXACT = new Set(["/index.html", "/README.md"]);

// Paths whose case and trailing slash are NOT ours to normalise: Streamlit's
// multipage nav generates /analytics/Market_Direction, and asset filenames are
// case-sensitive. Lower-casing those would 404 the dashboard.
const NO_NORMALISE = ["/analytics", "/files/", "/data/", "/fonts/", "/_olx", "/webhook/"];

function isInternalAsset(pathname) {
  return INTERNAL_ASSET_EXACT.has(pathname)
    || INTERNAL_ASSET_PREFIXES.some(pre => pathname.startsWith(pre));
}

const PT_PREFIX = "/pt";

const PT_CHILD_PREFIXES = ["/preco/", "/precos/", "/depreciacao/", "/comparar/", "/liquidez/",
  "/vender/", "/historico/", "/guias/", "/importar/", "/mercado/", "/widget/preco/", "/ir/",
  ...Object.values(DUELS).map(d => `/${d.path}/`)];

function ptRest(pathname) {
  if (pathname === PT_PREFIX) return "/";
  if (pathname.startsWith(`${PT_PREFIX}/`)) return pathname.slice(PT_PREFIX.length);
  return null;
}

function ptLegacyPath(pathname) {
  if (pathname === "/") return PT_PREFIX;
  if (PRODUCT_PATHS.has(pathname) || PT_CHILD_PREFIXES.some(pre => pathname.startsWith(pre))) {
    return `${PT_PREFIX}${pathname}`;
  }
  return null;
}

function canonicalPath(pathname) {
  if (NO_NORMALISE.some(pre => pathname.startsWith(pre))) return pathname;
  let norm = pathname;
  if (norm.length > 1 && norm.endsWith("/")) norm = norm.replace(/\/+$/, "") || "/";
  if (/[A-Z]/.test(norm)) norm = norm.toLowerCase();
  return ptLegacyPath(norm) || norm;
}

async function handlePt(request, env, url, pathname, method) {
  const u = new URL(url);
  u.pathname = pathname;
  // Per-model SEO pages — prefix route, BEFORE the asset gate.
  //   /pt/preco/{slug}            model
  //   /pt/preco/{slug}.json       model, machine-readable
  //   /pt/preco/{slug}/{ano}      one model year
  //   /pt/preco/{slug}/{ano}.json one model year, machine-readable
  if (pathname.startsWith("/preco/") && method === "GET") {
    return handleModelPage(request, env, u);
  }
  // District pages (/pt/precos/{distrito}). Prefix route; the bare /pt/precos hub
  // stays in PRODUCT_PATHS below.
  if (pathname.startsWith("/precos/") && method === "GET") {
    return handleDistrict(request, env, u);
  }
  // Depreciation curve + its hub.
  if (pathname.startsWith("/depreciacao/") && method === "GET") {
    return handleDepreciation(request, env, u);
  }
  // Head-to-head comparisons + their hub.
  if (pathname.startsWith("/comparar/") && method === "GET") {
    return handleCompare(request, env, u);
  }
  if (pathname.startsWith("/liquidez/") && method === "GET") {
    return handleLiquidityPage(request, env, u);
  }
  if (pathname.startsWith("/vender/") && method === "GET") {
    return handleVenderPage(request, env, u);
  }
  if (pathname.startsWith("/historico/") && method === "GET") {
    return handleArchive(request, env, u);
  }
  if (pathname.startsWith("/guias/") && method === "GET") {
    return handleGuide(request, env, u);
  }
  if (pathname.startsWith("/importar/") && method === "GET") {
    return handleImportPage(request, env, u);
  }
  const duelPrefix = Object.values(DUELS).find(d => pathname.startsWith(`/${d.path}/`));
  if (duelPrefix && method === "GET") {
    return handleDuel(request, env, u, duelPrefix);
  }
  // Market index archive (/pt/mercado/indice[/{YYYY-Www}|/{YYYY-MM}]).
  if (pathname === "/mercado/indice" || pathname.startsWith("/mercado/indice/")) {
    if (method !== "GET") return notFound();
    return handleMarketIndex(request, env, u);
  }
  // Embeddable valuation widget (/pt/widget/preco/{slug}) — public, iframe-able,
  // cached, cookie-less. Also a prefix route before the asset gate.
  if (pathname.startsWith("/widget/preco/") && method === "GET") {
    return handleModelWidget(request, env, u);
  }
  if (pathname === "/sitemap.xml" && method === "GET") return handleSitemap(request, env, u);

  // Internal assets keep the Basic-Auth gate. Everything else that is not a
  // product route is genuinely not here → real 404 (see notFoundPage).
  if (!PRODUCT_PATHS.has(pathname)) return notFoundPage(request, env, url);

  if (pathname === "/" && method === "GET") return handleLanding(request, env, u);
  if (pathname === "/depreciacao" && method === "GET") return handleDepreciationHub(request, env, u);
  if (pathname === "/comparar" && method === "GET") return handleCompareHub(request, env, u);
  const duelHub = Object.values(DUELS).find(d => pathname === `/${d.path}`);
  if (duelHub && method === "GET") return handleDuelHub(request, env, u, duelHub);
  if (pathname === "/liquidez" && method === "GET") return handleLiquidity(request, env, u);
  if (pathname === "/vender" && method === "GET") return handleVenderHub(request, env, u);
  if (pathname === "/guias" && method === "GET") return handleGuidesHub(request, env, u);
  if (pathname === "/historico" && method === "GET") return handleArchiveHub(request, env, u);
  if (pathname === "/sobrevalorizados" && method === "GET") return handleValuationGap(request, env, u);
  if (pathname === "/importar" && method === "GET") return handleImportHub(request, env, u);
  if (pathname === "/metodologia" && method === "GET") return handleMethodology(request, env, u);
  if (pathname === "/sobre" && method === "GET") return handleAbout(request, env, u);
  if (pathname === "/isv" && method === "GET") return handleIsv(request, env, u);
  if (pathname === "/precos" && method === "GET") return handleModelsHub(request, env, u);
  if (pathname === "/avaliar" && method === "GET") return handleAvaliar(request, env, u);
  if (pathname === "/mercado" && method === "GET") return handleFeed(request, env, u);
  if (pathname === "/car" && method === "GET") return handleCar(request, env, u);
  if (pathname === "/lead" && method === "POST") return handleLead(request, env, u);
  if (pathname === "/lead" && method === "GET") return redirect("/pt/avaliar#escolher", 302);
  if (pathname === "/ir/historico" && method === "GET") return handleHistoryRedirect(request, env, u);
  // Страница приватности публичная и индексируемая: на неё ссылается баннер
  // согласия, и за Basic-Auth она отдавала бы 401 и Googlebot, и человеку.
  if (pathname === "/privacidade" && method === "GET") {
    return publicHtml(renderPrivacy({ depositCount: null, host: u.host, contact: env.SITE_CONTACT_EMAIL || null }));
  }

  // A known path reached with a method it does not answer.
  return notFoundPage(request, env, url);
}

const worker = {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    const { pathname } = url;
    const method = request.method;

    if (method === "HEAD") {
      const res = await worker.fetch(
        new Request(url.toString(), { method: "GET", headers: request.headers }), env, ctx);
      return new Response(null, { status: res.status, statusText: res.statusText, headers: res.headers });
    }

    // Measurement ID для GA4. Ставится до любой ветки, потому что сниппет
    // рендерится в общей обёртке страниц. Пусто = аналитики нет вообще.
    setAnalyticsId(env.GA4_MEASUREMENT_ID);

    try {
      if (pathname === "/healthz") {
        // С флагом verbose отдаём только булевы признаки настройки, без самих
        // значений. Иначе опечатку в имени секрета или обрезанную вставку не
        // отличить от рабочей настройки до первой оплаты, а серверный purchase
        // в этом случае просто молчит. Тело без флага оставлено прежним: на
        // него может смотреть внешний монитор. Ответ не кешируется - кеш здесь
        // означал бы устаревшую правду о настройках.
        if (url.searchParams.get("verbose") === "1") {
          return new Response(JSON.stringify({
            ok: true,
            ga4_tag: Boolean((env.GA4_MEASUREMENT_ID || "").trim()),
            ga4_mp_secret: Boolean((env.GA4_API_SECRET || "").trim()),
          }, null, 2), {
            status: 200,
            headers: {
              "content-type": "application/json; charset=utf-8",
              "cache-control": "no-store",
            },
          });
        }
        return new Response("ok", { status: 200 });
      }

      // Scraper egress relay. Must sit here, above both the canonical-host
      // redirect and the PRODUCT_PATHS asset gate: it is neither a product
      // page nor an internal asset, and gating it behind Basic-Auth (which is
      // what the fallthrough does) makes it answer 401 to its only caller.
      if (pathname === "/_olx" && method === "GET") return handleOlxRelay(request, env, url);

      // Canonical host — the product answers on both carsbuyer.org and the
      // workers.dev subdomain, and identical content on two hostnames splits
      // ranking signals. Send everything to CANONICAL_HOST, preserving path and
      // query. Deliberately placed AFTER /healthz, which must answer 200 on
      // every hostname a monitor may be pinned to rather than a redirect.
      // Dormant until CANONICAL_HOST is set (see wrangler.toml [vars]).
      const canonicalHost = (env.CANONICAL_HOST || "").trim();
      let visitorScheme = null;
      try { visitorScheme = JSON.parse(request.headers.get("cf-visitor") || "{}").scheme; } catch (_) { visitorScheme = null; }
      const insecure = url.protocol === "http:" || visitorScheme === "http";
      if (canonicalHost && !isLocalHost(url.hostname)
          && (url.hostname !== canonicalHost || insecure)) {
        const dest = new URL(url);
        dest.hostname = canonicalHost;
        dest.protocol = "https:";
        dest.port = "";
        dest.pathname = canonicalPath(pathname);
        // 301 on GET/HEAD (cacheable, passes ranking signals to the new host);
        // 308 elsewhere, which is the only redirect that preserves method+body.
        const perm = (method === "GET" || method === "HEAD") ? 301 : 308;
        return Response.redirect(dest.toString(), perm);
      }

      // Identity for the trust pages (/pt/sobre, /pt/metodologia). Unset in [vars]
      // ⇒ those blocks render the brand-level version instead of inventing a
      // name and an address the site cannot honour.
      setSiteIdentity({ author: env.SITE_AUTHOR, contact: env.SITE_CONTACT_EMAIL });
      // Staged rollout of the second SEO layer. Empty ⇒ everything is live.
      setWave(env.SEO_WAVE_MODELS);
      setLiqWave(env.LIQ_WAVE_MODELS);
      setVenderWave(env.VENDER_WAVE_MODELS);
      setIntlLocales(env.INTL_LOCALES);
      setIntlWave(env.INTL_WAVE_MODELS);

      // Internal stlite dashboard + its assets — Basic-Auth gated, fail-closed.
      if (pathname === "/analytics/leads.json") {
        if (!checkBasicAuth(request, env)) return unauthorized();
        return leadsJson(env);
      }
      if (pathname === "/analytics/clicks.json") {
        if (!checkBasicAuth(request, env)) return unauthorized();
        return clicksJson(env);
      }
      if (pathname === "/analytics/ai.json") {
        if (!checkBasicAuth(request, env)) return unauthorized();
        return aiJson(env);
      }
      if (pathname === "/analytics" || pathname.startsWith("/analytics/")) {
        return handleAnalytics(request, env, url);
      }

      if (method === "GET") {
        const tracked = trackAiFetch(env, request, pathname);
        if (ctx && typeof ctx.waitUntil === "function") ctx.waitUntil(tracked);
        else await tracked;
      }

      // URL normalisation — one canonical spelling per page, 301 to it.
      //
      // /pt/precos/ and /PRECO/AUDI-A1 used to fall past every branch into the
      // asset gate and answer 401. They are not errors, they are the same page
      // typed differently: a trailing slash from a pasted link, an upper-case
      // path from a copied-out-of-a-document URL. A 301 keeps whatever link
      // equity they carry instead of throwing it away.
      //
      // Skipped for /analytics and the asset paths, whose case is meaningful.
      if (method === "GET" || method === "HEAD") {
        const norm = canonicalPath(pathname);
        if (norm !== pathname) {
          const dest = new URL(url);
          dest.pathname = norm;
          return redirect(dest.toString(), 301);
        }
      }

      const intl = localeForPath(pathname, intlLocales());
      if (intl) return handleIntl(request, env, url, intl);

      if (pathname === "/geo" && method === "GET") return geoJson(request);

      const inKey = (env.INDEXNOW_KEY || "").trim();
      if (inKey && method === "GET" && pathname === `/${inKey}.txt`) {
        return new Response(inKey, {
          status: 200,
          headers: { "content-type": "text/plain; charset=utf-8", "cache-control": "public, max-age=86400" },
        });
      }

      // Self-hosted webfonts. Public and un-gated for the same reason as the
      // share card: the Basic-Auth fallthrough would answer 401, and a 401 on a
      // preloaded font is a page that renders in the fallback face.
      if (pathname.startsWith("/fonts/") && (method === "GET" || method === "HEAD")) {
        const res = await env.ASSETS.fetch(request);
        const out = new Response(res.body, res);
        // Content-addressed by name (the variable font file never changes under
        // this name), so a year is safe and the second page view pays nothing.
        out.headers.set("Cache-Control", "public, max-age=31536000, immutable");
        return out;
      }

      // Google Search Console ownership proof for the https://carsbuyer.org/
      // URL-prefix property. Googlebot re-checks this file periodically, so it
      // has to stay reachable for as long as the property exists — removing it
      // silently un-verifies the site. Sits above the asset gate for the same
      // reason as the share card: the Basic-Auth fallthrough would answer 401.
      if (pathname === "/google153fadf0c569abd1.html" && method === "GET") {
        return new Response("google-site-verification: google153fadf0c569abd1.html", {
          status: 200,
          headers: {
            "content-type": "text/html; charset=utf-8",
            "cache-control": "public, max-age=3600",
          },
        });
      }
      // Public social-share card (og:image / twitter:image). Served from the
      // ASSETS bucket WITHOUT the analytics Basic-Auth gate — social scrapers
      // (Facebook, Telegram, LinkedIn, Reddit) fetch it unauthenticated.
      if (pathname === "/og-default.png" && method === "GET") {
        const res = await env.ASSETS.fetch(request);
        const out = new Response(res.body, res);
        out.headers.set("Cache-Control", "public, max-age=86400");
        return out;
      }
      if (ICON_PATHS.has(pathname) && method === "GET") {
        const res = await env.ASSETS.fetch(request);
        const out = new Response(res.body, res);
        out.headers.set("Cache-Control", "public, max-age=604800");
        return out;
      }
      if (pathname === "/sitemap.xml" && method === "GET") return handleSitemapIndex(request, env, url);
      if (pathname === "/robots.txt" && method === "GET") return handleRobots(request, env, url);
      if (pathname === "/llms.txt" && method === "GET") return handleLlmsTxt(request, env, url);

      const pt = ptRest(pathname);
      if (pt !== null) return handlePt(request, env, url, pt, method);

      if (isInternalAsset(pathname)) return handleAssetGated(request, env);

      const moved = ptLegacyPath(pathname);
      if (moved) {
        const dest = new URL(url);
        dest.pathname = moved;
        return redirect(dest.toString(), (method === "GET" || method === "HEAD") ? 301 : 308);
      }
      return notFoundPage(request, env, url);
    } catch (err) {
      console.error("worker error", err && err.stack || err);
      return new Response("Internal error", { status: 500 });
    }
  },

  /**
   * Cron. Records the weekly market cut without waiting for a visitor.
   *
   * The snapshot used to be written on read, which quietly made the archive a
   * function of traffic: a week nobody browsed was a week that never got a row,
   * and a missing row is never backfilled because the numbers for a past week no
   * longer exist. On a site whose whole pitch for these URLs is "cite this with
   * a date", a hole in the series is the failure that matters.
   *
   * Runs DAILY, not weekly, and writes only when the current week is absent. So
   * a failed Monday has six more chances, and the extra runs cost one KV read
   * each. See [triggers] in wrangler.toml.
   */
  async scheduled(controller, env, ctx) {
    const now = new Date(controller.scheduledTime || Date.now());
    const r = await recordWeeklyIndex(env, now, "cron");
    // Logged either way: silence here is indistinguishable from a cron that
    // stopped firing, and that is precisely the failure we are guarding against.
    console.log(`index cron ${r.week}: ${r.written ? "written" : "skipped (" + r.reason + ")"}`);
  },
};

export default worker;

// ── Product handlers ────────────────────────────────────────────────────────

// Landing (/pt) — marketing hero with live market stats + a featured top deal.
async function handleLanding(request, env, url) {
  const { deals, degraded } = await getDeals(env, "all");

  if (degraded || !Array.isArray(deals) || deals.length === 0) {
    return html(renderInfo({
      zone: "all", depositCount: null,
      title: "Serviço indisponível",
      message: "Não foi possível carregar os negócios neste momento. Tenta novamente dentro de instantes.",
    }), degraded ? 503 : 200);
  }

  const sorted = sortDeals(deals, "score");
  const withProfit = deals.filter(d => d.est_profit_eur != null);
  const totalProfit = withProfit.reduce((s, d) => s + d.est_profit_eur, 0);
  const withDisc = deals.filter(d => d.discount_pct != null);
  const avgDisc = withDisc.length
    ? Math.round(withDisc.reduce((s, d) => s + d.discount_pct, 0) / withDisc.length * 100)
    : 0;

  return html(renderLanding({
    stats: {
      deals: deals.length,
      avgDisc: avgDisc + "%",
      totalProfit: "€" + Math.round(totalProfit).toLocaleString("pt-PT"),
    },
    featured: sorted[0],
    depositCount: null, host: url.host,
  }), 200);
}

// Avaliar (/pt/avaliar) — paste-a-link valuation of ANY OLX listing (Tier-2).
// ?q = an OLX URL or a bare olx_id; we extract the id, look it up in the
// precomputed valuations blob, and render a fair-price verdict (or a graceful
// "not found / ask by email" fallback). No q ⇒ just the lookup form + teaser.
async function handleAvaliar(request, env, url) {
  const query = (url.searchParams.get("q") || "").toString().trim();
  const modelo = (url.searchParams.get("modelo") || "").toString().trim().toLowerCase();
  const anoRaw = parseInt(url.searchParams.get("ano") || "", 10);
  const ano = Number.isFinite(anoRaw) ? anoRaw : null;

  // Paste-a-link path (an existing OLX listing).
  let rec = null, olxId = null, sourceUrl = null;
  if (query) {
    olxId = parseOlxId(query);
    if (/^https?:\/\//i.test(query)) sourceUrl = query;
    if (olxId) {
      const cars = await getValuations(env);
      rec = cars ? (cars[olxId] || null) : null;
    }
  }

  // Model index — for the paste-hit's contextual /pt/preco link, the spec-form
  // options, and the spec lookup. (cf-cached; cheap.)
  const mdoc = await getModels(env);
  const models = mdoc && mdoc.models;
  let spec = null;
  if (!rec && modelo && models && models[modelo]) {
    const mrec = models[modelo];
    spec = { rec: mrec, slug: modelo, year: ano, cell: pickYearCell(mrec, ano),
             vender: publishedVender(models, modelo, mrec, mdoc.built_at) };
  }
  return html(renderAvaliar({
    rec, olxId, sourceUrl, query, models, spec, depositCount: null,
    host: url.host, builtAt: mdoc && mdoc.built_at,
    contact: env.SITE_CONTACT_EMAIL,
    historyUrl: env.HISTORY_REPORT_URL || null,
    market: (mdoc && mdoc.lqm) || null,
    stats: models ? corpusStats(models, mdoc.built_at) : null,
  }), 200);
}

// Find the per-year cell for `ano` in a model record: exact year, or a band
// ("y0-y1") containing it. Returns null when no year given or no match.
function pickYearCell(mrec, ano) {
  if (ano == null || !Array.isArray(mrec.yr)) return null;
  for (const c of mrec.yr) {
    if (typeof c.y === "number") {
      if (c.y === ano) return c;
    } else if (typeof c.y === "string") {
      const m = c.y.match(/^(\d{4})-(\d{4})$/);
      if (m && ano >= +m[1] && ano <= +m[2]) return c;
    }
  }
  return null;
}

// Extract our olx_id from a pasted OLX URL (".../-ID<id>.html") or a bare id.
function parseOlxId(q) {
  const m = q.match(/-ID([A-Za-z0-9]+)\.html/i);
  if (m) return m[1];
  const t = q.trim();
  return /^[A-Za-z0-9]{4,14}$/.test(t) ? t : null;
}

// Per-model SEO pages under /pt/preco/.
//
//   /pt/preco/{slug}              the model page
//   /pt/preco/{slug}.json         same figures, machine-readable
//   /pt/preco/{slug}/{ano}        one model year (10+ active listings)
//   /pt/preco/{slug}/{ano}.json   same, machine-readable
//
// Exact slug lookup, never a re-split on "-" (models contain hyphens). Unknown
// slug, unknown year, or a year below the publishing floor → real 404: a page
// nobody can reach from the site and nothing links to must not exist just
// because someone typed it.
async function handleModelPage(request, env, url) {
  let rest;
  try {
    // decodeURIComponent throws URIError on a malformed %-escape (/pt/preco/%) —
    // a garbage URL must 404, not 500.
    rest = decodeURIComponent(url.pathname.slice("/preco/".length)).replace(/\/+$/, "").toLowerCase();
  } catch (_) {
    return notFoundPage(request, env, url);
  }
  const wantsJson = rest.endsWith(".json");
  if (wantsJson) rest = rest.slice(0, -".json".length);

  // Split a trailing /{year}. Slugs never contain "/", so this is unambiguous.
  let slug = rest, year = null, facet = null;
  const slash = rest.lastIndexOf("/");
  if (slash > 0) {
    const tail = rest.slice(slash + 1);
    slug = rest.slice(0, slash);
    // Four digits is a model year; anything else is a facet key (fuel or
    // district) and is resolved against the blob below — a segment that matches
    // neither is a 404, not a guess.
    if (/^\d{4}$/.test(tail)) year = parseInt(tail, 10);
    else if (/^[a-z0-9-]{2,40}$/.test(tail)) facet = tail;
    else return notFoundPage(request, env, url);
  }

  const mdoc = await getModels(env);
  const models = mdoc && mdoc.models;
  const rec = models ? models[slug] : null;
  if (!rec) return notFoundPage(request, env, url);
  const builtAt = mdoc && mdoc.built_at;

  if (year != null) return renderYear({ request, env, url, models, rec, slug, year, builtAt, wantsJson });
  if (facet != null) return renderFacet({ request, env, url, models, rec, slug, facet, builtAt, wantsJson });

  if (wantsJson) return jsonResponse(modelJson(rec, slug, { host: url.host, builtAt, models }));

  await primeAltMarkets(env, "pt", models);
  const stats = corpusStats(models, builtAt);

  // Conversion bridge: live hot_deals matching this model (already below-fair).
  let liveDeals = [];
  try {
    const { deals } = await getDeals(env, "all");
    liveDeals = (deals || []).filter(d => slugify(`${d.brand}-${d.model}`) === slug).slice(0, 3);
  } catch (_) { /* bridge is best-effort */ }

  // Same-brand siblings (unchanged) …
  const siblings = Object.entries(models)
    .filter(([sl, r]) => r.b === rec.b && sl !== slug)
    .sort((a, b) => (b[1].n || 0) - (a[1].n || 0))
    .slice(0, 8)
    .map(([sl, r]) => ({ slug: sl, m: r.m, fm: r.fm, n: r.n }));

  const cls = modelClass(slug);
  let competitors = [], competitorKind = "price";
  if (cls) {
    competitors = Object.entries(models)
      .filter(([sl, r]) => sl !== slug && r.b !== rec.b && r.fm > 0 && modelClass(sl) === cls)
      .map(([sl, r]) => ({ sl, r, gap: comparePriceGap(r, rec) }))
      .filter(x => x.gap && x.gap.years >= COMPETITOR_MIN_YEARS && x.gap.dist <= COMPETITOR_TOL)
      .sort((a, b) => (b.r.n || 0) - (a.r.n || 0))
      .slice(0, 6)
      .map(x => ({ slug: x.sl, b: x.r.b, m: x.r.m, fm: x.r.fm, ratio: x.gap.ratio }));
    if (competitors.length) competitorKind = "segment";
  }
  if (!competitors.length) {
    competitors = Object.entries(models)
      .filter(([sl, r]) => sl !== slug && r.b !== rec.b && r.fm > 0
        && Math.abs(r.fm - rec.fm) / Math.max(r.fm, rec.fm) <= 0.22)
      .sort((a, b) => (b[1].n || 0) - (a[1].n || 0))
      .slice(0, 6)
      .map(([sl, r]) => ({ slug: sl, b: r.b, m: r.m, fm: r.fm }));
    competitorKind = "price";
  }

  // Generated comparison pages that include this model.
  const comparisons = publishedPairs(models)
    .filter(([a, b]) => a === slug || b === slug)
    .map(([a, b]) => {
      const other = a === slug ? b : a;
      return { href: `${a}-vs-${b}`, m: `${models[other].b} ${models[other].m}` };
    });

  return publicHtml(renderModelPage({
    guides: guideBlock("preco"),
    rec, slug, liveDeals, siblings, host: url.host, depositCount: null, builtAt,
    insights: modelInsights(rec, stats),
    yearPages: publishedYearPages(models, slug, rec, builtAt),
    competitors, competitorKind, comparisons,
    // Fuel/district cuts, tagged with their kind so the page can label them.
    // Gated by the same wave as everything else, so the page never links a URL
    // the router would refuse.
    facets: publishedFacets(models, slug, rec, builtAt).map(k => {
      const kind = facetKind(rec, k);
      const cell = facetCell(rec, kind, k);
      return { k, kind, lbl: cell.lbl, n: cell.n, fm: cell.fm };
    }),
    hasDepreciation: publishedDepreciation(models, slug, rec, builtAt),
    hasLiquidity: publishedLiquidity(models, slug, rec, builtAt),
    hasVender: publishedVender(models, slug, rec, builtAt),
    duels: duelsFor(models, slug, rec, builtAt).map(d => ({ path: d.path, kind: d.kind })),
    provenanceHtml: provenance({ n: rec.n, builtAt }),
    altJson: `https://${url.host}/pt/preco/${slug}.json`,
  }));
}

// /pt/preco/{slug}/{ano} — one model year.
async function renderYear({ request, env, url, models, rec, slug, year, builtAt, wantsJson }) {
  const published = publishedYearPages(models, slug, rec, builtAt);
  const cell = published.includes(year) ? yearCell(rec, year) : null;
  if (!cell) {
    if (published.length && (rec.yr || []).some(c => c.y === year)) {
      return new Response(null, {
        status: 301,
        headers: { location: `https://${url.host}/pt/preco/${encodeURIComponent(slug)}${wantsJson ? ".json" : ""}` },
      });
    }
    return notFoundPage(request, env, url);
  }
  if (wantsJson) return jsonResponse(yearJson(rec, slug, year, cell, { host: url.host, builtAt }));

  const stats = corpusStats(models, builtAt);
  // Neighbours come from ALL year cells (n>=5), not only those with pages: the
  // comparison "is 2013 worth the premium over 2012" is still true when 2013 is
  // too thin for a page of its own, and dropping it would leave a gap that
  // reads as missing data.
  const all = yearCells(rec, 1).slice().sort((a, b) => a.y - b.y);
  const idx = all.findIndex(c => c.y === year);
  const older = idx > 0 ? all[idx - 1] : null;
  const newer = (idx >= 0 && idx < all.length - 1) ? all[idx + 1] : null;
  const win = all.slice(Math.max(0, idx - 3), idx + 4).slice().sort((a, b) => b.y - a.y);

  // Deals for this exact year, else the nearest years of the same model. A year
  // page usually has no deal of its own (the feed carries ~30 cars per zone),
  // and an empty block was the page's only answer to "and what can I buy now".
  // The fallback is labelled as neighbouring years, never passed off as this one.
  let liveDeals = [], dealsNear = false;
  try {
    const { deals } = await getDeals(env, "all");
    const mine = (deals || []).filter(d => slugify(`${d.brand}-${d.model}`) === slug);
    liveDeals = mine.filter(d => Number(d.year) === year).slice(0, 3);
    if (!liveDeals.length) {
      liveDeals = mine
        .filter(d => Number.isFinite(Number(d.year)) && Math.abs(Number(d.year) - year) <= DEALS_NEAR_YEARS)
        .sort((a, b) => Math.abs(Number(a.year) - year) - Math.abs(Number(b.year) - year))
        .slice(0, 3);
      dealsNear = liveDeals.length > 0;
    }
  } catch (_) { /* best-effort */ }

  return publicHtml(renderYearPage({
    guides: guideBlock("preco"),
    rec, slug, year, cell,
    neighbours: { older, newer, window: win },
    liveDeals, dealsNear, pageYears: publishedYearPages(models, slug, rec, builtAt), stats,
    host: url.host, depositCount: null, builtAt,
    historyUrl: env.HISTORY_REPORT_URL || null,
    hasVender: publishedVender(models, slug, rec, builtAt),
  }));
}

// Embeddable widget (/pt/widget/preco/{slug}) — a standalone valuation card other
// sites iframe. Public + cacheable + cookie-less; permissive frame-ancestors so
// any host can embed. Unknown/sub-threshold slug → 404 (never an empty frame).
async function handleModelWidget(request, env, url) {
  let slug;
  try {
    slug = decodeURIComponent(url.pathname.slice("/widget/preco/".length)).replace(/\/+$/, "").toLowerCase();
  } catch (_) {
    return notFound();
  }
  const mdoc = await getModels(env);
  const rec = mdoc && mdoc.models ? mdoc.models[slug] : null;
  if (!rec) return notFound();
  return new Response(renderModelWidget({ rec, slug, host: url.host }), {
    status: 200,
    headers: {
      "Content-Type": "text/html; charset=utf-8",
      "Cache-Control": "public, max-age=3600",
      // Allow embedding on any site (this is the backlink lever); no X-Frame-Options.
      "Content-Security-Policy": "frame-ancestors *",
    },
  });
}

// /pt/preco/{slug}/{combustivel} and /pt/preco/{slug}/{distrito}.
//
// Both live off `fx` / `dt` cells in models.json. Until the pipeline that emits
// them has run, facetKind() finds nothing and every such URL 404s — which is
// the correct answer, and means the pages switch on by themselves at the next
// data build with no deploy.
async function renderFacet({ request, env, url, models, rec, slug, facet, builtAt, wantsJson }) {
  const kind = publishedFacets(models, slug, rec, builtAt).includes(facet) ? facetKind(rec, facet) : null;
  if (!kind) {
    if (retiredFacetKind(rec, facet)) {
      return new Response(null, {
        status: 301,
        headers: { location: `https://${url.host}/pt/preco/${encodeURIComponent(slug)}` },
      });
    }
    return notFoundPage(request, env, url);
  }
  const cell = facetCell(rec, kind, facet);
  if (wantsJson) {
    return jsonResponse(facetJson(rec, slug, kind, cell, publishedCells(rec, kind),
                                 { host: url.host, builtAt }));
  }
  return publicHtml(renderFacetPage({
    rec, slug, kind, cell, altJson: `https://${url.host}/pt/preco/${slug}/${cell.k}.json`,
    duelSpec: (kind === "fuel" && publishedDuel(models, slug, rec, builtAt, "fuel")) ? DUELS.fuel
            : (kind === "transmission" && publishedDuel(models, slug, rec, builtAt, "gear")) ? DUELS.gear
            : null,
    siblingsCells: publishedCells(rec, kind),
    stats: corpusStats(models, builtAt),
    host: url.host, depositCount: null, builtAt,
  }));
}

// /pt/precos/{distrito}
async function handleDistrict(request, env, url) {
  let key;
  try {
    key = decodeURIComponent(url.pathname.slice("/precos/".length)).replace(/\/+$/, "").toLowerCase();
  } catch (_) { return notFoundPage(request, env, url); }
  const mdoc = await getModels(env);
  const models = mdoc && mdoc.models;
  const districts = (mdoc && mdoc.districts) || null;
  const rec = (districts && districts[key]) || null;
  if (!models || !rec) return notFoundPage(request, env, url);
  return publicHtml(renderDistrictPage({
    key, rec, models, districts, stats: corpusStats(models, mdoc.built_at),
    host: url.host, depositCount: null, builtAt: mdoc.built_at,
  }));
}

// ── Second-layer SEO handlers ───────────────────────────────────────────────
//
// All of them read the same models.json the /pt/preco pages read. Each answers a
// query the model pages could not: how fast this loses value, which of these two
// to buy, how long either takes to sell, and what the market as a whole did this
// week.

// A page that only makes sense once models.json exists. Degrades to a 503 with
// an honest message rather than a half-empty page or a 500.
async function withModels(request, env, url, fn) {
  const mdoc = await getModels(env);
  const models = mdoc && mdoc.models;
  if (!models) {
    return html(renderInfo({
      zone: "all", depositCount: null, title: "Serviço indisponível",
      message: "Os dados de mercado estão a ser preparados. Volta dentro de instantes.",
    }), 503);
  }
  return fn({ models, builtAt: mdoc.built_at, depositCount: null, setCookie: null,
              mq: mdoc.mq || null, market: mdoc.lqm || null,
              stats: corpusStats(models, mdoc.built_at) });
}

// /pt/depreciacao/{slug} and /pt/depreciacao/{slug}.json
async function handleDepreciation(request, env, url) {
  let slug;
  try {
    slug = decodeURIComponent(url.pathname.slice("/depreciacao/".length)).replace(/\/+$/, "").toLowerCase();
  } catch (_) { return notFoundPage(request, env, url); }
  const wantsJson = slug.endsWith(".json");
  if (wantsJson) slug = slug.slice(0, -".json".length);
  return withModels(request, env, url, ({ models, builtAt, depositCount, setCookie, stats }) => {
    const rec = models[slug];
    if (!rec || !publishedDepreciation(models, slug, rec, builtAt)) return notFoundPage(request, env, url, setCookie);
    const fit = depreciationFit(rec);
    if (wantsJson) {
      return jsonResponse(depreciationJson(rec, slug, fit, depreciationAge(rec, fit, builtAt),
                                           { host: url.host, builtAt }));
    }
    return publicHtml(renderDepreciationPage({
      rec, slug, fit, stats,
      pageYears: publishedYearPages(models, slug, rec, builtAt),
      host: url.host, depositCount, builtAt,
    }));
  });
}

// /pt/depreciacao — ranked hub, fastest-losing first.
async function handleDepreciationHub(request, env, url) {
  return withModels(request, env, url, ({ models, builtAt, depositCount, setCookie, stats }) => {
    const rows = depreciationSlugs(models).filter(slug => publishedDepreciation(models, slug, models[slug], builtAt)).map(slug => {
      const r = models[slug], f = depreciationFit(r), av = depreciationAge(r, f, builtAt);
      return { slug, b: r.b, m: r.m, n: r.n, rate: f.rate, span: f.span,
               half: av && av.halfLife, cheapAge: av && av.cheapFrom ? av.cheapFrom.age : null };
    }).sort((a, b) => b.rate - a.rate);
    return publicHtml(renderDepreciationHub({
      rows, stats, host: url.host, depositCount, builtAt,
      duelHubs: Object.values(DUELS).filter(d => duelSlugs(models, d.kind, builtAt).length),
    }));
  });
}

async function handleDuel(request, env, url, spec) {
  let slug;
  try {
    slug = decodeURIComponent(url.pathname.slice(`/${spec.path}/`.length)).replace(/\/+$/, "").toLowerCase();
  } catch (_) { return notFoundPage(request, env, url); }
  const wantsJson = slug.endsWith(".json");
  if (wantsJson) slug = slug.slice(0, -".json".length);
  return withModels(request, env, url, ({ models, builtAt, depositCount, setCookie, stats }) => {
    const rec = models[slug];
    if (!rec || !publishedDuel(models, slug, rec, builtAt, spec.kind)) return notFoundPage(request, env, url, setCookie);
    const av = duel(rec, spec.kind, builtAt);
    if (!av) return notFoundPage(request, env, url, setCookie);
    if (wantsJson) return jsonResponse(duelJson(rec, slug, av, { host: url.host, builtAt }));
    return publicHtml(renderDuelPage({
      rec, slug, av, stats, host: url.host, depositCount, builtAt,
      facetKeys: publishedFacets(models, slug, rec, builtAt),
    }));
  });
}

async function handleDuelHub(request, env, url, spec) {
  return withModels(request, env, url, ({ models, builtAt, depositCount, setCookie, stats }) => {
    const rows = duelSlugs(models, spec.kind, builtAt).map(slug => {
      const r = models[slug];
      return { slug, b: r.b, m: r.m, av: duel(r, spec.kind, builtAt) };
    }).filter(r => r.av);
    if (!rows.length) return notFoundPage(request, env, url, setCookie);
    const other = Object.values(DUELS)
      .find(d => d.kind !== spec.kind && duelSlugs(models, d.kind, builtAt).length) || null;
    return publicHtml(renderDuelHub({ spec, rows, other, stats, host: url.host, depositCount, builtAt }));
  });
}

// /pt/comparar/{a}-vs-{b}
async function handleCompare(request, env, url) {
  let rest;
  try {
    rest = decodeURIComponent(url.pathname.slice("/comparar/".length)).replace(/\/+$/, "").toLowerCase();
  } catch (_) { return notFoundPage(request, env, url); }
  return withModels(request, env, url, ({ models, builtAt, depositCount, setCookie, stats }) => {
    const pairSet = new Set(publishedPairs(models).map(([a, b]) => comparePairKey(a, b)));
    const pair = parseComparePath(rest, models, pairSet);
    if (!pair) return notFoundPage(request, env, url, setCookie);
    return publicHtml(renderComparePage({
      a: pair.a, b: pair.b, ra: models[pair.a], rb: models[pair.b],
      stats, host: url.host, depositCount, builtAt,
    }));
  });
}

// /pt/comparar
async function handleCompareHub(request, env, url) {
  return withModels(request, env, url, ({ models, builtAt, depositCount, setCookie }) =>
    publicHtml(renderCompareHub({ pairs: publishedPairs(models), models, host: url.host, depositCount, builtAt })));
}

// /pt/liquidez
async function handleLiquidity(request, env, url) {
  return withModels(request, env, url, ({ models, builtAt, depositCount, setCookie, market }) => {
    const rows = Object.entries(models)
      .filter(([, r]) => (r.lq && r.lq.s30 != null) || (r.sd != null && r.sn != null))
      .map(([slug, r]) => ({
        slug, b: r.b, m: r.m, sd: r.sd, sn: r.sn, fm: r.fm,
        lq: (r.lq && r.lq.s30 != null) ? r.lq : null,
        page: publishedLiquidity(models, slug, r, builtAt),
      }))
      .sort((a, b) => {
        if (a.lq && b.lq) return b.lq.s30 - a.lq.s30 || b.lq.n - a.lq.n;
        if (a.lq) return -1;
        if (b.lq) return 1;
        return (a.sd || 0) - (b.sd || 0);
      });
    return publicHtml(renderLiquidityHub({ rows, market, host: url.host, depositCount, builtAt }));
  });
}

async function handleLiquidityPage(request, env, url) {
  let slug;
  try {
    slug = decodeURIComponent(url.pathname.slice("/liquidez/".length)).replace(/\/+$/, "").toLowerCase();
  } catch (_) { return notFoundPage(request, env, url); }
  const wantsJson = slug.endsWith(".json");
  if (wantsJson) slug = slug.slice(0, -".json".length);
  return withModels(request, env, url, ({ models, builtAt, depositCount, setCookie, market }) => {
    const rec = models[slug];
    if (!rec || !publishedLiquidity(models, slug, rec, builtAt)) return notFoundPage(request, env, url, setCookie);
    if (wantsJson) return jsonResponse(liquidityJson(rec, slug, { host: url.host, builtAt }));
    return publicHtml(renderLiquidityPage({
      rec, slug, market,
      hasDepreciation: publishedDepreciation(models, slug, rec, builtAt),
      host: url.host, depositCount, builtAt,
    }));
  });
}

// /pt/sobrevalorizados — both directions of the asking-vs-estimate gap.
async function handleValuationGap(request, env, url) {
  return withModels(request, env, url, ({ models, builtAt, depositCount, setCookie, stats, market }) => {
    const withGap = Object.entries(models)
      .filter(([, r]) => r.gm > 0 && r.fm > 0)
      .map(([slug, r]) => ({
        slug, b: r.b, m: r.m, fm: r.fm, gm: r.gm, n: r.n, gap: r.fm / r.gm - 1,
        s30: (r.lq && r.lq.s30 != null) ? r.lq.s30 : null,
        page: publishedLiquidity(models, slug, r, builtAt),
      }))
      .sort((a, b) => b.gap - a.gap);
    return publicHtml(renderValuationGap({
      over: withGap.slice(0, 25),
      under: withGap.slice(-25).reverse(),
      market, stats, host: url.host, depositCount, builtAt,
    }));
  });
}

async function getImports(env) {
  const url = `${HOT_DEALS_BASE}/import.json`;
  try {
    const r = await fetch(url, {
      cf: { cacheEverything: true, cacheTtlByStatus: { "200-299": 300, "300-399": 0, "400-499": 0, "500-599": 0 } },
    });
    if (!r.ok) return null;
    const data = await r.json();
    return data && data.models ? data : null;
  } catch (err) {
    console.warn("import fetch error", err && err.message);
    return null;
  }
}

async function handleImportHub(request, env, url) {
  const doc = await getImports(env);
  if (!doc) return notFoundPage(request, env, url);
  const rows = importSlugs(doc).map(slug => {
    const r = doc.models[slug];
    return { slug, b: r.b, m: r.m, med_gap: r.med_gap, wins: r.wins,
             cells: (r.yr || []).length, nde: r.nde, npt: r.npt };
  });
  if (!rows.length) return notFoundPage(request, env, url);
  return publicHtml(renderImportHub({
    rows, costs: doc.costs, host: url.host, depositCount: null, builtAt: doc.built_at,
  }));
}

async function handleImportPage(request, env, url) {
  let slug;
  try {
    slug = decodeURIComponent(url.pathname.slice("/importar/".length)).replace(/\/+$/, "").toLowerCase();
  } catch (_) { return notFoundPage(request, env, url); }
  const wantsJson = slug.endsWith(".json");
  if (wantsJson) slug = slug.slice(0, -".json".length);
  const doc = await getImports(env);
  const rec = doc && doc.models ? doc.models[slug] : null;
  if (!rec || !importOk(rec)) return notFoundPage(request, env, url);
  if (wantsJson) return jsonResponse(importJson(rec, slug, doc.costs, { host: url.host, builtAt: doc.built_at }));
  const mdoc = await getModels(env);
  const hasModelPage = !!(mdoc && mdoc.models && mdoc.models[slug]);
  return publicHtml(renderImportPage({
      guides: guideBlock("importar"),
    rec, slug, costs: doc.costs, hasModelPage,
    host: url.host, depositCount: null, builtAt: doc.built_at,
    historyUrl: env.HISTORY_REPORT_URL || null,
  }));
}

// /pt/metodologia, /pt/sobre, /pt/isv
async function handleMethodology(request, env, url) {
  return withModels(request, env, url, ({ models, builtAt, depositCount, stats, mq }) =>
    publicHtml(renderMethodology({
      stats, mq, host: url.host, depositCount, builtAt,
      duelHubs: Object.values(DUELS).filter(d => duelSlugs(models, d.kind, builtAt).length),
      wave: (() => {
        const w = waveSlugs(models, builtAt);
        if (!w) return null;
        let pages = 0;
        for (const s of w) {
          pages += publishedYearPages(models, s, models[s], builtAt).length
                 + publishedFacets(models, s, models[s], builtAt).length
                 + (publishedDepreciation(models, s, models[s], builtAt) ? 1 : 0);
        }
        const sample = Object.entries(models)
          .filter(([s, r]) => !w.has(s) && yearPageYears(r).length)
          .sort((a, b) => (b[1].n || 0) - (a[1].n || 0))
          .map(([s]) => s)[0] || null;
        return { models: w.size, total: Object.keys(models).length, pages, sample };
      })(),
    })));
}
async function handleAbout(request, env, url) {
  return withModels(request, env, url, ({ builtAt, depositCount, setCookie, stats, mq }) =>
    publicHtml(renderAbout({ stats, mq, host: url.host, depositCount, builtAt })));
}
async function handleIsv(request, env, url) {
  return withModels(request, env, url, ({ models, builtAt, depositCount, setCookie }) => {
    const topModels = Object.entries(models)
      .sort((a, b) => (b[1].n || 0) - (a[1].n || 0)).slice(0, 12)
      .map(([slug, r]) => ({ slug, b: r.b, m: r.m, fm: r.fm }));
    const refYear = parseInt((builtAt || "").slice(0, 4), 10) || null;
    return publicHtml(renderIsv({ topModels, host: url.host, depositCount, builtAt, refYear }));
  });
}

// ── /pt/mercado/indice — weekly market index with a permanent archive ───────
//
// The index has to be citable, and a figure that changes under its own URL is
// not. So each ISO week is written ONCE to KV, at the first request that sees
// that week, and never rewritten — /pt/mercado/indice/2026-W35 says the same thing
// next year as it does today. The bare /pt/mercado/indice shows the current week
// plus the trend.
//
// One KV write per week (guarded by a read), so the write-on-read is bounded no
// matter how much traffic the page gets.
const IDX_WEEK_PREFIX = "idx:week:";
const IDX_LIST_KEY = "idx:weeks";
const IDX_MAX_WEEKS = 120;

// `src` records which path wrote the row: the cron, or a visitor's request.
//
// Not editorial, so it does not appear on the page — it exists so the question
// "did the cron actually fire, or did a passing crawler write this?" has an
// answer in the data instead of an argument from timing.
function snapshotFrom(models, builtAt, week, date, src) {
  const stats = corpusStats(models, builtAt);
  return {
    week, date, src: src || "web", builtAt: builtAt || null,
    models: stats.models, listings: stats.listings,
    priceMed: stats.priceMed, kmMed: stats.kmMed,
    sellMed: stats.sellMed, depMed: stats.depMed,
  };
}

/**
 * Record this week's cut, once.
 *
 * Shared by the request path and the cron below, because the archive's whole
 * value is that a week's URL keeps saying the same thing — and two code paths
 * writing it are two chances to disagree about what "this week" means.
 *
 * Never overwrites an existing week. Returns the history either way, so the
 * caller can render without a second read.
 */
const SNAP_MODELS_PREFIX = "snap:models:";
const SNAP_LIST_KEY = "snap:models:weeks";

function archiveKeys(country) {
  const c = String(country || "PT").toUpperCase();
  if (c === "PT") {
    return { snapList: SNAP_LIST_KEY, snapWeek: SNAP_MODELS_PREFIX,
             idxList: IDX_LIST_KEY, idxWeek: IDX_WEEK_PREFIX };
  }
  const k = c.toLowerCase();
  return {
    snapList: `${SNAP_MODELS_PREFIX}${k}:weeks`, snapWeek: `${SNAP_MODELS_PREFIX}${k}:`,
    idxList: `${IDX_LIST_KEY}:${k}`, idxWeek: `${IDX_WEEK_PREFIX}${k}:`,
  };
}

function modelsCut(models) {
  const out = {};
  for (const [slug, r] of Object.entries(models)) {
    out[slug] = {
      b: r.b, m: r.m, fm: r.fm, fl: r.fl, fh: r.fh, n: r.n, kmm: r.kmm,
      yr: (r.yr || []).map(c => ({ y: c.y, fm: c.fm, fl: c.fl, fh: c.fh, n: c.n })),
    };
  }
  return out;
}

async function snapshotWeeks(env, country = null) {
  try {
    const listed = await env.KV.get(archiveKeys(country).snapList, "json");
    return Array.isArray(listed) ? listed : [];
  } catch (_) { return []; }
}

async function recordModelsSnapshot(env, models, builtAt, week, date, src, country = null) {
  const keys = archiveKeys(country);
  const weeks = await snapshotWeeks(env, country);
  if (weeks.includes(week)) return false;
  const body = JSON.stringify({
    week, date, builtAt: builtAt || null, src, models: modelsCut(models),
  });
  try {
    await env.KV.put(`${keys.snapWeek}${week}`, body);
    await env.KV.put(keys.snapList, JSON.stringify([...weeks, week].sort()));
  } catch (err) {
    console.warn("models snapshot write failed", err && err.message);
    return false;
  }
  return true;
}

async function recordWeeklyIndex(env, now, src = "web", country = null) {
  const keys = archiveKeys(country);
  const mdoc = await getModels(env, country);
  const models = mdoc && mdoc.models;
  const week = isoWeek(now);
  const today = now.toISOString().slice(0, 10);

  let history = [];
  try {
    const listed = await env.KV.get(keys.idxList, "json");
    if (Array.isArray(listed)) history = listed;
  } catch (_) { /* the archive is a nice-to-have, never a 500 */ }

  // No data means no snapshot. A row of nulls is worse than a gap: the gap is
  // visible and honest, the nulls look like a market that stopped existing.
  if (!models) return { week, history, written: false, reason: "no-models" };
  await recordModelsSnapshot(env, models, mdoc.built_at, week, today, src, country);
  if (history.some(h => h.week === week)) return { week, history, written: false, reason: "already" };

  const snap = snapshotFrom(models, mdoc.built_at, week, today, src);
  const next = [...history, snap].sort((a, b) => a.week < b.week ? -1 : 1).slice(-IDX_MAX_WEEKS);
  try {
    await env.KV.put(`${keys.idxWeek}${week}`, JSON.stringify(snap));
    await env.KV.put(keys.idxList, JSON.stringify(next));
  } catch (err) {
    console.warn("index snapshot write failed", err && err.message);
    return { week, history, written: false, reason: "kv-error" };
  }
  return { week, history: next, written: true, snapshot: snap };
}

function weekToken(raw) {
  const m = /^(\d{4})-w(\d{2})$/i.exec(raw || "");
  return m ? `${m[1]}-W${m[2]}` : null;
}

async function readSnapshot(env, week, country = null) {
  try { return await env.KV.get(`${archiveKeys(country).snapWeek}${week}`, "json"); }
  catch (_) { return null; }
}

async function handleArchiveHub(request, env, url) {
  const weeks = await snapshotWeeks(env);
  return publicHtml(renderArchiveHub({ weeks, host: url.host }));
}

async function handleArchive(request, env, url) {
  let tail;
  try {
    tail = decodeURIComponent(url.pathname.slice("/historico/".length)).replace(/\/+$/, "").toLowerCase();
  } catch (_) { return notFoundPage(request, env, url); }
  if (!tail.endsWith(".json")) return notFoundPage(request, env, url);
  const parts = tail.slice(0, -".json".length).split("/");
  if (parts.length > 2) return notFoundPage(request, env, url);

  const week = weekToken(parts[0]);
  if (!week) return notFoundPage(request, env, url);
  const snap = await readSnapshot(env, week);
  if (!snap) return notFoundPage(request, env, url);

  const base = {
    week: snap.week, date: snap.date, built_at: snap.builtAt,
    source: `https://${url.host}/pt/historico/${parts[0]}.json`,
    note: "Valores congelados nesta semana. Não mudam: cita-os com a data.",
  };
  const frozen = payload => {
    const res = jsonResponse(payload);
    res.headers.set("cache-control", "public, max-age=31536000, immutable");
    return res;
  };
  if (parts.length === 1) {
    return frozen({ ...base, models: snap.models || {} });
  }
  const rec = (snap.models || {})[parts[1]];
  if (!rec) return notFoundPage(request, env, url);
  return frozen({
    ...base,
    source: `https://${url.host}/pt/historico/${parts[0]}/${parts[1]}.json`,
    slug: parts[1], ...rec,
  });
}

async function handleMarketIndex(request, env, url) {
  const tail = url.pathname.slice("/mercado/indice".length).replace(/^\/+|\/+$/g, "");
  return withModels(request, env, url, async ({ models, builtAt, depositCount, setCookie }) => {
    const now = new Date();
    const { week, history } = await recordWeeklyIndex(env, now);
    const gaps = missingWeeks(history, week);
    const months = monthlyCuts(history, week);

    if (tail) {
      const mm = /^(\d{4})-(\d{2})$/.exec(tail);
      if (mm) {
        const cut = months.find(c => c.month === tail);
        if (!cut) return notFoundPage(request, env, url, setCookie);
        return publicHtml(renderMarketMonth({ cut, months, host: url.host, depositCount }));
      }
      // An archived week. Only weeks we actually recorded exist — an invented
      // /pt/mercado/indice/1999-w03 is a 404, not an empty page.
      // Normalisation has already lower-cased the path, so the URL token is
      // "2026-w35"; the stored key and the display form are ISO ("2026-W35").
      const m = /^(\d{4})-w(\d{2})$/i.exec(tail);
      if (!m) return notFoundPage(request, env, url, setCookie);
      const wk = `${m[1]}-W${m[2]}`;
      let snap = null;
      try { snap = await env.KV.get(`${IDX_WEEK_PREFIX}${wk}`, "json"); } catch (_) {}
      if (!snap) snap = history.find(h => h.week === wk) || null;
      if (!snap) return notFoundPage(request, env, url, setCookie);
      return publicHtml(renderMarketIndex({
        snapshot: snap, history, host: url.host, depositCount,
        isArchive: true, currentWeek: week, gaps, months,
      }));
    }

    const current = history.find(h => h.week === week)
      || snapshotFrom(models, builtAt, week, now.toISOString().slice(0, 10), "web");
    return publicHtml(renderMarketIndex({
      snapshot: current, history, host: url.host, depositCount, currentWeek: week, gaps, months,
    }));
  });
}

// ── 404 ─────────────────────────────────────────────────────────────────────
//
// A styled 404 with real links, not a bare string — and never a 401. Best-effort
// suggestions from models.json; if that fetch fails the page still renders.
async function notFoundPage(request, env, url, setCookie = null) {
  let suggestions = [];
  let depositCount = 0;
  try {
    const mdoc = await getModels(env);
    if (mdoc && mdoc.models) {
      suggestions = Object.entries(mdoc.models)
        .sort((a, b) => (b[1].n || 0) - (a[1].n || 0)).slice(0, 12)
        .map(([slug, r]) => ({ slug, m: `${r.b} ${r.m}`, fm: r.fm }));
    }
  } catch (_) { /* a 404 must never depend on a network call */ }
  return html(renderNotFound({
    suggestions, depositCount, host: url.host, path: url.pathname,
  }), 404, setCookie);
}

function jsonResponse(payload, status = 200) {
  return new Response(JSON.stringify(payload, null, 2), {
    status,
    headers: {
      "content-type": "application/json; charset=utf-8",
      // Public, cacheable, and cross-origin readable: the point of this endpoint
      // is that other people's tools can take the numbers.
      "cache-control": "public, max-age=1800",
      "access-control-allow-origin": "*",
    },
  });
}

export function registerIntlPages(mod) {
  registerIntlPageModule(mod);
  return mod;
}

function matchIntlModule(loc, path) {
  for (const mod of intlPageModules()) {
    for (const route of mod.routes) {
      const seg = loc.routes[route.routeKey];
      if (typeof seg !== "string" || !seg) continue;
      const base = `/${seg}`;
      if (path !== base && !path.startsWith(`${base}/`)) continue;
      const rest = path.slice(base.length);
      let params = null;
      try {
        params = route.match(rest);
      } catch (err) {
        console.warn(`intl module ${mod.id} match error`, err && err.message);
        params = null;
      }
      if (params) return { mod, route, params, rest };
    }
  }
  return null;
}

async function runIntlModule(request, env, url, loc, path, claimed) {
  const mdoc = await getModels(env, loc.country);
  const models = mdoc && mdoc.models;
  if (!models) return intlUnavailable(loc, url.host, "info.preparing_models");
  const builtAt = mdoc.built_at;
  refreshIntlNav(loc, models, builtAt, mdoc);
  return claimed.route.handle({
    request, env, url, loc, models, builtAt, mdoc,
    stats: corpusStats(models, builtAt),
    path, rest: claimed.rest, params: claimed.params,
    helpers: {
      publicHtml,
      jsonResponse,
      notFoundIntl: (e = env, u = url, l = loc) => intlNotFound(e, u, l),
      getModels,
    },
  });
}

async function handleIntl(request, env, url, intl) {
  const { loc, rest } = intl;
  if (request.method !== "GET") return intlNotFound(env, url, loc);
  const R = loc.routes;
  const path = rest === "/" ? "" : rest;
  if (path === "") return intlLanding(env, url, loc);
  if (path === `/${R.hub}`) return intlHub(env, url, loc);
  if (path === `/${R.avaliar}`) return intlAvaliar(env, url, loc);
  if (path === `/${R.mercado}`) return intlFeed(env, url, loc);
  if (path === `/${R.car}`) return intlCar(env, url, loc);
  if (path === `/${R.arquivo}`) return intlArchive(env, url, loc, "");
  if (path.startsWith(`/${R.arquivo}/`)) {
    return intlArchive(env, url, loc, path.slice(`/${R.arquivo}/`.length));
  }
  if (path === `/${R.indice}`) return intlMarketIndex(env, url, loc, "");
  if (path.startsWith(`/${R.indice}/`)) {
    return intlMarketIndex(env, url, loc, path.slice(`/${R.indice}/`.length));
  }
  if (path === `/${R.metodologia}`) return intlTrustPage(env, url, loc, "method");
  if (path === `/${R.sobre}`) return intlTrustPage(env, url, loc, "about");
  if (path === `/${R.privacidade}`) return intlPrivacy(env, url, loc);
  if (path === "/sitemap.xml") return intlSitemap(env, url, loc);
  const claimed = matchIntlModule(loc, path);
  if (claimed) return runIntlModule(request, env, url, loc, path, claimed);
  if (path.startsWith(`/${R.model}/`)) {
    return intlModel(env, url, loc, path.slice(`/${R.model}/`.length));
  }
  return intlNotFound(env, url, loc);
}

const NAV_STAMP = new Map();

function refreshIntlNav(loc, models, builtAt, mdoc) {
  if (NAV_STAMP.get(loc.code) === builtAt) return;
  const declared = [], live = [];
  for (const mod of intlPageModules()) {
    if (!Array.isArray(mod.navRouteKeys) || !mod.navRouteKeys.length) continue;
    for (const k of mod.navRouteKeys) declared.push(k);
    if (typeof mod.navAvailable !== "function") {
      for (const k of mod.navRouteKeys) live.push(k);
      continue;
    }
    try {
      for (const k of mod.navAvailable(loc, models, builtAt, mdoc) || []) live.push(k);
    } catch (err) {
      console.warn(`intl nav ${mod.id} error`, err && err.message);
      for (const k of mod.navRouteKeys) live.push(k);
    }
  }
  setNavLive(loc.code, declared, live);
  NAV_STAMP.set(loc.code, builtAt);
}

async function intlCorpus(env, loc) {
  const mdoc = await getModels(env, loc.country);
  const models = mdoc && mdoc.models;
  if (!models) return null;
  refreshIntlNav(loc, models, mdoc.built_at, mdoc);
  return { models, mdoc, builtAt: mdoc.built_at, stats: corpusStats(models, mdoc.built_at) };
}

function intlUnavailable(loc, host, key) {
  return html(renderIntlInfo({
    loc, host, title: it(loc, "info.unavailable"), message: it(loc, key),
  }), 503);
}

async function intlNotFound(env, url, loc) {
  let suggestions = [];
  try {
    const mdoc = await getModels(env, loc.country);
    if (mdoc && mdoc.models) {
      refreshIntlNav(loc, mdoc.models, mdoc.built_at, mdoc);
      suggestions = Object.entries(mdoc.models)
        .sort((a, b) => (b[1].n || 0) - (a[1].n || 0)).slice(0, 12)
        .map(([slug, r]) => ({ slug, m: `${r.b} ${r.m}`, fm: r.fm }));
    }
  } catch (_) { /* a 404 must never depend on a network call */ }
  return html(renderIntlNotFound({
    loc, host: url.host, path: url.pathname, suggestions,
  }), 404);
}

async function intlPrivacy(env, url, loc) {
  await intlCorpus(env, loc);
  return publicHtml(renderIntlPrivacy({ loc, host: url.host }));
}

async function intlLanding(env, url, loc) {
  const c = await intlCorpus(env, loc);
  if (!c) return intlUnavailable(loc, url.host, "info.preparing_models");
  return publicHtml(renderIntlLanding({ loc, host: url.host, stats: c.stats, builtAt: c.builtAt }));
}

async function intlHub(env, url, loc) {
  const c = await intlCorpus(env, loc);
  if (!c) return intlUnavailable(loc, url.host, "info.preparing_hub");
  return publicHtml(renderIntlHub({
    loc, host: url.host, models: c.models, builtAt: c.builtAt, stats: c.stats,
  }));
}

async function intlTrustPage(env, url, loc, kind) {
  const c = await intlCorpus(env, loc);
  const stats = c ? c.stats : null;
  const builtAt = c ? c.builtAt : null;
  const page = kind === "method"
    ? renderIntlMethodology({ loc, host: url.host, stats, builtAt })
    : renderIntlAbout({ loc, host: url.host, stats, builtAt });
  return publicHtml(page);
}

async function intlModel(env, url, loc, tail) {
  let rest;
  try {
    rest = decodeURIComponent(tail).replace(/\/+$/, "").toLowerCase();
  } catch (_) {
    return intlNotFound(env, url, loc);
  }
  const wantsJson = rest.endsWith(".json");
  if (wantsJson) rest = rest.slice(0, -".json".length);
  let slug = rest, year = null;
  const slash = rest.lastIndexOf("/");
  if (slash > 0) {
    const seg = rest.slice(slash + 1);
    slug = rest.slice(0, slash);
    if (!/^\d{4}$/.test(seg)) return intlNotFound(env, url, loc);
    year = parseInt(seg, 10);
  }
  const c = await intlCorpus(env, loc);
  const rec = c ? c.models[slug] : null;
  if (!rec) return intlNotFound(env, url, loc);
  const modelUrl = `https://${url.host}${ihref(loc, "model", slug)}`;
  if (year != null) {
    const published = intlPublishedYears(loc, c.models, slug, rec, c.builtAt);
    const cell = published.includes(year) ? intlYearCell(rec, year) : null;
    if (!cell) {
      if (published.length && (rec.yr || []).some(x => x.y === year)) {
        return new Response(null, {
          status: 301,
          headers: { location: `${modelUrl}${wantsJson ? ".json" : ""}` },
        });
      }
      return intlNotFound(env, url, loc);
    }
    if (wantsJson) {
      return jsonResponse(intlYearJson(loc, rec, slug, year, cell, { host: url.host, builtAt: c.builtAt }));
    }
    return publicHtml(renderIntlYearPage({
      loc, host: url.host, models: c.models, rec, slug, year, cell, stats: c.stats, builtAt: c.builtAt,
    }));
  }
  if (wantsJson) {
    return jsonResponse(intlModelJson(loc, rec, slug, {
      host: url.host, builtAt: c.builtAt, models: c.models,
    }));
  }
  await primeAltMarkets(env, loc.code, c.models);
  return publicHtml(renderIntlModelPage({
    loc, host: url.host, models: c.models, rec, slug, builtAt: c.builtAt, stats: c.stats,
    siblings: intlSiblings(c.models, slug, rec),
    extras: intlModelCutLinks(loc, rec, slug, c.mdoc && c.mdoc.districts,
      intlInWave(loc, c.models, slug, c.builtAt))
      + intlModelCurveLinks(loc, c.models, rec, slug, c.builtAt),
  }));
}

async function intlAvaliar(env, url, loc) {
  const query = (url.searchParams.get("q") || "").toString().trim();
  const modelo = (url.searchParams.get("modelo") || "").toString().trim().toLowerCase();
  const anoRaw = parseInt(url.searchParams.get("ano") || "", 10);
  const ano = Number.isFinite(anoRaw) ? anoRaw : null;

  let rec = null, carId = null, sourceUrl = null;
  if (query) {
    carId = parseAs24Id(query);
    if (/^https?:\/\//i.test(query)) sourceUrl = query;
    if (carId) {
      const cars = await getValuations(env, loc.country);
      rec = cars ? (cars[valuationKey(loc, carId)] || null) : null;
    }
  }
  const c = await intlCorpus(env, loc);
  const models = c ? c.models : null;
  let spec = null;
  if (!rec && modelo && models && models[modelo]) {
    const mrec = models[modelo];
    spec = { rec: mrec, slug: modelo, year: ano, cell: pickYearCell(mrec, ano) };
  }
  return html(renderIntlAvaliar({
    loc, host: url.host, models, builtAt: c && c.builtAt, stats: c && c.stats,
    rec, carId, sourceUrl, query, spec,
  }), 200);
}

async function intlFeed(env, url, loc) {
  const { deals, degraded, builtAt } = await getDeals(env, "all", loc.country);
  if (degraded) return intlUnavailable(loc, url.host, "info.deals_unavailable");
  const c = await intlCorpus(env, loc);
  const districts = (c && c.mdoc && c.mdoc.districts) || null;
  return publicHtml(renderIntlFeed({
    loc, host: url.host, deals: sortDeals(deals || [], "score"),
    builtAt: builtAt || (c && c.builtAt), models: c && c.models,
    districts, region: (url.searchParams.get("region") || "").toLowerCase() || null,
  }));
}

async function intlCar(env, url, loc) {
  const id = (url.searchParams.get("olx_id") || "").toString();
  const { deals, degraded, builtAt } = await getDeals(env, "all", loc.country);
  if (degraded) return intlUnavailable(loc, url.host, "info.deals_unavailable");
  const deal = (deals || []).find(d => String(d.olx_id) === id);
  if (!deal) return redirect(ihref(loc, "mercado"), 302);
  const doc = await getValuations(env, loc.country);
  const cars = (doc && doc.cars) ? doc.cars : doc;
  const rec = (cars && cars[String(deal.olx_id)]) || null;
  const c = await intlCorpus(env, loc);
  return publicHtml(renderIntlCar({
    loc, host: url.host, deal, rec, builtAt: builtAt || (c && c.builtAt),
  }));
}

async function intlArchive(env, url, loc, tail) {
  if (!tail) {
    const weeks = await snapshotWeeks(env, loc.country);
    return publicHtml(renderIntlArchive({ loc, host: url.host, weeks }));
  }
  let rest;
  try { rest = decodeURIComponent(tail).replace(/\/+$/, "").toLowerCase(); }
  catch (_) { return intlNotFound(env, url, loc); }
  if (!rest.endsWith(".json")) return intlNotFound(env, url, loc);
  const parts = rest.slice(0, -".json".length).split("/");
  if (parts.length > 2) return intlNotFound(env, url, loc);
  const week = weekToken(parts[0]);
  if (!week) return intlNotFound(env, url, loc);
  const snap = await readSnapshot(env, week, loc.country);
  if (!snap) return intlNotFound(env, url, loc);
  const base = {
    market: loc.country, language: loc.code,
    week: snap.week, date: snap.date, built_at: snap.builtAt,
    source: `https://${url.host}${ihref(loc, "arquivo")}/${parts[0]}.json`,
    note: it(loc, "arch.note"),
  };
  const frozen = payload => {
    const res = jsonResponse(payload);
    res.headers.set("cache-control", "public, max-age=31536000, immutable");
    return res;
  };
  if (parts.length === 1) return frozen({ ...base, models: snap.models || {} });
  const rec = (snap.models || {})[parts[1]];
  if (!rec) return intlNotFound(env, url, loc);
  return frozen({
    ...base,
    source: `https://${url.host}${ihref(loc, "arquivo")}/${parts[0]}/${parts[1]}.json`,
    slug: parts[1], ...rec,
  });
}

async function intlMarketIndex(env, url, loc, tail) {
  const { week, history } = await recordWeeklyIndex(env, new Date(), "web", loc.country);
  const months = monthlyCuts(history, week);
  const rest = String(tail || "").replace(/\/+$/, "").toLowerCase();
  if (rest) {
    const mm = /^(\d{4})-(\d{2})$/.exec(rest);
    if (mm) {
      const cut = months.find(c => c.month === rest);
      if (!cut) return intlNotFound(env, url, loc);
      return publicHtml(renderIntlMarketIndex({
        loc, host: url.host, snapshot: null, history, months, month: cut,
      }));
    }
    const pinned = weekToken(rest);
    const row = pinned ? history.find(h => h.week === pinned) : null;
    if (!row) return intlNotFound(env, url, loc);
    return publicHtml(renderIntlMarketIndex({
      loc, host: url.host, snapshot: row, history: [], months: [], pinned,
    }));
  }
  return publicHtml(renderIntlMarketIndex({
    loc, host: url.host, snapshot: history[history.length - 1] || null,
    history, gaps: missingWeeks(history, week), months,
  }));
}

async function intlSitemap(env, url, loc) {
  const base = `https://${url.host}`;
  const mdoc = await getModels(env, loc.country);
  const models = (mdoc && mdoc.models) || null;
  const lastmod = ((mdoc && mdoc.built_at) || "").slice(0, 10);
  const lm = /^\d{4}-\d{2}-\d{2}$/.test(lastmod) ? `<lastmod>${lastmod}</lastmod>` : "";
  let hasDeals = false;
  try {
    const feed = await getDeals(env, "all", loc.country);
    hasDeals = !feed.degraded && Array.isArray(feed.deals) && feed.deals.length > 0;
  } catch (_) { hasDeals = false; }
  const paths = intlSitemapPaths(loc, models, hasDeals, (mdoc && mdoc.built_at) || null);
  const weeks = await snapshotWeeks(env, loc.country);
  if (weeks.length) {
    paths.push({ path: ihref(loc, "arquivo"), freq: "weekly", prio: "0.5" });
    paths.push({ path: ihref(loc, "indice"), freq: "weekly", prio: "0.6" });
  }
  if (models) {
    for (const mod of intlPageModules()) {
      if (typeof mod.sitemap !== "function") continue;
      try {
        for (const u of mod.sitemap(loc, models, (mdoc && mdoc.built_at) || null, mdoc) || []) {
          if (u && typeof u.path === "string" && u.path) paths.push(u);
        }
      } catch (err) {
        console.warn(`intl sitemap ${mod.id} error`, err && err.message);
      }
    }
  }
  const seen = new Set();
  const urls = paths.filter(u => {
    if (seen.has(u.path)) return false;
    seen.add(u.path);
    return true;
  }).map(u =>
    `<url><loc>${base}${u.path}</loc>${lm}<changefreq>${u.freq}</changefreq><priority>${u.prio}</priority></url>`);
  const xml = `<?xml version="1.0" encoding="UTF-8"?>\n`
    + `<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n${urls.join("\n")}\n</urlset>`;
  return new Response(xml, {
    status: 200,
    headers: { "content-type": "application/xml; charset=utf-8", "cache-control": "public, max-age=3600" },
  });
}

// Models hub (/pt/precos) — the crawl spine linking every model page.
async function handleModelsHub(request, env, url) {
  const mdoc = await getModels(env);
  const models = mdoc && mdoc.models;
  if (!models) {
    return html(renderInfo({
      zone: "all", depositCount: null, title: "Serviço indisponível",
      message: "Os preços por modelo estão a ser preparados. Volta dentro de instantes.",
    }), 503);
  }
  const list = Object.entries(models)
    .map(([s, r]) => ({
      slug: s, b: r.b, m: r.m, fm: r.fm, n: r.n,
      ys: publishedYearPages(models, s, r, mdoc.built_at).slice().sort((x, y) => y - x),
    }))
    .sort((a, b) => (b.n || 0) - (a.n || 0));
  // District pages hang off this hub. Without the row they would exist only in
  // the sitemap — crawlable in principle, orphaned in practice.
  const districts = Object.entries((mdoc && mdoc.districts) || {})
    .map(([k, d]) => ({ k, lbl: d.lbl, n: d.n, fm: d.fm }))
    .sort((a, b) => (b.n || 0) - (a.n || 0));
  return publicHtml(renderModelsHub({
    models: list, depositCount: null, builtAt: mdoc.built_at, host: url.host, districts,
  }));
}

// /sitemap.xml — every indexable URL, generated from the same selection
// functions the router uses.
//
// Sharing those functions is not tidiness, it is the whole correctness argument:
// if the sitemap advertised a model-year the router 404s (or the router served
// one the sitemap never listed), Search Console would report it as a site-wide
// error and the crawler would learn to distrust the file. One source, both
// consumers. Degrades to the static set (never 500) if models.json is missing.
async function handleSitemapIndex(request, env, url) {
  const base = `https://${url.host}`;
  const markets = [{ prefix: PT_PREFIX, country: null },
                   ...liveLocales().map(l => ({ prefix: l.prefix, country: l.country }))];
  const rows = [];
  for (const m of markets) {
    let built = "";
    try {
      const mdoc = await getModels(env, m.country);
      built = String((mdoc && mdoc.built_at) || "").slice(0, 10);
    } catch (_) { built = ""; }
    const lm = /^\d{4}-\d{2}-\d{2}$/.test(built) ? `<lastmod>${built}</lastmod>` : "";
    rows.push(`<sitemap><loc>${base}${m.prefix}/sitemap.xml</loc>${lm}</sitemap>`);
  }
  const xml = `<?xml version="1.0" encoding="UTF-8"?>\n`
    + `<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n${rows.join("\n")}\n</sitemapindex>`;
  return new Response(xml, {
    status: 200,
    headers: { "content-type": "application/xml; charset=utf-8", "cache-control": "public, max-age=3600" },
  });
}

async function handleSitemap(request, env, url) {
  const base = `https://${url.host}`;
  const mdoc = await getModels(env);
  const models = mdoc && mdoc.models;
  // Real content-change stamp: the models.json build date, NOT the request date.
  // A request-time "today" on every URL makes lastmod a lie Google learns to
  // ignore. Emit <lastmod> only when we actually have a build stamp.
  const lastmodSrc = (mdoc && mdoc.built_at) || "";
  const lastmod = lastmodSrc.slice(0, 10);
  const lm = lastmod ? `<lastmod>${lastmod}</lastmod>` : "";
  const iso = d => {
    const v = String(d || "").slice(0, 10);
    return /^\d{4}-\d{2}-\d{2}$/.test(v) ? v : null;
  };
  const stamp = d => {
    if (d === null) return "";
    const v = (Array.isArray(d) ? d : [d]).map(iso).find(Boolean);
    if (v) return `<lastmod>${v}</lastmod>`;
    return Array.isArray(d) ? "" : lm;
  };
  const urls = [];
  const add = (path, freq, prio, when) =>
    urls.push(`<url><loc>${base}${path}</loc>${when === undefined ? lm : stamp(when)}<changefreq>${freq}</changefreq><priority>${prio}</priority></url>`);

  add("/pt", "daily", "1.0");
  add("/pt/mercado", "daily", "0.9");
  add("/pt/avaliar", "weekly", "0.8");
  add("/pt/precos", "weekly", "0.7");
  add("/pt/mercado/indice", "weekly", "0.7");
  // Trust pages: they change rarely but they are what an evaluator looks for.
  add("/pt/historico", "weekly", "0.5");
  add("/pt/metodologia", "monthly", "0.6");
  add("/pt/sobre", "monthly", "0.6");
  add("/pt/isv", "monthly", "0.6");
  // Consent banner links here, so it has to be crawlable and listed.
  add("/pt/privacidade", "yearly", "0.2", null);

  if (models) {
    add("/pt/depreciacao", "weekly", "0.7");
    add("/pt/comparar", "weekly", "0.7");
    for (const d of Object.values(DUELS)) {
      if (duelSlugs(models, d.kind, lastmodSrc).length) add(`/pt/${d.path}`, "weekly", "0.7");
    }
    add("/pt/liquidez", "weekly", "0.7");
    add("/pt/vender", "weekly", "0.7");
    add("/pt/guias", "monthly", "0.6");
    for (const g of GUIDES) add(`/pt/guias/${g.slug}`, "monthly", "0.6", GUIDES_UPDATED);
    add("/pt/sobrevalorizados", "weekly", "0.6");

    for (const [slug, rec] of Object.entries(models)) {
      add(`/pt/preco/${encodeURIComponent(slug)}`, "daily", "0.6", rec.u);
      // Model-year pages — only the ones that clear the publishing floor, which
      // is exactly the set handleModelPage will serve.
      for (const y of publishedYearPages(models, slug, rec, lastmodSrc)) {
        const c = yearCell(rec, y);
        add(`/pt/preco/${encodeURIComponent(slug)}/${y}`, "daily", "0.5", (c && c.u) || rec.u);
      }
      // Fuel / district facets, where the sample supports them.
      for (const k of publishedFacets(models, slug, rec, lastmodSrc)) {
        add(`/pt/preco/${encodeURIComponent(slug)}/${encodeURIComponent(k)}`, "daily", "0.5", rec.u);
      }
      if (publishedDepreciation(models, slug, rec, lastmodSrc)) add(`/pt/depreciacao/${encodeURIComponent(slug)}`, "weekly", "0.5", rec.u);
      if (publishedLiquidity(models, slug, rec, lastmodSrc)) add(`/pt/liquidez/${encodeURIComponent(slug)}`, "weekly", "0.5", rec.u);
      if (publishedVender(models, slug, rec, lastmodSrc)) add(`/pt/vender/${encodeURIComponent(slug)}`, "weekly", "0.6", rec.u);
      for (const d of Object.values(DUELS)) {
        if (publishedDuel(models, slug, rec, lastmodSrc, d.kind)) add(`/pt/${d.path}/${encodeURIComponent(slug)}`, "weekly", "0.5", rec.u);
      }
    }
    for (const k of Object.keys((mdoc && mdoc.districts) || {})) {
      add(`/pt/precos/${encodeURIComponent(k)}`, "weekly", "0.6");
    }
    for (const [a, b] of publishedPairs(models)) {
      const pairStamp = [models[a] && models[a].u, models[b] && models[b].u].filter(Boolean).sort().pop();
      add(`/pt/comparar/${encodeURIComponent(a)}-vs-${encodeURIComponent(b)}`, "weekly", "0.5", pairStamp);
    }
    // Archived index weeks: permanent URLs, so they belong in the sitemap.
    try {
      const history = await env.KV.get(IDX_LIST_KEY, "json");
      if (Array.isArray(history)) {
        // The current week is still exactly what /pt/mercado/indice serves, so listing
        // it here is what created the duplicate. It joins the sitemap next week,
        // when it has closed and become a permanent cut.
        const liveWeek = isoWeek(new Date());
        for (const h of history.slice(-52)) {
          if (h.week === liveWeek) continue;
          add(`/pt/mercado/indice/${h.week.toLowerCase()}`, "yearly", "0.3", [h.builtAt, h.date]);
        }
        for (const c of monthlyCuts(history, liveWeek).slice(-24)) {
          const last = c.rows && c.rows.length ? c.rows[c.rows.length - 1] : null;
          add(`/pt/mercado/indice/${c.month}`, "yearly", "0.4", [c.builtAt, last && last.date, c.to]);
        }
      }
    } catch (_) { /* archive is optional */ }
  }

  try {
    const idoc = await getImports(env);
    if (idoc) {
      const slugs = importSlugs(idoc);
      if (slugs.length) {
        add("/pt/importar", "weekly", "0.7");
        for (const slug of slugs) add(`/pt/importar/${encodeURIComponent(slug)}`, "weekly", "0.6");
      }
    }
  } catch (_) { /* the import layer is optional */ }

  const xml = `<?xml version="1.0" encoding="UTF-8"?>\n`
    + `<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n${urls.join("\n")}\n</urlset>`;
  return new Response(xml, {
    status: 200,
    headers: { "content-type": "application/xml; charset=utf-8", "cache-control": "public, max-age=3600" },
  });
}

// /_olx — narrow egress relay for the scraper.
//
// OLX sits behind CloudFront and its WAF answers "Request blocked" to both the
// scrape host's address and a GitHub-hosted one, so the scraper has no clean
// address of its own. This forwards a single, tightly-shaped request from
// Cloudflare's network instead.
//
// Deliberately not a general proxy, because a general proxy on a public domain
// is an open relay someone else will find and use:
//   * requires RELAY_TOKEN, compared in constant time;
//   * GET only;
//   * one hardcoded origin and path prefix — the offers API, nothing else;
//   * query string is forwarded verbatim, everything else is dropped.
// Absent RELAY_TOKEN the route does not exist at all (404), so deploying this
// without setting the secret changes nothing.
const RELAY_ORIGIN = "https://www.olx.pt";
// Two prefixes, both read-only and both needed by the scrape path:
//   /api/v1/offers — the listings API the scraper enumerates
//   /d/anuncio/    — a listing page, fetched only to read its og:image URLs
// Everything else is refused. Keep this list minimal: each entry widens what
// a leaked token could reach.
const RELAY_PATH_PREFIXES = ["/api/v1/offers", "/d/anuncio/"];

async function handleOlxRelay(request, env, url) {
  const expected = (env.RELAY_TOKEN || "").trim();
  if (!expected) return notFound();
  const given = request.headers.get("X-Relay-Token") || "";
  if (given.length !== expected.length || !constantTimeEqStr(given, expected)) {
    return forbidden();
  }
  const path = url.searchParams.get("path") || "";
  if (!RELAY_PATH_PREFIXES.some(pre => path.startsWith(pre))) return forbidden();
  const target = new URL(RELAY_ORIGIN + path);
  const upstream = await fetch(target.toString(), {
    method: "GET",
    headers: {
      "User-Agent": request.headers.get("X-Relay-UA")
        || "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
           + "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36",
      "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8",
      "Accept-Language": "pt-PT,pt;q=0.9,en;q=0.5",
    },
    // No edge caching: the scraper needs what OLX says now, and a cached 403
    // would be worse than the 403 itself.
    cf: { cacheEverything: false },
  });
  return new Response(upstream.body, {
    status: upstream.status,
    headers: {
      "content-type": upstream.headers.get("content-type") || "text/plain",
      "cache-control": "no-store",
    },
  });
}

function constantTimeEqStr(a, b) {
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}

// /llms.txt — a plain-language map of the site for tool-using models.
//
// Deliberately modest expectations: the 2026 evidence is that no engine ranks
// or cites on the strength of this file, so it is not an SEO lever. What it
// does do is answer, in one fetch, the question an agent actually has — what
// is here, how current is it, what may I quote — and that is cheap to serve.
//
// It is generated, not a static asset, so the figures cannot drift away from
// what the pages show: the model count and freshness come from the same blob
// the pages render from.
async function handleLlmsTxt(request, env, url) {
  const mdoc = await getModels(env);
  const models = (mdoc && mdoc.models) || null;
  const count = models ? Object.keys(models).length : 0;
  const built = (mdoc && mdoc.built_at) ? String(mdoc.built_at).slice(0, 10) : null;
  const base = `https://${url.host}`;
  const duelHubs = models ? Object.values(DUELS).filter(d => duelSlugs(models, d.kind, mdoc && mdoc.built_at).length) : [];
  const wave = models ? waveSlugs(models, mdoc && mdoc.built_at) : null;
  const waveCount = wave ? wave.size : 0;
  const liqWave = models ? liqWaveSlugs(models, mdoc && mdoc.built_at) : null;
  const liqCount = liqWave ? liqWave.size : 0;
  // A handful of the best-covered models, so an agent has real entry points
  // rather than a bare index it would have to crawl to be useful.
  const top = models
    ? Object.entries(models)
        .sort((a, b) => (b[1].n || 0) - (a[1].n || 0))
        .slice(0, 12)
        .map(([slug, r]) => `- [${r.b} ${r.m}](${base}/pt/preco/${slug}): mediana pedida €${r.fm}, ${r.n} anúncios ativos`)
    : [];
  const body = [
    "# Carsbuyer",
    "",
    "> Avaliação independente de carros usados em Portugal. Preços medianos e",
    "> intervalos calculados a partir de anúncios ativos do OLX Portugal,",
    "> recolhidos e atualizados diariamente por nós.",
    "",
    "## O que estes números são",
    "",
    "- Preços **pedidos** em anúncios ativos, não preços de venda fechados.",
    "- Mediana e intervalo interquartil (P25-P75) por modelo e por ano.",
    "- Dias medianos até vender, quando há amostra de vendas suficiente.",
    "- Valor justo estimado por um modelo GBM treinado nos mesmos dados,",
    "  para quilometragem e specs típicas do modelo — não para uma viatura concreta.",
    count ? `- Cobertura: ${count} modelos.` : null,
    built ? `- Dados atualizados a ${built}.` : null,
    "",
    "## Atribuição",
    "",
    "Os números podem ser citados com atribuição a Carsbuyer (" + base + ")",
    "e indicação da data, porque mudam diariamente.",
    "",
    "## Páginas",
    "",
    `- [Índice de preços por modelo](${base}/pt/precos)`,
    `- [Avaliar um anúncio concreto](${base}/pt/avaliar)`,
    `- [Mercado: carros abaixo do valor justo](${base}/pt/mercado)`,
    `- [Índice do mercado, com arquivo semanal e mensal permanente](${base}/pt/mercado/indice)`,
    `- [Arquivo datado por modelo: os números como estavam nessa semana](${base}/pt/historico)`,
    `- [Desvalorização por modelo](${base}/pt/depreciacao)`,
    `- [Quanto tempo demora a vender cada modelo](${base}/pt/liquidez)`,
    `- [Vender: quanto pedir por modelo e em quantos dias vende](${base}/pt/vender)`,
    `- [Guias para vender um carro usado: documentos, registo, crédito, importados, burlas](${base}/pt/guias)`,
    `- [Comparações diretas entre modelos](${base}/pt/comparar)`,
    ...duelHubs.map(d => `- [${d.hubTitle}](${base}/pt/${d.path})`),
    `- [Preço pedido vs. valor justo estimado](${base}/pt/sobrevalorizados)`,
    `- [Importar da Alemanha: em que modelos a conta fecha](${base}/pt/importar)`,
    `- [Simulador de ISV](${base}/pt/isv)`,
    `- [Metodologia](${base}/pt/metodologia)`,
    `- [Quem somos](${base}/pt/sobre)`,
    `- [Sitemap](${base}/pt/sitemap.xml)`,
    "",
    "## Estrutura dos endereços",
    "",
    "Constrói o endereço em vez de percorrer o índice. `{slug}` é",
    "`marca-modelo` sem acentos e em minúsculas (`volkswagen-golf`,",
    "`alfa-romeo-giulietta`); `{ano}` são quatro dígitos.",
    "",
    `- \`${base}/pt/preco/{slug}\` — preços de um modelo, por ano`,
    `- \`${base}/pt/preco/{slug}/{ano}\` — um modelo num ano concreto (a partir de 10 anúncios ativos nesse ano${waveCount ? ", nos modelos já publicados: ver \"Publicação por vagas\"" : ""})`,
    `- \`${base}/pt/preco/{slug}/{combustivel}\` — o mesmo modelo só em diesel, gasolina ou GPL`,
    `- \`${base}/pt/preco/{slug}/{caixa}\` — o mesmo modelo só com caixa manual ou automática`,
    ...duelHubs.map(d => `- \`${base}/pt/${d.path}/{slug}\` — ${d.question}: qual segura melhor o preço desse modelo (+ .json)`),
    `- \`${base}/pt/preco/{slug}/{distrito}\` — o mesmo modelo num distrito`,
    `- \`${base}/pt/precos/{distrito}\` — o mercado de um distrito`,
    `- \`${base}/pt/depreciacao/{slug}\` — curva de desvalorização, custo de cada ano de idade e onde a queda abranda (existe onde há histórico suficiente)`,
    `- \`${base}/pt/liquidez/{slug}\` — quanto tempo esse modelo demora a sair do OLX: percentagem que sai em 30/60/90 dias, mediana, e os mesmos cortes por faixa de preço, idade e distrito (+ .json)`,
    `- \`${base}/pt/importar/{slug}\` — preço pedido na Alemanha + ISV + legalização contra o preço pedido em Portugal, ano a ano (+ .json)`,
    `- \`${base}/pt/comparar/{slug-a}-vs-{slug-b}\` — comparação entre dois modelos`,
    `- \`${base}/pt/mercado/indice/{AAAA}-W{SS}\` — corte semanal permanente do mercado`,
    `- \`${base}/pt/mercado/indice/{AAAA}-{MM}\` — corte mensal permanente, mediana dos cortes semanais desse mês`,
    `- \`${base}/pt/historico/{AAAA}-w{SS}.json\` — corte semanal congelado de todos os modelos`,
    `- \`${base}/pt/historico/{AAAA}-w{SS}/{slug}.json\` — um modelo como estava nessa semana`,
    "",
    "## Como citar um número sem que envelheça",
    "",
    "As páginas de preço mostram o mercado de hoje e hoje muda: a mediana que",
    "citares esta semana já não é a que lá está na próxima. Isso não torna a",
    "citação falsa, torna-a sem data — e sem data parece errada.",
    "",
    `Para citar um valor estável usa o arquivo: [${base}/pt/historico](${base}/pt/historico).`,
    "Cada semana fica congelada e o seu endereço não volta a mudar. Cada",
    "resposta traz `week`, `date` e `built_at`, por isso a data da citação não",
    "depende de quem cita.",
    "",
    "Por modelo o corte tem mediana pedida, intervalo interquartil, anúncios",
    "ativos e quilometragem mediana, e o mesmo por ano do carro.",
    "",
    waveCount ? "" : null,
    waveCount ? "## Publicação por vagas" : null,
    waveCount ? "" : null,
    waveCount ? `As páginas por ano, por corte (combustível, caixa, distrito) e de` : null,
    waveCount ? `desvalorização não existem para todos os modelos ao mesmo tempo: são` : null,
    waveCount ? `publicadas por vagas e neste momento existem para os **${waveCount} modelos**` : null,
    waveCount ? `com mais anúncios ativos. Nos restantes, um ano com amostra suficiente` : null,
    waveCount ? `devolve **404** — mas os números desse ano estão na mesma em` : null,
    waveCount ? `\`${base}/pt/preco/{slug}.json\`, no campo \`by_year\`, com \`page: null\`.` : null,
    waveCount ? `O mesmo vale para \`by_fuel\`, \`by_transmission\` e \`by_district\`.` : null,
    waveCount ? `Constrói o endereço a partir de \`page\`, não a partir do padrão.` : null,
    waveCount ? `` : null,
    liqCount ? `## Publicação por vagas (liquidez)` : null,
    liqCount ? `` : null,
    liqCount ? `\`${base}/pt/liquidez/{slug}\` tem uma vaga própria, separada da de cima:` : null,
    liqCount ? `${liqCount} modelos publicados. O número de dias até vender aparece na mesma` : null,
    liqCount ? `na página do modelo, mesmo quando o modelo ainda não tem página de liquidez.` : null,
    liqCount ? `` : null,
    waveCount ? "" : null,
    "## Dados em JSON",
    "",
    "Cada página de modelo e de modelo-ano tem uma versão JSON no mesmo",
    "endereço com o sufixo `.json`, ligada no HTML por",
    "`<link rel=\"alternate\" type=\"application/json\">`. Traz os mesmos números",
    "com nomes por extenso, mais o tamanho da amostra e a data de recolha.",
    "",
    `- \`${base}/pt/preco/{slug}.json\``,
    `- \`${base}/pt/preco/{slug}/{ano}.json\``,
    `- \`${base}/pt/preco/{slug}/{combustivel|caixa|distrito}.json\` — o mesmo corte, com a razão contra o modelo controlada pela idade (ano a ano onde a amostra chega, anúncio a anúncio onde não chega)`,
    `- \`${base}/pt/depreciacao/{slug}.json\` — taxa anual, meia-vida do valor, custo de um ano de idade por idade, e se a taxa quebra em alguma idade`,
    `- \`${base}/pt/liquidez/{slug}.json\` — dias até sair do anúncio, percentagem que sai em 30/60/90 dias, quantos voltam a ser anunciados e quanto se costuma baixar no preço`,
    "",
    waveCount
      ? "Um endereço que não exista devolve 404: ou a amostra é fina demais, ou o modelo ainda não entrou na vaga de publicação."
      : "Um endereço que não exista devolve 404 — não há páginas geradas para combinações sem amostra suficiente.",
    top.length ? "" : null,
    top.length ? "## Modelos com mais dados" : null,
    top.length ? "" : null,
    ...top,
    ...llmsLocaleSection(base),
    "",
  ].filter(v => v !== null).join("\n");
  return new Response(body, {
    status: 200,
    headers: {
      "content-type": "text/plain; charset=utf-8",
      "cache-control": "public, max-age=3600",
    },
  });
}

function llmsLocaleSection(base) {
  const live = liveLocales();
  if (!live.length) return [];
  return [
    "",
    "## Outros mercados",
    "",
    "O mesmo produto, com dados do mercado local e em língua local:",
    "",
    ...live.map(l =>
      `- [${l.countryName} — ${l.source.name}](${base}${ihref(l, "hub")}): ${it(l, "hub.h1", { country: l.countryName })} (${l.lang})`),
    `- Cada mercado tem o seu sitemap em \`${base}/{cc}/sitemap.xml\`.`,
  ];
}

// /robots.txt — allow public, block transactional/internal, point at the sitemap.
const ROBOTS_RULES = [
  "Allow: /",
  "Disallow: /analytics", "Disallow: /_olx", "Disallow: /geo", "Disallow: /pt/lead", "Disallow: /pt/ir/",
];

function geoJson(request) {
  const raw = (request.cf && request.cf.country) || request.headers.get("cf-ipcountry") || "";
  const cc = /^[A-Z]{2}$/.test(raw) ? raw : null;
  return new Response(JSON.stringify({ c: cc }), {
    status: 200,
    headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" },
  });
}

async function handleRobots(request, env, url) {
  const body = [
    "User-agent: *", ...ROBOTS_RULES,
    // /pt/widget stays crawlable on purpose: it is noindex,follow and links back to
    // the canonical /pt/preco page, so it works as a backlink lever when embedded.
    "",
    // Answer engines are named explicitly rather than left to the wildcard.
    // The wildcard already allows them, but naming them states the intent so a
    // later tightening of `*` cannot silently cut off AI citations — the one
    // distribution channel where being the ORIGINAL source of the numbers
    // (median asking price, IQR, days-to-sell, per-year table) is the whole
    // advantage. Blocking these is how sites vanish from AI answers.
    "User-agent: GPTBot", ...ROBOTS_RULES,          // OpenAI crawler (training/index)
    "User-agent: OAI-SearchBot", ...ROBOTS_RULES,   // OpenAI, powers ChatGPT search
    "User-agent: ChatGPT-User", ...ROBOTS_RULES,    // live fetch on a user's request
    "User-agent: PerplexityBot", ...ROBOTS_RULES,
    "User-agent: ClaudeBot", ...ROBOTS_RULES,
    "User-agent: Google-Extended", ...ROBOTS_RULES, // Gemini / AI Overviews grounding
    "User-agent: CCBot", ...ROBOTS_RULES,
    "",
    `Sitemap: https://${url.host}/sitemap.xml`,
    ...liveLocales().map(l => `Sitemap: https://${url.host}${l.prefix}/sitemap.xml`),
    // Not a search-ranking signal — no engine ranks on it. It is an
    // agent-readiness convenience: a single fetch that tells a tool-using
    // model what this site holds and how to reach it.
    `LLM-Content: https://${url.host}/llms.txt`, "",
  ].join("\n");
  return new Response(body, {
    status: 200,
    headers: { "content-type": "text/plain; charset=utf-8", "cache-control": "public, max-age=3600" },
  });
}

// Mercado feed (/pt/mercado) — the grid of car tiles, zone + sort filtered.
async function handleFeed(request, env, url) {
  const zone = pickZone(url.searchParams.get("zone"));
  const view = pickView(url.searchParams.get("view"));
  // Fetch every zone in parallel so the filter chips can show live per-zone
  // counts ("Norte (12)"). Each getDeals call is edge/KV-cached, so the extra
  // three fetches are cheap. The current zone's result drives the feed itself.
  // Per-zone .catch: a sibling zone failing (e.g. a KV.get reject — getDeals
  // only guards the fetch, not the cache read) must NOT take down the whole
  // feed via Promise.all. A failed sibling → null count; a failed current zone
  // → degraded:true → the graceful 503 below, never an uncaught 500.
  const zoneResults = Object.fromEntries(
    await Promise.all(ZONES.map(z =>
      getDeals(env, z).then(r => [z, r]).catch(() => [z, { deals: [], degraded: true }]))));
  const zoneCounts = {};
  for (const z of ZONES) {
    const r = zoneResults[z];
    zoneCounts[z] = (r && !r.degraded && Array.isArray(r.deals)) ? r.deals.length : null;
  }
  const { deals, degraded, builtAt: feedBuiltAt } = zoneResults[zone] || { deals: [], degraded: true, builtAt: null };

  if (degraded) {
    return html(renderInfo({
      zone, depositCount: null,
      title: "Serviço indisponível",
      message: "Não foi possível carregar os negócios neste momento. Tenta novamente dentro de instantes.",
    }), 503);
  }

  const sort = url.searchParams.get("sort") || "score";
  const sorted = sortDeals(deals, sort);
  if (sorted.length === 0) {
    return html(renderInfo({
      zone, depositCount: null,
      title: "Sem negócios quentes",
      message: "Sem carros abaixo do preço nesta zona agora — a lista renova ao longo do dia. Entretanto, cola o link de qualquer anúncio em /pt/avaliar para saber se está bem cotado.",
    }), 200);
  }

  // Model links for the models on offer right now.
  //
  // Every link the feed emitted pointed at /pt/car?olx_id=… — noindex by design,
  // because the listing vanishes when the car sells. So the site's freshest page
  // passed nothing onward and read, to a crawler, as a static advert. These
  // chips connect it to the stable /pt/preco pages, which is also the next thing a
  // visitor wants to know ("is that a good price for this model?").
  const mdoc = await getModels(env);
  const mmap = (mdoc && mdoc.models) || null;
  const counted = new Map();
  if (mmap) {
    for (const d of sorted) {
      const sl = slugify(`${d.brand}-${d.model}`);
      const rec = mmap[sl];
      if (!rec) continue;
      const prev = counted.get(sl);
      if (prev) prev.count += 1;
      else counted.set(sl, { slug: sl, b: rec.b, m: rec.m, fm: rec.fm, count: 1 });
    }
  }
  const modelLinks = [...counted.values()].sort((a, b) => b.count - a.count || a.b.localeCompare(b.b));

  const mBuiltAt = mdoc && mdoc.built_at;
  const seenYear = new Set(), yearLinks = [];
  const contextLinks = [], districtLinks = [];
  if (mmap) {
    for (const [sl, c] of counted) {
      const rec = mmap[sl];
      const pub = publishedYearPages(mmap, sl, rec, mBuiltAt);
      for (const d of sorted) {
        if (slugify(`${d.brand}-${d.model}`) !== sl) continue;
        const y = Number(d.year);
        if (!pub.includes(y) || seenYear.has(`${sl}/${y}`)) continue;
        seenYear.add(`${sl}/${y}`);
        yearLinks.push({ href: `/pt/preco/${encodeURIComponent(sl)}/${y}`, name: `${rec.b} ${rec.m} ${y}` });
      }
      if (publishedDepreciation(mmap, sl, rec, mBuiltAt)) {
        contextLinks.push({ href: `/pt/depreciacao/${encodeURIComponent(sl)}`,
                            name: `Desvalorização ${rec.b} ${rec.m}` });
      }
      if (publishedLiquidity(mmap, sl, rec, mBuiltAt)) {
        contextLinks.push({ href: `/pt/liquidez/${encodeURIComponent(sl)}`,
                            name: `Tempo de venda ${rec.b} ${rec.m}` });
      }
      void c;
    }
    const dseen = new Set();
    for (const d of sorted) {
      const k = slugify(String(d.district || ""));
      if (!k || dseen.has(k) || !((mdoc && mdoc.districts) || {})[k]) continue;
      dseen.add(k);
      districtLinks.push({ href: `/pt/precos/${encodeURIComponent(k)}`,
                           name: `Preços em ${d.district}` });
    }
  }

  const canonicalView = zone === "all" && sort === "score" && view !== "revender";
  const items = [
    ...modelLinks.slice(0, 24).map(m => ({ href: `/pt/preco/${encodeURIComponent(m.slug)}`,
                                          name: `${m.b} ${m.m}` })),
    ...yearLinks.slice(0, 24), ...contextLinks.slice(0, 24), ...districtLinks.slice(0, 24),
  ];
  const origin = `https://${url.host}`;
  const jsonLd = (canonicalView && items.length) ? {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "CollectionPage",
        "@id": `${origin}/pt/mercado#page`,
        "url": `${origin}/pt/mercado`,
        "name": "Carros usados abaixo do preço em Portugal (OLX)",
        "description": "Carros usados no OLX Portugal abaixo do preço justo de mercado, com desconto, lucro estimado e nota de risco.",
        "inLanguage": "pt-PT",
        "isPartOf": { "@id": `${origin}/#site` },
        ...(feedBuiltAt || mBuiltAt ? { "dateModified": String(feedBuiltAt || mBuiltAt) } : {}),
      },
      {
        "@type": "ItemList",
        "@id": `${origin}/pt/mercado#lista`,
        "name": "Preço de mercado dos modelos com negócios agora",
        "numberOfItems": items.length,
        "itemListElement": items.map((it, i) => ({
          "@type": "ListItem", "position": i + 1, "name": it.name, "url": `${origin}${it.href}`,
        })),
      },
      breadcrumbLd(url.host, [{ name: "Início", href: "/pt" }, { name: "Carros abaixo do preço" }]),
    ],
  } : null;

  return html(renderGrid({
    deals: sorted, zone, sort, view, depositCount: null, zoneCounts,
    host: url.host,
    modelLinks, yearLinks, contextLinks, districtLinks, jsonLd,
    builtAt: mBuiltAt, feedBuiltAt,
  }), 200);
}

// Single-car detail page (opened by clicking a grid tile).
async function handleCar(request, env, url) {
  const zone = pickZone(url.searchParams.get("zone"));
  const view = pickView(url.searchParams.get("view"));
  const olxId = (url.searchParams.get("olx_id") || "").toString();
  const { deals, degraded } = await getDeals(env, zone);
  if (degraded) {
    return html(renderInfo({
      zone, depositCount: null, title: "Serviço indisponível",
      message: "Não foi possível carregar os negócios neste momento. Tenta novamente dentro de instantes.",
    }), 503);
  }
  const deal = (deals || []).find(d => d.olx_id === olxId);
  if (!deal) return redirect(`/pt/mercado?zone=${zone}`, 302);
  // Contextual link into the model SEO page, when this model has one.
  const mdoc = await getModels(env);
  const models = mdoc && mdoc.models;
  const mslug = slugify(`${deal.brand}-${deal.model}`);
  const modelHref = (models && models[mslug]) ? `/pt/preco/${encodeURIComponent(mslug)}` : null;
  return html(renderCarPage({
    deal, zone, view, depositCount: null, modelHref, host: url.host,
    historyUrl: env.HISTORY_REPORT_URL || null,
  }), 200);
}

// ── Internal dashboard / assets — Basic Auth, fail-closed ───────────────────

function checkBasicAuth(request, env) {
  const user = env.ANALYTICS_USER, pass = env.ANALYTICS_PASS;
  if (!user || !pass) return false; // not configured → deny (never expose data)
  const h = request.headers.get("Authorization") || "";
  if (!h.startsWith("Basic ")) return false;
  let decoded;
  try { decoded = atob(h.slice(6)); } catch { return false; }
  const i = decoded.indexOf(":");
  if (i < 0) return false;
  const u = decoded.slice(0, i), p = decoded.slice(i + 1);
  // Bitwise-AND both compares so a length/value mismatch can't short-circuit.
  return Number(constantTimeEq(u, user)) & Number(constantTimeEq(p, pass)) ? true : false;
}

function unauthorized() {
  return new Response("Unauthorized", {
    status: 401,
    headers: { "WWW-Authenticate": 'Basic realm="analytics", charset="UTF-8"' },
  });
}

async function handleAnalytics(request, env, url) {
  if (!checkBasicAuth(request, env)) return unauthorized();
  // Strip the /analytics prefix and forward to ASSETS. /analytics → /, /analytics/X → /X.
  let stripped = url.pathname.replace(/^\/analytics\/?/, "/");
  if (!stripped.startsWith("/")) stripped = "/" + stripped;
  const targetUrl = new URL(stripped + url.search, url.origin);
  return env.ASSETS.fetch(new Request(targetUrl, request));
}

async function handleAssetGated(request, env) {
  if (!checkBasicAuth(request, env)) return unauthorized();
  return env.ASSETS.fetch(request);
}

// ── Deals feed (GitHub Release → KV cache) ──────────────────────────────────

const HOT_DEALS_BASE =
  "https://github.com/nikit34/olx-car-parser/releases/download/latest-data";
const DEALS_CACHE_TTL_SEC = 900;
const DEALS_NEAR_YEARS = 2;

const COMPETITOR_MIN_YEARS = 4;
const COMPETITOR_TOL = 0.6;
const DEGRADED_CACHE_TTL_SEC = 30;

// Returns { deals, degraded }. `degraded: true` means we could not load the
// real feed — surfaced honestly rather than showing stale or fake listings.
async function getDeals(env, zone, country = null) {
  const cc = country ? String(country).toLowerCase() : null;
  const safeZone = cc ? "all" : (ZONES.includes(zone) ? zone : "all");
  const cacheKey = cc ? `cache:deals:${cc}:all` : `cache:deals:${safeZone}`;
  let cached = null;
  try { cached = await env.KV.get(cacheKey); } catch (err) { console.warn("deals cache read failed", err && err.message); }
  if (cached) {
    try {
      const parsed = JSON.parse(cached);
      if (parsed && parsed.__degraded) return { deals: [], degraded: true, builtAt: null };
      if (Array.isArray(parsed.deals)) return { deals: parsed.deals, degraded: false, builtAt: parsed.built_at || null };
    } catch {}
  }
  const url = `${HOT_DEALS_BASE}/${cc ? `hot_deals_${cc}_all` : `hot_deals_${safeZone}`}.json`;
  try {
    const r = await fetch(url, { cf: { cacheTtl: 60, cacheEverything: true } });
    if (!r.ok) {
      console.warn(`hot_deals fetch ${url} → ${r.status}`);
      return degrade(env, cacheKey);
    }
    const body = await r.text();
    let parsed;
    try {
      parsed = JSON.parse(body);
    } catch (e) {
      console.warn(`hot_deals parse fail ${url}`, e && e.message);
      return degrade(env, cacheKey);
    }
    if (!Array.isArray(parsed.deals)) return degrade(env, cacheKey);
    try {
      await env.KV.put(cacheKey, body, { expirationTtl: DEALS_CACHE_TTL_SEC });
    } catch (err) {
      console.warn("deals cache write failed", err && err.message);
    }
    return { deals: parsed.deals, degraded: false, builtAt: parsed.built_at || null };
  } catch (err) {
    console.warn("hot_deals fetch error", err && err.message);
    return degrade(env, cacheKey);
  }
}

async function degrade(env, cacheKey) {
  try {
    await env.KV.put(cacheKey, JSON.stringify({ __degraded: true }),
      { expirationTtl: DEGRADED_CACHE_TTL_SEC });
  } catch (err) {
    console.warn("degrade tombstone write failed", err && err.message);
  }
  return { deals: [], degraded: true, builtAt: null };
}

// valuations.json — the public "value any listing" lookup (Tier-2). ~0.9 MB
// gzipped; fetched from the Release and edge-cached. Parsed per request (the
// /pt/avaliar tool is low-traffic). Returns the {olx_id: rec} map, or null if the
// blob isn't published yet / fetch fails (handler then shows the fallback).
async function getValuations(env, country = null) {
  const cc = country ? String(country).toLowerCase() : null;
  for (const packed of cc ? [false] : [true, false]) {
    const url = `${HOT_DEALS_BASE}/${cc ? `valuations_${cc}` : "valuations"}.json${packed ? ".gz" : ""}`;
    try {
      // Cache only successful responses (cacheTtlByStatus) — never pin a 404 from
      // the pre-publish window, or a transient 5xx, into the edge cache for 10 min.
      const r = await fetch(url, {
        cf: { cacheEverything: true, cacheTtlByStatus: { "200-299": 300, "300-399": 0, "400-499": 0, "500-599": 0 } },
      });
      if (!r.ok) {
        console.warn(`valuations fetch ${url} → ${r.status}`);
        continue;
      }
      const body = packed
        ? new Response(r.body.pipeThrough(new DecompressionStream("gzip")))
        : r;
      const data = await body.json();
      if (data && data.cars) return data.cars;
      console.warn(`valuations ${url} → no cars in the blob`);
    } catch (err) {
      console.warn("valuations fetch error", url, err && err.message);
    }
  }
  return null;
}

// models.json — the per-model SEO blob (Tier-3) for /pt/preco/*, /pt/precos, /sitemap.
// Same edge-cache (success-only) pattern as getValuations; ~50 KB gzipped.
// Returns the full doc { models: {slug: rec}, built_at }, or null (handlers
// then 404/degrade). Callers read `.models`; `.built_at` drives the public
// "preços atualizados em …" freshness line.
const ALT_MARKETS_KEY = "altmarkets:v1";
const ALT_MARKETS_TTL_SEC = 3600;

async function primeAltMarkets(env, ownCode, ownModels) {
  setSlugMarkets(null);
  if (!intlLocales().size) return;
  try {
    const hit = await env.KV.get(ALT_MARKETS_KEY, "json");
    if (hit && hit[ownCode]) {
      setSlugMarkets(Object.fromEntries(Object.entries(hit).map(([k, v]) => [k, new Set(v)])));
      return;
    }
  } catch (err) {
    console.warn("alt markets cache read failed", err && err.message);
  }
  const lists = { [ownCode]: Object.keys(ownModels || {}) };
  const others = [{ code: "pt", country: null },
                  ...liveLocales().map(l => ({ code: l.code, country: l.country }))]
    .filter(m => m.code !== ownCode);
  await Promise.all(others.map(async m => {
    const doc = await getModels(env, m.country);
    if (doc && doc.models) lists[m.code] = Object.keys(doc.models);
  }));
  setSlugMarkets(Object.fromEntries(Object.entries(lists).map(([k, v]) => [k, new Set(v)])));
  try {
    await env.KV.put(ALT_MARKETS_KEY, JSON.stringify(lists), { expirationTtl: ALT_MARKETS_TTL_SEC });
  } catch (err) {
    console.warn("alt markets cache write failed", err && err.message);
  }
}

async function getModels(env, country = null) {
  const cc = country ? String(country).toLowerCase() : null;
  const url = `${HOT_DEALS_BASE}/${cc ? `models_${cc}` : "models"}.json`;
  try {
    const r = await fetch(url, {
      cf: { cacheEverything: true, cacheTtlByStatus: { "200-299": 300, "300-399": 0, "400-499": 0, "500-599": 0 } },
    });
    if (!r.ok) {
      console.warn(`models fetch ${url} → ${r.status}`);
      return null;
    }
    const data = await r.json();
    return data && data.models ? data : null;
  } catch (err) {
    console.warn("models fetch error", err && err.message);
    return null;
  }
}

// Risk-adjusted ranking — same default the old dashboard used. One car at a
// time means there's no sort toggle; we always lead with the best bet.
function sortDeals(deals, sort = "score") {
  const out = [...deals];
  if (sort === "newest") {
    out.sort((a, b) => new Date(b.first_seen_at || 0) - new Date(a.first_seen_at || 0));
  } else if (sort === "profit") {
    out.sort((a, b) => (b.est_profit_eur || 0) - (a.est_profit_eur || 0));
  } else {
    out.sort((a, b) =>
      (b.decision_score || 0) - (a.decision_score || 0)
      || (b.est_profit_eur || 0) - (a.est_profit_eur || 0));
  }
  return out;
}

// ── Small helpers ───────────────────────────────────────────────────────────

function pickZone(z) {
  z = (z || "").toString();
  return ZONES.includes(z) ? z : "all";
}

// Intent lens for the feed/detail: "comprar" (buyer, default) or "revender".
function pickView(v) {
  return (v || "").toString() === "revender" ? "revender" : "comprar";
}

function randomToken(bytes = 16) {
  const buf = new Uint8Array(bytes);
  crypto.getRandomValues(buf);
  return Array.from(buf, b => b.toString(16).padStart(2, "0")).join("");
}

function publicHtml(body, status = 200) {
  return new Response(body, {
    status,
    headers: {
      "Content-Type": "text/html; charset=utf-8",
      "Cache-Control": "public, max-age=300, stale-while-revalidate=86400",
    },
  });
}

function html(body, status = 200, setCookie = null) {
  // `private, no-cache` (not `no-store`) keeps per-visitor pages out of shared
  // caches while staying eligible for the browser back/forward cache — no-store
  // disables bfcache in Chrome, making the mercado→car→Back loop re-fetch/render.
  const headers = { "Content-Type": "text/html; charset=utf-8", "Cache-Control": "private, no-cache" };
  if (setCookie) headers["Set-Cookie"] = setCookie;
  return new Response(body, { status, headers });
}

function redirect(loc, status = 302, setCookie = null) {
  const headers = { "Location": loc };
  if (setCookie) headers["Set-Cookie"] = setCookie;
  return new Response(null, { status, headers });
}

// Local dev hostnames must never be redirected to the public domain.
function isLocalHost(h) {
  return h === "localhost" || h.endsWith(".localhost")
    || h === "127.0.0.1" || h === "::1" || h === "[::1]";
}

function notFound() { return new Response("Not found", { status: 404 }); }
function forbidden() { return new Response("Forbidden", { status: 403 }); }

// CSRF guard for the /pt/lead POST: verify the request came from our own host.
function sameOrigin(request, url) {
  const origin = request.headers.get("Origin");
  if (origin) {
    try { return new URL(origin).host === url.host; } catch { return false; }
  }
  const referer = request.headers.get("Referer");
  if (referer) {
    try { return new URL(referer).host === url.host; } catch { return false; }
  }
  return false;
}

function constantTimeEq(a, b) {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}

const LEAD_TTL_SEC = 90 * 24 * 3600;
const LEAD_RATE_MAX = 5;

async function handleLead(request, env, url) {
  if (!sameOrigin(request, url)) return forbidden();
  let form;
  try { form = await request.formData(); }
  catch { return leadError("O formulário chegou incompleto. Volta atrás e tenta outra vez.", 400); }
  const f = k => (form.get(k) || "").toString().trim();
  const thanks = (name, year) => html(renderLeadThanks({ name, year, depositCount: null, host: url.host }), 200);
  if (f("website")) return thanks("", null);

  const modelo = f("modelo").toLowerCase().slice(0, 60);
  const name = f("nome_modelo").slice(0, 80);
  const ano = parseInt(f("ano"), 10);
  const kmRaw = parseInt(f("km"), 10);
  const km = (Number.isFinite(kmRaw) && kmRaw >= 0 && kmRaw <= 1500000) ? kmRaw : null;
  const distrito = f("distrito").slice(0, 40);
  const contacto = f("contacto").slice(0, 120);
  const nome = f("nome").slice(0, 80);
  const consent = f("consent") === "1" || f("consent") === "on";
  if (!Number.isFinite(ano) || ano < 1980 || ano > 2027) {
    return leadError("Indica o ano do carro, entre 1980 e 2027.", 400);
  }
  const digits = contacto.replace(/\D/g, "");
  const looksPhone = digits.length >= 9 && digits.length <= 15;
  const looksEmail = /^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/.test(contacto);
  if (!looksPhone && !looksEmail) {
    return leadError("Deixa um telemóvel com 9 dígitos ou um email válido, para te podermos enviar as propostas.", 400);
  }
  if (!consent) {
    return leadError("Para enviarmos o teu contacto a compradores precisamos da tua autorização: marca a caixa e volta a enviar.", 400);
  }
  if (modelo && !/^[a-z0-9-]{2,60}$/.test(modelo)) {
    return leadError("O modelo não foi reconhecido. Escolhe-o outra vez na lista.", 400);
  }

  const ip = request.headers.get("cf-connecting-ip") || "0.0.0.0";
  const rlKey = `leadrl:${ip}`;
  let count = 0;
  try { count = parseInt((await env.KV.get(rlKey)) || "0", 10) || 0; } catch {}
  if (count >= LEAD_RATE_MAX) {
    return leadError("Já recebemos vários pedidos deste endereço na última hora. Tenta mais tarde.", 429);
  }
  try { await env.KV.put(rlKey, String(count + 1), { expirationTtl: 3600 }); } catch {}

  const ts = new Date().toISOString();
  const lead = {
    ts, modelo, name, ano, km, distrito, contacto, nome,
    ua: (request.headers.get("user-agent") || "").slice(0, 160),
    country: (request.cf && request.cf.country) || null,
  };
  try {
    await env.KV.put(`lead:${ts}:${randomToken(4)}`, JSON.stringify(lead), { expirationTtl: LEAD_TTL_SEC });
  } catch (err) {
    console.error("lead store failed", err && err.message);
    return leadError("Não conseguimos guardar o pedido. Tenta outra vez dentro de instantes.", 502);
  }
  await notifyLead(env, lead, url.host);
  return thanks(name, ano);
}

function leadError(message, status) {
  return html(renderInfo({ zone: "all", depositCount: null, title: "Falta um dado no pedido", message }), status);
}

async function notifyLead(env, lead, host) {
  const token = env.TELEGRAM_BOT_TOKEN, chat = env.LEADS_TELEGRAM_CHAT_ID;
  if (!token || !chat) return;
  const line = [lead.name || lead.modelo || "modelo?", lead.ano, lead.km != null ? `${lead.km} km` : null, lead.distrito || null]
    .filter(Boolean).join(" · ");
  const text = [
    "🚗 Novo pedido de propostas", line,
    `Contacto: ${lead.contacto}${lead.nome ? ` (${lead.nome})` : ""}`,
    `https://${host}/analytics/leads.json`,
  ].join("\n");
  try {
    const r = await fetch(`https://api.telegram.org/bot${token}/sendMessage`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ chat_id: chat, text, disable_web_page_preview: true }),
    });
    if (!r.ok) console.warn("lead telegram notify", r.status);
  } catch (err) { console.warn("lead telegram notify failed", err && err.message); }
}

async function leadsJson(env) {
  const out = [];
  let cursor;
  for (let i = 0; i < 5; i++) {
    const page = await env.KV.list({ prefix: "lead:", limit: 200, cursor });
    for (const k of page.keys || []) {
      const v = await env.KV.get(k.name, "json").catch(() => null);
      if (v) out.push({ key: k.name, ...v });
    }
    if (page.list_complete || !page.cursor) break;
    cursor = page.cursor;
  }
  out.sort((a, b) => (a.ts < b.ts ? 1 : -1));
  return new Response(JSON.stringify({ count: out.length, leads: out }, null, 2), {
    status: 200,
    headers: { "Content-Type": "application/json; charset=utf-8", "Cache-Control": "no-store" },
  });
}

async function handleVenderHub(request, env, url) {
  return withModels(request, env, url, ({ models, builtAt, depositCount, market }) => {
    const rows = Object.entries(models)
      .filter(([slug, r]) => publishedVender(models, slug, r, builtAt))
      .map(([slug, r]) => ({
        slug, b: r.b, m: r.m, n: r.n, fm: r.fm, fl: r.fl, fh: r.fh, sd: r.sd,
        s30: (r.lq && r.lq.s30 != null) ? r.lq.s30 : null,
        cu: (r.lq && r.lq.cu != null) ? r.lq.cu : null,
        cp: (r.lq && r.lq.cp != null) ? r.lq.cp : null,
      }))
      .sort((a, b) => b.n - a.n || (a.slug < b.slug ? -1 : 1));
    return publicHtml(renderVenderHub({ rows, market, host: url.host, depositCount, builtAt }));
  });
}

async function handleVenderPage(request, env, url) {
  let slug;
  try {
    slug = decodeURIComponent(url.pathname.slice("/vender/".length)).replace(/\/+$/, "").toLowerCase();
  } catch (_) { return notFoundPage(request, env, url); }
  const wantsJson = slug.endsWith(".json");
  if (wantsJson) slug = slug.slice(0, -".json".length);
  return withModels(request, env, url, ({ models, builtAt, depositCount, setCookie, market }) => {
    const rec = models[slug];
    if (!rec || !publishedVender(models, slug, rec, builtAt)) return notFoundPage(request, env, url, setCookie);
    if (wantsJson) return jsonResponse(venderJson(rec, slug, { host: url.host, builtAt }));
    return publicHtml(renderVenderPage({
      guides: guideBlock("vender"),
      rec, slug, market,
      pageYears: publishedYearPages(models, slug, rec, builtAt),
      hasLiquidity: publishedLiquidity(models, slug, rec, builtAt),
      hasDepreciation: publishedDepreciation(models, slug, rec, builtAt),
      host: url.host, depositCount, builtAt,
    }));
  });
}

async function handleGuidesHub(request, env, url) {
  return withModels(request, env, url, ({ builtAt, depositCount, stats, market }) =>
    publicHtml(renderGuidesHub({ market, stats, host: url.host, depositCount, builtAt })));
}

async function handleGuide(request, env, url) {
  let slug;
  try {
    slug = decodeURIComponent(url.pathname.slice("/guias/".length)).replace(/\/+$/, "").toLowerCase();
  } catch (_) { return notFoundPage(request, env, url); }
  const guide = guideBySlug(slug);
  if (!guide) return notFoundPage(request, env, url);
  return withModels(request, env, url, ({ models, builtAt, depositCount, stats, market }) =>
    publicHtml(renderGuide({ guide, models, market, stats, host: url.host, depositCount, builtAt })));
}

const AI_TTL_SEC = 180 * 24 * 3600;
const AI_HIT_TTL_SEC = 30 * 24 * 3600;
const AI_AGENTS = [
  ["chatgpt-user", "ask", /ChatGPT-User/i],
  ["perplexity-user", "ask", /Perplexity-User/i],
  ["claude-user", "ask", /Claude-User/i],
  ["oai-searchbot", "index", /OAI-SearchBot/i],
  ["perplexitybot", "index", /PerplexityBot/i],
  ["claude-searchbot", "index", /Claude-SearchBot/i],
  ["gptbot", "bulk", /GPTBot/i],
  ["claudebot", "bulk", /ClaudeBot/i],
  ["ccbot", "bulk", /CCBot/i],
  ["bytespider", "bulk", /Bytespider/i],
  ["amazonbot", "bulk", /Amazonbot/i],
  ["meta-ai", "bulk", /meta-externalagent/i],
];
const AI_CAP = { ask: 120, index: 120, bulk: 20 };
const AI_SAMPLE_CAP = 30;
const AI_SKIP = ["/analytics", "/_olx", "/pt/ir/", "/fonts/", "/healthz", "/og-default.png"];

function aiAgent(ua) {
  for (const [name, kind, re] of AI_AGENTS) if (re.test(ua)) return { name, kind };
  return null;
}

async function trackAiFetch(env, request, pathname) {
  if (ICON_PATHS.has(pathname)) return;
  if (AI_SKIP.some(p => pathname === p || pathname.startsWith(p))) return;
  const agent = aiAgent(request.headers.get("user-agent") || "");
  if (!agent) return;
  const day = new Date().toISOString().slice(0, 10);
  const key = `ai:hit:${day}:${agent.name}`;
  let seen;
  try {
    seen = parseInt((await env.KV.get(key)) || "0", 10) || 0;
    if (seen < AI_CAP[agent.kind]) await env.KV.put(key, String(seen + 1), { expirationTtl: AI_TTL_SEC });
  } catch (err) { console.warn("ai count failed", err && err.message); return; }
  if (agent.kind !== "ask" || seen >= AI_SAMPLE_CAP) return;
  const rec = {
    t: new Date().toISOString(),
    agent: agent.name,
    path: pathname,
    country: (request.cf && request.cf.country) || null,
  };
  try {
    await env.KV.put(`aihit:${rec.t}:${randomToken(3)}`, JSON.stringify(rec), { expirationTtl: AI_HIT_TTL_SEC });
  } catch (err) { console.warn("ai sample failed", err && err.message); }
}

async function aiJson(env) {
  const days = await kvCountsByDay(env, "ai:hit:");
  const names = [];
  let cursor;
  for (let i = 0; i < 5; i++) {
    const page = await env.KV.list({ prefix: "aihit:", limit: 500, cursor });
    for (const k of page.keys || []) names.push(k.name);
    if (page.list_complete || !page.cursor) break;
    cursor = page.cursor;
  }
  const hits = [];
  for (const name of names.slice(-HIT_SAMPLE_MAX).reverse()) {
    const v = await env.KV.get(name, "json").catch(() => null);
    if (v) hits.push(v);
  }
  return new Response(JSON.stringify({ days, hits }, null, 2), {
    status: 200,
    headers: { "Content-Type": "application/json; charset=utf-8", "Cache-Control": "no-store" },
  });
}

const CLICK_TTL_SEC = 180 * 24 * 3600;
const HIT_TTL_SEC = 30 * 24 * 3600;
const HIT_SAMPLE_MAX = 150;
const HIT_SAMPLE_PER_BUCKET = 20;
const CLICK_SOURCES = new Set(["avaliar", "ano", "car", "importar", "vender", "modelo", "outro"]);
const BOT_UA = /bot|crawl|spider|slurp|fetch|monitor|headless|curl|wget|python/i;
const PREFETCH_HINT = /prefetch|prerender|preview/i;

function hitPurpose(request) {
  return (request.headers.get("sec-purpose")
    || request.headers.get("purpose")
    || request.headers.get("x-purpose")
    || request.headers.get("x-moz")
    || "").trim();
}

function clickDrop(request) {
  const ua = (request.headers.get("user-agent") || "").trim();
  if (!ua) return "sem-ua";
  if (PREFETCH_HINT.test(hitPurpose(request))) return "prefetch";
  if (BOT_UA.test(ua)) return "bot";
  return null;
}

async function handleHistoryRedirect(request, env, url) {
  const target = (env.HISTORY_REPORT_URL || "").trim();
  if (!target) return notFoundPage(request, env, url);
  const raw = (url.searchParams.get("from") || "outro").toString().toLowerCase();
  const from = CLICK_SOURCES.has(raw) ? raw : "outro";
  const drop = clickDrop(request);
  const day = new Date().toISOString().slice(0, 10);
  const key = drop ? `click:drop:${day}:${drop}` : `click:hist:${day}:${from}`;
  let seen = HIT_SAMPLE_PER_BUCKET;
  try {
    seen = parseInt((await env.KV.get(key)) || "0", 10) || 0;
    await env.KV.put(key, String(seen + 1), { expirationTtl: CLICK_TTL_SEC });
  } catch (err) { console.warn("click count failed", err && err.message); }
  if (seen < HIT_SAMPLE_PER_BUCKET) await recordHistoryHit(env, request, from, drop);
  return new Response(null, { status: 302, headers: { Location: target, "Cache-Control": "no-store" } });
}

async function recordHistoryHit(env, request, from, drop) {
  const cf = request.cf || {};
  const trim = (v, n) => ((v == null ? "" : String(v)).slice(0, n) || null);
  const rec = {
    t: new Date().toISOString(),
    from,
    drop: drop || null,
    ua: trim(request.headers.get("user-agent"), 200),
    ref: trim(request.headers.get("referer"), 200),
    purpose: trim(hitPurpose(request), 60),
    country: trim(cf.country, 4),
    net: trim(cf.asOrganization, 80),
  };
  try {
    await env.KV.put(`histhit:${rec.t}:${randomToken(3)}`, JSON.stringify(rec), { expirationTtl: HIT_TTL_SEC });
  } catch (err) { console.warn("click sample failed", err && err.message); }
}

async function kvCountsByDay(env, prefix) {
  const out = {};
  let cursor;
  for (let i = 0; i < 5; i++) {
    const page = await env.KV.list({ prefix, limit: 500, cursor });
    for (const k of page.keys || []) {
      const parts = k.name.split(":");
      const day = parts[2], bucket = parts[3] || "outro";
      const v = parseInt((await env.KV.get(k.name)) || "0", 10) || 0;
      (out[day] = out[day] || {})[bucket] = v;
    }
    if (page.list_complete || !page.cursor) break;
    cursor = page.cursor;
  }
  return out;
}

async function clicksJson(env) {
  const days = await kvCountsByDay(env, "click:hist:");
  const drops = await kvCountsByDay(env, "click:drop:");
  const names = [];
  let cursor;
  for (let i = 0; i < 5; i++) {
    const page = await env.KV.list({ prefix: "histhit:", limit: 500, cursor });
    for (const k of page.keys || []) names.push(k.name);
    if (page.list_complete || !page.cursor) break;
    cursor = page.cursor;
  }
  const hits = [];
  for (const name of names.slice(-HIT_SAMPLE_MAX).reverse()) {
    const v = await env.KV.get(name, "json").catch(() => null);
    if (v) hits.push(v);
  }
  return new Response(JSON.stringify({ days, drops, hits }, null, 2), {
    status: 200,
    headers: { "Content-Type": "application/json; charset=utf-8", "Cache-Control": "no-store" },
  });
}
