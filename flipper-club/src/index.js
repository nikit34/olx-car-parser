// Cloudflare Worker entry — public, one-car-at-a-time flip-deal feed with a
// Stripe deposit gate on each listing.
//
// Product model (no auth, no PINs):
//   GET  /                  → one car at a time (top-ranked by decision_score).
//                             Photos/specs/signals are public; the seller's OLX
//                             link is paywalled until a deposit is paid.
//   POST /reserve           → create a Stripe Checkout Session for one car's
//                             deposit, 303 → Stripe-hosted checkout.
//   GET  /unlocked          → Stripe success redirect; verify the session was
//                             paid, record the unlock, reveal the contact.
//   POST /webhook/stripe    → async checkout.session.completed → record unlock
//                             authoritatively (survives a closed success tab).
//   GET  /healthz           → unauthenticated liveness.
//
// Internal-only (NOT the product):
//   GET  /analytics/*  and  /files/* /data/* …  → the stlite analytics bundle
//   and its raw data assets. Gated by HTTP Basic Auth against the Worker
//   secrets ANALYTICS_USER / ANALYTICS_PASS. Fail-closed: if those secrets are
//   unset, access is denied — raw parquets/model internals never go public.
//
// What we sell: the *find* (an unlocked seller contact for one specific car),
// never the car. The deposit unlocks one olx_id's contact link, nothing else.
//
// Data source: getDeals() fetches hot_deals_{zone}.json from the latest-data
// GitHub Release and caches it in KV for 5 min. A missing/broken feed renders a
// degraded banner (no fake data).

import {
  renderGrid, renderCarPage, renderInfo,
  renderLanding, renderClaim, renderClaimSuccess, renderReservations,
  renderAvaliar, renderModelPage, renderModelsHub, renderModelWidget, slugify,
} from "./templates.js";
import {
  stripeConfigured, createCheckoutSession,
  retrieveCheckoutSession, verifyWebhookSignature,
} from "./stripe.js";

const ZONES = ["norte", "centro", "sul", "all"];
const COOKIE_UID = "fc_uid";
const UNLOCK_TTL_SEC = 90 * 24 * 3600; // a paid reservation stays unlocked 90d
const DEFAULT_DEPOSIT_CENTS = 500;     // €5 — overridable via env.DEPOSIT_AMOUNT_CENTS
const DEFAULT_CURRENCY = "eur";

// Product routes the Worker owns directly. Everything else (that isn't
// /analytics or /webhook) is treated as an internal asset request.
//   /          landing (marketing)        /claim     claim-confirm interstitial
//   /mercado   deal feed (grid)           /reserve   POST → Stripe checkout
//   /car       single-car detail          /unlocked  Stripe success → claimed
//   /reservas  my claimed cars
const PRODUCT_PATHS = new Set([
  "/", "/mercado", "/car", "/claim", "/reserve", "/unlocked", "/reservas", "/avaliar",
  "/precos", "/sitemap.xml", "/robots.txt", "/llms.txt",
]);

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const { pathname } = url;
    const method = request.method;

    try {
      if (pathname === "/healthz") return new Response("ok", { status: 200 });

      if (pathname === "/webhook/stripe") {
        if (method !== "POST") return notFound();
        return handleWebhook(request, env);
      }

      // Canonical host — the product answers on both carsbuyer.org and the
      // workers.dev subdomain, and identical content on two hostnames splits
      // ranking signals. Send everything to CANONICAL_HOST, preserving path and
      // query. Deliberately placed AFTER /healthz and /webhook/stripe so neither
      // can ever be redirected: Stripe's signed POST has to reach the exact
      // origin the endpoint was registered against, and a redirect would drop
      // the body and silently break deposit confirmation.
      // Dormant until CANONICAL_HOST is set (see wrangler.toml [vars]).
      const canonicalHost = (env.CANONICAL_HOST || "").trim();
      if (canonicalHost && url.hostname !== canonicalHost && !isLocalHost(url.hostname)) {
        const dest = new URL(url);
        dest.hostname = canonicalHost;
        dest.protocol = "https:";
        dest.port = "";
        // 301 on GET/HEAD (cacheable, passes ranking signals to the new host);
        // 308 elsewhere, which is the only redirect that preserves method+body.
        const perm = (method === "GET" || method === "HEAD") ? 301 : 308;
        return Response.redirect(dest.toString(), perm);
      }

      // Internal stlite dashboard + its assets — Basic-Auth gated, fail-closed.
      if (pathname === "/analytics" || pathname.startsWith("/analytics/")) {
        return handleAnalytics(request, env, url);
      }
      // Per-model SEO pages (/preco/{slug}) — prefix route, BEFORE the asset gate.
      if (pathname.startsWith("/preco/") && method === "GET") {
        return handleModelPage(request, env, url);
      }
      // Embeddable valuation widget (/widget/preco/{slug}) — public, iframe-able,
      // cached, cookie-less. Also a prefix route before the asset gate.
      if (pathname.startsWith("/widget/preco/") && method === "GET") {
        return handleModelWidget(request, env, url);
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
      if (!PRODUCT_PATHS.has(pathname)) {
        return handleAssetGated(request, env);
      }

      if (pathname === "/" && method === "GET") return handleLanding(request, env, url);
      if (pathname === "/precos" && method === "GET") return handleModelsHub(request, env, url);
      if (pathname === "/sitemap.xml" && method === "GET") return handleSitemap(request, env, url);
      if (pathname === "/robots.txt" && method === "GET") return handleRobots(request, env, url);
      if (pathname === "/llms.txt" && method === "GET") return handleLlmsTxt(request, env, url);
      if (pathname === "/avaliar" && method === "GET") return handleAvaliar(request, env, url);
      if (pathname === "/mercado" && method === "GET") return handleFeed(request, env, url);
      if (pathname === "/car" && method === "GET") return handleCar(request, env, url);
      if (pathname === "/claim" && method === "GET") return handleClaim(request, env, url);
      if (pathname === "/reserve" && method === "POST") return handleReserve(request, env, url);
      if (pathname === "/unlocked" && method === "GET") return handleUnlocked(request, env, url);
      if (pathname === "/reservas" && method === "GET") return handleReservas(request, env, url);

      return notFound();
    } catch (err) {
      console.error("worker error", err && err.stack || err);
      return new Response("Internal error", { status: 500 });
    }
  },
};

// ── Product handlers ────────────────────────────────────────────────────────

// Landing (/) — marketing hero with live market stats + a featured top deal.
async function handleLanding(request, env, url) {
  const { uid, setCookie } = ensureUid(request);
  const { deals, degraded } = await getDeals(env, "all");
  const depositCount = (await listUnlocked(env, uid)).size;

  if (degraded || !Array.isArray(deals) || deals.length === 0) {
    return html(renderInfo({
      zone: "all", depositCount,
      title: "Serviço indisponível",
      message: "Não foi possível carregar os negócios neste momento. Tenta novamente dentro de instantes.",
    }), degraded ? 503 : 200, setCookie);
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
    depositEur: depositCents(env) / 100,
    depositCount, host: url.host,
  }), 200, setCookie);
}

// Avaliar (/avaliar) — paste-a-link valuation of ANY OLX listing (Tier-2).
// ?q = an OLX URL or a bare olx_id; we extract the id, look it up in the
// precomputed valuations blob, and render a fair-price verdict (or a graceful
// "not found / ask by email" fallback). No q ⇒ just the lookup form + teaser.
async function handleAvaliar(request, env, url) {
  const { uid, setCookie } = ensureUid(request);
  const depositCount = (await listUnlocked(env, uid)).size;
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

  // Model index — for the paste-hit's contextual /preco link, the spec-form
  // options, and the spec lookup. (cf-cached; cheap.)
  const mdoc = await getModels(env);
  const models = mdoc && mdoc.models;
  let spec = null;
  if (!rec && modelo && models && models[modelo]) {
    const mrec = models[modelo];
    spec = { rec: mrec, slug: modelo, year: ano, cell: pickYearCell(mrec, ano) };
  }
  return html(renderAvaliar({
    rec, olxId, sourceUrl, query, models, spec, depositCount,
    host: url.host, builtAt: mdoc && mdoc.built_at,
  }), 200, setCookie);
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

// Per-model SEO page (/preco/{slug}). Exact slug lookup (never re-split the path
// — models contain hyphens); unknown/sub-threshold slug → real 404 (never indexed).
async function handleModelPage(request, env, url) {
  const { uid, setCookie } = ensureUid(request);
  // strip the /preco/ prefix + any trailing slash; never split on '-'.
  // decodeURIComponent throws URIError on a malformed %-escape (e.g. /preco/%) —
  // a garbage URL must 404, not 500.
  let slug;
  try {
    slug = decodeURIComponent(url.pathname.slice("/preco/".length)).replace(/\/+$/, "").toLowerCase();
  } catch (_) {
    return notFound();
  }
  const mdoc = await getModels(env);
  const models = mdoc && mdoc.models;
  const rec = models ? models[slug] : null;
  if (!rec) return notFound();
  const depositCount = (await listUnlocked(env, uid)).size;

  // Conversion bridge: live hot_deals matching this model (already curated below-fair).
  let liveDeals = [];
  try {
    const { deals } = await getDeals(env, "all");
    liveDeals = (deals || []).filter(d => slugify(`${d.brand}-${d.model}`) === slug).slice(0, 3);
  } catch (_) { /* bridge is best-effort */ }

  // Sibling models (same brand, by listing count desc, excluding self).
  const siblings = Object.entries(models)
    .filter(([s, r]) => r.b === rec.b && s !== slug)
    .sort((a, b) => (b[1].n || 0) - (a[1].n || 0))
    .slice(0, 8)
    .map(([s, r]) => ({ slug: s, m: r.m, fm: r.fm, n: r.n }));

  return html(renderModelPage({
    rec, slug, liveDeals, siblings, host: url.host, depositCount,
    builtAt: mdoc && mdoc.built_at,
  }), 200, setCookie);
}

// Embeddable widget (/widget/preco/{slug}) — a standalone valuation card other
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

// Models hub (/precos) — the crawl spine linking every model page.
async function handleModelsHub(request, env, url) {
  const { uid, setCookie } = ensureUid(request);
  const depositCount = (await listUnlocked(env, uid)).size;
  const mdoc = await getModels(env);
  const models = mdoc && mdoc.models;
  if (!models) {
    return html(renderInfo({
      zone: "all", depositCount, title: "Serviço indisponível",
      message: "Os preços por modelo estão a ser preparados. Volta dentro de instantes.",
    }), 503, setCookie);
  }
  const list = Object.entries(models)
    .map(([s, r]) => ({ slug: s, b: r.b, m: r.m, fm: r.fm, n: r.n }))
    .sort((a, b) => (b.n || 0) - (a.n || 0));
  return html(renderModelsHub({ models: list, depositCount, builtAt: mdoc.built_at, host: url.host }), 200, setCookie);
}

// /sitemap.xml — static indexable URLs + one per model page. Degrades to the
// static set (never 500) if models.json isn't published yet.
async function handleSitemap(request, env, url) {
  const base = `https://${url.host}`;
  const mdoc = await getModels(env);
  const models = mdoc && mdoc.models;
  // Real content-change stamp: the models.json build date, NOT the request date.
  // A request-time "today" on every URL makes lastmod a lie Google learns to
  // ignore. Emit <lastmod> only when we actually have a build stamp.
  const lastmod = ((mdoc && mdoc.built_at) || "").slice(0, 10);
  const lm = lastmod ? `<lastmod>${lastmod}</lastmod>` : "";
  const urls = [
    `<url><loc>${base}/</loc>${lm}<changefreq>daily</changefreq><priority>1.0</priority></url>`,
    `<url><loc>${base}/mercado</loc>${lm}<changefreq>daily</changefreq><priority>0.9</priority></url>`,
    `<url><loc>${base}/avaliar</loc>${lm}<changefreq>weekly</changefreq><priority>0.8</priority></url>`,
    `<url><loc>${base}/precos</loc>${lm}<changefreq>weekly</changefreq><priority>0.7</priority></url>`,
  ];
  if (models) {
    for (const slug of Object.keys(models)) {
      urls.push(`<url><loc>${base}/preco/${encodeURIComponent(slug)}</loc>${lm}<changefreq>weekly</changefreq><priority>0.6</priority></url>`);
    }
  }
  const xml = `<?xml version="1.0" encoding="UTF-8"?>\n`
    + `<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n${urls.join("\n")}\n</urlset>`;
  return new Response(xml, {
    status: 200,
    headers: { "content-type": "application/xml; charset=utf-8", "cache-control": "public, max-age=3600" },
  });
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
  // A handful of the best-covered models, so an agent has real entry points
  // rather than a bare index it would have to crawl to be useful.
  const top = models
    ? Object.entries(models)
        .sort((a, b) => (b[1].n || 0) - (a[1].n || 0))
        .slice(0, 12)
        .map(([slug, r]) => `- [${r.b} ${r.m}](${base}/preco/${slug}): mediana pedida €${r.fm}, ${r.n} anúncios ativos`)
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
    `- [Índice de preços por modelo](${base}/precos)`,
    `- [Avaliar um anúncio concreto](${base}/avaliar)`,
    `- [Mercado: carros abaixo do valor justo](${base}/mercado)`,
    `- [Sitemap](${base}/sitemap.xml)`,
    top.length ? "" : null,
    top.length ? "## Modelos com mais dados" : null,
    top.length ? "" : null,
    ...top,
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

// /robots.txt — allow public, block transactional/internal, point at the sitemap.
async function handleRobots(request, env, url) {
  const body = [
    "User-agent: *", "Allow: /",
    "Disallow: /analytics", "Disallow: /claim", "Disallow: /reserve",
    "Disallow: /unlocked", "Disallow: /reservas",
    // /widget stays crawlable on purpose: it is noindex,follow and links back to
    // the canonical /preco page, so it works as a backlink lever when embedded.
    "",
    // Answer engines are named explicitly rather than left to the wildcard.
    // The wildcard already allows them, but naming them states the intent so a
    // later tightening of `*` cannot silently cut off AI citations — the one
    // distribution channel where being the ORIGINAL source of the numbers
    // (median asking price, IQR, days-to-sell, per-year table) is the whole
    // advantage. Blocking these is how sites vanish from AI answers.
    "User-agent: GPTBot", "Allow: /",          // OpenAI crawler (training/index)
    "User-agent: OAI-SearchBot", "Allow: /",   // OpenAI, powers ChatGPT search
    "User-agent: ChatGPT-User", "Allow: /",    // live fetch on a user's request
    "User-agent: PerplexityBot", "Allow: /",
    "User-agent: ClaudeBot", "Allow: /",
    "User-agent: Google-Extended", "Allow: /", // Gemini / AI Overviews grounding
    "User-agent: CCBot", "Allow: /",
    "",
    `Sitemap: https://${url.host}/sitemap.xml`,
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

// Mercado feed (/mercado) — the grid of car tiles, zone + sort filtered.
async function handleFeed(request, env, url) {
  const zone = pickZone(url.searchParams.get("zone"));
  const view = pickView(url.searchParams.get("view"));
  const { uid, setCookie } = ensureUid(request);
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
  const { deals, degraded } = zoneResults[zone] || { deals: [], degraded: true };
  const unlockedSet = await listUnlocked(env, uid);
  const depositCount = unlockedSet.size;

  if (degraded) {
    return html(renderInfo({
      zone, depositCount,
      title: "Serviço indisponível",
      message: "Não foi possível carregar os negócios neste momento. Tenta novamente dentro de instantes.",
    }), 503, setCookie);
  }

  const sort = url.searchParams.get("sort") || "score";
  const sorted = sortDeals(deals, sort);
  if (sorted.length === 0) {
    return html(renderInfo({
      zone, depositCount,
      title: "Sem negócios quentes",
      message: "Sem carros abaixo do preço nesta zona agora — a lista renova ao longo do dia. Entretanto, cola o link de qualquer anúncio em /avaliar para saber se está bem cotado.",
    }), 200, setCookie);
  }

  return html(renderGrid({
    deals: sorted, zone, sort, view, unlockedSet, depositCount, zoneCounts,
    depositEur: depositCents(env) / 100,
    stripeReady: stripeConfigured(env), host: url.host,
  }), 200, setCookie);
}

// Single-car detail page (opened by clicking a grid tile).
async function handleCar(request, env, url) {
  const zone = pickZone(url.searchParams.get("zone"));
  const view = pickView(url.searchParams.get("view"));
  const olxId = (url.searchParams.get("olx_id") || "").toString();
  const { uid, setCookie } = ensureUid(request);
  const { deals, degraded } = await getDeals(env, zone);
  const depositCount = (await listUnlocked(env, uid)).size;
  if (degraded) {
    return html(renderInfo({
      zone, depositCount, title: "Serviço indisponível",
      message: "Não foi possível carregar os negócios neste momento. Tenta novamente dentro de instantes.",
    }), 503, setCookie);
  }
  const deal = (deals || []).find(d => d.olx_id === olxId);
  if (!deal) return redirect(`/mercado?zone=${zone}`, 302, setCookie);
  const rec = await getUnlock(env, uid, deal.olx_id);
  // Contextual link into the model SEO page, when this model has one.
  const mdoc = await getModels(env);
  const models = mdoc && mdoc.models;
  const mslug = slugify(`${deal.brand}-${deal.model}`);
  const modelHref = (models && models[mslug]) ? `/preco/${encodeURIComponent(mslug)}` : null;
  return html(renderCarPage({
    deal, zone, view, unlocked: !!rec, justReserved: false,
    claimedAtMs: claimedAtMs(rec), depositCount, modelHref,
    depositEur: depositCents(env) / 100,
    stripeReady: stripeConfigured(env), host: url.host,
  }), 200, setCookie);
}

// Claim confirm interstitial (/claim) — deposit breakdown + benefits, then a
// form that POSTs to /reserve → Stripe. If already unlocked, jump to detail.
async function handleClaim(request, env, url) {
  const zone = pickZone(url.searchParams.get("zone"));
  const olxId = (url.searchParams.get("olx_id") || "").toString();
  const { uid, setCookie } = ensureUid(request);
  const { deals, degraded } = await getDeals(env, zone);
  const depositCount = (await listUnlocked(env, uid)).size;
  if (degraded) {
    return html(renderInfo({
      zone, depositCount, title: "Serviço indisponível",
      message: "Não foi possível carregar os negócios neste momento. Tenta novamente dentro de instantes.",
    }), 503, setCookie);
  }
  const deal = (deals || []).find(d => d.olx_id === olxId);
  if (!deal) return redirect(`/mercado?zone=${zone}`, 302, setCookie);
  if (await isUnlocked(env, uid, deal.olx_id)) {
    return redirect(`/car?zone=${zone}&olx_id=${encodeURIComponent(olxId)}`, 302, setCookie);
  }
  return html(renderClaim({
    deal, zone, depositCount,
    depositEur: depositCents(env) / 100,
    stripeReady: stripeConfigured(env),
  }), 200, setCookie);
}

async function handleReserve(request, env, url) {
  if (!sameOrigin(request, url)) return forbidden();
  const { uid, setCookie } = ensureUid(request);
  const form = await request.formData();
  const olxId = (form.get("olx_id") || "").toString();
  const zone = pickZone(form.get("zone"));

  if (!stripeConfigured(env)) {
    return html(renderInfo({
      zone,
      title: "Pagamentos indisponíveis",
      message: "A reserva por depósito ainda não está ativa. Tenta novamente mais tarde.",
    }), 503, setCookie);
  }

  // Validate the olx_id is a real, current deal before charging anyone.
  const { deals } = await getDeals(env, zone);
  const deal = (deals || []).find(d => d.olx_id === olxId);
  if (!deal) return redirect(`/mercado?zone=${zone}`, 303, setCookie);

  const carName = deal.title
    || [deal.brand, deal.model, deal.year].filter(Boolean).join(" ")
    || "Viatura";
  // Stripe substitutes the literal {CHECKOUT_SESSION_ID} on redirect — must not
  // be URL-encoded, so it's appended after the encoded params.
  const successUrl =
    `${url.origin}/unlocked?zone=${zone}&olx_id=${encodeURIComponent(olxId)}`
    + `&session_id={CHECKOUT_SESSION_ID}`;
  const cancelUrl = `${url.origin}/car?zone=${zone}&olx_id=${encodeURIComponent(olxId)}`;

  try {
    const session = await createCheckoutSession(env, {
      uid, olxId, carName,
      amountCents: depositCents(env),
      currency: env.CURRENCY || DEFAULT_CURRENCY,
      successUrl, cancelUrl,
    });
    return redirect(session.url, 303, setCookie);
  } catch (err) {
    console.error("checkout create failed", err && err.message);
    return html(renderInfo({
      zone,
      title: "Erro no pagamento",
      message: "Não foi possível iniciar o pagamento. Tenta novamente dentro de instantes.",
    }), 502, setCookie);
  }
}

async function handleUnlocked(request, env, url) {
  const zone = pickZone(url.searchParams.get("zone"));
  const view = pickView(url.searchParams.get("view"));
  const olxId = (url.searchParams.get("olx_id") || "").toString();
  const sessionId = (url.searchParams.get("session_id") || "").toString();
  const { uid, setCookie } = ensureUid(request);

  // Verify on the redirect too (belt-and-suspenders with the webhook): if the
  // user lands here with a genuinely paid session, record the unlock now so it
  // works even if the webhook is delayed.
  if (sessionId && stripeConfigured(env)) {
    try {
      const s = await retrieveCheckoutSession(env, sessionId);
      if (s && s.payment_status === "paid") {
        const m = s.metadata || {};
        await recordUnlock(env, m.uid || uid, m.olx_id || olxId, {
          stripe_session_id: s.id, amount: s.amount_total, currency: s.currency,
        });
      }
    } catch (err) {
      console.warn("unlocked verify failed", err && err.message);
    }
  }

  const { deals, degraded } = await getDeals(env, zone);
  const depositCount = (await listUnlocked(env, uid)).size;
  if (degraded || !Array.isArray(deals) || deals.length === 0) {
    return html(renderInfo({
      zone, depositCount,
      title: "Reserva registada",
      message: "O teu depósito foi recebido. Recarrega a página dentro de instantes para ver o contacto.",
    }), 200, setCookie);
  }
  const deal = (deals || []).find(d => d.olx_id === olxId);
  if (!deal) return redirect(`/mercado?zone=${zone}`, 302, setCookie);
  const rec = await getUnlock(env, uid, deal.olx_id);

  // Paid+unlocked → the design's celebratory success screen. Not yet recorded
  // (webhook lag) → reuse the detail page's locked module so they can retry.
  if (rec) {
    return html(renderClaimSuccess({
      deal, zone, claimedAtMs: claimedAtMs(rec), depositCount,
      depositEur: depositCents(env) / 100,
    }), 200, setCookie);
  }
  return html(renderCarPage({
    deal, zone, view, unlocked: false, justReserved: false,
    claimedAtMs: null, depositCount,
    depositEur: depositCents(env) / 100,
    stripeReady: stripeConfigured(env),
  }), 200, setCookie);
}

// Reservas (/reservas) — every car this visitor has claimed (paid-unlocked),
// each with its 24h-exclusivity countdown anchored on the deposit timestamp.
async function handleReservas(request, env, url) {
  const { uid, setCookie } = ensureUid(request);
  const { deals, degraded } = await getDeals(env, "all");
  const records = await listUnlockedRecords(env, uid);
  const depositCount = records.length;

  let claims = [];
  if (!degraded && Array.isArray(deals)) {
    const byId = new Map(deals.map(d => [d.olx_id, d]));
    claims = records
      .map(r => ({ deal: byId.get(r.olxId), claimedAtMs: r.claimedAtMs }))
      .filter(c => c.deal)
      .sort((a, b) => (b.claimedAtMs || 0) - (a.claimedAtMs || 0));
  }

  return html(renderReservations({
    claims, depositCount,
    depositEur: depositCents(env) / 100,
  }), 200, setCookie);
}

async function handleWebhook(request, env) {
  const raw = await request.text();
  const sig = request.headers.get("Stripe-Signature");
  let event;
  try {
    event = await verifyWebhookSignature(env, raw, sig);
  } catch (err) {
    console.warn("webhook rejected", err && err.message);
    return new Response("invalid signature", { status: 400 });
  }
  if (event.type === "checkout.session.completed"
      || event.type === "checkout.session.async_payment_succeeded") {
    const s = event.data && event.data.object;
    if (s && s.payment_status === "paid") {
      const m = s.metadata || {};
      await recordUnlock(env, m.uid, m.olx_id, {
        stripe_session_id: s.id, amount: s.amount_total, currency: s.currency,
      });
    }
  }
  return new Response("ok", { status: 200 });
}

// ── Unlock state (KV) ─────────────────────────────────────────────────────

async function recordUnlock(env, uid, olxId, info) {
  if (!uid || !olxId) return;
  await env.KV.put(
    `unlock:${uid}:${olxId}`,
    JSON.stringify({ paid_at: new Date().toISOString(), ...info }),
    { expirationTtl: UNLOCK_TTL_SEC },
  );
}

async function isUnlocked(env, uid, olxId) {
  if (!uid || !olxId) return false;
  return !!(await env.KV.get(`unlock:${uid}:${olxId}`));
}

// Full unlock record ({ paid_at, … }) or null — used to drive the per-car 24h
// exclusivity countdown from when the deposit was actually taken.
async function getUnlock(env, uid, olxId) {
  if (!uid || !olxId) return null;
  const raw = await env.KV.get(`unlock:${uid}:${olxId}`);
  if (!raw) return null;
  try { return JSON.parse(raw); } catch { return {}; }
}

// Epoch-ms of the deposit, for the countdown's data-claimed-at. Null if the
// record predates paid_at (legacy) — the UI then shows a static 24:00:00.
function claimedAtMs(rec) {
  if (!rec || !rec.paid_at) return null;
  const t = Date.parse(rec.paid_at);
  return Number.isFinite(t) ? t : null;
}

// Every unlock for this visitor as { olxId, claimedAtMs } — one prefix scan
// plus a get per key (Reservas only, so the fan-out stays small).
async function listUnlockedRecords(env, uid) {
  const ids = [...(await listUnlocked(env, uid))];
  return Promise.all(ids.map(async olxId => ({
    olxId, claimedAtMs: claimedAtMs(await getUnlock(env, uid, olxId)),
  })));
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
const DEALS_CACHE_TTL_SEC = 300;
const DEGRADED_CACHE_TTL_SEC = 30;

// Returns { deals, degraded }. `degraded: true` means we could not load the
// real feed — surfaced honestly rather than showing stale or fake listings.
async function getDeals(env, zone) {
  const safeZone = ZONES.includes(zone) ? zone : "all";
  const cacheKey = `cache:deals:${safeZone}`;
  const cached = await env.KV.get(cacheKey);
  if (cached) {
    try {
      const parsed = JSON.parse(cached);
      if (parsed && parsed.__degraded) return { deals: [], degraded: true };
      if (Array.isArray(parsed.deals)) return { deals: parsed.deals, degraded: false };
    } catch {}
  }
  const url = `${HOT_DEALS_BASE}/hot_deals_${safeZone}.json`;
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
    await env.KV.put(cacheKey, body, { expirationTtl: DEALS_CACHE_TTL_SEC });
    return { deals: parsed.deals, degraded: false };
  } catch (err) {
    console.warn("hot_deals fetch error", err && err.message);
    return degrade(env, cacheKey);
  }
}

async function degrade(env, cacheKey) {
  await env.KV.put(cacheKey, JSON.stringify({ __degraded: true }),
    { expirationTtl: DEGRADED_CACHE_TTL_SEC });
  return { deals: [], degraded: true };
}

// valuations.json — the public "value any listing" lookup (Tier-2). ~0.9 MB
// gzipped; fetched from the Release and edge-cached. Parsed per request (the
// /avaliar tool is low-traffic). Returns the {olx_id: rec} map, or null if the
// blob isn't published yet / fetch fails (handler then shows the fallback).
async function getValuations(env) {
  const url = `${HOT_DEALS_BASE}/valuations.json`;
  try {
    // Cache only successful responses (cacheTtlByStatus) — never pin a 404 from
    // the pre-publish window, or a transient 5xx, into the edge cache for 10 min.
    const r = await fetch(url, {
      cf: { cacheEverything: true, cacheTtlByStatus: { "200-299": 300, "300-399": 0, "400-499": 0, "500-599": 0 } },
    });
    if (!r.ok) {
      console.warn(`valuations fetch ${url} → ${r.status}`);
      return null;
    }
    const data = await r.json();
    return data && data.cars ? data.cars : null;
  } catch (err) {
    console.warn("valuations fetch error", err && err.message);
    return null;
  }
}

// models.json — the per-model SEO blob (Tier-3) for /preco/*, /precos, /sitemap.
// Same edge-cache (success-only) pattern as getValuations; ~50 KB gzipped.
// Returns the full doc { models: {slug: rec}, built_at }, or null (handlers
// then 404/degrade). Callers read `.models`; `.built_at` drives the public
// "preços atualizados em …" freshness line.
async function getModels(env) {
  const url = `${HOT_DEALS_BASE}/models.json`;
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

// Every olx_id this visitor has paid-unlocked, via one KV prefix scan instead
// of N per-deal gets — used to badge tiles in the grid. Returns a Set.
async function listUnlocked(env, uid) {
  const set = new Set();
  if (!uid) return set;
  const prefix = `unlock:${uid}:`;
  let cursor;
  do {
    const page = await env.KV.list({ prefix, cursor });
    for (const k of page.keys) set.add(k.name.slice(prefix.length));
    cursor = page.list_complete ? null : page.cursor;
  } while (cursor);
  return set;
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

function depositCents(env) {
  const n = parseInt(env.DEPOSIT_AMOUNT_CENTS, 10);
  return Number.isFinite(n) && n > 0 ? n : DEFAULT_DEPOSIT_CENTS;
}

function randomToken(bytes = 16) {
  const buf = new Uint8Array(bytes);
  crypto.getRandomValues(buf);
  return Array.from(buf, b => b.toString(16).padStart(2, "0")).join("");
}

// Read the visitor cookie, minting one if absent. The returned setCookie (if
// any) must be attached to the response so the same visitor's unlocks persist.
function ensureUid(request) {
  const cookie = request.headers.get("cookie") || "";
  const m = cookie.match(new RegExp(`${COOKIE_UID}=([a-f0-9]+)`));
  if (m) return { uid: m[1], setCookie: null };
  const uid = randomToken(16);
  const setCookie = [
    `${COOKIE_UID}=${uid}`,
    "Path=/", "HttpOnly", "SameSite=Lax", "Secure",
    `Max-Age=${365 * 24 * 3600}`,
  ].join("; ");
  return { uid, setCookie };
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

// CSRF guard for the /reserve POST: verify the request came from our own host.
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
