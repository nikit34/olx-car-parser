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
  "/", "/mercado", "/car", "/claim", "/reserve", "/unlocked", "/reservas",
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

      // Internal stlite dashboard + its assets — Basic-Auth gated, fail-closed.
      if (pathname === "/analytics" || pathname.startsWith("/analytics/")) {
        return handleAnalytics(request, env, url);
      }
      if (!PRODUCT_PATHS.has(pathname)) {
        return handleAssetGated(request, env);
      }

      if (pathname === "/" && method === "GET") return handleLanding(request, env, url);
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
    depositCount,
  }), 200, setCookie);
}

// Mercado feed (/mercado) — the grid of car tiles, zone + sort filtered.
async function handleFeed(request, env, url) {
  const zone = pickZone(url.searchParams.get("zone"));
  const { uid, setCookie } = ensureUid(request);
  const { deals, degraded } = await getDeals(env, zone);
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
      message: "Sem negócios com margem na tua zona neste momento. Volta dentro de 4h — o próximo scrape vai colocar novos.",
    }), 200, setCookie);
  }

  return html(renderGrid({
    deals: sorted, zone, sort, unlockedSet, depositCount,
    depositEur: depositCents(env) / 100,
    stripeReady: stripeConfigured(env),
  }), 200, setCookie);
}

// Single-car detail page (opened by clicking a grid tile).
async function handleCar(request, env, url) {
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
  const rec = await getUnlock(env, uid, deal.olx_id);
  return html(renderCarPage({
    deal, zone, unlocked: !!rec, justReserved: false,
    claimedAtMs: claimedAtMs(rec), depositCount,
    depositEur: depositCents(env) / 100,
    stripeReady: stripeConfigured(env),
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
    deal, zone, unlocked: false, justReserved: false,
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
  const headers = { "Content-Type": "text/html; charset=utf-8", "Cache-Control": "no-store" };
  if (setCookie) headers["Set-Cookie"] = setCookie;
  return new Response(body, { status, headers });
}

function redirect(loc, status = 302, setCookie = null) {
  const headers = { "Location": loc };
  if (setCookie) headers["Set-Cookie"] = setCookie;
  return new Response(null, { status, headers });
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
