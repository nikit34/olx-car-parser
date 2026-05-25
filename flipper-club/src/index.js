// Cloudflare Worker entry — closed B2B flip-deal feed.
//
// Routes:
//   GET  /                  → dashboard (requires session)
//   GET  /login             → PIN form
//   POST /login             → validate PIN, mint session cookie
//   POST /logout            → clear session
//   GET  /admin             → PIN list + create form (admin-only)
//   POST /admin/pins/create → issue new PIN, redirect with value in toast
//   POST /admin/pins/:id/revoke → flip revoked=true; takes effect on next req
//   GET  /healthz           → unauthenticated liveness
//
// Data source: chunk 1 uses mock.js. Chunk 2 will swap getDeals() to fetch
// hot_deals_{zone}.json from the latest-data GitHub Release.

import {
  COOKIE_NAME, cookieHeader, clearCookieHeader,
  readSession, findPinByValue, createSession, destroySession,
  listPins, createPin, revokePin, getPinById,
  hasAnyAdmin, countActiveAdmins, isExpired,
} from "./auth.js";
import { renderLogin, renderDashboard, renderAdmin, renderSetup } from "./templates.js";
import { getMockDeals } from "./mock.js";

const ZONES_DEFAULT = "norte,centro,sul,all";

// Paths owned by the flipper-club app. Anything else is treated as a static
// asset request and gated on admin session (the internal analytics dashboard).
const FLIPPER_PATHS = new Set(["/", "/login", "/logout", "/setup", "/healthz"]);
function isFlipperPath(p) {
  if (FLIPPER_PATHS.has(p)) return true;
  if (p === "/admin" || p.startsWith("/admin/")) return true;
  return false;
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const { pathname } = url;
    const method = request.method;

    try {
      if (pathname === "/healthz") return new Response("ok", { status: 200 });

      // /analytics/* — admin-only proxy to the stlite dashboard bundle. Strip
      // the prefix and delegate to the ASSETS binding so the bundle's
      // relative paths (./files/*, ./data/*) resolve correctly under
      // `<base href="/analytics/">` set in dashboard-static/index.html.
      if (pathname === "/analytics" || pathname.startsWith("/analytics/")) {
        return handleAnalytics(request, env, url);
      }

      // Catch-all for non-flipper static-asset paths (e.g. /files/foo.py,
      // /data/dashboard/listings.parquet). These are referenced by the
      // analytics bundle once it boots under /analytics/, so they ALSO
      // require an admin session. Without this, a non-admin sharing a deep
      // link could fetch parquets directly.
      if (!isFlipperPath(pathname)) {
        return handleAssetGated(request, env, url);
      }

      // Setup gate. /setup is only reachable when no admin exists.
      // Conversely, any other path on a fresh install redirects to /setup.
      if (pathname === "/setup") {
        const adminExists = await hasAnyAdmin(env);
        if (adminExists) return notFound();
        if (method === "GET") return html(renderSetup({}));
        if (method === "POST") return handleSetup(request, env);
        return notFound();
      }
      // For every other route, if there's no admin in KV, force setup first.
      // We only check on unauthenticated paths — if a session is valid, an
      // admin definitionally exists (the PIN behind it).
      if (pathname === "/login" || pathname === "/") {
        const session0 = await readSession(request, env);
        if (!session0) {
          const adminExists = await hasAnyAdmin(env);
          if (!adminExists) return redirect("/setup");
        }
      }

      if (pathname === "/login" && method === "GET") {
        const s = await readSession(request, env);
        if (s) return redirect("/");
        return html(renderLogin({}));
      }

      if (pathname === "/login" && method === "POST") {
        return handleLogin(request, env);
      }

      if (pathname === "/logout") {
        return handleLogout(request, env);
      }

      // Everything below this requires a valid session.
      const session = await readSession(request, env);

      if (pathname === "/" && method === "GET") {
        if (!session) return redirect("/login");
        const sort = url.searchParams.get("sort") || "discount";
        const zone = session.pin.zone || "all";
        const deals = await getDeals(env, zone);
        return html(renderDashboard({
          deals,
          zone,
          sort,
          isAdmin: !!session.pin.is_admin,
        }));
      }

      if (pathname === "/admin" && method === "GET") {
        if (!session || !session.pin.is_admin) return notFound();
        const pins = await listPins(env);
        const newPinValue = url.searchParams.get("new");
        const newPin = newPinValue ? { value: newPinValue } : null;
        const error = url.searchParams.get("error") || "";
        const zones = env.ZONES || ZONES_DEFAULT;
        return html(renderAdmin({ pins, newPin, error, zones, isAdmin: true }));
      }

      if (pathname === "/admin/pins/create" && method === "POST") {
        if (!session || !session.pin.is_admin) return notFound();
        const form = await request.formData();
        const isAdminFlag = form.get("is_admin") === "1";
        // Admins don't expire (createPin sets expires_at: null). Flippers
        // default to 24h; user can override per-PIN.
        const ttlHours = parseFloat(form.get("ttl_hours"));
        const pin = await createPin(env, {
          label: form.get("label") || "",
          zone: isAdminFlag ? "all" : (form.get("zone") || "all"),
          ttl_hours: Number.isFinite(ttlHours) && ttlHours > 0 ? ttlHours : 24,
          notes: form.get("notes") || "",
          is_admin: isAdminFlag,
        });
        return redirect(`/admin?new=${encodeURIComponent(pin.value)}`);
      }

      const revokeMatch = pathname.match(/^\/admin\/pins\/([a-f0-9]+)\/revoke$/);
      if (revokeMatch && method === "POST") {
        if (!session || !session.pin.is_admin) return notFound();
        const target = await getPinById(env, revokeMatch[1]);
        if (!target) return redirect("/admin");
        // Block revoking the last active admin — would lock the user out.
        if (target.is_admin) {
          const activeAdmins = await countActiveAdmins(env);
          if (activeAdmins <= 1) return redirect("/admin?error=last_admin");
        }
        await revokePin(env, revokeMatch[1]);
        return redirect("/admin");
      }

      return notFound();
    } catch (err) {
      console.error("worker error", err && err.stack || err);
      return new Response("Internal error", { status: 500 });
    }
  },
};

async function handleLogin(request, env) {
  const form = await request.formData();
  const enteredRaw = (form.get("pin") || "").toString().trim().toUpperCase();
  if (!enteredRaw) return html(renderLogin({ error: "PIN obrigatório." }), 400);
  // basic shape check before hitting KV — PIN alphabet is uppercase alnum
  if (!/^[A-Z0-9]{4,16}$/.test(enteredRaw)) {
    await sleep(400); // soft throttle on garbage submissions
    return html(renderLogin({ error: "PIN inválido." }), 401);
  }
  const pin = await findPinByValue(env, enteredRaw);
  if (!pin) {
    await sleep(400);
    return html(renderLogin({ error: "PIN inválido ou expirado." }), 401);
  }
  if (pin.revoked) return html(renderLogin({ error: "PIN revogado." }), 401);
  if (isExpired(pin)) {
    return html(renderLogin({ error: "PIN expirado." }), 401);
  }
  const ip = request.headers.get("cf-connecting-ip") || "";
  const session = await createSession(env, pin, ip);
  return new Response(null, {
    status: 302,
    headers: {
      "Location": pin.is_admin ? "/admin" : "/",
      "Set-Cookie": cookieHeader(COOKIE_NAME, session.token, {
        maxAgeSec: Math.floor((new Date(session.expires_at) - Date.now()) / 1000),
      }),
    },
  });
}

async function handleAnalytics(request, env, url) {
  const session = await readSession(request, env);
  if (!session || !session.pin.is_admin) {
    // HTML navigation → friendly redirect to login. XHR/asset → 401 to keep
    // the network tab tidy and avoid surprise HTML in image/parquet slots.
    const wantsHtml = (request.headers.get("Accept") || "").includes("text/html");
    if (wantsHtml) return redirect("/login");
    return new Response("Unauthorized", { status: 401 });
  }
  // Strip the /analytics prefix and forward to ASSETS. /analytics → /, /analytics/X → /X.
  let stripped = url.pathname.replace(/^\/analytics\/?/, "/");
  if (!stripped.startsWith("/")) stripped = "/" + stripped;
  const targetUrl = new URL(stripped + url.search, url.origin);
  return env.ASSETS.fetch(new Request(targetUrl, request));
}

async function handleAssetGated(request, env, url) {
  const session = await readSession(request, env);
  if (!session || !session.pin.is_admin) {
    const wantsHtml = (request.headers.get("Accept") || "").includes("text/html");
    if (wantsHtml) return redirect("/login");
    return new Response("Unauthorized", { status: 401 });
  }
  return env.ASSETS.fetch(request);
}

async function handleSetup(request, env) {
  // Idempotency safety: re-check inside POST in case two parallel installs
  // race past the gate. First admin wins; second submit hits notFound.
  const adminExists = await hasAnyAdmin(env);
  if (adminExists) return notFound();

  const form = await request.formData();
  const label = (form.get("label") || "Admin").toString().slice(0, 80);
  const pin = await createPin(env, {
    label,
    zone: "all",
    is_admin: true,
    notes: "Initial admin — created via /setup",
  });
  // Show the PIN once on a confirmation page, then it lives only in KV (and
  // wherever the user copies it). No way to retrieve plaintext again.
  return html(renderSetup({ newPin: { value: pin.value, label: pin.label } }));
}

async function handleLogout(request, env) {
  const cookie = request.headers.get("cookie") || "";
  const m = cookie.match(new RegExp(`${COOKIE_NAME}=([a-f0-9]+)`));
  if (m) await destroySession(env, m[1]);
  return new Response(null, {
    status: 302,
    headers: {
      "Location": "/login",
      "Set-Cookie": clearCookieHeader(COOKIE_NAME),
    },
  });
}

// GitHub Release "latest-data" download base. The scrape.yml workflow uploads
// hot_deals_{zone}.json there after every successful train-model + build_hot_deals.
const HOT_DEALS_BASE =
  "https://github.com/nikit34/olx-car-parser/releases/download/latest-data";
const DEALS_CACHE_TTL_SEC = 300;

async function getDeals(env, zone) {
  const safeZone = ["norte", "centro", "sul", "all"].includes(zone) ? zone : "all";
  // Try KV cache first — survives across requests but ages out after 5 min so
  // a fresh CI run propagates fast.
  const cacheKey = `cache:deals:${safeZone}`;
  const cached = await env.KV.get(cacheKey);
  if (cached) {
    try {
      const parsed = JSON.parse(cached);
      if (Array.isArray(parsed.deals)) return parsed.deals;
    } catch {}
  }
  // Miss → fetch from Release.
  try {
    const url = `${HOT_DEALS_BASE}/hot_deals_${safeZone}.json`;
    const r = await fetch(url, { cf: { cacheTtl: 60, cacheEverything: true } });
    if (!r.ok) {
      console.warn(`hot_deals fetch ${url} → ${r.status}`);
      // Fall back to mock so a bad release doesn't blank the UI.
      return getMockDeals(safeZone);
    }
    const body = await r.text();
    await env.KV.put(cacheKey, body, { expirationTtl: DEALS_CACHE_TTL_SEC });
    const parsed = JSON.parse(body);
    return Array.isArray(parsed.deals) ? parsed.deals : [];
  } catch (err) {
    console.warn("hot_deals fetch error", err && err.message);
    return getMockDeals(safeZone);
  }
}

function html(body, status = 200) {
  return new Response(body, {
    status,
    headers: { "Content-Type": "text/html; charset=utf-8", "Cache-Control": "no-store" },
  });
}

function redirect(loc) {
  return new Response(null, { status: 302, headers: { "Location": loc } });
}

function notFound() {
  return new Response("Not found", { status: 404 });
}

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}
