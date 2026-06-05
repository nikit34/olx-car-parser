// Auth & session helpers — KV-backed PINs and session cookies.
//
// Storage layout in the single KV binding:
//   pin:{id}         → JSON { value, label, zone, created_at, expires_at,
//                              revoked, is_admin, notes,
//                              login_count, last_login, last_login_ip,
//                              login_history }
//   session:{token}  → JSON { pin_id, expires_at, created_at, ip }
//
// login_count/last_login/login_history are written by recordLogin() on every
// successful sign-in. They're the only persistent record of who came back —
// sessions are TTL'd and GC'd, so without this a "returning user" (login_count
// > 1) is unobservable. Older PINs created before this existed simply have the
// fields absent until their next login.
//
// PINs are looked up by iterating `pin:*` on login (n ≤ 50 in practice).
// Revocation is enforced on EVERY request by re-reading the PIN record —
// flipping `revoked: true` invalidates all live sessions on next action.

const SESSION_TTL_HOURS = 24;
const PIN_TTL_HOURS_DEFAULT = 24;

export const COOKIE_NAME = "fclub_session";

export function randomToken(bytes = 24) {
  const buf = new Uint8Array(bytes);
  crypto.getRandomValues(buf);
  return Array.from(buf, b => b.toString(16).padStart(2, "0")).join("");
}

export function generatePinValue() {
  // 8 chars, alphabet excludes ambiguous 0/O/1/I/l.
  const alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789";
  const buf = new Uint8Array(8);
  crypto.getRandomValues(buf);
  return Array.from(buf, b => alphabet[b % alphabet.length]).join("");
}

export async function listPins(env) {
  const out = [];
  let cursor;
  while (true) {
    const page = await env.KV.list({ prefix: "pin:", cursor });
    for (const k of page.keys) {
      const raw = await env.KV.get(k.name);
      if (raw) {
        try {
          const p = JSON.parse(raw);
          p.id = k.name.slice(4);
          out.push(p);
        } catch {}
      }
    }
    if (page.list_complete) break;
    cursor = page.cursor;
  }
  return out.sort((a, b) => (b.created_at || "").localeCompare(a.created_at || ""));
}

export async function findPinByValue(env, value) {
  if (!value) return null;
  const pins = await listPins(env);
  // Constant-time compare each value to avoid timing leaks.
  let match = null;
  for (const p of pins) {
    if (constantTimeEq(p.value || "", value)) match = p;
  }
  return match;
}

function constantTimeEq(a, b) {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}

export async function createPin(env, { id, label, zone, ttl_hours, is_admin, notes }) {
  // Callers may pass a fixed `id` to make the write idempotent (the initial
  // /setup admin uses this so a double-submit collides on one KV key instead
  // of minting N admin records). Everything else gets a random id.
  const pinId = id || randomToken(8);
  const value = generatePinValue();
  const now = new Date();
  // Admin PINs don't expire — they're managed by revoke. Flippers get a TTL.
  const expiresAt = is_admin
    ? null
    : new Date(now.getTime() + (ttl_hours || PIN_TTL_HOURS_DEFAULT) * 3600 * 1000).toISOString();
  const record = {
    value,
    label: label || "",
    zone: zone || "all",
    notes: notes || "",
    is_admin: !!is_admin,
    revoked: false,
    created_at: now.toISOString(),
    expires_at: expiresAt,
  };
  await env.KV.put(`pin:${pinId}`, JSON.stringify(record));
  return { id: pinId, ...record };
}

// Returns true iff the PIN's expires_at exists and is in the past.
// Null/missing expires_at means "never expires" (admin PINs).
export function isExpired(pin) {
  return !!pin.expires_at && new Date(pin.expires_at) < new Date();
}

export async function getPinById(env, id) {
  const raw = await env.KV.get(`pin:${id}`);
  if (!raw) return null;
  const p = JSON.parse(raw);
  p.id = id;
  return p;
}

export async function revokePin(env, id) {
  const p = await getPinById(env, id);
  if (!p) return false;
  p.revoked = true;
  p.revoked_at = new Date().toISOString();
  await env.KV.put(`pin:${id}`, JSON.stringify(p));
  return true;
}

export async function deletePin(env, id) {
  await env.KV.delete(`pin:${id}`);
}

// Bounded recent-login tail kept per PIN. login_count holds the lifetime
// total; login_history is only the trailing window so a frequently-used PIN's
// KV value can't grow without bound.
const LOGIN_HISTORY_MAX = 20;

// Stamp a successful login onto the PIN record: bump the lifetime counter,
// record last_login / last_login_ip, and append to the bounded history tail.
// "Returning user" is then simply login_count > 1.
//
// Read-modify-write on one KV key. On the rare concurrent-login race (same PIN,
// two requests) the later write wins and one increment can be lost — acceptable
// for this low-traffic B2B tool. Re-put carries NO expirationTtl, matching
// createPin/revokePin: pin:* keys persist and expiry is logical (isExpired
// against expires_at), not KV-level. Best-effort — never throws into the login
// path, so a tracking hiccup can't block a valid sign-in.
export async function recordLogin(env, pin, ip) {
  try {
    const raw = await env.KV.get(`pin:${pin.id}`);
    if (!raw) return;
    const rec = JSON.parse(raw);
    const now = new Date().toISOString();
    rec.login_count = (rec.login_count || 0) + 1;
    rec.last_login = now;
    rec.last_login_ip = ip || "";
    const history = Array.isArray(rec.login_history) ? rec.login_history : [];
    history.push({ at: now, ip: ip || "" });
    rec.login_history = history.slice(-LOGIN_HISTORY_MAX);
    await env.KV.put(`pin:${pin.id}`, JSON.stringify(rec));
  } catch {
    // tracking is best-effort; a valid PIN must still be able to log in
  }
}

// Used by the setup-gate: returns true iff at least one un-revoked,
// non-expired admin PIN exists. Calling code uses this to decide whether
// /setup is reachable.
export async function hasAnyAdmin(env) {
  let cursor;
  while (true) {
    const page = await env.KV.list({ prefix: "pin:", cursor });
    for (const k of page.keys) {
      const raw = await env.KV.get(k.name);
      if (!raw) continue;
      try {
        const p = JSON.parse(raw);
        if (p.is_admin && !p.revoked && !isExpired(p)) return true;
      } catch {}
    }
    if (page.list_complete) break;
    cursor = page.cursor;
  }
  return false;
}

// Used to block revoking the last active admin (would lock the user out
// of the UI). countActiveAdmins counts un-revoked, non-expired admin PINs.
export async function countActiveAdmins(env) {
  const pins = await listPins(env);
  let n = 0;
  for (const p of pins) {
    if (p.is_admin && !p.revoked && !isExpired(p)) n++;
  }
  return n;
}

export async function createSession(env, pin, ip) {
  const token = randomToken(24);
  const now = new Date();
  // Session expires at min(now + 24h, pin.expires_at). Admin PINs have
  // null expires_at — for them, the session caps at the 24h ceiling.
  const sessionExpiresMs = now.getTime() + SESSION_TTL_HOURS * 3600 * 1000;
  const pinExpiresMs = pin.expires_at ? new Date(pin.expires_at).getTime() : Infinity;
  const expiresMs = Math.min(sessionExpiresMs, pinExpiresMs);
  const record = {
    pin_id: pin.id,
    created_at: now.toISOString(),
    expires_at: new Date(expiresMs).toISOString(),
    ip: ip || "",
  };
  // KV expirationTtl in seconds — let KV garbage-collect expired sessions for us.
  const ttlSec = Math.max(60, Math.floor((expiresMs - now.getTime()) / 1000));
  await env.KV.put(`session:${token}`, JSON.stringify(record), { expirationTtl: ttlSec });
  return { token, ...record };
}

export async function readSession(request, env) {
  const cookie = request.headers.get("cookie") || "";
  const m = cookie.match(new RegExp(`${COOKIE_NAME}=([a-f0-9]+)`));
  if (!m) return null;
  const token = m[1];
  const raw = await env.KV.get(`session:${token}`);
  if (!raw) return null;
  const session = JSON.parse(raw);
  session.token = token;
  // Cross-check the linked PIN — this is what makes revocation work.
  const pin = await getPinById(env, session.pin_id);
  if (!pin) return null;
  if (pin.revoked) return null;
  if (isExpired(pin)) return null;
  if (new Date(session.expires_at) < new Date()) return null;
  session.pin = pin;
  return session;
}

export async function destroySession(env, token) {
  if (token) await env.KV.delete(`session:${token}`);
}

export function cookieHeader(name, value, { maxAgeSec, secure = true } = {}) {
  const parts = [
    `${name}=${value}`,
    "Path=/",
    "HttpOnly",
    "SameSite=Lax",
  ];
  if (secure) parts.push("Secure");
  if (typeof maxAgeSec === "number") parts.push(`Max-Age=${maxAgeSec}`);
  return parts.join("; ");
}

export function clearCookieHeader(name) {
  return cookieHeader(name, "", { maxAgeSec: 0 });
}
