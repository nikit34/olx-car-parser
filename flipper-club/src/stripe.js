// Stripe helpers — raw REST against api.stripe.com (no SDK; the official
// stripe-node package pulls in Node built-ins that don't exist in Workers).
// Everything here is fetch + WebCrypto, which the Workers runtime provides.
//
// Three things the Worker needs:
//   1. createCheckoutSession() — mint a hosted-checkout URL for one car's deposit
//   2. retrieveCheckoutSession() — verify on the success redirect that it was paid
//   3. verifyWebhookSignature() — authenticate the async checkout.session.completed
//      webhook so an attacker can't forge "this car is paid" by POSTing /webhook.
//
// Secrets (set with `wrangler secret put`, never committed):
//   STRIPE_SECRET_KEY      sk_test_… / sk_live_…
//   STRIPE_WEBHOOK_SECRET  whsec_… (from the webhook endpoint in the Stripe dash)

const STRIPE_API = "https://api.stripe.com/v1";

export function stripeConfigured(env) {
  return !!(env.STRIPE_SECRET_KEY && env.STRIPE_SECRET_KEY.startsWith("sk_"));
}

// Flatten a nested object into Stripe's bracket form-encoding, e.g.
// { line_items: [{ quantity: 1 }] } → "line_items[0][quantity]=1".
function formEncode(obj, prefix = "", out = new URLSearchParams()) {
  for (const [k, v] of Object.entries(obj)) {
    if (v == null) continue;
    const key = prefix ? `${prefix}[${k}]` : k;
    if (Array.isArray(v)) {
      v.forEach((item, i) =>
        typeof item === "object"
          ? formEncode(item, `${key}[${i}]`, out)
          : out.append(`${key}[${i}]`, String(item)));
    } else if (typeof v === "object") {
      formEncode(v, key, out);
    } else {
      out.append(key, String(v));
    }
  }
  return out;
}

async function stripeFetch(env, method, path, body) {
  const init = {
    method,
    headers: {
      Authorization: `Bearer ${env.STRIPE_SECRET_KEY}`,
      "Content-Type": "application/x-www-form-urlencoded",
    },
  };
  if (body) init.body = formEncode(body).toString();
  const r = await fetch(`${STRIPE_API}${path}`, init);
  const json = await r.json();
  if (!r.ok) {
    const msg = json && json.error && json.error.message || r.status;
    throw new Error(`stripe ${path} → ${msg}`);
  }
  return json;
}

// Create a one-off payment Checkout Session for a single car's deposit.
// `uid` is the visitor cookie id; it rides in client_reference_id + metadata so
// the webhook (which has no cookies) can attribute the unlock back to this user.
export async function createCheckoutSession(env, {
  uid, olxId, carName, amountCents, currency, successUrl, cancelUrl,
}) {
  const session = await stripeFetch(env, "POST", "/checkout/sessions", {
    mode: "payment",
    success_url: successUrl,
    cancel_url: cancelUrl,
    client_reference_id: uid,
    metadata: { uid, olx_id: olxId },
    // Restrict the payment record so refunds/disputes are traceable to the car.
    payment_intent_data: { metadata: { uid, olx_id: olxId } },
    line_items: [{
      quantity: 1,
      price_data: {
        currency,
        unit_amount: amountCents,
        product_data: {
          name: `Reserva — ${carName}`.slice(0, 250),
          description: "Depósito reembolsável para desbloquear o contacto do vendedor.",
        },
      },
    }],
  });
  return session; // { id, url, ... }
}

export async function retrieveCheckoutSession(env, id) {
  return stripeFetch(env, "GET", `/checkout/sessions/${encodeURIComponent(id)}`);
}

// Stripe signs webhooks as `t=<unix>,v1=<hex-hmac>` over `${t}.${rawBody}`.
// We recompute the HMAC with the endpoint secret and constant-time compare.
// Returns the parsed event on success, throws otherwise. `toleranceSec` guards
// against replay of an old captured request.
export async function verifyWebhookSignature(env, rawBody, sigHeader, toleranceSec = 300, nowMs = Date.now()) {
  if (!env.STRIPE_WEBHOOK_SECRET) throw new Error("webhook secret not configured");
  if (!sigHeader) throw new Error("missing Stripe-Signature");
  const parts = Object.fromEntries(
    sigHeader.split(",").map(kv => kv.split("=").map(s => s.trim()))
  );
  const t = parts.t;
  const v1 = parts.v1;
  if (!t || !v1) throw new Error("malformed Stripe-Signature");
  if (Math.abs(nowMs / 1000 - Number(t)) > toleranceSec) throw new Error("signature timestamp outside tolerance");
  const expected = await hmacSha256Hex(env.STRIPE_WEBHOOK_SECRET, `${t}.${rawBody}`);
  if (!constantTimeEq(expected, v1)) throw new Error("signature mismatch");
  return JSON.parse(rawBody);
}

async function hmacSha256Hex(secret, message) {
  const key = await crypto.subtle.importKey(
    "raw", new TextEncoder().encode(secret),
    { name: "HMAC", hash: "SHA-256" }, false, ["sign"]
  );
  const sig = await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(message));
  return Array.from(new Uint8Array(sig), b => b.toString(16).padStart(2, "0")).join("");
}

function constantTimeEq(a, b) {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}
