# flipper-club

Public web-feed of OLX/StandVirtual flip candidates, shown **one car at a
time**. The photos, specs, fair-price estimate and risk signals are open; the
**seller's OLX link is paywalled** behind a small refundable **Stripe deposit**
charged per listing. Paying the deposit reserves that one car and unlocks its
contact — we sell the *find*, never the car.

Separate Cloudflare Worker from the public stlite dashboard at the repo root —
both live alongside each other on the same domain and deploy together.

## Stack

- Cloudflare Workers — `src/index.js`
- Cloudflare KV (one binding) for the deals cache + per-visitor unlock records
- Stripe Checkout (hosted) for the deposit, verified by a signed webhook —
  `src/stripe.js`, raw REST + WebCrypto (no SDK; stripe-node needs Node built-ins)
- Server-rendered HTML — no build step, no framework — `src/templates.js`
- Data comes from `hot_deals_{zone}.json` artifacts uploaded to the `latest-data`
  Release by `scrape.yml`; the Worker fetches per-zone at request time and caches
  in KV for 15 min. A missing/broken feed renders a degraded banner (no fake data).

## Routes

Public product:

- `GET  /` — one car at a time (top-ranked by `decision_score`). `?zone=` picks
  the geo-shard (`norte`/`centro`/`sul`/`all`); `?i=N` / `?olx_id=…` pick the car.
- `POST /reserve` — create a Stripe Checkout Session for one car's deposit, then
  303 → the Stripe-hosted checkout page.
- `GET  /unlocked` — Stripe success redirect. Verifies the session was paid,
  records the unlock, reveals the contact.
- `POST /webhook/stripe` — async `checkout.session.completed`; records the unlock
  authoritatively (survives a closed success tab). Signature-verified.
- `GET  /healthz` — unauthenticated liveness.

Internal (NOT the product), HTTP Basic-Auth gated, **fail-closed**:

- `GET /analytics` `/analytics/*` — internal stlite analytics dashboard.
- `/files/*` `/data/*` and other asset paths — used by the analytics bundle.

If `ANALYTICS_USER` / `ANALYTICS_PASS` are unset, these return `401` — the raw
parquets and model internals never go public.

## Secrets

Set once per environment (never committed):

```bash
cd ..                                   # repo root (wrangler.toml lives here)
npx wrangler secret put STRIPE_SECRET_KEY      # sk_live_… / sk_test_…
npx wrangler secret put STRIPE_WEBHOOK_SECRET  # whsec_… (from the webhook endpoint)
npx wrangler secret put ANALYTICS_USER
npx wrangler secret put ANALYTICS_PASS
```

The deposit amount is a plain var in `wrangler.toml` (`DEPOSIT_AMOUNT_CENTS`,
default `500` = €5.00; `CURRENCY`, default `eur`) — change it there, no secret
needed.

### Stripe setup

1. Stripe Dashboard → Developers → API keys → copy the **secret** key →
   `wrangler secret put STRIPE_SECRET_KEY`.
2. Developers → Webhooks → **Add endpoint**:
   `https://olx-car-parser.permikov134.workers.dev/webhook/stripe`,
   events `checkout.session.completed` and
   `checkout.session.async_payment_succeeded` → copy the **Signing secret**
   (`whsec_…`) → `wrangler secret put STRIPE_WEBHOOK_SECRET`.
3. Use **test mode** keys + `stripe listen --forward-to …/webhook/stripe` while
   developing; swap to live keys for production.

Without `STRIPE_SECRET_KEY` the site still works for browsing — the reserve
button shows "Reservas em breve" instead of charging.

## Run locally

```bash
cd ..                         # repo root
npm install                    # installs wrangler from root package.json
npx wrangler dev --local       # → http://localhost:8787
```

No setup wizard, no login — `/` shows the first car immediately. Local KV state
lives under `.wrangler/state/`. To exercise the full Stripe flow locally, run
`stripe listen --forward-to http://localhost:8787/webhook/stripe` and set the
test secrets via `.dev.vars` or `wrangler secret put`.

## Deploy

CF Pages auto-deploys on every push to `master` (the existing pipeline that
serves the dashboard). Locally:

```bash
cd ..                  # repo root
npx wrangler deploy    # → https://olx-car-parser.<subdomain>.workers.dev/
```

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│  Cloudflare Worker (this folder)                            │
│  ┌───────────────────────────────────────────────────────┐ │
│  │  Public product                                        │ │
│  │  GET  /          → renderCarPage (one car, paywalled)  │ │
│  │  POST /reserve   → Stripe Checkout Session → 303       │ │
│  │  GET  /unlocked  → verify paid → recordUnlock → reveal │ │
│  │  POST /webhook/stripe → verify sig → recordUnlock      │ │
│  ├───────────────────────────────────────────────────────┤ │
│  │  Internal (Basic Auth)                                 │ │
│  │  GET /analytics/*  /files/*  /data/*  → env.ASSETS     │ │
│  └───────────────────────────────────────────────────────┘ │
│             │                          │                     │
│             ▼                          ▼                     │
│  ┌────────────────────────┐   ┌──────────────────────────┐  │
│  │  KV binding "KV"       │   │  getDeals(zone)          │  │
│  │  cache:deals:{zone}    │   │  fetch hot_deals_{zone}  │  │
│  │  unlock:{uid}:{olx_id} │   │  from latest-data Release│  │
│  └────────────────────────┘   └──────────────────────────┘  │
└────────────────────────────────────────────────────────────┘
```

### Unlock flow

1. First visit sets an `fc_uid` cookie (random 16-byte id) — the only thing
   tying a browser to its paid unlocks. No accounts, no PINs.
2. `POST /reserve` mints a Checkout Session carrying `uid` + `olx_id` in
   `metadata` / `client_reference_id`, redirects to Stripe.
3. On payment, two independent paths record `unlock:{uid}:{olx_id}` in KV (90-day
   TTL): the `/unlocked` success redirect (verifies the session live) **and** the
   signed `/webhook/stripe` event (authoritative; survives a closed tab).
4. Rendering a car checks `unlock:{uid}:{olx_id}` — present ⇒ the OLX link is
   revealed and the card shows "Reservado ✓".

### Geo-sharding

`?zone=` selects `hot_deals_{zone}.json` (`norte`/`centro`/`sul`/`all`, default
`all`). Switchable from the header.

## What is NOT here (and why)

- **Selling the car itself.** The listings are scraped third-party OLX/SV ads —
  we don't own the cars and can't transfer them. The deposit buys a reserved,
  unlocked seller contact for one car (a lead), nothing more.
- **Accounts / login.** Unlocks are tracked by the `fc_uid` cookie. Clearing
  cookies loses the local record (the KV record + Stripe receipt still exist).
- **Refund automation.** The deposit is described as refundable; issue refunds
  from the Stripe Dashboard (the `payment_intent` metadata carries `olx_id`).
- **Push notifications.** Browse-when-you-want surface; push stays in Telegram.
```
