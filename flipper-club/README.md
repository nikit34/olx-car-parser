# flipper-club

Closed B2B web-feed of OLX/StandVirtual flip candidates for a small group of paying flippers. PIN-based auth, geo-sharded by district zone (`norte` / `centro` / `sul`), revocable from an admin panel.

Separate Cloudflare Worker from the public stlite dashboard at the repo root — both live alongside each other and deploy independently.

## Stack

- Cloudflare Workers (the Worker script itself) — `src/index.js`
- Cloudflare KV (one binding) for PIN records and session tokens
- Server-rendered HTML — no build step, no framework, no client framework
- Data comes from `hot_deals_{zone}.json` artifacts uploaded to the `latest-data` Release by `scrape.yml`; the Worker fetches per-zone at request time and caches in KV for 5 min. A missing/broken feed renders a degraded banner (no fake data).

## First-time setup

```bash
cd ..                                # repo root
npm install                          # installs wrangler from root package.json
npx wrangler kv namespace create KV  # returns { id = "abc123..." }
```

Edit the repo-root `wrangler.toml`, replace `REPLACE_WITH_REAL_KV_ID` with the returned KV id. For `wrangler dev --local` the id can be any string (local state lives at `.wrangler/state/`).

## Run locally

> The deploy config lives at the **repo root** `wrangler.toml` now — the worker
> serves both the flipper-club at `/` and (admin-only) the internal stlite
> analytics dashboard at `/analytics/*` from the same domain. Run wrangler
> from the root, not from this folder.

```bash
cd ..                        # repo root
npm install                   # installs wrangler from root package.json
npx wrangler dev --local      # → http://localhost:8787
```

**First visit is a setup wizard, not a login form.** Because no admin PIN exists in KV yet, any route redirects to `/setup`. Click "Criar PIN de admin", and the server generates an 8-char admin PIN and shows it once on the next page — copy it immediately, it won't be shown in plaintext again.

After that:
- `/setup` returns 404 forever (or until you wipe KV).
- `/login` accepts your admin PIN → lands on `/admin`.
- From `/admin` issue test PINs for `norte` / `centro` / `sul` and verify each lands on the dashboard with the right zone-filtered deal list.
- Tick "Admin PIN" in the create form if you ever want a second admin (e.g. for rotation). The system blocks revoking the last active admin so you can't lock yourself out.

## Lockout recovery

If you somehow lose every admin PIN, the only escape hatch is wiping the KV entries for admin PINs so the `/setup` gate re-opens:

```bash
# Lists all PINs (look for is_admin: true)
npx wrangler kv key list --binding=KV --local
# Delete the admin PIN by id
npx wrangler kv key delete --binding=KV --local pin:<id>
# Then refresh the browser — /setup will be available again.
```

For production (`--remote`), same commands without `--local`.

## Deploy

CF Pages auto-deploys on every push to `master` — that's the existing pipeline that already serves the dashboard. Locally:

```bash
cd ..                  # repo root
npx wrangler deploy
# → https://olx-car-parser.<your-subdomain>.workers.dev/
```

The same domain serves:
- `/` `/login` `/setup` `/admin` `/admin/pins/*` — flipper-club (product surface, this folder's worker)
- `/analytics` `/analytics/*` — internal stlite analytics (admin-only proxy to `dashboard-static/`)
- `/files/*` `/data/*` and other asset paths — also admin-only (used by the analytics bundle)

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Cloudflare Worker (this folder)                         │
│  ┌──────────────────────────────────────────────────┐    │
│  │  Routes                                          │    │
│  │  GET  /          → renderDashboard (gated)       │    │
│  │  GET  /login     → renderLogin                   │    │
│  │  POST /login     → findPinByValue → createSession│    │
│  │  POST /logout    → destroySession                │    │
│  │  GET  /admin     → renderAdmin (admin-only)      │    │
│  │  POST /admin/pins/create     → createPin         │    │
│  │  POST /admin/pins/:id/revoke → revokePin         │    │
│  └──────────────────────────────────────────────────┘    │
│             │                          │                  │
│             ▼                          ▼                  │
│  ┌──────────────────┐      ┌──────────────────────────┐   │
│  │  KV binding "KV" │      │  getDeals(zone)          │   │
│  │  pin:{id} → PIN  │      │  chunk 1: getMockDeals() │   │
│  │  session:{tok}→S │      │  chunk 2: fetch from     │   │
│  └──────────────────┘      │  latest-data Release     │   │
│                            └──────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### Auth flow

1. User submits PIN at `/login`.
2. Worker iterates `KV.list({ prefix: "pin:" })` (n ≤ 50 for our scale), constant-time-compares each `value` field.
3. On match: random 24-byte session token minted, stored at `KV[session:{token}]` with `expirationTtl` set so KV garbage-collects on expiry, returned as HttpOnly Secure SameSite=Lax cookie.
4. Every subsequent request: cookie → KV session → KV pin → check `revoked` / `expires_at`. Any fail → redirect `/login`.
5. Admin revoke: flip `revoked: true` on the PIN record. Next request from that session sees `revoked` and gets kicked.

### Geo-sharding

PIN record has `zone: "norte" | "centro" | "sul" | "all"`. Dashboard calls `getDeals(env, session.pin.zone)`, which returns the cohort filtered to that zone. Two flippers in different zones never see each other's deals — alpha decay is structurally contained.

## What is NOT here (and why)

- **Payment / Stripe integration.** Manual for 10 paying users. Admin creates PIN after receiving payment out-of-band (SEPA / Revolut / Stripe Payment Link).
- **Email auth / OTP.** PINs are intentionally low-friction. They're short-lived and revocable; if a flipper shares their PIN, you revoke it and issue a new one.
- **Push notifications.** This is the "browse-when-you-want" surface. Push lives in WhatsApp (manual or future Business API) so the flipper opens it next to their seller-conversation thread.
- **Analytics.** Out of scope for chunk 1. Wrangler Tail gives raw request logs for now; structured analytics later if needed.
