import worker from "../../flipper-club/src/index.js";

const TOKEN = "0123456789abcdef0123456789abcdef";
const BASE = `https://carsbuyer.org/relatorio/${TOKEN}`;
const META = {
  title: "Relatório de teste",
  amount_cents: 7500,
  currency: "eur",
  payment_link: "https://buy.stripe.com/test_abc",
  payment_link_id: "plink_test_1",
  filename: "relatorio-teste.pdf",
  summary: ["linha de resumo"],
  includes: ["linha de conteúdo"],
};

let failures = 0;
function assert(cond, msg) { if (!cond) throw new Error(msg); }
async function check(name, fn) {
  try { await fn(); console.log(`  ok   ${name}`); }
  catch (err) { failures++; console.error(`  FAIL ${name}\n       ${err && err.message}`); }
}

function makeEnv(extra = {}) {
  const kv = new Map();
  const env = {
    CANONICAL_HOST: "carsbuyer.org",
    KV: {
      async get(k, type) {
        if (!kv.has(k)) return null;
        const v = kv.get(k);
        if (type === "json") return JSON.parse(v);
        if (type === "arrayBuffer") return typeof v === "string" ? new TextEncoder().encode(v).buffer : v;
        return v;
      },
      async put(k, v) { kv.set(k, v); },
      async list() { return { keys: [] }; },
      async delete(k) { kv.delete(k); },
    },
    ...extra,
  };
  kv.set(`report:${TOKEN}`, JSON.stringify(META));
  return { env, kv };
}

const get = (env, url, init) => worker.fetch(new Request(url, init), env, { waitUntil() {} });

await check("unknown token is 404", async () => {
  const { env } = makeEnv();
  const r = await get(env, "https://carsbuyer.org/relatorio/ffffffffffffffffffffffffffffffff");
  assert(r.status === 404, `status ${r.status}`);
});

await check("malformed token and bare prefix are 404", async () => {
  const { env } = makeEnv();
  for (const p of ["/relatorio", "/relatorio/", "/relatorio/abc", `/relatorio/${TOKEN.toUpperCase()}`, `/relatorio/${TOKEN}/outro.pdf`]) {
    const r = await get(env, `https://carsbuyer.org${p}`);
    assert(r.status === 404, `${p} -> ${r.status}`);
  }
});

await check("unpaid page offers payment and is private", async () => {
  const { env } = makeEnv();
  const r = await get(env, BASE);
  const body = await r.text();
  assert(r.status === 200, `status ${r.status}`);
  assert(body.includes("Pagar 75,00 €"), "no pay button");
  assert(body.includes(`client_reference_id=${TOKEN}`), "pay link lacks client_reference_id");
  assert(!body.includes("Descarregar PDF"), "report leaked before payment");
  assert((r.headers.get("x-robots-tag") || "").includes("noindex"), "missing noindex header");
  assert((r.headers.get("cache-control") || "").includes("no-store"), "cacheable");
});

await check("pdf before payment redirects to the page", async () => {
  const { env, kv } = makeEnv();
  kv.set(`report:${TOKEN}:pdf`, "%PDF-1.4 secret");
  const r = await get(env, `${BASE}/relatorio.pdf`, { redirect: "manual" });
  assert(r.status === 303, `status ${r.status}`);
  assert(r.headers.get("location") === `/relatorio/${TOKEN}`, `location ${r.headers.get("location")}`);
});

await check("non-GET is 405", async () => {
  const { env } = makeEnv();
  const r = await get(env, BASE, { method: "POST", body: "x" });
  assert(r.status === 405, `status ${r.status}`);
});

await check("return without auto verification shows manual notice", async () => {
  const { env } = makeEnv();
  const r = await get(env, `${BASE}?session_id=cs_live_abc123`);
  const body = await r.text();
  assert(r.status === 200, `status ${r.status}`);
  assert(body.includes("Obrigado pelo pagamento"), "no thank-you notice");
  assert(!body.includes("Descarregar PDF"), "report leaked without confirmation");
});

await check("paid flag serves the report and the pdf", async () => {
  const { env, kv } = makeEnv();
  kv.set(`report:${TOKEN}:paid`, JSON.stringify({ via: "manual" }));
  kv.set(`report:${TOKEN}:html`, "<html><a href='x'>Descarregar PDF</a></html>");
  kv.set(`report:${TOKEN}:pdf`, "%PDF-1.4 body");
  const page = await get(env, BASE);
  assert(page.status === 200 && (await page.text()).includes("Descarregar PDF"), "report not served");
  const pdf = await get(env, `${BASE}/relatorio.pdf`);
  assert(pdf.status === 200, `pdf status ${pdf.status}`);
  assert(pdf.headers.get("content-type") === "application/pdf", "pdf content-type");
  assert((pdf.headers.get("content-disposition") || "").includes("relatorio-teste.pdf"), "pdf filename");
  assert((await pdf.text()).startsWith("%PDF"), "pdf body");
});

await check("paid without uploaded report shows preparing page", async () => {
  const { env, kv } = makeEnv();
  kv.set(`report:${TOKEN}:paid`, JSON.stringify({ via: "manual" }));
  const r = await get(env, BASE);
  assert(r.status === 200 && (await r.text()).includes("Pagamento confirmado"), "no preparing notice");
});

async function withStripe(session, fn) {
  const real = globalThis.fetch;
  const calls = [];
  globalThis.fetch = async (url, init) => {
    calls.push({ url: String(url), auth: init && init.headers && init.headers.authorization });
    return new Response(JSON.stringify(session), { status: 200, headers: { "content-type": "application/json" } });
  };
  try { return await fn(calls); } finally { globalThis.fetch = real; }
}

await check("verified Stripe session unlocks and is remembered", async () => {
  const { env, kv } = makeEnv({ STRIPE_SECRET_KEY: "rk_test_x" });
  kv.set(`report:${TOKEN}:html`, "<html>Descarregar PDF</html>");
  const session = { id: "cs_test_ok1", status: "complete", payment_status: "paid", amount_total: 7500, currency: "eur", payment_link: "plink_test_1" };
  await withStripe(session, async calls => {
    const r = await get(env, `${BASE}?session_id=cs_test_ok1`);
    assert(r.status === 200 && (await r.text()).includes("Descarregar PDF"), "not unlocked");
    assert(calls.length === 1 && calls[0].url.endsWith("/checkout/sessions/cs_test_ok1"), "wrong Stripe call");
    assert(calls[0].auth === "Bearer rk_test_x", "no auth header");
  });
  assert(kv.has(`report:${TOKEN}:paid`), "paid flag not stored");
});

await check("Stripe session for another link, amount or unpaid does not unlock", async () => {
  const bad = [
    { id: "cs_test_b1", status: "complete", payment_status: "paid", amount_total: 7500, currency: "eur", payment_link: "plink_other" },
    { id: "cs_test_b2", status: "complete", payment_status: "paid", amount_total: 100, currency: "eur", payment_link: "plink_test_1" },
    { id: "cs_test_b3", status: "open", payment_status: "unpaid", amount_total: 7500, currency: "eur", payment_link: "plink_test_1" },
  ];
  for (const session of bad) {
    const { env, kv } = makeEnv({ STRIPE_SECRET_KEY: "rk_test_x" });
    await withStripe(session, async () => {
      const r = await get(env, `${BASE}?session_id=${session.id}`);
      const body = await r.text();
      assert(!body.includes("Descarregar PDF"), `${session.id} unlocked`);
      assert(body.includes("Ainda não foi possível confirmar"), `${session.id} no pending notice`);
    });
    assert(!kv.has(`report:${TOKEN}:paid`), `${session.id} stored paid flag`);
  }
});

await check("garbage session id is not sent to Stripe", async () => {
  const { env } = makeEnv({ STRIPE_SECRET_KEY: "rk_test_x" });
  await withStripe({}, async calls => {
    await get(env, `${BASE}?session_id=../../v1/charges`);
    assert(calls.length === 0, "called Stripe with an untrusted id");
  });
});

await check("workers.dev host redirects to the canonical report URL", async () => {
  const { env } = makeEnv();
  const r = await get(env, `https://olx-car-parser.permikov134.workers.dev/relatorio/${TOKEN}`, { redirect: "manual" });
  assert(r.status === 301, `status ${r.status}`);
  assert(r.headers.get("location") === BASE, `location ${r.headers.get("location")}`);
});

if (failures) {
  console.error(`\n${failures} report check(s) failed`);
  process.exit(1);
}
console.log("\nreport routes OK");
