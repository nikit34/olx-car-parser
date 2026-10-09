const STRIPE_API = "https://api.stripe.com/v1";
const REPORT_PATH = /^\/relatorio\/([a-f0-9]{32})(\/relatorio\.pdf)?$/;
const SESSION_ID = /^cs_(live|test)_[A-Za-z0-9]+$/;

const BASE_HEADERS = {
  "cache-control": "private, no-store",
  "x-robots-tag": "noindex, nofollow",
  "referrer-policy": "no-referrer",
};

export function isReportPath(pathname) {
  return pathname === "/relatorio" || pathname.startsWith("/relatorio/");
}

function respond(body, status, type, extra = {}) {
  return new Response(body, { status, headers: { ...BASE_HEADERS, "content-type": type, ...extra } });
}

function notFound() {
  return respond("Not found", 404, "text/plain; charset=utf-8");
}

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function eur(cents) {
  const whole = Math.floor(cents / 100).toString().replace(/\B(?=(\d{3})+(?!\d))/g, ".");
  return `${whole},${String(cents % 100).padStart(2, "0")} €`;
}

function payHref(meta, token) {
  if (!meta.payment_link) return "";
  const u = new URL(meta.payment_link);
  u.searchParams.set("client_reference_id", token);
  return u.toString();
}

function acceptSession(s, meta, token) {
  if (!s) return null;
  const ours = meta.payment_link_id ? s.payment_link === meta.payment_link_id : s.client_reference_id === token;
  if (ours && s.status === "complete" && s.payment_status === "paid"
      && s.amount_total === meta.amount_cents && s.currency === meta.currency) {
    return { session: s.id, amount: s.amount_total, at: new Date().toISOString(), via: "stripe" };
  }
  return null;
}

async function stripeGet(env, path) {
  const r = await fetch(`${STRIPE_API}${path}`, { headers: { authorization: `Bearer ${env.STRIPE_SECRET_KEY}` } });
  return r.ok ? r.json() : null;
}

async function verifySession(env, meta, token, sessionId) {
  try {
    return acceptSession(await stripeGet(env, `/checkout/sessions/${encodeURIComponent(sessionId)}`), meta, token);
  } catch (err) {
    console.warn("report payment check failed", err && err.message);
  }
  return null;
}

async function findPaidSession(env, meta, token) {
  if (!meta.payment_link_id) return null;
  try {
    const list = await stripeGet(env,
      `/checkout/sessions?payment_link=${encodeURIComponent(meta.payment_link_id)}&status=complete&limit=10`);
    for (const s of (list && Array.isArray(list.data)) ? list.data : []) {
      const ok = acceptSession(s, meta, token);
      if (ok) return ok;
    }
  } catch (err) {
    console.warn("report payment lookup failed", err && err.message);
  }
  return null;
}

const PAGE_CSS = `
:root{--ink:#14181d;--muted:#5b6470;--line:#d9dde3;--soft:#f4f5f7;--accent:#0b5cad}
*{box-sizing:border-box}
html,body{margin:0;background:#fff;color:var(--ink)}
body{font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
main{max-width:720px;margin:0 auto;padding:28px 16px 64px}
h1{font-size:24px;line-height:1.25;margin:0 0 8px}
h2{font-size:17px;margin:28px 0 8px}
p{margin:8px 0}
ul{margin:6px 0;padding-left:20px}
li{margin:4px 0}
.muted{color:var(--muted)}
.box{border:1px solid var(--line);background:var(--soft);border-radius:8px;padding:12px 14px;margin:16px 0}
.box.ok{border-color:#9fd3ad;background:#effaf2}
.price{font-size:20px;font-weight:700;margin:4px 0}
.btn{display:inline-block;background:var(--accent);color:#fff;text-decoration:none;font-weight:600;font-size:16px;padding:12px 22px;border-radius:8px;margin:8px 0}
a{color:var(--accent)}
`;

function page(title, body) {
  return `<!doctype html><html lang="pt-PT"><head><meta charset="utf-8">`
    + `<meta name="viewport" content="width=device-width,initial-scale=1">`
    + `<meta name="robots" content="noindex,nofollow"><title>${esc(title)}</title>`
    + `<style>${PAGE_CSS}</style></head><body><main>${body}</main></body></html>`;
}

function list(items) {
  return Array.isArray(items) && items.length ? `<ul>${items.map(i => `<li>${esc(i)}</li>`).join("")}</ul>` : "";
}

function renderOrder(meta, token, { returned, auto }) {
  const price = eur(meta.amount_cents);
  const href = payHref(meta, token);
  let notice = "";
  if (returned && auto) {
    notice = `<div class="box">Ainda não foi possível confirmar o pagamento. Se já pagou, aguarde alguns minutos e volte a abrir esta página.</div>`;
  } else if (returned) {
    notice = `<div class="box ok"><strong>Obrigado pelo pagamento.</strong> A confirmação é feita manualmente e o relatório ficará disponível nesta mesma página, normalmente no próprio dia e no máximo em 2 dias úteis. Enviaremos também um aviso por email.</div>`;
  }
  const after = auto
    ? "Depois do pagamento regressa automaticamente a esta página, onde o relatório fica disponível de imediato para consultar e descarregar em PDF."
    : "Depois do pagamento regressa a esta página. Confirmado o pagamento, o relatório fica disponível aqui para consultar e descarregar em PDF, no prazo de 2 dias úteis (normalmente no próprio dia), e enviamos um aviso por email.";
  const pay = meta.closed
    ? `<p class="muted">Esta encomenda já não está disponível para pagamento. Contacte ola@carsbuyer.org.</p>`
    : href
      ? `<a class="btn" href="${esc(href)}">Pagar ${esc(price)}</a><p class="muted">Pagamento seguro na página da Stripe. ${esc(after)} Se fechar a janela antes de regressar, basta voltar a abrir este endereço.</p>`
      : `<p class="muted">O link de pagamento está a ser preparado.</p>`;
  const body = `${notice}<h1>${esc(meta.title)}</h1>`
    + (meta.ordered ? `<p class="muted">${esc(meta.ordered)}</p>` : "")
    + (meta.summary ? `<h2>${esc(meta.summary_title || "Resumo dos dados")}</h2>${list(meta.summary)}` : "")
    + (meta.includes ? `<h2>O relatório inclui</h2>${list(meta.includes)}` : "")
    + `<h2>Pagamento</h2><p class="price">${esc(price)}</p>`
    + `<p>${esc(meta.price_note || "Preço final, impostos incluídos.")}</p>${pay}`
    + `<p class="muted">Dúvidas: <a href="mailto:ola@carsbuyer.org">ola@carsbuyer.org</a></p>`;
  return page(meta.title, body);
}

function renderPreparing(meta) {
  return page(meta.title, `<div class="box ok"><strong>Pagamento confirmado.</strong> O relatório está a ser preparado e ficará disponível nesta página.</div><h1>${esc(meta.title)}</h1><p class="muted">Dúvidas: <a href="mailto:ola@carsbuyer.org">ola@carsbuyer.org</a></p>`);
}

export async function handleReport(request, env, url) {
  const m = REPORT_PATH.exec(url.pathname);
  if (!m || !env.KV) return notFound();
  if (request.method !== "GET") {
    return respond("Method not allowed", 405, "text/plain; charset=utf-8", { allow: "GET" });
  }
  const token = m[1];
  const meta = await env.KV.get(`report:${token}`, "json");
  if (!meta) return notFound();

  let paid = await env.KV.get(`report:${token}:paid`, "json");
  const sessionId = (url.searchParams.get("session_id") || "").trim();
  const auto = Boolean((env.STRIPE_SECRET_KEY || "").trim());
  if (!paid && auto) {
    paid = (SESSION_ID.test(sessionId) ? await verifySession(env, meta, token, sessionId) : null)
      || await findPaidSession(env, meta, token);
    if (paid) await env.KV.put(`report:${token}:paid`, JSON.stringify(paid));
  }

  if (m[2]) {
    if (!paid) return respond("", 303, "text/plain; charset=utf-8", { location: `/relatorio/${token}` });
    const pdf = await env.KV.get(`report:${token}:pdf`, "arrayBuffer");
    if (!pdf) return respond(renderPreparing(meta), 200, "text/html; charset=utf-8");
    return respond(pdf, 200, "application/pdf", {
      "content-disposition": `attachment; filename="${(meta.filename || "relatorio.pdf").replace(/[^A-Za-z0-9._-]/g, "-")}"`,
    });
  }

  if (paid) {
    const doc = await env.KV.get(`report:${token}:html`);
    return respond(doc || renderPreparing(meta), 200, "text/html; charset=utf-8");
  }
  return respond(renderOrder(meta, token, { returned: Boolean(sessionId), auto }), 200, "text/html; charset=utf-8");
}
