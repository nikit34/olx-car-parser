// Server-rendered HTML templates. Mobile-first, single-page master-detail
// layout for the dashboard. Minimal inline JS — only used to toggle detail
// panels on the dashboard list.

const ZONE_LABEL = {
  norte: "Norte (Porto · Braga · Aveiro)",
  centro: "Centro (Coimbra · Viseu · Leiria)",
  sul: "Sul (Lisboa · Setúbal · Algarve)",
  all: "Portugal — todas as zonas",
};

export function escapeHtml(s) {
  if (s == null) return "";
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function fmtEur(n) {
  if (n == null) return "—";
  return "€" + Math.round(n).toLocaleString("pt-PT");
}

function fmtKm(n) {
  if (n == null) return "—";
  return Math.round(n).toLocaleString("pt-PT") + " km";
}

function fmtPct(p) {
  if (p == null) return "—";
  return (p * 100).toFixed(1) + "%";
}

function fmtRelative(iso) {
  if (!iso) return "—";
  const diffH = (Date.now() - new Date(iso).getTime()) / 3600 / 1000;
  if (diffH < 1) return "há menos de 1h";
  if (diffH < 24) return `há ${Math.floor(diffH)}h`;
  const d = Math.floor(diffH / 24);
  return `há ${d}d`;
}

function discountClass(p) {
  if (p == null) return "discount-neutral";
  if (p >= 0.25) return "discount-strong";
  if (p >= 0.15) return "discount-medium";
  return "discount-mild";
}

const BASE_CSS = `
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; color: #111; background: #f5f6f8; }
  a { color: #1d4ed8; text-decoration: none; }
  a:hover { text-decoration: underline; }
  header { background: #fff; border-bottom: 1px solid #e5e7eb; padding: 12px 20px; display: flex; align-items: center; justify-content: space-between; gap: 16px; flex-wrap: wrap; }
  header h1 { margin: 0; font-size: 18px; font-weight: 600; }
  header .zone-tag { font-size: 12px; padding: 4px 10px; background: #eef2ff; color: #4338ca; border-radius: 999px; font-weight: 500; }
  header nav a { margin-left: 16px; font-size: 14px; color: #374151; }
  header form { display: inline; }
  header button.logout { background: none; border: 0; color: #6b7280; font-size: 14px; cursor: pointer; padding: 0; }
  header button.logout:hover { color: #b91c1c; }
  main { padding: 20px; max-width: 1200px; margin: 0 auto; }
  .toolbar { display: flex; gap: 8px; margin-bottom: 16px; align-items: center; flex-wrap: wrap; }
  .toolbar a { font-size: 13px; padding: 6px 12px; background: #fff; border: 1px solid #d1d5db; border-radius: 6px; color: #374151; }
  .toolbar a.active { background: #1d4ed8; border-color: #1d4ed8; color: #fff; }
  .toolbar .count { margin-left: auto; font-size: 13px; color: #6b7280; }
  .card { background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; margin-bottom: 10px; overflow: hidden; transition: box-shadow 0.15s; }
  .card:hover { box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
  .card-summary { display: grid; grid-template-columns: 120px 1fr auto; gap: 14px; padding: 12px; cursor: pointer; align-items: center; }
  .card-summary img { width: 120px; height: 90px; object-fit: cover; border-radius: 4px; background: #e5e7eb; }
  .card-summary .meta h3 { margin: 0 0 4px 0; font-size: 15px; font-weight: 600; line-height: 1.3; }
  .card-summary .meta .sub { font-size: 13px; color: #6b7280; margin-bottom: 4px; }
  .card-summary .meta .tags { font-size: 12px; color: #6b7280; }
  .card-summary .meta .tags span { display: inline-block; margin-right: 10px; }
  .card-summary .price-block { text-align: right; min-width: 130px; }
  .card-summary .price { font-size: 19px; font-weight: 700; color: #111; }
  .card-summary .fair { font-size: 12px; color: #6b7280; margin-top: 2px; }
  .discount-chip { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 12px; font-weight: 600; margin-top: 4px; }
  .discount-strong { background: #d1fae5; color: #065f46; }
  .discount-medium { background: #fef3c7; color: #92400e; }
  .discount-mild { background: #f3f4f6; color: #4b5563; }
  .discount-neutral { background: #f3f4f6; color: #9ca3af; }
  .profit-chip { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: 500; background: #ecfdf5; color: #047857; margin-left: 6px; }
  .card-detail { display: none; padding: 16px 20px 20px 20px; border-top: 1px solid #e5e7eb; background: #fafbfc; }
  .card.open .card-detail { display: block; }
  .card-detail h4 { margin: 0 0 8px 0; font-size: 14px; color: #6b7280; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; }
  .card-detail .desc { font-size: 14px; line-height: 1.55; color: #1f2937; margin-bottom: 16px; word-wrap: break-word; }
  .gallery { position: relative; margin: 0 0 16px 0; border-radius: 6px; overflow: hidden; background: #000; }
  .gallery-track { display: flex; overflow-x: auto; scroll-snap-type: x mandatory; scroll-behavior: smooth; -webkit-overflow-scrolling: touch; }
  .gallery-track::-webkit-scrollbar { display: none; }
  .gallery-track { scrollbar-width: none; }
  .gallery-track img { flex: 0 0 100%; width: 100%; max-height: 480px; object-fit: contain; scroll-snap-align: center; user-select: none; -webkit-user-drag: none; }
  .gallery-nav { position: absolute; top: 50%; transform: translateY(-50%); width: 36px; height: 36px; border-radius: 50%; border: 0; background: rgba(0,0,0,0.5); color: #fff; font-size: 22px; line-height: 1; cursor: pointer; display: flex; align-items: center; justify-content: center; padding: 0; }
  .gallery-nav.prev { left: 8px; }
  .gallery-nav.next { right: 8px; }
  .gallery-nav:hover { background: rgba(0,0,0,0.75); }
  .gallery-counter { position: absolute; bottom: 8px; right: 8px; background: rgba(0,0,0,0.6); color: #fff; font-size: 12px; padding: 3px 8px; border-radius: 4px; pointer-events: none; font-variant-numeric: tabular-nums; }
  .gallery.single .gallery-nav, .gallery.single .gallery-counter { display: none; }
  .signals { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 10px; margin-bottom: 16px; }
  .signal { background: #fff; border: 1px solid #e5e7eb; border-radius: 6px; padding: 8px 12px; }
  .signal .label { font-size: 11px; color: #6b7280; text-transform: uppercase; letter-spacing: 0.04em; }
  .signal .value { font-size: 15px; font-weight: 600; color: #111; margin-top: 2px; }
  .signal.warning .value { color: #b45309; }
  .signal.danger .value { color: #b91c1c; }
  .open-link { display: inline-flex; align-items: center; gap: 6px; padding: 8px 16px; background: #1d4ed8; color: #fff; border-radius: 6px; font-size: 14px; font-weight: 500; }
  .open-link:hover { background: #1e40af; text-decoration: none; }
  .empty { text-align: center; padding: 80px 20px; color: #6b7280; }
  .login-wrap { max-width: 360px; margin: 80px auto; padding: 32px; background: #fff; border-radius: 12px; box-shadow: 0 8px 32px rgba(0,0,0,0.06); }
  .login-wrap h1 { font-size: 22px; margin: 0 0 4px 0; }
  .login-wrap p.tag { color: #6b7280; font-size: 14px; margin: 0 0 24px 0; }
  .login-wrap input { width: 100%; padding: 12px 14px; font-size: 18px; border: 1px solid #d1d5db; border-radius: 8px; letter-spacing: 0.2em; font-family: ui-monospace, "SF Mono", Menlo, monospace; text-transform: uppercase; }
  .login-wrap input:focus { outline: 2px solid #1d4ed8; outline-offset: -1px; border-color: #1d4ed8; }
  .login-wrap button { width: 100%; padding: 12px; margin-top: 12px; background: #1d4ed8; color: #fff; border: 0; border-radius: 8px; font-size: 15px; font-weight: 500; cursor: pointer; }
  .login-wrap button:hover { background: #1e40af; }
  .login-wrap .error { color: #b91c1c; font-size: 13px; margin-top: 12px; min-height: 18px; }
  table.pins { width: 100%; border-collapse: collapse; background: #fff; border-radius: 8px; overflow: hidden; }
  table.pins th, table.pins td { padding: 10px 12px; text-align: left; border-bottom: 1px solid #e5e7eb; font-size: 13px; }
  table.pins th { background: #f9fafb; color: #6b7280; font-weight: 600; text-transform: uppercase; font-size: 11px; letter-spacing: 0.04em; }
  table.pins .pin-value { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 14px; background: #f3f4f6; padding: 2px 8px; border-radius: 4px; }
  table.pins .pill { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 600; }
  table.pins .pill-active { background: #d1fae5; color: #065f46; }
  table.pins .pill-revoked { background: #fee2e2; color: #991b1b; }
  table.pins .pill-expired { background: #fef3c7; color: #92400e; }
  table.pins .pill-admin { background: #ede9fe; color: #5b21b6; margin-left: 4px; }
  table.pins button.revoke { background: #fee2e2; color: #991b1b; border: 0; padding: 4px 10px; border-radius: 4px; font-size: 12px; cursor: pointer; }
  table.pins button.revoke:hover { background: #fecaca; }
  .admin-form { background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 16px; margin-bottom: 24px; }
  .admin-form h2 { font-size: 16px; margin: 0 0 12px 0; }
  .admin-form .row { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; margin-bottom: 12px; }
  .admin-form label { display: block; font-size: 12px; color: #6b7280; margin-bottom: 4px; }
  .admin-form input, .admin-form select { width: 100%; padding: 8px 10px; font-size: 14px; border: 1px solid #d1d5db; border-radius: 6px; }
  .admin-form button { background: #1d4ed8; color: #fff; border: 0; padding: 8px 18px; border-radius: 6px; cursor: pointer; font-size: 14px; }
  .toast { background: #ecfdf5; border: 1px solid #6ee7b7; color: #065f46; padding: 10px 14px; border-radius: 6px; margin-bottom: 16px; font-size: 14px; }
  .toast .new-pin { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 18px; font-weight: 700; letter-spacing: 0.15em; background: #fff; padding: 4px 10px; border-radius: 4px; margin-left: 6px; }
  @media (max-width: 640px) {
    .card-summary { grid-template-columns: 80px 1fr; }
    .card-summary img { width: 80px; height: 60px; }
    .card-summary .price-block { grid-column: 1 / -1; text-align: left; padding-top: 6px; border-top: 1px dashed #e5e7eb; }
    .card-summary .price-block .price { font-size: 17px; }
    header nav { display: none; }
  }
`;

function layout({ title, body, zone, isAdmin, pageType }) {
  const zoneLabel = zone ? ZONE_LABEL[zone] || zone : "";
  return `<!doctype html>
<html lang="pt">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex,nofollow">
<title>${escapeHtml(title)} · Flipper Club</title>
<style>${BASE_CSS}</style>
</head>
<body>
${pageType === "login" ? "" : `<header>
  <div>
    <h1>Flipper Club</h1>
    ${zoneLabel ? `<span class="zone-tag">${escapeHtml(zoneLabel)}</span>` : ""}
  </div>
  <nav>
    <a href="/">Deals</a>
    ${isAdmin ? `<a href="/admin">Admin</a>` : ""}
    ${isAdmin ? `<a href="/analytics">Analytics</a>` : ""}
    <form action="/logout" method="post" style="display:inline">
      <button class="logout" type="submit">Sair</button>
    </form>
  </nav>
</header>`}
<main>${body}</main>
${pageType === "dashboard" ? `<script>
document.querySelectorAll(".card-summary").forEach(el => {
  el.addEventListener("click", e => {
    if (e.target.tagName === "A") return;
    el.parentElement.classList.toggle("open");
  });
});
document.querySelectorAll(".gallery").forEach(g => {
  const track = g.querySelector(".gallery-track");
  const counter = g.querySelector(".gallery-counter");
  const total = parseInt(g.dataset.count, 10) || 1;
  const update = () => {
    if (!counter || !track.clientWidth) return;
    const idx = Math.min(total - 1, Math.max(0, Math.round(track.scrollLeft / track.clientWidth)));
    counter.textContent = (idx + 1) + " / " + total;
  };
  track.addEventListener("scroll", update, { passive: true });
  g.querySelectorAll(".gallery-nav").forEach(btn => {
    btn.addEventListener("click", e => {
      e.stopPropagation();
      const dir = btn.classList.contains("next") ? 1 : -1;
      track.scrollBy({ left: dir * track.clientWidth, behavior: "smooth" });
    });
  });
});
</script>` : ""}
</body></html>`;
}

export function renderSetup({ error, newPin } = {}) {
  let body;
  if (newPin) {
    body = `
    <div class="login-wrap">
      <h1>Admin criado ✓</h1>
      <p class="tag">Guarda este PIN <strong>agora</strong>. Não voltará a ser mostrado em texto claro.</p>
      <div style="font-family:ui-monospace,'SF Mono',Menlo,monospace;font-size:28px;font-weight:700;letter-spacing:0.2em;background:#f3f4f6;padding:18px;border-radius:8px;text-align:center;margin:16px 0;">${escapeHtml(newPin.value)}</div>
      <a href="/login" style="display:block;text-align:center;background:#1d4ed8;color:#fff;padding:12px;border-radius:8px;font-size:15px;font-weight:500;">Continuar para login →</a>
    </div>`;
  } else {
    body = `
    <div class="login-wrap">
      <h1>Configuração inicial</h1>
      <p class="tag">Primeira visita ao Flipper Club. Cria o teu PIN de admin — gerencias tudo a partir do dashboard depois disto.</p>
      <form action="/setup" method="post" autocomplete="off">
        <label style="font-size:12px;color:#6b7280;display:block;margin-bottom:6px;font-family:inherit;letter-spacing:normal;text-transform:none;">Label (opcional)</label>
        <input type="text" name="label" placeholder="Admin" maxlength="80" style="text-transform:none;letter-spacing:normal;font-family:inherit;font-size:15px;">
        <button type="submit" style="margin-top:16px;">Criar PIN de admin</button>
        ${error ? `<div class="error">${escapeHtml(error)}</div>` : `<div class="error"></div>`}
      </form>
    </div>`;
  }
  return layout({ title: "Setup", body, pageType: "login" });
}

export function renderLogin({ error } = {}) {
  const body = `
  <div class="login-wrap">
    <h1>Flipper Club</h1>
    <p class="tag">Acesso por PIN — clube fechado de revenda de viaturas em Portugal.</p>
    <form action="/login" method="post" autocomplete="off">
      <input type="text" name="pin" placeholder="PIN" maxlength="16" autofocus required>
      <button type="submit">Entrar</button>
      ${error ? `<div class="error">${escapeHtml(error)}</div>` : `<div class="error"></div>`}
    </form>
  </div>`;
  return layout({ title: "Entrar", body, pageType: "login" });
}

function renderCard(deal) {
  const photos = Array.isArray(deal.photo_urls) ? deal.photo_urls : [];
  const cover = photos[0] || "";
  const galleryHtml = photos.length > 0 ? `<div class="gallery ${photos.length === 1 ? 'single' : ''}" data-count="${photos.length}">
        <div class="gallery-track">${photos.map(u => `<img loading="lazy" src="${escapeHtml(u)}" alt="">`).join("")}</div>
        <button type="button" class="gallery-nav prev" aria-label="Anterior">‹</button>
        <button type="button" class="gallery-nav next" aria-label="Próxima">›</button>
        <div class="gallery-counter">1 / ${photos.length}</div>
      </div>` : "";
  return `<div class="card">
    <div class="card-summary">
      ${cover
        ? `<img loading="lazy" src="${escapeHtml(cover)}" alt="">`
        : `<div style="width:120px;height:90px;background:#e5e7eb;border-radius:4px"></div>`}
      <div class="meta">
        <h3>${escapeHtml(deal.title || (deal.brand + " " + deal.model))}</h3>
        <div class="sub">${escapeHtml(deal.brand)} ${escapeHtml(deal.model)} · ${deal.year} · ${fmtKm(deal.mileage_km)} · ${escapeHtml(deal.fuel_type || "")}</div>
        <div class="tags">
          <span>📍 ${escapeHtml(deal.city || "")}, ${escapeHtml(deal.district || "")}</span>
          <span>${fmtRelative(deal.first_seen_at)}</span>
          <span>${escapeHtml(deal.seller_type || "")}</span>
          ${deal.photo_damage_flagged ? `<span style="color:#b91c1c">⚠ photo damage</span>` : ""}
        </div>
      </div>
      <div class="price-block">
        <div class="price">${fmtEur(deal.price_eur)}</div>
        <div class="fair">justo ${fmtEur(deal.fair_low)}–${fmtEur(deal.fair_high)}</div>
        <span class="discount-chip ${discountClass(deal.discount_pct)}">↓ ${fmtPct(deal.discount_pct)}</span>
        ${deal.est_profit_eur ? `<span class="profit-chip">+${fmtEur(deal.est_profit_eur)}</span>` : ""}
      </div>
    </div>
    <div class="card-detail">
      ${galleryHtml}
      <h4>Descrição</h4>
      <div class="desc">${escapeHtml(deal.description_excerpt || "")}</div>
      <h4>Sinais</h4>
      <div class="signals">
        <div class="signal"><div class="label">Preço pedido</div><div class="value">${fmtEur(deal.price_eur)}</div></div>
        <div class="signal"><div class="label">Justo (mediana)</div><div class="value">${fmtEur(deal.fair_median)}</div></div>
        <div class="signal"><div class="label">Desconto</div><div class="value">${fmtPct(deal.discount_pct)}</div></div>
        <div class="signal"><div class="label">Lucro estimado</div><div class="value">${fmtEur(deal.est_profit_eur)}</div></div>
        <div class="signal ${deal.damage_severity >= 2 ? 'warning' : ''} ${deal.damage_severity >= 3 ? 'danger' : ''}">
          <div class="label">Damage severity</div>
          <div class="value">${deal.damage_severity ?? "—"} / 3</div>
        </div>
        <div class="signal ${deal.photo_damage_flagged ? 'danger' : ''}">
          <div class="label">Photo damage p</div>
          <div class="value">${fmtPct(deal.photo_damage_p)}</div>
        </div>
        <div class="signal"><div class="label">Vendedor</div><div class="value">${escapeHtml(deal.seller_type || "—")}</div></div>
        <div class="signal"><div class="label">Dias no mercado</div><div class="value">${deal.days_on_market ?? "—"}</div></div>
      </div>
      <a class="open-link" href="${escapeHtml(deal.url)}" target="_blank" rel="noopener">Abrir no OLX →</a>
    </div>
  </div>`;
}

export function renderDashboard({ deals, zone, sort, isAdmin }) {
  const sorted = [...deals];
  if (sort === "newest") {
    sorted.sort((a, b) => (b.first_seen_at || "").localeCompare(a.first_seen_at || ""));
  } else if (sort === "profit") {
    sorted.sort((a, b) => (b.est_profit_eur || 0) - (a.est_profit_eur || 0));
  } else {
    sorted.sort((a, b) => (b.discount_pct || 0) - (a.discount_pct || 0));
  }
  const tab = s => `<a href="/?sort=${s}" class="${sort === s ? 'active' : ''}">${s === 'discount' ? 'Maior desconto' : s === 'newest' ? 'Mais recentes' : 'Maior lucro'}</a>`;
  const cards = sorted.length === 0
    ? `<div class="empty">Sem deals quentes na tua zona neste momento. Volta dentro de 4h — o próximo scrape vai colocar novos.</div>`
    : sorted.map(renderCard).join("\n");
  const body = `
    <div class="toolbar">
      ${tab("discount")}
      ${tab("newest")}
      ${tab("profit")}
      <span class="count">${sorted.length} ${sorted.length === 1 ? 'deal' : 'deals'}</span>
    </div>
    ${cards}`;
  return layout({ title: "Deals", body, zone, isAdmin, pageType: "dashboard" });
}

export function renderAdmin({ pins, newPin, error, zones, isAdmin }) {
  const zoneOpts = zones.split(",").map(z => `<option value="${z}">${z}</option>`).join("");
  const now = new Date();
  const statusFor = p => {
    if (p.revoked) return `<span class="pill pill-revoked">revoked</span>`;
    if (p.expires_at && new Date(p.expires_at) < now) return `<span class="pill pill-expired">expired</span>`;
    return `<span class="pill pill-active">active</span>`;
  };
  const rows = pins.map(p => `<tr>
    <td><span class="pin-value">${escapeHtml(p.value)}</span></td>
    <td>${escapeHtml(p.label || "—")}${p.is_admin ? `<span class="pill pill-admin">admin</span>` : ""}</td>
    <td>${escapeHtml(p.zone || "—")}</td>
    <td>${statusFor(p)}</td>
    <td>${p.expires_at ? escapeHtml(p.expires_at.slice(0, 16).replace("T", " ")) : "—"}</td>
    <td>${escapeHtml(p.created_at?.slice(0, 16).replace("T", " ") || "—")}</td>
    <td>
      ${!p.revoked && !p.is_admin ? `<form action="/admin/pins/${p.id}/revoke" method="post" style="display:inline" onsubmit="return confirm('Revogar PIN ${p.value}?')">
        <button class="revoke" type="submit">Revoke</button>
      </form>` : ""}
    </td>
  </tr>`).join("");

  const toast = newPin ? `<div class="toast">PIN criado: <span class="new-pin">${escapeHtml(newPin.value)}</span> — guarda-o agora, depois não volta a aparecer em texto claro fora desta tabela.</div>` : "";
  const errorToast = error === "last_admin"
    ? `<div class="toast" style="background:#fee2e2;border-color:#fecaca;color:#991b1b;">Não podes revogar o último admin ativo. Cria outro admin antes de revogar este.</div>`
    : "";

  const body = `
    ${toast}
    ${errorToast}
    <div class="admin-form">
      <h2>Novo PIN</h2>
      <form action="/admin/pins/create" method="post">
        <div class="row">
          <div>
            <label>Label</label>
            <input type="text" name="label" placeholder="João — Porto" required>
          </div>
          <div>
            <label>Zona</label>
            <select name="zone">${zoneOpts}</select>
          </div>
          <div>
            <label>Validade (horas)</label>
            <input type="number" name="ttl_hours" placeholder="24" min="1" max="876000">
          </div>
          <div>
            <label>Notas</label>
            <input type="text" name="notes" placeholder="opcional">
          </div>
        </div>
        <div style="margin-bottom:12px;font-size:13px;">
          <label style="display:inline-flex;align-items:center;gap:6px;color:#374151;">
            <input type="checkbox" name="is_admin" value="1" style="width:auto;">
            Admin PIN (vê admin panel, zona ignorada, não expira — apenas revoke)
          </label>
        </div>
        <button type="submit">Criar PIN</button>
      </form>
    </div>

    <table class="pins">
      <thead><tr>
        <th>PIN</th><th>Label</th><th>Zona</th><th>Status</th><th>Expira</th><th>Criado</th><th></th>
      </tr></thead>
      <tbody>${rows || `<tr><td colspan="7" style="text-align:center;color:#9ca3af;padding:40px">Sem PINs ainda — cria o primeiro acima.</td></tr>`}</tbody>
    </table>`;
  return layout({ title: "Admin", body, isAdmin: true, pageType: "admin" });
}
