// Server-rendered HTML. Public flip feed: a GRID of cars at /, each tile links
// to a single-car detail page (/car?olx_id=…) whose seller OLX link is paywalled
// behind a Stripe deposit. Everything else (photos, specs, signals) is visible.

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
  header h1 a { color: inherit; }
  header .zone-tag { font-size: 12px; padding: 4px 10px; background: #eef2ff; color: #4338ca; border-radius: 999px; font-weight: 500; }
  header .zones a { font-size: 12px; margin-left: 10px; color: #6b7280; }
  header .zones a.active { color: #1d4ed8; font-weight: 600; }
  main { padding: 20px; max-width: 1180px; margin: 0 auto; }
  .toolbar { display: flex; gap: 8px; margin-bottom: 16px; align-items: center; flex-wrap: wrap; }
  .toolbar a { font-size: 13px; padding: 6px 12px; background: #fff; border: 1px solid #d1d5db; border-radius: 6px; color: #374151; }
  .toolbar a.active { background: #1d4ed8; border-color: #1d4ed8; color: #fff; }
  .toolbar a:hover { text-decoration: none; }
  .toolbar .count { margin-left: auto; font-size: 13px; color: #6b7280; }
  .back { display: inline-block; margin-bottom: 14px; font-size: 14px; color: #374151; }

  /* Grid of car tiles */
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); gap: 14px; }
  .tile { background: #fff; border: 1px solid #e5e7eb; border-radius: 10px; overflow: hidden; display: block; color: inherit; transition: box-shadow .15s, transform .15s; }
  .tile:hover { box-shadow: 0 4px 14px rgba(0,0,0,.08); transform: translateY(-2px); text-decoration: none; }
  .tile .thumb { position: relative; width: 100%; aspect-ratio: 4 / 3; background: #e5e7eb; }
  .tile .thumb img { width: 100%; height: 100%; object-fit: cover; display: block; }
  .tile .badge { position: absolute; top: 8px; right: 8px; font-size: 12px; font-weight: 600; padding: 3px 8px; border-radius: 999px; background: rgba(17,17,17,.72); color: #fff; }
  .tile .badge.unlocked { background: #047857; }
  .tile .tbody { padding: 10px 12px 12px; }
  .tile h3 { margin: 0 0 2px; font-size: 14px; font-weight: 600; line-height: 1.3; }
  .tile .sub { font-size: 12px; color: #6b7280; margin-bottom: 8px; min-height: 16px; }
  .tile .row { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
  .tile .price { font-size: 17px; font-weight: 700; color: #111; }
  .discount-chip { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 12px; font-weight: 600; }
  .discount-strong { background: #d1fae5; color: #065f46; }
  .discount-medium { background: #fef3c7; color: #92400e; }
  .discount-mild { background: #f3f4f6; color: #4b5563; }
  .discount-neutral { background: #f3f4f6; color: #9ca3af; }
  .verdict-chip { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 700; letter-spacing: 0.03em; }
  .verdict-buy { background: #d1fae5; color: #065f46; }
  .verdict-watch { background: #fef3c7; color: #92400e; }
  .profit-chip { display: inline-block; padding: 3px 10px; border-radius: 6px; font-size: 13px; font-weight: 600; background: #ecfdf5; color: #047857; }

  /* Single-car detail */
  .card { background: #fff; border: 1px solid #e5e7eb; border-radius: 12px; overflow: hidden; max-width: 760px; margin: 0 auto; }
  .card-body { padding: 16px 20px 20px 20px; }
  .card-body h2 { margin: 0 0 4px 0; font-size: 21px; font-weight: 700; line-height: 1.25; }
  .card-body .sub { font-size: 14px; color: #6b7280; margin-bottom: 6px; }
  .card-body .tags { font-size: 13px; color: #6b7280; margin-bottom: 14px; }
  .card-body .tags span { display: inline-block; margin-right: 12px; }
  .price-row { display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; margin-bottom: 16px; }
  .price-row .price { font-size: 26px; font-weight: 800; color: #111; }
  .price-row .fair { font-size: 13px; color: #6b7280; }
  .gallery { position: relative; margin: 0 0 16px 0; border-radius: 8px; overflow: hidden; background: #000; }
  .gallery-track { display: flex; overflow-x: auto; scroll-snap-type: x mandatory; scroll-behavior: smooth; -webkit-overflow-scrolling: touch; }
  .gallery-track::-webkit-scrollbar { display: none; }
  .gallery-track { scrollbar-width: none; }
  .gallery-track img { flex: 0 0 100%; width: 100%; max-height: 460px; object-fit: contain; scroll-snap-align: center; user-select: none; -webkit-user-drag: none; }
  .gallery-nav { position: absolute; top: 50%; transform: translateY(-50%); width: 36px; height: 36px; border-radius: 50%; border: 0; background: rgba(0,0,0,0.5); color: #fff; font-size: 22px; line-height: 1; cursor: pointer; display: flex; align-items: center; justify-content: center; padding: 0; }
  .gallery-nav.prev { left: 8px; }
  .gallery-nav.next { right: 8px; }
  .gallery-nav:hover { background: rgba(0,0,0,0.75); }
  .gallery-counter { position: absolute; bottom: 8px; right: 8px; background: rgba(0,0,0,0.6); color: #fff; font-size: 12px; padding: 3px 8px; border-radius: 4px; pointer-events: none; font-variant-numeric: tabular-nums; }
  .gallery.single .gallery-nav, .gallery.single .gallery-counter { display: none; }
  h4.section { margin: 18px 0 8px 0; font-size: 13px; color: #6b7280; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; }
  .desc { font-size: 14px; line-height: 1.55; color: #1f2937; word-wrap: break-word; white-space: pre-wrap; }
  .signals { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; }
  .signal { background: #fafbfc; border: 1px solid #e5e7eb; border-radius: 6px; padding: 8px 12px; }
  .signal .label { font-size: 11px; color: #6b7280; text-transform: uppercase; letter-spacing: 0.04em; }
  .signal .value { font-size: 15px; font-weight: 600; color: #111; margin-top: 2px; }
  .signal.warning .value { color: #b45309; }
  .signal.danger .value { color: #b91c1c; }
  .paywall { margin-top: 20px; border: 1px solid #e5e7eb; border-radius: 10px; padding: 18px 20px; background: #fcfcfd; text-align: center; }
  .paywall .lock { font-size: 15px; font-weight: 600; color: #374151; margin-bottom: 4px; }
  .paywall .note { font-size: 13px; color: #6b7280; margin-bottom: 14px; }
  .reserve-btn { display: inline-block; width: 100%; max-width: 420px; padding: 14px 18px; background: #1d4ed8; color: #fff; border: 0; border-radius: 10px; font-size: 16px; font-weight: 600; cursor: pointer; }
  .reserve-btn:hover { background: #1e40af; }
  .reserve-btn[disabled] { background: #cbd5e1; cursor: not-allowed; }
  .reserved { margin-top: 20px; border: 1px solid #6ee7b7; border-radius: 10px; padding: 18px 20px; background: #ecfdf5; text-align: center; }
  .reserved .ok { font-size: 16px; font-weight: 700; color: #065f46; margin-bottom: 4px; }
  .reserved .note { font-size: 13px; color: #047857; margin-bottom: 14px; }
  .open-link { display: inline-flex; align-items: center; gap: 6px; padding: 12px 22px; background: #047857; color: #fff; border-radius: 8px; font-size: 15px; font-weight: 600; }
  .open-link:hover { background: #065f46; text-decoration: none; }
  .toast { background: #ecfdf5; border: 1px solid #6ee7b7; color: #065f46; padding: 12px 16px; border-radius: 8px; margin-bottom: 16px; font-size: 14px; font-weight: 500; max-width: 760px; margin-left: auto; margin-right: auto; }
  .empty { text-align: center; padding: 80px 20px; color: #6b7280; }
  @media (max-width: 640px) {
    .grid { grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 10px; }
    .card-body h2 { font-size: 19px; }
    .price-row .price { font-size: 22px; }
    header .zones { width: 100%; }
  }
`;

function zoneSwitcher(active) {
  return `<span class="zones">` + Object.keys(ZONE_LABEL).map(z =>
    `<a href="/?zone=${z}" class="${z === active ? "active" : ""}">${z}</a>`
  ).join("") + `</span>`;
}

function layout({ title, body, zone, pageType }) {
  const zoneLabel = zone ? ZONE_LABEL[zone] || zone : "";
  return `<!doctype html>
<html lang="pt">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex,nofollow">
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🚗</text></svg>">
<title>${escapeHtml(title)} · Flipper Club</title>
<style>${BASE_CSS}</style>
</head>
<body>
<header>
  <div>
    <h1><a href="/?zone=${escapeHtml(zone || "all")}">Flipper Club</a></h1>
    ${zoneLabel ? `<span class="zone-tag">${escapeHtml(zoneLabel)}</span>` : ""}
  </div>
  ${zoneSwitcher(zone || "all")}
</header>
<main>${body}</main>
${pageType === "car" ? `<script>
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

// Generic single-message page — degraded feed, empty market, payments off, etc.
export function renderInfo({ zone, title, message }) {
  const body = `<div class="empty">${escapeHtml(message)}</div>`;
  return layout({ title, body, zone, pageType: "info" });
}

// The grid of car tiles at /.
export function renderGrid({ deals, zone, sort, unlockedSet, depositEur, stripeReady }) {
  const tabLabel = s => s === "score" ? "🏆 Melhor aposta"
    : s === "profit" ? "💰 Maior lucro"
    : "🆕 Mais recentes";
  const tab = s => `<a href="/?zone=${zone}&sort=${s}" class="${sort === s ? "active" : ""}">${tabLabel(s)}</a>`;

  const tiles = deals.map(deal => {
    const photos = Array.isArray(deal.photo_urls) ? deal.photo_urls : [];
    const cover = photos[0] || "";
    const name = deal.title || [deal.brand, deal.model].filter(Boolean).join(" ") || "Viatura";
    const unlocked = unlockedSet && unlockedSet.has(deal.olx_id);
    const badge = unlocked
      ? `<span class="badge unlocked">✓ Reservado</span>`
      : `<span class="badge">🔒 ${fmtEur(depositEur)}</span>`;
    const href = `/car?zone=${zone}&olx_id=${encodeURIComponent(deal.olx_id)}`;
    return `<a class="tile" href="${href}">
      <div class="thumb">
        ${cover ? `<img loading="lazy" src="${escapeHtml(cover)}" alt="${escapeHtml(name)}">` : ""}
        ${badge}
      </div>
      <div class="tbody">
        <h3>${escapeHtml(name)}</h3>
        <div class="sub">${deal.year ?? "—"} · ${fmtKm(deal.mileage_km)} · ${escapeHtml(deal.fuel_type || "")}</div>
        <div class="row">
          <span class="price">${fmtEur(deal.price_eur)}</span>
          <span class="discount-chip ${discountClass(deal.discount_pct)}">↓ ${fmtPct(deal.discount_pct)}</span>
        </div>
      </div>
    </a>`;
  }).join("\n");

  const body = `
    <div class="toolbar">
      ${tab("score")}
      ${tab("profit")}
      ${tab("newest")}
      <span class="count">${deals.length} ${deals.length === 1 ? "carro" : "carros"}</span>
    </div>
    <div class="grid">${tiles}</div>`;
  return layout({ title: "Carros", body, zone, pageType: "grid" });
}

// Single-car detail page (paywalled contact). Reached from a grid tile.
export function renderCarPage({ deal, zone, unlocked, justReserved, depositEur, stripeReady }) {
  const photos = Array.isArray(deal.photo_urls) ? deal.photo_urls : [];
  const name = deal.title || [deal.brand, deal.model].filter(Boolean).join(" ") || "Viatura";
  const galleryHtml = photos.length > 0 ? `<div class="gallery ${photos.length === 1 ? "single" : ""}" data-count="${photos.length}">
      <div class="gallery-track">${photos.map((u, i) => `<img loading="lazy" src="${escapeHtml(u)}" alt="${escapeHtml(name)} — foto ${i + 1}">`).join("")}</div>
      <button type="button" class="gallery-nav prev" aria-label="Anterior">‹</button>
      <button type="button" class="gallery-nav next" aria-label="Próxima">›</button>
      <div class="gallery-counter">1 / ${photos.length}</div>
    </div>` : "";

  let contact;
  if (unlocked) {
    contact = `<div class="reserved">
      <div class="ok">✓ Reservado — contacto desbloqueado</div>
      <div class="note">Depósito recebido. Abre o anúncio e fala diretamente com o vendedor.</div>
      <a class="open-link" href="${escapeHtml(deal.url)}" target="_blank" rel="noopener">Abrir anúncio e contactar vendedor →</a>
    </div>`;
  } else if (stripeReady) {
    contact = `<div class="paywall">
      <div class="lock">🔒 Contacto do vendedor bloqueado</div>
      <div class="note">Paga um depósito reembolsável de ${fmtEur(depositEur)} para reservar este carro e desbloquear o link direto ao anúncio.</div>
      <form action="/reserve" method="post">
        <input type="hidden" name="olx_id" value="${escapeHtml(deal.olx_id)}">
        <input type="hidden" name="zone" value="${escapeHtml(zone)}">
        <button type="submit" class="reserve-btn">Reservar e desbloquear — ${fmtEur(depositEur)}</button>
      </form>
    </div>`;
  } else {
    contact = `<div class="paywall">
      <div class="lock">🔒 Contacto do vendedor bloqueado</div>
      <div class="note">As reservas por depósito estarão disponíveis em breve.</div>
      <button class="reserve-btn" disabled>Reservas em breve</button>
    </div>`;
  }

  const toast = justReserved
    ? `<div class="toast">✓ Pagamento confirmado. Este carro está reservado para ti.</div>`
    : "";

  const body = `
    ${toast}
    <a class="back" href="/?zone=${escapeHtml(zone)}">‹ Voltar à lista</a>
    <div class="card">
      <div class="card-body">
        ${galleryHtml}
        <h2>${escapeHtml(name)}</h2>
        <div class="sub">${escapeHtml(deal.brand)} ${escapeHtml(deal.model)} · ${deal.year ?? "—"} · ${fmtKm(deal.mileage_km)} · ${escapeHtml(deal.fuel_type || "")}</div>
        <div class="tags">
          <span>📍 ${escapeHtml(deal.city || "")}${deal.district ? ", " + escapeHtml(deal.district) : ""}</span>
          <span>${fmtRelative(deal.first_seen_at)}</span>
          <span>${escapeHtml(deal.seller_type || "")}</span>
          ${deal.photo_damage_flagged ? `<span style="color:#b91c1c">⚠ photo damage</span>` : ""}
        </div>
        <div class="price-row">
          <span class="price">${fmtEur(deal.price_eur)}</span>
          <span class="fair">justo ${fmtEur(deal.fair_low)}–${fmtEur(deal.fair_high)}</span>
          ${deal.verdict ? `<span class="verdict-chip verdict-${escapeHtml(String(deal.verdict).toLowerCase())}">${deal.verdict === "BUY" ? "🟢" : "🟡"} ${escapeHtml(deal.verdict)}</span>` : ""}
          <span class="discount-chip ${discountClass(deal.discount_pct)}">↓ ${fmtPct(deal.discount_pct)}</span>
          ${deal.est_profit_eur ? `<span class="profit-chip">+${fmtEur(deal.est_profit_eur)}</span>` : ""}
        </div>

        ${contact}

        <h4 class="section">Sinais</h4>
        <div class="signals">
          <div class="signal"><div class="label">Preço pedido</div><div class="value">${fmtEur(deal.price_eur)}</div></div>
          <div class="signal"><div class="label">Justo (mediana)</div><div class="value">${fmtEur(deal.fair_median)}</div></div>
          <div class="signal"><div class="label">Desconto</div><div class="value">${fmtPct(deal.discount_pct)}</div></div>
          <div class="signal"><div class="label">Lucro estimado</div><div class="value">${fmtEur(deal.est_profit_eur)}</div></div>
          <div class="signal ${deal.damage_severity >= 2 ? "warning" : ""} ${deal.damage_severity >= 3 ? "danger" : ""}">
            <div class="label">Damage severity</div>
            <div class="value">${deal.damage_severity ?? "—"} / 3</div>
          </div>
          <div class="signal ${deal.photo_damage_flagged ? "danger" : ""}">
            <div class="label">Photo damage p</div>
            <div class="value">${fmtPct(deal.photo_damage_p)}</div>
          </div>
          <div class="signal"><div class="label">Vendedor</div><div class="value">${escapeHtml(deal.seller_type || "—")}</div></div>
          <div class="signal"><div class="label">Dias no mercado</div><div class="value">${deal.days_on_market ?? "—"}</div></div>
        </div>

        ${(deal.description ?? deal.description_excerpt) ? `<h4 class="section">Descrição</h4>
        <div class="desc">${escapeHtml(deal.description ?? deal.description_excerpt ?? "")}</div>` : ""}
      </div>
    </div>`;

  return layout({ title: name, body, zone, pageType: "car" });
}
