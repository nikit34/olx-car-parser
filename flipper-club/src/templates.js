// Server-rendered HTML for the public Flipper Club flip-deal product.
//
// Visual system — "deal terminal for car flippers" (from the Claude Design
// handoff "Flipper Club.dc.html"): a calm, trustworthy light UI with a
// quant/fintech edge. Warm paper background, near-black ink, money-green as the
// signal accent (amber/red for risk grades). Type: Space Grotesk (display),
// Hanken Grotesk (UI), JetBrains Mono (all prices/metrics). Portuguese UI.
//
// Screens: Landing (/) → Mercado feed (/mercado) → Car detail (/car) → Claim
// confirm (/claim) → Unlocked success (/unlocked) → Reservas (/reservas).
//
// The €5 deposit is reframed as a refundable "claim" that buys 24h exclusivity —
// the deal is presented as hidden from other members while you decide. (The
// exclusivity copy is product-vision framing; the backend records a per-visitor
// unlock of the seller's OLX link, which is what the deposit actually buys.)

const ZONE_LABEL = {
  norte: "Norte",
  centro: "Centro",
  sul: "Sul",
  all: "todas as zonas",
};
const ZONE_FULL = {
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

// ── Number formatting (matches the design's JetBrains-Mono number style) ──────
const NB = " "; // narrow no-break space — the design's thousands separator

// Group thousands with a thin space, but only for |n| ≥ 10000 (design quirk:
// €4500 stays ungrouped, €15 499 groups). Keeps the mock's exact look.
export function fmtNum(n) {
  if (n == null) return "—";
  n = Math.round(n);
  const s = Math.abs(n).toString();
  const g = (n < 10000 && n > -10000) ? s : s.replace(/\B(?=(\d{3})+(?!\d))/g, NB);
  return (n < 0 ? "-" : "") + g;
}
export function fmtEur(n) { return n == null ? "—" : "€" + fmtNum(n); }
export function fmtKm(n) {
  if (n == null) return "—";
  return Math.round(n).toString().replace(/\B(?=(\d{3})+(?!\d))/g, NB) + " km";
}
export function fmtPct(p) { return p == null ? "—" : Math.round(p * 100) + "%"; }
function fmtPct1(p) { return p == null ? "—" : (p * 100).toFixed(1) + "%"; }

function fmtRelativeDays(iso) {
  if (!iso) return null;
  const d = Math.max(0, Math.floor((Date.now() - new Date(iso).getTime()) / 86400000));
  return d;
}

// ── Deal → presentation model (grade, risk, gauge, formatted strings) ─────────
// Replicates the design's scoring/grade formula on real deal fields.
function riskOf(deal) {
  const sev = deal.damage_severity || 0;
  if (sev >= 2 || deal.photo_damage_flagged) return "high";
  if (sev >= 1) return "med";
  return "low";
}

const GRADE_COLORS = {
  green: { fg: "#177A47", bg: "#E4F2E9", br: "#BFE3CE" },
  amber: { fg: "#9A6B12", bg: "#F6EEDA", br: "#E8D6A8" },
  red:   { fg: "#AA4632", bg: "#F6E6E1", br: "#E6C7BD" },
};
const RISK_META = {
  low:  { label: "Risco baixo", c: GRADE_COLORS.green },
  med:  { label: "Risco médio", c: GRADE_COLORS.amber },
  high: { label: "Risco alto",  c: GRADE_COLORS.red },
};

// ── Imported-car detector (Tier-0, text-only) ────────────────────────────────
// fair_median is the price of a PT-REGISTERED car. A foreign-plate / not-yet-
// legalized import is cheaper because the Portuguese import tax (ISV) +
// legalização are still unpaid — so its "discount" and "profit" overstate the
// real margin. We have no CO₂ in the feed, so we FLAG the car and show a hedged
// cost RANGE; we never fabricate a precise euro haircut. Detection is text-only
// over title+description (accent-stripped) with a negation guard so genuinely
// Portuguese cars ("matrícula portuguesa", "nacional desde novo") stay clean.
// Validated on the live feed: 7/29 flagged, 0 false positives, Mustang
// "matrícula portuguesa" correctly cleared. See feedback_quality_over_coverage:
// flag, never fake.
function stripAccents(s) {
  return (s == null ? "" : String(s)).toLowerCase()
    .normalize("NFD").replace(/[\u0300-\u036f]/g, "");
}
// URL slug for /preco/{slug}. LOCK-STEP with src/analytics/model_pages.py::slugify
// \u2014 keep byte-identical (NFD-strip \u2192 lower \u2192 non-alnum runs to '-' \u2192 trim).
export function slugify(s) {
  return stripAccents(s).replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");
}
const IMPORT_POS = /\b(importad[ao]s?|importacao|nacionaliz\w*|legaliza(?:r|cao|do|da)|por\s+legalizar|matricul(?:ar|a(?:do|da)?\s+(?:na|nos|em)\s+(?:alemanha|franca|belgica|holanda|espanha|italia|suica))|matricula\s+(?:nl|de|be|fr|es|it|alem\w*|estrangeira|holandesa|alema|francesa|belga)|ainda\s+(?:com|por)\s+matricula\s+estrangeira|vindo\s+d[ao]\s+estrangeiro)\b/;
// Clears cars that are NATIVELY Portuguese (never imported). Deliberately does
// NOT include "já legalizado/nacionalizado" — those are imported-but-legalized
// cars we still want to flag (as legalized), handled by IMPORT_LEGAL instead.
const IMPORT_NEG = /matricula\s+(?:portuguesa|nacional)|nacional\s+desde\s+novo|sempre\s+(?:em\s+)?portugal|documentacao\s+(?:regularizada|portuguesa)|matriculado\s+em\s+portugal|nao\s+(?:e\s+)?importad|sem\s+importacao/;
// Completion words only — bare "vou legalizar" must NOT count as already done.
const IMPORT_LEGAL = /\bja\s+(?:legalizad[oa]|nacionalizad[oa])|legalizacao\s+(?:feita|concluida|paga)|isv\s+pag/;

// The structured `origin` field (OLX/SV param, "national"|"imported") reinforces
// BOTH sides when present: an "imported" origin is a positive even if the text is
// silent; a "national" origin clears a text false-positive. The text still
// supplies the catch-all and the legalized/not-legalized nuance. LOCK-STEP with
// src/analytics/valuations.py::_import_flags.
function importInfo(deal) {
  const hay = stripAccents(deal.title) + " " + stripAccents(deal.description);
  const o = deal.origin; // "national" | "imported" | undefined (until shipped in the feed)
  const pos = o === "imported" || IMPORT_POS.test(hay);
  const neg = o === "national" || IMPORT_NEG.test(hay);
  const flag = pos && !neg;
  return { flag, legalized: flag && IMPORT_LEGAL.test(hay) };
}

// Qualitative legalization-cost band by price tier (no CO₂ ⇒ never a single
// number; always hedged + pointed at the Finanças table).
function isvTier(price) {
  if (price == null) return null;
  if (price < 12000) return "popular";
  if (price <= 30000) return "medio";
  return "premium";
}
const ISV_RANGE = {
  popular: "Custo de legalização (estimativa grosseira): ~€1.500–€4.000 para um carro popular/pequeno. O ISV depende do CO₂ e da idade — confirma na tabela das Finanças.",
  medio:   "Custo de legalização (estimativa grosseira): ~€4.000–€9.000 para um médio/familiar. O ISV depende do CO₂ e da idade — confirma na tabela das Finanças.",
  premium: "Custo de legalização (estimativa grosseira): pode passar de €10.000 num premium/grande cilindrada. O ISV depende do CO₂ e da idade — confirma na tabela das Finanças.",
};

export function present(deal) {
  const price = deal.price_eur;
  const fairMedian = deal.fair_median;
  const disc = deal.discount_pct;            // fraction, e.g. 0.355
  const profit = deal.est_profit_eur;
  const risk = riskOf(deal);
  const days = deal.days_on_market;

  // Bet grade — discount-driven, lightly adjusted for profit, risk, freshness.
  const discN = disc || 0;
  const profN = profit || 0;
  const riskPenalty = risk === "high" ? 14 : risk === "med" ? 6 : 0;
  const freshBonus = (days != null && days < 7) ? 4 : 0;
  const profitBonus = profN > 8000 ? 14 : profN > 4000 ? 7 : 0;
  let score = Math.round(38 + discN * 120 + profitBonus - riskPenalty + freshBonus);
  score = Math.max(42, Math.min(99, score));
  const grade = score >= 82 ? "A+" : score >= 72 ? "A" : score >= 60 ? "B" : "C";
  const gc = (grade === "A+" || grade === "A") ? GRADE_COLORS.green
    : grade === "B" ? GRADE_COLORS.amber : GRADE_COLORS.red;
  const rk = RISK_META[risk];

  // Discount bar — 45% discount = full bar.
  const barW = Math.min(100, Math.round((discN / 0.45) * 100));

  // Fair-price gauge — use the real fair range, fall back to ±band on median.
  const fairLow = deal.fair_low ?? (fairMedian != null ? Math.round(fairMedian * 0.86) : null);
  const fairHigh = deal.fair_high ?? (fairMedian != null ? Math.round(fairMedian * 1.12) : null);
  let gaugePos = 50;
  if (price != null && fairLow != null && fairHigh != null && fairHigh > fairLow) {
    gaugePos = Math.max(3, Math.min(96, Math.round((price - fairLow) / (fairHigh - fairLow) * 100)));
  }

  const photos = Array.isArray(deal.photo_urls) ? deal.photo_urls : [];
  const name = deal.title
    || [deal.brand, deal.model].filter(Boolean).join(" ")
    || "Viatura";
  const days0 = fmtRelativeDays(deal.first_seen_at);

  // Imported-car flag (text-only; never fabricates an ISV number) + km bands.
  const imp = importInfo(deal);
  const km = deal.mileage_km;
  const highKm = km != null && km >= 200000;
  const veryHighKm = km != null && km >= 280000;
  const tier = isvTier(price);
  // Display-only grade clamp: an unpriced import cost can't justify A+/A, so the
  // SHOWN grade caps at B with a dagger. The numeric score/grade that drive the
  // gauge and discount bar are left intact (no double-counting, no fabrication).
  const clampImport = imp.flag && !imp.legalized && (grade === "A+" || grade === "A");
  const gradeDisplay = clampImport ? "B" : grade;
  const gradeDisplayFull = clampImport ? "B †" : `${grade} · ${score}`;
  const gcDisplay = clampImport ? GRADE_COLORS.amber : gc;
  // Mileage is caption-only: fair_median already prices km in (see
  // project_mileage_not_the_lever) — we amber-tint the figure, never re-penalize.
  const kmStr = fmtKm(km);
  const kmSpan = veryHighKm
    ? `<span style="color:${GRADE_COLORS.amber.fg};">${kmStr}</span>`
    : escapeHtml(kmStr);
  const subHtml = `${deal.year ?? "—"} · ${kmSpan} · ${escapeHtml(deal.fuel_type || "—")}`;
  // Buyer-lens framing: euros saved vs the fair median (same magnitude as profit,
  // different verb). Falls back to the est_profit figure if median/price missing.
  const saving = (fairMedian != null && price != null)
    ? Math.round(fairMedian - price) : (profit ?? null);

  return {
    deal, name,
    make: deal.brand || "",
    sub: `${deal.year ?? "—"} · ${fmtKm(deal.mileage_km)} · ${deal.fuel_type || "—"}`,
    subHtml,
    price, fairMedian, disc, profit, risk,
    priceStr: fmtEur(price),
    fairStr: fmtEur(fairMedian),
    profitStr: profit != null ? "+" + fmtEur(profit) : "—",
    saving, savingStr: saving != null ? "+" + fmtEur(saving) : "—",
    discStr: "↓ " + fmtPct(disc),
    grade, score, gradeFull: `${grade} · ${score}`,
    gradeDisplay, gradeDisplayFull, gcDisplay,
    importFlag: imp.flag, importLegalized: imp.legalized,
    isvTier: tier, isvRange: tier ? ISV_RANGE[tier] : null,
    isvEur: deal.isv_eur ?? null,   // computed ISV (imports w/ CO2); else null → qualitative range
    highKm, veryHighKm,
    gc, rk,
    barW,
    fairLow, fairHigh,
    fairLowStr: fmtEur(fairLow), fairHighStr: fmtEur(fairHigh),
    gaugePos,
    photos, cover: photos[0] || "",
    zone: deal.district ? districtZone(deal.district) : null,
    zoneLabel: deal.district || deal.city || "",
    daysOnMarket: deal.days_on_market,
    daysLabel: deal.days_on_market != null ? `${deal.days_on_market}d no mercado` : "",
    sellDays: deal.sell_days ?? null,   // median days-to-sell for this brand+model
    sellN: deal.sell_n ?? null,
    sellerType: deal.seller_type || "Particular",
    sellerInitial: (deal.seller_type || "?").charAt(0).toUpperCase(),
    loc: deal.city || deal.district || "—",
    firstSeenDays: days0,
    verdict: deal.verdict || null,
    href: olxId => `/car?olx_id=${encodeURIComponent(deal.olx_id)}`,
  };
}

// District → zone bucket (best-effort, for the feed zone label).
function districtZone(district) {
  const d = (district || "").toLowerCase();
  if (/porto|braga|aveiro|viana|vila real|bragan/.test(d)) return "norte";
  if (/lisboa|set[uú]bal|faro|[ée]vora|beja|santar[ée]m/.test(d)) return "sul";
  if (/coimbra|viseu|leiria|guarda|castelo branco|aveiro/.test(d)) return "centro";
  return null;
}

// ─────────────────────────────────────────────────────────────────────────────
// Fonts are self-hosted from /fonts/ (dashboard-static/fonts/, served by a
// public Worker route that bypasses the analytics auth gate).
//
// Was: three Google-hosted families behind two cross-origin handshakes
// (fonts.googleapis.com for the CSS, then fonts.gstatic.com for each face).
// Even loaded non-blocking that is two connections and a CSS round-trip before
// the first glyph can be requested — on mobile 4G it was the largest avoidable
// piece of LCP. Now: same-origin, already-warm connection, and the face URL is
// in the HTML instead of behind a stylesheet we have to fetch first.
//
// Three families became two. JetBrains Mono existed only to give prices and
// captions a tabular-ish look; the system monospace stack (var(--mono)) does
// that with zero bytes. The two that carry the actual design — Space Grotesk
// for display, Hanken Grotesk for text — stay.
//
// Both files are the VARIABLE font (one file covers every weight we use), latin
// subset only: pt-PT lives entirely inside U+0000–U+00FF, so the latin-ext and
// vietnamese subsets Google was serving were dead weight.
const FONT_LATIN = "U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,"
  + "U+0304,U+0308,U+0329,U+2000-206F,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD";
const FONT_FACES = `
@font-face{font-family:'Hanken Grotesk';font-style:normal;font-weight:100 900;font-display:swap;src:url(/fonts/hanken-grotesk-var.woff2) format('woff2');unicode-range:${FONT_LATIN};}
@font-face{font-family:'Space Grotesk';font-style:normal;font-weight:300 700;font-display:swap;src:url(/fonts/space-grotesk-var.woff2) format('woff2');unicode-range:${FONT_LATIN};}`;
// Preload both faces: they are same-origin and used above the fold, so the
// browser should not wait to discover them inside the inlined <style>.
const FONT_LINKS = `
<link rel="preload" href="/fonts/hanken-grotesk-var.woff2" as="font" type="font/woff2" crossorigin>
<link rel="preload" href="/fonts/space-grotesk-var.woff2" as="font" type="font/woff2" crossorigin>`;

const CSS = `
${FONT_FACES}
:root{--mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,"Liberation Mono",monospace;}
*{box-sizing:border-box;}
html,body{margin:0;padding:0;}
body{background:#F7F6F3;color:#16181D;font-family:'Hanken Grotesk',sans-serif;-webkit-font-smoothing:antialiased;min-height:100vh;display:flex;flex-direction:column;}
main{flex:1 0 auto;}
::selection{background:#CDEAD8;}
a{color:inherit;text-decoration:none;}
img{max-width:100%;}
.mono{font-family:var(--mono);}
.fc-url{overflow-wrap:anywhere;}
.disp{font-family:'Space Grotesk',sans-serif;}
.wrap{max-width:1180px;margin:0 auto;}

/* Header */
.fc-header{position:sticky;top:0;z-index:30;background:rgba(247,246,243,0.85);backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);border-bottom:1px solid #E8E6E1;}
.fc-header-in{max-width:1180px;margin:0 auto;padding:13px 22px;display:flex;flex-wrap:wrap;align-items:center;gap:16px;}
.fc-brand{display:flex;align-items:center;gap:10px;cursor:pointer;flex-shrink:0;}
.fc-logo{width:30px;height:30px;border-radius:50%;background:#177A47;color:#fff;display:flex;align-items:center;justify-content:center;font-family:var(--mono);font-weight:700;font-size:15px;box-shadow:0 3px 8px rgba(23,122,71,0.32);}
.fc-word{font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:18px;letter-spacing:-0.025em;white-space:nowrap;}
.fc-nav{display:flex;gap:2px;margin-left:6px;}
.fc-nav a{font-family:'Hanken Grotesk',sans-serif;font-weight:500;font-size:13px;padding:7px 12px;border-radius:9px;border:none;background:none;color:#5B606B;cursor:pointer;white-space:nowrap;}
.fc-nav a.active{font-weight:600;background:#ECEAE4;color:#16181D;}
.fc-right{margin-left:auto;display:flex;flex-wrap:wrap;justify-content:flex-end;align-items:center;gap:11px;}
.fc-deposit{display:flex;align-items:center;gap:7px;padding:6px 11px;border-radius:999px;border:1px solid #E0DDD6;background:#fff;cursor:pointer;}
.fc-dot{width:7px;height:7px;border-radius:50%;background:#177A47;flex-shrink:0;}
.fc-deposit span.mono{font-size:12px;font-weight:500;color:#3A3F47;white-space:nowrap;}
.fc-cta-dark{font-family:'Hanken Grotesk',sans-serif;font-weight:600;font-size:13px;padding:8px 15px;border-radius:10px;border:none;background:#16181D;color:#fff;cursor:pointer;white-space:nowrap;}

/* Buttons */
.btn-dark{font-family:'Hanken Grotesk',sans-serif;font-weight:600;border:none;background:#16181D;color:#fff;cursor:pointer;border-radius:12px;}
.btn-green{font-family:'Hanken Grotesk',sans-serif;font-weight:600;border:none;background:#177A47;color:#fff;cursor:pointer;border-radius:12px;}
.btn-green:hover{background:#13633a;}
.btn-outline{font-family:'Hanken Grotesk',sans-serif;font-weight:600;border:1px solid #E2DFD8;background:#fff;color:#16181D;cursor:pointer;border-radius:12px;}
.btn-bright{font-family:'Hanken Grotesk',sans-serif;font-weight:600;border:none;background:#23C268;color:#06301A;cursor:pointer;border-radius:12px;}

/* Landing */
.hero{max-width:1180px;margin:0 auto;padding:64px 22px 36px;}
.hero-grid{display:flex;flex-wrap:wrap;gap:48px;align-items:flex-start;}
.hero-copy{flex:1 1 460px;min-width:min(300px,100%);container-type:inline-size;}
.eyebrow{display:inline-flex;align-items:center;gap:8px;padding:6px 12px;border-radius:999px;background:#E4F2E9;border:1px solid #BFE3CE;margin-bottom:22px;}
.eyebrow .mono{font-size:11px;font-weight:500;color:#177A47;letter-spacing:0.02em;}
.eyebrow .e-dot{width:6px;height:6px;border-radius:50%;background:#177A47;}
h1.hero-title{font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:54px;line-height:1.02;letter-spacing:-0.035em;margin:0 0 18px;color:#16181D;text-wrap:balance;}
.lede{font-size:18px;line-height:1.55;color:#5B606B;margin:0 0 30px;max-width:480px;text-wrap:pretty;}
.hero-actions{display:flex;flex-wrap:wrap;gap:12px;align-items:center;}
.hero-actions .btn-dark{font-size:15px;padding:14px 24px;box-shadow:0 6px 18px -6px rgba(20,24,29,0.4);}
.note{font-family:var(--mono);font-size:13px;color:#8A8F98;}
.hero-stats{display:flex;flex-wrap:wrap;gap:32px;margin-top:42px;}
.stat-num{font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:30px;letter-spacing:-0.02em;}
.stat-num.green{color:#177A47;}
.stat-cap{font-size:13px;color:#8A8F98;margin-top:2px;max-width:128px;line-height:1.3;}
.stat-div{width:1px;background:#E2DFD8;}
/* A divider left at the end of a wrapped stats row points at nothing. */
@container (max-width:520px){.stat-div{display:none;}}
.feature-wrap{flex:1 1 420px;min-width:min(300px,100%);display:flex;flex-direction:column;align-items:stretch;}
.feature-lead{font-family:var(--mono);font-size:11px;letter-spacing:0.06em;text-transform:uppercase;color:#177A47;margin:0 0 10px 2px;display:flex;align-items:center;gap:7px;}
.feature-card{width:100%;max-width:none;background:#fff;border:1px solid #E8E6E1;border-radius:20px;box-shadow:0 30px 60px -30px rgba(20,24,29,0.28);overflow:hidden;cursor:pointer;}

.section{max-width:1180px;margin:0 auto;}
.sec-label{font-family:var(--mono);font-size:12px;color:#8A8F98;letter-spacing:0.06em;margin-bottom:22px;}
.steps{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:16px;}
.step-card{background:#fff;border:1px solid #E8E6E1;border-radius:16px;padding:22px;}
.step-n{font-family:var(--mono);font-weight:700;font-size:13px;color:#177A47;}
.step-t{font-family:'Space Grotesk',sans-serif;font-weight:600;font-size:17px;margin:12px 0 7px;letter-spacing:-0.01em;}
.step-d{font-size:14px;line-height:1.5;color:#5B606B;text-wrap:pretty;}
.cta-banner{background:#16181D;border-radius:22px;padding:44px;display:flex;flex-wrap:wrap;gap:24px;align-items:center;justify-content:space-between;}
.cta-banner h2{font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:32px;letter-spacing:-0.025em;color:#fff;margin:0 0 10px;text-wrap:balance;}
.cta-banner p{font-size:16px;line-height:1.55;color:#A8ADB6;margin:0;max-width:460px;text-wrap:pretty;}
.cta-banner .btn-bright{font-size:15px;padding:14px 26px;}
.indep-note{margin-top:16px;max-width:460px;}
.indep-note>summary{cursor:pointer;font-family:var(--mono);font-size:12px;letter-spacing:0.02em;color:#7FE3AB;list-style:none;display:inline-flex;align-items:center;gap:7px;}
.indep-note>summary::-webkit-details-marker{display:none;}
.indep-note>summary::before{content:"+";font-size:15px;line-height:1;width:16px;text-align:center;}
.indep-note[open]>summary::before{content:"–";}
.indep-note>summary:hover{color:#A6F0C6;}
.indep-note p{margin-top:12px;font-size:14px;color:#9CA1AA;}

/* Striped photo placeholder (fallback when a listing has no photos) */
.striped{background:repeating-linear-gradient(135deg,#EEEBE5 0 13px,#F2EFE9 13px 26px);display:flex;align-items:center;justify-content:center;}
.striped-label{font-family:'Space Grotesk',sans-serif;font-weight:700;color:rgba(20,24,29,0.07);letter-spacing:0.05em;text-transform:uppercase;}

/* Badges */
.grade{position:absolute;top:12px;left:12px;font-family:var(--mono);font-weight:700;font-size:12px;padding:4px 10px;border-radius:8px;}
.risk{position:absolute;top:12px;right:12px;font-family:'Hanken Grotesk',sans-serif;font-weight:600;font-size:11px;padding:4px 10px;border-radius:999px;}
.photo-count{position:absolute;bottom:10px;right:12px;font-family:var(--mono);font-size:10px;color:#7A7F88;background:rgba(255,255,255,0.86);padding:3px 7px;border-radius:6px;}

/* Feed */
.feed{max-width:1180px;margin:0 auto;padding:30px 22px 70px;}
.feed-head h1{font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:30px;letter-spacing:-0.03em;margin:0;}
.feed-head p{font-size:14px;color:#8A8F98;margin:5px 0 0;}
.toolbar{display:flex;flex-wrap:wrap;gap:16px;justify-content:space-between;align-items:center;margin:0 0 12px;}
/* Sticky filter bar — survives the long single-column feed scroll. */
.feed-tools{position:sticky;top:54px;z-index:20;margin:18px -22px 8px;padding:12px 22px 4px;background:rgba(247,246,243,0.92);backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);border-bottom:1px solid #EFECE6;}
.feed-tools .toolbar:last-child{margin-bottom:8px;}
.to-top{position:fixed;bottom:20px;right:16px;width:44px;height:44px;border-radius:50%;background:#16181D;color:#fff;border:none;cursor:pointer;font-size:20px;line-height:1;align-items:center;justify-content:center;box-shadow:0 6px 16px rgba(20,24,29,0.28);z-index:40;display:none;}
.to-top.show{display:flex;}
.chips{display:flex;gap:6px;flex-wrap:wrap;}
.chip{font-family:'Hanken Grotesk',sans-serif;font-weight:500;font-size:13px;padding:8px 14px;border-radius:10px;border:1px solid #E2DFD8;background:#fff;color:#5B606B;cursor:pointer;white-space:nowrap;}
.chip.active{font-weight:600;border-color:#16181D;background:#16181D;color:#fff;}
.chip-count{font-family:var(--mono);font-size:11px;color:#8A8F98;margin-left:2px;}
.chip.active .chip-count{color:#B8BDC6;}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(290px,1fr));gap:18px;}
.tile{cursor:pointer;background:#fff;border:1px solid #E8E6E1;border-radius:18px;overflow:hidden;display:flex;flex-direction:column;transition:transform .16s ease,box-shadow .16s ease,border-color .16s ease;}
.tile:hover{transform:translateY(-4px);box-shadow:0 16px 34px -16px rgba(20,24,29,0.22);border-color:#D8D5CE;}
.tile .thumb{position:relative;height:168px;}
.thumb img{width:100%;height:100%;object-fit:cover;display:block;}
.tile .tbody{padding:16px 17px 17px;display:flex;flex-direction:column;flex:1;}
.tile-title{font-family:'Space Grotesk',sans-serif;font-weight:600;font-size:16px;letter-spacing:-0.01em;line-height:1.25;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.tile-sub{font-family:var(--mono);font-size:11.5px;color:#8A8F98;margin-top:5px;}
.price-row{display:flex;flex-wrap:wrap;align-items:flex-end;gap:9px;margin-top:15px;}
.price{font-family:var(--mono);font-weight:700;font-size:22px;letter-spacing:-0.02em;}
.fair-strike{font-family:var(--mono);font-size:12px;color:#9A9FA8;text-decoration:line-through;margin-bottom:3px;}
.profit-pill{margin-left:auto;font-family:var(--mono);font-weight:700;font-size:13px;color:#177A47;background:#E4F2E9;padding:4px 9px;border-radius:8px;white-space:nowrap;}
.bar-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;}
.bar-head .cap{font-size:11px;color:#8A8F98;}
.bar-head .val{font-family:var(--mono);font-weight:700;font-size:12px;color:#16181D;}
.bar-track{height:6px;border-radius:999px;background:#EEEBE5;overflow:hidden;}
.bar-fill{height:100%;border-radius:999px;}
.tile-foot{display:flex;flex-wrap:wrap;align-items:center;gap:4px 8px;margin-top:auto;padding-top:13px;border-top:1px solid #F0EDE7;font-family:var(--mono);font-size:11px;color:#8A8F98;}
.tile-foot .sep{color:#D8D5CE;}
.tile-foot > span{white-space:nowrap;}
.tile-foot .seller{margin-left:auto;color:#16181D;font-weight:500;}
.badge-unlocked{position:absolute;bottom:10px;left:12px;font-family:var(--mono);font-size:10px;font-weight:700;color:#fff;background:#177A47;padding:3px 8px;border-radius:6px;}

/* Detail */
.detail{max-width:1180px;margin:0 auto;padding:22px 22px 70px;}
.back{font-family:'Hanken Grotesk',sans-serif;font-size:13px;font-weight:500;color:#5B606B;background:none;border:none;cursor:pointer;padding:6px 0;margin-bottom:14px;display:inline-block;}
.detail-grid{display:grid;gap:28px;grid-template-columns:minmax(0,1fr) 360px;grid-template-areas:"gallery side" "extra side";align-items:start;}
.dg-gallery{grid-area:gallery;min-width:0;}
.dg-side{grid-area:side;position:sticky;top:78px;min-width:0;}
.dg-extra{grid-area:extra;min-width:0;}
.hero-photo{position:relative;border-radius:20px;overflow:hidden;border:1px solid #E8E6E1;height:380px;}
.thumbs{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:9px;margin-top:9px;}
.thumb-cell{height:62px;border-radius:11px;overflow:hidden;border:1px solid #EAE7E1;background:repeating-linear-gradient(135deg,#EEEBE5 0 9px,#F2EFE9 9px 18px);}
.thumb-cell img{width:100%;height:100%;object-fit:cover;display:block;}
.panel{background:#fff;border:1px solid #E8E6E1;border-radius:18px;padding:24px;margin-top:22px;}
.panel-title{font-family:'Space Grotesk',sans-serif;font-weight:600;font-size:16px;margin-bottom:16px;letter-spacing:-0.01em;}
.signals{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));background:#fff;border:1px solid #F0EDE7;border-radius:12px;overflow:hidden;}
.signal{background:#fff;padding:14px 15px;box-shadow:-1px 0 0 #F0EDE7,0 -1px 0 #F0EDE7;}
.signal .k{font-size:12px;color:#8A8F98;margin-bottom:6px;}
.signal .v{font-family:var(--mono);font-weight:700;font-size:16px;color:#16181D;}
.signal .v.warn{color:#9A6B12;}
.signal .v.bad{color:#AA4632;}
.desc{font-size:14.5px;line-height:1.65;color:#3A3F47;margin:0;white-space:pre-line;text-wrap:pretty;word-wrap:break-word;}

.side-card{background:#fff;border:1px solid #E8E6E1;border-radius:20px;padding:24px;box-shadow:0 18px 40px -26px rgba(20,24,29,0.2);}
.side-head{display:flex;align-items:flex-start;gap:10px;}
.side-head h1{font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:23px;letter-spacing:-0.02em;line-height:1.18;margin:0;flex:1;text-wrap:balance;}
.grade-badge{flex-shrink:0;font-family:var(--mono);font-weight:700;font-size:12px;padding:5px 10px;border-radius:9px;}
.side-sub{font-family:var(--mono);font-size:12px;color:#8A8F98;margin-top:8px;}
.side-loc{font-size:12.5px;color:#8A8F98;margin-top:6px;}
.side-prices{display:flex;align-items:flex-end;gap:12px;margin-top:20px;}
.side-prices .cap{font-size:12px;color:#8A8F98;}
.side-prices .big{font-family:var(--mono);font-weight:700;font-size:32px;letter-spacing:-0.03em;line-height:1;}
.side-fair{margin-left:auto;text-align:right;flex-shrink:0;}
.side-fair .cap{white-space:nowrap;}
.side-fair .v{font-family:var(--mono);font-weight:500;font-size:18px;color:#5B606B;white-space:nowrap;}
.verdict-row{display:flex;flex-wrap:wrap;align-items:center;gap:10px;margin-top:16px;padding:13px 15px;border-radius:12px;background:#E4F2E9;border:1px solid #C9E7D5;}
.verdict-tag{font-family:var(--mono);font-weight:700;font-size:13px;color:#0F5C35;background:#fff;padding:4px 9px;border-radius:7px;white-space:nowrap;}
.verdict-disc{font-family:var(--mono);font-weight:500;font-size:13px;color:#177A47;white-space:nowrap;}
.verdict-profit{margin-left:auto;font-family:var(--mono);font-weight:700;font-size:17px;color:#0F5C35;white-space:nowrap;}
.gauge-head{display:flex;justify-content:space-between;font-size:11px;color:#8A8F98;font-family:var(--mono);margin-bottom:7px;}
.gauge-track{position:relative;height:8px;border-radius:999px;background:linear-gradient(90deg,#E4F2E9,#EEEBE5);}
.gauge-pin{position:absolute;top:-3px;width:14px;height:14px;border-radius:50%;background:#177A47;border:3px solid #fff;box-shadow:0 2px 6px rgba(20,24,29,0.3);transform:translateX(-50%);}
.side-foot{text-align:center;font-size:12px;color:#9A9FA8;margin-top:12px;line-height:1.5;}

/* Claim module (locked) */
.claim-mod{margin-top:22px;border:1px solid #E8E6E1;border-radius:16px;overflow:hidden;}
.claim-mod-head{padding:16px 17px;background:#FAFAF8;border-bottom:1px solid #EFECE6;display:flex;align-items:center;gap:10px;}
.claim-mod-head .t{font-family:'Space Grotesk',sans-serif;font-weight:600;font-size:15px;}
.claim-mod-head .d{font-size:12px;color:#8A8F98;margin-top:2px;}
.claim-mod-body{padding:16px 17px;}
.unlock-item{display:flex;align-items:flex-start;gap:10px;margin-bottom:11px;}
.tick{flex-shrink:0;width:18px;height:18px;border-radius:50%;background:#E4F2E9;color:#177A47;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;margin-top:1px;}
.unlock-item .t{font-size:13.5px;font-weight:600;color:#16181D;}
.unlock-item .d{font-size:13.5px;color:#5B606B;}
.exclusive{display:flex;align-items:center;gap:9px;margin:14px 0 4px;padding:11px 13px;border-radius:11px;background:#16181D;}
.exclusive .x{font-size:12.5px;color:#D6DAE0;line-height:1.4;}
.exclusive .x b{color:#fff;font-weight:600;}
.claim-mod .btn-green{width:100%;margin-top:14px;font-size:15px;padding:14px;box-shadow:0 8px 20px -8px rgba(23,122,71,0.5);display:block;text-align:center;}
.claim-fine{text-align:center;font-size:11.5px;color:#9A9FA8;margin-top:10px;line-height:1.4;}
.claim-mod[disabled] .btn-disabled,.btn-disabled{width:100%;margin-top:14px;font-size:15px;padding:14px;border-radius:12px;border:none;background:#D8D5CE;color:#fff;cursor:not-allowed;text-align:center;}

/* Claimed module (unlocked) */
.claimed-mod{margin-top:22px;border:1px solid #C9E7D5;border-radius:16px;overflow:hidden;background:#F4FBF6;}
.claimed-head{padding:16px 17px;display:flex;align-items:center;gap:11px;border-bottom:1px solid #DCEFE3;}
.claimed-check{width:30px;height:30px;border-radius:50%;background:#177A47;color:#fff;display:flex;align-items:center;justify-content:center;font-size:15px;flex-shrink:0;}
.claimed-head .t{font-family:'Space Grotesk',sans-serif;font-weight:600;font-size:15px;color:#0F5C35;}
.claimed-head .d{font-size:12px;color:#3F7A5B;margin-top:2px;}
.claimed-body{padding:16px 17px;}
.contact-label{font-family:var(--mono);font-size:11px;color:#8A8F98;letter-spacing:0.04em;margin-bottom:9px;}
.contact-card{display:flex;align-items:center;gap:11px;padding:12px 14px;border-radius:12px;background:#fff;border:1px solid #E2EFE7;}
.avatar{width:34px;height:34px;border-radius:50%;background:#16181D;color:#fff;display:flex;align-items:center;justify-content:center;font-family:'Space Grotesk',sans-serif;font-weight:600;font-size:14px;flex-shrink:0;}
.contact-card .nm{font-weight:600;font-size:14px;}
.contact-card .meta{font-family:var(--mono);font-size:13px;color:#177A47;}
.olx-row{display:flex;gap:9px;margin-top:12px;}
.olx-btn{flex:1;font-family:'Hanken Grotesk',sans-serif;font-weight:600;font-size:14px;padding:12px;border-radius:11px;border:none;background:#177A47;color:#fff;cursor:pointer;text-align:center;display:block;}
.olx-btn:hover{background:#13633a;}
.claimed-note{display:flex;align-items:center;gap:8px;margin-top:13px;font-size:12px;color:#3F7A5B;}

/* Claim confirm page */
.claim-page{max-width:560px;margin:0 auto;padding:42px 22px 70px;}
.claim-card{background:#fff;border:1px solid #E8E6E1;border-radius:22px;overflow:hidden;box-shadow:0 24px 50px -28px rgba(20,24,29,0.24);}
.claim-card-head{padding:26px 26px 22px;border-bottom:1px solid #EFECE6;}
.claim-card-head .eb{font-family:var(--mono);font-size:11px;color:#8A8F98;letter-spacing:0.06em;margin-bottom:12px;}
.claim-card-head .t{font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:26px;letter-spacing:-0.025em;text-wrap:balance;}
.claim-summary .pair{display:inline-flex;align-items:baseline;gap:7px;white-space:nowrap;}
.claim-summary{display:flex;flex-wrap:wrap;align-items:center;gap:6px 14px;margin-top:14px;padding:12px 14px;border-radius:12px;background:#FAFAF8;border:1px solid #EFECE6;}
.claim-summary .mono{font-family:var(--mono);font-weight:700;font-size:14px;}
.claim-summary .cap{font-size:12px;color:#8A8F98;}
.claim-summary .prof{margin-left:auto;font-family:var(--mono);font-weight:700;font-size:14px;color:#177A47;white-space:nowrap;}
.claim-card-body{padding:24px 26px;}
.dep-row{display:flex;justify-content:space-between;align-items:baseline;}
.dep-row .l{font-size:15px;color:#3A3F47;}
.dep-row .r{font-family:var(--mono);font-weight:700;font-size:24px;}
.hr{height:1px;background:#EFECE6;margin:18px 0;}
.benefit{display:flex;align-items:flex-start;gap:11px;margin-bottom:13px;}
.benefit .tick{width:20px;height:20px;}
.benefit .t{font-size:14px;font-weight:600;}
.benefit .d{font-size:13px;color:#5B606B;margin-top:1px;text-wrap:pretty;}
.claim-card-body .btn-green{width:100%;margin-top:8px;font-size:15px;padding:15px;border-radius:13px;box-shadow:0 10px 24px -10px rgba(23,122,71,0.55);}
.cancel-btn{width:100%;margin-top:9px;font-family:'Hanken Grotesk',sans-serif;font-weight:500;font-size:14px;padding:12px;border-radius:13px;border:none;background:none;color:#8A8F98;cursor:pointer;display:block;text-align:center;}
.secure{display:flex;align-items:center;justify-content:center;gap:7px;margin-top:14px;font-size:12px;color:#9A9FA8;}

/* Unlocked success */
.success{max-width:560px;margin:0 auto;padding:42px 22px 70px;}
.success-top{text-align:center;padding:14px 0 6px;}
.success-check{width:62px;height:62px;border-radius:50%;background:#177A47;color:#fff;display:flex;align-items:center;justify-content:center;font-size:30px;margin:0 auto 18px;box-shadow:0 12px 30px -10px rgba(23,122,71,0.5);}
.success-top h1{font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:28px;letter-spacing:-0.025em;margin:0 0 8px;}
.success-top p{font-size:15px;color:#5B606B;margin:0 auto;max-width:380px;text-wrap:pretty;}
.cd-banner{background:#16181D;border-radius:18px;padding:20px 24px;margin:24px 0 16px;display:flex;align-items:center;gap:16px;}
.cd-banner .cap{font-size:12px;color:#A8ADB6;}
.cd-banner .big{font-family:var(--mono);font-weight:700;font-size:34px;color:#23C268;letter-spacing:0.02em;line-height:1.1;}
.cd-banner .dep{text-align:right;}
.cd-banner .dep .v{font-family:var(--mono);font-weight:700;font-size:20px;color:#fff;}
.cd-banner .dep .s{font-size:11px;color:#7E848C;}
.next-steps .step{display:flex;align-items:flex-start;gap:11px;margin-bottom:11px;}
.next-steps .n{flex-shrink:0;font-family:var(--mono);font-weight:700;font-size:12px;color:#177A47;width:18px;}
.next-steps .tx{font-size:14px;color:#3A3F47;text-wrap:pretty;}
.success .btn-outline{width:100%;margin-top:16px;font-size:15px;padding:14px;border-radius:13px;display:block;text-align:center;}

/* Reservas */
.res{max-width:900px;margin:0 auto;padding:30px 22px 70px;}
.res h1{font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:30px;letter-spacing:-0.03em;margin:0;}
.res .lead{font-size:14px;color:#8A8F98;margin:5px 0 24px;}
.res-list{display:flex;flex-direction:column;gap:14px;}
.res-card{background:#fff;border:1px solid #E8E6E1;border-radius:18px;padding:17px;display:flex;flex-wrap:wrap;gap:16px;align-items:center;}
.res-thumb{width:104px;height:74px;border-radius:12px;flex-shrink:0;overflow:hidden;}
.res-thumb img{width:100%;height:100%;object-fit:cover;display:block;}
.res-mid{flex:1;min-width:180px;}
.res-mid .t{font-family:'Space Grotesk',sans-serif;font-weight:600;font-size:16px;letter-spacing:-0.01em;}
.res-mid .s{font-family:var(--mono);font-size:11.5px;color:#8A8F98;margin-top:4px;}
.res-prices{display:flex;gap:7px;margin-top:9px;align-items:center;}
.res-prices .p{font-family:var(--mono);font-weight:700;font-size:13px;}
.res-prices .pr{font-family:var(--mono);font-weight:700;font-size:12px;color:#177A47;background:#E4F2E9;padding:3px 8px;border-radius:7px;white-space:nowrap;}
.res-right{text-align:right;min-width:130px;margin-left:auto;}
.cd-pill{display:inline-flex;align-items:center;gap:6px;padding:5px 10px;border-radius:999px;background:#F4FBF6;border:1px solid #C9E7D5;margin-bottom:8px;}
.cd-pill .mono{font-family:var(--mono);font-size:12px;font-weight:700;color:#0F5C35;}
.res-right .ex{font-size:11px;color:#8A8F98;margin-bottom:9px;}
.res-right .btn-dark{font-size:13px;padding:9px 15px;border-radius:10px;white-space:nowrap;display:inline-block;}
.empty-card{background:#fff;border:1px dashed #DAD7D0;border-radius:18px;padding:54px 24px;text-align:center;}
.empty-card .ic{font-size:30px;margin-bottom:12px;opacity:0.5;}
.empty-card .t{font-family:'Space Grotesk',sans-serif;font-weight:600;font-size:18px;margin-bottom:6px;}
.empty-card p{font-size:14px;color:#8A8F98;margin:0 auto 20px;max-width:340px;text-wrap:pretty;}
.empty-card .btn-green{font-size:14px;padding:12px 22px;display:inline-block;}

/* Per-model year table (SEO model pages) */
.year-tbl{width:100%;border-collapse:collapse;font-size:14px;margin-top:4px;white-space:nowrap;}
.year-tbl th,.year-tbl td{border:1px solid #E8E6E1;padding:9px 12px;text-align:right;}
.year-tbl th{font-family:'Hanken Grotesk',sans-serif;font-weight:600;font-size:12px;color:#5B606B;background:#FAFAF8;text-align:right;}
.year-tbl th:first-child,.year-tbl td:first-child{text-align:left;font-weight:600;}
.year-tbl td{font-family:var(--mono);color:#16181D;}
.year-tbl td.mut{color:#8A8F98;}
.mchips{display:flex;flex-wrap:wrap;gap:7px;}
.mchip{display:inline-block;padding:8px 12px;border-radius:10px;border:1px solid #E2DFD8;background:#fff;font-size:13px;color:#16181D;white-space:nowrap;max-width:100%;overflow:hidden;text-overflow:ellipsis;}
.mchip .mut{color:#8A8F98;font-family:var(--mono);font-size:11.5px;}

/* Info / degraded */
.info{max-width:560px;margin:0 auto;padding:72px 22px;text-align:center;}
.info .ic{font-size:34px;margin-bottom:14px;opacity:0.55;}
.info h1{font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:24px;letter-spacing:-0.02em;margin:0 0 10px;}
.info p{font-size:15px;color:#5B606B;line-height:1.55;margin:0 auto 22px;max-width:420px;text-wrap:pretty;}
.info .btn-dark{font-size:14px;padding:12px 22px;display:inline-block;}

/* Footer */
.footer{border-top:1px solid #E8E6E1;margin-top:20px;}
.footer-in{max-width:1180px;margin:0 auto;padding:22px;display:flex;flex-wrap:wrap;gap:10px;align-items:center;justify-content:space-between;}
.footer .mono{font-family:var(--mono);font-size:11px;color:#9A9FA8;}

/* Gallery (detail hero) */
.gallery{position:relative;border-radius:20px;overflow:hidden;border:1px solid #E8E6E1;height:380px;background:#16181D;}
.gallery-track{display:flex;overflow-x:auto;scroll-snap-type:x mandatory;scroll-behavior:smooth;-webkit-overflow-scrolling:touch;height:100%;scrollbar-width:none;}
.gallery-track::-webkit-scrollbar{display:none;}
.gallery-track img{flex:0 0 100%;width:100%;height:380px;object-fit:cover;scroll-snap-align:center;user-select:none;-webkit-user-drag:none;}
.gallery-nav{position:absolute;top:50%;transform:translateY(-50%);width:36px;height:36px;border-radius:50%;border:0;background:rgba(20,24,29,0.5);color:#fff;font-size:22px;line-height:1;cursor:pointer;display:flex;align-items:center;justify-content:center;padding:0;}
.gallery-nav.prev{left:10px;}
.gallery-nav.next{right:10px;}
.gallery-nav:hover{background:rgba(20,24,29,0.78);}
.gallery-counter{position:absolute;bottom:12px;left:14px;background:rgba(255,255,255,0.88);color:#7A7F88;font-family:var(--mono);font-size:11px;padding:4px 10px;border-radius:7px;pointer-events:none;}
.gallery.single .gallery-nav{display:none;}
.gallery-track img,.thumb-cell img{cursor:zoom-in;}
.zoom-hint{position:absolute;top:10px;right:12px;background:rgba(20,24,29,0.55);color:#fff;font-family:'Hanken Grotesk',sans-serif;font-size:11px;padding:4px 9px;border-radius:7px;pointer-events:none;}
/* Full-screen photo lightbox (click any photo to enlarge) */
.lightbox{position:fixed;inset:0;background:rgba(10,12,15,0.94);z-index:100;display:none;align-items:center;justify-content:center;}
.lightbox.open{display:flex;}
.lb-img{max-width:94vw;max-height:88vh;object-fit:contain;border-radius:6px;box-shadow:0 24px 70px rgba(0,0,0,0.55);}
.lb-close{position:absolute;top:16px;right:18px;width:42px;height:42px;border-radius:50%;border:0;background:rgba(255,255,255,0.14);color:#fff;font-size:24px;line-height:1;cursor:pointer;}
.lb-nav{position:absolute;top:50%;transform:translateY(-50%);width:48px;height:48px;border-radius:50%;border:0;background:rgba(255,255,255,0.14);color:#fff;font-size:28px;line-height:1;cursor:pointer;}
.lb-prev{left:14px;}
.lb-next{right:14px;}
.lb-nav:hover,.lb-close:hover{background:rgba(255,255,255,0.28);}
.gallery.single + .lightbox .lb-nav,.gallery.single + .lightbox .lb-counter{display:none;}
.lb-counter{position:absolute;bottom:18px;left:50%;transform:translateX(-50%);color:#cfd3da;font-family:var(--mono);font-size:12px;}
@media (max-width:760px){.zoom-hint{display:none;}.lb-nav{width:42px;height:42px;}}

/* Below 900px the 360px sidebar is wider than what is left for the gallery,
   so the two-column detail layout stacks earlier than the rest of the site. */
@media (max-width:899px){
  .detail-grid{grid-template-columns:1fr;grid-template-areas:"gallery" "side" "extra";}
  .dg-side{position:static;}
}
@media (max-width:760px){
  h1.hero-title{font-size:38px;}
  .hero{padding:40px 22px 28px;}
  .cta-banner{padding:30px;}
  .cta-banner h2{font-size:26px;}
  /* Compact header so logo + deposit + CTA fit 390px (was overflowing → page-wide
     horizontal scroll). Nav links drop on mobile; destinations stay reachable via
     logo (home), the deposit pill (→ reservas) and the CTA (→ mercado). */
  .fc-nav{display:none;}
  .fc-word{font-size:16px;}
  .fc-header-in{gap:10px;padding:11px 16px;}
  .fc-deposit{padding:5px 9px;}
  .fc-deposit span.mono{font-size:11px;}
  .fc-cta-dark{padding:7px 12px;font-size:12px;}
}
/* A half-visible cell shows a plausible wrong number, so below 760px a table
   either drops the columns that do not fit or, when short enough, stacks into
   one label/value card per row. */
@media (max-width:760px){
  .fc-tbl:not(.tbl-labelled) th:nth-child(n+4),.fc-tbl:not(.tbl-labelled) td:nth-child(n+4){display:none;}
}
@media (max-width:660px){
  .fc-tbl{font-size:12px;}
  .fc-tbl th,.fc-tbl td:first-child,.fc-tbl td.nm{white-space:normal;}
  .fc-tbl th{padding:0 7px 7px 0;}
  .fc-tbl td{padding:7px 7px 7px 0;}
  .year-tbl{font-size:12.5px;}
  .year-tbl th,.year-tbl td{padding:7px 8px;}
}
@media (max-width:600px){
  .tbl-labelled,.tbl-labelled tbody,.tbl-labelled tr,.tbl-labelled td{display:block;width:auto;}
  .tbl-labelled thead{display:none;}
  .tbl-labelled tr{border:1px solid #E8E6E1;border-radius:12px;background:#fff;padding:11px 13px;margin-bottom:8px;}
  .tbl-labelled td{border:0;padding:3px 0;display:flex;align-items:baseline;justify-content:space-between;gap:14px;white-space:nowrap;}
  .tbl-labelled td::before{content:attr(data-l);color:#8A8F98;font-size:11px;font-family:var(--mono);white-space:normal;text-align:left;}
  .tbl-labelled td:first-child{font-weight:600;font-size:13.5px;padding:0 0 7px;border-bottom:1px solid #F2EFE9;margin-bottom:6px;white-space:normal;}
  .tbl-labelled td:first-child::before{content:none;}
  .tbl-labelled tr:hover td{background:none;}
  .fc-scroll:has(.tbl-labelled){margin-inline:0;padding-inline:0;background:none;}
}
@media (max-width:480px){
  .grid{grid-template-columns:1fr;}
  .hero-stats{gap:20px;}
  /* Single-line flex rows drag the grid track past the viewport at 320px. */
  .side-card{padding:18px;}
  .side-prices,.gauge-head{flex-wrap:wrap;}
  .gauge-head{gap:6px;}
  .gauge-head span:nth-child(2){order:3;flex-basis:100%;text-align:center;}
}
/* Between 661px and the two-column hero the card would stretch to the full
   line and the 200px photo becomes a sliver; the steps grid leaves 3+1. */
@media (min-width:661px) and (max-width:1023px){
  .feature-wrap{max-width:560px;}
  .steps{grid-template-columns:repeat(2,minmax(0,1fr));}
}
/* Tablet: the full-size header needs 856px, which at 768 scrolled the page. */
@media (min-width:761px) and (max-width:1023px){
  .fc-header-in{gap:10px;padding:12px 16px;}
  .fc-word{font-size:16px;}
  .fc-nav{margin-left:0;}
  .fc-nav a{font-size:12.5px;padding:7px 8px;}
  .fc-deposit{padding:5px 9px;}
  .fc-deposit span.mono{font-size:11px;}
  .fc-cta-dark{font-size:12px;padding:7px 12px;}
}
@media (max-width:400px){
  .fc-tbl:not(.tbl-labelled) th:nth-child(n+3),.fc-tbl:not(.tbl-labelled) td:nth-child(n+3){display:none;}
  .mchip{font-size:11.5px;padding:7px 7px;}
  .fc-word{display:none;}
  .fc-header-in{gap:8px;padding:11px 12px;}
  .fc-deposit span.mono{font-size:10.5px;}
  .fc-cta-dark{font-size:11.5px;padding:7px 10px;}
}

.fc-consent{position:fixed;left:12px;right:12px;bottom:12px;z-index:60;display:flex;
flex-wrap:wrap;gap:12px;align-items:center;justify-content:space-between;
background:#12161B;border:1px solid #2A3038;border-radius:12px;padding:12px 14px;
box-shadow:0 8px 30px rgba(0,0,0,.45);max-width:760px;margin:0 auto;}
.fc-consent[hidden]{display:none;}
body:has(.fc-consent:not([hidden])){padding-bottom:88px;}
@media (max-width:760px){body:has(.fc-consent:not([hidden])){padding-bottom:135px;}}
@media (max-width:400px){body:has(.fc-consent:not([hidden])){padding-bottom:155px;}}
.fc-consent-text{font-size:13px;color:#C9CFD6;line-height:1.45;flex:1 1 260px;}
.fc-consent-text a{color:#4FB477;}
.fc-consent-btns{display:flex;gap:8px;flex:0 0 auto;}
.fc-consent button{font:inherit;font-size:13px;padding:8px 14px;border-radius:8px;cursor:pointer;}
.fc-consent-no{background:transparent;color:#C9CFD6;border:1px solid #3A424C;}
.fc-consent-yes{background:#177A47;color:#fff;border:1px solid #177A47;}
/* ── Second-layer SEO pages (seo-pages.js) ────────────────────────────────── */
.fc-crumbs{max-width:760px;padding:22px 22px 0;font-size:12.5px;color:#8A8F98;}
/* Provenance line: same shape on every data page — sample, freshness, measure. */
.fc-prov{font-size:11.5px;color:#9A9FA8;line-height:1.6;margin:14px 0 0;}
.fc-chart{width:100%;height:auto;display:block;margin:6px 0 4px;}
.fc-chart .c-ax{font-family:var(--mono);font-size:10px;fill:#9A9FA8;}
@media (max-width:560px){.fc-chart .c-ax{font-size:15px;}}
@media (max-width:400px){.fc-chart .c-ax{font-size:21px;}}
.fc-chart .c-grid{stroke:#E8E6E1;stroke-width:1;}
.fc-chart .c-mark{stroke:#9A6B12;stroke-width:1.4;stroke-dasharray:4 3;}
.fc-chart .c-marklab{font-family:var(--mono);font-size:10px;fill:#9A6B12;}
@media (max-width:560px){.fc-chart .c-marklab{font-size:14px;}}
@media (max-width:400px){.fc-chart .c-marklab{font-size:19px;}}
.fc-wrap{max-width:760px;padding:24px 22px 0;}
.fc-wide{max-width:1180px;padding:24px 22px 0;}
.fc-h1{font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:30px;letter-spacing:-0.02em;margin:0 0 10px;line-height:1.12;}
.fc-h2{font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:19px;letter-spacing:-0.01em;margin:34px 0 12px;}
.fc-h3{font-weight:700;font-size:15px;margin:22px 0 8px;}
.fc-p{font-size:15px;line-height:1.65;color:#3A3F47;margin:0 0 14px;}
.fc-p a,.fc-li a{color:#177A47;font-weight:600;}
.fc-ul{margin:0 0 14px;padding-left:20px;}
.fc-li{font-size:15px;line-height:1.65;color:#3A3F47;margin-bottom:7px;}
/* Insight list: one sentence per rule that fired, so the block is a different
   shape on every page instead of the same paragraph with numbers swapped. */
.fc-insights{list-style:none;margin:0;padding:0;}
.fc-insights li{position:relative;padding:0 0 0 20px;margin-bottom:11px;font-size:15px;line-height:1.6;color:#3A3F47;}
.fc-insights li:before{content:"";position:absolute;left:2px;top:9px;width:6px;height:6px;border-radius:50%;background:#177A47;}
.fc-stat-row{display:flex;flex-wrap:wrap;gap:22px;margin:4px 0 2px;}
.fc-stat{min-width:120px;}
.fc-stat .k{font-size:11px;color:#8A8F98;font-family:var(--mono);letter-spacing:.03em;}
.fc-stat .v{font-family:var(--mono);font-weight:700;font-size:24px;letter-spacing:-.02em;}
.fc-stat .s{font-family:var(--mono);font-size:12px;color:#5B606B;}
.fc-tbl{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:13px;}
.fc-tbl th{text-align:left;font-weight:500;color:#8A8F98;font-size:11px;letter-spacing:.04em;padding:0 10px 8px 0;border-bottom:1px solid #E8E6E1;white-space:nowrap;}
.fc-tbl td{padding:9px 10px 9px 0;border-bottom:1px solid #F0EDE7;white-space:nowrap;}
.fc-tbl td.mut{color:#8A8F98;}
.fc-tbl tr:hover td{background:#FBFAF8;}
.fc-doc{max-width:760px;margin:0 auto;padding:34px 22px 60px;}
.fc-doc h1{font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:32px;letter-spacing:-0.025em;margin:0 0 14px;text-wrap:balance;}
.fc-doc h2{font-family:'Space Grotesk',sans-serif;font-weight:600;font-size:19px;letter-spacing:-0.015em;margin:26px 0 8px;}
.fc-doc p{font-size:15px;line-height:1.6;color:#3A3F47;margin:0 0 12px;text-wrap:pretty;}
.fc-doc a{color:#177A47;font-weight:600;}
.fc-doc code{font-family:var(--mono);font-size:13px;background:#EFECE6;padding:1px 5px;border-radius:5px;}
.fc-scroll{overflow-x:auto;-webkit-overflow-scrolling:touch;scrollbar-width:thin;margin-inline:-22px;padding-inline:22px;
background:linear-gradient(to right,#F7F6F3 30%,rgba(247,246,243,0)) left center,
linear-gradient(to right,rgba(247,246,243,0),#F7F6F3 70%) right center,
radial-gradient(farthest-side at 0 50%,rgba(20,24,29,.13),rgba(20,24,29,0)) left center,
radial-gradient(farthest-side at 100% 50%,rgba(20,24,29,.13),rgba(20,24,29,0)) right center;
background-repeat:no-repeat;background-size:34px 100%,34px 100%,12px 100%,12px 100%;
background-attachment:local,local,scroll,scroll;}
.fc-scroll::-webkit-scrollbar{height:6px;}
.fc-scroll::-webkit-scrollbar-thumb{background:#D8D5CE;border-radius:3px;}
.fc-scroll::-webkit-scrollbar-track{background:transparent;}
.fc-vs{display:grid;grid-template-columns:1fr 1fr;gap:14px;}
@media(max-width:620px){.fc-vs{grid-template-columns:1fr;}}
.fc-vs-card{border:1px solid #E8E6E1;border-radius:14px;background:#fff;padding:16px 18px;}
.fc-vs-card.win{border-color:#BFE3CC;background:#F6FBF8;}
.fc-win{display:inline-block;font-family:var(--mono);font-size:10.5px;font-weight:700;color:#0F5C35;background:#DCF0E4;padding:3px 7px;border-radius:6px;margin-left:6px;}
.fc-yearlinks{display:flex;flex-wrap:wrap;gap:7px;margin:2px 0 0;}
.fc-yearlinks a{font-family:var(--mono);font-size:12.5px;padding:6px 11px;border:1px solid #E8E6E1;border-radius:9px;background:#fff;color:#3A3F47;}
.fc-yearlinks a.on{background:#16181D;border-color:#16181D;color:#fff;}
.fc-yearlinks a:hover{border-color:#177A47;color:#177A47;}
.fc-form{display:flex;flex-wrap:wrap;gap:10px;align-items:flex-end;margin:6px 0 16px;}
.fc-field{display:flex;flex-direction:column;gap:5px;}
.fc-field label{font-family:var(--mono);font-size:11px;color:#8A8F98;letter-spacing:.03em;}
.fc-field input,.fc-field select{font-family:var(--mono);font-size:14px;padding:10px 12px;border:1px solid #E0DDD6;border-radius:10px;background:#fff;color:#16181D;min-width:132px;}
.fc-out{border:1px solid #E8E6E1;border-radius:14px;background:#fff;padding:18px 20px;margin-top:6px;}
.fc-out .big{font-family:var(--mono);font-weight:700;font-size:30px;letter-spacing:-.02em;}
.fc-404{max-width:640px;padding:56px 22px 70px;margin:0 auto;text-align:center;}
`;

// Live 24h-exclusivity countdown + detail gallery. Both guard on their own
// elements, so it's safe to ship on every page.


const PAGE_SCRIPT = `
(function(){
  // Each cell carries its column header so phones can show a label beside every
  // value. Long tables keep the scroller: stacked, they run tens of thousands
  // of pixels. Without JS every table stays a scroller.
  [].forEach.call(document.querySelectorAll('.fc-tbl,.year-tbl'),function(t){
    var hs=[].map.call(t.querySelectorAll('thead th'),function(th){return th.textContent.trim();});
    var rows=t.querySelectorAll('tbody tr');
    if(!hs.length||!rows.length||rows.length>80) return;
    [].forEach.call(rows,function(tr){
      [].forEach.call(tr.children,function(td,i){ if(hs[i]) td.setAttribute('data-l',hs[i]); });
    });
    t.classList.add('tbl-labelled');
  });
  function pad(x){return (x<10?'0':'')+x;}
  function tick(){
    var now=Date.now();
    document.querySelectorAll('.fc-countdown').forEach(function(el){
      var start=parseInt(el.getAttribute('data-claimed-at'),10);
      if(!start){return;}
      var ms=start+86400000-now; if(ms<0){ms=0;}
      var h=Math.floor(ms/3600000),m=Math.floor(ms/60000)%60,s=Math.floor(ms/1000)%60;
      el.textContent=pad(h)+':'+pad(m)+':'+pad(s);
    });
  }
  if(document.querySelector('.fc-countdown')){tick();setInterval(tick,1000);}
  document.querySelectorAll('.gallery').forEach(function(g){
    var track=g.querySelector('.gallery-track');
    var counter=g.querySelector('.gallery-counter');
    var total=parseInt(g.getAttribute('data-count'),10)||1;
    if(!track){return;}
    function update(){
      if(!counter||!track.clientWidth){return;}
      var idx=Math.min(total-1,Math.max(0,Math.round(track.scrollLeft/track.clientWidth)));
      counter.textContent=(idx+1)+' / '+total+' · FOTO DO ANÚNCIO';
    }
    track.addEventListener('scroll',update,{passive:true});
    g.querySelectorAll('.gallery-nav').forEach(function(btn){
      btn.addEventListener('click',function(e){
        e.stopPropagation();e.preventDefault();
        var dir=btn.classList.contains('next')?1:-1;
        track.scrollBy({left:dir*track.clientWidth,behavior:'smooth'});
      });
    });
  });
  // Feed back-to-top FAB — show after a screenful, scroll up on click.
  var toTop=document.querySelector('.to-top');
  if(toTop){
    var onScroll=function(){ if(window.scrollY>700){toTop.classList.add('show');}else{toTop.classList.remove('show');} };
    window.addEventListener('scroll',onScroll,{passive:true}); onScroll();
    toTop.addEventListener('click',function(){window.scrollTo({top:0,behavior:'smooth'});});
  }
  // Photo lightbox — click any gallery/thumb photo to view it full-screen.
  var lb=document.querySelector('.lightbox');
  if(lb){
    var lbImg=lb.querySelector('.lb-img'), lbCount=lb.querySelector('.lb-counter');
    var srcs=[].slice.call(document.querySelectorAll('.gallery-track img')).map(function(im){return im.getAttribute('src');});
    var idx=0;
    function lbShow(i){ if(!srcs.length){return;} idx=(i+srcs.length)%srcs.length; lbImg.setAttribute('src',srcs[idx]); if(lbCount){lbCount.textContent=(idx+1)+' / '+srcs.length;} }
    function lbOpen(i){ lbShow(i); lb.classList.add('open'); lb.setAttribute('aria-hidden','false'); document.body.style.overflow='hidden'; }
    function lbClose(){ lb.classList.remove('open'); lb.setAttribute('aria-hidden','true'); document.body.style.overflow=''; }
    document.querySelectorAll('.gallery-track img,.thumb-cell img').forEach(function(im){
      im.addEventListener('click',function(){ var i=srcs.indexOf(im.getAttribute('src')); lbOpen(i<0?0:i); });
    });
    lb.querySelector('.lb-close').addEventListener('click',lbClose);
    lb.querySelector('.lb-prev').addEventListener('click',function(e){e.stopPropagation();lbShow(idx-1);});
    lb.querySelector('.lb-next').addEventListener('click',function(e){e.stopPropagation();lbShow(idx+1);});
    lb.addEventListener('click',function(e){ if(e.target===lb){lbClose();} });
    document.addEventListener('keydown',function(e){ if(!lb.classList.contains('open')){return;} if(e.key==='Escape'){lbClose();}else if(e.key==='ArrowLeft'){lbShow(idx-1);}else if(e.key==='ArrowRight'){lbShow(idx+1);} });
  }
})();
`;

// Grade rubric — shown as a tooltip so the A+→C badge isn't unexplained authority.
const GRADE_RUBRIC = "Nota A+ → C: desconto vs. preço justo, ajustado ao risco (dano, fotos) e a importação por legalizar. † = limitada por custo de importação não incluído.";
function gradeBadge(p, cls) {
  const c = p.gcDisplay;
  return `<span class="${cls}" title="${GRADE_RUBRIC}" style="background:${c.bg};color:${c.fg};border:1px solid ${c.br};cursor:help;">${p.gradeDisplayFull}</span>`;
}
export function gradeChip(p) {
  const c = p.gcDisplay;
  return `<span class="grade" title="${GRADE_RUBRIC}" style="background:${c.bg};color:${c.fg};border:1px solid ${c.br};cursor:help;">${p.gradeDisplayFull}</span>`;
}
function riskChip(p) {
  return `<span class="risk" style="background:${p.rk.c.bg};color:${p.rk.c.fg};">${p.rk.label}</span>`;
}
// Amber "imported" inline tag — shown in BOTH lenses (a buyer must see it before
// clicking a fake-cheap import). Empty string when the car isn't flagged.
function importTag(p) {
  if (!p.importFlag) return "";
  const c = GRADE_COLORS.amber;
  const txt = p.importLegalized ? "🌍 IMPORTADO" : "🌍 IMPORTADO · ISV?";
  return `<span style="display:inline-block;font-family:var(--mono);font-weight:700;`
    + `font-size:10px;padding:3px 7px;border-radius:6px;margin-top:7px;`
    + `background:${c.bg};color:${c.fg};border:1px solid ${c.br};">${txt}</span>`;
}

// Net-of-ISV honesty line: the headline saving/margin on a not-yet-legalised
// import overstates the real margin (it ignores the ISV the banner warns about).
// Net it explicitly — with the computed € when we have it, qualitatively otherwise.
// `base` = the gross figure being netted (buyer: poupas; reseller: margin).
function netIsvNote(p, base) {
  if (!p.importFlag || p.importLegalized) return "";
  const a = GRADE_COLORS.amber, red = GRADE_COLORS.red;
  if (p.isvEur != null && base != null) {
    const real = Math.round(base - p.isvEur);
    const neg = real <= 0;
    const realStr = (neg ? "−" : "+") + fmtEur(Math.abs(real));
    return `<div style="font-size:12px;color:${neg ? red.fg : a.fg};margin-top:8px;line-height:1.5;">`
      + `Após o ISV estimado (~${fmtEur(p.isvEur)}), a margem real fica em <b>${realStr}</b>`
      + `${neg ? " — a poupança aparente desaparece com o imposto." : "."}</div>`;
  }
  return `<div style="font-size:12px;color:${a.fg};margin-top:8px;line-height:1.5;">`
    + `⚠️ É a poupança <b>antes de legalizar</b> — falta somar o ISV (vários milhares €), por isso a margem real será menor.</div>`;
}

// Photo block for a tile/card — real cover photo, else striped brand placeholder.
export function thumbBlock(p, h, labelSize, eager = false) {
  if (p.cover) {
    // eager+high-priority for above-the-fold LCP images (landing featured card);
    // lazy everywhere else (grid tiles, model-page live deals, reservas).
    const load = eager ? `loading="eager" fetchpriority="high"` : `loading="lazy"`;
    return `<img ${load} src="${escapeHtml(p.cover)}" alt="${escapeHtml(p.name)}">`;
  }
  return `<div class="striped" style="position:absolute;inset:0;"><span class="striped-label" style="font-size:${labelSize}px;">${escapeHtml(p.make)}</span></div>`;
}

// ── Analytics ─────────────────────────────────────────────────────────────────

// Measurement ID приходит из wrangler.toml [vars] и ставится один раз за
// запуск изолята: значение постоянно для деплоя, поэтому модульная переменная
// безопаснее, чем протаскивать env через одиннадцать вызовов layout().
// Пусто = аналитика полностью выключена, ни одного запроса наружу.
let GA4_MEASUREMENT_ID = "";

export function setAnalyticsId(id) {
  GA4_MEASUREMENT_ID = (id || "").trim();
}

// Аудитория продукта в Португалии, то есть в ЕЭЗ, поэтому Consent Mode v2
// объявляется ДО загрузки gtag и по умолчанию запрещает и аналитическое, и
// рекламное хранилище. В этом состоянии GA4 шлёт обезличенные пинги без кук:
// измерение работает, согласия не требуется. Баннер согласия, когда он
// появится, должен звать gtag('consent','update',{...}) - трогать этот код не
// придётся.
function analyticsSnippet() {
  if (!GA4_MEASUREMENT_ID) return "";
  const id = escapeHtml(GA4_MEASUREMENT_ID);
  // Сохранённое согласие применяется сразу после default и до конца загрузки
  // gtag, поэтому у согласившегося измерение полноценное с первого хита.
  return `<script>
window.dataLayer=window.dataLayer||[];function gtag(){dataLayer.push(arguments);}
gtag('consent','default',{ad_storage:'denied',ad_user_data:'denied',ad_personalization:'denied',analytics_storage:'denied',wait_for_update:500});
try{if(localStorage.getItem('fc_consent')==='granted'){gtag('consent','update',{analytics_storage:'granted'});}}catch(e){}
gtag('js',new Date());
</script>
<script async src="https://www.googletagmanager.com/gtag/js?id=${id}"></script>
<script>gtag('config','${id}',{anonymize_ip:true});</script>`;
}

// Баннер согласия. Отказ такой же простой, как согласие - это требование, а не
// вежливость. Скрыт, если выбор уже сделан. Технический cookie fc_uid баннер не
// касается: без него не работает резервация, согласия он не требует.
function consentBanner() {
  if (!GA4_MEASUREMENT_ID) return "";
  return `<div id="fc-consent" hidden class="fc-consent">
  <div class="fc-consent-text">Usamos estatísticas anónimas para saber que páginas ajudam.
    Sem elas o site funciona igual. <a href="/privacidade">Privacidade</a></div>
  <div class="fc-consent-btns">
    <button type="button" data-consent="denied" class="fc-consent-no">Recusar</button>
    <button type="button" data-consent="granted" class="fc-consent-yes">Aceitar</button>
  </div>
</div>
<script>(function(){var box=document.getElementById('fc-consent');if(!box)return;
var saved=null;try{saved=localStorage.getItem('fc_consent');}catch(e){}
if(saved===null)box.hidden=false;
box.addEventListener('click',function(ev){var b=ev.target.closest('[data-consent]');if(!b)return;
var v=b.getAttribute('data-consent');try{localStorage.setItem('fc_consent',v);}catch(e){}
if(typeof gtag==='function')gtag('consent','update',{analytics_storage:v});
box.hidden=true;});})();</script>`;
}

// Событие GA4. Пусто, когда аналитика выключена, поэтому вызов безопасно
// вставлять в любой шаблон. Значения прогоняются через JSON.stringify: они
// попадают внутрь <script>, где escapeHtml не защищает.
function analyticsEvent(name, params = {}) {
  if (!GA4_MEASUREMENT_ID) return "";
  const payload = JSON.stringify(params).replace(/</g, "\\u003c");
  return `<script>window.dataLayer=window.dataLayer||[];` +
    `dataLayer.push(['event',${JSON.stringify(name)},${payload}]);</script>`;
}

// То же, но по отправке формы: событие должно уйти ДО ухода на Stripe.
// transport_type beacon, потому что обычный запрос браузер отменяет при
// навигации, и begin_checkout терялся бы ровно у тех, кто дошёл до оплаты.
function analyticsFormEvent(formSelector, name, params = {}) {
  if (!GA4_MEASUREMENT_ID) return "";
  const payload = JSON.stringify(params).replace(/</g, "\\u003c");
  return `<script>(function(){var f=document.querySelector(${JSON.stringify(formSelector)});` +
    `if(!f)return;f.addEventListener('submit',function(){` +
    `if(typeof gtag==='function')gtag('event',${JSON.stringify(name)},` +
    `Object.assign(${payload},{transport_type:'beacon'}));},{once:true});})();</script>`;
}

// Подставляет client_id GA4 в скрытое поле формы. Нужен вебхуку Stripe, чтобы
// серверное событие purchase легло в ту же сессию, а не создало отдельного
// «пользователя». Куки _ga есть только у согласившегося, поэтому у остальных
// поле остаётся пустым - и это правильно: без согласия отправлять нечего.
function gaClientIdFiller() {
  if (!GA4_MEASUREMENT_ID) return "";
  return `<script>(function(){var f=document.querySelector('input[name="ga_cid"]');if(!f)return;
var m=document.cookie.match(/_ga=GA1\.\d+\.(\d+\.\d+)/);if(m)f.value=m[1];})();</script>`;
}

// ── Shell ─────────────────────────────────────────────────────────────────────
export function layout({ title, body, zone, nav, depositCount, index = false, description = null,
                 canonical = null, jsonLd = null, host = null, image = null, type = "website",
                 ogUrl: ogUrlOverride = null, altJson = null }) {
  const personalised = depositCount != null;
  const dep = (depositCount || 0) * 5;
  const navItem = (key, label, href) =>
    `<a href="${href}" class="${nav === key ? "active" : ""}">${label}</a>`;
  // Public valuation pages are indexable (SEO). Everything else is noindex but
  // still "follow" — ephemeral /car and /avaliar-result pages link to the stable
  // /preco and /mercado pages, and follow keeps that link equity flowing.
  const robots = index ? "index,follow" : "noindex,follow";

  // Open Graph + Twitter cards — the growth channel is organic sharing in
  // Facebook groups, Telegram, Reddit, WhatsApp; a link with no preview card
  // gets a fraction of the clicks. Emitted whenever we know the host (all public
  // pages thread it in). og:image falls back to the branded 1200×630 card;
  // /car passes its real cover photo (raster, absolute) instead.
  const origin = host ? `https://${host}` : null;
  // og:url must identify THIS page. Falling back to the origin root made every
  // shared /car link report the homepage, so Facebook and Telegram attributed
  // the share to "/" instead of the car. Noindex pages have no canonical, hence
  // the explicit override.
  const ogUrl = ogUrlOverride || canonical || (origin ? `${origin}/` : null);
  const ogImage = image || (origin ? `${origin}/og-default.png` : null);
  const usingDefaultImage = !image; // only the default card is known 1200×630
  const social = origin ? [
    `<meta property="og:site_name" content="Carsbuyer">`,
    `<meta property="og:locale" content="pt_PT">`,
    `<meta property="og:type" content="${escapeHtml(type)}">`,
    `<meta property="og:title" content="${escapeHtml(title)}">`,
    description ? `<meta property="og:description" content="${escapeHtml(description)}">` : "",
    ogUrl ? `<meta property="og:url" content="${escapeHtml(ogUrl)}">` : "",
    ogImage ? `<meta property="og:image" content="${escapeHtml(ogImage)}">` : "",
    (ogImage && usingDefaultImage) ? `<meta property="og:image:width" content="1200">` : "",
    (ogImage && usingDefaultImage) ? `<meta property="og:image:height" content="630">` : "",
    ogImage ? `<meta property="og:image:alt" content="${escapeHtml(title)}">` : "",
    `<meta name="twitter:card" content="summary_large_image">`,
    `<meta name="twitter:title" content="${escapeHtml(title)}">`,
    description ? `<meta name="twitter:description" content="${escapeHtml(description)}">` : "",
    ogImage ? `<meta name="twitter:image" content="${escapeHtml(ogImage)}">` : "",
  ] : [];

  const head = [
    description ? `<meta name="description" content="${escapeHtml(description)}">` : "",
    canonical ? `<link rel="canonical" href="${escapeHtml(canonical)}">` : "",
    ...social,
    // jsonLd is our own data (no user input); JSON.stringify already escapes it,
    // and we additionally close-tag-escape to be safe inside <script>.
    jsonLd ? `<script type="application/ld+json">${JSON.stringify(jsonLd).replace(/</g, "\\u003c")}</script>` : "",
    // Machine-readable twin of this page. An agent that finds the HTML can take
    // the numbers without parsing our markup, and the link is what tells it the
    // endpoint exists — a JSON route nobody can discover gets used by nobody.
    altJson ? `<link rel="alternate" type="application/json" href="${escapeHtml(altJson)}" title="Dados desta página em JSON">` : "",
  ].filter(Boolean).join("\n");

  // Keep the SERP title under ~60 chars: append the brand suffix only when it
  // still fits, so keyword-led titles aren't truncated by " · Flipper Club".
  const suffix = " · Carsbuyer";
  const fullTitle = (title.length + suffix.length <= 60) ? title + suffix : title;
  return `<!doctype html>
<html lang="pt-PT">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="${robots}">
<meta name="theme-color" content="#177A47">
<link rel="icon" href="/favicon.ico" sizes="48x48 32x32 16x16">
<link rel="icon" type="image/png" href="/icon-96.png" sizes="96x96">
<link rel="icon" type="image/png" href="/icon-192.png" sizes="192x192">
<link rel="apple-touch-icon" href="/apple-touch-icon.png" sizes="180x180">
<title>${escapeHtml(fullTitle)}</title>
${head}
${analyticsSnippet()}
${FONT_LINKS}
<style>${CSS}</style>
</head>
<body>
${consentBanner()}
<header class="fc-header">
  <div class="fc-header-in">
    <a class="fc-brand" href="/">
      <div class="fc-logo">€</div>
      <span class="fc-word">Carsbuyer</span>
    </a>
    <nav class="fc-nav">
      ${navItem("feed", "Mercado", "/mercado")}
      ${navItem("precos", "Preços", "/precos")}
      ${navItem("avaliar", "Avaliar", "/avaliar")}
      ${navItem("reservas", "Reservas", "/reservas")}
      ${navItem("landing", "Como funciona", "/")}
    </nav>
    <div class="fc-right">
      ${personalised ? `<a class="fc-deposit" href="/reservas">
        <span class="fc-dot"></span>
        <span class="mono">€${dep} em depósito</span>
      </a>` : ""}
      <a class="fc-cta-dark" href="/mercado">Ver mercado</a>
    </div>
  </div>
</header>
<main>${body}</main>
<footer class="footer">
  <div class="footer-in">
    <span class="mono">AVALIAÇÃO INDEPENDENTE&nbsp;· dados de anúncios públicos OLX&nbsp;· estimativas indicativas, não vinculativas&nbsp;· não somos stand nem intermediário</span>
    <span class="mono"><a href="/precos" style="color:#5B606B;">Preços por modelo</a>&nbsp;· <a href="/depreciacao" style="color:#5B606B;">Desvalorização</a>&nbsp;· <a href="/comparar" style="color:#5B606B;">Comparar</a>&nbsp;· <a href="/liquidez" style="color:#5B606B;">Tempo de venda</a>&nbsp;· <a href="/mercado/indice" style="color:#5B606B;">Índice de mercado</a>&nbsp;· <a href="/avaliar" style="color:#5B606B;">Avaliar o meu carro</a>&nbsp;· <a href="/vender" style="color:#5B606B;">Vender o meu carro</a></span>
    <span class="mono"><a href="/metodologia" style="color:#5B606B;">Metodologia</a>&nbsp;· <a href="/sobre" style="color:#5B606B;">Quem somos</a>&nbsp;· <a href="/isv" style="color:#5B606B;">Simulador ISV</a>&nbsp;· <a href="/importar" style="color:#5B606B;">Importar da Alemanha</a>&nbsp;· Portugal&nbsp;🇵🇹</span>
  </div>
</footer>
<script>${PAGE_SCRIPT}</script>
</body></html>`;
}

// ── Landing (/) ───────────────────────────────────────────────────────────────
export function renderLanding({ stats, featured, depositEur, depositCount, host }) {
  const f = featured ? present(featured) : null;
  const featureCard = f ? `
    <div class="feature-wrap">
      <div class="feature-lead"><span class="e-dot" style="background:#177A47;"></span>Destaque de hoje</div>
      <a class="feature-card" href="/car?olx_id=${encodeURIComponent(featured.olx_id)}">
        <div class="thumb" style="position:relative;height:200px;">
          ${thumbBlock(f, 200, 34, true)}
          ${gradeChip(f)}
          ${riskChip(f)}
        </div>
        <div style="padding:18px;">
          <div class="disp" style="font-weight:600;font-size:17px;letter-spacing:-0.01em;">${escapeHtml(f.name)}</div>
          <div class="mono" style="font-size:12px;color:#8A8F98;margin-top:4px;">${f.subHtml}</div>
          ${importTag(f)}
          <div class="price-row">
            <div><div style="font-size:11px;color:#8A8F98;">Pedido</div><div class="mono" style="font-weight:700;font-size:24px;letter-spacing:-0.02em;">${f.priceStr}</div></div>
            <div style="margin-bottom:3px;"><span class="fair-strike">${f.fairStr}</span></div>
            <span class="profit-pill" style="font-size:14px;padding:5px 10px;">${f.saving != null ? "poupas " + fmtEur(f.saving) : f.profitStr}</span>
          </div>
          <div class="btn-outline" style="width:100%;margin-top:16px;font-size:14px;padding:11px;background:#FAFAF8;text-align:center;">Ver análise completa</div>
        </div>
      </a>
    </div>` : "";

  const steps = [
    { n: "01", t: "Varremos o OLX", d: "Milhares de anúncios de carros em Portugal, recolhidos e atualizados ao longo do dia." },
    { n: "02", t: "Calculamos o preço justo", d: "Cada carro é comparado com anúncios semelhantes para estimar a mediana de mercado." },
    { n: "03", t: "Avisamos-te dos riscos", d: "Importação por legalizar (ISV em falta), indícios de dano nas fotos, e há quanto tempo o carro está parado. Cada carro leva uma nota de A+ a C." },
    { n: "04", t: "Compras com confiança — ou revendes", d: `${fmtEur(depositEur)} reembolsáveis bloqueiam o contacto do vendedor para ti durante 24h. Falas, negoceias, e o depósito volta para a tua carteira.` },
  ];

  const body = `
    <section class="hero">
      <div class="hero-grid">
        <div class="hero-copy">
          <div class="eyebrow"><span class="e-dot"></span><span class="mono">OLX PORTUGAL · AVALIAÇÃO INDEPENDENTE · ${stats.deals} CARROS ANALISADOS HOJE</span></div>
          <h1 class="hero-title">Antes de comprares, sabe quanto vale mesmo.</h1>
          <p class="lede">Comparamos cada anúncio do OLX com dezenas de carros semelhantes e dizemos-te o preço justo de mercado — e o que o vendedor não te conta: importação por legalizar, indícios de dano, tempo a encalhar. Não pagues a mais.</p>
          <div class="hero-actions">
            <a class="btn-dark" href="/mercado">Ver os ${stats.deals} carros abaixo do preço&nbsp;&nbsp;→</a>
            <a class="btn-outline" href="/avaliar" style="font-size:15px;padding:14px 22px;">Quanto vale o meu carro?&nbsp;&nbsp;→</a>
          </div>
          <div class="note" style="margin-top:10px;">Comprar ou vender · sem registo · grátis</div>
          <div style="margin-top:24px;">
            <div class="mono" style="font-size:12px;color:#8A8F98;margin-bottom:9px;">O que queres fazer?</div>
            <div class="chips">
              <a class="chip active" href="/mercado?view=comprar">🛒 Comprar bem</a>
              <a class="chip" href="/mercado?view=revender">📈 Revender com margem</a>
              <a class="chip" href="/vender">Vender o meu carro</a>
            </div>
          </div>
          <div class="hero-stats">
            <div><div class="stat-num">${stats.avgDisc}</div><div class="stat-cap">abaixo do preço justo, em média</div></div>
            <div class="stat-div"></div>
            <div><div class="stat-num green">${stats.totalProfit}</div><div class="stat-cap">poupança total detetada</div></div>
            <div class="stat-div"></div>
            <div><div class="stat-num">${fmtEur(depositEur)}</div><div class="stat-cap">para reservar · reembolsável</div></div>
          </div>
        </div>
        ${featureCard}
      </div>
    </section>

    <section class="section" style="padding:40px 22px 30px;">
      <div class="sec-label">COMO FUNCIONA</div>
      <div class="steps">
        ${steps.map(s => `<div class="step-card"><div class="step-n">${s.n}</div><div class="step-t">${s.t}</div><div class="step-d">${escapeHtml(s.d)}</div></div>`).join("")}
      </div>
    </section>

    <section class="section" style="padding:8px 22px 0;">
      <div class="cta-banner" style="background:#fff;border:1px solid #E8E6E1;">
        <div style="flex:1 1 360px;">
          <h2 style="color:#16181D;">Vais vender o teu carro?</h2>
          <p style="color:#5B606B;">Diz-nos o modelo e o ano e fazemos-te uma avaliação independente — para saberes por quanto anunciar sem deixar dinheiro em cima da mesa.</p>
        </div>
        <a class="btn-dark" href="/avaliar" style="font-size:15px;padding:14px 26px;">Avaliar o meu carro&nbsp;&nbsp;→</a>
      </div>
    </section>

    <section class="section" style="padding:24px 22px 70px;">
      <div class="cta-banner">
        <div style="flex:1 1 360px;">
          <h2>A independência é o produto.</h2>
          <p>O OLX nunca te vai dizer que o anúncio está caro, nem que aquele preço baixo é de um carro ainda por legalizar. Nós dizemos — e mostramos o que o vendedor não escreve: quantas vezes já baixou o preço e há quanto tempo o carro está à venda. Reservar um carro (${fmtEur(depositEur)} reembolsáveis) desbloqueia o contacto do vendedor durante 24h.</p>
          <details class="indep-note">
            <summary>Como ganhamos dinheiro — e porque é que a avaliação não muda por isso</summary>
            <p>Não cobramos comissão ao vendedor e ninguém paga para aparecer melhor avaliado. Ganhamos de duas formas: quando um vendedor pede propostas de compra e um comprador profissional paga por esse contacto, e quando um comprador encomenda um relatório de histórico através da nossa ligação de parceiro. Se um dia houver publicidade nas páginas de preços, estará assinalada como tal. Nenhuma destas receitas depende do valor que mostramos — a avaliação sai dos anúncios e do modelo, não de quem paga.</p>
          </details>
        </div>
        <a class="btn-bright" href="/mercado">Ver os carros avaliados&nbsp;&nbsp;→</a>
      </div>
    </section>`;

  const origin = host ? `https://${host}` : "";
  const jsonLd = origin ? {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "Organization",
        "@id": `${origin}/#org`,
        "name": "Carsbuyer",
        "url": `${origin}/`,
        "description": "Avaliação independente de carros usados em Portugal a partir de anúncios ativos do OLX.",
        "logo": `${origin}/og-default.png`,
      },
      {
        "@type": "WebSite",
        "@id": `${origin}/#site`,
        "name": "Carsbuyer",
        "url": `${origin}/`,
        "inLanguage": "pt-PT",
        "publisher": { "@id": `${origin}/#org` },
        "potentialAction": {
          "@type": "SearchAction",
          "target": { "@type": "EntryPoint", "urlTemplate": `${origin}/avaliar?q={search_term_string}` },
          "query-input": "required name=search_term_string",
        },
      },
    ],
  } : null;
  return layout({
    title: "Avaliação grátis de carros usados em Portugal",
    description: "Avaliação independente e grátis de carros usados em Portugal a partir de anúncios do OLX: preço justo de mercado, avisos de importação/ISV e carros abaixo do preço. Não pagues a mais.",
    body, zone: "all", nav: "landing", depositCount, index: true,
    host, canonical: origin ? `${origin}/` : null, jsonLd,
  });
}

// ── Mercado feed (/mercado) ─────────────────────────────────────────────────────
// `view` is the intent lens: "comprar" (default, buyer-first) or "revender"
// (importer/flipper). It only RELABELS the same decision_score-ranked feed —
// never re-sorts or filters — so we never imply a precision the ranking lacks.
export function renderGrid({ deals, zone, sort, view, unlockedSet, depositEur, depositCount,
                             zoneCounts, host, modelLinks = [], builtAt = null,
                             yearLinks = [], contextLinks = [], districtLinks = [],
                             jsonLd = null, feedBuiltAt = null }) {
  const lens = view === "revender" ? "revender" : "comprar";
  const profitLabel = lens === "comprar" ? "💰 Maior poupança" : "💰 Maior margem";
  const tabLabel = s => s === "score" ? "🏆 Melhor aposta" : s === "profit" ? profitLabel : "🆕 Mais recentes";
  const q = extra => `/mercado?zone=${extra.zone ?? zone}&sort=${extra.sort ?? sort}&view=${lens}`;
  const sortChip = s => `<a href="${q({ sort: s })}" class="chip ${sort === s ? "active" : ""}">${tabLabel(s)}</a>`;
  const zoneChip = z => {
    const labels = { all: "Todas", norte: "Norte", centro: "Centro", sul: "Sul" };
    const c = zoneCounts && typeof zoneCounts[z] === "number" ? zoneCounts[z] : null;
    const count = c != null ? ` <span class="chip-count">${c}</span>` : "";
    return `<a href="${q({ zone: z })}" class="chip ${zone === z ? "active" : ""}">${labels[z]}${count}</a>`;
  };
  const lensChip = (v, label) => `<a href="/mercado?zone=${zone}&sort=${sort}&view=${v}" class="chip ${lens === v ? "active" : ""}">${label}</a>`;

  const tiles = deals.map(deal => {
    const p = present(deal);
    const unlocked = unlockedSet && unlockedSet.has(deal.olx_id);
    const href = `/car?zone=${zone}&view=${lens}&olx_id=${encodeURIComponent(deal.olx_id)}`;
    const photoCount = p.photos.length ? `<span class="photo-count">FOTO 1/${p.photos.length}</span>` : "";
    const unlockedBadge = unlocked ? `<span class="badge-unlocked">✓ RESERVADO</span>` : "";
    // Buyer cares "how much under fair" (poupas); reseller cares raw margin
    // (which a flagged import overstates → asterisk + footnote, never a fake cut).
    const pill = lens === "comprar"
      ? `<div class="profit-pill">${p.saving != null ? "poupas " + fmtEur(p.saving) : p.profitStr}</div>`
      : `<div class="profit-pill">${p.profitStr}${p.importFlag ? "*" : ""}</div>`;
    const importNote = (lens === "revender" && p.importFlag)
      ? `<div style="font-size:10.5px;color:${GRADE_COLORS.amber.fg};margin-top:5px;">* antes do ISV + legalização</div>`
      : "";
    return `<a class="tile" href="${href}">
      <div class="thumb">
        ${thumbBlock(p, 168, 28)}
        ${gradeChip(p)}
        ${riskChip(p)}
        ${photoCount}
        ${unlockedBadge}
      </div>
      <div class="tbody">
        <div class="tile-title">${escapeHtml(p.name)}</div>
        <div class="tile-sub">${p.subHtml}</div>
        ${importTag(p)}
        <div class="price-row">
          <div class="price">${p.priceStr}</div>
          <div class="fair-strike">${p.fairStr}</div>
          ${pill}
        </div>
        <div style="margin-top:14px;">
          <div class="bar-head"><span class="cap">Desconto vs. justo</span><span class="val">${p.discStr}</span></div>
          <div class="bar-track"><div class="bar-fill" style="width:${p.barW}%;background:${p.gc.fg};"></div></div>
          ${importNote}
        </div>
        <div class="tile-foot">
          <span>${escapeHtml(p.zoneLabel)}</span><span class="sep">·</span><span>${escapeHtml(p.daysLabel)}</span>
          <span class="seller">${escapeHtml(p.sellerType)}</span>
        </div>
      </div>
    </a>`;
  }).join("\n");

  const chipRow = (label, links) => (links && links.length
    ? `<div class="sec-label" style="margin-top:26px;">${label}</div>
      <div class="mchips">${links.slice(0, 24).map(l =>
        `<a class="mchip" href="${l.href}">${escapeHtml(l.name)}</a>`).join("")}</div>`
    : "");

  // Model links for the models in today's feed.
  //
  // Every link on this page used to point at /car?olx_id=… — pages that are
  // noindex by design because the listing disappears when the car sells. So the
  // site's most-updated page passed nothing onward: a crawler saw a wall of
  // dead-end links and a banner. These chips point at the stable /preco pages
  // for the models actually on offer right now, which is both a real crawl path
  // and the next question a visitor has ("is this a good price for a Golf?").
  const modelChips = (modelLinks && modelLinks.length) ? `
    <section class="section" style="padding:34px 0 0;">
      <div class="sec-label">PREÇO DE MERCADO DOS MODELOS COM NEGÓCIOS AGORA</div>
      <div class="mchips">${modelLinks.slice(0, 24).map(m =>
        `<a class="mchip" href="/preco/${encodeURIComponent(m.slug)}">${escapeHtml(m.b)} ${escapeHtml(m.m)} <span class="mut">mediana ${fmtEur(m.fm)}${m.count > 1 ? ` · ${m.count} negócios` : ""}</span></a>`).join("")}</div>
      ${chipRow("O ANO EXATO DESTES CARROS", yearLinks)}
      ${chipRow("QUANTO PERDEM E QUANTO DEMORAM A VENDER", contextLinks)}
      ${chipRow("O MERCADO DOS DISTRITOS COM NEGÓCIOS AGORA", districtLinks)}
      <p style="font-size:13.5px;color:#5B606B;margin:16px 0 0;">
        <a href="/precos" style="color:#177A47;font-weight:600;">Todos os modelos</a> ·
        <a href="/mercado/indice" style="color:#177A47;font-weight:600;">Índice semanal do mercado</a> ·
        <a href="/sobrevalorizados" style="color:#177A47;font-weight:600;">Onde se pede acima do valor justo</a>
      </p>
      ${(feedBuiltAt || builtAt) ? `<p class="mono fc-prov" data-updated="${escapeHtml(String(feedBuiltAt || builtAt).slice(0, 10))}" data-measure="live-deal-feed" data-source="OLX Portugal">Anúncios ativos recolhidos até ${escapeHtml(String(feedBuiltAt || builtAt).slice(0, 10))}${builtAt && feedBuiltAt && String(builtAt).slice(0, 10) !== String(feedBuiltAt).slice(0, 10) ? `, com os preços de referência de ${escapeHtml(String(builtAt).slice(0, 10))}` : ""} · fonte: OLX Portugal</p>` : ""}
    </section>` : "";

  const n = deals.length;
  const head = lens === "comprar"
    ? { h1: "Carros abaixo do preço", sub: `Portugal — ${ZONE_LABEL[zone] || zone} · ${n} ${n === 1 ? "carro" : "carros"} a valer mais do que custam` }
    : { h1: "Carros usados com margem de revenda", sub: `Portugal — ${ZONE_LABEL[zone] || zone} · ${n} ${n === 1 ? "carro" : "carros"} com margem de revenda` };

  const body = `
    <div class="feed">
      <div class="feed-head">
        <h1>${head.h1}</h1>
        <p>${head.sub}</p>
      </div>
      <div class="feed-tools">
        <div class="toolbar">
          <div class="chips">${lensChip("comprar", "🛒 Comprar")}${lensChip("revender", "📈 Revender")}</div>
          <div class="chips">${["score", "profit", "newest"].map(sortChip).join("")}</div>
        </div>
        <div class="toolbar">
          <div class="chips">${["all", "norte", "centro", "sul"].map(zoneChip).join("")}</div>
        </div>
      </div>
      <div class="grid">${tiles}</div>
      ${modelChips}
    </div>
    <button type="button" class="to-top" aria-label="Voltar ao topo">↑</button>`;
  const origin = host ? `https://${host}` : "";
  const title = lens === "comprar"
    ? "Carros usados abaixo do preço em Portugal (OLX)"
    : "Carros usados com margem de revenda (OLX Portugal)";
  const description = lens === "comprar"
    ? "Carros usados no OLX Portugal abaixo do preço justo de mercado, com desconto, lucro estimado e nota de risco de A+ a C. Avaliação independente, atualizada ao longo do dia."
    : "Carros usados no OLX Portugal com margem de revenda, comparados com o preço justo de mercado e nota de risco. Avaliação independente, atualizada ao longo do dia.";
  return layout({
    title, description, body, zone, nav: "feed", depositCount, index: true,
    // Collapse the zone×sort×view variants onto one indexable URL — the feed is
    // transient and not the SEO target, so folding the params avoids ~24 dupes.
    host, canonical: origin ? `${origin}/mercado` : null, jsonLd,
  });
}

// ── Car detail (/car) ───────────────────────────────────────────────────────────
export function renderCarPage({ deal, zone, view, unlocked, justReserved, depositEur, stripeReady, claimedAtMs, depositCount, modelHref, host }) {
  const p = present(deal);
  const lens = view === "revender" ? "revender" : "comprar";
  const photos = p.photos;
  const gallery = photos.length > 0
    ? `<div class="gallery ${photos.length === 1 ? "single" : ""}" data-count="${photos.length}">
        <div class="gallery-track">${photos.map((u, i) => `<img ${i === 0 ? `fetchpriority="high"` : `loading="lazy"`} src="${escapeHtml(u)}" alt="${escapeHtml(p.name)} — foto ${i + 1}">`).join("")}</div>
        <button type="button" class="gallery-nav prev" aria-label="Anterior">‹</button>
        <button type="button" class="gallery-nav next" aria-label="Próxima">›</button>
        <div class="gallery-counter">1 / ${photos.length} · FOTO DO ANÚNCIO</div>
        <div class="zoom-hint">⤢ Clica para ampliar</div>
      </div>
      <div class="lightbox" role="dialog" aria-modal="true" aria-label="Fotos do anúncio" aria-hidden="true">
        <button type="button" class="lb-close" aria-label="Fechar">×</button>
        <button type="button" class="lb-nav lb-prev" aria-label="Anterior">‹</button>
        <img class="lb-img" alt="${escapeHtml(p.name)}">
        <button type="button" class="lb-nav lb-next" aria-label="Próxima">›</button>
        <div class="lb-counter"></div>
      </div>`
    : `<div class="hero-photo striped"><span class="striped-label" style="font-size:54px;">${escapeHtml(p.make)}</span></div>`;

  // Thumbnail strip — up to 5 real photos (only when there's a real gallery).
  const thumbStrip = photos.length > 1
    ? `<div class="thumbs">${photos.slice(0, 5).map(u => `<div class="thumb-cell"><img loading="lazy" src="${escapeHtml(u)}" alt=""></div>`).join("")}</div>`
    : "";

  const sigClass = (warn, bad) => bad ? "v bad" : warn ? "v warn" : "v";
  const signals = [
    { k: "Preço pedido", v: p.priceStr },
    { k: "Justo (mediana)", v: p.fairStr },
    { k: "Desconto", v: fmtPct(p.disc) },
    { k: "Lucro estimado", v: p.profitStr },
    { k: "Severidade de dano", v: `${deal.damage_severity ?? 0} / 3`, cls: sigClass(deal.damage_severity >= 1, deal.damage_severity >= 2) },
    { k: "Dano em fotos", v: fmtPct1(deal.photo_damage_p), cls: deal.photo_damage_flagged ? "v bad" : "v" },
    { k: "Origem", v: p.importFlag ? (p.importLegalized ? "Importado (legalizado)" : "Importado · ISV em falta") : "Sem indício de importação", cls: (p.importFlag && !p.importLegalized) ? "v warn" : "v" },
    { k: "Quilometragem", v: fmtKm(deal.mileage_km), cls: sigClass(p.highKm, p.veryHighKm) },
    { k: "Vendedor", v: p.sellerType },
    { k: "Dias no mercado", v: deal.days_on_market != null ? String(deal.days_on_market) : "—" },
    ...(p.sellDays != null ? [{ k: "Tempo de venda", v: `~${p.sellDays}d` }] : []),
  ];
  const sellCaption = p.sellDays != null
    ? `<div style="font-size:12px;color:#8A8F98;margin-top:10px;line-height:1.5;">Carros deste modelo vendem, em mediana, em <b style="color:#16181D;">~${p.sellDays} dias</b> no mercado (${p.sellN} vendas analisadas). Boa referência se o pensas revender — ou para saberes a que ritmo este preço atrai compradores.</div>`
    : "";
  const kmCaption = p.veryHighKm
    ? `<div style="font-size:12px;color:#8A8F98;margin-top:10px;line-height:1.5;">Quilometragem alta para a idade — o preço justo já reflete a média do mercado, mas confirma histórico de manutenção e correia/distribuição na inspeção.</div>`
    : "";

  // Imported-car banner (amber) shown above the verdict when flagged. Two
  // variants; the not-legalized one carries the hedged, price-bucketed ISV range.
  const amber = GRADE_COLORS.amber;
  const importBanner = p.importFlag ? `
            <div class="exclusive" style="background:${amber.bg};border:1px solid ${amber.br};align-items:flex-start;margin-top:16px;">
              <span style="font-size:15px;">${p.importLegalized ? "🌍" : "⚠️"}</span>
              <span class="x" style="color:#6B4E12;">
                <b style="color:${amber.fg};">${p.importLegalized ? "Carro importado — já legalizado" : "Carro importado — ainda por legalizar"}.</b>
                ${p.importLegalized
                  ? "Este carro foi importado mas o anúncio indica matrícula/legalização portuguesa concluída. O preço já deve incluir o ISV. Confirma a documentação na inspeção."
                  : (p.isvEur
                      ? `O preço justo (${p.fairStr}) é de um carro já registado em Portugal. Como este parece ter matrícula estrangeira / por nacionalizar, falta somar o ISV — <b>estimado em ~${fmtEur(p.isvEur)}</b>${deal.co2_g_km ? ` (com base em ${deal.co2_g_km} g/km CO₂ + idade)` : ""}. Estimativa indicativa — confirma na tabela das Finanças.`
                      : `O preço justo (${p.fairStr}) é de um carro já registado em Portugal. Este anúncio parece ter matrícula estrangeira ou estar por nacionalizar, por isso é mais barato: o ISV e a legalização ainda não estão pagos. Conta com vários milhares de euros adicionais.<br><span style="display:block;margin-top:6px;">${p.isvRange || ""}</span>`)}
              </span>
            </div>` : "";

  // Verdict row (uses the real BUY/WATCH verdict; falls back to grade-driven).
  const verdictBuy = (deal.verdict || "").toUpperCase() === "BUY" || (!deal.verdict && (p.grade === "A+" || p.grade === "A"));
  const verdictTag = deal.verdict
    ? (verdictBuy ? "🟢 COMPRAR" : "🟡 OBSERVAR")
    : "🟢 COMPRAR";
  // Buyer lens leads with "how much under fair" (poupas); reseller lens keeps the
  // resale margin but asterisks it when an unpriced import cost is in play.
  const verdictProfit = lens === "comprar"
    ? (p.saving != null ? "poupas " + fmtEur(p.saving) : p.profitStr)
    : `${p.profitStr}${p.importFlag ? "*" : ""}`;
  const verdictFootnote = netIsvNote(p, lens === "comprar" ? p.saving : p.profit);

  // Claim / claimed module.
  let module;
  if (unlocked) {
    const cdAttr = claimedAtMs ? ` data-claimed-at="${claimedAtMs}"` : "";
    module = `
      <div class="claimed-mod">
        <div class="claimed-head">
          <div class="claimed-check">✓</div>
          <div style="flex:1;">
            <div class="t">Negócio reivindicado</div>
            <div class="d">Exclusivo para ti durante <span class="mono fc-countdown" style="font-weight:700;"${cdAttr}>24:00:00</span></div>
          </div>
        </div>
        <div class="claimed-body">
          <div class="contact-label">CONTACTO DO VENDEDOR</div>
          <div class="contact-card">
            <div class="avatar">${escapeHtml(p.sellerInitial)}</div>
            <div><div class="nm">${escapeHtml(p.sellerType)}</div><div class="meta">Anúncio OLX desbloqueado</div></div>
          </div>
          <div class="olx-row">
            <a class="olx-btn" href="${escapeHtml(deal.url || "#")}" target="_blank" rel="noopener">Abrir anúncio no OLX&nbsp;&nbsp;↗</a>
          </div>
          <div class="claimed-note"><span class="fc-dot"></span><span>${fmtEur(depositEur)} em depósito · reembolsado ao contactar o vendedor</span></div>
        </div>
      </div>`;
  } else if (stripeReady) {
    const unlockItems = [
      { t: "Link direto ao anúncio OLX", d: "sem intermediários" },
      { t: "Contacto do vendedor", d: "nome e telefone" },
      { t: "Galeria completa + verificação de matrícula", d: "todas as fotos e matrícula" },
      { t: "24h de exclusividade", d: "escondido dos outros membros" },
      ...(p.importFlag ? [{ t: "Aviso de importação incluído", d: "dizemos-te se falta pagar ISV antes de contactares o vendedor" }] : []),
    ];
    module = `
      <div class="claim-mod">
        <div class="claim-mod-head">
          <span style="font-size:18px;">🔒</span>
          <div><div class="t">Reivindicar este negócio</div><div class="d">Depósito de ${fmtEur(depositEur)} · 100% reembolsável</div></div>
        </div>
        <div class="claim-mod-body">
          ${unlockItems.map(u => `<div class="unlock-item"><span class="tick">✓</span><div><span class="t">${u.t}</span><span class="d"> — ${u.d}</span></div></div>`).join("")}
          <div class="exclusive"><span style="font-size:15px;">⏳</span><span class="x"><b>24h exclusivo.</b> Escondemos este negócio dos outros membros enquanto decides.</span></div>
          <a class="btn-green" href="/claim?zone=${zone}&olx_id=${encodeURIComponent(deal.olx_id)}">Desbloquear contacto do vendedor — ${fmtEur(depositEur)}</a>
          <div class="claim-fine"><b style="color:#3A3F47;">Pagas ${fmtEur(depositEur)} para falar direto com o vendedor — não é sinal do carro.</b> Reembolsado assim que o contactares (ou devolução automática em 48h).</div>
        </div>
      </div>`;
  } else {
    module = `
      <div class="claim-mod">
        <div class="claim-mod-head">
          <span style="font-size:18px;">🔒</span>
          <div><div class="t">Reivindicar este negócio</div><div class="d">As reservas por depósito estarão disponíveis em breve.</div></div>
        </div>
        <div class="claim-mod-body"><div class="btn-disabled">Reservas em breve</div></div>
      </div>`;
  }

  const locBits = [p.loc, p.firstSeenDays != null ? `há ${p.firstSeenDays}d` : null, p.sellerType].filter(Boolean).join(" · ");

  const body = `
    <div class="detail">
      <a class="back" href="/mercado?zone=${escapeHtml(zone)}&view=${lens}">‹&nbsp;&nbsp;Voltar ao mercado</a>
      <div class="detail-grid">
        <div class="dg-gallery">
          ${gallery}
          ${thumbStrip}
        </div>

        <div class="dg-side">
          <div class="side-card">
            <div class="side-head">
              <h1>${escapeHtml(p.name)}</h1>
              ${gradeBadge(p, "grade-badge")}
            </div>
            ${p.gradeDisplay !== p.grade ? `<div style="font-size:11.5px;color:${amber.fg};margin-top:8px;">† nota limitada: custo de importação não incluído</div>` : ""}
            <div class="side-sub">${p.subHtml}</div>
            <div class="side-loc">📍 ${escapeHtml(locBits)}</div>

            <div class="side-prices">
              <div><div class="cap">Preço pedido</div><div class="big">${p.priceStr}</div></div>
              <div class="side-fair"><div class="cap">Justo (mediana)${deal.sample_size != null ? ` · ${deal.sample_size} comp.` : ""}</div><div class="v">${p.fairStr}</div></div>
            </div>

            ${importBanner}

            <div class="verdict-row">
              <span class="verdict-tag">${verdictTag}</span>
              <span class="verdict-disc">${p.discStr}</span>
              <span class="verdict-profit">${verdictProfit}</span>
            </div>
            ${verdictFootnote}

            <div style="margin-top:16px;">
              <div class="gauge-head"><span>${p.fairLowStr}</span><span>intervalo justo de mercado</span><span>${p.fairHighStr}</span></div>
              <div class="gauge-track"><div class="gauge-pin" style="left:${p.gaugePos}%;"></div></div>
            </div>

            ${module}
          </div>
          <div class="side-foot">${deal.sample_size != null ? `Mediana de ${deal.sample_size} anúncios comparáveis (mesmo modelo) em Portugal` : `Avaliação a partir de anúncios comparáveis em Portugal`} · avaliação independente, não somos o vendedor.</div>
          ${modelHref ? `<div style="text-align:center;margin-top:10px;"><a href="${modelHref}" style="font-size:13px;color:#177A47;font-weight:600;">Ver preços de ${escapeHtml(p.make)} ${escapeHtml(deal.model || "")} por ano&nbsp;→</a></div>` : ""}
        </div>

        <div class="dg-extra">
          <div class="panel">
            <div class="panel-title">Sinais de avaliação</div>
            <div class="signals">
              ${signals.map(g => `<div class="signal"><div class="k">${g.k}</div><div class="${g.cls || "v"}">${g.v}</div></div>`).join("")}
            </div>
            ${kmCaption}
            ${sellCaption}
          </div>
          <div class="panel">
            <div class="panel-title">Descrição do vendedor</div>
            <p class="desc">${escapeHtml(deal.description ?? deal.description_excerpt ?? "Sem descrição disponível.")}</p>
          </div>
        </div>
      </div>
    </div>`;

  // noindex,follow: /car backs transient 5-min-rotating listings that vanish
  // (soft-404 risk) and republish the seller's OLX description verbatim
  // (duplicate content). Ranking consolidates on the stable /preco pages; the
  // page stays crawlable so link equity still flows to /preco and /mercado.
  // Share image is the car's own cover photo when present (absolute OLX URL).
  const ogImg = (typeof p.cover === "string" && /^https?:\/\//.test(p.cover)) ? p.cover : null;
  const desc = `${p.name}: pedido ${p.priceStr}, preço justo ${p.fairStr}${p.saving != null ? `, poupas ${fmtEur(p.saving)}` : ""}. Avaliação independente do anúncio no OLX Portugal.`;
  return layout({
    title: p.name, description: desc, body, zone, nav: "feed", depositCount,
    index: false, host, image: ogImg, type: ogImg ? "product" : "website",
    // Normalised: zone/view are display state, not part of the car's identity.
    ogUrl: host ? `https://${host}/car?olx_id=${encodeURIComponent(deal.olx_id)}` : null,
  });
}

// ── Claim confirm (/claim) ────────────────────────────────────────────────────
export function renderClaim({ deal, zone, depositEur, stripeReady, depositCount }) {
  const p = present(deal);
  const benefits = [
    { t: "Exclusividade de 24 horas", d: "Escondemos este carro de todos os outros utilizadores enquanto decides." },
    { t: "Contacto e link desbloqueados", d: "Nome, telefone e link direto ao anúncio OLX, mais a galeria completa." },
    ...(p.importFlag ? [{ t: "Aviso de importação", d: "Este anúncio parece ser um carro importado — dizemos-te se falta pagar ISV antes de avançares." }] : []),
    { t: "Totalmente reembolsável", d: "O depósito volta para a tua carteira ao contactar o vendedor, ou auto-devolução em 48h." },
  ];
  const cta = stripeReady
    ? `<button type="submit" class="btn-green">Desbloquear contacto — ${fmtEur(depositEur)}</button>`
    : `<div class="btn-disabled">Reservas em breve</div>`;

  const body = `
    <div class="claim-page">
      <div class="claim-card">
        <div class="claim-card-head">
          <div class="eb">DEPÓSITO REEMBOLSÁVEL</div>
          <div class="t">Reivindicar ${escapeHtml(p.name)}</div>
          <div class="claim-summary">
            <span class="pair"><span class="mono">${p.priceStr}</span><span class="cap">pedido</span></span>
            <span class="pair"><span class="prof">${p.profitStr}</span><span class="cap">margem</span></span>
          </div>
        </div>
        <form class="claim-card-body" action="/reserve" method="post">
          <input type="hidden" name="olx_id" value="${escapeHtml(deal.olx_id)}">
          <input type="hidden" name="zone" value="${escapeHtml(zone)}">
          <!-- Идентификатор клиента GA4. Существует только у согласившегося на
               аналитику: без согласия куки _ga нет, поле уходит пустым, и
               серверное событие не отправляется вовсе. -->
          <input type="hidden" name="ga_cid" value="">
          <div class="dep-row"><span class="l">${fmtEur(depositEur)} para falar com o vendedor</span><span class="r">${fmtEur(depositEur)}</span></div>
          <div style="font-size:12px;color:#8A8F98;margin-top:4px;">Não é sinal do carro — é o que pagas para desbloquear o contacto. Reembolsável.</div>
          <div class="hr"></div>
          ${benefits.map(b => `<div class="benefit"><span class="tick">✓</span><div><div class="t">${b.t}</div><div class="d">${b.d}</div></div></div>`).join("")}
          ${cta}
          <a class="cancel-btn" href="/car?zone=${zone}&olx_id=${encodeURIComponent(deal.olx_id)}">Cancelar</a>
          <div class="secure"><span>🔒</span><span>Pagamento seguro · depósito devolvido automaticamente em 48h</span></div>
        </form>
      </div>
    </div>`;
  return layout({
    title: `Reivindicar ${p.name}`,
    body: body + gaClientIdFiller() + analyticsFormEvent('form[action="/reserve"]', "begin_checkout", {
      currency: "EUR",
      value: depositEur,
      items: [{ item_id: deal.olx_id, item_name: p.name }],
    }),
    zone, nav: "feed", depositCount,
  });
}

// ── Unlocked success (/unlocked) ──────────────────────────────────────────────
export function renderClaimSuccess({ deal, zone, depositEur, claimedAtMs, depositCount, txnId }) {
  const p = present(deal);
  const cdAttr = claimedAtMs ? ` data-claimed-at="${claimedAtMs}"` : "";
  const nextSteps = [
    { n: "1", t: "Liga ao vendedor e confirma estado, documentos e margem de negociação." },
    { n: "2", t: "Agenda uma inspeção presencial antes de qualquer pagamento." },
    { n: "3", t: `Fecha o negócio — o teu depósito de ${fmtEur(depositEur)} é devolvido automaticamente.` },
  ];
  const body = `
    <div class="success">
      <div class="success-top">
        <div class="success-check">✓</div>
        <h1>Negócio reivindicado</h1>
        <p>É todo teu. Mais nenhum membro vê este carro durante as próximas 24h.</p>
      </div>
      <div class="cd-banner">
        <div style="flex:1;">
          <div class="cap">Exclusividade termina em</div>
          <div class="big mono fc-countdown"${cdAttr}>24:00:00</div>
        </div>
        <div class="dep">
          <div class="cap">Depósito</div>
          <div class="v">${fmtEur(depositEur)}</div>
          <div class="s">reembolsável</div>
        </div>
      </div>
      <div class="panel" style="margin-top:0;">
        <div class="contact-label">CONTACTO DESBLOQUEADO</div>
        <div style="display:flex;align-items:center;gap:12px;">
          <div class="avatar" style="width:42px;height:42px;font-size:16px;">${escapeHtml(p.sellerInitial)}</div>
          <div style="flex:1;"><div class="nm" style="font-weight:600;font-size:15px;">${escapeHtml(p.sellerType)}</div><div class="meta">Anúncio OLX desbloqueado</div></div>
        </div>
        <a class="olx-btn" style="margin-top:16px;padding:14px;border-radius:12px;font-size:15px;" href="${escapeHtml(deal.url || "#")}" target="_blank" rel="noopener">Abrir anúncio no OLX&nbsp;&nbsp;↗</a>
        <div class="hr"></div>
        <div class="contact-label">PRÓXIMOS PASSOS</div>
        <div class="next-steps">
          ${nextSteps.map(n => `<div class="step"><span class="n">${n.n}</span><span class="tx">${n.t}</span></div>`).join("")}
        </div>
      </div>
      <a class="btn-outline" href="/reservas">Ver as minhas reservas</a>
    </div>`;
  // purchase на странице после оплаты. transaction_id приходит из обработчика и
  // склеен из объявления и id сессии Stripe: он стабилен при обновлении
  // страницы и совпадает с тем, что шлёт вебхук, поэтому одна оплата не
  // посчитается дважды. Клиентское событие теряется, если человек закрыл
  // вкладку на редиректе Stripe - для этого и нужен серверный дубль.
  const purchase = analyticsEvent("purchase", {
    transaction_id: txnId || `${deal.olx_id}-${claimedAtMs || 0}`,
    currency: "EUR",
    value: depositEur,
    items: [{ item_id: deal.olx_id, item_name: p.name }],
  });
  return layout({ title: "Negócio reivindicado", body: body + purchase, zone, nav: "reservas", depositCount });
}

// ── Reservas (/reservas) ────────────────────────────────────────────────────────
export function renderReservations({ claims, depositEur, depositCount }) {
  // claims: [{ deal, claimedAtMs }]
  const hasClaims = claims && claims.length > 0;
  let inner;
  if (hasClaims) {
    inner = `<div class="res-list">${claims.map(({ deal, claimedAtMs }) => {
      const p = present(deal);
      const cdAttr = claimedAtMs ? ` data-claimed-at="${claimedAtMs}"` : "";
      const thumb = p.cover
        ? `<div class="res-thumb"><img loading="lazy" src="${escapeHtml(p.cover)}" alt="${escapeHtml(p.name)}"></div>`
        : `<div class="res-thumb striped"><span class="striped-label" style="font-size:15px;">${escapeHtml(p.make)}</span></div>`;
      return `<div class="res-card">
        ${thumb}
        <div class="res-mid">
          <div class="t">${escapeHtml(p.name)}</div>
          <div class="s">${escapeHtml(p.sub)}</div>
          ${importTag(p)}
          <div class="res-prices"><span class="p">${p.priceStr}</span><span class="pr">${p.profitStr}${p.importFlag ? "*" : ""}</span></div>
          ${p.importFlag ? `<div style="font-size:11px;color:${GRADE_COLORS.amber.fg};margin-top:5px;">* margem antes do ISV${p.isvEur ? ` (~${fmtEur(p.isvEur)})` : ""}</div>` : ""}
        </div>
        <div class="res-right">
          <div class="cd-pill"><span class="fc-dot"></span><span class="mono fc-countdown"${cdAttr}>24:00:00</span></div>
          <div class="ex">exclusivo · ${fmtEur(depositEur)} em depósito</div>
          <a class="btn-dark" href="/car?olx_id=${encodeURIComponent(deal.olx_id)}">Ver contacto</a>
        </div>
      </div>`;
    }).join("")}</div>`;
  } else {
    inner = `<div class="empty-card">
      <div class="ic">🗝️</div>
      <div class="t">Ainda não reivindicaste nenhum negócio</div>
      <p>Reivindica um carro no mercado para o esconder dos outros membros e desbloquear o contacto do vendedor.</p>
      <a class="btn-green" href="/mercado">Explorar o mercado</a>
    </div>`;
  }
  const body = `
    <div class="res">
      <h1>As minhas reservas</h1>
      <p class="lead">€${(depositCount || 0) * 5} em depósito retido · reembolsado ao contactar cada vendedor</p>
      ${inner}
    </div>`;
  return layout({ title: "As minhas reservas", body, zone: "all", nav: "reservas", depositCount });
}

// ── Generic info / degraded page ──────────────────────────────────────────────
// Страница приватности. Содержание сверено с кодом: fc_uid это случайный
// токен без персональных данных, в KV лежит только «этот псевдоним разблокировал
// это объявление», email и имя сайт не собирает, платёж целиком у Stripe.
// Если поведение изменится, эту страницу надо править вместе с кодом.
export function renderPrivacy({ depositCount, host, contact = null }) {
  const origin = host ? `https://${host}` : "";
  const to = (contact || "").trim();
  const contactHtml = to
    ? `<a href="mailto:${escapeHtml(to)}">${escapeHtml(to)}</a>`
    : `o contacto indicado em <a href="/sobre">Quem somos</a>`;
  const body = `
    <section class="fc-doc">
      <h1>Privacidade</h1>
      <p>Esta página descreve exatamente o que o Carsbuyer guarda. Está escrita a
      partir do código do site, não de um modelo genérico.</p>

      <h2>O que guardamos sempre</h2>
      <p>Um cookie técnico <code>fc_uid</code> com um número aleatório. Serve para saber
      que reservas são tuas quando voltas ao site. Não contém nome, email nem
      qualquer dado que te identifique, e sem ele a reserva não funciona — por isso
      não depende de consentimento.</p>
      <p>Ligado a esse número aleatório guardamos apenas que anúncios desbloqueaste.
      Nada mais.</p>

      <h2>O que só guardamos com o teu consentimento</h2>
      <p>Estatísticas anónimas de utilização através do Google Analytics: que páginas
      são vistas e por que caminho. Até aceitares, o Google Analytics corre sem
      cookies e sem identificadores — recolhe apenas contagens agregadas. Podes
      recusar no aviso em baixo da página, e recusar é tão simples como aceitar.</p>

      <h2>Pagamentos</h2>
      <p>O depósito é processado pela Stripe na página da própria Stripe. Os dados do
      cartão nunca passam por este site nem ficam guardados aqui. Recebemos da Stripe
      apenas a confirmação de que o pagamento foi feito.</p>

      <h2>Anúncios de carros</h2>
      <p>Os anúncios mostrados são públicos e recolhidos de portais como o OLX e o
      StandVirtual. Não pertencem ao Carsbuyer e não contêm dados teus.</p>

      <h2>Pedidos de propostas de compra</h2>
      <p>Se pedires propostas para o teu carro, guardamos o que escreves no formulário:
      modelo, ano, quilómetros, distrito, o teu contacto e, se o deres, o nome. Servem
      para um fim só: enviá-los a compradores profissionais — stands e serviços de compra
      de carros — que te apresentam uma proposta. O pedido fica guardado 90 dias e é
      apagado depois. Não o usamos para publicidade nem o cedemos a mais ninguém. Podes
      pedir a eliminação antes do prazo pelo contacto em baixo.</p>

      <h2>Ligações de parceiros</h2>
      <p>Nas páginas de preços há ligações para um serviço de relatórios de histórico de
      veículos. Se comprares um relatório por essa ligação, o Carsbuyer recebe uma
      comissão; o preço para ti é o mesmo, e não enviamos ao parceiro nenhum dado teu —
      só a ligação que carregas.</p>

      <h2>Os teus direitos</h2>
      <p>Podes apagar o cookie <code>fc_uid</code> no teu navegador a qualquer momento —
      perdes o acesso às reservas antigas, e mais nada. Para pedir a eliminação do que
      está associado ao teu identificador ou de um pedido de propostas, escreve para
      ${contactHtml}.</p>
    </section>`;
  return layout({
    title: "Privacidade",
    description: "O que o Carsbuyer guarda: cookie técnico de reserva, estatísticas anónimas só com consentimento, pedidos de propostas guardados 90 dias, pagamentos processados pela Stripe.",
    body, zone: "all", nav: null, depositCount, index: true,
    host, canonical: origin ? `${origin}/privacidade` : null,
  });
}

export function renderInfo({ zone, title, message, depositCount }) {
  const body = `
    <div class="info">
      <div class="ic">🚗</div>
      <h1>${escapeHtml(title)}</h1>
      <p>${escapeHtml(message)}</p>
      <a class="btn-dark" href="/mercado">Ver mercado</a>
    </div>`;
  return layout({ title, body, zone, nav: null, depositCount });
}

// ── Avaliar (/avaliar) — seller-lens teaser ──────────────────────────────────
// Tier-0: no real "value MY car" tool yet (needs an inference endpoint). Ships a
// waitlist teaser whose CTA is a mailto. A real form is Tier-1.
// ── Avaliar (/avaliar) — paste-a-link valuation of ANY OLX listing (Tier-2) ──
// rec = the valuations.json record for the looked-up olx_id (or null). query =
// the raw user input (URL or id). The verdict is derived from where the asking
// price sits in the model's fair band [fl, fh].
export function renderAvaliar({ rec, olxId, sourceUrl, query, models, spec, depositCount, host, builtAt, contact, historyUrl = null }) {
  const to = (contact || "").trim();
  const mailto = to
    ? `mailto:${encodeURIComponent(to)}?subject=Avaliar%20o%20meu%20carro`
      + "&body=Marca%2Fmodelo%3A%0AAno%3A%0AQuilometragem%3A%0ACombust%C3%ADvel%3A%0ALink%20do%20an%C3%BAncio%20(se%20tiver)%3A"
    : "";

  const g = GRADE_COLORS.green, amber = GRADE_COLORS.amber, rd = GRADE_COLORS.red;

  let result = "";
  if (rec) {
    const price = rec.p, fl = rec.fl, fm = rec.fm, fh = rec.fh;
    const saving = (fm != null && price != null) ? fm - price : null;
    let gaugePos = 50;
    if (price != null && fl != null && fh != null && fh > fl)
      gaugePos = Math.max(3, Math.min(96, Math.round((price - fl) / (fh - fl) * 100)));
    let tag, vc, line;
    if (price != null && fh != null && price > fh) {
      tag = "🔴 ACIMA DO MERCADO"; vc = rd;
      line = saving != null ? `pagas ${fmtEur(-saving)} acima do justo` : "";
    } else if (price != null && fl != null && price < fl) {
      tag = "🟢 ABAIXO DO MERCADO"; vc = g;
      line = saving != null ? `poupas ${fmtEur(saving)} vs o justo` : "";
    } else {
      tag = "🟢 PREÇO JUSTO"; vc = g;
      line = (saving != null && saving > 0) ? `poupas ${fmtEur(saving)} vs a mediana` : "dentro do intervalo de mercado";
    }
    const isvR = ISV_RANGE[isvTier(price)] || "";
    const importBanner = rec.imp ? `
        <div class="exclusive" style="background:${amber.bg};border:1px solid ${amber.br};align-items:flex-start;margin-top:16px;">
          <span style="font-size:15px;">${rec.il ? "🌍" : "⚠️"}</span>
          <span class="x" style="color:#6B4E12;">
            <b style="color:${amber.fg};">${rec.il ? "Carro importado — já legalizado" : "Indícios de importação — possivelmente por legalizar"}.</b>
            ${rec.il
              ? " O preço já deve incluir o ISV; confirma a documentação na inspeção."
              : ` O preço justo acima é de um carro já registado em Portugal. Se a matrícula ainda for estrangeira, falta somar o ISV.<br><span style="display:block;margin-top:6px;">${isvR}</span>`}
          </span>
        </div>` : "";
    const sellLine = rec.sd != null ? `<div style="font-size:12px;color:#8A8F98;margin-top:14px;line-height:1.5;">Carros deste modelo vendem, em mediana, em <b style="color:#16181D;">~${rec.sd} dias</b> no mercado.</div>` : "";
    const track = Array.isArray(rec.ph) ? rec.ph : null;
    let cutBlock = "";
    if (track && track.length >= 2) {
      const first = track[0][1], last = track[track.length - 1][1];
      const cuts = track.length - 1;
      const drop = first > 0 ? Math.round((1 - last / first) * 100) : 0;
      const lastDays = track[track.length - 1][0];
      if (drop > 0) {
        cutBlock = `
        <div class="exclusive" style="background:${g.bg};border:1px solid ${g.br};align-items:flex-start;margin-top:16px;">
          <span style="font-size:15px;">📉</span>
          <span class="x" style="color:#1B5E3A;">
            <b style="color:${g.fg};">O vendedor já baixou o preço ${cuts === 1 ? "uma vez" : `${cuts} vezes`}.</b>
            De ${fmtEur(first)} para ${fmtEur(last)} (−${drop}%)${lastDays === 0 ? ", a última descida foi hoje" : `, a última há ${lastDays} ${lastDays === 1 ? "dia" : "dias"}`}.
            <span style="display:block;margin-top:6px;">Há margem para negociar.</span>
          </span>
        </div>`;
      }
    }
    let stallBlock = "";
    if (rec.dom != null) {
      const stalled = rec.sd != null && rec.dom > rec.sd * 1.5;
      const c = stalled ? amber : null;
      stallBlock = stalled
        ? `<div style="font-size:12.5px;color:#6B4E12;background:${c.bg};border:1px solid ${c.br};border-radius:10px;padding:10px 12px;margin-top:12px;line-height:1.5;">
             Anúncio online há <b>${rec.dom} dias</b> — carros deste modelo costumam vender em ~${rec.sd}. Quem está há tanto tempo à espera tende a aceitar menos.
           </div>`
        : `<div style="font-size:12px;color:#8A8F98;margin-top:12px;">Anúncio online há <b style="color:#16181D;">${rec.dom} ${rec.dom === 1 ? "dia" : "dias"}</b>.</div>`;
    }
    const quote = v => typeof v === "string" && v.trim() ? `«${escapeHtml(v.trim())}»` : "";
    const faultBlock = rec.hb
      ? `<div class="exclusive" style="background:${rd.bg};border:1px solid ${rd.br};align-items:flex-start;margin-top:16px;">
           <span style="font-size:15px;">🛑</span>
           <span class="x" style="color:#7A1B1B;"><b style="color:${rd.fg};">O próprio anúncio diz ${quote(rec.hb)}.</b> O intervalo justo acima é de um carro em condições normais e não se aplica a esta viatura.</span>
         </div>`
      : (rec.mf
        ? `<div style="font-size:12.5px;color:#6B4E12;background:${amber.bg};border:1px solid ${amber.br};border-radius:10px;padding:10px 12px;margin-top:12px;line-height:1.5;">
             O texto do anúncio menciona ${quote(rec.mf)}. Desconta o custo da reparação antes de comparar com o intervalo justo.
           </div>`
        : "");
    // Use the pasted URL when present. Only reconstruct an OLX URL for OLX-style
    // ids (SV ids start "8P" and live on standvirtual.com — a reconstructed
    // olx.pt URL would 404), otherwise omit the button.
    const olxHref = sourceUrl || (olxId && !olxId.startsWith("8") ? `https://www.olx.pt/d/anuncio/-ID${encodeURIComponent(olxId)}.html` : null);
    // Contextual link into the model SEO page, when this model has one.
    const modelHref = (rec.ms && models && models[rec.ms]) ? `/preco/${encodeURIComponent(rec.ms)}` : null;
    const sub = `${rec.y ?? "—"} · ${rec.km != null ? fmtKm(rec.km) : "—"} · ${escapeHtml(rec.fu || "—")}`;
    const hist = historyCheckBlock({ url: historyUrl, reasons: historyReasons(rec, models), price });
    const sellHref = (rec.ms && models && models[rec.ms])
      ? `/avaliar?modelo=${encodeURIComponent(rec.ms)}${rec.y ? `&ano=${encodeURIComponent(rec.y)}` : ""}#vender`
      : "/avaliar#escolher";
    result = `
    <div class="detail" style="max-width:640px;margin:0 auto;padding-top:0;">
      <div class="side-card">
        <div class="side-head"><h1 style="font-size:21px;">${escapeHtml(rec.t || "Viatura")}</h1></div>
        <div class="side-sub">${sub}</div>
        ${rec.ct ? `<div class="side-loc">📍 ${escapeHtml(rec.ct)}</div>` : ""}
        <div class="side-prices">
          <div><div class="cap">Preço pedido</div><div class="big">${fmtEur(price)}</div></div>
          <div class="side-fair"><div class="cap">Justo (mediana)</div><div class="v">${fmtEur(fm)}</div></div>
        </div>
        <div class="verdict-row" style="background:${vc.bg};border-color:${vc.br};">
          <span class="verdict-tag" style="color:${vc.fg};">${tag}</span>
          <span class="verdict-profit" style="color:${vc.fg};">${line}</span>
        </div>
        ${netIsvNote({ importFlag: !!rec.imp, importLegalized: !!rec.il, isvEur: rec.isv_eur ?? null, isvTier: isvTier(price) }, saving)}
        <div style="margin-top:16px;">
          <div class="gauge-head"><span>${fmtEur(fl)}</span><span>intervalo justo de mercado</span><span>${fmtEur(fh)}</span></div>
          <div class="gauge-track"><div class="gauge-pin" style="left:${gaugePos}%;"></div></div>
        </div>
        ${importBanner}
        ${cutBlock}
        ${faultBlock}
        ${sellLine}
        ${stallBlock}
        ${hist}
        ${olxHref ? `<a class="olx-btn" style="display:block;margin-top:18px;" href="${escapeHtml(olxHref)}" target="_blank" rel="noopener nofollow">Ver anúncio original&nbsp;&nbsp;↗</a>` : ""}
        ${modelHref ? `<a href="${modelHref}" style="display:block;text-align:center;margin-top:12px;font-size:13.5px;color:#177A47;font-weight:600;">Ver preços deste modelo por ano&nbsp;→</a>` : ""}
        <a href="${sellHref}" style="display:block;text-align:center;margin-top:10px;font-size:13.5px;color:#5B606B;">É o teu carro? Recebe propostas de compra&nbsp;→</a>
        <div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:14px;">
          <a class="btn-outline" style="flex:1 1 auto;padding:11px 14px;font-size:13.5px;text-align:center;" href="/avaliar">Avaliar outro carro</a>
          <a class="btn-dark" style="flex:1 1 auto;padding:11px 14px;font-size:13.5px;text-align:center;" href="/mercado">Ver carros abaixo do preço&nbsp;→</a>
        </div>
      </div>
      <div class="side-foot">Estimativa a partir de anúncios comparáveis em Portugal · avaliação independente, não somos o vendedor.</div>
    </div>`;
  }

  // Spec-based estimate (seller without a listing yet): pick the matching year
  // cell from the model record, else fall back to the model-level median.
  let specResult = "";
  if (!rec && spec && spec.rec) {
    const mr = spec.rec, SB = escapeHtml(mr.b), SM = escapeHtml(mr.m);
    const cell = spec.cell;
    const sfm = cell ? cell.fm : mr.fm, sfl = cell ? cell.fl : mr.fl, sfh = cell ? cell.fh : mr.fh;
    const yLabel = cell ? escapeHtml(String(cell.y)) : "";
    let pin = 50; if (sfh > sfl) pin = Math.max(6, Math.min(94, Math.round((sfm - sfl) / (sfh - sfl) * 100)));
    const caveat = (!cell && spec.year)
      ? `<div style="font-size:12.5px;color:#9A6B12;margin-top:10px;">Sem amostra suficiente para ${spec.year} — mostramos a mediana do modelo (todos os anos).</div>`
      : (cell && cell.w
        ? `<div style="font-size:12.5px;color:#9A6B12;margin-top:10px;">Poucos anúncios ativos deste ano: mediana dos últimos ${Math.max(1, Math.round(cell.w / 30))} meses, ativos e já fechados.</div>`
        : "");
    const sl = mr.sd != null ? `<div style="font-size:12px;color:#8A8F98;margin-top:14px;">Vende, em mediana, em ~${mr.sd} dias no OLX.</div>` : "";
    specResult = `
    <div class="detail" style="max-width:640px;margin:0 auto;padding-top:0;">
      <div class="side-card">
        <div class="side-head"><h1 style="font-size:21px;">${SB} ${SM}${cell ? ` · ${yLabel}` : ""}</h1></div>
        <div class="side-prices">
          <div><div class="cap">Preço mediano (pedido)</div><div class="big">${fmtEur(sfm)}</div></div>
          <div class="side-fair"><div class="cap">intervalo típico</div><div class="v">${fmtEur(sfl)} – ${fmtEur(sfh)}</div></div>
        </div>
        <div style="margin-top:16px;">
          <div class="gauge-head"><span>${fmtEur(sfl)}</span><span>intervalo típico (P25–P75)</span><span>${fmtEur(sfh)}</span></div>
          <div class="gauge-track"><div class="gauge-pin" style="left:${pin}%;"></div></div>
        </div>
        ${caveat}${sl}
        <div style="margin-top:16px;display:flex;gap:10px;flex-wrap:wrap;">
          <a class="btn-outline" style="padding:11px 16px;font-size:14px;" href="/preco/${encodeURIComponent(spec.slug)}">Ver preço por ano&nbsp;→</a>
          <a class="btn-dark" style="padding:11px 16px;font-size:14px;" href="/avaliar">Tens o anúncio? Cola o link</a>
        </div>
        ${spec.vender ? `<a href="/vender/${encodeURIComponent(spec.slug)}" style="display:block;text-align:center;margin-top:12px;font-size:13.5px;color:#177A47;font-weight:600;">Quanto pedir e em quantos dias vende&nbsp;→</a>` : ""}
      </div>
      <div class="side-foot">Preços PEDIDOS em anúncios ativos do OLX — estimativa indicativa, não o valor da tua viatura concreta.</div>
      ${leadFormBlock({
        slug: spec.slug, name: `${mr.b} ${mr.m}`,
        year: spec.year || (cell && typeof cell.y === "number" ? cell.y : null),
        median: sfm,
      })}
    </div>`;
  }

  // Spec form (model select grouped by brand + year), built from the models map.
  let specForm = "";
  if (!rec && models) {
    const byBrand = {};
    for (const [slug, r] of Object.entries(models)) (byBrand[r.b] = byBrand[r.b] || []).push([slug, r.m]);
    const brands = Object.keys(byBrand).sort((a, b) => a.localeCompare(b, "pt"));
    const opts = brands.map(b => `<optgroup label="${escapeHtml(b)}">`
      + byBrand[b].sort((a, c) => a[1].localeCompare(c[1], "pt")).map(([slug, m]) =>
          `<option value="${escapeHtml(slug)}"${spec && spec.slug === slug ? " selected" : ""}>${escapeHtml(m)}</option>`).join("")
      + `</optgroup>`).join("");
    specForm = `
    <section id="escolher" class="section" style="padding:10px 22px 0;max-width:620px;">
      <div class="side-card">
        <div class="panel-title" style="font-size:16px;margin-bottom:12px;">Não tens anúncio? Escolhe o teu carro</div>
        <form action="/avaliar" method="get" style="display:flex;gap:10px;flex-wrap:wrap;">
          <select name="modelo" required style="flex:1 1 230px;min-width:180px;padding:12px;border:1px solid #E2DFD8;border-radius:11px;font-size:15px;background:#fff;color:#16181D;">
            <option value="">Modelo…</option>${opts}
          </select>
          <input type="number" name="ano" min="1990" max="2026" value="${spec && spec.year ? spec.year : ""}" placeholder="Ano"
            style="width:108px;padding:12px;border:1px solid #E2DFD8;border-radius:11px;font-size:15px;">
          <button type="submit" class="btn-dark" style="padding:12px 20px;font-size:15px;">Estimar&nbsp;→</button>
        </form>
        <div class="mono" style="font-size:11px;color:#9A9FA8;margin-top:10px;">Estimativa pela mediana de mercado do modelo/ano. Para o teu carro exato, cola o link do anúncio acima.</div>
      </div>
    </section>`;
  }

  const notice = (!rec && query) ? `
    <div class="info" style="padding:8px 22px 0;max-width:620px;">
      <p style="margin:0 auto;">Ainda não temos este anúncio na nossa base. Cobrimos a maioria dos carros ativos no OLX Portugal, mas nem todos — confirma o link, ou pede uma avaliação por email.</p>
    </div>` : "";

  const form = `
    <form action="/avaliar" method="get" style="display:flex;gap:10px;flex-wrap:wrap;justify-content:center;max-width:620px;margin:0 auto;">
      <input name="q" value="${escapeHtml(query || "")}" placeholder="Cola o link OLX ou StandVirtual (ou o ID)" autocomplete="off"
        style="flex:1 1 340px;min-width:220px;padding:13px 15px;border:1px solid #E2DFD8;border-radius:12px;font-family:'Hanken Grotesk',sans-serif;font-size:15px;background:#fff;color:#16181D;">
      <button type="submit" class="btn-dark" style="font-size:15px;padding:13px 24px;">Avaliar&nbsp;&nbsp;→</button>
    </form>`;

  const body = `
    <section class="hero" style="padding-bottom:18px;">
      <div class="hero-copy" style="max-width:660px;margin:0 auto;text-align:center;">
        <div class="eyebrow" style="margin:0 auto 22px;"><span class="e-dot"></span><span class="mono">AVALIAÇÃO INDEPENDENTE · OLX PORTUGAL</span></div>
        <h1 class="hero-title" style="font-size:40px;">Quanto vale o teu carro usado?</h1>
        <p class="lede" style="margin:0 auto 26px;">Cola o link de qualquer anúncio de carro do OLX ou StandVirtual e dizemos-te o preço justo de mercado, quanto estás a poupar (ou a pagar a mais) e se é importado com ISV por pagar. Grátis, sem registo.</p>
        ${form}
        <div class="mono" style="font-size:12px;color:#8A8F98;margin-top:14px;">Estimativa indicativa · independente · não somos stand nem intermediário${builtAt && fmtBuilt(builtAt) ? ` · atualizado a ${fmtBuilt(builtAt).replace(/ /g, "\u00a0")}` : ""}</div>
      </div>
    </section>
    ${notice}
    ${result}
    ${specResult}
    ${!rec ? specForm : ""}
    ${rec ? "" : `
    <section class="section" style="padding:18px 22px 70px;">
      <div class="cta-banner">
        <div style="flex:1 1 360px;">
          <h2>Vais vender o teu carro?</h2>
          <p>Escolhe o modelo e o ano acima: vês o que o mercado está a pedir e podes pedir propostas de compra a compradores profissionais, sem compromisso${mailto ? " — ou pede uma avaliação por email" : ""}.</p>
        </div>
        ${mailto ? `<a class="btn-bright" href="${mailto}">Pedir avaliação por email&nbsp;&nbsp;→</a>` : `<a class="btn-bright" href="#escolher">Escolher o meu carro&nbsp;&nbsp;→</a>`}
      </div>
    </section>`}`;
  const origin = host ? `https://${host}` : "";
  // Only the bare tool page is indexable; every ?q=/​?modelo=/?ano= variant is a
  // thin/transient per-listing view — noindex + canonical-to-bare consolidates
  // that unbounded (user-pasted-URL) param space onto the single sitemap URL.
  const isBare = !rec && !spec && !query;
  // Бесплатная оценка это то, ради чего человек приходит, и первый шаг воронки.
  // Событие шлём только когда оценка реально посчиталась, иначе оно означало бы
  // просто открытие формы.
  // Оценка показывается двумя путями: по ссылке на объявление (rec) и по
  // выбору модели с годом (spec). Событие должно быть на обоих - сначала оно
  // висело только на rec, а это более редкий путь.
  const shown = rec || (spec && spec.rec);
  // У объявления название лежит в t, у записи модели названия нет вовсе - только
  // марка и модель по отдельности. Взял t на обоих путях, и на пути модели
  // параметр уходил пустым.
  const valuation = shown ? analyticsEvent("valuation_result", {
    model: rec ? (rec.t || "")
                : (`${spec.rec.b || ""} ${spec.rec.m || ""}`.trim() || spec.slug || ""),
    source: rec ? "listing" : "model",
    has_listing: Boolean(olxId),
  }) : "";
  // Structured data for the site's most commercial query ("quanto vale o meu
  // carro"). This page had none at all — it is a free tool, so WebApplication
  // with a zero-price Offer is the accurate type, and the FAQ answers the three
  // objections people actually arrive with. Only on the bare page: the ?q=
  // variants are noindex, and schema on a noindex page is wasted markup.
  const jsonLd = (origin && isBare) ? {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "WebApplication",
        "name": "Avaliação de carros usados",
        "url": `${origin}/avaliar`,
        "applicationCategory": "FinanceApplication",
        "operatingSystem": "Web",
        "inLanguage": "pt-PT",
        "isAccessibleForFree": true,
        "offers": { "@type": "Offer", "price": "0", "priceCurrency": "EUR" },
        "provider": { "@type": "Organization", "name": "Carsbuyer", "url": `${origin}/` },
        "description": "Cola o link de um anúncio de carro usado do OLX Portugal e recebe o preço justo estimado, o desvio face ao mercado e o aviso de importação por legalizar.",
        "featureList": [
          "Preço justo estimado para o anúncio concreto",
          "Diferença face ao preço pedido",
          "Aviso de viatura importada com ISV por pagar",
          "Preço mediano do modelo e do ano",
        ],
      },
      {
        "@type": "BreadcrumbList",
        "itemListElement": [
          { "@type": "ListItem", "position": 1, "name": "Início", "item": `${origin}/` },
          { "@type": "ListItem", "position": 2, "name": "Avaliar o meu carro" },
        ],
      },
      {
        "@type": "FAQPage",
        "mainEntity": [
          ["Quanto vale o meu carro usado em Portugal?",
           "Cola o link do anúncio do teu carro no OLX e devolvemos o valor justo estimado para essa viatura concreta, com os seus quilómetros, ano e versão, além do preço mediano pedido pelo mesmo modelo no mercado. Se ainda não tens anúncio, escolhe o modelo e o ano para veres a mediana do mercado."],
          ["A avaliação é grátis?",
           "Sim. A avaliação de um anúncio e os preços por modelo são gratuitos e sem registo. Só se paga um depósito reembolsável de 5 € para desbloquear o contacto de um vendedor no mercado de negócios, e esse depósito é devolvido."],
          ["Esta avaliação serve para vender ao meu stand ou ao seguro?",
           "É uma estimativa independente a partir de anúncios reais e serve para saber por quanto anunciar ou quanto oferecer. Não é uma avaliação oficial para efeitos de seguro, sinistro ou fiscais, e não somos stand nem intermediário."],
          ["De onde vêm os valores?",
           "De anúncios ativos de carros no OLX Portugal, recolhidos diariamente. Trabalhamos com preços pedidos, com a mediana e o intervalo interquartil, e com um modelo estatístico para o valor justo. O método completo está publicado na página de metodologia."],
        ].map(([q, a]) => ({
          "@type": "Question", "name": q,
          "acceptedAnswer": { "@type": "Answer", "text": a },
        })),
      },
    ],
  } : null;
  return layout({
    title: rec
      ? `${rec.t}: preço justo e avaliação`
      : "Quanto vale o meu carro? Avaliação grátis de carros usados",
    description: "Cola o link de qualquer anúncio OLX ou StandVirtual e sabe o preço justo do carro, quanto poupas ou pagas a mais, e se tem ISV por pagar. Avaliação independente e grátis.",
    body: body + valuation, zone: "all", nav: "avaliar", depositCount, index: isBare, jsonLd,
    host, canonical: origin ? `${origin}/avaliar` : null,
  });
}

// Freshness: ISO build stamp → PT short date ("1 jul 2026"). "" when absent.
const PT_MON = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"];
export function fmtBuilt(iso) {
  const m = (typeof iso === "string") ? iso.match(/^(\d{4})-(\d{2})-(\d{2})/) : null;
  return m ? `${+m[3]} ${PT_MON[+m[2] - 1]} ${m[1]}` : "";
}

// ── Per-model SEO valuation page (/preco/{slug}) ─────────────────────────────
// rec = the models.json record. liveDeals = raw hot_deals matching this model
// (below fair), already filtered by the worker. siblings = same-brand models.
// builtAt = models.json build stamp (freshness signal). rec.gl/gm/gh = the
// MODEL fair-value band, present only when it cleared the build-time guards.
export function renderModelPage({ rec, slug, liveDeals, siblings, host, depositCount, builtAt,
                                  insights = [], yearPages = [], competitors = [],
                                  competitorKind = "price", comparisons = [],
                                  facets = [], hasDepreciation = false, duels = [],
                                  hasLiquidity = false, hasVender = false,
                                  provenanceHtml = "", altJson = null }) {
  const B = escapeHtml(rec.b), M = escapeHtml(rec.m);
  const FM = fmtEur(rec.fm), FL = fmtEur(rec.fl), FH = fmtEur(rec.fh);
  const FRESH = fmtBuilt(builtAt);
  const hasG = rec.gm != null && rec.gl != null && rec.gh != null;
  const yr0 = rec.y0, yr1 = rec.y1;
  const yrRange = (yr0 && yr1) ? `${yr0}-${yr1}` : "";
  // median pin within the IQR band
  let pin = 50;
  if (rec.fh > rec.fl) pin = Math.max(6, Math.min(94, Math.round((rec.fm - rec.fl) / (rec.fh - rec.fl) * 100)));
  const fuelChips = Array.isArray(rec.fu)
    ? rec.fu.map(([f, frac]) => `<span class="chip">${escapeHtml(f)} ${Math.round(frac * 100)}%</span>`).join("")
    : "";

  // 1. Hero verdict card
  const hero = `
    <div class="side-card" style="max-width:680px;margin:0 auto;">
      <div class="eyebrow" style="margin-bottom:14px;"><span class="e-dot"></span><span class="mono">AVALIAÇÃO INDEPENDENTE · OLX PORTUGAL</span></div>
      <h1 style="font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:30px;letter-spacing:-0.02em;margin:0 0 10px;line-height:1.1;">Quanto vale um ${B} ${M} usado em Portugal?</h1>
      <p class="lede" style="font-size:16px;margin:0 0 20px;">Com base em ${rec.n} anúncios ativos no OLX, um ${B} ${M} usado pede em mediana <b>${FM}</b> — não é o valor da tua viatura concreta, é o que o mercado está a pedir hoje. Estimativa independente e indicativa.</p>
      <div class="side-prices">
        <div><div class="cap">Preço mediano (pedido)</div><div class="big">${FM}</div></div>
        <div class="side-fair"><div class="cap">${rec.n} anúncios${yrRange ? " · " + yrRange : ""}</div><div class="v">${rec.kmm != null ? fmtKm(rec.kmm) + " med." : ""}</div></div>
      </div>
      <div style="margin-top:16px;">
        <div class="gauge-head"><span>${FL}</span><span>intervalo típico (50% dos anúncios)</span><span>${FH}</span></div>
        <div class="gauge-track"><div class="gauge-pin" style="left:${pin}%;"></div></div>
      </div>
      ${fuelChips ? `<div class="chips" style="margin-top:16px;">${fuelChips}</div>` : ""}
      <div class="mono" style="font-size:11.5px;color:#9A9FA8;margin-top:14px;line-height:1.5;">Preços PEDIDOS em anúncios ativos do OLX — não preço de venda fechado. Estimativa indicativa.${FRESH ? ` · Atualizado a ${FRESH}` : ""}</div>
      ${provenanceHtml}
    </div>`;

  // 1b. Model fair-value band — only when it cleared the build-time guards
  // (asking €5k–45k, agrees with the asking IQR). Absent → asking-only, as before.
  //
  // Framing: the asking MEDIAN of a whole model virtually always sits INSIDE
  // this (wide, CQR) fair band — comparing two central tendencies can't yield a
  // hard "overpaying €X vs fair value" claim (that signal is per-LISTING, on
  // /mercado). So we state WHERE in the fair range the market sits, never a
  // fabricated €-overpay against the point estimate.
  let bandMsg = "";
  if (hasG) {
    const span = rec.gh - rec.gl;
    const pos = span > 0 ? (rec.fm - rec.gl) / span : 0.5;   // asking's place in [gl,gh]
    const BAND = `${fmtEur(rec.gl)} – ${fmtEur(rec.gh)}`;
    if (pos > 0.667)
      bandMsg = `<p style="font-size:14px;color:#177A47;font-weight:600;margin:14px 0 0;">O mercado pede em mediana ${FM} — no <b>terço superior</b> do intervalo justo (${BAND}). Há margem para negociar; não pagues a mais.</p>`;
    else if (pos < 0.333)
      bandMsg = `<p style="font-size:14px;color:#3A3F47;margin:14px 0 0;">O mercado pede em mediana ${FM} — no <b>terço inferior</b> do intervalo justo (${BAND}). Se muito abaixo, confirma o estado e o histórico.</p>`;
    else
      bandMsg = `<p style="font-size:14px;color:#3A3F47;margin:14px 0 0;">O preço pedido em mercado (${FM}) está <b>em linha</b> com o valor justo estimado (${BAND}).</p>`;
  }
  const gbmCard = hasG ? `
    <section class="section" style="padding:22px 22px 0;max-width:680px;">
      <div class="side-card" style="border-color:#DDEBE1;background:#F6FBF8;">
        <div class="eyebrow" style="margin-bottom:12px;"><span class="e-dot" style="background:#177A47;"></span><span class="mono">VALOR JUSTO ESTIMADO · MODELO</span></div>
        <div class="side-prices">
          <div><div class="cap">Valor justo (mediana)</div><div class="big">${fmtEur(rec.gm)}</div></div>
          <div class="side-fair"><div class="cap">intervalo estimado</div><div class="v">${fmtEur(rec.gl)} – ${fmtEur(rec.gh)}</div></div>
        </div>
        ${bandMsg}
        <div class="mono" style="font-size:11.5px;color:#9A9FA8;margin-top:12px;line-height:1.5;">Estimativa do nosso modelo para um ${B} ${M} com quilometragem e specs típicas deste modelo — não considera o estado específico da tua viatura. Para o teu carro concreto, <a href="/avaliar" style="color:#177A47;font-weight:600;">avalia o anúncio</a>.</div>
      </div>
    </section>` : "";

  // 1c. What these numbers mean for THIS model.
  //
  // Generated from the model's own figures against the corpus (seo-pages.js
  // modelInsights), not written once and reused: the previous page had ~49%
  // seven-gram overlap with every other model page, which is the reason a
  // thousand-page expansion would have been partly de-indexed. A rule only
  // speaks when this model actually differs, so the block is a different length
  // and a different shape on every page.
  const insightBlock = (insights && insights.length) ? `
    <section class="section" style="padding:30px 22px 0;max-width:680px;">
      <h2 class="panel-title" style="font-size:18px;margin:0 0 12px;">O que estes números dizem sobre o ${B} ${M}</h2>
      <ul class="fc-insights">${insights.map(t => `<li>${t}</li>`).join("")}</ul>
    </section>` : "";

  // 2. Live matching listings (conversion bridge #1)
  let bridge1;
  if (liveDeals && liveDeals.length) {
    const cards = liveDeals.slice(0, 3).map(d => {
      const p = present(d);
      return `<a class="tile" href="/car?olx_id=${encodeURIComponent(d.olx_id)}" style="max-width:none;">
        <div class="thumb">${thumbBlock(p, 168, 28)}${gradeChip(p)}</div>
        <div class="tbody">
          <div class="tile-title">${escapeHtml(p.name)}</div>
          <div class="tile-sub">${p.subHtml}</div>
          <div class="price-row"><div class="price">${p.priceStr}</div><div class="fair-strike">${p.fairStr}</div>${p.saving != null ? `<div class="profit-pill">poupas ${fmtEur(p.saving)}</div>` : ""}</div>
        </div></a>`;
    }).join("");
    bridge1 = `
      <section class="section" style="padding:30px 22px 0;max-width:1180px;">
        <div class="sec-label">${B} ${M} ABAIXO DO PREÇO JUSTO AGORA</div>
        <div class="grid">${cards}</div>
        <a class="btn-dark" href="/mercado" style="display:inline-block;margin-top:18px;font-size:14px;padding:12px 22px;">Ver todos os ${B} ${M} no mercado&nbsp;&nbsp;→</a>
      </section>`;
  } else {
    bridge1 = `
      <section class="section" style="padding:24px 22px 0;max-width:680px;">
        <div class="info" style="padding:18px 0 0;"><p style="margin:0;">Sem ${B} ${M} abaixo do preço justo neste momento. <a href="/avaliar" style="color:#177A47;font-weight:600;">Avalia o teu ${B} ${M}</a> ou <a href="/mercado" style="color:#177A47;font-weight:600;">vê o mercado completo</a>.</p></div>
      </section>`;
  }

  // 3. Per-year table
  // Years that cleared the year-page floor become links: this is the crawl path
  // into the 565 model-year pages, and the one a reader wants anyway.
  const yearHref = y => `/preco/${slug}/${y}`;
  const yrRows = (rec.yr || []).map(c => `<tr>
      <td>${(typeof c.y === "number" && yearPages.includes(c.y))
        ? `<a href="${yearHref(c.y)}" style="color:#177A47;font-weight:600;">${c.y}</a>`
        : escapeHtml(String(c.y))}</td>
      <td>${c.n}${c.w ? "*" : ""}</td>
      <td>${fmtEur(c.fm)}</td>
      <td class="mut">${c.gm != null ? fmtEur(c.gm) : "—"}</td>
      <td class="mut">${fmtEur(c.fl)}&nbsp;–&nbsp;${fmtEur(c.fh)}</td>
      <td class="mut">${c.km != null ? fmtKm(c.km) : "—"}</td>
    </tr>`).join("");
  const table = yrRows ? `
    <section class="section" style="padding:34px 22px 0;max-width:780px;">
      <h2 class="panel-title" style="font-size:18px;margin:0 0 12px;">Preço de um ${B} ${M} usado por ano</h2>
      <div class="fc-scroll"><table class="year-tbl">
        <thead><tr><th>Ano</th><th>Anúncios</th><th>Mediano (pedido)</th><th>Valor justo</th><th>Intervalo (P25–P75)</th><th>Km mediano</th></tr></thead>
        <tbody>${yrRows}</tbody>
      </table></div>
      ${(rec.yr || []).some(c => c.w) ? `<div class="mono" style="font-size:11.5px;color:#9A9FA8;margin-top:10px;">* anos com poucos anúncios ativos: mediana dos anúncios dos últimos 6 meses, ativos e já fechados.</div>` : ""}
      ${rec.yt ? `<div class="mono" style="font-size:11.5px;color:#9A9FA8;margin-top:10px;">Mais ${rec.yt} ano(s) com poucos anúncios para mostrar um preço fiável.</div>` : ""}
      <div style="font-size:13px;color:#5B606B;margin-top:12px;">Tens um ${B} ${M}${yrRange ? " de " + yrRange : ""}? <a href="/avaliar" style="color:#177A47;font-weight:600;">Avalia o teu&nbsp;→</a></div>
    </section>` : "";

  // 3b. Fuel and district cuts of this same model.
  //
  // Without these links the facet pages are orphans: in the sitemap, reachable
  // by a crawler, and reachable by a reader only if they guess the URL. They are
  // also the natural next question on this page — "and what does the diesel one
  // go for" — so the link earns its place twice.
  const fuelFacets = (facets || []).filter(f => f.kind === "fuel");
  const gearFacets = (facets || []).filter(f => f.kind === "transmission");
  const geoFacets = (facets || []).filter(f => f.kind === "district");
  const facetRow = (label, list) => list.length ? `
      <div style="margin-bottom:16px;">
        <div class="sec-label" style="margin:0 0 9px;">${label}</div>
        <div class="fc-yearlinks">${list.map(f =>
          `<a href="/preco/${encodeURIComponent(slug)}/${encodeURIComponent(f.k)}">${escapeHtml(f.lbl)} <span class="mut">${fmtEur(f.fm)}</span></a>`).join("")}</div>
      </div>` : "";
  const facetBlock = (fuelFacets.length || gearFacets.length || geoFacets.length) ? `
    <section class="section" style="padding:30px 22px 0;max-width:680px;">
      ${facetRow(`${B} ${M} POR COMBUSTÍVEL`, fuelFacets)}
      ${facetRow(`${B} ${M} POR CAIXA`, gearFacets)}
      ${facetRow(`${B} ${M} POR DISTRITO`, geoFacets)}
    </section>` : "";

  // 4. Paste-a-link CTA (bridge #2)
  const bridge2 = `
    <section class="section" style="padding:30px 22px 0;max-width:1180px;">
      <div class="cta-banner">
        <div style="flex:1 1 360px;">
          <h2>Vê o preço exato do TEU anúncio</h2>
          <p>Esta é a média do modelo. Cola o link do teu ${B} ${M} no OLX e dizemos-te o preço justo desse carro específico — quanto poupas ou pagas a mais.</p>
        </div>
        <a class="btn-bright" href="/avaliar">Avaliar o meu anúncio&nbsp;&nbsp;→</a>
      </div>
    </section>`;

  // 5/6. Sell-speed + trust box
  const lqRec = (rec.lq && rec.lq.s30 != null) ? rec.lq : null;
  const sellLine = lqRec
    ? `<p style="font-size:14px;color:#3A3F47;margin:0 0 14px;"><b>${Math.round(lqRec.s30 * 100)} em cada 100</b> anúncios deste modelo saem do OLX no primeiro mês${lqRec.md != null ? `, com uma mediana de <b>${lqRec.md} dias</b>` : ""} (${fmtNum(lqRec.n)} anúncios acompanhados).</p>`
    : rec.sd != null
      ? `<p style="font-size:14px;color:#3A3F47;margin:0 0 14px;">Carros deste modelo vendem, em mediana, em <b>~${rec.sd} dias</b> no OLX (amostra de ${rec.sn} vendas).</p>` : "";
  const trust = `
    <section class="section" style="padding:30px 22px 0;max-width:680px;">
      ${sellLine}
      <div class="exclusive" style="background:#FAFAF8;border:1px solid #EFECE6;align-items:flex-start;">
        <span style="font-size:15px;">📊</span>
        <span class="x" style="color:#5B606B;"><b style="color:#16181D;">Como lemos estes números.</b> Mediana e intervalo P25-P75 dos preços <b>pedidos</b> em ${rec.n} anúncios ativos do OLX — não preços de venda fechados, e não uma avaliação da tua viatura: quilómetros, estado, histórico e o ISV por pagar num importado mudam o valor real. <a href="/metodologia" style="color:#177A47;font-weight:600;">O método, em detalhe</a> · <a href="/avaliar" style="color:#177A47;font-weight:600;">avaliar o meu anúncio</a>.</span>
      </div>
    </section>`;

  // 7. Seller CTA
  const sellerCta = `
    <section class="section" style="padding:30px 22px 0;max-width:1180px;">
      <div class="cta-banner" style="background:#fff;border:1px solid #E8E6E1;">
        <div style="flex:1 1 360px;">
          <h2 style="color:#16181D;">Vais vender o teu ${B} ${M}?</h2>
          <p style="color:#5B606B;">Metade dos ${B} ${M} anunciados no OLX pede entre ${FL} e ${FH}. Sabe onde cai o teu antes de anunciar — grátis.</p>
        </div>
        <a class="btn-dark" href="/avaliar" style="font-size:15px;padding:14px 26px;">Avaliar o meu carro&nbsp;&nbsp;→</a>
      </div>
    </section>`;

  const bySegment = competitorKind === "segment";
  const compChips = (competitors || []).slice(0, 6).map(c => {
    const rel = (bySegment && c.ratio > 0)
      ? (Math.abs(c.ratio - 1) < 0.025 ? "igual"
        : `${c.ratio > 1 ? "+" : "−"}${Math.round((c.ratio > 1 ? c.ratio - 1 : 1 - c.ratio) * 100)}%`)
      : fmtEur(c.fm);
    return `<a class="mchip" href="/preco/${encodeURIComponent(c.slug)}">${escapeHtml(c.b)} ${escapeHtml(c.m)} <span class="mut">${rel}</span></a>`;
  }).join("");
  const cmpChips = (comparisons || []).slice(0, 4).map(c =>
    `<a class="mchip" href="/comparar/${c.href}">${escapeHtml(rec.m)} <span class="mut">vs</span> ${escapeHtml(c.m)}</a>`).join("");
  const rivals = (compChips || cmpChips) ? `
    <section class="section" style="padding:34px 22px 0;max-width:1180px;">
      ${compChips ? `<div class="sec-label">${bySegment ? `ALTERNATIVAS NO MESMO SEGMENTO · PREÇO AO MESMO ANO QUE O ${B} ${M}` : "ALTERNATIVAS NA MESMA FAIXA DE PREÇO"}</div><div class="mchips">${compChips}</div>` : ""}
      ${cmpChips ? `<div class="sec-label" style="margin-top:22px;">COMPARAÇÕES DIRETAS</div><div class="mchips">${cmpChips}</div>` : ""}
    </section>` : "";

  // 7c. Depreciation, where the curve exists.
  const depLink = hasDepreciation ? `
    <section class="section" style="padding:26px 22px 0;max-width:680px;">
      <div class="exclusive" style="background:#F6FBF8;border:1px solid #DDEBE1;align-items:flex-start;">
        <span style="font-size:15px;">📉</span>
        <span class="x" style="color:#3A3F47;"><b style="color:#16181D;">Quanto perde por ano.</b> Temos histórico suficiente para traçar a curva de desvalorização do ${B} ${M} e dizer quanto custa, em euros, cada ano de matrícula. <a href="/depreciacao/${slug}" style="color:#177A47;font-weight:600;">Ver a curva&nbsp;→</a></span>
      </div>
    </section>` : "";

  const venderLink = hasVender ? `
    <section class="section" style="padding:26px 22px 0;max-width:680px;">
      <div class="exclusive" style="background:#F4F6FB;border:1px solid #D9E0F0;align-items:flex-start;">
        <span style="font-size:15px;">🏷️</span>
        <span class="x" style="color:#3A3F47;"><b style="color:#16181D;">Vais vender o teu ${B} ${M}?</b> Quanto pedir por ano, em quantos dias sai e quantos vendedores acabam por baixar o preço — e propostas de compra sem compromisso. <a href="/vender/${slug}" style="color:#177A47;font-weight:600;">Ver quanto pedir&nbsp;→</a></span>
      </div>
    </section>` : "";

  const liqLink = hasLiquidity ? `
    <section class="section" style="padding:26px 22px 0;max-width:680px;">
      <div class="exclusive" style="background:#F6FBF8;border:1px solid #DDEBE1;align-items:flex-start;">
        <span style="font-size:15px;">⏱️</span>
        <span class="x" style="color:#3A3F47;"><b style="color:#16181D;">Quanto tempo demora a vender.</b> Acompanhámos anúncios de ${B} ${M} suficientes para dizer quantos saem no primeiro mês, e como isso muda com o preço a que os pões. <a href="/liquidez/${slug}" style="color:#177A47;font-weight:600;">Ver o tempo de venda&nbsp;→</a></span>
      </div>
    </section>` : "";

  const DUEL_COPY = {
    fuel: { icon: "⛽", head: "Diesel ou gasolina?", what: "os dois combustíveis", which: "qual dos dois" },
    gear: { icon: "⚙️", head: "Caixa manual ou automática?", what: "as duas caixas", which: "qual das duas" },
  };
  const duelLink = (duels || []).map(d => {
    const c = DUEL_COPY[d.kind];
    if (!c) return "";
    return `
    <section class="section" style="padding:26px 22px 0;max-width:680px;">
      <div class="exclusive" style="background:#F6FBF8;border:1px solid #DDEBE1;align-items:flex-start;">
        <span style="font-size:15px;">${c.icon}</span>
        <span class="x" style="color:#3A3F47;"><b style="color:#16181D;">${c.head}</b> Neste modelo há anúncios ativos que cheguem para ajustar ${c.what} em separado, com a quilometragem igualada, e dizer ${c.which} segura melhor o preço. <a href="/${d.path}/${encodeURIComponent(slug)}" style="color:#177A47;font-weight:600;">Ver a resposta&nbsp;→</a></span>
      </div>
    </section>`;
  }).join("");

  // 8. Sibling models footer
  const sibChips = (siblings || []).slice(0, 8).map(s =>
    `<a class="mchip" href="/preco/${encodeURIComponent(s.slug)}">${escapeHtml(s.m)} <span class="mut">${fmtEur(s.fm)}</span></a>`).join("");
  const sib = sibChips ? `
    <section class="section" style="padding:34px 22px 70px;max-width:1180px;">
      <div class="sec-label">OUTROS MODELOS ${B.toUpperCase()}</div>
      <div class="mchips">${sibChips}</div>
      <div style="margin-top:16px;"><a href="/precos" style="font-size:13px;color:#177A47;font-weight:600;">Ver todos os modelos&nbsp;→</a></div>
    </section>` : `<section class="section" style="padding:34px 22px 70px;"><a href="/precos" style="font-size:13px;color:#177A47;font-weight:600;">Ver preços de todos os modelos&nbsp;→</a></section>`;

  // Visible breadcrumb — mirrors the BreadcrumbList JSON-LD and adds real
  // internal links back to / and /precos (reinforcing the crawl spine).
  const crumb = `<nav class="section" aria-label="Breadcrumb" style="max-width:680px;padding:22px 22px 0;font-size:12.5px;color:#8A8F98;">`
    + `<a href="/" style="color:#8A8F98;">Início</a> › <a href="/precos" style="color:#8A8F98;">Preços</a> › <span style="color:#16181D;">${B} ${M}</span></nav>`;
  const body = `${crumb}<div style="padding-top:14px;">${hero}</div>${gbmCard}${insightBlock}${bridge1}${table}${facetBlock}${duelLink}${depLink}${liqLink}${venderLink}${bridge2}${trust}${rivals}${sellerCta}${sib}`;

  const canonical = `https://${host}/preco/${slug}`;
  const faq = (q, a) => ({
    "@type": "Question", "name": q,
    "acceptedAnswer": { "@type": "Answer", "text": a },
  });
  const faqEntries = [
    faq(
      `Quanto vale um ${rec.b} ${rec.m} usado em Portugal?`,
      `Com base em ${rec.n} anúncios ativos no OLX, um ${rec.b} ${rec.m} usado pede em mediana ${FM}, com um intervalo típico entre ${FL} e ${FH}. É o preço pedido no mercado hoje, não o valor de uma viatura concreta. Estimativa independente e indicativa.`,
    ),
    faq(
      `O preço mediano de um ${rec.b} ${rec.m} é o preço de venda?`,
      `Não. Mostramos a mediana e o intervalo dos preços PEDIDOS em ${rec.n} anúncios ativos do OLX, não preços de venda fechados. O valor real depende de quilómetros, estado, extras, histórico e de ser importado com ISV por pagar.`,
    ),
  ];
  if (rec.fl != null && rec.fh != null) {
    faqEntries.push(faq(
      `Qual é o intervalo de preços de um ${rec.b} ${rec.m} usado?`,
      `Metade dos ${rec.b} ${rec.m} anunciados no OLX pede entre ${FL} e ${FH} (intervalo interquartil P25–P75)${yrRange ? `, para anos ${yrRange}` : ""}. Fora deste intervalo ficam os 25% mais baratos e os 25% mais caros.`,
    ));
  }
  if (lqRec) {
    faqEntries.push(faq(
      `Quanto tempo demora a vender um ${rec.b} ${rec.m} em Portugal?`,
      `${Math.round(lqRec.s30 * 100)} em cada 100 anúncios de ${rec.b} ${rec.m} saem do OLX no primeiro mês${lqRec.md != null ? `, com uma mediana de ${lqRec.md} dias` : ""}, medido em ${lqRec.n} anúncios acompanhados até ao fim. Sair do OLX não prova a venda: um anúncio corre em ciclos de 30 dias e pode expirar sem ter vendido.`,
    ));
  } else if (rec.sd != null && rec.sn != null) {
    faqEntries.push(faq(
      `Quanto tempo demora a vender um ${rec.b} ${rec.m} em Portugal?`,
      `Um ${rec.b} ${rec.m} vende, em mediana, em cerca de ${rec.sd} dias no OLX, medido numa amostra de ${rec.sn} vendas deste modelo. Modelos que demoram mais a vender costumam exigir preço mais agressivo.`,
    ));
  }
  if (rec.kmm != null) {
    faqEntries.push(faq(
      `Qual é a quilometragem típica de um ${rec.b} ${rec.m} à venda?`,
      `A quilometragem mediana dos ${rec.b} ${rec.m} anunciados é de ${fmtKm(rec.kmm)}${yrRange ? `, para anos ${yrRange}` : ""}. Acima desse valor espera-se um preço abaixo da mediana, e vice-versa.`,
    ));
  }
  if (hasG) {
    faqEntries.push(faq(
      `O ${rec.b} ${rec.m} está caro ou barato neste momento?`,
      `O preço pedido em mercado (${FM}) compara com um valor justo estimado de ${fmtEur(rec.gm)} (intervalo ${fmtEur(rec.gl)} a ${fmtEur(rec.gh)}), calculado pelo nosso modelo para quilometragem e specs típicas deste modelo. Para uma leitura da tua viatura concreta é preciso avaliar o anúncio específico.`,
    ));
  }

  const jsonLd = {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "Dataset",
        // Те же условия, что и на /metodologia#licenca. Строкой, а не импортом
        // из seo-pages.js: зависимость между модулями односторонняя.
        "license": `https://${host}/metodologia#licenca`,
        "name": `Preços de ${rec.b} ${rec.m} usado em Portugal`,
        "description": `Resumo estatístico (mediana, P25–P75) de ${rec.n} anúncios ativos de ${rec.b} ${rec.m} no OLX Portugal, por ano.`,
        "creator": { "@type": "Organization", "name": "Carsbuyer" },
        "isAccessibleForFree": true,
        "temporalCoverage": yrRange ? `${yr0}/${yr1}` : undefined,
        "variableMeasured": hasG ? ["Preço pedido (EUR)", "Valor justo estimado (EUR)"] : "Preço pedido (EUR)",
        "dateModified": builtAt || undefined,
        "url": canonical,
        ...(altJson ? {
          "distribution": {
            "@type": "DataDownload",
            "encodingFormat": "application/json",
            "contentUrl": altJson,
          },
        } : {}),
      },
      // AggregateOffer over the asking prices, NOT a Product.
      //
      // Product/Offer would assert that we sell these cars; we don't, and
      // marking up merchandise you don't sell is what earns a manual action.
      // AggregateOffer with lowPrice/highPrice/offerCount and no seller is the
      // accurate description of what this page holds: the range of asking
      // prices across N third-party listings. itemOffered names the vehicle so
      // the offer isn't floating free of its subject.
      {
        "@type": "AggregateOffer",
        "priceCurrency": "EUR",
        "lowPrice": rec.fl, "highPrice": rec.fh, "offerCount": rec.n,
        "url": canonical,
        "itemOffered": {
          "@type": "Car",
          "name": `${rec.b} ${rec.m}`,
          "brand": { "@type": "Brand", "name": rec.b },
          "model": rec.m,
          ...(rec.kmm != null ? {
            "mileageFromOdometer": { "@type": "QuantitativeValue", "value": rec.kmm, "unitCode": "KMT" },
          } : {}),
          ...(Array.isArray(rec.fu) && rec.fu.length ? { "fuelType": rec.fu[0][0] } : {}),
        },
      },
      {
        "@type": "BreadcrumbList",
        "itemListElement": [
          { "@type": "ListItem", "position": 1, "name": "Início", "item": `https://${host}/` },
          { "@type": "ListItem", "position": 2, "name": "Preços", "item": `https://${host}/precos` },
          { "@type": "ListItem", "position": 3, "name": `${rec.b} ${rec.m}` },
        ],
      },
      // FAQPage — the H1 is a literal question; every answer mirrors a figure
      // the page already shows (no content mismatch) and can win a FAQ rich
      // result. Generative engines cite data-backed Q&A far more readily than
      // prose, but only entries whose numbers actually exist are emitted:
      // a fabricated answer is worse than a missing one, and a model with no
      // sell-speed sample must not claim one.
      { "@type": "FAQPage", "mainEntity": faqEntries },
    ],
  };
  return layout({
    title: `${rec.b} ${rec.m} usado: ${FM} (${rec.n} anúncios) · quanto vale em Portugal`,
    description: `${rec.b} ${rec.m} usado em Portugal: preço mediano ${FM} (intervalo ${FL}–${FH}), com base em ${rec.n} anúncios ativos no OLX. Preços por ano e avaliação independente grátis.`,
    canonical, jsonLd, body, zone: "all", nav: "precos", depositCount, index: true, host, altJson,
  });
}

// ── Models hub (/precos) — the crawl spine: one link to every model page ─────
export function renderModelsHub({ models, depositCount, builtAt, host, districts = [] }) {
  const FRESH = fmtBuilt(builtAt);
  // models = [{slug, b, m, fm, n}], pre-sorted by the worker. Group by brand.
  const byBrand = new Map();
  for (const m of models) {
    if (!byBrand.has(m.b)) byBrand.set(m.b, []);
    byBrand.get(m.b).push(m);
  }
  const brands = [...byBrand.keys()].sort((a, b) => a.localeCompare(b, "pt"));
  const groups = brands.map(b => {
    const chips = byBrand.get(b).map(m =>
      `<a class="mchip" href="/preco/${encodeURIComponent(m.slug)}">${escapeHtml(m.m)} <span class="mut">· mediana ${fmtEur(m.fm)} · ${m.n}</span></a>`).join("");
    return `<div style="margin-bottom:22px;"><h2 class="sec-label" style="margin:0 0 10px;">${escapeHtml(b)}</h2><div class="mchips">${chips}</div></div>`;
  }).join("");

  const body = `
    <section class="hero" style="padding-bottom:18px;">
      <div class="hero-copy" style="max-width:760px;">
        <div class="eyebrow" style="margin-bottom:18px;"><span class="e-dot"></span><span class="mono">${models.length} MODELOS · OLX PORTUGAL${FRESH ? ` · ATUALIZADO A ${FRESH}` : ""}</span></div>
        <h1 class="hero-title" style="font-size:38px;">Preço de carros usados em Portugal por modelo</h1>
        <p class="lede">Avaliação independente a partir de anúncios ativos do OLX. Escolhe o modelo para ver o preço mediano e o intervalo por ano.</p>
        <div class="hero-actions">
          <a class="btn-dark" href="/avaliar">Avaliar o TEU carro&nbsp;&nbsp;→</a>
          <a class="chip" href="/mercado">Ver mercado</a>
        </div>
        <p style="font-size:13.5px;color:#5B606B;margin:20px 0 0;line-height:1.9;">
          Também aqui:
          <a href="/depreciacao" style="color:#177A47;font-weight:600;">que modelos perdem mais valor</a> ·
          <a href="/liquidez" style="color:#177A47;font-weight:600;">quanto tempo demoram a vender</a> ·
          <a href="/comparar" style="color:#177A47;font-weight:600;">comparações diretas</a> ·
          <a href="/sobrevalorizados" style="color:#177A47;font-weight:600;">pedido vs. valor justo</a> ·
          <a href="/mercado/indice" style="color:#177A47;font-weight:600;">índice semanal</a> ·
          <a href="/metodologia" style="color:#177A47;font-weight:600;">como calculamos</a>
        </p>
      </div>
    </section>
    ${districts.length ? `<section class="section" style="padding:18px 22px 0;max-width:1180px;">
      <h2 class="sec-label" style="margin:0 0 10px;">PREÇOS POR DISTRITO</h2>
      <div class="mchips">${districts.map(d =>
        `<a class="mchip" href="/precos/${encodeURIComponent(d.k)}">${escapeHtml(d.lbl)} <span class="mut">· mediana ${fmtEur(d.fm)} · ${fmtNum(d.n)}</span></a>`).join("")}</div>
    </section>` : ""}
    <section class="section" style="padding:18px 22px 70px;max-width:1180px;">${groups}</section>`;
  const origin = host ? `https://${host}` : "";
  // Lightweight CollectionPage (no 465-item ItemList — the visible mchips + the
  // sitemap already give Google every /preco link; a full ItemList would ~double
  // this page's HTML weight for marginal gain).
  const jsonLd = origin ? {
    "@context": "https://schema.org",
    "@type": "CollectionPage",
    "name": "Preço de carros usados em Portugal por modelo",
    "description": "Preço mediano e intervalo por ano de carros usados em Portugal, a partir de anúncios ativos do OLX.",
    "url": `${origin}/precos`,
    "inLanguage": "pt-PT",
    "isPartOf": { "@id": `${origin}/#site` },
  } : null;
  return layout({
    title: "Preço de carros usados em Portugal por modelo",
    description: "Preço mediano e intervalo por ano de carros usados em Portugal, a partir de anúncios ativos do OLX. Avaliação independente e grátis por modelo.",
    canonical: origin ? `${origin}/precos` : null, jsonLd,
    body, zone: "all", nav: "precos", depositCount, index: true, host,
  });
}

// ── Embeddable widget (/widget/preco/{slug}) — the backlink lever ────────────
// A self-contained, iframe-friendly valuation card any site can embed:
//   <iframe src="https://HOST/widget/preco/opel-corsa" width="340" height="300"
//           style="border:0" loading="lazy"></iframe>
// Own minimal HTML (no shared header/footer/cookie), noindex (the /preco page is
// canonical), and a prominent link back to the full page for attribution.
export function renderModelWidget({ rec, slug, host }) {
  const B = escapeHtml(rec.b), M = escapeHtml(rec.m);
  const hasG = rec.gm != null && rec.gl != null && rec.gh != null;
  const full = `https://${escapeHtml(host)}/preco/${encodeURIComponent(slug)}`;
  const fairRow = hasG ? `
    <div class="w-fair">
      <div class="w-cap">Valor justo estimado</div>
      <div class="w-fair-v">${fmtEur(rec.gm)}</div>
      <div class="w-band">${fmtEur(rec.gl)} – ${fmtEur(rec.gh)}</div>
    </div>` : "";
  const sellRow = rec.sd != null
    ? `<div class="w-sell">Vende, em mediana, em ~${rec.sd} dias no OLX</div>` : "";
  return `<!doctype html><html lang="pt"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex,follow">
<link rel="canonical" href="${full}">
<title>${B} ${M}: quanto vale · Carsbuyer</title>
<style>
*{box-sizing:border-box;margin:0}
body{font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;color:#16181D;background:#fff;padding:14px}
.w-card{border:1px solid #E8E6E1;border-radius:14px;padding:16px 18px;max-width:360px}
.w-eyebrow{font:600 10.5px/1 ui-monospace,monospace;letter-spacing:.08em;color:#177A47;text-transform:uppercase;margin-bottom:10px}
.w-h{font-weight:700;font-size:17px;letter-spacing:-.01em;margin-bottom:12px}
.w-cap{font-size:11px;color:#8A8F98}
.w-ask{font-weight:700;font-size:26px;letter-spacing:-.02em}
.w-band{font-size:12px;color:#5B606B}
.w-fair{margin-top:12px;padding-top:12px;border-top:1px solid #EFECE6}
.w-fair-v{font-weight:700;font-size:20px;color:#177A47;letter-spacing:-.01em}
.w-sell{margin-top:12px;font-size:12.5px;color:#3A3F47}
.w-cta{display:block;margin-top:14px;text-align:center;background:#16181D;color:#fff;text-decoration:none;font-weight:600;font-size:13px;padding:11px;border-radius:9px}
.w-credit{margin-top:9px;text-align:center;font-size:11px;color:#9A9FA8}
.w-credit a{color:#177A47;text-decoration:none;font-weight:600}
.w-note{margin-top:10px;font:400 10.5px/1.5 ui-monospace,monospace;color:#9A9FA8}
</style></head>
<body>
<div class="w-card">
  <div class="w-eyebrow">● Avaliação independente</div>
  <div class="w-h">Quanto vale um ${B} ${M} usado?</div>
  <div class="w-cap">Preço mediano (pedido) · ${rec.n} anúncios OLX</div>
  <div class="w-ask">${fmtEur(rec.fm)}</div>
  <div class="w-band">intervalo ${fmtEur(rec.fl)} – ${fmtEur(rec.fh)}</div>
  ${fairRow}
  ${sellRow}
  <a class="w-cta" href="${full}" target="_blank" rel="noopener">Ver avaliação completa&nbsp;&nbsp;→</a>
  <div class="w-credit">via <a href="${full}" target="_blank" rel="noopener">Carsbuyer</a></div>
  <div class="w-note">Preços pedidos em anúncios ativos do OLX — estimativa indicativa, não vinculativa.</div>
</div>
</body></html>`;
}

export const PT_DISTRICTS = [
  "Aveiro", "Beja", "Braga", "Bragança", "Castelo Branco", "Coimbra", "Évora", "Faro",
  "Guarda", "Leiria", "Lisboa", "Portalegre", "Porto", "Santarém", "Setúbal",
  "Viana do Castelo", "Vila Real", "Viseu", "Açores", "Madeira",
];

export function leadFormBlock({ slug = "", name = "", year = null, median = null, district = "" }) {
  const opts = PT_DISTRICTS.map(d =>
    `<option value="${escapeHtml(d)}"${d === district ? " selected" : ""}>${escapeHtml(d)}</option>`).join("");
  const inp = "padding:12px;border:1px solid #E2DFD8;border-radius:11px;font-size:15px;background:#fff;color:#16181D;min-width:0;";
  return `
      <section id="vender" class="side-card" style="margin-top:16px;">
        <div class="panel-title" style="font-size:16px;margin-bottom:6px;">Queres vender${name ? ` o teu ${escapeHtml(name)}` : " este carro"}?</div>
        <p style="font-size:14px;color:#5B606B;margin:0 0 14px;line-height:1.5;">Deixa o contacto e recebes propostas de compra de compradores profissionais — stands e serviços de compra imediata — normalmente em 24 a 48 horas. Sem compromisso: comparas com ${median != null ? `a mediana de ${fmtEur(median)}` : "o valor acima"} e decides.</p>
        <form action="/lead" method="post" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;">
          <input type="hidden" name="modelo" value="${escapeHtml(slug)}">
          <input type="hidden" name="nome_modelo" value="${escapeHtml(name)}">
          <input type="text" name="website" value="" tabindex="-1" autocomplete="off" aria-hidden="true" style="position:absolute;left:-9999px;width:1px;height:1px;">
          <input type="number" name="ano" min="1980" max="2027" required value="${year ? escapeHtml(String(year)) : ""}" placeholder="Ano" style="${inp}">
          <input type="number" name="km" min="0" max="1500000" step="1000" placeholder="Quilómetros" style="${inp}">
          <select name="distrito" style="${inp}"><option value="">Distrito…</option>${opts}</select>
          <input type="text" name="contacto" required minlength="5" maxlength="120" placeholder="Telemóvel ou email" autocomplete="tel" style="${inp}">
          <input type="text" name="nome" maxlength="80" placeholder="Nome (opcional)" autocomplete="name" style="${inp}grid-column:1 / -1;">
          <label style="grid-column:1 / -1;display:flex;gap:9px;align-items:flex-start;font-size:13px;color:#5B606B;line-height:1.45;">
            <input type="checkbox" name="consent" value="1" required style="margin-top:3px;">
            <span>Autorizo o Carsbuyer a guardar estes dados durante 90 dias e a enviá-los a compradores profissionais para me apresentarem propostas. <a href="/privacidade">Privacidade</a>.</span>
          </label>
          <button type="submit" class="btn-dark" style="grid-column:1 / -1;padding:13px 20px;font-size:15px;">Receber propostas de compra&nbsp;&nbsp;→</button>
        </form>
        <div class="mono" style="font-size:11px;color:#9A9FA8;margin-top:10px;line-height:1.5;">Não vendemos o teu contacto para publicidade. Um comprador profissional paga ao Carsbuyer por este contacto — é assim que o site se financia, e a avaliação não muda por isso.</div>
      </section>`;
}

export function historyCheckBlock({ url, reasons = [], price = null, title = null }) {
  if (!url) return "";
  const items = reasons.filter(Boolean).map(r => `<li style="margin:0 0 4px;">${r}</li>`).join("");
  const head = title || `Antes de pagar${price != null ? ` ${fmtEur(price)}` : ""}, verifica o histórico`;
  return `
        <div style="background:#F4F6FB;border:1px solid #D9E0F0;border-radius:12px;margin-top:16px;padding:14px 16px;">
          <div style="font-weight:600;color:#16181D;font-size:14.5px;">${escapeHtml(head)}</div>
          <p style="font-size:13.5px;color:#5B606B;margin:6px 0 0;line-height:1.5;">Quilómetros reais, sinistros, número de donos e se veio do estrangeiro: o anúncio não diz, o relatório pela matrícula ou pelo VIN diz.${items ? " Neste caso há motivos concretos:" : ""}</p>
          ${items ? `<ul style="margin:8px 0 0;padding-left:18px;font-size:13.5px;color:#16181D;line-height:1.5;">${items}</ul>` : ""}
          <a href="${escapeHtml(url)}" target="_blank" rel="nofollow sponsored noopener" class="btn-outline" style="display:inline-block;margin-top:12px;padding:10px 16px;font-size:13.5px;">Verificar o histórico do carro&nbsp;&nbsp;↗</a>
          <div class="mono" style="font-size:11px;color:#9A9FA8;margin-top:8px;line-height:1.5;">Ligação de parceiro: se comprares um relatório, o Carsbuyer recebe uma comissão. O preço para ti é o mesmo e a avaliação não muda.</div>
        </div>`;
}

export function historyReasons(rec, models) {
  if (!rec) return [];
  const out = [];
  if (rec.imp && !rec.il) out.push("Há indícios de importação e não está claro se o ISV foi pago: confirma a origem e a data de entrada em Portugal.");
  else if (rec.imp) out.push("Carro importado: o que aconteceu antes de entrar em Portugal só aparece num relatório internacional.");
  const mr = (rec.ms && models) ? models[rec.ms] : null;
  let refKm = null, refYear = false;
  if (mr && Array.isArray(mr.yr) && typeof rec.y === "number") {
    const c = mr.yr.find(x => x.y === rec.y);
    if (c && c.km != null) { refKm = c.km; refYear = true; }
  }
  if (refKm == null && mr && mr.kmm != null) refKm = mr.kmm;
  if (refKm != null && rec.km != null && rec.km > 0 && rec.km < refKm * 0.6) {
    out.push(`Marca ${fmtKm(rec.km)}, bem abaixo do habitual para este modelo${refYear ? " e ano" : ""} (${fmtKm(refKm)}): confirma o conta-quilómetros nas inspeções anteriores.`);
  }
  const track = Array.isArray(rec.ph) ? rec.ph : null;
  if (track && track.length >= 3) out.push(`O preço já baixou ${track.length - 1} vezes: pergunta porquê antes de aceitar a descida.`);
  if (rec.dom != null && rec.sd != null && rec.dom > rec.sd * 1.5) {
    out.push(`Está à venda há ${rec.dom} dias, mais do que o normal para o modelo (~${rec.sd}): um carro que não sai costuma ter uma razão.`);
  }
  return out;
}

export function renderLeadThanks({ name = "", year = null, depositCount = null, host = null }) {
  const car = [name, year].filter(Boolean).join(" ");
  const body = `
    <div class="info">
      <div class="ic">✅</div>
      <h1>Pedido recebido</h1>
      <p>${car ? `Vamos apresentar o teu ${escapeHtml(car)} a compradores profissionais. ` : ""}Se houver interesse, recebes propostas em 24 a 48 horas pelo contacto que deixaste. Não há compromisso: aceitas só se a proposta te servir.</p>
      <p style="font-size:13.5px;color:#5B606B;">Entretanto, vê <a href="/avaliar">o que o mercado pede por carros como o teu</a> — é a referência para comparar as propostas.</p>
      <a class="btn-dark" href="/avaliar">Voltar à avaliação</a>
    </div>
    ${analyticsEvent("generate_lead", { model: car })}`;
  return layout({ title: "Pedido recebido", body, zone: "all", nav: "avaliar", depositCount, index: false, host });
}
