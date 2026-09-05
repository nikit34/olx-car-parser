// Second-layer SEO pages — everything narrower or wider than "one model".
//
// Why a second file: templates.js owns the product (feed, car, checkout) and the
// design system. This owns the pages that exist to answer a search query, all of
// which are derived from the SAME models.json blob the /preco pages already read.
// No new data source, no new pipeline — the numbers are re-cut, not re-collected.
//
// What lives here:
//   /preco/{slug}/{ano}      one model in one year  — the query is "quanto vale
//                            um Golf de 2012", and the model page answers it for
//                            every year at once, which wins none of them.
//   /depreciacao/{slug}      how fast this model loses value (+ the hub)
//   /comparar/{a}-vs-{b}     two models side by side (+ the hub)
//   /liquidez                how long each model takes to sell
//   /sobrevalorizados        where asking price and estimated fair value diverge
//   /mercado/indice          market-wide medians, with a permanent weekly archive
//   /metodologia /sobre      how the numbers are made, and by whom
//   /isv                     ISV estimator for an imported car
//   404                      a real not-found page (see index.js: unknown paths
//                            used to fall into the analytics Basic-Auth gate and
//                            answer 401, which Googlebot reads as "forbidden",
//                            not "gone")
//
// THE PUBLISHING RULE, applied everywhere below: a page exists only where the
// sample behind it is big enough for its number to mean something. A model-year
// with 4 listings is a row in the parent table, never a URL of its own. Minting
// URLs for thin cells is how a 250-page site becomes a 1300-page site that ranks
// for less than it did before.

import {
  layout, escapeHtml, fmtEur, fmtKm, fmtNum, fmtBuilt, slugify,
  present, thumbBlock, gradeChip, historyCheckBlock, leadFormBlock,
} from "./templates.js";

// ── Publishing thresholds ────────────────────────────────────────────────────
// A model-year gets its own URL at 10+ active listings. Rationale: at n=5 (the
// blob's own floor for showing a row) a single outlier moves the median by more
// than the year-over-year step we are claiming to measure, so the page would be
// asserting a difference it cannot see. 10 is where the per-year medians stop
// crossing each other out of order in the corpus.
export const MIN_YEAR_PAGE_N = 10;

// Условия повторного использования опубликованных цифр. Полный текст лежит на
// /metodologia#licenca и продублирован в поле licence каждого .json-эндпоинта,
// чтобы машиночитаемая и человекочитаемая формулировки не разъехались.
const licenseUrl = (host) => `https://${host}/metodologia#licenca`;

// A depreciation curve needs enough points, over enough time, that a straight
// line through them is a description rather than an interpolation.
const DEP_MIN_CELLS = 8;      // year cells (n>=5) feeding the fit
const DEP_MIN_SPAN = 8;       // years between the oldest and newest cell
const DEP_MIN_R2 = 0.55;      // fit must actually explain the points
const DEP_MIN_RATE = 0.03;    // <3%/yr is noise or a collector car, not depreciation
const DEP_MAX_RATE = 0.22;    // >22%/yr means the year mix, not age, is driving it

// Comparison pages are generated, so they need a hard cap or they become the
// site: 243 models is 29 403 unordered pairs. These bounds keep only pairs a
// buyer would actually put against each other.
const CMP_POOL = 60;          // deepest-sampled models only
const CMP_PRICE_TOL = 0.50;
const CMP_MIN_YEARS = 6;
const CMP_PER_MODEL = 3;      // so one popular model can't monopolise the set
const CMP_MAX = 50;
const CMP_TABLE_YEARS = 12;

const MODEL_CLASS = new Map(Object.entries({
  "smart-fortwo-coupe": "a", "smart-fortwo-cabrio": "a", "citroen-c1": "a",
  "renault-twingo": "a", "fiat-panda": "a",
  "opel-corsa": "b", "renault-clio": "b", "seat-ibiza": "b", "volkswagen-polo": "b",
  "citroen-c3": "b", "ford-fiesta": "b", "peugeot-208": "b", "peugeot-207": "b",
  "peugeot-206": "b", "fiat-punto": "b", "fiat-grande-punto": "b", "toyota-yaris": "b",
  "dacia-sandero": "b", "nissan-micra": "b", "citroen-c2": "b", "citroen-saxo": "b",
  "chevrolet-aveo": "b",
  "fiat-500": "b-premium", "mini-3-portas": "b-premium", "mini-cooper": "b-premium",
  "mini-clubman": "b-premium", "mini-cabrio": "b-premium", "mini-coupe": "b-premium",
  "alfa-romeo-mito": "b-premium",
  "volkswagen-golf": "c", "opel-astra": "c", "renault-megane": "c", "ford-focus": "c",
  "peugeot-308": "c", "peugeot-307": "c", "seat-leon": "c", "citroen-c4": "c",
  "citroen-c4-cactus": "c", "honda-civic": "c", "toyota-corolla": "c",
  "audi-a3": "c", "audi-a3-sportback": "c", "bmw-116": "c", "bmw-118": "c",
  "bmw-120": "c", "volvo-v40": "c", "mercedes-benz-a-180": "c",
  "volkswagen-golf-variant": "c-estate", "opel-astra-sports-tourer": "c-estate",
  "opel-astra-caravan": "c-estate", "renault-megane-sport-tourer": "c-estate",
  "renault-megane-break": "c-estate", "ford-focus-sw": "c-estate",
  "peugeot-308-sw": "c-estate", "peugeot-307-sw": "c-estate",
  "skoda-octavia-break": "c-estate", "volvo-v50": "c-estate",
  "bmw-316": "d", "bmw-318": "d", "bmw-320": "d", "bmw-330": "d", "audi-a4": "d",
  "mercedes-benz-c-200": "d", "mercedes-benz-c-220": "d", "mercedes-benz-c-300": "d",
  "mercedes-benz-220": "ambiguo",
  "volkswagen-passat": "d", "citroen-c5": "d", "tesla-model-3": "d",
  "audi-a4-avant": "d-estate", "volkswagen-passat-variant": "d-estate",
  "peugeot-508-sw": "d-estate", "volvo-v60": "d-estate",
  "bmw-520": "e", "bmw-525": "e", "bmw-530": "e",
  "mercedes-benz-e-220": "e", "mercedes-benz-e-300": "e",
  "audi-a6-avant": "e-estate",
  "peugeot-2008": "suv-b", "renault-captur": "suv-b", "nissan-juke": "suv-b",
  "nissan-qashqai": "suv-c", "peugeot-3008": "suv-c", "bmw-x1": "suv-c",
  "bmw-x3": "suv-c", "mini-countryman": "suv-c",
  "peugeot-5008": "suv-d", "bmw-x5": "suv-d",
  "renault-scenic": "mpv", "renault-grand-scenic": "mpv", "opel-zafira": "mpv",
  "citroen-c4-grand-picasso": "mpv", "ford-s-max": "mpv",
}));

// ── Page-set selection ───────────────────────────────────────────────────────
// These are pure functions of the blob, and BOTH the router and the sitemap call
// them. That is deliberate: if the two disagreed, we would either serve pages no
// sitemap advertises or advertise pages that 404 — and the second is the failure
// mode Search Console reports as a site-wide error.

// ── Staged rollout ───────────────────────────────────────────────────────────
//
// 565 model-year pages appearing on one crawl is a bad way to find out whether
// the layer works: if indexation comes back at 40% there is no way to tell an
// unconvincing page template from a crawl-budget wall.
//
// SEO_WAVE_MODELS (wrangler.toml [vars]) caps the PER-MODEL layer to the N
// deepest-sampled models. The gate is applied in ONE place and everything reads
// it — router, sitemap, and the year links on the model page — so a page outside
// the current wave is not merely unlisted, it is unreachable and unlinked.
// Widening the number is the whole release step; no deploy of code, no data
// rebuild.
//
//
// Empty ⇒ no cap, every qualifying page is live.
let WAVE_MODELS = 0;
export function setWave(n) {
  const v = parseInt(n, 10);
  WAVE_MODELS = Number.isFinite(v) && v > 0 ? v : 0;
}

let _waveKey = null, _waveVal = null;
/** Slugs inside the current wave. Everything, when no cap is set. */
export function waveSlugs(models, builtAt) {
  if (!WAVE_MODELS) return null;                 // null = "no gate", not "empty"
  const key = `${builtAt || ""}:${Object.keys(models).length}:${WAVE_MODELS}`;
  if (_waveKey === key && _waveVal) return _waveVal;
  _waveVal = new Set(Object.entries(models)
    .sort((a, b) => (b[1].n || 0) - (a[1].n || 0) || (a[0] < b[0] ? -1 : 1))
    .slice(0, WAVE_MODELS)
    .map(([slug]) => slug));
  _waveKey = key;
  return _waveVal;
}

/**
 * Year pages actually published for this model right now.
 *
 * The single source the router, the sitemap and the model page's year links all
 * call. Splitting them is how a sitemap ends up advertising 404s.
 */
export function publishedYearPages(models, slug, rec, builtAt) {
  const wave = waveSlugs(models, builtAt);
  if (wave && !wave.has(slug)) return [];
  return yearPageYears(rec);
}

/** Whether this model's depreciation page is published in the current wave. */
export function publishedDepreciation(models, slug, rec, builtAt) {
  const wave = waveSlugs(models, builtAt);
  if (wave && !wave.has(slug)) return false;
  return depreciationOk(rec);
}

export function publishedPairs(models) {
  return comparePairs(models);
}

/** Facet URLs published for this model in the current wave. */
export function publishedFacets(models, slug, rec, builtAt) {
  const wave = waveSlugs(models, builtAt);
  if (wave && !wave.has(slug)) return [];
  return facetKeys(rec);
}

export function liquidityOk(rec) {
  const lq = rec && rec.lq;
  return !!(lq && lq.n > 0 && lq.s30 != null);
}

let LIQ_WAVE = 0;
export function setLiqWave(n) {
  const v = parseInt(n, 10);
  LIQ_WAVE = Number.isFinite(v) && v > 0 ? v : 0;
}

let _liqKey = null, _liqVal = null;
export function liqWaveSlugs(models, builtAt) {
  if (!LIQ_WAVE) return null;
  const key = `${builtAt || ""}:${Object.keys(models).length}:${LIQ_WAVE}`;
  if (_liqKey === key && _liqVal) return _liqVal;
  _liqVal = new Set(Object.entries(models)
    .filter(([, r]) => liquidityOk(r))
    .sort((a, b) => ((b[1].lq && b[1].lq.n) || 0) - ((a[1].lq && a[1].lq.n) || 0)
                 || (a[0] < b[0] ? -1 : 1))
    .slice(0, LIQ_WAVE)
    .map(([slug]) => slug));
  _liqKey = key;
  return _liqVal;
}

export function publishedLiquidity(models, slug, rec, builtAt) {
  const wave = liqWaveSlugs(models, builtAt);
  if (wave && !wave.has(slug)) return false;
  return liquidityOk(rec);
}

/** Integer-year cells (bands excluded), newest first. */
export function yearCells(rec, minN = 1) {
  return (rec && Array.isArray(rec.yr) ? rec.yr : [])
    .filter(c => typeof c.y === "number" && c.fm != null && (c.n || 0) >= minN)
    .sort((a, b) => b.y - a.y);
}

function hasYearPage(c) {
  return c.pg !== undefined ? !!c.pg : (c.n || 0) >= MIN_YEAR_PAGE_N;
}

/** Years of this model that clear the year-page floor. */
export function yearPageYears(rec) {
  return yearCells(rec, 1).filter(hasYearPage).map(c => c.y);
}

/** The one cell for {slug}/{year}, or null when that year has no page. */
export function yearCell(rec, year) {
  return yearCells(rec, 1).filter(hasYearPage).find(c => c.y === year) || null;
}

/**
 * Log-linear fit of median asking price on year → annual depreciation rate.
 *
 * Log-linear, not straight-line, because depreciation is a percentage of what
 * is left: a car that loses 10%/yr loses far more euros in year two than in
 * year nine, and a straight line through euros would report the average of two
 * different things. Returns { rate, r2, cells, span, newest, oldest } or null.
 */
export function depreciationFit(rec) {
  const cs = yearCells(rec, 5).slice().sort((a, b) => a.y - b.y);
  if (cs.length < 2) return null;
  const xs = cs.map(c => c.y);
  const ys = cs.map(c => Math.log(c.fm));
  const n = xs.length;
  const mx = xs.reduce((a, b) => a + b, 0) / n;
  const my = ys.reduce((a, b) => a + b, 0) / n;
  let sxy = 0, sxx = 0;
  for (let i = 0; i < n; i++) { sxy += (xs[i] - mx) * (ys[i] - my); sxx += (xs[i] - mx) ** 2; }
  if (sxx === 0) return null;
  const b = sxy / sxx;                 // d(log price)/d(year), positive = newer costs more
  const a = my - b * mx;
  let ssRes = 0, ssTot = 0;
  for (let i = 0; i < n; i++) {
    ssRes += (ys[i] - (a + b * xs[i])) ** 2;
    ssTot += (ys[i] - my) ** 2;
  }
  const r2 = ssTot > 0 ? 1 - ssRes / ssTot : 0;
  return {
    rate: 1 - Math.exp(-b),            // fraction of value lost per year of age
    r2, cells: cs, span: xs[n - 1] - xs[0],
    oldest: cs[0], newest: cs[n - 1],
    predict: y => Math.exp(a + b * y),
  };
}

/** Whether this model earns a /depreciacao page. */
export function depreciationOk(rec) {
  const f = depreciationFit(rec);
  return !!(f
    && f.cells.length >= DEP_MIN_CELLS
    && f.span >= DEP_MIN_SPAN
    && f.r2 >= DEP_MIN_R2
    && f.rate >= DEP_MIN_RATE
    && f.rate <= DEP_MAX_RATE);
}

/** Slugs with a depreciation page, deepest sample first. */
export function depreciationSlugs(models) {
  return Object.entries(models)
    .filter(([, r]) => depreciationOk(r))
    .sort((a, b) => (b[1].n || 0) - (a[1].n || 0))
    .map(([s]) => s);
}

const COST_FLOOR_EUR = 500;
const CHEAP_MAX_AGE = 15;
const BEND_MIN_AGE = 4;
const BEND_MAX_AGE = 15;
const BEND_MIN_SIDE = 4;
const BEND_MIN_SPAN = 3;
const BEND_MIN_F = 10;
const BEND_MIN_GAP = 0.03;

function lstsq(X, y) {
  const p = X[0].length, n = X.length;
  const M = Array.from({ length: p }, () => new Array(p + 1).fill(0));
  for (let i = 0; i < p; i++) {
    for (let j = 0; j < p; j++) {
      let s = 0; for (let k = 0; k < n; k++) s += X[k][i] * X[k][j];
      M[i][j] = s;
    }
    let s = 0; for (let k = 0; k < n; k++) s += X[k][i] * y[k];
    M[i][p] = s;
  }
  for (let c = 0; c < p; c++) {
    let piv = c;
    for (let r = c + 1; r < p; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
    if (Math.abs(M[piv][c]) < 1e-10) return null;
    [M[c], M[piv]] = [M[piv], M[c]];
    for (let r = 0; r < p; r++) {
      if (r === c) continue;
      const f = M[r][c] / M[c][c];
      for (let j = c; j <= p; j++) M[r][j] -= f * M[c][j];
    }
  }
  return M.map((row, i) => row[p] / M[i][i]);
}

export function depreciationBend(pts) {
  if (!pts || pts.length < BEND_MIN_SIDE * 2) return null;
  const xs = pts.map(p => p.age), ys = pts.map(p => Math.log(p.fm));
  const n = xs.length;
  const one = lstsq(xs.map(x => [1, x]), ys);
  if (!one) return null;
  const sse1 = ys.reduce((s, v, i) => s + (v - (one[0] + one[1] * xs[i])) ** 2, 0);
  let best = null;
  for (let k = BEND_MIN_AGE; k <= BEND_MAX_AGE; k++) {
    if (k - xs[0] < BEND_MIN_SPAN || xs[n - 1] - k < BEND_MIN_SPAN) continue;
    if (xs.filter(x => x <= k).length < BEND_MIN_SIDE) continue;
    if (xs.filter(x => x >= k).length < BEND_MIN_SIDE) continue;
    const c = lstsq(xs.map(x => [1, x, Math.max(0, x - k)]), ys);
    if (!c) continue;
    const sse = ys.reduce((s, v, i) =>
      s + (v - (c[0] + c[1] * xs[i] + c[2] * Math.max(0, xs[i] - k))) ** 2, 0);
    if (!best || sse < best.sse) best = { k, sse, c };
  }
  if (!best || !(best.sse > 0) || n <= 3) return null;
  const F = (sse1 - best.sse) / (best.sse / (n - 3));
  const early = 1 - Math.exp(best.c[1]);
  const late = 1 - Math.exp(best.c[1] + best.c[2]);
  const sane = early > 0 && early < 0.5 && late >= 0 && late < 0.5;
  return {
    age: best.k, early, late, F, dir: early > late ? "slows" : "speeds",
    published: sane && F >= BEND_MIN_F && Math.abs(early - late) >= BEND_MIN_GAP,
  };
}

export function depreciationAge(rec, fit, builtAt) {
  if (!fit || !fit.cells || fit.cells.length < 2) return null;
  const built = parseInt((builtAt || "").slice(0, 4), 10);
  const ref = Math.max(Number.isFinite(built) ? built : 0, fit.newest.y);
  const pts = fit.cells
    .map(c => ({ age: ref - c.y, y: c.y, fm: c.fm, n: c.n, km: c.km }))
    .filter(p => p.age >= 0)
    .sort((a, b) => a.age - b.age);
  if (pts.length < 2) return null;
  const rate = fit.rate;
  const price = age => fit.predict(ref - age);
  const cost = age => Math.max(0, price(age) * rate);
  const minAge = pts[0].age, maxAge = pts[pts.length - 1].age;
  const at = age => (age >= minAge && age <= maxAge) ? Math.round(cost(age)) : null;
  const cand = depreciationBend(pts);
  const capAge = Math.min(Math.floor(maxAge), CHEAP_MAX_AGE);
  let cheapFrom = null;
  for (let a = Math.ceil(minAge); a <= capAge; a++) {
    if (Math.round(cost(a)) <= COST_FLOOR_EUR) {
      cheapFrom = { age: a, cost: Math.round(cost(a)), price: Math.round(price(a)) };
      break;
    }
  }
  return {
    ref, pts, minAge, maxAge, rate, price, cost, at, cheapFrom, capAge,
    costFloor: COST_FLOOR_EUR,
    halfLife: rate > 0 && rate < 1 ? Math.log(2) / -Math.log(1 - rate) : null,
    oldestCost: Math.round(cost(maxAge)),
    capCost: Math.round(cost(capAge)),
    base: { age: Math.ceil(minAge), year: ref - Math.ceil(minAge),
            price: Math.round(price(Math.ceil(minAge))) },
    bend: cand && cand.published ? cand : null,
    bendCandidate: cand,
  };
}

export function comparePool(models) {
  return Object.entries(models)
    .filter(([, r]) => r.fm > 0)
    .sort((a, b) => (b[1].n || 0) - (a[1].n || 0))
    .slice(0, CMP_POOL);
}

export function modelClass(slug) {
  return MODEL_CLASS.get(slug) || null;
}

export function comparePriceGap(ra, rb) {
  const byYear = new Map();
  for (const c of (rb && Array.isArray(rb.yr) ? rb.yr : [])) {
    if (typeof c.y === "number" && c.fm > 0) byYear.set(c.y, c);
  }
  const cells = [];
  for (const ca of (ra && Array.isArray(ra.yr) ? ra.yr : [])) {
    if (typeof ca.y !== "number" || !(ca.fm > 0)) continue;
    const cb = byYear.get(ca.y);
    if (!cb) continue;
    cells.push({
      y: ca.y, fa: ca.fm, fb: cb.fm,
      na: ca.n || 0, nb: cb.n || 0, n: Math.min(ca.n || 0, cb.n || 0),
    });
  }
  if (!cells.length) return null;
  cells.sort((p, q) => q.y - p.y);
  const ratio = weightedMedian(cells.map(c => [c.fa / c.fb, c.n || 1]));
  return {
    cells, years: cells.length, ratio,
    dist: 1 - Math.min(ratio, 1 / ratio),
    n: cells.reduce((t, c) => t + (c.n || 0), 0),
  };
}

function weightedMedian(points) {
  const s = points.slice().sort((p, q) => p[0] - q[0]);
  const total = s.reduce((t, p) => t + p[1], 0);
  let acc = 0;
  for (const p of s) {
    acc += p[1];
    if (acc * 2 >= total) return p[0];
  }
  return s[s.length - 1][0];
}

/**
 * The comparison set: deterministic, so routing and the sitemap agree.
 *
 * Three gates, and a pair has to clear all of them:
 *   cross-brand   — a Golf against a Polo is the same brand's own ladder, which
 *                   the model page's sibling chips already cover;
 *   same class    — what makes two cars substitutes, and the reason the earlier
 *                   price-only rule shipped a Mégane against a Smart Fortwo;
 *   price, at the same model year — bounds how far apart two cars of one class
 *                   may be before "{A} ou {B}" stops being one person's choice.
 *
 * Ordering inside a pair is alphabetical so /comparar/a-vs-b and /comparar/b-vs-a
 * can never both exist.
 */
export function comparePairs(models) {
  const pool = comparePool(models);
  const cand = [];
  for (let i = 0; i < pool.length; i++) {
    for (let j = i + 1; j < pool.length; j++) {
      const [sa, ra] = pool[i], [sb, rb] = pool[j];
      if (ra.b === rb.b) continue;
      const cls = modelClass(sa);
      if (!cls || cls !== modelClass(sb)) continue;
      const gap = comparePriceGap(ra, rb);
      if (!gap || gap.years < CMP_MIN_YEARS || gap.dist > CMP_PRICE_TOL) continue;
      const [x, y] = sa < sb ? [sa, sb] : [sb, sa];
      cand.push({ a: x, b: y, dist: gap.dist, depth: Math.min(ra.n || 0, rb.n || 0) });
    }
  }
  // Deepest samples first; ties broken by how close the two prices are, then by
  // slug so the set is stable across builds with identical inputs.
  cand.sort((p, q) => (q.depth - p.depth) || (p.dist - q.dist)
    || (p.a + p.b).localeCompare(q.a + q.b));
  const used = new Map();
  const out = [];
  for (const c of cand) {
    if ((used.get(c.a) || 0) >= CMP_PER_MODEL || (used.get(c.b) || 0) >= CMP_PER_MODEL) continue;
    used.set(c.a, (used.get(c.a) || 0) + 1);
    used.set(c.b, (used.get(c.b) || 0) + 1);
    out.push([c.a, c.b]);
    if (out.length >= CMP_MAX) break;
  }
  return out;
}

/**
 * Resolve "{a}-vs-{b}" back to two slugs.
 *
 * Slugs contain hyphens ("alfa-romeo-giulietta"), so a split on the first "-vs-"
 * is not enough — try every "-vs-" boundary and accept the one where BOTH halves
 * are real models. Anything not in the generated pair set is a 404: a URL a
 * scraper invented must not become a page.
 */
export function parseComparePath(rest, models, pairSet) {
  const parts = rest.split("-vs-");
  for (let i = 1; i < parts.length; i++) {
    const a = parts.slice(0, i).join("-vs-");
    const b = parts.slice(i).join("-vs-");
    if (!models[a] || !models[b]) continue;
    const [x, y] = a < b ? [a, b] : [b, a];
    if (pairSet && !pairSet.has(`${x}-vs-${y}`)) continue;
    return { a: x, b: y };
  }
  return null;
}

export function comparePairKey(a, b) {
  return a < b ? `${a}-vs-${b}` : `${b}-vs-${a}`;
}

// ── Corpus statistics ────────────────────────────────────────────────────────
// Every comparative sentence on a page ("faster than the market") needs a
// market to compare against. That is this: medians over all 243 models, from
// the same blob, computed once per build and memoised — 243 records is nothing,
// but recomputing it on every request of every page is still waste.
let _statsKey = null, _statsVal = null;

function bandRates(recs, builtAt) {
  const built = parseInt((builtAt || "").slice(0, 4), 10);
  const out = {};
  for (const [key, lo, hi] of [["depYoung", 4, 10], ["depOld", 10, 20]]) {
    const rates = [];
    for (const rec of recs) {
      const ref = Math.max(Number.isFinite(built) ? built : 0, ...yearCells(rec, 5).map(c => c.y), 0);
      const pts = yearCells(rec, 5)
        .map(c => ({ age: ref - c.y, fm: c.fm }))
        .filter(p => p.age >= lo && p.age < hi && p.fm > 0);
      if (pts.length < 3) continue;
      const c = lstsq(pts.map(p => [1, p.age]), pts.map(p => Math.log(p.fm)));
      if (!c) continue;
      const rate = 1 - Math.exp(c[1]);
      if (rate > -0.5 && rate < 0.6) rates.push(rate);
    }
    rates.sort((a, b) => a - b);
    out[key] = rates.length >= 8
      ? { rate: rates[Math.floor(rates.length / 2)], models: rates.length, from: lo, to: hi }
      : null;
  }
  return out;
}

export function corpusStats(models, builtAt) {
  const key = `${builtAt || ""}:${models ? Object.keys(models).length : 0}`;
  if (_statsKey === key && _statsVal) return _statsVal;
  const med = arr => {
    const v = arr.filter(x => x != null && isFinite(x)).sort((a, b) => a - b);
    return v.length ? v[Math.floor(v.length / 2)] : null;
  };
  const recs = Object.values(models || {});
  const rates = [];
  for (const r of recs) {
    const f = depreciationFit(r);
    if (f && f.cells.length >= 5 && f.rate > 0 && f.rate < 0.30) rates.push(f.rate);
  }
  const stats = {
    models: recs.length,
    priceMed: med(recs.map(r => r.fm)),
    kmMed: med(recs.map(r => r.kmm)),
    sellMed: med(recs.map(r => r.sd)),
    goneMed: med(recs.map(r => (r.lq && r.lq.s30 != null) ? r.lq.s30 : null)),
    // Spread = interquartile width as a share of the median. A wide spread means
    // condition/spec/history decide the price more than the badge does.
    spreadMed: med(recs.map(r => (r.fm > 0 && r.fl != null && r.fh != null)
      ? (r.fh - r.fl) / r.fm : null)),
    depMed: med(rates),
    ...bandRates(recs, builtAt),
    // Asking vs estimated fair value, where the model's estimate was publishable.
    gapMed: med(recs.map(r => (r.gm > 0 && r.fm > 0) ? r.fm / r.gm - 1 : null)),
    listings: recs.reduce((s, r) => s + (r.n || 0), 0),
  };
  _statsKey = key; _statsVal = stats;
  return stats;
}

// ── Prose generated from THIS page's numbers ─────────────────────────────────
//
// The problem this solves: with one hand-written template behind 243 pages, the
// median 7-gram overlap between any two model pages was 49% — only the figures
// differed. At 1 000+ pages that is the reason Google keeps a fraction of them.
//
// The fix is not a rewrite (nobody rewrites 565 pages, and a rewrite drifts away
// from the data the moment the data moves). It is to derive the sentences FROM
// the figures: each rule below reads this model's own numbers, compares them to
// the corpus, and only speaks when it has something true and specific to say.
// A model that depreciates at the market rate simply does not get that sentence.
//
// Every rule states a consequence for the reader, because a number without a
// consequence is still boilerplate — just boilerplate with a variable in it.
export function modelInsights(rec, stats) {
  const out = [];
  const pct = x => Math.round(Math.abs(x) * 100);
  const B = escapeHtml(rec.b), M = escapeHtml(rec.m);

  // 1. Depreciation against the market.
  const fit = depreciationFit(rec);
  if (fit && fit.cells.length >= 5 && stats.depMed && fit.rate > 0 && fit.rate < 0.30) {
    const mine = fit.rate, mkt = stats.depMed;
    const rel = (mine - mkt) / mkt;
    if (rel > 0.18) {
      out.push(`Perde valor depressa: cerca de <b>${pct(mine)}% por cada ano de idade</b>, contra ${pct(mkt)}% na mediana do mercado. Comprar um exemplar dois ou três anos mais velho poupa mais aqui do que na maioria dos modelos.`);
    } else if (rel < -0.18) {
      out.push(`Segura bem o valor: cerca de <b>${pct(mine)}% por ano de idade</b>, contra ${pct(mkt)}% na mediana do mercado. Pagar por um ano mais recente custa menos do que noutros modelos — e revender também dói menos.`);
    } else {
      out.push(`Desvaloriza ao ritmo do mercado: cerca de <b>${pct(mine)}% por ano de idade</b> (mediana do mercado: ${pct(mkt)}%).`);
    }
  }

  // 2. Time to sell — the seller's question, and the buyer's leverage.
  if (rec.lq && rec.lq.s30 != null && stats.goneMed) {
    const mine = rec.lq.s30, mkt = stats.goneMed;
    const tail = `${pct(mine)} em cada 100 saem do OLX no primeiro mês (mercado: ${pct(mkt)}), em ${fmtNum(rec.lq.n)} anúncios acompanhados`;
    if (mine >= mkt * 1.12) {
      out.push(`Sai depressa: <b>${tail}</b>. A anunciar, tens pouca pressão para descer o preço; a comprar, os bons exemplares desaparecem em dias.`);
    } else if (mine <= mkt * 0.88) {
      out.push(`Demora a sair: <b>${tail}</b>. Quem vende costuma ter de ceder no preço, e quem compra tem margem para negociar.`);
    } else {
      out.push(`Tempo até sair em linha com o mercado: <b>${tail}</b>.`);
    }
  } else if (rec.sd != null && rec.sn != null && stats.sellMed) {
    const d = rec.sd, mkt = stats.sellMed;
    if (d <= mkt * 0.75) {
      out.push(`Vende rápido: mediana de <b>${d} dias</b> no OLX (mercado: ${mkt} dias), em ${rec.sn} vendas observadas. A anunciar, tens pouca pressão para descer o preço; a comprar, os bons exemplares desaparecem em dias.`);
    } else if (d >= mkt * 1.25) {
      out.push(`Demora a sair: mediana de <b>${d} dias</b> no OLX contra ${mkt} do mercado, em ${rec.sn} vendas observadas. Quem vende costuma ter de ceder no preço, e quem compra tem margem para negociar.`);
    } else {
      out.push(`Tempo até vender em linha com o mercado: mediana de <b>${d} dias</b> (mercado: ${mkt}), em ${rec.sn} vendas observadas.`);
    }
  }

  // 3. Price dispersion — how much the individual car matters versus the badge.
  if (rec.fm > 0 && rec.fl != null && rec.fh != null && stats.spreadMed) {
    const sp = (rec.fh - rec.fl) / rec.fm, mkt = stats.spreadMed;
    if (sp > mkt * 1.25) {
      out.push(`Os preços estão muito dispersos: metade dos anúncios ocupa uma faixa de ${pct(sp)}% em torno da mediana, contra ${pct(mkt)}% no mercado. Aqui o estado, os quilómetros e a versão pesam mais do que o modelo — a mediana sozinha diz pouco sobre o carro à tua frente.`);
    } else if (sp < mkt * 0.75) {
      out.push(`Os preços são apertados: metade dos anúncios cabe em ${pct(sp)}% em torno da mediana, contra ${pct(mkt)}% no mercado. Um anúncio muito fora desta faixa tem quase sempre uma razão — versão rara, quilometragem extrema ou algo por dizer.`);
    }
  }

  // 4. Mileage on offer.
  if (rec.kmm != null && stats.kmMed) {
    const rel = (rec.kmm - stats.kmMed) / stats.kmMed;
    if (rel > 0.20) {
      out.push(`A quilometragem mediana à venda é alta: <b>${fmtKm(rec.kmm)}</b> contra ${fmtKm(stats.kmMed)} no mercado. Um exemplar abaixo disso justifica pedir mais do que a mediana.`);
    } else if (rel < -0.20) {
      out.push(`A quilometragem mediana à venda é baixa: <b>${fmtKm(rec.kmm)}</b> contra ${fmtKm(stats.kmMed)} no mercado, por isso a mediana de preço aqui é a de carros pouco rodados.`);
    }
  }

  // 5. Fuel mix — only when it is genuinely lopsided, i.e. when it constrains
  //    what you can actually find.
  if (Array.isArray(rec.fu) && rec.fu.length && rec.fu[0][1] >= 0.80) {
    const [f, share] = rec.fu[0];
    out.push(`O mercado deste modelo é praticamente todo ${escapeHtml(String(f).toLowerCase())}: <b>${pct(share)}% dos anúncios</b>. As outras motorizações aparecem pouco, e quando aparecem o preço não segue esta mediana.`);
  }

  // 6. The steepest year-over-year step in the table — where the money is.
  const cs = yearCells(rec, MIN_YEAR_PAGE_N).slice().sort((a, b) => a.y - b.y);
  let best = null;
  for (let i = 1; i < cs.length; i++) {
    if (cs[i].y !== cs[i - 1].y + 1) continue;
    const step = (cs[i].fm - cs[i - 1].fm) / cs[i].fm;
    if (step > 0.12 && (!best || step > best.step)) best = { step, lo: cs[i - 1], hi: cs[i] };
  }
  if (best) {
    out.push(`O maior degrau está entre <b>${best.lo.y} e ${best.hi.y}</b>: ${pct(best.step)}% de diferença na mediana (${fmtEur(best.lo.fm)} contra ${fmtEur(best.hi.fm)}). Se o ano não é decisivo para ti, é aqui que se poupa mais de uma vez só.`);
  }

  // 7. Where the market sits against our own fair-value estimate.
  if (rec.gm > 0 && rec.fm > 0 && stats.gapMed != null) {
    const gap = rec.fm / rec.gm - 1;
    if (gap > (stats.gapMed + 0.10)) {
      out.push(`O que se pede está acima do valor justo que estimamos (${fmtEur(rec.gm)}) mais do que é habitual neste mercado. Trata a mediana como preço de partida, não como preço final.`);
    } else if (gap < (stats.gapMed - 0.10)) {
      out.push(`O que se pede está abaixo do valor justo que estimamos (${fmtEur(rec.gm)}) mais do que é habitual — sinal de oferta a mais ou de procura fraca neste momento.`);
    }
  }

  return out;
}

// ── Small shared blocks ──────────────────────────────────────────────────────

/**
 * The provenance line, identical in shape on every data page.
 *
 * Same words, same order, same data-* attributes everywhere, so a model reading
 * the page can lift "how many, how fresh, of what" without parsing prose — and a
 * human gets the caveat in the same place every time.
 */
export function provenance({ n, builtAt, measure = "Preço pedido em anúncios ativos (mediana e P25-P75)",
                             unit = "anúncios ativos", measureId = "asking-price-median",
                             source = "OLX Portugal", extra = "" }) {
  const day = (builtAt || "").slice(0, 10);
  return `<p class="mono fc-prov" data-sample="${n != null ? n : ""}" data-updated="${escapeHtml(day)}" data-measure="${escapeHtml(measureId)}" data-source="${escapeHtml(source)}">`
    + `Amostra: ${n != null ? fmtNum(n) + " " + escapeHtml(unit) : "n/d"} · Recolhido até: ${day || "n/d"} · Medida: ${escapeHtml(measure)} · Fonte: ${escapeHtml(source)}${extra ? " · " + extra : ""}`
    + `</p>`;
}

/** Visible breadcrumb; mirrors whatever BreadcrumbList the page emits. */
export function crumbs(items) {
  const inner = items.map((it, i) => {
    const last = i === items.length - 1;
    return last
      ? `<span style="color:#16181D;">${escapeHtml(it.name)}</span>`
      : `<a href="${escapeHtml(it.href)}" style="color:#8A8F98;">${escapeHtml(it.name)}</a>`;
  }).join(" › ");
  return `<nav class="section fc-crumbs" aria-label="Breadcrumb">${inner}</nav>`;
}

export function breadcrumbLd(host, items) {
  return {
    "@type": "BreadcrumbList",
    "itemListElement": items.map((it, i) => ({
      "@type": "ListItem", "position": i + 1, "name": it.name,
      ...(it.href ? { "item": `https://${host}${it.href}` } : {}),
    })),
  };
}

export function faqLd(pairs) {
  return {
    "@type": "FAQPage",
    "mainEntity": pairs.map(([q, a]) => ({
      "@type": "Question", "name": q,
      "acceptedAnswer": { "@type": "Answer", "text": a },
    })),
  };
}

/**
 * Median-price-by-year line chart, inline SVG.
 *
 * Inline because it must render with the HTML (no JS, no request) — this chart
 * IS the page's argument, and a chart that paints late is a chart nobody sees.
 * viewBox + preserveAspectRatio make it fluid without a media query.
 */
export function priceChart(cells, { w = 640, h = 220, color = "#177A47" } = {}) {
  const pts = cells.slice().sort((a, b) => a.y - b.y);
  if (pts.length < 2) return "";
  const padL = 16, padR = 14, padT = 20, padB = 30;
  const xs = pts.map(p => p.y);
  const ys = pts.map(p => p.fm);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const y1 = Math.max(...ys), y0 = 0;
  const X = y => padL + ((y - x0) / Math.max(1, x1 - x0)) * (w - padL - padR);
  const Y = v => padT + (1 - (v - y0) / Math.max(1, y1 - y0)) * (h - padT - padB);
  const line = pts.map((p, i) => `${i ? "L" : "M"}${X(p.y).toFixed(1)},${Y(p.fm).toFixed(1)}`).join("");
  const area = `${line}L${X(x1).toFixed(1)},${Y(0).toFixed(1)}L${X(x0).toFixed(1)},${Y(0).toFixed(1)}Z`;
  const dots = pts.map(p =>
    `<circle cx="${X(p.y).toFixed(1)}" cy="${Y(p.fm).toFixed(1)}" r="3" fill="${color}"><title>${p.y}: ${fmtEur(p.fm)} (${p.n} anúncios)</title></circle>`).join("");
  // At most six year labels, otherwise they collide on a phone.
  const step = Math.max(1, Math.ceil(pts.length / 6));
  const xlab = pts.filter((_, i) => i % step === 0 || i === pts.length - 1).map(p => {
    // The end labels are anchored inwards or half of them falls outside the viewBox.
    const anchor = p.y === x0 ? "start" : p.y === x1 ? "end" : "middle";
    return `<text x="${X(p.y).toFixed(1)}" y="${h - 9}" text-anchor="${anchor}" class="c-ax">${p.y}</text>`;
  }).join("");
  const ticks = [0, 0.5, 1].map(f => {
    const v = y1 * f;
    return `<line x1="${padL}" x2="${w - padR}" y1="${Y(v).toFixed(1)}" y2="${Y(v).toFixed(1)}" class="c-grid"/>`
      + `<text x="${padL + 2}" y="${(Y(v) - 5).toFixed(1)}" text-anchor="start" class="c-ax">${fmtEur(Math.round(v))}</text>`;
  }).join("");
  return `<svg class="fc-chart" viewBox="0 0 ${w} ${h}" role="img"
    aria-label="Preço mediano pedido por ano de fabrico">${ticks}
    <path d="${area}" fill="${color}" opacity="0.10"/>
    <path d="${line}" fill="none" stroke="${color}" stroke-width="2.2" stroke-linejoin="round"/>
    ${dots}${xlab}</svg>`;
}

export function depreciationChart(av, { w = 640, h = 240, color = "#177A47" } = {}) {
  if (!av || av.pts.length < 2) return "";
  const padL = 16, padR = 14, padT = 38, padB = 38;
  const a0 = av.minAge, a1 = av.maxAge;
  const top = Math.max(...av.pts.map(p => p.fm), av.price(a0));
  const X = a => padL + ((a - a0) / Math.max(1, a1 - a0)) * (w - padL - padR);
  const Y = v => padT + (1 - v / Math.max(1, top)) * (h - padT - padB);
  let curve = "";
  for (let a = a0; a <= a1 + 1e-9; a += Math.max(0.25, (a1 - a0) / 120)) {
    curve += `${curve ? "L" : "M"}${X(a).toFixed(1)},${Y(av.price(a)).toFixed(1)}`;
  }
  curve += `L${X(a1).toFixed(1)},${Y(av.price(a1)).toFixed(1)}`;
  const area = `${curve}L${X(a1).toFixed(1)},${Y(0).toFixed(1)}L${X(a0).toFixed(1)},${Y(0).toFixed(1)}Z`;
  const dots = av.pts.map(p =>
    `<circle cx="${X(p.age).toFixed(1)}" cy="${Y(p.fm).toFixed(1)}" r="3" fill="${color}">`
    + `<title>${p.age} anos (${p.y}): ${fmtEur(p.fm)} · ${p.n} anúncios</title></circle>`).join("");
  const ticks = [0, 0.5, 1].map(f => {
    const v = top * f;
    return `<line x1="${padL}" x2="${w - padR}" y1="${Y(v).toFixed(1)}" y2="${Y(v).toFixed(1)}" class="c-grid"/>`
      + `<text x="${padL + 2}" y="${(Y(v) - 5).toFixed(1)}" text-anchor="start" class="c-ax">${fmtEur(Math.round(v))}</text>`;
  }).join("");
  const step = Math.max(1, Math.ceil((a1 - a0) / 6));
  let xlab = "";
  for (let a = Math.ceil(a0); a <= a1; a += step) {
    const anchor = a - a0 < step / 2 ? "start" : "middle";
    xlab += `<text x="${X(a).toFixed(1)}" y="${h - 17}" text-anchor="${anchor}" class="c-ax">${a}</text>`;
  }
  xlab += `<text x="${w - padR}" y="${h - 5}" text-anchor="end" class="c-ax">anos de idade</text>`;
  const mark = (age, label, row) => {
    if (age == null || age < a0 || age > a1) return "";
    const x = X(age), lab = x > w - 90 ? "end" : x < padL + 70 ? "start" : "middle";
    return `<line x1="${x.toFixed(1)}" x2="${x.toFixed(1)}" y1="${padT - 6}" y2="${Y(0).toFixed(1)}" class="c-mark"/>`
      + `<text x="${x.toFixed(1)}" y="${padT - (row ? 10 : 24)}" text-anchor="${lab}" class="c-marklab">${label}</text>`;
  };
  const marks = (av.bend ? mark(av.bend.age, `quebra aos ${av.bend.age} anos`, 0) : "")
    + (av.cheapFrom ? mark(av.cheapFrom.age, `${fmtEur(av.costFloor)}/ano aos ${av.cheapFrom.age}`, 1) : "");
  return `<svg class="fc-chart" viewBox="0 0 ${w} ${h}" role="img"
    aria-label="Preço mediano pedido por idade do carro, com a curva ajustada">${ticks}
    <path d="${area}" fill="${color}" opacity="0.10"/>
    <path d="${curve}" fill="none" stroke="${color}" stroke-width="2.2" stroke-linejoin="round"/>
    ${dots}${marks}${xlab}</svg>`;
}

// ── Comparing two adjacent year cells ────────────────────────────────────────
//
// Two neighbouring years are two samples of ~20 cars on sale today, not the same
// car measured a year apart. Their medians can sit far apart while the asking
// ranges behind them are one cloud: Golf 2012 asks 7500-14 850 and Golf 2013 asks
// 9000-17 990, so "+82% for one year of age" is a step the page cannot see, and
// the GBM agrees with the medians because it is pricing the same skewed samples.
//
// So the percentage is only phrased as a step when the two P25-P75 ranges are
// mostly disjoint. Otherwise the page states both medians, both ranges and both
// sample sizes, and says what the distance actually measures: which cars happen
// to be for sale in each year. Same fact, without the claim it cannot carry.
const GAP_MAX_OVERLAP = 0.5;

export function yearGap(a, b) {
  if (!a || !b || !(a.fm > 0) || !(b.fm > 0)) return null;
  let overlap = null;
  if (a.fl != null && a.fh != null && b.fl != null && b.fh != null) {
    const lo = Math.max(a.fl, b.fl), hi = Math.min(a.fh, b.fh);
    const span = Math.min(a.fh - a.fl, b.fh - b.fl);
    overlap = span > 0 ? Math.max(0, hi - lo) / span : (hi >= lo ? 1 : 0);
  }
  return {
    pct: (b.fm - a.fm) / a.fm,
    overlap,
    separated: overlap != null && overlap <= GAP_MAX_OVERLAP,
    dkm: (a.km != null && b.km != null) ? b.km - a.km : null,
  };
}

// ═══ /preco/{slug}/{ano} ═════════════════════════════════════════════════════
//
// The model page answers "quanto vale um Golf" for eighteen model years at once,
// which means it competes with itself on every one of them and wins none. This
// page answers exactly one: "quanto vale um Golf de 2012".
//
// It only exists where the year has 10+ active listings (MIN_YEAR_PAGE_N).
// Thinner years stay as a row in the parent table — visible, linked, honest, and
// not a URL asking to be indexed on four data points.
export function renderYearPage({ rec, slug, year, cell, neighbours, liveDeals, dealsNear, pageYears,
                                 stats, host, depositCount, builtAt, historyUrl = null, hasVender = false }) {
  const B = escapeHtml(rec.b), M = escapeHtml(rec.m);
  const FM = fmtEur(cell.fm), FL = fmtEur(cell.fl), FH = fmtEur(cell.fh);
  const canonical = `https://${host}/preco/${slug}/${year}`;
  const hasG = cell.gm != null && cell.gl != null && cell.gh != null;
  const refYear = parseInt((builtAt || "").slice(0, 4), 10) || null;
  const age = refYear ? refYear - year : null;
  const share = rec.n ? Math.round((cell.n / rec.n) * 100) : null;
  const win = cell.w ? Math.max(1, Math.round(cell.w / 30)) : null;
  const sample = win ? `anúncios dos últimos ${win} meses` : "anúncios ativos";
  const sampleNote = win ? " São preços pedidos em anúncios ativos e já fechados, não preços de venda." : "";

  let pin = 50;
  if (cell.fh > cell.fl) pin = Math.max(6, Math.min(94, Math.round((cell.fm - cell.fl) / (cell.fh - cell.fl) * 100)));

  const hero = `
    <div class="side-card" style="max-width:680px;margin:0 auto;">
      <div class="eyebrow" style="margin-bottom:14px;"><span class="e-dot"></span><span class="mono">${B.toUpperCase()} ${M.toUpperCase()} · ${year} · OLX PORTUGAL</span></div>
      <h1 class="fc-h1">Quanto vale um ${B} ${M} de ${year}?</h1>
      <p class="lede" style="font-size:16px;margin:0 0 20px;">Nos <b>${cell.n} ${sample}</b> de ${B} ${M} do ano ${year} no OLX${win && cell.na ? ` (${cell.na} ainda ativos)` : ""}, o preço pedido mediano é <b>${FM}</b>${cell.km != null ? `, com ${fmtKm(cell.km)} de quilometragem mediana` : ""}. É o que o mercado pede hoje por este ano concreto, não uma avaliação da tua viatura.</p>
      <div class="side-prices">
        <div><div class="cap">Preço mediano (pedido) · ${year}${win ? ` · últimos ${win} meses` : ""}</div><div class="big">${FM}</div></div>
        <div class="side-fair"><div class="cap">${cell.n} anúncios${age != null ? ` · ${age} ano${age === 1 ? "" : "s"}` : ""}</div><div class="v">${cell.km != null ? fmtKm(cell.km) : "—"}</div></div>
      </div>
      <div style="margin-top:16px;">
        <div class="gauge-head"><span>${FL}</span><span>intervalo típico (50% dos anúncios de ${year})</span><span>${FH}</span></div>
        <div class="gauge-track"><div class="gauge-pin" style="left:${pin}%;"></div></div>
      </div>
      ${hasG ? `<div style="margin-top:16px;padding-top:14px;border-top:1px solid #EFECE6;"><div class="cap">Valor justo estimado para ${year}</div><div class="mono" style="font-weight:700;font-size:20px;color:#177A47;">${fmtEur(cell.gm)}</div><div class="mono" style="font-size:12px;color:#5B606B;">intervalo ${fmtEur(cell.gl)} – ${fmtEur(cell.gh)}</div></div>` : ""}
      ${provenance({ n: cell.n, builtAt, measure: `Preço pedido, ${B} ${M} do ano ${year} (mediana e P25-P75)${win ? `, anúncios ativos e já fechados dos últimos ${win} meses` : ""}`, unit: win ? "anúncios (ativos e fechados)" : "anúncios ativos" })}
    </div>`;

  // The one comparison a buyer on this page is actually making: is the next year
  // up worth its premium, and how much does the year below save?
  const stepBlock = (() => {
    const bits = [];
    const older = neighbours.older, newer = neighbours.newer;
    const rng = c => `${fmtEur(c.fl)}–${fmtEur(c.fh)}`;
    if (newer) {
      const g = yearGap(cell, newer);
      const pct = Math.round(g.pct * 100);
      const href = pageYears.includes(newer.y) ? `/preco/${slug}/${newer.y}` : `/preco/${slug}`;
      const link = `<a href="${href}">${B} ${M} de ${newer.y}</a>`;
      const km = g.dkm != null && g.dkm !== 0
        ? `, com ${fmtKm(Math.abs(g.dkm))} ${g.dkm < 0 ? "a menos" : "a mais"} no conta-quilómetros`
        : "";
      bits.push(g.separated
        ? `<li>Um ${link} pede em mediana ${fmtEur(newer.fm)}, <b>${pct >= 0 ? "+" : ""}${pct}%</b> face a ${year}${km}${pct <= 0 ? " — mais recente e ainda assim não mais caro" : ""}.</li>`
        : `<li>Um ${link} pede em mediana ${fmtEur(newer.fm)} contra ${FM} em ${year}, mas as faixas de preço dos dois anos sobrepõem-se (${year}: ${rng(cell)}; ${newer.y}: ${rng(newer)})${km}. Com ${cell.n === newer.n ? `${cell.n} anúncios de cada lado` : `${cell.n} e ${newer.n} anúncios de cada lado`}, essa distância mede sobretudo que carros estão à venda em cada ano, não quanto vale um ano de matrícula.</li>`);
    }
    if (older) {
      const g = yearGap(older, cell);
      const save = Math.round((cell.fm - older.fm) / cell.fm * 100);
      const href = pageYears.includes(older.y) ? `/preco/${slug}/${older.y}` : `/preco/${slug}`;
      const link = `<a href="${href}">${older.y}</a>`;
      const km = older.km != null && cell.km != null && older.km !== cell.km
        ? `, com ${fmtKm(Math.abs(older.km - cell.km))} ${older.km > cell.km ? "a mais" : "a menos"} no conta-quilómetros`
        : "";
      bits.push(save <= 0
        ? `<li>Descer para ${link} não poupa nada: a mediana desse ano é ${fmtEur(older.fm)}, ${save === 0 ? `a mesma de ${year}` : `acima da de ${year}`}${km}. Entre estes dois anos quem manda no preço é o carro, não a matrícula.</li>`
        : g.separated
          ? `<li>Descer para ${link} poupa cerca de <b>${save}%</b> (mediana ${fmtEur(older.fm)})${km}.</li>`
          : `<li>Descer para ${link} baixa a mediana em <b>${save}%</b> (${fmtEur(older.fm)})${km}, mas as faixas dos dois anos sobrepõem-se (${older.y}: ${rng(older)}; ${year}: ${rng(cell)}): há exemplares de ${older.y} a pedir mais do que boa parte dos de ${year}.</li>`);
    }
    // The honest answer to the heading. Two adjacent cells cannot carry a
    // one-year step on most models (their P25-P75 overlap), but the model's own
    // curve can: it is the same question measured over the whole series instead
    // of over two samples of ~20 cars. Published on the same terms as the
    // depreciation page itself — 8+ year cells, 8+ years of span, R2 >= 0.55 —
    // so a page never cites a rate the site would not publish on its own.
    const fit = depreciationOk(rec) ? depreciationFit(rec) : null;
    if (fit) {
      const r = Math.round(fit.rate * 100);
      const series = `${fit.cells.length} anos com amostra, de ${fit.oldest.y} a ${fit.newest.y}`;
      const href = `<a href="/depreciacao/${slug}">ritmo medido em toda a série</a>`;
      const gapNewer = newer ? yearGap(cell, newer) : null;
      bits.push(gapNewer && !gapNewer.separated
        ? `<li>Ao ${href} do ${B} ${M} (${series}), um ano de idade vale cerca de <b>${r}%</b>. É a melhor resposta que os dados dão à pergunta acima: entre dois anos concretos a diferença de preço fica dominada por quem pôs o carro à venda, e só a curva inteira mede o ano.</li>`
        : `<li>Ao ${href} do ${B} ${M} (${series}), um ano de idade vale cerca de <b>${r}%</b>${gapNewer ? `, contra os ${Math.round(gapNewer.pct * 100)}% entre ${year} e ${newer.y}` : ""}.</li>`);
    }
    if (share != null) {
      bits.push(`<li>O ano ${year} representa <b>${share}%</b> de todos os ${B} ${M} à venda agora (${cell.n} de ${rec.n} anúncios)${share >= 15 ? " — é dos anos com mais escolha" : share <= 5 ? " — há pouca oferta, por isso conta com menos margem para escolher" : ""}.</li>`);
    }
    return bits.length ? `<section class="section fc-wrap"><h2 class="fc-h2">Vale a pena pagar por um ano mais recente?</h2><ul class="fc-insights">${bits.join("")}</ul></section>` : "";
  })();

  // Neighbouring years in a table, each linked where it has its own page.
  const rows = neighbours.window.map(c => {
    const isSelf = c.y === year;
    const label = pageYears.includes(c.y) && !isSelf
      ? `<a href="/preco/${slug}/${c.y}" style="color:#177A47;font-weight:600;">${c.y}</a>`
      : `${isSelf ? `<b>${c.y}</b>` : c.y}`;
    return `<tr${isSelf ? ' style="background:#F6FBF8;"' : ""}>
      <td>${label}</td><td>${c.n}</td><td>${fmtEur(c.fm)}</td>
      <td class="mut">${fmtEur(c.fl)} – ${fmtEur(c.fh)}</td>
      <td class="mut">${c.km != null ? fmtKm(c.km) : "—"}</td></tr>`;
  }).join("");
  const table = rows ? `
    <section class="section fc-wrap">
      <h2 class="fc-h2">${B} ${M}: anos vizinhos</h2>
      <div class="fc-scroll"><table class="fc-tbl">
        <thead><tr><th>Ano</th><th>Anúncios</th><th>Mediano (pedido)</th><th>P25–P75</th><th>Km mediano</th></tr></thead>
        <tbody>${rows}</tbody></table></div>
      <p class="fc-p" style="margin-top:12px;"><a href="/preco/${slug}">Ver todos os anos de ${B} ${M}&nbsp;→</a></p>
    </section>` : "";

  const yearNav = pageYears.length > 1 ? `
    <section class="section fc-wrap">
      <div class="sec-label" style="margin-bottom:10px;">OUTROS ANOS COM DADOS SUFICIENTES</div>
      <div class="fc-yearlinks">${pageYears.map(y =>
        y === year ? `<a class="on" href="/preco/${slug}/${y}">${y}</a>` : `<a href="/preco/${slug}/${y}">${y}</a>`).join("")}</div>
    </section>` : "";

  const deals = (liveDeals || []).length ? `
    <section class="section fc-wide">
      <div class="sec-label">${dealsNear ? `${B} ${M} DE ANOS PRÓXIMOS ABAIXO DO PREÇO JUSTO AGORA` : `${B} ${M} DE ${year} ABAIXO DO PREÇO JUSTO AGORA`}</div>
      ${dealsNear ? `<p class="fc-p" style="margin:0 0 12px;">Nenhum de ${year} neste momento. Estes são ${B} ${M} de anos próximos cujo preço pedido está abaixo do valor justo que estimamos.</p>` : ""}
      <div class="grid">${liveDeals.slice(0, 3).map(d => {
        const p = present(d);
        return `<a class="tile" href="/car?olx_id=${encodeURIComponent(d.olx_id)}" style="max-width:none;">
          <div class="thumb">${thumbBlock(p, 168, 28)}${gradeChip(p)}</div>
          <div class="tbody"><div class="tile-title">${escapeHtml(p.name)}</div>
          <div class="tile-sub">${p.subHtml}</div>
          <div class="price-row"><div class="price">${p.priceStr}</div><div class="fair-strike">${p.fairStr}</div></div>
          </div></a>`;
      }).join("")}</div>
    </section>` : `
    <section class="section fc-wrap">
      <p class="fc-p">Sem ${B} ${M} abaixo do preço justo neste momento, nem de ${year} nem dos anos à volta. <a href="/avaliar">Avalia um anúncio concreto</a> ou <a href="/mercado">vê o mercado completo</a>.</p>
    </section>`;

  const cta = `
    <section class="section fc-wide">
      <div class="cta-banner">
        <div style="flex:1 1 360px;">
          <h2>Tens um ${B} ${M} de ${year}?</h2>
          <p>Metade dos ${cell.n} anúncios de ${year} pede entre ${FL} e ${FH}. Cola o link do teu e dizemos onde cai — quilómetros, versão e estado incluídos.</p>
        </div>
        <a class="btn-bright" href="/avaliar?modelo=${encodeURIComponent(slug)}&ano=${year}">Avaliar o meu ${year}&nbsp;&nbsp;→</a>
      </div>
    </section>`;

  const links = `
    <section class="section fc-wrap" style="padding-bottom:70px;">
      <p class="fc-p"><a href="/preco/${slug}">Preços de ${B} ${M} por ano</a>${depreciationOk(rec) ? ` · <a href="/depreciacao/${slug}">Curva de desvalorização</a>` : ""} · <a href="/metodologia">Como calculamos</a> · <a href="/precos">Todos os modelos</a></p>
    </section>`;

  const histBlock = historyUrl ? `<section class="section fc-wrap" style="padding-top:0;">${historyCheckBlock({
    url: historyUrl, from: "ano",
    title: `Antes de pagar ${FM} por um ${rec.b} ${rec.m} de ${year}, verifica o histórico`,
    reasons: [
      cell.km != null ? `Um ${B} ${M} de ${year} anda em mediana com ${fmtKm(cell.km)}: bem abaixo disso, confirma o conta-quilómetros nas inspeções anteriores.` : "",
      "Sinistros, número de donos e importação não aparecem no anúncio; aparecem no relatório pela matrícula.",
    ],
  })}</section>` : "";

  const sellHref = hasVender ? `/vender/${slug}#vender` : `/avaliar?modelo=${encodeURIComponent(slug)}&ano=${year}#vender`;
  const sellBlock = `
    <section class="section" style="padding:18px 22px 0;max-width:680px;margin:0 auto;">
      <div class="exclusive" style="background:#F4F6FB;border:1px solid #D9E0F0;align-items:flex-start;">
        <span style="font-size:15px;">🏷️</span>
        <span class="x" style="color:#3A3F47;"><b style="color:#16181D;">Vais vender o teu ${B} ${M} de ${year}?</b> Metade dos anúncios deste ano pede entre ${FL} e ${FH}${rec.sd != null ? `, e um ${B} ${M} sai do OLX em ~${rec.sd} dias` : ""}. Pede propostas de compra a compradores profissionais, sem compromisso. <a href="${sellHref}" style="color:#177A47;font-weight:600;">Receber propostas&nbsp;→</a></span>
      </div>
    </section>`;
  const sellForm = `
    <section class="section" style="padding:0 22px;max-width:680px;margin:0 auto;">
      ${leadFormBlock({ slug, name: `${rec.b} ${rec.m}`, year, median: cell.fm })}
    </section>`;

  const body = crumbs([
    { name: "Início", href: "/" }, { name: "Preços", href: "/precos" },
    { name: `${rec.b} ${rec.m}`, href: `/preco/${slug}` }, { name: String(year) },
  ]) + `<div style="padding-top:14px;">${hero}</div>${sellBlock}${histBlock}${stepBlock}${table}${yearNav}${deals}${cta}${sellForm}${links}`;

  const faqs = [[
    `Quanto vale um ${rec.b} ${rec.m} de ${year} em Portugal?`,
    `Nos ${cell.n} ${sample} de ${rec.b} ${rec.m} do ano ${year} no OLX Portugal, o preço pedido mediano é ${FM}, com metade dos anúncios entre ${FL} e ${FH}.${sampleNote} São preços pedidos em anúncios ativos, não preços de venda fechados.`,
  ]];
  if (cell.km != null) faqs.push([
    `Qual é a quilometragem típica de um ${rec.b} ${rec.m} de ${year}?`,
    `A quilometragem mediana dos ${rec.b} ${rec.m} de ${year} à venda é ${fmtKm(cell.km)}. Um exemplar bastante abaixo desse valor justifica um preço acima da mediana do ano, e vice-versa.`,
  ]);
  const depFit = depreciationOk(rec) ? depreciationFit(rec) : null;
  if (neighbours.newer) {
    const nb = neighbours.newer, g = yearGap(cell, nb);
    const d = Math.round(g.pct * 100);
    faqs.push([
      `Compensa comprar um ${rec.b} ${rec.m} de ${nb.y} em vez de ${year}?`,
      g.separated
        ? `Um ${rec.b} ${rec.m} de ${nb.y} pede em mediana ${fmtEur(nb.fm)}, ou seja ${d >= 0 ? "+" : ""}${d}% face aos ${FM} de ${year}. A diferença compensa se a quilometragem e o estado acompanharem; caso contrário estás a pagar pelo ano na matrícula.`
        : `Os ${rec.b} ${rec.m} de ${nb.y} pedem em mediana ${fmtEur(nb.fm)} e os de ${year} ${FM}, mas as faixas de preço sobrepõem-se (${year}: ${fmtEur(cell.fl)} a ${fmtEur(cell.fh)}; ${nb.y}: ${fmtEur(nb.fl)} a ${fmtEur(nb.fh)}), com ${cell.n} e ${nb.n} anúncios ativos. Entre estes dois anos a diferença de preço vem sobretudo de que carros estão à venda em cada um, por isso a escolha decide-se no exemplar concreto (quilómetros, versão, estado) e não no ano.${depFit ? ` Medido em toda a série do modelo (${depFit.cells.length} anos com amostra), um ano de idade vale cerca de ${Math.round(depFit.rate * 100)}%.` : ""}`,
    ]);
  }
  if (hasG) faqs.push([
    `Um ${rec.b} ${rec.m} de ${year} a ${FM} é caro?`,
    `Para o ano ${year} estimamos um valor justo de ${fmtEur(cell.gm)} (intervalo ${fmtEur(cell.gl)} a ${fmtEur(cell.gh)}) com base nas características típicas dos exemplares deste ano. O preço pedido mediano é ${FM}. Um carro concreto pode estar acima ou abaixo consoante quilómetros, versão e histórico.`,
  ]);

  const jsonLd = {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "Dataset",
        "license": licenseUrl(host),
        "name": `Preços de ${rec.b} ${rec.m} de ${year} em Portugal`,
        "description": `Mediana e intervalo interquartil dos preços pedidos em ${cell.n} ${sample} de ${rec.b} ${rec.m} do ano ${year} no OLX Portugal.`,
        "creator": { "@type": "Organization", "name": "Carsbuyer", "url": `https://${host}/` },
        "isAccessibleForFree": true,
        "temporalCoverage": String(year),
        "variableMeasured": hasG ? ["Preço pedido (EUR)", "Valor justo estimado (EUR)"] : "Preço pedido (EUR)",
        "dateModified": builtAt || undefined,
        "url": canonical,
        "distribution": {
          "@type": "DataDownload",
          "encodingFormat": "application/json",
          "contentUrl": `${canonical}.json`,
        },
      },
      // AggregateOffer, not Product: we are describing the ASK across a market of
      // listings we do not own or sell. itemOffered names the car; no seller is
      // claimed, no availability, no single price — which is exactly what an
      // aggregate of third-party asking prices is.
      {
        "@type": "AggregateOffer",
        "priceCurrency": "EUR",
        "lowPrice": cell.fl, "highPrice": cell.fh, "offerCount": cell.n,
        "url": canonical,
        "itemOffered": {
          "@type": "Car", "name": `${rec.b} ${rec.m} ${year}`,
          "brand": { "@type": "Brand", "name": rec.b },
          "model": rec.m, "vehicleModelDate": String(year),
          ...(cell.km != null ? {
            "mileageFromOdometer": { "@type": "QuantitativeValue", "value": cell.km, "unitCode": "KMT" },
          } : {}),
        },
      },
      breadcrumbLd(host, [
        { name: "Início", href: "/" }, { name: "Preços", href: "/precos" },
        { name: `${rec.b} ${rec.m}`, href: `/preco/${slug}` }, { name: String(year) },
      ]),
      faqLd(faqs),
    ],
  };

  return layout({
    title: `${rec.b} ${rec.m} ${year} usado: ${FM} (${cell.n} anúncios) · quanto vale`,
    description: `${rec.b} ${rec.m} de ${year} usado: preço mediano ${FM} (${FL}–${FH}) em ${cell.n} ${sample} do OLX Portugal${cell.km != null ? `, ${fmtKm(cell.km)} medianos` : ""}. Avaliação independente.`,
    canonical, jsonLd, body, zone: "all", nav: "precos", depositCount, index: true, host,
    altJson: `${canonical}.json`,
  });
}

// ═══ 404 ═════════════════════════════════════════════════════════════════════
//
// Unknown paths used to fall through to the analytics Basic-Auth gate and answer
// 401. To a crawler 401 means "there is something here and you may not have it",
// so every typo'd link, every old URL, every stray trailing slash was spending
// crawl budget and logging a site-wide access error instead of resolving.
//
// This is a real 404: correct status, useful body, noindex,follow so the links
// on it still carry signal back into the site.
export function renderNotFound({ suggestions = [], depositCount = 0, host = null, path = "" } = {}) {
  const chips = suggestions.slice(0, 12).map(s =>
    `<a class="mchip" href="/preco/${encodeURIComponent(s.slug)}">${escapeHtml(s.m)} <span class="mut">${fmtEur(s.fm)}</span></a>`).join("");
  const body = `
    <div class="fc-404">
      <div class="eyebrow" style="justify-content:center;margin-bottom:16px;"><span class="e-dot"></span><span class="mono">ERRO 404</span></div>
      <h1 class="fc-h1">Esta página não existe</h1>
      <p class="fc-p">${path ? `Não temos nada em <span class="mono">${escapeHtml(path)}</span>. ` : ""}Pode ter sido um link antigo ou um endereço mal escrito. O que existe está tudo a partir daqui:</p>
      <div class="hero-actions" style="justify-content:center;margin:22px 0 30px;">
        <a class="btn-dark" href="/precos">Preços por modelo</a>
        <a class="chip" href="/avaliar">Avaliar o meu carro</a>
        <a class="chip" href="/mercado">Mercado</a>
      </div>
      ${chips ? `<div class="sec-label" style="text-align:left;">MODELOS MAIS PROCURADOS</div><div class="mchips" style="justify-content:center;">${chips}</div>` : ""}
    </div>`;
  return layout({
    title: "Página não encontrada",
    description: "A página que procuras não existe. Vê os preços de carros usados por modelo em Portugal.",
    body, zone: "all", nav: null, depositCount, index: false, host,
  });
}

// ═══ /depreciacao/{slug} ═════════════════════════════════════════════════════
//
// "{modelo} desvalorização" is a query nobody on this market answers with real
// numbers, and it is the single most linkable thing this dataset can produce:
// "a Golf loses X% a year, and the fall flattens after N" is a sentence forums
// and press quote, which is where the links come from.
//
// Published only where the curve is a measurement (DEP_* guards above) — a fit
// through six scattered points would be a drawing, not a finding.
export function renderDepreciationPage({ rec, slug, fit, stats, pageYears, host, depositCount, builtAt }) {
  const B = escapeHtml(rec.b), M = escapeHtml(rec.m);
  const canonical = `https://${host}/depreciacao/${slug}`;
  const cs = fit.cells;                       // oldest → newest, n>=5
  const newest = fit.newest, oldest = fit.oldest;
  const ratePct = Math.round(fit.rate * 100);
  const keep = yrs => Math.round(Math.pow(1 - fit.rate, yrs) * 100);
  const av = depreciationAge(rec, fit, builtAt);
  const base = av ? av.base.price : newest.fm;
  const baseYear = av ? av.base.year : newest.y;
  const loseEur = yrs => Math.round(base * (1 - Math.pow(1 - fit.rate, yrs)));
  const dec = x => x.toFixed(1).replace(".", ",");

  const mkt = stats.depMed;
  const vsMarket = mkt
    ? (fit.rate > mkt * 1.15
        ? `Mais rápido do que o mercado (mediana ${Math.round(mkt * 100)}% ao ano).`
        : fit.rate < mkt * 0.85
          ? `Mais devagar do que o mercado (mediana ${Math.round(mkt * 100)}% ao ano) — segura melhor o valor do que a média.`
          : `Em linha com o mercado (mediana ${Math.round(mkt * 100)}% ao ano).`)
    : "";

  const ladderAges = av
    ? [...new Set([Math.ceil(av.minAge), 3, 5, 8, 10, 12, 15, 20, Math.floor(av.maxAge)])]
        .filter(a => a >= av.minAge && a <= av.maxAge).sort((a, b) => a - b)
    : [];
  const ladder = ladderAges.map(a => `<tr>
      <td>${a} ano${a === 1 ? "" : "s"} <span class="mut">(${av.ref - a})</span></td>
      <td>${fmtEur(av.at(a))}</td>
      <td class="mut">${fmtEur(Math.round(av.price(a)))}</td></tr>`).join("");

  const rows = cs.slice().sort((a, b) => b.y - a.y).map(c => {
    const vs = Math.round((c.fm / newest.fm - 1) * 100);
    const ageGap = newest.y - c.y;
    const link = pageYears.includes(c.y)
      ? `<a href="/preco/${slug}/${c.y}" style="color:#177A47;font-weight:600;">${c.y}</a>` : c.y;
    return `<tr><td>${link}</td>
      <td>${fmtEur(c.fm)}</td>
      <td>${c.n}</td>
      <td class="mut">${ageGap}</td>
      <td class="mut">${vs === 0 ? "—" : `${vs}%`}</td>
      <td class="mut">${c.km != null ? fmtKm(c.km) : "—"}</td></tr>`;
  }).join("");

  const bend = av && av.bend;
  const bendPara = bend
    ? (bend.dir === "slows"
        ? `<p class="fc-p"><b>Sim, e aos ${bend.age} anos.</b> Até essa idade o ${B} ${M} perde cerca de <b>${Math.round(bend.early * 100)}% por ano</b>; a partir daí, <b>${Math.round(bend.late * 100)}%</b>. Um exemplar já do lado direito dessa quebra custa-te menos por cada ano que o tiveres — é o troço onde comprar sai barato.</p>`
        : `<p class="fc-p"><b>Há uma quebra aos ${bend.age} anos, mas ao contrário do esperado:</b> antes disso o ${B} ${M} perde cerca de <b>${Math.round(bend.early * 100)}% por ano</b> e depois <b>${Math.round(bend.late * 100)}%</b> — a percentagem acelera com a idade em vez de abrandar. Nos exemplares mais velhos a conta é outra: o preço já é baixo, por isso a perda em euros continua a encolher.</p>`)
    : `<p class="fc-p"><b>Não, não em percentagem.</b> Testámos um ajuste com quebra em cada idade entre os ${BEND_MIN_AGE} e os ${BEND_MAX_AGE} anos e nenhum explica os preços do ${B} ${M} melhor do que uma queda constante de ${ratePct}% ao ano (R²=${fit.r2.toFixed(2).replace(".", ",")}).${av && av.bendCandidate ? ` O melhor candidato ficava aos ${av.bendCandidate.age} anos — ${Math.round(av.bendCandidate.early * 100)}% ao ano antes, ${Math.round(av.bendCandidate.late * 100)}% depois — e essa diferença é do tamanho dos saltos que a mediana já dá entre dois anos seguidos.` : ""} O que abranda é a fatura em euros, não a percentagem.</p>`;

  const cheapPara = !av ? "" : (av.cheapFrom
    ? `<p class="fc-p">Onde isso deixa de doer dá para datar: a partir dos <b>${av.cheapFrom.age} anos</b> — matrículas de ${av.ref - av.cheapFrom.age} e mais antigas, à volta de ${fmtEur(av.cheapFrom.price)} — cada ano de idade a mais custa menos de ${fmtEur(av.costFloor)}. É menos do que um jogo de pneus ou uma correia de distribuição, ou seja: a partir daí a matrícula pesa menos no orçamento do que o estado em que o carro está, e é por aí que a escolha se decide.</p>`
    : `<p class="fc-p">Aqui isso não chega a acontecer dentro do que alguém procura: aos ${av.capAge} anos um ano de idade ainda vale cerca de ${fmtEur(av.capCost)}, e só muito mais tarde desceria abaixo de ${fmtEur(av.costFloor)}. Neste modelo a matrícula manda no preço em toda a gama que se compra — esticar o orçamento por um ano mais recente continua a custar dinheiro a sério.</p>`);

  const body = crumbs([
    { name: "Início", href: "/" }, { name: "Desvalorização", href: "/depreciacao" },
    { name: `${rec.b} ${rec.m}` },
  ]) + `
    <div style="padding-top:14px;">
      <div class="side-card" style="max-width:680px;margin:0 auto;">
        <div class="eyebrow" style="margin-bottom:14px;"><span class="e-dot"></span><span class="mono">CURVA DE DESVALORIZAÇÃO · OLX PORTUGAL</span></div>
        <h1 class="fc-h1">Quanto se desvaloriza um ${B} ${M}?</h1>
        <p class="lede" style="font-size:16px;margin:0 0 18px;">Medido nos preços pedidos de <b>${rec.n} anúncios ativos</b> entre ${oldest.y} e ${newest.y}, um ${B} ${M} perde cerca de <b>${ratePct}% por cada ano de idade</b>${av && av.halfLife ? `, ou seja metade do valor a cada <b>${dec(av.halfLife)} anos</b>` : ""}. ${vsMarket}</p>
        <div class="fc-stat-row">
          <div class="fc-stat"><div class="k">POR ANO</div><div class="v">${ratePct}%</div><div class="s">-${fmtEur(loseEur(1))} por ano de idade</div></div>
          <div class="fc-stat"><div class="k">AOS 3 ANOS</div><div class="v">${keep(3)}%</div><div class="s">-${fmtEur(loseEur(3))} sobre ${fmtEur(base)}</div></div>
          <div class="fc-stat"><div class="k">AOS 5 ANOS</div><div class="v">${keep(5)}%</div><div class="s">-${fmtEur(loseEur(5))}</div></div>
          <div class="fc-stat"><div class="k">AOS 8 ANOS</div><div class="v">${keep(8)}%</div><div class="s">-${fmtEur(loseEur(8))}</div></div>
        </div>
        ${provenance({ n: rec.n, builtAt, measure: `Preço pedido mediano por ano de fabrico, ${oldest.y}-${newest.y}`, extra: `Ajuste log-linear, R²=${fit.r2.toFixed(2)}; valores em euros sobre ${fmtEur(base)}, o que a curva dá a um exemplar de ${baseYear}` })}
      </div>
    </div>
    <section class="section fc-wrap">
      <h2 class="fc-h2">A curva</h2>
      ${av ? depreciationChart(av) : priceChart(cs)}
      <p class="fc-p" style="margin-top:10px;">Cada ponto é a mediana dos preços pedidos nessa idade; a linha é o ajuste log-linear que gera a percentagem acima${av && (av.cheapFrom || av.bend) ? ", e as marcas verticais são as idades sobre as quais esta página faz uma afirmação" : ""}. A queda é uma percentagem do que resta, por isso é grande em euros nos primeiros anos e pequena no fim — mesmo quando a percentagem não muda.</p>
      <p class="fc-p">Uma ressalva que a curva não mostra: entre ${oldest.y} e ${newest.y} o ${B} ${M} mudou de geração mais do que uma vez, e um exemplar de cada ponta não é o mesmo carro com mais uns anos. Parte da queda é desgaste e idade, parte é um modelo diferente com outro equipamento — a série mede o que o mercado pede por cada ano de matrícula, não o envelhecimento de um carro concreto.</p>
    </section>
    ${ladder ? `
    <section class="section fc-wrap">
      <h2 class="fc-h2">Quanto custa um ano de idade</h2>
      <p class="fc-p">A mesma curva lida ao contrário: o que pagas, em euros, por cada ano de matrícula mais recente. É esta coluna que decide se vale a pena esticar o orçamento por um ano a mais.</p>
      <div class="fc-scroll"><table class="fc-tbl">
        <thead><tr><th>Idade (matrícula)</th><th>Custo de +1 ano</th><th>Preço na curva</th></tr></thead>
        <tbody>${ladder}</tbody></table></div>
      <p class="fc-prov mono">Valores da curva ajustada, não medianas observadas: entre dois anos seguidos a mediana salta mais do que o passo que estamos a medir. As medianas em bruto estão na tabela por ano, mais abaixo.</p>
    </section>` : ""}
    <section class="section fc-wrap">
      <h2 class="fc-h2">Há um ponto de inflexão?</h2>
      ${bendPara}
      ${cheapPara}
    </section>
    <section class="section fc-wrap">
      <h2 class="fc-h2">Preço mediano por ano</h2>
      <div class="fc-scroll"><table class="fc-tbl">
        <thead><tr><th>Ano</th><th>Mediano (pedido)</th><th>Anúncios</th><th>Anos vs. ${newest.y}</th><th>vs. ${newest.y}</th><th>Km mediano</th></tr></thead>
        <tbody>${rows}</tbody></table></div>
    </section>
    <section class="section fc-wide">
      <div class="cta-banner">
        <div style="flex:1 1 360px;">
          <h2>Quanto vale o TEU ${B} ${M} hoje?</h2>
          <p>Esta curva é do modelo. Cola o link do teu anúncio e dizemos o valor justo do teu carro concreto, com os teus quilómetros e a tua versão.</p>
        </div>
        <a class="btn-bright" href="/avaliar?modelo=${encodeURIComponent(slug)}">Avaliar o meu carro&nbsp;&nbsp;→</a>
      </div>
    </section>
    <section class="section fc-wrap" style="padding-bottom:70px;">
      <p class="fc-p"><a href="/preco/${slug}">Preço de ${B} ${M} por ano</a> · <a href="/depreciacao">Desvalorização de outros modelos</a> · <a href="/precos">Todos os modelos</a> · <a href="${canonical}.json">Dados em JSON</a></p>
    </section>`;

  const faqs = [
    [`Quanto se desvaloriza um ${rec.b} ${rec.m} por ano?`,
     `Cerca de ${ratePct}% do valor restante por cada ano de idade, medido nos preços pedidos de ${rec.n} anúncios ativos de ${rec.b} ${rec.m} entre ${oldest.y} e ${newest.y} no OLX Portugal. A percentagem é constante, mas em euros a perda é muito maior nos primeiros anos.`],
    [`Quanto vale um ${rec.b} ${rec.m} ao fim de 5 anos?`,
     `Ao ritmo medido, um ${rec.b} ${rec.m} mantém cerca de ${keep(5)}% do valor ao fim de 5 anos. Sobre os ${fmtEur(base)} que a curva dá a um exemplar de ${baseYear}, isso são cerca de ${fmtEur(loseEur(5))} perdidos.`],
  ];
  if (av && av.halfLife) faqs.push([
    `Em quantos anos um ${rec.b} ${rec.m} perde metade do valor?`,
    `Cerca de ${dec(av.halfLife)} anos, ao ritmo de ${ratePct}% ao ano medido nos anúncios ativos do OLX Portugal. É a mesma taxa dita de outra maneira: a cada ${dec(av.halfLife)} anos de idade o preço pedido mediano fica a metade.`]);
  if (av) faqs.push([
    `A partir de que idade um ${rec.b} ${rec.m} deixa de perder valor?`,
    bend && bend.dir === "slows"
      ? `A queda abranda aos ${bend.age} anos: até lá são cerca de ${Math.round(bend.early * 100)}% ao ano, depois ${Math.round(bend.late * 100)}%.${av.cheapFrom ? ` Em euros, a partir dos ${av.cheapFrom.age} anos cada ano de idade custa menos de ${fmtEur(av.costFloor)}.` : ""}`
      : `Nunca deixa de perder, mas deixa de doer. Em percentagem a queda mantém-se em cerca de ${ratePct}% ao ano em toda a série medida — não há idade a partir da qual a percentagem trave.${av.cheapFrom ? ` Em euros é outra história: a partir dos ${av.cheapFrom.age} anos (matrículas de ${av.ref - av.cheapFrom.age} e mais antigas) cada ano de idade custa menos de ${fmtEur(av.costFloor)}, e aí o estado do carro pesa mais do que o ano.` : ` Em euros a perda encolhe, mas devagar: aos ${av.capAge} anos um ano de idade ainda custa cerca de ${fmtEur(av.capCost)}.`}`]);
  if (mkt) faqs.push([
    `O ${rec.b} ${rec.m} desvaloriza mais do que a média?`,
    `A mediana do mercado português de usados que medimos é ${Math.round(mkt * 100)}% por ano de idade. O ${rec.b} ${rec.m} está nos ${ratePct}%, ou seja ${fit.rate > mkt ? "acima" : fit.rate < mkt ? "abaixo" : "em linha com"} a média.`]);

  const jsonLd = {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "Dataset",
        "license": licenseUrl(host),
        "name": `Desvalorização de ${rec.b} ${rec.m} em Portugal`,
        "description": `Preço pedido mediano de ${rec.b} ${rec.m} por ano de fabrico (${oldest.y}-${newest.y}), taxa de desvalorização anual de ${ratePct}% e custo em euros de cada ano de idade, a partir de ${rec.n} anúncios ativos do OLX Portugal.`,
        "creator": { "@type": "Organization", "name": "Carsbuyer", "url": `https://${host}/` },
        "isAccessibleForFree": true,
        "temporalCoverage": `${oldest.y}/${newest.y}`,
        "variableMeasured": ["Preço pedido (EUR)", "Desvalorização anual (%)", "Custo de um ano de idade (EUR)"],
        "distribution": [{ "@type": "DataDownload", "encodingFormat": "application/json", "contentUrl": `${canonical}.json` }],
        "dateModified": builtAt || undefined,
        "url": canonical,
      },
      breadcrumbLd(host, [
        { name: "Início", href: "/" }, { name: "Desvalorização", href: "/depreciacao" },
        { name: `${rec.b} ${rec.m}` },
      ]),
      faqLd(faqs),
    ],
  };
  return layout({
    title: `${rec.b} ${rec.m}: desvalorização por ano`,
    description: `Um ${rec.b} ${rec.m} perde cerca de ${ratePct}% por ano de idade e mantém ${keep(5)}% ao fim de 5 anos, medido em ${rec.n} anúncios ativos do OLX Portugal. Curva completa, custo de cada ano de idade e onde a queda abranda.`,
    canonical, jsonLd, body, zone: "all", nav: "precos", depositCount, index: true, host,
    altJson: `${canonical}.json`,
  });
}

// ── /depreciacao — hub, ranked ───────────────────────────────────────────────
export function renderDepreciationHub({ rows, stats, host, depositCount, builtAt, duelHubs = [] }) {
  const canonical = `https://${host}/depreciacao`;
  const dec = x => x.toFixed(1).replace(".", ",");
  const tr = rows.map(r => `<tr>
      <td><a href="/depreciacao/${r.slug}" style="color:#177A47;font-weight:600;">${escapeHtml(r.b)} ${escapeHtml(r.m)}</a></td>
      <td>${Math.round(r.rate * 100)}%</td>
      <td class="mut">${Math.round(Math.pow(1 - r.rate, 5) * 100)}%</td>
      <td class="mut">${r.half ? `${dec(r.half)} anos` : "—"}</td>
      <td class="mut">${r.cheapAge ? `${r.cheapAge} anos` : "—"}</td>
      <td class="mut">${r.n}</td>
      <td class="mut">${r.span} anos</td></tr>`).join("");

  const yg = stats.depYoung, od = stats.depOld;
  const ageBand = (yg && od) ? (() => {
    const a = Math.round(yg.rate * 100), b = Math.round(od.rate * 100);
    const verdict = b >= a - 1
      ? `A percentagem <b>não abranda com a idade</b> — o que abranda é a fatura em euros, porque ${a}% de um carro de 12 000 € e ${a}% de um de 3 000 € não são a mesma conta.`
      : `A percentagem abranda com a idade, mas menos do que se costuma dizer: são ${a - b} pontos entre um troço e o outro.`;
    return `<p class="fc-p">Entre os ${yg.from} e os ${yg.to} anos de idade, a mediana destes modelos perde <b>${a}% ao ano</b> (${yg.models} modelos com amostra nesse troço); dos ${od.from} anos em diante, <b>${b}%</b> (${od.models} modelos). ${verdict}</p>`;
  })() : "";

  const body = crumbs([{ name: "Início", href: "/" }, { name: "Desvalorização" }]) + `
    <section class="section fc-wrap" style="padding-top:16px;">
      <h1 class="fc-h1">Que carros se desvalorizam mais em Portugal</h1>
      <p class="fc-p">Taxa de desvalorização por ano de idade, medida nos preços pedidos de anúncios ativos do OLX. Só entram modelos com histórico suficiente para a curva significar alguma coisa: pelo menos ${DEP_MIN_CELLS} anos com amostra e ${DEP_MIN_SPAN} anos de intervalo.</p>
      ${stats.depMed ? `<p class="fc-p">A mediana do mercado é <b>${Math.round(stats.depMed * 100)}% por ano</b>. Acima disso, o carro custa-te mais a ter; abaixo, revendes com menos perda.</p>` : ""}
      ${ageBand}
      <div class="fc-scroll"><table class="fc-tbl">
        <thead><tr><th>Modelo</th><th>Por ano</th><th>Valor aos 5 anos</th><th>Metade do valor</th><th>Um ano custa &lt;500 €</th><th>Anúncios</th><th>Histórico</th></tr></thead>
        <tbody>${tr}</tbody></table></div>
      <p class="fc-prov mono">"Metade do valor" é o tempo que o modelo leva a valer metade, ao ritmo medido. "Um ano custa &lt;500 €" é a idade a partir da qual mais um ano de matrícula vale menos de 500 € na curva ajustada; um travessão significa que isso não acontece antes dos ${CHEAP_MAX_AGE} anos, ou seja a matrícula manda no preço em toda a gama que se compra. As curvas atravessam gerações: parte da queda é modelo diferente, não idade.</p>
      ${provenance({ n: stats.listings, builtAt, measure: "Preço pedido mediano por ano de fabrico, ajuste log-linear" })}
      ${duelHubs.length ? `<p class="fc-p" style="margin-top:18px;">Esta tabela mede o modelo inteiro, com todas as versões juntas. Onde a amostra chega para separar as curvas, separamo-las: ${duelHubs.map(d => `<a href="/${d.path}">${escapeHtml(d.question)}</a>`).join(" · ")}.</p>` : ""}
      <p class="fc-p" style="margin-top:18px;"><a href="/precos">Todos os modelos</a> · <a href="/liquidez">Quanto tempo demoram a vender</a> · <a href="/metodologia">Como calculamos</a></p>
    </section>
    <div style="height:60px;"></div>`;
  return layout({
    title: "Desvalorização de carros usados em Portugal",
    description: `Que modelos perdem mais valor por ano em Portugal, medido em anúncios ativos do OLX. ${rows.length} modelos com curva completa, valor retido aos 5 anos e a idade a partir da qual o ano de matrícula deixa de mandar no preço.`,
    canonical, body, zone: "all", nav: "precos", depositCount, index: true, host,
    jsonLd: {
      "@context": "https://schema.org",
      "@graph": [
        {
          "@type": "CollectionPage", "url": canonical, "inLanguage": "pt-PT",
          "name": "Desvalorização de carros usados em Portugal",
        },
        breadcrumbLd(host, [{ name: "Início", href: "/" }, { name: "Desvalorização" }]),
      ],
    },
  });
}

// ═══ /comparar/{a}-vs-{b} ════════════════════════════════════════════════════
//
// "{A} ou {B} qual comprar usado" is how people actually shop, and no page on
// this market answers it with both cars' real numbers side by side.
//
// The page does NOT pick a winner overall — we measure price, dispersion, time
// to sell and depreciation, and none of those is "the better car". It names the
// winner PER DIMENSION and says what each one costs you, which is the honest
// version and also the more useful one.
export function renderComparePage({ a, b, ra, rb, stats, host, depositCount, builtAt }) {
  const canonical = `https://${host}/comparar/${a}-vs-${b}`;
  const nameA = `${ra.b} ${ra.m}`, nameB = `${rb.b} ${rb.m}`;
  const A = escapeHtml(nameA), Bn = escapeHtml(nameB);
  const fitA = depreciationFit(ra), fitB = depreciationFit(rb);
  const depA = (fitA && fitA.cells.length >= 5 && fitA.rate > 0 && fitA.rate < 0.30) ? fitA.rate : null;
  const depB = (fitB && fitB.cells.length >= 5 && fitB.rate > 0 && fitB.rate < 0.30) ? fitB.rate : null;
  const sprA = ra.fm > 0 ? (ra.fh - ra.fl) / ra.fm : null;
  const sprB = rb.fm > 0 ? (rb.fh - rb.fl) / rb.fm : null;
  const gap = comparePriceGap(ra, rb);
  const gPct = gap ? Math.round((Math.max(gap.ratio, 1 / gap.ratio) - 1) * 100) : null;
  const gDearer = gap ? (gap.ratio > 1 ? "a" : "b") : null;

  const card = (slug, r, name, wins) => `
    <div class="fc-vs-card${wins ? " win" : ""}">
      <div class="mono" style="font-size:11px;color:#8A8F98;letter-spacing:.04em;">${escapeHtml(r.b).toUpperCase()}</div>
      <div style="font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:19px;margin:3px 0 12px;">${escapeHtml(r.m)}</div>
      <div class="cap">Preço mediano (pedido)</div>
      <div class="mono" style="font-weight:700;font-size:27px;letter-spacing:-.02em;">${fmtEur(r.fm)}</div>
      <div class="mono" style="font-size:12px;color:#5B606B;margin-top:2px;">${fmtEur(r.fl)} – ${fmtEur(r.fh)} · ${r.n} anúncios</div>
      <div style="margin-top:12px;font-size:13.5px;line-height:1.8;color:#3A3F47;">
        ${r.kmm != null ? `Km mediano · <b>${fmtKm(r.kmm)}</b><br>` : ""}
        ${r.sd != null ? `Vende em · <b>${r.sd} dias</b><br>` : ""}
        ${r.y0 && r.y1 ? `Anos à venda · <b>${r.y0}–${r.y1}</b><br>` : ""}
        ${Array.isArray(r.fu) && r.fu.length ? `Combustível · <b>${escapeHtml(String(r.fu[0][0]))} ${Math.round(r.fu[0][1] * 100)}%</b>` : ""}
      </div>
      <a class="btn-dark" href="/preco/${slug}" style="display:block;text-align:center;margin-top:14px;font-size:13.5px;padding:11px;">Ver ${escapeHtml(r.m)}&nbsp;→</a>
    </div>`;

  // Per-dimension verdicts. Each row states which side wins AND why that matters,
  // because "cheaper" is only an advantage once you know what it costs elsewhere.
  const verdicts = [];
  const cheaper = gap ? (gDearer === "a" ? "b" : "a") : (ra.fm <= rb.fm ? "a" : "b");
  verdicts.push({
    k: "Preço ao mesmo ano",
    w: cheaper,
    t: gPct === 0
      ? `Ao mesmo ano de modelo pedem praticamente o mesmo, nos ${gap.years} anos em que ambos têm amostra.`
      : `Ao mesmo ano de modelo, ${gDearer === "a" ? A : Bn} pede ${gPct}% mais do que ${gDearer === "a" ? Bn : A} — mediana dos ${gap.years} anos em que ambos têm anúncios suficientes.`,
  });
  if (ra.sd != null && rb.sd != null) {
    const w = ra.sd <= rb.sd ? "a" : "b";
    verdicts.push({
      k: "Tempo até vender",
      w,
      t: `${w === "a" ? A : Bn} sai mais depressa (${Math.min(ra.sd, rb.sd)} contra ${Math.max(ra.sd, rb.sd)} dias medianos). Quem vende tem menos pressão no preço; quem compra tem menos tempo para decidir.`,
    });
  }
  if (depA != null && depB != null) {
    const w = depA <= depB ? "a" : "b";
    verdicts.push({
      k: "Desvalorização",
      w,
      t: `${w === "a" ? A : Bn} segura melhor o valor: ${Math.round(Math.min(depA, depB) * 100)}% por ano contra ${Math.round(Math.max(depA, depB) * 100)}%. Em cinco anos a diferença vale mais do que o desconto na compra.`,
    });
  }
  if (sprA != null && sprB != null) {
    const w = sprA <= sprB ? "a" : "b";
    verdicts.push({
      k: "Previsibilidade do preço",
      w,
      t: `${w === "a" ? A : Bn} tem preços mais concentrados (${Math.round(Math.min(sprA, sprB) * 100)}% de dispersão contra ${Math.round(Math.max(sprA, sprB) * 100)}%). Onde a dispersão é maior, o estado e a versão decidem mais do que o modelo — dá para fazer melhores negócios, e piores.`,
    });
  }
  if (ra.kmm != null && rb.kmm != null) {
    const w = ra.kmm <= rb.kmm ? "a" : "b";
    verdicts.push({
      k: "Quilometragem à venda",
      w,
      t: `Os ${w === "a" ? A : Bn} à venda estão menos rodados (${fmtKm(Math.min(ra.kmm, rb.kmm))} contra ${fmtKm(Math.max(ra.kmm, rb.kmm))} medianos). A comparação de preço acima já é ao mesmo ano, mas não ao mesmo quilómetro: parte do que sobra é isto.`,
    });
  }
  const scoreA = verdicts.filter(v => v.w === "a").length;
  const scoreB = verdicts.filter(v => v.w === "b").length;

  const vrows = verdicts.map(v => `<tr>
      <td>${escapeHtml(v.k)}</td>
      <td class="nm">${v.w === "a" ? `<b>${escapeHtml(ra.m)}</b>` : escapeHtml(ra.m)}${v.w === "a" ? '<span class="fc-win">+</span>' : ""}</td>
      <td class="nm">${v.w === "b" ? `<b>${escapeHtml(rb.m)}</b>` : escapeHtml(rb.m)}${v.w === "b" ? '<span class="fc-win">+</span>' : ""}</td>
    </tr>`).join("");

  const yrows = gap ? gap.cells.slice(0, CMP_TABLE_YEARS).map(c => {
    const r = c.fa / c.fb;
    const dearer = r > 1 ? escapeHtml(ra.m) : escapeHtml(rb.m);
    const pct = Math.round((Math.max(r, 1 / r) - 1) * 100);
    return `<tr>
      <td class="mono">${c.y}</td>
      <td class="mono">${fmtEur(c.fa)}</td>
      <td class="mono">${fmtEur(c.fb)}</td>
      <td>${pct === 0 ? "iguais" : `${dearer} +${pct}%`}</td>
      <td class="mut mono">${c.na}&nbsp;/&nbsp;${c.nb}</td>
    </tr>`;
  }).join("") : "";

  const body = crumbs([
    { name: "Início", href: "/" }, { name: "Comparar", href: "/comparar" },
    { name: `${nameA} vs ${nameB}` },
  ]) + `
    <section class="section fc-wrap" style="padding-top:16px;">
      <div class="eyebrow" style="margin-bottom:14px;"><span class="e-dot"></span><span class="mono">COMPARAÇÃO A PARTIR DE ANÚNCIOS ATIVOS · OLX PORTUGAL</span></div>
      <h1 class="fc-h1">${A} ou ${Bn}: qual comprar usado?</h1>
      <p class="fc-p">Comparação com números do mercado português de hoje: ${ra.n} anúncios ativos de ${A} e ${rb.n} de ${Bn}. Não dizemos qual é o melhor carro — dizemos o que cada um custa a comprar, a ter e a revender, e deixamos a escolha contigo.</p>
      <div class="fc-vs">
        ${card(a, ra, nameA, scoreA > scoreB)}
        ${card(b, rb, nameB, scoreB > scoreA)}
      </div>
      ${gap ? `<p class="fc-prov mono">As duas medianas acima são de carros de idades diferentes: ${ra.y0 && ra.y1 ? `${ra.m} ${ra.y0}–${ra.y1}` : ra.m} contra ${rb.y0 && rb.y1 ? `${rb.m} ${rb.y0}–${rb.y1}` : rb.m}. A comparação de preço desta página é feita ano a ano, mais abaixo.</p>` : ""}
      ${provenance({ n: ra.n + rb.n, builtAt, measure: "Preço pedido em anúncios ativos dos dois modelos (mediana e P25-P75)" })}
    </section>
    <section class="section fc-wrap">
      <h2 class="fc-h2">Quem ganha em quê</h2>
      <div class="fc-scroll"><table class="fc-tbl">
        <thead><tr><th>Critério</th><th>${escapeHtml(ra.m)}</th><th>${escapeHtml(rb.m)}</th></tr></thead>
        <tbody>${vrows}</tbody></table></div>
      <ul class="fc-insights" style="margin-top:16px;">${verdicts.map(v => `<li>${v.t}</li>`).join("")}</ul>
    </section>
    ${gap ? `
    <section class="section fc-wrap">
      <h2 class="fc-h2">Preço lado a lado, ano a ano</h2>
      <p class="fc-p">O mesmo ano de matrícula dos dois lados, que é a única forma de a diferença ser sobre os carros e não sobre a idade de quem está a vender. ${gPct === 0 ? `No conjunto dos ${gap.years} anos comuns, as medianas empatam.` : `No conjunto dos ${gap.years} anos comuns, ${gDearer === "a" ? A : Bn} pede <b>${gPct}% mais</b>.`}</p>
      <div class="fc-scroll"><table class="fc-tbl">
        <thead><tr><th>Ano</th><th>${escapeHtml(ra.m)}</th><th>${escapeHtml(rb.m)}</th><th>Diferença</th><th>Anúncios</th></tr></thead>
        <tbody>${yrows}</tbody></table></div>
      <p class="fc-prov mono">A última coluna é o número de anúncios de cada lado nesse ano — a diferença de um ano com 5 e 6 anúncios diz muito menos do que a de um com 40. A percentagem do conjunto pesa cada ano por essa amostra, e não pela sua ordem.${gap.cells.length > CMP_TABLE_YEARS ? ` Mostrados os ${CMP_TABLE_YEARS} anos mais recentes dos ${gap.cells.length} em que ambos têm amostra; a percentagem usa todos.` : ""}</p>
    </section>` : ""}
    <section class="section fc-wide">
      <div class="cta-banner">
        <div style="flex:1 1 360px;">
          <h2>Já tens um anúncio em vista?</h2>
          <p>A mediana compara modelos. Para saber se AQUELE carro está bem de preço, cola o link do anúncio.</p>
        </div>
        <a class="btn-bright" href="/avaliar">Avaliar um anúncio&nbsp;&nbsp;→</a>
      </div>
    </section>
    <section class="section fc-wrap" style="padding-bottom:70px;">
      <p class="fc-p"><a href="/preco/${a}">Preços de ${A}</a> · <a href="/preco/${b}">Preços de ${Bn}</a> · <a href="/comparar">Outras comparações</a></p>
    </section>`;

  const faqs = [
    [`${nameA} ou ${nameB}: qual é mais barato em Portugal?`,
     gPct === 0
       ? `Ao mesmo ano de modelo os dois pedem praticamente o mesmo, na mediana dos ${gap.years} anos em que ambos têm amostra no OLX Portugal.`
       : `Ao mesmo ano de modelo, o ${gDearer === "a" ? nameA : nameB} pede cerca de ${gPct}% mais do que o ${gDearer === "a" ? nameB : nameA}, na mediana dos ${gap.years} anos em que ambos têm amostra no OLX Portugal. Nas medianas de tudo o que está à venda a diferença parece outra (${fmtEur(ra.fm)} contra ${fmtEur(rb.fm)}), porque os dois modelos não estão à venda com a mesma idade.`],
  ];
  if (depA != null && depB != null) faqs.push([
    `${nameA} ou ${nameB}: qual perde menos valor?`,
    `O ${depA <= depB ? nameA : nameB} desvaloriza cerca de ${Math.round(Math.min(depA, depB) * 100)}% por ano de idade, contra ${Math.round(Math.max(depA, depB) * 100)}% do outro. Sobre cinco anos, essa diferença costuma pesar mais do que o desconto na compra.`]);
  if (ra.sd != null && rb.sd != null) faqs.push([
    `Qual se vende mais depressa, ${nameA} ou ${nameB}?`,
    `O ${ra.sd <= rb.sd ? nameA : nameB} vende em mediana em ${Math.min(ra.sd, rb.sd)} dias no OLX, contra ${Math.max(ra.sd, rb.sd)} dias do outro. Um modelo que sai depressa dá menos margem de negociação a quem compra.`]);

  return layout({
    title: `${nameA} ou ${nameB}? Comparação de preços usados`,
    description: gap
      ? `${nameA} contra ${nameB} no mercado português de usados: preço comparado ao mesmo ano de modelo (${gPct === 0 ? "empate" : `${gDearer === "a" ? nameA : nameB} +${gPct}%`}), quilometragem, tempo até vender e desvalorização, em anúncios ativos do OLX.`
      : `${nameA} (${fmtEur(ra.fm)}) contra ${nameB} (${fmtEur(rb.fm)}) no mercado português de usados: preço, quilometragem, tempo até vender e desvalorização, em anúncios ativos do OLX.`,
    canonical, body, zone: "all", nav: "precos", depositCount, index: true, host,
    jsonLd: {
      "@context": "https://schema.org",
      "@graph": [
        breadcrumbLd(host, [
          { name: "Início", href: "/" }, { name: "Comparar", href: "/comparar" },
          { name: `${nameA} vs ${nameB}` },
        ]),
        faqLd(faqs),
      ],
    },
  });
}

const CLASS_LABEL = new Map(Object.entries({
  a: "Citadinos", b: "Utilitários", "b-premium": "Utilitários premium",
  c: "Compactos", "c-estate": "Carrinhas compactas",
  d: "Berlinas médias", "d-estate": "Carrinhas médias",
  e: "Berlinas grandes", "e-estate": "Carrinhas grandes",
  "suv-b": "SUV pequenos", "suv-c": "SUV médios", "suv-d": "SUV grandes",
  mpv: "Monovolumes",
}));

export function renderCompareHub({ pairs, models, host, depositCount, builtAt }) {
  const canonical = `https://${host}/comparar`;
  const groups = new Map();
  for (const [a, b] of pairs) {
    const k = modelClass(a) || "outros";
    if (!groups.has(k)) groups.set(k, []);
    groups.get(k).push([a, b]);
  }
  const order = [...CLASS_LABEL.keys(), "outros"];
  const items = order.filter(k => groups.has(k)).map(k => {
    const chips = groups.get(k).map(([a, b]) => {
      const ra = models[a], rb = models[b];
      return `<a class="mchip" href="/comparar/${a}-vs-${b}">${escapeHtml(ra.b)} ${escapeHtml(ra.m)} <span class="mut">vs</span> ${escapeHtml(rb.b)} ${escapeHtml(rb.m)}</a>`;
    }).join("");
    return `<h2 class="fc-h2" style="font-size:16px;margin:22px 0 10px;">${escapeHtml(CLASS_LABEL.get(k) || "Outros")}</h2><div class="mchips">${chips}</div>`;
  }).join("");
  const body = crumbs([{ name: "Início", href: "/" }, { name: "Comparar" }]) + `
    <section class="section fc-wrap" style="padding-top:16px;">
      <h1 class="fc-h1">Comparar carros usados em Portugal</h1>
      <p class="fc-p">Cada comparação usa os anúncios ativos dos dois modelos no OLX: preço, dispersão, quilometragem, tempo até vender e desvalorização. Só pomos frente a frente modelos de marcas diferentes que jogam no mesmo segmento — é entre esses que a escolha existe de facto, e é por isso que não vais encontrar aqui um citadino contra uma berlina.</p>
      <p class="fc-p">O preço é comparado <b>ao mesmo ano de modelo</b>, não pela mediana de tudo o que está à venda. Um modelo cujos anúncios são em média mais velhos parece mais barato sem o ser, e essa é a comparação que toda a gente faz por engano.</p>
      ${items}
      ${provenance({ n: null, builtAt, measure: "Preço pedido mediano dos dois modelos comparados" })}
      <p class="fc-p" style="margin-top:18px;"><a href="/precos">Todos os modelos</a> · <a href="/depreciacao">Desvalorização</a> · <a href="/metodologia">Como calculamos</a></p>
    </section>
    <div style="height:60px;"></div>`;
  return layout({
    title: "Comparar carros usados em Portugal",
    description: `${pairs.length} comparações de carros usados em Portugal com preço, quilometragem, tempo até vender e desvalorização, a partir de anúncios ativos do OLX.`,
    canonical, body, zone: "all", nav: "precos", depositCount, index: true, host,
    jsonLd: {
      "@context": "https://schema.org",
      "@graph": [
        { "@type": "CollectionPage", "url": canonical, "inLanguage": "pt-PT", "name": "Comparar carros usados em Portugal" },
        breadcrumbLd(host, [{ name: "Início", href: "/" }, { name: "Comparar" }]),
      ],
    },
  });
}

// ═══ /liquidez — how long each model takes to sell ═══════════════════════════
//
// Days-to-sell is the one number here that neither Standvirtual nor a valuation
// book publishes, because it needs listings watched over time rather than a
// snapshot. It answers the seller's real question ("how long will this take")
// and hands the buyer their leverage ("this one sits, so offer less").
const LIQ_HORIZONS = [30, 60, 90];

function liqPct(x) { return Math.round(x * 100); }

export function liquidityChart(lq, { w = 640, h = 210, color = "#177A47" } = {}) {
  const pts = [[0, 1]];
  for (const d of LIQ_HORIZONS) {
    const s = lq[`s${d}`];
    if (s != null) pts.push([d, 1 - s]);
  }
  if (pts.length < 3) return "";
  const padL = 36, padR = 14, padT = 18, padB = 30;
  const x1 = pts[pts.length - 1][0];
  const X = d => padL + (d / x1) * (w - padL - padR);
  const Y = v => padT + (1 - v) * (h - padT - padB);
  const line = pts.map((p, i) => `${i ? "L" : "M"}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join("");
  const area = `${line}L${X(x1).toFixed(1)},${Y(0).toFixed(1)}L${X(0).toFixed(1)},${Y(0).toFixed(1)}Z`;
  const dots = pts.slice(1).map(p =>
    `<circle cx="${X(p[0]).toFixed(1)}" cy="${Y(p[1]).toFixed(1)}" r="3.2" fill="${color}">`
    + `<title>Aos ${p[0]} dias ainda estão à venda ${liqPct(p[1])} em cada 100</title></circle>`).join("");
  const ticks = [0, 0.5, 1].map(f =>
    `<line x1="${padL}" x2="${w - padR}" y1="${Y(f).toFixed(1)}" y2="${Y(f).toFixed(1)}" class="c-grid"/>`
    + `<text x="${padL - 5}" y="${(Y(f) + 4).toFixed(1)}" text-anchor="end" class="c-ax">${liqPct(f)}%</text>`).join("");
  const xlab = pts.map(p =>
    `<text x="${X(p[0]).toFixed(1)}" y="${h - 9}" text-anchor="${p[0] === 0 ? "start" : p[0] === x1 ? "end" : "middle"}" class="c-ax">${p[0]} dias</text>`).join("");
  const mark = (lq.md != null && lq.md > 0 && lq.md <= x1)
    ? `<line x1="${X(lq.md).toFixed(1)}" x2="${X(lq.md).toFixed(1)}" y1="${padT}" y2="${Y(0).toFixed(1)}" class="c-mark"/>`
      + `<text x="${X(lq.md).toFixed(1)}" y="${padT - 5}" text-anchor="${X(lq.md) > w - 100 ? "end" : "middle"}" class="c-marklab">mediana: ${lq.md} dias</text>`
    : "";
  return `<svg class="fc-chart" viewBox="0 0 ${w} ${h}" role="img"
    aria-label="Percentagem de anúncios ainda à venda ao longo dos dias">${ticks}
    <path d="${area}" fill="${color}" opacity="0.10"/>
    <path d="${line}" fill="none" stroke="${color}" stroke-width="2.2" stroke-linejoin="round"/>
    ${dots}${mark}${xlab}</svg>`;
}

export function liquidityJson(rec, slug, { host, builtAt } = {}) {
  const lq = rec.lq || {};
  const cut = cells => (cells || []).map(c => ({
    key: c.k, label: c.lbl, sample: c.n,
    gone_in_30d: c.s30, median_days: c.md != null ? c.md : null,
  }));
  return {
    slug, brand: rec.b, model: rec.m,
    url: `https://${host}/liquidez/${slug}`,
    measure: "days from the listing appearing on OLX to the last scrape cycle that saw it live",
    estimator: "kaplan-meier, listings still on sale censored at the last scrape",
    sample_ended: lq.n != null ? lq.n : null,
    sample_still_listed: lq.cn != null ? lq.cn : null,
    median_days: lq.md != null ? lq.md : null,
    p25_days: lq.q1 != null ? lq.q1 : null,
    p75_days: lq.q3 != null ? lq.q3 : null,
    gone_in_30d: lq.s30 != null ? lq.s30 : null,
    gone_in_60d: lq.s60 != null ? lq.s60 : null,
    gone_in_90d: lq.s90 != null ? lq.s90 : null,
    relisted_share: lq.rb != null ? lq.rb : null,
    price_cut_share: lq.cu != null ? lq.cu : null,
    price_cut_median: lq.cp != null ? lq.cp : null,
    by_price: cut(lq.pb), by_age: cut(lq.ab), by_district: cut(lq.dt),
    updated: builtAt || null,
    licence: licenseUrl(host),
    caveat: "a listing leaving OLX is not proof of a sale: an ad runs in 30-day cycles and can simply expire",
  };
}

function liqCutTable(cells, head) {
  if (!cells || !cells.length) return "";
  return `<div class="fc-scroll"><table class="fc-tbl">
    <thead><tr><th>${head}</th><th>Sai nos primeiros 30 dias</th><th>Mediana</th><th>Anúncios observados</th></tr></thead>
    <tbody>${cells.map(c => `<tr>
      <td>${escapeHtml(c.lbl)}</td>
      <td>${liqPct(c.s30)}%</td>
      <td class="mut">${c.md != null ? `${c.md} dias` : "—"}</td>
      <td class="mut">${fmtNum(c.n)}</td></tr>`).join("")}</tbody></table></div>`;
}

export function renderLiquidityPage({ rec, slug, market, hasDepreciation = false,
                                      host, depositCount, builtAt }) {
  const lq = rec.lq;
  const B = escapeHtml(rec.b), M = escapeHtml(rec.m);
  const canonical = `https://${host}/liquidez/${slug}`;
  const mkt = market || {};
  const s30 = liqPct(lq.s30);
  const still90 = lq.s90 != null ? liqPct(1 - lq.s90) : null;
  const vsMkt = mkt.s30 != null
    ? (lq.s30 >= mkt.s30 * 1.12 ? "acima" : lq.s30 <= mkt.s30 * 0.88 ? "abaixo" : "media")
    : null;
  const verdict = vsMkt === "acima"
    ? `Sai mais depressa do que a média do mercado (${liqPct(mkt.s30)} em cada 100). A anunciar, tens pouca razão para começar abaixo do valor; a comprar, os exemplares bem preçados desaparecem antes de teres tempo de pensar.`
    : vsMkt === "abaixo"
      ? `Sai mais devagar do que a média do mercado (${liqPct(mkt.s30)} em cada 100). A anunciar, conta com um segundo ciclo e com ter de ceder; a comprar, o tempo joga a teu favor — quem já lá está há semanas ouve uma proposta.`
      : vsMkt === "media" ? `É o ritmo médio deste mercado (${liqPct(mkt.s30)} em cada 100).` : "";

  const horizonRows = LIQ_HORIZONS.filter(d => lq[`s${d}`] != null).map(d => `<tr>
      <td>${d} dias</td>
      <td>${liqPct(lq[`s${d}`])} em cada 100</td>
      <td class="mut">${liqPct(1 - lq[`s${d}`])} ainda à venda</td></tr>`).join("");

  const pb = lq.pb || [];
  const cheap = pb.length ? pb[0] : null, dear = pb.length > 1 ? pb[pb.length - 1] : null;
  const spread = (cheap && dear && cheap.k !== dear.k && Math.abs(cheap.s30 - dear.s30) >= 0.05);
  const priceLine = spread
    ? `<p class="fc-p">O preço é o que mais mexe com isto: na faixa <b>${escapeHtml(cheap.lbl.toLowerCase())}</b> saem ${liqPct(cheap.s30)} em cada 100 no primeiro mês; na faixa <b>${escapeHtml(dear.lbl.toLowerCase())}</b> são ${liqPct(dear.s30)}. Não é o mesmo carro a demorar mais: é outro comprador, com mais alternativas e menos pressa.</p>`
    : "";

  const ab = lq.ab || [];
  const dt = lq.dt || [];
  const districtLine = dt.length >= 2
    ? `<p class="fc-p">Entre ${escapeHtml(dt[0].lbl)} e ${escapeHtml(dt[dt.length - 1].lbl)} a diferença no primeiro mês é de ${liqPct(dt[0].s30)}% para ${liqPct(dt[dt.length - 1].s30)}%. Onde há mais oferta há normalmente mais procura, por isso o distrito costuma mexer mais com o preço do que com o tempo — os preços locais estão em <a href="/precos/${encodeURIComponent(dt[0].k)}">carros usados ${emDistrito(dt[0].k, dt[0].lbl)}</a>.</p>`
    : "";

  const cutBlock = (lq.cu != null) ? `
    <section class="section fc-wrap">
      <h2 class="fc-h2">Quantos acabam por baixar o preço</h2>
      <p class="fc-p"><b>${liqPct(lq.cu)} em cada 100</b> anúncios de ${B} ${M} que acompanhámos até ao fim desceram o preço pelo caminho${lq.cp != null ? `, com um corte mediano de <b>${liqPct(lq.cp)}%</b>` : ""}. É a margem que este modelo costuma ceder, e a referência mais honesta que temos para o que dá para negociar.</p>
      ${(lq.cd != null && lq.hd != null) ? `<p class="fc-p">Os anúncios que baixaram estiveram no ar ${lq.cd} dias, contra ${lq.hd} dos que nunca mexeram no preço. Lê-se na direção certa: baixa-se o preço porque o anúncio está parado, não fica parado por se ter baixado o preço.</p>` : ""}
    </section>` : "";

  const relistBlock = (lq.rb != null) ? `<p class="fc-p">De uns quantos sabemos que não venderam: pelo menos <b>${liqPct(lq.rb)}%</b> reapareceram semanas depois como anúncio novo do mesmo carro, que conseguimos emparelhar com o anterior. É um mínimo e não uma taxa — só contamos os reaparecimentos que identificámos, e quem saiu na semana passada ainda não teve tempo de voltar.</p>` : "";

  const body = crumbs([
    { name: "Início", href: "/" }, { name: "Tempo de venda", href: "/liquidez" },
    { name: `${rec.b} ${rec.m}` },
  ]) + `
    <div style="padding-top:14px;">
      <div class="side-card" style="max-width:680px;margin:0 auto;">
        <div class="eyebrow" style="margin-bottom:14px;"><span class="e-dot"></span><span class="mono">TEMPO DE VENDA · OLX PORTUGAL</span></div>
        <h1 class="fc-h1">Quanto tempo demora a vender um ${B} ${M}?</h1>
        <p class="lede" style="font-size:16px;margin:0 0 18px;">Acompanhámos <b>${fmtNum(lq.n)} anúncios</b> de ${B} ${M} até saírem do OLX${lq.cn ? `, mais ${fmtNum(lq.cn)} que ainda lá estão` : ""}: <b>${s30} em cada 100 desaparecem no primeiro mês</b>${lq.md != null ? `, e a mediana está nos <b>${lq.md} dias</b>` : ""}. ${verdict}</p>
        <div class="fc-stat-row">
          <div class="fc-stat"><div class="k">EM 30 DIAS</div><div class="v">${s30}%</div><div class="s">${mkt.s30 != null ? `mercado: ${liqPct(mkt.s30)}%` : "dos anúncios saem"}</div></div>
          ${lq.md != null ? `<div class="fc-stat"><div class="k">MEDIANA</div><div class="v">${lq.md} d</div><div class="s">${(lq.q1 != null && lq.q3 != null) ? `metade sai entre ${lq.q1} e ${lq.q3} dias` : "até sair do OLX"}</div></div>` : ""}
          ${still90 != null ? `<div class="fc-stat"><div class="k">AOS 90 DIAS</div><div class="v">${still90}%</div><div class="s">ainda à venda</div></div>` : ""}
          ${lq.rb != null ? `<div class="fc-stat"><div class="k">VOLTAM A ANUNCIAR</div><div class="v">${liqPct(lq.rb)}%</div><div class="s">estes não venderam</div></div>` : ""}
        </div>
        ${provenance({ n: lq.n, builtAt, unit: "anúncios acompanhados até saírem",
                       measureId: "days-on-market-km",
                       measure: "Dias entre o anúncio aparecer no OLX e o último ciclo que o viu no ar",
                       extra: `Kaplan-Meier, com ${lq.cn ? fmtNum(lq.cn) + " anúncios" : "os anúncios"} ainda à venda contados como censurados` })}
      </div>
    </div>
    <section class="section fc-wrap">
      <h2 class="fc-h2">A que ritmo saem</h2>
      ${liquidityChart(lq)}
      <div class="fc-scroll" style="margin-top:12px;"><table class="fc-tbl">
        <thead><tr><th>Ao fim de</th><th>Já saíram</th><th>Continuam no ar</th></tr></thead>
        <tbody>${horizonRows}</tbody></table></div>
      <p class="fc-p" style="margin-top:12px;">A conta inclui os anúncios que ainda estão à venda, não só os que já acabaram. É a diferença entre medir o mercado e medir apenas os anúncios que tiveram pressa — olhar só para os que já saíram encurta o resultado em cerca de dez dias.</p>
    </section>
    <section class="section fc-wrap">
      <h2 class="fc-h2">Sair do OLX não é o mesmo que vender</h2>
      <p class="fc-p">Um anúncio do OLX corre em ciclos de 30 dias, e isso vê-se nos dados: há uma acumulação de saídas exatamente nesse dia, em todos os modelos. Um anúncio que acaba aí tanto pode ter vendido como ter expirado sem que ninguém o renovasse, e nós não distinguimos as duas coisas — quem sabe é o vendedor. É por isso que a primeira frase desta página é a percentagem que sai no primeiro mês e não uma mediana: a mediana cai dentro desse degrau em quase todos os modelos e acaba a descrever o ciclo do OLX em vez do ${B} ${M}.</p>
      ${relistBlock}
    </section>
    ${pb.length ? `<section class="section fc-wrap">
      <h2 class="fc-h2">Por faixa de preço</h2>
      ${priceLine}
      ${liqCutTable(pb, "Faixa de preço")}
    </section>` : ""}
    ${ab.length ? `<section class="section fc-wrap">
      <h2 class="fc-h2">Por idade do carro</h2>
      <p class="fc-p">Idade e preço andam juntos neste mercado, por isso esta tabela lê-se com a de cima: em boa parte é o mesmo efeito visto do outro lado.</p>
      ${liqCutTable(ab, "Idade")}
    </section>` : ""}
    ${dt.length ? `<section class="section fc-wrap">
      <h2 class="fc-h2">Por distrito</h2>
      ${districtLine}
      ${liqCutTable(dt, "Distrito")}
    </section>` : ""}
    ${cutBlock}
    <section class="section fc-wide">
      <div class="cta-banner">
        <div style="flex:1 1 360px;">
          <h2>Vais anunciar o teu ${B} ${M}?</h2>
          <p>O tempo que vai demorar depende sobretudo do preço a que o pões. Cola o link do anúncio e dizemos-te onde ele está em relação ao valor justo.</p>
        </div>
        <a class="btn-bright" href="/avaliar?modelo=${encodeURIComponent(slug)}">Avaliar o meu carro&nbsp;&nbsp;→</a>
      </div>
    </section>
    <section class="section fc-wrap" style="padding-bottom:70px;">
      <p class="fc-p"><a href="/preco/${slug}">Preços de ${B} ${M} por ano</a>${hasDepreciation ? ` · <a href="/depreciacao/${slug}">Quanto se desvaloriza</a>` : ""} · <a href="/liquidez">Tempo de venda de outros modelos</a> · <a href="/sobrevalorizados">Onde se pede acima do valor justo</a> · <a href="${canonical}.json">Dados em JSON</a></p>
    </section>`;

  const faqs = [
    [`Quanto tempo demora a vender um ${rec.b} ${rec.m} em Portugal?`,
     `${s30} em cada 100 anúncios de ${rec.b} ${rec.m} saem do OLX no primeiro mês${lq.md != null ? `, e a mediana é de ${lq.md} dias` : ""}, medido em ${lq.n} anúncios acompanhados até ao fim${lq.cn ? ` e ${lq.cn} ainda à venda` : ""}. Sair do OLX não prova a venda: um anúncio corre em ciclos de 30 dias e pode expirar sem ter vendido.`],
  ];
  if (spread) faqs.push([
    `O preço muda o tempo que um ${rec.b} ${rec.m} demora a vender?`,
    `Muda. Na faixa ${cheap.lbl.toLowerCase()} saem ${liqPct(cheap.s30)} em cada 100 no primeiro mês; na faixa ${dear.lbl.toLowerCase()}, ${liqPct(dear.s30)}. Medido nos mesmos anúncios de ${rec.b} ${rec.m} do OLX Portugal.`]);
  if (lq.cu != null) faqs.push([
    `Quanto se costuma baixar no preço de um ${rec.b} ${rec.m}?`,
    `${liqPct(lq.cu)}% dos anúncios de ${rec.b} ${rec.m} que acompanhámos baixaram o preço antes de sair${lq.cp != null ? `, com um corte mediano de ${liqPct(lq.cp)}%` : ""}. É a margem que este modelo costuma ceder, e o ponto de partida para negociar.`]);
  if (lq.rb != null) faqs.push([
    `Os anúncios de ${rec.b} ${rec.m} que desaparecem do OLX foram todos vendidos?`,
    `Não. Pelo menos ${liqPct(lq.rb)}% dos que saíram voltaram a aparecer depois como anúncio novo do mesmo carro, ou seja não tinham vendido. É um mínimo: só contamos os reaparecimentos que conseguimos emparelhar.`]);

  return layout({
    title: `${rec.b} ${rec.m}: quanto tempo demora a vender`,
    description: `${s30}% dos anúncios de ${rec.b} ${rec.m} saem do OLX Portugal no primeiro mês${lq.md != null ? ` e a mediana é de ${lq.md} dias` : ""}, medido em ${lq.n} anúncios acompanhados ao longo do tempo. Por faixa de preço, idade e distrito.`,
    canonical, body, zone: "all", nav: "precos", depositCount, index: true, host,
    altJson: `${canonical}.json`,
    jsonLd: {
      "@context": "https://schema.org",
      "@graph": [
        {
          "@type": "Dataset", "license": licenseUrl(host), "url": canonical, "inLanguage": "pt-PT",
          "name": `Tempo até vender de ${rec.b} ${rec.m} em Portugal`,
          "description": `Dias que um anúncio de ${rec.b} ${rec.m} fica no OLX Portugal antes de sair, estimado por Kaplan-Meier sobre ${lq.n} anúncios terminados${lq.cn ? ` e ${lq.cn} ainda ativos` : ""}, com cortes por faixa de preço, idade e distrito.`,
          "creator": { "@type": "Organization", "name": "Carsbuyer", "url": `https://${host}/` },
          "isAccessibleForFree": true, "dateModified": builtAt || undefined,
          "variableMeasured": ["Dias até o anúncio sair (mediana)", "Percentagem que sai em 30 dias"],
          "distribution": [{ "@type": "DataDownload", "encodingFormat": "application/json", "contentUrl": `${canonical}.json` }],
        },
        breadcrumbLd(host, [
          { name: "Início", href: "/" }, { name: "Tempo de venda", href: "/liquidez" },
          { name: `${rec.b} ${rec.m}` },
        ]),
        faqLd(faqs),
      ],
    },
  });
}

export function renderLiquidityHub({ rows, market, host, depositCount, builtAt }) {
  const canonical = `https://${host}/liquidez`;
  const mkt = market || {};
  const withCurve = rows.filter(r => r.lq);
  const name = r => `<a href="${r.page ? `/liquidez/${r.slug}` : `/preco/${r.slug}`}" style="color:#177A47;font-weight:600;">${escapeHtml(r.b)} ${escapeHtml(r.m)}</a>`;
  const tr = rows.map(r => r.lq ? `<tr>
      <td>${name(r)}</td>
      <td>${liqPct(r.lq.s30)}%</td>
      <td class="mut">${r.lq.md != null ? `${r.lq.md} dias` : "—"}</td>
      <td class="mut">${mkt.s30 ? (r.lq.s30 > mkt.s30 ? `${Math.round((r.lq.s30 / mkt.s30 - 1) * 100)}% acima` : r.lq.s30 < mkt.s30 ? `${Math.round((1 - r.lq.s30 / mkt.s30) * 100)}% abaixo` : "na média") : "—"}</td>
      <td class="mut">${fmtNum(r.lq.n)}</td>
      <td class="mut">${fmtEur(r.fm)}</td></tr>` : `<tr>
      <td>${name(r)}</td>
      <td class="mut">—</td>
      <td class="mut">${r.sd != null ? `${r.sd} dias` : "—"}</td>
      <td class="mut">—</td>
      <td class="mut">${r.sn != null ? fmtNum(r.sn) : "—"}</td>
      <td class="mut">${fmtEur(r.fm)}</td></tr>`).join("");
  const fastest = withCurve[0], slowest = withCurve[withCurve.length - 1];
  const body = crumbs([{ name: "Início", href: "/" }, { name: "Tempo de venda" }]) + `
    <section class="section fc-wrap" style="padding-top:16px;">
      <h1 class="fc-h1">Quanto tempo demora a vender cada carro em Portugal</h1>
      <p class="fc-p">Quantos anúncios de cada modelo saem do OLX no primeiro mês, e ao fim de quantos dias sai metade. Acompanhamos os anúncios ao longo do tempo — isto não é estimado a partir do preço, é o que aconteceu — e a conta inclui os que ainda estão à venda, que é o que a impede de ficar curta.</p>
      ${(fastest && slowest && fastest !== slowest) ? `<p class="fc-p">Do mais rápido ao mais lento: um ${escapeHtml(fastest.b)} ${escapeHtml(fastest.m)} sai no primeiro mês em <b>${liqPct(fastest.lq.s30)}%</b> dos casos, um ${escapeHtml(slowest.b)} ${escapeHtml(slowest.m)} em <b>${liqPct(slowest.lq.s30)}%</b>. Se vais anunciar, é a diferença entre pedir o preço todo e ter de ceder; se vais comprar, é onde tens margem para negociar.</p>` : ""}
      ${mkt.s30 != null ? `<p class="fc-p">No conjunto do mercado saem <b>${liqPct(mkt.s30)} em cada 100</b> no primeiro mês${mkt.md != null ? `, com uma mediana de <b>${mkt.md} dias</b>` : ""}. Um anúncio do OLX corre em ciclos de 30 dias e muitos desaparecem exatamente aí, por isso desaparecer não prova que vendeu.</p>` : ""}
      <div class="fc-scroll"><table class="fc-tbl">
        <thead><tr><th>Modelo</th><th>Sai em 30 dias</th><th>Mediana</th><th>vs. mercado</th><th>Anúncios observados</th><th>Preço mediano</th></tr></thead>
        <tbody>${tr}</tbody></table></div>
      ${provenance({ n: rows.reduce((s, r) => s + ((r.lq && r.lq.n) || r.sn || 0), 0), builtAt,
                     unit: "anúncios acompanhados até saírem", measureId: "days-on-market-km",
                     measure: "Dias entre o anúncio aparecer no OLX e o último ciclo que o viu no ar" })}
      <p class="fc-p" style="margin-top:18px;"><a href="/precos">Preços por modelo</a> · <a href="/depreciacao">Desvalorização</a> · <a href="/sobrevalorizados">Pedido vs. valor justo</a> · <a href="/metodologia">Como medimos</a></p>
    </section>
    <div style="height:60px;"></div>`;
  return layout({
    title: "Quanto tempo demora a vender um carro em Portugal",
    description: `Quantos anúncios de cada modelo saem do OLX Portugal no primeiro mês${mkt.s30 != null ? ` (mercado: ${liqPct(mkt.s30)}%)` : ""} e ao fim de quantos dias sai metade. Medido em anúncios reais acompanhados ao longo do tempo.`,
    canonical, body, zone: "all", nav: "precos", depositCount, index: true, host,
    jsonLd: {
      "@context": "https://schema.org",
      "@graph": [
        {
          "@type": "Dataset", "license": licenseUrl(host), "url": canonical, "inLanguage": "pt-PT",
          "name": "Tempo até vender, por modelo (Portugal)",
          "description": "Percentagem de anúncios que saem no primeiro mês e dias medianos até sair, por modelo, no OLX Portugal.",
          "creator": { "@type": "Organization", "name": "Carsbuyer", "url": `https://${host}/` },
          "isAccessibleForFree": true, "dateModified": builtAt || undefined,
          "variableMeasured": ["Percentagem que sai em 30 dias", "Dias até sair (mediana)"],
        },
        breadcrumbLd(host, [{ name: "Início", href: "/" }, { name: "Tempo de venda" }]),
      ],
    },
  });
}

// ═══ /sobrevalorizados — asking price against our own estimate ═══════════════
//
// The mirror of /mercado: that page finds individual listings below fair value,
// this one finds where a whole MODEL is systematically asked above (or below)
// what we estimate it is worth. Both directions on one page on purpose — a list
// of only "overpriced" reads as an accusation, and the underpriced half is the
// actionable one for a buyer.
export function renderValuationGap({ over, under, market, stats, host, depositCount, builtAt }) {
  const canonical = `https://${host}/sobrevalorizados`;
  const mkt = market || {};
  const row = r => `<tr>
      <td><a href="/preco/${r.slug}" style="color:#177A47;font-weight:600;">${escapeHtml(r.b)} ${escapeHtml(r.m)}</a></td>
      <td>${fmtEur(r.fm)}</td>
      <td class="mut">${fmtEur(r.gm)}</td>
      <td>${r.gap > 0 ? "+" : ""}${Math.round(r.gap * 100)}%</td>
      <td class="mut">${r.s30 != null ? `${r.page ? `<a href="/liquidez/${r.slug}" style="color:#177A47;font-weight:600;">${liqPct(r.s30)}%</a>` : `${liqPct(r.s30)}%`}` : "—"}</td>
      <td class="mut">${r.n}</td></tr>`;
  const wanted = over.filter(r => r.s30 != null && mkt.s30 != null && r.s30 >= mkt.s30 * 1.12);
  const stuck = over.filter(r => r.s30 != null && mkt.s30 != null && r.s30 <= mkt.s30 * 0.88);
  const readLine = (mkt.s30 != null && (wanted.length || stuck.length))
    ? `<p class="fc-p">A coluna do tempo de venda separa duas coisas que a percentagem sozinha confunde. ${wanted.length ? `Pede-se acima da estimativa e mesmo assim sai depressa — ${wanted.slice(0, 3).map(r => `<b>${escapeHtml(r.b)} ${escapeHtml(r.m)}</b>`).join(", ")} — e aí o prémio é procura a sério: o mercado paga-o e não vais negociá-lo para baixo com uma tabela na mão. ` : ""}${stuck.length ? `Pede-se acima da estimativa <i>e</i> fica no mercado — ${stuck.slice(0, 3).map(r => `<b>${escapeHtml(r.b)} ${escapeHtml(r.m)}</b>`).join(", ")} — e aí é preço a mais à espera de descer.` : ""}</p>`
    : "";
  const body = crumbs([{ name: "Início", href: "/" }, { name: "Preço pedido vs. valor justo" }]) + `
    <section class="section fc-wrap" style="padding-top:16px;">
      <h1 class="fc-h1">Onde o preço pedido se afasta do valor justo</h1>
      <p class="fc-p">Para cada modelo comparamos o que o mercado <b>pede</b> com o que o nosso modelo <b>estima</b> que vale, para quilometragem e versões típicas desse modelo. Um desvio grande não significa que alguém esteja a enganar ninguém: significa que a oferta e a procura desse modelo estão desalinhadas neste momento, e é aí que se negoceia.</p>
      ${stats.gapMed != null ? `<p class="fc-p">No conjunto do mercado, o preço pedido está ${stats.gapMed >= 0 ? "acima" : "abaixo"} da estimativa em <b>${Math.abs(Math.round(stats.gapMed * 100))}%</b> na mediana — é o normal, e é a referência contra a qual ler a tabela.</p>` : ""}
      <h2 class="fc-h2">Pedem mais do que estimamos</h2>
      <p class="fc-p">Se vais comprar, entra a negociar. Se vais vender, o mercado está a teu favor.</p>
      <div class="fc-scroll"><table class="fc-tbl">
        <thead><tr><th>Modelo</th><th>Pedido (mediana)</th><th>Valor justo estimado</th><th>Desvio</th><th>Sai em 30 dias</th><th>Anúncios</th></tr></thead>
        <tbody>${over.map(row).join("")}</tbody></table></div>
      ${readLine}
      <h2 class="fc-h2">Pedem menos do que estimamos</h2>
      <p class="fc-p">Oferta a mais ou procura fraca. Bom momento para comprar, mau para anunciar.</p>
      <div class="fc-scroll"><table class="fc-tbl">
        <thead><tr><th>Modelo</th><th>Pedido (mediana)</th><th>Valor justo estimado</th><th>Desvio</th><th>Sai em 30 dias</th><th>Anúncios</th></tr></thead>
        <tbody>${under.map(row).join("")}</tbody></table></div>
      ${provenance({ n: null, builtAt, measure: "Preço pedido mediano vs. valor justo estimado pelo modelo" })}
      <p class="fc-p" style="margin-top:18px;">A estimativa só é publicada onde passa os nossos limites de fiabilidade — ver <a href="/metodologia">metodologia</a>. Modelos onde não passa não aparecem aqui.</p>
      <p class="fc-p"><a href="/mercado">Anúncios concretos abaixo do valor justo</a> · <a href="/precos">Todos os modelos</a></p>
    </section>
    <div style="height:60px;"></div>`;
  return layout({
    title: "Preço pedido vs. valor justo por modelo",
    description: "Que modelos de carros usados em Portugal são pedidos acima ou abaixo do valor justo estimado, a partir de anúncios ativos do OLX.",
    canonical, body, zone: "all", nav: "precos", depositCount, index: true, host,
    jsonLd: {
      "@context": "https://schema.org",
      "@graph": [
        { "@type": "CollectionPage", "url": canonical, "inLanguage": "pt-PT", "name": "Preço pedido vs. valor justo por modelo" },
        breadcrumbLd(host, [{ name: "Início", href: "/" }, { name: "Preço pedido vs. valor justo" }]),
      ],
    },
  });
}

// ── ISO week arithmetic ──────────────────────────────────────────────────────
//
// One implementation, imported by index.js. Two copies of a week calculation is
// exactly how an archive ends up with a row labelled the wrong week.

/** "2026-W35" for a Date. Thursday-anchored, which is what makes the week that
 *  contains 1 January come out right: 28 Dec 2026 and 3 Jan 2027 are both W53. */
export function isoWeek(d) {
  const t = new Date(Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate()));
  const day = t.getUTCDay() || 7;
  t.setUTCDate(t.getUTCDate() + 4 - day);
  const yearStart = new Date(Date.UTC(t.getUTCFullYear(), 0, 1));
  const week = Math.ceil((((t - yearStart) / 86400000) + 1) / 7);
  return `${t.getUTCFullYear()}-W${String(week).padStart(2, "0")}`;
}

/** The Monday of an ISO week label. ISO week 1 is the one containing 4 January. */
export function isoWeekStart(wk) {
  const m = /^(\d{4})-W(\d{2})$/.exec(wk || "");
  if (!m) return null;
  const jan4 = new Date(Date.UTC(+m[1], 0, 4));
  const day = jan4.getUTCDay() || 7;
  const week1Mon = new Date(jan4);
  week1Mon.setUTCDate(jan4.getUTCDate() - day + 1);
  const out = new Date(week1Mon);
  out.setUTCDate(week1Mon.getUTCDate() + (+m[2] - 1) * 7);
  return out;
}

/**
 * Weeks between the earliest recorded one and `upTo` that the archive lacks.
 *
 * They are NOT backfilled, and that is deliberate. The numbers for a week that
 * has passed no longer exist; writing today's figures under last week's label
 * would be inventing data, which is the one thing this site is built not to do.
 * So a gap stays a gap, the page says so, and the cron is what stops new ones.
 */
export function missingWeeks(history, upTo) {
  const have = new Set((history || []).map(h => h.week));
  if (!have.size || !upTo) return [];
  let cur = isoWeekStart([...have].sort()[0]);
  const end = isoWeekStart(upTo);
  if (!cur || !end) return [];
  const gaps = [];
  while (cur < end) {
    cur = new Date(cur.getTime() + 7 * 86400000);
    const wk = isoWeek(cur);
    if (wk !== upTo && !have.has(wk)) gaps.push(wk);
  }
  return gaps;
}

export function isoWeekMonth(wk) {
  const mon = isoWeekStart(wk);
  if (!mon) return null;
  const thu = new Date(mon.getTime() + 3 * 86400000);
  return `${thu.getUTCFullYear()}-${String(thu.getUTCMonth() + 1).padStart(2, "0")}`;
}

export function weeksOfMonth(month) {
  const m = /^(\d{4})-(\d{2})$/.exec(month || "");
  if (!m || +m[2] < 1 || +m[2] > 12) return [];
  const out = [];
  const d = new Date(Date.UTC(+m[1], +m[2] - 1, 1));
  const end = new Date(Date.UTC(+m[1], +m[2], 0));
  for (; d <= end; d.setUTCDate(d.getUTCDate() + 1)) {
    if (d.getUTCDay() === 4) out.push(isoWeek(d));
  }
  return out;
}

const MONTH_NAMES_PT = ["janeiro", "fevereiro", "março", "abril", "maio", "junho",
  "julho", "agosto", "setembro", "outubro", "novembro", "dezembro"];

export function monthLabel(month) {
  const m = /^(\d{4})-(\d{2})$/.exec(month || "");
  if (!m || +m[2] < 1 || +m[2] > 12) return month || "";
  return `${MONTH_NAMES_PT[+m[2] - 1]} de ${m[1]}`;
}

export const IDX_MIN_MONTH_WEEKS = 2;

function medianOf(values) {
  const v = (values || []).filter(x => x != null && Number.isFinite(x)).sort((a, b) => a - b);
  if (!v.length) return null;
  const mid = v.length >> 1;
  return v.length % 2 ? v[mid] : (v[mid - 1] + v[mid]) / 2;
}

export function monthlyCuts(history, currentWeek = null) {
  const openMonth = currentWeek ? isoWeekMonth(currentWeek) : null;
  const byMonth = new Map();
  for (const h of history || []) {
    const month = isoWeekMonth(h.week);
    if (!month) continue;
    if (openMonth && month >= openMonth) continue;
    if (!byMonth.has(month)) byMonth.set(month, []);
    byMonth.get(month).push(h);
  }
  const cuts = [];
  for (const [month, rows] of byMonth) {
    if (rows.length < IDX_MIN_MONTH_WEEKS) continue;
    rows.sort((a, b) => a.week < b.week ? -1 : 1);
    const med = (k, round = false) => {
      const v = medianOf(rows.map(r => r[k]));
      return v == null ? null : (round ? Math.round(v) : v);
    };
    const first = isoWeekStart(rows[0].week);
    const last = isoWeekStart(rows[rows.length - 1].week);
    const all = weeksOfMonth(month);
    cuts.push({
      month, rows, weeks: rows.map(r => r.week), n: rows.length, monthWeeks: all.length,
      missing: all.filter(w => !rows.some(r => r.week === w)),
      from: first ? first.toISOString().slice(0, 10) : null,
      to: last ? new Date(last.getTime() + 6 * 86400000).toISOString().slice(0, 10) : null,
      priceMed: med("priceMed", true), listings: med("listings", true), models: med("models", true),
      sellMed: med("sellMed", true), kmMed: med("kmMed", true), depMed: med("depMed"),
      builtAt: rows[rows.length - 1].builtAt || null,
    });
  }
  return cuts.sort((a, b) => a.month < b.month ? -1 : 1);
}

// ═══ /mercado/indice — the market index, with a permanent weekly archive ═════
//
// Journalists and forums link to a number they can cite with a date. A page whose
// figures change under the link is not citable, so every week gets its OWN
// permanent URL (/mercado/indice/2026-W35) that never changes again, and the
// bare /mercado/indice always shows the latest plus the trend.
export function renderMarketIndex({ snapshot, history, host, depositCount, isArchive = false, currentWeek = null, gaps = [], months = [] }) {
  const wk = snapshot.week;                 // display form, ISO: "2026-W35"
  // URL form is lower-case, because the router normalises every public path to
  // lower case and a canonical that disagreed with its own URL would 301 to
  // itself. The page still SHOWS the ISO spelling.
  const wkSlug = wk.toLowerCase();
  // The week's own address is permanent and stays what the page tells people to
  // cite, whether or not it is the canonical at this moment.
  const permalink = `https://${host}/mercado/indice/${wkSlug}`;
  // While a week is still the CURRENT week, its archive page and the bare
  // /mercado/indice are the same cut. Two self-canonical URLs over identical
  // numbers split the signal, and Search Console showed the symptom: the hub URL
  // listed under the archive's title. So the live cut defers to the hub and only
  // becomes its own canonical once the week closes and the hub moves on.
  const isLiveCut = isArchive && currentWeek != null && wk === currentWeek;
  const canonical = (isArchive && !isLiveCut)
    ? permalink
    : `https://${host}/mercado/indice`;
  const prev = history.filter(h => h.week < wk).sort((a, b) => a.week < b.week ? 1 : -1)[0] || null;
  const delta = (now, then, fmt) => {
    if (then == null || now == null || !then) return "";
    const d = (now - then) / then;
    if (Math.abs(d) < 0.005) return `<div class="s">estável vs. semana anterior</div>`;
    return `<div class="s" style="color:${d > 0 ? "#B4551F" : "#177A47"};">${d > 0 ? "+" : ""}${(d * 100).toFixed(1)}% vs. semana anterior</div>`;
  };
  const rows = history.slice().sort((a, b) => a.week < b.week ? 1 : -1).slice(0, 26).map(h => `<tr>
      <td>${h.week === wk && !isArchive ? `<b>${escapeHtml(h.week)}</b>` : `<a href="/mercado/indice/${escapeHtml(h.week.toLowerCase())}" style="color:#177A47;font-weight:600;">${escapeHtml(h.week)}</a>`}</td>
      <td class="mut">${escapeHtml(h.date || "")}</td>
      <td>${fmtEur(h.priceMed)}</td>
      <td class="mut">${fmtNum(h.listings)}</td>
      <td class="mut">${h.models}</td>
      <td class="mut">${h.sellMed != null ? h.sellMed + " dias" : "—"}</td></tr>`).join("");


  const monthRows = months.slice().sort((a, b) => a.month < b.month ? 1 : -1).slice(0, 24).map(c => `<tr>
      <td><a href="/mercado/indice/${escapeHtml(c.month)}" style="color:#177A47;font-weight:600;">${escapeHtml(monthLabel(c.month))}</a></td>
      <td class="mut">${escapeHtml(c.from || "")} — ${escapeHtml(c.to || "")}</td>
      <td>${fmtEur(c.priceMed)}</td>
      <td class="mut">${fmtNum(c.listings)}</td>
      <td class="mut">${c.sellMed != null ? c.sellMed + " dias" : "—"}</td>
      <td class="mut">${c.n}/${c.monthWeeks}</td></tr>`).join("");

  const body = crumbs(isArchive
    ? [{ name: "Início", href: "/" }, { name: "Índice de mercado", href: "/mercado/indice" }, { name: wk }]
    : [{ name: "Início", href: "/" }, { name: "Índice de mercado" }]) + `
    <section class="section fc-wrap" style="padding-top:16px;">
      <div class="eyebrow" style="margin-bottom:14px;"><span class="e-dot"></span><span class="mono">SEMANA ${escapeHtml(wk)} · ${escapeHtml(snapshot.date || "")}</span></div>
      <h1 class="fc-h1">Índice do mercado de usados em Portugal${isArchive ? ` — ${escapeHtml(wk)}` : ""}</h1>
      <p class="fc-p">Retrato semanal do que está à venda no OLX Portugal: quanto se pede, quanto há e quanto demora a sair.${isArchive ? " Este é o registo permanente desta semana — os números desta página não voltam a mudar." : " Cada semana e cada mês fechado ficam guardados num endereço próprio, para poderes citar um número com data."}</p>
      <div class="fc-stat-row" style="margin:20px 0 8px;">
        <div class="fc-stat"><div class="k">PREÇO MEDIANO</div><div class="v">${fmtEur(snapshot.priceMed)}</div>${delta(snapshot.priceMed, prev && prev.priceMed)}</div>
        <div class="fc-stat"><div class="k">ANÚNCIOS ATIVOS</div><div class="v">${fmtNum(snapshot.listings)}</div>${delta(snapshot.listings, prev && prev.listings)}</div>
        <div class="fc-stat"><div class="k">MODELOS COBERTOS</div><div class="v">${snapshot.models}</div><div class="s">com amostra suficiente</div></div>
        <div class="fc-stat"><div class="k">DIAS ATÉ VENDER</div><div class="v">${snapshot.sellMed != null ? snapshot.sellMed : "—"}</div><div class="s">mediana do mercado</div></div>
        <div class="fc-stat"><div class="k">KM MEDIANO</div><div class="v">${snapshot.kmMed != null ? fmtNum(snapshot.kmMed) : "—"}</div><div class="s">à venda</div></div>
        <div class="fc-stat"><div class="k">DESVALORIZAÇÃO</div><div class="v">${snapshot.depMed != null ? Math.round(snapshot.depMed * 100) + "%" : "—"}</div><div class="s">por ano de idade</div></div>
      </div>
      ${provenance({ n: snapshot.listings, builtAt: snapshot.builtAt, measure: "Mediana das medianas de preço pedido por modelo" })}
    </section>
    ${rows ? `<section class="section fc-wrap">
      <h2 class="fc-h2">Histórico semanal</h2>
      <div class="fc-scroll"><table class="fc-tbl">
        <thead><tr><th>Semana</th><th>Data</th><th>Preço mediano</th><th>Anúncios</th><th>Modelos</th><th>Dias até vender</th></tr></thead>
        <tbody>${rows}</tbody></table></div>
      <p class="fc-p" style="margin-top:12px;">O histórico começa na semana em que passámos a guardar os cortes. Cresce uma linha por semana, e nenhuma linha antiga é reescrita.</p>
      ${gaps.length ? `<p class="fc-p" style="color:#B4551F;">Faltam ${gaps.length} semana${gaps.length === 1 ? "" : "s"} no histórico: ${gaps.map(escapeHtml).join(", ")}. Não as preenchemos, e é de propósito: os números dessas semanas já não existem, e escrever os de hoje com a data de então seria inventá-los.</p>` : ""}
    </section>` : ""}
    ${rows ? `<section class="section fc-wrap">
      <h2 class="fc-h2">Arquivo mensal</h2>
      ${monthRows ? `<p class="fc-p">Cada mês fechado tem o seu próprio endereço permanente, com a mediana dos cortes semanais desse mês. É o corte a citar quando a frase é sobre um mês e não sobre uma semana.</p>
      <div class="fc-scroll"><table class="fc-tbl">
        <thead><tr><th>Mês</th><th>Período</th><th>Preço mediano</th><th>Anúncios</th><th>Dias até vender</th><th>Semanas</th></tr></thead>
        <tbody>${monthRows}</tbody></table></div>
      <p class="fc-p" style="margin-top:12px;">O período é o das semanas ISO que fecham dentro do mês, por isso não coincide com o dia 1 nem com o último dia. A última coluna diz quantas semanas do mês entraram no cálculo.</p>`
        : `<p class="fc-p">Ainda não há nenhum mês fechado com pelo menos ${IDX_MIN_MONTH_WEEKS} cortes semanais guardados. O primeiro abre assim que houver, no endereço <span class="mono">/mercado/indice/{AAAA}-{MM}</span>, e também não volta a mudar.</p>`}
    </section>` : ""}
    <section class="section fc-wrap" style="padding-bottom:70px;">
      <h2 class="fc-h2">Podes citar isto</h2>
      <p class="fc-p">Estes números podem ser usados com atribuição a Carsbuyer e indicação da data — mudam todas as semanas, por isso a data faz parte do número. ${isArchive ? `Endereço permanente desta semana: <span class="mono fc-url">${escapeHtml(permalink)}</span>.` : ""}</p>
      <p class="fc-p"><a href="/precos">Preços por modelo</a> · <a href="/liquidez">Tempo de venda</a> · <a href="/sobrevalorizados">Pedido vs. valor justo</a> · <a href="/metodologia">Metodologia</a></p>
    </section>`;

  return layout({
    title: isArchive
      ? `Índice do mercado de usados em Portugal — ${wk}`
      : "Índice do mercado de carros usados em Portugal",
    description: `Semana ${wk}: preço mediano ${fmtEur(snapshot.priceMed)} em ${fmtNum(snapshot.listings)} anúncios ativos de ${snapshot.models} modelos no OLX Portugal${snapshot.sellMed != null ? `, ${snapshot.sellMed} dias medianos até vender` : ""}.`,
    canonical, body, zone: "all", nav: "feed", depositCount, index: true, host,
    jsonLd: {
      "@context": "https://schema.org",
      "@graph": [
        {
          "@type": "Dataset", "license": licenseUrl(host), "url": canonical, "inLanguage": "pt-PT",
          "name": `Índice do mercado de carros usados em Portugal — ${wk}`,
          "description": "Preço pedido mediano, número de anúncios ativos, quilometragem mediana e dias medianos até vender no mercado português de carros usados.",
          "creator": { "@type": "Organization", "name": "Carsbuyer", "url": `https://${host}/` },
          "isAccessibleForFree": true,
          "temporalCoverage": snapshot.date || undefined,
          "dateModified": snapshot.builtAt || undefined,
          "variableMeasured": ["Preço pedido mediano (EUR)", "Anúncios ativos", "Dias até vender (mediana)"],
        },
        breadcrumbLd(host, isArchive
          ? [{ name: "Início", href: "/" }, { name: "Índice de mercado", href: "/mercado/indice" }, { name: wk }]
          : [{ name: "Início", href: "/" }, { name: "Índice de mercado" }]),
      ],
    },
  });
}

export function renderMarketMonth({ cut, months = [], host, depositCount }) {
  const label = monthLabel(cut.month);
  const permalink = `https://${host}/mercado/indice/${cut.month}`;
  const idx = months.findIndex(c => c.month === cut.month);
  const prev = idx > 0 ? months[idx - 1] : null;
  const next = idx >= 0 && idx < months.length - 1 ? months[idx + 1] : null;

  const delta = (now, then) => {
    if (!prev || then == null || now == null || !then) return "";
    const d = (now - then) / then;
    if (Math.abs(d) < 0.005) return `<div class="s">estável vs. ${escapeHtml(monthLabel(prev.month))}</div>`;
    return `<div class="s" style="color:${d > 0 ? "#B4551F" : "#177A47"};">${d > 0 ? "+" : ""}${(d * 100).toFixed(1)}% vs. ${escapeHtml(monthLabel(prev.month))}</div>`;
  };

  const weekRows = cut.rows.slice().sort((a, b) => a.week < b.week ? 1 : -1).map(h => `<tr>
      <td><a href="/mercado/indice/${escapeHtml(h.week.toLowerCase())}" style="color:#177A47;font-weight:600;">${escapeHtml(h.week)}</a></td>
      <td class="mut">${escapeHtml(h.date || "")}</td>
      <td>${fmtEur(h.priceMed)}</td>
      <td class="mut">${fmtNum(h.listings)}</td>
      <td class="mut">${h.sellMed != null ? h.sellMed + " dias" : "—"}</td></tr>`).join("");

  const crumbItems = [
    { name: "Início", href: "/" },
    { name: "Índice de mercado", href: "/mercado/indice" },
    { name: label },
  ];

  const body = crumbs(crumbItems) + `
    <section class="section fc-wrap" style="padding-top:16px;">
      <div class="eyebrow" style="margin-bottom:14px;"><span class="e-dot"></span><span class="mono">MÊS ${escapeHtml(cut.month)} · ${escapeHtml(cut.from || "")} — ${escapeHtml(cut.to || "")}</span></div>
      <h1 class="fc-h1">Índice do mercado de usados em Portugal — ${escapeHtml(label)}</h1>
      <p class="fc-p">Corte mensal do que estava à venda no OLX Portugal em ${escapeHtml(label)}: a mediana dos ${cut.n} cortes semanais desse mês. Este é o registo permanente do mês — os números desta página não voltam a mudar.</p>
      <div class="fc-stat-row" style="margin:20px 0 8px;">
        <div class="fc-stat"><div class="k">PREÇO MEDIANO</div><div class="v">${fmtEur(cut.priceMed)}</div>${delta(cut.priceMed, prev && prev.priceMed)}</div>
        <div class="fc-stat"><div class="k">ANÚNCIOS ATIVOS</div><div class="v">${fmtNum(cut.listings)}</div>${delta(cut.listings, prev && prev.listings)}</div>
        <div class="fc-stat"><div class="k">MODELOS COBERTOS</div><div class="v">${cut.models != null ? cut.models : "—"}</div><div class="s">com amostra suficiente</div></div>
        <div class="fc-stat"><div class="k">DIAS ATÉ VENDER</div><div class="v">${cut.sellMed != null ? cut.sellMed : "—"}</div><div class="s">mediana do mercado</div></div>
        <div class="fc-stat"><div class="k">KM MEDIANO</div><div class="v">${cut.kmMed != null ? fmtNum(cut.kmMed) : "—"}</div><div class="s">à venda</div></div>
        <div class="fc-stat"><div class="k">DESVALORIZAÇÃO</div><div class="v">${cut.depMed != null ? Math.round(cut.depMed * 100) + "%" : "—"}</div><div class="s">por ano de idade</div></div>
      </div>
      ${provenance({ n: cut.listings, builtAt: cut.builtAt, measure: `Mediana dos ${cut.n} cortes semanais de ${label}` })}
    </section>
    <section class="section fc-wrap">
      <h2 class="fc-h2">As semanas deste mês</h2>
      <div class="fc-scroll"><table class="fc-tbl">
        <thead><tr><th>Semana</th><th>Data</th><th>Preço mediano</th><th>Anúncios</th><th>Dias até vender</th></tr></thead>
        <tbody>${weekRows}</tbody></table></div>
      <p class="fc-p" style="margin-top:12px;">Entram as semanas ISO que fecham dentro do mês, por isso o período vai de ${escapeHtml(cut.from || "")} a ${escapeHtml(cut.to || "")} e não do dia 1 ao último dia.${cut.missing.length ? ` Falta${cut.missing.length === 1 ? "" : "m"} ${cut.missing.length} das ${cut.monthWeeks} semanas (${cut.missing.map(escapeHtml).join(", ")}): não ${cut.missing.length === 1 ? "a guardámos" : "as guardámos"} na altura e não ${cut.missing.length === 1 ? "a inventamos" : "as inventamos"} agora, por isso a mediana deste mês é a de ${cut.n} semanas.` : ` Estão cá as ${cut.monthWeeks} semanas do mês.`}</p>
    </section>
    <section class="section fc-wrap" style="padding-bottom:70px;">
      <h2 class="fc-h2">Podes citar isto</h2>
      <p class="fc-p">Estes números podem ser usados com atribuição a Carsbuyer e indicação do mês. Endereço permanente: <span class="mono fc-url">${escapeHtml(permalink)}</span>.</p>
      <p class="fc-p">${prev ? `<a href="/mercado/indice/${escapeHtml(prev.month)}" style="color:#177A47;font-weight:600;">← ${escapeHtml(monthLabel(prev.month))}</a> · ` : ""}<a href="/mercado/indice">Índice e semana atual</a>${next ? ` · <a href="/mercado/indice/${escapeHtml(next.month)}" style="color:#177A47;font-weight:600;">${escapeHtml(monthLabel(next.month))} →</a>` : ""}</p>
      <p class="fc-p"><a href="/precos">Preços por modelo</a> · <a href="/liquidez">Tempo de venda</a> · <a href="/sobrevalorizados">Pedido vs. valor justo</a> · <a href="/metodologia">Metodologia</a></p>
    </section>`;

  return layout({
    title: `Índice do mercado de usados em Portugal — ${label}`,
    description: `${label}: preço mediano ${fmtEur(cut.priceMed)} em ${fmtNum(cut.listings)} anúncios ativos no OLX Portugal${cut.sellMed != null ? `, ${cut.sellMed} dias medianos até vender` : ""} — mediana de ${cut.n} cortes semanais.`,
    canonical: permalink, body, zone: "all", nav: "feed", depositCount, index: true, host,
    jsonLd: {
      "@context": "https://schema.org",
      "@graph": [
        {
          "@type": "Dataset", "license": licenseUrl(host), "url": permalink, "inLanguage": "pt-PT",
          "name": `Índice do mercado de carros usados em Portugal — ${label}`,
          "description": "Preço pedido mediano, número de anúncios ativos, quilometragem mediana e dias medianos até vender no mercado português de carros usados, agregados por mês a partir dos cortes semanais.",
          "creator": { "@type": "Organization", "name": "Carsbuyer", "url": `https://${host}/` },
          "isAccessibleForFree": true,
          "temporalCoverage": cut.from && cut.to ? `${cut.from}/${cut.to}` : undefined,
          "dateModified": cut.builtAt || undefined,
          "variableMeasured": ["Preço pedido mediano (EUR)", "Anúncios ativos", "Dias até vender (mediana)"],
        },
        breadcrumbLd(host, crumbItems),
      ],
    },
  });
}

// ═══ Identity ════════════════════════════════════════════════════════════════
//
// Who stands behind the numbers is an E-E-A-T signal, and for a money topic its
// absence is a real deduction. But a name and a contact address are the site
// operator's to publish, not ours to invent: a made-up address is worse than
// none, because it is a promise the site cannot keep.
//
// So both come from env (SITE_AUTHOR / SITE_CONTACT_EMAIL in wrangler.toml
// [vars]); unset, the pages render the brand-level version and simply omit the
// blocks that would otherwise state something untrue.
let SITE_AUTHOR = "", SITE_CONTACT = "";
export function setSiteIdentity({ author, contact } = {}) {
  SITE_AUTHOR = (author || "").trim();
  SITE_CONTACT = (contact || "").trim();
}
function authorBlock() {
  if (!SITE_AUTHOR && !SITE_CONTACT) return "";
  return `<div class="exclusive" style="background:#FAFAF8;border:1px solid #EFECE6;align-items:flex-start;margin-top:8px;">
      <span style="font-size:15px;">✍️</span>
      <span class="x" style="color:#5B606B;">${SITE_AUTHOR ? `<b style="color:#16181D;">${escapeHtml(SITE_AUTHOR)}</b> mantém o Carsbuyer: recolhe os dados, escreve o código que os trata e responde por eles. ` : ""}${SITE_CONTACT ? `Erros, dúvidas de método e pedidos de dados: <a href="mailto:${escapeHtml(SITE_CONTACT)}" style="color:#177A47;font-weight:600;">${escapeHtml(SITE_CONTACT)}</a>.` : ""}</span>
    </div>`;
}

function modelQualityBlock(mq) {
  if (!mq || mq.mae == null || mq.mape == null || mq.cov == null) {
    return `<p class="fc-p">A medição do último treino do modelo ainda não chegou a esta página. Até chegar, o que garante as estimativas são os filtros da secção seguinte, e não um número de erro que não te mostrámos.</p>`;
  }
  const dec = (v, n) => v.toFixed(n).replace(".", ",");
  const rows = [
    [`Erro médio, em euros`, fmtEur(mq.mae),
      `A diferença média entre a estimativa e o preço pedido, em valor absoluto.`],
    [`Erro médio, em percentagem`, `${dec(mq.mape, 1)}%`,
      `Puxado para cima pelos carros baratos: 700 € de diferença num carro de 2 000 € são 35%. É uma das razões para não publicarmos estimativa abaixo dos 5 000 €.`],
    ...(mq.r2 != null ? [[`Variação explicada (R²)`, dec(mq.r2, 3),
      `Quanto da diferença de preço entre carros o modelo consegue explicar. 1,000 seria acertar sempre.`]] : []),
    [`Banda de 80%: acerto real`, `${Math.round(mq.cov * 100)}%`,
      `Em cada 100 carros, a banda publicada contém o preço real em ${Math.round(mq.cov * 100)}. Uma banda de 80% que apanhasse 95% seria larga de mais para dizer alguma coisa.`],
  ];
  return `<ul class="fc-ul">${rows.map(([k, v, why]) =>
    `<li class="fc-li"><b>${k}: <span class="mono">${v}</span></b> — ${why}</li>`).join("")}</ul>
    <p class="mono" style="color:#5B606B;font-size:13px;margin-top:-4px;">Medido em ${mq.n ? fmtNum(mq.n) + " anúncios" : "todos os anúncios com preço"}${mq.folds ? `, em ${mq.folds} partes` : ""}${mq.ts ? ` · treino de ${escapeHtml(mq.ts)}` : ""}</p>`;
}

// ═══ /metodologia ════════════════════════════════════════════════════════════
//
// Every number on this site is an estimate, and the honest move is to publish
// where it comes from and where it stops working — including the thresholds that
// make us DROP a figure rather than show a weak one.
export function renderMethodology({ stats, mq, host, depositCount, builtAt, duelHubs = [], wave = null }) {
  const duelList = duelHubs.length
    ? duelHubs.map(d => `<a href="/${d.path}">${escapeHtml(d.question)}</a>`).join(" e ")
    : "diesel ou gasolina e caixa manual ou automática";
  const canonical = `https://${host}/metodologia`;
  const body = crumbs([{ name: "Início", href: "/" }, { name: "Metodologia" }]) + `
    <section class="section fc-wrap" style="padding-top:16px;">
      <h1 class="fc-h1">Como calculamos os preços</h1>
      <p class="fc-p">Sem caixa preta: aqui está de onde vêm os números, o que cada um mede, e em que casos preferimos não mostrar nada a mostrar um valor fraco.</p>

      <h2 class="fc-h2">1. De onde vêm os dados</h2>
      <p class="fc-p">Recolhemos diariamente os anúncios de automóveis do <b>OLX Portugal</b> e guardamos o histórico de cada um: preço, alterações de preço, e o dia em que o anúncio desaparece. Usamos apenas anúncios <b>ativos</b> no momento do cálculo${stats.listings ? `; hoje são ${fmtNum(stats.listings)} anúncios em ${stats.models} modelos` : ""}. Não compramos nem vendemos carros, não somos stand e não recebemos de nenhum vendedor.</p>

      <h2 class="fc-h2">2. O que é um "preço" aqui</h2>
      <p class="fc-p">É o <b>preço pedido</b> num anúncio ativo — não o preço a que o carro foi vendido. Ninguém em Portugal publica preços de transação, e inventá-los seria pior do que dizer o que temos. Preços pedidos e preços de venda não são a mesma coisa: a diferença costuma ser a margem de negociação, e é maior nos modelos que demoram a sair (ver <a href="/liquidez">tempo de venda</a>).</p>

      <h2 class="fc-h2">3. Mediana e intervalo, nunca um número sozinho</h2>
      <p class="fc-p">Para cada modelo mostramos a <b>mediana</b> (o valor que divide os anúncios ao meio) e o <b>intervalo interquartil P25-P75</b>, onde cabe metade dos anúncios. Usamos a mediana e não a média porque um único carro de coleção ou um anúncio com preço simbólico destrói uma média e não mexe numa mediana. O intervalo vai sempre junto: uma mediana sem dispersão parece uma precisão que não existe.</p>

      <h2 class="fc-h2">4. Quando é que uma página existe</h2>
      <p class="fc-p">Um número só é publicado quando a amostra por trás dele significa alguma coisa:</p>
      <ul class="fc-ul">
        <li class="fc-li"><b>20 anúncios ativos</b> — mínimo para um modelo <b>ganhar</b> página. Uma vez publicada, a página mantém-se enquanto houver <b>14</b>: o stock de um modelo oscila de dia para dia, e deixar o endereço morrer e ressuscitar ao sabor de um anúncio a mais ou a menos é pior do que publicar 14 e dizer que são 14. O número de anúncios por trás de cada mediana está sempre à vista.</li>
        <li class="fc-li"><b>5 anúncios</b> — mínimo para uma linha por ano na tabela, <b>3</b> para uma linha já publicada se manter. Anos mais finos são juntados em intervalos de dois ou mais anos, ou omitidos e contados no rodapé da tabela.</li>
        <li class="fc-li"><b>${MIN_YEAR_PAGE_N} anúncios</b> — mínimo para um ano <b>ganhar página própria</b>, <b>7</b> para a manter depois de a ter. Abaixo de ${MIN_YEAR_PAGE_N}, um único anúncio fora do normal move a mediana mais do que a diferença entre anos que estaríamos a afirmar; e um ano que já tem endereço não o deve perder por causa de um carro vendido esta semana.</li>
        <li class="fc-li"><b>${DEP_MIN_CELLS} anos com amostra e ${DEP_MIN_SPAN} anos de intervalo</b> — mínimo para uma <a href="/depreciacao">curva de desvalorização</a>, mais um ajuste que explique de facto os pontos (R² ≥ ${DEP_MIN_R2}).</li>
        <li class="fc-li"><b>15 anúncios</b> — mínimo para um <b>corte</b> do modelo (combustível, caixa, distrito) ganhar página própria, <b>11</b> para a manter. Um corte que é praticamente o modelo inteiro — a única motorização, ou a única caixa, com mais de 85% dos anúncios — não ganha página nenhuma: seria a página do modelo outra vez noutro endereço.</li>
        <li class="fc-li"><b>3 anos com amostra dos dois lados</b> — mínimo para comparar dois cortes em percentagem. Sem isso a página mostra as duas medianas e diz que a distância entre elas ainda inclui a diferença de idades.</li>
        <li class="fc-li"><b>20 anúncios de cada lado</b>, mais uma margem estreita o suficiente para a resposta significar alguma coisa — mínimo para uma página de duelo (${duelList}).</li>
        <li class="fc-li"><b>Uma quebra na taxa só é publicada se for medida</b> — ver abaixo.</li>
      </ul>
      ${wave ? `<h3 class="fc-h3" id="vagas">Amostra suficiente e ainda sem página</h3>
      <p class="fc-p">Passar o limite da amostra é condição necessária, não suficiente. As páginas por ano, por corte e de desvalorização são publicadas <b>por vagas</b>, começando pelos modelos com mais anúncios: neste momento existem para <b>${fmtNum(wave.models)} dos ${fmtNum(wave.total)} modelos</b>, ${fmtNum(wave.pages)} páginas ao todo. A página de <a href="/liquidez">tempo de venda</a> tem uma vaga própria, separada desta. Publicamos por vagas para conseguirmos medir se cada camada é lida antes de multiplicá-la — mil páginas de uma vez só dizem que algo não funcionou, não o quê.</p>
      <p class="fc-p">Nos modelos que ainda não entraram, um ano com amostra suficiente devolve <b>404</b> e não aparece ligado em lado nenhum: preferimos uma página em falta a uma ligação partida. Os números desses anos não estão escondidos — estão na versão JSON de cada modelo${wave.sample ? ` (<a href="/preco/${encodeURIComponent(wave.sample)}.json">exemplo</a>)` : ""}, em <code>by_year</code>, com <code>page: null</code> a dizer que a página ainda não existe.</p>` : ""}

      <h3 class="fc-h3">Comparar dois cortes do mesmo modelo</h3>
      <p class="fc-p">As medianas de dois cortes não se subtraem. Os automáticos à venda são muito mais novos do que os manuais, por isso a razão em bruto entre as duas medianas mede sobretudo a diferença de idades e chamar-lhe prémio da caixa seria inventar um número que os dados não dizem. Cada comparação entre cortes é feita <b>dentro de cada ano de matrícula</b> e só depois juntada, ponderada pela amostra mais fina de cada ano — o mesmo método das <a href="/comparar">comparações entre modelos</a>.</p>
      <p class="fc-p">Um corte fino — um distrito com trinta anúncios espalhados por vinte anos — raramente tem três anos com amostra dos dois lados, e é precisamente onde a mistura de idades mais engana. Nesses casos comparamos <b>anúncio a anúncio</b>: cada carro do corte é dividido pela mediana do modelo no seu próprio ano de matrícula, e a resposta é a mediana desses quocientes. Um ano só entra se o modelo tiver aí pelo menos o dobro dos anúncios do corte, senão o corte estaria a dividir-se por si próprio. A página diz sempre qual dos dois métodos usou, e abaixo de <b>5%</b> não afirmamos direção nenhuma: a essa distância os dois métodos discordam de sinal com demasiada frequência para a diferença significar alguma coisa.</p>
      <p class="fc-p">Nas páginas de duelo (${duelList}) vamos um passo mais longe, porque aí a pergunta é sobre o <i>ritmo</i> da queda e não sobre o preço de hoje: ajustamos o preço à idade <b>e</b> à quilometragem em simultâneo, e a taxa de cada lado é lida com a quilometragem igualada. Sem isso mediríamos o facto de os diesels à venda andarem muito mais e os automáticos muito menos. Cada página traz a margem de 95% da diferença que afirma, e um modelo só tem página quando essa margem é estreita o suficiente para que "não há diferença" queira dizer <b>não há vantagem apreciável</b> e não <b>não conseguimos ver</b>.</p>

<h2 class="fc-h2" id="modelo">5. O "valor justo estimado"</h2>
      <p class="fc-p">Além dos preços pedidos, treinamos um modelo estatístico — <i>gradient boosting</i>, uma soma de muitas árvores de decisão em que cada uma corrige o erro das anteriores — sobre todos os anúncios com preço que recolhemos, incluindo os que já saíram do mercado. Lê marca, modelo, ano, quilometragem, cilindrada, potência, combustível, caixa, geração, versão, nível de equipamento, segmento e distrito, e devolve um valor com uma banda à volta, nunca um número seco. Nas páginas de modelo, esse valor é a <b>mediana das estimativas dos anúncios reais</b> daquele modelo — não a estimativa de um carro-tipo inventado.</p>

      <h3 class="fc-h3">O que lhe impomos à força</h3>
      <p class="fc-p">Duas regras não são aprendidas, são obrigatórias no valor que publicamos: <b>mais um ano de idade nunca o aumenta</b> e <b>mais quilómetros nunca o aumentam</b>, tudo o resto igual. Sem elas, um ano com poucos anúncios produz saltos que o mercado não tem. E o modelo aprende sobre o <b>logaritmo</b> do preço, para que o erro conte em percentagem: enganar-se em 1 000 € num carro de 3 000 € não é o mesmo que enganar-se em 1 000 € num de 30 000 €.</p>

      <h3 class="fc-h3">Quanto erra — medido, não afirmado</h3>
      <p class="fc-p">O erro é medido por <b>validação cruzada em cinco partes</b>: o modelo é treinado em quatro quintos dos anúncios e avaliado no quinto que não viu, cinco vezes seguidas. Publicar uma estimativa sem dizer quanto ela erra é publicar meia verdade, por isso aqui está a última medição:</p>
      ${modelQualityBlock(mq)}

      <h3 class="fc-h3">O que o modelo não vê</h3>
      <p class="fc-p">Um anúncio não diz o estado da embraiagem, o histórico de manutenção, se houve batida, como estão os pneus, nem se o ISV de um importado já foi pago. Nada disso entra no modelo, porque não existe nos dados. <b>Dois carros com a mesma ficha recebem a mesma estimativa, mesmo que um precise de caixa nova.</b> Daí a estimativa ser um ponto de partida para negociar e para saber onde olhar — nunca a avaliação de uma viatura concreta.</p>

      <p class="fc-p">Não o publicamos sempre. A estimativa é <b>retirada</b> quando:</p>
      <ul class="fc-ul">
        <li class="fc-li">o preço fica abaixo de <b>5 000 €</b> — no fundo do mercado o modelo sobrestima sistematicamente;</li>
        <li class="fc-li">o preço fica acima de <b>45 000 €</b> — no topo o modelo satura e carros muito diferentes colapsam no mesmo valor;</li>
        <li class="fc-li">a estimativa discorda demasiado dos anúncios reais (fora de 0,70x a 1,40x da mediana pedida, ou fora do contexto do intervalo P25-P75);</li>
        <li class="fc-li">faltam características suficientes nos anúncios para o modelo distinguir versões.</li>
      </ul>
      <p class="fc-p">Nesses casos a página mostra só os preços pedidos. Preferimos uma página com menos números a uma página com um número errado.</p>

      <h2 class="fc-h2">6. Dias até vender</h2>
      <p class="fc-p">Acompanhamos cada anúncio desde que aparece até ao último dia em que o vimos no ar, e daí sai o <a href="/liquidez">tempo de venda</a>. Três coisas nesta conta não são óbvias e mudam o resultado:</p>
      <ul class="fc-ul">
        <li class="fc-li"><b>Os anúncios ainda à venda contam.</b> Se olhássemos só para os que já acabaram, ficávamos com os que tiveram pressa — num mercado que recebe anúncios novos todos os dias, isso encurta a conta em cerca de dez dias. Cada anúncio ainda no ar entra com os dias que já leva (é o método de Kaplan-Meier, o mesmo que se usa para não deitar fora quem ainda não teve o desfecho).</li>
        <li class="fc-li"><b>O dia que conta é o último em que o vimos vivo</b>, não o dia em que reparámos que tinha saído. Quando a recolha fica bloqueada uns dias, a varredura seguinte marca tudo de uma vez, e usar essa data acrescentaria a duração da avaria ao tempo de venda de milhares de carros.</li>
        <li class="fc-li"><b>Um anúncio do OLX corre em ciclos de 30 dias</b>, e vê-se: há uma acumulação de saídas exatamente aí. Por isso a figura principal é a <b>percentagem que sai no primeiro mês</b> e não a mediana — a mediana cai dentro desse degrau em quase todos os modelos e acaba a descrever o ciclo do OLX.</li>
      </ul>
      <p class="fc-p">Um anúncio pode desaparecer por venda ou por desistência, e não distinguimos os dois: a leitura correta é <b>tempo até sair do mercado</b>. O que conseguimos afirmar é um mínimo do que não vendeu — os anúncios que reaparecem semanas depois como anúncio novo do mesmo carro, que emparelhamos pela ficha e pela quilometragem. Uma página de tempo de venda existe a partir de <b>40 anúncios acompanhados até ao fim</b>, e cada corte dentro dela (preço, idade, distrito) precisa dos seus próprios 40.</p>

      <h2 class="fc-h2" id="importar">7. Importar da Alemanha</h2>
      <p class="fc-p">A conta das páginas de <a href="/importar">importação</a> tem três parcelas e todas são medidas, não estimadas por regra de três: o <b>preço pedido na Alemanha</b> vem de anúncios do AutoScout24 lidos pelo nosso próprio recolhedor, ano a ano; o <b>ISV</b> é calculado anúncio a anúncio a partir do CO2, da cilindrada e do ano de cada carro alemão (nunca a partir de um carro-tipo, porque as tabelas são progressivas e a média não passa por elas); a <b>legalização</b> é uma lista de rubricas com valores de 2026, publicada como intervalo porque transporte e certificado de conformidade variam.</p>
      <p class="fc-p">Emparelhamos sempre o <b>mesmo ano de matrícula</b> dos dois lados, e cada linha traz as duas amostras e as duas quilometragens medianas. Sem isso a comparação mediria sobretudo o facto de a oferta alemã ser mais nova e menos rodada do que a portuguesa. Um modelo só tem página com pelo menos dois anos comparáveis, dez anúncios alemães por ano, cinco portugueses e seis carros alemães com CO2 utilizável — o CO2 é campo livre lá, e um valor impossível é descartado em vez de virar imposto.</p>

      <h2 class="fc-h2">8. Desvalorização</h2>
      <p class="fc-p">Ajustamos uma reta ao <b>logaritmo</b> da mediana de preço contra o ano de fabrico, e reportamos a taxa anual daí resultante. Logaritmo porque a desvalorização é uma percentagem do que resta: em euros perde-se muito mais no primeiro ano do que no nono, e uma reta sobre euros misturaria as duas coisas. Como é um corte transversal de anos diferentes hoje (e não o mesmo carro seguido ao longo do tempo), a leitura correta é "quanto custa a mais um ano mais recente", não "quanto vou perder no próximo ano".</p>

      <h3 class="fc-h3">"A partir dos N anos deixa de perder valor"</h3>
<p class="fc-p">A desvalorização é uma percentagem <b>do que resta</b>, por isso a mesma taxa dá uma perda enorme em euros no início e pequena no fim. É daí que vem a frase feita: em euros a queda abranda sempre, em qualquer modelo, sem que a percentagem mude nada. Dizer "a partir dos oito anos já não perde" a partir disso é confundir as duas coisas.</p>
      <p class="fc-p">Por isso medimos as duas em separado. O <b>custo de um ano de idade em euros</b> sai da curva ajustada e está em todas as páginas de desvalorização. A <b>quebra na percentagem</b> é testada: ajustamos uma curva com um joelho em cada idade entre os ${BEND_MIN_AGE} e os ${BEND_MAX_AGE} anos, ficamos com a que melhor explica os preços, e só a publicamos se o joelho compensar o parâmetro extra (F ≥ ${BEND_MIN_F}, com pelo menos ${BEND_MIN_SIDE} anos com amostra de cada lado) e se as duas taxas diferirem em ${Math.round(BEND_MIN_GAP * 100)} pontos ou mais. Na maioria dos modelos deste mercado não compensa, e a página di-lo em vez de desenhar um ponto de inflexão que ninguém mediu.</p>

      <h2 class="fc-h2">9. O que isto não faz</h2>
      <ul class="fc-ul">
        <li class="fc-li">Não avalia a <b>tua</b> viatura. A mediana de um modelo não sabe do teu histórico, dos teus extras nem do estado da tua embraiagem. Para o carro concreto, <a href="/avaliar">avalia o anúncio</a>.</li>
        <li class="fc-li">Não distingue carros <b>importados por legalizar</b> na mediana do modelo. Um preço muito abaixo do normal costuma ter ISV por pagar — e o <a href="/isv">ISV</a> pode valer milhares.</li>
        <li class="fc-li">Não cobre carros vendidos fora do OLX (stands com stock próprio, particulares em grupos fechados, leilões).</li>
        <li class="fc-li">Não é aconselhamento financeiro nem uma avaliação para efeitos legais ou de seguro.</li>
      </ul>

      <h2 class="fc-h2" id="licenca">10. Reutilização e atribuição</h2>
      <p class="fc-p">Os números destas páginas são estatísticas nossas, calculadas a partir de anúncios públicos do OLX Portugal. <b>Podes citá-los e reutilizá-los</b>, desde que a fonte seja atribuída ao Carsbuyer e seja indicada a data de recolha: a mediana de um modelo muda ao longo do tempo, e uma citação sem data deixa de ser verificável. Se precisares dos valores sem a marcação da página, cada modelo publica o mesmo conteúdo em JSON, bastando acrescentar <b>.json</b> ao endereço, como em <a href="/preco/opel-corsa.json">/preco/opel-corsa.json</a>.</p>

      <h2 class="fc-h2">10. Correções</h2>
      <p class="fc-p">Se um número parece errado, provavelmente vale a pena olhar: a amostra pode estar contaminada por anúncios repetidos ou por uma versão mal classificada. Todas as páginas indicam o tamanho da amostra e a data de recolha, para que qualquer afirmação nossa seja verificável.</p>
      ${authorBlock()}
      ${provenance({ n: stats.listings, builtAt, measure: "Preço pedido em anúncios ativos (mediana e P25-P75)" })}
      <p class="fc-p" style="margin-top:18px;"><a href="/sobre">Quem faz isto</a> · <a href="/precos">Preços por modelo</a> · <a href="/mercado/indice">Índice de mercado</a></p>
    </section>
    <div style="height:60px;"></div>`;
  return layout({
    title: "Metodologia: como calculamos os preços",
    description: "De onde vêm os dados, o que é a mediana e o intervalo P25-P75, quando publicamos um valor justo estimado e quando o retiramos. Método completo do Carsbuyer.",
    canonical, body, zone: "all", nav: null, depositCount, index: true, host,
    jsonLd: {
      "@context": "https://schema.org",
      "@graph": [
        {
          "@type": "TechArticle", "headline": "Como calculamos os preços de carros usados em Portugal",
          "url": canonical, "inLanguage": "pt-PT", "dateModified": builtAt || undefined,
          "publisher": { "@type": "Organization", "name": "Carsbuyer", "url": `https://${host}/` },
          ...(SITE_AUTHOR ? { "author": { "@type": "Person", "name": SITE_AUTHOR } } : {}),
        },
        breadcrumbLd(host, [{ name: "Início", href: "/" }, { name: "Metodologia" }]),
      ],
    },
  });
}

// ═══ /sobre ══════════════════════════════════════════════════════════════════
export function renderAbout({ stats, mq, host, depositCount, builtAt }) {
  const canonical = `https://${host}/sobre`;
  const body = crumbs([{ name: "Início", href: "/" }, { name: "Quem somos" }]) + `
    <section class="section fc-wrap" style="padding-top:16px;">
      <h1 class="fc-h1">Quem faz isto, e porquê</h1>
      <p class="fc-p">O Carsbuyer é um projeto independente que mede o mercado português de carros usados a partir dos anúncios que estão de facto à venda. Nasceu de uma pergunta simples que ninguém em Portugal respondia com números: <i>quanto vale mesmo este carro?</i></p>

      <h2 class="fc-h2">Independentes de quem?</h2>
      <p class="fc-p">Não somos stand, não somos intermediário e não representamos nenhum vendedor. Não temos carros para colocar, por isso não temos motivo para inflacionar nem para desvalorizar nenhum modelo. Os números que publicamos são os mesmos que usamos para as nossas próprias decisões — se estivessem enviesados, seríamos os primeiros prejudicados.</p>

      <h2 class="fc-h2">Como nos pagamos</h2>
      <p class="fc-p">As avaliações e os preços por modelo são gratuitos e ficam gratuitos: ver um anúncio avaliado, o <a href="/mercado">mercado</a> ou os preços por modelo não custa nada e não exige registo. O site paga-se de duas formas: quando um vendedor pede propostas de compra e um comprador profissional paga por esse contacto, e com ligações de parceiros para relatórios de histórico do veículo. Não vendemos os teus dados, não temos publicidade paga por marcas e não aceitamos pagamento para mexer numa avaliação — nenhuma destas receitas muda os números que mostramos.</p>

      <h2 class="fc-h2">O que temos hoje</h2>
      <ul class="fc-ul">
        <li class="fc-li"><b>${stats.models}</b> modelos com amostra suficiente para publicar preços${stats.listings ? `, sobre ${fmtNum(stats.listings)} anúncios ativos` : ""}.</li>
        <li class="fc-li">Preço mediano por modelo e <b>por ano de fabrico</b>, sempre com o intervalo onde cabe metade dos anúncios.</li>
        <li class="fc-li"><a href="/liquidez">Tempo mediano até vender</a> — medido em anúncios reais, não estimado.</li>
        <li class="fc-li"><a href="/depreciacao">Curvas de desvalorização</a> para os modelos com histórico suficiente.</li>
        <li class="fc-li">Um <a href="/mercado/indice">índice semanal do mercado</a>, com registo permanente de cada semana.</li>
      </ul>

<h2 class="fc-h2">Erramos?</h2>
      <p class="fc-p">Sim, e dizemos quanto. ${mq && mq.mape != null && mq.cov != null
        ? `Na última medição, a estimativa de valor justo erra em média <b>${mq.mape.toFixed(1).replace(".", ",")}%</b> em anúncios que o modelo não viu no treino, e a banda de 80% que publicamos contém o preço real em <b>${Math.round(mq.cov * 100)}%</b> dos casos. Como isso é medido está na <a href="/metodologia#modelo">metodologia</a>.`
        : `Como medimos o erro da estimativa, e o que ele deu, está na <a href="/metodologia#modelo">metodologia</a>.`}</p>
      <p class="fc-p">Publicamos também o <a href="/metodologia">método completo</a>, o tamanho de cada amostra e a data de recolha em todas as páginas. Quando um número não é fiável, retiramo-lo em vez de o disfarçar — há modelos onde verás preços pedidos e nenhuma estimativa de valor justo, e isso é intencional.</p>

      <h2 class="fc-h2">Podes usar os nossos números</h2>
      <p class="fc-p">Com atribuição ao Carsbuyer e a data — os valores mudam todos os dias. Cada página de modelo tem uma <b>versão em JSON</b> ligada no cabeçalho, e há um <a href="/llms.txt">llms.txt</a> com a estrutura completa para quem lê o site com ferramentas automáticas.</p>
      ${authorBlock()}
      ${provenance({ n: stats.listings, builtAt, measure: "Preço pedido em anúncios ativos (mediana e P25-P75)" })}
      <p class="fc-p" style="margin-top:18px;"><a href="/metodologia">Metodologia</a> · <a href="/precos">Preços por modelo</a> · <a href="/avaliar">Avaliar um carro</a></p>
    </section>
    <div style="height:60px;"></div>`;
  return layout({
    title: "Quem somos — avaliação independente de usados",
    description: `Projeto independente que mede o mercado português de carros usados a partir de anúncios ativos do OLX: ${stats.models} modelos, método publicado, sem ligação a stands.`,
    canonical, body, zone: "all", nav: null, depositCount, index: true, host,
    jsonLd: {
      "@context": "https://schema.org",
      "@graph": [
        {
          "@type": "AboutPage", "url": canonical, "inLanguage": "pt-PT", "name": "Quem somos",
          "mainEntity": {
            "@type": "Organization", "name": "Carsbuyer", "alternateName": "Flipper Club",
            "url": `https://${host}/`, "areaServed": "PT",
            "description": "Avaliação independente de carros usados em Portugal a partir de anúncios ativos do OLX.",
            ...(SITE_CONTACT ? { "email": SITE_CONTACT } : {}),
            ...(SITE_AUTHOR ? { "founder": { "@type": "Person", "name": SITE_AUTHOR } } : {}),
          },
        },
        breadcrumbLd(host, [{ name: "Início", href: "/" }, { name: "Quem somos" }]),
      ],
    },
  });
}

// ═══ /isv — imported-car tax estimator ═══════════════════════════════════════
//
// "simulador ISV" is a large query and the reason importing looks cheap until it
// isn't: a €9 000 German car with €4 000 of ISV is not a €9 000 car. We already
// compute this server-side for flagged imports on /mercado; this exposes the
// same tables as a calculator, next to the one thing the ISV simulators do not
// have — what that model actually costs in Portugal today.
//
// TABLES ARE LOCK-STEP WITH src/analytics/isv.py. Same brackets, same order,
// same age-reduction table. If one changes, change both (paired-comment pact,
// like slugify ↔ model_pages.slugify). Verified against Autoridade Tributária
// 2026 rates, which are unchanged from 2024/2025.
const ISV_TABLES = {
  cc: [[1000, 1.09, 849.03], [1250, 1.18, 850.69], [null, 5.61, 6194.88]],
  co2: {
    "WLTP:petrol": [[110, 0.44, 43.02], [115, 1.10, 115.80], [120, 1.38, 147.79],
      [130, 5.27, 619.17], [145, 6.38, 762.73], [175, 41.54, 5819.56],
      [195, 51.38, 7247.39], [235, 193.01, 34190.52], [null, 233.81, 41910.96]],
    "WLTP:diesel": [[110, 1.72, 11.50], [120, 18.96, 1906.19], [140, 65.04, 7360.85],
      [150, 127.40, 16080.57], [160, 160.81, 21176.06], [170, 221.69, 29227.38],
      [190, 274.08, 36987.98], [null, 282.35, 38271.32]],
    "NEDC:petrol": [[99, 4.62, 427.00], [115, 8.09, 750.99], [145, 52.56, 5903.94],
      [175, 61.24, 7140.17], [195, 155.97, 23627.27], [null, 205.65, 33390.12]],
    "NEDC:diesel": [[79, 5.78, 439.04], [95, 23.45, 1848.58], [120, 79.22, 7195.63],
      [140, 175.73, 18924.92], [160, 195.43, 21720.92], [null, 268.42, 33447.90]],
  },
  particulas: 500,
  reducao: [[1, 0.10], [2, 0.20], [3, 0.28], [4, 0.35], [5, 0.43], [6, 0.52],
    [7, 0.60], [8, 0.65], [9, 0.70], [10, 0.75], [null, 0.80]],
};

/**
 * ISV estimate. Pure, exported, and SERIALISED INTO THE PAGE via toString()
 * below — so the number the browser shows and the number Node tests are
 * produced by the same function, not by two copies that drift.
 *
 * Mirrors src/analytics/isv.py::compute_isv, including its refusals: a BEV is
 * exempt (0), a PHEV returns null because its reduced regime needs per-car
 * eligibility we do not have, and missing cilindrada/CO2 returns null rather
 * than a guess.
 *
 * Self-contained on purpose: it receives the tables as an argument and calls
 * nothing from module scope, because its source is what ships to the browser.
 */
export function estimateIsv(T, { cc, co2, fuel, regYear, asOfYear, isEu }) {
  if (fuel === "bev") return { exempt: true, isv: 0 };
  if (fuel === "phev") return null;
  if (!(cc > 0) || !(co2 > 0) || !(regYear > 0)) return null;
  var bracket = function (v, tbl) {
    for (var i = 0; i < tbl.length; i++) if (tbl[i][0] === null || v <= tbl[i][0]) return tbl[i];
    return tbl[tbl.length - 1];
  };
  var fclass = (fuel === "diesel") ? "diesel" : "petrol";
  var cycle = (regYear <= 2019) ? "NEDC" : "WLTP";
  var bcc = bracket(cc, T.cc);
  var compCc = Math.max(0, cc * bcc[1] - bcc[2]);
  var bco2 = bracket(co2, T.co2[cycle + ":" + fclass]);
  var compCo2 = Math.max(0, co2 * bco2[1] - bco2[2]);
  var part = (fclass === "diesel") ? T.particulas : 0;
  var gross = compCc + compCo2 + part;
  var age = Math.max(0, asOfYear - regYear);
  var red = 0;
  if (isEu) {
    for (var j = 0; j < T.reducao.length; j++) {
      if (T.reducao[j][0] === null || age <= T.reducao[j][0]) { red = T.reducao[j][1]; break; }
    }
  }
  return {
    exempt: false, cycle: cycle, age: age, reduction: red,
    cilindrada: compCc, co2: compCo2, particulas: part,
    gross: gross, isv: gross * (1 - red),
    ambiguousCycle: regYear >= 2018 && regYear <= 2019,
  };
}

// Exported only so the tests can drive estimateIsv() with the shipped tables.
export const ISV_TABLES_FOR_TEST = ISV_TABLES;

export function renderIsv({ topModels, host, depositCount, builtAt, refYear }) {
  const canonical = `https://${host}/isv`;
  const year = refYear || new Date().getUTCFullYear();
  const years = [];
  for (let y = year; y >= year - 25; y--) years.push(y);
  const modelLinks = (topModels || []).slice(0, 12).map(m =>
    `<a class="mchip" href="/preco/${encodeURIComponent(m.slug)}">${escapeHtml(m.b)} ${escapeHtml(m.m)} <span class="mut">${fmtEur(m.fm)}</span></a>`).join("");

  const body = crumbs([{ name: "Início", href: "/" }, { name: "Simulador ISV" }]) + `
    <section class="section fc-wrap" style="padding-top:16px;">
      <h1 class="fc-h1">Simulador de ISV: quanto custa legalizar um carro importado</h1>
      <p class="fc-p">O ISV é a fatura que transforma um bom negócio na Alemanha num mau negócio em Portugal. Calcula-se a partir da <b>cilindrada</b>, das <b>emissões de CO2</b> e da <b>idade</b> do carro — e para um usado com alguns anos a redução por idade corta uma boa parte.</p>
      <div class="fc-form">
        <div class="fc-field"><label for="isv-cc">Cilindrada (cm³)</label><input id="isv-cc" type="number" inputmode="numeric" min="1" max="9000" step="1" placeholder="1598"></div>
        <div class="fc-field"><label for="isv-co2">CO2 (g/km)</label><input id="isv-co2" type="number" inputmode="numeric" min="1" max="500" step="1" placeholder="110"></div>
        <div class="fc-field"><label for="isv-fuel">Combustível</label><select id="isv-fuel">
          <option value="petrol">Gasolina</option>
          <option value="diesel">Gasóleo (diesel)</option>
          <option value="petrol-hybrid">Híbrido (não plug-in)</option>
          <option value="petrol-lpg">GPL</option>
          <option value="bev">Elétrico</option>
          <option value="phev">Híbrido plug-in</option>
        </select></div>
        <div class="fc-field"><label for="isv-year">1.ª matrícula</label><select id="isv-year">${years.map(y => `<option value="${y}">${y}</option>`).join("")}</select></div>
        <div class="fc-field"><label for="isv-eu">Matriculado na UE</label><select id="isv-eu"><option value="1">Sim</option><option value="0">Não (sem redução)</option></select></div>
      </div>
      <div class="fc-out" id="isv-out" aria-live="polite">
        <div class="cap">Preenche a cilindrada e o CO2</div>
        <div class="big">—</div>
      </div>
      <div class="exclusive" style="background:#F6FBF8;border:1px solid #DDEBE1;align-items:flex-start;margin-top:18px;">
        <span style="font-size:15px;">🇩🇪</span>
        <span class="x" style="color:#3A3F47;"><b style="color:#16181D;">O ISV sozinho não responde à pergunta.</b> O que decide é o preço alemão mais o imposto mais a legalização, contra o que o mesmo carro pede em Portugal hoje. Fizemos essa conta modelo a modelo, com anúncios reais dos dois lados. <a href="/importar" style="color:#177A47;font-weight:600;">Ver em que modelos compensa&nbsp;→</a></span>
      </div>
      <p class="fc-p" style="margin-top:14px;">A cilindrada e o CO2 estão no certificado de conformidade e no documento único do carro. Para carros de 2018 e 2019 o ciclo de medição (NEDC ou WLTP) é ambíguo pelo ano — assumimos NEDC, o que pode subestimar ou sobrestimar; confirma no documento.</p>

      <h2 class="fc-h2">O que este número inclui e o que não inclui</h2>
      <ul class="fc-ul">
        <li class="fc-li"><b>Inclui:</b> componente de cilindrada, componente ambiental (CO2), taxa de partículas de 500 € nos gasóleos, e a redução por anos de uso da Tabela D (só para carros matriculados na UE).</li>
        <li class="fc-li"><b>Não inclui:</b> transporte, inspeção, legalização e chapas, IUC do ano, eventual IVA em carros novos ou quase-novos, nem a margem de quem vende.</li>
        <li class="fc-li"><b>Híbridos plug-in</b> têm um regime reduzido que depende da autonomia elétrica e do CO2 homologado; não o calculamos porque exige dados por viatura que não temos. <b>Elétricos</b> estão isentos.</li>
      </ul>

      <h2 class="fc-h2">A pergunta que interessa: compensa importar?</h2>
      <p class="fc-p">O ISV sozinho não responde. O que responde é: <b>preço lá fora + ISV + transporte + legalização</b> contra <b>o que esse modelo custa em Portugal hoje</b>. A segunda metade dessa conta é o que medimos todos os dias:</p>
      <div class="mchips">${modelLinks}</div>
      <p class="fc-p" style="margin-top:14px;"><a href="/precos">Ver preço de qualquer modelo em Portugal&nbsp;→</a></p>
      ${provenance({ n: null, builtAt, measure: "Tabelas de ISV 2026 (Código do ISV, art.º 7.º e 11.º)", extra: "Estimativa, não vinculativa" })}
      <p class="fc-p" style="margin-top:18px;">Estimativa indicativa a partir das tabelas em vigor. O valor liquidado pela Autoridade Tributária no processo de admissão é o que conta.</p>
    </section>
    <script>
    (function(){
      var T = ${JSON.stringify(ISV_TABLES)};
      var el = function(id){ return document.getElementById(id); };
      var out = el('isv-out');
      if(!out) return;
      var estimateIsv = ${estimateIsv.toString()};
      function eur(n){ return '€' + Math.round(n).toLocaleString('pt-PT'); }
      function calc(){
        var fuel = el('isv-fuel').value;
        var r = estimateIsv(T, {
          cc: parseFloat(el('isv-cc').value),
          co2: parseFloat(el('isv-co2').value),
          fuel: fuel,
          regYear: parseInt(el('isv-year').value, 10),
          asOfYear: ${year},
          isEu: el('isv-eu').value === '1',
        });
        if (r && r.exempt){
          out.innerHTML = '<div class="cap">Veículo elétrico</div><div class="big">€0</div>'
            + '<div class="mono" style="font-size:12px;color:#5B606B;margin-top:6px;">Isento de ISV.</div>';
          return;
        }
        if (!r && fuel === 'phev'){
          out.innerHTML = '<div class="cap">Híbrido plug-in</div><div class="big">—</div>'
            + '<div class="mono" style="font-size:12px;color:#5B606B;margin-top:6px;">Regime reduzido dependente da autonomia elétrica e do CO2 homologado. Não estimamos para não dar um número errado.</div>';
          return;
        }
        if (!r){
          out.innerHTML = '<div class="cap">Preenche a cilindrada e o CO2</div><div class="big">—</div>';
          return;
        }
        out.innerHTML =
          '<div class="cap">ISV estimado · ' + r.cycle + ' · ' + r.age + ' ano' + (r.age===1?'':'s') + ' de uso</div>' +
          '<div class="big">' + eur(r.isv) + '</div>' +
          '<div class="mono" style="font-size:12.5px;color:#5B606B;margin-top:10px;line-height:1.7;">' +
          'Cilindrada ' + eur(r.cilindrada) + ' · CO2 ' + eur(r.co2) +
          (r.particulas ? ' · Partículas ' + eur(r.particulas) : '') +
          '<br>Bruto ' + eur(r.gross) + (r.reduction ? ' − ' + Math.round(r.reduction*100) + '% por idade' : ' · sem redução (fora da UE)') +
          (r.ambiguousCycle ? '<br>Atenção: 2018-2019 é ambíguo entre NEDC e WLTP — confirma no documento.' : '') +
          '</div>';
      }
      ['isv-cc','isv-co2','isv-fuel','isv-year','isv-eu'].forEach(function(id){
        var n = el(id); if(!n) return;
        n.addEventListener('input', calc); n.addEventListener('change', calc);
      });
      calc();
    })();
    </script>
    <div style="height:60px;"></div>`;

  const faqs = [
    ["Como se calcula o ISV de um carro importado?",
     "O ISV soma uma componente de cilindrada e uma componente ambiental de CO2, mais 500 € de taxa de partículas nos gasóleos. Ao total aplica-se a redução por anos de uso da Tabela D, que vai de 10% no primeiro ano até 80% a partir dos dez anos, e que só se aplica a carros já matriculados na União Europeia."],
    ["Um carro elétrico paga ISV em Portugal?",
     "Não. Os veículos exclusivamente elétricos estão isentos de ISV. Os híbridos plug-in têm um regime reduzido que depende da autonomia elétrica e do CO2 homologado de cada viatura."],
    ["Compensa importar um carro da Alemanha para Portugal?",
     "Depende da conta completa: preço lá fora mais ISV, transporte e legalização, contra o que esse modelo custa em Portugal hoje. O ISV é a parcela que costuma ser subestimada, e num gasóleo com CO2 alto pode ultrapassar metade do preço de compra."],
  ];

  return layout({
    title: "Simulador de ISV 2026 para carro importado",
    description: "Calcula o ISV de um carro importado a partir da cilindrada, do CO2 e da idade, com as tabelas em vigor. Depois compara com o que esse modelo custa em Portugal hoje.",
    canonical, body, zone: "all", nav: null, depositCount, index: true, host,
    jsonLd: {
      "@context": "https://schema.org",
      "@graph": [
        {
          "@type": "WebApplication", "name": "Simulador de ISV", "url": canonical,
          "applicationCategory": "FinanceApplication", "operatingSystem": "Web",
          "inLanguage": "pt-PT", "isAccessibleForFree": true,
          "offers": { "@type": "Offer", "price": "0", "priceCurrency": "EUR" },
          "description": "Estimativa do Imposto Sobre Veículos para admissão de um carro usado importado em Portugal.",
        },
        breadcrumbLd(host, [{ name: "Início", href: "/" }, { name: "Simulador ISV" }]),
        faqLd(faqs),
      ],
    },
  });
}

// ═══ Machine-readable twins ══════════════════════════════════════════════════
//
// /preco/{slug}.json and /preco/{slug}/{ano}.json — the same figures the HTML
// shows, without the markup.
//
// This is the cheapest thing on the list and probably the highest-leverage for
// the one channel where being the ORIGINAL source of a number is the whole
// advantage: an answer engine that can take the figures in one parse, with the
// sample size and the date attached, cites them with attribution far more often
// than one that has to infer them from a table. Every field is named in full —
// the blob's two-letter keys are a size optimisation for our own transport, not
// an interface anyone else should have to decode.
function facetSummary(rec, kind, slug, base, published) {
  const live = new Set(published || []);
  return facetCells(rec, kind).map(c => ({
    key: c.k, label: c.lbl, sample_size: c.n,
    share_of_model_listings: rec.n ? Math.round((c.n / rec.n) * 1000) / 1000 : null,
    asking_price: { median: c.fm, p25: c.fl, p75: c.fh },
    mileage_km_median: c.km != null ? c.km : null,
    model_years: (c.y0 && c.y1) ? { from: c.y0, to: c.y1 } : null,
    vs_model_year_matched: Array.isArray(c.vsm) ? { ratio: c.vsm[0], shared_years: c.vsm[1] } : null,
    vs_model_age_normalized: Array.isArray(c.dr) ? { ratio: c.dr[0], listings_used: c.dr[1] } : null,
    page: live.has(c.k) ? `${base}/preco/${slug}/${c.k}` : null,
  }));
}

export function modelJson(rec, slug, { host, builtAt, models = null }) {
  const base = `https://${host}`;
  const pageYears = new Set(models ? publishedYearPages(models, slug, rec, builtAt) : []);
  const facets = models ? publishedFacets(models, slug, rec, builtAt) : [];
  const duels = models ? duelsFor(models, slug, rec, builtAt) : [];
  return {
    source: "Carsbuyer",
    source_url: `${base}/preco/${slug}`,
    licence: "Citação permitida com atribuição a Carsbuyer e indicação da data.",
    measured: "asking_price",
    measured_note: "Preços PEDIDOS em anúncios ativos do OLX Portugal, não preços de venda fechados.",
    collected_until: (builtAt || "").slice(0, 10) || null,
    updated_at: builtAt || null,
    market: "PT", currency: "EUR",
    brand: rec.b, model: rec.m, slug,
    sample_size: rec.n,
    asking_price: { median: rec.fm, p25: rec.fl, p75: rec.fh },
    fair_value_estimate: rec.gm != null
      ? { median: rec.gm, low: rec.gl, high: rec.gh,
          note: "Estimativa do nosso modelo para specs típicas deste modelo; publicada apenas quando passa os limites de fiabilidade descritos em /metodologia." }
      : null,
    mileage_km_median: rec.kmm != null ? rec.kmm : null,
    model_years: (rec.y0 && rec.y1) ? { from: rec.y0, to: rec.y1 } : null,
    fuel_mix: Array.isArray(rec.fu) ? rec.fu.map(([f, share]) => ({ fuel: f, share })) : null,
    by_fuel: facetSummary(rec, "fuel", slug, base, facets),
    by_transmission: facetSummary(rec, "transmission", slug, base, facets),
    by_district: facetSummary(rec, "district", slug, base, facets),
    days_to_sell: rec.sd != null ? { median_days: rec.sd, sample_size: rec.sn } : null,
    by_year: (rec.yr || []).map(c => ({
      year: c.y, sample_size: c.n,
      asking_price: { median: c.fm, p25: c.fl, p75: c.fh },
      fair_value_estimate: c.gm != null ? { median: c.gm, low: c.gl, high: c.gh } : null,
      mileage_km_median: c.km != null ? c.km : null,
      page: pageYears.has(c.y) ? `${base}/preco/${slug}/${c.y}` : null,
      page_absent_because: pageYears.has(c.y) ? null
        : typeof c.y !== "number" ? "merged_band"
        : (c.n || 0) < MIN_YEAR_PAGE_N ? "below_year_floor"
        : "outside_publication_wave",
    })),
    years_omitted_thin_sample: rec.yt || 0,
    page_coverage_note: `\`page\` é o endereço que existe agora; quando é nulo, \`page_absent_because\` diz porquê. \"merged_band\" é uma linha que junta anos vizinhos e nunca terá endereço próprio; \"below_year_floor\" é um ano com menos de ${MIN_YEAR_PAGE_N} anúncios, que não publicamos de propósito; \"outside_publication_wave\" é amostra suficiente num modelo que ainda não entrou na vaga de publicação — esse volta a aparecer. Em qualquer dos casos os números do ano estão aqui na mesma. Os limiares estão em /metodologia.`,
    related: {
      depreciation: (models && publishedDepreciation(models, slug, rec, builtAt))
        ? `${base}/depreciacao/${slug}` : null,
      liquidity: (models && publishedLiquidity(models, slug, rec, builtAt))
        ? `${base}/liquidez/${slug}` : null,
      facets: facets.map(k => `${base}/preco/${slug}/${k}`),
      duels: duels.map(d => `${base}/${d.path}/${slug}`),
      methodology: `${base}/metodologia`,
    },
  };
}

export function depreciationJson(rec, slug, fit, av, { host, builtAt }) {
  const base = `https://${host}`;
  const keep = yrs => Math.round(Math.pow(1 - fit.rate, yrs) * 1000) / 1000;
  const r3 = x => Math.round(x * 1000) / 1000;
  return {
    source: "Carsbuyer",
    source_url: `${base}/depreciacao/${slug}`,
    licence: "Citação permitida com atribuição a Carsbuyer e indicação da data.",
    measured: "asking_price",
    measured_note: "Preços PEDIDOS em anúncios ativos do OLX Portugal, não preços de venda fechados.",
    collected_until: (builtAt || "").slice(0, 10) || null,
    updated_at: builtAt || null,
    market: "PT", currency: "EUR",
    brand: rec.b, model: rec.m, slug,
    sample_size: rec.n,
    method: "Ajuste log-linear do preço pedido mediano sobre o ano de fabrico; a taxa é uma percentagem do valor restante por ano de idade.",
    annual_depreciation_rate: r3(fit.rate),
    fit_r2: r3(fit.r2),
    reference_year: av ? av.ref : null,
    measured_age_range: av ? { from: av.minAge, to: av.maxAge } : null,
    model_year_range: { from: fit.oldest.y, to: fit.newest.y },
    half_life_years: av && av.halfLife ? Math.round(av.halfLife * 10) / 10 : null,
    retained_value: { after_1y: keep(1), after_3y: keep(3), after_5y: keep(5), after_8y: keep(8) },
    retained_value_base: av
      ? { model_year: av.base.year, age: av.base.age, fitted_price_eur: av.base.price }
      : null,
    cost_of_one_year_of_age: av
      ? av.pts.map(p => ({ age: p.age, model_year: p.y, cost_eur: av.at(p.age) }))
      : [],
    age_from_which_a_year_costs_under: av && av.cheapFrom
      ? { threshold_eur: av.costFloor, age: av.cheapFrom.age,
          model_year: av.ref - av.cheapFrom.age, fitted_price_eur: av.cheapFrom.price,
          max_age_considered: av.capAge }
      : null,
    caveats: [
      "A série é um corte transversal de anos de fabrico diferentes hoje, não o mesmo carro seguido ao longo do tempo.",
      "Entre o ano mais antigo e o mais recente o modelo mudou de geração, por isso parte da diferença de preço é equipamento e não idade.",
      "Os valores em euros saem da curva ajustada; as medianas observadas por ano estão em by_year.",
    ],
    rate_bend: av && av.bend
      ? { age: av.bend.age, rate_before: r3(av.bend.early), rate_after: r3(av.bend.late),
          f_statistic: Math.round(av.bend.F * 10) / 10,
          note: "Ajuste log-linear com uma quebra contínua; publicado só quando a quebra explica os pontos melhor do que uma taxa constante." }
      : null,
    rate_bend_note: av && av.bend ? null
      : "Nenhuma quebra entre os 4 e os 15 anos explica os preços melhor do que uma taxa constante: neste modelo a percentagem não abranda com a idade.",
    by_year: fit.cells.slice().sort((a, b) => b.y - a.y).map(c => ({
      year: c.y, sample_size: c.n, asking_price_median: c.fm,
      mileage_km_median: c.km != null ? c.km : null,
    })),
    related: {
      model: `${base}/preco/${slug}`,
      hub: `${base}/depreciacao`,
      methodology: `${base}/metodologia`,
    },
  };
}

export function facetJson(rec, slug, kind, cell, siblings, { host, builtAt }) {
  const base = `https://${host}`;
  const seg = kind === "fuel" ? "combustivel" : kind === "transmission" ? "caixa" : "distrito";
  return {
    source: "Carsbuyer",
    source_url: `${base}/preco/${slug}/${cell.k}`,
    licence: "Citação permitida com atribuição a Carsbuyer e indicação da data.",
    measured: "asking_price",
    measured_note: "Preços PEDIDOS em anúncios ativos do OLX Portugal, não preços de venda fechados.",
    collected_until: (builtAt || "").slice(0, 10) || null,
    updated_at: builtAt || null,
    market: "PT", currency: "EUR",
    brand: rec.b, model: rec.m, slug,
    facet: { kind: seg, key: cell.k, label: cell.lbl },
    sample_size: cell.n,
    share_of_model_listings: rec.n ? Math.round((cell.n / rec.n) * 1000) / 1000 : null,
    asking_price: { median: cell.fm, p25: cell.fl, p75: cell.fh },
    mileage_km_median: cell.km != null ? cell.km : null,
    model_years: (cell.y0 && cell.y1) ? { from: cell.y0, to: cell.y1 } : null,
    vs_model_year_matched: Array.isArray(cell.vsm)
      ? { ratio: cell.vsm[0], shared_years: cell.vsm[1],
          note: "Razão medida dentro de cada ano de matrícula e só depois juntada; as medianas em bruto não se subtraem porque descrevem misturas de idades diferentes." }
      : null,
    vs_model_age_normalized: Array.isArray(cell.dr)
      ? { ratio: cell.dr[0], listings_used: cell.dr[1],
          note: "Cada anúncio dividido pela mediana do modelo no seu próprio ano de matrícula; usado onde a amostra é fina demais para a razão ano a ano." }
      : null,
    siblings: (siblings || []).filter(c => c.k !== cell.k).map(c => ({
      key: c.k, label: c.lbl, sample_size: c.n,
      asking_price_median: c.fm, page: `${base}/preco/${slug}/${c.k}`,
      vs_this_cut_year_matched: (cell.vs && Array.isArray(cell.vs[c.k]))
        ? { ratio: cell.vs[c.k][0], shared_years: cell.vs[c.k][1] } : null,
    })),
    related: { model: `${base}/preco/${slug}`, methodology: `${base}/metodologia` },
  };
}

export function yearJson(rec, slug, year, cell, { host, builtAt }) {
  const base = `https://${host}`;
  return {
    source: "Carsbuyer",
    source_url: `${base}/preco/${slug}/${year}`,
    licence: "Citação permitida com atribuição a Carsbuyer e indicação da data.",
    measured: "asking_price",
    measured_note: cell.w
      ? `Preços PEDIDOS em anúncios do OLX Portugal dos últimos ${cell.w} dias, ativos e já fechados, não preços de venda.`
      : "Preços PEDIDOS em anúncios ativos do OLX Portugal, não preços de venda fechados.",
    collected_until: (builtAt || "").slice(0, 10) || null,
    updated_at: builtAt || null,
    market: "PT", currency: "EUR",
    brand: rec.b, model: rec.m, slug, year,
    sample_size: cell.n,
    window_days: cell.w || null,
    active_in_sample: cell.na != null ? cell.na : null,
    asking_price: { median: cell.fm, p25: cell.fl, p75: cell.fh },
    fair_value_estimate: cell.gm != null ? { median: cell.gm, low: cell.gl, high: cell.gh } : null,
    mileage_km_median: cell.km != null ? cell.km : null,
    share_of_model_listings: rec.n ? Math.round((cell.n / rec.n) * 1000) / 1000 : null,
    related: { model: `${base}/preco/${slug}`, methodology: `${base}/metodologia` },
  };
}

// ═══ Facets: /preco/{slug}/{combustivel} and /preco/{slug}/{distrito} ════════
//
// Two query clusters the model page cannot win because it answers them mixed
// together: "Golf diesel usado preço" and "carros usados Porto preços".
//
// Both are gated on the blob carrying `fx` / `dt` cells (added to
// model_pages.py alongside this). Until the pipeline publishes them these
// functions return nothing and the routes 404 — the pages appear by themselves
// on the next build, with no deploy.

/** Facet cells of one kind: "fuel" → rec.fx, "district" → rec.dt. */
export function facetCells(rec, kind) {
  const arr = kind === "fuel" ? rec.fx : kind === "transmission" ? rec.tx : rec.dt;
  return Array.isArray(arr) ? arr : [];
}

/** Find a facet cell by key, or null. */
export function facetCell(rec, kind, key) {
  return facetCells(rec, kind).find(c => c.k === key) || null;
}

/** Which facet kind a path segment belongs to, if any. */
export function facetKind(rec, key) {
  for (const kind of ["fuel", "transmission", "district"]) {
    if (publishedCells(rec, kind).some(c => c.k === key)) return kind;
  }
  return null;
}

export function retiredFacetKind(rec, key) {
  for (const kind of ["fuel", "transmission", "district"]) {
    if (facetCells(rec, kind).some(c => c.k === key)
        && !publishedCells(rec, kind).some(c => c.k === key)) return kind;
  }
  return null;
}

const FACET_SOLO_MAX_SHARE = 0.85;
const FACET_AGE_GAP_MIN = 0.05;

export function publishedCells(rec, kind) {
  const cells = facetCells(rec, kind);
  if (kind === "district" || cells.length !== 1) return cells;
  const only = cells[0];
  const share = rec && rec.n ? (only.n || 0) / rec.n : 0;
  return share >= FACET_SOLO_MAX_SHARE ? [] : cells;
}

/** Every facet URL segment this model publishes (for the sitemap). */
export function facetKeys(rec) {
  return [...publishedCells(rec, "fuel"), ...publishedCells(rec, "transmission"),
          ...publishedCells(rec, "district")].map(c => c.k);
}

export function renderFacetPage({ rec, slug, kind, cell, siblingsCells, stats, host, depositCount, builtAt, duelSpec = null, altJson = null }) {
  const B = escapeHtml(rec.b), M = escapeHtml(rec.m);
  const label = escapeHtml(cell.lbl);
  const canonical = `https://${host}/preco/${slug}/${cell.k}`;
  const isFuel = kind === "fuel";
  const isGear = kind === "transmission";
  // "um Golf diesel" vs "um Golf no Porto" — the preposition is the difference
  // between a sentence and a slot-filled template.
  const phrase = isFuel ? `${B} ${M} ${label.toLowerCase()}`
    : isGear ? `${B} ${M} com caixa ${label.toLowerCase()}`
    : `${B} ${M} no distrito de ${label}`;
  const titlePhrase = isFuel ? `${rec.b} ${rec.m} ${cell.lbl.toLowerCase()}`
    : isGear ? `${rec.b} ${rec.m} caixa ${cell.lbl.toLowerCase()}`
    : `${rec.b} ${rec.m} em ${cell.lbl}`;
  const sibHead = isFuel ? "Contra as outras motorizações"
    : isGear ? "Contra a outra caixa" : "Contra o resto do país";
  const sibLabel = isFuel ? "OUTRAS MOTORIZAÇÕES"
    : isGear ? "OUTRAS CAIXAS" : "OUTROS DISTRITOS";
  const share = rec.n ? Math.round(cell.n / rec.n * 100) : null;
  const matched = Array.isArray(cell.vsm) ? { pct: cell.vsm[0] - 1, years: cell.vsm[1] } : null;
  const normalized = (!matched && Array.isArray(cell.dr))
    ? { pct: cell.dr[0] - 1, used: cell.dr[1] } : null;
  const age = matched || normalized;
  const agePct = age ? Math.abs(Math.round(age.pct * 100)) : null;
  const ageMoves = !!(age && Math.abs(age.pct) >= FACET_AGE_GAP_MIN);
  const refAll = matched
    ? ((isFuel || isGear) ? "o modelo todo" : "o modelo no país inteiro")
    : ((isFuel || isGear) ? "os restantes cortes deste modelo" : "o mesmo modelo fora deste distrito");
  const ageMethod = matched
    ? `comparando ano a ano, sobre ${matched && matched.years} anos com amostra dos dois lados`
    : normalized ? `dividindo cada um dos ${normalized.used} anúncios pela mediana do resto do modelo no seu próprio ano de matrícula`
    : "";
  const vsAll = rec.fm > 0 ? (cell.fm - rec.fm) / rec.fm : null;
  const more = x => x >= 0 ? "mais" : "menos";
  const dearer = x => x >= 0 ? "mais caro" : "mais barato";

  let pin = 50;
  if (cell.fh > cell.fl) pin = Math.max(6, Math.min(94, Math.round((cell.fm - cell.fl) / (cell.fh - cell.fl) * 100)));

  // The comparison that makes this page worth existing: this facet against the
  // model's other facets of the same kind. "Diesel or petrol, which holds its
  // price" is answered everywhere with opinion and nowhere with a number.
  const others = siblingsCells.filter(c => c.k !== cell.k);
  const pairOf = o => (cell.vs && Array.isArray(cell.vs[o.k]))
    ? { pct: cell.vs[o.k][0] - 1, years: cell.vs[o.k][1] } : null;
  const compare = others.map(o => {
    const href = `/preco/${slug}/${o.k}`;
    const link = `<a href="${href}">${escapeHtml(o.lbl)}</a>`;
    const km = (o.km != null && cell.km != null)
      ? ` A quilometragem mediana difere em ${fmtKm(Math.abs(cell.km - o.km))} (${cell.km > o.km ? "mais" : "menos"} deste lado).` : "";
    const m = pairOf(o);
    if (m) {
      return `<li>Contra ${link} (${o.n} anúncios, mediana ${fmtEur(o.fm)}): <b>${Math.abs(Math.round(m.pct * 100))}% ${dearer(m.pct)}</b> comparando ano a ano, sobre ${m.years} anos com amostra dos dois lados. As medianas em bruto (${fmtEur(cell.fm)} contra ${fmtEur(o.fm)}) dizem outra coisa porque cada lado tem a sua mistura de idades.${km}</li>`;
    }
    return `<li>Contra ${link}: mediana ${fmtEur(o.fm)} em ${o.n} anúncios${o.y0 && o.y1 ? ` (anos ${o.y0}-${o.y1})` : ""}, contra ${fmtEur(cell.fm)} aqui${cell.y0 && cell.y1 ? ` (anos ${cell.y0}-${cell.y1})` : ""}. Não há anos que cheguem com amostra dos dois lados para comparar à mesma idade, por isso a distância entre as duas medianas ainda inclui a diferença de anos.${km}</li>`;
  }).join("");

  const ageNote = !age ? ""
    : ageMoves ? ` Ajustado pela idade, este corte pede ${more(age.pct)} ${agePct}% do que ${refAll}: a mediana acima é em bruto e inclui a diferença de anos.`
    : ` Ajustado pela idade, este corte pede o mesmo que ${refAll} — a distância entre as duas medianas em bruto é a diferença de anos.`;
  const body = crumbs([
    { name: "Início", href: "/" }, { name: "Preços", href: "/precos" },
    { name: `${rec.b} ${rec.m}`, href: `/preco/${slug}` }, { name: cell.lbl },
  ]) + `
    <div style="padding-top:14px;">
      <div class="side-card" style="max-width:680px;margin:0 auto;">
        <div class="eyebrow" style="margin-bottom:14px;"><span class="e-dot"></span><span class="mono">${escapeHtml(cell.lbl).toUpperCase()} · OLX PORTUGAL</span></div>
        <h1 class="fc-h1">Quanto vale um ${phrase} usado?</h1>
        <p class="lede" style="font-size:16px;margin:0 0 20px;">Em <b>${cell.n} anúncios ativos</b>${isFuel || isGear ? "" : " no distrito"}, um ${phrase} pede em mediana <b>${fmtEur(cell.fm)}</b>${cell.km != null ? `, com ${fmtKm(cell.km)} medianos` : ""}${cell.y0 && cell.y1 ? `, para anos ${cell.y0}-${cell.y1}` : ""}.${ageNote}</p>
        <div class="side-prices">
          <div><div class="cap">Preço mediano (pedido)</div><div class="big">${fmtEur(cell.fm)}</div></div>
          <div class="side-fair"><div class="cap">${cell.n} anúncios${share != null ? ` · ${share}% do modelo` : ""}</div><div class="v">${cell.km != null ? fmtKm(cell.km) : "—"}</div></div>
        </div>
        <div style="margin-top:16px;">
          <div class="gauge-head"><span>${fmtEur(cell.fl)}</span><span>intervalo típico (50% dos anúncios)</span><span>${fmtEur(cell.fh)}</span></div>
          <div class="gauge-track"><div class="gauge-pin" style="left:${pin}%;"></div></div>
        </div>
        ${provenance({ n: cell.n, builtAt, measure: `Preço pedido, ${titlePhrase} (mediana e P25-P75)` })}
      </div>
    </div>
    ${(compare || vsAll != null) ? `
    <section class="section fc-wrap">
      <h2 class="fc-h2">${sibHead}</h2>
      <ul class="fc-insights">
        ${age ? `<li>Face ${matched ? `a todos os ${B} ${M} do país (mediana ${fmtEur(rec.fm)})` : `${refAll}`}, este corte ${ageMoves ? `pede <b>${more(age.pct)} ${agePct}%</b>` : "pede <b>o mesmo</b>"}, ${ageMethod}.${Math.abs(cell.fm / rec.fm - 1) >= FACET_AGE_GAP_MIN ? ` As medianas em bruto (${fmtEur(cell.fm)} contra ${fmtEur(rec.fm)}) estão mais afastadas do que isso porque descrevem misturas de idades diferentes.` : ""}</li>`
          : vsAll != null ? `<li>Face a todos os ${B} ${M} do país, este corte pede em mediana ${fmtEur(cell.fm)} contra ${fmtEur(rec.fm)}${cell.y0 && cell.y1 ? `, em anos ${cell.y0}-${cell.y1}` : ""}. Os dois números não se subtraem: descrevem misturas de idades diferentes.</li>` : ""}
        ${compare}
      </ul>
    </section>` : ""}
    ${siblingsCells.length > 1 ? `
    <section class="section fc-wrap">
      <div class="sec-label" style="margin-bottom:10px;">${sibLabel}</div>
      <div class="fc-yearlinks">${siblingsCells.map(c =>
        c.k === cell.k ? `<a class="on" href="/preco/${slug}/${c.k}">${escapeHtml(c.lbl)}</a>`
                       : `<a href="/preco/${slug}/${c.k}">${escapeHtml(c.lbl)}</a>`).join("")}</div>
    </section>` : ""}
    <section class="section fc-wide">
      <div class="cta-banner">
        <div style="flex:1 1 360px;">
          <h2>Tens um ${phrase}?</h2>
          <p>Esta é a mediana do corte. Cola o link do teu anúncio e dizemos o valor justo desse carro em concreto.</p>
        </div>
        <a class="btn-bright" href="/avaliar?modelo=${encodeURIComponent(slug)}">Avaliar o meu carro&nbsp;&nbsp;→</a>
      </div>
    </section>
    <section class="section fc-wrap" style="padding-bottom:70px;">
      <p class="fc-p"><a href="/preco/${slug}">Todos os ${B} ${M}</a>${duelSpec ? ` · <a href="/${duelSpec.path}/${slug}">${escapeHtml(duelSpec.crumb)} neste modelo</a>` : ""}${isFuel || isGear ? "" : ` · <a href="/precos/${cell.k}">Carros usados ${emDistrito(cell.k, escapeHtml(cell.lbl))}</a>`} · <a href="/precos">Todos os modelos</a></p>
    </section>`;

  const faqs = [[
    `Quanto vale um ${titlePhrase} usado?`,
    `Em ${cell.n} anúncios ativos no OLX Portugal, um ${titlePhrase} pede em mediana ${fmtEur(cell.fm)}, com metade dos anúncios entre ${fmtEur(cell.fl)} e ${fmtEur(cell.fh)}. São preços pedidos, não preços de venda fechados.`,
  ]];
  if (others.length) {
    const o = others[0];
    const m = pairOf(o);
    const d = Math.round((m ? m.pct : (cell.fm - o.fm) / o.fm) * 100);
    faqs.push([
      isFuel ? `${rec.b} ${rec.m}: ${cell.lbl.toLowerCase()} ou ${o.lbl.toLowerCase()}?`
             : isGear ? `${rec.b} ${rec.m}: caixa ${cell.lbl.toLowerCase()} ou ${o.lbl.toLowerCase()}?`
             : `Um ${rec.b} ${rec.m} é mais caro em ${cell.lbl} ou em ${o.lbl}?`,
      m
        ? `Comparando ano a ano — só anos em que ambos têm amostra, ${m.years} ao todo — o corte ${cell.lbl.toLowerCase()} fica ${Math.abs(d)}% ${dearer(m.pct)} do que ${o.lbl.toLowerCase()}. As medianas em bruto (${fmtEur(cell.fm)} e ${fmtEur(o.fm)}) não se comparam directamente porque cada lado tem a sua mistura de anos.`
        : `A mediana pedida é ${fmtEur(cell.fm)} para ${cell.lbl.toLowerCase()} e ${fmtEur(o.fm)} para ${o.lbl.toLowerCase()}, mas em anos diferentes${cell.y0 && cell.y1 && o.y0 && o.y1 ? ` (${cell.y0}-${cell.y1} contra ${o.y0}-${o.y1})` : ""} e sem anos suficientes com amostra dos dois lados para comparar à mesma idade. A distância entre os dois números inclui a diferença de idades.`,
    ]);
  }

  return layout({
    title: `${titlePhrase} usado: ${fmtEur(cell.fm)} (${cell.n} anúncios) · quanto vale`,
    description: `${titlePhrase}: preço mediano ${fmtEur(cell.fm)} (${fmtEur(cell.fl)}–${fmtEur(cell.fh)}) em ${cell.n} anúncios ativos do OLX Portugal.${ageMoves ? ` Ajustado pela idade, o corte pede ${more(age.pct)} ${agePct}% do que ${refAll}.` : ""}`,
    canonical, body, zone: "all", nav: "precos", depositCount, index: true, host, altJson,
    jsonLd: {
      "@context": "https://schema.org",
      "@graph": [
        {
          "@type": "Dataset",
        "license": licenseUrl(host),
          "name": `Preços de ${titlePhrase} em Portugal`,
          "description": `Mediana e intervalo interquartil dos preços pedidos em ${cell.n} anúncios ativos de ${titlePhrase} no OLX Portugal.`,
          "creator": { "@type": "Organization", "name": "Carsbuyer", "url": `https://${host}/` },
          "isAccessibleForFree": true, "dateModified": builtAt || undefined,
          "variableMeasured": "Preço pedido (EUR)", "url": canonical,
          "distribution": [{ "@type": "DataDownload", "encodingFormat": "application/json", "contentUrl": `${canonical}.json` }],
          ...(cell.y0 && cell.y1 ? { "temporalCoverage": `${cell.y0}/${cell.y1}` } : {}),
        },
        {
          "@type": "AggregateOffer", "priceCurrency": "EUR",
          "lowPrice": cell.fl, "highPrice": cell.fh, "offerCount": cell.n, "url": canonical,
          "itemOffered": {
            "@type": "Car", "name": titlePhrase,
            "brand": { "@type": "Brand", "name": rec.b }, "model": rec.m,
            ...(isFuel ? { "fuelType": cell.lbl } : {}),
            ...(isGear ? { "vehicleTransmission": cell.lbl } : {}),
          },
        },
        breadcrumbLd(host, [
          { name: "Início", href: "/" }, { name: "Preços", href: "/precos" },
          { name: `${rec.b} ${rec.m}`, href: `/preco/${slug}` }, { name: cell.lbl },
        ]),
        faqLd(faqs),
      ],
    },
  });
}

// ═══ /precos/{distrito} — the market in one district ═════════════════════════
// Portuguese district names and the definite article.
//
// Most districts take a bare "em Lisboa" / "em Braga" / "em Faro". Porto takes
// the article — "no Porto" — and "em Porto" reads wrong to any Portuguese
// speaker, which on a site whose whole claim is local credibility is a worse
// look than it is a grammar slip. An exception list rather than a rule, because
// that is honestly what it is: the article is lexical, not derivable.
const DISTRICT_ARTICLE = { porto: "no" };
function emDistrito(key, label) {
  return `${DISTRICT_ARTICLE[key] || "em"} ${label}`;
}

export function districtRanking(districts, key) {
  const rows = Object.entries(districts || {})
    .filter(([, r]) => r && r.fm > 0)
    .map(([k, r]) => ({ k, lbl: r.lbl, n: r.n, fm: r.fm, kmm: r.kmm != null ? r.kmm : null }))
    .sort((a, b) => b.fm - a.fm);
  const pos = rows.findIndex(r => r.k === key);
  return { rows, pos: pos < 0 ? null : pos + 1, total: rows.length };
}

export function renderDistrictPage({ key, rec, models, districts, stats, host, depositCount, builtAt }) {
  const canonical = `https://${host}/precos/${key}`;
  const L = escapeHtml(rec.lbl);
  const vsNational = stats.priceMed ? (rec.fm - stats.priceMed) / stats.priceMed : null;
  const rows = (rec.top || []).map(([slug, n, fm]) => {
    const m = models[slug];
    if (!m) return "";
    const d = m.fm > 0 ? (fm - m.fm) / m.fm : null;
    return `<tr>
      <td><a href="/preco/${slug}" style="color:#177A47;font-weight:600;">${escapeHtml(m.b)} ${escapeHtml(m.m)}</a></td>
      <td>${n}</td><td>${fmtEur(fm)}</td>
      <td class="mut">${fmtEur(m.fm)}</td>
      <td>${d == null ? "—" : `${d >= 0 ? "+" : ""}${Math.round(d * 100)}%`}</td></tr>`;
  }).join("");

  const rank = districtRanking(districts || {}, key);
  const ranking = rank.rows.length >= 3 ? rank.rows.map(r => `<tr>
      <td>${r.k === key ? `<b>${escapeHtml(r.lbl)}</b>` : `<a href="/precos/${encodeURIComponent(r.k)}" style="color:#177A47;font-weight:600;">${escapeHtml(r.lbl)}</a>`}</td>
      <td>${fmtEur(r.fm)}</td>
      <td class="mut">${r.k === key ? "—" : `${r.fm >= rec.fm ? "+" : ""}${Math.round((r.fm / rec.fm - 1) * 100)}%`}</td>
      <td class="mut">${fmtNum(r.n)}</td></tr>`).join("") : "";
  const rankLine = (rank.pos && rank.rows.length >= 3)
    ? `Entre os ${rank.total} distritos com amostra suficiente, ${L} é o <b>${rank.pos}.º mais caro</b>: ${fmtEur(rank.rows[0].fm)} no topo (${escapeHtml(rank.rows[0].lbl)}) contra ${fmtEur(rank.rows[rank.rows.length - 1].fm)} no fim (${escapeHtml(rank.rows[rank.rows.length - 1].lbl)}). A diferença entre pontas é de ${Math.round((rank.rows[0].fm / rank.rows[rank.rows.length - 1].fm - 1) * 100)}%, e não é o mesmo carro a custar mais: onde os preços são mais altos anuncia-se também outro tipo de carro, mais recente e com menos quilómetros.`
    : "";

  const body = crumbs([
    { name: "Início", href: "/" }, { name: "Preços", href: "/precos" }, { name: rec.lbl },
  ]) + `
    <section class="section fc-wrap" style="padding-top:16px;">
      <div class="eyebrow" style="margin-bottom:14px;"><span class="e-dot"></span><span class="mono">${L.toUpperCase()} · OLX PORTUGAL</span></div>
      <h1 class="fc-h1">Preços de carros usados ${emDistrito(key, L)}</h1>
      <p class="fc-p">Nos <b>${fmtNum(rec.n)} anúncios ativos</b> com localização ${emDistrito(key, L)}, o preço pedido mediano é <b>${fmtEur(rec.fm)}</b>${rec.kmm != null ? `, com ${fmtKm(rec.kmm)} de quilometragem mediana` : ""}.${vsNational != null ? ` Isso é <b>${Math.abs(Math.round(vsNational * 100))}% ${vsNational >= 0 ? "acima" : "abaixo"}</b> da mediana nacional (${fmtEur(stats.priceMed)}).` : ""}</p>
      <div class="fc-stat-row" style="margin:18px 0 6px;">
        <div class="fc-stat"><div class="k">MEDIANO AQUI</div><div class="v">${fmtEur(rec.fm)}</div><div class="s">${fmtEur(rec.fl)} – ${fmtEur(rec.fh)}</div></div>
        <div class="fc-stat"><div class="k">ANÚNCIOS</div><div class="v">${fmtNum(rec.n)}</div><div class="s">ativos agora</div></div>
        ${rec.kmm != null ? `<div class="fc-stat"><div class="k">KM MEDIANO</div><div class="v">${fmtNum(rec.kmm)}</div><div class="s">à venda aqui</div></div>` : ""}
      </div>
      ${provenance({ n: rec.n, builtAt, measure: `Preço pedido em anúncios com localização em ${rec.lbl}` })}
    </section>
    ${rows ? `<section class="section fc-wrap">
      <h2 class="fc-h2">Modelos mais anunciados ${emDistrito(key, L)}</h2>
      <p class="fc-p">A última coluna é o que interessa: onde o preço local se afasta do nacional, ou há mais oferta aqui, ou o mesmo carro custa mais por causa da procura.</p>
      <div class="fc-scroll"><table class="fc-tbl">
        <thead><tr><th>Modelo</th><th>Anúncios aqui</th><th>Mediano aqui</th><th>Mediano nacional</th><th>Diferença</th></tr></thead>
        <tbody>${rows}</tbody></table></div>
    </section>` : `<section class="section fc-wrap">
      <h2 class="fc-h2">Modelo a modelo, aqui não dá</h2>
      <p class="fc-p">Os ${fmtNum(rec.n)} anúncios ${emDistrito(key, L)} chegam para uma mediana do distrito, mas não para uma mediana por modelo: nenhum modelo tem anúncios que cheguem aqui para que a sua mediana signifique alguma coisa. Em vez de a inventar com quatro carros, ficamos pelo que o distrito diz no seu conjunto e pela comparação com o resto do país, aqui em baixo. Para um modelo concreto, os <a href="/precos">preços nacionais</a> são a referência mais firme que temos.</p>
    </section>`}
    ${ranking ? `<section class="section fc-wrap">
      <h2 class="fc-h2">Onde ${L} fica no país</h2>
      <p class="fc-p">${rankLine}</p>
      <div class="fc-scroll"><table class="fc-tbl">
        <thead><tr><th>Distrito</th><th>Preço mediano</th><th>vs. ${L}</th><th>Anúncios</th></tr></thead>
        <tbody>${ranking}</tbody></table></div>
    </section>` : ""}
    <section class="section fc-wrap" style="padding-bottom:70px;">
      <p class="fc-p"><a href="/precos">Todos os modelos</a> · <a href="/mercado">Carros abaixo do valor justo</a> · <a href="/avaliar">Avaliar um anúncio</a></p>
    </section>`;

  return layout({
    title: `Preços de carros usados ${emDistrito(key, rec.lbl)}`,
    description: `Carros usados ${emDistrito(key, rec.lbl)}: preço mediano ${fmtEur(rec.fm)} em ${fmtNum(rec.n)} anúncios ativos do OLX, e como cada modelo se compara com a mediana nacional.`,
    canonical, body, zone: "all", nav: "precos", depositCount, index: true, host,
    jsonLd: {
      "@context": "https://schema.org",
      "@graph": [
        {
          "@type": "Dataset", "license": licenseUrl(host), "url": canonical, "inLanguage": "pt-PT",
          "name": `Preços de carros usados ${emDistrito(key, rec.lbl)}`,
          "description": `Preço pedido mediano e intervalo interquartil em ${rec.n} anúncios ativos de carros usados com localização ${emDistrito(key, rec.lbl)}, OLX Portugal.`,
          "creator": { "@type": "Organization", "name": "Carsbuyer", "url": `https://${host}/` },
          "isAccessibleForFree": true, "dateModified": builtAt || undefined,
          "spatialCoverage": { "@type": "Place", "name": rec.lbl, "containedInPlace": { "@type": "Country", "name": "Portugal" } },
          "variableMeasured": "Preço pedido (EUR)",
        },
        {
          "@type": "AggregateOffer", "priceCurrency": "EUR",
          "lowPrice": rec.fl, "highPrice": rec.fh, "offerCount": rec.n, "url": canonical,
          "areaServed": { "@type": "Place", "name": rec.lbl },
        },
        breadcrumbLd(host, [
          { name: "Início", href: "/" }, { name: "Preços", href: "/precos" }, { name: rec.lbl },
        ]),
      ],
    },
  });
}

export const IMPORT_MIN_CELLS = 2;

export function importOk(rec) {
  return !!(rec && Array.isArray(rec.yr) && rec.yr.length >= IMPORT_MIN_CELLS);
}

export function importSlugs(doc) {
  const models = (doc && doc.models) || {};
  return Object.entries(models)
    .filter(([, r]) => importOk(r))
    .sort((a, b) => (b[1].med_gap || 0) - (a[1].med_gap || 0))
    .map(([slug]) => slug);
}

function importVerdict(rec) {
  const cells = rec.yr || [];
  const wins = cells.filter(c => c.gl > 0).length;
  const best = cells.slice().sort((a, b) => b.gl - a.gl)[0] || null;
  const worst = cells.slice().sort((a, b) => a.gl - b.gl)[0] || null;
  return { wins, total: cells.length, best, worst, always: wins === cells.length, never: wins === 0 };
}

export function importJson(rec, slug, costs, { host, builtAt } = {}) {
  return {
    slug, brand: rec.b, model: rec.m,
    url: `https://${host}/importar/${slug}`,
    question: "does importing this model from Germany land under the Portuguese asking price",
    sample_de: rec.nde, sample_pt: rec.npt,
    fixed_costs_eur: costs ? { low: costs.lo, high: costs.hi, items: costs.items } : null,
    years: (rec.yr || []).map(c => ({
      year: c.y,
      de_listings: c.nde, pt_listings: c.npt,
      de_asking_p25: c.dl, de_asking_median: c.dm, de_asking_p75: c.dh,
      isv_median_eur: c.isv, isv_sample: c.isvn,
      landed_low_eur: c.ll, landed_high_eur: c.lh,
      pt_asking_median: c.ptm,
      saving_low_eur: c.gl, saving_high_eur: c.gh,
    })),
    updated: builtAt || null,
    licence: licenseUrl(host),
    caveat: "asking prices on both sides, not transaction prices; the ISV is our own estimate from each German listing's CO2, engine size and first registration",
  };
}

export function renderImportPage({ rec, slug, costs, stats, hasModelPage = true,
                                  host, depositCount, builtAt, historyUrl = null }) {
  const B = escapeHtml(rec.b), M = escapeHtml(rec.m);
  const canonical = `https://${host}/importar/${slug}`;
  const v = importVerdict(rec);
  const cells = rec.yr || [];
  const lo = costs ? costs.lo : null, hi = costs ? costs.hi : null;

  const verdictLine = v.always
    ? `Nos <b>${v.total} anos</b> que conseguimos comparar, importar sai mais barato — entre ${fmtEur(v.worst.gl)} e ${fmtEur(v.best.gh)} conforme o ano e conforme o que pagares de transporte e papelada.`
    : v.never
      ? `Em <b>nenhum</b> dos ${v.total} anos comparados a conta fecha a favor da importação: depois do ISV e da legalização, o ${B} ${M} alemão chega cá mais caro do que o que já está à venda em Portugal.`
      : `Depende do ano, e essa é a resposta honesta: em <b>${v.wins} dos ${v.total}</b> anos comparados importar sai mais barato, no melhor deles cerca de ${fmtEur(v.best.gl)}; nos outros a conta fecha contra.`;

  const rows = cells.map(c => `<tr>
      <td>${c.y}</td>
      <td>${fmtEur(c.dm)} <span class="mut">${fmtEur(c.dl)}–${fmtEur(c.dh)}</span></td>
      <td class="mut">${fmtEur(c.isv)}</td>
      <td class="mut">${fmtEur(lo)}–${fmtEur(hi)}</td>
      <td>${fmtEur(c.ll)}–${fmtEur(c.lh)}</td>
      <td>${fmtEur(c.ptm)}</td>
      <td style="font-weight:600;color:${c.gl > 0 ? "#177A47" : "#9B2C2C"};">${c.gl > 0 ? `poupa ${fmtEur(c.gl)}` : `mais ${fmtEur(-c.gl)}`}</td>
      <td class="mut">${(c.dkm != null && c.ptkm != null) ? `${fmtKm(c.dkm)} / ${fmtKm(c.ptkm)}` : "—"}</td>
      <td class="mut">${c.nde} / ${c.npt}</td></tr>`).join("");
  const kmGap = rec.km_gap;
  const kmLine = (kmGap != null && Math.abs(kmGap) >= 0.15)
    ? `<p class="fc-p">Uma diferença que a tabela mostra e a conta não corrige: os ${B} ${M} à venda na Alemanha têm, na mediana, <b>${Math.abs(Math.round(kmGap * 100))}% ${kmGap > 0 ? "mais" : "menos"} quilómetros</b> do que os portugueses do mesmo ano. ${kmGap > 0 ? "Ou seja, parte do que parece poupança é um carro mais rodado." : "Ou seja, o carro alemão tende a estar menos rodado, e a poupança na tabela é conservadora."} Comparamos o mesmo ano de matrícula, não a mesma quilometragem.</p>`
    : "";

  const costRows = (costs && costs.items ? costs.items : []).map(i => `<tr>
      <td>${escapeHtml(i.lbl)}</td>
      <td>${i.lo === i.hi ? fmtEur(i.lo) : `${fmtEur(i.lo)} – ${fmtEur(i.hi)}`}</td>
      <td class="mut">${escapeHtml(i.src || "")}</td></tr>`).join("");

  const body = crumbs([
    { name: "Início", href: "/" }, { name: "Importar", href: "/importar" },
    { name: `${rec.b} ${rec.m}` },
  ]) + `
    <div style="padding-top:14px;">
      <div class="side-card" style="max-width:680px;margin:0 auto;">
        <div class="eyebrow" style="margin-bottom:14px;"><span class="e-dot"></span><span class="mono">IMPORTAR DA ALEMANHA · ${B.toUpperCase()} ${M.toUpperCase()}</span></div>
        <h1 class="fc-h1">Vale a pena importar um ${B} ${M} da Alemanha?</h1>
        <p class="lede" style="font-size:16px;margin:0 0 18px;">Comparámos <b>${fmtNum(rec.nde)} anúncios alemães</b> com <b>${fmtNum(rec.npt)} anúncios portugueses</b> do mesmo modelo, ano a ano, somando ao preço alemão o ISV que esse carro concreto pagaria e a legalização. ${verdictLine}</p>
        <div class="fc-stat-row">
          <div class="fc-stat"><div class="k">ANOS A FAVOR</div><div class="v">${v.wins}/${v.total}</div><div class="s">anos comparados</div></div>
          ${v.best ? `<div class="fc-stat"><div class="k">MELHOR ANO</div><div class="v">${v.best.y}</div><div class="s">${v.best.gl > 0 ? `poupa ${fmtEur(v.best.gl)}` : `perde ${fmtEur(-v.best.gl)}`}</div></div>` : ""}
          <div class="fc-stat"><div class="k">ISV MEDIANO</div><div class="v">${fmtEur(rec.isv_med != null ? rec.isv_med : cells[0].isv)}</div><div class="s">calculado por anúncio</div></div>
          ${lo != null ? `<div class="fc-stat"><div class="k">RESTO DA CONTA</div><div class="v">${fmtEur(lo)}+</div><div class="s">até ${fmtEur(hi)} sem o ISV</div></div>` : ""}
        </div>
        ${provenance({ n: rec.nde, builtAt, unit: "anúncios alemães", measureId: "import-landed-cost",
                       source: "AutoScout24 (Alemanha) e OLX (Portugal)",
                       measure: "Preço pedido na Alemanha + ISV estimado + custos de legalização, contra o preço pedido em Portugal" })}
      </div>
    </div>
    <section class="section fc-wrap">
      <h2 class="fc-h2">A conta, ano a ano</h2>
      <div class="fc-scroll"><table class="fc-tbl">
        <thead><tr><th>Ano</th><th>Alemanha (pedido)</th><th>ISV</th><th>Legalização</th><th>Total à porta</th><th>Portugal (pedido)</th><th>Diferença</th><th>Km DE/PT</th><th>Anúncios DE/PT</th></tr></thead>
        <tbody>${rows}</tbody></table></div>
      ${kmLine}
      <p class="fc-p" style="margin-top:12px;">A coluna que decide é a última mas uma: <b>total à porta</b> é o que o carro alemão te fica depois de pago tudo, e é isso que se compara com o preço pedido em Portugal — não o preço alemão sozinho, que é o número com que toda a gente vende a ideia da importação.</p>
      <p class="fc-p">Cada linha compara o <b>mesmo ano de matrícula</b> dos dois lados. Sem isso a conta não significa nada: a oferta alemã costuma ser mais nova do que a portuguesa, e uma diferença que fosse só idade apareceria como poupança.</p>
    </section>
    <section class="section fc-wrap">
      <h2 class="fc-h2">O que está nesta conta</h2>
      <div class="fc-scroll"><table class="fc-tbl">
        <thead><tr><th>Rubrica</th><th>Valor</th><th>Nota</th></tr></thead>
        <tbody>${costRows}
          <tr><td><b>ISV</b></td><td><b>varia com o carro</b></td><td class="mut">calculado a partir do CO2, cilindrada e ano de cada anúncio — <a href="/isv">simulador</a></td></tr>
        </tbody></table></div>
      <p class="fc-p" style="margin-top:12px;">O IVA não aparece aqui de propósito: num usado com mais de seis meses e mais de 6 000 km o IVA é pago no país onde se compra e não volta a ser pago em Portugal. O preço alemão que usamos é o preço pedido ao público, com o IVA alemão lá dentro quando o vendedor é um stand.</p>
    </section>
    <section class="section fc-wrap">
      <h2 class="fc-h2">O que esta conta não sabe</h2>
      <ul class="fc-ul">
        <li class="fc-li"><b>São preços pedidos dos dois lados</b>, não preços de venda. Negoceia-se na Alemanha e negoceia-se cá, e não sabemos de que lado se cede mais.</li>
        <li class="fc-li"><b>O estado do carro concreto.</b> Um anúncio não diz o histórico de manutenção nem o que vai aparecer na inspeção — e um carro comprado à distância é comprado com menos informação, não com mais.</li>
        <li class="fc-li"><b>A versão e os quilómetros.</b> Emparelhamos pelo ano de matrícula, e dentro do mesmo ano cabe muita coisa — motorizações diferentes, níveis de equipamento diferentes e quilometragens diferentes. É por isso que a coluna alemã traz o intervalo P25-P75 e não só a mediana.</li>
        <li class="fc-li"><b>O teu tempo e as tuas deslocações</b>, a garantia que perdes ou ganhas, e o risco de a legalização correr mal.</li>
        <li class="fc-li"><b>O ISV é a nossa estimativa</b>, a partir do CO2 e da cilindrada que o vendedor alemão declarou. Para carros de 2018 e 2019 o ciclo de medição é ambíguo, o que mexe no valor.</li>
      </ul>
    </section>
    ${historyUrl ? `<section class="section fc-wrap">${historyCheckBlock({
      url: historyUrl, from: "importar",
      title: `Num importado, o histórico é a parte que mais importa`,
      reasons: [
        "Um carro vindo de fora chega sem o registo português de inspeções e donos; o relatório internacional é a única forma de ver quilómetros e sinistros anteriores.",
        "Pede o VIN ao vendedor antes da viagem: o relatório custa menos do que uma ida à Alemanha.",
      ],
    })}</section>` : ""}
    <section class="section fc-wide">
      <div class="cta-banner">
        <div style="flex:1 1 360px;">
          <h2>Tens um carro concreto em vista?</h2>
          <p>Esta página é a mediana do modelo. Para o anúncio que estás a ver, mete a cilindrada, o CO2 e o ano no simulador e fica com o ISV desse carro.</p>
        </div>
        <a class="btn-bright" href="/isv">Simular o ISV&nbsp;&nbsp;→</a>
      </div>
    </section>
    <section class="section fc-wrap" style="padding-bottom:70px;">
      <p class="fc-p">${hasModelPage ? `<a href="/preco/${slug}">Preços de ${B} ${M} em Portugal</a> · ` : ""}<a href="/importar">Outros modelos que vale a pena comparar</a> · <a href="/isv">Simulador de ISV</a> · <a href="${canonical}.json">Dados em JSON</a></p>
    </section>`;

  const faqs = [
    [`Vale a pena importar um ${rec.b} ${rec.m} da Alemanha?`,
     v.never
       ? `Pelos nossos números, não: em nenhum dos ${v.total} anos comparados o preço alemão mais ISV e legalização fica abaixo do que se pede em Portugal por um ${rec.b} ${rec.m} do mesmo ano.`
       : `Em ${v.wins} dos ${v.total} anos comparados, sim. No melhor ano (${v.best.y}) a diferença a favor da importação é de cerca de ${fmtEur(v.best.gl)} depois de somar o ISV e a legalização ao preço pedido na Alemanha.`],
    [`Quanto custa legalizar um ${rec.b} ${rec.m} importado?`,
     `Fora o ISV, entre ${fmtEur(lo)} e ${fmtEur(hi)}: transporte, certificado de conformidade, inspeção tipo B, matrícula no IMT e registo de propriedade. O ISV depende do carro — para este modelo a mediana dos anúncios alemães que conseguimos calcular anda pelos ${fmtEur(rec.isv_med != null ? rec.isv_med : cells[0].isv)}.`],
    [`O preço alemão já inclui o IVA?`,
     `Nos anúncios de stand normalmente sim, e é esse o valor que usamos. Num usado com mais de seis meses e 6 000 km o IVA fica pago na Alemanha e não é cobrado outra vez em Portugal; só um comprador com IVA dedutível recupera a parte alemã.`],
  ];

  return layout({
    title: `Importar ${rec.b} ${rec.m} da Alemanha: vale a pena?`,
    description: `${rec.b} ${rec.m}: preço pedido na Alemanha mais ISV e legalização, comparado ano a ano com o preço pedido em Portugal. ${v.never ? "Pelos nossos números não compensa." : `Compensa em ${v.wins} de ${v.total} anos.`}`,
    canonical, body, zone: "all", nav: "precos", depositCount, index: true, host,
    altJson: `${canonical}.json`,
    jsonLd: {
      "@context": "https://schema.org",
      "@graph": [
        {
          "@type": "Dataset", "license": licenseUrl(host), "url": canonical, "inLanguage": "pt-PT",
          "name": `Importar ${rec.b} ${rec.m} da Alemanha: custo total contra o preço português`,
          "description": `Preço pedido mediano de ${rec.b} ${rec.m} na Alemanha por ano de matrícula, ISV estimado por anúncio, custos de legalização e a diferença para o preço pedido em Portugal.`,
          "creator": { "@type": "Organization", "name": "Carsbuyer", "url": `https://${host}/` },
          "isAccessibleForFree": true, "dateModified": builtAt || undefined,
          "variableMeasured": ["Preço pedido na Alemanha (EUR)", "ISV estimado (EUR)", "Custo total à porta (EUR)", "Preço pedido em Portugal (EUR)"],
          "distribution": [{ "@type": "DataDownload", "encodingFormat": "application/json", "contentUrl": `${canonical}.json` }],
        },
        breadcrumbLd(host, [
          { name: "Início", href: "/" }, { name: "Importar", href: "/importar" },
          { name: `${rec.b} ${rec.m}` },
        ]),
        faqLd(faqs),
      ],
    },
  });
}

export function renderImportHub({ rows, costs, host, depositCount, builtAt }) {
  const canonical = `https://${host}/importar`;
  const lo = costs ? costs.lo : null, hi = costs ? costs.hi : null;
  const tr = rows.map(r => `<tr>
      <td><a href="/importar/${r.slug}" style="color:#177A47;font-weight:600;">${escapeHtml(r.b)} ${escapeHtml(r.m)}</a></td>
      <td style="font-weight:600;color:${r.med_gap > 0 ? "#177A47" : "#9B2C2C"};">${r.med_gap > 0 ? "−" : "+"}${fmtEur(Math.abs(r.med_gap))}</td>
      <td class="mut">${r.wins}/${r.cells}</td>
      <td class="mut">${fmtNum(r.nde)}</td>
      <td class="mut">${fmtNum(r.npt)}</td></tr>`).join("");
  const winners = rows.filter(r => r.med_gap > 0).length;
  const body = crumbs([{ name: "Início", href: "/" }, { name: "Importar" }]) + `
    <section class="section fc-wrap" style="padding-top:16px;">
      <h1 class="fc-h1">Importar da Alemanha: em que modelos a conta fecha</h1>
      <p class="fc-p">Toda a gente que vende importação mostra o mesmo: um simulador de ISV. Um ISV sozinho não decide nada — o que decide é o preço alemão <b>mais</b> o imposto <b>mais</b> a legalização, contra o que o mesmo carro pede em Portugal hoje. É essa conta que está aqui, ano a ano, com as duas pontas medidas em anúncios reais: AutoScout24 de um lado, OLX do outro.</p>
      <p class="fc-p">Em <b>${winners} dos ${rows.length}</b> modelos que conseguimos comparar a importação fecha a favor na mediana dos anos. Nos outros, não — e isso também é resposta.${lo != null ? ` Fora o ISV, a legalização anda entre ${fmtEur(lo)} e ${fmtEur(hi)}.` : ""}</p>
      <div class="fc-scroll"><table class="fc-tbl">
        <thead><tr><th>Modelo</th><th>Diferença mediana</th><th>Anos a favor</th><th>Anúncios DE</th><th>Anúncios PT</th></tr></thead>
        <tbody>${tr}</tbody></table></div>
      ${provenance({ n: rows.reduce((s, r) => s + (r.nde || 0), 0), builtAt, unit: "anúncios alemães",
                     measureId: "import-landed-cost",
                     source: "AutoScout24 (Alemanha) e OLX (Portugal)",
                     measure: "Preço pedido na Alemanha + ISV estimado + legalização, contra o preço pedido em Portugal" })}
      <p class="fc-p" style="margin-top:18px;"><a href="/isv">Simulador de ISV</a> · <a href="/precos">Preços em Portugal por modelo</a> · <a href="/metodologia">Como calculamos</a></p>
    </section>
    <div style="height:60px;"></div>`;
  return layout({
    title: "Importar carro da Alemanha: em que modelos compensa",
    description: `Preço alemão mais ISV e legalização contra o preço pedido em Portugal, modelo a modelo e ano a ano. Compensa em ${winners} dos ${rows.length} modelos comparados.`,
    canonical, body, zone: "all", nav: "precos", depositCount, index: true, host,
    jsonLd: {
      "@context": "https://schema.org",
      "@graph": [
        {
          "@type": "Dataset", "license": licenseUrl(host), "url": canonical, "inLanguage": "pt-PT",
          "name": "Custo total de importar da Alemanha, por modelo",
          "description": "Preço pedido mediano na Alemanha, ISV estimado, custos de legalização e a diferença para o preço pedido em Portugal, por modelo.",
          "creator": { "@type": "Organization", "name": "Carsbuyer", "url": `https://${host}/` },
          "isAccessibleForFree": true, "dateModified": builtAt || undefined,
          "variableMeasured": ["Custo total à porta (EUR)", "Preço pedido em Portugal (EUR)"],
        },
        breadcrumbLd(host, [{ name: "Início", href: "/" }, { name: "Importar" }]),
      ],
    },
  });
}

export const DUEL_SIG_T = 1.96;

export const DUELS = {
  fuel: {
    kind: "fuel", key: "dg", path: "diesel-ou-gasolina",
    crumb: "Diesel ou gasolina", eyebrow: "DIESEL vs. GASOLINA",
    question: "diesel ou gasolina",
    hubH1: "Diesel ou gasolina: onde a escolha muda mesmo o preço",
    hubTitle: "Diesel ou gasolina: qual segura melhor o preço, modelo a modelo",
    hubLede: "combustível",
    a: { lbl: "Diesel", chip: "DIESEL", low: "diesel", subj: "o diesel",
         facet: "diesel", only: "Só diesel", json: "diesel" },
    b: { lbl: "Gasolina", chip: "GASOLINA", low: "gasolina", subj: "a gasolina",
         facet: "gasolina", only: "Só gasolina", json: "gasoline" },
  },
  gear: {
    kind: "gear", key: "cx", path: "manual-ou-automatica",
    crumb: "Manual ou automática", eyebrow: "CAIXA MANUAL vs. AUTOMÁTICA",
    question: "caixa manual ou automática",
    hubH1: "Manual ou automática: onde a caixa muda mesmo o preço",
    hubTitle: "Caixa manual ou automática: qual segura melhor o preço, modelo a modelo",
    hubLede: "caixa",
    a: { lbl: "Manual", chip: "MANUAL", low: "caixa manual", subj: "a caixa manual",
         facet: "manual", only: "Só manual", json: "manual" },
    b: { lbl: "Automática", chip: "AUTOMÁTICA", low: "caixa automática",
         subj: "a caixa automática", facet: "automatica", only: "Só automática",
         json: "automatic" },
  },
};

export function duelSpec(kind) {
  return DUELS[kind] || null;
}

export function duelByPath(segment) {
  return Object.values(DUELS).find(d => d.path === segment) || null;
}

export function duel(rec, kind, builtAt) {
  const spec = DUELS[kind];
  const g = spec && rec ? rec[spec.key] : null;
  if (!g || !g.a || !g.b || !(g.a.r > 0) || !(g.b.r > 0)) return null;
  const built = parseInt((builtAt || "").slice(0, 4), 10);
  const ref = Number.isFinite(built) ? built : new Date().getUTCFullYear();
  const a0 = Math.max(0, ref - (g.y1 || ref));
  const a1 = Math.max(a0 + 1, ref - (g.y0 || ref));
  const diff = g.b.r - g.a.r;
  return {
    spec, a: g.a, b: g.b, ci: g.ci || 0, t: g.t || 0, r2: g.r2 || 0,
    y0: g.y0, y1: g.y1, gap: Array.isArray(g.gap) ? g.gap : [],
    ref, a0, a1, diff, n: (g.a.n || 0) + (g.b.n || 0),
    decisive: Math.abs(g.t || 0) >= DUEL_SIG_T,
    winner: diff > 0 ? "a" : "b",
  };
}

export function duelOk(rec, kind) {
  const spec = DUELS[kind];
  const g = spec && rec ? rec[spec.key] : null;
  return !!(g && g.a && g.b && g.a.r > 0 && g.b.r > 0);
}

/**
 * Duel pages are OUTSIDE the SEO_WAVE_MODELS gate, like the comparison pages
 * and unlike everything per-model: the whole layer is a few dozen URLs, and its
 * own publishing gate (20 listings a side, a 95% interval under 3pp/yr, R²) is
 * already stricter than anything a wave could stage. There is nothing here to
 * release in batches.
 */
export function publishedDuel(models, slug, rec, builtAt, kind) {
  return duelOk(rec, kind);
}

export function duelSlugs(models, kind, builtAt) {
  const spec = DUELS[kind];
  if (!spec) return [];
  return Object.entries(models)
    .filter(([slug, r]) => publishedDuel(models, slug, r, builtAt, kind))
    .sort((x, y) => Math.abs(y[1][spec.key].b.r - y[1][spec.key].a.r)
                  - Math.abs(x[1][spec.key].b.r - x[1][spec.key].a.r))
    .map(([s]) => s);
}

/** Every duel this model publishes, for the model page's own links. */
export function duelsFor(models, slug, rec, builtAt) {
  return Object.keys(DUELS)
    .filter(k => publishedDuel(models, slug, rec, builtAt, k))
    .map(k => DUELS[k]);
}

const CONTRACTIONS = { "de|o": "do", "de|a": "da", "a|o": "ao", "a|a": "à" };

export function withPrep(prep, subject) {
  const m = /^(o|a)\s+(.*)$/.exec(subject || "");
  if (!m) return `${prep} ${subject}`;
  return `${CONTRACTIONS[`${prep}|${m[1]}`] || `${prep} ${m[1]}`} ${m[2]}`;
}

const keepPct = (rate, years) => Math.round(Math.pow(1 - rate, years) * 100);
const ppc = x => (Math.abs(x) * 100).toFixed(1).replace(".", ",");
const pctc = x => (x * 100).toFixed(1).replace(".", ",");

export function retentionChart(av, { w = 640, h = 240 } = {}) {
  if (!av) return "";
  const padL = 34, padR = 14, padT = 22, padB = 34;
  const a0 = av.a0, a1 = av.a1;
  const X = a => padL + ((a - a0) / Math.max(1, a1 - a0)) * (w - padL - padR);
  const Y = v => padT + (1 - v) * (h - padT - padB);
  const curve = (rate) => {
    let dpath = "";
    const step = Math.max(0.25, (a1 - a0) / 120);
    for (let a = a0; a <= a1 + 1e-9; a += step) {
      dpath += `${dpath ? "L" : "M"}${X(a).toFixed(1)},${Y(Math.pow(1 - rate, a - a0)).toFixed(1)}`;
    }
    return dpath + `L${X(a1).toFixed(1)},${Y(Math.pow(1 - rate, a1 - a0)).toFixed(1)}`;
  };
  const ticks = [0, 0.25, 0.5, 0.75, 1].map(f =>
    `<line x1="${padL}" x2="${w - padR}" y1="${Y(f).toFixed(1)}" y2="${Y(f).toFixed(1)}" class="c-grid"/>`
    + `<text x="${padL - 5}" y="${(Y(f) + 4).toFixed(1)}" text-anchor="end" class="c-ax">${Math.round(f * 100)}%</text>`).join("");
  const step = Math.max(1, Math.ceil((a1 - a0) / 6));
  let xlab = "";
  for (let a = a0; a <= a1; a += step) {
    xlab += `<text x="${X(a).toFixed(1)}" y="${h - 13}" text-anchor="${a === a0 ? "start" : "middle"}" class="c-ax">${a}</text>`;
  }
  xlab += `<text x="${w - padR}" y="${h - 2}" text-anchor="end" class="c-ax">anos de idade</text>`;
  return `<svg class="fc-chart" viewBox="0 0 ${w} ${h}" role="img"
    aria-label="Percentagem do preço mantida por idade, ${escapeHtml(av.spec.question)}">${ticks}
    <path d="${curve(av.a.r)}" fill="none" stroke="#177A47" stroke-width="2.4" stroke-linejoin="round"/>
    <path d="${curve(av.b.r)}" fill="none" stroke="#B4661E" stroke-width="2.4" stroke-dasharray="5 4" stroke-linejoin="round"/>
    ${xlab}
    <text x="${w - padR}" y="${padT}" text-anchor="end" class="c-ax" fill="#177A47">— ${escapeHtml(av.spec.a.lbl.toLowerCase())}</text>
    <text x="${w - padR}" y="${padT + 14}" text-anchor="end" class="c-ax" fill="#B4661E">- - ${escapeHtml(av.spec.b.lbl.toLowerCase())}</text></svg>`;
}

export function duelJson(rec, slug, av, { host, builtAt }) {
  const S = av.spec;
  const canonical = `https://${host}/${S.path}/${slug}`;
  const side = s => ({ sample_size: s.n, annual_depreciation_rate: s.r,
                       median_asking_eur: s.fm, median_mileage_km: s.km });
  return {
    url: canonical, slug, brand: rec.b, model: rec.m,
    compares: S.kind === "fuel" ? "fuel_type" : "transmission",
    measured: "asking_price",
    method: `log(price) ~ age + log(mileage) + side + age*side, OLS on active listings`,
    collected_until: (builtAt || "").slice(0, 10) || null,
    model_years: { from: av.y0, to: av.y1 },
    [S.a.json]: side(av.a),
    [S.b.json]: side(av.b),
    rate_difference_pp_per_year: Math.round(av.diff * 1000) / 10,
    rate_difference_ci95_half_width_pp: Math.round(av.ci * 1000) / 10,
    distinguishable_at_95: av.decisive,
    holds_value_better: av.decisive ? S[av.winner].json : null,
    premium_by_age: av.gap.map(([age, est, half]) =>
      ({ age, premium_of: S.a.json, premium: est, ci95_half_width: half })),
    fit_r2: av.r2,
    licence: licenseUrl(host),
  };
}

export function renderDuelPage({ rec, slug, av, stats, host, depositCount, builtAt, facetKeys: liveFacets = null }) {
  const S = av.spec;
  const B = escapeHtml(rec.b), M = escapeHtml(rec.m);
  const canonical = `https://${host}/${S.path}/${slug}`;
  const aPct = pctc(av.a.r), bPct = pctc(av.b.r);
  const win = S[av.winner], lose = S[av.winner === "a" ? "b" : "a"];
  const winSide = av[av.winner], loseSide = av[av.winner === "a" ? "b" : "a"];

  const verdict = av.decisive
    ? `<p class="fc-p"><b>Neste modelo, ${win.subj} segura melhor o preço.</b> Com a quilometragem igualada, ${S.a.subj} perde <b>${aPct}% por ano de idade</b> e ${S.b.subj} <b>${bPct}%</b> — uma diferença de <b>${ppc(av.diff)} pontos por ano</b> a favor ${withPrep("de", win.subj)} (intervalo de 95%: ${ppc(Math.abs(av.diff) - av.ci)} a ${ppc(Math.abs(av.diff) + av.ci)} pontos). Ao fim de cinco anos são ${keepPct(winSide.r, 5)}% do preço mantidos contra ${keepPct(loseSide.r, 5)}%.</p>`
    : `<p class="fc-p"><b>Neste modelo, ${S.kind === "fuel" ? "a escolha do combustível" : "a escolha da caixa"} não decide a desvalorização.</b> ${S.a.subj[0].toUpperCase()}${S.a.subj.slice(1)} perde ${aPct}% por ano de idade e ${S.b.subj} ${bPct}%, e a diferença de ${ppc(av.diff)} pontos cabe dentro da margem da própria medição (±${ppc(av.ci)} pontos). Não é "não sabemos": a amostra chega para dizer que, se existe vantagem, ela é menor do que ${ppc(av.ci + Math.abs(av.diff))} pontos por ano — pouco ao lado do que separa dois exemplares do mesmo ano.</p>`;

  const gapRows = av.gap.map(([age, est, half]) => {
    const lo = est - half, hi = est + half;
    const sure = lo > 0 || hi < 0;
    const read = !sure ? "indistinguível"
      : est > 0 ? `${escapeHtml(S.a.lbl)} pede mais` : `${escapeHtml(S.b.lbl)} pede mais`;
    return `<tr><td>${age} anos <span class="mut">(${av.ref - age})</span></td>
      <td>${est >= 0 ? "+" : "−"}${ppc(est)}%</td>
      <td class="mut">${(lo >= 0 ? "+" : "−")}${ppc(lo)}% a ${(hi >= 0 ? "+" : "−")}${ppc(hi)}%</td>
      <td class="mut">${read}</td></tr>`;
  }).join("");

  const drift = av.gap.length >= 2 ? (() => {
    const [g0, e0] = av.gap[0], [g1, e1] = av.gap[av.gap.length - 1];
    const move = e1 > e0 ? `vai encarecendo face ${withPrep("a", S.b.subj)}` : `vai ficando mais barat${S.a.subj.startsWith("a ") ? "a" : "o"} face ${withPrep("a", S.b.subj)}`;
    const winnerStartsCheaper = av.winner === "a" ? e0 < 0 : e0 > 0;
    const tail = winnerStartsCheaper
      ? "O que segura melhor o preço é também o mais barato à partida, por isso a vantagem soma-se: pagas menos hoje e perdes menos depois."
      : "O que segura melhor o preço é também o mais caro à partida — o prémio paga-se na compra e devolve-se em desvalorização, e quanto tempo ficas com o carro decide se compensa.";
    const say = (a, e) => `aos ${a} anos ${S.a.subj} pede ${e >= 0 ? "mais" : "menos"} ${ppc(e)}% do que ${S.b.subj}`;
    return `<p class="fc-p">As duas leituras são a mesma conta vista de dois lados: ${say(g0, e0)}, e ${say(g1, e1)}. Ou seja, com a idade ${S.a.subj} <b>${move}</b> — que é exactamente o que a diferença de ${ppc(av.diff)} pontos por ano diz, escrita em preço em vez de em taxa. ${av.decisive ? tail : ""}</p>`;
  })() : "";

  const gapBlock = av.gap.length ? `
    <section class="section fc-wrap">
      <h2 class="fc-h2">E hoje, qual pede mais?</h2>
      <p class="fc-p">A pergunta anterior era sobre o ritmo da queda; esta é sobre o preço no balcão. Cada linha compara ${S.a.subj} e ${S.b.subj} <b>da mesma idade e com os mesmos quilómetros</b> — a quilometragem entra no ajuste, por isso a diferença abaixo já não é ${S.kind === "fuel" ? 'o "diesel anda mais"' : 'o "automático é de outro segmento"'}.</p>
      <div class="fc-scroll"><table class="fc-tbl">
        <thead><tr><th>Idade</th><th>${escapeHtml(S.a.lbl)} vs. ${escapeHtml(S.b.lbl.toLowerCase())}</th><th>Intervalo (95%)</th><th>Leitura</th></tr></thead>
        <tbody>${gapRows}</tbody></table></div>
      ${drift}
      <p class="fc-prov mono">Valores do ajuste, não medianas em bruto: as medianas por idade misturam versões e quilometragens diferentes de cada lado.</p>
    </section>` : "";

  const tbl = `
    <div class="fc-scroll"><table class="fc-tbl">
      <thead><tr><th>&nbsp;</th><th>${escapeHtml(S.a.lbl)}</th><th>${escapeHtml(S.b.lbl)}</th></tr></thead>
      <tbody>
        <tr><td>Anúncios ativos no ajuste</td><td>${av.a.n}</td><td>${av.b.n}</td></tr>
        <tr><td>Preço pedido mediano</td><td>${fmtEur(av.a.fm)}</td><td>${fmtEur(av.b.fm)}</td></tr>
        <tr><td>Quilometragem mediana</td><td>${fmtKm(av.a.km)}</td><td>${fmtKm(av.b.km)}</td></tr>
        <tr><td>Perda por ano de idade</td><td>${aPct}%</td><td>${bPct}%</td></tr>
        <tr><td>Mantém ao fim de 5 anos</td><td>${keepPct(av.a.r, 5)}%</td><td>${keepPct(av.b.r, 5)}%</td></tr>
        <tr><td>Mantém ao fim de 10 anos</td><td>${keepPct(av.a.r, 10)}%</td><td>${keepPct(av.b.r, 10)}%</td></tr>
      </tbody></table></div>`;

  const body = crumbs([
    { name: "Início", href: "/" }, { name: S.crumb, href: `/${S.path}` },
    { name: `${rec.b} ${rec.m}` },
  ]) + `
    <div style="padding-top:14px;">
      <div class="side-card" style="max-width:680px;margin:0 auto;">
        <div class="eyebrow" style="margin-bottom:14px;"><span class="e-dot"></span><span class="mono">${S.eyebrow} · OLX PORTUGAL</span></div>
        <h1 class="fc-h1">${B} ${M}: ${S.question} segura melhor o preço?</h1>
        <p class="lede" style="font-size:16px;margin:0 0 18px;">Ajustámos as duas curvas em separado sobre <b>${av.n} anúncios ativos</b> de ${B} ${M} (${av.a.n} ${escapeHtml(S.a.low)}, ${av.b.n} ${escapeHtml(S.b.low)}, matrículas de ${av.y0} a ${av.y1}), com a quilometragem igualada. ${S.a.subj[0].toUpperCase()}${S.a.subj.slice(1)} perde <b>${aPct}% por ano de idade</b>, ${S.b.subj} <b>${bPct}%</b>.</p>
        <div class="fc-stat-row">
          <div class="fc-stat"><div class="k">${S.a.chip}</div><div class="v">${aPct}%</div><div class="s">por ano · ${av.a.n} anúncios</div></div>
          <div class="fc-stat"><div class="k">${S.b.chip}</div><div class="v">${bPct}%</div><div class="s">por ano · ${av.b.n} anúncios</div></div>
          <div class="fc-stat"><div class="k">DIFERENÇA</div><div class="v">${ppc(av.diff)} pp</div><div class="s">±${ppc(av.ci)} pp · ${av.decisive ? `a favor ${withPrep("de", win.subj)}` : "indistinguível"}</div></div>
        </div>
        ${provenance({ n: av.n, builtAt, measure: `Preço pedido de ${rec.b} ${rec.m}, ${S.a.low} e ${S.b.low} em separado (${av.y0}-${av.y1})`, extra: `Ajuste log-linear com quilometragem controlada, R²=${av.r2.toFixed(2)}` })}
      </div>
    </div>
    <section class="section fc-wrap">
      <h2 class="fc-h2">A resposta</h2>
      ${verdict}
      <p class="fc-p">A comparação directa das medianas não responde a isto: no ${B} ${M} ${S.a.subj} à venda tem ${fmtKm(av.a.km)} medianos e ${S.b.subj} ${fmtKm(av.b.km)}, e uma curva ajustada sem contar com isso mede a mistura de quilometragens e chama-lhe ${S.kind === "fuel" ? "combustível" : "caixa"}. Por isso o ajuste usa idade <b>e</b> quilometragem, e as duas curvas abaixo estão à mesma quilometragem.</p>
    </section>
    <section class="section fc-wrap">
      <h2 class="fc-h2">As duas curvas</h2>
      ${retentionChart(av)}
      <p class="fc-p" style="margin-top:10px;">Percentagem do preço mantida à medida que o carro envelhece, a partir do exemplar mais novo com amostra (${av.ref - av.a0}). São preços pedidos em anúncios ativos, não vendas fechadas: medem o que o mercado pede hoje por cada idade, não o que um dono concreto recebeu.</p>
    </section>
    ${gapBlock}
    <section class="section fc-wrap">
      <h2 class="fc-h2">Os dois lados, lado a lado</h2>
      ${tbl}
    </section>
    <section class="section fc-wide">
      <div class="cta-banner">
        <div style="flex:1 1 360px;">
          <h2>E o TEU ${B} ${M}?</h2>
          <p>Estas são as curvas do modelo. Cola o link do teu anúncio e dizemos o valor justo desse carro concreto — com a tua versão, os teus quilómetros e a tua caixa.</p>
        </div>
        <a class="btn-bright" href="/avaliar?modelo=${encodeURIComponent(slug)}">Avaliar o meu carro&nbsp;&nbsp;→</a>
      </div>
    </section>
    <section class="section fc-wrap" style="padding-bottom:70px;">
      <p class="fc-p"><a href="/preco/${slug}">Todos os ${B} ${M}</a>${[S.a, S.b].filter(side => !liveFacets || liveFacets.includes(side.facet)).map(side => ` · <a href="/preco/${slug}/${side.facet}">${side.only}</a>`).join("")} · <a href="/${S.path}">Outros modelos</a> · <a href="/metodologia">Como calculamos</a></p>
    </section>`;

  const faqs = [
    [`Num ${rec.b} ${rec.m}, ${S.a.low} desvaloriza mais do que ${S.b.low}?`,
     av.decisive
       ? `Não da forma que se costuma dizer: neste modelo é ${win.subj} que segura melhor o preço. Em ${av.n} anúncios ativos do OLX Portugal, com a quilometragem igualada, ${S.a.subj} perde ${aPct}% por ano de idade e ${S.b.subj} ${bPct}% — ${ppc(av.diff)} pontos de diferença por ano a favor ${withPrep("de", win.subj)}, contra ${lose.subj}.`
       : `Neste modelo a diferença não é distinguível: ${S.a.subj} perde ${aPct}% por ano de idade e ${S.b.subj} ${bPct}%, uma distância de ${ppc(av.diff)} pontos que cabe na margem da medição (±${ppc(av.ci)} pontos) sobre ${av.n} anúncios ativos do OLX Portugal.`],
    [`Um ${rec.b} ${rec.m} ${S.a.low} é mais caro do que ${S.b.low}?`,
     av.gap.length
       ? `Aos ${av.gap[0][0]} anos e com a mesma quilometragem, ${S.a.subj} pede ${av.gap[0][1] >= 0 ? "mais" : "menos"} ${ppc(av.gap[0][1])}% do que ${S.b.subj}. Em bruto, sem igualar quilómetros, a mediana pedida é ${fmtEur(av.a.fm)} (${fmtKm(av.a.km)} medianos) contra ${fmtEur(av.b.fm)} (${fmtKm(av.b.km)}).`
       : `A mediana pedida é ${fmtEur(av.a.fm)} (${fmtKm(av.a.km)} medianos) contra ${fmtEur(av.b.fm)} (${fmtKm(av.b.km)} medianos). São quilometragens muito diferentes, por isso a diferença de preço em bruto não é só ${S.kind === "fuel" ? "do combustível" : "da caixa"}.`],
  ];

  return layout({
    title: `${rec.b} ${rec.m}: ${S.question} segura melhor o preço?`,
    description: `Num ${rec.b} ${rec.m}, ${S.a.low} perde ${aPct}% por ano de idade e ${S.b.low} ${bPct}%, medido em ${av.n} anúncios ativos do OLX Portugal com a quilometragem igualada.${av.decisive ? ` Vantagem ${withPrep("de", win.subj)}: ${ppc(av.diff)} pontos por ano.` : " A diferença não é distinguível da margem da medição."}`,
    canonical, body, zone: "all", nav: "precos", depositCount, index: true, host,
    altJson: `${canonical}.json`,
    jsonLd: {
      "@context": "https://schema.org",
      "@graph": [
        {
          "@type": "Dataset", "license": licenseUrl(host), "url": canonical, "inLanguage": "pt-PT",
          "name": `Desvalorização de ${rec.b} ${rec.m} por ${S.kind === "fuel" ? "combustível" : "caixa"}`,
          "description": `Taxa de desvalorização anual de ${rec.b} ${rec.m} em ${S.a.low} (${aPct}%) e em ${S.b.low} (${bPct}%), ajustada à quilometragem, sobre ${av.n} anúncios ativos do OLX Portugal entre ${av.y0} e ${av.y1}.`,
          "creator": { "@type": "Organization", "name": "Carsbuyer", "url": `https://${host}/` },
          "isAccessibleForFree": true, "dateModified": builtAt || undefined,
          "temporalCoverage": `${av.y0}/${av.y1}`,
          "variableMeasured": ["Desvalorização anual (%)", "Preço pedido (EUR)", "Quilometragem (km)"],
          "distribution": [{ "@type": "DataDownload", "encodingFormat": "application/json", "contentUrl": `${canonical}.json` }],
        },
        breadcrumbLd(host, [
          { name: "Início", href: "/" }, { name: S.crumb, href: `/${S.path}` },
          { name: `${rec.b} ${rec.m}` },
        ]),
        faqLd(faqs),
      ],
    },
  });
}

export function renderDuelHub({ spec, rows, other, stats, host, depositCount, builtAt }) {
  const S = spec;
  const canonical = `https://${host}/${S.path}`;
  const aWins = rows.filter(r => r.av.decisive && r.av.winner === "a").length;
  const bWins = rows.filter(r => r.av.decisive && r.av.winner === "b").length;
  const draws = rows.length - aWins - bWins;

  const tr = rows.map(r => `<tr>
      <td><a href="/${S.path}/${r.slug}" style="color:#177A47;font-weight:600;">${escapeHtml(r.b)} ${escapeHtml(r.m)}</a></td>
      <td>${pctc(r.av.a.r)}%</td>
      <td>${pctc(r.av.b.r)}%</td>
      <td class="mut">${ppc(r.av.diff)} pp ±${ppc(r.av.ci)}</td>
      <td>${r.av.decisive ? escapeHtml(S[r.av.winner].lbl) : "<span class=\"mut\">Empate</span>"}</td>
      <td class="mut">${r.av.a.n} / ${r.av.b.n}</td></tr>`).join("");

  const body = crumbs([{ name: "Início", href: "/" }, { name: S.crumb }]) + `
    <div style="padding-top:14px;">
      <div class="side-card" style="max-width:680px;margin:0 auto;">
        <div class="eyebrow" style="margin-bottom:14px;"><span class="e-dot"></span><span class="mono">${S.eyebrow} · ${rows.length} MODELOS</span></div>
        <h1 class="fc-h1">${S.hubH1}</h1>
        <p class="lede" style="font-size:16px;margin:0 0 18px;">A resposta não é uma para todos os carros — é uma por modelo. Nos <b>${rows.length} modelos</b> com anúncios ativos suficientes para ajustar as duas curvas em separado, ${S.a.subj} segura melhor o preço em <b>${aWins}</b>, ${S.b.subj} em <b>${bWins}</b>, e em <b>${draws}</b> a diferença não se distingue da margem da medição.</p>
        ${provenance({ n: rows.reduce((a, r) => a + r.av.n, 0), builtAt, measure: `Preço pedido por idade, ${S.a.low} e ${S.b.low} em separado, com a quilometragem controlada` })}
      </div>
    </div>
    <section class="section fc-wrap">
      <h2 class="fc-h2">Modelo a modelo</h2>
      <p class="fc-p">Ordenado pela distância entre as duas curvas. A coluna do meio é o que a página do modelo desenvolve: quantos pontos percentuais por ano separam os dois lados, e com que margem foram medidos.</p>
      <div class="fc-scroll"><table class="fc-tbl">
        <thead><tr><th>Modelo</th><th>${escapeHtml(S.a.lbl)} /ano</th><th>${escapeHtml(S.b.lbl)} /ano</th><th>Diferença</th><th>Segura melhor</th><th>Anúncios ${escapeHtml(S.a.lbl[0])}/${escapeHtml(S.b.lbl[0])}</th></tr></thead>
        <tbody>${tr}</tbody></table></div>
    </section>
    <section class="section fc-wrap">
      <h2 class="fc-h2">Porque é que "empate" também é uma resposta</h2>
      <p class="fc-p">Uma diferença de meio ponto por ano entre duas curvas ajustadas em algumas dezenas de anúncios não é um resultado, é ruído com três casas decimais. Por isso cada linha traz a sua margem, e um modelo só entra nesta tabela quando essa margem é estreita o suficiente para que "empate" signifique <b>não há vantagem apreciável</b> e não <b>não conseguimos ver</b>. Os modelos onde a amostra não chega para essa distinção simplesmente não têm página aqui.</p>
      <p class="fc-p">A quilometragem entra no ajuste em todos eles. Sem isso, a tabela mediria sobretudo o facto de ${S.kind === "fuel" ? "os diesels à venda andarem muito mais" : "os automáticos à venda serem mais recentes e andarem muito menos"} — e chamaria a isso ${S.kind === "fuel" ? "combustível" : "caixa"}.</p>
    </section>
    <section class="section fc-wide">
      <div class="cta-banner">
        <div style="flex:1 1 360px;">
          <h2>Estás a escolher entre dois carros concretos?</h2>
          <p>Cola o link de cada anúncio e dizemos o valor justo de cada um — com a motorização, os quilómetros e a versão de cada exemplar.</p>
        </div>
        <a class="btn-bright" href="/avaliar">Avaliar um anúncio&nbsp;&nbsp;→</a>
      </div>
    </section>
    <section class="section fc-wrap" style="padding-bottom:70px;">
      <p class="fc-p">${other ? `<a href="/${other.path}">${other.crumb}</a> · ` : ""}<a href="/depreciacao">Desvalorização por modelo</a> · <a href="/precos">Preços por modelo</a> · <a href="/metodologia">Como calculamos</a></p>
    </section>`;

  return layout({
    title: S.hubTitle,
    description: `Em ${rows.length} modelos com amostra para separar as duas curvas, ${S.a.low} segura melhor o preço em ${aWins} e ${S.b.low} em ${bWins}. Taxas por ano de idade medidas em anúncios ativos do OLX Portugal, com a quilometragem controlada.`,
    canonical, body, zone: "all", nav: "precos", depositCount, index: true, host,
    jsonLd: {
      "@context": "https://schema.org",
      "@graph": [
        {
          "@type": "Dataset", "license": licenseUrl(host), "url": canonical, "inLanguage": "pt-PT",
          "name": `Desvalorização por ${S.kind === "fuel" ? "combustível" : "caixa"}, modelo a modelo (Portugal)`,
          "description": `Taxa de desvalorização anual em ${S.a.low} e em ${S.b.low} para ${rows.length} modelos, ajustada à quilometragem, a partir de anúncios ativos do OLX Portugal.`,
          "creator": { "@type": "Organization", "name": "Carsbuyer", "url": `https://${host}/` },
          "isAccessibleForFree": true, "dateModified": builtAt || undefined,
          "variableMeasured": ["Desvalorização anual (%)", S.kind === "fuel" ? "Combustível" : "Caixa"],
        },
        breadcrumbLd(host, [{ name: "Início", href: "/" }, { name: S.crumb }]),
      ],
    },
  });
}

let VENDER_WAVE = 0;
export function setVenderWave(n) {
  const v = parseInt(n, 10);
  VENDER_WAVE = Number.isFinite(v) && v > 0 ? v : 0;
}

export function venderOk(rec) {
  return !!(rec && rec.fm > 0 && rec.n >= 20 && ((rec.sd != null && rec.sn >= 8) || liquidityOk(rec)));
}

let _venderKey = null, _venderVal = null;
export function venderWaveSlugs(models, builtAt) {
  if (!VENDER_WAVE) return null;
  const key = `${builtAt || ""}:${Object.keys(models).length}:${VENDER_WAVE}`;
  if (_venderKey === key && _venderVal) return _venderVal;
  _venderVal = new Set(Object.entries(models)
    .filter(([, r]) => venderOk(r))
    .sort((a, b) => (b[1].n || 0) - (a[1].n || 0) || (a[0] < b[0] ? -1 : 1))
    .slice(0, VENDER_WAVE)
    .map(([slug]) => slug));
  _venderKey = key;
  return _venderVal;
}

export function publishedVender(models, slug, rec, builtAt) {
  const wave = venderWaveSlugs(models, builtAt);
  if (wave && !wave.has(slug)) return false;
  return venderOk(rec);
}

function venderFacts(rec, market) {
  const lq = liquidityOk(rec) ? rec.lq : null;
  const mkt = market || {};
  const pick = k => (lq && lq[k] != null) ? lq[k] : null;
  return {
    lq,
    days: (lq && lq.md != null) ? lq.md : (rec.sd != null ? rec.sd : null),
    s30: pick("s30"), s60: pick("s60"), s90: pick("s90"),
    cu: pick("cu"), cp: pick("cp"), cd: pick("cd"), hd: pick("hd"),
    mktS30: mkt.s30 != null ? mkt.s30 : null,
    mktCu: mkt.cu != null ? mkt.cu : null,
    mktCp: mkt.cp != null ? mkt.cp : null,
  };
}

export function venderJson(rec, slug, { host, builtAt } = {}) {
  const f = venderFacts(rec, null);
  return {
    source: "Carsbuyer",
    licence: "Citação permitida com atribuição a Carsbuyer e indicação da data.",
    url: host ? `https://${host}/vender/${slug}` : undefined,
    built_at: builtAt || undefined,
    brand: rec.b, model: rec.m, listings: rec.n,
    asking: { median: rec.fm, p25: rec.fl, p75: rec.fh },
    days_to_sell_median: f.days,
    sold_within: { d30: f.s30, d60: f.s60, d90: f.s90 },
    price_cut: { share: f.cu, median_pct: f.cp },
    years: yearCells(rec, 1).map(c => ({ year: c.y, median: c.fm, p25: c.fl, p75: c.fh, listings: c.n })),
  };
}

const VENDER_CHECKLIST = [
  ["Fotografias à luz do dia, carro lavado, os quatro cantos e o interior", "os anúncios sem fotos do interior são os primeiros a ser ignorados."],
  ["Quilómetros, ano, combustível e caixa no título", "é o que o comprador filtra antes de abrir o anúncio."],
  ["Inspeção em dia e livro de revisões à mão", "um comprador que vê papéis negoceia menos."],
  ["Diz o que tem de errado", "quem esconde um risco perde o comprador na inspeção, quem o diz fecha o negócio com desconto menor."],
  ["Responde no próprio dia", "o comprador que pergunta hoje compra amanhã, a outro."],
];

export function renderVenderPage({ rec, slug, market, pageYears = [], hasLiquidity = false, hasDepreciation = false,
                                   host, depositCount, builtAt }) {
  const B = escapeHtml(rec.b), M = escapeHtml(rec.m);
  const FM = fmtEur(rec.fm), FL = fmtEur(rec.fl), FH = fmtEur(rec.fh);
  const canonical = `https://${host}/vender/${slug}`;
  const f = venderFacts(rec, market);
  const years = yearCells(rec, 1);

  const yearRows = years.map(c => {
    const y = pageYears.includes(c.y) ? `<a href="/preco/${slug}/${c.y}">${c.y}</a>` : String(c.y);
    return `<tr><td>${y}</td><td><b>${fmtEur(c.fm)}</b></td><td class="mut">${fmtEur(c.fl)} – ${fmtEur(c.fh)}</td><td class="mut">${c.n}</td></tr>`;
  }).join("");

  const vsMkt = (f.s30 != null && f.mktS30 != null)
    ? (f.s30 >= f.mktS30 * 1.12 ? "acima" : f.s30 <= f.mktS30 * 0.88 ? "abaixo" : "media")
    : null;
  const speedRows = [["30", f.s30], ["60", f.s60], ["90", f.s90]].filter(([, v]) => v != null)
    .map(([d, v]) => `<tr><td>${d} dias</td><td><b>${liqPct(v)} em cada 100</b></td><td class="mut">${liqPct(1 - v)} ainda à venda</td></tr>`).join("");
  const speedLead = f.days != null
    ? `Metade dos ${B} ${M} que saem do OLX sai em <b>${f.days} dias</b>.`
    : "";
  const speedMkt = f.s30 != null
    ? ` No primeiro mês saem <b>${liqPct(f.s30)} em cada 100</b>${
        vsMkt === "acima" ? `, mais do que a média do mercado (${liqPct(f.mktS30)})`
        : vsMkt === "abaixo" ? `, menos do que a média do mercado (${liqPct(f.mktS30)})`
        : vsMkt === "media" ? `, o ritmo médio do mercado` : ""}.`
    : "";
  const speedAdvice = vsMkt === "acima"
    ? `Tens pouca razão para começar abaixo da mediana: um ${B} ${M} ao preço certo vende no primeiro ciclo do anúncio.`
    : vsMkt === "abaixo"
      ? `Conta com um segundo ciclo de 30 dias e com ter de ceder. Começar acima de ${FH} é pedir para ficar parado.`
      : `O preço a que o pões decide em que metade ficas.`;

  const cutBlock = f.cu != null ? `
      <h2 class="fc-h2">Quanto costumam baixar o preço</h2>
      <p class="fc-p"><b>${liqPct(f.cu)} em cada 100</b> anúncios de ${B} ${M} baixaram o preço antes de sair${f.mktCu != null ? ` (no mercado, ${liqPct(f.mktCu)})` : ""}.${f.cp != null ? ` Quando baixam, a descida mediana é de <b>${liqPct(f.cp)}%</b>${f.mktCp != null ? ` (mercado: ${liqPct(f.mktCp)}%)` : ""} — é a margem que o comprador costuma conseguir, e por isso é a folga que faz sentido deixar entre o preço que pedes e o que aceitas.` : ""}</p>
      ${(f.cd != null && f.hd != null) ? `<p class="fc-p">Os que baixaram estiveram <b>${f.cd} dias</b> no ar; os que aguentaram o preço, <b>${f.hd}</b>. Não é a descida que atrasa a venda: baixa-se porque o carro não está a sair.</p>` : ""}` : "";

  const dt = (f.lq && Array.isArray(f.lq.dt)) ? f.lq.dt.filter(d => d.s30 != null).slice().sort((a, b) => b.s30 - a.s30) : [];
  const dtBlock = dt.length >= 3 ? `
      <h2 class="fc-h2">Onde vende mais depressa</h2>
      <p class="fc-p">Entre os distritos com amostra, um ${B} ${M} sai mais depressa em <b>${escapeHtml(dt[0].lbl)}</b> (${liqPct(dt[0].s30)} em cada 100 no primeiro mês${dt[0].md != null ? `, mediana ${dt[0].md} dias` : ""}) e mais devagar em <b>${escapeHtml(dt[dt.length - 1].lbl)}</b> (${liqPct(dt[dt.length - 1].s30)}${dt[dt.length - 1].md != null ? `, ${dt[dt.length - 1].md} dias` : ""}). Um anúncio vê-se em todo o país; o comprador que se desloca é o que já decidiu.</p>` : "";

  const pb = (f.lq && Array.isArray(f.lq.pb)) ? f.lq.pb.filter(p => p.s30 != null) : [];
  const pbBlock = pb.length >= 2 ? `
      <h2 class="fc-h2">O preço muda o comprador</h2>
      <div class="fc-scroll"><table class="fc-tbl">
        <thead><tr><th>Faixa de preço</th><th>Sai em 30 dias</th><th>Mediana</th><th>Anúncios</th></tr></thead>
        <tbody>${pb.map(p => `<tr><td>${escapeHtml(p.lbl)}</td><td><b>${liqPct(p.s30)} em cada 100</b></td><td class="mut">${p.md != null ? `${p.md} dias` : "—"}</td><td class="mut">${fmtNum(p.n)}</td></tr>`).join("")}</tbody>
      </table></div>` : "";

  const check = VENDER_CHECKLIST.map(([t, d]) => `<li class="fc-li"><b>${t}</b> — ${d}</li>`).join("");

  const body = crumbs([{ name: "Início", href: "/" }, { name: "Vender", href: "/vender" }, { name: `${rec.b} ${rec.m}` }]) + `
    <section class="section fc-wrap" style="padding-top:16px;">
      <h1 class="fc-h1">Vender um ${B} ${M}: quanto pedir e em quantos dias vende</h1>
      <p class="fc-p">Nos <b>${rec.n} anúncios ativos</b> de ${B} ${M} no OLX, metade pede entre <b>${FL}</b> e <b>${FH}</b>, com mediana de <b>${FM}</b>. ${speedLead}${speedMkt} Estes são os números contra os quais o teu anúncio vai ser lido.</p>
      ${provenance({ n: rec.n, builtAt, measure: `Preço pedido, ${B} ${M} (mediana e P25-P75); dias até sair do OLX` })}

      <h2 class="fc-h2">Quanto pedir</h2>
      <p class="fc-p">Acima de ${FH} ficas na quarta parte mais cara dos anúncios e competes com carros mais novos ou com menos quilómetros. Abaixo de ${FL} estás na quarta parte mais barata, onde o comprador desconfia antes de perguntar. O ponto de partida mais comum é a mediana do teu ano:</p>
      <div class="fc-scroll"><table class="fc-tbl">
        <thead><tr><th>Ano</th><th>Mediana pedida</th><th>Metade pede entre</th><th>Anúncios</th></tr></thead>
        <tbody>${yearRows}</tbody>
      </table></div>
      <p class="fc-p">Preços <b>pedidos</b> em anúncios ativos, não preços de venda fechados: quem vende cede em média o que está na secção seguinte. Para o teu carro concreto, com os teus quilómetros, usa a <a href="/avaliar?modelo=${encodeURIComponent(slug)}">avaliação por modelo e ano</a>.</p>

      <h2 class="fc-h2">Em quantos dias vende</h2>
      <p class="fc-p">${speedLead}${speedMkt} ${speedAdvice}</p>
      ${speedRows ? `<div class="fc-scroll"><table class="fc-tbl"><thead><tr><th>Ao fim de</th><th>Já saíram</th><th>Ainda à venda</th></tr></thead><tbody>${speedRows}</tbody></table></div>` : ""}
      <p class="fc-p mut" style="font-size:13.5px;">Um anúncio do OLX corre em ciclos de 30 dias; contamos como saída o último ciclo em que o vimos no ar, e a conta inclui os que ainda estão à venda, que é o que a impede de ficar curta.${hasLiquidity ? ` Detalhe por preço, idade e distrito: <a href="/liquidez/${slug}">tempo de venda do ${B} ${M}</a>.` : ""}</p>
      ${cutBlock}
      ${dtBlock}
      ${pbBlock}

      <h2 class="fc-h2">O anúncio que vende</h2>
      <ul class="fc-ul">${check}</ul>
    </section>
    <section class="section" style="padding:0 22px;max-width:680px;margin:0 auto;">
      ${leadFormBlock({ slug, name: `${rec.b} ${rec.m}`, year: null, median: rec.fm })}
    </section>
    <section class="section fc-wrap" style="padding-bottom:70px;">
      <p class="fc-p"><a href="/preco/${slug}">Preços de ${B} ${M} por ano</a>${hasDepreciation ? ` · <a href="/depreciacao/${slug}">Desvalorização</a>` : ""}${hasLiquidity ? ` · <a href="/liquidez/${slug}">Tempo de venda</a>` : ""} · <a href="/vender">Outros modelos</a> · <a href="/guias">Guias para vender</a> · <a href="/metodologia">Como medimos</a> · <a href="${canonical}.json">Dados em JSON</a></p>
    </section>`;

  const faqs = [
    [`Quanto pedir por um ${rec.b} ${rec.m} usado?`,
     `A mediana pedida nos ${rec.n} anúncios ativos do OLX é ${FM}; metade dos anúncios pede entre ${FL} e ${FH}. O ano concreto muda o número: a tabela desta página tem a mediana de cada ano.`],
    ...(f.days != null ? [[`Em quantos dias se vende um ${rec.b} ${rec.m}?`,
     `Metade dos ${rec.b} ${rec.m} que saem do OLX sai em ${f.days} dias${f.s30 != null ? `; no primeiro mês saem ${liqPct(f.s30)} em cada 100` : ""}. Medido em anúncios reais acompanhados até saírem.`]] : []),
    ...(f.cu != null ? [[`Vale a pena baixar o preço de um ${rec.b} ${rec.m}?`,
     `${liqPct(f.cu)} em cada 100 anúncios deste modelo baixaram o preço antes de sair${f.cp != null ? `, em mediana ${liqPct(f.cp)}%` : ""}. Baixa-se porque o carro não está a sair; começar perto da mediana do ano evita a descida.`]] : []),
  ];

  return layout({
    title: `Vender ${rec.b} ${rec.m}: quanto pedir (${FM}) e em quantos dias vende`,
    description: `${rec.b} ${rec.m} usado: mediana pedida ${FM} (${FL}–${FH}) em ${rec.n} anúncios${f.days != null ? `, vende em ~${f.days} dias` : ""}${f.cu != null ? `, ${liqPct(f.cu)}% baixam o preço` : ""}. Quanto pedir por ano e propostas de compra sem compromisso.`,
    canonical, body, zone: "all", nav: "avaliar", depositCount, index: true, host,
    altJson: `${canonical}.json`,
    jsonLd: {
      "@context": "https://schema.org",
      "@graph": [
        {
          "@type": "Dataset", "license": licenseUrl(host), "url": canonical, "inLanguage": "pt-PT",
          "name": `Vender ${rec.b} ${rec.m}: preço pedido e tempo de venda em Portugal`,
          "description": `Mediana e intervalo do preço pedido por ano, dias até sair do OLX e frequência de descidas de preço para ${rec.b} ${rec.m}.`,
          "creator": { "@type": "Organization", "name": "Carsbuyer", "url": `https://${host}/` },
          "isAccessibleForFree": true, "dateModified": builtAt || undefined,
          "variableMeasured": ["Preço pedido (EUR)", "Dias até sair do OLX", "Anúncios com descida de preço (%)"],
          "distribution": [{ "@type": "DataDownload", "encodingFormat": "application/json", "contentUrl": `${canonical}.json` }],
        },
        breadcrumbLd(host, [{ name: "Início", href: "/" }, { name: "Vender", href: "/vender" }, { name: `${rec.b} ${rec.m}` }]),
        faqLd(faqs),
      ],
    },
  });
}

export function renderVenderHub({ rows, market, host, depositCount, builtAt }) {
  const canonical = `https://${host}/vender`;
  const mkt = market || {};
  const tr = rows.map(r => `<tr>
      <td><a href="/vender/${r.slug}" style="color:#177A47;font-weight:600;">${escapeHtml(r.b)} ${escapeHtml(r.m)}</a></td>
      <td><b>${fmtEur(r.fm)}</b></td>
      <td class="mut">${fmtEur(r.fl)} – ${fmtEur(r.fh)}</td>
      <td class="mut">${r.s30 != null ? `${liqPct(r.s30)} em cada 100` : (r.sd != null ? `~${r.sd} dias` : "—")}</td>
      <td class="mut">${r.cu != null ? `${liqPct(r.cu)}%${r.cp != null ? ` · −${liqPct(r.cp)}%` : ""}` : "—"}</td>
      <td class="mut">${fmtNum(r.n)}</td></tr>`).join("");
  const body = crumbs([{ name: "Início", href: "/" }, { name: "Vender" }]) + `
    <section class="section fc-wrap" style="padding-top:16px;">
      <h1 class="fc-h1">Vender carro usado em Portugal: quanto pedir por modelo</h1>
      <p class="fc-p">Para cada modelo com amostra suficiente no OLX: o que os outros vendedores estão a pedir, em quantos dias os anúncios saem e quantos acabam por baixar o preço. É a referência contra a qual o teu anúncio vai ser comparado — e a que usas para ler uma proposta de compra.</p>
      ${mkt.s30 != null ? `<p class="fc-p">No conjunto do mercado saem <b>${liqPct(mkt.s30)} em cada 100</b> anúncios no primeiro mês${mkt.md != null ? `, com mediana de <b>${mkt.md} dias</b>` : ""}${mkt.cu != null ? `, e <b>${liqPct(mkt.cu)} em cada 100</b> baixam o preço antes de sair${mkt.cp != null ? ` (em mediana ${liqPct(mkt.cp)}%)` : ""}` : ""}.</p>` : ""}
      <div class="fc-scroll"><table class="fc-tbl">
        <thead><tr><th>Modelo</th><th>Mediana pedida</th><th>Metade pede entre</th><th>Sai em 30 dias</th><th>Baixam o preço</th><th>Anúncios</th></tr></thead>
        <tbody>${tr}</tbody></table></div>
      <p class="fc-p" style="margin-top:18px;">O teu modelo não está na lista? <a href="/avaliar#escolher">Escolhe-o na avaliação por modelo e ano</a>: mostra a mediana e deixa-te pedir propostas de compra.</p>
      ${provenance({ n: rows.reduce((s, r) => s + (r.n || 0), 0), builtAt, measure: "Preço pedido em anúncios ativos (mediana e P25-P75); dias até sair do OLX" })}
      <p class="fc-p" style="margin-top:18px;"><a href="/precos">Preços por modelo</a> · <a href="/liquidez">Tempo de venda</a> · <a href="/depreciacao">Desvalorização</a> · <a href="/metodologia">Como medimos</a></p>
    </section>
    <div style="height:60px;"></div>`;
  return layout({
    title: "Vender carro usado: quanto pedir e em quantos dias vende, por modelo",
    description: `Quanto pedir por um carro usado em Portugal, modelo a modelo: mediana pedida no OLX, em quantos dias os anúncios saem e quantos baixam o preço. ${rows.length} modelos.`,
    canonical, body, zone: "all", nav: "avaliar", depositCount, index: true, host,
    jsonLd: {
      "@context": "https://schema.org",
      "@graph": [
        {
          "@type": "Dataset", "license": licenseUrl(host), "url": canonical, "inLanguage": "pt-PT",
          "name": "Quanto pedir por um carro usado, por modelo (Portugal)",
          "description": "Mediana do preço pedido, dias até sair e frequência de descidas de preço por modelo, no OLX Portugal.",
          "creator": { "@type": "Organization", "name": "Carsbuyer", "url": `https://${host}/` },
          "isAccessibleForFree": true, "dateModified": builtAt || undefined,
        },
        breadcrumbLd(host, [{ name: "Início", href: "/" }, { name: "Vender" }]),
      ],
    },
  });
}
