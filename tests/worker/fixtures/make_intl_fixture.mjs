import { readFileSync, writeFileSync } from "node:fs";

const SRC = new URL("./models.json", import.meta.url);
const OUT = new URL("./models_de.json", import.meta.url);

const BUNDESLAENDER = [
  ["baden-wuerttemberg", "Baden-Württemberg"],
  ["bayern", "Bayern"],
  ["berlin", "Berlin"],
  ["brandenburg", "Brandenburg"],
  ["bremen", "Bremen"],
  ["hamburg", "Hamburg"],
  ["hessen", "Hessen"],
  ["mecklenburg-vorpommern", "Mecklenburg-Vorpommern"],
  ["niedersachsen", "Niedersachsen"],
  ["nordrhein-westfalen", "Nordrhein-Westfalen"],
  ["rheinland-pfalz", "Rheinland-Pfalz"],
  ["saarland", "Saarland"],
  ["sachsen", "Sachsen"],
  ["sachsen-anhalt", "Sachsen-Anhalt"],
  ["schleswig-holstein", "Schleswig-Holstein"],
  ["thueringen", "Thüringen"],
];

const src = JSON.parse(readFileSync(SRC, "utf8"));
const ptKeys = Object.keys(src.districts || {}).sort();
const map = new Map();
ptKeys.forEach((k, i) => map.set(k, BUNDESLAENDER[i % BUNDESLAENDER.length]));

const region = k => map.get(k) || [k, k];

const out = {
  v: src.v,
  built_at: src.built_at,
  models: {},
  lqm: src.lqm,
  mq: src.mq,
  districts: {},
};

for (const [slug, rec] of Object.entries(src.models)) {
  const copy = JSON.parse(JSON.stringify(rec));
  if (Array.isArray(copy.dt)) {
    copy.dt = copy.dt.map(cell => {
      const [k, lbl] = region(cell.k);
      const next = Object.assign({}, cell, { k, lbl });
      if (next.vs) {
        const vs = {};
        for (const [other, val] of Object.entries(next.vs)) vs[region(other)[0]] = val;
        next.vs = vs;
      }
      return next;
    });
  }
  out.models[slug] = copy;
}

for (const [k, d] of Object.entries(src.districts || {})) {
  const [nk, lbl] = region(k);
  out.districts[nk] = Object.assign({}, d, { lbl });
}

writeFileSync(OUT, JSON.stringify(out, null, 0) + "\n");
console.log(`models_de.json: ${Object.keys(out.models).length} models, `
  + `${Object.keys(out.districts).length} regions, built_at ${out.built_at}`);
