import { escapeHtml, layout } from "./templates.js";
import {
  crumbs, breadcrumbLd, faqLd, facetCell, facetKind, publishedCells, districtRanking,
} from "./seo-pages.js";
import { intlProvenance, intlInWave } from "./pages-intl.js";
import {
  t, href, labelL, fmtEurL, fmtKmL, fmtNumL, fmtPctL, monthTagL,
  registerStrings, registerRoutes, registerNav,
} from "./i18n.js";
import { registerIntlPages } from "./index.js";

const CC_LICENCE = "https://creativecommons.org/licenses/by/4.0/";
const AGE_GAP_MIN = 0.05;
const REGION_TOP_MODELS = 20;
const RANK_MIN_ROWS = 3;

const ROUTES = {
  de: { regions: "regionen", regionsJson: "regionen.json" },
  fr: { regions: "regions", regionsJson: "regions.json" },
  it: { regions: "regioni", regionsJson: "regioni.json" },
};

const REGION_PREPOSITION = {
  de: { saarland: "im" },
  fr: {
    "hauts-de-france": "dans les", "pays-de-la-loire": "dans les",
    "grand-est": "dans le", "centre-val-de-loire": "dans le",
  },
  it: { lazio: "nel", molise: "nel", marche: "nelle", "valle-d-aosta": "nella" },
  pt: { porto: "no" },
};

const REGION_PREPOSITION_DEFAULT = { de: "in", fr: "en", it: "in", pt: "em" };

const STRINGS = {
  de: {
    "footer.regions": "Preise nach Bundesland",
    "cut.dearer": "teurer",
    "cut.cheaper": "günstiger",
    "facet.eyebrow": "{label} · {source}",
    "facet.phrase_fuel": "{brand} {model} {label}",
    "facet.phrase_gear": "{brand} {model} mit {label}",
    "facet.phrase_region": "{brand} {model} {region}",
    "facet.h1": "Was kostet ein gebrauchter {phrase}?",
    "facet.title": "{phrase} gebraucht: Preis {price}{month} ({n} Angebote)",
    "facet.title_month": " im {month}",
    "facet.desc": "{phrase}: Medianpreis {price} ({lo} bis {hi}) aus {n} aktiven Angeboten auf {source}.{age}",
    "facet.lede": "In {n} aktiven Angeboten verlangt ein {phrase} im Median {price}{km}{years}.{age}",
    "facet.lede_km": ", bei {km} Kilometerstand im Median",
    "facet.lede_years": ", Erstzulassung {from} bis {to}",
    "facet.cap_median": "Verlangter Preis, Median",
    "facet.cap_sample": "{n} Angebote",
    "facet.cap_share": "{n} Angebote · {share} des Modells",
    "facet.cap_km": "Kilometerstand, Median",
    "facet.range_label": "typische Spanne (die mittlere Hälfte der Angebote)",
    "facet.age_more": "Altersbereinigt ist dieser Preis {pct} teurer {ref}.",
    "facet.age_less": "Altersbereinigt ist dieser Preis {pct} günstiger {ref}.",
    "facet.age_same": "Altersbereinigt bleibt kein Unterschied übrig: der Abstand zwischen den beiden rohen Medianen ist reiner Altersunterschied.",
    "facet.ref_model": "als das ganze Modell",
    "facet.ref_model_country": "als das Modell im ganzen Land",
    "facet.ref_rest_cuts": "als die übrigen Varianten dieses Modells",
    "facet.ref_rest_country": "als dasselbe Modell außerhalb dieser Region",
    "facet.method_matched": "Jahr für Jahr verglichen, über {years} Zulassungsjahre mit Stichprobe auf beiden Seiten",
    "facet.method_normalized": "jedes der {used} Angebote geteilt durch den Median des übrigen Modells in seinem eigenen Zulassungsjahr",
    "facet.raw_caveat": " Die rohen Mediane ({a} gegen {b}) stehen weiter auseinander, weil jede Seite ihre eigene Altersmischung hat.",
    "facet.h2_fuel": "Gegen die anderen Motorisierungen",
    "facet.h2_gear": "Gegen die andere Getriebeart",
    "facet.h2_region": "Gegen den Rest des Landes",
    "facet.sib_label_fuel": "ANDERE MOTORISIERUNGEN",
    "facet.sib_label_gear": "ANDERE GETRIEBEARTEN",
    "facet.sib_label_region": "ANDERE REGIONEN",
    "facet.vs_model_matched": "Gegenüber allen {brand} {model} im Land (Median {all}) wird hier {verdict} verlangt – {method}.{caveat}",
    "facet.vs_model_raw": "Gegenüber allen {brand} {model} im Land werden hier im Median {mine} verlangt statt {all}{years}. Die beiden Zahlen lassen sich nicht voneinander abziehen: sie beschreiben unterschiedliche Altersmischungen.",
    "facet.sib_matched": "Gegen {link} ({n} Angebote, Median {fm}): <b>{pct} {dir}</b>, {method}. Die rohen Mediane ({mine} gegen {fm}) sagen etwas anderes, weil jede Seite ihre eigene Altersmischung hat.{km}",
    "facet.sib_raw": "Gegen {link}: Median {fm} in {n} Angeboten{oyears}, hier {mine}{myears}. Es gibt nicht genug Zulassungsjahre mit Stichprobe auf beiden Seiten, um bei gleichem Alter zu vergleichen – der Abstand zwischen den beiden Medianen enthält noch den Altersunterschied.{km}",
    "facet.verdict_more": "{pct} mehr",
    "facet.verdict_less": "{pct} weniger",
    "facet.verdict_same": "genauso viel",
    "facet.km_more": " Der mediane Kilometerstand liegt hier {km} höher.",
    "facet.km_less": " Der mediane Kilometerstand liegt hier {km} niedriger.",
    "facet.cta_h": "Du hast einen {phrase}?",
    "facet.cta_p": "Oben steht der Median dieser Auswahl. Für den Wert deines konkreten Autos gib Modell und Erstzulassung ein.",
    "facet.cta_btn": "Mein Auto bewerten",
    "facet.links": "Alle {brand} {model}",
    "facet.links_region": "Gebrauchtwagen {region}",
    "facet.links_hub": "Alle Modelle",
    "facet.measure": "Verlangter Preis, {phrase} (Median und P25 bis P75)",
    "facet.faq_q1": "Was kostet ein gebrauchter {phrase}?",
    "facet.faq_a1": "In {n} aktiven Angeboten auf {source} verlangt ein {phrase} im Median {price}, die mittlere Hälfte der Angebote liegt zwischen {lo} und {hi}. Das sind verlangte Preise, keine erzielten Verkaufspreise.",
    "facet.faq_q_fuel": "{brand} {model}: {a} oder {b}?",
    "facet.faq_q_gear": "{brand} {model}: {a} oder {b}?",
    "facet.faq_q_region": "Ist ein {brand} {model} {a} teurer als {b}?",
    "facet.faq_a_matched": "Jahr für Jahr verglichen – nur Zulassungsjahre mit Stichprobe auf beiden Seiten, {years} insgesamt – ist {a} {pct} {dir} als {b}. Die rohen Mediane ({mine} und {other}) lassen sich nicht direkt vergleichen, weil jede Seite ihre eigene Altersmischung hat.",
    "facet.faq_a_raw": "Der verlangte Median liegt bei {mine} für {a} und bei {other} für {b}, aber in unterschiedlichen Zulassungsjahren und ohne genug gemeinsame Jahre für einen Vergleich bei gleichem Alter. Der Abstand zwischen beiden Zahlen enthält den Altersunterschied.",
    "facet.ds_name": "Gebrauchtwagenpreise {phrase} in {country}",
    "facet.ds_desc": "Median und Interquartilsabstand der verlangten Preise in {n} aktiven Angeboten für {phrase} auf {source}.",
    "facet.model_links_h": "Preise nach Schnitt",
    "facet.model_links_fuel": "NACH KRAFTSTOFF",
    "facet.model_links_gear": "NACH GETRIEBE",
    "facet.model_links_region": "NACH REGION",
    "region.crumb": "Regionen",
    "region.hub_eyebrow": "PREISE NACH BUNDESLAND · {source}",
    "region.hub_h1": "Gebrauchtwagenpreise nach Bundesland",
    "region.hub_title": "Gebrauchtwagenpreise nach Bundesland in {country}",
    "region.hub_desc": "Verlangter Medianpreis für Gebrauchtwagen in jedem Bundesland mit ausreichender Stichprobe: {regions} Regionen, {n} aktive Angebote auf {source}.",
    "region.hub_lede": "In {n} aktiven Angeboten liegt der verlangte Median bei {price}. So verteilt er sich auf die {regions} Regionen, die genug Angebote für eine belastbare Zahl haben.",
    "region.hub_measure": "Verlangter Preis nach Region (Median und P25 bis P75)",
    "region.th_region": "Region",
    "region.th_median": "Median",
    "region.th_range": "Mittlere Hälfte",
    "region.th_n": "Angebote",
    "region.th_km": "Kilometerstand",
    "region.eyebrow": "{label} · {source}",
    "region.h1": "Gebrauchtwagenpreise {region}",
    "region.title": "Gebrauchtwagenpreise {region}: Median {price}",
    "region.desc": "Gebrauchtwagen {region}: verlangter Medianpreis {price} in {n} aktiven Angeboten auf {source}, und wie jedes Modell gegen den Landesmedian steht.",
    "region.lede": "In {n} aktiven Angeboten {region} liegt der verlangte Median bei {price}{km}.{vs}",
    "region.lede_km": ", bei {km} Kilometerstand im Median",
    "region.vs_above": " Das sind {pct} über dem Median des ganzen Landes ({nat}).",
    "region.vs_below": " Das sind {pct} unter dem Median des ganzen Landes ({nat}).",
    "region.cap_median": "Median hier",
    "region.cap_n": "Angebote",
    "region.cap_active": "jetzt aktiv",
    "region.cap_km": "Kilometerstand, Median",
    "region.cap_here": "hier im Angebot",
    "region.models_h": "Meistangebotene Modelle {region}",
    "region.models_p": "Die letzte Spalte ist die interessante: wo der örtliche Preis vom Landesmedian abweicht, gibt es hier entweder mehr Angebot oder mehr Nachfrage für dasselbe Auto.",
    "region.models_none": "Die {n} Angebote {region} reichen für einen Median der Region, aber nicht für einen Median je Modell: kein einziges Modell hat hier genug Angebote, damit seine Zahl etwas bedeuten würde. Statt sie aus einer Handvoll Autos zu erfinden, bleibt es bei dem, was die Region als Ganzes sagt.",
    "region.th_model": "Modell",
    "region.th_n_here": "Angebote hier",
    "region.th_median_here": "Median hier",
    "region.th_median_nat": "Median im Land",
    "region.th_diff": "Abstand",
    "region.rank_h": "Wo {label} im Land steht",
    "region.rank_line": "Unter den {total} Regionen mit ausreichender Stichprobe steht {label} auf <b>Platz {pos} der teuersten</b>: {top_price} an der Spitze ({top}) gegen {bot_price} am Ende ({bot}). Zwischen den Enden liegen {gap}, und es ist nicht dasselbe Auto, das mehr kostet: wo die Preise höher sind, werden auch andere Autos angeboten, jünger und mit weniger Kilometern.",
    "region.th_vs": "gegen {label}",
    "region.links_hub": "Alle Modelle",
    "region.links_regions": "Alle Regionen",
    "region.links_avaliar": "Ein Inserat bewerten",
    "region.measure": "Verlangter Preis in Angeboten mit Standort {region}",
    "region.faq_q": "Was kostet ein Gebrauchtwagen {region}?",
    "region.faq_a": "In {n} aktiven Angeboten {region} liegt der verlangte Median bei {price}, die mittlere Hälfte der Angebote zwischen {lo} und {hi}. Das sind verlangte Preise, keine erzielten Verkaufspreise.",
    "region.ds_name": "Gebrauchtwagenpreise {region}",
    "region.ds_desc": "Verlangter Medianpreis und Interquartilsabstand in {n} aktiven Gebrauchtwagenangeboten mit Standort {region}, Quelle {source}.",
    "json.facet_matched_note": "Verhältnis innerhalb jedes Zulassungsjahres gemessen und erst danach zusammengefasst; rohe Mediane lassen sich nicht voneinander abziehen, weil sie unterschiedliche Altersmischungen beschreiben.",
    "json.facet_normalized_note": "Jedes Angebot geteilt durch den Median des Modells in seinem eigenen Zulassungsjahr; verwendet, wo die Stichprobe für einen Vergleich Jahr für Jahr zu dünn ist.",
    "json.facet_measured_note": "VERLANGTE Preise aus aktiven Angeboten auf {source}, keine erzielten Verkaufspreise.",
    "json.region_note": "Regionen unter der nationalen Mindeststichprobe haben keine Seite und erscheinen hier nicht.",
  },
  fr: {
    "footer.regions": "Prix par région",
    "cut.dearer": "plus cher",
    "cut.cheaper": "moins cher",
    "facet.eyebrow": "{label} · {source}",
    "facet.phrase_fuel": "{brand} {model} {label}",
    "facet.phrase_gear": "{brand} {model} boîte {label}",
    "facet.phrase_region": "{brand} {model} {region}",
    "facet.h1": "{phrase} d'occasion : quel est le prix ?",
    "facet.title": "{phrase} d'occasion : prix {price}{month} ({n} annonces)",
    "facet.title_month": " en {month}",
    "facet.desc": "{phrase} : prix médian {price} ({lo} à {hi}) sur {n} annonces actives {source}.{age}",
    "facet.lede": "Sur {n} annonces actives, une {phrase} est affichée à {price} en médiane{km}{years}.{age}",
    "facet.lede_km": ", avec {km} au compteur en médiane",
    "facet.lede_years": ", première immatriculation de {from} à {to}",
    "facet.cap_median": "Prix affiché, médiane",
    "facet.cap_sample": "{n} annonces",
    "facet.cap_share": "{n} annonces · {share} du modèle",
    "facet.cap_km": "Kilométrage médian",
    "facet.range_label": "fourchette courante (la moitié centrale des annonces)",
    "facet.age_more": "À âge égal, ce prix est {pct} plus cher {ref}.",
    "facet.age_less": "À âge égal, ce prix est {pct} moins cher {ref}.",
    "facet.age_same": "À âge égal, l'écart disparaît : la distance entre les deux médianes brutes n'est que la différence d'âge.",
    "facet.ref_model": "que l'ensemble du modèle",
    "facet.ref_model_country": "que le modèle sur tout le pays",
    "facet.ref_rest_cuts": "que les autres versions de ce modèle",
    "facet.ref_rest_country": "que le même modèle hors de cette région",
    "facet.method_matched": "comparaison année par année, sur {years} années de première immatriculation avec de l'échantillon des deux côtés",
    "facet.method_normalized": "chacune des {used} annonces divisée par la médiane du reste du modèle dans sa propre année de première immatriculation",
    "facet.raw_caveat": " Les médianes brutes ({a} contre {b}) sont plus éloignées que cela : chaque côté a son propre mélange d'âges.",
    "facet.h2_fuel": "Face aux autres motorisations",
    "facet.h2_gear": "Face à l'autre boîte",
    "facet.h2_region": "Face au reste du pays",
    "facet.sib_label_fuel": "AUTRES MOTORISATIONS",
    "facet.sib_label_gear": "AUTRES BOÎTES",
    "facet.sib_label_region": "AUTRES RÉGIONS",
    "facet.vs_model_matched": "Face à toutes les {brand} {model} du pays (médiane {all}), on demande ici {verdict} – {method}.{caveat}",
    "facet.vs_model_raw": "Face à toutes les {brand} {model} du pays, on affiche ici {mine} en médiane contre {all}{years}. Les deux chiffres ne se soustraient pas : ils décrivent des mélanges d'âges différents.",
    "facet.sib_matched": "Face à {link} ({n} annonces, médiane {fm}) : <b>{pct} {dir}</b>, {method}. Les médianes brutes ({mine} contre {fm}) disent autre chose, parce que chaque côté a son propre mélange d'âges.{km}",
    "facet.sib_raw": "Face à {link} : médiane {fm} sur {n} annonces{oyears}, contre {mine} ici{myears}. Il n'y a pas assez d'années de première immatriculation avec de l'échantillon des deux côtés pour comparer à âge égal – l'écart entre les deux médianes contient encore la différence d'âge.{km}",
    "facet.verdict_more": "{pct} de plus",
    "facet.verdict_less": "{pct} de moins",
    "facet.verdict_same": "la même chose",
    "facet.km_more": " Le kilométrage médian est ici supérieur de {km}.",
    "facet.km_less": " Le kilométrage médian est ici inférieur de {km}.",
    "facet.cta_h": "{phrase} : combien vaut la vôtre ?",
    "facet.cta_p": "Ci-dessus, c'est la médiane de cette sélection. Pour la valeur de votre voiture précise, indiquez le modèle et l'année de première immatriculation.",
    "facet.cta_btn": "Estimer ma voiture",
    "facet.links": "Toutes les {brand} {model}",
    "facet.links_region": "Voitures d'occasion {region}",
    "facet.links_hub": "Tous les modèles",
    "facet.measure": "Prix affiché, {phrase} (médiane et P25 à P75)",
    "facet.faq_q1": "Quel est le prix d'une {phrase} d'occasion ?",
    "facet.faq_a1": "Sur {n} annonces actives {source}, une {phrase} est affichée à {price} en médiane, la moitié centrale des annonces se situant entre {lo} et {hi}. Ce sont des prix affichés, pas des prix de vente conclus.",
    "facet.faq_q_fuel": "{brand} {model} : {a} ou {b} ?",
    "facet.faq_q_gear": "{brand} {model} : boîte {a} ou {b} ?",
    "facet.faq_q_region": "Une {brand} {model} est-elle plus chère {a} ou {b} ?",
    "facet.faq_a_matched": "En comparant année par année – uniquement les années où les deux côtés ont de l'échantillon, {years} au total – {a} ressort {pct} {dir} que {b}. Les médianes brutes ({mine} et {other}) ne se comparent pas directement, chaque côté ayant son propre mélange d'âges.",
    "facet.faq_a_raw": "La médiane affichée est de {mine} pour {a} et de {other} pour {b}, mais sur des années de première immatriculation différentes et sans assez d'années communes pour comparer à âge égal. L'écart entre les deux chiffres contient la différence d'âge.",
    "facet.ds_name": "Prix des {phrase} d'occasion en {country}",
    "facet.ds_desc": "Médiane et écart interquartile des prix affichés sur {n} annonces actives de {phrase} {source}.",
    "facet.model_links_h": "Prix par version",
    "facet.model_links_fuel": "PAR CARBURANT",
    "facet.model_links_gear": "PAR BOÎTE",
    "facet.model_links_region": "PAR RÉGION",
    "region.crumb": "Régions",
    "region.hub_eyebrow": "PRIX PAR RÉGION · {source}",
    "region.hub_h1": "Prix des voitures d'occasion par région",
    "region.hub_title": "Prix des voitures d'occasion par région en {country}",
    "region.hub_desc": "Prix affiché médian des voitures d'occasion dans chaque région où l'échantillon suffit : {regions} régions, {n} annonces actives {source}.",
    "region.hub_lede": "Sur {n} annonces actives, la médiane affichée est de {price}. Voici comment elle se répartit entre les {regions} régions qui ont assez d'annonces pour que le chiffre tienne.",
    "region.hub_measure": "Prix affiché par région (médiane et P25 à P75)",
    "region.th_region": "Région",
    "region.th_median": "Médiane",
    "region.th_range": "Moitié centrale",
    "region.th_n": "Annonces",
    "region.th_km": "Kilométrage",
    "region.eyebrow": "{label} · {source}",
    "region.h1": "Prix des voitures d'occasion {region}",
    "region.title": "Prix des voitures d'occasion {region} : médiane {price}",
    "region.desc": "Voitures d'occasion {region} : prix affiché médian {price} sur {n} annonces actives {source}, et l'écart de chaque modèle avec la médiane nationale.",
    "region.lede": "Sur {n} annonces actives {region}, la médiane affichée est de {price}{km}.{vs}",
    "region.lede_km": ", avec {km} au compteur en médiane",
    "region.vs_above": " Soit {pct} au-dessus de la médiane nationale ({nat}).",
    "region.vs_below": " Soit {pct} en dessous de la médiane nationale ({nat}).",
    "region.cap_median": "Médiane ici",
    "region.cap_n": "Annonces",
    "region.cap_active": "actives maintenant",
    "region.cap_km": "Kilométrage médian",
    "region.cap_here": "en vente ici",
    "region.models_h": "Modèles les plus annoncés {region}",
    "region.models_p": "La dernière colonne est celle qui compte : là où le prix local s'écarte du national, soit l'offre est plus abondante ici, soit la même voiture se paie plus cher.",
    "region.models_none": "Les {n} annonces {region} suffisent pour une médiane de région, pas pour une médiane par modèle : aucun modèle n'a ici assez d'annonces pour que son chiffre veuille dire quelque chose. Plutôt que de l'inventer avec une poignée de voitures, on s'en tient à ce que dit la région dans son ensemble.",
    "region.th_model": "Modèle",
    "region.th_n_here": "Annonces ici",
    "region.th_median_here": "Médiane ici",
    "region.th_median_nat": "Médiane nationale",
    "region.th_diff": "Écart",
    "region.rank_h": "Où se situe {label} dans le pays",
    "region.rank_line": "Parmi les {total} régions dont l'échantillon suffit, {label} est la <b>{pos}e plus chère</b> : {top_price} en tête ({top}) contre {bot_price} en fin de classement ({bot}). L'écart entre les extrêmes est de {gap}, et ce n'est pas la même voiture qui coûte plus cher : là où les prix sont plus élevés, on annonce aussi d'autres voitures, plus récentes et moins kilométrées.",
    "region.th_vs": "contre {label}",
    "region.links_hub": "Tous les modèles",
    "region.links_regions": "Toutes les régions",
    "region.links_avaliar": "Estimer une annonce",
    "region.measure": "Prix affiché dans les annonces localisées {region}",
    "region.faq_q": "Combien coûte une voiture d'occasion {region} ?",
    "region.faq_a": "Sur {n} annonces actives {region}, la médiane affichée est de {price}, la moitié centrale des annonces se situant entre {lo} et {hi}. Ce sont des prix affichés, pas des prix de vente conclus.",
    "region.ds_name": "Prix des voitures d'occasion {region}",
    "region.ds_desc": "Prix affiché médian et écart interquartile sur {n} annonces actives de voitures d'occasion localisées {region}, source {source}.",
    "json.facet_matched_note": "Rapport mesuré à l'intérieur de chaque année de première immatriculation puis seulement agrégé ; les médianes brutes ne se soustraient pas, elles décrivent des mélanges d'âges différents.",
    "json.facet_normalized_note": "Chaque annonce divisée par la médiane du modèle dans sa propre année de première immatriculation ; utilisé là où l'échantillon est trop mince pour un rapport année par année.",
    "json.facet_measured_note": "Prix AFFICHÉS dans des annonces actives {source}, pas des prix de vente conclus.",
    "json.region_note": "Les régions sous l'échantillon minimal national n'ont pas de page et n'apparaissent pas ici.",
  },
  it: {
    "footer.regions": "Prezzi per regione",
    "cut.dearer": "più caro",
    "cut.cheaper": "più economico",
    "facet.eyebrow": "{label} · {source}",
    "facet.phrase_fuel": "{brand} {model} {label}",
    "facet.phrase_gear": "{brand} {model} cambio {label}",
    "facet.phrase_region": "{brand} {model} {region}",
    "facet.h1": "{phrase} usata: quanto costa?",
    "facet.title": "{phrase} usata: prezzo {price}{month} ({n} annunci)",
    "facet.title_month": " a {month}",
    "facet.desc": "{phrase}: prezzo mediano {price} (da {lo} a {hi}) su {n} annunci attivi su {source}.{age}",
    "facet.lede": "Su {n} annunci attivi il prezzo richiesto mediano per {phrase} è {price}{km}{years}.{age}",
    "facet.lede_km": ", con {km} di chilometraggio mediano",
    "facet.lede_years": ", immatricolazione dal {from} al {to}",
    "facet.cap_median": "Prezzo richiesto, mediana",
    "facet.cap_sample": "{n} annunci",
    "facet.cap_share": "{n} annunci · {share} del modello",
    "facet.cap_km": "Chilometraggio mediano",
    "facet.range_label": "fascia tipica (la metà centrale degli annunci)",
    "facet.age_more": "A parità di età questo prezzo è del {pct} più caro {ref}.",
    "facet.age_less": "A parità di età questo prezzo è del {pct} più economico {ref}.",
    "facet.age_same": "A parità di età la differenza sparisce: la distanza fra le due mediane grezze è solo differenza di età.",
    "facet.ref_model": "rispetto a tutto il modello",
    "facet.ref_model_country": "rispetto al modello in tutto il Paese",
    "facet.ref_rest_cuts": "rispetto alle altre versioni di questo modello",
    "facet.ref_rest_country": "rispetto allo stesso modello fuori da questa regione",
    "facet.method_matched": "confronto anno per anno, su {years} anni di immatricolazione con campione da entrambe le parti",
    "facet.method_normalized": "ogni annuncio, {used} in tutto, diviso per la mediana del resto del modello nel suo anno di immatricolazione",
    "facet.raw_caveat": " Le mediane grezze ({a} contro {b}) sono più distanti di così, perché ogni lato ha la sua miscela di età.",
    "facet.h2_fuel": "Contro le altre alimentazioni",
    "facet.h2_gear": "Contro l'altro cambio",
    "facet.h2_region": "Contro il resto del Paese",
    "facet.sib_label_fuel": "ALTRE ALIMENTAZIONI",
    "facet.sib_label_gear": "ALTRI CAMBI",
    "facet.sib_label_region": "ALTRE REGIONI",
    "facet.vs_model_matched": "Rispetto a tutte le {brand} {model} del Paese (mediana {all}), qui si chiede {verdict} – {method}.{caveat}",
    "facet.vs_model_raw": "Rispetto a tutte le {brand} {model} del Paese, qui si chiedono {mine} di mediana contro {all}{years}. I due numeri non si sottraggono: descrivono miscele di età diverse.",
    "facet.sib_matched": "Contro {link} ({n} annunci, mediana {fm}): <b>{pct} {dir}</b>, {method}. Le mediane grezze ({mine} contro {fm}) dicono altro, perché ogni lato ha la sua miscela di età.{km}",
    "facet.sib_raw": "Contro {link}: mediana {fm} su {n} annunci{oyears}, qui {mine}{myears}. Non ci sono abbastanza anni di immatricolazione con campione da entrambe le parti per confrontare a parità di età: la distanza fra le due mediane contiene ancora la differenza di età.{km}",
    "facet.verdict_more": "il {pct} in più",
    "facet.verdict_less": "il {pct} in meno",
    "facet.verdict_same": "lo stesso",
    "facet.km_more": " Il chilometraggio mediano qui è più alto di {km}.",
    "facet.km_less": " Il chilometraggio mediano qui è più basso di {km}.",
    "facet.cta_h": "{phrase}: quanto vale la tua?",
    "facet.cta_p": "Quella sopra è la mediana di questa selezione. Per il valore della tua auto precisa indica modello e anno di immatricolazione.",
    "facet.cta_btn": "Valuta la mia auto",
    "facet.links": "Tutte le {brand} {model}",
    "facet.links_region": "Auto usate {region}",
    "facet.links_hub": "Tutti i modelli",
    "facet.measure": "Prezzo richiesto, {phrase} (mediana e P25-P75)",
    "facet.faq_q1": "Quanto costa {phrase} usata?",
    "facet.faq_a1": "Su {n} annunci attivi su {source} il prezzo richiesto mediano per {phrase} è {price}, con la metà centrale degli annunci fra {lo} e {hi}. Sono prezzi richiesti, non prezzi di vendita conclusi.",
    "facet.faq_q_fuel": "{brand} {model}: {a} o {b}?",
    "facet.faq_q_gear": "{brand} {model}: cambio {a} o {b}?",
    "facet.faq_q_region": "Una {brand} {model} costa di più {a} o {b}?",
    "facet.faq_a_matched": "Confrontando anno per anno – solo gli anni in cui entrambi hanno campione, {years} in tutto – {a} risulta del {pct} {dir} di {b}. Le mediane grezze ({mine} e {other}) non si confrontano direttamente, perché ogni lato ha la sua miscela di anni.",
    "facet.faq_a_raw": "La mediana richiesta è {mine} per {a} e {other} per {b}, ma su anni di immatricolazione diversi e senza abbastanza anni in comune per confrontare a parità di età. La distanza fra i due numeri contiene la differenza di età.",
    "facet.ds_name": "Prezzi di {phrase} usate in {country}",
    "facet.ds_desc": "Mediana e scarto interquartile dei prezzi richiesti su {n} annunci attivi di {phrase} su {source}.",
    "facet.model_links_h": "Prezzi per versione",
    "facet.model_links_fuel": "PER ALIMENTAZIONE",
    "facet.model_links_gear": "PER CAMBIO",
    "facet.model_links_region": "PER REGIONE",
    "region.crumb": "Regioni",
    "region.hub_eyebrow": "PREZZI PER REGIONE · {source}",
    "region.hub_h1": "Prezzi delle auto usate per regione",
    "region.hub_title": "Prezzi delle auto usate per regione in {country}",
    "region.hub_desc": "Prezzo richiesto mediano delle auto usate in ogni regione con campione sufficiente: {regions} regioni, {n} annunci attivi su {source}.",
    "region.hub_lede": "Su {n} annunci attivi la mediana richiesta è {price}. Ecco come si distribuisce fra le {regions} regioni che hanno abbastanza annunci perché il numero regga.",
    "region.hub_measure": "Prezzo richiesto per regione (mediana e P25-P75)",
    "region.th_region": "Regione",
    "region.th_median": "Mediana",
    "region.th_range": "Metà centrale",
    "region.th_n": "Annunci",
    "region.th_km": "Chilometraggio",
    "region.eyebrow": "{label} · {source}",
    "region.h1": "Prezzi delle auto usate {region}",
    "region.title": "Prezzi delle auto usate {region}: mediana {price}",
    "region.desc": "Auto usate {region}: prezzo richiesto mediano {price} su {n} annunci attivi su {source}, e come ogni modello si colloca rispetto alla mediana nazionale.",
    "region.lede": "Su {n} annunci attivi {region} la mediana richiesta è {price}{km}.{vs}",
    "region.lede_km": ", con {km} di chilometraggio mediano",
    "region.vs_above": " Sono il {pct} sopra la mediana nazionale ({nat}).",
    "region.vs_below": " Sono il {pct} sotto la mediana nazionale ({nat}).",
    "region.cap_median": "Mediana qui",
    "region.cap_n": "Annunci",
    "region.cap_active": "attivi ora",
    "region.cap_km": "Chilometraggio mediano",
    "region.cap_here": "in vendita qui",
    "region.models_h": "Modelli più annunciati {region}",
    "region.models_p": "L'ultima colonna è quella che conta: dove il prezzo locale si stacca da quello nazionale, o qui c'è più offerta, oppure la stessa auto si paga di più.",
    "region.models_none": "I {n} annunci {region} bastano per una mediana della regione, non per una mediana per modello: nessun modello ha qui abbastanza annunci perché il suo numero significhi qualcosa. Invece di inventarlo con una manciata di auto, ci fermiamo a quello che dice la regione nel suo insieme.",
    "region.th_model": "Modello",
    "region.th_n_here": "Annunci qui",
    "region.th_median_here": "Mediana qui",
    "region.th_median_nat": "Mediana nazionale",
    "region.th_diff": "Scarto",
    "region.rank_h": "Dove si colloca {label} nel Paese",
    "region.rank_line": "Fra le {total} regioni con campione sufficiente, {label} è la <b>{pos}ª più cara</b>: {top_price} in testa ({top}) contro {bot_price} in coda ({bot}). Fra i due estremi ci sono {gap}, e non è la stessa auto a costare di più: dove i prezzi sono più alti si annuncia anche altra roba, più recente e con meno chilometri.",
    "region.th_vs": "contro {label}",
    "region.links_hub": "Tutti i modelli",
    "region.links_regions": "Tutte le regioni",
    "region.links_avaliar": "Valuta un annuncio",
    "region.measure": "Prezzo richiesto negli annunci localizzati {region}",
    "region.faq_q": "Quanto costa un'auto usata {region}?",
    "region.faq_a": "Su {n} annunci attivi {region} la mediana richiesta è {price}, con la metà centrale degli annunci fra {lo} e {hi}. Sono prezzi richiesti, non prezzi di vendita conclusi.",
    "region.ds_name": "Prezzi delle auto usate {region}",
    "region.ds_desc": "Prezzo richiesto mediano e scarto interquartile su {n} annunci attivi di auto usate localizzate {region}, fonte {source}.",
    "json.facet_matched_note": "Rapporto misurato dentro ogni anno di immatricolazione e solo dopo aggregato; le mediane grezze non si sottraggono, perché descrivono miscele di età diverse.",
    "json.facet_normalized_note": "Ogni annuncio diviso per la mediana del modello nel suo anno di immatricolazione; usato dove il campione è troppo sottile per un rapporto anno per anno.",
    "json.facet_measured_note": "Prezzi RICHIESTI in annunci attivi su {source}, non prezzi di vendita conclusi.",
    "json.region_note": "Le regioni sotto il campione minimo nazionale non hanno pagina e qui non compaiono.",
  },
  pt: {
    "footer.regions": "Preços por região",
    "cut.dearer": "mais caro",
    "cut.cheaper": "mais barato",
    "facet.eyebrow": "{label} · {source}",
    "facet.phrase_fuel": "{brand} {model} {label}",
    "facet.phrase_gear": "{brand} {model} com caixa {label}",
    "facet.phrase_region": "{brand} {model} {region}",
    "facet.h1": "Quanto vale um {phrase} usado?",
    "facet.title": "{phrase} usado: preço {price}{month} ({n} anúncios)",
    "facet.title_month": " em {month}",
    "facet.desc": "{phrase}: preço mediano {price} ({lo} a {hi}) em {n} anúncios ativos do {source}.{age}",
    "facet.lede": "Em {n} anúncios ativos, um {phrase} pede em mediana {price}{km}{years}.{age}",
    "facet.lede_km": ", com {km} medianos",
    "facet.lede_years": ", para anos {from} a {to}",
    "facet.cap_median": "Preço pedido, mediana",
    "facet.cap_sample": "{n} anúncios",
    "facet.cap_share": "{n} anúncios · {share} do modelo",
    "facet.cap_km": "Quilometragem mediana",
    "facet.range_label": "intervalo típico (metade dos anúncios)",
    "facet.age_more": "Ajustado pela idade, este preço é {pct} mais caro {ref}.",
    "facet.age_less": "Ajustado pela idade, este preço é {pct} mais barato {ref}.",
    "facet.age_same": "Ajustado pela idade não sobra diferença: a distância entre as duas medianas em bruto é só diferença de anos.",
    "facet.ref_model": "do que o modelo todo",
    "facet.ref_model_country": "do que o modelo no país inteiro",
    "facet.ref_rest_cuts": "do que os restantes cortes deste modelo",
    "facet.ref_rest_country": "do que o mesmo modelo fora desta região",
    "facet.method_matched": "comparando ano a ano, sobre {years} anos de matrícula com amostra dos dois lados",
    "facet.method_normalized": "cada um dos {used} anúncios dividido pela mediana do resto do modelo no seu próprio ano de matrícula",
    "facet.raw_caveat": " As medianas em bruto ({a} contra {b}) estão mais afastadas do que isso porque cada lado tem a sua mistura de idades.",
    "facet.h2_fuel": "Contra as outras motorizações",
    "facet.h2_gear": "Contra a outra caixa",
    "facet.h2_region": "Contra o resto do país",
    "facet.sib_label_fuel": "OUTRAS MOTORIZAÇÕES",
    "facet.sib_label_gear": "OUTRAS CAIXAS",
    "facet.sib_label_region": "OUTRAS REGIÕES",
    "facet.vs_model_matched": "Face a todos os {brand} {model} do país (mediana {all}), aqui pede-se {verdict} – {method}.{caveat}",
    "facet.vs_model_raw": "Face a todos os {brand} {model} do país, aqui pede-se em mediana {mine} contra {all}{years}. Os dois números não se subtraem: descrevem misturas de idades diferentes.",
    "facet.sib_matched": "Contra {link} ({n} anúncios, mediana {fm}): <b>{pct} {dir}</b>, {method}. As medianas em bruto ({mine} contra {fm}) dizem outra coisa porque cada lado tem a sua mistura de idades.{km}",
    "facet.sib_raw": "Contra {link}: mediana {fm} em {n} anúncios{oyears}, aqui {mine}{myears}. Não há anos de matrícula que cheguem com amostra dos dois lados para comparar à mesma idade: a distância entre as duas medianas ainda inclui a diferença de anos.{km}",
    "facet.verdict_more": "mais {pct}",
    "facet.verdict_less": "menos {pct}",
    "facet.verdict_same": "o mesmo",
    "facet.km_more": " A quilometragem mediana é aqui superior em {km}.",
    "facet.km_less": " A quilometragem mediana é aqui inferior em {km}.",
    "facet.cta_h": "Tens um {phrase}?",
    "facet.cta_p": "Acima está a mediana desta seleção. Para o valor do teu carro em concreto, indica modelo e ano de matrícula.",
    "facet.cta_btn": "Avaliar o meu carro",
    "facet.links": "Todos os {brand} {model}",
    "facet.links_region": "Carros usados {region}",
    "facet.links_hub": "Todos os modelos",
    "facet.measure": "Preço pedido, {phrase} (mediana e P25-P75)",
    "facet.faq_q1": "Quanto vale um {phrase} usado?",
    "facet.faq_a1": "Em {n} anúncios ativos do {source}, um {phrase} pede em mediana {price}, com metade dos anúncios entre {lo} e {hi}. São preços pedidos, não preços de venda fechados.",
    "facet.faq_q_fuel": "{brand} {model}: {a} ou {b}?",
    "facet.faq_q_gear": "{brand} {model}: caixa {a} ou {b}?",
    "facet.faq_q_region": "Um {brand} {model} é mais caro {a} ou {b}?",
    "facet.faq_a_matched": "Comparando ano a ano - só anos em que ambos têm amostra, {years} ao todo - {a} fica {pct} {dir} do que {b}. As medianas em bruto ({mine} e {other}) não se comparam directamente porque cada lado tem a sua mistura de anos.",
    "facet.faq_a_raw": "A mediana pedida é {mine} para {a} e {other} para {b}, mas em anos de matrícula diferentes e sem anos que cheguem em comum para comparar à mesma idade. A distância entre os dois números inclui a diferença de idades.",
    "facet.ds_name": "Preços de {phrase} usados em {country}",
    "facet.ds_desc": "Mediana e intervalo interquartil dos preços pedidos em {n} anúncios ativos de {phrase} no {source}.",
    "facet.model_links_h": "Preços por corte",
    "facet.model_links_fuel": "POR COMBUSTÍVEL",
    "facet.model_links_gear": "POR CAIXA",
    "facet.model_links_region": "POR REGIÃO",
    "region.crumb": "Regiões",
    "region.hub_eyebrow": "PREÇOS POR REGIÃO · {source}",
    "region.hub_h1": "Preços de carros usados por região",
    "region.hub_title": "Preços de carros usados por região em {country}",
    "region.hub_desc": "Preço pedido mediano de carros usados em cada região com amostra suficiente: {regions} regiões, {n} anúncios ativos do {source}.",
    "region.hub_lede": "Em {n} anúncios ativos, a mediana pedida é {price}. É assim que se reparte pelas {regions} regiões com anúncios que cheguem para o número aguentar.",
    "region.hub_measure": "Preço pedido por região (mediana e P25-P75)",
    "region.th_region": "Região",
    "region.th_median": "Mediana",
    "region.th_range": "Metade central",
    "region.th_n": "Anúncios",
    "region.th_km": "Quilometragem",
    "region.eyebrow": "{label} · {source}",
    "region.h1": "Preços de carros usados {region}",
    "region.title": "Preços de carros usados {region}: mediana {price}",
    "region.desc": "Carros usados {region}: preço pedido mediano {price} em {n} anúncios ativos do {source}, e como cada modelo se compara com a mediana nacional.",
    "region.lede": "Em {n} anúncios ativos {region}, a mediana pedida é {price}{km}.{vs}",
    "region.lede_km": ", com {km} de quilometragem mediana",
    "region.vs_above": " Isso é {pct} acima da mediana nacional ({nat}).",
    "region.vs_below": " Isso é {pct} abaixo da mediana nacional ({nat}).",
    "region.cap_median": "Mediana aqui",
    "region.cap_n": "Anúncios",
    "region.cap_active": "ativos agora",
    "region.cap_km": "Quilometragem mediana",
    "region.cap_here": "à venda aqui",
    "region.models_h": "Modelos mais anunciados {region}",
    "region.models_p": "A última coluna é o que interessa: onde o preço local se afasta do nacional, ou há mais oferta aqui, ou o mesmo carro custa mais.",
    "region.models_none": "Os {n} anúncios {region} chegam para uma mediana da região, mas não para uma mediana por modelo: nenhum modelo tem aqui anúncios que cheguem para que a sua mediana signifique alguma coisa. Em vez de a inventar com quatro carros, ficamos pelo que a região diz no seu conjunto.",
    "region.th_model": "Modelo",
    "region.th_n_here": "Anúncios aqui",
    "region.th_median_here": "Mediana aqui",
    "region.th_median_nat": "Mediana nacional",
    "region.th_diff": "Diferença",
    "region.rank_h": "Onde {label} fica no país",
    "region.rank_line": "Entre as {total} regiões com amostra suficiente, {label} é a <b>{pos}.ª mais cara</b>: {top_price} no topo ({top}) contra {bot_price} no fim ({bot}). A diferença entre pontas é de {gap}, e não é o mesmo carro a custar mais: onde os preços são mais altos anuncia-se também outro tipo de carro, mais recente e com menos quilómetros.",
    "region.th_vs": "contra {label}",
    "region.links_hub": "Todos os modelos",
    "region.links_regions": "Todas as regiões",
    "region.links_avaliar": "Avaliar um anúncio",
    "region.measure": "Preço pedido em anúncios com localização {region}",
    "region.faq_q": "Quanto custa um carro usado {region}?",
    "region.faq_a": "Em {n} anúncios ativos {region}, a mediana pedida é {price}, com metade dos anúncios entre {lo} e {hi}. São preços pedidos, não preços de venda fechados.",
    "region.ds_name": "Preços de carros usados {region}",
    "region.ds_desc": "Preço pedido mediano e intervalo interquartil em {n} anúncios ativos de carros usados com localização {region}, fonte {source}.",
    "json.facet_matched_note": "Razão medida dentro de cada ano de matrícula e só depois juntada; as medianas em bruto não se subtraem porque descrevem misturas de idades diferentes.",
    "json.facet_normalized_note": "Cada anúncio dividido pela mediana do modelo no seu próprio ano de matrícula; usado onde a amostra é fina demais para a razão ano a ano.",
    "json.facet_measured_note": "Preços PEDIDOS em anúncios ativos do {source}, não preços de venda fechados.",
    "json.region_note": "Regiões abaixo da amostra mínima nacional não têm página e não aparecem aqui.",
  },
};

for (const [code, dict] of Object.entries(STRINGS)) registerStrings(code, dict);
for (const [code, routes] of Object.entries(ROUTES)) registerRoutes(code, routes);
registerNav([{ routeKey: "regions", labelKey: "footer.regions" }]);

const KIND_SUFFIX = { fuel: "fuel", transmission: "gear", district: "region" };

function kindKey(kind, stem) {
  return `${stem}_${KIND_SUFFIX[kind] || "fuel"}`;
}

function sentenceLabel(loc, lbl) {
  const s = String(lbl == null ? "" : lbl);
  return loc.code === "de" ? s : s.toLowerCase();
}

function regionPhrase(loc, key, label) {
  const table = REGION_PREPOSITION[loc.code] || {};
  const prep = table[key] || REGION_PREPOSITION_DEFAULT[loc.code] || "in";
  return `${prep} ${label}`;
}

function day(builtAt) {
  return (builtAt || "").slice(0, 10);
}

function statBlock(items) {
  return `<div class="fc-stat-row">${items.filter(Boolean).map(item =>
    `<div class="fc-stat"><div class="k">${item.k}</div><div class="v">${item.v}</div>`
    + `${item.s ? `<div class="s">${item.s}</div>` : ""}</div>`).join("")}</div>`;
}

function eyebrow(text) {
  return `<div class="eyebrow"><span class="e-dot"></span><span class="mono">${text}</span></div>`;
}

function gauge(loc, lo, hi, at, label) {
  if (!(lo > 0) || !(hi > lo) || !(at > 0)) return "";
  const pos = Math.min(94, Math.max(6, ((at - lo) / (hi - lo)) * 100));
  return `<div style="margin-top:16px;">
    <div class="gauge-head"><span>${escapeHtml(fmtEurL(loc, lo))}</span><span>${escapeHtml(label)}</span><span>${escapeHtml(fmtEurL(loc, hi))}</span></div>
    <div class="gauge-track"><span class="gauge-pin" style="left:${pos.toFixed(1)}%;"></span></div>
  </div>`;
}

function homeCrumb(loc) {
  return { name: t(loc, "common.crumb_home"), href: href(loc, "landing") };
}

function hubCrumb(loc) {
  return { name: t(loc, "common.crumb_hub"), href: href(loc, "hub") };
}

function graph(nodes) {
  return { "@context": "https://schema.org", "@graph": nodes.filter(Boolean) };
}

function monthSuffix(loc, builtAt) {
  const tag = monthTagL(loc, builtAt);
  return tag ? t(loc, "facet.title_month", { month: tag }) : "";
}

export function facetPath(loc, slug, key) {
  return `${href(loc, "model", slug)}/${encodeURIComponent(key)}`;
}

export function regionPath(loc, key) {
  return href(loc, "regions", key);
}

export function intlFacetKeys(rec) {
  return [...publishedCells(rec, "fuel"), ...publishedCells(rec, "transmission"),
          ...publishedCells(rec, "district")].map(c => c.k);
}

export function regionKeys(districts) {
  return Object.entries(districts || {})
    .filter(([, r]) => r && r.fm > 0 && r.n > 0)
    .sort((a, b) => (b[1].n || 0) - (a[1].n || 0))
    .map(([k]) => k);
}

const REGION_INDEX = new Map();

function rememberRegions(loc, mdoc) {
  const districts = mdoc && mdoc.districts;
  if (!districts) return;
  REGION_INDEX.set(loc.code, { builtAt: (mdoc.built_at || null), keys: regionKeys(districts) });
}

function rememberedRegions(loc, builtAt) {
  const hit = REGION_INDEX.get(loc.code);
  return (hit && hit.builtAt === (builtAt || null)) ? hit.keys : [];
}

function ageComparison(loc, rec, kind, cell) {
  const matched = Array.isArray(cell.vsm) ? { ratio: cell.vsm[0], years: cell.vsm[1] } : null;
  const normalized = (!matched && Array.isArray(cell.dr))
    ? { ratio: cell.dr[0], used: cell.dr[1] } : null;
  const src = matched || normalized;
  if (!src || !isFinite(src.ratio) || src.ratio <= 0) return null;
  const isRegion = kind === "district";
  const gap = src.ratio - 1;
  const method = matched
    ? t(loc, "facet.method_matched", { years: matched.years })
    : t(loc, "facet.method_normalized", { used: normalized.used });
  const ref = matched
    ? t(loc, isRegion ? "facet.ref_model_country" : "facet.ref_model")
    : t(loc, isRegion ? "facet.ref_rest_country" : "facet.ref_rest_cuts");
  return { matched, normalized, gap, method, ref, moves: Math.abs(gap) >= AGE_GAP_MIN };
}

function ageSentence(loc, age) {
  if (!age) return "";
  if (!age.moves) return t(loc, "facet.age_same");
  const key = age.gap > 0 ? "facet.age_more" : "facet.age_less";
  return t(loc, key, { pct: fmtPctL(loc, Math.abs(age.gap)), ref: age.ref });
}

function pairRatio(cell, otherKey) {
  const v = cell.vs && cell.vs[otherKey];
  return Array.isArray(v) && isFinite(v[0]) && v[0] > 0 ? { ratio: v[0], years: v[1] } : null;
}

function facetPhrase(loc, rec, cell, kind, esc = escapeHtml) {
  const brand = esc(rec.b), model = esc(rec.m);
  if (kind === "district") {
    return t(loc, "facet.phrase_region", {
      brand, model, region: regionPhrase(loc, cell.k, esc(cell.lbl)),
    });
  }
  const label = esc(sentenceLabel(loc, labelL(loc, cell.lbl)));
  return t(loc, kindKey(kind, "facet.phrase"), { brand, model, label });
}

function plainText(s) {
  return String(s == null ? "" : s);
}

export function intlFacetJson(loc, rec, slug, kind, cell, siblings, { host, builtAt }) {
  const base = `https://${host}`;
  const canonical = `${base}${facetPath(loc, slug, cell.k)}`;
  return {
    source: "Carsbuyer",
    source_url: canonical,
    licence: t(loc, "json.licence"),
    measured: "asking_price",
    measured_note: t(loc, "json.facet_measured_note", { source: loc.source.name }),
    collected_until: day(builtAt) || null,
    updated_at: builtAt || null,
    market: loc.country, currency: "EUR", language: loc.code,
    brand: rec.b, model: rec.m, slug,
    facet: {
      kind: kind === "fuel" ? "fuel" : kind === "transmission" ? "transmission" : "region",
      key: cell.k, label: labelL(loc, cell.lbl), label_source: cell.lbl,
    },
    sample_size: cell.n,
    share_of_model_listings: rec.n ? Math.round((cell.n / rec.n) * 1000) / 1000 : null,
    asking_price: { median: cell.fm, p25: cell.fl, p75: cell.fh },
    mileage_km_median: cell.km != null ? cell.km : null,
    model_years: (cell.y0 && cell.y1) ? { from: cell.y0, to: cell.y1 } : null,
    vs_model_year_matched: Array.isArray(cell.vsm)
      ? { ratio: cell.vsm[0], shared_years: cell.vsm[1], note: t(loc, "json.facet_matched_note") }
      : null,
    vs_model_age_normalized: Array.isArray(cell.dr)
      ? { ratio: cell.dr[0], listings_used: cell.dr[1], note: t(loc, "json.facet_normalized_note") }
      : null,
    siblings: (siblings || []).filter(c => c.k !== cell.k).map(c => {
      const m = pairRatio(cell, c.k);
      return {
        key: c.k, label: labelL(loc, c.lbl), sample_size: c.n,
        asking_price_median: c.fm, page: `${base}${facetPath(loc, slug, c.k)}`,
        vs_this_cut_year_matched: m ? { ratio: m.ratio, shared_years: m.years } : null,
      };
    }),
    related: {
      model: `${base}${href(loc, "model", slug)}`,
      models_index: `${base}${href(loc, "hub")}`,
      methodology: `${base}${href(loc, "metodologia")}`,
    },
  };
}

export function renderIntlFacetPage({ loc, host, rec, slug, kind, cell, siblings, builtAt }) {
  const brand = escapeHtml(rec.b), model = escapeHtml(rec.m);
  const label = escapeHtml(labelL(loc, cell.lbl));
  const phrase = facetPhrase(loc, rec, cell, kind);
  const plain = facetPhrase(loc, rec, cell, kind, plainText);
  const canonical = `https://${host}${facetPath(loc, slug, cell.k)}`;
  const altJson = `${canonical}.json`;
  const price = escapeHtml(fmtEurL(loc, cell.fm));
  const lo = escapeHtml(fmtEurL(loc, cell.fl));
  const hi = escapeHtml(fmtEurL(loc, cell.fh));
  const age = ageComparison(loc, rec, kind, cell);
  const ageLine = ageSentence(loc, age);
  const share = rec.n ? fmtPctL(loc, cell.n / rec.n) : null;

  const lede = t(loc, "facet.lede", {
    n: fmtNumL(loc, cell.n), phrase, price,
    km: cell.km != null ? t(loc, "facet.lede_km", { km: escapeHtml(fmtKmL(loc, cell.km)) }) : "",
    years: (cell.y0 && cell.y1) ? t(loc, "facet.lede_years", { from: cell.y0, to: cell.y1 }) : "",
    age: ageLine ? ` ${ageLine}` : "",
  });

  const others = (siblings || []).filter(c => c.k !== cell.k);
  const rawCaveat = (rec.fm > 0 && Math.abs(cell.fm / rec.fm - 1) >= AGE_GAP_MIN)
    ? t(loc, "facet.raw_caveat", {
        a: escapeHtml(fmtEurL(loc, cell.fm)), b: escapeHtml(fmtEurL(loc, rec.fm)),
      })
    : "";
  const vsModel = age
    ? t(loc, "facet.vs_model_matched", {
        brand, model, all: escapeHtml(fmtEurL(loc, rec.fm)),
        verdict: age.moves
          ? t(loc, age.gap > 0 ? "facet.verdict_more" : "facet.verdict_less",
              { pct: fmtPctL(loc, Math.abs(age.gap)) })
          : t(loc, "facet.verdict_same"),
        method: age.method, caveat: age.moves ? rawCaveat : "",
      })
    : rec.fm > 0
      ? t(loc, "facet.vs_model_raw", {
          brand, model, mine: escapeHtml(fmtEurL(loc, cell.fm)),
          all: escapeHtml(fmtEurL(loc, rec.fm)),
          years: (cell.y0 && cell.y1) ? t(loc, "facet.lede_years", { from: cell.y0, to: cell.y1 }) : "",
        })
      : "";

  const kmTail = other => {
    if (other.km == null || cell.km == null || other.km === cell.km) return "";
    const diff = escapeHtml(fmtKmL(loc, Math.abs(cell.km - other.km)));
    return t(loc, cell.km > other.km ? "facet.km_more" : "facet.km_less", { km: diff });
  };
  const compare = others.map(o => {
    const linkText = kind === "district"
      ? escapeHtml(o.lbl) : escapeHtml(sentenceLabel(loc, labelL(loc, o.lbl)));
    const link = `<a href="${facetPath(loc, slug, o.k)}">${linkText}</a>`;
    const m = pairRatio(cell, o.k);
    if (m) {
      return `<li>${t(loc, "facet.sib_matched", {
        link, n: fmtNumL(loc, o.n), fm: escapeHtml(fmtEurL(loc, o.fm)),
        pct: fmtPctL(loc, Math.abs(m.ratio - 1)),
        dir: t(loc, m.ratio >= 1 ? "cut.dearer" : "cut.cheaper"),
        method: t(loc, "facet.method_matched", { years: m.years }),
        mine: escapeHtml(fmtEurL(loc, cell.fm)), km: kmTail(o),
      })}</li>`;
    }
    return `<li>${t(loc, "facet.sib_raw", {
      link, n: fmtNumL(loc, o.n), fm: escapeHtml(fmtEurL(loc, o.fm)),
      mine: escapeHtml(fmtEurL(loc, cell.fm)),
      oyears: (o.y0 && o.y1) ? t(loc, "facet.lede_years", { from: o.y0, to: o.y1 }) : "",
      myears: (cell.y0 && cell.y1) ? t(loc, "facet.lede_years", { from: cell.y0, to: cell.y1 }) : "",
      km: kmTail(o),
    })}</li>`;
  }).join("");

  const chips = (siblings || []).length > 1
    ? (siblings || []).map(c => `<a${c.k === cell.k ? ' class="on"' : ""} href="${facetPath(loc, slug, c.k)}">${escapeHtml(labelL(loc, c.lbl))}</a>`).join("")
    : "";

  const regionLink = kind === "district"
    ? ` · <a href="${regionPath(loc, cell.k)}">${t(loc, "facet.links_region", {
        region: regionPhrase(loc, cell.k, escapeHtml(cell.lbl)),
      })}</a>`
    : "";

  const body = crumbs([
    homeCrumb(loc), hubCrumb(loc),
    { name: `${rec.b} ${rec.m}`, href: href(loc, "model", slug) },
    { name: labelL(loc, cell.lbl) },
  ]) + `
    <section class="section fc-wrap" style="padding-top:16px;">
      ${eyebrow(t(loc, "facet.eyebrow", { label: label.toUpperCase(), source: loc.source.name.toUpperCase() }))}
      <h1 class="fc-h1">${t(loc, "facet.h1", { phrase })}</h1>
      <p class="fc-p">${lede}</p>
      ${statBlock([
        { k: t(loc, "facet.cap_median"), v: price,
          s: share ? escapeHtml(t(loc, "facet.cap_share", { n: fmtNumL(loc, cell.n), share }))
                   : escapeHtml(t(loc, "facet.cap_sample", { n: fmtNumL(loc, cell.n) })) },
        { k: t(loc, "facet.range_label"), v: `${lo} – ${hi}`, s: "" },
        cell.km != null
          ? { k: t(loc, "facet.cap_km"), v: escapeHtml(fmtKmL(loc, cell.km)), s: "" }
          : null,
      ])}
      ${gauge(loc, cell.fl, cell.fh, cell.fm, t(loc, "facet.range_label"))}
      ${intlProvenance(loc, {
        n: cell.n, builtAt,
        measure: t(loc, "facet.measure", { phrase: plain }),
      })}
    </section>
    ${(vsModel || compare) ? `<section class="section fc-wrap">
      <h2 class="fc-h2">${t(loc, kindKey(kind, "facet.h2"))}</h2>
      <ul class="fc-insights">${vsModel ? `<li>${vsModel}</li>` : ""}${compare}</ul>
    </section>` : ""}
    ${chips ? `<section class="section fc-wrap">
      <div class="sec-label" style="margin-bottom:10px;">${t(loc, kindKey(kind, "facet.sib_label"))}</div>
      <div class="fc-yearlinks">${chips}</div>
    </section>` : ""}
    <section class="section fc-wrap">
      <div class="cta-banner" style="background:#fff;border:1px solid #E8E6E1;">
        <div style="flex:1 1 360px;">
          <h2 style="color:#16181D;">${t(loc, "facet.cta_h", { phrase })}</h2>
          <p style="color:#5B606B;">${t(loc, "facet.cta_p")}</p>
        </div>
        <a class="btn-dark" href="${href(loc, "avaliar")}">${t(loc, "facet.cta_btn")}&nbsp;&nbsp;→</a>
      </div>
    </section>
    <section class="section fc-wrap" style="padding-bottom:70px;">
      <p class="fc-p"><a href="${href(loc, "model", slug)}">${t(loc, "facet.links", { brand, model })}</a>${regionLink} · <a href="${href(loc, "hub")}">${t(loc, "facet.links_hub")}</a></p>
    </section>`;

  const faqs = [[
    t(loc, "facet.faq_q1", { phrase: plain }),
    t(loc, "facet.faq_a1", {
      n: fmtNumL(loc, cell.n), phrase: plain, price: fmtEurL(loc, cell.fm),
      lo: fmtEurL(loc, cell.fl), hi: fmtEurL(loc, cell.fh), source: loc.source.name,
    }),
  ]];
  if (others.length) {
    const o = others[0];
    const m = pairRatio(cell, o.k);
    const a = kind === "district"
      ? regionPhrase(loc, cell.k, cell.lbl) : sentenceLabel(loc, labelL(loc, cell.lbl));
    const b = kind === "district"
      ? regionPhrase(loc, o.k, o.lbl) : sentenceLabel(loc, labelL(loc, o.lbl));
    faqs.push([
      t(loc, kindKey(kind, "facet.faq_q"), { brand: rec.b, model: rec.m, a, b }),
      m
        ? t(loc, "facet.faq_a_matched", {
            years: m.years, a, b, pct: fmtPctL(loc, Math.abs(m.ratio - 1)),
            dir: t(loc, m.ratio >= 1 ? "cut.dearer" : "cut.cheaper"),
            mine: fmtEurL(loc, cell.fm), other: fmtEurL(loc, o.fm),
          })
        : t(loc, "facet.faq_a_raw", {
            a, b, mine: fmtEurL(loc, cell.fm), other: fmtEurL(loc, o.fm),
          }),
    ]);
  }

  const descAge = (age && age.moves) ? ` ${ageLine.replace(/<[^>]+>/g, "")}` : "";
  return layout({
    title: t(loc, "facet.title", {
      phrase: plain, price: fmtEurL(loc, cell.fm),
      month: monthSuffix(loc, builtAt), n: fmtNumL(loc, cell.n),
    }),
    description: t(loc, "facet.desc", {
      phrase: plain, price: fmtEurL(loc, cell.fm),
      lo: fmtEurL(loc, cell.fl), hi: fmtEurL(loc, cell.fh),
      n: fmtNumL(loc, cell.n), source: loc.source.name, age: descAge,
    }),
    body, zone: "all", nav: "precos", depositCount: null, index: true,
    host, locale: loc, canonical, altJson,
    jsonLd: graph([
      breadcrumbLd(host, [
        homeCrumb(loc), hubCrumb(loc),
        { name: `${rec.b} ${rec.m}`, href: href(loc, "model", slug) },
        { name: labelL(loc, cell.lbl), href: facetPath(loc, slug, cell.k) },
      ]),
      faqLd(faqs),
      {
        "@type": "Dataset",
        "name": t(loc, "facet.ds_name", { phrase: plain, country: loc.countryName }),
        "description": t(loc, "facet.ds_desc", {
          n: cell.n, phrase: plain, source: loc.source.name,
        }),
        "url": canonical,
        "inLanguage": loc.lang,
        "license": CC_LICENCE,
        "creator": { "@type": "Organization", "name": "Carsbuyer", "url": `https://${host}/` },
        "isAccessibleForFree": true,
        "variableMeasured": "asking price (EUR)",
        ...(builtAt ? { "dateModified": builtAt } : {}),
        ...(cell.y0 && cell.y1 ? { "temporalCoverage": `${cell.y0}/${cell.y1}` } : {}),
        "distribution": [{
          "@type": "DataDownload", "encodingFormat": "application/json", "contentUrl": altJson,
        }],
      },
      {
        "@type": "AggregateOffer", "priceCurrency": "EUR",
        "lowPrice": cell.fl, "highPrice": cell.fh, "offerCount": cell.n, "url": canonical,
        "itemOffered": {
          "@type": "Car", "name": plain,
          "brand": { "@type": "Brand", "name": rec.b }, "model": rec.m,
          ...(kind === "fuel" ? { "fuelType": labelL(loc, cell.lbl) } : {}),
          ...(kind === "transmission" ? { "vehicleTransmission": labelL(loc, cell.lbl) } : {}),
        },
        ...(kind === "district"
          ? { "areaServed": { "@type": "Place", "name": cell.lbl } } : {}),
      },
    ]),
  });
}

export function intlModelCutLinks(loc, rec, slug, districts = null, inWave = true) {
  if (!inWave) return "";
  const groups = [["fuel", "facet.model_links_fuel"], ["transmission", "facet.model_links_gear"],
                  ["district", "facet.model_links_region"]];
  const blocks = groups.map(([kind, labelKey]) => {
    const cells = publishedCells(rec, kind);
    if (!cells.length) return "";
    const links = cells.map(c =>
      `<a href="${facetPath(loc, slug, c.k)}">${escapeHtml(labelL(loc, c.lbl))} <span class="mut">${escapeHtml(fmtEurL(loc, c.fm))}</span></a>`).join("");
    return `<div class="sec-label" style="margin-top:14px;">${t(loc, labelKey)}</div>`
      + `<div class="fc-yearlinks">${links}</div>`;
  }).join("");
  if (!blocks) return "";
  const hubLink = regionKeys(districts).length
    ? `<p class="fc-p" style="margin-top:14px;"><a href="${href(loc, "regions")}">${t(loc, "footer.regions")}</a></p>`
    : "";
  return `<section class="section fc-wrap">
      <h2 class="fc-h2">${t(loc, "facet.model_links_h")}</h2>
      ${blocks}
      ${hubLink}
    </section>`;
}

function regionRow(loc, key, rec) {
  return `<tr>
      <td><a href="${regionPath(loc, key)}" style="color:#177A47;font-weight:600;">${escapeHtml(rec.lbl)}</a></td>
      <td>${escapeHtml(fmtEurL(loc, rec.fm))}</td>
      <td class="mut">${escapeHtml(fmtEurL(loc, rec.fl))} – ${escapeHtml(fmtEurL(loc, rec.fh))}</td>
      <td class="mut">${fmtNumL(loc, rec.n)}</td>
      <td class="mut">${rec.kmm != null ? escapeHtml(fmtKmL(loc, rec.kmm)) : "—"}</td>
    </tr>`;
}

export function intlRegionHubJson(loc, districts, { host, builtAt }) {
  const base = `https://${host}`;
  const keys = regionKeys(districts);
  return {
    source: "Carsbuyer",
    source_url: `${base}${href(loc, "regions")}`,
    licence: t(loc, "json.licence"),
    measured: "asking_price",
    measured_note: t(loc, "json.facet_measured_note", { source: loc.source.name }),
    collected_until: day(builtAt) || null,
    updated_at: builtAt || null,
    market: loc.country, currency: "EUR", language: loc.code,
    coverage_note: t(loc, "json.region_note"),
    regions: keys.map(k => {
      const r = districts[k];
      return {
        key: k, label: r.lbl, sample_size: r.n,
        asking_price: { median: r.fm, p25: r.fl, p75: r.fh },
        mileage_km_median: r.kmm != null ? r.kmm : null,
        page: `${base}${regionPath(loc, k)}`,
      };
    }),
    related: {
      models_index: `${base}${href(loc, "hub")}`,
      methodology: `${base}${href(loc, "metodologia")}`,
    },
  };
}

export function renderIntlRegionHub({ loc, host, districts, stats, builtAt }) {
  const keys = regionKeys(districts);
  const canonical = `https://${host}${href(loc, "regions")}`;
  const altJson = `${canonical}.json`;
  const total = keys.reduce((a, k) => a + (districts[k].n || 0), 0);
  const rows = keys.map(k => regionRow(loc, k, districts[k])).join("");
  const price = fmtEurL(loc, stats && stats.priceMed != null ? stats.priceMed : districts[keys[0]].fm);
  const body = crumbs([homeCrumb(loc), hubCrumb(loc), { name: t(loc, "region.crumb") }]) + `
    <section class="section fc-wrap" style="padding-top:16px;">
      ${eyebrow(t(loc, "region.hub_eyebrow", { source: loc.source.name.toUpperCase() }))}
      <h1 class="fc-h1">${t(loc, "region.hub_h1")}</h1>
      <p class="fc-p">${t(loc, "region.hub_lede", {
        n: fmtNumL(loc, total), price: escapeHtml(price), regions: keys.length,
      })}</p>
      <div class="fc-scroll"><table class="fc-tbl">
        <thead><tr><th>${t(loc, "region.th_region")}</th><th>${t(loc, "region.th_median")}</th><th>${t(loc, "region.th_range")}</th><th>${t(loc, "region.th_n")}</th><th>${t(loc, "region.th_km")}</th></tr></thead>
        <tbody>${rows}</tbody></table></div>
      ${intlProvenance(loc, { n: total, builtAt, measure: t(loc, "region.hub_measure") })}
      <p class="fc-p" style="padding-bottom:70px;"><a href="${href(loc, "hub")}">${t(loc, "region.links_hub")}</a> · <a href="${href(loc, "avaliar")}">${t(loc, "region.links_avaliar")}</a></p>
    </section>`;
  return layout({
    title: t(loc, "region.hub_title", { country: loc.countryName }),
    description: t(loc, "region.hub_desc", {
      regions: keys.length, n: fmtNumL(loc, total), source: loc.source.name,
    }),
    body, zone: "all", nav: "precos", depositCount: null, index: true,
    host, locale: loc, canonical, altJson,
    jsonLd: graph([
      breadcrumbLd(host, [
        homeCrumb(loc), hubCrumb(loc), { name: t(loc, "region.crumb"), href: href(loc, "regions") },
      ]),
      {
        "@type": "ItemList",
        "name": t(loc, "region.hub_h1"),
        "numberOfItems": keys.length,
        "itemListElement": keys.map((k, i) => ({
          "@type": "ListItem", "position": i + 1, "name": districts[k].lbl,
          "item": `https://${host}${regionPath(loc, k)}`,
        })),
      },
    ]),
  });
}

export function intlRegionJson(loc, key, rec, models, districts, { host, builtAt }) {
  const base = `https://${host}`;
  const rank = districtRanking(districts || {}, key);
  return {
    source: "Carsbuyer",
    source_url: `${base}${regionPath(loc, key)}`,
    licence: t(loc, "json.licence"),
    measured: "asking_price",
    measured_note: t(loc, "json.facet_measured_note", { source: loc.source.name }),
    collected_until: day(builtAt) || null,
    updated_at: builtAt || null,
    market: loc.country, currency: "EUR", language: loc.code,
    region: { key, label: rec.lbl },
    sample_size: rec.n,
    asking_price: { median: rec.fm, p25: rec.fl, p75: rec.fh },
    mileage_km_median: rec.kmm != null ? rec.kmm : null,
    rank_by_median: rank.pos, regions_ranked: rank.total,
    deepest_models: (rec.top || []).slice(0, REGION_TOP_MODELS).map(([slug, n, fm]) => {
      const m = models && models[slug];
      return {
        slug, brand: m ? m.b : null, model: m ? m.m : null,
        listings_here: n, asking_price_median_here: fm,
        asking_price_median_country: m ? m.fm : null,
        page: `${base}${href(loc, "model", slug)}`,
      };
    }),
    related: {
      regions_index: `${base}${href(loc, "regions")}`,
      models_index: `${base}${href(loc, "hub")}`,
      methodology: `${base}${href(loc, "metodologia")}`,
    },
  };
}

export function renderIntlRegionPage({ loc, host, key, rec, models, districts, stats, builtAt }) {
  const canonical = `https://${host}${regionPath(loc, key)}`;
  const altJson = `${canonical}.json`;
  const label = escapeHtml(rec.lbl);
  const region = regionPhrase(loc, key, label);
  const plainRegion = regionPhrase(loc, key, rec.lbl);
  const price = escapeHtml(fmtEurL(loc, rec.fm));
  const nat = stats && stats.priceMed ? stats.priceMed : null;
  const vsNat = nat ? (rec.fm - nat) / nat : null;
  const vs = vsNat == null ? "" : t(loc, vsNat >= 0 ? "region.vs_above" : "region.vs_below", {
    pct: fmtPctL(loc, Math.abs(vsNat)), nat: escapeHtml(fmtEurL(loc, nat)),
  });

  const modelRows = (rec.top || []).slice(0, REGION_TOP_MODELS).map(([slug, n, fm]) => {
    const m = models && models[slug];
    if (!m) return "";
    const d = m.fm > 0 ? (fm - m.fm) / m.fm : null;
    return `<tr>
      <td><a href="${href(loc, "model", slug)}" style="color:#177A47;font-weight:600;">${escapeHtml(m.b)} ${escapeHtml(m.m)}</a></td>
      <td>${fmtNumL(loc, n)}</td><td>${escapeHtml(fmtEurL(loc, fm))}</td>
      <td class="mut">${escapeHtml(fmtEurL(loc, m.fm))}</td>
      <td>${d == null ? "—" : `${d >= 0 ? "+" : "−"}${escapeHtml(fmtPctL(loc, Math.abs(d)))}`}</td></tr>`;
  }).join("");

  const rank = districtRanking(districts || {}, key);
  const rankRows = rank.rows.length >= RANK_MIN_ROWS ? rank.rows.map(r => `<tr>
      <td>${r.k === key ? `<b>${escapeHtml(r.lbl)}</b>` : `<a href="${regionPath(loc, r.k)}" style="color:#177A47;font-weight:600;">${escapeHtml(r.lbl)}</a>`}</td>
      <td>${escapeHtml(fmtEurL(loc, r.fm))}</td>
      <td class="mut">${r.k === key ? "—" : `${r.fm >= rec.fm ? "+" : "−"}${escapeHtml(fmtPctL(loc, Math.abs(r.fm / rec.fm - 1)))}`}</td>
      <td class="mut">${fmtNumL(loc, r.n)}</td></tr>`).join("") : "";
  const top = rank.rows[0], bot = rank.rows[rank.rows.length - 1];
  const rankLine = (rank.pos && rankRows)
    ? t(loc, "region.rank_line", {
        total: rank.total, label, pos: rank.pos,
        top_price: escapeHtml(fmtEurL(loc, top.fm)), top: escapeHtml(top.lbl),
        bot_price: escapeHtml(fmtEurL(loc, bot.fm)), bot: escapeHtml(bot.lbl),
        gap: bot.fm > 0 ? fmtPctL(loc, top.fm / bot.fm - 1) : "—",
      })
    : "";

  const body = crumbs([
    homeCrumb(loc), hubCrumb(loc),
    { name: t(loc, "region.crumb"), href: href(loc, "regions") },
    { name: rec.lbl },
  ]) + `
    <section class="section fc-wrap" style="padding-top:16px;">
      ${eyebrow(t(loc, "region.eyebrow", { label: label.toUpperCase(), source: loc.source.name.toUpperCase() }))}
      <h1 class="fc-h1">${t(loc, "region.h1", { region })}</h1>
      <p class="fc-p">${t(loc, "region.lede", {
        n: fmtNumL(loc, rec.n), region, price,
        km: rec.kmm != null ? t(loc, "region.lede_km", { km: escapeHtml(fmtKmL(loc, rec.kmm)) }) : "",
        vs,
      })}</p>
      ${statBlock([
        { k: t(loc, "region.cap_median"), v: price,
          s: `${escapeHtml(fmtEurL(loc, rec.fl))} – ${escapeHtml(fmtEurL(loc, rec.fh))}` },
        { k: t(loc, "region.cap_n"), v: fmtNumL(loc, rec.n), s: t(loc, "region.cap_active") },
        rec.kmm != null
          ? { k: t(loc, "region.cap_km"), v: escapeHtml(fmtKmL(loc, rec.kmm)), s: t(loc, "region.cap_here") }
          : null,
      ])}
      ${intlProvenance(loc, { n: rec.n, builtAt, measure: t(loc, "region.measure", { region: plainRegion }) })}
    </section>
    ${modelRows ? `<section class="section fc-wrap">
      <h2 class="fc-h2">${t(loc, "region.models_h", { region })}</h2>
      <p class="fc-p">${t(loc, "region.models_p")}</p>
      <div class="fc-scroll"><table class="fc-tbl">
        <thead><tr><th>${t(loc, "region.th_model")}</th><th>${t(loc, "region.th_n_here")}</th><th>${t(loc, "region.th_median_here")}</th><th>${t(loc, "region.th_median_nat")}</th><th>${t(loc, "region.th_diff")}</th></tr></thead>
        <tbody>${modelRows}</tbody></table></div>
    </section>` : `<section class="section fc-wrap">
      <p class="fc-p">${t(loc, "region.models_none", { n: fmtNumL(loc, rec.n), region })}</p>
    </section>`}
    ${rankRows ? `<section class="section fc-wrap">
      <h2 class="fc-h2">${t(loc, "region.rank_h", { label })}</h2>
      <p class="fc-p">${rankLine}</p>
      <div class="fc-scroll"><table class="fc-tbl">
        <thead><tr><th>${t(loc, "region.th_region")}</th><th>${t(loc, "region.th_median")}</th><th>${t(loc, "region.th_vs", { label })}</th><th>${t(loc, "region.th_n")}</th></tr></thead>
        <tbody>${rankRows}</tbody></table></div>
    </section>` : ""}
    <section class="section fc-wrap" style="padding-bottom:70px;">
      <p class="fc-p"><a href="${href(loc, "regions")}">${t(loc, "region.links_regions")}</a> · <a href="${href(loc, "hub")}">${t(loc, "region.links_hub")}</a> · <a href="${href(loc, "avaliar")}">${t(loc, "region.links_avaliar")}</a></p>
    </section>`;

  return layout({
    title: t(loc, "region.title", { region: plainRegion, price: fmtEurL(loc, rec.fm) }),
    description: t(loc, "region.desc", {
      region: plainRegion, price: fmtEurL(loc, rec.fm),
      n: fmtNumL(loc, rec.n), source: loc.source.name,
    }),
    body, zone: "all", nav: "precos", depositCount: null, index: true,
    host, locale: loc, canonical, altJson,
    jsonLd: graph([
      breadcrumbLd(host, [
        homeCrumb(loc), hubCrumb(loc),
        { name: t(loc, "region.crumb"), href: href(loc, "regions") },
        { name: rec.lbl, href: regionPath(loc, key) },
      ]),
      faqLd([[
        t(loc, "region.faq_q", { region: plainRegion }),
        t(loc, "region.faq_a", {
          n: fmtNumL(loc, rec.n), region: plainRegion, price: fmtEurL(loc, rec.fm),
          lo: fmtEurL(loc, rec.fl), hi: fmtEurL(loc, rec.fh),
        }),
      ]]),
      {
        "@type": "Dataset",
        "name": t(loc, "region.ds_name", { region: plainRegion }),
        "description": t(loc, "region.ds_desc", {
          n: rec.n, region: plainRegion, source: loc.source.name,
        }),
        "url": canonical,
        "inLanguage": loc.lang,
        "license": CC_LICENCE,
        "creator": { "@type": "Organization", "name": "Carsbuyer", "url": `https://${host}/` },
        "isAccessibleForFree": true,
        "variableMeasured": "asking price (EUR)",
        ...(builtAt ? { "dateModified": builtAt } : {}),
        "spatialCoverage": {
          "@type": "Place", "name": rec.lbl,
          "containedInPlace": { "@type": "Country", "name": loc.countryName },
        },
        "distribution": [{
          "@type": "DataDownload", "encodingFormat": "application/json", "contentUrl": altJson,
        }],
      },
      {
        "@type": "AggregateOffer", "priceCurrency": "EUR",
        "lowPrice": rec.fl, "highPrice": rec.fh, "offerCount": rec.n, "url": canonical,
        "areaServed": { "@type": "Place", "name": rec.lbl },
      },
    ]),
  });
}

const FACET_TAIL = /^\/([a-z0-9][a-z0-9-]*)\/([a-z][a-z0-9-]*)(\.json)?$/;
const REGION_TAIL = /^\/([a-z][a-z0-9-]*)(\.json)?$/;

function normaliseTail(rest) {
  let s = String(rest == null ? "" : rest);
  try {
    s = decodeURIComponent(s);
  } catch (_) {
    return null;
  }
  s = s.replace(/\/+$/, "").toLowerCase();
  return s;
}

function matchFacet(rest) {
  const s = normaliseTail(rest);
  if (s === null) return null;
  const m = FACET_TAIL.exec(s);
  return m ? { slug: m[1], key: m[2], json: !!m[3] } : null;
}

function matchRegion(rest) {
  const s = normaliseTail(rest);
  if (s === null) return null;
  if (s === "") return { hub: true, json: false };
  const m = REGION_TAIL.exec(s);
  return m ? { hub: false, key: m[1], json: !!m[2] } : null;
}

function matchRegionHubJson(rest) {
  const s = normaliseTail(rest);
  return s === "" ? { hub: true, json: true } : null;
}

async function handleFacet(ctx) {
  const { loc, url, models, builtAt, mdoc, params, helpers } = ctx;
  rememberRegions(loc, mdoc);
  const rec = models[params.slug];
  if (!rec) return helpers.notFoundIntl();
  if (!intlInWave(loc, models, params.slug, builtAt)) return helpers.notFoundIntl();
  const kind = facetKind(rec, params.key);
  if (!kind) return helpers.notFoundIntl();
  const cell = facetCell(rec, kind, params.key);
  if (!cell || !(cell.fm > 0) || !(cell.n > 0)) return helpers.notFoundIntl();
  const siblings = publishedCells(rec, kind);
  if (params.json) {
    return helpers.jsonResponse(intlFacetJson(loc, rec, params.slug, kind, cell, siblings, {
      host: url.host, builtAt,
    }));
  }
  return helpers.publicHtml(renderIntlFacetPage({
    loc, host: url.host, rec, slug: params.slug, kind, cell, siblings, builtAt,
  }));
}

async function handleRegion(ctx) {
  const { loc, url, models, builtAt, mdoc, stats, params, helpers } = ctx;
  rememberRegions(loc, mdoc);
  const districts = (mdoc && mdoc.districts) || null;
  if (!districts || !regionKeys(districts).length) return helpers.notFoundIntl();
  if (params.hub) {
    if (params.json) {
      return helpers.jsonResponse(intlRegionHubJson(loc, districts, { host: url.host, builtAt }));
    }
    return helpers.publicHtml(renderIntlRegionHub({
      loc, host: url.host, districts, stats, builtAt,
    }));
  }
  const rec = districts[params.key];
  if (!rec || !(rec.fm > 0) || !(rec.n > 0)) return helpers.notFoundIntl();
  if (params.json) {
    return helpers.jsonResponse(intlRegionJson(loc, params.key, rec, models, districts, {
      host: url.host, builtAt,
    }));
  }
  return helpers.publicHtml(renderIntlRegionPage({
    loc, host: url.host, key: params.key, rec, models, districts, stats, builtAt,
  }));
}

export const intlFacetsModule = registerIntlPages({
  id: "intl-facets",
  routes: [
    { routeKey: "model", match: matchFacet, handle: handleFacet },
    { routeKey: "regions", match: matchRegion, handle: handleRegion },
    { routeKey: "regionsJson", match: matchRegionHubJson, handle: handleRegion },
  ],
  navRouteKeys: ["regions"],
  navAvailable(loc, models, builtAt, mdoc) {
    const live = mdoc && mdoc.districts
      ? regionKeys(mdoc.districts)
      : rememberedRegions(loc, builtAt);
    return live.length ? ["regions"] : [];
  },
  sitemap(loc, models, builtAt, mdoc) {
    const out = [];
    for (const [slug, rec] of Object.entries(models || {})) {
      if (!intlInWave(loc, models, slug, builtAt)) continue;
      for (const key of intlFacetKeys(rec)) {
        out.push({ path: facetPath(loc, slug, key), freq: "daily", prio: "0.5" });
      }
    }
    const live = mdoc && mdoc.districts
      ? regionKeys(mdoc.districts)
      : rememberedRegions(loc, builtAt);
    if (live.length) {
      out.push({ path: href(loc, "regions"), freq: "weekly", prio: "0.6" });
      for (const key of live) {
        out.push({ path: regionPath(loc, key), freq: "weekly", prio: "0.55" });
      }
    }
    return out;
  },
});
