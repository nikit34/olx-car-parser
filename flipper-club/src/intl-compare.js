import { escapeHtml, layout } from "./templates.js";
import {
  crumbs, breadcrumbLd, faqLd, depreciationFit,
  comparePairs, comparePairKey, parseComparePath, comparePriceGap, modelClass,
} from "./seo-pages.js";
import {
  t, href, labelL, fmtEurL, fmtKmL, fmtNumL, fmtPctL,
  registerStrings, registerRoutes, registerNav,
} from "./i18n.js";
import { registerIntlPages } from "./index.js";
import { intlProvenance } from "./pages-intl.js";

const CMP_YEARS_SHOWN = 12;
const GAP_MIN_MODELS = 4;
const GAP_TABLE_ROWS = 25;
const DEP_MIN_CELLS = 5;
const DEP_MAX_RATE = 0.30;

const CLASS_KEY = new Map(Object.entries({
  "a": "cmp.class_a",
  "b": "cmp.class_b",
  "b-premium": "cmp.class_b_premium",
  "c": "cmp.class_c",
  "c-estate": "cmp.class_c_estate",
  "d": "cmp.class_d",
  "d-estate": "cmp.class_d_estate",
  "e": "cmp.class_e",
  "e-estate": "cmp.class_e_estate",
  "suv-b": "cmp.class_suv_b",
  "suv-c": "cmp.class_suv_c",
  "suv-d": "cmp.class_suv_d",
  "mpv": "cmp.class_mpv",
}));

const STRINGS_DE = {
  "footer.compare": "Modellvergleich",
  "cmp.hub_title": "Gebrauchtwagen vergleichen in {country}",
  "cmp.hub_desc": "{n} Modellvergleiche für Gebrauchtwagen in {country}: verlangter Preis beim gleichen Modelljahr, Kilometerstand, Preisstreuung und Wertverlust, gerechnet aus aktiven Angeboten auf {source}.",
  "cmp.hub_h1": "Gebrauchtwagen vergleichen in {country}",
  "cmp.hub_lede": "Jeder Vergleich rechnet mit den aktiven Angeboten beider Modelle auf {source}: verlangter Preis, Streuung, Kilometerstand und Wertverlust. Gegenüber stehen sich nur Modelle verschiedener Marken aus derselben Klasse, denn nur dort besteht die Wahl wirklich.",
  "cmp.hub_rule": "Verglichen wird <b>beim gleichen Modelljahr</b>, nicht über den Median aller Angebote. Ein Modell, dessen Angebote im Schnitt älter sind, sieht billiger aus, ohne es zu sein, und genau diesen Fehler macht fast jeder Vergleich.",
  "cmp.hub_crumb": "Vergleich",
  "cmp.hub_measure": "Verlangter Preis beider Modelle beim gleichen Modelljahr (Median)",
  "cmp.link_hub": "Alle Modelle",
  "cmp.link_method": "Wie wir rechnen",
  "cmp.link_gap": "Verlangter Preis gegen Schätzwert",
  "cmp.link_prices": "Preise {model}",
  "cmp.link_more": "Weitere Vergleiche",
  "cmp.class_a": "Kleinstwagen",
  "cmp.class_b": "Kleinwagen",
  "cmp.class_b_premium": "Kleinwagen Premium",
  "cmp.class_c": "Kompaktklasse",
  "cmp.class_c_estate": "Kompaktkombis",
  "cmp.class_d": "Mittelklasse",
  "cmp.class_d_estate": "Mittelklasse-Kombis",
  "cmp.class_e": "Obere Mittelklasse",
  "cmp.class_e_estate": "Kombis der oberen Mittelklasse",
  "cmp.class_suv_b": "Kleine SUV",
  "cmp.class_suv_c": "Kompakt-SUV",
  "cmp.class_suv_d": "Große SUV",
  "cmp.class_mpv": "Vans",
  "cmp.class_other": "Andere",
  "cmp.eyebrow": "VERGLEICH AUS AKTIVEN ANGEBOTEN · {source}",
  "cmp.title": "{a} oder {b}? Gebrauchtpreise im Vergleich",
  "cmp.desc": "{a} gegen {b} auf dem Gebrauchtwagenmarkt: verlangter Preis beim gleichen Modelljahr ({verdict}), Kilometerstand, Preisstreuung und Wertverlust, aus aktiven Angeboten auf {source}.",
  "cmp.h1": "{a} oder {b}: welcher Gebrauchtwagen ist der bessere Kauf?",
  "cmp.lede": "Ein Vergleich mit den Zahlen des heutigen Marktes: {na} aktive Angebote für den {a} und {nb} für den {b} auf {source}. Wir sagen dir nicht, welches das bessere Auto ist, sondern was jedes im Kauf, im Unterhalt und im Wiederverkauf kostet.",
  "cmp.pair_crumb": "{a} gegen {b}",
  "cmp.card_median": "Verlangter Preis (Median)",
  "cmp.card_sub": "{lo} – {hi} · {n} Angebote",
  "cmp.card_km": "Kilometerstand (Median)",
  "cmp.card_years": "Baujahre im Angebot",
  "cmp.card_fuel": "Kraftstoff",
  "cmp.card_cta": "{model} ansehen&nbsp;→",
  "cmp.mixed_ages": "Die beiden Mediane oben stammen aus Autos unterschiedlichen Alters: {a} gegen {b}. Der Preisvergleich dieser Seite wird weiter unten Jahr für Jahr gerechnet.",
  "cmp.verdicts_h": "Wer worin vorn liegt",
  "cmp.th_criterion": "Kriterium",
  "cmp.v_price": "Preis beim gleichen Modelljahr",
  "cmp.v_price_tie": "Beim gleichen Modelljahr wird praktisch dasselbe verlangt, über die {years} Jahre, in denen beide genug Angebote haben.",
  "cmp.v_price_gap": "Beim gleichen Modelljahr verlangt der {dearer} {pct} mehr als der {cheaper}, im Median über die {years} Jahre, in denen beide genug Angebote haben.",
  "cmp.v_dep": "Wertverlust",
  "cmp.v_dep_t": "Der {winner} hält den Wert besser: {lo} pro Jahr gegen {hi}. Über fünf Jahre wiegt dieser Unterschied mehr als der Nachlass beim Kauf.",
  "cmp.v_spread": "Preisstreuung",
  "cmp.v_spread_t": "Beim {winner} liegen die Preise enger beieinander ({lo} Streuung gegen {hi}). Wo die Streuung größer ist, entscheiden Zustand und Ausstattung mehr als das Modell, und es gibt dort bessere Geschäfte und schlechtere.",
  "cmp.v_km": "Kilometerstand im Angebot",
  "cmp.v_km_t": "Die angebotenen {winner} sind weniger gelaufen ({lo} gegen {hi} im Median). Der Preisvergleich oben ist beim gleichen Modelljahr, aber nicht beim gleichen Kilometerstand, und ein Teil des Rests ist genau das.",
  "cmp.table_h": "Preis nebeneinander, Jahr für Jahr",
  "cmp.table_lede_tie": "Dasselbe Zulassungsjahr auf beiden Seiten, denn nur so geht es im Unterschied um die Autos und nicht um das Alter dessen, was gerade verkauft wird. Über die {years} gemeinsamen Jahre stehen die Mediane gleichauf.",
  "cmp.table_lede_gap": "Dasselbe Zulassungsjahr auf beiden Seiten, denn nur so geht es im Unterschied um die Autos und nicht um das Alter dessen, was gerade verkauft wird. Über die {years} gemeinsamen Jahre verlangt der {dearer} <b>{pct} mehr</b>.",
  "cmp.th_year": "Jahr",
  "cmp.th_diff": "Unterschied",
  "cmp.th_listings": "Angebote",
  "cmp.cell_tie": "gleichauf",
  "cmp.cell_diff": "{name} +{pct}",
  "cmp.table_note": "Die letzte Spalte ist die Zahl der Angebote je Seite in diesem Jahr: ein Jahr mit 5 und 6 Angeboten sagt viel weniger als eines mit 40. Der Gesamtwert gewichtet jedes Jahr mit dieser Stichprobe, nicht mit seiner Reihenfolge.{tail}",
  "cmp.table_note_more": " Gezeigt sind die {shown} jüngsten der {total} Jahre, in denen beide eine Stichprobe haben; der Prozentwert nutzt alle.",
  "cmp.cta_h": "Schon ein konkretes Angebot im Blick?",
  "cmp.cta_p": "Der Median vergleicht Modelle. Ob genau dieses Auto gut im Preis liegt, sagt dir der Link-Check.",
  "cmp.cta_btn": "Ein Angebot bewerten&nbsp;&nbsp;→",
  "cmp.measure": "Verlangter Preis beider Modelle beim gleichen Modelljahr (Median)",
  "cmp.verdict_tie": "gleichauf",
  "cmp.verdict_pct": "{name} +{pct}",
  "cmp.faq1_q": "{a} oder {b}: was ist in {country} günstiger?",
  "cmp.faq1_a_tie": "Beim gleichen Modelljahr verlangen beide praktisch dasselbe, im Median über die {years} Jahre, in denen beide auf {source} genug Angebote haben.",
  "cmp.faq1_a_gap": "Beim gleichen Modelljahr verlangt der {dearer} rund {pct} mehr als der {cheaper}, im Median über die {years} Jahre mit Stichprobe auf beiden Seiten auf {source}. Über alle Angebote hinweg sieht der Abstand anders aus ({fa} gegen {fb}), weil die beiden Modelle nicht im gleichen Alter zum Verkauf stehen.",
  "cmp.faq2_q": "{a} oder {b}: welcher verliert weniger an Wert?",
  "cmp.faq2_a": "Der {winner} verliert rund {lo} pro Lebensjahr, der andere {hi}. Über fünf Jahre wiegt das meist schwerer als der Nachlass beim Kauf.",
  "cmp.faq3_q": "Welcher Preis ist besser einzuschätzen, {a} oder {b}?",
  "cmp.faq3_a": "Beim {winner} liegen die verlangten Preise enger beieinander ({lo} Streuung gegen {hi}). Bei größerer Streuung entscheidet das einzelne Angebot mehr als das Modell.",
  "gap.title": "Verlangter Preis gegen Schätzwert in {country}",
  "gap.desc": "Bei welchen Gebrauchtwagenmodellen in {country} über oder unter dem geschätzten Wert verlangt wird: der Median der Angebote gegen unseren Schätzwert, aus aktiven Angeboten auf {source}.",
  "gap.h1": "Wo der verlangte Preis vom Schätzwert abweicht",
  "gap.crumb": "Preis gegen Schätzwert",
  "gap.lede": "Für jedes Modell stellen wir das, was der Markt <b>verlangt</b>, dem gegenüber, was unser Modell für typischen Kilometerstand und typische Ausstattung <b>schätzt</b>. Eine große Abweichung heißt nicht, dass jemand betrügt: sie heißt, dass Angebot und Nachfrage bei diesem Modell gerade auseinanderlaufen, und genau dort wird verhandelt.",
  "gap.market_over": "Über den ganzen Markt liegt der verlangte Preis im Median <b>{pct}</b> über der Schätzung. Das ist der Normalfall und der Maßstab, an dem die Tabelle zu lesen ist.",
  "gap.market_under": "Über den ganzen Markt liegt der verlangte Preis im Median <b>{pct}</b> unter der Schätzung. Das ist der Normalfall und der Maßstab, an dem die Tabelle zu lesen ist.",
  "gap.scope": "Grundlage: {n} von {total} Modellen dieses Marktes, nämlich die, bei denen die Schätzung unsere Zuverlässigkeitsgrenzen besteht.",
  "gap.over_h": "Hier wird mehr verlangt, als wir schätzen",
  "gap.over_p": "Wenn du kaufst, geh mit dieser Zahl ins Gespräch. Wenn du verkaufst, steht der Markt gerade auf deiner Seite.",
  "gap.under_h": "Hier wird weniger verlangt, als wir schätzen",
  "gap.under_p": "Zu viel Angebot oder schwache Nachfrage. Guter Moment zum Kaufen, schlechter zum Inserieren.",
  "gap.none_over": "Gerade wird bei keinem Modell mit belastbarer Schätzung mehr verlangt, als wir schätzen.",
  "gap.none_under": "Gerade wird bei keinem Modell mit belastbarer Schätzung weniger verlangt, als wir schätzen.",
  "gap.th_model": "Modell",
  "gap.th_asking": "Verlangt (Median)",
  "gap.th_fair": "Geschätzter Wert",
  "gap.th_dev": "Abweichung",
  "gap.th_n": "Angebote",
  "gap.note": "Der Schätzwert wird nur dort veröffentlicht, wo er unsere Zuverlässigkeitsgrenzen besteht, siehe <a href=\"{method}\">Methodik</a>. Modelle, bei denen er das nicht tut, stehen hier nicht.",
  "gap.link_market": "Konkrete Angebote unter dem Schätzwert",
  "gap.link_hub": "Alle Modelle",
  "gap.link_compare": "Modelle vergleichen",
  "gap.measure": "Verlangter Preis im Median gegen den vom Modell geschätzten Wert",
  "gap.faq1_q": "Welche Gebrauchtwagen sind in {country} zu teuer?",
  "gap.faq1_a": "Wir nennen kein Modell zu teuer. Wir zeigen, wo der verlangte Median über dem Wert liegt, den unser Preismodell für dieses Modell schätzt, und wo er darunter liegt, gerechnet aus aktiven Angeboten auf {source}. Eine Abweichung nach oben ist ein Ansatz zum Verhandeln, kein Vorwurf.",
  "gap.faq2_q": "Wie entsteht der geschätzte Wert?",
  "gap.faq2_a": "Aus einem Preismodell, das Marke, Modell, Erstzulassung, Kilometerstand, Kraftstoff und Getriebe aller erfassten Angebote auswertet. Veröffentlicht wird er nur dort, wo die geschätzte Spanne eng genug ist; die Grenzen stehen in der Methodik.",
  "wid.title_meta": "{brand} {model}: was ist er wert",
  "wid.desc": "Was ein gebrauchter {brand} {model} in Angeboten auf {source} kostet: verlangter Preis im Median {price} aus {n} aktiven Angeboten, dazu der geschätzte Wert.",
  "wid.eyebrow": "● Unabhängige Bewertung",
  "wid.h": "Was ist ein gebrauchter {brand} {model} wert?",
  "wid.cap": "Verlangter Preis (Median) · {n} Angebote auf {source}",
  "wid.band": "Spanne {lo} – {hi}",
  "wid.fair_cap": "Geschätzter Wert",
  "wid.cta": "Vollständige Bewertung ansehen&nbsp;&nbsp;→",
  "wid.note": "Verlangte Preise aus aktiven Angeboten auf {source} – unverbindlicher Richtwert.",
};

const STRINGS_FR = {
  "footer.compare": "Comparatifs",
  "cmp.hub_title": "Comparer des voitures d'occasion en {country}",
  "cmp.hub_desc": "{n} comparatifs de voitures d'occasion en {country} : prix demandé au même millésime, kilométrage, dispersion des prix et décote, calculés sur les annonces actives de {source}.",
  "cmp.hub_h1": "Comparer des voitures d'occasion en {country}",
  "cmp.hub_lede": "Chaque comparatif part des annonces actives des deux modèles sur {source} : prix demandé, dispersion, kilométrage et décote. Nous ne mettons face à face que des modèles de marques différentes du même segment, parce que c'est là que le choix se pose vraiment.",
  "cmp.hub_rule": "Le prix est comparé <b>au même millésime</b>, pas sur la médiane de tout ce qui est en vente. Un modèle dont les annonces sont en moyenne plus anciennes paraît moins cher sans l'être, et c'est l'erreur que fait presque tout le monde.",
  "cmp.hub_crumb": "Comparatifs",
  "cmp.hub_measure": "Prix demandé des deux modèles au même millésime (médiane)",
  "cmp.link_hub": "Tous les modèles",
  "cmp.link_method": "Comment nous calculons",
  "cmp.link_gap": "Prix demandé face à l'estimation",
  "cmp.link_prices": "Prix {model}",
  "cmp.link_more": "Autres comparatifs",
  "cmp.class_a": "Citadines",
  "cmp.class_b": "Polyvalentes",
  "cmp.class_b_premium": "Polyvalentes premium",
  "cmp.class_c": "Compactes",
  "cmp.class_c_estate": "Breaks compacts",
  "cmp.class_d": "Berlines familiales",
  "cmp.class_d_estate": "Breaks familiaux",
  "cmp.class_e": "Grandes berlines",
  "cmp.class_e_estate": "Grands breaks",
  "cmp.class_suv_b": "Petits SUV",
  "cmp.class_suv_c": "SUV compacts",
  "cmp.class_suv_d": "Grands SUV",
  "cmp.class_mpv": "Monospaces",
  "cmp.class_other": "Autres",
  "cmp.eyebrow": "COMPARATIF À PARTIR D'ANNONCES ACTIVES · {source}",
  "cmp.title": "{a} ou {b} ? Comparatif des prix d'occasion",
  "cmp.desc": "{a} face à {b} sur le marché de l'occasion : prix demandé au même millésime ({verdict}), kilométrage, dispersion des prix et décote, à partir des annonces actives de {source}.",
  "cmp.h1": "{a} ou {b} : quel modèle acheter d'occasion ?",
  "cmp.lede": "Comparatif avec les chiffres du marché d'aujourd'hui : {na} annonces actives de {a} et {nb} de {b} sur {source}. Nous ne disons pas quelle est la meilleure voiture, nous disons ce que chacune coûte à l'achat, à l'usage et à la revente.",
  "cmp.pair_crumb": "{a} face à {b}",
  "cmp.card_median": "Prix demandé (médiane)",
  "cmp.card_sub": "{lo} – {hi} · {n} annonces",
  "cmp.card_km": "Kilométrage (médiane)",
  "cmp.card_years": "Années en vente",
  "cmp.card_fuel": "Carburant",
  "cmp.card_cta": "Voir {model}&nbsp;→",
  "cmp.mixed_ages": "Les deux médianes ci-dessus portent sur des voitures d'âges différents : {a} face à {b}. Le comparatif de prix de cette page se fait année par année, plus bas.",
  "cmp.verdicts_h": "Qui gagne sur quoi",
  "cmp.th_criterion": "Critère",
  "cmp.v_price": "Prix au même millésime",
  "cmp.v_price_tie": "Au même millésime, les deux demandent pratiquement la même chose, sur les {years} années où les deux ont assez d'annonces.",
  "cmp.v_price_gap": "Au même millésime, {dearer} demande {pct} de plus que {cheaper}, en médiane sur les {years} années où les deux ont assez d'annonces.",
  "cmp.v_dep": "Décote",
  "cmp.v_dep_t": "{winner} tient mieux sa valeur : {lo} par an contre {hi}. Sur cinq ans, cet écart pèse plus lourd que la remise obtenue à l'achat.",
  "cmp.v_spread": "Dispersion des prix",
  "cmp.v_spread_t": "Chez {winner} les prix sont plus resserrés ({lo} de dispersion contre {hi}). Là où la dispersion est large, l'état et la finition décident plus que le modèle : on y fait de meilleures affaires, et de pires.",
  "cmp.v_km": "Kilométrage en vente",
  "cmp.v_km_t": "Les {winner} en vente ont moins roulé ({lo} contre {hi} en médiane). Le comparatif de prix ci-dessus est au même millésime, pas au même kilométrage, et une partie de l'écart restant vient de là.",
  "cmp.table_h": "Prix côte à côte, année par année",
  "cmp.table_lede_tie": "La même année de mise en circulation des deux côtés : c'est la seule façon que l'écart parle des voitures et non de l'âge de ce qui est en vente. Sur les {years} années communes, les médianes sont à égalité.",
  "cmp.table_lede_gap": "La même année de mise en circulation des deux côtés : c'est la seule façon que l'écart parle des voitures et non de l'âge de ce qui est en vente. Sur les {years} années communes, {dearer} demande <b>{pct} de plus</b>.",
  "cmp.th_year": "Année",
  "cmp.th_diff": "Écart",
  "cmp.th_listings": "Annonces",
  "cmp.cell_tie": "à égalité",
  "cmp.cell_diff": "{name} +{pct}",
  "cmp.table_note": "La dernière colonne donne le nombre d'annonces de chaque côté pour cette année : un millésime à 5 et 6 annonces dit bien moins qu'un millésime à 40. Le pourcentage d'ensemble pondère chaque année par cet échantillon, pas par son rang.{tail}",
  "cmp.table_note_more": " Les {shown} millésimes les plus récents sur les {total} où les deux ont un échantillon sont affichés ; le pourcentage les utilise tous.",
  "cmp.cta_h": "Une annonce précise en tête ?",
  "cmp.cta_p": "La médiane compare des modèles. Pour savoir si cette voiture-là est au bon prix, colle le lien de l'annonce.",
  "cmp.cta_btn": "Estimer une annonce&nbsp;&nbsp;→",
  "cmp.measure": "Prix demandé des deux modèles au même millésime (médiane)",
  "cmp.verdict_tie": "égalité",
  "cmp.verdict_pct": "{name} +{pct}",
  "cmp.faq1_q": "{a} ou {b} : lequel des deux est le moins cher en {country} ?",
  "cmp.faq1_a_tie": "Au même millésime, les deux demandent pratiquement la même chose, en médiane sur les {years} années où les deux ont un échantillon suffisant sur {source}.",
  "cmp.faq1_a_gap": "Au même millésime, {dearer} demande environ {pct} de plus que {cheaper}, en médiane sur les {years} années où les deux ont un échantillon sur {source}. Sur la médiane de tout ce qui est en vente, l'écart paraît différent ({fa} contre {fb}), parce que les deux modèles ne sont pas en vente au même âge.",
  "cmp.faq2_q": "{a} ou {b} : lequel décote le moins ?",
  "cmp.faq2_a": "{winner} perd environ {lo} par année d'âge, contre {hi} pour l'autre. Sur cinq ans, cet écart pèse en général plus lourd que la remise obtenue à l'achat.",
  "cmp.faq3_q": "Quel prix est le plus prévisible, {a} ou {b} ?",
  "cmp.faq3_a": "Chez {winner} les prix demandés sont plus resserrés ({lo} de dispersion contre {hi}). Quand la dispersion est large, l'annonce elle-même décide plus que le modèle.",
  "gap.title": "Prix demandé face à l'estimation en {country}",
  "gap.desc": "Quels modèles d'occasion en {country} sont demandés au-dessus ou en dessous de la valeur estimée : la médiane des annonces face à notre estimation, à partir des annonces actives de {source}.",
  "gap.h1": "Là où le prix demandé s'écarte de l'estimation",
  "gap.crumb": "Prix face à l'estimation",
  "gap.lede": "Pour chaque modèle, nous mettons face à face ce que le marché <b>demande</b> et ce que notre modèle <b>estime</b>, pour un kilométrage et des finitions typiques. Un écart large ne veut pas dire que quelqu'un trompe quelqu'un : il veut dire que l'offre et la demande de ce modèle sont désalignées en ce moment, et c'est là que la négociation se joue.",
  "gap.market_over": "Sur l'ensemble du marché, le prix demandé est en médiane <b>{pct}</b> au-dessus de l'estimation. C'est la normale, et c'est la référence pour lire le tableau.",
  "gap.market_under": "Sur l'ensemble du marché, le prix demandé est en médiane <b>{pct}</b> en dessous de l'estimation. C'est la normale, et c'est la référence pour lire le tableau.",
  "gap.scope": "Base : {n} modèles sur les {total} de ce marché, ceux dont l'estimation passe nos seuils de fiabilité.",
  "gap.over_h": "On y demande plus que ce que nous estimons",
  "gap.over_p": "Si tu achètes, entre en négociation avec ce chiffre. Si tu vends, le marché est de ton côté.",
  "gap.under_h": "On y demande moins que ce que nous estimons",
  "gap.under_p": "Trop d'offre ou demande faible. Bon moment pour acheter, mauvais pour mettre en vente.",
  "gap.none_over": "En ce moment, aucun modèle à estimation fiable n'est demandé au-dessus de notre estimation.",
  "gap.none_under": "En ce moment, aucun modèle à estimation fiable n'est demandé en dessous de notre estimation.",
  "gap.th_model": "Modèle",
  "gap.th_asking": "Demandé (médiane)",
  "gap.th_fair": "Valeur estimée",
  "gap.th_dev": "Écart",
  "gap.th_n": "Annonces",
  "gap.note": "L'estimation n'est publiée que là où elle passe nos seuils de fiabilité, voir la <a href=\"{method}\">méthodologie</a>. Les modèles où elle ne passe pas ne figurent pas ici.",
  "gap.link_market": "Annonces concrètes sous la valeur estimée",
  "gap.link_hub": "Tous les modèles",
  "gap.link_compare": "Comparer des modèles",
  "gap.measure": "Prix demandé médian face à la valeur estimée par le modèle",
  "gap.faq1_q": "Quelles voitures d'occasion sont trop chères en {country} ?",
  "gap.faq1_a": "Nous ne qualifions aucun modèle de trop cher. Nous montrons où la médiane demandée passe au-dessus de la valeur que notre modèle de prix estime, et où elle passe en dessous, à partir des annonces actives de {source}. Un écart vers le haut est un point de négociation, pas une accusation.",
  "gap.faq2_q": "D'où vient la valeur estimée ?",
  "gap.faq2_a": "D'un modèle de prix qui lit la marque, le modèle, la première immatriculation, le kilométrage, le carburant et la boîte de toutes les annonces relevées. Elle n'est publiée que là où la fourchette estimée est assez serrée ; les seuils sont dans la méthodologie.",
  "wid.title_meta": "{brand} {model} : quelle est sa cote",
  "wid.desc": "Ce que coûte {brand} {model} d'occasion dans les annonces de {source} : prix demandé médian de {price} sur {n} annonces actives, et la valeur estimée en face.",
  "wid.eyebrow": "● Cote indépendante",
  "wid.h": "{brand} {model} d'occasion : combien ça vaut ?",
  "wid.cap": "Prix demandé (médiane) · {n} annonces sur {source}",
  "wid.band": "fourchette {lo} – {hi}",
  "wid.fair_cap": "Valeur estimée",
  "wid.cta": "Voir la cote complète&nbsp;&nbsp;→",
  "wid.note": "Prix demandés dans les annonces actives de {source} – estimation indicative, sans valeur contractuelle.",
};

const STRINGS_IT = {
  "footer.compare": "Confronti",
  "cmp.hub_title": "Confrontare auto usate in {country}",
  "cmp.hub_desc": "{n} confronti tra auto usate in {country}: prezzo richiesto a parità di anno di immatricolazione, chilometraggio, dispersione dei prezzi e svalutazione, dagli annunci attivi su {source}.",
  "cmp.hub_h1": "Confrontare auto usate in {country}",
  "cmp.hub_lede": "Ogni confronto parte dagli annunci attivi dei due modelli su {source}: prezzo richiesto, dispersione, chilometraggio e svalutazione. Mettiamo uno di fronte all'altro solo modelli di marche diverse dello stesso segmento, perché è lì che la scelta esiste davvero.",
  "cmp.hub_rule": "Il prezzo si confronta <b>a parità di anno</b>, non sulla mediana di tutto ciò che è in vendita. Un modello i cui annunci sono in media più vecchi sembra più economico senza esserlo, ed è l'errore che fanno quasi tutti.",
  "cmp.hub_crumb": "Confronti",
  "cmp.hub_measure": "Prezzo richiesto dei due modelli a parità di anno (mediana)",
  "cmp.link_hub": "Tutti i modelli",
  "cmp.link_method": "Come calcoliamo",
  "cmp.link_gap": "Prezzo richiesto e valore stimato",
  "cmp.link_prices": "Prezzi {model}",
  "cmp.link_more": "Altri confronti",
  "cmp.class_a": "Citycar",
  "cmp.class_b": "Utilitarie",
  "cmp.class_b_premium": "Utilitarie premium",
  "cmp.class_c": "Compatte",
  "cmp.class_c_estate": "Station wagon compatte",
  "cmp.class_d": "Berline medie",
  "cmp.class_d_estate": "Station wagon medie",
  "cmp.class_e": "Berline grandi",
  "cmp.class_e_estate": "Station wagon grandi",
  "cmp.class_suv_b": "SUV piccoli",
  "cmp.class_suv_c": "SUV compatti",
  "cmp.class_suv_d": "SUV grandi",
  "cmp.class_mpv": "Monovolumi",
  "cmp.class_other": "Altri",
  "cmp.eyebrow": "CONFRONTO DA ANNUNCI ATTIVI · {source}",
  "cmp.title": "{a} o {b}? Confronto dei prezzi dell'usato",
  "cmp.desc": "{a} contro {b} sul mercato dell'usato: prezzo richiesto a parità di anno ({verdict}), chilometraggio, dispersione dei prezzi e svalutazione, dagli annunci attivi su {source}.",
  "cmp.h1": "{a} o {b}: quale conviene comprare nell'usato?",
  "cmp.lede": "Confronto con i numeri del mercato di oggi: {na} annunci attivi di {a} e {nb} di {b} su {source}. Non diciamo quale sia l'auto migliore: diciamo quanto costa ciascuna da comprare, da tenere e da rivendere.",
  "cmp.pair_crumb": "{a} contro {b}",
  "cmp.card_median": "Prezzo richiesto (mediana)",
  "cmp.card_sub": "{lo} – {hi} · {n} annunci",
  "cmp.card_km": "Chilometraggio (mediana)",
  "cmp.card_years": "Anni in vendita",
  "cmp.card_fuel": "Alimentazione",
  "cmp.card_cta": "Vedi {model}&nbsp;→",
  "cmp.mixed_ages": "Le due mediane qui sopra vengono da auto di età diverse: {a} contro {b}. Il confronto di prezzo di questa pagina si fa anno per anno, più sotto.",
  "cmp.verdicts_h": "Chi vince su cosa",
  "cmp.th_criterion": "Criterio",
  "cmp.v_price": "Prezzo a parità di anno",
  "cmp.v_price_tie": "A parità di anno chiedono praticamente lo stesso, sui {years} anni in cui entrambe hanno abbastanza annunci.",
  "cmp.v_price_gap": "A parità di anno {dearer} chiede {pct} in più di {cheaper}, in mediana sui {years} anni in cui entrambe hanno abbastanza annunci.",
  "cmp.v_dep": "Svalutazione",
  "cmp.v_dep_t": "{winner} tiene meglio il valore: {lo} all'anno contro {hi}. Su cinque anni questa differenza pesa più dello sconto ottenuto all'acquisto.",
  "cmp.v_spread": "Dispersione dei prezzi",
  "cmp.v_spread_t": "Su {winner} i prezzi sono più concentrati ({lo} di dispersione contro {hi}). Dove la dispersione è ampia, stato e allestimento decidono più del modello: si fanno affari migliori, e peggiori.",
  "cmp.v_km": "Chilometraggio in vendita",
  "cmp.v_km_t": "Gli esemplari di {winner} in vendita hanno meno chilometri ({lo} contro {hi} in mediana). Il confronto di prezzo qui sopra è a parità di anno, ma non a parità di chilometri, e parte di ciò che resta è proprio questo.",
  "cmp.table_h": "Prezzo affiancato, anno per anno",
  "cmp.table_lede_tie": "Lo stesso anno di immatricolazione da entrambe le parti: è l'unico modo perché la differenza parli delle auto e non dell'età di ciò che è in vendita. Sui {years} anni in comune le mediane sono pari.",
  "cmp.table_lede_gap": "Lo stesso anno di immatricolazione da entrambe le parti: è l'unico modo perché la differenza parli delle auto e non dell'età di ciò che è in vendita. Sui {years} anni in comune {dearer} chiede <b>{pct} in più</b>.",
  "cmp.th_year": "Anno",
  "cmp.th_diff": "Differenza",
  "cmp.th_listings": "Annunci",
  "cmp.cell_tie": "pari",
  "cmp.cell_diff": "{name} +{pct}",
  "cmp.table_note": "L'ultima colonna è il numero di annunci per parte in quell'anno: un anno con 5 e 6 annunci dice molto meno di uno con 40. La percentuale complessiva pesa ogni anno con quel campione, non con la sua posizione.{tail}",
  "cmp.table_note_more": " Mostrati i {shown} anni più recenti dei {total} in cui entrambe hanno campione; la percentuale li usa tutti.",
  "cmp.cta_h": "Hai già un annuncio in mente?",
  "cmp.cta_p": "La mediana confronta modelli. Per sapere se quell'auto è al prezzo giusto, incolla il link dell'annuncio.",
  "cmp.cta_btn": "Valuta un annuncio&nbsp;&nbsp;→",
  "cmp.measure": "Prezzo richiesto dei due modelli a parità di anno (mediana)",
  "cmp.verdict_tie": "pari",
  "cmp.verdict_pct": "{name} +{pct}",
  "cmp.faq1_q": "{a} o {b}: quale costa meno in {country}?",
  "cmp.faq1_a_tie": "A parità di anno chiedono praticamente lo stesso, in mediana sui {years} anni in cui entrambe hanno campione sufficiente su {source}.",
  "cmp.faq1_a_gap": "A parità di anno {dearer} chiede circa {pct} in più di {cheaper}, in mediana sui {years} anni con campione da entrambe le parti su {source}. Sulla mediana di tutto ciò che è in vendita la distanza sembra un'altra ({fa} contro {fb}), perché i due modelli non sono in vendita alla stessa età.",
  "cmp.faq2_q": "{a} o {b}: quale si svaluta di meno?",
  "cmp.faq2_a": "{winner} perde circa {lo} per ogni anno di età, l'altra {hi}. Su cinque anni questa differenza pesa di solito più dello sconto ottenuto all'acquisto.",
  "cmp.faq3_q": "Quale prezzo è più prevedibile, {a} o {b}?",
  "cmp.faq3_a": "Su {winner} i prezzi richiesti sono più concentrati ({lo} di dispersione contro {hi}). Quando la dispersione è ampia, decide più il singolo annuncio che il modello.",
  "gap.title": "Prezzo richiesto e valore stimato in {country}",
  "gap.desc": "Quali modelli usati in {country} vengono chiesti sopra o sotto il valore stimato: la mediana degli annunci contro la nostra stima, dagli annunci attivi su {source}.",
  "gap.h1": "Dove il prezzo richiesto si allontana dalla stima",
  "gap.crumb": "Prezzo e stima",
  "gap.lede": "Per ogni modello mettiamo a confronto quello che il mercato <b>chiede</b> con quello che il nostro modello <b>stima</b>, per chilometraggio e allestimenti tipici. Uno scarto grande non significa che qualcuno stia imbrogliando: significa che domanda e offerta di quel modello sono disallineate in questo momento, ed è lì che si tratta.",
  "gap.market_over": "Su tutto il mercato il prezzo richiesto sta in mediana <b>{pct}</b> sopra la stima. È la normalità, ed è il riferimento con cui leggere la tabella.",
  "gap.market_under": "Su tutto il mercato il prezzo richiesto sta in mediana <b>{pct}</b> sotto la stima. È la normalità, ed è il riferimento con cui leggere la tabella.",
  "gap.scope": "Base: {n} modelli sui {total} di questo mercato, quelli la cui stima supera le nostre soglie di affidabilità.",
  "gap.over_h": "Qui si chiede più di quanto stimiamo",
  "gap.over_p": "Se compri, entra a trattare con questo numero. Se vendi, il mercato è dalla tua parte.",
  "gap.under_h": "Qui si chiede meno di quanto stimiamo",
  "gap.under_p": "Troppa offerta o domanda debole. Buon momento per comprare, cattivo per mettere in vendita.",
  "gap.none_over": "In questo momento nessun modello con stima affidabile viene chiesto sopra la nostra stima.",
  "gap.none_under": "In questo momento nessun modello con stima affidabile viene chiesto sotto la nostra stima.",
  "gap.th_model": "Modello",
  "gap.th_asking": "Richiesto (mediana)",
  "gap.th_fair": "Valore stimato",
  "gap.th_dev": "Scarto",
  "gap.th_n": "Annunci",
  "gap.note": "La stima viene pubblicata solo dove supera le nostre soglie di affidabilità, vedi la <a href=\"{method}\">metodologia</a>. I modelli in cui non le supera qui non compaiono.",
  "gap.link_market": "Annunci concreti sotto il valore stimato",
  "gap.link_hub": "Tutti i modelli",
  "gap.link_compare": "Confrontare modelli",
  "gap.measure": "Prezzo richiesto mediano contro il valore stimato dal modello",
  "gap.faq1_q": "Quali auto usate costano troppo in {country}?",
  "gap.faq1_a": "Non definiamo troppo caro nessun modello. Mostriamo dove la mediana richiesta passa sopra il valore che il nostro modello di prezzo stima, e dove passa sotto, calcolato dagli annunci attivi su {source}. Uno scarto verso l'alto è un margine di trattativa, non un'accusa.",
  "gap.faq2_q": "Da dove viene il valore stimato?",
  "gap.faq2_a": "Da un modello di prezzo che legge marca, modello, immatricolazione, chilometraggio, alimentazione e cambio di tutti gli annunci raccolti. Viene pubblicato solo dove la fascia stimata è abbastanza stretta; le soglie sono nella metodologia.",
  "wid.title_meta": "{brand} {model}: quanto vale",
  "wid.desc": "Quanto vale {brand} {model} negli annunci su {source}: prezzo richiesto mediano di {price} su {n} annunci attivi, con accanto il valore stimato.",
  "wid.eyebrow": "● Valutazione indipendente",
  "wid.h": "Quanto vale {brand} {model} nell'usato?",
  "wid.cap": "Prezzo richiesto (mediana) · {n} annunci su {source}",
  "wid.band": "fascia {lo} – {hi}",
  "wid.fair_cap": "Valore stimato",
  "wid.cta": "Vedi la valutazione completa&nbsp;&nbsp;→",
  "wid.note": "Prezzi richiesti negli annunci attivi su {source} – stima indicativa, non vincolante.",
};

const STRINGS_PT = {
  "footer.compare": "Comparar",
  "cmp.hub_title": "Comparar carros usados em {country}",
  "cmp.hub_desc": "{n} comparações de carros usados em {country}: preço pedido ao mesmo ano de modelo, quilometragem, dispersão de preços e desvalorização, a partir de anúncios ativos do {source}.",
  "cmp.hub_h1": "Comparar carros usados em {country}",
  "cmp.hub_lede": "Cada comparação usa os anúncios ativos dos dois modelos no {source}: preço pedido, dispersão, quilometragem e desvalorização. Só pomos frente a frente modelos de marcas diferentes do mesmo segmento, porque é entre esses que a escolha existe de facto.",
  "cmp.hub_rule": "O preço é comparado <b>ao mesmo ano de modelo</b>, não pela mediana de tudo o que está à venda. Um modelo cujos anúncios são em média mais velhos parece mais barato sem o ser, e essa é a comparação que toda a gente faz por engano.",
  "cmp.hub_crumb": "Comparar",
  "cmp.hub_measure": "Preço pedido dos dois modelos ao mesmo ano de modelo (mediana)",
  "cmp.link_hub": "Todos os modelos",
  "cmp.link_method": "Como calculamos",
  "cmp.link_gap": "Preço pedido vs. valor justo",
  "cmp.link_prices": "Preços {model}",
  "cmp.link_more": "Outras comparações",
  "cmp.class_a": "Citadinos",
  "cmp.class_b": "Utilitários",
  "cmp.class_b_premium": "Utilitários premium",
  "cmp.class_c": "Compactos",
  "cmp.class_c_estate": "Carrinhas compactas",
  "cmp.class_d": "Berlinas médias",
  "cmp.class_d_estate": "Carrinhas médias",
  "cmp.class_e": "Berlinas grandes",
  "cmp.class_e_estate": "Carrinhas grandes",
  "cmp.class_suv_b": "SUV pequenos",
  "cmp.class_suv_c": "SUV médios",
  "cmp.class_suv_d": "SUV grandes",
  "cmp.class_mpv": "Monovolumes",
  "cmp.class_other": "Outros",
  "cmp.eyebrow": "COMPARAÇÃO A PARTIR DE ANÚNCIOS ATIVOS · {source}",
  "cmp.title": "{a} ou {b}? Comparação de preços usados",
  "cmp.desc": "{a} contra {b} no mercado de usados: preço comparado ao mesmo ano de modelo ({verdict}), quilometragem, dispersão de preços e desvalorização, em anúncios ativos do {source}.",
  "cmp.h1": "{a} ou {b}: qual comprar usado?",
  "cmp.lede": "Comparação com números do mercado de hoje: {na} anúncios ativos de {a} e {nb} de {b} no {source}. Não dizemos qual é o melhor carro, dizemos o que cada um custa a comprar, a ter e a revender.",
  "cmp.pair_crumb": "{a} contra {b}",
  "cmp.card_median": "Preço pedido (mediana)",
  "cmp.card_sub": "{lo} – {hi} · {n} anúncios",
  "cmp.card_km": "Quilometragem (mediana)",
  "cmp.card_years": "Anos à venda",
  "cmp.card_fuel": "Combustível",
  "cmp.card_cta": "Ver {model}&nbsp;→",
  "cmp.mixed_ages": "As duas medianas acima são de carros de idades diferentes: {a} contra {b}. A comparação de preço desta página é feita ano a ano, mais abaixo.",
  "cmp.verdicts_h": "Quem ganha em quê",
  "cmp.th_criterion": "Critério",
  "cmp.v_price": "Preço ao mesmo ano",
  "cmp.v_price_tie": "Ao mesmo ano de modelo pedem praticamente o mesmo, nos {years} anos em que ambos têm amostra.",
  "cmp.v_price_gap": "Ao mesmo ano de modelo, {dearer} pede {pct} mais do que {cheaper}, na mediana dos {years} anos em que ambos têm anúncios suficientes.",
  "cmp.v_dep": "Desvalorização",
  "cmp.v_dep_t": "{winner} segura melhor o valor: {lo} por ano contra {hi}. Em cinco anos a diferença vale mais do que o desconto na compra.",
  "cmp.v_spread": "Dispersão de preços",
  "cmp.v_spread_t": "{winner} tem preços mais concentrados ({lo} de dispersão contra {hi}). Onde a dispersão é maior, o estado e a versão decidem mais do que o modelo, e dá para fazer melhores negócios e piores.",
  "cmp.v_km": "Quilometragem à venda",
  "cmp.v_km_t": "Os {winner} à venda estão menos rodados ({lo} contra {hi} medianos). A comparação de preço acima já é ao mesmo ano, mas não ao mesmo quilómetro, e parte do que sobra é isto.",
  "cmp.table_h": "Preço lado a lado, ano a ano",
  "cmp.table_lede_tie": "O mesmo ano de matrícula dos dois lados, que é a única forma de a diferença ser sobre os carros e não sobre a idade de quem está a vender. No conjunto dos {years} anos comuns, as medianas empatam.",
  "cmp.table_lede_gap": "O mesmo ano de matrícula dos dois lados, que é a única forma de a diferença ser sobre os carros e não sobre a idade de quem está a vender. No conjunto dos {years} anos comuns, {dearer} pede <b>{pct} mais</b>.",
  "cmp.th_year": "Ano",
  "cmp.th_diff": "Diferença",
  "cmp.th_listings": "Anúncios",
  "cmp.cell_tie": "iguais",
  "cmp.cell_diff": "{name} +{pct}",
  "cmp.table_note": "A última coluna é o número de anúncios de cada lado nesse ano: um ano com 5 e 6 anúncios diz muito menos do que um com 40. A percentagem do conjunto pesa cada ano por essa amostra, e não pela sua ordem.{tail}",
  "cmp.table_note_more": " Mostrados os {shown} anos mais recentes dos {total} em que ambos têm amostra; a percentagem usa todos.",
  "cmp.cta_h": "Já tens um anúncio em vista?",
  "cmp.cta_p": "A mediana compara modelos. Para saber se aquele carro está bem de preço, cola o link do anúncio.",
  "cmp.cta_btn": "Avaliar um anúncio&nbsp;&nbsp;→",
  "cmp.measure": "Preço pedido dos dois modelos ao mesmo ano de modelo (mediana)",
  "cmp.verdict_tie": "empate",
  "cmp.verdict_pct": "{name} +{pct}",
  "cmp.faq1_q": "{a} ou {b}: qual é mais barato em {country}?",
  "cmp.faq1_a_tie": "Ao mesmo ano de modelo os dois pedem praticamente o mesmo, na mediana dos {years} anos em que ambos têm amostra no {source}.",
  "cmp.faq1_a_gap": "Ao mesmo ano de modelo, {dearer} pede cerca de {pct} mais do que {cheaper}, na mediana dos {years} anos em que ambos têm amostra no {source}. Nas medianas de tudo o que está à venda a diferença parece outra ({fa} contra {fb}), porque os dois modelos não estão à venda com a mesma idade.",
  "cmp.faq2_q": "{a} ou {b}: qual perde menos valor?",
  "cmp.faq2_a": "{winner} desvaloriza cerca de {lo} por ano de idade, contra {hi} do outro. Sobre cinco anos, essa diferença costuma pesar mais do que o desconto na compra.",
  "cmp.faq3_q": "Qual preço é mais previsível, {a} ou {b}?",
  "cmp.faq3_a": "{winner} tem preços pedidos mais concentrados ({lo} de dispersão contra {hi}). Onde a dispersão é maior, o anúncio decide mais do que o modelo.",
  "gap.title": "Preço pedido vs. valor justo em {country}",
  "gap.desc": "Que modelos de carros usados em {country} são pedidos acima ou abaixo do valor justo estimado: a mediana dos anúncios contra a nossa estimativa, a partir de anúncios ativos do {source}.",
  "gap.h1": "Onde o preço pedido se afasta do valor justo",
  "gap.crumb": "Preço pedido vs. valor justo",
  "gap.lede": "Para cada modelo comparamos o que o mercado <b>pede</b> com o que o nosso modelo <b>estima</b> que vale, para quilometragem e versões típicas desse modelo. Um desvio grande não significa que alguém esteja a enganar ninguém: significa que a oferta e a procura desse modelo estão desalinhadas neste momento, e é aí que se negoceia.",
  "gap.market_over": "No conjunto do mercado, o preço pedido está <b>{pct}</b> acima da estimativa na mediana. É o normal, e é a referência contra a qual ler a tabela.",
  "gap.market_under": "No conjunto do mercado, o preço pedido está <b>{pct}</b> abaixo da estimativa na mediana. É o normal, e é a referência contra a qual ler a tabela.",
  "gap.scope": "Base: {n} de {total} modelos deste mercado, aqueles em que a estimativa passa os nossos limites de fiabilidade.",
  "gap.over_h": "Pedem mais do que estimamos",
  "gap.over_p": "Se vais comprar, entra a negociar. Se vais vender, o mercado está a teu favor.",
  "gap.under_h": "Pedem menos do que estimamos",
  "gap.under_p": "Oferta a mais ou procura fraca. Bom momento para comprar, mau para anunciar.",
  "gap.none_over": "Neste momento nenhum modelo com estimativa fiável é pedido acima da nossa estimativa.",
  "gap.none_under": "Neste momento nenhum modelo com estimativa fiável é pedido abaixo da nossa estimativa.",
  "gap.th_model": "Modelo",
  "gap.th_asking": "Pedido (mediana)",
  "gap.th_fair": "Valor justo estimado",
  "gap.th_dev": "Desvio",
  "gap.th_n": "Anúncios",
  "gap.note": "A estimativa só é publicada onde passa os nossos limites de fiabilidade, ver <a href=\"{method}\">metodologia</a>. Modelos onde não passa não aparecem aqui.",
  "gap.link_market": "Anúncios concretos abaixo do valor justo",
  "gap.link_hub": "Todos os modelos",
  "gap.link_compare": "Comparar modelos",
  "gap.measure": "Preço pedido mediano vs. valor justo estimado pelo modelo",
  "gap.faq1_q": "Que carros usados estão caros a mais em {country}?",
  "gap.faq1_a": "Não chamamos caro a mais a nenhum modelo. Mostramos onde a mediana pedida passa acima do valor que o nosso modelo de preço estima, e onde passa abaixo, a partir de anúncios ativos do {source}. Um desvio para cima é margem de negociação, não uma acusação.",
  "gap.faq2_q": "De onde vem o valor justo estimado?",
  "gap.faq2_a": "De um modelo de preço que lê marca, modelo, ano, quilometragem, combustível e caixa de todos os anúncios recolhidos. Só é publicado onde o intervalo estimado é suficientemente estreito; os limites estão na metodologia.",
  "wid.title_meta": "{brand} {model}: quanto vale",
  "wid.desc": "Quanto custa um {brand} {model} usado nos anúncios do {source}: preço pedido mediano de {price} em {n} anúncios ativos, e ao lado o valor justo estimado.",
  "wid.eyebrow": "● Avaliação independente",
  "wid.h": "Quanto vale um {brand} {model} usado?",
  "wid.cap": "Preço pedido (mediana) · {n} anúncios no {source}",
  "wid.band": "intervalo {lo} – {hi}",
  "wid.fair_cap": "Valor justo estimado",
  "wid.cta": "Ver avaliação completa&nbsp;&nbsp;→",
  "wid.note": "Preços pedidos em anúncios ativos do {source} – estimativa indicativa, não vinculativa.",
};

registerStrings("de", STRINGS_DE);
registerStrings("fr", STRINGS_FR);
registerStrings("it", STRINGS_IT);
registerStrings("pt", STRINGS_PT);

registerRoutes("de", { compare: "vergleich", compareJson: "vergleich.json", overvalued: "ueberteuert", overvaluedJson: "ueberteuert.json", widget: "widget" });
registerRoutes("fr", { compare: "comparer", compareJson: "comparer.json", overvalued: "surcotes", overvaluedJson: "surcotes.json", widget: "widget" });
registerRoutes("it", { compare: "confronta", compareJson: "confronta.json", overvalued: "sopravvalutate", overvaluedJson: "sopravvalutate.json", widget: "widget" });

registerNav([{ routeKey: "compare", labelKey: "footer.compare" }]);

function graph(nodes) {
  return { "@context": "https://schema.org", "@graph": nodes.filter(Boolean) };
}

function eyebrow(text) {
  return `<div class="eyebrow"><span class="e-dot"></span><span class="mono">${escapeHtml(text)}</span></div>`;
}

function homeCrumb(loc) {
  return { name: t(loc, "common.crumb_home"), href: href(loc, "landing") };
}

function stripTags(s) {
  return String(s).replace(/<[^>]+>/g, "").replace(/&nbsp;/g, " ");
}

function modelName(rec) {
  return `${rec.b} ${rec.m}`;
}

function day(builtAt) {
  return (builtAt || "").slice(0, 10);
}

function depRate(rec) {
  const fit = depreciationFit(rec);
  return (fit && fit.cells.length >= DEP_MIN_CELLS && fit.rate > 0 && fit.rate < DEP_MAX_RATE)
    ? fit.rate : null;
}

function spreadOf(rec) {
  return (rec.fm > 0 && rec.fl != null && rec.fh != null) ? (rec.fh - rec.fl) / rec.fm : null;
}

function ageLabel(rec) {
  const name = escapeHtml(modelName(rec));
  return (rec.y0 && rec.y1) ? `${name} ${rec.y0}–${rec.y1}` : name;
}

function classLabel(loc, slug) {
  const key = CLASS_KEY.get(modelClass(slug) || "");
  return t(loc, key || "cmp.class_other");
}

function comparePath(loc, a, b) {
  return href(loc, "compare", `${a}-vs-${b}`);
}

export function gapRows(models) {
  return Object.entries(models || {})
    .filter(([, r]) => r.gm > 0 && r.fm > 0)
    .map(([slug, r]) => ({
      slug, b: r.b, m: r.m, fm: r.fm, gm: r.gm, gl: r.gl, gh: r.gh,
      n: r.n || 0, gap: r.fm / r.gm - 1,
    }))
    .sort((x, y) => y.gap - x.gap);
}

function gapSplit(rows) {
  return {
    over: rows.filter(r => r.gap > 0).slice(0, GAP_TABLE_ROWS),
    under: rows.filter(r => r.gap < 0).reverse().slice(0, GAP_TABLE_ROWS),
  };
}

function pairVerdicts(loc, ra, rb, gap) {
  const A = escapeHtml(modelName(ra)), B = escapeHtml(modelName(rb));
  const out = [];
  const dearer = gap ? (gap.ratio > 1 ? "a" : "b") : null;
  const gPct = gap ? Math.max(gap.ratio, 1 / gap.ratio) - 1 : null;
  const tie = gPct != null && Math.round(gPct * 100) === 0;
  if (gap) {
    out.push({
      k: t(loc, "cmp.v_price"),
      w: tie ? null : (dearer === "a" ? "b" : "a"),
      text: tie
        ? t(loc, "cmp.v_price_tie", { years: gap.years })
        : t(loc, "cmp.v_price_gap", {
            dearer: dearer === "a" ? A : B, cheaper: dearer === "a" ? B : A,
            pct: fmtPctL(loc, gPct), years: gap.years,
          }),
    });
  }
  const depA = depRate(ra), depB = depRate(rb);
  if (depA != null && depB != null) {
    const w = depA <= depB ? "a" : "b";
    out.push({
      k: t(loc, "cmp.v_dep"), w,
      text: t(loc, "cmp.v_dep_t", {
        winner: w === "a" ? A : B,
        lo: fmtPctL(loc, Math.min(depA, depB)), hi: fmtPctL(loc, Math.max(depA, depB)),
      }),
    });
  }
  const sprA = spreadOf(ra), sprB = spreadOf(rb);
  if (sprA != null && sprB != null) {
    const w = sprA <= sprB ? "a" : "b";
    out.push({
      k: t(loc, "cmp.v_spread"), w,
      text: t(loc, "cmp.v_spread_t", {
        winner: w === "a" ? A : B,
        lo: fmtPctL(loc, Math.min(sprA, sprB)), hi: fmtPctL(loc, Math.max(sprA, sprB)),
      }),
    });
  }
  if (ra.kmm != null && rb.kmm != null) {
    const w = ra.kmm <= rb.kmm ? "a" : "b";
    out.push({
      k: t(loc, "cmp.v_km"), w,
      text: t(loc, "cmp.v_km_t", {
        winner: w === "a" ? A : B,
        lo: escapeHtml(fmtKmL(loc, Math.min(ra.kmm, rb.kmm))),
        hi: escapeHtml(fmtKmL(loc, Math.max(ra.kmm, rb.kmm))),
      }),
    });
  }
  return out;
}

function pairCard(loc, slug, rec, wins) {
  const fuel = (Array.isArray(rec.fu) && rec.fu.length)
    ? `${escapeHtml(labelL(loc, rec.fu[0][0]))} ${Math.round(rec.fu[0][1] * 100)}%` : null;
  const lines = [
    rec.kmm != null ? `${t(loc, "cmp.card_km")} · <b>${escapeHtml(fmtKmL(loc, rec.kmm))}</b>` : "",
    (rec.y0 && rec.y1) ? `${t(loc, "cmp.card_years")} · <b>${rec.y0}–${rec.y1}</b>` : "",
    fuel ? `${t(loc, "cmp.card_fuel")} · <b>${fuel}</b>` : "",
  ].filter(Boolean).join("<br>");
  return `
    <div class="fc-vs-card${wins ? " win" : ""}">
      <div class="mono" style="font-size:11px;color:#8A8F98;letter-spacing:.04em;">${escapeHtml(String(rec.b)).toUpperCase()}</div>
      <div style="font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:19px;margin:3px 0 12px;">${escapeHtml(rec.m)}</div>
      <div class="cap">${t(loc, "cmp.card_median")}</div>
      <div class="mono" style="font-weight:700;font-size:27px;letter-spacing:-.02em;">${escapeHtml(fmtEurL(loc, rec.fm))}</div>
      <div class="mono" style="font-size:12px;color:#5B606B;margin-top:2px;">${t(loc, "cmp.card_sub", {
        lo: escapeHtml(fmtEurL(loc, rec.fl)), hi: escapeHtml(fmtEurL(loc, rec.fh)), n: fmtNumL(loc, rec.n),
      })}</div>
      <div style="margin-top:12px;font-size:13.5px;line-height:1.8;color:#3A3F47;">${lines}</div>
      <a class="btn-dark" href="${href(loc, "model", slug)}" style="display:block;text-align:center;margin-top:14px;font-size:13.5px;padding:11px;">${t(loc, "cmp.card_cta", { model: escapeHtml(rec.m) })}</a>
    </div>`;
}

export function renderIntlComparePage({ loc, host, a, b, ra, rb, builtAt }) {
  const canonical = `https://${host}${comparePath(loc, a, b)}`;
  const nameA = modelName(ra), nameB = modelName(rb);
  const A = escapeHtml(nameA), B = escapeHtml(nameB);
  const gap = comparePriceGap(ra, rb);
  const gPct = gap ? Math.max(gap.ratio, 1 / gap.ratio) - 1 : null;
  const tie = gPct != null && Math.round(gPct * 100) === 0;
  const dearer = gap ? (gap.ratio > 1 ? "a" : "b") : null;
  const dearerName = dearer === "a" ? A : B;
  const dearerRaw = dearer === "a" ? nameA : nameB;
  const cheaperRaw = dearer === "a" ? nameB : nameA;
  const verdicts = pairVerdicts(loc, ra, rb, gap);
  const scoreA = verdicts.filter(v => v.w === "a").length;
  const scoreB = verdicts.filter(v => v.w === "b").length;

  const vrows = verdicts.map(v => `<tr>
      <td>${escapeHtml(v.k)}</td>
      <td class="nm">${v.w === "a" ? `<b>${escapeHtml(ra.m)}</b><span class="fc-win">+</span>` : escapeHtml(ra.m)}</td>
      <td class="nm">${v.w === "b" ? `<b>${escapeHtml(rb.m)}</b><span class="fc-win">+</span>` : escapeHtml(rb.m)}</td>
    </tr>`).join("");

  const shown = gap ? gap.cells.slice(0, CMP_YEARS_SHOWN) : [];
  const yrows = shown.map(c => {
    const r = c.fa / c.fb;
    const pct = Math.max(r, 1 / r) - 1;
    const flat = Math.round(pct * 100) === 0;
    return `<tr>
      <td class="mono">${c.y}</td>
      <td class="mono">${escapeHtml(fmtEurL(loc, c.fa))}</td>
      <td class="mono">${escapeHtml(fmtEurL(loc, c.fb))}</td>
      <td>${flat ? t(loc, "cmp.cell_tie") : t(loc, "cmp.cell_diff", {
        name: escapeHtml(r > 1 ? ra.m : rb.m), pct: escapeHtml(fmtPctL(loc, pct)),
      })}</td>
      <td class="mut mono">${fmtNumL(loc, c.na)}&nbsp;/&nbsp;${fmtNumL(loc, c.nb)}</td>
    </tr>`;
  }).join("");

  const tail = (gap && gap.cells.length > CMP_YEARS_SHOWN)
    ? t(loc, "cmp.table_note_more", { shown: CMP_YEARS_SHOWN, total: gap.cells.length })
    : "";

  const crumbName = t(loc, "cmp.pair_crumb", { a: nameA, b: nameB });
  const body = crumbs([
    homeCrumb(loc),
    { name: t(loc, "cmp.hub_crumb"), href: href(loc, "compare") },
    { name: crumbName },
  ]) + `
    <section class="section fc-wrap" style="padding-top:16px;">
      ${eyebrow(t(loc, "cmp.eyebrow", { source: loc.source.name }))}
      <h1 class="fc-h1">${t(loc, "cmp.h1", { a: A, b: B })}</h1>
      <p class="fc-p">${t(loc, "cmp.lede", {
        a: A, b: B, na: fmtNumL(loc, ra.n), nb: fmtNumL(loc, rb.n), source: loc.source.name,
      })}</p>
      <div class="fc-vs">
        ${pairCard(loc, a, ra, scoreA > scoreB)}
        ${pairCard(loc, b, rb, scoreB > scoreA)}
      </div>
      ${gap ? `<p class="fc-prov mono">${t(loc, "cmp.mixed_ages", { a: ageLabel(ra), b: ageLabel(rb) })}</p>` : ""}
      ${intlProvenance(loc, { n: (ra.n || 0) + (rb.n || 0), builtAt, measure: t(loc, "cmp.measure") })}
    </section>
    <section class="section fc-wrap">
      <h2 class="fc-h2">${t(loc, "cmp.verdicts_h")}</h2>
      <div class="fc-scroll"><table class="fc-tbl">
        <thead><tr><th>${t(loc, "cmp.th_criterion")}</th><th>${escapeHtml(ra.m)}</th><th>${escapeHtml(rb.m)}</th></tr></thead>
        <tbody>${vrows}</tbody></table></div>
      <ul class="fc-insights" style="margin-top:16px;">${verdicts.map(v => `<li>${v.text}</li>`).join("")}</ul>
    </section>
    ${gap ? `
    <section class="section fc-wrap">
      <h2 class="fc-h2">${t(loc, "cmp.table_h")}</h2>
      <p class="fc-p">${tie
        ? t(loc, "cmp.table_lede_tie", { years: gap.years })
        : t(loc, "cmp.table_lede_gap", { years: gap.years, dearer: dearerName, pct: escapeHtml(fmtPctL(loc, gPct)) })}</p>
      <div class="fc-scroll"><table class="fc-tbl">
        <thead><tr><th>${t(loc, "cmp.th_year")}</th><th>${escapeHtml(ra.m)}</th><th>${escapeHtml(rb.m)}</th><th>${t(loc, "cmp.th_diff")}</th><th>${t(loc, "cmp.th_listings")}</th></tr></thead>
        <tbody>${yrows}</tbody></table></div>
      <p class="fc-prov mono">${t(loc, "cmp.table_note", { tail })}</p>
    </section>` : ""}
    <section class="section fc-wide">
      <div class="cta-banner">
        <div style="flex:1 1 360px;">
          <h2>${t(loc, "cmp.cta_h")}</h2>
          <p>${t(loc, "cmp.cta_p")}</p>
        </div>
        <a class="btn-bright" href="${href(loc, "avaliar")}">${t(loc, "cmp.cta_btn")}</a>
      </div>
    </section>
    <section class="section fc-wrap" style="padding-bottom:70px;">
      <p class="fc-p"><a href="${href(loc, "model", a)}">${t(loc, "cmp.link_prices", { model: A })}</a> · <a href="${href(loc, "model", b)}">${t(loc, "cmp.link_prices", { model: B })}</a> · <a href="${href(loc, "compare")}">${t(loc, "cmp.link_more")}</a></p>
    </section>`;

  const verdict = tie
    ? t(loc, "cmp.verdict_tie")
    : t(loc, "cmp.verdict_pct", { name: dearerRaw, pct: fmtPctL(loc, gPct) });
  const faqs = [];
  if (gap) {
    faqs.push([
      t(loc, "cmp.faq1_q", { a: nameA, b: nameB, country: loc.countryName }),
      tie
        ? t(loc, "cmp.faq1_a_tie", { years: gap.years, source: loc.source.name })
        : t(loc, "cmp.faq1_a_gap", {
            dearer: dearerRaw, cheaper: cheaperRaw,
            pct: fmtPctL(loc, gPct), years: gap.years, source: loc.source.name,
            fa: fmtEurL(loc, ra.fm), fb: fmtEurL(loc, rb.fm),
          }),
    ]);
  }
  const depA = depRate(ra), depB = depRate(rb);
  if (depA != null && depB != null) {
    faqs.push([
      t(loc, "cmp.faq2_q", { a: nameA, b: nameB }),
      t(loc, "cmp.faq2_a", {
        winner: depA <= depB ? nameA : nameB,
        lo: fmtPctL(loc, Math.min(depA, depB)), hi: fmtPctL(loc, Math.max(depA, depB)),
      }),
    ]);
  }
  const sprA = spreadOf(ra), sprB = spreadOf(rb);
  if (sprA != null && sprB != null) {
    faqs.push([
      t(loc, "cmp.faq3_q", { a: nameA, b: nameB }),
      t(loc, "cmp.faq3_a", {
        winner: sprA <= sprB ? nameA : nameB,
        lo: fmtPctL(loc, Math.min(sprA, sprB)), hi: fmtPctL(loc, Math.max(sprA, sprB)),
      }),
    ]);
  }

  return layout({
    title: t(loc, "cmp.title", { a: nameA, b: nameB }),
    description: t(loc, "cmp.desc", { a: nameA, b: nameB, verdict, source: loc.source.name }),
    body, zone: "all", nav: "precos", depositCount: null, index: true, host, locale: loc,
    canonical, altJson: `${canonical}.json`,
    jsonLd: graph([
      breadcrumbLd(host, [
        homeCrumb(loc),
        { name: t(loc, "cmp.hub_crumb"), href: href(loc, "compare") },
        { name: crumbName, href: comparePath(loc, a, b) },
      ]),
      faqs.length ? faqLd(faqs.map(([q, ans]) => [stripTags(q), stripTags(ans)])) : null,
    ]),
  });
}

export function renderIntlCompareHub({ loc, host, pairs, models, builtAt, hasGap }) {
  const canonical = `https://${host}${href(loc, "compare")}`;
  const groups = new Map();
  for (const [a, b] of pairs) {
    const k = modelClass(a) || "other";
    if (!groups.has(k)) groups.set(k, []);
    groups.get(k).push([a, b]);
  }
  const order = [...CLASS_KEY.keys(), "other"];
  const items = order.filter(k => groups.has(k)).map(k => {
    const chips = groups.get(k).map(([a, b]) => {
      const ra = models[a], rb = models[b];
      return `<a class="mchip" href="${comparePath(loc, a, b)}">${escapeHtml(modelName(ra))} <span class="mut">vs</span> ${escapeHtml(modelName(rb))}</a>`;
    }).join("");
    const label = k === "other" ? t(loc, "cmp.class_other") : t(loc, CLASS_KEY.get(k));
    return `<h2 class="fc-h2" style="font-size:16px;margin:22px 0 10px;">${escapeHtml(label)}</h2><div class="mchips">${chips}</div>`;
  }).join("");
  const links = [
    `<a href="${href(loc, "hub")}">${t(loc, "cmp.link_hub")}</a>`,
    hasGap ? `<a href="${href(loc, "overvalued")}">${t(loc, "cmp.link_gap")}</a>` : "",
    `<a href="${href(loc, "metodologia")}">${t(loc, "cmp.link_method")}</a>`,
  ].filter(Boolean).join(" · ");
  const body = crumbs([homeCrumb(loc), { name: t(loc, "cmp.hub_crumb") }]) + `
    <section class="section fc-wrap" style="padding-top:16px;">
      ${eyebrow(t(loc, "cmp.eyebrow", { source: loc.source.name }))}
      <h1 class="fc-h1">${t(loc, "cmp.hub_h1", { country: loc.countryName })}</h1>
      <p class="fc-p">${t(loc, "cmp.hub_lede", { source: loc.source.name })}</p>
      <p class="fc-p">${t(loc, "cmp.hub_rule")}</p>
      ${items}
      ${intlProvenance(loc, { n: null, builtAt, measure: t(loc, "cmp.hub_measure") })}
      <p class="fc-p" style="margin-top:18px;">${links}</p>
    </section>
    <div style="height:60px;"></div>`;
  return layout({
    title: t(loc, "cmp.hub_title", { country: loc.countryName }),
    description: t(loc, "cmp.hub_desc", {
      n: fmtNumL(loc, pairs.length), country: loc.countryName, source: loc.source.name,
    }),
    body, zone: "all", nav: "precos", depositCount: null, index: true, host, locale: loc,
    canonical, altJson: `${canonical}.json`,
    jsonLd: graph([
      {
        "@type": "CollectionPage", "url": canonical, "inLanguage": loc.lang,
        "name": t(loc, "cmp.hub_h1", { country: loc.countryName }),
      },
      breadcrumbLd(host, [homeCrumb(loc), { name: t(loc, "cmp.hub_crumb"), href: href(loc, "compare") }]),
    ]),
  });
}

export function renderIntlValuationGap({ loc, host, rows, stats, models, builtAt, hasPairs }) {
  const canonical = `https://${host}${href(loc, "overvalued")}`;
  const { over, under } = gapSplit(rows);
  const row = r => `<tr>
      <td><a href="${href(loc, "model", r.slug)}" style="color:#177A47;font-weight:600;">${escapeHtml(modelName(r))}</a></td>
      <td>${escapeHtml(fmtEurL(loc, r.fm))}</td>
      <td class="mut">${escapeHtml(fmtEurL(loc, r.gm))}</td>
      <td>${r.gap > 0 ? "+" : "-"}${escapeHtml(fmtPctL(loc, Math.abs(r.gap)))}</td>
      <td class="mut">${fmtNumL(loc, r.n)}</td></tr>`;
  const table = list => `<div class="fc-scroll"><table class="fc-tbl">
        <thead><tr><th>${t(loc, "gap.th_model")}</th><th>${t(loc, "gap.th_asking")}</th><th>${t(loc, "gap.th_fair")}</th><th>${t(loc, "gap.th_dev")}</th><th>${t(loc, "gap.th_n")}</th></tr></thead>
        <tbody>${list.map(row).join("")}</tbody></table></div>`;
  const marketLine = stats && stats.gapMed != null
    ? `<p class="fc-p">${t(loc, stats.gapMed >= 0 ? "gap.market_over" : "gap.market_under", {
        pct: escapeHtml(fmtPctL(loc, Math.abs(stats.gapMed))),
      })}</p>`
    : "";
  const links = [
    `<a href="${href(loc, "mercado")}">${t(loc, "gap.link_market")}</a>`,
    `<a href="${href(loc, "hub")}">${t(loc, "gap.link_hub")}</a>`,
    hasPairs ? `<a href="${href(loc, "compare")}">${t(loc, "gap.link_compare")}</a>` : "",
  ].filter(Boolean).join(" · ");
  const body = crumbs([homeCrumb(loc), { name: t(loc, "gap.crumb") }]) + `
    <section class="section fc-wrap" style="padding-top:16px;">
      ${eyebrow(t(loc, "common.eyebrow", { source: loc.source.name }))}
      <h1 class="fc-h1">${t(loc, "gap.h1")}</h1>
      <p class="fc-p">${t(loc, "gap.lede")}</p>
      ${marketLine}
      <p class="fc-p mono" style="font-size:12px;">${t(loc, "gap.scope", {
        n: fmtNumL(loc, rows.length), total: fmtNumL(loc, Object.keys(models || {}).length),
      })}</p>
      <h2 class="fc-h2">${t(loc, "gap.over_h")}</h2>
      ${over.length ? `<p class="fc-p">${t(loc, "gap.over_p")}</p>${table(over)}` : `<p class="fc-p">${t(loc, "gap.none_over")}</p>`}
      <h2 class="fc-h2">${t(loc, "gap.under_h")}</h2>
      ${under.length ? `<p class="fc-p">${t(loc, "gap.under_p")}</p>${table(under)}` : `<p class="fc-p">${t(loc, "gap.none_under")}</p>`}
      ${intlProvenance(loc, { n: null, builtAt, measure: t(loc, "gap.measure") })}
      <p class="fc-p" style="margin-top:18px;">${t(loc, "gap.note", { method: href(loc, "metodologia") })}</p>
      <p class="fc-p">${links}</p>
    </section>
    <div style="height:60px;"></div>`;
  const faqs = [
    [t(loc, "gap.faq1_q", { country: loc.countryName }), t(loc, "gap.faq1_a", { source: loc.source.name })],
    [t(loc, "gap.faq2_q"), t(loc, "gap.faq2_a")],
  ];
  return layout({
    title: t(loc, "gap.title", { country: loc.countryName }),
    description: t(loc, "gap.desc", { country: loc.countryName, source: loc.source.name }),
    body, zone: "all", nav: "precos", depositCount: null, index: true, host, locale: loc,
    canonical, altJson: `${canonical}.json`,
    jsonLd: graph([
      {
        "@type": "CollectionPage", "url": canonical, "inLanguage": loc.lang,
        "name": t(loc, "gap.h1"),
      },
      breadcrumbLd(host, [homeCrumb(loc), { name: t(loc, "gap.crumb"), href: href(loc, "overvalued") }]),
      faqLd(faqs.map(([q, a]) => [stripTags(q), stripTags(a)])),
    ]),
  });
}

export function renderIntlModelWidget({ loc, host, rec, slug }) {
  const brand = escapeHtml(rec.b), model = escapeHtml(rec.m);
  const full = `https://${escapeHtml(host)}${href(loc, "model", slug)}`;
  const hasBand = rec.gm != null && rec.gl != null && rec.gh != null;
  const fairRow = hasBand ? `
    <div class="w-fair">
      <div class="w-cap">${t(loc, "wid.fair_cap")}</div>
      <div class="w-fair-v">${escapeHtml(fmtEurL(loc, rec.gm))}</div>
      <div class="w-band">${escapeHtml(fmtEurL(loc, rec.gl))} – ${escapeHtml(fmtEurL(loc, rec.gh))}</div>
    </div>` : "";
  const title = `${t(loc, "wid.title_meta", { brand: rec.b, model: rec.m })} · Carsbuyer`;
  const desc = t(loc, "wid.desc", {
    brand: rec.b, model: rec.m, source: loc.source.name,
    price: fmtEurL(loc, rec.fm), n: fmtNumL(loc, rec.n),
  });
  return `<!doctype html><html lang="${escapeHtml(loc.lang)}"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex,follow">
<meta name="description" content="${escapeHtml(desc)}">
<link rel="canonical" href="${full}">
<title>${escapeHtml(title)}</title>
<style>
*{box-sizing:border-box;margin:0}
body{font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;color:#16181D;background:#fff;padding:14px}
.w-card{border:1px solid #E8E6E1;border-radius:14px;padding:16px 18px;max-width:360px}
.w-eyebrow{font:600 10.5px/1 ui-monospace,monospace;letter-spacing:.08em;color:#177A47;text-transform:uppercase;margin-bottom:10px}
.w-h{font-weight:700;font-size:17px;letter-spacing:-.01em;margin-bottom:12px;line-height:1.3}
.w-cap{font-size:11px;color:#8A8F98}
.w-ask{font-weight:700;font-size:26px;letter-spacing:-.02em}
.w-band{font-size:12px;color:#5B606B}
.w-fair{margin-top:12px;padding-top:12px;border-top:1px solid #EFECE6}
.w-fair-v{font-weight:700;font-size:20px;color:#177A47;letter-spacing:-.01em}
.w-cta{display:block;margin-top:14px;text-align:center;background:#16181D;color:#fff;text-decoration:none;font-weight:600;font-size:13px;padding:11px;border-radius:9px}
.w-credit{margin-top:9px;text-align:center;font-size:11px;color:#9A9FA8}
.w-credit a{color:#177A47;text-decoration:none;font-weight:600}
.w-note{margin-top:10px;font:400 10.5px/1.5 ui-monospace,monospace;color:#9A9FA8}
</style></head>
<body>
<div class="w-card">
  <div class="w-eyebrow">${t(loc, "wid.eyebrow")}</div>
  <h1 class="w-h">${t(loc, "wid.h", { brand, model })}</h1>
  <div class="w-cap">${t(loc, "wid.cap", { n: fmtNumL(loc, rec.n), source: escapeHtml(loc.source.name) })}</div>
  <div class="w-ask">${escapeHtml(fmtEurL(loc, rec.fm))}</div>
  <div class="w-band">${t(loc, "wid.band", {
    lo: escapeHtml(fmtEurL(loc, rec.fl)), hi: escapeHtml(fmtEurL(loc, rec.fh)),
  })}</div>
  ${fairRow}
  <a class="w-cta" href="${full}" target="_blank" rel="noopener">${t(loc, "wid.cta")}</a>
  <div class="w-credit">via <a href="${full}" target="_blank" rel="noopener">Carsbuyer</a></div>
  <div class="w-note">${t(loc, "wid.note", { source: escapeHtml(loc.source.name) })}</div>
</div>
</body></html>`;
}

function modelBrief(loc, slug, rec, base) {
  return {
    slug, brand: rec.b, model: rec.m,
    sample_size: rec.n || 0,
    asking_price: { median: rec.fm, p25: rec.fl, p75: rec.fh },
    fair_value_estimate: rec.gm != null ? { median: rec.gm, low: rec.gl, high: rec.gh } : null,
    mileage_km_median: rec.kmm != null ? rec.kmm : null,
    model_years: (rec.y0 && rec.y1) ? { from: rec.y0, to: rec.y1 } : null,
    annual_depreciation: depRate(rec),
    price_spread: spreadOf(rec),
    page: `${base}${href(loc, "model", slug)}`,
  };
}

export function intlCompareJson(loc, a, b, ra, rb, { host, builtAt }) {
  const base = `https://${host}`;
  const gap = comparePriceGap(ra, rb);
  return {
    source: "Carsbuyer",
    source_url: `${base}${comparePath(loc, a, b)}`,
    licence: t(loc, "json.licence"),
    measured: "asking_price",
    measured_note: t(loc, "json.measured_note", { source: loc.source.name }),
    collected_until: day(builtAt) || null,
    updated_at: builtAt || null,
    market: loc.country, currency: "EUR", language: loc.code,
    comparison: {
      method: "same_model_year",
      common_years: gap ? gap.years : 0,
      compared_sample: gap ? gap.n : 0,
      dearer: gap ? (gap.ratio > 1 ? a : b) : null,
      premium: gap ? Math.max(gap.ratio, 1 / gap.ratio) - 1 : null,
      note: t(loc, "cmp.measure"),
    },
    models: [modelBrief(loc, a, ra, base), modelBrief(loc, b, rb, base)],
    by_year: gap ? gap.cells.map(c => ({
      year: c.y,
      asking_price_median: { [a]: c.fa, [b]: c.fb },
      sample_size: { [a]: c.na, [b]: c.nb },
      dearer: c.fa === c.fb ? null : (c.fa > c.fb ? a : b),
      premium: Math.max(c.fa / c.fb, c.fb / c.fa) - 1,
    })) : [],
    related: {
      comparisons_index: `${base}${href(loc, "compare")}`,
      models_index: `${base}${href(loc, "hub")}`,
      methodology: `${base}${href(loc, "metodologia")}`,
    },
  };
}

export function intlCompareHubJson(loc, pairs, models, { host, builtAt }) {
  const base = `https://${host}`;
  return {
    source: "Carsbuyer",
    source_url: `${base}${href(loc, "compare")}`,
    licence: t(loc, "json.licence"),
    measured: "asking_price",
    measured_note: t(loc, "json.measured_note", { source: loc.source.name }),
    collected_until: day(builtAt) || null,
    updated_at: builtAt || null,
    market: loc.country, currency: "EUR", language: loc.code,
    method: "same_model_year",
    comparisons: pairs.map(([a, b]) => ({
      pair: comparePairKey(a, b),
      segment: modelClass(a) || null,
      models: [
        { slug: a, brand: models[a].b, model: models[a].m },
        { slug: b, brand: models[b].b, model: models[b].m },
      ],
      page: `${base}${comparePath(loc, a, b)}`,
    })),
    related: {
      models_index: `${base}${href(loc, "hub")}`,
      methodology: `${base}${href(loc, "metodologia")}`,
    },
  };
}

export function intlValuationGapJson(loc, rows, stats, models, { host, builtAt }) {
  const base = `https://${host}`;
  const { over, under } = gapSplit(rows);
  const entry = r => ({
    slug: r.slug, brand: r.b, model: r.m,
    sample_size: r.n,
    asking_price_median: r.fm,
    fair_value_estimate: { median: r.gm, low: r.gl, high: r.gh },
    deviation: r.gap,
    page: `${base}${href(loc, "model", r.slug)}`,
  });
  return {
    source: "Carsbuyer",
    source_url: `${base}${href(loc, "overvalued")}`,
    licence: t(loc, "json.licence"),
    measured: "asking_price_vs_estimate",
    measured_note: t(loc, "gap.measure"),
    fair_value_note: t(loc, "json.gbm_note"),
    collected_until: day(builtAt) || null,
    updated_at: builtAt || null,
    market: loc.country, currency: "EUR", language: loc.code,
    models_with_estimate: rows.length,
    models_in_market: Object.keys(models || {}).length,
    market_median_deviation: (stats && stats.gapMed != null) ? stats.gapMed : null,
    asked_above_estimate: over.map(entry),
    asked_below_estimate: under.map(entry),
    related: {
      deals: `${base}${href(loc, "mercado")}`,
      models_index: `${base}${href(loc, "hub")}`,
      methodology: `${base}${href(loc, "metodologia")}`,
    },
  };
}

function normalise(rest) {
  let s;
  try {
    s = decodeURIComponent(rest);
  } catch (_) {
    return null;
  }
  return s.replace(/\/+$/, "");
}

function matchBare(rest) {
  const s = normalise(rest);
  return s === "" ? { kind: "bare" } : null;
}

function matchBareJson(rest) {
  const s = normalise(rest);
  return s === "" ? { kind: "bare", wantsJson: true } : null;
}

function matchCompare(rest) {
  const s = normalise(rest);
  if (s == null) return null;
  if (s === "") return { kind: "hub" };
  if (!s.startsWith("/")) return null;
  let tail = s.slice(1).toLowerCase();
  let wantsJson = false;
  if (tail.endsWith(".json")) {
    wantsJson = true;
    tail = tail.slice(0, -".json".length);
  }
  if (!/^[a-z0-9-]+$/.test(tail) || !tail.includes("-vs-")) return null;
  return { kind: "pair", tail, wantsJson };
}

function matchWidget(rest) {
  const s = normalise(rest);
  if (s == null || !s.startsWith("/")) return null;
  const slug = s.slice(1).toLowerCase();
  return /^[a-z0-9-]+$/.test(slug) ? { kind: "widget", slug } : null;
}

async function handleCompare(ctx) {
  const { loc, url, models, builtAt, params, helpers } = ctx;
  const pairs = comparePairs(models);
  if (!pairs.length) return helpers.notFoundIntl();
  if (params.kind === "pair") {
    const pairSet = new Set(pairs.map(([a, b]) => comparePairKey(a, b)));
    const pair = parseComparePath(params.tail, models, pairSet);
    if (!pair) return helpers.notFoundIntl();
    const ra = models[pair.a], rb = models[pair.b];
    if (params.wantsJson) {
      return helpers.jsonResponse(intlCompareJson(loc, pair.a, pair.b, ra, rb, {
        host: url.host, builtAt,
      }));
    }
    return helpers.publicHtml(renderIntlComparePage({
      loc, host: url.host, a: pair.a, b: pair.b, ra, rb, builtAt,
    }));
  }
  if (params.wantsJson) {
    return helpers.jsonResponse(intlCompareHubJson(loc, pairs, models, { host: url.host, builtAt }));
  }
  return helpers.publicHtml(renderIntlCompareHub({
    loc, host: url.host, pairs, models, builtAt,
    hasGap: gapRows(models).length >= GAP_MIN_MODELS,
  }));
}

async function handleGap(ctx) {
  const { loc, url, models, builtAt, stats, params, helpers } = ctx;
  const rows = gapRows(models);
  if (rows.length < GAP_MIN_MODELS) return helpers.notFoundIntl();
  if (params.wantsJson) {
    return helpers.jsonResponse(intlValuationGapJson(loc, rows, stats, models, {
      host: url.host, builtAt,
    }));
  }
  return helpers.publicHtml(renderIntlValuationGap({
    loc, host: url.host, rows, stats, models, builtAt,
    hasPairs: comparePairs(models).length > 0,
  }));
}

async function handleWidget(ctx) {
  const { loc, url, models, params, helpers } = ctx;
  const rec = models[params.slug];
  if (!rec) return helpers.notFoundIntl();
  return new Response(renderIntlModelWidget({ loc, host: url.host, rec, slug: params.slug }), {
    status: 200,
    headers: {
      "Content-Type": "text/html; charset=utf-8",
      "Cache-Control": "public, max-age=3600",
      "Content-Security-Policy": "frame-ancestors *",
    },
  });
}

export function intlCompareSitemap(loc, models) {
  const out = [];
  const pairs = comparePairs(models);
  if (pairs.length) {
    out.push({ path: href(loc, "compare"), freq: "weekly", prio: "0.6" });
    for (const [a, b] of pairs) {
      out.push({ path: comparePath(loc, a, b), freq: "weekly", prio: "0.5" });
    }
  }
  if (gapRows(models).length >= GAP_MIN_MODELS) {
    out.push({ path: href(loc, "overvalued"), freq: "daily", prio: "0.6" });
  }
  return out;
}

export const INTL_COMPARE_MODULE = registerIntlPages({
  id: "intl-compare",
  routes: [
    { routeKey: "compare", match: matchCompare, handle: handleCompare },
    { routeKey: "compareJson", match: matchBareJson, handle: handleCompare },
    { routeKey: "overvalued", match: matchBare, handle: handleGap },
    { routeKey: "overvaluedJson", match: matchBareJson, handle: handleGap },
    { routeKey: "widget", match: matchWidget, handle: handleWidget },
  ],
  navRouteKeys: ["compare"],
  navAvailable(loc, models) {
    return comparePairs(models).length ? ["compare"] : [];
  },
  sitemap: intlCompareSitemap,
});
