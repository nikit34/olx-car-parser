import { escapeHtml, layout } from "./templates.js";
import {
  crumbs, breadcrumbLd, faqLd, yearPageYears,
  depreciationFit, depreciationOk, depreciationSlugs, depreciationAge,
  DUELS, duel, duelSlugs, publishedDuel,
} from "./seo-pages.js";
import { intlProvenance } from "./pages-intl.js";
import {
  t, href, fmtEurL, fmtKmL, fmtNumL, fmtPctL,
  registerStrings, registerRoutes, registerNav,
} from "./i18n.js";
import { registerIntlPages } from "./index.js";

const CURVE_MIN_CELLS = 8;
const CURVE_MIN_SPAN = 8;
const BEND_FROM_AGE = 4;
const BEND_TO_AGE = 15;
const KINDS = ["fuel", "gear"];
const PAGE_ROUTE = { dep: "depreciation", fuel: "duel_fuel", gear: "duel_gear" };
const JSON_ROUTE = { dep: "depreciation_json", fuel: "duel_fuel_json", gear: "duel_gear_json" };
const GREEN = "#177A47";
const AMBER = "#B4661E";

const STRINGS_DE = {
  "footer.depreciation": "Wertverlust nach Modell",
  "dep.crumb": "Wertverlust",
  "dep.hub_eyebrow": "WERTVERLUSTKURVEN · {n} MODELLE",
  "dep.hub_title": "Wertverlust bei Gebrauchtwagen in {country}",
  "dep.hub_h1": "Welche Autos in {country} am schnellsten an Wert verlieren",
  "dep.hub_desc": "Welche Modelle in {country} pro Jahr am meisten Wert verlieren, gemessen an aktiven Angeboten auf {source}. {n} Modelle mit vollständiger Kurve, Restwert nach fünf Jahren und das Alter, ab dem die Erstzulassung den Preis nicht mehr bestimmt.",
  "dep.hub_lede": "Wertverlust je Jahr Fahrzeugalter, gemessen an den verlangten Preisen aktiver Angebote auf {source}. Aufgenommen wird nur, wessen Historie für eine belastbare Kurve reicht: mindestens {cells} Baujahre mit Stichprobe und {span} Jahre zwischen dem ältesten und dem jüngsten.",
  "dep.hub_market": "Der Median des Marktes liegt bei {mkt} pro Jahr. Darüber kostet dich das Auto im Besitz mehr, darunter verkaufst du es mit weniger Verlust weiter.",
  "dep.hub_band": "Zwischen {from} und {to} Jahren Alter verlieren diese Modelle im Median {a} pro Jahr ({na} Modelle mit Stichprobe in dieser Spanne), ab {cut} Jahren sind es {b} ({nb} Modelle).",
  "dep.hub_band_flat": "Der Prozentsatz bremst also nicht mit dem Alter. Was bremst, ist die Rechnung in Euro: {a} von 12.000 € und {a} von 3.000 € sind nicht derselbe Betrag.",
  "dep.hub_band_slower": "Der Prozentsatz bremst mit dem Alter, aber schwächer als es heißt: {gap} liegen zwischen den beiden Spannen.",
  "dep.hub_table_h": "Modell für Modell",
  "dep.th_model": "Modell",
  "dep.th_rate": "Pro Jahr",
  "dep.th_keep5": "Restwert nach 5 Jahren",
  "dep.th_half": "Hälfte des Werts",
  "dep.th_cheap": "Ein Jahr kostet unter {floor}",
  "dep.th_n": "Angebote",
  "dep.th_span": "Historie",
  "dep.hub_note": "„Hälfte des Werts“ ist die Zeit, die ein Modell im gemessenen Tempo braucht, bis der verlangte Medianpreis halbiert ist. „Ein Jahr kostet unter {floor}“ ist das Alter, ab dem ein weiteres Jahr Erstzulassung auf der Kurve weniger als {floor} wert ist; ein Gedankenstrich heißt, dass das vor {max_age} Jahren nicht eintritt, die Erstzulassung also über die ganze kaufbare Spanne den Preis bestimmt. Die Kurven laufen über Modellgenerationen hinweg: ein Teil des Rückgangs ist ein anderes Auto, nicht mehr Alter.",
  "dep.hub_duels": "Diese Tabelle misst das ganze Modell, alle Varianten zusammengefasst. Wo die Stichprobe reicht, trennen wir die Kurven:",
  "dep.hub_prov": "Verlangter Medianpreis nach Erstzulassung, log-lineare Anpassung",
  "dep.eyebrow": "WERTVERLUSTKURVE · {source}",
  "dep.title": "{brand} {model}: Wertverlust pro Jahr",
  "dep.desc": "Ein {brand} {model} verliert rund {rate} je Jahr Fahrzeugalter und hält nach fünf Jahren noch {keep_five} seines Werts, gemessen an {n} aktiven Angeboten auf {source}. Vollständige Kurve, Preis jedes Jahres Erstzulassung und die Stelle, an der der Rückgang nachlässt.",
  "dep.h1": "Wie schnell verliert ein {brand} {model} an Wert?",
  "dep.lede": "Gemessen an den verlangten Preisen von {n} aktiven Angeboten mit Erstzulassung zwischen {from} und {to} verliert ein {brand} {model} rund <b>{rate} je Jahr Fahrzeugalter</b>{half}. {vs}",
  "dep.half_clause": ", also die Hälfte seines Werts alle <b>{half} Jahre</b>",
  "dep.vs_fast": "Das ist schneller als der Markt (Median {mkt} pro Jahr).",
  "dep.vs_slow": "Das ist langsamer als der Markt (Median {mkt} pro Jahr), dieses Modell hält seinen Wert besser als der Durchschnitt.",
  "dep.vs_mid": "Das liegt auf dem Niveau des Marktes (Median {mkt} pro Jahr).",
  "dep.stat_year": "PRO JAHR",
  "dep.stat_year_s": "−{eur} je Jahr Fahrzeugalter",
  "dep.stat_age": "NACH {years} JAHREN",
  "dep.stat_left": "−{eur}",
  "dep.stat_left_base": "−{eur} von {base}",
  "dep.prov_measure": "Verlangter Medianpreis nach Erstzulassung, {from}–{to}",
  "dep.prov_extra": "Log-lineare Anpassung, R²={rsq}; Eurobeträge bezogen auf {base}, den Kurvenwert für ein Fahrzeug aus {year}",
  "dep.curve_h": "Die Kurve",
  "dep.curve_p": "Jeder Punkt ist der Medianpreis der Angebote in diesem Alter, die Linie ist die log-lineare Anpassung, aus der der Prozentsatz oben stammt{marks}. Der Rückgang ist ein Anteil des Restwerts: in den ersten Jahren sind das viele Euro, am Ende wenige, auch wenn der Prozentsatz derselbe bleibt.",
  "dep.curve_marks": ", und die senkrechten Markierungen sind die Altersangaben, über die diese Seite eine Aussage trifft",
  "dep.curve_caveat": "Ein Vorbehalt, den die Kurve nicht zeigt: zwischen {from} und {to} hat der {brand} {model} mehr als einen Generationswechsel hinter sich, und ein Fahrzeug von jedem Ende der Reihe ist nicht dasselbe Auto mit ein paar Jahren mehr. Ein Teil des Rückgangs ist Alter und Verschleiß, ein Teil schlicht ein anderes Modell mit anderer Ausstattung. Die Reihe misst, was der Markt für jedes Baujahr verlangt, nicht das Altern eines einzelnen Autos.",
  "dep.ladder_h": "Was ein Jahr Erstzulassung kostet",
  "dep.ladder_p": "Dieselbe Kurve rückwärts gelesen: was du in Euro dafür zahlst, dass die Erstzulassung ein Jahr jünger ist. An dieser Spalte entscheidet sich, ob ein Aufschlag für das nächstjüngere Baujahr lohnt.",
  "dep.ladder_th_age": "Alter (Erstzulassung)",
  "dep.ladder_th_cost": "Kosten für +1 Jahr",
  "dep.ladder_th_price": "Preis auf der Kurve",
  "dep.ladder_note": "Werte aus der angepassten Kurve, keine rohen Mediane: zwischen zwei benachbarten Baujahren springt der Median stärker als der Schritt, den wir hier messen. Die rohen Mediane stehen weiter unten in der Tabelle nach Baujahr.",
  "dep.age_n": "{n} Jahre",
  "dep.age_one": "1 Jahr",
  "dep.bend_h": "Gibt es einen Knick?",
  "dep.bend_slows": "<b>Ja, bei {age} Jahren.</b> Bis dahin verliert der {brand} {model} rund <b>{early} pro Jahr</b>, danach <b>{late}</b>. Ein Exemplar jenseits dieses Knicks kostet dich für jedes Jahr, das du es fährst, weniger. Das ist die Strecke, auf der der Kauf günstig wird.",
  "dep.bend_speeds": "<b>Bei {age} Jahren gibt es einen Knick, aber andersherum als erwartet:</b> davor verliert der {brand} {model} rund <b>{early} pro Jahr</b>, danach <b>{late}</b>. Der Prozentsatz zieht mit dem Alter an, statt nachzulassen. Bei den ältesten Fahrzeugen sieht die Rechnung trotzdem anders aus: der Preis ist längst niedrig, also schrumpft der Verlust in Euro weiter.",
  "dep.bend_none": "<b>Nein, nicht in Prozent.</b> Wir haben eine Anpassung mit Knick bei jedem Alter zwischen {min_age} und {max_age} Jahren geprüft, und keine erklärt die Preise des {brand} {model} besser als ein gleichbleibender Rückgang von {rate} pro Jahr (R²={rsq}).",
  "dep.bend_none_cand": " Der beste Kandidat lag bei {age} Jahren, {early} pro Jahr davor und {late} danach, und dieser Abstand hat die Größe der Sprünge, die der Median zwischen zwei benachbarten Baujahren ohnehin macht.",
  "dep.bend_none_tail": " Was nachlässt, ist die Rechnung in Euro, nicht der Prozentsatz.",
  "dep.cheap_yes": "Ab wann das nicht mehr wehtut, lässt sich datieren: ab <b>{age} Jahren</b>, Erstzulassung {year} und älter, rund {price}, kostet jedes weitere Jahr Alter weniger als {floor}. Das ist weniger als ein Satz Reifen oder ein Zahnriemenwechsel. Ab hier wiegt der Zustand des Autos schwerer als das Baujahr, und daran entscheidet sich der Kauf.",
  "dep.cheap_no": "So weit kommt es hier nicht, jedenfalls nicht in der Spanne, die jemand sucht: mit {age} Jahren ist ein Jahr Alter immer noch rund {cost} wert, und erst viel später fiele es unter {floor}. Bei diesem Modell bestimmt die Erstzulassung den Preis über die ganze kaufbare Spanne, ein jüngeres Baujahr kostet weiter richtig Geld.",
  "dep.table_h": "Medianpreis nach Erstzulassung",
  "dep.th_year": "Baujahr",
  "dep.th_median": "Median (verlangt)",
  "dep.th_listings": "Angebote",
  "dep.th_gap": "Jahre vor {year}",
  "dep.th_vs": "gegenüber {year}",
  "dep.th_km": "Kilometerstand (Median)",
  "dep.cta_h": "Was ist DEIN {brand} {model} heute wert?",
  "dep.cta_p": "Diese Kurve gilt für das Modell. Füg den Link deines Inserats ein und wir sagen dir den fairen Wert deines konkreten Autos, mit deinem Kilometerstand und deiner Ausstattung.",
  "dep.cta_btn": "Mein Auto bewerten&nbsp;&nbsp;→",
  "dep.links_model": "{brand} {model} nach Baujahr",
  "dep.links_others": "Wertverlust anderer Modelle",
  "dep.faq1_q": "Wie viel verliert ein {brand} {model} pro Jahr an Wert?",
  "dep.faq1_a": "Rund {rate} des Restwerts je Jahr Fahrzeugalter, gemessen an den verlangten Preisen von {n} aktiven Angeboten eines {brand} {model} mit Erstzulassung zwischen {from} und {to} auf {source} in {country}. Der Prozentsatz bleibt gleich, in Euro ist der Verlust in den ersten Jahren aber deutlich größer.",
  "dep.faq2_q": "Was ist ein {brand} {model} nach fünf Jahren noch wert?",
  "dep.faq2_a": "Im gemessenen Tempo hält ein {brand} {model} nach fünf Jahren noch rund {keep} seines Werts. Bezogen auf die {base}, die die Kurve einem Fahrzeug aus {year} gibt, sind das rund {eur} weniger.",
  "dep.faq3_q": "In wie vielen Jahren verliert ein {brand} {model} die Hälfte seines Werts?",
  "dep.faq3_a": "In rund {half} Jahren, bei den gemessenen {rate} pro Jahr aus den aktiven Angeboten auf {source}. Das ist dieselbe Rate anders ausgedrückt: alle {half} Jahre Fahrzeugalter halbiert sich der verlangte Medianpreis.",
  "dep.faq4_q": "Ab welchem Alter verliert ein {brand} {model} kaum noch an Wert?",
  "dep.faq4_bend": "Der Rückgang lässt bei {age} Jahren nach: bis dahin rund {early} pro Jahr, danach {late}.",
  "dep.faq4_flat": "Er hört nie auf zu verlieren, es tut nur irgendwann nicht mehr weh. In Prozent bleibt der Rückgang über die ganze gemessene Reihe bei rund {rate} pro Jahr, es gibt kein Alter, ab dem der Prozentsatz bremst.",
  "dep.faq4_cheap": " In Euro sieht es anders aus: ab {age} Jahren, Erstzulassung {year} und älter, kostet jedes Jahr Alter weniger als {floor}, und dann wiegt der Zustand des Autos schwerer als das Baujahr.",
  "dep.faq4_nocheap": " In Euro schrumpft der Verlust, aber langsam: mit {age} Jahren kostet ein Jahr Alter immer noch rund {cost}.",
  "dep.faq5_q": "Verliert ein {brand} {model} mehr an Wert als der Durchschnitt?",
  "dep.faq5_a": "Der Median des von uns gemessenen Gebrauchtwagenmarktes liegt bei {mkt} je Jahr Fahrzeugalter. Der {brand} {model} liegt bei {rate}. {verdict}",
  "dep.above": "Das ist mehr als der Durchschnitt.",
  "dep.below": "Das ist weniger als der Durchschnitt.",
  "dep.inline": "Das ist genau der Durchschnitt.",
  "dep.ds_name": "Wertverlust von {brand} {model} in {country}",
  "dep.ds_desc": "Verlangter Medianpreis eines {brand} {model} nach Erstzulassung ({from}–{to}), jährlicher Wertverlust von {rate} und der Eurobetrag, den jedes Jahr Fahrzeugalter kostet, aus {n} aktiven Angeboten auf {source}.",
  "dep.var_price": "Verlangter Preis (EUR)",
  "dep.var_rate": "Jährlicher Wertverlust (%)",
  "dep.var_cost": "Kosten eines Jahres Fahrzeugalter (EUR)",
  "dep.json_method": "log(Preis) ~ Baujahr, kleinste Quadrate über die Medianpreise aller Baujahre mit mindestens fünf Angeboten",
  "dep.json_ladder_note": "Werte aus der angepassten Kurve, keine rohen Mediane.",
  "dep.json_bend_note": "Eine Anpassung mit Knick wurde für jedes Alter zwischen {min_age} und {max_age} Jahren geprüft und nur veröffentlicht, wenn sie die gleichbleibende Rate deutlich schlägt.",
  "dep.json_hub_note": "Aufgenommen wird ein Modell mit mindestens {cells} Baujahren mit Stichprobe, {span} Jahren Historie und einer Anpassung, die die Punkte tatsächlich erklärt.",
  "dep.chart_alt": "Verlangter Medianpreis nach Fahrzeugalter mit angepasster Kurve",
  "dep.chart_axis": "Jahre Alter",
  "dep.chart_bend": "Knick bei {age} Jahren",
  "dep.chart_cheap": "{floor}/Jahr ab {age}",
  "dep.chart_tip": "{age} Jahre ({year}): {price} · {n} Angebote",
  "duel.fuel_crumb": "Diesel oder Benziner",
  "duel.fuel_eyebrow": "DIESEL VS. BENZINER",
  "duel.fuel_question": "Diesel oder Benziner",
  "duel.fuel_hub_h1": "Diesel oder Benziner: wo die Wahl den Preis wirklich verändert",
  "duel.fuel_hub_title": "Diesel oder Benziner: was hält den Preis besser, Modell für Modell",
  "duel.fuel_choice": "die Wahl des Kraftstoffs",
  "duel.fuel_facet": "Kraftstoff",
  "duel.fuel_facet_of": "vom Kraftstoff",
  "duel.fuel_mixup": "dass die angebotenen Diesel deutlich mehr gelaufen sind",
  "duel.fuel_a": "Diesel",
  "duel.fuel_b": "Benziner",
  "duel.fuel_a_low": "Diesel",
  "duel.fuel_b_low": "Benziner",
  "duel.fuel_a_subj": "der Diesel",
  "duel.fuel_b_subj": "der Benziner",
  "duel.fuel_a_scap": "Der Diesel",
  "duel.fuel_b_scap": "Der Benziner",
  "duel.fuel_a_fav": "zugunsten des Diesels",
  "duel.fuel_b_fav": "zugunsten des Benziners",
  "duel.fuel_a_chip": "DIESEL",
  "duel.fuel_b_chip": "BENZINER",
  "duel.fuel_b_than": "als der Benziner",
  "duel.gear_crumb": "Schaltung oder Automatik",
  "duel.gear_eyebrow": "HANDSCHALTUNG VS. AUTOMATIK",
  "duel.gear_question": "Handschaltung oder Automatik",
  "duel.gear_hub_h1": "Handschaltung oder Automatik: wo das Getriebe den Preis wirklich verändert",
  "duel.gear_hub_title": "Handschaltung oder Automatik: was hält den Preis besser, Modell für Modell",
  "duel.gear_choice": "die Wahl des Getriebes",
  "duel.gear_facet": "Getriebe",
  "duel.gear_facet_of": "vom Getriebe",
  "duel.gear_mixup": "dass die angebotenen Automatikwagen jünger sind und weniger gelaufen haben",
  "duel.gear_a": "Handschaltung",
  "duel.gear_b": "Automatik",
  "duel.gear_a_low": "mit Handschaltung",
  "duel.gear_b_low": "mit Automatik",
  "duel.gear_a_subj": "die Handschaltung",
  "duel.gear_b_subj": "die Automatik",
  "duel.gear_a_scap": "Die Handschaltung",
  "duel.gear_b_scap": "Die Automatik",
  "duel.gear_a_fav": "zugunsten der Handschaltung",
  "duel.gear_b_fav": "zugunsten der Automatik",
  "duel.gear_a_chip": "HANDSCHALTUNG",
  "duel.gear_b_chip": "AUTOMATIK",
  "duel.gear_b_than": "als die Automatik",
  "duel.title": "{brand} {model}: {question} – was hält den Preis besser?",
  "duel.h1": "{brand} {model}: {question} – was hält den Preis besser?",
  "duel.desc": "Bei einem {brand} {model} verliert {a_subj} {a_pct} je Jahr Fahrzeugalter, {b_subj} {b_pct}, gemessen an {n} aktiven Angeboten auf {source} bei angeglichenem Kilometerstand. {tail}",
  "duel.desc_win": "Der Abstand: {pp} pro Jahr {fav}.",
  "duel.desc_draw": "Der Abstand ist von der Messgenauigkeit nicht zu trennen.",
  "duel.lede": "Wir haben beide Kurven getrennt über <b>{n} aktive Angebote</b> eines {brand} {model} angepasst ({an} {a_low}, {bn} {b_low}, Erstzulassung {from} bis {to}), mit angeglichenem Kilometerstand. {a_scap} verliert <b>{a_pct} je Jahr Fahrzeugalter</b>, {b_subj} <b>{b_pct}</b>.",
  "duel.stat_diff": "ABSTAND",
  "duel.stat_side": "pro Jahr · {n} Angebote",
  "duel.stat_ci": "±{ci} · {verdict}",
  "duel.draw_word": "nicht unterscheidbar",
  "duel.prov_measure": "Verlangter Preis eines {brand} {model}, {a_low} und {b_low} getrennt ({from}–{to})",
  "duel.prov_extra": "Log-lineare Anpassung mit kontrolliertem Kilometerstand, R²={rsq}",
  "duel.answer_h": "Die Antwort",
  "duel.verdict_win": "<b>Bei diesem Modell hält {win_subj} den Preis besser.</b> Bei gleichem Kilometerstand verliert {a_subj} <b>{a_pct} je Jahr Fahrzeugalter</b> und {b_subj} <b>{b_pct}</b>. Der Abstand: <b>{pp} pro Jahr</b> {fav} (95-%-Intervall: {lo} bis {hi}). Nach fünf Jahren sind das {keep_win} des Preises gegenüber {keep_lose}.",
  "duel.verdict_draw": "<b>Bei diesem Modell entscheidet {choice} den Wertverlust nicht.</b> {a_scap} verliert {a_pct} je Jahr Fahrzeugalter und {b_subj} {b_pct}, und die {pp} Abstand liegen innerhalb der Messgenauigkeit (±{ci}). Das heißt nicht „wir wissen es nicht“: die Stichprobe reicht für die Aussage, dass ein Vorteil, falls es ihn gibt, kleiner ist als {bound} pro Jahr, und das ist wenig neben dem, was zwei Fahrzeuge desselben Baujahrs trennt.",
  "duel.mileage_p": "Der direkte Vergleich der Mediane beantwortet das nicht: beim {brand} {model} hat {a_subj} im Angebot {a_km} auf der Uhr und {b_subj} {b_km}, und eine Kurve, die das nicht berücksichtigt, misst die Mischung der Laufleistungen und nennt sie {facet}. Deshalb rechnet die Anpassung mit Alter <b>und</b> Kilometerstand, und die beiden Kurven unten stehen beim gleichen Kilometerstand.",
  "duel.curves_h": "Die beiden Kurven",
  "duel.curves_p": "Anteil des Preises, den das Auto mit zunehmendem Alter hält, ausgehend vom jüngsten Fahrzeug mit Stichprobe ({year}). Es sind verlangte Preise aus aktiven Angeboten, keine abgeschlossenen Verkäufe: sie messen, was der Markt heute für jedes Alter verlangt, nicht was ein bestimmter Besitzer bekommen hat.",
  "duel.chart_alt": "Gehaltener Preisanteil nach Alter, {question}",
  "duel.chart_axis": "Jahre Alter",
  "duel.gap_h": "Und heute, wer verlangt mehr?",
  "duel.gap_p": "Die Frage davor galt dem Tempo des Rückgangs, diese gilt dem Preis im Schaufenster. In jeder Zeile stehen sich {a_subj} und {b_subj} <b>gleichen Alters und mit gleichem Kilometerstand</b> gegenüber. Der Kilometerstand steckt in der Anpassung, der Abstand unten ist also nicht {mixup}.",
  "duel.gap_th_age": "Alter",
  "duel.gap_th_diff": "{a} gegenüber {b}",
  "duel.gap_th_ci": "Intervall (95 %)",
  "duel.gap_th_read": "Lesart",
  "duel.gap_read_none": "nicht unterscheidbar",
  "duel.gap_read_more": "{side} verlangt mehr",
  "duel.more": "{pct} mehr",
  "duel.less": "{pct} weniger",
  "duel.drift": "Beide Lesarten sind dieselbe Rechnung von zwei Seiten: mit {age_lo} Jahren verlangt {a_subj} {gap_lo} {b_than}, mit {age_hi} Jahren {gap_hi}. Der Aufschlag für {a} gegenüber {b} fällt mit dem Alter also <b>{move}</b> aus, und genau das sagt der gemessene Abstand: {pp} pro Jahr, nur in Preis statt in Rate geschrieben. {tail}",
  "duel.move_up": "höher",
  "duel.move_down": "niedriger",
  "duel.drift_cheap": "Was den Preis besser hält, ist hier auch das Günstigere beim Kauf: der Vorteil summiert sich, du zahlst heute weniger und verlierst später weniger.",
  "duel.drift_dear": "Was den Preis besser hält, ist hier auch das Teurere beim Kauf: der Aufschlag wird bei der Anschaffung bezahlt und über den Wertverlust zurückgegeben, und wie lange du das Auto behältst, entscheidet, ob sich das lohnt.",
  "duel.gap_note": "Werte aus der Anpassung, keine rohen Mediane: die Mediane je Alter mischen auf beiden Seiten verschiedene Ausstattungen und Laufleistungen.",
  "duel.table_h": "Beide Seiten nebeneinander",
  "duel.tr_n": "Angebote in der Anpassung",
  "duel.tr_price": "Verlangter Medianpreis",
  "duel.tr_km": "Kilometerstand (Median)",
  "duel.tr_rate": "Verlust je Jahr Fahrzeugalter",
  "duel.tr_keep5": "Restwert nach 5 Jahren",
  "duel.tr_keep10": "Restwert nach 10 Jahren",
  "duel.cta_h": "Und DEIN {brand} {model}?",
  "duel.cta_p": "Das sind die Kurven des Modells. Füg den Link deines Inserats ein und wir sagen dir den fairen Wert dieses konkreten Autos, mit deiner Motorisierung, deinem Kilometerstand und deinem Getriebe.",
  "duel.cta_btn": "Mein Auto bewerten&nbsp;&nbsp;→",
  "duel.links_model": "Alle {brand} {model}",
  "duel.links_others": "Andere Modelle",
  "duel.links_dep": "Wertverlustkurve",
  "duel.faq1_q": "Verliert bei einem {brand} {model} {a_subj} schneller an Wert als {b_subj}?",
  "duel.faq1_win": "Nicht so, wie man es gern sagt: bei diesem Modell hält {win_subj} den Preis besser. Über {n} aktive Angebote auf {source} verliert {a_subj} bei angeglichenem Kilometerstand {a_pct} je Jahr Fahrzeugalter und {b_subj} {b_pct}, {pp} Abstand pro Jahr {fav}.",
  "duel.faq1_draw": "Bei diesem Modell ist der Abstand nicht zu unterscheiden: {a_scap} verliert {a_pct} je Jahr Fahrzeugalter und {b_subj} {b_pct}, {pp} Abstand, die in die Messgenauigkeit passen (±{ci}), über {n} aktive Angebote auf {source}.",
  "duel.faq2_q": "{brand} {model}: verlangt {a_subj} mehr als {b_subj}?",
  "duel.faq2_gap": "Mit {age} Jahren und gleichem Kilometerstand verlangt {a_subj} {e} {b_than}. Roh, ohne Angleich der Laufleistung, liegt der verlangte Median bei {a_price} ({a_km} im Median) gegenüber {b_price} ({b_km}).",
  "duel.faq2_raw": "Der verlangte Median liegt bei {a_price} ({a_km} im Median) gegenüber {b_price} ({b_km} im Median). Das sind sehr unterschiedliche Laufleistungen, der rohe Preisunterschied kommt also nicht nur {facet_of}.",
  "duel.ds_name": "Wertverlust von {brand} {model} nach {facet}",
  "duel.ds_desc": "Jährlicher Wertverlust eines {brand} {model} {a_low} ({a_pct}) und {b_low} ({b_pct}), um den Kilometerstand bereinigt, über {n} aktive Angebote auf {source} mit Erstzulassung zwischen {from} und {to}.",
  "duel.var_rate": "Jährlicher Wertverlust (%)",
  "duel.var_price": "Verlangter Preis (EUR)",
  "duel.var_km": "Kilometerstand (km)",
  "duel.json_method": "log(Preis) ~ Alter + log(Kilometerstand) + Seite + Alter×Seite, kleinste Quadrate über aktive Angebote",
  "duel.hub_eyebrow": "{n} MODELLE",
  "duel.hub_lede": "Die Antwort gilt nicht für alle Autos, sondern für jedes Modell einzeln. Bei den <b>{n} Modellen</b> mit genug aktiven Angeboten, um beide Kurven getrennt anzupassen, hält {a_subj} den Preis in <b>{awins}</b> besser, {b_subj} in <b>{bwins}</b>, und in <b>{draws}</b> ist der Abstand von der Messgenauigkeit nicht zu trennen.",
  "duel.hub_desc": "Bei {n} Modellen mit ausreichender Stichprobe hält {a_subj} den Preis bei {awins} Modellen besser und {b_subj} bei {bwins}. Raten je Jahr Fahrzeugalter, gemessen an aktiven Angeboten auf {source}, mit kontrolliertem Kilometerstand.",
  "duel.hub_prov": "Verlangter Preis nach Alter, {a_low} und {b_low} getrennt, mit kontrolliertem Kilometerstand",
  "duel.hub_table_h": "Modell für Modell",
  "duel.hub_table_p": "Sortiert nach dem Abstand zwischen den beiden Kurven. Die mittlere Spalte ist das, was die Modellseite ausführt: wie viele Prozentpunkte pro Jahr die beiden Seiten trennen und mit welcher Genauigkeit das gemessen wurde.",
  "duel.hub_th_side": "{side} / Jahr",
  "duel.hub_th_diff": "Abstand",
  "duel.hub_th_wins": "Hält besser",
  "duel.hub_th_n": "Angebote {a}/{b}",
  "duel.hub_draw": "Unentschieden",
  "duel.hub_draw_h": "Warum „unentschieden“ auch eine Antwort ist",
  "duel.hub_draw_p1": "Ein halber Prozentpunkt Abstand zwischen zwei Kurven, die über ein paar Dutzend Angebote angepasst wurden, ist kein Ergebnis, sondern Rauschen mit drei Nachkommastellen. Deshalb trägt jede Zeile ihre Genauigkeit mit, und ein Modell kommt erst in diese Tabelle, wenn diese Genauigkeit eng genug ist, damit „unentschieden“ <b>kein spürbarer Vorteil</b> heißt und nicht <b>wir konnten keinen sehen</b>. Modelle, bei denen die Stichprobe für diese Unterscheidung nicht reicht, haben hier schlicht keine Seite.",
  "duel.hub_draw_p2": "Der Kilometerstand steckt überall in der Anpassung. Ohne ihn würde die Tabelle vor allem messen, {mixup}, und das dann {facet} nennen.",
  "duel.hub_cta_h": "Du stehst zwischen zwei konkreten Autos?",
  "duel.hub_cta_p": "Füg den Link jedes Inserats ein und wir sagen dir den fairen Wert von beiden, mit Motorisierung, Kilometerstand und Ausstattung des jeweiligen Fahrzeugs.",
  "duel.hub_cta_btn": "Ein Inserat bewerten&nbsp;&nbsp;→",
  "duel.pp": "{v} Prozentpunkte",
  "duel.pp_chip": "{v} %-Pkt.",
};

const STRINGS_FR = {
  "footer.depreciation": "Décote par modèle",
  "dep.crumb": "Décote",
  "dep.hub_eyebrow": "COURBES DE DÉCOTE · {n} MODÈLES",
  "dep.hub_title": "Décote des voitures d'occasion en {country}",
  "dep.hub_h1": "Quelles voitures décotent le plus vite en {country}",
  "dep.hub_desc": "Quels modèles perdent le plus de valeur par an en {country}, mesuré sur les annonces actives publiées sur {source}. {n} modèles avec une courbe complète, la valeur restante à cinq ans et l'âge à partir duquel l'année de première immatriculation ne fait plus le prix.",
  "dep.hub_lede": "Décote par année d'âge, mesurée sur les prix demandés des annonces actives publiées sur {source}. N'entrent que les modèles dont l'historique tient : au moins {cells} millésimes avec échantillon et {span} ans entre le plus ancien et le plus récent.",
  "dep.hub_market": "La médiane du marché est de {mkt} par an. Au-dessus, la voiture te coûte plus cher à garder ; en dessous, tu la revends en perdant moins.",
  "dep.hub_band": "Entre {from} et {to} ans d'âge, ces modèles perdent en médiane {a} par an ({na} modèles avec échantillon sur cette tranche) ; à partir de {cut} ans, {b} ({nb} modèles).",
  "dep.hub_band_flat": "Le pourcentage ne ralentit donc pas avec l'âge. Ce qui ralentit, c'est la facture en euros : {a} de 12 000 € et {a} de 3 000 €, ce n'est pas la même somme.",
  "dep.hub_band_slower": "Le pourcentage ralentit avec l'âge, mais moins qu'on ne le dit : {gap} séparent les deux tranches.",
  "dep.hub_table_h": "Modèle par modèle",
  "dep.th_model": "Modèle",
  "dep.th_rate": "Par an",
  "dep.th_keep5": "Valeur à 5 ans",
  "dep.th_half": "Moitié de la valeur",
  "dep.th_cheap": "Une année coûte moins de {floor}",
  "dep.th_n": "Annonces",
  "dep.th_span": "Historique",
  "dep.hub_note": "« Moitié de la valeur », c'est le temps qu'un modèle met, au rythme mesuré, à voir son prix demandé médian divisé par deux. « Une année coûte moins de {floor} », c'est l'âge à partir duquel une année de première immatriculation en plus vaut moins de {floor} sur la courbe ; un tiret veut dire que cela n'arrive pas avant {max_age} ans, donc que le millésime fait le prix sur toute la gamme qu'on achète. Les courbes traversent les générations : une partie de la baisse, c'est une autre voiture, pas de l'âge en plus.",
  "dep.hub_duels": "Ce tableau mesure le modèle entier, toutes versions confondues. Là où l'échantillon le permet, nous séparons les courbes :",
  "dep.hub_prov": "Prix demandé médian par année de première immatriculation, ajustement log-linéaire",
  "dep.eyebrow": "COURBE DE DÉCOTE · {source}",
  "dep.title": "{brand} {model} : décote par an",
  "dep.desc": "Une {brand} {model} perd environ {rate} par année d'âge et conserve {keep_five} de sa valeur à cinq ans, mesuré sur {n} annonces actives publiées sur {source}. Courbe complète, prix de chaque millésime et l'endroit où la baisse ralentit.",
  "dep.h1": "À quelle vitesse une {brand} {model} perd-elle de la valeur ?",
  "dep.lede": "Mesuré sur les prix demandés de {n} annonces actives immatriculées entre {from} et {to}, une {brand} {model} perd environ <b>{rate} par année d'âge</b>{half}. {vs}",
  "dep.half_clause": ", soit la moitié de sa valeur tous les <b>{half} ans</b>",
  "dep.vs_fast": "C'est plus rapide que le marché (médiane {mkt} par an).",
  "dep.vs_slow": "C'est plus lent que le marché (médiane {mkt} par an) : ce modèle tient sa valeur mieux que la moyenne.",
  "dep.vs_mid": "C'est le rythme du marché (médiane {mkt} par an).",
  "dep.stat_year": "PAR AN",
  "dep.stat_year_s": "−{eur} par année d'âge",
  "dep.stat_age": "À {years} ANS",
  "dep.stat_left": "−{eur}",
  "dep.stat_left_base": "−{eur} sur {base}",
  "dep.prov_measure": "Prix demandé médian par année de première immatriculation, {from}–{to}",
  "dep.prov_extra": "Ajustement log-linéaire, R²={rsq} ; montants en euros rapportés à {base}, ce que la courbe donne à un exemplaire de {year}",
  "dep.curve_h": "La courbe",
  "dep.curve_p": "Chaque point est la médiane des prix demandés à cet âge, la ligne est l'ajustement log-linéaire d'où sort le pourcentage ci-dessus{marks}. La baisse est un pourcentage de ce qui reste : beaucoup d'euros les premières années, peu à la fin, même quand le pourcentage ne bouge pas.",
  "dep.curve_marks": ", et les repères verticaux sont les âges sur lesquels cette page avance une affirmation",
  "dep.curve_caveat": "Une réserve que la courbe ne montre pas : entre {from} et {to}, la {brand} {model} a changé de génération plus d'une fois, et un exemplaire de chaque extrémité n'est pas la même voiture avec quelques années de plus. Une partie de la baisse, c'est l'âge et l'usure ; une autre, c'est un modèle différent avec un autre équipement. La série mesure ce que le marché demande pour chaque millésime, pas le vieillissement d'une voiture donnée.",
  "dep.ladder_h": "Ce que coûte une année d'âge",
  "dep.ladder_p": "La même courbe lue à l'envers : ce que tu paies, en euros, pour une année de première immatriculation en plus. C'est cette colonne qui décide s'il vaut la peine de pousser le budget d'un millésime.",
  "dep.ladder_th_age": "Âge (millésime)",
  "dep.ladder_th_cost": "Coût de +1 an",
  "dep.ladder_th_price": "Prix sur la courbe",
  "dep.ladder_note": "Valeurs issues de la courbe ajustée, pas des médianes brutes : d'un millésime au suivant, la médiane saute plus que le pas que nous mesurons ici. Les médianes brutes sont dans le tableau par millésime, plus bas.",
  "dep.age_n": "{n} ans",
  "dep.age_one": "1 an",
  "dep.bend_h": "Y a-t-il un point d'inflexion ?",
  "dep.bend_slows": "<b>Oui, à {age} ans.</b> Jusque-là, la {brand} {model} perd environ <b>{early} par an</b> ; ensuite, <b>{late}</b>. Un exemplaire déjà passé de l'autre côté te coûte moins cher pour chaque année où tu le gardes : c'est la tranche où l'achat devient bon marché.",
  "dep.bend_speeds": "<b>Il y a un point d'inflexion à {age} ans, mais à l'envers de ce qu'on attend :</b> avant, la {brand} {model} perd environ <b>{early} par an</b>, après <b>{late}</b> — le pourcentage accélère avec l'âge au lieu de ralentir. Sur les exemplaires les plus vieux, le calcul change quand même : le prix est déjà bas, donc la perte en euros continue de se réduire.",
  "dep.bend_none": "<b>Non, pas en pourcentage.</b> Nous avons testé un ajustement avec cassure à chaque âge entre {min_age} et {max_age} ans, et aucun n'explique les prix de la {brand} {model} mieux qu'une baisse constante de {rate} par an (R²={rsq}).",
  "dep.bend_none_cand": " Le meilleur candidat tombait à {age} ans — {early} par an avant, {late} après — et cet écart a la taille des sauts que la médiane fait déjà d'un millésime au suivant.",
  "dep.bend_none_tail": " Ce qui ralentit, c'est la facture en euros, pas le pourcentage.",
  "dep.cheap_yes": "Le moment où ça cesse de faire mal se date : à partir de <b>{age} ans</b> — immatriculations de {year} et plus anciennes, autour de {price} — chaque année d'âge en plus coûte moins de {floor}. C'est moins qu'un train de pneus ou qu'une courroie de distribution : à partir de là, l'état de la voiture pèse plus que le millésime, et c'est là-dessus que l'achat se décide.",
  "dep.cheap_no": "Ici, ça n'arrive pas dans la gamme que l'on cherche : à {age} ans, une année d'âge vaut encore environ {cost}, et elle ne passerait sous {floor} que bien plus tard. Sur ce modèle, le millésime commande le prix sur toute la gamme achetable — pousser le budget d'une année plus récente continue de coûter cher.",
  "dep.table_h": "Prix médian par millésime",
  "dep.th_year": "Millésime",
  "dep.th_median": "Médiane (demandé)",
  "dep.th_listings": "Annonces",
  "dep.th_gap": "Ans avant {year}",
  "dep.th_vs": "face à {year}",
  "dep.th_km": "Kilométrage (médiane)",
  "dep.cta_h": "Et TA {brand} {model} aujourd'hui ?",
  "dep.cta_p": "Cette courbe est celle du modèle. Colle le lien de ton annonce et nous te disons la valeur juste de cette voiture précise, avec ton kilométrage et ta finition.",
  "dep.cta_btn": "Estimer ma voiture&nbsp;&nbsp;→",
  "dep.links_model": "{brand} {model} par millésime",
  "dep.links_others": "La décote des autres modèles",
  "dep.faq1_q": "Combien une {brand} {model} perd-elle de valeur par an ?",
  "dep.faq1_a": "Environ {rate} de la valeur restante par année d'âge, mesuré sur les prix demandés de {n} annonces actives de {brand} {model} immatriculées entre {from} et {to} sur {source} en {country}. Le pourcentage reste le même, mais en euros la perte est bien plus lourde les premières années.",
  "dep.faq2_q": "Que vaut une {brand} {model} au bout de cinq ans ?",
  "dep.faq2_a": "Au rythme mesuré, une {brand} {model} conserve environ {keep} de sa valeur à cinq ans. Rapporté aux {base} que la courbe donne à un exemplaire de {year}, cela fait environ {eur} de moins.",
  "dep.faq3_q": "En combien d'années une {brand} {model} perd-elle la moitié de sa valeur ?",
  "dep.faq3_a": "En environ {half} ans, au rythme de {rate} par an mesuré sur les annonces actives publiées sur {source}. C'est le même taux dit autrement : tous les {half} ans d'âge, le prix demandé médian est divisé par deux.",
  "dep.faq4_q": "À partir de quel âge une {brand} {model} ne perd-elle presque plus de valeur ?",
  "dep.faq4_bend": "La baisse ralentit à {age} ans : environ {early} par an jusque-là, {late} ensuite.",
  "dep.faq4_flat": "Elle ne cesse jamais de perdre, elle cesse seulement de faire mal. En pourcentage, la baisse reste autour de {rate} par an sur toute la série mesurée : il n'y a pas d'âge à partir duquel le pourcentage freine.",
  "dep.faq4_cheap": " En euros, c'est autre chose : à partir de {age} ans (immatriculations de {year} et plus anciennes), chaque année d'âge coûte moins de {floor}, et l'état de la voiture pèse alors plus que le millésime.",
  "dep.faq4_nocheap": " En euros, la perte se réduit, mais lentement : à {age} ans, une année d'âge coûte encore environ {cost}.",
  "dep.faq5_q": "Une {brand} {model} décote-t-elle plus que la moyenne ?",
  "dep.faq5_a": "La médiane du marché de l'occasion que nous mesurons est de {mkt} par année d'âge. La {brand} {model} est à {rate}. {verdict}",
  "dep.above": "C'est au-dessus de la moyenne.",
  "dep.below": "C'est en dessous de la moyenne.",
  "dep.inline": "C'est exactement la moyenne.",
  "dep.ds_name": "Décote de {brand} {model} en {country}",
  "dep.ds_desc": "Prix demandé médian d'une {brand} {model} par année de première immatriculation ({from}–{to}), décote annuelle de {rate} et coût en euros de chaque année d'âge, à partir de {n} annonces actives publiées sur {source}.",
  "dep.var_price": "Prix demandé (EUR)",
  "dep.var_rate": "Décote annuelle (%)",
  "dep.var_cost": "Coût d'une année d'âge (EUR)",
  "dep.json_method": "log(prix) ~ millésime, moindres carrés sur les prix médians de chaque millésime comptant au moins cinq annonces",
  "dep.json_ladder_note": "Valeurs issues de la courbe ajustée, pas des médianes brutes.",
  "dep.json_bend_note": "Un ajustement avec cassure a été testé à chaque âge entre {min_age} et {max_age} ans ; il n'est publié que s'il bat nettement le taux constant.",
  "dep.json_hub_note": "Un modèle entre s'il a au moins {cells} millésimes avec échantillon, {span} ans d'historique et un ajustement qui explique réellement les points.",
  "dep.chart_alt": "Prix demandé médian par âge de la voiture, avec la courbe ajustée",
  "dep.chart_axis": "ans d'âge",
  "dep.chart_bend": "cassure à {age} ans",
  "dep.chart_cheap": "{floor}/an à {age} ans",
  "dep.chart_tip": "{age} ans ({year}) : {price} · {n} annonces",
  "duel.fuel_crumb": "Diesel ou essence",
  "duel.fuel_eyebrow": "DIESEL VS ESSENCE",
  "duel.fuel_question": "diesel ou essence",
  "duel.fuel_hub_h1": "Diesel ou essence : où le choix change vraiment le prix",
  "duel.fuel_hub_title": "Diesel ou essence : lequel tient le mieux le prix, modèle par modèle",
  "duel.fuel_choice": "le choix du carburant",
  "duel.fuel_facet": "carburant",
  "duel.fuel_facet_of": "du carburant",
  "duel.fuel_mixup": "le fait que les diesels en vente ont beaucoup plus roulé",
  "duel.fuel_a": "Diesel",
  "duel.fuel_b": "Essence",
  "duel.fuel_a_low": "diesel",
  "duel.fuel_b_low": "essence",
  "duel.fuel_a_subj": "le diesel",
  "duel.fuel_b_subj": "l'essence",
  "duel.fuel_a_scap": "Le diesel",
  "duel.fuel_b_scap": "L'essence",
  "duel.fuel_a_fav": "en faveur du diesel",
  "duel.fuel_b_fav": "en faveur de l'essence",
  "duel.fuel_a_chip": "DIESEL",
  "duel.fuel_b_chip": "ESSENCE",
  "duel.fuel_b_than": "que l'essence",
  "duel.gear_crumb": "Manuelle ou automatique",
  "duel.gear_eyebrow": "BOÎTE MANUELLE VS AUTOMATIQUE",
  "duel.gear_question": "boîte manuelle ou automatique",
  "duel.gear_hub_h1": "Manuelle ou automatique : où la boîte change vraiment le prix",
  "duel.gear_hub_title": "Boîte manuelle ou automatique : laquelle tient le mieux le prix, modèle par modèle",
  "duel.gear_choice": "le choix de la boîte",
  "duel.gear_facet": "boîte",
  "duel.gear_facet_of": "de la boîte",
  "duel.gear_mixup": "le fait que les automatiques en vente sont plus récentes et ont beaucoup moins roulé",
  "duel.gear_a": "Manuelle",
  "duel.gear_b": "Automatique",
  "duel.gear_a_low": "en boîte manuelle",
  "duel.gear_b_low": "en boîte automatique",
  "duel.gear_a_subj": "la boîte manuelle",
  "duel.gear_b_subj": "la boîte automatique",
  "duel.gear_a_scap": "La boîte manuelle",
  "duel.gear_b_scap": "La boîte automatique",
  "duel.gear_a_fav": "en faveur de la boîte manuelle",
  "duel.gear_b_fav": "en faveur de la boîte automatique",
  "duel.gear_a_chip": "MANUELLE",
  "duel.gear_b_chip": "AUTOMATIQUE",
  "duel.gear_b_than": "que la boîte automatique",
  "duel.title": "{brand} {model} : {question}, qu'est-ce qui tient le mieux le prix ?",
  "duel.h1": "{brand} {model} : {question}, qu'est-ce qui tient le mieux le prix ?",
  "duel.desc": "Sur une {brand} {model}, {a_subj} perd {a_pct} par année d'âge et {b_subj} {b_pct}, mesuré sur {n} annonces actives publiées sur {source} à kilométrage égal. {tail}",
  "duel.desc_win": "L'écart : {pp} par an {fav}.",
  "duel.desc_draw": "L'écart ne se distingue pas de la précision de la mesure.",
  "duel.lede": "Nous avons ajusté les deux courbes séparément sur <b>{n} annonces actives</b> de {brand} {model} ({an} {a_low}, {bn} {b_low}, immatriculations de {from} à {to}), à kilométrage égal. {a_scap} perd <b>{a_pct} par année d'âge</b>, {b_subj} <b>{b_pct}</b>.",
  "duel.stat_diff": "ÉCART",
  "duel.stat_side": "par an · {n} annonces",
  "duel.stat_ci": "±{ci} · {verdict}",
  "duel.draw_word": "indistinguable",
  "duel.prov_measure": "Prix demandé d'une {brand} {model}, {a_low} et {b_low} séparément ({from}–{to})",
  "duel.prov_extra": "Ajustement log-linéaire à kilométrage contrôlé, R²={rsq}",
  "duel.answer_h": "La réponse",
  "duel.verdict_win": "<b>Sur ce modèle, c'est {win_subj} qui tient le mieux le prix.</b> À kilométrage égal, {a_subj} perd <b>{a_pct} par année d'âge</b> et {b_subj} <b>{b_pct}</b> — un écart de <b>{pp} par an</b> {fav} (intervalle à 95 % : {lo} à {hi}). À cinq ans, cela fait {keep_win} du prix conservés contre {keep_lose}.",
  "duel.verdict_draw": "<b>Sur ce modèle, {choice} ne décide pas de la décote.</b> {a_scap} perd {a_pct} par année d'âge et {b_subj} {b_pct}, et l'écart de {pp} tient dans la précision de la mesure elle-même (±{ci}). Ce n'est pas « nous ne savons pas » : l'échantillon suffit à dire que s'il y a un avantage, il est inférieur à {bound} par an — peu de chose à côté de ce qui sépare deux exemplaires du même millésime.",
  "duel.mileage_p": "La comparaison directe des médianes ne répond pas à ça : sur la {brand} {model}, {a_subj} en vente affiche {a_km} au compteur et {b_subj} {b_km}, et une courbe qui n'en tient pas compte mesure le mélange des kilométrages et appelle ça {facet}. L'ajustement travaille donc sur l'âge <b>et</b> le kilométrage, et les deux courbes ci-dessous sont au même kilométrage.",
  "duel.curves_h": "Les deux courbes",
  "duel.curves_p": "Part du prix conservée à mesure que la voiture vieillit, à partir du plus récent exemplaire avec échantillon ({year}). Ce sont des prix demandés dans des annonces actives, pas des ventes conclues : ils mesurent ce que le marché demande aujourd'hui à chaque âge, pas ce qu'un propriétaire a touché.",
  "duel.chart_alt": "Part du prix conservée par âge, {question}",
  "duel.chart_axis": "ans d'âge",
  "duel.gap_h": "Et aujourd'hui, qui demande le plus ?",
  "duel.gap_p": "La question précédente portait sur le rythme de la baisse ; celle-ci porte sur le prix en vitrine. Chaque ligne compare {a_subj} et {b_subj} <b>du même âge et au même kilométrage</b> — le kilométrage entre dans l'ajustement, donc ce que tu vois ci-dessous n'est pas {mixup}.",
  "duel.gap_th_age": "Âge",
  "duel.gap_th_diff": "{a} face à {b}",
  "duel.gap_th_ci": "Intervalle (95 %)",
  "duel.gap_th_read": "Lecture",
  "duel.gap_read_none": "indistinguable",
  "duel.gap_read_more": "{side} demande plus",
  "duel.more": "{pct} de plus",
  "duel.less": "{pct} de moins",
  "duel.drift": "Les deux lectures sont le même calcul vu de deux côtés : à {age_lo} ans, {a_subj} demande {gap_lo} {b_than} ; à {age_hi} ans, {gap_hi}. Avec l'âge, le supplément de {a} face à {b} est donc <b>{move}</b>, et c'est exactement ce que dit l'écart de {pp} par an, écrit en prix au lieu d'être écrit en taux. {tail}",
  "duel.move_up": "plus élevé",
  "duel.move_down": "plus faible",
  "duel.drift_cheap": "Ce qui tient le mieux le prix est ici aussi le moins cher à l'achat : l'avantage s'additionne, tu paies moins aujourd'hui et tu perds moins ensuite.",
  "duel.drift_dear": "Ce qui tient le mieux le prix est ici le plus cher à l'achat : le supplément se paie à l'achat et se récupère en décote, et c'est la durée de garde qui décide si le compte y est.",
  "duel.gap_note": "Valeurs issues de l'ajustement, pas des médianes brutes : les médianes par âge mélangent des finitions et des kilométrages différents des deux côtés.",
  "duel.table_h": "Les deux côtés, côte à côte",
  "duel.tr_n": "Annonces dans l'ajustement",
  "duel.tr_price": "Prix demandé médian",
  "duel.tr_km": "Kilométrage (médiane)",
  "duel.tr_rate": "Perte par année d'âge",
  "duel.tr_keep5": "Valeur conservée à 5 ans",
  "duel.tr_keep10": "Valeur conservée à 10 ans",
  "duel.cta_h": "Et TA {brand} {model} ?",
  "duel.cta_p": "Ce sont les courbes du modèle. Colle le lien de ton annonce et nous te disons la valeur juste de cette voiture précise, avec ta motorisation, ton kilométrage et ta boîte.",
  "duel.cta_btn": "Estimer ma voiture&nbsp;&nbsp;→",
  "duel.links_model": "Toutes les {brand} {model}",
  "duel.links_others": "Autres modèles",
  "duel.links_dep": "Courbe de décote",
  "duel.faq1_q": "Sur une {brand} {model}, qu'est-ce qui décote le plus vite, {a_subj} ou {b_subj} ?",
  "duel.faq1_win": "Pas comme on le dit d'habitude : sur ce modèle, c'est {win_subj} qui tient le mieux le prix. Parmi {n} annonces actives publiées sur {source}, à kilométrage égal, {a_subj} perd {a_pct} par année d'âge et {b_subj} {b_pct} — {pp} d'écart par an {fav}.",
  "duel.faq1_draw": "Sur ce modèle, l'écart ne se distingue pas : {a_scap} perd {a_pct} par année d'âge et {b_subj} {b_pct}, soit {pp} d'écart, ce qui tient dans la précision de la mesure (±{ci}), parmi {n} annonces actives publiées sur {source}.",
  "duel.faq2_q": "{brand} {model} : qui demande le plus cher, {a_subj} ou {b_subj} ?",
  "duel.faq2_gap": "À {age} ans et à kilométrage égal, {a_subj} demande {e} {b_than}. Brut, sans égaliser les kilomètres, la médiane demandée est de {a_price} ({a_km} en médiane) contre {b_price} ({b_km}).",
  "duel.faq2_raw": "La médiane demandée est de {a_price} ({a_km} en médiane) contre {b_price} ({b_km} en médiane). Ce sont des kilométrages très différents : l'écart de prix brut ne vient donc pas seulement {facet_of}.",
  "duel.ds_name": "Décote de {brand} {model} par {facet}",
  "duel.ds_desc": "Décote annuelle d'une {brand} {model} {a_low} ({a_pct}) et {b_low} ({b_pct}), corrigée du kilométrage, sur {n} annonces actives publiées sur {source}, immatriculées entre {from} et {to}.",
  "duel.var_rate": "Décote annuelle (%)",
  "duel.var_price": "Prix demandé (EUR)",
  "duel.var_km": "Kilométrage (km)",
  "duel.json_method": "log(prix) ~ âge + log(kilométrage) + côté + âge×côté, moindres carrés sur les annonces actives",
  "duel.hub_eyebrow": "{n} MODÈLES",
  "duel.hub_lede": "La réponse ne vaut pas pour toutes les voitures, elle vaut modèle par modèle. Sur les <b>{n} modèles</b> qui ont assez d'annonces actives pour ajuster les deux courbes séparément, {a_subj} tient mieux le prix sur <b>{awins}</b>, {b_subj} sur <b>{bwins}</b>, et sur <b>{draws}</b> l'écart ne se sépare pas de la précision de la mesure.",
  "duel.hub_desc": "Sur {n} modèles à l'échantillon suffisant, {a_subj} tient mieux le prix sur {awins} modèles et {b_subj} sur {bwins}. Taux par année d'âge mesurés sur les annonces actives publiées sur {source}, à kilométrage contrôlé.",
  "duel.hub_prov": "Prix demandé par âge, {a_low} et {b_low} séparément, à kilométrage contrôlé",
  "duel.hub_table_h": "Modèle par modèle",
  "duel.hub_table_p": "Classé par la distance entre les deux courbes. La colonne du milieu est ce que la page du modèle développe : combien de points de pourcentage par an séparent les deux côtés, et avec quelle précision cela a été mesuré.",
  "duel.hub_th_side": "{side} / an",
  "duel.hub_th_diff": "Écart",
  "duel.hub_th_wins": "Tient le mieux",
  "duel.hub_th_n": "Annonces {a}/{b}",
  "duel.hub_draw": "Égalité",
  "duel.hub_draw_h": "Pourquoi « égalité » est aussi une réponse",
  "duel.hub_draw_p1": "Un demi-point d'écart entre deux courbes ajustées sur quelques dizaines d'annonces n'est pas un résultat, c'est du bruit à trois décimales. Chaque ligne porte donc sa précision, et un modèle n'entre dans ce tableau que lorsque cette précision est assez serrée pour que « égalité » veuille dire <b>aucun avantage sensible</b> et non <b>nous n'avons pas réussi à en voir</b>. Les modèles dont l'échantillon ne permet pas cette distinction n'ont tout simplement pas de ligne ici.",
  "duel.hub_draw_p2": "Le kilométrage entre dans l'ajustement partout. Sans lui, le tableau mesurerait surtout {mixup}, pour appeler ça {facet} ensuite.",
  "duel.hub_cta_h": "Tu hésites entre deux voitures précises ?",
  "duel.hub_cta_p": "Colle le lien de chaque annonce et nous te disons la valeur juste des deux, avec la motorisation, le kilométrage et la finition de chaque exemplaire.",
  "duel.hub_cta_btn": "Estimer une annonce&nbsp;&nbsp;→",
  "duel.pp": "{v} points",
  "duel.pp_chip": "{v} pts",
};

const STRINGS_IT = {
  "footer.depreciation": "Svalutazione per modello",
  "dep.crumb": "Svalutazione",
  "dep.hub_eyebrow": "CURVE DI SVALUTAZIONE · {n} MODELLI",
  "dep.hub_title": "Svalutazione delle auto usate in {country}",
  "dep.hub_h1": "Quali auto in {country} perdono valore più in fretta",
  "dep.hub_desc": "Quali modelli in {country} perdono più valore ogni anno, misurato sugli annunci attivi di {source}. {n} modelli con curva completa, valore residuo a cinque anni e l'età oltre la quale l'anno di immatricolazione non fa più il prezzo.",
  "dep.hub_lede": "Svalutazione per ogni anno di età, misurata sui prezzi richiesti negli annunci attivi di {source}. Entra solo chi ha storia sufficiente perché la curva significhi qualcosa: almeno {cells} annate con campione e {span} anni tra la più vecchia e la più recente.",
  "dep.hub_market": "La mediana del mercato è {mkt} all'anno. Sopra, tenere l'auto ti costa di più; sotto, la rivendi perdendo meno.",
  "dep.hub_band": "Tra {from} e {to} anni di età questi modelli perdono in mediana {a} all'anno ({na} modelli con campione in questa fascia); da {cut} anni in poi, {b} ({nb} modelli).",
  "dep.hub_band_flat": "La percentuale quindi non rallenta con l'età. Quello che rallenta è il conto in euro: {a} di 12.000 € e {a} di 3.000 € non sono la stessa cifra.",
  "dep.hub_band_slower": "La percentuale rallenta con l'età, ma meno di quanto si dica: tra le due fasce ci sono {gap}.",
  "dep.hub_table_h": "Modello per modello",
  "dep.th_model": "Modello",
  "dep.th_rate": "All'anno",
  "dep.th_keep5": "Valore a 5 anni",
  "dep.th_half": "Metà del valore",
  "dep.th_cheap": "Un anno costa meno di {floor}",
  "dep.th_n": "Annunci",
  "dep.th_span": "Storico",
  "dep.hub_note": "«Metà del valore» è il tempo che un modello impiega, al ritmo misurato, perché il prezzo richiesto mediano si dimezzi. «Un anno costa meno di {floor}» è l'età dalla quale un anno di immatricolazione in più vale meno di {floor} sulla curva; un trattino significa che non succede prima dei {max_age} anni, cioè che l'immatricolazione comanda il prezzo su tutta la gamma che si compra. Le curve attraversano le generazioni: una parte del calo è un'auto diversa, non più età.",
  "dep.hub_duels": "Questa tabella misura il modello intero, con tutte le versioni insieme. Dove il campione basta, separiamo le curve:",
  "dep.hub_prov": "Prezzo richiesto mediano per anno di immatricolazione, adattamento log-lineare",
  "dep.eyebrow": "CURVA DI SVALUTAZIONE · {source}",
  "dep.title": "{brand} {model}: svalutazione all'anno",
  "dep.desc": "Una {brand} {model} perde circa {rate} per ogni anno di età e a cinque anni conserva {keep_five} del valore, misurato su {n} annunci attivi di {source}. Curva completa, prezzo di ogni annata e il punto in cui il calo rallenta.",
  "dep.h1": "Quanto in fretta perde valore una {brand} {model}?",
  "dep.lede": "Misurato sui prezzi richiesti di {n} annunci attivi immatricolati tra il {from} e il {to}, una {brand} {model} perde circa <b>{rate} per ogni anno di età</b>{half}. {vs}",
  "dep.half_clause": ", cioè metà del valore ogni <b>{half} anni</b>",
  "dep.vs_fast": "È più veloce del mercato (mediana {mkt} all'anno).",
  "dep.vs_slow": "È più lento del mercato (mediana {mkt} all'anno): questo modello tiene il valore meglio della media.",
  "dep.vs_mid": "È in linea con il mercato (mediana {mkt} all'anno).",
  "dep.stat_year": "ALL'ANNO",
  "dep.stat_year_s": "−{eur} per anno di età",
  "dep.stat_age": "A {years} ANNI",
  "dep.stat_left": "−{eur}",
  "dep.stat_left_base": "−{eur} su {base}",
  "dep.prov_measure": "Prezzo richiesto mediano per anno di immatricolazione, {from}–{to}",
  "dep.prov_extra": "Adattamento log-lineare, R²={rsq}; importi in euro riferiti a {base}, quanto la curva dà a un esemplare del {year}",
  "dep.curve_h": "La curva",
  "dep.curve_p": "Ogni punto è la mediana dei prezzi richiesti a quell'età, la linea è l'adattamento log-lineare da cui esce la percentuale qui sopra{marks}. Il calo è una quota di quel che resta: tanti euro nei primi anni, pochi alla fine, anche quando la percentuale non cambia.",
  "dep.curve_marks": ", e i segni verticali sono le età su cui questa pagina fa un'affermazione",
  "dep.curve_caveat": "Un'avvertenza che la curva non mostra: tra il {from} e il {to} la {brand} {model} ha cambiato generazione più di una volta, e un esemplare di ciascun estremo non è la stessa auto con qualche anno in più. Una parte del calo è età e usura, un'altra è semplicemente un modello diverso con un altro allestimento. La serie misura quanto il mercato chiede per ogni annata, non l'invecchiamento di una singola auto.",
  "dep.ladder_h": "Quanto costa un anno di età",
  "dep.ladder_p": "La stessa curva letta al contrario: quanto paghi, in euro, per un anno di immatricolazione più recente. È questa colonna a decidere se conviene allungare il budget di un'annata.",
  "dep.ladder_th_age": "Età (immatricolazione)",
  "dep.ladder_th_cost": "Costo di +1 anno",
  "dep.ladder_th_price": "Prezzo sulla curva",
  "dep.ladder_note": "Valori dalla curva adattata, non mediane grezze: tra due annate vicine la mediana salta più del passo che stiamo misurando. Le mediane grezze sono nella tabella per annata, più sotto.",
  "dep.age_n": "{n} anni",
  "dep.age_one": "1 anno",
  "dep.bend_h": "C'è un punto di svolta?",
  "dep.bend_slows": "<b>Sì, a {age} anni.</b> Fino a quell'età la {brand} {model} perde circa <b>{early} all'anno</b>; da lì in poi <b>{late}</b>. Un esemplare già oltre quella svolta ti costa meno per ogni anno in cui lo tieni: è il tratto in cui comprare viene a poco.",
  "dep.bend_speeds": "<b>Una svolta a {age} anni c'è, ma al contrario di come ci si aspetta:</b> prima la {brand} {model} perde circa <b>{early} all'anno</b>, dopo <b>{late}</b> — la percentuale accelera con l'età invece di frenare. Sugli esemplari più vecchi il conto cambia lo stesso: il prezzo è già basso, quindi la perdita in euro continua a ridursi.",
  "dep.bend_none": "<b>No, non in percentuale.</b> Abbiamo provato un adattamento con svolta a ogni età tra {min_age} e {max_age} anni e nessuno spiega i prezzi della {brand} {model} meglio di un calo costante del {rate} all'anno (R²={rsq}).",
  "dep.bend_none_cand": " Il candidato migliore cadeva a {age} anni — {early} all'anno prima, {late} dopo — e quella distanza è grande quanto i salti che la mediana fa già tra due annate vicine.",
  "dep.bend_none_tail": " A rallentare è il conto in euro, non la percentuale.",
  "dep.cheap_yes": "Il momento in cui smette di far male si può datare: da <b>{age} anni</b> — immatricolazioni {year} e più vecchie, intorno a {price} — ogni anno di età in più costa meno di {floor}. È meno di un treno di gomme o di una cinghia di distribuzione: da lì in avanti pesa più lo stato dell'auto che l'annata, ed è lì che si decide l'acquisto.",
  "dep.cheap_no": "Qui non succede, non nella fascia che si cerca davvero: a {age} anni un anno di età vale ancora circa {cost}, e scenderebbe sotto {floor} solo molto più tardi. Su questo modello l'immatricolazione comanda il prezzo su tutta la gamma acquistabile: allungare il budget per un'annata più recente continua a costare parecchio.",
  "dep.table_h": "Prezzo mediano per annata",
  "dep.th_year": "Annata",
  "dep.th_median": "Mediana (richiesta)",
  "dep.th_listings": "Annunci",
  "dep.th_gap": "Anni prima del {year}",
  "dep.th_vs": "rispetto al {year}",
  "dep.th_km": "Chilometraggio (mediana)",
  "dep.cta_h": "E la TUA {brand} {model} oggi?",
  "dep.cta_p": "Questa curva è del modello. Incolla il link del tuo annuncio e ti diciamo il valore giusto di quella singola auto, con il tuo chilometraggio e il tuo allestimento.",
  "dep.cta_btn": "Valuta la mia auto&nbsp;&nbsp;→",
  "dep.links_model": "{brand} {model} per annata",
  "dep.links_others": "Svalutazione di altri modelli",
  "dep.faq1_q": "Quanto perde di valore all'anno una {brand} {model}?",
  "dep.faq1_a": "Circa {rate} del valore residuo per ogni anno di età, misurato sui prezzi richiesti di {n} annunci attivi di {brand} {model} immatricolati tra il {from} e il {to} su {source} in {country}. La percentuale resta la stessa, ma in euro la perdita è molto più pesante nei primi anni.",
  "dep.faq2_q": "Quanto vale una {brand} {model} dopo cinque anni?",
  "dep.faq2_a": "Al ritmo misurato, dopo cinque anni una {brand} {model} conserva circa {keep} del valore. Sui {base} che la curva dà a un esemplare del {year}, sono circa {eur} in meno.",
  "dep.faq3_q": "In quanti anni una {brand} {model} perde metà del valore?",
  "dep.faq3_a": "In circa {half} anni, al ritmo di {rate} all'anno misurato sugli annunci attivi di {source}. È lo stesso tasso detto in un altro modo: ogni {half} anni di età il prezzo richiesto mediano si dimezza.",
  "dep.faq4_q": "Da che età una {brand} {model} smette quasi di perdere valore?",
  "dep.faq4_bend": "Il calo rallenta a {age} anni: fino a lì circa {early} all'anno, poi {late}.",
  "dep.faq4_flat": "Non smette mai di perdere, smette solo di far male. In percentuale il calo resta intorno a {rate} all'anno su tutta la serie misurata: non c'è un'età dalla quale la percentuale freni.",
  "dep.faq4_cheap": " In euro è un'altra storia: da {age} anni (immatricolazioni {year} e più vecchie) ogni anno di età costa meno di {floor}, e allora pesa più lo stato dell'auto che l'annata.",
  "dep.faq4_nocheap": " In euro la perdita si riduce, ma piano: a {age} anni un anno di età costa ancora circa {cost}.",
  "dep.faq5_q": "Una {brand} {model} si svaluta più della media?",
  "dep.faq5_a": "La mediana del mercato dell'usato che misuriamo è {mkt} per anno di età. La {brand} {model} sta a {rate}. {verdict}",
  "dep.above": "È sopra la media.",
  "dep.below": "È sotto la media.",
  "dep.inline": "È esattamente la media.",
  "dep.ds_name": "Svalutazione di {brand} {model} in {country}",
  "dep.ds_desc": "Prezzo richiesto mediano di una {brand} {model} per anno di immatricolazione ({from}–{to}), svalutazione annua del {rate} e costo in euro di ogni anno di età, da {n} annunci attivi di {source}.",
  "dep.var_price": "Prezzo richiesto (EUR)",
  "dep.var_rate": "Svalutazione annua (%)",
  "dep.var_cost": "Costo di un anno di età (EUR)",
  "dep.json_method": "log(prezzo) ~ annata, minimi quadrati sui prezzi mediani di ogni annata con almeno cinque annunci",
  "dep.json_ladder_note": "Valori dalla curva adattata, non mediane grezze.",
  "dep.json_bend_note": "Un adattamento con svolta è stato provato a ogni età tra {min_age} e {max_age} anni e viene pubblicato solo se batte nettamente il tasso costante.",
  "dep.json_hub_note": "Entra un modello con almeno {cells} annate con campione, {span} anni di storico e un adattamento che spiega davvero i punti.",
  "dep.chart_alt": "Prezzo richiesto mediano per età dell'auto, con la curva adattata",
  "dep.chart_axis": "anni di età",
  "dep.chart_bend": "svolta a {age} anni",
  "dep.chart_cheap": "{floor}/anno a {age}",
  "dep.chart_tip": "{age} anni ({year}): {price} · {n} annunci",
  "duel.fuel_crumb": "Diesel o benzina",
  "duel.fuel_eyebrow": "DIESEL VS BENZINA",
  "duel.fuel_question": "diesel o benzina",
  "duel.fuel_hub_h1": "Diesel o benzina: dove la scelta cambia davvero il prezzo",
  "duel.fuel_hub_title": "Diesel o benzina: chi tiene meglio il prezzo, modello per modello",
  "duel.fuel_choice": "la scelta dell'alimentazione",
  "duel.fuel_facet": "alimentazione",
  "duel.fuel_facet_of": "dall'alimentazione",
  "duel.fuel_mixup": "il fatto che i diesel in vendita hanno molti più chilometri",
  "duel.fuel_a": "Diesel",
  "duel.fuel_b": "Benzina",
  "duel.fuel_a_low": "diesel",
  "duel.fuel_b_low": "a benzina",
  "duel.fuel_a_subj": "il diesel",
  "duel.fuel_b_subj": "la benzina",
  "duel.fuel_a_scap": "Il diesel",
  "duel.fuel_b_scap": "La benzina",
  "duel.fuel_a_fav": "a favore del diesel",
  "duel.fuel_b_fav": "a favore della benzina",
  "duel.fuel_a_chip": "DIESEL",
  "duel.fuel_b_chip": "BENZINA",
  "duel.fuel_b_than": "della benzina",
  "duel.gear_crumb": "Manuale o automatico",
  "duel.gear_eyebrow": "CAMBIO MANUALE VS AUTOMATICO",
  "duel.gear_question": "cambio manuale o automatico",
  "duel.gear_hub_h1": "Manuale o automatico: dove il cambio sposta davvero il prezzo",
  "duel.gear_hub_title": "Cambio manuale o automatico: quale tiene meglio il prezzo, modello per modello",
  "duel.gear_choice": "la scelta del cambio",
  "duel.gear_facet": "cambio",
  "duel.gear_facet_of": "dal cambio",
  "duel.gear_mixup": "il fatto che gli automatici in vendita sono più recenti e hanno molti meno chilometri",
  "duel.gear_a": "Manuale",
  "duel.gear_b": "Automatico",
  "duel.gear_a_low": "con cambio manuale",
  "duel.gear_b_low": "con cambio automatico",
  "duel.gear_a_subj": "il cambio manuale",
  "duel.gear_b_subj": "il cambio automatico",
  "duel.gear_a_scap": "Il cambio manuale",
  "duel.gear_b_scap": "Il cambio automatico",
  "duel.gear_a_fav": "a favore del cambio manuale",
  "duel.gear_b_fav": "a favore del cambio automatico",
  "duel.gear_a_chip": "MANUALE",
  "duel.gear_b_chip": "AUTOMATICO",
  "duel.gear_b_than": "del cambio automatico",
  "duel.title": "{brand} {model}: {question}, chi tiene meglio il prezzo?",
  "duel.h1": "{brand} {model}: {question}, chi tiene meglio il prezzo?",
  "duel.desc": "Su una {brand} {model}, {a_subj} perde {a_pct} per anno di età e {b_subj} {b_pct}, misurato su {n} annunci attivi di {source} a pari chilometraggio. {tail}",
  "duel.desc_win": "La distanza: {pp} all'anno {fav}.",
  "duel.desc_draw": "La distanza non si distingue dalla precisione della misura.",
  "duel.lede": "Abbiamo adattato le due curve separatamente su <b>{n} annunci attivi</b> di {brand} {model} ({an} {a_low}, {bn} {b_low}, immatricolazioni dal {from} al {to}), a pari chilometraggio. {a_scap} perde <b>{a_pct} per anno di età</b>, {b_subj} <b>{b_pct}</b>.",
  "duel.stat_diff": "DISTANZA",
  "duel.stat_side": "all'anno · {n} annunci",
  "duel.stat_ci": "±{ci} · {verdict}",
  "duel.draw_word": "indistinguibile",
  "duel.prov_measure": "Prezzo richiesto di {brand} {model}, {a_low} e {b_low} separatamente ({from}–{to})",
  "duel.prov_extra": "Adattamento log-lineare con chilometraggio controllato, R²={rsq}",
  "duel.answer_h": "La risposta",
  "duel.verdict_win": "<b>Su questo modello è {win_subj} a tenere meglio il prezzo.</b> A pari chilometraggio {a_subj} perde <b>{a_pct} per anno di età</b> e {b_subj} <b>{b_pct}</b>: una distanza di <b>{pp} all'anno</b> {fav} (intervallo al 95 %: da {lo} a {hi}). Dopo cinque anni sono {keep_win} del prezzo conservati contro {keep_lose}.",
  "duel.verdict_draw": "<b>Su questo modello {choice} non decide la svalutazione.</b> {a_scap} perde {a_pct} per anno di età e {b_subj} {b_pct}, e la distanza di {pp} sta dentro la precisione della misura stessa (±{ci}). Non è «non lo sappiamo»: il campione basta a dire che un vantaggio, se c'è, è minore di {bound} all'anno, poca cosa accanto a quello che separa due esemplari della stessa annata.",
  "duel.mileage_p": "Il confronto diretto delle mediane non risponde: su una {brand} {model} {a_subj} in vendita ha {a_km} sul contachilometri e {b_subj} {b_km}, e una curva che non ne tiene conto misura il miscuglio dei chilometraggi e lo chiama {facet}. Per questo l'adattamento usa età <b>e</b> chilometraggio, e le due curve qui sotto stanno allo stesso chilometraggio.",
  "duel.curves_h": "Le due curve",
  "duel.curves_p": "Quota di prezzo che l'auto conserva invecchiando, a partire dall'esemplare più recente con campione ({year}). Sono prezzi richiesti in annunci attivi, non vendite concluse: misurano quanto il mercato chiede oggi a ogni età, non quanto ha incassato un proprietario.",
  "duel.chart_alt": "Quota di prezzo conservata per età, {question}",
  "duel.chart_axis": "anni di età",
  "duel.gap_h": "E oggi, chi chiede di più?",
  "duel.gap_p": "La domanda di prima riguardava il ritmo del calo, questa riguarda il prezzo in vetrina. Ogni riga confronta {a_subj} e {b_subj} <b>della stessa età e con gli stessi chilometri</b>: il chilometraggio entra nell'adattamento, quindi la distanza qui sotto non è {mixup}.",
  "duel.gap_th_age": "Età",
  "duel.gap_th_diff": "{a} rispetto a {b}",
  "duel.gap_th_ci": "Intervallo (95 %)",
  "duel.gap_th_read": "Lettura",
  "duel.gap_read_none": "indistinguibile",
  "duel.gap_read_more": "{side} chiede di più",
  "duel.more": "il {pct} in più",
  "duel.less": "il {pct} in meno",
  "duel.drift": "Le due letture sono lo stesso conto visto da due lati: a {age_lo} anni {a_subj} chiede {gap_lo} {b_than}, a {age_hi} anni {gap_hi}. Con l'età il sovrapprezzo di {a} rispetto a {b} è quindi <b>{move}</b>, ed è esattamente quello che dice la distanza di {pp} all'anno, scritta in prezzo invece che in tasso. {tail}",
  "duel.move_up": "più alto",
  "duel.move_down": "più basso",
  "duel.drift_cheap": "Quello che tiene meglio il prezzo qui è anche il più economico all'acquisto: il vantaggio si somma, paghi meno oggi e perdi meno dopo.",
  "duel.drift_dear": "Quello che tiene meglio il prezzo qui è anche il più caro all'acquisto: il sovrapprezzo si paga comprando e torna indietro in svalutazione, e quanto a lungo tieni l'auto decide se conviene.",
  "duel.gap_note": "Valori dall'adattamento, non mediane grezze: le mediane per età mescolano allestimenti e chilometraggi diversi dalle due parti.",
  "duel.table_h": "I due lati, uno accanto all'altro",
  "duel.tr_n": "Annunci nell'adattamento",
  "duel.tr_price": "Prezzo richiesto mediano",
  "duel.tr_km": "Chilometraggio (mediana)",
  "duel.tr_rate": "Perdita per anno di età",
  "duel.tr_keep5": "Valore conservato a 5 anni",
  "duel.tr_keep10": "Valore conservato a 10 anni",
  "duel.cta_h": "E la TUA {brand} {model}?",
  "duel.cta_p": "Queste sono le curve del modello. Incolla il link del tuo annuncio e ti diciamo il valore giusto di quella singola auto, con la tua motorizzazione, i tuoi chilometri e il tuo cambio.",
  "duel.cta_btn": "Valuta la mia auto&nbsp;&nbsp;→",
  "duel.links_model": "Tutte le {brand} {model}",
  "duel.links_others": "Altri modelli",
  "duel.links_dep": "Curva di svalutazione",
  "duel.faq1_q": "Su una {brand} {model} perde valore più in fretta {a_subj} o {b_subj}?",
  "duel.faq1_win": "Non come si dice di solito: su questo modello è {win_subj} a tenere meglio il prezzo. Su {n} annunci attivi di {source}, a pari chilometraggio, {a_subj} perde {a_pct} per anno di età e {b_subj} {b_pct}: {pp} di distanza all'anno {fav}.",
  "duel.faq1_draw": "Su questo modello la distanza non si distingue: {a_scap} perde {a_pct} per anno di età e {b_subj} {b_pct}, {pp} di distanza che stanno dentro la precisione della misura (±{ci}), su {n} annunci attivi di {source}.",
  "duel.faq2_q": "{brand} {model}: chiede di più {a_subj} o {b_subj}?",
  "duel.faq2_gap": "A {age} anni e a pari chilometraggio {a_subj} chiede {e} {b_than}. In grezzo, senza pareggiare i chilometri, la mediana richiesta è {a_price} ({a_km} mediani) contro {b_price} ({b_km}).",
  "duel.faq2_raw": "La mediana richiesta è {a_price} ({a_km} mediani) contro {b_price} ({b_km} mediani). Sono chilometraggi molto diversi, quindi la differenza di prezzo grezza non dipende solo {facet_of}.",
  "duel.ds_name": "Svalutazione di {brand} {model} per {facet}",
  "duel.ds_desc": "Svalutazione annua di una {brand} {model} {a_low} ({a_pct}) e {b_low} ({b_pct}), corretta per il chilometraggio, su {n} annunci attivi di {source} immatricolati tra il {from} e il {to}.",
  "duel.var_rate": "Svalutazione annua (%)",
  "duel.var_price": "Prezzo richiesto (EUR)",
  "duel.var_km": "Chilometraggio (km)",
  "duel.json_method": "log(prezzo) ~ età + log(chilometraggio) + lato + età×lato, minimi quadrati sugli annunci attivi",
  "duel.hub_eyebrow": "{n} MODELLI",
  "duel.hub_lede": "La risposta non vale per tutte le auto, vale modello per modello. Sui <b>{n} modelli</b> con abbastanza annunci attivi per adattare le due curve separatamente, {a_subj} tiene meglio il prezzo su <b>{awins}</b>, {b_subj} su <b>{bwins}</b>, e su <b>{draws}</b> la distanza non si separa dalla precisione della misura.",
  "duel.hub_desc": "Su {n} modelli con campione sufficiente, {a_subj} tiene meglio il prezzo su {awins} modelli e {b_subj} su {bwins}. Tassi per anno di età misurati sugli annunci attivi di {source}, con il chilometraggio controllato.",
  "duel.hub_prov": "Prezzo richiesto per età, {a_low} e {b_low} separatamente, con chilometraggio controllato",
  "duel.hub_table_h": "Modello per modello",
  "duel.hub_table_p": "Ordinato per la distanza tra le due curve. La colonna centrale è quello che la pagina del modello sviluppa: quanti punti percentuali all'anno separano i due lati e con che precisione sono stati misurati.",
  "duel.hub_th_side": "{side} / anno",
  "duel.hub_th_diff": "Distanza",
  "duel.hub_th_wins": "Tiene meglio",
  "duel.hub_th_n": "Annunci {a}/{b}",
  "duel.hub_draw": "Pari",
  "duel.hub_draw_h": "Perché anche «pari» è una risposta",
  "duel.hub_draw_p1": "Mezzo punto di distanza tra due curve adattate su qualche decina di annunci non è un risultato, è rumore con tre decimali. Per questo ogni riga porta la sua precisione, e un modello entra in questa tabella solo quando quella precisione è abbastanza stretta perché «pari» significhi <b>nessun vantaggio apprezzabile</b> e non <b>non siamo riusciti a vederlo</b>. I modelli il cui campione non basta per questa distinzione qui semplicemente non hanno una riga.",
  "duel.hub_draw_p2": "Il chilometraggio entra nell'adattamento dappertutto. Senza, la tabella misurerebbe soprattutto {mixup}, per poi chiamarlo {facet}.",
  "duel.hub_cta_h": "Sei indeciso tra due auto precise?",
  "duel.hub_cta_p": "Incolla il link di ogni annuncio e ti diciamo il valore giusto di entrambe, con motorizzazione, chilometri e allestimento di ciascun esemplare.",
  "duel.hub_cta_btn": "Valuta un annuncio&nbsp;&nbsp;→",
  "duel.pp": "{v} punti",
  "duel.pp_chip": "{v} p.p.",
};

const STRINGS_PT = {
  "footer.depreciation": "Desvalorização por modelo",
  "dep.crumb": "Desvalorização",
  "dep.hub_eyebrow": "CURVAS DE DESVALORIZAÇÃO · {n} MODELOS",
  "dep.hub_title": "Desvalorização de carros usados em {country}",
  "dep.hub_h1": "Que carros se desvalorizam mais depressa em {country}",
  "dep.hub_desc": "Que modelos perdem mais valor por ano em {country}, medido em anúncios ativos do {source}. {n} modelos com curva completa, valor retido aos cinco anos e a idade a partir da qual a matrícula deixa de mandar no preço.",
  "dep.hub_lede": "Desvalorização por ano de idade, medida nos preços pedidos de anúncios ativos do {source}. Só entram modelos com histórico que chegue para a curva significar alguma coisa: pelo menos {cells} anos com amostra e {span} anos entre o mais antigo e o mais recente.",
  "dep.hub_market": "A mediana do mercado é {mkt} ao ano. Acima disso, o carro custa-te mais a ter; abaixo, revendes com menos perda.",
  "dep.hub_band": "Entre os {from} e os {to} anos de idade, estes modelos perdem na mediana {a} ao ano ({na} modelos com amostra nesse troço); dos {cut} anos em diante, {b} ({nb} modelos).",
  "dep.hub_band_flat": "A percentagem não abranda com a idade. O que abranda é a fatura em euros: {a} de 12 000 € e {a} de 3 000 € não são a mesma conta.",
  "dep.hub_band_slower": "A percentagem abranda com a idade, mas menos do que se costuma dizer: são {gap} entre um troço e o outro.",
  "dep.hub_table_h": "Modelo a modelo",
  "dep.th_model": "Modelo",
  "dep.th_rate": "Por ano",
  "dep.th_keep5": "Valor aos 5 anos",
  "dep.th_half": "Metade do valor",
  "dep.th_cheap": "Um ano custa menos de {floor}",
  "dep.th_n": "Anúncios",
  "dep.th_span": "Histórico",
  "dep.hub_note": "«Metade do valor» é o tempo que o modelo leva, ao ritmo medido, até o preço pedido mediano ficar a metade. «Um ano custa menos de {floor}» é a idade a partir da qual mais um ano de matrícula vale menos de {floor} na curva ajustada; um travessão significa que isso não acontece antes dos {max_age} anos, ou seja a matrícula manda no preço em toda a gama que se compra. As curvas atravessam gerações: parte da queda é um modelo diferente, não mais idade.",
  "dep.hub_duels": "Esta tabela mede o modelo inteiro, com todas as versões juntas. Onde a amostra chega, separamos as curvas:",
  "dep.hub_prov": "Preço pedido mediano por ano de matrícula, ajuste log-linear",
  "dep.eyebrow": "CURVA DE DESVALORIZAÇÃO · {source}",
  "dep.title": "{brand} {model}: desvalorização por ano",
  "dep.desc": "Um {brand} {model} perde cerca de {rate} por cada ano de idade e mantém {keep_five} do valor ao fim de cinco anos, medido em {n} anúncios ativos do {source}. Curva completa, preço de cada ano de matrícula e onde a queda abranda.",
  "dep.h1": "Quanto se desvaloriza um {brand} {model}?",
  "dep.lede": "Medido nos preços pedidos de {n} anúncios ativos com matrícula entre {from} e {to}, um {brand} {model} perde cerca de <b>{rate} por cada ano de idade</b>{half}. {vs}",
  "dep.half_clause": ", ou seja metade do valor a cada <b>{half} anos</b>",
  "dep.vs_fast": "É mais rápido do que o mercado (mediana {mkt} ao ano).",
  "dep.vs_slow": "É mais devagar do que o mercado (mediana {mkt} ao ano): este modelo segura melhor o valor do que a média.",
  "dep.vs_mid": "Está em linha com o mercado (mediana {mkt} ao ano).",
  "dep.stat_year": "POR ANO",
  "dep.stat_year_s": "−{eur} por ano de idade",
  "dep.stat_age": "AOS {years} ANOS",
  "dep.stat_left": "−{eur}",
  "dep.stat_left_base": "−{eur} sobre {base}",
  "dep.prov_measure": "Preço pedido mediano por ano de matrícula, {from}–{to}",
  "dep.prov_extra": "Ajuste log-linear, R²={rsq}; valores em euros sobre {base}, o que a curva dá a um exemplar de {year}",
  "dep.curve_h": "A curva",
  "dep.curve_p": "Cada ponto é a mediana dos preços pedidos nessa idade e a linha é o ajuste log-linear de onde sai a percentagem acima{marks}. A queda é uma percentagem do que resta: muitos euros nos primeiros anos, poucos no fim, mesmo quando a percentagem não muda.",
  "dep.curve_marks": ", e as marcas verticais são as idades sobre as quais esta página faz uma afirmação",
  "dep.curve_caveat": "Uma ressalva que a curva não mostra: entre {from} e {to} o {brand} {model} mudou de geração mais do que uma vez, e um exemplar de cada ponta não é o mesmo carro com mais uns anos. Parte da queda é idade e desgaste, parte é um modelo diferente com outro equipamento. A série mede o que o mercado pede por cada ano de matrícula, não o envelhecimento de um carro concreto.",
  "dep.ladder_h": "Quanto custa um ano de idade",
  "dep.ladder_p": "A mesma curva lida ao contrário: o que pagas, em euros, por cada ano de matrícula mais recente. É esta coluna que decide se vale a pena esticar o orçamento por um ano a mais.",
  "dep.ladder_th_age": "Idade (matrícula)",
  "dep.ladder_th_cost": "Custo de +1 ano",
  "dep.ladder_th_price": "Preço na curva",
  "dep.ladder_note": "Valores da curva ajustada, não medianas em bruto: entre dois anos seguidos a mediana salta mais do que o passo que estamos a medir. As medianas em bruto estão na tabela por ano, mais abaixo.",
  "dep.age_n": "{n} anos",
  "dep.age_one": "1 ano",
  "dep.bend_h": "Há um ponto de inflexão?",
  "dep.bend_slows": "<b>Sim, aos {age} anos.</b> Até essa idade o {brand} {model} perde cerca de <b>{early} por ano</b>; a partir daí, <b>{late}</b>. Um exemplar já do lado direito dessa quebra custa-te menos por cada ano que o tiveres: é o troço onde comprar sai barato.",
  "dep.bend_speeds": "<b>Há uma quebra aos {age} anos, mas ao contrário do esperado:</b> antes disso o {brand} {model} perde cerca de <b>{early} por ano</b> e depois <b>{late}</b> — a percentagem acelera com a idade em vez de abrandar. Nos exemplares mais velhos a conta é outra: o preço já é baixo, por isso a perda em euros continua a encolher.",
  "dep.bend_none": "<b>Não, não em percentagem.</b> Testámos um ajuste com quebra em cada idade entre os {min_age} e os {max_age} anos e nenhum explica os preços do {brand} {model} melhor do que uma queda constante de {rate} ao ano (R²={rsq}).",
  "dep.bend_none_cand": " O melhor candidato ficava aos {age} anos — {early} ao ano antes, {late} depois — e essa diferença é do tamanho dos saltos que a mediana já dá entre dois anos seguidos.",
  "dep.bend_none_tail": " O que abranda é a fatura em euros, não a percentagem.",
  "dep.cheap_yes": "Onde isso deixa de doer dá para datar: a partir dos <b>{age} anos</b> — matrículas de {year} e mais antigas, à volta de {price} — cada ano de idade a mais custa menos de {floor}. É menos do que um jogo de pneus ou uma correia de distribuição: a partir daí o estado do carro pesa mais do que a matrícula, e é por aí que a escolha se decide.",
  "dep.cheap_no": "Aqui isso não chega a acontecer dentro do que alguém procura: aos {age} anos um ano de idade ainda vale cerca de {cost}, e só muito mais tarde desceria abaixo de {floor}. Neste modelo a matrícula manda no preço em toda a gama que se compra: esticar o orçamento por um ano mais recente continua a custar dinheiro a sério.",
  "dep.table_h": "Preço mediano por ano",
  "dep.th_year": "Ano",
  "dep.th_median": "Mediano (pedido)",
  "dep.th_listings": "Anúncios",
  "dep.th_gap": "Anos vs. {year}",
  "dep.th_vs": "vs. {year}",
  "dep.th_km": "Km mediano",
  "dep.cta_h": "Quanto vale o TEU {brand} {model} hoje?",
  "dep.cta_p": "Esta curva é do modelo. Cola o link do teu anúncio e dizemos o valor justo do teu carro concreto, com os teus quilómetros e a tua versão.",
  "dep.cta_btn": "Avaliar o meu carro&nbsp;&nbsp;→",
  "dep.links_model": "{brand} {model} por ano",
  "dep.links_others": "Desvalorização de outros modelos",
  "dep.faq1_q": "Quanto se desvaloriza um {brand} {model} por ano?",
  "dep.faq1_a": "Cerca de {rate} do valor restante por cada ano de idade, medido nos preços pedidos de {n} anúncios ativos de {brand} {model} com matrícula entre {from} e {to} no {source} em {country}. A percentagem é constante, mas em euros a perda é muito maior nos primeiros anos.",
  "dep.faq2_q": "Quanto vale um {brand} {model} ao fim de cinco anos?",
  "dep.faq2_a": "Ao ritmo medido, um {brand} {model} mantém cerca de {keep} do valor ao fim de cinco anos. Sobre os {base} que a curva dá a um exemplar de {year}, isso são cerca de {eur} perdidos.",
  "dep.faq3_q": "Em quantos anos um {brand} {model} perde metade do valor?",
  "dep.faq3_a": "Cerca de {half} anos, ao ritmo de {rate} ao ano medido nos anúncios ativos do {source}. É a mesma taxa dita de outra maneira: a cada {half} anos de idade o preço pedido mediano fica a metade.",
  "dep.faq4_q": "A partir de que idade um {brand} {model} deixa de perder valor?",
  "dep.faq4_bend": "A queda abranda aos {age} anos: até lá são cerca de {early} ao ano, depois {late}.",
  "dep.faq4_flat": "Nunca deixa de perder, mas deixa de doer. Em percentagem a queda mantém-se em cerca de {rate} ao ano em toda a série medida: não há idade a partir da qual a percentagem trave.",
  "dep.faq4_cheap": " Em euros é outra história: a partir dos {age} anos (matrículas de {year} e mais antigas) cada ano de idade custa menos de {floor}, e aí o estado do carro pesa mais do que o ano.",
  "dep.faq4_nocheap": " Em euros a perda encolhe, mas devagar: aos {age} anos um ano de idade ainda custa cerca de {cost}.",
  "dep.faq5_q": "O {brand} {model} desvaloriza mais do que a média?",
  "dep.faq5_a": "A mediana do mercado de usados que medimos é {mkt} por ano de idade. O {brand} {model} está nos {rate}. {verdict}",
  "dep.above": "Está acima da média.",
  "dep.below": "Está abaixo da média.",
  "dep.inline": "Está em linha com a média.",
  "dep.ds_name": "Desvalorização de {brand} {model} em {country}",
  "dep.ds_desc": "Preço pedido mediano de {brand} {model} por ano de matrícula ({from}–{to}), desvalorização anual de {rate} e custo em euros de cada ano de idade, a partir de {n} anúncios ativos do {source}.",
  "dep.var_price": "Preço pedido (EUR)",
  "dep.var_rate": "Desvalorização anual (%)",
  "dep.var_cost": "Custo de um ano de idade (EUR)",
  "dep.json_method": "log(preço) ~ ano de matrícula, mínimos quadrados sobre os preços medianos de cada ano com pelo menos cinco anúncios",
  "dep.json_ladder_note": "Valores da curva ajustada, não medianas em bruto.",
  "dep.json_bend_note": "Um ajuste com quebra foi testado em cada idade entre os {min_age} e os {max_age} anos e só é publicado quando bate claramente a taxa constante.",
  "dep.json_hub_note": "Entra um modelo com pelo menos {cells} anos com amostra, {span} anos de histórico e um ajuste que explique mesmo os pontos.",
  "dep.chart_alt": "Preço mediano pedido por idade do carro, com a curva ajustada",
  "dep.chart_axis": "anos de idade",
  "dep.chart_bend": "quebra aos {age} anos",
  "dep.chart_cheap": "{floor}/ano aos {age}",
  "dep.chart_tip": "{age} anos ({year}): {price} · {n} anúncios",
  "duel.fuel_crumb": "Diesel ou gasolina",
  "duel.fuel_eyebrow": "DIESEL VS. GASOLINA",
  "duel.fuel_question": "diesel ou gasolina",
  "duel.fuel_hub_h1": "Diesel ou gasolina: onde a escolha muda mesmo o preço",
  "duel.fuel_hub_title": "Diesel ou gasolina: qual segura melhor o preço, modelo a modelo",
  "duel.fuel_choice": "a escolha do combustível",
  "duel.fuel_facet": "combustível",
  "duel.fuel_facet_of": "do combustível",
  "duel.fuel_mixup": "o facto de os diesels à venda andarem muito mais",
  "duel.fuel_a": "Diesel",
  "duel.fuel_b": "Gasolina",
  "duel.fuel_a_low": "diesel",
  "duel.fuel_b_low": "a gasolina",
  "duel.fuel_a_subj": "o diesel",
  "duel.fuel_b_subj": "a gasolina",
  "duel.fuel_a_scap": "O diesel",
  "duel.fuel_b_scap": "A gasolina",
  "duel.fuel_a_fav": "a favor do diesel",
  "duel.fuel_b_fav": "a favor da gasolina",
  "duel.fuel_a_chip": "DIESEL",
  "duel.fuel_b_chip": "GASOLINA",
  "duel.fuel_b_than": "do que a gasolina",
  "duel.gear_crumb": "Manual ou automática",
  "duel.gear_eyebrow": "CAIXA MANUAL VS. AUTOMÁTICA",
  "duel.gear_question": "caixa manual ou automática",
  "duel.gear_hub_h1": "Manual ou automática: onde a caixa muda mesmo o preço",
  "duel.gear_hub_title": "Caixa manual ou automática: qual segura melhor o preço, modelo a modelo",
  "duel.gear_choice": "a escolha da caixa",
  "duel.gear_facet": "caixa",
  "duel.gear_facet_of": "da caixa",
  "duel.gear_mixup": "o facto de os automáticos à venda serem mais recentes e andarem muito menos",
  "duel.gear_a": "Manual",
  "duel.gear_b": "Automática",
  "duel.gear_a_low": "com caixa manual",
  "duel.gear_b_low": "com caixa automática",
  "duel.gear_a_subj": "a caixa manual",
  "duel.gear_b_subj": "a caixa automática",
  "duel.gear_a_scap": "A caixa manual",
  "duel.gear_b_scap": "A caixa automática",
  "duel.gear_a_fav": "a favor da caixa manual",
  "duel.gear_b_fav": "a favor da caixa automática",
  "duel.gear_a_chip": "MANUAL",
  "duel.gear_b_chip": "AUTOMÁTICA",
  "duel.gear_b_than": "do que a caixa automática",
  "duel.title": "{brand} {model}: {question} segura melhor o preço?",
  "duel.h1": "{brand} {model}: {question} segura melhor o preço?",
  "duel.desc": "Num {brand} {model}, {a_subj} perde {a_pct} por ano de idade e {b_subj} {b_pct}, medido em {n} anúncios ativos do {source} com a quilometragem igualada. {tail}",
  "duel.desc_win": "A diferença: {pp} por ano {fav}.",
  "duel.desc_draw": "A diferença não se distingue da margem da medição.",
  "duel.lede": "Ajustámos as duas curvas em separado sobre <b>{n} anúncios ativos</b> de {brand} {model} ({an} {a_low}, {bn} {b_low}, matrículas de {from} a {to}), com a quilometragem igualada. {a_scap} perde <b>{a_pct} por ano de idade</b>, {b_subj} <b>{b_pct}</b>.",
  "duel.stat_diff": "DIFERENÇA",
  "duel.stat_side": "por ano · {n} anúncios",
  "duel.stat_ci": "±{ci} · {verdict}",
  "duel.draw_word": "indistinguível",
  "duel.prov_measure": "Preço pedido de {brand} {model}, {a_low} e {b_low} em separado ({from}–{to})",
  "duel.prov_extra": "Ajuste log-linear com quilometragem controlada, R²={rsq}",
  "duel.answer_h": "A resposta",
  "duel.verdict_win": "<b>Neste modelo, {win_subj} segura melhor o preço.</b> Com a quilometragem igualada, {a_subj} perde <b>{a_pct} por ano de idade</b> e {b_subj} <b>{b_pct}</b> — uma diferença de <b>{pp} por ano</b> {fav} (intervalo de 95%: {lo} a {hi}). Ao fim de cinco anos são {keep_win} do preço mantidos contra {keep_lose}.",
  "duel.verdict_draw": "<b>Neste modelo, {choice} não decide a desvalorização.</b> {a_scap} perde {a_pct} por ano de idade e {b_subj} {b_pct}, e a diferença de {pp} cabe dentro da margem da própria medição (±{ci}). Não é «não sabemos»: a amostra chega para dizer que, se existe vantagem, ela é menor do que {bound} por ano, pouco ao lado do que separa dois exemplares do mesmo ano.",
  "duel.mileage_p": "A comparação directa das medianas não responde a isto: no {brand} {model}, {a_subj} à venda tem {a_km} medianos e {b_subj} {b_km}, e uma curva ajustada sem contar com isso mede a mistura de quilometragens e chama-lhe {facet}. Por isso o ajuste usa idade <b>e</b> quilometragem, e as duas curvas abaixo estão à mesma quilometragem.",
  "duel.curves_h": "As duas curvas",
  "duel.curves_p": "Percentagem do preço mantida à medida que o carro envelhece, a partir do exemplar mais novo com amostra ({year}). São preços pedidos em anúncios ativos, não vendas fechadas: medem o que o mercado pede hoje por cada idade, não o que um dono concreto recebeu.",
  "duel.chart_alt": "Percentagem do preço mantida por idade, {question}",
  "duel.chart_axis": "anos de idade",
  "duel.gap_h": "E hoje, qual pede mais?",
  "duel.gap_p": "A pergunta anterior era sobre o ritmo da queda; esta é sobre o preço no balcão. Cada linha compara {a_subj} e {b_subj} <b>da mesma idade e com os mesmos quilómetros</b>: a quilometragem entra no ajuste, por isso a diferença abaixo já não é {mixup}.",
  "duel.gap_th_age": "Idade",
  "duel.gap_th_diff": "{a} vs. {b}",
  "duel.gap_th_ci": "Intervalo (95%)",
  "duel.gap_th_read": "Leitura",
  "duel.gap_read_none": "indistinguível",
  "duel.gap_read_more": "{side} pede mais",
  "duel.more": "mais {pct}",
  "duel.less": "menos {pct}",
  "duel.drift": "As duas leituras são a mesma conta vista de dois lados: aos {age_lo} anos {a_subj} pede {gap_lo} {b_than}, aos {age_hi} anos {gap_hi}. Com a idade, o prémio de {a} sobre {b} fica <b>{move}</b>, que é exactamente o que a diferença de {pp} por ano diz, escrita em preço em vez de em taxa. {tail}",
  "duel.move_up": "mais alto",
  "duel.move_down": "mais baixo",
  "duel.drift_cheap": "O que segura melhor o preço é também o mais barato à partida, por isso a vantagem soma-se: pagas menos hoje e perdes menos depois.",
  "duel.drift_dear": "O que segura melhor o preço é também o mais caro à partida: o prémio paga-se na compra e devolve-se em desvalorização, e quanto tempo ficas com o carro decide se compensa.",
  "duel.gap_note": "Valores do ajuste, não medianas em bruto: as medianas por idade misturam versões e quilometragens diferentes de cada lado.",
  "duel.table_h": "Os dois lados, lado a lado",
  "duel.tr_n": "Anúncios ativos no ajuste",
  "duel.tr_price": "Preço pedido mediano",
  "duel.tr_km": "Quilometragem mediana",
  "duel.tr_rate": "Perda por ano de idade",
  "duel.tr_keep5": "Mantém ao fim de 5 anos",
  "duel.tr_keep10": "Mantém ao fim de 10 anos",
  "duel.cta_h": "E o TEU {brand} {model}?",
  "duel.cta_p": "Estas são as curvas do modelo. Cola o link do teu anúncio e dizemos o valor justo desse carro concreto, com a tua versão, os teus quilómetros e a tua caixa.",
  "duel.cta_btn": "Avaliar o meu carro&nbsp;&nbsp;→",
  "duel.links_model": "Todos os {brand} {model}",
  "duel.links_others": "Outros modelos",
  "duel.links_dep": "Curva de desvalorização",
  "duel.faq1_q": "Num {brand} {model}, o que desvaloriza mais depressa, {a_subj} ou {b_subj}?",
  "duel.faq1_win": "Não da forma que se costuma dizer: neste modelo é {win_subj} que segura melhor o preço. Em {n} anúncios ativos do {source}, com a quilometragem igualada, {a_subj} perde {a_pct} por ano de idade e {b_subj} {b_pct} — {pp} de diferença por ano {fav}.",
  "duel.faq1_draw": "Neste modelo a diferença não é distinguível: {a_scap} perde {a_pct} por ano de idade e {b_subj} {b_pct}, uma distância de {pp} que cabe na margem da medição (±{ci}), sobre {n} anúncios ativos do {source}.",
  "duel.faq2_q": "{brand} {model}: qual pede mais, {a_subj} ou {b_subj}?",
  "duel.faq2_gap": "Aos {age} anos e com a mesma quilometragem, {a_subj} pede {e} {b_than}. Em bruto, sem igualar quilómetros, a mediana pedida é {a_price} ({a_km} medianos) contra {b_price} ({b_km}).",
  "duel.faq2_raw": "A mediana pedida é {a_price} ({a_km} medianos) contra {b_price} ({b_km} medianos). São quilometragens muito diferentes, por isso a diferença de preço em bruto não é só {facet_of}.",
  "duel.ds_name": "Desvalorização de {brand} {model} por {facet}",
  "duel.ds_desc": "Desvalorização anual de {brand} {model} {a_low} ({a_pct}) e {b_low} ({b_pct}), ajustada à quilometragem, sobre {n} anúncios ativos do {source} com matrícula entre {from} e {to}.",
  "duel.var_rate": "Desvalorização anual (%)",
  "duel.var_price": "Preço pedido (EUR)",
  "duel.var_km": "Quilometragem (km)",
  "duel.json_method": "log(preço) ~ idade + log(quilometragem) + lado + idade×lado, mínimos quadrados sobre anúncios ativos",
  "duel.hub_eyebrow": "{n} MODELOS",
  "duel.hub_lede": "A resposta não é uma para todos os carros, é uma por modelo. Nos <b>{n} modelos</b> com anúncios ativos suficientes para ajustar as duas curvas em separado, {a_subj} segura melhor o preço em <b>{awins}</b>, {b_subj} em <b>{bwins}</b>, e em <b>{draws}</b> a diferença não se distingue da margem da medição.",
  "duel.hub_desc": "Em {n} modelos com amostra suficiente, {a_subj} segura melhor o preço em {awins} modelos e {b_subj} em {bwins}. Taxas por ano de idade medidas em anúncios ativos do {source}, com a quilometragem controlada.",
  "duel.hub_prov": "Preço pedido por idade, {a_low} e {b_low} em separado, com a quilometragem controlada",
  "duel.hub_table_h": "Modelo a modelo",
  "duel.hub_table_p": "Ordenado pela distância entre as duas curvas. A coluna do meio é o que a página do modelo desenvolve: quantos pontos percentuais por ano separam os dois lados, e com que margem foram medidos.",
  "duel.hub_th_side": "{side} /ano",
  "duel.hub_th_diff": "Diferença",
  "duel.hub_th_wins": "Segura melhor",
  "duel.hub_th_n": "Anúncios {a}/{b}",
  "duel.hub_draw": "Empate",
  "duel.hub_draw_h": "Porque é que «empate» também é uma resposta",
  "duel.hub_draw_p1": "Uma diferença de meio ponto por ano entre duas curvas ajustadas em algumas dezenas de anúncios não é um resultado, é ruído com três casas decimais. Por isso cada linha traz a sua margem, e um modelo só entra nesta tabela quando essa margem é estreita o suficiente para que «empate» signifique <b>não há vantagem apreciável</b> e não <b>não conseguimos ver</b>. Os modelos onde a amostra não chega para essa distinção simplesmente não têm linha aqui.",
  "duel.hub_draw_p2": "A quilometragem entra no ajuste em todos eles. Sem isso, a tabela mediria sobretudo {mixup}, e chamaria a isso {facet}.",
  "duel.hub_cta_h": "Estás a escolher entre dois carros concretos?",
  "duel.hub_cta_p": "Cola o link de cada anúncio e dizemos o valor justo de cada um, com a motorização, os quilómetros e a versão de cada exemplar.",
  "duel.hub_cta_btn": "Avaliar um anúncio&nbsp;&nbsp;→",
  "duel.pp": "{v} pontos",
  "duel.pp_chip": "{v} pp",
};

const STRINGS = { de: STRINGS_DE, fr: STRINGS_FR, it: STRINGS_IT, pt: STRINGS_PT };

const ROUTES = {
  de: {
    depreciation: "wertverlust", depreciation_json: "wertverlust.json",
    duel_fuel: "diesel-oder-benziner", duel_fuel_json: "diesel-oder-benziner.json",
    duel_gear: "schaltung-oder-automatik", duel_gear_json: "schaltung-oder-automatik.json",
  },
  fr: {
    depreciation: "decote", depreciation_json: "decote.json",
    duel_fuel: "diesel-ou-essence", duel_fuel_json: "diesel-ou-essence.json",
    duel_gear: "manuelle-ou-automatique", duel_gear_json: "manuelle-ou-automatique.json",
  },
  it: {
    depreciation: "svalutazione", depreciation_json: "svalutazione.json",
    duel_fuel: "diesel-o-benzina", duel_fuel_json: "diesel-o-benzina.json",
    duel_gear: "manuale-o-automatico", duel_gear_json: "manuale-o-automatico.json",
  },
};

for (const [code, dict] of Object.entries(STRINGS)) registerStrings(code, dict);
for (const [code, routes] of Object.entries(ROUTES)) registerRoutes(code, routes);
registerNav([{ routeKey: "depreciation", labelKey: "footer.depreciation" }]);

const COST_FLOOR = 500;
const CHEAP_CAP_AGE = 15;
const LADDER_STOPS = [3, 5, 8, 10, 12, 15, 20];
const KEEP_STOPS = [3, 5, 8];
const CC_LICENCE = "https://creativecommons.org/licenses/by/4.0/";

const day = b => (b || "").slice(0, 10);
const r3 = x => (x == null || !isFinite(x)) ? null : Math.round(x * 1000) / 1000;
const r1 = x => (x == null || !isFinite(x)) ? null : Math.round(x * 10) / 10;

function decL(loc, x, digits = 1) {
  const v = Number(x).toFixed(digits);
  return loc.code === "pt" ? v : v.replace(".", ",");
}

function ppNum(loc, x) {
  return decL(loc, Math.abs(x) * 100, 1);
}

function pp(loc, x) {
  return t(loc, "duel.pp", { v: ppNum(loc, x) });
}

function ppChip(loc, x) {
  return t(loc, "duel.pp_chip", { v: ppNum(loc, x) });
}

function graph(nodes) {
  return { "@context": "https://schema.org", "@graph": nodes.filter(Boolean) };
}

function statBlock(items) {
  return `<div class="fc-stat-row">${items.filter(Boolean).map(item =>
    `<div class="fc-stat"><div class="k">${item.k}</div><div class="v">${item.v}</div>`
    + `${item.s ? `<div class="s">${item.s}</div>` : ""}</div>`).join("")}</div>`;
}

function eyebrow(text) {
  return `<div class="eyebrow"><span class="e-dot"></span><span class="mono">${text}</span></div>`;
}

function homeCrumb(loc) {
  return { name: t(loc, "common.crumb_home"), href: href(loc, "landing") };
}

function plainText(s) {
  return String(s).replace(/<[^>]+>/g, "").replace(/&nbsp;/g, " ").replace(/\s+/g, " ").trim();
}

function ageLabel(loc, years) {
  return years === 1 ? t(loc, "dep.age_one") : t(loc, "dep.age_n", { n: years });
}

export function depreciationPath(loc, slug) {
  return href(loc, "depreciation", slug);
}

export function duelPath(loc, kind, slug) {
  return href(loc, PAGE_ROUTE[kind], slug);
}

export function intlDepreciationRows(models, builtAt) {
  return depreciationSlugs(models || {}).map(slug => {
    const rec = models[slug];
    const fit = depreciationFit(rec);
    const av = depreciationAge(rec, fit, builtAt);
    return { slug, rec, fit, av };
  }).filter(r => r.fit && r.av);
}

export function intlDuelRows(models, kind, builtAt) {
  return duelSlugs(models || {}, kind, builtAt).map(slug => {
    const rec = models[slug];
    return { slug, rec, av: duel(rec, kind, builtAt) };
  }).filter(r => r.av);
}

export function duelCopy(loc, kind) {
  const p = k => t(loc, `duel.${kind}_${k}`);
  return {
    kind,
    crumb: p("crumb"), eyebrow: p("eyebrow"), question: p("question"),
    hubH1: p("hub_h1"), hubTitle: p("hub_title"), choice: p("choice"),
    facet: p("facet"), facetOf: p("facet_of"), mixup: p("mixup"), than: p("b_than"),
    a: { lbl: p("a"), low: p("a_low"), subj: p("a_subj"), scap: p("a_scap"),
         fav: p("a_fav"), chip: p("a_chip"), json: DUELS[kind].a.json },
    b: { lbl: p("b"), low: p("b_low"), subj: p("b_subj"), scap: p("b_scap"),
         fav: p("b_fav"), chip: p("b_chip"), json: DUELS[kind].b.json },
  };
}

export function intlModelCurveLinks(loc, models, rec, slug, builtAt) {
  const links = [];
  if (depreciationOk(rec)) {
    const fit = depreciationFit(rec);
    const av = fit ? depreciationAge(rec, fit, builtAt) : null;
    if (fit && av) {
      links.push(`<a href="${depreciationPath(loc, slug)}">${t(loc, "duel.links_dep")}</a>`);
    }
  }
  for (const kind of KINDS) {
    if (!publishedDuel(models || {}, slug, rec, builtAt, kind)) continue;
    if (!duel(rec, kind, builtAt)) continue;
    links.push(`<a href="${duelPath(loc, kind, slug)}">${escapeHtml(duelCopy(loc, kind).question)}</a>`);
  }
  if (!links.length) return "";
  return `<section class="section fc-wrap" style="padding-top:0;">
      <p class="fc-p">${links.join(" &middot; ")}</p>
    </section>`;
}

function depChart(loc, av, { w = 640, h = 240, color = GREEN } = {}) {
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
    + `<title>${escapeHtml(t(loc, "dep.chart_tip", {
        age: p.age, year: p.y, price: fmtEurL(loc, p.fm), n: fmtNumL(loc, p.n),
      }))}</title></circle>`).join("");
  const ticks = [0, 0.5, 1].map(f => {
    const v = top * f;
    return `<line x1="${padL}" x2="${w - padR}" y1="${Y(v).toFixed(1)}" y2="${Y(v).toFixed(1)}" class="c-grid"/>`
      + `<text x="${padL + 2}" y="${(Y(v) - 5).toFixed(1)}" text-anchor="start" class="c-ax">${escapeHtml(fmtEurL(loc, Math.round(v)))}</text>`;
  }).join("");
  const step = Math.max(1, Math.ceil((a1 - a0) / 6));
  let xlab = "";
  for (let a = Math.ceil(a0); a <= a1; a += step) {
    const anchor = a - a0 < step / 2 ? "start" : "middle";
    xlab += `<text x="${X(a).toFixed(1)}" y="${h - 17}" text-anchor="${anchor}" class="c-ax">${a}</text>`;
  }
  xlab += `<text x="${w - padR}" y="${h - 5}" text-anchor="end" class="c-ax">${escapeHtml(t(loc, "dep.chart_axis"))}</text>`;
  const mark = (age, label, row) => {
    if (age == null || age < a0 || age > a1) return "";
    const x = X(age), anchor = x > w - 90 ? "end" : x < padL + 70 ? "start" : "middle";
    return `<line x1="${x.toFixed(1)}" x2="${x.toFixed(1)}" y1="${padT - 6}" y2="${Y(0).toFixed(1)}" class="c-mark"/>`
      + `<text x="${x.toFixed(1)}" y="${padT - (row ? 10 : 24)}" text-anchor="${anchor}" class="c-marklab">${escapeHtml(label)}</text>`;
  };
  const marks = (av.bend ? mark(av.bend.age, t(loc, "dep.chart_bend", { age: av.bend.age }), 0) : "")
    + (av.cheapFrom ? mark(av.cheapFrom.age, t(loc, "dep.chart_cheap", {
        floor: fmtEurL(loc, av.costFloor), age: av.cheapFrom.age }), 1) : "");
  return `<svg class="fc-chart" viewBox="0 0 ${w} ${h}" role="img"`
    + ` aria-label="${escapeHtml(t(loc, "dep.chart_alt"))}">${ticks}`
    + `<path d="${area}" fill="${color}" opacity="0.10"/>`
    + `<path d="${curve}" fill="none" stroke="${color}" stroke-width="2.2" stroke-linejoin="round"/>`
    + `${dots}${marks}${xlab}</svg>`;
}

function retentionChart(loc, av, C, { w = 640, h = 240 } = {}) {
  if (!av) return "";
  const padL = 34, padR = 14, padT = 22, padB = 34;
  const a0 = av.a0, a1 = av.a1;
  const X = a => padL + ((a - a0) / Math.max(1, a1 - a0)) * (w - padL - padR);
  const Y = v => padT + (1 - v) * (h - padT - padB);
  const curve = rate => {
    let d = "";
    const step = Math.max(0.25, (a1 - a0) / 120);
    for (let a = a0; a <= a1 + 1e-9; a += step) {
      d += `${d ? "L" : "M"}${X(a).toFixed(1)},${Y(Math.pow(1 - rate, a - a0)).toFixed(1)}`;
    }
    return d + `L${X(a1).toFixed(1)},${Y(Math.pow(1 - rate, a1 - a0)).toFixed(1)}`;
  };
  const ticks = [0, 0.25, 0.5, 0.75, 1].map(f =>
    `<line x1="${padL}" x2="${w - padR}" y1="${Y(f).toFixed(1)}" y2="${Y(f).toFixed(1)}" class="c-grid"/>`
    + `<text x="${padL - 5}" y="${(Y(f) + 4).toFixed(1)}" text-anchor="end" class="c-ax">${escapeHtml(fmtPctL(loc, f))}</text>`).join("");
  const step = Math.max(1, Math.ceil((a1 - a0) / 6));
  let xlab = "";
  for (let a = a0; a <= a1; a += step) {
    xlab += `<text x="${X(a).toFixed(1)}" y="${h - 13}" text-anchor="${a === a0 ? "start" : "middle"}" class="c-ax">${a}</text>`;
  }
  xlab += `<text x="${w - padR}" y="${h - 2}" text-anchor="end" class="c-ax">${escapeHtml(t(loc, "duel.chart_axis"))}</text>`;
  return `<svg class="fc-chart" viewBox="0 0 ${w} ${h}" role="img"`
    + ` aria-label="${escapeHtml(t(loc, "duel.chart_alt", { question: C.question }))}">${ticks}`
    + `<path d="${curve(av.a.r)}" fill="none" stroke="${GREEN}" stroke-width="2.4" stroke-linejoin="round"/>`
    + `<path d="${curve(av.b.r)}" fill="none" stroke="${AMBER}" stroke-width="2.4" stroke-dasharray="5 4" stroke-linejoin="round"/>`
    + `${xlab}`
    + `<text x="${w - padR}" y="${padT}" text-anchor="end" class="c-ax" fill="${GREEN}">— ${escapeHtml(C.a.lbl)}</text>`
    + `<text x="${w - padR}" y="${padT + 14}" text-anchor="end" class="c-ax" fill="${AMBER}">- - ${escapeHtml(C.b.lbl)}</text></svg>`;
}

function depCrumb(loc) {
  return { name: t(loc, "dep.crumb"), href: href(loc, "depreciation") };
}

function ladderAges(av) {
  return [...new Set([Math.ceil(av.minAge), ...LADDER_STOPS, Math.floor(av.maxAge)])]
    .filter(a => a >= av.minAge && a <= av.maxAge)
    .sort((a, b) => a - b);
}

function depVars(loc, rec, fit, av) {
  return {
    brand: escapeHtml(rec.b), model: escapeHtml(rec.m),
    country: loc.countryName, source: loc.source.name,
    n: fmtNumL(loc, rec.n), rate: fmtPctL(loc, fit.rate),
    from: fit.oldest.y, to: fit.newest.y,
    keep_five: fmtPctL(loc, Math.pow(1 - fit.rate, 5)),
  };
}

export function intlDepreciationJson(loc, rec, slug, fit, av, { host, builtAt }) {
  const base = `https://${host}`;
  const canonical = `${base}${depreciationPath(loc, slug)}`;
  return {
    source: "Carsbuyer",
    source_url: canonical,
    licence: t(loc, "json.licence"),
    measured: "asking_price",
    measured_note: t(loc, "json.measured_note", { source: loc.source.name }),
    method: t(loc, "dep.json_method"),
    collected_until: day(builtAt) || null,
    updated_at: builtAt || null,
    market: loc.country, currency: "EUR", language: loc.code,
    brand: rec.b, model: rec.m, slug,
    sample_size: rec.n,
    model_years: { from: fit.oldest.y, to: fit.newest.y },
    annual_depreciation_rate: r3(fit.rate),
    fit_r2: r3(fit.r2),
    half_life_years: av.halfLife ? r1(av.halfLife) : null,
    value_retained: KEEP_STOPS.concat([10]).map(y => ({
      after_years: y, share_of_value: r3(Math.pow(1 - fit.rate, y)),
    })),
    curve_reference: { age: av.base.age, model_year: av.base.year, price_eur: av.base.price },
    year_of_age_cost: {
      note: t(loc, "dep.json_ladder_note"),
      points: ladderAges(av).map(a => ({
        age: a, model_year: av.ref - a,
        cost_of_one_more_year_eur: av.at(a),
        curve_price_eur: Math.round(av.price(a)),
      })),
    },
    bend: av.bend
      ? { age: av.bend.age, rate_before: r3(av.bend.early), rate_after: r3(av.bend.late),
          direction: av.bend.dir }
      : { age: null, note: t(loc, "dep.json_bend_note", { min_age: BEND_FROM_AGE, max_age: BEND_TO_AGE }) },
    cheap_from: av.cheapFrom
      ? { age: av.cheapFrom.age, model_year: av.ref - av.cheapFrom.age,
          cost_of_one_more_year_eur: av.cheapFrom.cost,
          curve_price_eur: av.cheapFrom.price, threshold_eur: av.costFloor }
      : null,
    by_model_year: fit.cells.slice().sort((a, b) => b.y - a.y).map(c => ({
      model_year: c.y, sample_size: c.n,
      asking_price: { median: c.fm, p25: c.fl, p75: c.fh },
      mileage_km_median: c.km != null ? c.km : null,
    })),
    related: {
      model: `${base}${href(loc, "model", slug)}`,
      depreciation_index: `${base}${href(loc, "depreciation")}`,
      methodology: `${base}${href(loc, "metodologia")}`,
    },
  };
}

export function renderIntlDepreciationPage({ loc, host, rec, slug, fit, av, stats, builtAt }) {
  const canonical = `https://${host}${depreciationPath(loc, slug)}`;
  const v = depVars(loc, rec, fit, av);
  const newest = fit.newest, oldest = fit.oldest;
  const base = av.base.price, baseYear = av.base.year;
  const loseEur = yrs => Math.round(base * (1 - Math.pow(1 - fit.rate, yrs)));
  const keep = yrs => fmtPctL(loc, Math.pow(1 - fit.rate, yrs));
  const mkt = stats && stats.depMed;
  const vsMarket = mkt
    ? t(loc, fit.rate > mkt * 1.15 ? "dep.vs_fast" : fit.rate < mkt * 0.85 ? "dep.vs_slow" : "dep.vs_mid",
        { mkt: fmtPctL(loc, mkt) })
    : "";
  const half = av.halfLife ? t(loc, "dep.half_clause", { half: decL(loc, av.halfLife) }) : "";
  const pageYears = new Set(yearPageYears(rec));

  const ladder = ladderAges(av).map(a => `<tr>
      <td>${escapeHtml(ageLabel(loc, a))} <span class="mut">(${av.ref - a})</span></td>
      <td>${escapeHtml(fmtEurL(loc, av.at(a)))}</td>
      <td class="mut">${escapeHtml(fmtEurL(loc, Math.round(av.price(a))))}</td></tr>`).join("");

  const rows = fit.cells.slice().sort((a, b) => b.y - a.y).map(c => {
    const rel = c.fm / newest.fm - 1;
    const link = pageYears.has(c.y)
      ? `<a href="${href(loc, "model", slug)}/${c.y}">${c.y}</a>` : String(c.y);
    return `<tr><td>${link}</td>
      <td>${escapeHtml(fmtEurL(loc, c.fm))}</td>
      <td class="mut">${fmtNumL(loc, c.n)}</td>
      <td class="mut">${newest.y - c.y}</td>
      <td class="mut">${Math.abs(rel) < 0.005 ? "—" : escapeHtml(fmtPctL(loc, rel))}</td>
      <td class="mut">${c.km != null ? escapeHtml(fmtKmL(loc, c.km)) : "—"}</td></tr>`;
  }).join("");

  const bend = av.bend;
  const bendPara = bend
    ? t(loc, bend.dir === "slows" ? "dep.bend_slows" : "dep.bend_speeds", {
        age: bend.age, brand: v.brand, model: v.model,
        early: fmtPctL(loc, bend.early), late: fmtPctL(loc, bend.late),
      })
    : t(loc, "dep.bend_none", {
        min_age: BEND_FROM_AGE, max_age: BEND_TO_AGE, brand: v.brand, model: v.model,
        rate: v.rate, rsq: decL(loc, fit.r2, 2),
      })
      + (av.bendCandidate ? t(loc, "dep.bend_none_cand", {
          age: av.bendCandidate.age,
          early: fmtPctL(loc, av.bendCandidate.early),
          late: fmtPctL(loc, av.bendCandidate.late),
        }) : "")
      + t(loc, "dep.bend_none_tail");

  const cheapPara = av.cheapFrom
    ? t(loc, "dep.cheap_yes", {
        age: av.cheapFrom.age, year: av.ref - av.cheapFrom.age,
        price: fmtEurL(loc, av.cheapFrom.price), floor: fmtEurL(loc, av.costFloor),
      })
    : t(loc, "dep.cheap_no", {
        age: av.capAge, cost: fmtEurL(loc, av.capCost), floor: fmtEurL(loc, av.costFloor),
      });

  const duels = KINDS.filter(k => publishedDuel(null, slug, rec, builtAt, k));
  const duelLinks = duels.map(k =>
    ` · <a href="${duelPath(loc, k, slug)}">${escapeHtml(duelCopy(loc, k).crumb)}</a>`).join("");

  const body = crumbs([homeCrumb(loc), depCrumb(loc), { name: `${rec.b} ${rec.m}` }]) + `
    <div style="padding-top:14px;">
      <div class="side-card" style="max-width:680px;margin:0 auto;">
        ${eyebrow(escapeHtml(t(loc, "dep.eyebrow", { source: loc.source.name })))}
        <h1 class="fc-h1">${t(loc, "dep.h1", v)}</h1>
        <p class="lede" style="font-size:16px;margin:0 0 18px;">${t(loc, "dep.lede", Object.assign({}, v, { half, vs: vsMarket }))}</p>
        ${statBlock([
          { k: escapeHtml(t(loc, "dep.stat_year")), v: escapeHtml(v.rate),
            s: escapeHtml(t(loc, "dep.stat_year_s", { eur: fmtEurL(loc, loseEur(1)) })) },
          ...KEEP_STOPS.map((y, i) => ({
            k: escapeHtml(t(loc, "dep.stat_age", { years: y })),
            v: escapeHtml(keep(y)),
            s: escapeHtml(i === 0
              ? t(loc, "dep.stat_left_base", { eur: fmtEurL(loc, loseEur(y)), base: fmtEurL(loc, base) })
              : t(loc, "dep.stat_left", { eur: fmtEurL(loc, loseEur(y)) })),
          })),
        ])}
        ${intlProvenance(loc, {
          n: rec.n, builtAt,
          measure: t(loc, "dep.prov_measure", { from: oldest.y, to: newest.y }),
          measureId: "depreciation-rate",
        })}
        <p class="fc-prov mono">${escapeHtml(t(loc, "dep.prov_extra", {
          rsq: decL(loc, fit.r2, 2), base: fmtEurL(loc, base), year: baseYear,
        }))}</p>
      </div>
    </div>
    <section class="section fc-wrap">
      <h2 class="fc-h2">${t(loc, "dep.curve_h")}</h2>
      ${depChart(loc, av)}
      <p class="fc-p" style="margin-top:10px;">${t(loc, "dep.curve_p", {
        marks: (av.bend || av.cheapFrom) ? t(loc, "dep.curve_marks") : "",
      })}</p>
      <p class="fc-p">${t(loc, "dep.curve_caveat", v)}</p>
    </section>
    ${ladder ? `
    <section class="section fc-wrap">
      <h2 class="fc-h2">${t(loc, "dep.ladder_h")}</h2>
      <p class="fc-p">${t(loc, "dep.ladder_p")}</p>
      <div class="fc-scroll"><table class="fc-tbl">
        <thead><tr><th>${t(loc, "dep.ladder_th_age")}</th><th>${t(loc, "dep.ladder_th_cost")}</th><th>${t(loc, "dep.ladder_th_price")}</th></tr></thead>
        <tbody>${ladder}</tbody></table></div>
      <p class="fc-prov mono">${escapeHtml(t(loc, "dep.ladder_note"))}</p>
    </section>` : ""}
    <section class="section fc-wrap">
      <h2 class="fc-h2">${t(loc, "dep.bend_h")}</h2>
      <p class="fc-p">${bendPara}</p>
      <p class="fc-p">${cheapPara}</p>
    </section>
    <section class="section fc-wrap">
      <h2 class="fc-h2">${t(loc, "dep.table_h")}</h2>
      <div class="fc-scroll"><table class="fc-tbl">
        <thead><tr><th>${t(loc, "dep.th_year")}</th><th>${t(loc, "dep.th_median")}</th><th>${t(loc, "dep.th_listings")}</th><th>${t(loc, "dep.th_gap", { year: newest.y })}</th><th>${t(loc, "dep.th_vs", { year: newest.y })}</th><th>${t(loc, "dep.th_km")}</th></tr></thead>
        <tbody>${rows}</tbody></table></div>
    </section>
    <section class="section fc-wide">
      <div class="cta-banner">
        <div style="flex:1 1 360px;">
          <h2>${t(loc, "dep.cta_h", v)}</h2>
          <p>${t(loc, "dep.cta_p")}</p>
        </div>
        <a class="btn-bright" href="${href(loc, "avaliar")}">${t(loc, "dep.cta_btn")}</a>
      </div>
    </section>
    <section class="section fc-wrap" style="padding-bottom:70px;">
      <p class="fc-p"><a href="${href(loc, "model", slug)}">${t(loc, "dep.links_model", v)}</a>${duelLinks} · <a href="${href(loc, "depreciation")}">${t(loc, "dep.links_others")}</a> · <a href="${href(loc, "hub")}">${t(loc, "common.all_models")}</a></p>
    </section>`;

  const faqs = [
    [t(loc, "dep.faq1_q", v), plainText(t(loc, "dep.faq1_a", v))],
    [t(loc, "dep.faq2_q", v), plainText(t(loc, "dep.faq2_a", Object.assign({}, v, {
      keep: keep(5), base: fmtEurL(loc, base), year: baseYear, eur: fmtEurL(loc, loseEur(5)),
    })))],
  ];
  if (av.halfLife) {
    faqs.push([t(loc, "dep.faq3_q", v), plainText(t(loc, "dep.faq3_a", Object.assign({}, v, {
      half: decL(loc, av.halfLife),
    })))]);
  }
  faqs.push([t(loc, "dep.faq4_q", v), plainText(
    (bend && bend.dir === "slows"
      ? t(loc, "dep.faq4_bend", { age: bend.age, early: fmtPctL(loc, bend.early), late: fmtPctL(loc, bend.late) })
      : t(loc, "dep.faq4_flat", v))
    + (av.cheapFrom
      ? t(loc, "dep.faq4_cheap", { age: av.cheapFrom.age, year: av.ref - av.cheapFrom.age, floor: fmtEurL(loc, av.costFloor) })
      : t(loc, "dep.faq4_nocheap", { age: av.capAge, cost: fmtEurL(loc, av.capCost) })))]);
  if (mkt) {
    faqs.push([t(loc, "dep.faq5_q", v), plainText(t(loc, "dep.faq5_a", Object.assign({}, v, {
      mkt: fmtPctL(loc, mkt),
      verdict: t(loc, fit.rate > mkt ? "dep.above" : fit.rate < mkt ? "dep.below" : "dep.inline"),
    })))]);
  }

  return layout({
    title: t(loc, "dep.title", v),
    description: plainText(t(loc, "dep.desc", v)),
    body, zone: "all", nav: "precos", depositCount: null, index: true, host, locale: loc,
    canonical, altJson: `${canonical}.json`,
    jsonLd: graph([
      {
        "@type": "Dataset",
        "name": t(loc, "dep.ds_name", v),
        "description": plainText(t(loc, "dep.ds_desc", v)),
        "url": canonical,
        "inLanguage": loc.lang,
        "license": CC_LICENCE,
        "isAccessibleForFree": true,
        "creator": { "@type": "Organization", "name": "Carsbuyer", "url": `https://${host}${href(loc, "landing")}` },
        "temporalCoverage": `${oldest.y}/${newest.y}`,
        "dateModified": builtAt || undefined,
        "variableMeasured": [t(loc, "dep.var_price"), t(loc, "dep.var_rate"), t(loc, "dep.var_cost")],
        "distribution": [{ "@type": "DataDownload", "encodingFormat": "application/json",
                           "contentUrl": `${canonical}.json` }],
      },
      breadcrumbLd(host, [homeCrumb(loc), depCrumb(loc),
                          { name: `${rec.b} ${rec.m}`, href: depreciationPath(loc, slug) }]),
      faqLd(faqs),
    ]),
  });
}

export function intlDepreciationHubJson(loc, rows, { host, builtAt }) {
  const base = `https://${host}`;
  const canonical = `${base}${href(loc, "depreciation")}`;
  return {
    source: "Carsbuyer",
    source_url: canonical,
    licence: t(loc, "json.licence"),
    measured: "asking_price",
    measured_note: t(loc, "json.measured_note", { source: loc.source.name }),
    method: t(loc, "dep.json_method"),
    inclusion_note: t(loc, "dep.json_hub_note", { cells: CURVE_MIN_CELLS, span: CURVE_MIN_SPAN }),
    collected_until: day(builtAt) || null,
    updated_at: builtAt || null,
    market: loc.country, currency: "EUR", language: loc.code,
    models: rows.map(({ slug, rec, fit, av }) => ({
      slug, brand: rec.b, model: rec.m,
      url: `${base}${depreciationPath(loc, slug)}`,
      sample_size: rec.n,
      model_years: { from: fit.oldest.y, to: fit.newest.y },
      annual_depreciation_rate: r3(fit.rate),
      share_after_5_years: r3(Math.pow(1 - fit.rate, 5)),
      half_life_years: av.halfLife ? r1(av.halfLife) : null,
      cheap_from_age: av.cheapFrom ? av.cheapFrom.age : null,
    })),
    related: {
      models_index: `${base}${href(loc, "hub")}`,
      methodology: `${base}${href(loc, "metodologia")}`,
    },
  };
}

export function renderIntlDepreciationHub({ loc, host, rows, models, stats, builtAt }) {
  const canonical = `https://${host}${href(loc, "depreciation")}`;
  const sorted = rows.slice().sort((a, b) => b.fit.rate - a.fit.rate);
  const floor = fmtEurL(loc, COST_FLOOR);
  const tr = sorted.map(({ slug, rec, fit, av }) => `<tr>
      <td><a href="${depreciationPath(loc, slug)}" style="color:#177A47;font-weight:600;">${escapeHtml(rec.b)} ${escapeHtml(rec.m)}</a></td>
      <td>${escapeHtml(fmtPctL(loc, fit.rate))}</td>
      <td class="mut">${escapeHtml(fmtPctL(loc, Math.pow(1 - fit.rate, 5)))}</td>
      <td class="mut">${av.halfLife ? escapeHtml(ageLabel(loc, decL(loc, av.halfLife))) : "—"}</td>
      <td class="mut">${av.cheapFrom ? escapeHtml(ageLabel(loc, av.cheapFrom.age)) : "—"}</td>
      <td class="mut">${fmtNumL(loc, rec.n)}</td>
      <td class="mut">${escapeHtml(ageLabel(loc, fit.span))}</td></tr>`).join("");

  const yg = stats && stats.depYoung, od = stats && stats.depOld;
  const band = (yg && od) ? (() => {
    const a = fmtPctL(loc, yg.rate), b = fmtPctL(loc, od.rate);
    const verdict = od.rate >= yg.rate - 0.01
      ? t(loc, "dep.hub_band_flat", { a })
      : t(loc, "dep.hub_band_slower", { gap: pp(loc, yg.rate - od.rate) });
    return `<p class="fc-p">${t(loc, "dep.hub_band", {
      from: yg.from, to: yg.to, a, na: fmtNumL(loc, yg.models),
      cut: od.from, b, nb: fmtNumL(loc, od.models),
    })} ${verdict}</p>`;
  })() : "";

  const duelHubs = KINDS
    .filter(k => intlDuelRows(models, k, builtAt).length > 0)
    .map(k => `<a href="${href(loc, PAGE_ROUTE[k])}">${escapeHtml(duelCopy(loc, k).question)}</a>`);

  const sample = sorted.reduce((s, r) => s + (r.rec.n || 0), 0);
  const body = crumbs([homeCrumb(loc), { name: t(loc, "dep.crumb") }]) + `
    <div style="padding-top:14px;">
      <div class="side-card" style="max-width:680px;margin:0 auto;">
        ${eyebrow(escapeHtml(t(loc, "dep.hub_eyebrow", { n: fmtNumL(loc, sorted.length) })))}
        <h1 class="fc-h1">${t(loc, "dep.hub_h1", { country: loc.countryName })}</h1>
        <p class="lede" style="font-size:16px;margin:0 0 18px;">${t(loc, "dep.hub_lede", {
          source: loc.source.name, cells: CURVE_MIN_CELLS, span: CURVE_MIN_SPAN,
        })}</p>
        ${stats && stats.depMed ? `<p class="fc-p">${t(loc, "dep.hub_market", { mkt: fmtPctL(loc, stats.depMed) })}</p>` : ""}
        ${band}
        ${intlProvenance(loc, {
          n: sample, builtAt, measure: t(loc, "dep.hub_prov"), measureId: "depreciation-rate",
        })}
      </div>
    </div>
    <section class="section fc-wrap">
      <h2 class="fc-h2">${t(loc, "dep.hub_table_h")}</h2>
      <div class="fc-scroll"><table class="fc-tbl">
        <thead><tr><th>${t(loc, "dep.th_model")}</th><th>${t(loc, "dep.th_rate")}</th><th>${t(loc, "dep.th_keep5")}</th><th>${t(loc, "dep.th_half")}</th><th>${t(loc, "dep.th_cheap", { floor: escapeHtml(floor) })}</th><th>${t(loc, "dep.th_n")}</th><th>${t(loc, "dep.th_span")}</th></tr></thead>
        <tbody>${tr}</tbody></table></div>
      <p class="fc-prov mono">${escapeHtml(t(loc, "dep.hub_note", { floor, max_age: CHEAP_CAP_AGE }))}</p>
      ${duelHubs.length ? `<p class="fc-p" style="margin-top:18px;">${t(loc, "dep.hub_duels")} ${duelHubs.join(" · ")}</p>` : ""}
      <p class="fc-p" style="margin-top:18px;"><a href="${href(loc, "hub")}">${t(loc, "common.all_models")}</a> · <a href="${href(loc, "metodologia")}">${t(loc, "footer.metodologia")}</a></p>
    </section>
    <div style="height:60px;"></div>`;

  return layout({
    title: t(loc, "dep.hub_title", { country: loc.countryName }),
    description: plainText(t(loc, "dep.hub_desc", {
      country: loc.countryName, source: loc.source.name, n: fmtNumL(loc, sorted.length),
    })),
    body, zone: "all", nav: "precos", depositCount: null, index: true, host, locale: loc,
    canonical, altJson: `${canonical}.json`,
    jsonLd: graph([
      {
        "@type": "Dataset",
        "name": t(loc, "dep.hub_title", { country: loc.countryName }),
        "description": plainText(t(loc, "dep.hub_desc", {
          country: loc.countryName, source: loc.source.name, n: fmtNumL(loc, sorted.length),
        })),
        "url": canonical,
        "inLanguage": loc.lang,
        "license": CC_LICENCE,
        "isAccessibleForFree": true,
        "creator": { "@type": "Organization", "name": "Carsbuyer", "url": `https://${host}${href(loc, "landing")}` },
        "dateModified": builtAt || undefined,
        "variableMeasured": [t(loc, "dep.var_rate"), t(loc, "dep.var_price")],
        "distribution": [{ "@type": "DataDownload", "encodingFormat": "application/json",
                           "contentUrl": `${canonical}.json` }],
      },
      breadcrumbLd(host, [homeCrumb(loc), { name: t(loc, "dep.crumb"), href: href(loc, "depreciation") }]),
    ]),
  });
}

function duelCrumb(loc, C, kind) {
  return { name: C.crumb, href: href(loc, PAGE_ROUTE[kind]) };
}

function duelVars(loc, rec, av, C) {
  return {
    brand: escapeHtml(rec.b), model: escapeHtml(rec.m),
    source: loc.source.name, country: loc.countryName,
    question: C.question, choice: C.choice, facet: C.facet, facet_of: C.facetOf, mixup: C.mixup,
    a: C.a.lbl, b: C.b.lbl, a_low: C.a.low, b_low: C.b.low,
    a_subj: C.a.subj, b_subj: C.b.subj, a_scap: C.a.scap, b_scap: C.b.scap, b_than: C.than,
    a_pct: fmtPctL(loc, av.a.r, 1), b_pct: fmtPctL(loc, av.b.r, 1),
    an: fmtNumL(loc, av.a.n), bn: fmtNumL(loc, av.b.n), n: fmtNumL(loc, av.n),
    from: av.y0, to: av.y1, pp: pp(loc, av.diff), ci: pp(loc, av.ci),
    a_km: fmtKmL(loc, av.a.km), b_km: fmtKmL(loc, av.b.km),
    a_price: fmtEurL(loc, av.a.fm), b_price: fmtEurL(loc, av.b.fm),
    fav: av.decisive ? C[av.winner].fav : "",
  };
}

function gapWord(loc, est) {
  return t(loc, est >= 0 ? "duel.more" : "duel.less", { pct: fmtPctL(loc, Math.abs(est), 1) });
}

function signedPctL(loc, x) {
  return `${x >= 0 ? "+" : "−"}${fmtPctL(loc, Math.abs(x), 1)}`;
}

export function intlDuelJson(loc, rec, slug, kind, av, { host, builtAt }) {
  const base = `https://${host}`;
  const C = duelCopy(loc, kind);
  const canonical = `${base}${duelPath(loc, kind, slug)}`;
  const side = s => ({ sample_size: s.n, annual_depreciation_rate: r3(s.r),
                       median_asking_eur: s.fm, median_mileage_km: s.km });
  return {
    source: "Carsbuyer",
    source_url: canonical,
    licence: t(loc, "json.licence"),
    measured: "asking_price",
    measured_note: t(loc, "json.measured_note", { source: loc.source.name }),
    compares: kind === "fuel" ? "fuel_type" : "transmission",
    method: t(loc, "duel.json_method"),
    collected_until: day(builtAt) || null,
    updated_at: builtAt || null,
    market: loc.country, currency: "EUR", language: loc.code,
    brand: rec.b, model: rec.m, slug,
    model_years: { from: av.y0, to: av.y1 },
    [C.a.json]: side(av.a),
    [C.b.json]: side(av.b),
    rate_difference_pp_per_year: r1(av.diff * 100),
    rate_difference_ci95_half_width_pp: r1(av.ci * 100),
    distinguishable_at_95: av.decisive,
    holds_value_better: av.decisive ? C[av.winner].json : null,
    fit_r2: r3(av.r2),
    price_gap_by_age: av.gap.map(([age, est, half]) => ({
      age, model_year: av.ref - age,
      gap_pct: r3(est), ci95_half_width_pct: r3(half),
      distinguishable_at_95: (est - half > 0) || (est + half < 0),
    })),
    related: {
      model: `${base}${href(loc, "model", slug)}`,
      duel_index: `${base}${href(loc, PAGE_ROUTE[kind])}`,
      methodology: `${base}${href(loc, "metodologia")}`,
    },
  };
}

export function renderIntlDuelPage({ loc, host, rec, slug, kind, av, builtAt }) {
  const C = duelCopy(loc, kind);
  const canonical = `https://${host}${duelPath(loc, kind, slug)}`;
  const v = duelVars(loc, rec, av, C);
  const win = av.decisive ? C[av.winner] : null;
  const winSide = av[av.winner], loseSide = av[av.winner === "a" ? "b" : "a"];
  const keepL = (r, y) => fmtPctL(loc, Math.pow(1 - r, y));

  const verdict = av.decisive
    ? t(loc, "duel.verdict_win", Object.assign({}, v, {
        win_subj: win.subj,
        lo: ppNum(loc, Math.abs(av.diff) - av.ci), hi: pp(loc, Math.abs(av.diff) + av.ci),
        keep_win: keepL(winSide.r, 5), keep_lose: keepL(loseSide.r, 5),
      }))
    : t(loc, "duel.verdict_draw", Object.assign({}, v, {
        bound: pp(loc, av.ci + Math.abs(av.diff)),
      }));

  const gapRows = av.gap.map(([age, est, half]) => {
    const lo = est - half, hi = est + half;
    const sure = lo > 0 || hi < 0;
    const read = sure
      ? t(loc, "duel.gap_read_more", { side: escapeHtml(est > 0 ? C.a.lbl : C.b.lbl) })
      : t(loc, "duel.gap_read_none");
    return `<tr><td>${escapeHtml(ageLabel(loc, age))} <span class="mut">(${av.ref - age})</span></td>
      <td>${escapeHtml(signedPctL(loc, est))}</td>
      <td class="mut">${escapeHtml(signedPctL(loc, lo))} – ${escapeHtml(signedPctL(loc, hi))}</td>
      <td class="mut">${read}</td></tr>`;
  }).join("");

  const drift = av.gap.length >= 2 ? (() => {
    const [g0, e0] = av.gap[0];
    const [g1, e1] = av.gap[av.gap.length - 1];
    const winnerStartsCheaper = av.winner === "a" ? e0 < 0 : e0 > 0;
    const tail = !av.decisive ? ""
      : t(loc, winnerStartsCheaper ? "duel.drift_cheap" : "duel.drift_dear");
    return `<p class="fc-p">${t(loc, "duel.drift", Object.assign({}, v, {
      age_lo: g0, age_hi: g1, gap_lo: gapWord(loc, e0), gap_hi: gapWord(loc, e1),
      move: t(loc, e1 > e0 ? "duel.move_up" : "duel.move_down"), tail,
    }))}</p>`;
  })() : "";

  const gapBlock = av.gap.length ? `
    <section class="section fc-wrap">
      <h2 class="fc-h2">${t(loc, "duel.gap_h")}</h2>
      <p class="fc-p">${t(loc, "duel.gap_p", v)}</p>
      <div class="fc-scroll"><table class="fc-tbl">
        <thead><tr><th>${t(loc, "duel.gap_th_age")}</th><th>${t(loc, "duel.gap_th_diff", { a: escapeHtml(C.a.lbl), b: escapeHtml(C.b.lbl) })}</th><th>${t(loc, "duel.gap_th_ci")}</th><th>${t(loc, "duel.gap_th_read")}</th></tr></thead>
        <tbody>${gapRows}</tbody></table></div>
      ${drift}
      <p class="fc-prov mono">${escapeHtml(t(loc, "duel.gap_note"))}</p>
    </section>` : "";

  const sideTable = `
    <div class="fc-scroll"><table class="fc-tbl">
      <thead><tr><th>&nbsp;</th><th>${escapeHtml(C.a.lbl)}</th><th>${escapeHtml(C.b.lbl)}</th></tr></thead>
      <tbody>
        <tr><td>${t(loc, "duel.tr_n")}</td><td>${fmtNumL(loc, av.a.n)}</td><td>${fmtNumL(loc, av.b.n)}</td></tr>
        <tr><td>${t(loc, "duel.tr_price")}</td><td>${escapeHtml(fmtEurL(loc, av.a.fm))}</td><td>${escapeHtml(fmtEurL(loc, av.b.fm))}</td></tr>
        <tr><td>${t(loc, "duel.tr_km")}</td><td>${escapeHtml(fmtKmL(loc, av.a.km))}</td><td>${escapeHtml(fmtKmL(loc, av.b.km))}</td></tr>
        <tr><td>${t(loc, "duel.tr_rate")}</td><td>${escapeHtml(v.a_pct)}</td><td>${escapeHtml(v.b_pct)}</td></tr>
        <tr><td>${t(loc, "duel.tr_keep5")}</td><td>${escapeHtml(keepL(av.a.r, 5))}</td><td>${escapeHtml(keepL(av.b.r, 5))}</td></tr>
        <tr><td>${t(loc, "duel.tr_keep10")}</td><td>${escapeHtml(keepL(av.a.r, 10))}</td><td>${escapeHtml(keepL(av.b.r, 10))}</td></tr>
      </tbody></table></div>`;

  const otherKinds = KINDS.filter(k => k !== kind && publishedDuel(null, slug, rec, builtAt, k));
  const extraLinks = otherKinds.map(k =>
    ` · <a href="${duelPath(loc, k, slug)}">${escapeHtml(duelCopy(loc, k).crumb)}</a>`).join("")
    + (depreciationOk(rec) ? ` · <a href="${depreciationPath(loc, slug)}">${t(loc, "duel.links_dep")}</a>` : "");

  const body = crumbs([homeCrumb(loc), duelCrumb(loc, C, kind), { name: `${rec.b} ${rec.m}` }]) + `
    <div style="padding-top:14px;">
      <div class="side-card" style="max-width:680px;margin:0 auto;">
        ${eyebrow(`${escapeHtml(C.eyebrow)} · ${escapeHtml(loc.source.name)}`)}
        <h1 class="fc-h1">${t(loc, "duel.h1", v)}</h1>
        <p class="lede" style="font-size:16px;margin:0 0 18px;">${t(loc, "duel.lede", v)}</p>
        ${statBlock([
          { k: escapeHtml(C.a.chip), v: escapeHtml(v.a_pct),
            s: escapeHtml(t(loc, "duel.stat_side", { n: fmtNumL(loc, av.a.n) })) },
          { k: escapeHtml(C.b.chip), v: escapeHtml(v.b_pct),
            s: escapeHtml(t(loc, "duel.stat_side", { n: fmtNumL(loc, av.b.n) })) },
          { k: escapeHtml(t(loc, "duel.stat_diff")), v: escapeHtml(ppChip(loc, av.diff)),
            s: escapeHtml(t(loc, "duel.stat_ci", {
              ci: ppChip(loc, av.ci),
              verdict: av.decisive ? C[av.winner].fav : t(loc, "duel.draw_word"),
            })) },
        ])}
        ${intlProvenance(loc, {
          n: av.n, builtAt,
          measure: t(loc, "duel.prov_measure", v),
          measureId: kind === "fuel" ? "fuel-retention" : "gearbox-retention",
        })}
        <p class="fc-prov mono">${escapeHtml(t(loc, "duel.prov_extra", { rsq: decL(loc, av.r2, 2) }))}</p>
      </div>
    </div>
    <section class="section fc-wrap">
      <h2 class="fc-h2">${t(loc, "duel.answer_h")}</h2>
      <p class="fc-p">${verdict}</p>
      <p class="fc-p">${t(loc, "duel.mileage_p", v)}</p>
    </section>
    <section class="section fc-wrap">
      <h2 class="fc-h2">${t(loc, "duel.curves_h")}</h2>
      ${retentionChart(loc, av, C)}
      <p class="fc-p" style="margin-top:10px;">${t(loc, "duel.curves_p", { year: av.ref - av.a0 })}</p>
    </section>
    ${gapBlock}
    <section class="section fc-wrap">
      <h2 class="fc-h2">${t(loc, "duel.table_h")}</h2>
      ${sideTable}
    </section>
    <section class="section fc-wide">
      <div class="cta-banner">
        <div style="flex:1 1 360px;">
          <h2>${t(loc, "duel.cta_h", v)}</h2>
          <p>${t(loc, "duel.cta_p")}</p>
        </div>
        <a class="btn-bright" href="${href(loc, "avaliar")}">${t(loc, "duel.cta_btn")}</a>
      </div>
    </section>
    <section class="section fc-wrap" style="padding-bottom:70px;">
      <p class="fc-p"><a href="${href(loc, "model", slug)}">${t(loc, "duel.links_model", v)}</a>${extraLinks} · <a href="${href(loc, PAGE_ROUTE[kind])}">${t(loc, "duel.links_others")}</a> · <a href="${href(loc, "metodologia")}">${t(loc, "footer.metodologia")}</a></p>
    </section>`;

  const faqs = [
    [plainText(t(loc, "duel.faq1_q", v)),
     plainText(av.decisive
       ? t(loc, "duel.faq1_win", Object.assign({}, v, { win_subj: win.subj }))
       : t(loc, "duel.faq1_draw", v))],
    [plainText(t(loc, "duel.faq2_q", v)),
     plainText(av.gap.length
       ? t(loc, "duel.faq2_gap", Object.assign({}, v, {
           age: av.gap[0][0], e: gapWord(loc, av.gap[0][1]),
         }))
       : t(loc, "duel.faq2_raw", v))],
  ];

  const tail = av.decisive
    ? t(loc, "duel.desc_win", { fav: C[av.winner].fav, pp: v.pp })
    : t(loc, "duel.desc_draw");

  return layout({
    title: plainText(t(loc, "duel.title", v)),
    description: plainText(t(loc, "duel.desc", Object.assign({}, v, { tail }))),
    body, zone: "all", nav: "precos", depositCount: null, index: true, host, locale: loc,
    canonical, altJson: `${canonical}.json`,
    jsonLd: graph([
      {
        "@type": "Dataset",
        "name": plainText(t(loc, "duel.ds_name", v)),
        "description": plainText(t(loc, "duel.ds_desc", v)),
        "url": canonical,
        "inLanguage": loc.lang,
        "license": CC_LICENCE,
        "isAccessibleForFree": true,
        "creator": { "@type": "Organization", "name": "Carsbuyer", "url": `https://${host}${href(loc, "landing")}` },
        "temporalCoverage": `${av.y0}/${av.y1}`,
        "dateModified": builtAt || undefined,
        "variableMeasured": [t(loc, "duel.var_rate"), t(loc, "duel.var_price"), t(loc, "duel.var_km")],
        "distribution": [{ "@type": "DataDownload", "encodingFormat": "application/json",
                           "contentUrl": `${canonical}.json` }],
      },
      breadcrumbLd(host, [homeCrumb(loc), duelCrumb(loc, C, kind),
                          { name: `${rec.b} ${rec.m}`, href: duelPath(loc, kind, slug) }]),
      faqLd(faqs),
    ]),
  });
}

function duelTally(rows) {
  const aWins = rows.filter(r => r.av.decisive && r.av.winner === "a").length;
  const bWins = rows.filter(r => r.av.decisive && r.av.winner === "b").length;
  return { aWins, bWins, draws: rows.length - aWins - bWins };
}

export function intlDuelHubJson(loc, kind, rows, { host, builtAt }) {
  const base = `https://${host}`;
  const C = duelCopy(loc, kind);
  const canonical = `${base}${href(loc, PAGE_ROUTE[kind])}`;
  const tally = duelTally(rows);
  return {
    source: "Carsbuyer",
    source_url: canonical,
    licence: t(loc, "json.licence"),
    measured: "asking_price",
    measured_note: t(loc, "json.measured_note", { source: loc.source.name }),
    compares: kind === "fuel" ? "fuel_type" : "transmission",
    method: t(loc, "duel.json_method"),
    collected_until: day(builtAt) || null,
    updated_at: builtAt || null,
    market: loc.country, currency: "EUR", language: loc.code,
    sides: [C.a.json, C.b.json],
    holds_value_better_count: { [C.a.json]: tally.aWins, [C.b.json]: tally.bWins,
                                indistinguishable: tally.draws },
    models: rows.map(({ slug, rec, av }) => ({
      slug, brand: rec.b, model: rec.m,
      url: `${base}${duelPath(loc, kind, slug)}`,
      sample_size: av.n,
      [`${C.a.json}_annual_depreciation_rate`]: r3(av.a.r),
      [`${C.b.json}_annual_depreciation_rate`]: r3(av.b.r),
      rate_difference_pp_per_year: r1(av.diff * 100),
      rate_difference_ci95_half_width_pp: r1(av.ci * 100),
      distinguishable_at_95: av.decisive,
      holds_value_better: av.decisive ? C[av.winner].json : null,
    })),
    related: {
      depreciation_index: `${base}${href(loc, "depreciation")}`,
      models_index: `${base}${href(loc, "hub")}`,
      methodology: `${base}${href(loc, "metodologia")}`,
    },
  };
}

export function renderIntlDuelHub({ loc, host, kind, rows, models, builtAt }) {
  const C = duelCopy(loc, kind);
  const canonical = `https://${host}${href(loc, PAGE_ROUTE[kind])}`;
  const { aWins, bWins, draws } = duelTally(rows);
  const n = fmtNumL(loc, rows.length);

  const tr = rows.map(({ slug, rec, av }) => `<tr>
      <td><a href="${duelPath(loc, kind, slug)}" style="color:#177A47;font-weight:600;">${escapeHtml(rec.b)} ${escapeHtml(rec.m)}</a></td>
      <td>${escapeHtml(fmtPctL(loc, av.a.r, 1))}</td>
      <td>${escapeHtml(fmtPctL(loc, av.b.r, 1))}</td>
      <td class="mut">${escapeHtml(ppChip(loc, av.diff))} ±${escapeHtml(ppChip(loc, av.ci))}</td>
      <td>${av.decisive ? escapeHtml(C[av.winner].lbl) : `<span class="mut">${escapeHtml(t(loc, "duel.hub_draw"))}</span>`}</td>
      <td class="mut">${fmtNumL(loc, av.a.n)} / ${fmtNumL(loc, av.b.n)}</td></tr>`).join("");

  const others = KINDS.filter(k => k !== kind && intlDuelRows(models, k, builtAt).length > 0)
    .map(k => `<a href="${href(loc, PAGE_ROUTE[k])}">${escapeHtml(duelCopy(loc, k).crumb)}</a> · `).join("");

  const body = crumbs([homeCrumb(loc), { name: C.crumb }]) + `
    <div style="padding-top:14px;">
      <div class="side-card" style="max-width:680px;margin:0 auto;">
        ${eyebrow(`${escapeHtml(C.eyebrow)} · ${escapeHtml(t(loc, "duel.hub_eyebrow", { n }))}`)}
        <h1 class="fc-h1">${escapeHtml(C.hubH1)}</h1>
        <p class="lede" style="font-size:16px;margin:0 0 18px;">${t(loc, "duel.hub_lede", {
          n, a_subj: C.a.subj, b_subj: C.b.subj,
          awins: fmtNumL(loc, aWins), bwins: fmtNumL(loc, bWins), draws: fmtNumL(loc, draws),
        })}</p>
        ${intlProvenance(loc, {
          n: rows.reduce((s, r) => s + (r.av.n || 0), 0), builtAt,
          measure: t(loc, "duel.hub_prov", { a_low: C.a.low, b_low: C.b.low }),
          measureId: kind === "fuel" ? "fuel-retention" : "gearbox-retention",
        })}
      </div>
    </div>
    <section class="section fc-wrap">
      <h2 class="fc-h2">${t(loc, "duel.hub_table_h")}</h2>
      <p class="fc-p">${t(loc, "duel.hub_table_p")}</p>
      <div class="fc-scroll"><table class="fc-tbl">
        <thead><tr><th>${t(loc, "dep.th_model")}</th><th>${t(loc, "duel.hub_th_side", { side: escapeHtml(C.a.lbl) })}</th><th>${t(loc, "duel.hub_th_side", { side: escapeHtml(C.b.lbl) })}</th><th>${t(loc, "duel.hub_th_diff")}</th><th>${t(loc, "duel.hub_th_wins")}</th><th>${t(loc, "duel.hub_th_n", { a: escapeHtml(C.a.lbl.slice(0, 1)), b: escapeHtml(C.b.lbl.slice(0, 1)) })}</th></tr></thead>
        <tbody>${tr}</tbody></table></div>
    </section>
    <section class="section fc-wrap">
      <h2 class="fc-h2">${t(loc, "duel.hub_draw_h")}</h2>
      <p class="fc-p">${t(loc, "duel.hub_draw_p1")}</p>
      <p class="fc-p">${t(loc, "duel.hub_draw_p2", { mixup: C.mixup, facet: C.facet })}</p>
    </section>
    <section class="section fc-wide">
      <div class="cta-banner">
        <div style="flex:1 1 360px;">
          <h2>${t(loc, "duel.hub_cta_h")}</h2>
          <p>${t(loc, "duel.hub_cta_p")}</p>
        </div>
        <a class="btn-bright" href="${href(loc, "avaliar")}">${t(loc, "duel.hub_cta_btn")}</a>
      </div>
    </section>
    <section class="section fc-wrap" style="padding-bottom:70px;">
      <p class="fc-p">${others}<a href="${href(loc, "depreciation")}">${t(loc, "footer.depreciation")}</a> · <a href="${href(loc, "hub")}">${t(loc, "common.all_models")}</a> · <a href="${href(loc, "metodologia")}">${t(loc, "footer.metodologia")}</a></p>
    </section>`;

  const desc = plainText(t(loc, "duel.hub_desc", {
    n, a_subj: C.a.subj, b_subj: C.b.subj, source: loc.source.name,
    awins: fmtNumL(loc, aWins), bwins: fmtNumL(loc, bWins),
  }));
  return layout({
    title: C.hubTitle,
    description: desc,
    body, zone: "all", nav: "precos", depositCount: null, index: true, host, locale: loc,
    canonical, altJson: `${canonical}.json`,
    jsonLd: graph([
      {
        "@type": "Dataset",
        "name": C.hubTitle,
        "description": desc,
        "url": canonical,
        "inLanguage": loc.lang,
        "license": CC_LICENCE,
        "isAccessibleForFree": true,
        "creator": { "@type": "Organization", "name": "Carsbuyer", "url": `https://${host}${href(loc, "landing")}` },
        "dateModified": builtAt || undefined,
        "variableMeasured": [t(loc, "duel.var_rate"), t(loc, "duel.var_price"), t(loc, "duel.var_km")],
        "distribution": [{ "@type": "DataDownload", "encodingFormat": "application/json",
                           "contentUrl": `${canonical}.json` }],
      },
      breadcrumbLd(host, [homeCrumb(loc), { name: C.crumb, href: href(loc, PAGE_ROUTE[kind]) }]),
    ]),
  });
}

const SLUG_TAIL = /^\/([a-z0-9][a-z0-9-]*)(\.json)?$/;

function normaliseTail(rest) {
  let s = String(rest == null ? "" : rest);
  try {
    s = decodeURIComponent(s);
  } catch (_) {
    return null;
  }
  return s.replace(/\/+$/, "").toLowerCase();
}

function matchHubOrSlug(rest) {
  const s = normaliseTail(rest);
  if (s === null) return null;
  if (s === "") return { hub: true, json: false };
  const m = SLUG_TAIL.exec(s);
  return m ? { hub: false, slug: m[1], json: !!m[2] } : null;
}

function matchHubJson(rest) {
  const s = normaliseTail(rest);
  return s === "" ? { hub: true, json: true } : null;
}

function duelMatcher(kind) {
  return function match(rest) {
    const hit = matchHubOrSlug(rest);
    return hit ? Object.assign({ kind }, hit) : null;
  };
}

function duelHubJsonMatcher(kind) {
  return function match(rest) {
    const hit = matchHubJson(rest);
    return hit ? Object.assign({ kind }, hit) : null;
  };
}

async function handleDepreciation(ctx) {
  const { loc, url, models, builtAt, stats, params, helpers } = ctx;
  if (params.hub) {
    const rows = intlDepreciationRows(models, builtAt);
    if (!rows.length) return helpers.notFoundIntl();
    if (params.json) {
      return helpers.jsonResponse(intlDepreciationHubJson(loc, rows, { host: url.host, builtAt }));
    }
    return helpers.publicHtml(renderIntlDepreciationHub({
      loc, host: url.host, rows, models, stats, builtAt,
    }));
  }
  const rec = models[params.slug];
  if (!rec || !depreciationOk(rec)) return helpers.notFoundIntl();
  const fit = depreciationFit(rec);
  const av = fit ? depreciationAge(rec, fit, builtAt) : null;
  if (!fit || !av) return helpers.notFoundIntl();
  if (params.json) {
    return helpers.jsonResponse(intlDepreciationJson(loc, rec, params.slug, fit, av, {
      host: url.host, builtAt,
    }));
  }
  return helpers.publicHtml(renderIntlDepreciationPage({
    loc, host: url.host, rec, slug: params.slug, fit, av, stats, builtAt,
  }));
}

async function handleDuel(ctx) {
  const { loc, url, models, builtAt, params, helpers } = ctx;
  const kind = params.kind;
  if (params.hub) {
    const rows = intlDuelRows(models, kind, builtAt);
    if (!rows.length) return helpers.notFoundIntl();
    if (params.json) {
      return helpers.jsonResponse(intlDuelHubJson(loc, kind, rows, { host: url.host, builtAt }));
    }
    return helpers.publicHtml(renderIntlDuelHub({
      loc, host: url.host, kind, rows, models, builtAt,
    }));
  }
  const rec = models[params.slug];
  if (!rec || !publishedDuel(models, params.slug, rec, builtAt, kind)) return helpers.notFoundIntl();
  const av = duel(rec, kind, builtAt);
  if (!av) return helpers.notFoundIntl();
  if (params.json) {
    return helpers.jsonResponse(intlDuelJson(loc, rec, params.slug, kind, av, {
      host: url.host, builtAt,
    }));
  }
  return helpers.publicHtml(renderIntlDuelPage({
    loc, host: url.host, rec, slug: params.slug, kind, av, builtAt,
  }));
}

export const intlCurvesModule = registerIntlPages({
  id: "intl-curves",
  routes: [
    { routeKey: "depreciation", match: matchHubOrSlug, handle: handleDepreciation },
    { routeKey: "depreciation_json", match: matchHubJson, handle: handleDepreciation },
    ...KINDS.map(kind => ({
      routeKey: PAGE_ROUTE[kind], match: duelMatcher(kind), handle: handleDuel,
    })),
    ...KINDS.map(kind => ({
      routeKey: JSON_ROUTE[kind], match: duelHubJsonMatcher(kind), handle: handleDuel,
    })),
  ],
  navRouteKeys: ["depreciation"],
  navAvailable(loc, models, builtAt) {
    return intlDepreciationRows(models, builtAt).length ? ["depreciation"] : [];
  },
  sitemap(loc, models, builtAt) {
    const out = [];
    const dep = intlDepreciationRows(models, builtAt);
    if (dep.length) {
      out.push({ path: href(loc, "depreciation"), freq: "weekly", prio: "0.6" });
      for (const { slug } of dep) {
        out.push({ path: depreciationPath(loc, slug), freq: "weekly", prio: "0.55" });
      }
    }
    for (const kind of KINDS) {
      const rows = intlDuelRows(models, kind, builtAt);
      if (!rows.length) continue;
      out.push({ path: href(loc, PAGE_ROUTE[kind]), freq: "weekly", prio: "0.6" });
      for (const { slug } of rows) {
        out.push({ path: duelPath(loc, kind, slug), freq: "weekly", prio: "0.5" });
      }
    }
    return out;
  },
});
