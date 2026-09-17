import { escapeHtml, layout } from "./templates.js";
import { crumbs, breadcrumbLd, faqLd } from "./seo-pages.js";
import {
  t, href, registerStrings, registerRoutes, registerNav,
} from "./i18n.js";
import { registerIntlPages } from "./index.js";

const UPDATED = "2026-09-17";

const STRINGS_DE = {
  "g.crumb": "Ratgeber",
  "g.hub_title": "Ratgeber: ein Auto privat verkaufen in {country}",
  "g.hub_desc": "Was ein Privatverkäufer in {country} wissen muss: welchen Preis der Markt gerade trägt, wie lange ein Verkauf dauert und wie das Geld sicher ankommt.",
  "g.hub_h1": "Privat verkaufen: Preis und Bezahlung",
  "g.hub_lede": "Zwei Fragen entscheiden über den Verkauf, und beide haben nichts mit Papierkram zu tun: was der Markt gerade zahlt, und wie das Geld ankommt, ohne dass du Auto und Geld gleichzeitig verlierst.",
  "g.updated": "Stand: {date} · Ratgeber für Privatverkäufer · ersetzt keine Rechtsberatung",
  "g.faq_h": "Häufige Fragen",
  "g.other_h": "Weitere Ratgeber",
  "g.links": "Preise nach Modell und Baujahr",
  "g.cta": "Mein Auto bewerten",
};

const STRINGS_FR = {
  "g.crumb": "Guides",
  "g.hub_title": "Guides : vendre sa voiture entre particuliers en {country}",
  "g.hub_desc": "Ce qu'un vendeur particulier doit savoir en {country} : quel prix le marché porte en ce moment, combien de temps prend une vente et comment être payé sans risque.",
  "g.hub_h1": "Vendre entre particuliers : prix et paiement",
  "g.hub_lede": "Deux questions décident de la vente, et aucune ne concerne la paperasse : ce que le marché paie aujourd'hui, et comment l'argent arrive sans que tu perdes en même temps la voiture et le paiement.",
  "g.updated": "Mis à jour le {date} · guide pour vendeurs particuliers · ne remplace pas un conseil juridique",
  "g.faq_h": "Questions fréquentes",
  "g.other_h": "Autres guides",
  "g.links": "Prix par modèle et par année",
  "g.cta": "Estimer ma voiture",
};

const STRINGS_IT = {
  "g.crumb": "Guide",
  "g.hub_title": "Guide: vendere l'auto tra privati in {country}",
  "g.hub_desc": "Quello che un venditore privato in {country} deve sapere: che prezzo regge il mercato adesso, quanto tempo serve per vendere e come farsi pagare senza rischi.",
  "g.hub_h1": "Vendere tra privati: prezzo e pagamento",
  "g.hub_lede": "Due domande decidono la vendita, e nessuna riguarda le scartoffie: quanto paga il mercato oggi, e come arrivano i soldi senza perdere insieme l'auto e il pagamento.",
  "g.updated": "Aggiornato il {date} · guida per venditori privati · non sostituisce una consulenza legale",
  "g.faq_h": "Domande frequenti",
  "g.other_h": "Altre guide",
  "g.links": "Prezzi per modello e per anno",
  "g.cta": "Valuta la mia auto",
};

const STRINGS_PT = {
  "g.crumb": "Guias",
  "g.hub_title": "Guias: vender um carro entre particulares em {country}",
  "g.hub_desc": "O que um vendedor particular em {country} precisa de saber: que preço o mercado aguenta agora, quanto tempo demora uma venda e como receber o dinheiro em segurança.",
  "g.hub_h1": "Vender entre particulares: preço e pagamento",
  "g.hub_lede": "Duas perguntas decidem a venda, e nenhuma é de papelada: quanto paga o mercado hoje, e como chega o dinheiro sem perderes ao mesmo tempo o carro e o pagamento.",
  "g.updated": "Atualizado a {date} · guia para vendedores particulares · não substitui aconselhamento jurídico",
  "g.faq_h": "Perguntas frequentes",
  "g.other_h": "Outros guias",
  "g.links": "Preços por modelo e por ano",
  "g.cta": "Avaliar o meu carro",
};

registerStrings("de", STRINGS_DE);
registerStrings("fr", STRINGS_FR);
registerStrings("it", STRINGS_IT);
registerStrings("pt", STRINGS_PT);

registerRoutes("de", { guides: "ratgeber" });
registerRoutes("fr", { guides: "guides" });
registerRoutes("it", { guides: "guide" });

registerNav([{ routeKey: "guides", labelKey: "g.crumb" }]);

const GUIDES = {
  de: [
    {
      slug: "preis-finden",
      title: "Welchen Preis verlangen - und wie lange der Verkauf dauert",
      description: "Wie du den Preis für dein Auto aus dem Markt ableitest statt aus dem Kaufpreis, was die Spanne P25-P75 bedeutet und wann ein Preis nachgeben muss.",
      body: loc => `
      <p class="fc-p">Der häufigste Fehler beim Privatverkauf ist, vom eigenen Kaufpreis auszugehen. Der Markt kennt deinen Kaufpreis nicht. Er kennt nur, was für dasselbe Modell mit demselben Baujahr und ähnlichem Kilometerstand gerade verlangt wird - und genau das steht auf den <a href="${href(loc, "hub")}">Preisseiten</a>.</p>
      <h2 class="fc-h2">Vom Median aus, nicht vom Wunsch</h2>
      <p class="fc-p">Nimm den Median deines Modells im Baujahr deines Autos. Der Median ist der mittlere Preis: die Hälfte der Angebote liegt darüber, die Hälfte darunter. Er ist robuster als ein Durchschnitt, den ein einzelnes überteuertes Inserat nach oben zieht.</p>
      <p class="fc-p">Dazu gehört die Spanne P25-P75: die mittleren fünfzig Prozent des Marktes. Liegt dein Auto in Ausstattung, Zustand und Kilometerstand im Schnitt, gehörst du in die Mitte dieser Spanne. Ein Auto mit Scheckheft, zwei Sätzen Reifen und wenig Kilometern darf ans obere Ende, eines mit anstehender Hauptuntersuchung und Blechschäden gehört ans untere.</p>
      <h2 class="fc-h2">Die ersten zwei Wochen entscheiden</h2>
      <p class="fc-p">Ein Inserat bekommt die meiste Aufmerksamkeit in den ersten Tagen. Wer zu hoch einsteigt, verbrennt genau diese Tage und verkauft am Ende oft unter dem Preis, den ein realistischer Start gebracht hätte. Kommt in zwei Wochen keine ernsthafte Anfrage, ist es nicht der Markt - es ist der Preis.</p>
      <p class="fc-p">Senke dann in einem sichtbaren Schritt, nicht in fünf kleinen: mehrere Mini-Senkungen signalisieren, dass weitere folgen, und Interessenten warten darauf.</p>
      <h2 class="fc-h2">Was den Preis wirklich bewegt</h2>
      <p class="fc-p">Kilometerstand und Baujahr wiegen am schwersten, danach Getriebe und Kraftstoff. Sonderausstattung bringt beim Wiederverkauf fast nie das zurück, was sie neu gekostet hat. Eine frische Hauptuntersuchung ist dagegen bares Geld: sie nimmt dem Käufer ein Risiko ab, über das er sonst verhandelt.</p>`,
      faq: [
        ["Soll ich Verhandlungsspielraum einpreisen?", "Ein kleiner Puffer ist üblich, aber zehn Prozent über dem Median sind kein Puffer, sondern ein Filter, der die meisten Anfragen aussortiert, bevor sie entstehen."],
        ["Was, wenn mein Auto besser ist als der Durchschnitt?", "Dann gehört es ans obere Ende der Spanne P25-P75, nicht darüber. Was über der Spanne steht, vergleicht der Käufer mit Autos, die es dort auch gibt."],
        ["Wie lange dauert ein Privatverkauf?", "Ein marktgerecht bepreistes Alltagsauto geht meist in wenigen Wochen weg. Zieht es sich deutlich länger, liegt es fast immer am Preis oder an Fotos, auf denen das Auto nicht zu erkennen ist."],
      ],
    },
    {
      slug: "sicher-bezahlt-werden",
      title: "Sicher bezahlt werden: Übergabe ohne böse Überraschung",
      description: "Wie das Geld sicher ankommt: Übergabe in der Bankfiliale, Echtzeitüberweisung vor der Schlüsselübergabe, und die Maschen, an denen Privatverkäufer scheitern.",
      body: loc => `
      <p class="fc-p">Beim Privatverkauf gibt es einen Moment, in dem beide Seiten etwas hergeben müssen. Alles, was du für deine Sicherheit tun kannst, entscheidet sich davor - danach hilft nur noch die Polizei.</p>
      <h2 class="fc-h2">Die einfachste sichere Übergabe</h2>
      <p class="fc-p">Trefft euch in einer Bankfiliale zu Öffnungszeiten. Das Geld wird dort überwiesen oder eingezahlt und gezählt, Fahrzeugpapiere und Schlüssel wechseln erst den Besitzer, wenn die Gutschrift auf deinem Konto steht - nicht wenn ein Beleg gezeigt wird.</p>
      <p class="fc-p">Eine Echtzeitüberweisung ist in Sekunden da und nicht rückholbar. Eine normale Überweisung braucht Zeit; bis sie gebucht ist, gibst du nichts heraus. Ein Screenshot einer App ist kein Zahlungseingang.</p>
      <h2 class="fc-h2">Bargeld</h2>
      <p class="fc-p">Bargeld ist zulässig, aber es hat zwei Risiken: Falschgeld und der Weg nach Hause. Beides löst sich am Bankschalter, wo geprüft und direkt eingezahlt wird. Zähle nie auf einem Parkplatz, und fahre nicht mit einer größeren Summe im Auto durch die Gegend.</p>
      <h2 class="fc-h2">Maschen, die immer wieder funktionieren</h2>
      <ul class="fc-ul">
        <li class="fc-li"><b>Überzahlung.</b> Der "Käufer" überweist zu viel und bittet um Rücküberweisung der Differenz. Die erste Zahlung wird später zurückgeholt, deine Rücküberweisung nicht.</li>
        <li class="fc-li"><b>Transporteur oder Treuhand.</b> Ein angeblicher Spediteur oder Treuhanddienst will das Auto abholen und später zahlen. Solche Dienste gibt es im Privatverkauf nicht; es gibt nur dich, den Käufer und die Bank.</li>
        <li class="fc-li"><b>Anzahlung ohne Besichtigung.</b> Wer ein Auto kauft, das er nie gesehen hat, kauft es nicht - er sucht deine Kontodaten oder deinen Ausweis.</li>
      </ul>
      <h2 class="fc-h2">Probefahrt</h2>
      <p class="fc-p">Lass dir vor der Probefahrt den Ausweis zeigen und fahre mit. Halte im Kaufvertrag Datum, Preis, Kilometerstand und den Zustand fest, den ihr besprochen habt: das ist deine einzige schriftliche Version der Geschichte, wenn später jemand eine andere erzählt.</p>`,
      faq: [
        ["Echtzeitüberweisung oder Bargeld?", "Echtzeitüberweisung: das Geld ist sofort da, nicht rückholbar und du trägst es nicht durch die Stadt. Bargeld ist in Ordnung, wenn es am Bankschalter geprüft und eingezahlt wird."],
        ["Reicht ein Screenshot der Überweisung?", "Nein. Maßgeblich ist die Gutschrift auf deinem Konto. Bis dahin bleiben Schlüssel und Papiere bei dir."],
        ["Was, wenn der Käufer aus dem Ausland kommt?", "Das ist normal - aber die Regel bleibt: Zahlung vor Übergabe, Übergabe in der Bank, kein Transporteur, der für den Käufer zahlt."],
      ],
    },
  ],
  fr: [
    {
      slug: "fixer-le-prix",
      title: "Quel prix demander - et combien de temps ça prend",
      description: "Comment tirer le prix du marché plutôt que de ton prix d'achat, ce que veut dire la fourchette P25-P75, et quand il faut baisser.",
      body: loc => `
      <p class="fc-p">L'erreur la plus fréquente entre particuliers est de partir de son prix d'achat. Le marché ne le connaît pas. Il connaît ce qui se demande aujourd'hui pour le même modèle, la même année et un kilométrage comparable - et c'est exactement ce que montrent les <a href="${href(loc, "hub")}">pages de prix</a>.</p>
      <h2 class="fc-h2">Partir de la médiane, pas de l'envie</h2>
      <p class="fc-p">Prends la médiane de ton modèle pour l'année de ta voiture. La médiane est le prix du milieu : la moitié des annonces est au-dessus, la moitié en dessous. Elle résiste mieux qu'une moyenne, qu'une seule annonce hors de prix tire vers le haut.</p>
      <p class="fc-p">Avec elle vient la fourchette P25-P75 : les cinquante pour cent du milieu du marché. Si ta voiture est dans la moyenne pour la finition, l'état et le kilométrage, ta place est au milieu de cette fourchette. Un carnet d'entretien suivi, deux trains de pneus et peu de kilomètres justifient le haut ; un contrôle technique à refaire et de la tôle abîmée, le bas.</p>
      <h2 class="fc-h2">Les deux premières semaines décident</h2>
      <p class="fc-p">Une annonce est surtout vue les premiers jours. Partir trop haut brûle ces jours-là et finit souvent sous le prix qu'un départ réaliste aurait obtenu. Si en deux semaines aucune demande sérieuse n'arrive, ce n'est pas le marché : c'est le prix.</p>
      <p class="fc-p">Baisse alors d'un pas visible, pas en cinq petits : des micro-baisses annoncent qu'il y en aura d'autres, et les acheteurs attendent.</p>
      <h2 class="fc-h2">Ce qui bouge vraiment le prix</h2>
      <p class="fc-p">Le kilométrage et l'année pèsent le plus, puis la boîte et l'énergie. Les options ne rendent presque jamais à la revente ce qu'elles ont coûté neuves. Un contrôle technique récent, lui, vaut de l'argent : il enlève à l'acheteur un risque sur lequel il négocierait sinon.</p>`,
      faq: [
        ["Faut-il prévoir une marge de négociation ?", "Une petite marge est habituelle, mais dix pour cent au-dessus de la médiane ne sont pas une marge : c'est un filtre qui supprime la plupart des demandes avant qu'elles n'existent."],
        ["Et si ma voiture est meilleure que la moyenne ?", "Alors elle va en haut de la fourchette P25-P75, pas au-dessus. Ce qui dépasse la fourchette est comparé à des voitures qui s'y trouvent aussi."],
        ["Combien de temps dure une vente entre particuliers ?", "Une voiture courante au prix du marché part généralement en quelques semaines. Quand ça traîne nettement, c'est presque toujours le prix ou des photos où l'on ne voit pas la voiture."],
      ],
    },
    {
      slug: "paiement-securise",
      title: "Être payé sans risque : la remise des clés",
      description: "Comment l'argent arrive en sécurité : rendez-vous à la banque, virement instantané avant la remise des clés, et les arnaques qui visent les vendeurs particuliers.",
      body: loc => `
      <p class="fc-p">Dans une vente entre particuliers, il y a un instant où chacun doit lâcher quelque chose. Tout ce que tu peux faire pour ta sécurité se joue avant - après, il ne reste que la plainte.</p>
      <h2 class="fc-h2">La remise la plus simple et la plus sûre</h2>
      <p class="fc-p">Donnez-vous rendez-vous dans une agence bancaire, aux heures d'ouverture. L'argent y est viré ou déposé et compté ; la carte grise et les clés ne changent de mains que lorsque la somme est créditée sur ton compte - pas quand on te montre un justificatif.</p>
      <p class="fc-p">Un virement instantané arrive en quelques secondes et n'est pas rappelable. Un virement classique prend du temps : tant qu'il n'est pas crédité, tu ne remets rien. Une capture d'écran d'application n'est pas un paiement.</p>
      <h2 class="fc-h2">Les espèces</h2>
      <p class="fc-p">Les espèces sont possibles, avec deux risques : les faux billets et le trajet du retour. Les deux disparaissent au guichet, où l'argent est vérifié et déposé immédiatement. Ne compte jamais sur un parking et ne roule pas avec une grosse somme dans la voiture.</p>
      <h2 class="fc-h2">Les arnaques qui marchent encore</h2>
      <ul class="fc-ul">
        <li class="fc-li"><b>Le trop-perçu.</b> "L'acheteur" verse trop et demande le remboursement de la différence. Le premier paiement est ensuite annulé, ton remboursement non.</li>
        <li class="fc-li"><b>Le transporteur ou le séquestre.</b> Un prétendu transporteur ou service de séquestre veut enlever la voiture et payer plus tard. Ces services n'existent pas entre particuliers : il y a toi, l'acheteur et la banque.</li>
        <li class="fc-li"><b>L'acompte sans visite.</b> Qui achète une voiture qu'il n'a jamais vue ne l'achète pas : il cherche tes coordonnées bancaires ou une pièce d'identité.</li>
      </ul>
      <h2 class="fc-h2">L'essai</h2>
      <p class="fc-p">Demande la pièce d'identité avant l'essai et monte à bord. Note dans le certificat de cession la date, le prix, le kilométrage et l'état dont vous avez parlé : c'est ta seule version écrite de l'histoire si quelqu'un en raconte une autre plus tard.</p>`,
      faq: [
        ["Virement instantané ou espèces ?", "Virement instantané : l'argent est là tout de suite, non rappelable, et tu ne le transportes pas. Les espèces conviennent si elles sont vérifiées et déposées au guichet."],
        ["Une capture d'écran du virement suffit-elle ?", "Non. Seul compte le crédit sur ton compte. Jusque-là, les clés et les papiers restent chez toi."],
        ["Et si l'acheteur vient de l'étranger ?", "C'est courant, mais la règle ne change pas : paiement avant remise, remise à la banque, et aucun transporteur qui paierait à la place de l'acheteur."],
      ],
    },
  ],
  it: [
    {
      slug: "fissare-il-prezzo",
      title: "Che prezzo chiedere - e quanto tempo serve",
      description: "Come ricavare il prezzo dal mercato invece che da quanto hai speso, cosa significa la fascia P25-P75 e quando conviene scendere.",
      body: loc => `
      <p class="fc-p">L'errore più comune tra privati è partire dal proprio prezzo d'acquisto. Il mercato non lo conosce. Conosce quello che oggi si chiede per lo stesso modello, lo stesso anno e un chilometraggio simile - ed è esattamente quello che mostrano le <a href="${href(loc, "hub")}">pagine dei prezzi</a>.</p>
      <h2 class="fc-h2">Partire dalla mediana, non dal desiderio</h2>
      <p class="fc-p">Prendi la mediana del tuo modello per l'anno della tua auto. La mediana è il prezzo di mezzo: metà degli annunci sta sopra, metà sotto. Regge meglio di una media, che un singolo annuncio fuori prezzo tira verso l'alto.</p>
      <p class="fc-p">Con lei viene la fascia P25-P75: il cinquanta per cento centrale del mercato. Se la tua auto è nella media per allestimento, stato e chilometri, il tuo posto è in mezzo a quella fascia. Tagliandi regolari, due treni di gomme e pochi chilometri giustificano la parte alta; revisione da rifare e lamiere segnate, quella bassa.</p>
      <h2 class="fc-h2">Le prime due settimane decidono</h2>
      <p class="fc-p">Un annuncio viene visto soprattutto nei primi giorni. Partire troppo alti brucia proprio quei giorni e spesso si chiude sotto il prezzo che un avvio realistico avrebbe portato. Se in due settimane non arriva una richiesta seria, non è il mercato: è il prezzo.</p>
      <p class="fc-p">Allora scendi con un passo visibile, non con cinque piccoli: i micro-ribassi annunciano che ne arriveranno altri, e chi guarda aspetta.</p>
      <h2 class="fc-h2">Cosa muove davvero il prezzo</h2>
      <p class="fc-p">Chilometri e anno pesano di più, poi cambio e alimentazione. Gli optional quasi mai restituiscono alla rivendita quello che sono costati da nuovi. Una revisione appena fatta, invece, vale denaro: toglie a chi compra un rischio su cui altrimenti tratterebbe.</p>`,
      faq: [
        ["Devo lasciare margine di trattativa?", "Un piccolo margine è normale, ma il dieci per cento sopra la mediana non è margine: è un filtro che elimina la maggior parte delle richieste prima ancora che nascano."],
        ["E se la mia auto è migliore della media?", "Allora va nella parte alta della fascia P25-P75, non sopra. Ciò che sta sopra la fascia viene confrontato con auto che lì ci sono davvero."],
        ["Quanto dura una vendita tra privati?", "Un'auto comune a prezzo di mercato se ne va di solito in poche settimane. Se si trascina molto più a lungo, quasi sempre è il prezzo o sono foto in cui l'auto non si vede."],
      ],
    },
    {
      slug: "pagamento-sicuro",
      title: "Farsi pagare senza rischi: la consegna",
      description: "Come arrivano i soldi in sicurezza: appuntamento in banca, bonifico istantaneo prima delle chiavi, e le truffe che colpiscono i venditori privati.",
      body: loc => `
      <p class="fc-p">In una vendita tra privati c'è un momento in cui entrambi devono cedere qualcosa. Tutto quello che puoi fare per la tua sicurezza si decide prima - dopo resta solo la denuncia.</p>
      <h2 class="fc-h2">La consegna più semplice e più sicura</h2>
      <p class="fc-p">Datevi appuntamento in una filiale bancaria, negli orari di apertura. Lì i soldi si bonificano o si versano e si contano; libretto, documenti e chiavi passano di mano solo quando la somma è accreditata sul tuo conto - non quando ti mostrano una ricevuta.</p>
      <p class="fc-p">Un bonifico istantaneo arriva in pochi secondi e non è revocabile. Un bonifico ordinario richiede tempo: finché non è accreditato, non consegni nulla. Lo screenshot di un'app non è un pagamento.</p>
      <h2 class="fc-h2">Il contante</h2>
      <p class="fc-p">Il contante è possibile, con due rischi: banconote false e il viaggio di ritorno. Entrambi si risolvono allo sportello, dove i soldi si verificano e si versano subito. Non contare mai in un parcheggio e non girare con una somma grossa in auto.</p>
      <h2 class="fc-h2">Le truffe che funzionano ancora</h2>
      <ul class="fc-ul">
        <li class="fc-li"><b>Il pagamento in eccesso.</b> Il "compratore" versa troppo e chiede indietro la differenza. Il primo pagamento viene poi annullato, il tuo rimborso no.</li>
        <li class="fc-li"><b>Il trasportatore o il deposito fiduciario.</b> Un presunto trasportatore o servizio di garanzia vuole ritirare l'auto e pagare dopo. Tra privati questi servizi non esistono: ci siete tu, chi compra e la banca.</li>
        <li class="fc-li"><b>La caparra senza vedere l'auto.</b> Chi compra un'auto che non ha mai visto non la sta comprando: sta cercando le tue coordinate bancarie o un documento.</li>
      </ul>
      <h2 class="fc-h2">La prova su strada</h2>
      <p class="fc-p">Fatti mostrare un documento prima della prova e sali in auto. Metti per iscritto data, prezzo, chilometri e lo stato di cui avete parlato: è la tua unica versione scritta della storia, se più avanti qualcuno ne racconta un'altra.</p>`,
      faq: [
        ["Bonifico istantaneo o contante?", "Bonifico istantaneo: i soldi ci sono subito, non sono revocabili e non li porti in giro. Il contante va bene se verificato e versato allo sportello."],
        ["Basta lo screenshot del bonifico?", "No. Conta l'accredito sul tuo conto. Fino ad allora chiavi e documenti restano tuoi."],
        ["E se chi compra viene dall'estero?", "Capita spesso, ma la regola non cambia: pagamento prima della consegna, consegna in banca, e nessun trasportatore che paghi al posto di chi compra."],
      ],
    },
  ],
};

function guidesFor(loc) {
  return GUIDES[loc.code] || [];
}

function matchHubOrSlug(rest) {
  const s = String(rest || "").replace(/\/+$/, "");
  if (s === "") return { hub: true, slug: null };
  const m = /^\/([a-z0-9][a-z0-9-]{1,60})$/.exec(s);
  return m ? { hub: false, slug: m[1] } : null;
}

function updatedLine(loc) {
  return `<div class="mono" style="font-size:11.5px;color:#9A9FA8;margin:-6px 0 18px;">`
    + `${escapeHtml(t(loc, "g.updated", { date: UPDATED }))}</div>`;
}

function otherGuides(loc, current) {
  const rest = guidesFor(loc).filter(g => g.slug !== current);
  if (!rest.length) return "";
  const items = rest.map(g =>
    `<li class="fc-li"><a href="${href(loc, "guides")}/${g.slug}">${escapeHtml(g.title)}</a></li>`).join("");
  return `<h2 class="fc-h2">${t(loc, "g.other_h")}</h2><ul class="fc-ul">${items}</ul>`;
}

export function renderIntlGuidesHub({ loc, host }) {
  const path = href(loc, "guides");
  const canonical = `https://${host}${path}`;
  const items = guidesFor(loc).map(g =>
    `<li class="fc-li"><a href="${path}/${g.slug}"><b>${escapeHtml(g.title)}</b></a><br>`
    + `${escapeHtml(g.description)}</li>`).join("");
  const crumbItems = [
    { name: t(loc, "common.crumb_home"), href: href(loc, "landing") },
    { name: t(loc, "g.crumb") },
  ];
  const body = crumbs(crumbItems) + `
    <section class="fc-sec">
      <h1 class="fc-h1">${t(loc, "g.hub_h1")}</h1>
      ${updatedLine(loc)}
      <p class="fc-p">${t(loc, "g.hub_lede")}</p>
      <ul class="fc-ul">${items}</ul>
      <p class="fc-p"><a href="${href(loc, "hub")}">${t(loc, "g.links")}</a> · <a href="${href(loc, "avaliar")}">${t(loc, "g.cta")}</a></p>
    </section>`;
  return layout({
    title: t(loc, "g.hub_title", { country: loc.countryName }),
    description: t(loc, "g.hub_desc", { country: loc.countryName }),
    body, zone: "all", nav: "avaliar", depositCount: null, index: true, host, locale: loc,
    canonical,
    jsonLd: {
      "@context": "https://schema.org",
      "@graph": [
        {
          "@type": "CollectionPage", "url": canonical, "inLanguage": loc.lang,
          "name": t(loc, "g.hub_title", { country: loc.countryName }),
        },
        breadcrumbLd(host, crumbItems),
      ],
    },
  });
}

export function renderIntlGuide({ loc, host, guide }) {
  const path = `${href(loc, "guides")}/${guide.slug}`;
  const canonical = `https://${host}${path}`;
  const crumbItems = [
    { name: t(loc, "common.crumb_home"), href: href(loc, "landing") },
    { name: t(loc, "g.crumb"), href: href(loc, "guides") },
    { name: guide.title },
  ];
  const body = crumbs(crumbItems) + `
    <section class="fc-sec">
      <h1 class="fc-h1">${escapeHtml(guide.title)}</h1>
      ${updatedLine(loc)}
      ${guide.body(loc)}
      <h2 class="fc-h2">${t(loc, "g.faq_h")}</h2>
      ${guide.faq.map(([q, a]) =>
        `<details class="indep-note" style="margin:0 0 8px;"><summary>${escapeHtml(q)}</summary>`
        + `<p style="margin:8px 0 0;">${escapeHtml(a)}</p></details>`).join("")}
      ${otherGuides(loc, guide.slug)}
      <p class="fc-p"><a href="${href(loc, "hub")}">${t(loc, "g.links")}</a> · <a href="${href(loc, "avaliar")}">${t(loc, "g.cta")}</a></p>
    </section>`;
  return layout({
    title: guide.title,
    description: guide.description,
    body, zone: "all", nav: "avaliar", depositCount: null, index: true, host, locale: loc,
    canonical,
    jsonLd: {
      "@context": "https://schema.org",
      "@graph": [
        {
          "@type": "Article", "url": canonical, "inLanguage": loc.lang,
          "headline": guide.title, "description": guide.description,
          "datePublished": UPDATED, "dateModified": UPDATED,
          "author": { "@type": "Organization", "name": "Carsbuyer", "url": `https://${host}/` },
          "publisher": { "@type": "Organization", "name": "Carsbuyer", "url": `https://${host}/` },
          "mainEntityOfPage": canonical,
        },
        breadcrumbLd(host, crumbItems),
        faqLd(guide.faq),
      ],
    },
  });
}

async function handleGuides(ctx) {
  const { loc, url, params, helpers } = ctx;
  if (!guidesFor(loc).length) return helpers.notFoundIntl();
  if (params.hub) return helpers.publicHtml(renderIntlGuidesHub({ loc, host: url.host }));
  const guide = guidesFor(loc).find(g => g.slug === params.slug);
  if (!guide) return helpers.notFoundIntl();
  return helpers.publicHtml(renderIntlGuide({ loc, host: url.host, guide }));
}

export function intlGuidesSitemap(loc) {
  const list = guidesFor(loc);
  if (!list.length) return [];
  const path = href(loc, "guides");
  return [
    { path, freq: "monthly", prio: "0.5" },
    ...list.map(g => ({ path: `${path}/${g.slug}`, freq: "monthly", prio: "0.4" })),
  ];
}

export const INTL_GUIDES_MODULE = registerIntlPages({
  id: "intl-guides",
  routes: [{ routeKey: "guides", match: matchHubOrSlug, handle: handleGuides }],
  navRouteKeys: ["guides"],
  navAvailable(loc) {
    return guidesFor(loc).length ? ["guides"] : [];
  },
  sitemap: intlGuidesSitemap,
});
