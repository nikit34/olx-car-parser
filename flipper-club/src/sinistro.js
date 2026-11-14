import { layout, escapeHtml, PT_DISTRICTS } from "./templates.js";
import { crumbs, breadcrumbLd } from "./seo-pages.js";

export const REPORT_PATH = "/relatorio-valor-venal";
export const REPORT_PRICE_EUR = 75;
export const REPORT_DATA_SINCE = "2026-03-27";
export const REPORT_LEAD_TTL_SEC = 180 * 24 * 3600;
export const REPORT_SAMPLE_TOKEN = "bc7c53fb390f5195a858b42ba5371004";

export const REPORT_PURPOSES = [
  ["perda-total", "Perda total: contestar o valor venal da seguradora"],
  ["litigio", "Reclamação, CIMPAS ou tribunal"],
  ["partilha", "Partilha, herança ou divórcio"],
  ["outro", "Outro motivo"],
];

const EMAIL_RE = /^[^\s@<>]{1,64}@[^\s@<>]{1,190}\.[a-z]{2,24}$/i;

function clip(v, n) {
  return String(v == null ? "" : v).replace(/[\u0000-\u001f\u007f]/g, " ").trim().slice(0, n);
}

export function parseReportOrder(form, today) {
  const get = (k, n) => clip(form.get(k), n);
  const order = {
    modelo: get("modelo", 80),
    versao: get("versao", 80),
    ano: parseInt(get("ano", 4), 10) || null,
    km: parseInt(get("km", 7).replace(/\D/g, ""), 10) || null,
    data: get("data", 10),
    distrito: get("distrito", 40),
    finalidade: get("finalidade", 20),
    email: get("email", 254).toLowerCase(),
    nome: get("nome", 80),
    mensagem: get("mensagem", 1000),
  };
  const errors = [];
  if (order.modelo.length < 2) errors.push("Indica a marca e o modelo.");
  const maxYear = parseInt(today.slice(0, 4), 10) + 1;
  if (!order.ano || order.ano < 1950 || order.ano > maxYear) errors.push("Indica o ano do carro.");
  if (!/^\d{4}-\d{2}-\d{2}$/.test(order.data) || order.data < REPORT_DATA_SINCE || order.data > today) {
    errors.push(`A data tem de estar entre ${fmtDate(REPORT_DATA_SINCE)} e hoje.`);
  }
  if (!EMAIL_RE.test(order.email)) errors.push("Indica um email válido para te respondermos.");
  if (!REPORT_PURPOSES.some(([k]) => k === order.finalidade)) order.finalidade = "outro";
  if (order.distrito && !PT_DISTRICTS.includes(order.distrito)) order.distrito = "";
  if (form.get("consentimento") !== "1") errors.push("Precisamos do teu consentimento para guardar o pedido.");
  return { order, errors, trap: clip(form.get("website"), 200) !== "" };
}

export function fmtDate(iso) {
  const [y, m, d] = String(iso || "").split("-");
  return y && m && d ? `${d}/${m}/${y}` : "";
}

function field(label, input, hint = "") {
  return `<label style="display:block;margin:0 0 14px;">
          <span style="display:block;font-size:13.5px;font-weight:600;margin-bottom:5px;">${label}</span>
          ${input}
          ${hint ? `<span style="display:block;font-size:12.5px;color:#5B606B;margin-top:4px;">${hint}</span>` : ""}
        </label>`;
}

const INPUT = "width:100%;box-sizing:border-box;padding:11px 12px;font-size:15px;border:1px solid #C9CDD4;border-radius:8px;background:#fff;color:#14161A;";

function orderForm({ values = {}, errors = [], today }) {
  const v = (k) => escapeHtml(values[k] == null ? "" : String(values[k]));
  const purposeOpts = REPORT_PURPOSES.map(([k, label]) =>
    `<option value="${k}"${(values.finalidade || "perda-total") === k ? " selected" : ""}>${escapeHtml(label)}</option>`).join("");
  const districtOpts = [`<option value="">Escolhe o distrito</option>`].concat(PT_DISTRICTS.map(d =>
    `<option${values.distrito === d ? " selected" : ""}>${escapeHtml(d)}</option>`)).join("");
  const errorBox = errors.length
    ? `<div role="alert" style="background:#FDECEC;border:1px solid #E7A3A3;border-radius:8px;padding:12px 14px;margin:0 0 16px;font-size:14px;color:#7A1F1F;">${errors.map(escapeHtml).join("<br>")}</div>`
    : "";
  return `
      <form id="pedido" method="post" action="/pt${REPORT_PATH}" class="side-card" style="margin-top:8px;">
        <div class="panel-title" style="font-size:17px;margin-bottom:12px;">Pedir o relatório</div>
        ${errorBox}
        ${field("Marca e modelo", `<input name="modelo" required maxlength="80" value="${v("modelo")}" placeholder="Audi A4 Avant" style="${INPUT}">`)}
        ${field("Versão e motor", `<input name="versao" maxlength="80" value="${v("versao")}" placeholder="2.0 TDI 136 cv" style="${INPUT}">`, "Opcional, mas ajuda a escolher anúncios comparáveis.")}
        <div style="display:flex;gap:12px;flex-wrap:wrap;">
          <div style="flex:1 1 140px;">${field("Ano", `<input name="ano" type="number" inputmode="numeric" required min="1950" max="${parseInt(today.slice(0, 4), 10) + 1}" value="${v("ano")}" placeholder="2010" style="${INPUT}">`)}</div>
          <div style="flex:1 1 140px;">${field("Quilómetros", `<input name="km" inputmode="numeric" maxlength="9" value="${v("km")}" placeholder="137 800" style="${INPUT}">`)}</div>
        </div>
        <div style="display:flex;gap:12px;flex-wrap:wrap;">
          <div style="flex:1 1 160px;">${field("Data do valor", `<input name="data" type="date" required min="${REPORT_DATA_SINCE}" max="${today}" value="${v("data")}" style="${INPUT}">`, "Por exemplo, a data do sinistro.")}</div>
          <div style="flex:1 1 160px;">${field("Distrito", `<select name="distrito" style="${INPUT}">${districtOpts}</select>`)}</div>
        </div>
        ${field("Para que precisas do relatório", `<select name="finalidade" style="${INPUT}">${purposeOpts}</select>`)}
        ${field("Email", `<input name="email" type="email" required maxlength="254" value="${v("email")}" placeholder="nome@exemplo.pt" style="${INPUT}">`, "Respondemos para aqui em 1 dia útil.")}
        ${field("Nome", `<input name="nome" maxlength="80" value="${v("nome")}" style="${INPUT}">`, "Opcional. Usado só na fatura-recibo.")}
        ${field("Algo que devamos saber", `<textarea name="mensagem" rows="3" maxlength="1000" style="${INPUT}">${v("mensagem")}</textarea>`, "Por exemplo, o valor que a seguradora propôs.")}
        <div style="position:absolute;left:-9999px;" aria-hidden="true"><input name="website" tabindex="-1" autocomplete="off"></div>
        <label style="display:flex;gap:10px;align-items:flex-start;font-size:13.5px;color:#3A3F47;margin:4px 0 16px;line-height:1.45;">
          <input type="checkbox" name="consentimento" value="1" required style="margin-top:3px;">
          <span>Aceito que o Carsbuyer guarde estes dados até 180 dias para responder ao pedido e preparar o relatório. Ver <a href="/pt/privacidade">privacidade</a>.</span>
        </label>
        <button type="submit" class="btn-dark" style="width:100%;padding:13px 18px;font-size:15.5px;border:0;cursor:pointer;">Verificar grátis se há dados&nbsp;&nbsp;→</button>
        <p style="font-size:12.5px;color:#5B606B;margin:10px 0 0;line-height:1.45;">Não pagas nada agora. Primeiro confirmamos por email quantos anúncios comparáveis existem nessa data. Só depois, se quiseres, enviamos o link de pagamento.</p>
      </form>`;
}

export function renderReportOffer({ host, depositCount = null, today, values = {}, errors = [], sent = false }) {
  const canonical = `https://${host}/pt${REPORT_PATH}`;
  const trail = [{ name: "Início", href: "/pt" }, { name: "Relatório de valor de mercado" }];
  const done = sent ? `
      <div class="side-card" role="status" style="margin-top:8px;border-color:#9FD3B5;background:#F1FAF4;">
        <div class="panel-title" style="font-size:17px;margin-bottom:8px;">Pedido recebido</div>
        <p class="fc-p" style="margin:0;">Vamos ver quantos anúncios comparáveis existem nessa data e respondemos para o teu email em 1 dia útil, com o número de anúncios, o intervalo de preços e o link de pagamento. Se não houver dados suficientes, dizemos-te isso e não cobramos nada.</p>
      </div>` : orderForm({ values, errors, today });
  const body = crumbs(trail) + `
    <section class="section fc-wrap" style="padding-top:16px;">
      <h1 class="fc-h1">Quanto valia o teu carro numa data passada? Relatório com anúncios dessa data</h1>
      <p class="fc-p">A seguradora declarou perda total e o valor venal proposto parece baixo? A indemnização parte do valor do carro <b>antes do sinistro</b>, e para discordar precisas de mostrar com dados quanto custava comprar um carro igual nessa altura. Os anúncios dessa data já desapareceram dos sites. Nós guardamos os anúncios de carros usados do OLX e do Standvirtual desde ${fmtDate(REPORT_DATA_SINCE)}, com as datas em que os vimos e os preços pedidos.</p>
      <p class="fc-p">Com isso montamos um relatório datado com os anúncios comparáveis ao teu carro na data que indicares: os que estavam à venda nesse dia, não os de hoje. Mostra o preço de substituição, isto é, quanto custava comprar um carro igual, e não o valor a que um comerciante o compraria. <b>${REPORT_PRICE_EUR} €</b>, valor final, com fatura-recibo. Entrega em 2 dias úteis depois do pagamento, em PDF e numa página web.</p>
      <p class="fc-p"><a href="/relatorio/${REPORT_SAMPLE_TOKEN}" rel="nofollow">Vê um relatório de exemplo</a>: Volkswagen Golf VII 1.6 TDI a 15/10/2026, 23 carros comparáveis, com todas as secções que recebes.</p>
      ${done}

      <h2 class="fc-h2">O que vem no relatório</h2>
      <ul class="fc-ul">
        <li class="fc-li">Cada anúncio comparável com a data de recolha mais próxima antes e depois da data pedida, o preço pedido nessa data, quilómetros, ano, potência, distrito e a fonte com ligação.</li>
        <li class="fc-li">Os anúncios repetidos no OLX e no Standvirtual contados uma vez, com a regra usada para os juntar.</li>
        <li class="fc-li">Mediana e intervalo dos preços, e onde fica o teu carro face aos comparáveis em quilómetros, ano e motor.</li>
        <li class="fc-li">Método, critérios de escolha, data do relatório e responsável.</li>
        <li class="fc-li">As limitações dos dados, escritas no próprio relatório.</li>
        <li class="fc-li">Um modelo de carta à seguradora com os números do relatório já preenchidos, para contestares a proposta de indemnização.</li>
      </ul>

      <h2 class="fc-h2">O que o relatório não é</h2>
      <ul class="fc-ul">
        <li class="fc-li">Não é uma peritagem nem aconselhamento jurídico. É um levantamento documentado do mercado que podes juntar à tua reclamação.</li>
        <li class="fc-li">Mostra preços <b>pedidos</b> em anúncios, não preços a que os carros foram vendidos.</li>
        <li class="fc-li">Não guardamos cópias das páginas dos anúncios, e algumas ligações deixam de abrir quando o anúncio sai. Os dados que recolhemos ficam no relatório.</li>
        <li class="fc-li">Só cobre anúncios em Portugal e datas a partir de ${fmtDate(REPORT_DATA_SINCE)}. Houve dias sem recolha, e o relatório diz quais afetam a tua data.</li>
      </ul>

      <h2 class="fc-h2">Como funciona</h2>
      <ol class="fc-ul">
        <li class="fc-li">Preenches o formulário. Não pagas nada.</li>
        <li class="fc-li">Em 1 dia útil respondemos com quantos anúncios comparáveis existem e o intervalo de preços. Se forem poucos, dizemos-te e acaba aí.</li>
        <li class="fc-li">Se quiseres avançar, pagas ${REPORT_PRICE_EUR} € por cartão ou MB WAY no Stripe.</li>
        <li class="fc-li">Em 2 dias úteis recebes o relatório em PDF e numa página só tua.</li>
      </ol>
      <p class="fc-p" style="margin-top:18px;">Dúvidas antes de pedir: <a href="mailto:ola@carsbuyer.org">ola@carsbuyer.org</a> · <a href="/pt/guias/perda-total-contestar-valor-venal">Como contestar o valor venal</a> · <a href="/pt/metodologia">Metodologia</a> · <a href="/pt/sobre">Quem somos</a></p>
    </section>
    <div style="height:60px;"></div>`;
  return layout({
    title: "Valor de mercado do carro numa data passada: relatório para perda total",
    description: `Relatório datado com anúncios comparáveis do OLX e do Standvirtual na data do sinistro, para contestar o valor venal. ${REPORT_PRICE_EUR} € com fatura-recibo, verificação grátis antes de pagar.`,
    canonical, body, zone: "all", nav: null, depositCount, index: !sent && !errors.length, host,
    jsonLd: {
      "@context": "https://schema.org",
      "@graph": [
        {
          "@type": "Service", "url": canonical, "inLanguage": "pt-PT",
          "name": "Relatório de valor de mercado de um carro usado numa data passada",
          "areaServed": "PT", "provider": { "@type": "Organization", "name": "Carsbuyer", "url": `https://${host}/` },
          "offers": { "@type": "Offer", "price": String(REPORT_PRICE_EUR), "priceCurrency": "EUR" },
        },
        breadcrumbLd(host, trail),
      ],
    },
  });
}
