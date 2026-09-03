import { layout, escapeHtml, fmtEur, fmtNum, leadFormBlock } from "./templates.js";
import { crumbs, breadcrumbLd, faqLd, provenance } from "./seo-pages.js";

const UPDATED = "2026-09-04";
const pct = x => Math.round(x * 100);

const SRC = {
  govRegisto: ["Registar veículo automóvel (gov.pt)", "https://www2.gov.pt/servicos/registar-veiculo-automovel-servico-automovel-online"],
  irnCustos: ["Custos dos serviços (IRN)", "https://irn.justica.gov.pt/Custos-dos-servicos"],
  irnImpressos: ["Impressos e modelos (IRN)", "https://irn.justica.gov.pt/Impressos-e-Modelos"],
  decoContrato: ["Contrato de compra e venda de veículo, minuta (DECO PROteste)", "https://www.deco.proteste.pt/auto/carros-eletricos/cartas-tipo/contrato-compra-venda-veiculo-automovel"],
  decoReserva: ["Reserva de propriedade: para que serve e como cancelar (DECO PROteste)", "https://www.deco.proteste.pt/auto/financiamento-automovel/dicas/reserva-propriedade-automovel-serve-cancelar"],
  atIsencao: ["Isenção de ISV por transferência de residência (Portal das Finanças)", "https://info-aduaneiro.portaldasfinancas.gov.pt/pt/informacao_aduaneira/Veiculos/isencao_res/Pages/isv-res-03.aspx"],
  govMatricula: ["Pedir atribuição de matrícula (gov.pt)", "https://www2.gov.pt/servicos/pedir-atribuicao-de-matricula-para-um-veiculo"],
  anecraIsv: ["Perguntas frequentes sobre ISV (ANECRA)", "https://www.anecra.pt/AL/PDF/faqISV.PDF"],
  atMaisValias: ["Mais-valias, perguntas frequentes (Portal das Finanças)", "https://info.portaldasfinancas.gov.pt/pt/apoio_contribuinte/questoes_frequentes/pages/faqs-00566.aspx"],
  decoSeguro: ["Cessação do seguro automóvel por venda, minuta (DECO PROteste)", "https://www.deco.proteste.pt/auto/seguro-automovel/cartas-tipo/cessacao-contrato-seguro-automovel-venda-veiculo"],
  viaVerde: ["Perguntas frequentes (Via Verde)", "https://www.viaverde.pt/particulares/apoio-ao-cliente/perguntas-frequentes/perguntas-frequentes?themeId=181&faqId=1862"],
  gnrMbway: ["Alerta da GNR sobre burlas com MB WAY em anúncios (pplware)", "https://pplware.sapo.pt/informacao/alerta-gnr-usa-mbway-atencao-as-burlas-no-olx-e-custo-justo/"],
  pspBurla: ["Alerta da PSP sobre burla na compra e venda de carros (Notícias ao Minuto)", "https://www.noticiasaominuto.com/auto/2511293/psp-alerta-para-burla-com-compra-e-venda-de-carros-saiba-do-que-se-trata"],
  decoMbway: ["MB WAY: cuidado com a nova burla (DECO PROteste)", "https://www.deco.proteste.pt/tecnologia/ciberseguranca/noticias/mb-way-cuidado-nova-burla"],
  olxAjuda: ["Anúncios de carros no OLX (Ajuda OLX)", "https://help.olx.pt/hc/pt/articles/206187279"],
};

function marketLine(mk) {
  if (!mk || mk.s30 == null) return "";
  return `No conjunto do mercado de particulares que acompanhamos, <b>${pct(mk.s30)} em cada 100</b> anúncios saem do OLX no primeiro mês${mk.md != null ? `, com mediana de <b>${mk.md} dias</b>` : ""}${mk.cu != null ? `, e <b>${pct(mk.cu)} em cada 100</b> vendedores baixam o preço antes de vender${mk.cp != null ? `, em mediana ${pct(mk.cp)}%` : ""}` : ""}.`;
}

function fastestTable(models) {
  const rows = Object.entries(models || {})
    .filter(([, r]) => r.lq && r.lq.s30 != null && r.lq.n >= 100 && r.fm > 0)
    .sort((a, b) => b[1].lq.s30 - a[1].lq.s30)
    .slice(0, 8)
    .map(([slug, r]) => `<tr><td><a href="/vender/${slug}">${escapeHtml(r.b)} ${escapeHtml(r.m)}</a></td><td><b>${pct(r.lq.s30)} em cada 100</b></td><td class="mut">${r.lq.md != null ? `${r.lq.md} dias` : "—"}</td><td class="mut">${fmtEur(r.fm)}</td></tr>`)
    .join("");
  if (!rows) return "";
  return `<div class="fc-scroll"><table class="fc-tbl">
      <thead><tr><th>Modelo</th><th>Sai em 30 dias</th><th>Mediana</th><th>Mediana pedida</th></tr></thead>
      <tbody>${rows}</tbody></table></div>`;
}

export const GUIDES = [
  {
    slug: "documentos-para-vender-carro",
    title: "Documentos para vender um carro usado em Portugal",
    h1: "Documentos para vender um carro usado em Portugal",
    description: "O que o vendedor particular tem de ter e entregar: DUA, inspeção em dia, IUC, requerimento de registo e o contrato de compra e venda. Lista de verificação e onde obter cada documento.",
    body: () => `
      <p class="fc-p">Vender um carro entre particulares não exige advogado nem notário. Exige ter três coisas em ordem antes de anunciar e entregar duas no dia da venda. Esta lista segue o que o IRN e o gov.pt pedem para registar a transferência de propriedade.</p>
      <h2 class="fc-h2">Antes de anunciar</h2>
      <ul class="fc-ul">
        <li class="fc-li"><b>Documento Único Automóvel (DUA).</b> É o documento do carro desde 2005, e substitui o antigo par livrete mais título de registo de propriedade. Se ainda tens os documentos antigos, servem, mas a transferência vai emitir um DUA novo em nome do comprador. Confirma que o nome no DUA é o teu: se o carro ainda está registado em nome de outra pessoa, primeiro tens de regularizar isso.</li>
        <li class="fc-li"><b>Inspeção periódica (IPO) válida.</b> Carros até 4 anos estão isentos; dos 4 aos 8 anos a inspeção é de dois em dois anos; a partir dos 8, anual. Um carro com inspeção fora de prazo vende-se pior e o comprador vai usar isso na negociação.</li>
        <li class="fc-li"><b>Imposto Único de Circulação (IUC) pago.</b> O IUC segue quem está registado como proprietário. Vender com o imposto em atraso é entregar ao comprador uma dívida que continua em teu nome até o registo mudar.</li>
        <li class="fc-li"><b>Reserva de propriedade cancelada</b>, se o carro foi comprado a crédito. Enquanto o banco tiver a reserva, a venda não pode ser registada. O passo a passo está no guia <a href="/guias/vender-carro-com-credito">vender um carro com crédito</a>.</li>
      </ul>
      <h2 class="fc-h2">No dia da venda</h2>
      <ul class="fc-ul">
        <li class="fc-li"><b>Requerimento de Registo Automóvel</b>, assinado por ambos, ou o pedido feito online no Automóvel Online com Chave Móvel Digital. O impresso oficial está nos <a href="${SRC.irnImpressos[1]}" rel="noopener" target="_blank">impressos do IRN</a>.</li>
        <li class="fc-li"><b>Contrato de compra e venda.</b> Não é obrigatório para registar a venda, mas é a tua prova da data, do preço, dos quilómetros e do estado declarado. A DECO PROteste publica uma <a href="${SRC.decoContrato[1]}" rel="noopener" target="_blank">minuta gratuita</a>. Duas cópias, uma para cada um, com cópia do documento de identificação do comprador.</li>
        <li class="fc-li"><b>DUA, chaves, livro de revisões e faturas</b> que tenhas. Não vendem o carro, mas fecham o negócio mais depressa e com menos desconto.</li>
      </ul>
      <h2 class="fc-h2">Depois da venda</h2>
      <p class="fc-p">O registo da transferência tem de ser feito em 60 dias e é, por lei, obrigação do comprador; o vendedor também o pode fazer com a prova da venda. Enquanto o registo não muda, o carro continua em teu nome para efeitos de IUC e de multas. O guia <a href="/guias/registo-de-propriedade-automovel">registo de propriedade</a> explica como te protegeres.</p>`,
    faq: [
      ["Preciso do livrete e do título de registo se já tenho o DUA?", "Não. O DUA substitui os dois documentos antigos. Quem ainda só tem livrete e título de registo pode vender com eles, e a transferência emite um DUA novo em nome do comprador."],
      ["O contrato de compra e venda é obrigatório?", "Não é exigido para registar a transferência, mas é a única prova escrita da data, do preço e do estado do carro. A DECO PROteste tem uma minuta gratuita; assinem os dois e fiquem cada um com uma cópia."],
      ["Posso vender um carro com a inspeção fora de prazo?", "Podes, mas o comprador não pode circular legalmente com ele até fazer a inspeção, e vai descontar isso no preço. Fazer a inspeção antes de anunciar costuma compensar."],
    ],
    sources: ["govRegisto", "irnImpressos", "decoContrato"],
  },
  {
    slug: "registo-de-propriedade-automovel",
    title: "Registo de propriedade automóvel: prazo, custo e quem trata",
    h1: "Registo de propriedade: quem trata, o prazo de 60 dias e quanto custa",
    description: "Como se transfere a propriedade de um carro vendido entre particulares: Automóvel Online ou conservatória, prazo de 60 dias, obrigação do comprador, custo aproximado e como o vendedor se protege.",
    body: () => `
      <p class="fc-p">A venda fica completa quando o registo automóvel passa para o nome do comprador. Até lá, o carro é teu perante o Estado: o IUC, as multas de radar e a responsabilidade por uma viatura estacionada onde não devia chegam ao nome que está no registo.</p>
      <h2 class="fc-h2">Onde e como</h2>
      <ul class="fc-ul">
        <li class="fc-li"><b>Online</b>, no serviço Automóvel Online do IRN, com Chave Móvel Digital ou Cartão de Cidadão com leitor. É o caminho mais rápido e o mais barato.</li>
        <li class="fc-li"><b>Ao balcão</b>, numa Conservatória do Registo Automóvel ou Loja do Cidadão, com o Requerimento de Registo Automóvel assinado pelos dois.</li>
      </ul>
      <h2 class="fc-h2">Prazo e quem é responsável</h2>
      <p class="fc-p">Segundo o gov.pt, o pedido de registo tem de ser feito no prazo de <b>60 dias</b> após a venda e a obrigação é do <b>comprador</b>; o vendedor também pode fazê-lo, desde que tenha a prova da venda. Na prática, muitos vendedores preferem tratar do registo no próprio dia, com o comprador ao lado, precisamente para não ficarem dependentes dele.</p>
      <h2 class="fc-h2">Quanto custa</h2>
      <p class="fc-p">A taxa de registo da transferência ronda os <b>55 € online</b> e os <b>65 € ao balcão</b> dentro do prazo, e sobe para cerca de 120 € quando o pedido é feito fora dos 60 dias. Estes valores são os publicados na tabela de custos do IRN; confirma-os na <a href="${SRC.irnCustos[1]}" rel="noopener" target="_blank">página oficial</a> antes de pagar, porque mudam com o Orçamento do Estado. Quem paga é, em regra, o comprador, mas nada impede que fique acordado de outra forma no contrato.</p>
      <h2 class="fc-h2">Se o comprador não regista</h2>
      <p class="fc-p">É a situação mais comum de conflito depois da venda. Enquanto o registo não muda, as notificações de IUC e de contraordenações continuam a chegar ao vendedor. A proteção do vendedor é ter a prova da venda (contrato assinado, cópia do documento do comprador, comprovativo do pagamento) e usá-la: pedir ele próprio o registo, ou apresentar a declaração de venda junto das entidades que o notificam. Sem um papel assinado com data, o vendedor fica a discutir a data da venda sem prova.</p>
      <h2 class="fc-h2">Lista curta</h2>
      <ul class="fc-ul">
        <li class="fc-li">Contrato assinado com data, preço e quilómetros, mais cópia da identificação do comprador.</li>
        <li class="fc-li">Registo feito no próprio dia, online, com o comprador presente, sempre que possível.</li>
        <li class="fc-li">Se não for possível, guardar todas as provas e verificar ao fim de 60 dias se o registo mudou.</li>
      </ul>`,
    faq: [
      ["Quem tem de registar a venda do carro, o comprador ou o vendedor?", "A obrigação legal é do comprador, no prazo de 60 dias. O vendedor também pode pedir o registo com a prova da venda, e é o que mais o protege, porque até o registo mudar o carro continua em nome dele."],
      ["Quanto custa o registo de propriedade de um carro usado?", "Cerca de 55 € online e 65 € ao balcão dentro do prazo, subindo para cerca de 120 € fora dos 60 dias, segundo a tabela de custos do IRN. Confirma o valor atual na página oficial antes de pagar."],
      ["O que acontece se a venda não for registada?", "O vendedor continua a ser o proprietário registado: o IUC e as multas chegam-lhe a ele. Com o contrato assinado e a identificação do comprador consegue provar a data da venda e pedir ele próprio o registo."],
    ],
    sources: ["govRegisto", "irnCustos", "irnImpressos"],
  },
  {
    slug: "vender-carro-com-credito",
    title: "Vender um carro com crédito ou reserva de propriedade",
    h1: "Vender um carro com crédito: o distrate e a reserva de propriedade",
    description: "Um carro comprado a crédito costuma ter reserva de propriedade a favor do banco. O que é preciso para o vender: liquidar o crédito, obter o distrate e cancelar a reserva no registo.",
    body: () => `
      <p class="fc-p">Quando um carro é comprado com crédito, o banco fica normalmente com uma <b>reserva de propriedade</b> inscrita no registo: o carro é teu, mas não o podes vender livremente enquanto a dívida existir. A boa notícia é que o processo é conhecido e cabe em três passos.</p>
      <h2 class="fc-h2">1. Saber quanto falta pagar</h2>
      <p class="fc-p">Pede ao banco o valor de liquidação antecipada à data em que pretendes vender. É esse o número que interessa, não a soma das prestações que faltam: o crédito ao consumo permite liquidar antes do fim com uma comissão limitada por lei.</p>
      <h2 class="fc-h2">2. Liquidar e obter o distrate</h2>
      <p class="fc-p">Depois de paga a dívida, o banco emite o <b>distrate</b>, a declaração de que o crédito está liquidado e de que renuncia à reserva. Se não tens o dinheiro antes da venda, há duas formas habituais de o fazer: o comprador transfere o valor da liquidação diretamente para o banco e o resto para ti, com tudo escrito no contrato; ou faz-se o negócio num balcão do banco. Nunca entregues o carro antes de o distrate existir.</p>
      <h2 class="fc-h2">3. Cancelar a reserva e registar a venda</h2>
      <p class="fc-p">Com o distrate, pede-se a extinção da reserva no Automóvel Online ou numa conservatória; só depois é possível registar a transferência para o comprador. A DECO PROteste descreve o cancelamento na <a href="${SRC.decoReserva[1]}" rel="noopener" target="_blank">sua página sobre reserva de propriedade</a>. O cancelamento tem um custo de registo próprio, mais baixo do que a transferência; alguns bancos tratam do cancelamento a troco de uma comissão, o que fica mais caro do que fazê-lo tu.</p>
      <h2 class="fc-h2">O que dizer ao comprador</h2>
      <p class="fc-p">Diz logo no anúncio que o carro tem reserva de propriedade e que a venda inclui o distrate. Esconder isso faz perder o comprador no dia em que ele consulta o registo, e qualquer comprador informado consulta.</p>`,
    faq: [
      ["Posso vender um carro que ainda estou a pagar?", "Sim, mas a venda só pode ser registada depois de o crédito ser liquidado e a reserva de propriedade cancelada. O caminho habitual é o comprador pagar o valor de liquidação ao banco e o restante ao vendedor, com tudo escrito no contrato."],
      ["O que é o distrate?", "É a declaração do banco de que o crédito está pago e de que renuncia à reserva de propriedade. Com ela pede-se o cancelamento da reserva no registo automóvel e só depois a transferência para o comprador."],
      ["Quem paga o cancelamento da reserva?", "Normalmente o vendedor, porque é uma condição para poder vender. Feito diretamente no IRN custa menos do que através do banco."],
    ],
    sources: ["decoReserva", "govRegisto"],
  },
  {
    slug: "vender-carro-importado",
    title: "Vender um carro importado em Portugal",
    h1: "Vender um carro importado: matrícula, ISV e o prazo de 12 meses",
    description: "O que muda quando o carro veio de fora: só se vende com matrícula portuguesa, o que a legalização exige, e a regra de 12 meses para quem beneficiou da isenção de ISV por mudança de residência.",
    body: () => `
      <p class="fc-p">Um em cada cinco anúncios de particulares que acompanhamos é de um carro importado. Vendem-se todos os dias, mas com duas condições que o vendedor tem de conhecer antes de anunciar.</p>
      <h2 class="fc-h2">Só com matrícula portuguesa</h2>
      <p class="fc-p">Um comprador em Portugal precisa de um carro que possa registar em seu nome e com o qual possa circular: isso exige matrícula portuguesa. Um carro ainda com matrícula estrangeira não está legalizado, e quem o comprar terá de pagar o ISV e tratar da legalização. Por isso, ou legalizas antes de vender, ou vendes por um preço que desconte o ISV e todo o trabalho, e dizes isso no anúncio. O nosso <a href="/isv">simulador de ISV</a> dá o valor para um carro concreto.</p>
      <h2 class="fc-h2">O que a legalização exige</h2>
      <ul class="fc-ul">
        <li class="fc-li">Certificado de conformidade (COC) do fabricante, ou a homologação nacional quando não existe.</li>
        <li class="fc-li">Inspeção técnica de categoria B, que emite o modelo 112.</li>
        <li class="fc-li">Declaração aduaneira de veículo (DAV) e pagamento do ISV, com prazo contado a partir da entrada do carro em Portugal; fora do prazo há coima.</li>
        <li class="fc-li">Atribuição de matrícula no IMT e registo de propriedade na conservatória; os passos oficiais estão no <a href="${SRC.govMatricula[1]}" rel="noopener" target="_blank">gov.pt</a>.</li>
      </ul>
      <h2 class="fc-h2">Se tiveste isenção de ISV por mudança de residência</h2>
      <p class="fc-p">Quem trouxe o carro ao mudar-se para Portugal pode ter beneficiado da isenção de ISV. O Portal das Finanças exige, entre outras condições, que o carro tenha sido propriedade do requerente no país de origem durante <b>pelo menos 6 meses</b> antes da mudança. E a mesma isenção traz uma contrapartida citada pela ANECRA e pelos despachantes: o carro não pode ser vendido durante <b>12 meses</b> depois da atribuição da matrícula sem devolver o imposto. Antes de anunciar um carro nessa situação, confirma a data e as condições na <a href="${SRC.atIsencao[1]}" rel="noopener" target="_blank">página da Autoridade Tributária</a>.</p>
      <h2 class="fc-h2">O que o comprador vai perguntar</h2>
      <p class="fc-p">Historial do carro no país de origem, quilómetros, sinistros, número de donos. Um relatório de histórico pela matrícula estrangeira ou pelo VIN responde a isso e evita que o desconto seja «por desconfiança». Se o carro ficou mais barato do que o mesmo modelo nacional, a página <a href="/importar">importar da Alemanha</a> mostra a diferença de preços que o mercado pratica.</p>`,
    faq: [
      ["Posso vender um carro com matrícula estrangeira em Portugal?", "Na prática só a quem esteja disposto a legalizá-lo: o comprador terá de pagar o ISV e tratar da matrícula portuguesa. Ou legalizas antes de vender, ou vendes com desconto e dizes isso no anúncio."],
      ["Tive isenção de ISV por mudança de residência. Posso vender já?", "As condições da isenção incluem ter tido o carro pelo menos 6 meses no país de origem, e a venda nos 12 meses seguintes à matrícula implica, segundo a ANECRA e os despachantes, a devolução do imposto. Confirma na Autoridade Tributária antes de anunciar."],
      ["Um carro importado vale menos do que um nacional?", "Depende do modelo. Nas nossas páginas de importação comparamos o preço pedido em Portugal com o preço alemão mais ISV e legalização; para alguns modelos a diferença é grande, para outros quase nula."],
    ],
    sources: ["atIsencao", "govMatricula", "anecraIsv"],
  },
  {
    slug: "burlas-e-pagamento-seguro",
    title: "Burlas na venda de carros e como receber o dinheiro com segurança",
    h1: "Burlas na venda de carros e como receber o dinheiro",
    description: "As burlas mais comuns em anúncios de carros no OLX e no CustoJusto, segundo os alertas da GNR, da PSP e da DECO, e as regras simples para receber o pagamento sem risco.",
    body: () => `
      <p class="fc-p">Quase todas as burlas na venda de carros entre particulares têm o mesmo desenho: alguém que não está presente pede-te para fazeres algo com o teu telemóvel ou com o teu dinheiro antes de veres o comprador. As forças de segurança e a DECO descrevem as variantes; a defesa é uma só.</p>
      <h2 class="fc-h2">As burlas mais comuns</h2>
      <ul class="fc-ul">
        <li class="fc-li"><b>O «pagamento» por MB WAY.</b> O falso comprador diz que já pagou e que tens de «aceitar» ou «confirmar» a transferência seguindo instruções ao telefone; na verdade estás a autorizar um pagamento ou um levantamento a favor dele. A GNR e a DECO alertaram para esta burla em anúncios do OLX e do CustoJusto.</li>
        <li class="fc-li"><b>O anúncio clonado.</b> Copiam as fotos do teu carro, publicam-no mais barato noutro sítio e cobram sinais a compradores que nunca vão receber carro nenhum. O teu nome fica associado à burla. Pesquisa as tuas fotos de vez em quando e denuncia.</li>
        <li class="fc-li"><b>O comprador do estrangeiro.</b> Paga «a mais» por cheque ou transferência e pede que reenvies a diferença ao transportador. O cheque é falso e o transportador é o burlão.</li>
        <li class="fc-li"><b>O ensaio sem regresso.</b> Uma pessoa pede para experimentar o carro sozinha. Regra simples: vais sempre no carro, com o documento de identificação da pessoa fotografado antes de arrancar.</li>
      </ul>
      <h2 class="fc-h2">Como receber o dinheiro</h2>
      <ul class="fc-ul">
        <li class="fc-li"><b>Transferência bancária confirmada na tua conta</b>, não numa captura de ecrã. Para valores altos, fazer a transferência num balcão, com os dois presentes, é o método mais tranquilo.</li>
        <li class="fc-li"><b>MB WAY só para valores pequenos</b> e só depois de veres o dinheiro na tua aplicação. Nunca sigas instruções de um desconhecido sobre «como usar o MB WAY».</li>
        <li class="fc-li"><b>Cheque só se for visado</b> pelo banco emissor, e mesmo assim confirma-o antes de entregar o carro.</li>
        <li class="fc-li"><b>Sinal</b> apenas por transferência, com recibo escrito que diga o que acontece se o negócio não se concretizar.</li>
      </ul>
      <p class="fc-p">E, em qualquer caso, os documentos só mudam de mãos quando o dinheiro está na tua conta. Um comprador sério não tem pressa nesse ponto.</p>`,
    faq: [
      ["Como funciona a burla do MB WAY na venda de carros?", "O falso comprador diz que já pagou e pede que confirmes a receção seguindo passos ao telefone. Esses passos autorizam um pagamento ou levantamento a favor dele. Nenhuma receção de dinheiro exige ações da tua parte: se o dinheiro chegou, aparece na tua aplicação sem fazeres nada."],
      ["Qual é a forma mais segura de receber o pagamento de um carro?", "Transferência bancária confirmada na tua conta antes de entregar documentos e chaves; para valores altos, a transferência feita ao balcão com os dois presentes. Capturas de ecrã e comprovativos enviados por mensagem não provam nada."],
      ["Devo deixar o comprador experimentar o carro sozinho?", "Não. Vai sempre no carro, fotografa o documento de identificação antes de arrancar e escolhe um percurso conhecido."],
    ],
    sources: ["gnrMbway", "pspBurla", "decoMbway"],
  },
  {
    slug: "quanto-pedir-e-quanto-tempo-demora",
    title: "Quanto pedir pelo carro e quanto tempo demora a vender",
    h1: "Quanto pedir pelo carro e quanto tempo demora a vender",
    description: "Como fixar o preço de um carro usado a partir dos anúncios reais do mesmo modelo e ano, quantos vendedores acabam por baixar o preço e em quantos dias os anúncios saem do OLX.",
    body: ({ market, models, stats }) => `
      <p class="fc-p">O preço certo não é o que o carro te custou nem o que gostavas de receber: é o que o mercado está a pedir hoje por carros como o teu, corrigido pelo teu estado e pelos teus quilómetros. Medimos isso todos os dias${stats && stats.listings ? ` em ${fmtNum(stats.listings)} anúncios ativos` : ""} e publicamos por modelo e ano.</p>
      <h2 class="fc-h2">Três números antes de escrever o preço</h2>
      <ul class="fc-ul">
        <li class="fc-li"><b>A mediana do teu modelo e ano.</b> Metade dos anúncios pede menos, metade pede mais. Está em cada página de <a href="/precos">preços por modelo</a>, e a <a href="/avaliar">avaliação por modelo e ano</a> dá-a para o teu caso.</li>
        <li class="fc-li"><b>O intervalo onde fica metade dos anúncios.</b> Acima do limite de cima competes com carros mais novos ou com menos quilómetros; abaixo do de baixo, o comprador desconfia antes de perguntar.</li>
        <li class="fc-li"><b>Quantos baixam o preço.</b> ${marketLine(market) || "A percentagem de anúncios com descida de preço está em cada página de modelo."} A descida mediana é a folga que faz sentido deixar entre o preço que pedes e o que aceitas.</li>
      </ul>
      <h2 class="fc-h2">Quanto tempo demora</h2>
      <p class="fc-p">Um anúncio do OLX corre em ciclos de 30 dias e muitos desaparecem exatamente aí; contamos como saída o último ciclo em que vimos o anúncio no ar. Os modelos que saem mais depressa, entre os que acompanhamos com amostra grande:</p>
      ${fastestTable(models)}
      <p class="fc-p">A lista completa, com a percentagem que sai em 30, 60 e 90 dias por modelo, está em <a href="/liquidez">tempo de venda por modelo</a>; a página do teu modelo em <a href="/vender">vender</a> junta o preço a pedir, os dias e o pedido de propostas.</p>
      <h2 class="fc-h2">Há uma melhor altura do ano?</h2>
      <p class="fc-p">Ainda não temos um ano completo de dados para o afirmar com números, e não vamos inventar. O que já medimos é o que decide dentro de cada mês: o preço a que pões o carro e os quilómetros face à idade. Um carro ao preço da mediana sai no primeiro ciclo em muitos modelos; um carro acima do intervalo espera um segundo ciclo e acaba por baixar.</p>`,
    faq: [
      ["Como sei quanto pedir pelo meu carro?", "Pela mediana e pelo intervalo dos anúncios do mesmo modelo e ano, corrigidos pelos teus quilómetros e estado. A avaliação por modelo e ano dá esse número em segundos; a página do modelo mostra a evolução por ano."],
      ["Quantos vendedores baixam o preço?", "No mercado de particulares que acompanhamos, cerca de um terço dos anúncios baixa o preço antes de sair, em mediana perto de 8%. Por modelo o valor varia e está em cada página de vender."],
      ["Em quantos dias se vende um carro usado em Portugal?", "A mediana do mercado anda perto de um mês, com grandes diferenças entre modelos: os pequenos citadinos populares saem em duas semanas, os carros caros e os menos comuns esperam dois ciclos ou mais."],
    ],
    sources: [],
  },
  {
    slug: "depois-de-vender-seguro-iuc-via-verde",
    title: "Depois de vender o carro: seguro, IUC, Via Verde e a declaração de venda",
    h1: "Depois de vender: seguro, IUC, Via Verde e a prova da venda",
    description: "O que o vendedor tem de fazer nos dias a seguir à venda: avisar a seguradora, guardar a prova da venda, saber quem paga o IUC e o que fazer ao identificador da Via Verde.",
    body: () => `
      <p class="fc-p">O dinheiro entrou e o carro saiu, mas ainda há quatro coisas que ficam em teu nome até tratares delas.</p>
      <h2 class="fc-h2">Seguro</h2>
      <p class="fc-p">Avisa a seguradora por escrito com a data exata da venda: a cobertura termina no fim desse dia e a apólice deixa de ser cobrada. Se vais comprar outro carro, pergunta pela transferência da apólice para a matrícula nova, que costuma ser possível dentro de um prazo. A DECO PROteste tem uma <a href="${SRC.decoSeguro[1]}" rel="noopener" target="_blank">minuta de carta</a> para a cessação por venda.</p>
      <h2 class="fc-h2">IUC</h2>
      <p class="fc-p">O imposto segue o proprietário registado. Enquanto o registo não passar para o comprador, as notificações continuam a chegar-te. Depois de a transferência ser registada dentro dos 60 dias, o comprador passa a ser o sujeito do imposto. Se o comprador se atrasar, usa o contrato assinado para provar a data da venda.</p>
      <h2 class="fc-h2">Via Verde</h2>
      <p class="fc-p">Tira o identificador do carro antes de o entregar. Se vais ter outro carro, a Via Verde permite associar o mesmo identificador à matrícula nova sem custo, pela área de cliente; as instruções estão nas <a href="${SRC.viaVerde[1]}" rel="noopener" target="_blank">perguntas frequentes da Via Verde</a>. Se não vais ter carro, cancela o contrato do identificador para não continuares a pagar as passagens de outra pessoa.</p>
      <h2 class="fc-h2">A prova da venda</h2>
      <p class="fc-p">Guarda durante anos: o contrato assinado, a cópia da identificação do comprador, o comprovativo da transferência bancária e, se o fizeste, o comprovativo do pedido de registo. É com isso que respondes a uma multa ou a um IUC que chegue em teu nome por um carro que já não é teu.</p>
      <h2 class="fc-h2">E o IRS?</h2>
      <p class="fc-p">A venda ocasional do carro pessoal não é um rendimento tributável em IRS: a lista de mais-valias do Código do IRS não inclui automóveis, e as <a href="${SRC.atMaisValias[1]}" rel="noopener" target="_blank">perguntas frequentes do Portal das Finanças</a> sobre mais-valias não os mencionam. Quem compra e vende carros com regularidade já não é um particular a vender o seu carro, e essa atividade tem outro enquadramento.</p>`,
    faq: [
      ["Tenho de avisar a seguradora quando vendo o carro?", "Sim, por escrito e com a data da venda. A cobertura termina nesse dia e a apólice deixa de ser cobrada; se comprares outro carro, pergunta pela transferência para a matrícula nova."],
      ["Quem paga o IUC no ano da venda?", "O proprietário registado. Depois de a transferência ser registada, o comprador passa a ser responsável; até lá, as notificações chegam ao vendedor, que responde com a prova da data da venda."],
      ["Pago IRS por vender o meu carro?", "Não, quando é a venda ocasional do carro pessoal: os automóveis não estão na lista de mais-valias do Código do IRS. Comprar e vender carros com regularidade é uma atividade económica com outro enquadramento."],
    ],
    sources: ["decoSeguro", "viaVerde", "atMaisValias", "govRegisto"],
  },
];

export function guideBySlug(slug) {
  return GUIDES.find(g => g.slug === slug) || null;
}

function guideNav(current) {
  return GUIDES.filter(g => g.slug !== current).map(g => `<li class="fc-li"><a href="/guias/${g.slug}">${escapeHtml(g.title)}</a></li>`).join("");
}

function sourcesBlock(keys) {
  const items = (keys || []).map(k => SRC[k]).filter(Boolean)
    .map(([label, url]) => `<li class="fc-li"><a href="${url}" rel="noopener" target="_blank">${escapeHtml(label)}</a></li>`).join("");
  return items ? `<h2 class="fc-h2">Fontes</h2><ul class="fc-ul">${items}</ul>` : "";
}

export function renderGuide({ guide, models, market, stats, host, depositCount, builtAt }) {
  const canonical = `https://${host}/guias/${guide.slug}`;
  const body = crumbs([{ name: "Início", href: "/" }, { name: "Guias", href: "/guias" }, { name: guide.title }]) + `
    <section class="section fc-wrap" style="padding-top:16px;">
      <h1 class="fc-h1">${escapeHtml(guide.h1)}</h1>
      <div class="mono" style="font-size:11.5px;color:#9A9FA8;margin:-6px 0 18px;">Atualizado a ${UPDATED} · guia para vendedores particulares · não substitui aconselhamento jurídico</div>
      ${guide.body({ models, market, stats })}
      <h2 class="fc-h2">Perguntas frequentes</h2>
      ${guide.faq.map(([q, a]) => `<details class="indep-note" style="margin:0 0 8px;"><summary>${escapeHtml(q)}</summary><p style="margin:8px 0 0;">${escapeHtml(a)}</p></details>`).join("")}
      ${sourcesBlock(guide.sources)}
    </section>
    <section class="section" style="padding:0 22px;max-width:680px;margin:0 auto;">
      ${leadFormBlock({ slug: "", name: "", year: null, median: null, heading: "Queres vender o teu carro? Recebe propostas de compra" })}
    </section>
    <section class="section fc-wrap" style="padding-bottom:70px;">
      <h2 class="fc-h2">Outros guias</h2>
      <ul class="fc-ul">${guideNav(guide.slug)}</ul>
      <p class="fc-p"><a href="/vender">Quanto pedir por modelo</a> · <a href="/avaliar">Avaliar o meu carro</a> · <a href="/guias">Todos os guias</a></p>
    </section>`;
  return layout({
    title: guide.title,
    description: guide.description,
    canonical, body, zone: "all", nav: "avaliar", depositCount, index: true, host,
    jsonLd: {
      "@context": "https://schema.org",
      "@graph": [
        {
          "@type": "Article", "url": canonical, "inLanguage": "pt-PT",
          "headline": guide.title, "description": guide.description,
          "datePublished": UPDATED, "dateModified": UPDATED,
          "author": { "@type": "Organization", "name": "Carsbuyer", "url": `https://${host}/` },
          "publisher": { "@type": "Organization", "name": "Carsbuyer", "url": `https://${host}/` },
          "mainEntityOfPage": canonical,
        },
        breadcrumbLd(host, [{ name: "Início", href: "/" }, { name: "Guias", href: "/guias" }, { name: guide.title }]),
        faqLd(guide.faq),
      ],
    },
  });
}

export function renderGuidesHub({ market, stats, host, depositCount, builtAt }) {
  const canonical = `https://${host}/guias`;
  const items = GUIDES.map(g => `<li class="fc-li"><a href="/guias/${g.slug}"><b>${escapeHtml(g.title)}</b></a><br>${escapeHtml(g.description)}</li>`).join("");
  const body = crumbs([{ name: "Início", href: "/" }, { name: "Guias" }]) + `
    <section class="section fc-wrap" style="padding-top:16px;">
      <h1 class="fc-h1">Guias para vender um carro usado em Portugal</h1>
      <p class="fc-p">O que um vendedor particular precisa de saber, do preço aos papéis: documentos, registo de propriedade, crédito e reserva, carros importados, burlas e o que fazer depois da venda. Escritos a partir das páginas oficiais do IRN, do gov.pt e da Autoridade Tributária, com os números do nosso acompanhamento do mercado.</p>
      ${market && market.s30 != null ? `<p class="fc-p">${marketLine(market)}</p>` : ""}
      <ul class="fc-ul">${items}</ul>
      ${stats && stats.listings ? provenance({ n: stats.listings, builtAt, measure: "Preço pedido em anúncios ativos (mediana e P25-P75); dias até sair do OLX" }) : ""}
      <p class="fc-p" style="margin-top:18px;"><a href="/vender">Quanto pedir por modelo</a> · <a href="/avaliar">Avaliar o meu carro</a> · <a href="/liquidez">Tempo de venda</a></p>
    </section>
    <div style="height:60px;"></div>`;
  return layout({
    title: "Guias para vender um carro usado em Portugal",
    description: "Documentos, registo de propriedade, crédito e reserva, carros importados, burlas e o que fazer depois da venda: guias para vendedores particulares, com os números do mercado.",
    canonical, body, zone: "all", nav: "avaliar", depositCount, index: true, host,
    jsonLd: {
      "@context": "https://schema.org",
      "@graph": [
        { "@type": "CollectionPage", "url": canonical, "inLanguage": "pt-PT", "name": "Guias para vender um carro usado em Portugal" },
        breadcrumbLd(host, [{ name: "Início", href: "/" }, { name: "Guias" }]),
      ],
    },
  });
}
