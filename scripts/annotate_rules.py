#!/usr/bin/env python3
"""Rule-based annotation of 7 new fields for training data.

Fast, local, no API needed. Conservative: outputs null when uncertain.
"""

import json
import re
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _lower_no_accents(text: str) -> str:
    """Lowercase and normalize common Portuguese accents for matching."""
    t = text.lower()
    for a, b in [("á", "a"), ("à", "a"), ("ã", "a"), ("â", "a"),
                 ("é", "e"), ("ê", "e"), ("í", "i"), ("ó", "o"),
                 ("ô", "o"), ("õ", "o"), ("ú", "u"), ("ç", "c")]:
        t = t.replace(a, b)
    return t


def extract_urgency(desc: str) -> str | None:
    t = _lower_no_accents(desc)
    # High urgency
    high_patterns = [
        r"\burgente\b", r"preciso\s+vender", r"vend[ao]\s+r[aá]pido",
        r"emigra[cç][aã]o", r"pre[cç]o\s+para\s+despachar",
        r"motivo\s+de\s+viagem", r"vou\s+emigrar",
        r"ultima\s+semana", r"preciso\s+do\s+dinheiro",
        r"venda\s+urgente", r"urgencia",
    ]
    for p in high_patterns:
        if re.search(p, t):
            return "high"

    # Medium urgency
    medium_patterns = [
        r"aceito\s+propostas", r"preco\s+negociavel", r"negociavel",
        r"oportunidade", r"abaixo\s+d[oe]\s+mercado",
        r"preco\s+fixo\s+e\s+justo", r"melhor\s+oferta",
        r"faco\s+bom\s+preco",
    ]
    for p in medium_patterns:
        if re.search(p, t):
            return "medium"

    return "low"


def extract_warranty(desc: str) -> bool | None:
    t = _lower_no_accents(desc)
    # Negative first
    if re.search(r"sem\s+garantia", t):
        return False
    if re.search(r"nao\s+(tem|inclui|oferec)\w*\s+garantia", t):
        return False
    # Positive
    if re.search(r"garantia\s+d[eo]\s+\d+", t):
        return True
    if re.search(r"garantia\s+d[eo]\s+(motor|caixa|stand|fabrica|marca)", t):
        return True
    if re.search(r"garantia\s+(incluida|oferecida)", t):
        return True
    if re.search(r"oferec\w+\s+garantia", t):
        return True
    if re.search(r"com\s+garantia", t):
        return True
    if re.search(r"\bgarantia\b", t):
        # Mentioned but context unclear — still likely positive
        # Check it's not "sem garantia" (already handled above)
        return True
    return None


def extract_tuning_or_mods(desc: str) -> list:
    t = _lower_no_accents(desc)
    mods = []
    patterns = {
        r"reprogramac[aã]o\s*(stage\s*\d)?(\s*\w+)?": "reprogramação",
        r"stage\s*[12]\b[^,.)]*": "stage",
        r"\bremap\b": "remap",
        r"chip\s*tuning": "chip tuning",
        r"suspens[aã]o\s+rebaixada": "suspensão rebaixada",
        r"\bcoilovers?\b": "coilovers",
        r"escape\s+desportivo(?!\s+original)": "escape desportivo",
        r"\bdownpipe\b": "downpipe",
        r"\bwrap\b": "wrap",
        r"\bbodykit\b": "bodykit",
        r"\bbody\s*kit\b": "bodykit",
        r"turbo\s+upgrade": "turbo upgrade",
        r"intercooler\s+(aftermarket|upgrade|maior)": "intercooler upgrade",
        r"sem\s+catalisador": "sem catalisador",
        r"catalisador\s+removido": "catalisador removido",
        r"pop\s*&?\s*bang": "pop & bang",
        r"admiss[aã]o\s+(desportiv|aftermarket|aberta)": "admissão desportiva",
    }
    for pat, label in patterns.items():
        m = re.search(pat, t)
        if m:
            # Use the actual match text for stage/reprogramação to preserve details
            if label in ("reprogramação", "stage"):
                matched = m.group(0).strip()
                # Avoid duplicates like "reprogramação stage 1" + "stage 1"
                if not any(matched in existing or existing in matched for existing in mods):
                    mods.append(matched)
            else:
                if label not in mods:
                    mods.append(label)

    # Special: "escape desportivo original" is factory, not a mod
    # Already handled by negative lookahead above

    # H&R, KW, Bilstein etc. aftermarket suspension brands
    # KW: exclude "kWh" (battery) and standalone "kW" (power) — common in EV listings
    suspension_brands = re.findall(r"\b(h&r|bilstein|eibach|tein)\b", t)
    if re.search(r"\bkw\b(?!\s*h\b)(?!\s*\d)", t) and not re.search(r"\d+\s*kw\b", t):
        suspension_brands.append("kw")
    for brand in suspension_brands:
        label = f"suspensão {brand.upper()}"
        if label not in mods:
            mods.append(label)

    return mods


def extract_taxi_fleet_rental(desc: str) -> bool | None:
    t = _lower_no_accents(desc)
    patterns = [
        r"\bex[\s-]?taxi\b", r"\btaxi\b", r"\btvde\b",
        r"\buber\b", r"\bbolt\b", r"\bcabify\b",
        r"rent[\s-]?a[\s-]?car", r"\bfrota\b",
        r"carro\s+de\s+empresa", r"viatura\s+de\s+empresa",
        r"leasing\s+terminado", r"fim\s+de\s+leasing",
        r"ex[\s-]?aluguer",
    ]
    for p in patterns:
        if re.search(p, t):
            return True
    return None


def extract_recent_maintenance(desc: str) -> list:
    """Extract specific completed maintenance work."""
    t = _lower_no_accents(desc)
    maint = []

    # Pattern: "X trocado/a/os/as" or "trocou-se X" or "X novo/a/os/as"
    maintenance_items = [
        (r"correia\s+d[eo]\s+distribui[cç][aã]o[\w\s,]*(troca|feita|nova|substituida|aos\s+\d+)", "correia distribuição"),
        (r"kit\s+distribui[cç][aã]o[\w\s,]*(troca|feita|nova|substituida|aos\s+\d+)", "kit distribuição"),
        (r"embreagem\s+(nova|trocada|substituida|feita)", "embreagem nova"),
        (r"embrai?agem\s+(nova|trocada|substituida|feita)", "embreagem nova"),
        (r"(trocou|trocad[oa]s?|novos?|novas?).{0,30}(trav[oõ]es|pastilhas)", "travões"),
        (r"(pastilhas|discos)\s+d[eo]\s+trav[aã]o\s+(novos?|novas?|trocad[oa]s?)", "pastilhas/discos travão"),
        (r"bomba\s+d[eo]\s+[aá]gua\s+(nova|trocada)", "bomba de água"),
        (r"revis[aã]o\s+(feita|completa|geral|na\s+marca)", "revisão"),
        (r"(óleo|oleo)\s+(e\s+filtros?\s+)?(trocad|mudad|feita)", "óleo e filtros"),
        (r"filtros?\s+(trocad|novos?|mudad)", "filtros"),
        (r"amortecedores?\s+(novos?|trocad)", "amortecedores"),
        (r"molas?\s+(novas?|trocad)", "molas"),
        (r"bateria\s+(nova|trocada)", "bateria nova"),
        (r"pneus?\s+(novos?|trocad)", "pneus"),
        (r"bomba\s+d[eo]\s+combust[ií]vel\s+(nova|trocada)", "bomba combustível"),
        (r"juntas?\s+homo[\s-]?cin[eé]ticas?\s+(novas?|trocad)", "juntas homocinéticas"),
        (r"volante\s+d[eo]\s+motor\s+(novo|trocado)", "volante do motor"),
        (r"turbo\s+(novo|recondicionado|trocad)", "turbo"),
        (r"injetores?\s+(novos?|recondicionad|trocad)", "injetores"),
        (r"bomba\s+d[eo]\s+direc[cç][aã]o\s+(nova|trocada)", "bomba direção"),
        (r"radiador\s+(novo|trocad)", "radiador"),
        (r"alternador\s+(novo|trocad)", "alternador"),
        (r"motor\s+d[eo]\s+arranque\s+(novo|trocad)", "motor de arranque"),
        (r"valvulina\s+(trocada|nova|mudada)", "valvulina caixa"),
    ]

    for pat, label in maintenance_items:
        m = re.search(pat, t)
        if m:
            # Try to extract km context
            ctx = t[max(0, m.start() - 40):m.end() + 40]
            km_match = re.search(r"(\d[\d.]*)\s*km", ctx)
            if km_match:
                km_val = km_match.group(1).replace(".", "")
                label_with_km = f"{label} ({km_val}km)"
                if label_with_km not in maint and label not in [m.split(" (")[0] for m in maint]:
                    maint.append(label_with_km)
            else:
                if label not in maint and label not in [m.split(" (")[0] for m in maint]:
                    maint.append(label)

    # Also check for block-style lists of replaced parts
    # "Peças trocadas recentemente:" followed by a list
    block_match = re.search(r"(pecas|material)\s+(trocad|substituíd)\w*[\s:]*\n", t)
    if block_match and not maint:
        # There's a list but our patterns didn't catch specifics
        # Mark as generic maintenance done
        maint.append("múltiplas peças trocadas (ver descrição)")

    return maint


def extract_tires_condition(desc: str) -> str | None:
    t = _lower_no_accents(desc)
    # New
    if re.search(r"(4|quatro)?\s*pneus?\s+novos?", t):
        return "new"
    if re.search(r"pneus?\s+(acabad|recente)\w*\s+d[eo]\s+(trocar|colocar|montar)", t):
        return "new"
    # Good
    if re.search(r"pneus?\s+(em\s+)?(bom|optimo|otimo|excelente)\s+estado", t):
        return "good"
    if re.search(r"(8|9)0\s*%\s*(de\s+)?piso", t):
        return "good"
    if re.search(r"pneus?\s+(bons?|recentes?)", t):
        return "good"
    # Fair
    if re.search(r"pneus?\s+(em\s+)?estado\s+razo[aá]vel", t):
        return "fair"
    if re.search(r"(5|6)0\s*%\s*(de\s+)?piso", t):
        return "fair"
    # Poor
    if re.search(r"pneus?\s+(gast|desgastad)", t):
        return "poor"
    if re.search(r"precisa\s+d[eo]\s+pneus?", t):
        return "poor"
    return None


def extract_first_owner_selling(desc: str) -> bool | None:
    t = _lower_no_accents(desc)
    patterns = [
        r"1\s*dono\s+(desde\s+novo|desde\s+0\s*km)",
        r"(unico|primeiro)\s+dono",
        r"comprad[oa]\s+nov[oa]\s+por\s+mim",
        r"\b1\s*dono\b",
        r"um\s+dono",
        r"sempre\s+o\s+mesmo\s+dono",
        r"unica\s+dona",
        r"um\s+registo",
    ]
    for p in patterns:
        if re.search(p, t):
            return True
    return None


def annotate_example(messages: list[dict]) -> list[dict]:
    """Add 7 new fields to the assistant response JSON."""
    if len(messages) < 2:
        return messages

    user_content = messages[0]["content"]
    # Extract description (after the prompt prefix)
    prompt_end = user_content.find("\n\n")
    if prompt_end > 0:
        desc = user_content[prompt_end + 2:]
    else:
        desc = user_content

    try:
        existing = json.loads(messages[1]["content"])
    except json.JSONDecodeError:
        return messages

    # Extract new fields
    existing["urgency"] = extract_urgency(desc)
    existing["warranty"] = extract_warranty(desc)
    existing["tuning_or_mods"] = extract_tuning_or_mods(desc)
    existing["taxi_fleet_rental"] = extract_taxi_fleet_rental(desc)
    existing["recent_maintenance"] = extract_recent_maintenance(desc)
    existing["tires_condition"] = extract_tires_condition(desc)
    existing["first_owner_selling"] = extract_first_owner_selling(desc)

    messages[1]["content"] = json.dumps(existing, ensure_ascii=False)
    return messages


def main():
    import argparse as _ap
    _p = _ap.ArgumentParser()
    _p.add_argument("--input", default=str(DATA_DIR / "training_data.jsonl"))
    _p.add_argument("--output", default=str(DATA_DIR / "training_data_v2.jsonl"))
    _args = _p.parse_args()
    input_path = Path(_args.input)
    output_path = Path(_args.output)

    with open(input_path) as f:
        examples = [json.loads(line) for line in f]

    print(f"Loaded {len(examples)} examples")

    stats = {
        "urgency_high": 0, "urgency_medium": 0, "urgency_low": 0,
        "warranty_true": 0, "warranty_false": 0,
        "has_tuning": 0, "taxi": 0,
        "has_maintenance": 0,
        "tires_found": 0, "first_owner": 0,
    }

    with open(output_path, "w") as f:
        for ex in examples:
            ex["messages"] = annotate_example(ex["messages"])

            # Collect stats
            assistant = json.loads(ex["messages"][1]["content"])
            u = assistant.get("urgency")
            if u == "high": stats["urgency_high"] += 1
            elif u == "medium": stats["urgency_medium"] += 1
            elif u == "low": stats["urgency_low"] += 1
            if assistant.get("warranty") is True: stats["warranty_true"] += 1
            if assistant.get("warranty") is False: stats["warranty_false"] += 1
            if assistant.get("tuning_or_mods"): stats["has_tuning"] += 1
            if assistant.get("taxi_fleet_rental"): stats["taxi"] += 1
            if assistant.get("recent_maintenance"): stats["has_maintenance"] += 1
            if assistant.get("tires_condition"): stats["tires_found"] += 1
            if assistant.get("first_owner_selling"): stats["first_owner"] += 1

            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\nAnnotated → {output_path}")
    print(f"\n--- Stats ---")
    print(f"Urgency:     high={stats['urgency_high']}, medium={stats['urgency_medium']}, low={stats['urgency_low']}")
    print(f"Warranty:    yes={stats['warranty_true']}, no={stats['warranty_false']}")
    print(f"Tuning/mods: {stats['has_tuning']} listings")
    print(f"Taxi/fleet:  {stats['taxi']} listings")
    print(f"Maintenance: {stats['has_maintenance']} listings")
    print(f"Tires info:  {stats['tires_found']} listings")
    print(f"First owner: {stats['first_owner']} listings")


if __name__ == "__main__":
    main()
