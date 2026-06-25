"""Condition NLP: disclosed minor mechanical faults a spec-only model can't see.

The 2026-06-25 cheap-tail audit (50 blind live-OLX appraisals + 12.8k-car
population pass) found the price model is condition-blind: it predicts ≈ the
comparable median and ignores faults disclosed in the listing text. A
``check-engine`` light, a ``catalisador a precisar`` or a ``precisa de
reparação`` lands at ``damage_severity == 1`` (normal wear) today, so the model
over-prices it — and these concentrate in the <€4k tail where a €300-800 fault
is a large fraction of the car's value.

This module is the missing tier *between* normal wear and the severity-3 hard
blocks in :mod:`src.parser.llm_enrichment` (parts-only / non-runner / severe
mechanical). It does NOT block the deal — the car runs — it translates a
disclosed fault into a repair provision the deal scorer subtracts from net
margin (:func:`src.analytics.decision.decide` step 5b). Deliberately
precision-biased: a false positive only makes the scorer more cautious, which
matches the project's "quality over coverage" stance.
"""

import re

# Unambiguous "runs but has a disclosed fault" phrases. Salvage / non-runner /
# severe-mechanical phrasings are intentionally NOT here — those hit the
# severity-3 hard blocks upstream. Each alternative is written to be self-
# evidently a fault so the negation surface is small.
_FAULT_PHRASES = [
    # dashboard warning light ON (guard against "apagada/desligada/sem")
    r"(?<!sem\s)luz\s+(?:da\s+|de\s+|do\s+)?"
    r"(?:inje[çc][ãa]o|avarias?|motor|airbag|abs|esp)\b"
    r"(?!\s*(?:apagad|desligad|off))",
    r"check\s*engine",
    r"avaria\s+(?:el[ée]tr[óo]nica|electr[óo]nica|el[ée]ctrica)",
    # catalisador / DPF / EGR / turbo issues (lighter than 'fundido' = severe)
    r"catalisador\s+(?:avariad|fundid|partid|roubad|cortad|a\s+precisar|"
    r"para\s+(?:trocar|substituir)|estragad)",
    r"(?:fap|dpf)\s+(?:entupid|avariad|a\s+precisar)",
    r"v[áa]lvula\s+egr\s+(?:avariad|entupid|a\s+precisar)",
    r"turbo\s+(?:a\s+precisar|a\s+apitar|com\s+folga|avariad|fundid)",
    # clutch / gearbox slipping (not 'avariada' = severe)
    r"embra(?:i|e)agem\s+(?:gasta|a\s+patinar|a\s+precisar|"
    r"para\s+(?:trocar|substituir))",
    # smoke / overheat / leaks
    r"(?:deita|deitar|a\s+deitar)\s+fumo",
    r"sobreaquec",
    r"(?:fuga|perde|perda|queima)\s+(?:de\s+)?[óo]leo",
    # failed / pending inspection
    r"(?:reprovad[oa]|chumb(?:ou|ado))\s+(?:na|em|à)?\s*inspe[çc][ãa]o",
    r"inspe[çc][ãa]o\s+(?:reprovad|com\s+(?:defeitos|anomalias))",
]
_FAULT_PATTERN = re.compile("|".join(_FAULT_PHRASES), re.IGNORECASE)

# "needs repair / work" family — guarded against immediate negation
# ("não precisa de …", "sem precisar de …"). Fixed-width lookbehind only, so
# the guard catches the common direct-negation cases.
_NEEDS_REPAIR_PATTERN = re.compile(
    r"(?<!n[ãa]o\s)(?<!sem\s)"
    r"(?:precisa|necessita|a\s+precisar)\s+de\s+"
    r"(?:uma?\s+|alguns?\s+|pequen[oa]s?\s+|alguma\s+)?"
    r"(?:repara[çc][ãa]o|repara[çc][õo]es|arranjo|conserto|interven[çc][ãa]o|"
    r"m[ãa]o\s+de\s+obra|mec[âa]nica|obras?\s+de\s+mec[âa]nica)",
    re.IGNORECASE,
)

# Repair provision sizing. Floor covers a typical cat / sensor / clutch job;
# the %-of-price term scales it on pricier cars; the cap and the half-the-car
# guard keep it sane on both ends.
_FAULT_COST_FLOOR = 400.0
_FAULT_COST_PCT = 0.18
_FAULT_COST_CAP = 1500.0


def detect_minor_fault(title: str | None, description: str | None) -> str | None:
    """Return a short label for the first disclosed minor fault, else None."""
    text = f"{title or ''} {description or ''}"
    m = _FAULT_PATTERN.search(text)
    if m:
        return re.sub(r"\s+", " ", m.group(0)).strip().lower()[:60]
    m = _NEEDS_REPAIR_PATTERN.search(text)
    if m:
        return re.sub(r"\s+", " ", m.group(0)).strip().lower()[:60]
    return None


def minor_fault_cost(
    title: str | None, description: str | None, price
) -> tuple[float, str | None]:
    """Repair provision (€, ``flag``) for a disclosed minor fault.

    Returns ``(0.0, None)`` when no fault phrase is present. The provision is
    ``clamp(price * 0.18, 400, 1500)`` and never exceeds half the asking price.
    """
    flag = detect_minor_fault(title, description)
    if not flag:
        return 0.0, None
    try:
        p = float(price)
    except (TypeError, ValueError):
        p = 0.0
    if p <= 0:
        return _FAULT_COST_FLOOR, flag
    cost = min(max(_FAULT_COST_FLOOR, p * _FAULT_COST_PCT), _FAULT_COST_CAP)
    cost = min(cost, p * 0.5)
    return round(cost, 0), flag
