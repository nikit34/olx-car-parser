"""Enrich listing data using a local Ollama model.

Extracts 15 structured fields from title + description text:
sub_model, trim_level, mileage, accident/repair flags, condition,
urgency, warranty, tuning, taxi/fleet, first owner, customs, RHD,
damage_severity.
"""

import json
import logging
import re
import threading
from pathlib import Path

import httpx
import yaml

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "settings.yaml"


# ---------------------------------------------------------------------------
# Post-rules — run *after* the LLM emits its JSON, to make the result robust
# to known model mistakes without needing a retrain.  Each pattern is derived
# from failures observed on the golden eval set at /tmp/olx_golden/.
# ---------------------------------------------------------------------------

_PARTS_CAR_PATTERN = re.compile(
    r"para\s+pe[çc]as|vender\s+as\s+pe[çc]as|venda\s+de\s+pe[çc]as|"
    r"para\s+desmanchar|s[óo]\s+pe[çc]as|abate|salvado|"
    r"\bavariado\b|\bimobilizado\b",
    re.IGNORECASE,
)

_DAMAGE_PATTERN = re.compile(
    r"\bavariado\b|\bimobilizado\b|\bpartido\b|\bdanificado\b|\bestragado\b|"
    r"fuga\s+de|para\s+pe[çc]as|salvado|sinistro|acidente|batido|"
    r"precisa\s+de\s+(?:reparo|arranjo|conserto)|necessita\s+de\s+repara|"
    r"\brepara(?:r|do|da|dos|das)\b|\bsubstitu(?:ir|ido|ida|idos|idas)\b",
    re.IGNORECASE,
)

_RHD_PATTERN = re.compile(
    r"m[ãa]o\s+inglesa|volante\s+[àa]\s+direita|matr[ií]cula\s+inglesa|"
    r"condu[çc][ãa]o\s+[àa]\s+direita|\bRHD\b|right[\s-]hand\s+drive",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Schema documentation (kept as a Python list so consumers — eval scripts,
# annotation tools — share one source of truth for the field set).
# ---------------------------------------------------------------------------

_FIELD_NAMES = [
    "sub_model", "trim_level", "desc_mentions_num_owners",
    "desc_mentions_accident", "desc_mentions_repair",
    "mileage_in_description_km", "desc_mentions_customs_cleared",
    "right_hand_drive", "mechanical_condition", "urgency",
    "warranty", "tuning_or_mods", "taxi_fleet_rental",
    "first_owner_selling",
    # 0-3 severity inferred from the description as a whole — replaces the
    # rule-based damage_score for the price model. Captures "para peças",
    # "sem matrícula", "sinistro com toque", and similar phrasings the
    # boolean accident/repair flags miss because they fire as a whole only
    # on the most explicit mentions.
    "damage_severity",
]


_SYSTEM_PROMPT = """\
Extract structured features from a Portuguese (pt-PT) car listing. Return ONE JSON object with all of the following keys; use null for anything not mentioned or unclear.

Field rules:
sub_model: engine/body variant only (displacement+fuel+power code), e.g. "320d","1.6 TDI","2.0 TFSI","A 200","CLA 45". NOT a trim/package like "AMG Line","M Sport". NOT a bare model name like "DS3","Qashqai".
trim_level: equipment line, e.g. "AMG Line","M Sport","S-Line","GTI","FR","Tekna". null if basic.
mileage_in_description_km: integer km. "mil"=thousand ONLY as a separate word ("150 mil km"→150000; "89.500km"→89500; "4300 km"→4300; "127 mil km"→127000).
desc_mentions_accident: "sinistro","acidente","batido".
desc_mentions_repair: only if damage/breakdown is mentioned ("avariado","imobilizado","partido","danificado"). Routine maintenance ("óleo mudado","correia mudada","pastilhas novas","revisão feita") is NOT repair — keep false.
desc_mentions_customs_cleared: "desalfandegado","legalizado","por legalizar".
right_hand_drive: "mão inglesa","volante à direita","matrícula inglesa". Generic "importado" alone is NOT enough.
urgency: high="urgente","emigração","preço para despachar"; medium="aceito propostas","negociável","oportunidade"; low otherwise.
warranty: "garantia" mentioned positively (not "sem garantia").
tuning_or_mods: aftermarket mods only: ["reprogramação","stage 1","remap","coilovers","bodykit","escape desportivo","downpipe","wrap"]. Empty list if none.
taxi_fleet_rental: "ex-táxi","TVDE","Uber","Bolt","rent-a-car","frota","carro de empresa".
first_owner_selling: "1 dono desde novo","único dono","comprado novo por mim","vendo o meu".

damage_severity: 0 (pristine: "como novo","estado impecável", warranty mentioned, "primeiro dono comprado novo"); 1 (normal age-appropriate wear, no damage signals — DEFAULT for typical used-car language); 2 (needs significant repair OR accident history: "sinistro","embate","toque dianteiro/traseiro/lateral","necessita reparações","pintura fraca","amortecedores partidos","precisa óleo/correia"); 3 (salvage / parts-only / non-runner: "para peças","vender as peças","para sucata","para desmanchar","abate","salvado","motor fundido","imobilizado","não anda","sem matrícula","para exportação/utilização das peças"). When in doubt between two levels, pick the higher one.

PARTS-CAR OVERRIDE — if description contains ANY of "para peças","vender as peças","venda de peças","para desmanchar","só peças","abate","salvado","avariado","imobilizado", you MUST set ALL FOUR: mechanical_condition="poor", desc_mentions_accident=true, desc_mentions_repair=true, damage_severity=3.

Examples:

Listing: "BMW Série 3 320d Pack M com 180.000 km, 1 dono, garantia até 2026, sem sinistros."
→ sub_model="320d", trim_level="Pack M", desc_mentions_num_owners=1, desc_mentions_accident=false, desc_mentions_repair=false, mileage_in_description_km=180000, mechanical_condition="excellent", urgency="low", warranty=true, tuning_or_mods=[], first_owner_selling=true, damage_severity=0 (rest null).

Listing: "Vendo Audi A3 1.6 TDI S-Line, 150 mil km. Avariado motor, vendo para peças. Aceito propostas."
→ sub_model="1.6 TDI", trim_level="S-Line", desc_mentions_accident=true, desc_mentions_repair=true, mileage_in_description_km=150000, mechanical_condition="poor", urgency="medium", tuning_or_mods=[], damage_severity=3 (rest null).

Listing: "Seat Ibiza FR 1.4 TSI com 89.500km. Reprogramação stage 1, escape desportivo. Revisão feita, pneus novos."
→ sub_model="1.4 TSI", trim_level="FR", desc_mentions_accident=false, desc_mentions_repair=false, mileage_in_description_km=89500, mechanical_condition="good", urgency="low", tuning_or_mods=["reprogramação","stage 1","escape desportivo"], damage_severity=1 (rest null).

Listing: "Honda Civic 2009 com toque dianteiro esquerdo, mecânica boa, vendo para peças."
→ desc_mentions_accident=true, desc_mentions_repair=true, mechanical_condition="poor", damage_severity=3 (rest null).
"""


# ---------------------------------------------------------------------------
# Config / availability
# ---------------------------------------------------------------------------

def _get_config() -> dict:
    cfg = {}
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            data = yaml.safe_load(f) or {}
        cfg = data.get("llm", {})
    return {
        "ollama_model": cfg.get("ollama_model", "qwen3:4b-instruct"),
        "ollama_url": cfg.get("ollama_url", "http://localhost:11434"),
        "max_workers": cfg.get("max_workers", 6),
        "max_tokens": cfg.get("max_tokens", 600),
        "max_chars": cfg.get("max_chars", 4000),
    }


_ollama_status: bool | None = None

# Thread-local persistent httpx.Client. Reusing the TCP connection saves
# ~10-30 ms per call (handshake + slow-start) which on a 1700-listing batch
# adds up to ~30-50 s.
_thread_local = threading.local()


def _get_client(base_url: str) -> httpx.Client:
    client = getattr(_thread_local, "http_client", None)
    if client is None:
        client = httpx.Client(base_url=base_url,
                              timeout=httpx.Timeout(120.0, connect=10.0))
        _thread_local.http_client = client
    return client


def _ollama_available(_url: str = "") -> bool:
    """True iff the local Ollama HTTP API answers. Result cached per process
    so we don't probe `/api/tags` on every enrichment call."""
    global _ollama_status
    if _ollama_status is not None:
        return _ollama_status
    cfg = _get_config()
    url = _url or cfg.get("ollama_url", "http://localhost:11434")
    try:
        resp = _get_client(url).get("/api/tags", timeout=2.0)
        _ollama_status = resp.status_code == 200
    except Exception:
        _ollama_status = False
    if not _ollama_status:
        logger.warning("Ollama not reachable at %s — LLM enrichment disabled.", url)
    return _ollama_status


def _llm_available() -> bool:
    """Backwards-compat name. Same semantics as _ollama_available()."""
    return _ollama_available()


# ---------------------------------------------------------------------------
# Core API call
# ---------------------------------------------------------------------------

def _call_ollama(text: str, cfg: dict) -> dict | None:
    """Run one extraction round-trip against Ollama. Returns the parsed JSON
    payload as a dict, or None on any failure.

    Uses /api/generate (not /api/chat) on purpose:
      - the `system` field is byte-stable across calls, so Ollama keeps the
        same KV-cache slot and skips ~600 tokens of prefill every call;
      - no chat-template wrapping → fewer tokens, no template-version drift;
      - keep_alive holds the model in RAM between bursts so we don't pay the
        5 s reload cost on the M1 8 GB box.
    format=json constrains the output to parseable JSON; the system prompt
    already documents the 15-field schema, so any instruction-tuned model
    (e.g. qwen3:4b-instruct) matches it without a separate tool wrapper.
    """
    url = cfg.get("ollama_url", "http://localhost:11434")
    payload = {
        "model": cfg.get("ollama_model", "qwen3:4b-instruct"),
        "system": _SYSTEM_PROMPT,
        "prompt": text[:cfg.get("max_chars", 4000)],
        "format": "json",
        "stream": False,
        "keep_alive": "30m",
        "options": {
            "temperature": 0.0,
            "num_predict": cfg.get("max_tokens", 600),
        },
    }
    for attempt in range(2):
        try:
            resp = httpx.post(f"{url}/api/generate", json=payload, timeout=120.0)
            if resp.status_code != 200:
                logger.warning("Ollama HTTP %s (attempt %d)", resp.status_code, attempt + 1)
                continue
            content = resp.json().get("response", "")
            if not content:
                return None
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                # Some fine-tuned checkpoints occasionally wrap with ```json …```;
                # one strip pass before giving up.
                stripped = content.strip().strip("`").lstrip("json").strip()
                try:
                    parsed = json.loads(stripped)
                except json.JSONDecodeError:
                    logger.debug("Ollama returned non-JSON: %s", content[:200])
                    return None
            return parsed if isinstance(parsed, dict) else None
        except httpx.RequestError as e:
            logger.warning("Ollama connection error (attempt %d): %s", attempt + 1, e)
        except Exception as e:  # noqa: BLE001 — last-resort log so a worker never dies
            logger.debug("Ollama enrichment failed: %s", e)
            return None
    return None


# Public alias used by callers and tests; lets us swap the backend later
# without touching cli.py / enrich_local.py / the test patch targets.
def _call_llm(text: str, cfg: dict) -> dict | None:
    return _call_ollama(text, cfg)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_EXTRAS_KEY_ALIASES = {
    "desc_mentions_num_owners": "num_owners",
    "desc_mentions_accident": "had_accident",
    "desc_mentions_repair": "needs_repair",
    "desc_mentions_customs_cleared": "customs_cleared",
}


def enrich_from_description(description: str, title: str = "") -> dict | None:
    """Extract structured data from title + description via the local LLM.

    Returns dict with extracted fields, or None on failure / when Ollama is
    unreachable / when the description is too short to bother.
    """
    if not description or len(description.strip()) < 20:
        return None

    if not _llm_available():
        return None

    cfg = _get_config()
    text = f"{title}\n{description}" if title else description
    result = _call_llm(text, cfg)
    return normalize_llm_extras(result) if result else None


def normalize_llm_extras(extras: dict | None) -> dict | None:
    """Normalize legacy and current extraction keys to the current schema."""
    if not extras:
        return extras
    normalized = dict(extras)
    for new_key, old_key in _EXTRAS_KEY_ALIASES.items():
        if new_key not in normalized and old_key in normalized:
            normalized[new_key] = normalized[old_key]
    return normalized


def _get_extra(extras: dict, key: str):
    legacy_key = _EXTRAS_KEY_ALIASES.get(key)
    if key in extras:
        return extras.get(key)
    if legacy_key:
        return extras.get(legacy_key)
    return extras.get(key)


def enrich_listings_batch(listings: list, batch_size: int = 50) -> int:
    """Enrich a batch of RawListing objects with LLM-extracted data.

    Modifies listings in place. Returns count of enriched listings.
    """
    if not _llm_available():
        logger.info("Ollama not available. Skipping LLM enrichment.")
        return 0

    cfg = _get_config()
    logger.info("LLM enrichment using Ollama (%s) for up to %d listings",
                cfg["ollama_model"], min(batch_size, len(listings)))

    enriched = 0
    failures = 0
    for listing in listings[:batch_size]:
        if not listing.description:
            continue

        result = enrich_from_description(listing.description, getattr(listing, "title", ""))
        if result:
            listing._llm_extras = normalize_llm_extras(result)
            enriched += 1
            failures = 0
        else:
            failures += 1
            if failures >= 5:
                logger.warning("5 consecutive LLM failures, stopping enrichment.")
                break

    logger.info("LLM-enriched %d / %d listings", enriched, len(listings))
    return enriched


# ---------------------------------------------------------------------------
# Data correction — cross-check and fix attributes using LLM-extracted data
# ---------------------------------------------------------------------------

def correct_listing_data(listing) -> dict:
    """Cross-check listing attributes against LLM-extracted data and return corrections."""
    extras = getattr(listing, "_llm_extras", None)
    if extras is None:
        return {}
    extras = normalize_llm_extras(extras)

    corrections = {}

    # --- Mileage cross-check ---
    desc_km = extras.get("mileage_in_description_km")
    attr_km = listing.mileage_km

    if desc_km and isinstance(desc_km, (int, float)) and desc_km > 0:
        desc_km = int(desc_km)
        if attr_km and attr_km > 0:
            if desc_km > attr_km * 1.3 and (desc_km - attr_km) > 5000:
                corrections["real_mileage_km"] = desc_km
                logger.info(
                    "Mileage mismatch for %s: attribute=%d, description=%d → using description",
                    listing.url, attr_km, desc_km,
                )
            else:
                corrections["real_mileage_km"] = desc_km
        elif not attr_km or attr_km == 0:
            corrections["real_mileage_km"] = desc_km
    elif attr_km and attr_km > 0:
        corrections["real_mileage_km"] = attr_km

    sub_model = extras.get("sub_model")
    if sub_model and isinstance(sub_model, str) and sub_model.strip():
        corrections["sub_model"] = sub_model.strip()

    trim = extras.get("trim_level")
    if trim and isinstance(trim, str) and trim.strip():
        corrections["trim_level"] = trim.strip()

    num_owners = _get_extra(extras, "desc_mentions_num_owners")
    if num_owners and isinstance(num_owners, (int, float)) and num_owners > 0:
        corrections["desc_mentions_num_owners"] = int(num_owners)

    needs_repair = _get_extra(extras, "desc_mentions_repair")
    if needs_repair is not None:
        corrections["desc_mentions_repair"] = bool(needs_repair)

    had_accident = _get_extra(extras, "desc_mentions_accident")
    if had_accident is not None:
        corrections["desc_mentions_accident"] = bool(had_accident)

    customs = _get_extra(extras, "desc_mentions_customs_cleared")
    if customs is not None:
        corrections["desc_mentions_customs_cleared"] = bool(customs)

    rhd = extras.get("right_hand_drive")
    if rhd is not None:
        corrections["right_hand_drive"] = bool(rhd)

    urgency = extras.get("urgency")
    if urgency in ("high", "medium", "low"):
        corrections["urgency"] = urgency

    warranty = extras.get("warranty")
    if warranty is not None:
        corrections["warranty"] = bool(warranty)

    tuning = extras.get("tuning_or_mods")
    if tuning and isinstance(tuning, list) and len(tuning) > 0:
        corrections["tuning_or_mods"] = json.dumps(tuning, ensure_ascii=False)

    taxi = extras.get("taxi_fleet_rental")
    if taxi is not None:
        corrections["taxi_fleet_rental"] = bool(taxi)

    first_owner = extras.get("first_owner_selling")
    if first_owner is not None:
        corrections["first_owner_selling"] = bool(first_owner)

    mech = extras.get("mechanical_condition")
    if mech in ("excellent", "good", "fair", "poor"):
        corrections["mechanical_condition"] = mech

    severity = extras.get("damage_severity")
    if severity is not None and isinstance(severity, (int, float)):
        sev_int = int(severity)
        if 0 <= sev_int <= 3:
            corrections["damage_severity"] = sev_int

    if corrections.get("desc_mentions_accident") and not corrections.get("desc_mentions_repair"):
        corrections["desc_mentions_repair"] = True

    description = getattr(listing, "description", "") or ""
    if description:
        if _PARTS_CAR_PATTERN.search(description):
            corrections["mechanical_condition"] = "poor"
            corrections["desc_mentions_accident"] = True
            corrections["desc_mentions_repair"] = True
            # Belt-and-suspenders: any parts-car phrase in the raw text
            # forces severity 3, even if the LLM under-rated it.
            corrections["damage_severity"] = 3
        else:
            if corrections.get("desc_mentions_repair") and not _DAMAGE_PATTERN.search(description):
                corrections["desc_mentions_repair"] = False

        if corrections.get("right_hand_drive") and not _RHD_PATTERN.search(description):
            corrections["right_hand_drive"] = False

    return corrections


def apply_corrections(listings: list) -> int:
    """Apply data corrections to all listings that have LLM extras."""
    corrected = 0
    for listing in listings:
        corrections = correct_listing_data(listing)
        if not corrections:
            continue

        if not hasattr(listing, "_corrections"):
            listing._corrections = {}
        listing._corrections.update(corrections)
        corrected += 1

        if corrections.get("real_mileage_km") and corrections.get("real_mileage_km") != getattr(listing, "mileage_km", None):
            logger.info(
                "Corrected %s: real_mileage=%s, desc_mentions_repair=%s, desc_mentions_accident=%s",
                listing.olx_id,
                corrections.get("real_mileage_km"),
                corrections.get("desc_mentions_repair"),
                corrections.get("desc_mentions_accident"),
            )

    logger.info("Applied corrections to %d / %d listings", corrected, len(listings))
    return corrected
