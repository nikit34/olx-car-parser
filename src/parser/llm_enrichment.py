"""Enrich listing data using Claude (Haiku 4.5) via the Anthropic API.

Extracts 14 structured fields from title + description text:
sub_model, trim_level, mileage, accident/repair flags, condition,
urgency, warranty, tuning, taxi/fleet, first owner, customs, RHD.

Auth: Bearer token from `ANTHROPIC_AUTH_TOKEN` (set in .env). Sent via the
`auth_token` parameter so it goes as `Authorization: Bearer …` — required for
Claude Code OAuth / setup tokens, which `x-api-key` rejects.
"""

import json
import logging
import os
import re
import threading
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "settings.yaml"
ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"


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
# Tool schema — forces Claude to emit a single tool_use block with all fields.
# Unlike free-form JSON parsing this can never return "Sorry I can't" prose.
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

_EXTRACT_TOOL = {
    "name": "record_listing_features",
    "description": "Record extracted structured features from a Portuguese car listing. Always call this exactly once with all fields populated — use null when a field is not mentioned or cannot be inferred.",
    "input_schema": {
        "type": "object",
        "properties": {
            "sub_model": {
                "type": ["string", "null"],
                "description": "Engine/body variant (e.g. '320d', '1.6 TDI', '2.0 TFSI', 'A 200'). NOT a trim line, NOT a bare model name.",
            },
            "trim_level": {
                "type": ["string", "null"],
                "description": "Equipment line: 'AMG Line', 'M Sport', 'S-Line', 'GTI', 'FR', 'Tekna'. null if basic.",
            },
            "desc_mentions_num_owners": {"type": ["integer", "null"]},
            "desc_mentions_accident": {"type": ["boolean", "null"]},
            "desc_mentions_repair": {"type": ["boolean", "null"]},
            "mileage_in_description_km": {"type": ["integer", "null"]},
            "desc_mentions_customs_cleared": {"type": ["boolean", "null"]},
            "right_hand_drive": {"type": ["boolean", "null"]},
            "mechanical_condition": {
                "type": ["string", "null"],
                "enum": ["excellent", "good", "fair", "poor", None],
            },
            "urgency": {
                "type": ["string", "null"],
                "enum": ["high", "medium", "low", None],
            },
            "warranty": {"type": ["boolean", "null"]},
            "tuning_or_mods": {
                "type": ["array", "null"],
                "items": {"type": "string"},
            },
            "taxi_fleet_rental": {"type": ["boolean", "null"]},
            "first_owner_selling": {"type": ["boolean", "null"]},
            "damage_severity": {
                "type": ["integer", "null"],
                "enum": [0, 1, 2, 3, None],
                "description": "Overall vehicle state inferred from the description: 0=pristine/like-new, 1=normal age-appropriate wear (default for unspecified), 2=needs significant repair OR has accident history, 3=salvage / parts-only / non-runner.",
            },
        },
        "required": _FIELD_NAMES,
    },
}


_SYSTEM_PROMPT = """\
Extract structured features from a Portuguese (pt-PT) car listing. Always call the record_listing_features tool exactly once with all fields populated; use null for anything not mentioned or unclear.

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
# Client / config / availability
# ---------------------------------------------------------------------------

_client = None
_client_lock = threading.Lock()
_token_status: bool | None = None


def _read_env_file() -> dict:
    """Tiny .env reader so callers don't need python-dotenv installed."""
    if not ENV_PATH.exists():
        return {}
    out = {}
    for line in ENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def _get_token() -> str | None:
    """Resolve the Anthropic auth token. Env > .env file."""
    tok = os.environ.get("ANTHROPIC_AUTH_TOKEN")
    if tok:
        return tok
    return _read_env_file().get("ANTHROPIC_AUTH_TOKEN")


def _get_config() -> dict:
    cfg = {}
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            data = yaml.safe_load(f) or {}
        cfg = data.get("llm", {})
    return {
        "model": cfg.get("model", "claude-haiku-4-5"),
        "max_workers": cfg.get("max_workers", 16),
        "max_tokens": cfg.get("max_tokens", 600),
        "max_chars": cfg.get("max_chars", 4000),
    }


def _get_client():
    """Lazy, thread-safe Anthropic client. Returns None if SDK or token missing."""
    global _client
    if _client is not None:
        return _client
    with _client_lock:
        if _client is not None:
            return _client
        token = _get_token()
        if not token:
            return None
        try:
            import anthropic
        except ImportError:
            logger.error("`anthropic` package not installed. Run: pip install anthropic")
            return None
        # auth_token sets `Authorization: Bearer <token>`. Required for Claude
        # Code OAuth / setup tokens — `x-api-key` rejects them.
        _client = anthropic.Anthropic(auth_token=token, max_retries=2, timeout=60.0)
    return _client


def _llm_available() -> bool:
    """True iff a token is configured and the SDK loads. Cached for the process."""
    global _token_status
    if _token_status is not None:
        return _token_status
    _token_status = _get_client() is not None
    if not _token_status:
        logger.warning("Claude enrichment disabled: ANTHROPIC_AUTH_TOKEN missing or SDK unavailable.")
    return _token_status


# Backward-compat shim — older call sites in cli.py and tests reference this name.
def _ollama_available(_url: str = "") -> bool:
    return _llm_available()


# ---------------------------------------------------------------------------
# Core API call
# ---------------------------------------------------------------------------

def _call_llm(text: str, cfg: dict) -> dict | None:
    """Run one extraction round-trip. Returns the tool_use payload as dict, or None."""
    client = _get_client()
    if not client:
        return None

    try:
        import anthropic
    except ImportError:
        return None

    for attempt in range(2):
        try:
            resp = client.messages.create(
                model=cfg["model"],
                max_tokens=cfg.get("max_tokens", 600),
                # Prompt caching on the system prompt: after the first call in a
                # 5-minute window the ~600-token instruction block costs 0.1×.
                # The list-of-blocks form is required to attach cache_control.
                system=[{
                    "type": "text",
                    "text": _SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }],
                tools=[_EXTRACT_TOOL],
                tool_choice={"type": "tool", "name": "record_listing_features"},
                messages=[{"role": "user", "content": text[:cfg.get("max_chars", 4000)]}],
            )
            for block in resp.content:
                if getattr(block, "type", None) == "tool_use" and block.name == "record_listing_features":
                    return dict(block.input)
            logger.debug("Claude returned no tool_use block")
            return None

        except anthropic.RateLimitError as e:
            wait = 2 ** attempt
            logger.warning("Claude rate-limited: %s — sleeping %ds", e, wait)
            import time
            time.sleep(wait)
        except anthropic.APIStatusError as e:
            # 4xx (auth, request shape) won't be fixed by retry.
            if 400 <= e.status_code < 500:
                logger.error("Claude API error %s: %s", e.status_code, e.message)
                return None
            logger.warning("Claude server error %s (attempt %d)", e.status_code, attempt + 1)
        except anthropic.APIConnectionError as e:
            logger.warning("Claude connection error (attempt %d): %s", attempt + 1, e)
        except Exception as e:  # noqa: BLE001 — last-resort log so a worker never dies
            logger.debug("Claude enrichment failed: %s", e)
            return None
    return None


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
    """Extract structured data from title + description via Claude.

    Returns dict with extracted fields, or None on failure.
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
    """Enrich a batch of RawListing objects with Claude-extracted data.

    Modifies listings in place. Returns count of enriched listings.
    """
    if not _llm_available():
        logger.info("Claude not available. Skipping LLM enrichment.")
        return 0

    cfg = _get_config()
    logger.info("LLM enrichment using Claude (%s) for up to %d listings",
                cfg["model"], min(batch_size, len(listings)))

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
