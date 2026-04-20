"""Enrich listing data using a local Ollama model.

Extracts 14 structured fields from title + description text:
sub_model, trim_level, mileage, accident/repair flags, condition,
urgency, warranty, tuning, taxi/fleet, first owner, customs, RHD.
"""

import json
import logging
from pathlib import Path

import httpx
import yaml

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "settings.yaml"

OLLAMA_URL = "http://localhost:11434"

EXTRACTION_PROMPT = """\
Extract structured data from a Portuguese car listing. Output a single JSON object (null if unknown):
sub_model(str), trim_level(str), desc_mentions_num_owners(int), desc_mentions_accident(bool), desc_mentions_repair(bool), mileage_in_description_km(int), desc_mentions_customs_cleared(bool), right_hand_drive(bool), mechanical_condition("excellent"/"good"/"fair"/"poor"), urgency("high"/"medium"/"low"), warranty(bool), tuning_or_mods(list), taxi_fleet_rental(bool), first_owner_selling(bool).
Rules:
sub_model: engine/body variant only (displacement+fuel+power code), e.g. "320d","1.6 TDI","2.0 TFSI","A 200","CLA 45". NOT a trim/package like "AMG Line","M Sport". NOT a bare model name like "DS3","Qashqai".
trim_level: equipment line, e.g. "AMG Line","M Sport","S-Line","GTI","FR","Tekna". null if basic.
mileage_in_description_km: integer km. "mil"=thousand ONLY as a separate word ("150 mil km"→150000; "89.500km"→89500; "4300 km"→4300; "127 mil km"→127000).
desc_mentions_accident: "sinistro","acidente","batido".
desc_mentions_repair: only if damage/breakdown is mentioned ("avariado","imobilizado","partido","danificado"). Routine maintenance ("óleo mudado","correia mudada","pastilhas novas","revisão feita") is NOT repair — keep false.
desc_mentions_customs_cleared: "desalfandegado","legalizado","por legalizar".
right_hand_drive: "mão inglesa","volante à direita","matrícula inglesa".
urgency: high="urgente","emigração"; medium="aceito propostas","negociável".
warranty: "garantia" (not "sem garantia").
tuning_or_mods: ["reprogramação","stage 1","remap","coilovers","bodykit"].
taxi_fleet_rental: "ex-táxi","TVDE","Uber","Bolt".
first_owner_selling: "1 dono","único dono".
PARTS-CAR OVERRIDE — if the description contains ANY of "para peças","vender as peças","venda de peças","para desmanchar","só peças","abate","salvado","avariado","imobilizado", then you MUST set ALL THREE: mechanical_condition="poor", desc_mentions_accident=true, desc_mentions_repair=true.
"""


_EXTRAS_KEY_ALIASES = {
    "desc_mentions_num_owners": "num_owners",
    "desc_mentions_accident": "had_accident",
    "desc_mentions_repair": "needs_repair",
    "desc_mentions_customs_cleared": "customs_cleared",
}


def _get_config() -> dict:
    cfg = {}
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            data = yaml.safe_load(f) or {}
        cfg = data.get("llm", {})
    return {
        "ollama_model": cfg.get("ollama_model", "qwen3:4b-instruct"),
        "ollama_url": cfg.get("ollama_url", OLLAMA_URL),
        "llm_workers": cfg.get("llm_workers", 1),
    }


_ollama_status: bool | None = None

# Thread-local HTTP clients — one per thread for thread safety
import threading
_thread_local = threading.local()


def _get_client(base_url: str) -> httpx.Client:
    """Return a thread-local persistent httpx.Client."""
    client = getattr(_thread_local, "http_client", None)
    if client is None:
        client = httpx.Client(base_url=base_url, timeout=httpx.Timeout(60, connect=10))
        _thread_local.http_client = client
    return client


def _ollama_available(base_url: str) -> bool:
    """Check if Ollama is running locally. Result is cached for the process lifetime."""
    global _ollama_status
    if _ollama_status is not None:
        return _ollama_status
    try:
        client = _get_client(base_url)
        resp = client.get("/api/tags")
        _ollama_status = resp.status_code == 200
    except httpx.HTTPError:
        _ollama_status = False
    return _ollama_status


def _parse_llm_json(content: str) -> dict | None:
    """Extract JSON from LLM response, handling markdown code blocks and surrounding text."""
    # Try markdown code blocks first
    if "```" in content:
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        return json.loads(content.strip())
    # Extract first {...} object from response
    start = content.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(content)):
            if content[i] == "{":
                depth += 1
            elif content[i] == "}":
                depth -= 1
            if depth == 0:
                return json.loads(content[start : i + 1])
    return json.loads(content.strip())


def _call_ollama(description: str, cfg: dict) -> dict | None:
    """Call local Ollama model for extraction with retry."""
    for attempt in range(2):
        try:
            client = _get_client(cfg["ollama_url"])
            resp = client.post(
                "/api/generate",
                json={
                    "model": cfg["ollama_model"],
                    # System prompt must stay byte-identical across calls so
                    # Ollama reuses the same KV-cache slot (prefix caching),
                    # turning the ~240-token instruction block into a near-zero
                    # prefill cost after the first inference on each slot.
                    "system": EXTRACTION_PROMPT,
                    "prompt": description[:1200],
                    "stream": False,
                    "keep_alive": "30m",
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 220,       # JSON with 14 fields fits in ~180 tokens
                        "num_ctx": 1536,          # system(~240) + desc(~400) + reply(~220) + margin
                        "stop": ["} {", "}\n{"],
                    },
                },
            )
            if resp.status_code != 200:
                logger.warning("Ollama API error: %s", resp.status_code)
                return None

            content = resp.json()["response"]
            return _parse_llm_json(content)

        except httpx.TimeoutException:
            logger.warning("Ollama request timed out (attempt %d)", attempt + 1)
        except (httpx.HTTPError, json.JSONDecodeError, KeyError, IndexError) as e:
            logger.debug("Ollama enrichment failed: %s", e)
            return None
    return None


def enrich_from_description(description: str, title: str = "") -> dict | None:
    """Extract structured data from title + description using local Ollama.

    Returns dict with extracted fields, or None on failure.
    """
    if not description or len(description.strip()) < 20:
        return None

    cfg = _get_config()

    if not _ollama_available(cfg["ollama_url"]):
        logger.warning("Ollama not running at %s. Skipping enrichment.", cfg["ollama_url"])
        return None

    text = f"{title}\n{description}" if title else description
    result = _call_ollama(text, cfg)
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
    cfg = _get_config()

    if not _ollama_available(cfg["ollama_url"]):
        logger.info("Ollama not running. Skipping LLM enrichment.")
        return 0

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
            failures = 0  # reset on success
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
    """Cross-check listing attributes against LLM-extracted data and return corrections.

    Returns a dict with corrected/enriched fields to apply to the listing.
    The dict includes top-level column values derived from description mentions.
    and an updated llm_extras JSON.
    """
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

    # --- Sub-model ---
    sub_model = extras.get("sub_model")
    if sub_model and isinstance(sub_model, str) and sub_model.strip():
        corrections["sub_model"] = sub_model.strip()

    # --- Trim level ---
    trim = extras.get("trim_level")
    if trim and isinstance(trim, str) and trim.strip():
        corrections["trim_level"] = trim.strip()

    # --- Number of owners ---
    num_owners = _get_extra(extras, "desc_mentions_num_owners")
    if num_owners and isinstance(num_owners, (int, float)) and num_owners > 0:
        corrections["desc_mentions_num_owners"] = int(num_owners)

    # --- Description mentions repair ---
    needs_repair = _get_extra(extras, "desc_mentions_repair")
    if needs_repair is not None:
        corrections["desc_mentions_repair"] = bool(needs_repair)

    # --- Description mentions accident ---
    had_accident = _get_extra(extras, "desc_mentions_accident")
    if had_accident is not None:
        corrections["desc_mentions_accident"] = bool(had_accident)

    # --- Description mentions customs/legalization ---
    customs = _get_extra(extras, "desc_mentions_customs_cleared")
    if customs is not None:
        corrections["desc_mentions_customs_cleared"] = bool(customs)

    # --- Right-hand drive ---
    rhd = extras.get("right_hand_drive")
    if rhd is not None:
        corrections["right_hand_drive"] = bool(rhd)

    # --- Urgency ---
    urgency = extras.get("urgency")
    if urgency in ("high", "medium", "low"):
        corrections["urgency"] = urgency

    # --- Warranty ---
    warranty = extras.get("warranty")
    if warranty is not None:
        corrections["warranty"] = bool(warranty)

    # --- Tuning or modifications ---
    tuning = extras.get("tuning_or_mods")
    if tuning and isinstance(tuning, list) and len(tuning) > 0:
        corrections["tuning_or_mods"] = json.dumps(tuning, ensure_ascii=False)

    # --- Taxi / fleet / rental ---
    taxi = extras.get("taxi_fleet_rental")
    if taxi is not None:
        corrections["taxi_fleet_rental"] = bool(taxi)

    # --- First owner selling ---
    first_owner = extras.get("first_owner_selling")
    if first_owner is not None:
        corrections["first_owner_selling"] = bool(first_owner)

    # --- Mechanical condition ---
    mech = extras.get("mechanical_condition")
    if mech in ("excellent", "good", "fair", "poor"):
        corrections["mechanical_condition"] = mech

    # --- Fix internal LLM contradictions (only tighten, never loosen) ---
    if corrections.get("desc_mentions_accident") and not corrections.get("desc_mentions_repair"):
        corrections["desc_mentions_repair"] = True

    return corrections



def apply_corrections(listings: list) -> int:
    """Apply data corrections to all listings that have LLM extras.

    Modifies listings in place. Returns count of corrected listings.
    """
    corrected = 0
    for listing in listings:
        corrections = correct_listing_data(listing)
        if not corrections:
            continue

        # Store corrections as a separate attribute for the CLI to pick up
        if not hasattr(listing, "_corrections"):
            listing._corrections = {}
        listing._corrections.update(corrections)
        corrected += 1

        # Log significant corrections
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
