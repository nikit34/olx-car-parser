"""Enrich listing data using a local Ollama model.

Extracts additional structured info from description text:
- Accident history, service history, extras/features
- Number of owners, warranty, recent maintenance
- Repair needs, real mileage, customs status, red flags
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
Extract structured data from this Portuguese car listing. JSON fields (null if unknown):
num_owners(int), had_accident(bool), accident_details(str), service_history(bool), \
needs_repair(bool), repair_details(str), estimated_repair_cost_eur(int), \
mileage_in_description_km(int), customs_cleared(bool), imported(bool), \
mechanical_condition("excellent"/"good"/"fair"/"poor"), paint_condition(same), \
suspicious_signs(list), extras(list), issues(list), reason_for_sale(str).
Rules: mileage_in_description_km=any km mentioned (e.g. "150 mil km"→150000). \
needs_repair=true if ANY repair/damage mentioned. had_accident=true if collision mentioned. \
customs_cleared=look for "desalfandegado","legalizado","por legalizar". \
estimated_repair_cost_eur=estimate from issues (e.g. embraiagem→800).

"""


def _get_config() -> dict:
    cfg = {}
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            data = yaml.safe_load(f) or {}
        cfg = data.get("llm", {})
    return {
        "ollama_model": cfg.get("ollama_model", "qwen2.5:1.5b"),
        "ollama_url": cfg.get("ollama_url", OLLAMA_URL),
    }


_ollama_status: bool | None = None

# Persistent HTTP client — reuses TCP connection across all LLM calls
_http_client: httpx.Client | None = None


def _get_client(base_url: str) -> httpx.Client:
    """Return a persistent httpx.Client, creating one if needed."""
    global _http_client
    if _http_client is None:
        _http_client = httpx.Client(base_url=base_url, timeout=60)
    return _http_client


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
    """Extract JSON from LLM response, handling markdown code blocks."""
    if "```" in content:
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    return json.loads(content.strip())


def _call_ollama(description: str, cfg: dict) -> dict | None:
    """Call local Ollama model for extraction."""
    try:
        client = _get_client(cfg["ollama_url"])
        resp = client.post(
            "/api/generate",
            json={
                "model": cfg["ollama_model"],
                "prompt": EXTRACTION_PROMPT + description[:1200],
                "format": "json",
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "num_predict": 256,
                    "num_ctx": 1024,
                },
            },
        )
        if resp.status_code != 200:
            logger.warning("Ollama API error: %s", resp.status_code)
            return None

        content = resp.json()["response"]
        return _parse_llm_json(content)

    except httpx.TimeoutException:
        logger.warning("Ollama request timed out")
        return None
    except (httpx.HTTPError, json.JSONDecodeError, KeyError, IndexError) as e:
        logger.debug("Ollama enrichment failed: %s", e)
        return None


def enrich_from_description(description: str) -> dict | None:
    """Extract structured data from description using local Ollama.

    Returns dict with extracted fields, or None on failure.
    """
    if not description or len(description.strip()) < 20:
        return None

    cfg = _get_config()

    if not _ollama_available(cfg["ollama_url"]):
        logger.warning("Ollama not running at %s. Skipping enrichment.", cfg["ollama_url"])
        return None

    return _call_ollama(description, cfg)


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

        result = enrich_from_description(listing.description)
        if result:
            listing._llm_extras = result
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
    The dict includes both top-level column values (needs_repair, had_accident, etc.)
    and an updated llm_extras JSON.
    """
    extras = getattr(listing, "_llm_extras", None)
    if not extras:
        return {}

    corrections = {}

    # --- Mileage cross-check ---
    desc_km = extras.get("mileage_in_description_km")
    attr_km = listing.mileage_km

    if desc_km and isinstance(desc_km, (int, float)) and desc_km > 0:
        desc_km = int(desc_km)
        if attr_km and attr_km > 0:
            # Description mentions significantly higher mileage → attribute is suspect
            if desc_km > attr_km * 1.3 and (desc_km - attr_km) > 5000:
                corrections["real_mileage_km"] = desc_km
                corrections["mileage_suspect"] = True
                logger.info(
                    "Mileage mismatch for %s: attribute=%d, description=%d → using description",
                    listing.url, attr_km, desc_km,
                )
            # Description mentions significantly lower → could be seller's round number
            # but if attribute is suspiciously high (typo?), description might be correct
            elif attr_km > desc_km * 2 and (attr_km - desc_km) > 50000:
                corrections["real_mileage_km"] = desc_km
                corrections["mileage_suspect"] = True
                logger.info(
                    "Possible mileage typo for %s: attribute=%d, description=%d",
                    listing.url, attr_km, desc_km,
                )
        elif not attr_km or attr_km == 0:
            # No attribute mileage but description has it
            corrections["real_mileage_km"] = desc_km
            corrections["mileage_suspect"] = False

    # --- Needs repair ---
    needs_repair = extras.get("needs_repair")
    if needs_repair is not None:
        corrections["needs_repair"] = bool(needs_repair)
    elif extras.get("issues"):
        # If issues list is non-empty, car likely needs some work
        corrections["needs_repair"] = True

    # --- Accident history ---
    had_accident = extras.get("had_accident")
    if had_accident is not None:
        corrections["had_accident"] = bool(had_accident)
    elif extras.get("accident_free") is not None:
        corrections["had_accident"] = not extras["accident_free"]

    # --- Customs/legalization ---
    customs = extras.get("customs_cleared")
    if customs is not None:
        corrections["customs_cleared"] = bool(customs)
    elif listing.origin and "import" in (listing.origin or "").lower():
        # Imported car with no customs info → flag as unknown (None stays)
        pass

    # --- Estimated repair cost ---
    repair_cost = extras.get("estimated_repair_cost_eur")
    if repair_cost and isinstance(repair_cost, (int, float)) and repair_cost > 0:
        corrections["estimated_repair_cost_eur"] = int(repair_cost)

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
        if corrections.get("mileage_suspect"):
            logger.info(
                "Corrected %s: real_mileage=%s, needs_repair=%s, accident=%s",
                listing.olx_id,
                corrections.get("real_mileage_km"),
                corrections.get("needs_repair"),
                corrections.get("had_accident"),
            )

    logger.info("Applied corrections to %d / %d listings", corrected, len(listings))
    return corrected
