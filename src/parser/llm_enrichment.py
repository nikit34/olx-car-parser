"""Enrich listing data using a free LLM via OpenRouter.

Extracts additional structured info from description text:
- Accident history, service history, extras/features
- Number of owners, warranty, recent maintenance
"""

import json
import logging
import os
from pathlib import Path

import httpx
import yaml

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "settings.yaml"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

EXTRACTION_PROMPT = """\
You are a data extraction assistant. Given a car listing description in Portuguese, \
extract structured information. Return ONLY valid JSON with these fields (use null if not found):

{
  "num_owners": <int or null>,
  "accident_free": <bool or null>,
  "service_history": <bool or null>,
  "warranty_months": <int or null>,
  "recent_maintenance": <string or null>,
  "extras": [<list of notable extras/features as short strings>],
  "issues": [<list of mentioned problems/defects>],
  "reason_for_sale": <string or null>
}

Description:
"""


def _get_config() -> dict:
    cfg = {}
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            data = yaml.safe_load(f) or {}
        cfg = data.get("llm", {})
    return {
        "api_key": os.environ.get("OPENROUTER_API_KEY", cfg.get("openrouter_api_key", "")),
        "model": cfg.get("model", "nvidia/nemotron-3-super-120b-a12b:free"),
    }


def enrich_from_description(description: str) -> dict | None:
    """Call OpenRouter free model to extract structured data from description.

    Returns dict with extracted fields, or None on failure.
    """
    if not description or len(description.strip()) < 20:
        return None

    cfg = _get_config()
    if not cfg["api_key"]:
        return None

    try:
        resp = httpx.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {cfg['api_key']}",
                "Content-Type": "application/json",
            },
            json={
                "model": cfg["model"],
                "messages": [
                    {"role": "user", "content": EXTRACTION_PROMPT + description[:2000]},
                ],
                "max_tokens": 500,
                "temperature": 0.1,
            },
            timeout=30,
        )
        if resp.status_code != 200:
            logger.warning("OpenRouter API error: %s", resp.status_code)
            return None

        content = resp.json()["choices"][0]["message"]["content"]

        # Extract JSON from response (handle markdown code blocks)
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]

        return json.loads(content.strip())

    except (httpx.HTTPError, json.JSONDecodeError, KeyError, IndexError) as e:
        logger.debug("LLM enrichment failed: %s", e)
        return None


def enrich_listings_batch(listings: list, batch_size: int = 50) -> int:
    """Enrich a batch of RawListing objects with LLM-extracted data.

    Modifies listings in place. Returns count of enriched listings.
    """
    cfg = _get_config()
    if not cfg["api_key"]:
        logger.info("No OpenRouter API key configured. Skipping LLM enrichment.")
        return 0

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
            if failures >= 3:
                logger.warning("3 consecutive LLM failures, stopping enrichment.")
                break

    logger.info("LLM-enriched %d / %d listings", enriched, len(listings))
    return enriched
