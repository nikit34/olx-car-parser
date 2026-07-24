"""Value-gated LLM enrichment via OpenRouter (free-tier cloud models).

This is the cloud counterpart to :mod:`src.parser.llm_enrichment` (local
Ollama). The Ollama inference pool was retired 2026-07-24; scheduled scrape
runs now go raw-only. To recover *some* LLM signal without a self-hosted GPU,
we send only the small set of GENUINELY INTERESTING listings — the top deals
the GBM value model already flags as undervalued (see
:mod:`src.analytics.value_gate`) — to a free OpenRouter model.

Two reasons the call is gated hard:
  1. Free-tier OpenRouter caps requests at ~50/day and free models are
     frequently upstream-rate-limited (HTTP 429). A per-run + daily request
     budget (state file) keeps us under the ceiling.
  2. The value-add is on deals: reading a Portuguese description for
     condition / accident / owner / negotiability closes the "condition-blind"
     gap in the price model exactly where it matters (the surfaced deals).

Unlike the slim Ollama schema (sub_model / trim_level / mileage only), this
path also extracts the condition fields the decision engine already consumes:
mechanical_condition, desc_mentions_accident/repair/num_owners/customs_cleared,
right_hand_drive, warranty, urgency — plus display-only negotiable / red_flags
/ deal_note_pt kept in ``llm_extras``. The extracted keys are named to match
the DB columns so the write path is a plain ``setattr``.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import httpx
import yaml

# Reuse the validated primitives from the Ollama path — one source of truth
# for sub_model brand-family validation, mileage sanity bounds, and the
# deterministic damage_severity derivation.
from src.parser.llm_enrichment import (
    _validate_sub_model,
    _derive_damage_severity,
    _MILEAGE_SANITY_MAX_KM,
    _MILEAGE_SANITY_RELATIVE_MAX,
    correct_listing_data,
)

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "settings.yaml"

# Env var holding the OpenRouter API key (never stored in settings.yaml —
# same convention as TELEGRAM_BOT_TOKEN / CF_DEPLOY_HOOK_URL).
API_KEY_ENV = "OPENROUTER_API_KEY"


# ---------------------------------------------------------------------------
# Extraction prompt — base 3 fields + condition NLP
# ---------------------------------------------------------------------------
# The JSON keys are the DB column names on purpose (desc_mentions_*, urgency,
# warranty, mechanical_condition) so persistence is a straight setattr. Enums
# are English lowercase to match the exact vocab the consumers compare against
# (data_loader.compute_signals / _blocking_deal_reason, analytics.decision):
#   urgency ∈ {high, medium, low}; mechanical_condition ∈ {excellent, good,
#   fair, poor}. desc_mentions_accident / right_hand_drive / damage_severity≥3
#   are HARD funnel vetoes — the prompt demands they be true ONLY when the text
#   states it explicitly (high precision, never guessed).
_SYSTEM_PROMPT = """\
You extract structured facts from a Portuguese (pt-PT) used-car listing and return ONE JSON object. Use null when a field cannot be determined from the text. Output ONLY the JSON object, no markdown, no commentary.

Return exactly these keys:
sub_model: engine/body variant only (displacement+fuel+power), e.g. "320d","1.6 TDI","2.0 TFSI","A 200". NOT a trim/package, NOT a bare model name. Tech tags belong to specific brand families — never assign a tag from the wrong family: TDI/TFSI/TSI = VAG only (VW/Audi/Seat/Skoda); HDi/BlueHDi/PureTech = PSA only (Peugeot/Citroën/DS); CDI/BlueTec = Mercedes only; dCi/TCe = Renault/Dacia/Nissan only; M-Jet/Multijet = FCA only (Fiat/Alfa Romeo); TDCi/EcoBoost = Ford only. BMW uses model-code form (116d, 320d, 535d) — never "1.6 TDI", never just "Touring"/"xDrive". If unsure, null.
trim_level: equipment line e.g. "AMG Line","M Sport","S-Line","GTI","FR","Tekna". null if basic.
mileage_in_description_km: integer km. "mil"=thousand only as a separate word ("150 mil km"→150000; "89.500km"→89500). Service-interval km ("revisão aos 60.000 km") is NOT current mileage.
mechanical_condition: one of "excellent","good","fair","poor" — overall mechanical state as described. null if not indicated.
desc_mentions_accident: true ONLY if the text explicitly states the car had an accident / crash / "sinistrado" / "batido" / "acidente". Otherwise false. (This removes the car from deals — be strict.)
desc_mentions_repair: true if the text says it needs repair / has a known mechanical fault / "para arranjar" / "avaria". Otherwise false.
desc_mentions_num_owners: integer number of owners if stated ("2 donos"→2, "único dono"/"1 dono"→1). null if not stated.
desc_mentions_customs_cleared: true if it explicitly mentions being imported AND customs-cleared / "legalizado" / "nacionalizado" / "ISV pago". Otherwise false.
right_hand_drive: true ONLY if explicitly right-hand drive / "volante à direita" / UK import RHD. Otherwise false. (This removes the car from deals — be strict.)
warranty: true if the seller offers a warranty / "garantia" (not "sem garantia"). Otherwise false.
urgency: "high" if the seller signals urgency / quick sale / "venda urgente" / "preço negociável para venda rápida"; "medium" if mildly motivated; "low" otherwise.
negotiable: true if price is negotiable / "negociável" / "aceito retoma". Otherwise false.
red_flags: array of short pt-PT strings naming concrete concerns (high km, damage, many owners, import doubts). Empty array if none.
deal_note_pt: one short pt-PT phrase (≤120 chars) summarising the listing as a deal. null if nothing notable.

Example:
"BMW Série 3 320d Pack M, 180.000 km, 1 dono, garantia até 2026, nunca sinistrado, negociável."
→ {"sub_model":"320d","trim_level":"Pack M","mileage_in_description_km":180000,"mechanical_condition":"good","desc_mentions_accident":false,"desc_mentions_repair":false,"desc_mentions_num_owners":1,"desc_mentions_customs_cleared":false,"right_hand_drive":false,"warranty":true,"urgency":"low","negotiable":true,"red_flags":["quilometragem elevada"],"deal_note_pt":"320d Pack M com 1 dono e garantia, preço negociável"}
"""


# Which extracted keys map to real DB columns (written via setattr), with the
# type each column expects. Keys NOT here (negotiable, red_flags, deal_note_pt)
# live only in llm_extras JSON. sub_model / trim_level / real_mileage_km /
# damage_severity come from correct_listing_data (reused), so they are excluded
# here to avoid double-writing.
_BOOL_COLUMNS = (
    "desc_mentions_accident",
    "desc_mentions_repair",
    "desc_mentions_customs_cleared",
    "right_hand_drive",
    "warranty",
)
_STR_ENUM_COLUMNS = {
    "mechanical_condition": {"excellent", "good", "fair", "poor"},
    "urgency": {"high", "medium", "low"},
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def get_openrouter_config() -> dict:
    """Load the ``openrouter`` section of settings.yaml with safe defaults."""
    cfg = {}
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            data = yaml.safe_load(f) or {}
        cfg = data.get("openrouter", {}) or {}
    gate = cfg.get("gate", {}) or {}
    return {
        "base_url": cfg.get("base_url", "https://openrouter.ai/api/v1"),
        "models": cfg.get("models") or [
            "google/gemma-4-26b-a4b-it:free",
            "google/gemma-4-31b-it:free",
            "openai/gpt-oss-20b:free",
        ],
        "temperature": cfg.get("temperature", 0.1),
        "max_tokens": cfg.get("max_tokens", 500),
        "max_chars": cfg.get("max_chars", 2500),
        "timeout_seconds": cfg.get("timeout_seconds", 90),
        "max_retries": cfg.get("max_retries", 1),
        "referer": cfg.get("referer", "https://olx-car-parser.permikov134.workers.dev"),
        "title": cfg.get("title", "olx-car-parser"),
        "daily_request_budget": cfg.get("daily_request_budget", 45),
        "per_run_request_cap": cfg.get("per_run_request_cap", 15),
        "budget_state_file": cfg.get("budget_state_file", "data/openrouter_budget.json"),
        "gate": {
            "min_price_eur": gate.get("min_price_eur", 4000),
            "min_spec_fill": gate.get("min_spec_fill", 0.5),
            "max_band_pct": gate.get("max_band_pct", 0.40),
            "min_discount_pct": gate.get("min_discount_pct", 0.0),
            "max_discount_pct": gate.get("max_discount_pct", 60.0),
        },
    }


def get_api_key() -> str | None:
    key = os.environ.get(API_KEY_ENV, "").strip()
    return key or None


def openrouter_available() -> bool:
    return get_api_key() is not None


# ---------------------------------------------------------------------------
# Daily / per-run request budget (free-tier ~50/day ceiling)
# ---------------------------------------------------------------------------

def _today() -> str:
    return time.strftime("%Y-%m-%d", time.gmtime())


def load_budget(state_path: str | Path) -> dict:
    """Return {'date': 'YYYY-MM-DD', 'requests': int} for *today*.

    Resets to zero when the stored date is not today (UTC), so the free-tier
    daily allowance rolls over automatically.
    """
    path = Path(state_path)
    today = _today()
    if path.exists():
        try:
            data = json.loads(path.read_text())
            if data.get("date") == today:
                return {"date": today, "requests": int(data.get("requests", 0))}
        except (json.JSONDecodeError, ValueError, OSError):
            pass
    return {"date": today, "requests": 0}


def save_budget(state_path: str | Path, requests_spent: int) -> None:
    path = Path(state_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"date": _today(), "requests": int(requests_spent)}))


def remaining_daily(cfg: dict) -> int:
    b = load_budget(cfg["budget_state_file"])
    return max(0, int(cfg["daily_request_budget"]) - b["requests"])


# ---------------------------------------------------------------------------
# Core API call
# ---------------------------------------------------------------------------

def _parse_json_object(content: str) -> dict | None:
    """Parse a chat-completion string into a JSON object, tolerating ```fences```."""
    if not content:
        return None
    t = content.strip()
    if t.startswith("```"):
        t = t.lstrip("`")
        if t[:4].lower() == "json":
            t = t[4:]
        t = t.strip().rstrip("`").strip()
    try:
        parsed = json.loads(t)
    except json.JSONDecodeError:
        # Last resort: slice the outermost {...}.
        start, end = t.find("{"), t.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(t[start:end + 1])
            except json.JSONDecodeError:
                return None
        else:
            return None
    return parsed if isinstance(parsed, dict) else None


def call_openrouter(text: str, cfg: dict, *, request_cap: int | None = None) -> tuple[dict | None, int]:
    """Run one extraction against the OpenRouter free-model fallback chain.

    Tries each model in ``cfg['models']`` in order; on HTTP 429 (free model
    upstream-rate-limited) or 5xx it advances to the next model, retrying each
    up to ``cfg['max_retries']`` times. Returns ``(extras_dict_or_None,
    n_requests_made)`` — the caller uses ``n_requests_made`` for budget
    accounting (each HTTP POST counts against the free-tier daily allowance).

    ``request_cap`` hard-limits how many HTTP requests this single call may
    make (so one wedged listing can't drain the whole per-run budget).
    """
    key = get_api_key()
    if not key:
        return None, 0
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": cfg["referer"],
        "X-Title": cfg["title"],
    }
    truncated = text[: cfg.get("max_chars", 2500)]
    body_base = {
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": truncated},
        ],
        "temperature": cfg.get("temperature", 0.1),
        "max_tokens": cfg.get("max_tokens", 500),
        "response_format": {"type": "json_object"},
    }
    url = cfg["base_url"].rstrip("/") + "/chat/completions"
    n_requests = 0
    with httpx.Client(timeout=httpx.Timeout(float(cfg.get("timeout_seconds", 90)), connect=10.0)) as client:
        for model in cfg["models"]:
            for attempt in range(int(cfg.get("max_retries", 1)) + 1):
                if request_cap is not None and n_requests >= request_cap:
                    return None, n_requests
                body = dict(body_base, model=model)
                # gemma free rejects response_format upstream on some providers;
                # drop it for gemma and rely on the prompt's "ONLY JSON".
                if "gemma" in model:
                    body.pop("response_format", None)
                try:
                    resp = client.post(url, json=body, headers=headers)
                    n_requests += 1
                except httpx.RequestError as e:
                    logger.warning("OpenRouter connection error (%s): %s", model, e)
                    n_requests += 1
                    break  # try next model
                if resp.status_code == 200:
                    try:
                        j = resp.json()
                        content = j["choices"][0]["message"]["content"]
                    except (KeyError, IndexError, json.JSONDecodeError, ValueError):
                        logger.warning("OpenRouter malformed 200 body (%s)", model)
                        break
                    parsed = _parse_json_object(content)
                    if parsed is not None:
                        return parsed, n_requests
                    logger.info("OpenRouter %s returned non-JSON; trying next model", model)
                    break  # bad output → next model, not a retry
                if resp.status_code == 429 or resp.status_code >= 500:
                    logger.info("OpenRouter %s HTTP %s (attempt %d)", model, resp.status_code, attempt + 1)
                    continue  # retry same model, then fall through to next
                # 4xx other than 429 (401/402/400) → hard stop, no point retrying
                logger.warning("OpenRouter %s HTTP %s — aborting: %s",
                               model, resp.status_code, resp.text[:200])
                return None, n_requests
    return None, n_requests


# ---------------------------------------------------------------------------
# Corrections — base fields (reused) + condition columns
# ---------------------------------------------------------------------------

def openrouter_corrections(listing) -> dict:
    """Corrections dict for a listing whose ``_llm_extras`` came from OpenRouter.

    Combines the reused base corrections (sub_model / trim_level /
    real_mileage_km / damage_severity, via :func:`correct_listing_data`) with
    the condition columns unique to this richer schema. All keys are real
    Listing columns, so the caller writes them with ``setattr`` (same seam as
    the ``enrich`` command).
    """
    extras = getattr(listing, "_llm_extras", None)
    if not isinstance(extras, dict):
        return {}

    corrections = correct_listing_data(listing)  # sub_model/trim/mileage/damage_severity

    for col in _BOOL_COLUMNS:
        if col in extras and isinstance(extras[col], bool):
            corrections[col] = extras[col]

    n_owners = extras.get("desc_mentions_num_owners")
    if isinstance(n_owners, (int, float)) and not isinstance(n_owners, bool) and n_owners > 0:
        corrections["desc_mentions_num_owners"] = int(n_owners)

    for col, allowed in _STR_ENUM_COLUMNS.items():
        val = extras.get(col)
        if isinstance(val, str) and val.strip().lower() in allowed:
            corrections[col] = val.strip().lower()

    return corrections


def enrich_from_description(description: str, title: str, cfg: dict,
                           *, request_cap: int | None = None) -> tuple[dict | None, int]:
    """Extract structured data from title+description via OpenRouter.

    Returns ``(extras_or_None, n_requests_made)``. Mirrors the local-path
    ``enrich_from_description`` contract but with request accounting.
    """
    if not description or len(description.strip()) < 20:
        return None, 0
    text = f"{title}\n{description}" if title else description
    return call_openrouter(text, cfg, request_cap=request_cap)
