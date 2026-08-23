"""Value-gated cloud LLM enrichment — Gemini first, OpenRouter as backstop.

There is no local inference in this project any more. The Ollama pool (the
runner's own :11434 plus the Windows partner) was retired 2026-07-24 and
removed from both machines 2026-08-23; nothing here should ever dial
localhost for a model.

What replaced it is deliberately small. The GBM price model needs NO LLM to
value a car — mileage, power, displacement, fuel, brand, model and year are
structured scrape fields — so every fresh listing is priced first, ranked by
undervaluation, and only the top-K of that ranking (see
:mod:`src.analytics.value_gate`) is ever shown to a language model. The LLM's
whole job is reading the Portuguese free-text description for the things the
structured fields can't carry: condition, accident history, owners, warranty,
urgency. That closes the "condition-blind" gap in the price model exactly
where it changes a decision, and nowhere else.

Providers are a CASCADE, not a primary-plus-backup. Each one gets the same
prompt and must return the same JSON object; the first usable answer wins, and
any refusal — no API key, no rate-limit slot, spent daily budget, HTTP 429/5xx,
output that isn't parseable JSON — advances to the next provider and is logged
at WARNING so one grep explains a quiet run. Gemini leads because its free
tier is an order of magnitude larger than OpenRouter's ~50 requests/day;
OpenRouter catches the Gemini 503 bursts, which are real and come in threes.

Spend is metered per provider per UTC day in one state file (:class:`BudgetLedger`),
because the two free tiers have completely different ceilings and a shared
counter would either waste Gemini's headroom or overrun OpenRouter's.

The extracted schema matches the DB columns on purpose — mechanical_condition,
desc_mentions_accident/repair/num_owners/customs_cleared, right_hand_drive,
warranty, urgency — so persistence is a plain ``setattr``; display-only
negotiable / red_flags / deal_note_pt stay in ``llm_extras``.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from pathlib import Path

import httpx
import yaml

# One source of truth for turning a raw extraction into DB columns: sub_model
# brand-family validation, mileage sanity bounds and the deterministic
# damage_severity derivation all live in the domain module.
from src.parser.llm_enrichment import correct_listing_data

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "settings.yaml"

# Env vars holding the API keys (never stored in settings.yaml — same
# convention as TELEGRAM_BOT_TOKEN / CF_DEPLOY_HOOK_URL).
KEY_ENV = {
    "gemini": "GEMINI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


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

_GEMINI_DEFAULTS = {
    "base_url": "https://generativelanguage.googleapis.com/v1beta",
    "models": ["gemini-flash-latest", "gemini-flash-lite-latest"],
    "timeout_seconds": 60,
    "max_attempts": 2,
    "retry_backoff_seconds": 3.0,
    "max_rpm": 4,
    "slot_wait_seconds": 2.0,
    "daily_request_budget": 150,
}

_OPENROUTER_DEFAULTS = {
    "base_url": "https://openrouter.ai/api/v1",
    "models": [
        "google/gemma-4-26b-a4b-it:free",
        "google/gemma-4-31b-it:free",
        "openai/gpt-oss-20b:free",
    ],
    "timeout_seconds": 90,
    "max_attempts": 1,
    "retry_backoff_seconds": 1.0,
    "referer": "https://olx-car-parser.permikov134.workers.dev",
    "title": "olx-car-parser",
    "daily_request_budget": 40,
}


def get_llm_config() -> dict:
    """Load the ``llm`` section plus each provider's section, with defaults.

    Returns a flat dict: the shared knobs at the top level, provider settings
    under ``cfg['providers_cfg'][name]``, and the cascade order under
    ``cfg['providers']``. Unknown provider names in the configured order are
    dropped with a warning rather than crashing a scheduled run.
    """
    data = {}
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            data = yaml.safe_load(f) or {}
    llm = data.get("llm", {}) or {}
    gate = llm.get("gate", {}) or {}

    providers_cfg = {
        "gemini": {**_GEMINI_DEFAULTS, **(data.get("gemini", {}) or {})},
        "openrouter": {**_OPENROUTER_DEFAULTS, **(data.get("openrouter", {}) or {})},
    }

    order = []
    for name in (llm.get("providers") or ["gemini", "openrouter"]):
        name = str(name).strip().lower()
        if name in providers_cfg:
            order.append(name)
        else:
            logger.warning("Unknown LLM provider %r in llm.providers — ignoring", name)
    if not order:
        logger.warning("llm.providers resolved to nothing — falling back to gemini")
        order = ["gemini"]

    return {
        "providers": order,
        "providers_cfg": providers_cfg,
        "temperature": llm.get("temperature", 0.1),
        "max_tokens": llm.get("max_tokens", 500),
        "max_chars": llm.get("max_chars", 2500),
        "per_run_request_cap": llm.get("per_run_request_cap", 20),
        "budget_state_file": llm.get("budget_state_file", "data/llm_budget.json"),
        "gate": {
            "min_price_eur": gate.get("min_price_eur", 4000),
            "min_spec_fill": gate.get("min_spec_fill", 0.5),
            "max_band_pct": gate.get("max_band_pct", 0.40),
            "min_discount_pct": gate.get("min_discount_pct", 0.0),
            "max_discount_pct": gate.get("max_discount_pct", 60.0),
        },
    }


def get_api_key(provider: str) -> str | None:
    key = os.environ.get(KEY_ENV.get(provider, ""), "").strip()
    return key or None


def available_providers(cfg: dict) -> list[str]:
    """Configured providers that actually have a key in the environment."""
    return [p for p in cfg["providers"] if get_api_key(p)]


def llm_available(cfg: dict | None = None) -> bool:
    return bool(available_providers(cfg or get_llm_config()))


# ---------------------------------------------------------------------------
# Per-provider daily budget (the two free tiers have different ceilings)
# ---------------------------------------------------------------------------

def _today() -> str:
    return time.strftime("%Y-%m-%d", time.gmtime())


class BudgetLedger:
    """Requests spent per provider today, persisted to one JSON state file.

    Written after EVERY call, not at the end of the run: the enrichment step
    has a hard timeout and the workflow's concurrency rule can cancel it, so a
    single end-of-loop save would silently drop the spend and let the next run
    of the day re-spend past a free-tier ceiling.

    A stored date that isn't today means the allowance rolled over, so the
    counters start at zero — that is the whole reset mechanism.
    """

    def __init__(self, state_path: str | Path):
        self.path = Path(state_path)
        self.date = _today()
        self.spent: dict[str, int] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text())
        except (json.JSONDecodeError, ValueError, OSError):
            return
        if data.get("date") != self.date:
            return
        raw = data.get("providers")
        if isinstance(raw, dict):
            for name, n in raw.items():
                try:
                    self.spent[str(name)] = int(n)
                except (TypeError, ValueError):
                    continue

    def used(self, provider: str) -> int:
        return int(self.spent.get(provider, 0))

    def remaining(self, provider: str, cfg: dict) -> int:
        ceiling = int(cfg["providers_cfg"][provider].get("daily_request_budget", 0))
        return max(0, ceiling - self.used(provider))

    def charge(self, provider: str, n_requests: int) -> None:
        if n_requests <= 0:
            return
        self.spent[provider] = self.used(provider) + int(n_requests)
        self.save()

    def total(self) -> int:
        return sum(self.spent.values())

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"date": self.date, "providers": self.spent}))


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _parse_json_object(content) -> dict | None:
    """Parse a chat-completion string into a JSON object, tolerating ```fences```.

    Defensive against a provider returning a non-string ``content`` (null, or a
    structured content array) — anything that isn't a non-empty str yields None
    rather than an AttributeError that would abort the whole enrichment run.
    """
    if not isinstance(content, str) or not content.strip():
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


# ---------------------------------------------------------------------------
# Gemini — client-side pacing
# ---------------------------------------------------------------------------
# The free tier is a few requests per minute and answers an over-pace call with
# 429 plus a ~48 s retryDelay, so an immediate retry is a guaranteed second
# 429. Reserving a slot locally turns "over quota" into a 2-second discovery
# and hands the listing to the next provider instead of burning quota units.
_GEMINI_CALLS: deque[float] = deque()
_GEMINI_LOCK = threading.Lock()


def _reserve_gemini_slot(pcfg: dict) -> bool:
    """Take a slot in the rolling minute window. False if none frees up in time."""
    try:
        limit = max(1, int(pcfg.get("max_rpm", 4)))
    except (TypeError, ValueError):
        limit = 4
    try:
        max_wait = max(0.0, float(pcfg.get("slot_wait_seconds", 2)))
    except (TypeError, ValueError):
        max_wait = 2.0
    deadline = time.monotonic() + max_wait
    while True:
        with _GEMINI_LOCK:
            now = time.monotonic()
            while _GEMINI_CALLS and now - _GEMINI_CALLS[0] >= 60.0:
                _GEMINI_CALLS.popleft()
            if len(_GEMINI_CALLS) < limit:
                _GEMINI_CALLS.append(now)
                return True
            free_in = 60.0 - (now - _GEMINI_CALLS[0]) + 0.05
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            logger.warning("Gemini: no slot at %d req/min — moving to the next provider", limit)
            return False
        time.sleep(min(free_in, remaining, 0.5))


def call_gemini(text: str, cfg: dict, *, request_cap: int | None = None) -> tuple[dict | None, int]:
    """One extraction against Gemini. Returns ``(parsed_or_None, n_requests)``.

    Models in ``gemini.models`` are tried in order; each gets up to
    ``max_attempts`` tries. A 503 ("model overloaded") is worth retrying — the
    free tier serves them in bursts and the next attempt usually lands. A 429
    is not: the quota window is a minute wide, so we abandon the provider and
    let the cascade move on.
    """
    key = get_api_key("gemini")
    if not key:
        logger.warning("GEMINI_API_KEY not set — skipping Gemini")
        return None, 0
    pcfg = cfg["providers_cfg"]["gemini"]
    truncated = text[: cfg.get("max_chars", 2500)]
    payload_base = {
        "systemInstruction": {"parts": [{"text": _SYSTEM_PROMPT}]},
        "contents": [{"parts": [{"text": truncated}]}],
        "generationConfig": {
            "temperature": cfg.get("temperature", 0.1),
            "maxOutputTokens": cfg.get("max_tokens", 500),
            # Gemini 2.5 "thinks" out of the same token budget, which truncates
            # the JSON (finishReason=MAX_TOKENS). We want extraction, not
            # reasoning — spend the whole budget on the object.
            "thinkingConfig": {"thinkingBudget": 0},
            # Native JSON mode: the server constrains decoding, so there are no
            # ```fences``` or prose to strip. _parse_json_object still runs as
            # a belt-and-braces step.
            "responseMimeType": "application/json",
        },
    }
    headers = {"Content-Type": "application/json", "X-goog-api-key": key}
    base = str(pcfg["base_url"]).rstrip("/")
    attempts = max(1, int(pcfg.get("max_attempts", 2)))
    backoff = float(pcfg.get("retry_backoff_seconds", 3) or 0)
    n_requests = 0

    with httpx.Client(timeout=httpx.Timeout(float(pcfg.get("timeout_seconds", 60)), connect=10.0)) as client:
        for model in pcfg["models"]:
            url = f"{base}/models/{model}:generateContent"
            for attempt in range(1, attempts + 1):
                if request_cap is not None and n_requests >= request_cap:
                    return None, n_requests
                if not _reserve_gemini_slot(pcfg):
                    return None, n_requests
                try:
                    resp = client.post(url, json=payload_base, headers=headers)
                    n_requests += 1
                except httpx.RequestError as e:
                    logger.warning("Gemini connection error (%s): %s", model, e)
                    n_requests += 1
                    break  # next model
                if resp.status_code == 200:
                    parsed = _parse_json_object(_gemini_text(resp))
                    if parsed is not None:
                        return parsed, n_requests
                    logger.info("Gemini %s returned unusable text; trying next model", model)
                    break
                if resp.status_code == 429:
                    logger.warning("Gemini %s rate-limited (429) — not retrying within provider",
                                   model)
                    return None, n_requests
                if resp.status_code >= 500:
                    logger.info("Gemini %s HTTP %s (attempt %d/%d)",
                                model, resp.status_code, attempt, attempts)
                    if attempt < attempts and backoff:
                        time.sleep(backoff)
                    continue
                # An invalid Gemini key answers 400 INVALID_ARGUMENT, not 401 —
                # without this it looks like a per-model problem and we burn a
                # request on every model in the chain before giving up.
                if resp.status_code in (401, 403) or (
                    resp.status_code == 400 and "API key not valid" in resp.text
                ):
                    logger.warning("Gemini auth error HTTP %s — abandoning provider: %s",
                                   resp.status_code, resp.text[:200])
                    return None, n_requests
                logger.warning("Gemini %s HTTP %s — trying next model: %s",
                               model, resp.status_code, resp.text[:200])
                break
    return None, n_requests


def _gemini_text(resp) -> str | None:
    """Pull the answer text out of a generateContent 200 body."""
    try:
        candidate = (resp.json().get("candidates") or [{}])[0]
    except (json.JSONDecodeError, ValueError, AttributeError):
        return None
    parts = ((candidate.get("content") or {}).get("parts")) or []
    text = "".join(p.get("text", "") for p in parts if isinstance(p, dict)).strip()
    if not text:
        logger.info("Gemini returned no text (finishReason=%s)", candidate.get("finishReason"))
        return None
    return text


def call_openrouter(text: str, cfg: dict, *, request_cap: int | None = None) -> tuple[dict | None, int]:
    """One extraction against the OpenRouter free-model chain.

    Tries each model in ``openrouter.models`` in order; on HTTP 429 (free model
    upstream-rate-limited) or 5xx it advances, retrying each up to
    ``max_attempts`` times. Returns ``(parsed_or_None, n_requests)`` — every
    POST counts against the free-tier daily allowance, including the ones that
    came back 429.
    """
    key = get_api_key("openrouter")
    if not key:
        logger.warning("OPENROUTER_API_KEY not set — skipping OpenRouter")
        return None, 0
    pcfg = cfg["providers_cfg"]["openrouter"]
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": pcfg["referer"],
        "X-Title": pcfg["title"],
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
        # Without this, a reasoning-capable free model spends the whole token
        # budget thinking and returns an empty content field — the extraction
        # silently becomes None and we pay for the request anyway.
        "reasoning": {"enabled": False},
    }
    url = str(pcfg["base_url"]).rstrip("/") + "/chat/completions"
    attempts = max(1, int(pcfg.get("max_attempts", 1)))
    backoff = float(pcfg.get("retry_backoff_seconds", 0) or 0)
    n_requests = 0

    with httpx.Client(timeout=httpx.Timeout(float(pcfg.get("timeout_seconds", 90)), connect=10.0)) as client:
        for model in pcfg["models"]:
            for attempt in range(1, attempts + 1):
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
                    logger.info("OpenRouter %s HTTP %s (attempt %d/%d)",
                                model, resp.status_code, attempt, attempts)
                    if backoff and (attempt < attempts or model != pcfg["models"][-1]):
                        time.sleep(backoff)
                    continue  # retry same model (if attempts left), then next model
                if resp.status_code in (401, 402, 403):
                    # Account-level failure (bad key / no credits / forbidden) —
                    # no other model in the chain will help.
                    logger.warning("OpenRouter account error HTTP %s — abandoning provider: %s",
                                   resp.status_code, resp.text[:200])
                    return None, n_requests
                # Any other 4xx (400 bad request, 404 model delisted, …) is
                # model-specific — advance to the next model rather than abort.
                logger.warning("OpenRouter %s HTTP %s — trying next model: %s",
                               model, resp.status_code, resp.text[:200])
                break
    return None, n_requests


def _provider_call(provider: str):
    """Resolve a provider name to its call function AT CALL TIME.

    Deliberately not a module-level dict of function objects: that would
    capture the originals at import, so patching ``call_gemini`` in a test
    would leave the cascade dispatching to the real one — which is how a unit
    test ends up making live API calls.
    """
    return {"gemini": call_gemini, "openrouter": call_openrouter}[provider]


# ---------------------------------------------------------------------------
# The cascade
# ---------------------------------------------------------------------------

def call_llm(
    text: str,
    cfg: dict,
    *,
    ledger: BudgetLedger,
    request_cap: int | None = None,
) -> tuple[dict | None, int]:
    """Run one extraction through the provider cascade.

    Walks ``cfg['providers']`` in order and returns the first usable JSON
    object. A provider is skipped without spending anything when it has no API
    key or no daily budget left; it is abandoned mid-way when it rate-limits,
    errors, or answers with something that isn't JSON. Every request is charged
    to the provider that made it, whether or not it produced an answer.

    Returns ``(parsed_or_None, total_requests_made)``. ``request_cap`` bounds
    the requests this single listing may make across the whole cascade, so one
    wedged description can't drain the run.
    """
    spent = 0
    tried: list[str] = []
    for provider in cfg["providers"]:
        if request_cap is not None and spent >= request_cap:
            break
        if not get_api_key(provider):
            logger.info("Skipping %s: no API key", provider)
            continue
        provider_left = ledger.remaining(provider, cfg)
        if provider_left <= 0:
            logger.warning("Skipping %s: daily budget spent (%d used)",
                           provider, ledger.used(provider))
            continue
        cap = provider_left if request_cap is None else min(provider_left, request_cap - spent)
        tried.append(provider)
        try:
            parsed, n_req = _provider_call(provider)(text, cfg, request_cap=cap)
        except Exception as e:  # noqa: BLE001 — a provider must never abort the run
            logger.warning("%s raised (%s) — moving to the next provider", provider, e)
            parsed, n_req = None, 0
        ledger.charge(provider, n_req)
        spent += n_req
        if parsed is not None:
            if len(tried) > 1:
                logger.warning("Extraction came from %s after %s refused",
                               provider, ", ".join(tried[:-1]))
            return parsed, spent
        logger.warning("%s produced nothing (%d requests)", provider, n_req)
    if tried:
        logger.warning("No provider produced an extraction: tried %s", ", ".join(tried))
    return None, spent


# ---------------------------------------------------------------------------
# Corrections — base fields (reused) + condition columns
# ---------------------------------------------------------------------------

def cloud_corrections(listing) -> dict:
    """Corrections dict for a listing whose ``_llm_extras`` came from the cascade.

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


def enrich_from_description(
    description: str,
    title: str,
    cfg: dict,
    *,
    ledger: BudgetLedger,
    request_cap: int | None = None,
) -> tuple[dict | None, int]:
    """Extract structured data from title+description via the provider cascade.

    Returns ``(extras_or_None, n_requests_made)``. A description too short to
    carry any signal costs zero requests — the caller relies on that to keep
    filling the run's budget with the next candidate.
    """
    if not description or len(description.strip()) < 20:
        return None, 0
    text = f"{title}\n{description}" if title else description
    return call_llm(text, cfg, ledger=ledger, request_cap=request_cap)
