"""Enrich listing data using a local Ollama model.

Extracts 3 structured fields from title + description text:
sub_model, trim_level, mileage_in_description_km. damage_severity is
derived deterministically by ``_derive_damage_severity`` (regex), not
asked from the LLM — the 2026-04 ablation showed no model-quality
benefit from any of the LLM-extracted condition / urgency / warranty /
flag fields, so they were removed to free Ollama throughput.
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
# Rule-based damage_severity derivation (no LLM call)
# ---------------------------------------------------------------------------
# When a listing already has a populated `llm_extras` dict from a previous
# enrichment run but is missing the (newer) damage_severity field, we don't
# need a fresh LLM call to backfill it — the existing accident/repair/
# condition flags plus a keyword scan over title+description carry enough
# signal. This path is ~1000× faster than going to Ollama, and on the
# 30-listing oracle it matches the LLM's choice on damage_severity exactly.
# Schema: 0=pristine, 1=normal wear, 2=needs repair OR accident history,
#         3=salvage / parts-only / non-runner.

# A tighter parts-car pattern than _PARTS_CAR_PATTERN (above): the latter
# also matches plain "avariado"/"imobilizado" which on their own only mean
# "needs repair", not "selling for parts". This one only fires on phrasings
# that explicitly mark the listing as parts-only / scrap.
_PARTS_ONLY_HARD_PATTERN = re.compile(
    r"para\s+pe[çc]as|vender\s+as\s+pe[çc]as|venda\s+de\s+pe[çc]as|"
    r"para\s+sucata|para\s+desmanchar|s[óo]\s+pe[çc]as|abate|"
    r"para\s+exporta(?:r|[çc][ãa]o).{0,40}pe[çc]as|"
    r"sem\s+matr[ií]cula|sem\s+documentos",
    re.IGNORECASE,
)

# Severe mechanical / structural damage that's not "selling for parts" —
# the car is whole but seriously broken. Used to land on severity 2-3
# depending on whether mechanical_condition was also flagged "poor".
_SEVERE_DAMAGE_PATTERN = re.compile(
    r"motor\s+(?:fundido|avariad[oa])|caixa\s+avariad[oa]|"
    r"transmiss[ãa]o\s+avariad[oa]|n[ãa]o\s+anda|n[ãa]o\s+funciona|"
    r"n[ãa]o\s+pega|non[\s-]runner|engine\s+seized|capotamento",
    re.IGNORECASE,
)

# Pristine-car signals — used to override the default-1 fallback when the
# extras dict from a previous LLM run didn't set mechanical_condition but
# the description is unmistakably positive. Captures "como novo", "estado
# impecável", "FULL EXTRAS" and the like — phrasings the oracle marks as
# damage_severity=0.
_PRISTINE_PATTERN = re.compile(
    r"como\s+novo|estado\s+impec[áa]vel|\bimpec[áa]vel\b|"
    r"excelente\s+estado|estado\s+excelente|"
    r"perfeito\s+estado|estado\s+perfeito|"
    r"irrepreens[íi]vel|estado\s+de\s+novo|"
    r"\bfull\s+extras\b",
    re.IGNORECASE,
)


def _derive_damage_severity(extras: dict, title: str, description: str) -> int:
    """Return damage_severity 0-3 from already-extracted extras + raw text.

    Used for the backfill path: a listing has llm_extras from a previous
    enrich run but lacks damage_severity (added in DB schema v2 / model v5).
    Re-running the LLM just to recover one integer per row is wasteful; the
    boolean flags + condition + keyword scan deliver the same signal.

    Decision order (first match wins):
      1. Explicit parts-only / no-plates phrasing → 3 (salvage)
      2. Severe mechanical text → 2 (and 3 if condition is also "poor")
      3. desc_mentions_accident OR desc_mentions_repair → 2
      4. mechanical_condition == "excellent" + no damage flags → 0
      5. mechanical_condition == "poor" → 2
      6. fall through → 1 (normal age-appropriate wear)
    """
    text = f"{title or ''} {description or ''}"
    if _PARTS_ONLY_HARD_PATTERN.search(text):
        return 3
    if _SEVERE_DAMAGE_PATTERN.search(text):
        return 3 if extras.get("mechanical_condition") == "poor" else 2

    # Existing flags carry the explicit accident/repair signal that the LLM
    # extracted on the previous pass. Inline the legacy aliases (had_accident,
    # needs_repair) so this helper can be called before _EXTRAS_KEY_ALIASES /
    # _get_extra are defined later in the module.
    accident = extras.get("desc_mentions_accident")
    if accident is None:
        accident = extras.get("had_accident")
    repair = extras.get("desc_mentions_repair")
    if repair is None:
        repair = extras.get("needs_repair")
    if accident or repair:
        return 2

    cond = extras.get("mechanical_condition")
    if cond == "poor":
        return 2
    if cond == "excellent":
        return 0
    # Positive-signal scan — oracle marks "como novo" / "FULL EXTRAS" /
    # "estado impecável" listings as 0 even when the previous LLM pass
    # didn't set mechanical_condition, so we look at the raw text.
    if _PRISTINE_PATTERN.search(text):
        return 0
    # Warranty mention without any damage flag — warranty implies a clean,
    # dealer-grade car most of the time. The structured `warranty` flag
    # from the previous LLM pass is more reliable than a raw "garantia"
    # token in the text (avoids "sem garantia" false positives).
    if extras.get("warranty") is True:
        return 0
    return 1


# ---------------------------------------------------------------------------
# Schema documentation (kept as a Python list so consumers — eval scripts,
# annotation tools — share one source of truth for the field set).
# ---------------------------------------------------------------------------

_FIELD_NAMES = [
    "sub_model", "trim_level", "mileage_in_description_km",
]


_SYSTEM_PROMPT = """\
Extract structured features from a Portuguese (pt-PT) car listing as ONE JSON object. Use null when a field cannot be determined from the text.

Field rules:
sub_model: engine/body variant only (displacement+fuel+power), e.g. "320d","1.6 TDI","2.0 TFSI","A 200","CLA 45". NOT a trim/package, NOT a bare model name.
trim_level: equipment line e.g. "AMG Line","M Sport","S-Line","GTI","FR","Tekna". null if basic.
mileage_in_description_km: integer km. "mil"=thousand only as separate word ("150 mil km"→150000; "89.500km"→89500). Service-interval km ("revisão aos 60.000 km") is NOT current mileage.

Examples:

"BMW Série 3 320d Pack M com 180.000 km, 1 dono, garantia até 2026."
→ {"sub_model":"320d","trim_level":"Pack M","mileage_in_description_km":180000}

"Audi A3 1.6 TDI S-Line, 150 mil km."
→ {"sub_model":"1.6 TDI","trim_level":"S-Line","mileage_in_description_km":150000}

"Seat Ibiza FR 1.4 TSI com 89.500km."
→ {"sub_model":"1.4 TSI","trim_level":"FR","mileage_in_description_km":89500}

"Vendo Honda Civic Impecável." → {"sub_model":null,"trim_level":null,"mileage_in_description_km":null}
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
    urls = cfg.get("ollama_urls")
    if not urls:
        urls = [cfg.get("ollama_url", "http://localhost:11434")]
    return {
        "ollama_model": cfg.get("ollama_model", "qwen3:4b-instruct"),
        "ollama_url": cfg.get("ollama_url", "http://localhost:11434"),
        "ollama_urls": [u for u in urls if u],
        "ollama_weights": cfg.get("ollama_weights") or {},
        "max_workers": cfg.get("max_workers", 3),
        "max_tokens": cfg.get("max_tokens", 300),
        "max_chars": cfg.get("max_chars", 4000),
        "num_ctx": cfg.get("num_ctx", 4096),
    }


_ollama_status: bool | None = None
_resolved_ollama_url: str | None = None
_resolved_ollama_urls: list[str] | None = None
_resolved_assignment_pool: list[str] | None = None
_resolve_lock = threading.Lock()

# Maps thread native ID → assigned backend URL. First time a thread asks for
# a backend we hand it the next one in round-robin order; that pinning sticks
# for the thread's lifetime so each backend's KV-cache stays warm. Atomic via
# a single lock — assignment happens once per worker, not per call.
_thread_backend: dict[int, str] = {}
_thread_backend_lock = threading.Lock()
_next_backend_idx = [0]

# Thread-local persistent httpx.Client. Reusing the TCP connection saves
# ~10-30 ms per call (handshake + slow-start) which on a 1700-listing batch
# adds up to ~30-50 s.
_thread_local = threading.local()


def _get_client(base_url: str) -> httpx.Client:
    """Return a per-(thread, base_url) httpx.Client, creating it on first use.

    Keying by base_url lets us hold persistent connections to several Ollama
    backends in parallel — the local one and the LAN failover — without
    one client's base URL leaking into requests aimed at the other.
    """
    clients = getattr(_thread_local, "http_clients", None)
    if clients is None:
        clients = {}
        _thread_local.http_clients = clients
        # Also expose a single-client alias for back-compat with tests that
        # only check that _get_client was called (don't care about URL).
        _thread_local.http_client = None
    client = clients.get(base_url)
    if client is None:
        client = httpx.Client(base_url=base_url,
                              timeout=httpx.Timeout(120.0, connect=10.0))
        clients[base_url] = client
        _thread_local.http_client = client
    return client


def _resolve_all_ollama_urls() -> list[str]:
    """Probe every URL in `ollama_urls` once and cache the list of healthy ones.

    Used for load-balancing across multiple Ollama hosts (e.g. the M1 8 GB
    scraper at .77 + the Windows 16 GB box at .69). Per-process cache so
    we hit `/api/tags` once at startup, not on every enrichment call.
    """
    global _resolved_ollama_urls
    if _resolved_ollama_urls is not None:
        return _resolved_ollama_urls
    with _resolve_lock:
        if _resolved_ollama_urls is not None:
            return _resolved_ollama_urls
        cfg = _get_config()
        candidates = cfg.get("ollama_urls") or []
        healthy: list[str] = []
        for url in candidates:
            try:
                resp = _get_client(url).get("/api/tags", timeout=2.0)
                if resp.status_code == 200:
                    healthy.append(url)
            except Exception as e:
                logger.warning("Ollama at %s unreachable: %s", url, e)
        _resolved_ollama_urls = healthy
        if healthy:
            logger.info("Ollama backends ready (%d): %s", len(healthy), healthy)
    return healthy


def _resolve_ollama_url() -> str | None:
    """First reachable Ollama backend, or None. Kept for back-compat callers
    that just want any working URL (legacy `_call_ollama` fallback path,
    `_ollama_available()`, ad-hoc scripts). For per-call load balancing use
    `_pick_ollama_url()` instead."""
    healthy = _resolve_all_ollama_urls()
    return healthy[0] if healthy else None


def _build_assignment_pool() -> list[str]:
    """Healthy backends expanded by weight, used for round-robin pinning.

    Weights are read from ``ollama_weights`` in settings.yaml — keys are
    substring-matched against backend URLs (so ``"192.168.1.77": 2`` covers
    ``http://192.168.1.77:11434`` regardless of port). A backend with no
    matching weight defaults to 1. Cached per process; cleared together
    with the URL cache in :func:`_invalidate_ollama_url`.
    """
    global _resolved_assignment_pool
    if _resolved_assignment_pool is not None:
        return _resolved_assignment_pool
    with _resolve_lock:
        if _resolved_assignment_pool is not None:
            return _resolved_assignment_pool
        healthy = _resolve_all_ollama_urls()
        weights = _get_config().get("ollama_weights") or {}
        pool: list[str] = []
        for url in healthy:
            w = 1
            for key, value in weights.items():
                if key and key in url:
                    try:
                        w = max(int(value), 1)
                    except (TypeError, ValueError):
                        w = 1
                    break
            pool.extend([url] * w)
        _resolved_assignment_pool = pool
    return _resolved_assignment_pool


def _pick_ollama_url() -> str | None:
    """Sticky-per-thread round-robin across healthy Ollama backends.

    Each ThreadPoolExecutor worker is pinned to one backend on its first
    call and keeps hitting it for the rest of its life. That way every
    backend retains its own KV-cache for our 1210-token system prompt
    instead of paying re-prefill every time the load shifts.

    Backends listed in ``ollama_weights`` get repeated in the assignment
    pool, so a 2:1 weight gives twice as many workers to the faster host.
    Without weights, distribution is exact-uniform — first worker →
    backend[0], second → backend[1], etc.
    """
    healthy = _resolve_all_ollama_urls()
    if not healthy:
        return None
    tid = threading.get_ident()
    pinned = _thread_backend.get(tid)
    if pinned is not None and pinned in healthy:
        return pinned
    pool = _build_assignment_pool() or healthy
    with _thread_backend_lock:
        # Re-check under lock to avoid double-assignment under contention.
        pinned = _thread_backend.get(tid)
        if pinned is None or pinned not in healthy:
            pinned = pool[_next_backend_idx[0] % len(pool)]
            _next_backend_idx[0] += 1
            _thread_backend[tid] = pinned
    return pinned


def _invalidate_ollama_url() -> None:
    global _resolved_ollama_url, _resolved_ollama_urls, _resolved_assignment_pool, _ollama_status
    _resolved_ollama_url = None
    _resolved_ollama_urls = None
    _resolved_assignment_pool = None
    _ollama_status = None
    # Drop sticky pinning so the next probe re-distributes work among
    # whichever backends come back healthy.
    _thread_backend.clear()
    _next_backend_idx[0] = 0


def _ollama_available(_url: str = "") -> bool:
    """True iff at least one configured Ollama backend answers. Result cached
    per process so we don't probe `/api/tags` on every enrichment call."""
    global _ollama_status
    if _ollama_status is not None:
        return _ollama_status
    if _url:
        # Legacy single-URL probe path retained for callers that pass a URL
        # explicitly (tests, ad-hoc scripts).
        try:
            resp = _get_client(_url).get("/api/tags", timeout=2.0)
            _ollama_status = resp.status_code == 200
        except Exception:
            _ollama_status = False
        if not _ollama_status:
            logger.warning("Ollama not reachable at %s — LLM enrichment disabled.", _url)
        return _ollama_status
    _ollama_status = _resolve_ollama_url() is not None
    if not _ollama_status:
        cfg = _get_config()
        logger.warning("No Ollama backend reachable (tried %s) — LLM enrichment disabled.",
                       ", ".join(cfg.get("ollama_urls") or []))
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
        same KV-cache slot and skips ~700 tokens of prefill every call;
      - no chat-template wrapping → fewer tokens, no template-version drift;
      - keep_alive holds the model in RAM between bursts so we don't pay the
        5 s reload cost on the M1 8 GB box.
    format=json constrains the output to parseable JSON; the system prompt
    documents the 3-field schema, so any instruction-tuned model
    (e.g. qwen3:4b-instruct) matches it without a separate tool wrapper.

    Inference options are tuned for latency on M1 8 GB without quality loss:
      - num_ctx 2048 = budget for system(~280) + desc(≤1200) + reply(≤80).
        Halved from 4096 after the 2026-04 prompt slim — system shrank
        from 1210 → ~280 tokens.
      - num_predict 80 hard-caps generation; 3 fields × ~12 tokens + JSON
        wrapping ≈ 50, so 80 is a comfortable ceiling and cuts the rare
        "model loops" failure mode short.
      - top_k=1 + top_p=1 + repeat_penalty=1 disable every per-token
        sampling check; with temperature=0 the result is identical (greedy)
        and ~3-5 % faster decoding.
      - stop=["}\\n{","} {"] — belt-and-suspenders against the model
        emitting two JSON objects in a row, even though format=json should
        already prevent it.
    """
    # Sticky-per-thread backend pick — each worker prints to its own backend
    # so each backend's prompt cache stays warm. _resolve_ollama_url() is the
    # single-host fallback for environments with one configured backend.
    url = _pick_ollama_url() or _resolve_ollama_url() \
        or cfg.get("ollama_url", "http://localhost:11434")
    client = _get_client(url)
    truncated = text[:cfg.get("max_chars", 4000)]
    payload = {
        "model": cfg.get("ollama_model", "qwen3:4b-instruct"),
        "system": _SYSTEM_PROMPT,
        "prompt": truncated,
        "format": "json",
        "stream": False,
        "keep_alive": "30m",
        "options": {
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 1.0,
            "repeat_penalty": 1.0,
            "num_ctx": cfg.get("num_ctx", 2048),
            "num_predict": cfg.get("max_tokens", 80),
            "stop": ["}\n{", "} {"],
        },
    }
    for attempt in range(2):
        try:
            resp = client.post("/api/generate", json=payload)
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
            logger.warning("Ollama connection error at %s (attempt %d): %s",
                           url, attempt + 1, e)
            # The backend we picked is down. Try a different healthy backend
            # *without* invalidating the global cache (other threads may still
            # be reaching their own backend just fine). Only when no other
            # backend is available do we fall back to invalidating + reprobing.
            healthy = _resolve_all_ollama_urls()
            alt = next((u for u in healthy if u != url), None)
            if alt:
                url = alt
                client = _get_client(url)
                logger.info("Failing over to %s for this request", url)
            else:
                _invalidate_ollama_url()
                new_url = _resolve_ollama_url()
                if new_url and new_url != url:
                    url = new_url
                    client = _get_client(url)
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
    return _call_llm(text, cfg)


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
            listing._llm_extras = result
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
    """Cross-check listing attributes against LLM-extracted data and return corrections.

    Schema-v7-slim: LLM only returns sub_model, trim_level,
    mileage_in_description_km. damage_severity is derived deterministically
    via ``_derive_damage_severity`` (regex over title+description, with any
    legacy llm_extras flags taken into account when present).
    """
    extras = getattr(listing, "_llm_extras", None)
    if extras is None:
        return {}

    corrections = {}

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

    title = getattr(listing, "title", "") or ""
    description = getattr(listing, "description", "") or ""
    corrections["damage_severity"] = _derive_damage_severity(
        extras, title, description,
    )

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
                "Corrected %s: real_mileage=%s, damage_severity=%s",
                listing.olx_id,
                corrections.get("real_mileage_km"),
                corrections.get("damage_severity"),
            )

    logger.info("Applied corrections to %d / %d listings", corrected, len(listings))
    return corrected
