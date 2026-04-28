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
# Pre-LLM regex hints — for fields qwen3:4b returns null on too eagerly even
# when the trigger word is in plain sight. The hints are appended to the user
# message as a single line; the LLM is free to override if it disagrees, but
# in practice it echoes them, which lifts mechanical_condition and
# first_owner_selling out of the "I'm uncertain → null" trap.
# ---------------------------------------------------------------------------

_HINT_EXCELLENT = re.compile(
    r"\b(impec[áa]vel|como\s+novo|estado\s+irrepreens[íi]vel|"
    r"em\s+perfeito\s+estado|estado\s+perfeito|"
    r"em\s+excelente\s+estado|excelente\s+estado|estado\s+excelente|"
    r"em\s+[óo]timo\s+estado|[óo]timo\s+estado|"
    r"rigorosamente\s+novo)\b",
    re.IGNORECASE,
)

_HINT_GOOD = re.compile(
    r"\b(em\s+bom\s+estado|bom\s+estado|bem\s+cuidad[oa]|"
    r"bem\s+estimad[oa]|muito\s+estimad[oa]|tudo\s+a\s+funcionar)\b",
    re.IGNORECASE,
)

_HINT_FIRST_OWNER = re.compile(
    r"\b(1\s*dono|1[ºo]\s+dono|primeiro\s+dono|[úu]nico\s+dono|"
    r"comprado\s+novo\s+por\s+mim|vendo\s+o\s+meu)\b",
    re.IGNORECASE,
)

_HINT_DEALER = re.compile(
    # Dealer/stand markers — when present, first_owner_selling=false even if
    # the ad mentions "1 dono" (the dealer is reselling, not the original owner).
    r"\bPVP\b|\bIVA\s+(?:discriminado|dedut[íi]vel)\b|"
    r"intermedi[áa]rio\s+de\s+cr[ée]dito|\bstand\b|standcardeira|"
    r"financiamento\s+at[ée]\s+\d+\s+meses",
    re.IGNORECASE,
)


def _extract_hints(text: str) -> dict:
    """Run the cheap regex pre-extract pass.  Only emits hints that are
    *unambiguous* in the surface text — anything fuzzy gets left to the LLM."""
    hints: dict = {}

    if _HINT_EXCELLENT.search(text):
        hints["mechanical_condition"] = "excellent"
    elif _HINT_GOOD.search(text):
        hints["mechanical_condition"] = "good"

    is_dealer = bool(_HINT_DEALER.search(text))
    if _HINT_FIRST_OWNER.search(text) and not is_dealer:
        hints["first_owner_selling"] = True
    elif is_dealer:
        hints["first_owner_selling"] = False

    return hints


def _format_hints(hints: dict) -> str:
    """Render the hint line appended to the user message. Empty when nothing
    matched so we don't bloat the prompt for descriptions that hit no triggers."""
    if not hints:
        return ""
    pairs = ", ".join(
        f'{k}={json.dumps(v, ensure_ascii=False)}' for k, v in hints.items()
    )
    return f"\n\n[Hints (regex-extracted, override only if desc contradicts): {pairs}]"


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
Extract structured features from a Portuguese (pt-PT) car listing as ONE JSON object. Trigger keyword present in text → emit the corresponding value, NEVER null (this overrides "use null when unstated").

Field rules:
sub_model: engine/body variant only (displacement+fuel+power), e.g. "320d","1.6 TDI","2.0 TFSI","A 200","CLA 45". NOT a trim/package, NOT a bare model name.
trim_level: equipment line e.g. "AMG Line","M Sport","S-Line","GTI","FR","Tekna". null if basic.
mileage_in_description_km: integer km. "mil"=thousand only as separate word ("150 mil km"→150000; "89.500km"→89500). Service-interval km ("revisão aos 60.000 km") is NOT current mileage.
desc_mentions_num_owners: integer from "N dono(s)","1º/2º dono","primeiro/segundo dono","único dono"=1. Case-insensitive ("1 DONO"=1).
desc_mentions_accident: true if "sinistro","acidente","batido","embate","toque" appears positively. "sem sinistros" or absent → false.
desc_mentions_repair: true ONLY for damage/breakdown words ("avariado","imobilizado","partido","danificado","precisa reparação"). Routine maintenance ("óleo mudado","correia mudada","pastilhas novas","pneus novos","revisão feita","bateria nova") is NOT repair → false. Absent → false.
desc_mentions_customs_cleared: true if "desalfandegado","legalizado","por legalizar". "Importado"/"Nacional" alone → false. Absent → false.
right_hand_drive: true ONLY for explicit RHD phrases ("mão inglesa","volante à direita","matrícula inglesa","RHD"). "importado" / "matrícula portuguesa/suíça" → false. Absent → false.
urgency: "high" if "urgente","emigração","preço para despachar"; "medium" if "aceito propostas","negociável","oportunidade","aceito retomas"; "low" otherwise.
warranty: true if "garantia" positively ("garantia da marca","X meses de garantia"); "sem garantia" or absent → false.
tuning_or_mods: aftermarket only: ["reprogramação","stage 1","remap","coilovers","bodykit","escape desportivo aftermarket","downpipe","wrap"]. Factory packages ("Sport Chrono","AMG Line","S-Line","R-Line") NOT mods. [] if none.
taxi_fleet_rental: true if "ex-táxi","TVDE","Uber","Bolt","rent-a-car","frota","carro de empresa". Dealer ad / "particular" / absent → false.
first_owner_selling: true if "1 dono","1º dono","primeiro dono","único dono","comprado novo por mim","vendo o meu" (case-insensitive). Dealer markers ("PVP","financiamento","stand","IVA discriminado") OR no first-owner phrase → false. NEVER null.
mechanical_condition: "excellent" if text has any of "impecável","como novo","irrepreensível","perfeito","excelente","ótimo","rigorosamente novo" (with any qualifiers like "estado"/"geral"/"de conservação"); "good" only if ONLY weaker words present ("bom estado","bem cuidado","bem estimado","muito estimado","tudo a funcionar"); "fair" if "uso normal","ligeiros sinais"; "poor" if PARTS-CAR override OR "avariado". null if no condition phrase. Routine maintenance done ≠ excellent.
damage_severity: 0=pristine ("como novo","impecável", warranty + 1 dono); 1=normal wear (DEFAULT for typical used-car); 2=accident OR significant repair ("sinistro","toque","necessita reparações","pintura fraca"); 3=salvage/parts/non-runner ("para peças","sucata","abate","salvado","motor fundido","imobilizado","não anda","sem matrícula"). When in doubt pick higher.

PARTS-CAR OVERRIDE — text contains ANY of "para peças","vender as peças","venda de peças","para desmanchar","só peças","abate","salvado","avariado","imobilizado" → you MUST set: mechanical_condition="poor", desc_mentions_accident=true, desc_mentions_repair=true, damage_severity=3.

Examples:

"BMW Série 3 320d Pack M com 180.000 km, 1 dono, garantia até 2026, sem sinistros."
→ sub_model="320d", trim_level="Pack M", desc_mentions_num_owners=1, mileage_in_description_km=180000, mechanical_condition=null, warranty=true, first_owner_selling=true, damage_severity=0 (booleans false, lists []).

"Vendo Audi A3 1.6 TDI S-Line, 150 mil km. Avariado motor, vendo para peças. Aceito propostas."
→ sub_model="1.6 TDI", trim_level="S-Line", desc_mentions_accident=true, desc_mentions_repair=true, mileage_in_description_km=150000, mechanical_condition="poor", urgency="medium", damage_severity=3.

"Seat Ibiza FR 1.4 TSI com 89.500km. Reprogramação stage 1, escape desportivo. Revisão feita, pneus novos."
→ sub_model="1.4 TSI", trim_level="FR", mileage_in_description_km=89500, tuning_or_mods=["reprogramação","stage 1","escape desportivo"], damage_severity=1.

"Vendo Honda Civic Impecável." → mechanical_condition="excellent", damage_severity=0 (the trigger word IS in the text — do not return null for short ads).

"Mercedes E 53 AMG. Rigorosamente novo. Garantia da Marca." → sub_model="E 53", mechanical_condition="excellent", warranty=true, damage_severity=0.
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
    already documents the 15-field schema, so any instruction-tuned model
    (e.g. qwen3:4b-instruct) matches it without a separate tool wrapper.

    Inference options are tuned for latency on M1 8 GB without quality loss:
      - num_ctx 4096 = budget for system(1210, measured) + desc(≤1200) +
        reply(≤300). 2048 silently truncates the head of the system prompt
        when the description is long, which kills extraction quality.
      - num_predict 300 hard-caps generation; 15 fields × ~12 tokens + JSON
        wrapping ≈ 230, so 300 is a comfortable ceiling and cuts the rare
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
    # Regex pre-extract hints close the gap on the two fields where qwen3:4b
    # punts to null even when the trigger word ("Impecável","1 dono") is
    # right there. Appended to the user message rather than the system prompt
    # because they're per-listing and would otherwise break system-prompt
    # KV-cache reuse.
    prompt_with_hints = truncated + _format_hints(_extract_hints(truncated))
    payload = {
        "model": cfg.get("ollama_model", "qwen3:4b-instruct"),
        "system": _SYSTEM_PROMPT,
        "prompt": prompt_with_hints,
        "format": "json",
        "stream": False,
        "keep_alive": "30m",
        "options": {
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 1.0,
            "repeat_penalty": 1.0,
            "num_ctx": cfg.get("num_ctx", 4096),
            "num_predict": cfg.get("max_tokens", 300),
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

        # Regex hints override LLM on mechanical_condition + first_owner_selling.
        # Empirically (golden 30): when a hint regex fires it agrees with the
        # oracle 100 % of the time, while qwen3:4b still drifts in ~25 % of
        # those same cases. Trusting the regex over the LLM here is a pure-
        # upside override (no observed regressions). Title + desc both feed
        # the regex so a "1 DONO" in the title isn't lost.
        title = getattr(listing, "title", "") or ""
        hints = _extract_hints(f"{title}\n{description}")
        if "mechanical_condition" in hints:
            corrections["mechanical_condition"] = hints["mechanical_condition"]
        if "first_owner_selling" in hints:
            corrections["first_owner_selling"] = hints["first_owner_selling"]

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
