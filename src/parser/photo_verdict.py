"""Ask a vision model whether a surfaced deal's photos show real damage.

This is the veto. The cheap corpus-wide classifier no longer removes anything
from the feed — it was wrong four times out of five doing that (precision 0.20)
— and now only contributes a ranking weight. What remains is a precise check,
run on the handful of deals a buyer is actually shown.

Why a vision model is the right instrument HERE and nowhere else: it is
accurate but rate-limited. On a hand-labelled set of 30 listings it found all
three damaged cars and raised zero false alarms, and its evidence strings
matched what was visible ("dent and crease on rear quarter panel"). It is also
capped at ~20 requests per day per model on the free tier — useless for 90 000
listings, ample for the ~20 that reach the top of the feed.

That quota is per MODEL, measured off a 429 body, which is why this path is
pointed at a different model than the text enrichment: they draw on separate
buckets and do not starve each other.

The photos go up as ONE image — three frames side by side — because that is
the exact format the accuracy above was measured on, and because it costs one
request instead of three.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import time
from pathlib import Path

import httpx
import yaml

from src.parser.cloud_enrichment import (
    CONFIG_PATH,
    _NO_THINKING_MODELS,
    _parse_json_object,
    _reserve_gemini_slot,
    get_api_key,
)

logger = logging.getLogger(__name__)

# The ledger tracks spend per provider key. The vision channel gets its own
# key rather than sharing "gemini", because it draws on a different model's
# daily quota — sharing one counter would either starve the text path or
# overrun the vision one.
PROVIDER_KEY = "gemini_vision"

_DEFAULTS = {
    "base_url": "https://generativelanguage.googleapis.com/v1beta",
    # Deliberately NOT the model the text cascade leads with. Separate model,
    # separate free-tier bucket.
    "models": ["gemini-flash-lite-latest", "gemini-flash-latest"],
    "timeout_seconds": 90,
    "max_attempts": 2,
    "retry_backoff_seconds": 3,
    "max_rpm": 4,
    # Wait for a slot instead of giving up on one. The 2-second default in
    # the text cascade is right there — a person is waiting and another
    # provider can answer — and wrong here: this is a batch over the day's
    # surfaced deals, nothing else can do the job, and 4/min simply means
    # the pass takes minutes. Measured the hard way: at 2 s, a 24-deal pass
    # gave up on 17 of them without spending a single request.
    # 70 s, not 25: at 4 requests/minute the worst case is waiting out a full
    # window, and 25 s still abandoned 4 deals of 17 on the second pass.
    # The window is shared with the text cascade, so a deal needing a retry
    # can queue behind five other calls.
    "slot_wait_seconds": 70,
    "daily_request_budget": 20,
    "max_photos": 3,
    "sheet_height": 460,
    # Veto only at "needs repair" and above. Severity 1 is ordinary wear for
    # age — dirt, faded lamps, a scuffed bumper — and vetoing on it would
    # repeat the mistake this whole change is undoing.
    "veto_min_severity": 2,
}

_SYSTEM_PROMPT = """\
You assess a used car from its listing photos, for a resale marketplace. The photos are laid out side by side in one image. Return ONE JSON object:
{"severity": 0|1|2|3, "visible_damage": true|false, "interior_only": true|false, "evidence": "short phrase in English"}
severity 0 = pristine, no defect visible; 1 = normal wear for age (dirt, faded lamps, small scratches, dull paint); 2 = damage that needs repair (dent, crease, crash damage, panel or part missing, mid-repair, primer/filler); 3 = wreck / parts-only / non-runner.
visible_damage means collision or body damage, or missing/broken parts — NOT dirt, wear or age. interior_only = the photos show no exterior view of the car.
Be strict: do not invent damage you cannot see. A clean glossy car photographed in a showroom is severity 0.
"""


def get_vision_config() -> dict:
    data = {}
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            data = yaml.safe_load(f) or {}
    return {**_DEFAULTS, **(data.get("gemini_vision", {}) or {})}


def build_sheet(photo_paths: list[Path], height: int) -> bytes | None:
    """Concatenate photos horizontally into one JPEG.

    One image, one request. The frames keep their own aspect ratio so nothing
    is distorted; only the height is normalised.
    """
    from PIL import Image

    imgs = []
    for p in photo_paths:
        try:
            im = Image.open(p).convert("RGB")
        except Exception:  # noqa: BLE001 — a corrupt download must not abort the run
            continue
        imgs.append(im.resize((max(1, int(im.width * height / im.height)), height)))
    if not imgs:
        return None
    sheet = Image.new("RGB", (sum(i.width for i in imgs), height), "white")
    x = 0
    for im in imgs:
        sheet.paste(im, (x, 0))
        x += im.width
    buf = io.BytesIO()
    sheet.save(buf, format="JPEG", quality=88)
    return buf.getvalue()


def _payload(sheet: bytes, model: str, cfg: dict) -> dict:
    gen = {
        "temperature": 0.0,
        "maxOutputTokens": 300,
        "responseMimeType": "application/json",
    }
    if model not in _NO_THINKING_MODELS:
        gen["thinkingConfig"] = {"thinkingBudget": 0}
    return {
        "systemInstruction": {"parts": [{"text": _SYSTEM_PROMPT}]},
        "contents": [{"parts": [{"inlineData": {
            "mimeType": "image/jpeg",
            "data": base64.b64encode(sheet).decode(),
        }}]}],
        "generationConfig": gen,
    }


def judge_sheet(sheet: bytes, cfg: dict, *, request_cap: int | None = None) -> tuple[dict | None, int]:
    """Ask the vision model about one sheet. Returns (verdict_or_None, requests)."""
    key = get_api_key("gemini")
    if not key:
        logger.warning("GEMINI_API_KEY not set — skipping photo verification")
        return None, 0
    headers = {"Content-Type": "application/json", "X-goog-api-key": key}
    base = str(cfg["base_url"]).rstrip("/")
    attempts = max(1, int(cfg.get("max_attempts", 2)))
    backoff = float(cfg.get("retry_backoff_seconds", 3) or 0)
    n = 0

    with httpx.Client(timeout=httpx.Timeout(float(cfg.get("timeout_seconds", 90)), connect=10.0)) as client:
        for model in cfg["models"]:
            url = f"{base}/models/{model}:generateContent"
            for attempt in range(1, attempts + 1):
                if request_cap is not None and n >= request_cap:
                    return None, n
                if not _reserve_gemini_slot(cfg):
                    return None, n
                try:
                    resp = client.post(url, json=_payload(sheet, model, cfg), headers=headers)
                    n += 1
                except httpx.RequestError as e:
                    logger.warning("vision connection error (%s): %s", model, e)
                    n += 1
                    break
                if resp.status_code == 200:
                    try:
                        cand = (resp.json().get("candidates") or [{}])[0]
                        parts = (cand.get("content") or {}).get("parts") or []
                        text = "".join(p.get("text", "") for p in parts).strip()
                    except (json.JSONDecodeError, ValueError, AttributeError):
                        text = ""
                    parsed = _parse_json_object(text)
                    if parsed is not None:
                        parsed["model"] = model
                        return parsed, n
                    logger.info("vision %s returned unusable text; next model", model)
                    break
                if resp.status_code in (401, 403) or (
                    resp.status_code == 400 and "API key not valid" in resp.text
                ):
                    logger.warning("vision auth error HTTP %s — abandoning", resp.status_code)
                    return None, n
                if resp.status_code == 400 and "thinkingConfig" in _payload(sheet, model, cfg)["generationConfig"]:
                    logger.info("vision %s rejects thinkingConfig — retrying without it", model)
                    _NO_THINKING_MODELS.add(model)
                    continue
                if resp.status_code == 429:
                    logger.warning("vision %s rate-limited — next model", model)
                    break
                if resp.status_code >= 500:
                    logger.info("vision %s HTTP %s (attempt %d/%d)",
                                model, resp.status_code, attempt, attempts)
                    if attempt < attempts and backoff:
                        time.sleep(backoff)
                    continue
                logger.warning("vision %s HTTP %s — next model: %s",
                               model, resp.status_code, resp.text[:160])
                break
    return None, n


def verdict_blocks(verdict: dict | None, cfg: dict) -> bool:
    """Does this verdict remove the listing from the feed?"""
    if not isinstance(verdict, dict):
        return False
    try:
        sev = int(verdict.get("severity"))
    except (TypeError, ValueError):
        return False
    return sev >= int(cfg.get("veto_min_severity", 2))
