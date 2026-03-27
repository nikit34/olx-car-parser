"""Telegram deal alerts — notify on listings priced below market median."""

import json
import logging
import os
from pathlib import Path

import httpx
import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"
DEFAULT_STATE_FILE = PROJECT_ROOT / "data" / "alerts_state.json"


def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    alerts = cfg.get("alerts", {})
    return {
        "bot_token": os.environ.get("TELEGRAM_BOT_TOKEN", alerts.get("telegram_bot_token", "")),
        "chat_id": os.environ.get("TELEGRAM_CHAT_ID", alerts.get("telegram_chat_id", "")),
        "min_discount_pct": alerts.get("min_discount_pct", 15),
        "state_file": Path(alerts.get("state_file", str(DEFAULT_STATE_FILE))),
    }


def _load_seen_ids(state_file: Path) -> set:
    if state_file.exists():
        data = json.loads(state_file.read_text())
        return set(data.get("seen_ids", []))
    return set()


def _save_seen_ids(state_file: Path, ids: set):
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps({"seen_ids": sorted(ids)}, indent=2))


def _format_deal(deal: dict) -> str:
    price = int(deal.get("price_eur", 0))
    median = int(deal.get("median_price_eur", 0))
    discount = deal.get("discount_pct", 0)
    brand = deal.get("brand", "?")
    model = deal.get("model", "?")
    year = deal.get("year", "?")
    city = deal.get("city", "")
    district = deal.get("district", "")
    mileage = deal.get("mileage_km")
    url = deal.get("url", "")

    fire = "🔥🔥🔥" if discount > 25 else ("🔥🔥" if discount > 20 else "🔥")
    location = f"{city}, {district}" if city else district

    lines = [
        f"{fire} <b>{brand} {model} {year}</b>",
        f"💰 <b>{price:,} EUR</b> (median {median:,} EUR)",
        f"📉 <b>-{discount}%</b> below market",
    ]
    if mileage:
        lines.append(f"🛣 {int(mileage):,} km")
    if location:
        lines.append(f"📍 {location}")
    if url:
        lines.append(f"\n<a href=\"{url}\">Open on OLX</a>")

    return "\n".join(lines)


def _send_message(bot_token: str, chat_id: str, text: str) -> bool:
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    try:
        resp = httpx.post(url, json={
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }, timeout=15)
        if resp.status_code == 200:
            return True
        logger.warning("Telegram API error: %s %s", resp.status_code, resp.text)
        return False
    except Exception as e:
        logger.error("Failed to send Telegram message: %s", e)
        return False


def check_and_send_alerts() -> int:
    """Check for new deals and send Telegram alerts. Returns count sent."""
    cfg = _load_config()

    if not cfg["bot_token"] or not cfg["chat_id"]:
        logger.warning("Telegram bot_token or chat_id not configured. Skipping alerts.")
        return 0

    # Load data and compute signals
    from src.dashboard.data_loader import load_from_db, compute_signals
    from src.analytics.computed_columns import enrich_listings

    db_data = load_from_db()
    if db_data is None:
        logger.warning("No database data for alerts.")
        return 0

    listings, history = db_data
    listings = enrich_listings(listings)
    signals = compute_signals(listings, history)

    if signals.empty:
        logger.info("No deal signals found.")
        return 0

    # Filter by minimum discount
    signals = signals[signals["discount_pct"] >= cfg["min_discount_pct"]]

    # Filter out already-seen listings
    seen = _load_seen_ids(cfg["state_file"])
    new_signals = signals[~signals["olx_id"].isin(seen)]

    if new_signals.empty:
        logger.info("No new deals since last alert run.")
        return 0

    sent = 0
    for _, deal in new_signals.iterrows():
        msg = _format_deal(deal.to_dict())
        if _send_message(cfg["bot_token"], cfg["chat_id"], msg):
            seen.add(deal["olx_id"])
            sent += 1

    _save_seen_ids(cfg["state_file"], seen)
    logger.info("Sent %d deal alerts.", sent)
    return sent
