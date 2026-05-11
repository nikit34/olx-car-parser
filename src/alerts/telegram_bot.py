"""Telegram deal alerts — notify on listings priced below market median."""

import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import yaml
from sqlalchemy.exc import OperationalError

logger = logging.getLogger(__name__)

# Refresh window: candidates whose latest price snapshot is older than this
# get a detail-page re-fetch before the alert fires. Below this (i.e. the
# regular 4h shallow cycle just hit them) the snapshot is fresh enough that
# a probe is wasted HTTP.
ALERT_REFRESH_AGE_HOURS = 6
# Concurrency for the per-candidate detail fetch. Kept low because the
# alert-time refresh runs on top of an already-completed scrape and we
# don't want to elevate OLX's per-IP request rate right after a long
# scrape session.
ALERT_REFRESH_CONCURRENCY = 4
# Per-row commit retry budget. The scrape worker's market_stats commit
# can hold the SQLite write lock for 3-5 min and busy_timeout is 30s, so
# one stuck row used to roll back the whole 500+ candidate batch. The
# 2+4+8+16+32+60×7 ≈ 8 min envelope outlasts the longest lock we've seen.
_REFRESH_RETRY_MAX = 12
_REFRESH_RETRY_BASE_S = 2.0
_REFRESH_RETRY_MAX_WAIT_S = 60.0

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
    generation = deal.get("generation", "")
    city = deal.get("city", "")
    district = deal.get("district", "")
    mileage = deal.get("mileage_km")
    url = deal.get("url", "")

    fire = "🔥🔥🔥" if discount > 25 else ("🔥🔥" if discount > 20 else "🔥")
    location = f"{city}, {district}" if city else district
    gen_tag = f" ({generation})" if generation else ""

    lines = [
        f"{fire} <b>{brand} {model} {year}</b>{gen_tag}",
        f"💰 <b>{price:,} EUR</b> (median {median:,} EUR)",
        f"📉 <b>-{discount}%</b> below market",
    ]
    if mileage and not (isinstance(mileage, float) and mileage != mileage):
        lines.append(f"🛣 {int(mileage):,} km")
    if location:
        lines.append(f"📍 {location}")

    # Description mentions
    warnings = []
    if deal.get("desc_mentions_accident"):
        warnings.append("упоминание ДТП")
    if deal.get("desc_mentions_repair"):
        warnings.append("упоминание ремонта")
    if deal.get("desc_mentions_customs_cleared") is False:
        warnings.append("нет упоминания о растаможке")
    num_owners = deal.get("desc_mentions_num_owners")
    if num_owners and not (isinstance(num_owners, float) and num_owners != num_owners) and int(num_owners) >= 3:
        warnings.append(f"упоминание: {int(num_owners)} владельцев")
    if warnings:
        lines.append(f"📝 Из описания: {', '.join(warnings)}")

    # Seller-profile flags. Negatives fire on definitive disagreements
    # (pseudoprivate, parts-as-private, ≥3 brands as private); positives
    # surface identity strength so the buyer reads "this seller has
    # more identity than the median". Thresholds for definitive flags
    # are presence-only; the multi-brand cut at 3 sits in the top 1.6%
    # of the 2026-05-06 corpus.
    seller_warnings: list[str] = []
    seller_positives: list[str] = []
    if deal.get("seller_pseudoprivate"):
        seller_warnings.append("псевдочастник (Utilizador, но JSON=Empresa)")
    parts_count = deal.get("seller_parts_count")
    if (parts_count and parts_count > 0
            and not deal.get("seller_is_business")):
        seller_warnings.append(f"продаёт запчасти ({int(parts_count)})")
    n_brands = deal.get("seller_distinct_car_brands")
    if (n_brands and not (isinstance(n_brands, float) and n_brands != n_brands)
            and int(n_brands) >= 3
            and not deal.get("seller_is_business")):
        seller_warnings.append(f"{int(n_brands)} разных марок под Particular")
    fs = deal.get("flipper_score")
    fc = deal.get("flipper_confidence")
    if (fs is not None and fc is not None and fc >= 0.4
            and float(fs) >= 0.5):
        emoji = "🚨" if float(fs) >= 0.75 else "⚠️"
        seller_warnings.append(
            f"{emoji} flipper-score {float(fs):.2f} (conf {fc:.0%})"
        )
    social = deal.get("seller_social_account_type")
    if isinstance(social, str) and social:
        seller_positives.append(f"{social} link")
    if deal.get("seller_has_user_photo"):
        seller_positives.append("фото профиля")
    age_days = deal.get("seller_account_age_days")
    if (age_days and not (isinstance(age_days, float) and age_days != age_days)
            and int(age_days) >= 365 * 7):
        seller_positives.append(f"акк {int(age_days)//365}+ лет")
    if seller_warnings:
        lines.append(f"👤 Продавец: {', '.join(seller_warnings)}")
    if seller_positives:
        lines.append(f"✓ Доверие: {', '.join(seller_positives)}")

    # Photo classifier signal (set by `verify-photos` CLI, lives inside
    # llm_extras JSON). Borderline cases pass _blocking_deal_reason but
    # are still worth flagging visually in the alert.
    photo_p = deal.get("photo_damage_p")
    if photo_p is None:
        raw_extras = deal.get("llm_extras")
        if isinstance(raw_extras, str) and raw_extras.strip():
            try:
                photo_p = json.loads(raw_extras).get("photo_damage_p")
            except (TypeError, ValueError):
                photo_p = None
        elif isinstance(raw_extras, dict):
            photo_p = raw_extras.get("photo_damage_p")
    if isinstance(photo_p, (int, float)) and photo_p >= 0.10:
        lines.append(f"📷 Photo classifier: p_damaged={photo_p:.2f}")

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


def _refresh_stale_candidates(new_signals, listings_df,
                              min_discount_pct: float):
    """Re-fetch detail pages for candidates whose price snapshot is stale.

    Why: the shallow 4h scrape only refreshes top-N pages of OLX's "newest
    first" sort, so a listing that sank past page ~30 keeps an outdated
    price in our DB. If the seller raised the price after our last snapshot,
    the discount we compute is fictional. Probing the detail page right
    before the alert fires costs at most a few HTTP calls per cron and
    catches that exact failure mode.

    Returns ``(filtered_signals, refresh_log)``: ``filtered_signals`` is
    ``new_signals`` with any candidate that no longer qualifies after
    refresh dropped + ``price_eur``/``discount_pct`` updated for those that
    still qualify; ``refresh_log`` is a list of dicts for logging.
    """
    from concurrent.futures import ThreadPoolExecutor

    from src.parser.scraper import OlxScraper, ScraperConfig
    from src.storage.database import get_session
    from src.storage.repository import apply_freshness_refresh

    cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(
        hours=ALERT_REFRESH_AGE_HOURS,
    )
    by_olx = listings_df.set_index("olx_id")
    candidates = []
    for olx_id in new_signals["olx_id"]:
        if olx_id not in by_olx.index:
            continue
        last_seen = by_olx.at[olx_id, "last_seen_at"]
        url = by_olx.at[olx_id, "url"]
        # NaT-safe: any non-datetime falls through as "treat as stale".
        if not isinstance(last_seen, datetime) or last_seen < cutoff:
            candidates.append((olx_id, url))

    if not candidates:
        return new_signals, []

    logger.info("Alert refresh: probing %d stale candidates (>%.0fh old)",
                len(candidates), ALERT_REFRESH_AGE_HOURS)

    scraper_cfg = ScraperConfig(concurrency=ALERT_REFRESH_CONCURRENCY)
    refreshed: dict[str, dict] = {}
    with OlxScraper(scraper_cfg) as scraper:
        def _probe(item):
            olx_id, url = item
            if not url:
                return olx_id, None
            try:
                if "standvirtual.com" in url:
                    details = scraper.scrape_standvirtual_detail(url)
                else:
                    details = scraper.scrape_listing_detail(url)
                return olx_id, details if details else None
            except Exception as e:
                logger.warning("Alert refresh probe failed for %s: %s", olx_id, e)
                return olx_id, None

        with ThreadPoolExecutor(max_workers=ALERT_REFRESH_CONCURRENCY) as ex:
            for olx_id, details in ex.map(_probe, candidates):
                if details is not None:
                    refreshed[olx_id] = details

    if not refreshed:
        logger.info("Alert refresh: no detail pages returned usable data")
        return new_signals, []

    # Persist the fresh snapshots so the next dashboard load reflects truth.
    # Per-row commit + retry on SQLite lock: one stuck row used to roll back
    # the whole batch (the scrape worker can hold the write lock for minutes
    # during market_stats), so 500+ successful re-fetches were wasted on a
    # single contended UPDATE. Each row now retries independently and a
    # row that exhausts its budget is dropped, not poisoning the rest.
    session = get_session()
    refresh_log = []
    try:
        for olx_id, details in refreshed.items():
            for attempt in range(_REFRESH_RETRY_MAX):
                try:
                    res = apply_freshness_refresh(session, olx_id, details)
                    session.commit()
                    refresh_log.append(res)
                    break
                except OperationalError as e:
                    if "locked" not in str(e).lower():
                        raise
                    session.rollback()
                    wait = min(
                        _REFRESH_RETRY_BASE_S * (2 ** attempt),
                        _REFRESH_RETRY_MAX_WAIT_S,
                    )
                    logger.warning(
                        "Alert refresh: DB locked on %s, retry %d/%d in %.1fs",
                        olx_id, attempt + 1, _REFRESH_RETRY_MAX, wait,
                    )
                    time.sleep(wait)
            else:
                logger.error(
                    "Alert refresh: gave up persisting %s after %d retries",
                    olx_id, _REFRESH_RETRY_MAX,
                )
    finally:
        session.close()

    # Re-evaluate each candidate against the freshly-fetched price. Median
    # in `new_signals` is segment-level and barely shifts from one snapshot,
    # so we keep it and just recompute discount_pct from the new price.
    drop_ids: set[str] = set()
    new_signals = new_signals.copy()
    for res in refresh_log:
        olx_id = res["olx_id"]
        new_price = res["new_price"]
        if new_price is None:
            continue
        mask = new_signals["olx_id"] == olx_id
        if not mask.any():
            continue
        median = float(new_signals.loc[mask, "median_price_eur"].iloc[0])
        if median <= 0:
            continue
        new_discount = round((1 - new_price / median) * 100, 1)
        old_price = float(new_signals.loc[mask, "price_eur"].iloc[0])
        new_signals.loc[mask, "price_eur"] = new_price
        new_signals.loc[mask, "discount_pct"] = new_discount
        if new_discount < min_discount_pct:
            drop_ids.add(olx_id)
            logger.info(
                "Alert refresh: dropping %s — price %.0f→%.0f, discount now %.1f%% (<%.0f%%)",
                olx_id, old_price, new_price, new_discount, min_discount_pct,
            )
        elif res["price_changed"]:
            logger.info(
                "Alert refresh: %s price %.0f→%.0f, discount now %.1f%%",
                olx_id, old_price, new_price, new_discount,
            )

    if drop_ids:
        new_signals = new_signals[~new_signals["olx_id"].isin(drop_ids)]
    return new_signals, refresh_log


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
    signals, *_ = compute_signals(listings, history)

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

    # Re-probe stale candidates to catch sellers who raised the price after
    # our last snapshot. Filters out anything whose new price no longer
    # qualifies for the discount threshold.
    try:
        new_signals, _ = _refresh_stale_candidates(
            new_signals, listings, cfg["min_discount_pct"],
        )
    except Exception as e:
        logger.warning("Alert refresh stage failed (%s) — proceeding with stale prices", e)

    if new_signals.empty:
        logger.info("All candidates dropped by refresh stage (prices no longer qualify).")
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
