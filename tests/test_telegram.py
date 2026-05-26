"""Tests for Telegram alert formatting."""

from unittest.mock import patch

import pytest

from src.alerts.telegram_bot import ChatUnreachable, _format_deal, _send_message


class TestFormatDeal:
    def test_basic_format(self):
        msg = _format_deal({
            "brand": "Volkswagen", "model": "Golf", "year": 2015,
            "price_eur": 8000, "median_price_eur": 12000, "discount_pct": 33.3,
            "generation": "Mk7", "city": "Porto", "district": "Porto",
            "mileage_km": 150000, "url": "https://olx.pt/123",
        })
        assert "Volkswagen Golf 2015" in msg
        assert "(Mk7)" in msg
        assert "8,000 EUR" in msg
        assert "12,000 EUR" in msg
        assert "-33.3%" in msg
        assert "150,000 km" in msg
        assert "Porto" in msg
        assert "olx.pt/123" in msg

    def test_no_generation(self):
        msg = _format_deal({
            "brand": "Smart", "model": "ForTwo", "year": 2010,
            "price_eur": 3000, "median_price_eur": 5000, "discount_pct": 40,
            "generation": "",
        })
        assert "Smart ForTwo 2010" in msg
        assert "()" not in msg  # no empty parens

    def test_fire_emojis_high_discount(self):
        msg = _format_deal({
            "brand": "X", "model": "Y", "year": 2020,
            "price_eur": 5000, "median_price_eur": 10000, "discount_pct": 30,
            "generation": "",
        })
        assert "🔥🔥🔥" in msg

    def test_fire_emojis_medium_discount(self):
        msg = _format_deal({
            "brand": "X", "model": "Y", "year": 2020,
            "price_eur": 8000, "median_price_eur": 10000, "discount_pct": 22,
            "generation": "",
        })
        assert "🔥🔥" in msg
        assert "🔥🔥🔥" not in msg

    def test_fire_emojis_low_discount(self):
        msg = _format_deal({
            "brand": "X", "model": "Y", "year": 2020,
            "price_eur": 8500, "median_price_eur": 10000, "discount_pct": 16,
            "generation": "",
        })
        assert msg.count("🔥") == 1

    def test_no_mileage(self):
        msg = _format_deal({
            "brand": "X", "model": "Y", "year": 2020,
            "price_eur": 5000, "median_price_eur": 10000, "discount_pct": 50,
            "generation": "",
        })
        assert "km" not in msg


class TestSellerWarnings:
    """The seller_* fields default to None on rows whose backfill
    hasn't run yet; the alert formatter must stay silent on those.
    Definitive flags (pseudoprivate, parts-as-private) only fire when
    the underlying booleans/counts are unambiguously set."""

    def _base(self, **extra):
        return {
            "brand": "X", "model": "Y", "year": 2020,
            "price_eur": 5000, "median_price_eur": 10000, "discount_pct": 50,
            "generation": "",
            **extra,
        }

    def test_pseudoprivate_warning_fires(self):
        msg = _format_deal(self._base(seller_pseudoprivate=True))
        assert "псевдочастник" in msg
        assert "Продавец" in msg

    def test_pseudoprivate_silent_when_false(self):
        msg = _format_deal(self._base(seller_pseudoprivate=False))
        assert "псевдочастник" not in msg

    def test_pseudoprivate_silent_when_unknown(self):
        # Backfill hasn't filled this listing's seller yet — must NOT
        # fire (we never want spurious flags before data exists).
        msg = _format_deal(self._base(seller_pseudoprivate=None))
        assert "псевдочастник" not in msg

    def test_parts_warning_fires_for_private_seller_with_parts(self):
        msg = _format_deal(self._base(
            seller_parts_count=12, seller_is_business=False,
        ))
        assert "продаёт запчасти" in msg
        assert "12" in msg

    def test_parts_warning_silent_when_seller_is_business(self):
        # A registered parts-dealer is expected — only call out the
        # private-account-listing-parts case, which is the donor signal.
        msg = _format_deal(self._base(
            seller_parts_count=200, seller_is_business=True,
        ))
        assert "продаёт запчасти" not in msg

    def test_parts_warning_silent_when_zero_parts(self):
        msg = _format_deal(self._base(
            seller_parts_count=0, seller_is_business=False,
        ))
        assert "продаёт запчасти" not in msg

    def test_no_seller_section_when_all_fields_silent(self):
        # Minimal listing without ANY seller signal — the "Продавец:"
        # prefix line must not be emitted at all.
        msg = _format_deal(self._base())
        assert "Продавец" not in msg
        assert "Доверие" not in msg

    def test_distinct_brands_warning_under_private(self):
        msg = _format_deal(self._base(
            seller_distinct_car_brands=4, seller_is_business=False,
        ))
        assert "4 разных марок под Particular" in msg

    def test_distinct_brands_silent_under_business(self):
        msg = _format_deal(self._base(
            seller_distinct_car_brands=8, seller_is_business=True,
        ))
        assert "разных марок" not in msg

    def test_flipper_score_warning_at_strong_threshold(self):
        msg = _format_deal(self._base(
            flipper_score=0.78, flipper_confidence=0.85,
        ))
        assert "flipper-score 0.78" in msg
        assert "🚨" in msg

    def test_flipper_score_uses_warning_emoji_below_strong(self):
        msg = _format_deal(self._base(
            flipper_score=0.55, flipper_confidence=0.85,
        ))
        assert "⚠️" in msg
        assert "flipper-score 0.55" in msg

    def test_flipper_score_silent_below_05(self):
        msg = _format_deal(self._base(
            flipper_score=0.30, flipper_confidence=0.85,
        ))
        assert "flipper-score" not in msg

    def test_flipper_score_silent_below_confidence_gate(self):
        # Score ≥0.5 but confidence under 0.4 — too thin, must not fire.
        msg = _format_deal(self._base(
            flipper_score=0.80, flipper_confidence=0.20,
        ))
        assert "flipper-score" not in msg

    def test_facebook_positive_signal(self):
        msg = _format_deal(self._base(seller_social_account_type="facebook"))
        assert "Доверие" in msg
        assert "facebook link" in msg

    def test_account_age_veteran_positive(self):
        msg = _format_deal(self._base(seller_account_age_days=365 * 9))
        assert "акк 9+ лет" in msg

    def test_account_age_silent_below_seven_years(self):
        msg = _format_deal(self._base(seller_account_age_days=365 * 5))
        assert "лет" not in msg
        assert "Доверие" not in msg

    def test_user_photo_positive(self):
        msg = _format_deal(self._base(seller_has_user_photo=True))
        assert "фото профиля" in msg


class TestSendMessage:
    """A 403 from Telegram means the chat is permanently unreachable for
    this bot (blocked / never started / deactivated). It must escape as
    ChatUnreachable so the alert loop bails instead of burning the
    10-minute step budget retrying every remaining deal."""

    def _resp(self, status_code: int, text: str = ""):
        class _R:
            def __init__(self, sc, t):
                self.status_code = sc
                self.text = t
        return _R(status_code, text)

    def test_403_raises_chat_unreachable(self):
        with patch("src.alerts.telegram_bot.httpx.post",
                   return_value=self._resp(403, '{"description":"blocked"}')):
            with pytest.raises(ChatUnreachable):
                _send_message("tok", "chat", "msg")

    def test_200_returns_true(self):
        with patch("src.alerts.telegram_bot.httpx.post",
                   return_value=self._resp(200)):
            assert _send_message("tok", "chat", "msg") is True

    def test_non_403_error_returns_false(self):
        with patch("src.alerts.telegram_bot.httpx.post",
                   return_value=self._resp(429, "rate-limited")):
            assert _send_message("tok", "chat", "msg") is False
