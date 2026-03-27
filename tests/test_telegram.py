"""Tests for Telegram alert formatting."""

from src.alerts.telegram_bot import _format_deal


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
