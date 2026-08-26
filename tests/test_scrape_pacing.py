"""Pacing: spread requests instead of firing them in bursts.

The bounded thread pool was the only brake, and a pool is a burst limit, not a
rate limit — eight threads fire together, drain, fire together again. The
2026-08-25 run measured 905 requests in 6m07s in bursts of eight, which is the
traffic shape a CDN's reputation rules are built to notice. Going through the
Cloudflare relay does not make that free: hammering from those addresses would
just get those blocked instead.
"""

import threading
import time

import pytest

from src.parser.scraper import _RateLimiter, _parse_retry_after, _env_float, _env_int


class TestRateLimiter:
    def test_first_request_is_immediate(self):
        t0 = time.monotonic()
        _RateLimiter(2.0).acquire()
        assert time.monotonic() - t0 < 0.05, "a cold bucket must not stall the first call"

    def test_sustained_rate_is_capped(self):
        """8 acquisitions at 20/s must take at least the arithmetic minimum."""
        lim = _RateLimiter(20.0, burst=1)
        t0 = time.monotonic()
        for _ in range(8):
            lim.acquire()
        elapsed = time.monotonic() - t0
        # 7 gaps of 50ms after the first free token; allow scheduler slop.
        assert elapsed >= 0.25, f"ran too fast: {elapsed:.3f}s for 8 at 20/s"

    def test_burst_allowance_is_bounded(self):
        """A bucket that has been idle may burst up to capacity, never past."""
        lim = _RateLimiter(10.0, burst=3)
        time.sleep(0.5)                      # bank tokens (capped at capacity)
        t0 = time.monotonic()
        for _ in range(3):
            lim.acquire()
        assert time.monotonic() - t0 < 0.05, "banked tokens should be free"
        t1 = time.monotonic()
        lim.acquire()                        # 4th must wait for a refill
        assert time.monotonic() - t1 >= 0.05

    def test_threads_share_one_budget(self):
        """The limiter is global, not per-thread — otherwise N threads
        multiply the rate by N, which is exactly the old behaviour."""
        lim = _RateLimiter(20.0, burst=1)
        done = []
        def worker():
            lim.acquire()
            done.append(time.monotonic())
        t0 = time.monotonic()
        threads = [threading.Thread(target=worker) for _ in range(6)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert len(done) == 6
        assert max(done) - t0 >= 0.2, "six threads at 20/s cannot all finish at once"

    def test_rate_is_never_zero_or_negative(self):
        lim = _RateLimiter(0)
        assert lim.rate > 0
        lim2 = _RateLimiter(-5)
        assert lim2.rate > 0


class TestRetryAfter:
    @pytest.mark.parametrize("raw,expect", [
        ("30", 30.0), ("0", 0.0), (" 12.5 ", 12.5),
        (None, None), ("", None), ("Wed, 21 Oct 2026 07:28:00 GMT", None),
    ])
    def test_parsing(self, raw, expect):
        assert _parse_retry_after(raw) == expect

    def test_absurd_values_are_capped(self):
        """A header asking for hours would stall the run; the cron returns anyway."""
        assert _parse_retry_after("86400") == 300.0

    def test_negative_is_clamped_to_zero(self):
        assert _parse_retry_after("-10") == 0.0


class TestEnvKnobs:
    def test_defaults_when_unset(self, monkeypatch):
        monkeypatch.delenv("OLX_MAX_RPS", raising=False)
        assert _env_float("OLX_MAX_RPS", 2.0) == 2.0

    @pytest.mark.parametrize("raw", ["abc", "", "0", "-1"])
    def test_junk_falls_back_to_default(self, monkeypatch, raw):
        """A typo in a secret must not silently disable pacing."""
        monkeypatch.setenv("OLX_MAX_RPS", raw)
        assert _env_float("OLX_MAX_RPS", 2.0) == 2.0

    def test_override_is_honoured(self, monkeypatch):
        monkeypatch.setenv("OLX_MAX_RPS", "0.5")
        assert _env_float("OLX_MAX_RPS", 2.0) == 0.5
        monkeypatch.setenv("OLX_FETCH_WORKERS", "3")
        assert _env_int("OLX_FETCH_WORKERS", 8) == 3
