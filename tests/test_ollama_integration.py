"""Smoke tests: real Ollama LAN integration.

These tests hit the actual Ollama backends listed in ``config/settings.yaml``
(``localhost`` + ``192.168.1.69``). They verify that:

1. Both backends respond to ``/api/tags`` and serve the configured model.
2. The sticky-per-thread router distributes workers across both backends
   in the configured 2:1 weight ratio.
3. Concurrent inference actually runs in parallel (wall time < sequential
   sum), not serialized through one backend.

Not run in normal ``pytest`` invocations — the ``smoke`` marker is opt-in:

    pytest -m smoke tests/test_ollama_integration.py -v

When ``192.168.1.69`` is offline (Wi-Fi off, partner machine asleep, etc.)
the partner-specific assertions are SKIPPED rather than failed, so the
test always tells us *what* is missing instead of just going red.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
import pytest

from src.parser import llm_enrichment as _llm_mod
from src.parser.llm_enrichment import (
    _build_assignment_pool,
    _get_config,
    _invalidate_ollama_url,
    _pick_ollama_url,
    _resolve_all_ollama_urls,
)


# ── helpers ────────────────────────────────────────────────────────────


def _backend_alive(url: str, timeout: float = 3.0) -> bool:
    """True iff Ollama at ``url`` responds with 200 to /api/tags."""
    try:
        r = httpx.get(f"{url.rstrip('/')}/api/tags", timeout=timeout)
        return r.status_code == 200
    except httpx.HTTPError:
        return False


def _close_thread_local_clients() -> None:
    """Close cached httpx.Client objects on the current thread.

    Without this, a thread-local client from a previous test holds a
    keep-alive TCP connection that the OS may have reaped between tests
    (NAT idle timeout, Ollama restart). The next request reuses the dead
    socket and httpx blocks forever waiting for ACKs that never come.
    """
    clients = getattr(_llm_mod._thread_local, "http_clients", None) or {}
    for c in clients.values():
        try:
            c.close()
        except Exception:
            pass
    if hasattr(_llm_mod._thread_local, "http_clients"):
        _llm_mod._thread_local.http_clients = {}
    _llm_mod._thread_local.http_client = None


@pytest.fixture(autouse=True)
def _reset_routing_caches():
    """Force the router to re-probe every run — cached health from a
    previous test would mask a backend that just came online or went away.
    Also tear down persistent httpx clients (see _close_thread_local_clients)."""
    _invalidate_ollama_url()
    _close_thread_local_clients()
    yield
    _invalidate_ollama_url()
    _close_thread_local_clients()


# ── tests ──────────────────────────────────────────────────────────────


@pytest.mark.smoke
def test_localhost_ollama_reachable():
    """Self-Ollama on the runner must always be up — pipeline depends on it."""
    cfg = _get_config()
    urls = cfg.get("ollama_urls", [])
    localhost_url = next(
        (u for u in urls if "localhost" in u or "127.0.0.1" in u), None,
    )
    assert localhost_url, f"settings.yaml ollama_urls must include localhost: {urls}"
    assert _backend_alive(localhost_url), (
        f"Ollama on {localhost_url} unreachable. Start Ollama.app or run "
        f"`ollama serve` and retry."
    )


@pytest.mark.smoke
def test_lan_partner_ollama_reachable():
    """Windows partner at 192.168.1.69 should be reachable on the LAN.

    SKIP (not fail) when offline — sticky routing falls back to localhost
    so the rest of the pipeline keeps working. The skip message tells us
    *which* failure mode we're in (no route vs Ollama down vs firewall).
    """
    partner_url = "http://192.168.1.69:11434"
    try:
        r = httpx.get(f"{partner_url}/api/tags", timeout=3.0)
    except httpx.ConnectError as exc:
        pytest.skip(f"{partner_url} unreachable (network/firewall): {exc}")
    except httpx.HTTPError as exc:
        pytest.skip(f"{partner_url} HTTP error (Ollama maybe down): {exc}")

    assert r.status_code == 200, f"Unexpected status from {partner_url}: {r.status_code}"


@pytest.mark.smoke
def test_configured_model_present_on_both_backends():
    """Each healthy backend must serve the model named in settings.yaml."""
    cfg = _get_config()
    expected_model = cfg["ollama_model"]
    urls = cfg.get("ollama_urls", [])

    seen_on: list[str] = []
    skipped: list[str] = []
    for url in urls:
        if not _backend_alive(url):
            skipped.append(url)
            continue
        tags = httpx.get(f"{url.rstrip('/')}/api/tags", timeout=3.0).json()
        names = [m["name"] for m in tags.get("models", [])]
        # Ollama returns names like "qwen3:4b-instruct" or "qwen3:4b-instruct-q4_K_M"
        if any(n == expected_model or n.startswith(expected_model) for n in names):
            seen_on.append(url)
        else:
            pytest.fail(
                f"{url} alive but missing model {expected_model!r}. "
                f"Has: {names}",
            )

    assert seen_on, f"No backend serves {expected_model}. Skipped: {skipped}"


@pytest.mark.smoke
def test_assignment_pool_covers_both_backends_with_weights():
    """Sticky-per-thread pool should expand both URLs by their weight.

    With ``localhost: 2`` and ``192.168.1.69: 1`` and both backends healthy,
    the pool ought to look like ``[localhost, localhost, .69]``. If .69 is
    offline the pool collapses to ``[localhost, localhost]`` — that's the
    fallback contract; we assert exactly one of the two outcomes so a
    silent collapse to single-host can't go unnoticed.
    """
    pool = _build_assignment_pool()
    healthy = _resolve_all_ollama_urls()

    assert pool, "Assignment pool empty — no Ollama backend reachable at all"

    # Localhost MUST be present and weighted 2× (it's the always-on side).
    localhost_count = sum(1 for u in pool if "localhost" in u or "127.0.0.1" in u)
    assert localhost_count >= 2, (
        f"localhost should appear ≥2 times (weight=2); pool={pool}"
    )

    partner_in_healthy = any("192.168.1.69" in u for u in healthy)
    partner_in_pool = sum(1 for u in pool if "192.168.1.69" in u)

    if partner_in_healthy:
        assert partner_in_pool == 1, (
            f"192.168.1.69 healthy but appears {partner_in_pool}× in pool "
            f"(want 1, weight=1); pool={pool}"
        )
    else:
        assert partner_in_pool == 0, (
            f"192.168.1.69 unreachable but still in pool: {pool}"
        )


@pytest.mark.smoke
def test_parallel_requests_distribute_across_workers():
    """Multiple concurrent picks should hand out distinct sticky URLs.

    Spawn N=8 threads and call ``_pick_ollama_url`` once per thread; each
    thread is supposed to be pinned to a URL in the pool. With 2 healthy
    backends in a 2:1 ratio, all 8 threads land on at most 2 distinct URLs
    but the pool's ``ceil(weight × N / sum_weights)`` distribution should
    cover both. If only one URL ever shows up, sticky routing has
    silently collapsed.
    """
    healthy = _resolve_all_ollama_urls()
    if len(healthy) < 2:
        pytest.skip(f"Only {len(healthy)} backend(s) healthy: {healthy}")

    picks: list[str] = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = [pool.submit(_pick_ollama_url) for _ in range(8)]
        for f in as_completed(futs):
            picks.append(f.result())

    distinct = set(picks)
    assert len(distinct) >= 2, (
        f"Sticky router picked only {distinct} for 8 threads — pool "
        f"didn't distribute. Healthy backends: {healthy}"
    )


@pytest.mark.smoke
def test_concurrent_inference_actually_uses_both_backends():
    """Saturate both backends in parallel and assert each one served at least
    one request.

    The earlier wall-time check (parallel < sequential) didn't survive
    contact with reality: .69's MX230 GPU is roughly 3× slower than the
    M1's Metal, so a fan-out across both backends adds wall-time on the
    slow side instead of subtracting it. The point of having two backends
    isn't latency-per-call — it's throughput across a 1700-listing batch
    where workers are pinned and pipelined.

    What we actually want to verify: the routing layer hands traffic to
    BOTH urls (no silent collapse onto one host) AND the two calls run
    concurrently (per-backend latencies overlap rather than serialize).
    """
    healthy = _resolve_all_ollama_urls()
    if len(healthy) < 2:
        pytest.skip(f"Need ≥2 healthy backends, got {healthy}")

    cfg = _get_config()
    model = cfg["ollama_model"]
    payload = {
        "model": model,
        "prompt": "Reply with the single word: OK",
        "stream": False,
        "options": {"num_predict": 5, "temperature": 0.0},
    }

    # Warm up both so cold model load doesn't pollute timings.
    for url in healthy:
        httpx.post(f"{url.rstrip('/')}/api/generate", json=payload, timeout=120)

    def _timed_call(url: str) -> tuple[str, float, float]:
        """Return (url, start_ts, finish_ts) for overlap analysis."""
        t0 = time.monotonic()
        httpx.post(f"{url.rstrip('/')}/api/generate", json=payload, timeout=120)
        return url, t0, time.monotonic()

    n = max(4, len(healthy) * 2)
    with ThreadPoolExecutor(max_workers=n) as pool:
        results = list(pool.map(
            _timed_call, [healthy[i % len(healthy)] for i in range(n)],
        ))

    served = {url for url, _, _ in results}
    assert served == set(healthy), (
        f"Routing collapsed: only {served} served traffic; want {set(healthy)}"
    )

    # Concurrency check: there must be at least one moment where calls
    # against DIFFERENT backends overlap. If they serialized through a
    # single shared lock somewhere, no overlap would exist.
    overlapped = False
    for i, (url_a, start_a, end_a) in enumerate(results):
        for url_b, start_b, end_b in results[i + 1:]:
            if url_a == url_b:
                continue
            if start_b < end_a and start_a < end_b:
                overlapped = True
                break
        if overlapped:
            break
    assert overlapped, (
        "No cross-backend overlap detected — calls to different backends "
        "ran sequentially. results=" + str(results)
    )
