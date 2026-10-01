"""CLIP and the damage classifier must not run MPS inference concurrently.

Metal aborts the process — SIGABRT, "Abort trap: 6", exit 134, inside
``MTLReportFailure`` during ``GPU::EncodeDescriptor::getcomputeEncoder`` —
when two independent pipelines overlap. Each model used to hold its own
lock, which serialized same-model calls but left CLIP free to run against
the classifier on another worker thread. Run 36924350216 died 11 s into
the first-photo backfill and wrote nothing, and ``continue-on-error: true``
turned that into a green step.

These tests pin the one property that fixes it: both models hold the SAME
lock object.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest


@pytest.fixture
def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


class TestSharedLock:
    def test_gpu_lock_is_reentrant(self):
        from src.parser.gpu_lock import gpu_lock
        lock = gpu_lock()
        # Re-entrant: nested acquisition from the same thread must not deadlock.
        with lock:
            with lock:
                pass

    def test_gpu_lock_is_shared_singleton(self):
        from src.parser.gpu_lock import gpu_lock
        assert gpu_lock() is gpu_lock()

    def test_lock_serializes_across_threads(self):
        """The property that fixes the abort: while one thread holds the lock,
        another cannot enter."""
        from src.parser.gpu_lock import gpu_lock
        lock = gpu_lock()
        inside = threading.Event()
        released = threading.Event()

        def _holder():
            with lock:
                inside.set()
                released.wait(timeout=5)

        t = threading.Thread(target=_holder, daemon=True)
        t.start()
        assert inside.wait(timeout=5)
        acquired_other = threading.Event()

        def _other():
            with lock:
                acquired_other.set()

        t2 = threading.Thread(target=_other, daemon=True)
        t2.start()
        # Must NOT acquire while the holder is inside.
        assert not acquired_other.wait(timeout=0.3)
        released.set()
        assert acquired_other.wait(timeout=5)
        t.join(timeout=5)
        t2.join(timeout=5)


class TestModelsUseTheSharedLock:
    """Both model classes must install ``gpu_lock()`` as their inference lock.
    Read out of the source so the assertion needs no weights, no torch, and
    no CLIP download."""

    def _source(self, root: Path, rel: str) -> str:
        return (root / rel).read_text()

    def test_damage_classifier_uses_shared_lock(self, _repo_root):
        src = self._source(_repo_root, "src/parser/photo_damage.py")
        assert "self._inference_lock = gpu_lock()" in src
        assert "self._inference_lock = threading.Lock()" not in src

    def test_viewpoint_filter_uses_shared_lock(self, _repo_root):
        src = self._source(_repo_root, "src/parser/photo_viewpoint.py")
        assert "self._inference_lock = gpu_lock()" in src
        assert "self._inference_lock = threading.Lock()" not in src

    def test_both_import_the_same_accessor(self, _repo_root):
        for rel in ("src/parser/photo_damage.py", "src/parser/photo_viewpoint.py"):
            src = self._source(_repo_root, rel)
            assert "from src.parser.gpu_lock import gpu_lock" in src, rel