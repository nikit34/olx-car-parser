"""One process-wide lock around every MPS inference call.

Two separate models run in ``verify-photos``: the CLIP exterior filter and the
damage classifier. Each one guards its own forward pass, but those guards are
per-instance — so with ``--workers 4`` a thread can be inside CLIP's MPS
submission while another is inside the classifier's. Metal is not thread-safe
across independent pipelines, and on the scrape host that overlap aborts the
process rather than degrading: SIGABRT ("Abort trap: 6", exit 134) inside
``MTLReportFailure`` → ``-[AGXG13GFamilyCommandBuffer computeCommandEncoder…]``
during ``GPU::EncodeDescriptor::getcomputeEncoder``. Seen on run 36924350216,
where the step died 11 s in and silently wrote nothing.

The per-instance locks already in ``photo_viewpoint`` / ``photo_damage`` are
kept — they also serialize same-model calls — but they delegate to this shared
lock, so no two models overlap on the GPU. Photo I/O still parallelises across
worker threads, which is where the wall-clock win actually is.

CPU-only hosts are unaffected: the lock is uncontended there, and serializing
an already-serial path costs nothing.
"""
from __future__ import annotations

import threading

# Re-entrant because the plate reader and the classifier can be reached from
# the same worker thread in nested calls; a plain Lock would self-deadlock.
_GPU_LOCK = threading.RLock()


def gpu_lock() -> threading.RLock:
    """The shared MPS serialization lock."""
    return _GPU_LOCK