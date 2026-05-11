"""Portuguese license-plate detection from listing photos.

Runs EasyOCR on exterior photos (already filtered by the CLIP viewpoint
classifier) and looks for substrings matching one of the historic PT
plate formats. The signal we ultimately care about is binary —
"is at least one plate readable across this listing's photos?" — used
downstream as a soft hint (a listing whose every photo blurs/crops out
the plate is mildly suspicious).

Sits *after* ``DamageClassifier`` in the verify-photos pipeline. Single-
load pattern: instantiate one ``PlateReader`` per ``verify-photos``
invocation, call ``read_photo`` per photo.

The OCR backend is heavy (~50 MB of weights downloaded to ``~/.EasyOCR/``
on first run, plus opencv) so the import is deferred to ``__init__`` —
keeping ``photo_plate`` cheap to import in environments that just want
``normalize_plate`` (tests, dashboard label rendering).
"""

from __future__ import annotations

import re
import threading
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# easyocr's internal DataLoader hardcodes pin_memory=True (recognition.py
# lines 201, 215) — no API knob. On MPS that fires a UserWarning per
# readtext() call. Suppress the exact message, not all pin_memory warnings.
warnings.filterwarnings(
    "ignore",
    message=r"'pin_memory' argument is set as true but not supported on MPS.*",
    category=UserWarning,
)


# Portuguese plate eras (each group is 2 chars):
#   1937–1992: ``DD-DD-DD``           (digits only)
#   1992–2005: ``DD-DD-LL`` / ``LL-DD-DD``  (digit/letter mix at one end)
#   2005–2020: ``DD-LL-DD``           (digit-letter-digit sandwich)
#   2020+:     ``LL-DD-LL``           (letter-digit-letter sandwich)
# Anything else (e.g. ``LLLLLL`` or random alphanumeric soup from OCR
# noise) is rejected by ``_is_pt_layout`` below.
_PLATE_RE = re.compile(
    r"(?<![A-Z0-9])"
    r"([A-Z0-9]{2})[\s\-\.·_]?([A-Z0-9]{2})[\s\-\.·_]?([A-Z0-9]{2})"
    r"(?![A-Z0-9])"
)


def _is_pt_layout(a: str, b: str, c: str) -> bool:
    """True iff the three 2-char groups form a documented PT plate layout.

    Each group must be either all-digits or all-letters (mixed groups
    aren't part of any official format and are almost always OCR noise
    bleeding the separator into the group).
    """
    def kind(s: str) -> str:
        if s.isdigit():
            return "D"
        if s.isalpha():
            return "L"
        return "M"
    pattern = kind(a) + kind(b) + kind(c)
    return pattern in ("DDD", "DDL", "LDD", "DLD", "LDL")


def normalize_plate(text: str) -> str | None:
    """Return canonical ``XX-XX-XX`` plate string if ``text`` contains one, else None.

    Conservative: strips common OCR noise (parens, slashes, EU country
    band marks), uppercases, then scans for a 2-2-2 alphanumeric run
    that matches one of the four documented PT formats. Random 6-char
    alphanumeric strings (build numbers, hash fragments) are rejected
    by ``_is_pt_layout``.
    """
    if not text:
        return None
    cleaned = re.sub(r"[^A-Za-z0-9\s\-\.·_]", "", text).upper()
    for m in _PLATE_RE.finditer(cleaned):
        a, b, c = m.groups()
        if _is_pt_layout(a, b, c):
            return f"{a}-{b}-{c}"
    return None


@dataclass
class PlateRead:
    """One OCR detection on a single photo."""
    path: Path
    text: str            # canonical ``XX-XX-XX``
    confidence: float    # 0.0–1.0, from EasyOCR's per-detection score


class PlateReader:
    """Single-instance EasyOCR wrapper. Reuse one instance across photos.

    First instantiation downloads ~50 MB of EasyOCR weights to
    ``~/.EasyOCR/``; subsequent runs hit the cache. CPU-only by default
    — license plates are short and sparse, the GPU win is small and
    avoids contending with the damage classifier / CLIP filter for MPS
    bandwidth on the M1 scrape host.
    """

    # Confidence floor: easyocr returns scores in [0, 1]; below this the
    # detection is ignored even if it pattern-matches a plate. Permissive
    # on purpose — we'd rather a few false positives in the stored text
    # than miss readable plates that happen to OCR fuzzily on a glare.
    MIN_CONFIDENCE = 0.30

    def __init__(self, gpu: bool = False) -> None:
        import easyocr  # heavy: defer to instantiation
        self._reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)
        # Mirror DamageClassifier / ExteriorFilter — easyocr's underlying
        # torch model isn't safe to call concurrently from N threads.
        self._lock = threading.Lock()

    def read_photo(self, path: Path | str) -> PlateRead | None:
        """Highest-confidence PT-plate detection on ``path``, or None.

        Walks every OCR text fragment, normalizes via ``normalize_plate``,
        keeps the one with highest OCR confidence above ``MIN_CONFIDENCE``.
        """
        path = Path(path)
        with self._lock:
            results = self._reader.readtext(
                str(path), detail=1, paragraph=False,
            )
        best: PlateRead | None = None
        for entry in results:
            # easyocr returns (bbox, text, confidence)
            try:
                _, text, conf = entry
            except (ValueError, TypeError):
                continue
            if conf is None or conf < self.MIN_CONFIDENCE:
                continue
            plate = normalize_plate(text)
            if plate is None:
                continue
            if best is None or conf > best.confidence:
                best = PlateRead(path, plate, float(conf))
        return best

    def read_photos(self, paths: Iterable[Path | str]) -> list[PlateRead | None]:
        """Per-photo plate read, preserving order. ``None`` where no plate found."""
        return [self.read_photo(p) for p in paths]
