"""Tests for ``cli verify-photos`` per-photo persistence (issue #4).

The classifier itself is mocked — we only verify that the CLI plumbs
per-photo scores into ``llm_extras.photo_damages`` (idx + p) without
breaking the existing ``photo_damage_p`` / ``photo_damage_n_photos``
listing-level fields used by alerts/dashboard.
"""

from __future__ import annotations

import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest

# ``src.parser.photo_damage`` imports torch/torchvision at module load. The
# minimal CI test environment doesn't install torchvision (heavy GPU dep),
# so we shim the module before the CLI imports it. The ``DamageClassifier``
# attribute is replaced per-test with a deterministic stub.
_photo_damage_stub = types.ModuleType("src.parser.photo_damage")
_photo_damage_stub.DamageClassifier = None  # set per test
sys.modules.setdefault("src.parser.photo_damage", _photo_damage_stub)

# ``src.parser.photo_viewpoint`` imports transformers/torch/PIL at module
# load (issue #3 CLIP pre-filter). Same shim approach as photo_damage. The
# ``ExteriorFilter`` attribute is replaced per-test with a stub so we can
# control exactly which photos pass the filter without touching CLIP.
_photo_viewpoint_stub = types.ModuleType("src.parser.photo_viewpoint")
_photo_viewpoint_stub.ExteriorFilter = None  # set per test
sys.modules.setdefault("src.parser.photo_viewpoint", _photo_viewpoint_stub)

# ``src.parser.photo_plate`` imports easyocr at PlateReader.__init__ time
# (heavy: ~50 MB of weights downloaded on first use). Same shim approach
# as photo_damage / photo_viewpoint — the ``PlateReader`` attribute is
# replaced per-test with a stub so we can control exactly which photos
# emit a plate detection without touching the OCR backend. ``normalize_plate``
# is preserved on the stub so any caller importing it from this module
# (real or stubbed) gets a working implementation.
from src.parser.photo_plate import normalize_plate  # noqa: E402

_photo_plate_stub = types.ModuleType("src.parser.photo_plate")
_photo_plate_stub.PlateReader = None  # set per test
_photo_plate_stub.normalize_plate = normalize_plate
sys.modules.setdefault("src.parser.photo_plate", _photo_plate_stub)

from src import cli as cli_module  # noqa: E402
from src.models.listing import Listing  # noqa: E402


@dataclass
class _StubPhotoPred:
    path: Path
    p_damaged: float
    is_damaged: bool


@dataclass
class _StubListingPred:
    olx_id: str
    photos: list[_StubPhotoPred]
    max_p: float
    is_damaged: bool


class _StubClassifier:
    """Predict ``p_damaged = idx / 10`` for each downloaded photo path.

    Listing-level ``is_damaged`` mirrors the real multi-photo agreement
    rule (issue #2): True iff at least 2 photos have p ≥ 0.30. With the
    ``idx / 10`` scoring scheme that means listings with ≥4 photos are
    automatically flagged (idx 3 → p=0.30, idx 4 → p=0.40, …).
    """

    device = "cpu"
    classes = ["clean", "damaged"]

    def __init__(self, threshold: float = 0.2, **_kwargs):
        self.threshold = threshold

    def predict_listing(self, olx_id, photo_paths):
        photos = []
        for p in photo_paths:
            # _verify_one names files <olx_id>_<idx>.jpg — recover idx.
            stem = Path(p).stem
            idx = int(stem.rsplit("_", 1)[-1])
            prob = round(idx / 10.0, 4)
            photos.append(_StubPhotoPred(Path(p), prob, prob >= self.threshold))
        max_p = max((ph.p_damaged for ph in photos), default=0.0)
        # Match production listing rule from src.parser.photo_damage:
        # ≥ FLAG_MIN_PHOTOS photos at p ≥ FLAG_PHOTO_THRESHOLD.
        n_above = sum(1 for ph in photos if ph.p_damaged >= 0.30)
        is_damaged = n_above >= 2
        return _StubListingPred(olx_id, photos, max_p, is_damaged)


def _seed_listing(session, olx_id: str, url: str, llm_extras: dict | None = None):
    # Production filter requires non-NULL llm_extras (set by the LLM enrich
    # pass before verify-photos runs). Default to {} so every seeded listing
    # is eligible unless a test explicitly opts out.
    payload = {} if llm_extras is None else llm_extras
    listing = Listing(
        olx_id=olx_id,
        url=url,
        title="Stub car",
        brand="Volkswagen",
        model="Golf",
        is_active=True,
        llm_extras=json.dumps(payload),
    )
    session.add(listing)
    session.commit()
    return listing


class _StubExteriorFilter:
    """Deterministic CLIP pre-filter stub.

    By default keeps every photo — preserves pre-#3 behaviour for tests that
    don't care about filtering. Tests that *do* care set
    ``_StubExteriorFilter.ood_indices`` (per-olx_id sets of 1-based indices
    that should be classified as non-exterior) before calling ``_run_verify``.
    """

    device = "cpu"
    # Class-level so tests can configure without instantiation:
    # {olx_id: {idx, idx, ...}}
    ood_indices: dict[str, set[int]] = {}

    def __init__(self, *_a, **_kw):
        pass

    def is_exterior_batch(self, paths):
        results: list[bool] = []
        for p in paths:
            stem = Path(p).stem
            # Filenames are ``<olx_id>_<idx>.jpg``; recover both.
            olx_id, idx_str = stem.rsplit("_", 1)
            idx = int(idx_str)
            ood = idx in self.ood_indices.get(olx_id, set())
            results.append(not ood)
        return results

    def is_exterior(self, path):
        return self.is_exterior_batch([path])[0]


@dataclass
class _StubPlateRead:
    path: Path
    text: str
    confidence: float


class _StubPlateReader:
    """Deterministic plate OCR stub keyed off filename ``<olx_id>_<idx>.jpg``.

    Tests configure ``_StubPlateReader.plates_by_idx[olx_id] = {idx: (text, conf)}``
    before calling ``_run_verify``; absent entries return None (no plate
    detected). Default empty map → every photo returns None, mirroring the
    production case where no plate is readable on any photo.
    """

    # Class-level so tests can configure without instantiation:
    # {olx_id: {idx: (plate_text, confidence)}}
    plates_by_idx: dict[str, dict[int, tuple[str, float]]] = {}

    def __init__(self, *_a, **_kw):
        pass

    def read_photo(self, path):
        stem = Path(path).stem
        olx_id, idx_str = stem.rsplit("_", 1)
        idx = int(idx_str)
        match = self.plates_by_idx.get(olx_id, {}).get(idx)
        if match is None:
            return None
        text, conf = match
        return _StubPlateRead(Path(path), text, float(conf))

    def read_photos(self, paths):
        return [self.read_photo(p) for p in paths]


def _run_verify(session, monkeypatch, tmp_path, *,
                photo_urls_by_listing: dict[str, list[str]],
                fail_indices: dict[str, set[int]] | None = None,
                ood_indices: dict[str, set[int]] | None = None,
                plates_by_idx: dict[str, dict[int, tuple[str, float]]] | None = None,
                threshold: float = 0.2,
                classifier_cls=None,
                dry_run: bool = False,
                backfill_plates: bool = False):
    """Drive the typer command with stubbed photo IO + classifier + CLIP + plate OCR."""
    fail_indices = fail_indices or {}
    # Reset the class-level maps per call so tests don't leak state.
    _StubExteriorFilter.ood_indices = ood_indices or {}
    _StubPlateReader.plates_by_idx = plates_by_idx or {}

    def fake_fetch_photos(url):
        for olx_id, urls in photo_urls_by_listing.items():
            if url.endswith(olx_id) or olx_id in url:
                return list(urls)
        return []

    def fake_download_photo(purl, dest):
        # purl encodes "<olx_id>#<idx>" so we can simulate per-photo failures.
        olx_id, idx_str = purl.rsplit("#", 1)
        idx = int(idx_str)
        if idx in fail_indices.get(olx_id, set()):
            return False
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(b"stub")
        return True

    # Replace DB plumbing so the typer command operates on the in-memory
    # session from conftest.py rather than touching data/olx_cars.db.
    monkeypatch.setattr(cli_module, "init_db", lambda *a, **kw: None)
    monkeypatch.setattr(cli_module, "get_session", lambda: session)
    # Patch the per-module classes that ``cli.verify_photos`` resolves at
    # call time. The shim modules at the top of this file are installed via
    # ``sys.modules.setdefault`` — but other tests (notably
    # ``test_damage_v3_pipeline.py``) ``pop`` the cached entry to force-
    # reload the real module, leaving sys.modules without a key by the time
    # we get here. Re-install a fresh stub when the entry is missing so the
    # monkeypatch always has a target object.
    def _ensure_module(name: str):
        mod = sys.modules.get(name)
        if mod is None:
            mod = types.ModuleType(name)
            sys.modules[name] = mod
        return mod
    monkeypatch.setattr(
        _ensure_module("src.parser.photo_damage"),
        "DamageClassifier", classifier_cls or _StubClassifier, raising=False,
    )
    monkeypatch.setattr(
        _ensure_module("src.parser.photo_viewpoint"),
        "ExteriorFilter", _StubExteriorFilter, raising=False,
    )
    # Plate reader: photo_plate.py defers easyocr to PlateReader.__init__,
    # so the module imports cheaply at test-collect time and rarely gets
    # popped — but we route through the same helper for consistency.
    monkeypatch.setattr(
        _ensure_module("src.parser.photo_plate"),
        "PlateReader", _StubPlateReader, raising=False,
    )
    monkeypatch.setattr(
        "src.parser.photo_fetch.fetch_photos", fake_fetch_photos,
    )
    monkeypatch.setattr(
        "src.parser.photo_fetch.download_photo", fake_download_photo,
    )

    # Single worker keeps test deterministic and side-steps thread/session
    # interactions — the production code commits on the main thread anyway.
    cli_module.verify_photos(
        threshold=threshold,
        workers=1,
        only_text_flagged=False,
        upgrade_legacy=False,
        backfill_plates=backfill_plates,
        cache_dir=tmp_path / "cache",
        dry_run=dry_run,
        limit=None,
    )


class TestVerifyPhotosPerPhotoArray:
    def test_writes_photo_damages_in_idx_order(self, db_session, monkeypatch, tmp_path):
        """Listing with 4 photos gets a sorted ``photo_damages`` array,
        and the legacy listing-level fields keep their semantics."""
        olx_id = "olx-001"
        _seed_listing(
            db_session,
            olx_id=olx_id,
            url=f"https://standvirtual.com/test/{olx_id}",
            llm_extras={"damage_severity": 0},
        )
        _run_verify(
            db_session, monkeypatch, tmp_path,
            photo_urls_by_listing={
                olx_id: [f"{olx_id}#{i}" for i in range(1, 5)],
            },
        )

        listing = db_session.query(Listing).filter_by(olx_id=olx_id).one()
        extras = json.loads(listing.llm_extras)

        # Legacy listing-level fields untouched in semantics.
        assert extras["photo_damage_n_photos"] == 4
        assert extras["photo_damage_p"] == pytest.approx(0.4, rel=1e-3)
        # Backwards-compat: caller-set fields preserved.
        assert extras["damage_severity"] == 0

        # New per-photo array — ascending idx, scores stitched correctly.
        assert extras["photo_damages"] == [
            {"idx": 1, "p": 0.1},
            {"idx": 2, "p": 0.2},
            {"idx": 3, "p": 0.3},
            {"idx": 4, "p": 0.4},
        ]

    def test_skipped_indices_when_some_downloads_fail(self, db_session, monkeypatch, tmp_path):
        """If photo #2 fails to download, ``photo_damages`` keeps idx 1/3/4
        with their original positions — a downstream consumer that re-runs
        ``fetch_photos(url)`` can recover URLs by idx."""
        olx_id = "olx-002"
        _seed_listing(
            db_session,
            olx_id=olx_id,
            url=f"https://standvirtual.com/test/{olx_id}",
        )
        _run_verify(
            db_session, monkeypatch, tmp_path,
            photo_urls_by_listing={
                olx_id: [f"{olx_id}#{i}" for i in range(1, 5)],
            },
            fail_indices={olx_id: {2}},
        )

        listing = db_session.query(Listing).filter_by(olx_id=olx_id).one()
        extras = json.loads(listing.llm_extras)
        idxs = [d["idx"] for d in extras["photo_damages"]]
        assert idxs == [1, 3, 4]
        assert extras["photo_damage_n_photos"] == 3
        # max across surviving photos (idx 4 → 0.4)
        assert extras["photo_damage_p"] == pytest.approx(0.4, rel=1e-3)

    def test_no_photos_yields_empty_array(self, db_session, monkeypatch, tmp_path):
        """Listings whose photos all fail to download still get the new key
        (as ``[]``) so consumers can rely on its presence."""
        olx_id = "olx-003"
        _seed_listing(
            db_session,
            olx_id=olx_id,
            url=f"https://standvirtual.com/test/{olx_id}",
        )
        _run_verify(
            db_session, monkeypatch, tmp_path,
            photo_urls_by_listing={
                olx_id: [f"{olx_id}#{i}" for i in range(1, 3)],
            },
            fail_indices={olx_id: {1, 2}},
        )

        listing = db_session.query(Listing).filter_by(olx_id=olx_id).one()
        extras = json.loads(listing.llm_extras)
        assert extras["photo_damages"] == []
        assert extras["photo_damage_n_photos"] == 0
        assert extras["photo_damage_p"] == 0.0

    def test_skips_already_verified_listings(self, db_session, monkeypatch, tmp_path):
        """``needs_photo`` filter is preserved: a listing that already has
        ``photo_damage_p`` is not re-classified (no overwrite)."""
        olx_id = "olx-004"
        prior = {
            "photo_damage_p": 0.95,
            "photo_damage_n_photos": 7,
            # No ``photo_damages`` key — emulates pre-#4 record.
        }
        _seed_listing(
            db_session,
            olx_id=olx_id,
            url=f"https://standvirtual.com/test/{olx_id}",
            llm_extras=prior,
        )
        _run_verify(
            db_session, monkeypatch, tmp_path,
            photo_urls_by_listing={
                olx_id: [f"{olx_id}#{i}" for i in range(1, 5)],
            },
        )

        listing = db_session.query(Listing).filter_by(olx_id=olx_id).one()
        extras = json.loads(listing.llm_extras)
        # Pre-existing values intact, no new key added — confirms the
        # ``$.photo_damage_p IS NULL`` filter still gates writes.
        assert extras["photo_damage_p"] == 0.95
        assert extras["photo_damage_n_photos"] == 7
        assert "photo_damages" not in extras


class TestVerifyPhotosFlaggedField:
    """Issue #2: ``verify-photos`` writes a listing-level
    ``photo_damage_flagged`` boolean under the multi-photo agreement rule.

    The stub classifier scores ``p_damaged = idx / 10`` so:
      • a 4-photo listing has 2 photos at p ≥ 0.30 (idx 3, 4) → flagged
      • a 2-photo listing has 0 photos at p ≥ 0.30 → not flagged
    """

    def test_writes_flagged_true_when_two_photos_cross_threshold(
        self, db_session, monkeypatch, tmp_path,
    ):
        olx_id = "olx-flag-001"
        _seed_listing(
            db_session,
            olx_id=olx_id,
            url=f"https://standvirtual.com/test/{olx_id}",
        )
        _run_verify(
            db_session, monkeypatch, tmp_path,
            photo_urls_by_listing={
                olx_id: [f"{olx_id}#{i}" for i in range(1, 5)],
            },
        )
        listing = db_session.query(Listing).filter_by(olx_id=olx_id).one()
        extras = json.loads(listing.llm_extras)
        # 4 photos → idx 3 (p=0.3) and idx 4 (p=0.4) both ≥ 0.30 → flagged.
        assert extras["photo_damage_flagged"] is True
        # max_p semantics unchanged — still the peak per-photo score.
        assert extras["photo_damage_p"] == pytest.approx(0.4, rel=1e-3)

    def test_writes_flagged_false_when_only_one_photo_crosses(
        self, db_session, monkeypatch, tmp_path,
    ):
        """2-photo listing: max p=0.20 < 0.30, so 0 photos cross the
        per-photo threshold and the listing must not be flagged — the
        audit's main FP mode reduced to its smallest reproducer."""
        olx_id = "olx-flag-002"
        _seed_listing(
            db_session,
            olx_id=olx_id,
            url=f"https://standvirtual.com/test/{olx_id}",
        )
        _run_verify(
            db_session, monkeypatch, tmp_path,
            photo_urls_by_listing={
                olx_id: [f"{olx_id}#{i}" for i in range(1, 3)],
            },
        )
        listing = db_session.query(Listing).filter_by(olx_id=olx_id).one()
        extras = json.loads(listing.llm_extras)
        assert extras["photo_damage_flagged"] is False
        assert extras["photo_damage_p"] == pytest.approx(0.2, rel=1e-3)


class TestVerifyPhotosExteriorFilter:
    """Issue #3: ``verify-photos`` runs photos through a CLIP exterior
    pre-filter and writes ``photo_damage_n_exterior`` (additive). Photos
    classified as non-exterior (interior / engine bay / wheel close-up /
    etc.) are skipped so the v2 damage classifier — trained on full-vehicle
    exterior shots only — never scores OOD viewpoints.

    The stub ``ExteriorFilter`` filters out indices declared in
    ``ood_indices`` per olx_id; everything else passes through.
    """

    def test_writes_n_exterior_when_no_filtering(
        self, db_session, monkeypatch, tmp_path,
    ):
        """Default stub keeps every photo → ``n_exterior == n_photos`` and
        the per-photo array contains all 4 idx values, unchanged from #4."""
        olx_id = "olx-ext-001"
        _seed_listing(
            db_session,
            olx_id=olx_id,
            url=f"https://standvirtual.com/test/{olx_id}",
        )
        _run_verify(
            db_session, monkeypatch, tmp_path,
            photo_urls_by_listing={
                olx_id: [f"{olx_id}#{i}" for i in range(1, 5)],
            },
        )
        listing = db_session.query(Listing).filter_by(olx_id=olx_id).one()
        extras = json.loads(listing.llm_extras)
        assert extras["photo_damage_n_photos"] == 4
        assert extras["photo_damage_n_exterior"] == 4
        # Per-photo array unchanged: every idx still present.
        assert [d["idx"] for d in extras["photo_damages"]] == [1, 2, 3, 4]

    def test_filters_ood_photos_keeps_original_idx(
        self, db_session, monkeypatch, tmp_path,
    ):
        """4-photo listing where idx 2 and 4 are OOD (e.g. interior +
        wheel close-up). Only idx 1 and 3 reach the damage classifier.

        Crucially the persisted ``idx`` values stay 1 and 3 — NOT
        renumbered to 1 and 2. That preserves the issue #4 invariant that
        consumers can recover photo URLs by re-running ``fetch_photos(url)``
        and joining by 1-based position.
        """
        olx_id = "olx-ext-002"
        _seed_listing(
            db_session,
            olx_id=olx_id,
            url=f"https://standvirtual.com/test/{olx_id}",
        )
        _run_verify(
            db_session, monkeypatch, tmp_path,
            photo_urls_by_listing={
                olx_id: [f"{olx_id}#{i}" for i in range(1, 5)],
            },
            ood_indices={olx_id: {2, 4}},
        )
        listing = db_session.query(Listing).filter_by(olx_id=olx_id).one()
        extras = json.loads(listing.llm_extras)
        # n_photos = downloaded count (4); n_exterior = post-CLIP (2).
        assert extras["photo_damage_n_photos"] == 4
        assert extras["photo_damage_n_exterior"] == 2
        # Stub classifier scores p = idx/10. Original idx preserved so
        # idx=1 → 0.1, idx=3 → 0.3. The dropped idx 2/4 are absent.
        assert extras["photo_damages"] == [
            {"idx": 1, "p": 0.1},
            {"idx": 3, "p": 0.3},
        ]
        # max_p reflects the kept photos only (peak among idx 1, 3 = 0.3).
        assert extras["photo_damage_p"] == pytest.approx(0.3, rel=1e-3)

    def test_all_photos_filtered_out_writes_empty_record(
        self, db_session, monkeypatch, tmp_path,
    ):
        """Every photo OOD (e.g. dealer dumped 3 interior shots) → same
        persisted shape as the no-photos path semantically: max_p=0,
        empty per-photo array, not flagged. ``n_photos`` keeps the
        original download count so the listing still records "we did
        attempt N photos here, none survived the filter"."""
        olx_id = "olx-ext-003"
        _seed_listing(
            db_session,
            olx_id=olx_id,
            url=f"https://standvirtual.com/test/{olx_id}",
        )
        _run_verify(
            db_session, monkeypatch, tmp_path,
            photo_urls_by_listing={
                olx_id: [f"{olx_id}#{i}" for i in range(1, 4)],
            },
            ood_indices={olx_id: {1, 2, 3}},
        )
        listing = db_session.query(Listing).filter_by(olx_id=olx_id).one()
        extras = json.loads(listing.llm_extras)
        # We tried 3 photos; CLIP filter rejected all of them.
        assert extras["photo_damage_n_photos"] == 3
        assert extras["photo_damage_n_exterior"] == 0
        assert extras["photo_damages"] == []
        assert extras["photo_damage_p"] == 0.0
        assert extras["photo_damage_flagged"] is False

    def test_n_exterior_present_when_no_photos_downloaded(
        self, db_session, monkeypatch, tmp_path,
    ):
        """Even when every download fails (so the CLIP filter is never
        invoked), ``photo_damage_n_exterior`` must still be written so
        downstream consumers can rely on the field's presence."""
        olx_id = "olx-ext-004"
        _seed_listing(
            db_session,
            olx_id=olx_id,
            url=f"https://standvirtual.com/test/{olx_id}",
        )
        _run_verify(
            db_session, monkeypatch, tmp_path,
            photo_urls_by_listing={
                olx_id: [f"{olx_id}#{i}" for i in range(1, 3)],
            },
            fail_indices={olx_id: {1, 2}},
        )
        listing = db_session.query(Listing).filter_by(olx_id=olx_id).one()
        extras = json.loads(listing.llm_extras)
        assert extras["photo_damage_n_photos"] == 0
        assert extras["photo_damage_n_exterior"] == 0
        assert extras["photo_damages"] == []


class _AlwaysFailingClassifier:
    """Classifier whose every ``predict_listing`` call raises.

    Reproduces the issue #7 masking pattern at the unit level: the import
    /init succeeds (so the step "ran"), but every listing produces an
    error and zero ``photo_damage_p`` writes happen. Standalone the run
    looks like a successful run with N errors; in CI ``continue-on-error:
    true`` swallows the per-step exit and the whole workflow is green.
    """

    device = "cpu"
    classes = ["clean", "damaged"]

    def __init__(self, threshold: float = 0.2, **_kwargs):
        self.threshold = threshold

    def predict_listing(self, olx_id, photo_paths):
        raise RuntimeError(f"simulated classifier failure on {olx_id}")


class TestVerifyPhotosZeroOutputGuard:
    """Issue #7: verify-photos must exit non-zero with a workflow
    ``::warning::`` when it processes a non-trivial pending queue
    (≥50 listings) but writes zero ``photo_damage_p`` updates. Catches
    the masking pattern that hid two production bugs (transformers
    ImportError in run 25220681021, MPS thread-safety SIGTRAP in run
    25222655513) by burying their failure inside ``continue-on-error:
    true`` step results."""

    def test_warns_and_exits_when_zero_updates_against_50_pending(
        self, db_session, monkeypatch, tmp_path, capsys,
    ):
        photo_urls = {}
        for i in range(60):
            olx_id = f"olx-fail-{i:03d}"
            _seed_listing(
                db_session,
                olx_id=olx_id,
                url=f"https://standvirtual.com/test/{olx_id}",
            )
            photo_urls[olx_id] = [f"{olx_id}#1"]

        # typer.Exit is click.exceptions.Exit (RuntimeError subclass), not
        # SystemExit — typer's runner converts it to a process exit code,
        # but called directly in-process it propagates as the click type.
        import click
        with pytest.raises(click.exceptions.Exit) as exc_info:
            _run_verify(
                db_session, monkeypatch, tmp_path,
                photo_urls_by_listing=photo_urls,
                classifier_cls=_AlwaysFailingClassifier,
            )
        assert exc_info.value.exit_code == 2

        captured = capsys.readouterr()
        assert "::warning::" in captured.out
        assert "verify-photos processed 0" in captured.out
        # Listings should have no photo_damage_p — the guard fired
        # because every classifier call raised before extras was updated.
        for olx_id in photo_urls:
            listing = db_session.query(Listing).filter_by(olx_id=olx_id).one()
            extras = json.loads(listing.llm_extras)
            assert "photo_damage_p" not in extras

    def test_no_warning_when_pending_is_empty(
        self, db_session, monkeypatch, tmp_path, capsys,
    ):
        """Empty pending queue is a legitimate no-op (everything's already
        verified). The guard MUST NOT trip — would otherwise spam warnings
        on every steady-state cron after the backlog is drained."""
        # No listings seeded → pending == [] → command returns early.
        _run_verify(
            db_session, monkeypatch, tmp_path,
            photo_urls_by_listing={},
            classifier_cls=_AlwaysFailingClassifier,
        )
        captured = capsys.readouterr()
        assert "::warning::" not in captured.out

    def test_no_warning_when_dry_run_writes_nothing(
        self, db_session, monkeypatch, tmp_path, capsys,
    ):
        """Dry-run by definition writes nothing — the guard MUST be
        suppressed there or every dry-run smoke test would emit a false
        ``::warning::``."""
        photo_urls = {}
        for i in range(60):
            olx_id = f"olx-dry-{i:03d}"
            _seed_listing(
                db_session,
                olx_id=olx_id,
                url=f"https://standvirtual.com/test/{olx_id}",
            )
            photo_urls[olx_id] = [f"{olx_id}#1"]

        # Dry-run + always-failing classifier: no SystemExit raised even
        # though the queue is large and zero writes happen.
        _run_verify(
            db_session, monkeypatch, tmp_path,
            photo_urls_by_listing=photo_urls,
            classifier_cls=_AlwaysFailingClassifier,
            dry_run=True,
        )
        captured = capsys.readouterr()
        assert "::warning::" not in captured.out


class TestVerifyPhotosPlateOCR:
    """PT plate OCR runs on the same exterior set as the damage classifier
    and persists four ``plate_*`` keys to ``llm_extras``:
      • ``plate_texts`` — per-photo ``[{idx, text, confidence}]``
      • ``plate_n_readable`` — count of photos with a recognized plate
      • ``plate_readable`` — bool, ``plate_n_readable > 0``
      • ``plate_text_primary`` — highest-confidence plate text or None

    Photos sharing ``idx`` semantics with ``photo_damages`` so callers can
    join the two arrays by index.
    """

    def test_writes_plate_fields_when_one_photo_readable(
        self, db_session, monkeypatch, tmp_path,
    ):
        olx_id = "olx-plate-001"
        _seed_listing(
            db_session,
            olx_id=olx_id,
            url=f"https://standvirtual.com/test/{olx_id}",
        )
        _run_verify(
            db_session, monkeypatch, tmp_path,
            photo_urls_by_listing={
                olx_id: [f"{olx_id}#{i}" for i in range(1, 5)],
            },
            # Only photo #2 has a readable plate.
            plates_by_idx={olx_id: {2: ("AB-12-CD", 0.85)}},
        )
        listing = db_session.query(Listing).filter_by(olx_id=olx_id).one()
        extras = json.loads(listing.llm_extras)

        assert extras["plate_readable"] is True
        assert extras["plate_n_readable"] == 1
        assert extras["plate_text_primary"] == "AB-12-CD"
        assert extras["plate_texts"] == [
            {"idx": 2, "text": "AB-12-CD", "confidence": 0.85},
        ]

    def test_no_readable_plate_writes_empty_record(
        self, db_session, monkeypatch, tmp_path,
    ):
        """Listing whose every photo OCRs blank still gets all four keys
        — consumers can rely on field presence to distinguish "verified
        with no plate found" from "not yet verified" (key absent)."""
        olx_id = "olx-plate-002"
        _seed_listing(
            db_session,
            olx_id=olx_id,
            url=f"https://standvirtual.com/test/{olx_id}",
        )
        _run_verify(
            db_session, monkeypatch, tmp_path,
            photo_urls_by_listing={
                olx_id: [f"{olx_id}#{i}" for i in range(1, 4)],
            },
            # No plates_by_idx entries → every read_photo returns None.
        )
        listing = db_session.query(Listing).filter_by(olx_id=olx_id).one()
        extras = json.loads(listing.llm_extras)

        assert extras["plate_readable"] is False
        assert extras["plate_n_readable"] == 0
        assert extras["plate_text_primary"] is None
        assert extras["plate_texts"] == []

    def test_primary_picks_highest_confidence(
        self, db_session, monkeypatch, tmp_path,
    ):
        """Multiple photos with detected plates → ``plate_text_primary`` is
        the highest-confidence one; per-photo array preserves all of them
        ordered by idx ascending."""
        olx_id = "olx-plate-003"
        _seed_listing(
            db_session,
            olx_id=olx_id,
            url=f"https://standvirtual.com/test/{olx_id}",
        )
        _run_verify(
            db_session, monkeypatch, tmp_path,
            photo_urls_by_listing={
                olx_id: [f"{olx_id}#{i}" for i in range(1, 5)],
            },
            plates_by_idx={olx_id: {
                1: ("12-34-56", 0.45),
                3: ("AB-12-CD", 0.92),  # highest
                4: ("AB-12-CD", 0.60),
            }},
        )
        listing = db_session.query(Listing).filter_by(olx_id=olx_id).one()
        extras = json.loads(listing.llm_extras)

        assert extras["plate_n_readable"] == 3
        assert extras["plate_readable"] is True
        # Highest-confidence wins regardless of position in photo order.
        assert extras["plate_text_primary"] == "AB-12-CD"
        assert [d["idx"] for d in extras["plate_texts"]] == [1, 3, 4]

    def test_skips_ood_photos_so_plate_idx_aligns_with_damage_idx(
        self, db_session, monkeypatch, tmp_path,
    ):
        """Plate reader runs ONLY on the CLIP-exterior subset — same idx
        space as ``photo_damages`` so callers can join by ``idx`` without
        worrying about non-exterior frames sneaking back in."""
        olx_id = "olx-plate-004"
        _seed_listing(
            db_session,
            olx_id=olx_id,
            url=f"https://standvirtual.com/test/{olx_id}",
        )
        _run_verify(
            db_session, monkeypatch, tmp_path,
            photo_urls_by_listing={
                olx_id: [f"{olx_id}#{i}" for i in range(1, 5)],
            },
            # idx 2 and 4 are OOD (interior shots). Even though we
            # 'configure' a plate at idx 4, it must not appear because
            # the plate reader is never called on it.
            ood_indices={olx_id: {2, 4}},
            plates_by_idx={olx_id: {
                3: ("12-AB-34", 0.80),
                4: ("XX-XX-XX", 0.99),
            }},
        )
        listing = db_session.query(Listing).filter_by(olx_id=olx_id).one()
        extras = json.loads(listing.llm_extras)

        # Only idx 3 survives both filters.
        assert extras["plate_texts"] == [
            {"idx": 3, "text": "12-AB-34", "confidence": 0.8},
        ]
        assert extras["plate_text_primary"] == "12-AB-34"
        # And the damage array's exterior set matches — same idx universe.
        assert [d["idx"] for d in extras["photo_damages"]] == [1, 3]

    def test_no_photos_yields_empty_plate_record(
        self, db_session, monkeypatch, tmp_path,
    ):
        """Every download fails → the no-photos persistence shape includes
        the four plate keys so consumers don't need a fallback path."""
        olx_id = "olx-plate-005"
        _seed_listing(
            db_session,
            olx_id=olx_id,
            url=f"https://standvirtual.com/test/{olx_id}",
        )
        _run_verify(
            db_session, monkeypatch, tmp_path,
            photo_urls_by_listing={
                olx_id: [f"{olx_id}#{i}" for i in range(1, 3)],
            },
            fail_indices={olx_id: {1, 2}},
        )
        listing = db_session.query(Listing).filter_by(olx_id=olx_id).one()
        extras = json.loads(listing.llm_extras)
        assert extras["plate_readable"] is False
        assert extras["plate_n_readable"] == 0
        assert extras["plate_text_primary"] is None
        assert extras["plate_texts"] == []


class TestVerifyPhotosBackfillPlates:
    """``--backfill-plates`` retro-fits the four ``plate_*`` keys onto rows
    that already have ``photo_damage_p`` from a pre-plate run. Three
    invariants:

      1. Selection picks rows with damage but no plate — listings still
         in the steady-state damage queue (``photo_damage_p`` absent) are
         not touched.
      2. Existing damage scores are preserved bit-for-bit. The classifier
         should not re-run; even if it did, no damage_* extras key is
         allowed to change.
      3. The four plate_* keys land exactly as in the steady-state path.
    """

    def test_skips_rows_still_pending_damage_verification(
        self, db_session, monkeypatch, tmp_path,
    ):
        """A listing without ``photo_damage_p`` is not part of the backfill
        cohort — it'll be picked up by the steady-state cron, where the
        damage path runs alongside plate detection."""
        olx_id = "olx-bf-pending"
        _seed_listing(
            db_session,
            olx_id=olx_id,
            url=f"https://standvirtual.com/test/{olx_id}",
            llm_extras={"damage_severity": 0},  # no photo_damage_p
        )
        _run_verify(
            db_session, monkeypatch, tmp_path,
            photo_urls_by_listing={
                olx_id: [f"{olx_id}#{i}" for i in range(1, 4)],
            },
            plates_by_idx={olx_id: {1: ("AB-12-CD", 0.9)}},
            backfill_plates=True,
        )
        listing = db_session.query(Listing).filter_by(olx_id=olx_id).one()
        extras = json.loads(listing.llm_extras)
        # Untouched: no damage scores, no plate fields.
        assert "photo_damage_p" not in extras
        assert "plate_readable" not in extras

    def test_preserves_existing_damage_scores(
        self, db_session, monkeypatch, tmp_path,
    ):
        """Listing already verified for damage gets plate fields added
        without any change to existing photo_damage_* values — even though
        the stub classifier would score photos differently if invoked."""
        olx_id = "olx-bf-existing"
        # Populated with damage values from a pre-plate run. The stub
        # classifier scores p=idx/10, so a re-run on these 4 photos would
        # produce max_p=0.4 / flagged=True — values that must NOT appear
        # in the persisted record.
        prior = {
            "photo_damage_p": 0.85,
            "photo_damage_n_photos": 3,
            "photo_damage_n_exterior": 3,
            "photo_damages": [
                {"idx": 1, "p": 0.85},
                {"idx": 2, "p": 0.40},
                {"idx": 3, "p": 0.10},
            ],
            "photo_damage_flagged": True,
        }
        _seed_listing(
            db_session,
            olx_id=olx_id,
            url=f"https://standvirtual.com/test/{olx_id}",
            llm_extras=prior,
        )
        _run_verify(
            db_session, monkeypatch, tmp_path,
            photo_urls_by_listing={
                olx_id: [f"{olx_id}#{i}" for i in range(1, 5)],
            },
            plates_by_idx={olx_id: {2: ("AB-12-CD", 0.7)}},
            backfill_plates=True,
        )
        listing = db_session.query(Listing).filter_by(olx_id=olx_id).one()
        extras = json.loads(listing.llm_extras)

        # Damage fields untouched — would be 0.4 / 4 / 4 / [0.1, 0.2, 0.3, 0.4]
        # / True if the classifier had been invoked & persisted.
        assert extras["photo_damage_p"] == 0.85
        assert extras["photo_damage_n_photos"] == 3
        assert extras["photo_damage_n_exterior"] == 3
        assert extras["photo_damage_flagged"] is True
        assert [d["idx"] for d in extras["photo_damages"]] == [1, 2, 3]

        # Plate fields: written from the stub OCR.
        assert extras["plate_readable"] is True
        assert extras["plate_n_readable"] == 1
        assert extras["plate_text_primary"] == "AB-12-CD"
        assert extras["plate_texts"] == [
            {"idx": 2, "text": "AB-12-CD", "confidence": 0.7},
        ]

    def test_skips_rows_already_plated(
        self, db_session, monkeypatch, tmp_path,
    ):
        """A listing already through the plate reader (has plate_readable)
        is filtered out of the backfill selection — backfill is one-shot,
        not idempotent re-processing."""
        olx_id = "olx-bf-plated"
        prior = {
            "photo_damage_p": 0.10,
            "photo_damage_flagged": False,
            "plate_readable": False,
            "plate_n_readable": 0,
            "plate_text_primary": None,
            "plate_texts": [],
        }
        _seed_listing(
            db_session,
            olx_id=olx_id,
            url=f"https://standvirtual.com/test/{olx_id}",
            llm_extras=prior,
        )
        _run_verify(
            db_session, monkeypatch, tmp_path,
            photo_urls_by_listing={
                olx_id: [f"{olx_id}#{i}" for i in range(1, 4)],
            },
            # Even though we 'configure' a plate, this row should be skipped.
            plates_by_idx={olx_id: {1: ("AA-99-AA", 0.95)}},
            backfill_plates=True,
        )
        listing = db_session.query(Listing).filter_by(olx_id=olx_id).one()
        extras = json.loads(listing.llm_extras)
        # Selection filter excluded the row → all values match the prior.
        assert extras["plate_readable"] is False
        assert extras["plate_text_primary"] is None
        assert extras["plate_texts"] == []

    def test_mutually_exclusive_with_upgrade_legacy(
        self, db_session, monkeypatch, tmp_path,
    ):
        """``--backfill-plates`` and ``--upgrade-legacy`` model conflicting
        intents (skip vs re-run damage). Combining must raise so an
        operator typo doesn't silently misroute the cron."""
        import click
        with pytest.raises(click.exceptions.UsageError):
            cli_module.verify_photos(
                threshold=0.2,
                workers=1,
                only_text_flagged=False,
                upgrade_legacy=True,
                backfill_plates=True,
                cache_dir=tmp_path / "cache",
                dry_run=True,
                limit=None,
            )
