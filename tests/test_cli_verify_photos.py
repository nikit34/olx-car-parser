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


def _run_verify(session, monkeypatch, tmp_path, *,
                photo_urls_by_listing: dict[str, list[str]],
                fail_indices: dict[str, set[int]] | None = None,
                ood_indices: dict[str, set[int]] | None = None,
                threshold: float = 0.2,
                classifier_cls=None,
                dry_run: bool = False):
    """Drive the typer command with stubbed photo IO + classifier + CLIP."""
    fail_indices = fail_indices or {}
    # Reset the class-level OOD map per call so tests don't leak state.
    _StubExteriorFilter.ood_indices = ood_indices or {}

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
    # Patch ``DamageClassifier`` on whichever module object is currently in
    # ``sys.modules`` — could be our bare stub from this file's import-time
    # ``setdefault``, OR the *real* module if test_photo_damage.py ran first
    # and force-loaded it (it pops the cached stub to populate constants).
    # Either way, ``cli.verify_photos`` resolves ``from src.parser.photo_damage
    # import DamageClassifier`` against ``sys.modules`` at call time.
    monkeypatch.setattr(
        sys.modules["src.parser.photo_damage"],
        "DamageClassifier", classifier_cls or _StubClassifier, raising=False,
    )
    # Same pattern for the issue #3 CLIP pre-filter — patch whichever module
    # object is in ``sys.modules`` (bare stub from this file's setdefault, or
    # the real module if test_photo_viewpoint.py loaded it first).
    monkeypatch.setattr(
        sys.modules["src.parser.photo_viewpoint"],
        "ExteriorFilter", _StubExteriorFilter, raising=False,
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
