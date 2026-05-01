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
    """Predict ``p_damaged = idx / 10`` for each downloaded photo path."""

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
        return _StubListingPred(olx_id, photos, max_p, max_p >= self.threshold)


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


def _run_verify(session, monkeypatch, tmp_path, *,
                photo_urls_by_listing: dict[str, list[str]],
                fail_indices: dict[str, set[int]] | None = None,
                threshold: float = 0.2):
    """Drive the typer command with stubbed photo IO + classifier."""
    fail_indices = fail_indices or {}

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
    monkeypatch.setattr(
        _photo_damage_stub, "DamageClassifier", _StubClassifier, raising=False,
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
        cache_dir=tmp_path / "cache",
        dry_run=False,
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
