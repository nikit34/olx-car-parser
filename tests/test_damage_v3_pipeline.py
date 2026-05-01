"""Smoke tests for the v3 damage classifier mining/labelling pipeline (issue #5).

Exercises the pure-Python parts:

- ``mine_damage_fps`` — bucket math, manifest format, resume keys, per-listing
  scoring loop with mocked classifier + photo fetch.
- ``label_damage_vlm`` — VLM-to-binary mapping, manifest grouping, resumable
  ``(olx_id, photo_idx)`` skip-set, and ``call_vlm`` against a mock httpx
  client.
- ``build_v3_dataset`` — deterministic split, idempotent symlink creation,
  end-to-end assembly from a synthetic ``labels.jsonl``.

We don't exercise training (heavy torch loop, separate concern) or the live
Ollama path (the labelling script is just an HTTP client — covered via mock).
"""

from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Shared torchvision shim — mirrors tests/test_photo_damage.py. Lets the v3
# scripts import ``src.parser.photo_damage`` (which the pipeline uses for type
# hints + lazy classifier load) without needing the real torchvision wheel
# in the lightweight test env.
# ---------------------------------------------------------------------------
sys.modules.pop("src.parser.photo_damage", None)
if "torchvision" not in sys.modules:
    _tv = types.ModuleType("torchvision")
    _tv_models = types.ModuleType("torchvision.models")
    _tv_transforms = types.ModuleType("torchvision.transforms")

    def _noop(*_a, **_kw):  # pragma: no cover
        return None

    for _name in ("resnet50", "efficientnet_b0", "efficientnet_b3"):
        setattr(_tv_models, _name, _noop)
    for _name in ("Compose", "Resize", "ToTensor", "Normalize"):
        setattr(_tv_transforms, _name, _noop)
    _tv.models = _tv_models
    _tv.transforms = _tv_transforms
    sys.modules["torchvision"] = _tv
    sys.modules["torchvision.models"] = _tv_models
    sys.modules["torchvision.transforms"] = _tv_transforms


# ---------------------------------------------------------------------------
# Helpers — every script imports under its scripts/ path, which isn't a
# package by default. We use ``importlib.util`` to load each module by file
# path so the tests don't depend on a sys.path tweak inside the scripts dir.
# ---------------------------------------------------------------------------
def _load_script(name: str):
    path = PROJECT_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_v3_test_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mine_mod():
    return _load_script("mine_damage_fps")


@pytest.fixture(scope="module")
def label_mod():
    return _load_script("label_damage_vlm")


@pytest.fixture(scope="module")
def build_mod():
    return _load_script("build_v3_dataset")


# ---------------------------------------------------------------------------
# mine_damage_fps
# ---------------------------------------------------------------------------
class TestMine:
    def test_bucket_for_covers_all_canonical_scores(self, mine_mod):
        # One probe per bucket plus the boundary edges that bit us in #1.
        cases = {
            0.20: "0.20-0.35",
            0.34: "0.20-0.35",
            0.35: "0.35-0.50",
            0.50: "0.50-0.70",
            0.70: "0.70-0.90",
            0.89: "0.70-0.90",
            0.90: ">=0.90",
            1.00: ">=0.90",
        }
        for p, expected in cases.items():
            assert mine_mod.bucket_for(p) == expected, p

    def test_already_mined_ids_handles_missing_and_malformed(
        self, mine_mod, tmp_path: Path,
    ):
        manifest = tmp_path / "mining_manifest.jsonl"
        # Missing file → empty set (first run).
        assert mine_mod.already_mined_ids(manifest) == set()
        manifest.write_text(
            json.dumps({"olx_id": "AAA", "photo_idx": 1}) + "\n"
            + "  \n"  # blank line
            + "{not json}\n"  # malformed line
            + json.dumps({"olx_id": "BBB", "photo_idx": 1}) + "\n"
            + json.dumps({"olx_id": "AAA", "photo_idx": 2}) + "\n"
        )
        assert mine_mod.already_mined_ids(manifest) == {"AAA", "BBB"}

    def test_mine_listing_writes_canonical_manifest_rows(
        self, mine_mod, tmp_path: Path,
    ):
        # Stub fetch_photos: 3 photos at known URLs.
        urls = [f"https://cdn/example/{i}.jpg" for i in range(1, 4)]
        fetch = MagicMock(return_value=urls)
        # Stub download_photo: write a dummy blob and return True for all.
        def _download(url: str, dest: Path) -> bool:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"\xff\xd8\xff\xe0")  # JPEG SOI marker
            return True
        download = MagicMock(side_effect=_download)
        # Stub classifier: deterministic scores per photo so we can check
        # that listing_max_p is the per-fetch max, not the DB snapshot.
        scored_probs = [0.10, 0.42, 0.88]
        clf = MagicMock()
        clf.predict_photos_batch = MagicMock(return_value=[
            mine_mod.__dict__.get("PhotoPrediction") and None  # noqa
            or types.SimpleNamespace(p_damaged=p) for p in scored_probs
        ])

        listing = {
            "olx_id": "X1",
            "url": "https://standvirtual.com/x1",
            "listing_max_p": 0.42,
            "score_bucket": "0.35-0.50",
        }
        rows = mine_mod.mine_listing(listing, clf, tmp_path / "raw",
                                     fetch, download)
        assert len(rows) == 3
        # Every row carries the listing-wide fresh max, not the per-photo p.
        assert {r["listing_max_p"] for r in rows} == {0.88}
        # Per-photo fields populated.
        assert [r["v2_p_damaged"] for r in rows] == [0.10, 0.42, 0.88]
        assert [r["photo_idx"] for r in rows] == [1, 2, 3]
        assert all(r["score_bucket"] == "0.35-0.50" for r in rows)
        # Photos written under raw/<olx_id>/<idx>.jpg.
        for r in rows:
            assert Path(r["photo_path"]).exists()

    def test_mine_listing_skips_when_no_photos(self, mine_mod, tmp_path: Path):
        listing = {"olx_id": "Y", "url": "https://x", "listing_max_p": 0.5,
                   "score_bucket": "0.50-0.70"}
        rows = mine_mod.mine_listing(listing, MagicMock(), tmp_path / "raw",
                                     fetch_photos_fn=lambda _: [],
                                     download_photo_fn=lambda *_: False)
        assert rows == []


# ---------------------------------------------------------------------------
# label_damage_vlm
# ---------------------------------------------------------------------------
class TestLabel:
    def test_derive_damaged_truth_table(self, label_mod):
        f = label_mod.derive_damaged
        # severity=0 always clean, regardless of visible_damage flag.
        assert f({"severity": 0, "visible_damage": False}) is False
        assert f({"severity": 0, "visible_damage": True}) is False
        # severity≥1 + visible_damage=True → damaged.
        assert f({"severity": 1, "visible_damage": True}) is True
        assert f({"severity": 3, "visible_damage": True}) is True
        # severity≥1 but visible_damage=False → clean (the VLM contradicts
        # itself, treat as the conservative label).
        assert f({"severity": 2, "visible_damage": False}) is False
        # Errors / missing fields → None (skip the photo).
        assert f({"_error": "non-json"}) is None
        assert f({}) is None
        assert f({"severity": "huge"}) is None
        assert f({"severity": 1, "visible_damage": "yes"}) is None

    def test_already_labelled_keys_resume(self, label_mod, tmp_path: Path):
        labels = tmp_path / "labels.jsonl"
        # Empty first run.
        assert label_mod.already_labelled_keys(labels) == set()
        labels.write_text(
            json.dumps({"olx_id": "A", "photo_idx": 1, "damaged": False}) + "\n"
            + json.dumps({"olx_id": "A", "photo_idx": 2, "damaged": True}) + "\n"
            + "garbage\n"
            + json.dumps({"olx_id": "B", "photo_idx": 1, "damaged": False}) + "\n"
        )
        assert label_mod.already_labelled_keys(labels) == {("A", 1), ("A", 2),
                                                           ("B", 1)}

    def test_group_by_listing_preserves_order(self, label_mod):
        rows = [
            {"olx_id": "A", "photo_idx": 1},
            {"olx_id": "B", "photo_idx": 1},
            {"olx_id": "A", "photo_idx": 2},
            {"olx_id": "B", "photo_idx": 2},
        ]
        grouped = label_mod.group_by_listing(rows)
        assert [oid for oid, _ in grouped] == ["A", "B"]
        assert [p["photo_idx"] for p in grouped[0][1]] == [1, 2]
        assert [p["photo_idx"] for p in grouped[1][1]] == [1, 2]

    def test_call_vlm_parses_ollama_response(self, label_mod, tmp_path: Path):
        # Construct a tiny "image" — call_vlm only base64s the bytes.
        img = tmp_path / "p.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0_fake_")
        ollama_payload = {
            "message": {"content": json.dumps({
                "visible_damage": True, "severity": 2,
                "damage_areas": ["front bumper"],
                "evidence": "deep scratch on bumper",
                "is_studio_or_dealer": False,
            })}
        }
        client = MagicMock()
        client.post = MagicMock(return_value=MagicMock(
            status_code=200, json=lambda: ollama_payload,
        ))
        out = label_mod.call_vlm(img, client=client)
        assert out["severity"] == 2
        assert out["visible_damage"] is True
        assert out["damage_areas"] == ["front bumper"]
        # Latency is added regardless of source.
        assert "_latency_s" in out

    def test_call_vlm_handles_http_error(self, label_mod, tmp_path: Path):
        img = tmp_path / "p.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0")
        client = MagicMock()
        client.post = MagicMock(return_value=MagicMock(
            status_code=503, json=lambda: {},
        ))
        out = label_mod.call_vlm(img, client=client)
        assert "_error" in out and "503" in out["_error"]

    def test_call_vlm_handles_non_json(self, label_mod, tmp_path: Path):
        img = tmp_path / "p.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0")
        client = MagicMock()
        client.post = MagicMock(return_value=MagicMock(
            status_code=200,
            json=lambda: {"message": {"content": "not really json"}},
        ))
        out = label_mod.call_vlm(img, client=client)
        assert out["_error"] == "non-json"


# ---------------------------------------------------------------------------
# build_v3_dataset
# ---------------------------------------------------------------------------
class TestBuild:
    def test_deterministic_split_is_stable(self, build_mod):
        # Same key → same split, repeatedly.
        for key in ("AAA_1", "BBB_2", "vlm_X1_3"):
            assert build_mod.deterministic_split(key) == \
                   build_mod.deterministic_split(key)
        # 0% val → everything trains.
        assert build_mod.deterministic_split("anything", 0.0) == "train"
        # 100% val → everything val.
        assert build_mod.deterministic_split("anything", 1.0) == "val"

    def test_deterministic_split_distribution(self, build_mod):
        # ~10% val on a ~thousand-key sample. Loose bounds: deterministic
        # buckets are uniform across MD5 prefixes.
        n_val = sum(1 for i in range(1000)
                    if build_mod.deterministic_split(str(i), 0.10) == "val")
        assert 60 <= n_val <= 140

    def test_ensure_symlink_idempotent(self, build_mod, tmp_path: Path):
        src = tmp_path / "real.jpg"
        src.write_bytes(b"\xff\xd8")
        dst = tmp_path / "out" / "linked.jpg"
        build_mod.ensure_symlink(src, dst)
        assert dst.is_symlink()
        # Calling twice is a no-op (no exception).
        build_mod.ensure_symlink(src, dst)
        assert dst.is_symlink()

    def test_assembly_from_labels_only(self, build_mod, tmp_path: Path):
        # Synthesise a labels.jsonl with a mix of damaged/clean photos.
        photos = []
        for i in range(40):
            p = tmp_path / "raw" / f"{i}.jpg"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"\xff\xd8")
            photos.append(p)
        labels = tmp_path / "labels.jsonl"
        with labels.open("w") as f:
            for i, p in enumerate(photos):
                f.write(json.dumps({
                    "olx_id": f"L{i}",
                    "photo_idx": 1,
                    "photo_path": str(p),
                    "v2_p_damaged": 0.5,
                    "listing_max_p": 0.5,
                    "score_bucket": "0.35-0.50",
                    "vlm_severity": 2 if i % 4 == 0 else 0,
                    "vlm_visible_damage": (i % 4 == 0),
                    "damaged": (i % 4 == 0),
                }) + "\n")

        out = tmp_path / "v3"
        # Deliberately point at a non-existent v2 source — exercises the
        # "host /tmp pruned, no v2 source" branch.
        counts = build_mod.add_vlm_labels(labels, out, val_fraction=0.10)
        assert counts["skipped_missing_file"] == 0
        assert counts["skipped_no_label"] == 0
        # 1-in-4 photos labelled damaged.
        n_damaged = counts["train/damaged"] + counts["val/damaged"]
        n_clean = counts["train/clean"] + counts["val/clean"]
        assert n_damaged == 10
        assert n_clean == 30
        # Symlinks land in the expected directories.
        for cls in ("clean", "damaged"):
            for split in ("train", "val"):
                d = out / split / cls
                if d.exists():
                    for entry in d.iterdir():
                        assert entry.is_symlink()
                        assert entry.name.startswith("vlm_")

    def test_assembly_skips_unlabelled_and_missing(self, build_mod, tmp_path: Path):
        labels = tmp_path / "labels.jsonl"
        with labels.open("w") as f:
            # Photo with damaged=None (unusable) → skipped.
            f.write(json.dumps({"olx_id": "A", "photo_idx": 1,
                                "photo_path": "/does/not/exist.jpg",
                                "damaged": None}) + "\n")
            # Photo whose file doesn't exist on disk.
            f.write(json.dumps({"olx_id": "B", "photo_idx": 1,
                                "photo_path": "/also/missing.jpg",
                                "damaged": False}) + "\n")

        out = tmp_path / "v3"
        counts = build_mod.add_vlm_labels(labels, out, val_fraction=0.10)
        assert counts["skipped_no_label"] == 1
        assert counts["skipped_missing_file"] == 1

    def test_mirror_v2_no_op_when_missing(self, build_mod, tmp_path: Path):
        out = tmp_path / "v3"
        counts = build_mod.mirror_v2(tmp_path / "does_not_exist", out)
        assert all(v == 0 for v in counts.values())
