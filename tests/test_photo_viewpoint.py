"""Tests for the CLIP exterior pre-filter (issue #3).

The audit (#1) found the v2 damage classifier confidently mis-fires on
interiors / engine bays / wheel close-ups (it was trained on full-vehicle
exterior shots only). ``ExteriorFilter`` sits in front of the classifier
and drops OOD viewpoints by max-vs-max prompt-similarity comparison.

These tests stub the CLIP backbone entirely — no model download, no torch
backbone, no pillow open. We only verify the pure-Python decision rule
(``max(exterior_sims) > max(non_exterior_sims)``), batch ordering, and
the empty-batch / single-photo edge cases.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest


# --- CLIP / torch / PIL stubs (load before importing photo_viewpoint) -------
#
# Same pattern as ``tests/test_photo_damage.py``: shim heavy deps in
# ``sys.modules`` *before* the tested module imports them. We want the real
# ``photo_viewpoint`` module though — pop any cached stub so re-running the
# whole pytest suite doesn't pin a leftover.
sys.modules.pop("src.parser.photo_viewpoint", None)


def _install_stubs():
    # torch — keep the real module if already loaded (other tests import
    # the actual torch). We only need ``no_grad`` + ``device`` + the MPS
    # check, all of which the real torch provides on any platform.
    if "torch" not in sys.modules:
        torch_mod = types.ModuleType("torch")

        class _NoGrad:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        torch_mod.no_grad = lambda: _NoGrad()

        def _device(name):
            return f"device:{name}"

        torch_mod.device = _device
        backends = types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: False),
        )
        torch_mod.backends = backends
        sys.modules["torch"] = torch_mod

    # PIL may already be imported by other tests (torchvision pulls it in).
    # Don't replace the module — instead override ``Image.open`` per test
    # via the ``filt`` fixture so a real Image.open never gets called with
    # a bogus path. We do that in the fixture, not here, to keep this
    # function idempotent.

    # transformers.CLIPModel / CLIPProcessor — replaced per-test via
    # monkeypatch so each test injects its own logits matrix.
    if "transformers" not in sys.modules or not hasattr(
        sys.modules["transformers"], "_olx_clip_stub"
    ):
        tr_mod = types.ModuleType("transformers")
        tr_mod._olx_clip_stub = True

        class _FakeCLIPModel:
            @classmethod
            def from_pretrained(cls, _name):
                inst = cls()
                inst._logits = None
                return inst

            def to(self, _device):
                return self

            def eval(self):
                return self

            def __call__(self, **_kw):
                # Default: a single-image, exterior-wins logit row. Tests
                # override ``self._logits`` to drive each scenario.
                logits = self._logits if self._logits is not None else [[5.0, 4.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
                return types.SimpleNamespace(
                    logits_per_image=_FakeTensor(logits),
                )

        class _FakeProcessor:
            @classmethod
            def from_pretrained(cls, _name):
                return cls()

            def __call__(self, **kw):
                # Return a dict of fake tensors that supports ``.to(device)``
                # and unpacks via ``**inp``. Use a tiny class with the
                # ``.to`` method baked in.
                class _T:
                    def to(self, _d):
                        return self

                return {"pixel_values": _T(), "input_ids": _T(), "attention_mask": _T()}

        tr_mod.CLIPModel = _FakeCLIPModel
        tr_mod.CLIPProcessor = _FakeProcessor
        sys.modules["transformers"] = tr_mod


class _FakeTensor:
    """Stand-in for a torch logits-per-image tensor.

    The real code calls ``.cpu().numpy()`` and then iterates rows / slices
    columns + ``.max()``. We mimic the surface area with plain Python lists
    wrapped in a numpy-like proxy so tests don't need real numpy semantics.
    """

    def __init__(self, rows):
        self._rows = rows

    def cpu(self):
        return self

    def numpy(self):
        # Wrap each row in something that supports ``row[:n].max()``.
        return [_FakeRow(r) for r in self._rows]


class _FakeRow:
    def __init__(self, vals):
        self._vals = list(vals)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return _FakeRow(self._vals[item])
        return self._vals[item]

    def max(self):
        return max(self._vals)


_install_stubs()

from src.parser.photo_viewpoint import ExteriorFilter  # noqa: E402


@pytest.fixture
def filt(monkeypatch):
    """An ``ExteriorFilter`` whose CLIP forward pass is fully stubbed.

    The fake model carries an ``_logits`` attribute that each test sets to
    drive the per-photo decision. Returns ``(filter, set_logits)``: call
    ``set_logits([[...], [...], ...])`` before the call under test.

    Also stubs ``PIL.Image.open`` for the duration of the fixture so the
    filter doesn't actually try to read fake paths off disk.
    """

    class _FakeImage:
        def convert(self, _mode):
            return self

    import PIL.Image as _pil_image
    monkeypatch.setattr(_pil_image, "open", lambda _p: _FakeImage())

    f = ExteriorFilter()
    # 2 exterior + 6 non-exterior prompts → 8 columns per row.
    assert f._n_exterior == 2
    assert len(f._exterior_prompts) + len(f._non_exterior_prompts) == 8

    def set_logits(rows):
        f.model._logits = rows

    return f, set_logits


class TestExteriorFilterDecisionRule:
    def test_exterior_wins_returns_true(self, filt):
        """``max(ext_sims) > max(non_ext_sims)`` → True. Mirrors a clean
        side-profile photo: the "car body from outside" prompt scores
        highest, all interior/engine/wheel prompts trail."""
        f, set_logits = filt
        # ext: [3.0, 5.0]  non-ext: [4.5, 2.0, 1.0, 1.0, 1.0, 1.0]
        # max ext = 5.0, max non-ext = 4.5 → exterior.
        set_logits([[3.0, 5.0, 4.5, 2.0, 1.0, 1.0, 1.0, 1.0]])
        result = f.is_exterior(Path("fake.jpg"))
        assert result is True

    def test_non_exterior_wins_returns_false(self, filt):
        """Audit's failure mode: a wheel close-up where the "wheels and
        tyres" prompt blows past the exterior prompts."""
        f, set_logits = filt
        # ext: [2.0, 2.5]  non-ext: [1.0, 1.0, 6.0, 1.0, 1.0, 1.0]
        # max ext = 2.5, max non-ext = 6.0 → non-exterior.
        set_logits([[2.0, 2.5, 1.0, 1.0, 6.0, 1.0, 1.0, 1.0]])
        result = f.is_exterior(Path("fake.jpg"))
        assert result is False

    def test_tie_goes_to_non_exterior(self, filt):
        """``>`` (strict) means a tie counts as non-exterior. Documented
        edge: in practice CLIP logits effectively never tie, but the
        choice keeps the filter conservative on ambiguous shots."""
        f, set_logits = filt
        set_logits([[3.0, 3.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
        result = f.is_exterior(Path("fake.jpg"))
        assert result is False


class TestExteriorFilterBatch:
    def test_batch_preserves_order(self, filt):
        """Each row's decision is independent; output index matches input
        index regardless of how the rows are arranged."""
        f, set_logits = filt
        # Photo 0: non-exterior wins (engine bay scenario)
        # Photo 1: exterior wins (clean exterior)
        # Photo 2: non-exterior wins (interior dashboard)
        # Photo 3: exterior wins
        set_logits([
            [1.0, 1.0, 1.0, 6.0, 1.0, 1.0, 1.0, 1.0],   # → False
            [4.0, 5.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0],   # → True
            [2.0, 1.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0],   # → False
            [3.0, 6.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],   # → True
        ])
        paths = [Path(f"p{i}.jpg") for i in range(4)]
        result = f.is_exterior_batch(paths)
        assert result == [False, True, False, True]

    def test_empty_batch_returns_empty(self, filt):
        """Defensive: ``is_exterior_batch([])`` must not crash and must not
        invoke the model (zero photos → no work to do)."""
        f, _ = filt
        # Sentinel: if the model is called, this would raise because we set
        # logits to a 1-row matrix that wouldn't match an empty input.
        f.model._logits = [[5.0, 4.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
        assert f.is_exterior_batch([]) == []

    def test_single_photo_batch_matches_scalar(self, filt):
        """``is_exterior(p)`` is documented as a thin wrapper; the
        batch-of-one path must give the same answer as the scalar call."""
        f, set_logits = filt
        set_logits([[2.0, 5.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
        scalar = f.is_exterior(Path("p.jpg"))
        # Reset between calls — the fake model would be invoked twice with
        # the same logits, mirroring deterministic real-CLIP behaviour.
        set_logits([[2.0, 5.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
        batched = f.is_exterior_batch([Path("p.jpg")])
        assert batched == [scalar]
        assert batched == [True]


class TestExteriorFilterPromptStructure:
    """Pin the prompt counts: a casual edit that shrinks the exterior
    cluster to 1 prompt would silently bias the max-vs-max comparison.
    The test fails loudly so the change is deliberate."""

    def test_exterior_prompts_are_two(self, filt):
        f, _ = filt
        assert f._n_exterior == 2
        assert len(f._exterior_prompts) == 2

    def test_non_exterior_prompts_cover_audit_failure_modes(self, filt):
        """The audit's confident FPs spanned interior, engine bay, wheels,
        trunk, seat, dashboard/steering. Each failure mode must have a
        prompt or the cluster comparison can't catch it."""
        f, _ = filt
        joined = " ".join(f._non_exterior_prompts).lower()
        for keyword in ("interior", "engine", "wheel", "trunk", "seat", "steering"):
            assert keyword in joined, f"missing audit-failure prompt: {keyword}"
