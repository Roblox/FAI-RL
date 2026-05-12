"""Tests for ``core.peft_compat``.

Validates that :func:`patch_custom_linears_for_lora`:

1. Reproduces the production bug: PEFT/LoRA refuses
   ``Gemma4ClippableLinear`` as a target module.
2. After patching, PEFT can wrap it, the wrapped module preserves the
   original (clamping) forward behavior, and only LoRA adapter params
   become trainable.
3. Is a no-op on models that already use stock ``nn.Linear``.
4. Safely skips wrappers without a discoverable inner ``nn.Linear``.

No GPU, no real model weights, no HF Hub access required.
"""

from __future__ import annotations

import importlib.util
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

pytest.importorskip("peft")
from peft import LoraConfig, get_peft_model  # noqa: E402

try:
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ClippableLinear
    _HAS_GEMMA4 = True
except Exception:  # pragma: no cover - environment-dependent
    Gemma4ClippableLinear = None  # type: ignore[assignment]
    _HAS_GEMMA4 = False

# Load core/peft_compat.py directly so we don't pull in the rest of
# ``core`` (its ``__init__`` eagerly imports the full trainer stack,
# including wandb, which we don't need or want for unit tests).
_PEFT_COMPAT_PATH = Path(__file__).resolve().parents[1] / "core" / "peft_compat.py"
_spec = importlib.util.spec_from_file_location("peft_compat_under_test", _PEFT_COMPAT_PATH)
assert _spec is not None and _spec.loader is not None
_pc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pc)
patch_custom_linears_for_lora = _pc.patch_custom_linears_for_lora
_LinearShim = _pc._LinearShim


pytestmark = pytest.mark.skipif(
    not _HAS_GEMMA4,
    reason="Gemma4ClippableLinear requires transformers>=5.8.0",
)


def _make_tiny_model(use_clipped: bool = True) -> nn.Module:
    """Tiny model with one ``Gemma4ClippableLinear`` named ``proj``."""
    cfg = types.SimpleNamespace(use_clipped_linears=use_clipped)

    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = Gemma4ClippableLinear(cfg, in_features=8, out_features=4)
            if use_clipped:
                # Narrow clamp range so we can detect it numerically.
                with torch.no_grad():
                    self.proj.input_min.fill_(-0.5)
                    self.proj.input_max.fill_(0.5)
                    self.proj.output_min.fill_(-1.0)
                    self.proj.output_max.fill_(1.0)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.proj(x)

    return Tiny()


def _make_lora() -> LoraConfig:
    return LoraConfig(r=4, lora_alpha=8, target_modules=["proj"], lora_dropout=0.0)


def test_get_peft_model_without_patch_raises() -> None:
    """Reproduces the production bug: PEFT refuses ``Gemma4ClippableLinear``."""
    model = _make_tiny_model()
    with pytest.raises(ValueError, match="Gemma4ClippableLinear"):
        get_peft_model(model, _make_lora())


def test_patch_replaces_with_linear_shim() -> None:
    model = _make_tiny_model()
    assert type(model.proj).__name__ == "Gemma4ClippableLinear"

    n = patch_custom_linears_for_lora(model)
    assert n == 1
    assert isinstance(model.proj, _LinearShim)
    # Crucially, the shim is an ``nn.Linear`` subclass so PEFT's
    # ``isinstance(target, nn.Linear)`` check now succeeds.
    assert isinstance(model.proj, nn.Linear)
    # Shape/dtype mirror the inner projection.
    assert model.proj.in_features == 8
    assert model.proj.out_features == 4


def test_patched_model_preserves_clamp_behavior() -> None:
    """Forward output must be bitwise-identical after patching.

    Weight/bias are shared with the inner ``nn.Linear`` and the shim's
    ``forward`` delegates to the original module, so the clamping path
    is the exact same code.
    """
    model = _make_tiny_model()

    torch.manual_seed(0)
    x = torch.randn(2, 8) * 5.0  # well outside the ±0.5 input clamp
    y_before = model(x).detach().clone()

    n = patch_custom_linears_for_lora(model)
    assert n == 1

    y_after = model(x)
    torch.testing.assert_close(y_after, y_before)
    # Sanity: output is actually clamped to [-1.0, 1.0].
    assert y_after.abs().max().item() <= 1.0 + 1e-6


def test_get_peft_model_after_patch_succeeds() -> None:
    model = _make_tiny_model()
    patch_custom_linears_for_lora(model)
    peft_model = get_peft_model(model, _make_lora())

    wrapped = peft_model.base_model.model.proj
    # PEFT must keep our shim as ``base_layer`` — that's what guarantees
    # the clamp survives every training step (lora.Linear.forward calls
    # base_layer(x) for the residual).
    assert isinstance(wrapped.base_layer, _LinearShim)

    total = sum(p.numel() for p in peft_model.parameters())
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    assert 0 < trainable < total


def test_patch_is_noop_on_stock_linear() -> None:
    class Plain(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(8, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.proj(x)

    model = Plain()
    n = patch_custom_linears_for_lora(model)
    assert n == 0
    assert type(model.proj) is nn.Linear


def test_patch_skips_when_no_inner_linear(caplog: pytest.LogCaptureFixture) -> None:
    """Wrappers without a discoverable inner ``nn.Linear`` are left alone."""

    class FakeWrapper(nn.Module):
        # Doesn't actually wrap an nn.Linear; class name matches the
        # allow-list so we exercise the skip branch without depending on
        # any particular transformers version.
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    FakeWrapper.__name__ = "Gemma4ClippableLinear"

    class Wrapped(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = FakeWrapper()

    model = Wrapped()
    with caplog.at_level("WARNING"):
        n = patch_custom_linears_for_lora(model)
    assert n == 0
    # Original module untouched.
    assert type(model.proj).__name__ == "Gemma4ClippableLinear"
    assert any("no nn.Linear descendant" in rec.message for rec in caplog.records)
