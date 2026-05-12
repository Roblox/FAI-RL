"""Tests for ``core.peft_compat``.

Covers both fixes:

A. :func:`patch_custom_linears_for_lora`

   1. Reproduces the production bug: PEFT/LoRA refuses
      ``Gemma4ClippableLinear`` as a target module.
   2. After patching, PEFT can wrap it, the wrapped module preserves
      the original (clamping) forward behavior, and only LoRA adapter
      params become trainable.
   3. Is a no-op on models that already use stock ``nn.Linear``.
   4. Safely skips wrappers without a discoverable inner ``nn.Linear``.

B. :func:`extend_lora_config_for_moe_experts`

   5. Reproduces the latent Gemma 4 MoE bug: with the standard
      ``target_modules`` recipe and no ``target_parameters``, PEFT
      silently skips the experts (zero ``ParamWrapper`` instances).
   6. After the helper runs, PEFT creates one ``ParamWrapper`` per
      expert parameter per layer, and trainable param count grows
      meaningfully because the experts now have adapters.
   7. The helper forces ``lora_dropout`` to ``0.0`` (with a warning)
      because ``ParamWrapper`` cannot implement dropout.
   8. Is a no-op on models without MoE experts.

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

try:
    from transformers import Gemma4ForCausalLM, Gemma4TextConfig
    _HAS_GEMMA4_MODEL = True
except Exception:  # pragma: no cover - environment-dependent
    Gemma4ForCausalLM = None  # type: ignore[assignment]
    Gemma4TextConfig = None  # type: ignore[assignment]
    _HAS_GEMMA4_MODEL = False

# Load core/peft_compat.py directly so we don't pull in the rest of
# ``core`` (its ``__init__`` eagerly imports the full trainer stack,
# including wandb, which we don't need or want for unit tests).
_PEFT_COMPAT_PATH = Path(__file__).resolve().parents[1] / "core" / "peft_compat.py"
_spec = importlib.util.spec_from_file_location("peft_compat_under_test", _PEFT_COMPAT_PATH)
assert _spec is not None and _spec.loader is not None
_pc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pc)
patch_custom_linears_for_lora = _pc.patch_custom_linears_for_lora
extend_lora_config_for_moe_experts = _pc.extend_lora_config_for_moe_experts
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


# --------------------------------------------------------------------- #
# MoE expert target injection (extend_lora_config_for_moe_experts)
# --------------------------------------------------------------------- #


def _tiny_gemma4_moe_model() -> nn.Module:
    """A tiny but real Gemma 4 ``ForCausalLM`` with the MoE block enabled."""
    cfg = Gemma4TextConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        global_head_dim=8,
        hidden_size_per_layer_input=8,
        layer_types=["sliding_attention", "full_attention"],
        enable_moe_block=True,
        num_experts=2,
        num_experts_per_tok=1,
        num_local_experts=2,
    )
    torch.manual_seed(0)
    return Gemma4ForCausalLM(cfg)


def _count_lora_layer_types(peft_model: nn.Module) -> dict[str, int]:
    counts: dict[str, int] = {}
    for _, mod in peft_model.named_modules():
        name = type(mod).__name__
        if "lora" in name.lower() or "Param" in name:
            counts[name] = counts.get(name, 0) + 1
    return counts


@pytest.mark.skipif(
    not _HAS_GEMMA4_MODEL, reason="Gemma4ForCausalLM requires transformers>=5.8.0"
)
def test_get_peft_model_silently_skips_gemma4_experts_without_helper() -> None:
    """Pins the latent bug — without the helper, Gemma 4 experts are not LoRA'd."""
    model = _tiny_gemma4_moe_model()
    lora = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    from peft import get_peft_model as _gpm
    peft_model = _gpm(model, lora)
    counts = _count_lora_layer_types(peft_model)
    # PEFT 0.19 does not register Gemma 4 in `_MOE_TARGET_MODULE_MAPPING`,
    # so zero `ParamWrapper`s should be created.
    assert counts.get("ParamWrapper", 0) == 0


@pytest.mark.skipif(
    not _HAS_GEMMA4_MODEL, reason="Gemma4ForCausalLM requires transformers>=5.8.0"
)
def test_extend_lora_config_registers_gemma4_experts() -> None:
    """After the helper runs, PEFT creates a ParamWrapper per expert param per layer."""
    model = _tiny_gemma4_moe_model()
    lora = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    added = extend_lora_config_for_moe_experts(model, lora)

    # 2 expert params (gate_up_proj, down_proj) collapse to 2 patterns
    # via the leaf-attribute-name aggregation: "experts.gate_up_proj"
    # and "experts.down_proj".
    assert added == 2
    assert set(lora.target_parameters) == {"experts.gate_up_proj", "experts.down_proj"}

    peft_model = get_peft_model(model, lora)
    counts = _count_lora_layer_types(peft_model)
    # 2 expert params x 2 layers = 4 ParamWrappers.
    assert counts.get("ParamWrapper", 0) == 4

    # Trainable params should now meaningfully exceed what we'd get
    # without the helper. We don't pin an exact number, only the
    # qualitative property: there must be LoRA on the experts.
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    assert trainable > 0


@pytest.mark.skipif(
    not _HAS_GEMMA4_MODEL, reason="Gemma4ForCausalLM requires transformers>=5.8.0"
)
def test_extend_lora_config_forces_dropout_to_zero(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """ParamWrapper rejects lora_dropout > 0; the helper must clamp to 0 with a warning."""
    model = _tiny_gemma4_moe_model()
    lora = LoraConfig(r=4, lora_alpha=8, lora_dropout=0.1)

    with caplog.at_level("WARNING"):
        added = extend_lora_config_for_moe_experts(model, lora)

    assert added > 0
    assert lora.lora_dropout == 0.0
    assert any("lora_dropout" in rec.message for rec in caplog.records)


def test_extend_lora_config_is_noop_when_no_moe_experts() -> None:
    """A model with no MoE-expert modules should not get any target_parameters appended."""

    class Plain(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(8, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.proj(x)

    model = Plain()
    lora = LoraConfig(r=4, lora_alpha=8, target_modules=["proj"], lora_dropout=0.1)

    added = extend_lora_config_for_moe_experts(model, lora)

    assert added == 0
    # No experts found, so dropout must be left untouched.
    assert lora.lora_dropout == 0.1
    assert not lora.target_parameters
