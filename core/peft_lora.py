"""PEFT LoRA helpers for module types stock PEFT cannot wrap.

Gemma 4 E-series language stacks and Gemma 4 vision/audio towers use
``Gemma4ClippableLinear``: an ``nn.Module`` wrapping an inner ``nn.Linear``
(or bitsandbytes ``Linear4bit`` / ``Linear8bitLt``) with optional activation
clipping. PEFT LoRA only dispatches to ``nn.Linear`` and a few other stock
types, so ``target_modules`` like ``q_proj`` raise:

    Target module Gemma4ClippableLinear(...) is not supported.

PEFT 0.19 exposes experimental ``LoraConfig._register_custom_module``. We
register a factory that LoRA-wraps the inner ``.linear`` via PEFT's normal
dispatch (so QLoRA 4-bit/8-bit backends still run) and re-applies clipping
around that adapter. The outer wrapper is not replaced with a bare
``nn.Linear``. Custom modules are not serialized in adapter configs; re-register
before ``PeftModel.from_pretrained``.
"""

from __future__ import annotations

import types
from typing import Any, Optional

import torch
import torch.nn as nn

_CLIP_BUFFERS = ("input_min", "input_max", "output_min", "output_max")


def _gemma4_clippable_linear_cls() -> Optional[type]:
    try:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4ClippableLinear
    except ImportError:
        return None
    return Gemma4ClippableLinear


def _copy_clip_buffers(lora_layer: nn.Module, original: nn.Module) -> None:
    for name in _CLIP_BUFFERS:
        buf = getattr(original, name, None)
        if buf is None:
            continue
        tensor = buf.detach().clone() if torch.is_tensor(buf) else buf
        if name in lora_layer._buffers:
            del lora_layer._buffers[name]
        lora_layer.register_buffer(name, tensor)


def _install_clipping(lora_layer: nn.Module, original: nn.Module) -> nn.Module:
    """Keep Gemma4 clip semantics on the LoRA-wrapped inner linear."""
    use_clip = bool(getattr(original, "use_clipped_linears", False))
    lora_layer.use_clipped_linears = use_clip
    if not use_clip:
        return lora_layer

    _copy_clip_buffers(lora_layer, original)
    orig_forward = lora_layer.forward

    def clipped_forward(self, x, *args, **kwargs):
        x = torch.clamp(x, self.input_min, self.input_max)
        out = orig_forward(x, *args, **kwargs)
        return torch.clamp(out, self.output_min, self.output_max)

    lora_layer.forward = types.MethodType(clipped_forward, lora_layer)
    return lora_layer


def _lora_wrap_clippable_linear(base_layer, adapter_name, config, **kwargs):
    """Custom LoRA factory: adapt ``base_layer.linear``, keep clipping.

    Clears ``config._custom_modules`` while dispatching the inner layer so PEFT
    uses stock Linear / bitsandbytes QLoRA dispatchers instead of recursing.
    """
    inner = getattr(base_layer, "linear", None)
    if inner is None:
        raise TypeError(f"{type(base_layer).__name__} has no inner .linear to adapt with LoRA")

    from peft.tuners.lora.model import LoraModel

    saved = getattr(config, "_custom_modules", None)
    config._custom_modules = None
    try:
        lora_layer = LoraModel._create_new_module(config, adapter_name, inner, **kwargs)
    finally:
        config._custom_modules = saved

    return _install_clipping(lora_layer, base_layer)


def register_clippable_linear_lora(peft_config) -> Any:
    """Register Gemma4ClippableLinear on a PEFT config if the class exists.

    No-op when transformers has no Gemma 4 module or the config has no
    ``_register_custom_module`` (older peft). Call again after
    ``LoraConfig.from_pretrained`` because custom modules are not serialized.
    """
    register = getattr(peft_config, "_register_custom_module", None)
    if register is None:
        return peft_config
    cls = _gemma4_clippable_linear_cls()
    if cls is None:
        return peft_config
    register({cls: _lora_wrap_clippable_linear})
    return peft_config


def peft_model_from_pretrained(model, model_id, **kwargs):
    """``PeftModel.from_pretrained`` with Gemma4ClippableLinear re-registered."""
    from peft import PeftConfig, PeftModel

    config = kwargs.pop("config", None)
    if config is None:
        config = PeftConfig.from_pretrained(model_id)
    register_clippable_linear_lora(config)
    return PeftModel.from_pretrained(model, model_id, config=config, **kwargs)
