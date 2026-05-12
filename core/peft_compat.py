"""Compatibility shims so PEFT/LoRA can wrap custom Linear-like modules.

Background
----------
PEFT 0.19's LoRA dispatcher only accepts targets that are an exact instance
(or subclass) of ``torch.nn.Linear`` (plus a handful of other known types
like ``Conv1D``, ``Embedding``, bitsandbytes 4/8-bit linears, etc.). Some
upstream architectures wrap an internal ``nn.Linear`` inside an
``nn.Module`` subclass that adds extra behavior — most notably
``transformers.models.gemma4.modeling_gemma4.Gemma4ClippableLinear``, which
clamps its output for training stability. PEFT does not recognize that
class and raises ``ValueError: Target module Gemma4ClippableLinear(...) is
not supported``.

Strategy
--------
For each custom-linear instance, swap it out for ``_LinearShim`` — a thin
``nn.Linear`` subclass that

1. shares parameters (``weight``/``bias``) with the wrapped module's inner
   ``nn.Linear`` so PEFT sees real tensors with the correct
   ``in_features``/``out_features``/dtype, and
2. delegates ``forward`` back to the original module, so any extra
   behavior (output clipping for Gemma 4, etc.) is preserved end-to-end.
   After PEFT wraps the shim as ``lora.Linear.base_layer``, ``base_layer
   .forward`` is still invoked on every step, so the clamping survives
   LoRA injection.

Caveats
-------
- ``lora.Linear.merge_and_unload`` uses ``base_layer.weight`` directly
  (not ``base_layer.forward``), so a merged checkpoint loses the clamp.
  That only matters at export time; training/eval through the adapter
  preserves the original behavior.
- If a wrapper class doesn't expose an inner ``nn.Linear`` we can locate,
  we leave it alone and let PEFT raise its standard error. We don't try
  to silently no-op the LoRA path.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional

import torch
import torch.nn as nn


class _LinearShim(nn.Linear):
    """``nn.Linear``-typed wrapper that delegates forward to a custom module.

    Constructed on the meta device so ``super().__init__`` does not allocate
    fresh weight memory; the meta parameters are then immediately replaced
    with the inner linear's real parameters (weight sharing, no copy).
    """

    def __init__(self, original: nn.Module, inner_linear: nn.Linear):
        super().__init__(
            inner_linear.in_features,
            inner_linear.out_features,
            bias=inner_linear.bias is not None,
            device="meta",
        )
        # Share parameters with the inner linear. Optimizer/grad updates
        # therefore reach the same tensor `original` uses on forward.
        self.weight = inner_linear.weight
        if inner_linear.bias is not None:
            self.bias = inner_linear.bias
        # Register the original module as a child so its buffers/params
        # (e.g. clipping bounds) follow ``.to()`` and remain in the state
        # dict. Use a name unlikely to collide with PEFT's target regex.
        self._fai_inner = original

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._fai_inner(x)

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return f"{super().extra_repr()}, wraps={type(self._fai_inner).__name__}"


_DEFAULT_PATCH_CLASSES: tuple[str, ...] = ("Gemma4ClippableLinear",)


def _find_inner_linear(module: nn.Module) -> Optional[nn.Linear]:
    """Return the first ``nn.Linear`` descendant of ``module``, or ``None``.

    Direct children are checked first (the common case for wrappers that
    hold a single inner projection), then a depth-first walk.
    """
    for child in module.children():
        if isinstance(child, nn.Linear) and not isinstance(child, _LinearShim):
            return child
    for sub in module.modules():
        if sub is module:
            continue
        if isinstance(sub, nn.Linear) and not isinstance(sub, _LinearShim):
            return sub
    return None


def patch_custom_linears_for_lora(
    model: nn.Module,
    class_names: Iterable[str] = _DEFAULT_PATCH_CLASSES,
    logger: Optional[logging.Logger] = None,
) -> int:
    """Replace recognized custom Linear wrappers with ``_LinearShim``s.

    Args:
        model: The model to mutate in place.
        class_names: Class names (by ``type(m).__name__``) to consider for
            replacement. Defaults to Gemma 4's clippable linear.
        logger: Optional logger; falls back to the module-level logger.

    Returns:
        Number of modules successfully replaced.
    """
    log = logger or logging.getLogger(__name__)
    targets = tuple(class_names)
    if not targets:
        return 0

    # Collect first, mutate second — avoid invalidating the named_modules
    # iterator while we replace attributes on parent modules.
    to_patch: list[tuple[nn.Module, str, nn.Module]] = []
    for name, sub in model.named_modules():
        if type(sub).__name__ not in targets:
            continue
        if "." in name:
            parent_name, attr = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
        else:
            parent, attr = model, name
        to_patch.append((parent, attr, sub))

    if not to_patch:
        return 0

    patched = 0
    skipped = 0
    for parent, attr, original in to_patch:
        inner = _find_inner_linear(original)
        if inner is None:
            log.warning(
                "peft_compat: %s at '%s' has no nn.Linear descendant; "
                "leaving in place (PEFT will likely reject it).",
                type(original).__name__,
                attr,
            )
            skipped += 1
            continue
        setattr(parent, attr, _LinearShim(original, inner))
        patched += 1

    log.info(
        "peft_compat: replaced %d %s module(s) with nn.Linear shim "
        "for PEFT/LoRA compatibility%s",
        patched,
        "/".join(targets),
        f" (skipped {skipped} without inner Linear)" if skipped else "",
    )
    return patched
