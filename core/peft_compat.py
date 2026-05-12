"""Compatibility helpers so PEFT/LoRA can target Gemma 4 (dense + MoE).

This module bundles two independent fixes that together make
``transformers.models.gemma4`` LoRA-tunable in both dense and MoE flavors.
Each helper is opt-in and a no-op on models that don't need it, so it's
safe to call them unconditionally from the training pipeline.

1. ``patch_custom_linears_for_lora`` — custom-Linear wrapper shim
----------------------------------------------------------------
PEFT's LoRA dispatcher only accepts targets that are an exact instance
(or subclass) of ``torch.nn.Linear`` plus a handful of other known types
(``Conv1D``, ``Embedding``, bitsandbytes 4/8-bit linears, ...). Some
upstream architectures wrap an internal ``nn.Linear`` inside an
``nn.Module`` subclass that adds extra behavior — most notably
``Gemma4ClippableLinear``, which clamps its output for training
stability. PEFT does not recognize that class and raises
``ValueError: Target module Gemma4ClippableLinear(...) is not supported``.

Fix: swap each instance for ``_LinearShim`` — a thin ``nn.Linear``
subclass that (a) shares ``weight``/``bias`` with the wrapped module's
inner ``nn.Linear`` so PEFT sees real tensors with the correct
``in_features``/``out_features``/dtype, and (b) delegates ``forward``
back to the original module so the clamp (or any other custom logic)
survives LoRA injection — PEFT calls ``base_layer.forward`` each step.

Caveat: ``lora.Linear.merge_and_unload`` reads ``base_layer.weight``
directly (not ``base_layer.forward``), so a *merged* checkpoint loses
the clamp. That only matters at export time; training/eval through the
adapter preserves the original behavior.

2. ``extend_lora_config_for_moe_experts`` — MoE expert target injection
-----------------------------------------------------------------------
Modern MoE layers (Gemma 4, Qwen3-MoE, Mixtral, ...) store experts as
stacked 3D ``nn.Parameter`` tensors instead of per-expert ``nn.Linear``
modules. PEFT ships a dedicated ``ParamWrapper`` LoRA layer for this,
but it's only dispatched when ``LoraConfig.target_parameters`` includes
the expert parameter path — either set explicitly or auto-injected by
PEFT's ``_MOE_TARGET_MODULE_MAPPING`` registry.

That registry (in ``peft.utils.transformers_weight_conversion``) covers
Mixtral / Qwen2-MoE / Qwen3-MoE but, as of PEFT 0.19, **not** Gemma 4.
Without this helper, running the standard ``target_modules=[gate_proj,
up_proj, down_proj]`` recipe against a Gemma 4 MoE model silently
skips the experts — training "succeeds" but most of the model's
capacity is never updated.

Fix: walk the model, find every ``Gemma4TextExperts``-shaped module,
and append its raw expert parameters (``experts.gate_up_proj``,
``experts.down_proj``) to ``LoraConfig.target_parameters`` so PEFT's
``ParamWrapper`` kicks in. Also force ``lora_dropout=0.0`` because
``ParamWrapper`` cannot implement dropout — the math
``lora_B(lora_A(dropout(x)))`` does not factor out ``x``.

Scope notes
-----------
- These helpers target ``Gemma4ClippableLinear`` (Linear-shaped wrapper)
  and ``Gemma4TextExperts`` (Linear-shaped raw-parameter MoE) by default.
  Both class names are extensible via parameters on each helper.
- Models that are already in PEFT's MoE registry (Qwen3-MoE, Mixtral)
  are unaffected: ``patch_custom_linears_for_lora`` is a no-op when
  there are no custom wrappers, and the MoE helper's default class
  allow-list does not include ``Qwen3MoeExperts`` so we don't duplicate
  work PEFT is already doing.
- Genuinely non-Linear ops (fused QKV kernels, FP8 quant linears,
  etc.) are out of scope and need PEFT's own custom-modules API.
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


_DEFAULT_MOE_EXPERT_CLASSES: tuple[str, ...] = ("Gemma4TextExperts",)


def extend_lora_config_for_moe_experts(
    model: nn.Module,
    lora_config: object,
    module_class_names: Iterable[str] = _DEFAULT_MOE_EXPERT_CLASSES,
    logger: Optional[logging.Logger] = None,
) -> int:
    """Append MoE expert parameter paths to ``lora_config.target_parameters``.

    PEFT's ``ParamWrapper`` LoRA layer is what makes raw-``nn.Parameter``
    MoE experts trainable, but it only fires when the expert parameter
    path appears in ``LoraConfig.target_parameters``. PEFT auto-injects
    those paths for some architectures (Mixtral, Qwen2-MoE, Qwen3-MoE)
    via an internal registry — but **not** for Gemma 4 in PEFT 0.19.
    Without this call, Gemma 4 MoE LoRA training silently leaves the
    experts frozen.

    This function walks ``model``, finds every submodule whose class
    name is in ``module_class_names``, and registers each of its
    directly-owned ``nn.Parameter`` attributes (e.g. ``gate_up_proj``,
    ``down_proj``) onto ``lora_config.target_parameters`` using a
    "leaf-attribute.parameter" pattern (e.g. ``experts.gate_up_proj``),
    which matches every layer's experts at once.

    Side effect: if any expert paths are added and
    ``lora_config.lora_dropout > 0``, the dropout is forced to ``0.0``
    with a warning, because PEFT's ``ParamWrapper`` raises if
    ``lora_dropout != 0`` (the LoRA math ``lora_B(lora_A(dropout(x)))``
    cannot factor out ``x``).

    Args:
        model: The model to inspect (not mutated).
        lora_config: A ``peft.LoraConfig`` instance to update in place.
        module_class_names: Module class names (by
            ``type(m).__name__``) considered MoE-expert modules to
            register. Defaults to ``Gemma4TextExperts`` only —
            Qwen3-MoE / Mixtral are handled by PEFT's built-in
            registry, so listing them here would duplicate work.
        logger: Optional logger; falls back to the module-level logger.

    Returns:
        Number of new ``target_parameters`` entries appended.
    """
    log = logger or logging.getLogger(__name__)
    targets = tuple(module_class_names)
    if not targets:
        return 0

    # Discover patterns like "experts.gate_up_proj" once per (leaf-attr,
    # param-name) pair. We use the leaf attribute name (last component
    # of the module's qualified path) so the pattern matches every
    # layer's experts simultaneously, regardless of layer count.
    new_patterns: set[str] = set()
    for qualified_name, sub in model.named_modules():
        if type(sub).__name__ not in targets:
            continue
        if not qualified_name:
            # An MoE-experts class sitting at the model root would be
            # extremely unusual; skip rather than guess at a pattern.
            continue
        leaf = qualified_name.rsplit(".", 1)[-1]
        for pname, _ in sub.named_parameters(recurse=False):
            new_patterns.add(f"{leaf}.{pname}")

    if not new_patterns:
        return 0

    existing_raw = getattr(lora_config, "target_parameters", None)
    if existing_raw is None:
        existing: list[str] = []
    elif isinstance(existing_raw, str):
        existing = [existing_raw]
    else:
        existing = list(existing_raw)

    added = sorted(new_patterns - set(existing))
    if not added:
        return 0

    lora_config.target_parameters = existing + added

    # ParamWrapper does not support lora_dropout != 0.
    dropout = getattr(lora_config, "lora_dropout", 0.0) or 0.0
    if dropout > 0.0:
        log.warning(
            "peft_compat: forcing lora_dropout %.4f -> 0.0 because PEFT's "
            "ParamWrapper (used for MoE experts: %s) does not support "
            "dropout.",
            dropout,
            ", ".join(added),
        )
        lora_config.lora_dropout = 0.0

    log.info(
        "peft_compat: registered %d MoE expert parameter pattern(s) "
        "for LoRA via target_parameters: %s",
        len(added),
        ", ".join(added),
    )
    return len(added)
