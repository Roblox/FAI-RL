"""Loss functions for ``TrainingClient.forward_backward``.

Every loss, built-in or custom, has the same contract::

    loss_fn(data: list[Datum], logprobs: list[Tensor]) -> (scalar_loss, metrics)

where ``logprobs[i]`` are the model's log-probs of ``data[i]``'s
``target_tokens`` (shape ``(N,)`` or ``(N, K)``). Losses sum over tokens and
datums; scale via ``weights`` to get a mean.

Named losses are registered with ``register_loss``. ``separable=True`` declares
that the loss (and every metric) is a sum of independent per-datum terms, which
lets the client run it micro-batch by micro-batch in a single pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import torch

from .types import Datum

LossFn = Callable[[List[Datum], List[torch.Tensor]], Tuple[torch.Tensor, Dict[str, float]]]


@dataclass(frozen=True)
class RegisteredLoss:
    fn: LossFn
    separable: bool


_LOSSES: Dict[str, RegisteredLoss] = {}


def register_loss(name: str, *, separable: bool = False):
    """Register ``fn`` under ``name`` so it can be passed to ``forward_backward``."""

    def decorator(fn: LossFn) -> LossFn:
        if name in _LOSSES:
            raise ValueError(f"Loss {name!r} is already registered")
        _LOSSES[name] = RegisteredLoss(fn=fn, separable=separable)
        return fn

    return decorator


def get_loss(name: str) -> RegisteredLoss:
    try:
        return _LOSSES[name]
    except KeyError:
        raise ValueError(
            f"Unknown loss_fn {name!r}; available: {available_losses()}. "
            "Use forward_backward_custom for an arbitrary callable."
        ) from None


def available_losses() -> List[str]:
    return sorted(_LOSSES)


@register_loss("cross_entropy", separable=True)
def cross_entropy(data: List[Datum], logprobs: List[torch.Tensor]):
    """``sum(-weights * logprobs)``; ``weights`` default to 1 for every token."""
    total = None
    for datum, lp in zip(data, logprobs):
        if lp.dim() != 1:
            raise ValueError(
                "cross_entropy needs target_tokens of shape (N,); use "
                "forward_backward_custom for multi-target (N, K) losses"
            )
        weights = datum.loss_fn_inputs.get("weights")
        weights = torch.ones_like(lp) if weights is None else weights.to(lp.device, lp.dtype)
        term = -(weights * lp).sum()
        total = term if total is None else total + term
    if total is None:
        raise ValueError("cross_entropy needs at least one datum")
    return total, {"loss:sum": float(total.detach())}
