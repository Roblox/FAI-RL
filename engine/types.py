"""Data types for FAI-RL training primitives.

A ``Datum`` is one self-contained training example: the tokens the model reads
plus every per-token tensor its loss needs. Position ``i`` of ``model_input``
predicts ``target_tokens[i]`` (the full sequence shifted by one).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Sequence

import torch

# Per-token keys whose first dimension must equal ``model_input.length``. Other
# keys are free-form and passed to custom losses untouched (as tensors).
_TOKEN_ALIGNED_KEYS = {"target_tokens", "weights", "logprobs", "advantages"}
_FLOAT_KEYS = {"weights", "logprobs", "advantages"}


@dataclass(frozen=True)
class ModelInput:
    """Token ids fed to the model (text only in Phase 1)."""

    tokens: tuple

    @classmethod
    def from_ints(cls, tokens: Sequence[int]) -> "ModelInput":
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        return cls(tokens=tuple(int(token) for token in tokens))

    def to_ints(self) -> List[int]:
        return list(self.tokens)

    @property
    def length(self) -> int:
        return len(self.tokens)


def _as_tensor(key: str, value: Any) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if key == "target_tokens":
        return tensor.long()
    if key in _FLOAT_KEYS:
        return tensor.float()
    return tensor


@dataclass
class Datum:
    """One training example: model input plus the tensors its loss consumes.

    ``loss_fn_inputs`` always holds ``target_tokens`` with shape ``(N,)`` or
    ``(N, K)`` (``K`` candidate targets per position), plus whatever the loss
    reads, e.g. ``weights`` for ``cross_entropy``.
    """

    model_input: ModelInput
    loss_fn_inputs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if "target_tokens" not in self.loss_fn_inputs:
            raise ValueError("Datum.loss_fn_inputs must include 'target_tokens'")
        length = self.model_input.length
        if length == 0:
            raise ValueError("Datum.model_input must contain at least one token")
        converted = {}
        for key, value in self.loss_fn_inputs.items():
            tensor = _as_tensor(key, value)
            if key in _TOKEN_ALIGNED_KEYS:
                if tensor.dim() == 0 or tensor.shape[0] != length:
                    raise ValueError(
                        f"loss_fn_inputs[{key!r}] must have first dimension "
                        f"{length} (model_input.length), got shape {tuple(tensor.shape)}"
                    )
            converted[key] = tensor
        target = converted["target_tokens"]
        if target.dim() not in (1, 2):
            raise ValueError(
                "loss_fn_inputs['target_tokens'] must have shape (N,) or (N, K), "
                f"got {tuple(target.shape)}"
            )
        self.loss_fn_inputs = converted


@dataclass
class AdamParams:
    """AdamW hyperparameters for one ``optim_step``.

    Defaults follow HF ``TrainingArguments`` so native and TRL runs line up.
    ``grad_clip_norm <= 0`` disables clipping.
    """

    learning_rate: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0
    grad_clip_norm: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class ForwardBackwardOutput:
    """Result of ``forward`` / ``forward_backward`` / ``forward_backward_custom``.

    ``loss_fn_outputs[i]["logprobs"]`` holds the target-token log-probs of
    ``data[i]`` (detached), shaped like its ``target_tokens``.
    """

    loss_fn_outputs: List[Dict[str, torch.Tensor]] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class OptimStepOutput:
    metrics: Dict[str, float] = field(default_factory=dict)
