"""SPMD helpers: every rank runs the same loop on its own shard of the data.

The launcher (``torchrun`` via ``fai-rl-train --num-gpus N``) sets ``RANK`` /
``WORLD_SIZE``. ``TrainingClient.optim_step`` all-reduces gradients with SUM,
so the update equals the gradient of the loss summed over every rank's data.
"""

from __future__ import annotations

import os
from typing import List, Sequence, TypeVar

import torch
import torch.distributed as dist

T = TypeVar("T")


def world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return int(os.environ.get("WORLD_SIZE", 1))


def rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", 0))


def is_main_process() -> bool:
    return rank() == 0


def init_process_group_if_needed() -> None:
    """Join the torchrun process group when running with more than one rank."""
    if world_size() <= 1 or not dist.is_available() or dist.is_initialized():
        return
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def shard_for_rank(items: Sequence[T], group_size: int = 1) -> List[T]:
    """Return this rank's share of ``items``, keeping each group of
    ``group_size`` consecutive items (e.g. an AB/BA pair) on one rank.

    Groups are dealt round-robin, like ``DistributedSampler``.
    """
    if group_size < 1:
        raise ValueError("group_size must be >= 1")
    if len(items) % group_size:
        raise ValueError(
            f"len(items)={len(items)} is not a multiple of group_size={group_size}"
        )
    groups = [items[i:i + group_size] for i in range(0, len(items), group_size)]
    mine = groups[rank()::world_size()]
    return [item for group in mine for item in group]


def _device_for_collectives() -> torch.device:
    if dist.get_backend() == "nccl":
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def all_reduce_sum(value: float) -> float:
    """Sum a Python number across ranks (identity when not distributed)."""
    if not is_distributed():
        return float(value)
    tensor = torch.tensor([float(value)], dtype=torch.float64, device=_device_for_collectives())
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())
