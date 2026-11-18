"""FAI-RL training primitives for writing your own training loop.

Write the training loop yourself; the client handles the model::

    import engine
    from engine import types

    client = engine.ServiceClient().create_lora_training_client("Qwen/Qwen3-4B", rank=16)
    out = client.forward_backward(data, loss_fn="cross_entropy")
    client.optim_step(types.AdamParams(learning_rate=1e-4))
"""

from . import losses, types
from .distributed import all_reduce_sum, is_main_process, rank, shard_for_rank, world_size
from .losses import available_losses, register_loss
from .service_client import ServiceClient
from .training_client import TrainingClient
from .types import AdamParams, Datum, ForwardBackwardOutput, ModelInput, OptimStepOutput

__all__ = [
    "AdamParams",
    "Datum",
    "ForwardBackwardOutput",
    "ModelInput",
    "OptimStepOutput",
    "ServiceClient",
    "TrainingClient",
    "all_reduce_sum",
    "available_losses",
    "is_main_process",
    "losses",
    "rank",
    "register_loss",
    "shard_for_rank",
    "types",
    "world_size",
]
