"""Trainer implementations."""

from .cpt_trainer import CPTTrainer
from .dpo_trainer import DPOTrainer
from .grpo_trainer import GRPOTrainer
from .gspo_trainer import GSPOTrainer
from .sft_trainer import SFTTrainer

# PPO is imported lazily because the legacy trl PPOConfig/PPOTrainer API was
# removed in trl >= 1.0. Importing it eagerly would break every other algorithm
# (CPT, SFT, DPO, GRPO, GSPO) on modern trl. We only fail when PPOTrainer is
# actually accessed.
try:
    from .ppo_trainer import PPOTrainer  # noqa: F401
except ImportError as _ppo_import_error:
    _PPO_IMPORT_ERROR = _ppo_import_error

    class PPOTrainer:  # type: ignore[no-redef]
        """Placeholder raised only if PPO is actually used on an unsupported trl."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PPOTrainer is unavailable because the installed `trl` version "
                "removed the legacy PPOConfig/PPOTrainer API. Pin `trl<=0.23` "
                "or migrate FAI-RL's PPOTrainer to the new trl API. "
                f"Underlying error: {_PPO_IMPORT_ERROR}"
            )

__all__ = [
    "CPTTrainer",
    "DPOTrainer",
    "GRPOTrainer",
    "GSPOTrainer",
    "PPOTrainer",
    "SFTTrainer",
]

