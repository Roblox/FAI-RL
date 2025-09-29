"""
RL Fine-tuning Library

A modular library for reinforcement learning fine-tuning of language models.
"""

__version__ = "0.0.1"

from .core.config import ExperimentConfig, ModelConfig, DataConfig, TrainingConfig, WandbConfig
from .trainers.dpo_trainer import DPOTrainer

__all__ = [
    "ExperimentConfig",
    "ModelConfig",
    "DataConfig",
    "TrainingConfig",
    "WandbConfig",
    "DPOTrainer",
]
