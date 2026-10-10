"""Core modules for RL fine-tuning."""

from .config import ExperimentConfig, ModelConfig, DataConfig, TrainingConfig, WandbConfig


def __getattr__(name: str):
    """Load ML helpers lazily so configuration validation works on CPU-only installs."""
    from importlib import import_module

    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = ".trainer_base" if name == "BaseTrainer" else ".model_utils"
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value

__all__ = [
    "ExperimentConfig",
    "ModelConfig",
    "DataConfig",
    "TrainingConfig",
    "WandbConfig",
    "BaseTrainer",
    "load_model_and_tokenizer",
    "get_model_memory_usage",
    "count_trainable_parameters",
]
