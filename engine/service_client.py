"""Factory for ``TrainingClient``."""

from __future__ import annotations

import os
from typing import List, Optional, Union

from core.config import DataConfig, ExperimentConfig, ModelConfig, TrainingConfig, WandbConfig
from utils.logging_utils import setup_logging

from .training_client import TrainingClient


class ServiceClient:
    """Create training clients for a base model, a full config, or a saved state."""

    def __init__(self, output_dir: str = "outputs/engine", logger=None):
        self.output_dir = output_dir
        self.logger = logger or setup_logging("engine", file_output=False)

    def create_lora_training_client(
        self,
        base_model: str,
        rank: int = 32,
        *,
        lora_alpha: Optional[int] = None,
        target_modules: Union[str, List[str]] = "all-linear",
        training_config: Optional[TrainingConfig] = None,
        **model_overrides,
    ) -> TrainingClient:
        """LoRA on ``base_model`` (all linear layers by default).

        ``model_overrides`` are any other ``ModelConfig`` fields, e.g.
        ``load_in_4bit=True`` for QLoRA or ``torch_dtype="float32"``.
        """
        model_config = ModelConfig(
            base_model_name=base_model,
            use_lora=True,
            lora_r=rank,
            lora_alpha=lora_alpha if lora_alpha is not None else 2 * rank,
            lora_target_modules=target_modules,
            **model_overrides,
        )
        return self.create_training_client(model_config, training_config)

    def create_training_client(
        self,
        model_config: ModelConfig,
        training_config: Optional[TrainingConfig] = None,
    ) -> TrainingClient:
        """Full control over loading (LoRA/QLoRA/full fine-tuning) via ``ModelConfig``.

        ``training_config`` supplies ``output_dir``, ``micro_batch_tokens``,
        ``gradient_checkpointing`` and ``bf16``; learning rate and schedule
        are up to the caller's loop.
        """
        from .model_loading import build_training_client

        training = training_config or TrainingConfig(
            output_dir=self.output_dir,
            eval_enabled=False,
            early_stopping=False,
            gradient_checkpointing=False,
        )
        config = ExperimentConfig(
            model=model_config,
            data=DataConfig(),
            training=training,
            wandb=WandbConfig(enabled=False),
        )
        return build_training_client(config, logger=self.logger)

    def create_training_client_from_state(
        self,
        path: str,
        *,
        with_optimizer: bool = False,
        training_config: Optional[TrainingConfig] = None,
        **model_overrides,
    ) -> TrainingClient:
        """Resume from ``save_state`` output (PEFT adapter dir or full model dir)."""
        is_adapter = os.path.exists(os.path.join(path, "adapter_config.json"))
        model_config = ModelConfig(base_model_name=path, use_lora=is_adapter, **model_overrides)
        client = self.create_training_client(model_config, training_config)
        client._load_training_state(path, with_optimizer=with_optimizer)
        return client

    def create_training_client_from_state_with_optimizer(self, path: str, **kwargs) -> TrainingClient:
        return self.create_training_client_from_state(path, with_optimizer=True, **kwargs)
