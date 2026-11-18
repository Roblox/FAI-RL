"""Load models for ``TrainingClient`` with the same code path as the TRL trainers."""

from __future__ import annotations

from peft import TaskType

from core.config import ExperimentConfig
from core.trainer_base import BaseTrainer

from .training_client import TrainingClient


class _ModelLoader(BaseTrainer):
    """``BaseTrainer`` used only for its model/tokenizer/LoRA/S3/wandb setup."""

    def setup_model(self):
        quantization_config = self.create_quantization_config()
        model = self.load_base_model_for_training(self.prepare_model_kwargs(quantization_config))
        tokenizer = self.setup_tokenizer_with_model(model)
        self.prepare_model_for_modalities(model, use_vision=False, use_audio=False)
        model = self.apply_lora_to_model(model, TaskType.CAUSAL_LM, quantization_config)
        self.disable_cache_for_gradient_checkpointing(model)
        return model, tokenizer

    def setup_data(self):
        raise NotImplementedError("engine loops prepare their own data")

    def setup_trainer(self):
        raise NotImplementedError("engine loops drive TrainingClient directly")

    def train(self):
        raise NotImplementedError("engine loops drive TrainingClient directly")


def build_training_client(config: ExperimentConfig, logger=None) -> TrainingClient:
    """Load ``config.model`` and wrap it in a ``TrainingClient``.

    Wandb is initialized here when ``config.wandb.enabled`` (main process) and
    finished by ``TrainingClient.close()``.
    """
    loader = _ModelLoader(config, logger=logger)
    model, tokenizer = loader.setup_model()
    training = loader.config.training
    client = TrainingClient(
        model,
        tokenizer,
        micro_batch_tokens=training.micro_batch_tokens,
        gradient_checkpointing=training.gradient_checkpointing,
        bf16=training.bf16,
        output_dir=training.output_dir,
        logger=loader.logger,
    )
    client.add_close_callback(loader.cleanup_wandb)
    return client
