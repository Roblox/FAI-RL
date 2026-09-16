import logging
from types import SimpleNamespace

from core.config import DataConfig, TrainingConfig
from trainers import grpo_trainer


def test_grpo_forwards_beta_to_trl_config():
    trainer = object.__new__(grpo_trainer.GRPOTrainer)
    trainer.logger = logging.getLogger("test_grpo_beta")
    trainer.config = SimpleNamespace(
        training=TrainingConfig(
            output_dir="out",
            beta=0.04,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=1,
            bf16=False,
            fp16=False,
        ),
        data=DataConfig(),
        wandb=SimpleNamespace(enabled=False),
    )

    args = trainer.setup_training_args()

    assert args.beta == 0.04
