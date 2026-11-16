import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from core.config import DataConfig, TrainingConfig
from trainers import grpo_trainer, gspo_trainer

ROLLOUT_RECIPES = sorted(
    recipe
    for algorithm in ("grpo", "grpo_vlm", "gspo")
    for recipe in (Path(__file__).resolve().parents[1] / "recipes" / "training" / algorithm).glob("*.yaml")
)


def _trainer(module, cls_name, **training):
    trainer = object.__new__(getattr(module, cls_name))
    trainer.logger = logging.getLogger(f"test_{cls_name}")
    trainer.config = SimpleNamespace(
        training=TrainingConfig(
            output_dir="out",
            per_device_train_batch_size=8,
            gradient_accumulation_steps=1,
            bf16=False,
            fp16=False,
            **training,
        ),
        data=DataConfig(),
        wandb=SimpleNamespace(enabled=False),
    )
    return trainer


def test_grpo_forwards_beta_to_trl_config():
    args = _trainer(grpo_trainer, "GRPOTrainer", beta=0.04).setup_training_args()

    assert args.beta == 0.04


def test_grpo_defaults_rollout_temperature_to_one():
    args = _trainer(grpo_trainer, "GRPOTrainer").setup_training_args()

    assert args.temperature == 1.0


def test_grpo_forwards_temperature_to_trl_config():
    args = _trainer(grpo_trainer, "GRPOTrainer", temperature=0.7).setup_training_args()

    assert args.temperature == 0.7


def test_gspo_forwards_temperature_to_trl_config():
    args = _trainer(
        gspo_trainer, "GSPOTrainer", temperature=1.2, steps_per_generation=1
    ).setup_training_args()

    assert args.temperature == 1.2


@pytest.mark.parametrize("temperature", [0.0, -0.5])
def test_training_config_rejects_non_positive_temperature(temperature):
    with pytest.raises(ValueError, match="temperature"):
        TrainingConfig(output_dir="out", temperature=temperature)


def test_rollout_recipes_exist():
    assert len(ROLLOUT_RECIPES) >= 10


@pytest.mark.parametrize("recipe", ROLLOUT_RECIPES, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_rollout_recipes_set_temperature(recipe):
    training = yaml.safe_load(recipe.read_text())["training"]

    assert "temperature" in training
    assert TrainingConfig(**training).temperature == 1.0
