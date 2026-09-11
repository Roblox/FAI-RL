"""Tests for early-stopping helpers (no model load)."""

import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.early_stopping import hold_out_eval_dataset, training_args_for_early_stopping


class _FakeDataset:
    def __init__(self, n: int):
        self._n = n
        self.last_kwargs = None

    def __len__(self):
        return self._n

    def train_test_split(self, test_size, seed, shuffle):
        self.last_kwargs = {"test_size": test_size, "seed": seed, "shuffle": shuffle}
        eval_n = test_size if isinstance(test_size, int) else max(1, round(self._n * test_size))
        return {
            "train": _FakeDataset(self._n - eval_n),
            "test": _FakeDataset(eval_n),
        }


def test_hold_out_uses_at_least_one_eval_row():
    ds = _FakeDataset(10)
    train, ev = hold_out_eval_dataset(ds, ratio=0.1)
    assert ds.last_kwargs["test_size"] == 1
    assert len(ev) == 1
    assert len(train) == 9


def test_hold_out_rejects_empty_ratio():
    try:
        hold_out_eval_dataset(_FakeDataset(10), ratio=0.0)
    except ValueError as e:
        assert "between 0 and 1" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_hold_out_needs_two_examples():
    try:
        hold_out_eval_dataset(_FakeDataset(1), ratio=0.1)
    except ValueError as e:
        assert "at least 2" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_training_args_align_save_steps_to_eval_steps():
    t = SimpleNamespace(eval_steps=50, save_steps=120, metric_for_best_model="eval_loss")
    kwargs = training_args_for_early_stopping(t)
    assert kwargs["eval_strategy"] == "steps"
    assert kwargs["save_strategy"] == "steps"
    assert kwargs["eval_steps"] == 50
    assert kwargs["save_steps"] == 150
    assert kwargs["load_best_model_at_end"] is True
    assert kwargs["metric_for_best_model"] == "eval_loss"
    assert kwargs["greater_is_better"] is False


def test_training_args_keep_aligned_save_steps():
    t = SimpleNamespace(eval_steps=50, save_steps=100, metric_for_best_model="eval_loss")
    kwargs = training_args_for_early_stopping(t)
    assert kwargs["save_steps"] == 100


def test_training_args_drop_load_best_when_deepspeed_saves_model_only():
    t = SimpleNamespace(eval_steps=50, save_steps=100, metric_for_best_model="eval_loss")
    kwargs = training_args_for_early_stopping(
        t,
        base_kwargs={"deepspeed": "configs/deepspeed/zero3_config.json", "save_only_model": True},
    )
    assert kwargs["load_best_model_at_end"] is False
    assert kwargs["eval_strategy"] == "steps"


def test_training_args_keep_load_best_without_deepspeed():
    t = SimpleNamespace(eval_steps=50, save_steps=100, metric_for_best_model="eval_loss")
    kwargs = training_args_for_early_stopping(t, base_kwargs={"save_only_model": True})
    assert kwargs["load_best_model_at_end"] is True


def test_training_config_can_enable_evaluation_without_early_stopping():
    from core.config import TrainingConfig

    config = TrainingConfig(
        output_dir="out",
        eval_enabled=True,
        early_stopping=False,
    )

    assert config.eval_enabled is True
    assert config.early_stopping is False


def test_training_config_defaults_evaluation_to_early_stopping_for_compatibility():
    from core.config import TrainingConfig

    assert TrainingConfig(output_dir="out", early_stopping=True).eval_enabled is True
    assert TrainingConfig(output_dir="out", early_stopping=False).eval_enabled is False


def test_training_config_rejects_early_stopping_without_evaluation():
    from core.config import TrainingConfig

    with pytest.raises(ValueError, match="early_stopping requires eval_enabled"):
        TrainingConfig(
            output_dir="out",
            eval_enabled=False,
            early_stopping=True,
        )


def test_eval_holdout_runs_without_early_stopping():
    from core.trainer_base import BaseTrainer

    trainer = SimpleNamespace(
        config=SimpleNamespace(
            training=SimpleNamespace(
                eval_enabled=True,
                early_stopping=False,
                eval_split_ratio=0.2,
            )
        ),
        supports_early_stopping=True,
        logger=logging.getLogger("test"),
        eval_dataset=None,
    )

    train, evaluation = BaseTrainer.apply_eval_holdout(trainer, _FakeDataset(10))

    assert len(train) == 8
    assert len(evaluation) == 2
    assert trainer.eval_dataset is evaluation


def _stub_trainer(trainer_cls, **training_overrides):
    """A trainer instance with only the fields ``setup_training_args`` reads."""
    from core.config import DataConfig, TrainingConfig

    trainer = object.__new__(trainer_cls)
    trainer.logger = logging.getLogger("test")
    trainer.config = SimpleNamespace(
        training=TrainingConfig(output_dir="out", **training_overrides),
        data=DataConfig(),
        wandb=SimpleNamespace(enabled=False),
    )
    trainer._split_mode = False
    return trainer


def _supervised_trainer_classes():
    from trainers.cpt_trainer import CPTTrainer
    from trainers.dpo_trainer import DPOTrainer
    from trainers.sft_trainer import SFTTrainer
    from trainers.sft_vlm_trainer import SFTVLMTrainer

    return {
        "sft": SFTTrainer,
        "cpt": CPTTrainer,
        "dpo": DPOTrainer,
        "sft_vlm": SFTVLMTrainer,
    }


@pytest.mark.parametrize("algorithm", ["sft", "cpt", "dpo", "sft_vlm"])
def test_setup_training_args_applies_early_stopping_overrides(algorithm):
    """Early stopping must override eval/save steps, not collide with them.

    Passing both the recipe values and the early-stopping overrides to the TRL
    config raises ``TypeError: got multiple values for keyword argument``.
    """
    trainer_cls = _supervised_trainer_classes()[algorithm]
    trainer = _stub_trainer(trainer_cls, early_stopping=True, eval_steps=50, save_steps=120)

    args = trainer.setup_training_args()

    assert args.eval_strategy == "steps"
    assert args.eval_steps == 50
    assert args.save_steps == 150
    assert args.load_best_model_at_end is True
    assert args.metric_for_best_model == "eval_loss"


@pytest.mark.parametrize("algorithm", ["sft", "cpt", "dpo", "sft_vlm"])
def test_setup_training_args_without_early_stopping_keeps_recipe_steps(algorithm):
    trainer_cls = _supervised_trainer_classes()[algorithm]
    trainer = _stub_trainer(trainer_cls, early_stopping=False, eval_steps=50, save_steps=120)

    args = trainer.setup_training_args()

    assert args.save_steps == 120
    assert args.load_best_model_at_end is False


@pytest.mark.parametrize("algorithm", ["sft", "cpt", "dpo", "sft_vlm"])
def test_setup_training_args_evaluates_without_early_stopping(algorithm):
    trainer_cls = _supervised_trainer_classes()[algorithm]
    trainer = _stub_trainer(
        trainer_cls,
        eval_enabled=True,
        early_stopping=False,
        eval_steps=50,
        save_steps=100,
    )

    args = trainer.setup_training_args()

    assert args.eval_strategy == "steps"
    assert args.eval_steps == 50
    assert args.save_steps == 100
    assert args.load_best_model_at_end is False


@pytest.mark.parametrize("algorithm", ["sft", "cpt", "dpo", "sft_vlm"])
def test_setup_training_args_uses_recipe_eval_batch_size(algorithm):
    trainer_cls = _supervised_trainer_classes()[algorithm]
    trainer = _stub_trainer(
        trainer_cls,
        early_stopping=True,
        per_device_eval_batch_size=2,
    )

    args = trainer.setup_training_args()

    assert args.per_device_eval_batch_size == 2


def test_deepspeed_dpo_keeps_save_only_model_over_load_best():
    """DeepSpeed refuses to reload a best checkpoint saved without optimizer state."""
    # TrainingArguments rejects deepspeed= unless the (CUDA-only) package is installed.
    pytest.importorskip("deepspeed")
    from trainers.dpo_trainer import DPOTrainer

    trainer = _stub_trainer(
        DPOTrainer,
        early_stopping=True,
        save_only_model=True,
        deepspeed_config="configs/deepspeed/zero3_config.json",
    )

    args = trainer.setup_training_args()

    assert args.save_only_model is True
    assert args.load_best_model_at_end is False


def test_sft_recipe_loads_early_stopping_defaults():
    from core.config import ExperimentConfig

    config = ExperimentConfig.from_yaml(
        str(REPO_ROOT / "recipes" / "training" / "sft" / "llama3_3B_lora.yaml")
    )
    assert config.training.per_device_eval_batch_size == 1
    assert config.training.early_stopping is True
    assert config.training.early_stopping_patience == 3
    assert config.training.early_stopping_threshold == 0.0
    assert config.training.eval_split_ratio == 0.1
    assert config.training.metric_for_best_model == "eval_loss"


def test_supported_recipes_enable_early_stopping():
    import yaml

    for algorithm in ("sft", "sft_vlm", "cpt", "dpo"):
        for recipe in (REPO_ROOT / "recipes" / "training" / algorithm).glob("*.yaml"):
            config = yaml.safe_load(recipe.read_text())
            assert config["training"]["early_stopping"] is True, recipe
