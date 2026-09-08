"""Tests for early-stopping helpers (no model load)."""

import sys
from pathlib import Path
from types import SimpleNamespace

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


def test_sft_recipe_loads_early_stopping_defaults():
    from core.config import ExperimentConfig

    config = ExperimentConfig.from_yaml(
        str(REPO_ROOT / "recipes" / "training" / "sft" / "llama3_3B_lora.yaml")
    )
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
