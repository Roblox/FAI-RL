import logging
from types import SimpleNamespace

import pytest

from core.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
    WandbConfig,
)
from engine.cookbook import run_recipe, uses_native_engine
from trainers.train import load_recipe_with_overrides

CUSTOM_RUNS = []


def record_custom_run(config, logger):
    CUSTOM_RUNS.append((config.training.entrypoint, logger))
    return "custom-result"


def _config(**training):
    return ExperimentConfig(
        model=ModelConfig(base_model_name="tiny"),
        data=DataConfig(),
        training=TrainingConfig(output_dir="out", **training),
        wandb=WandbConfig(enabled=False),
    )


@pytest.mark.parametrize(
    "training, native",
    [
        ({"algorithm": "sft"}, False),
        ({"algorithm": "sft", "backend": "native"}, True),
        ({"algorithm": "custom", "entrypoint": "pkg.mod:fn"}, True),
        ({"algorithm": "dpo"}, False),
    ],
)
def test_uses_native_engine(training, native):
    assert uses_native_engine(_config(**training)) is native


@pytest.mark.parametrize(
    "training, message",
    [
        ({"algorithm": "sft", "backend": "jax"}, "backend"),
        ({"algorithm": "dpo", "backend": "native"}, "native"),
        ({"algorithm": "custom"}, "entrypoint"),
        ({"algorithm": "custom", "entrypoint": "no_colon"}, "module:function"),
        ({"algorithm": "sft", "micro_batch_tokens": 0}, "micro_batch_tokens"),
        ({"algorithm": "sft", "backend": "native", "deepspeed_config": "ds.json"}, "DeepSpeed"),
        (
            {"algorithm": "custom", "entrypoint": "pkg.mod:fn", "deepspeed_config": "ds.json"},
            "DeepSpeed",
        ),
    ],
)
def test_invalid_native_engine_configs_are_rejected(training, message):
    with pytest.raises(ValueError, match=message):
        _config(**training)


def test_run_recipe_calls_custom_entrypoint():
    CUSTOM_RUNS.clear()
    logger = logging.getLogger("custom")
    config = _config(algorithm="custom", entrypoint=f"{__name__}:record_custom_run")

    assert run_recipe(config, logger) == "custom-result"
    assert CUSTOM_RUNS == [(f"{__name__}:record_custom_run", logger)]


def test_run_recipe_rejects_unresolvable_entrypoint():
    config = _config(algorithm="custom", entrypoint=f"{__name__}:missing")
    with pytest.raises(ValueError, match="does not resolve"):
        run_recipe(config, logging.getLogger("custom"))


NATIVE_SFT_RECIPE = "recipes/training/sft/llama3_3B_lora_native.yaml"


def test_native_recipe_loads_through_launcher():
    from_yaml = ExperimentConfig.from_yaml(NATIVE_SFT_RECIPE)
    launched = load_recipe_with_overrides(SimpleNamespace(recipe=NATIVE_SFT_RECIPE, overrides=[]))

    assert uses_native_engine(from_yaml) and uses_native_engine(launched)
    assert launched.training.to_dict() == from_yaml.training.to_dict()


def test_launcher_overrides_native_engine_fields():
    args = SimpleNamespace(
        recipe="recipes/training/sft/llama3_3B_lora.yaml",
        overrides=["training.backend=native", "training.micro_batch_tokens=2048"],
    )

    config = load_recipe_with_overrides(args)

    assert uses_native_engine(config)
    assert config.training.micro_batch_tokens == 2048
