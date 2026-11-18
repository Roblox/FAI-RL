import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from core.config import DataConfig, ExperimentConfig, ModelConfig, TrainingConfig, WandbConfig
from trainers import grpo_trainer, grpo_vlm_trainer, gspo_trainer
from trainers.train import load_recipe_with_overrides
from utils import device_utils

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


ROLLOUT_TRAINERS = [
    pytest.param(grpo_trainer, "GRPOTrainer", {}, id="grpo"),
    pytest.param(gspo_trainer, "GSPOTrainer", {"steps_per_generation": 1}, id="gspo"),
]


@pytest.mark.parametrize("module, cls_name, extra", ROLLOUT_TRAINERS)
def test_rollouts_use_transformers_generate_by_default(module, cls_name, extra):
    args = _trainer(module, cls_name, **extra).setup_training_args()

    assert args.use_vllm is False


@pytest.mark.parametrize("module, cls_name, extra", ROLLOUT_TRAINERS)
def test_colocated_vllm_settings_reach_trl_config(module, cls_name, extra):
    args = _trainer(
        module,
        cls_name,
        use_vllm=True,
        vllm_gpu_memory_utilization=0.45,
        vllm_tensor_parallel_size=2,
        vllm_max_model_length=4096,
        vllm_enable_sleep_mode=True,
        **extra,
    ).setup_training_args()

    assert args.use_vllm is True
    assert args.vllm_mode == "colocate"
    assert args.vllm_gpu_memory_utilization == 0.45
    assert args.vllm_tensor_parallel_size == 2
    assert args.vllm_max_model_length == 4096
    assert args.vllm_enable_sleep_mode is True


@pytest.mark.parametrize("module, cls_name, extra", ROLLOUT_TRAINERS)
def test_vllm_server_settings_reach_trl_config(module, cls_name, extra):
    args = _trainer(
        module,
        cls_name,
        use_vllm=True,
        vllm_mode="server",
        vllm_server_base_url="http://vllm:8000",
        vllm_server_host="10.0.0.5",
        vllm_server_port=8001,
        vllm_server_timeout=60,
        **extra,
    ).setup_training_args()

    assert args.vllm_mode == "server"
    assert args.vllm_server_base_url == "http://vllm:8000"
    assert args.vllm_server_host == "10.0.0.5"
    assert args.vllm_server_port == 8001
    assert args.vllm_server_timeout == 60.0


def test_grpo_vlm_inherits_vllm_settings():
    args = _trainer(grpo_vlm_trainer, "GRPOVLMTrainer", use_vllm=True).setup_training_args()

    assert args.use_vllm is True
    assert args.remove_unused_columns is False


@pytest.mark.parametrize(
    "training, message",
    [
        ({"vllm_mode": "remote"}, "vllm_mode"),
        ({"vllm_gpu_memory_utilization": 0}, "vllm_gpu_memory_utilization"),
        ({"vllm_gpu_memory_utilization": 1.5}, "vllm_gpu_memory_utilization"),
        ({"vllm_tensor_parallel_size": 0}, "vllm_tensor_parallel_size"),
        ({"vllm_max_model_length": 0}, "vllm_max_model_length"),
    ],
)
def test_training_config_rejects_invalid_vllm_settings(training, message):
    with pytest.raises(ValueError, match=message):
        TrainingConfig(output_dir="out", **training)


def _experiment(algorithm, model=None, **training):
    return ExperimentConfig(
        model=ModelConfig(base_model_name="tiny", **(model or {})),
        data=DataConfig(),
        training=TrainingConfig(output_dir="out", algorithm=algorithm, **training),
        wandb=WandbConfig(enabled=False),
    )


@pytest.mark.parametrize("algorithm", ["grpo", "grpo_vlm", "gspo"])
def test_vllm_is_accepted_for_rollout_algorithms(algorithm):
    assert _experiment(algorithm, use_vllm=True).training.use_vllm is True


@pytest.mark.parametrize("algorithm", ["sft", "dpo", "cpt", "sft_vlm"])
def test_vllm_is_rejected_for_algorithms_without_rollouts(algorithm):
    with pytest.raises(ValueError, match="use_vllm supports grpo"):
        _experiment(algorithm, use_vllm=True)


def test_vllm_rejects_8bit_models():
    with pytest.raises(ValueError, match="load_in_8bit"):
        _experiment("grpo", model={"load_in_8bit": True, "use_lora": True}, use_vllm=True)


@pytest.mark.parametrize("device", ["mps", "cpu"])
def test_vllm_is_disabled_without_cuda(monkeypatch, device):
    monkeypatch.setattr(device_utils, "get_device_type", lambda: device)
    config = _experiment("grpo", use_vllm=True)

    device_utils.adapt_config_for_device(config)

    assert config.training.use_vllm is False


def test_vllm_stays_enabled_on_cuda(monkeypatch):
    monkeypatch.setattr(device_utils, "get_device_type", lambda: "cuda")
    config = _experiment("grpo", use_vllm=True)

    device_utils.adapt_config_for_device(config)

    assert config.training.use_vllm is True


VLLM_RECIPE = "recipes/training/grpo/llama3_3B_lora_vllm.yaml"
REWARD_ENDPOINT = "reward_api.endpoint=https://reward.example.com/v1/score"


def test_vllm_recipe_loads_through_launcher():
    config = load_recipe_with_overrides(
        SimpleNamespace(recipe=VLLM_RECIPE, overrides=[REWARD_ENDPOINT])
    )

    assert config.training.use_vllm is True
    assert config.training.vllm_mode == "colocate"
    assert config.training.vllm_max_model_length >= config.data.max_length


def test_launcher_overrides_switch_vllm_to_server_mode():
    config = load_recipe_with_overrides(
        SimpleNamespace(
            recipe=VLLM_RECIPE,
            overrides=[
                REWARD_ENDPOINT,
                "training.vllm_mode=server",
                "training.vllm_server_base_url=http://vllm:8000",
            ],
        )
    )

    assert config.training.vllm_mode == "server"
    assert config.training.vllm_server_base_url == "http://vllm:8000"
