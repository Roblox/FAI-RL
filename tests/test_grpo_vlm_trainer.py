"""Tests for image GRPO (grpo_vlm): data shape, training args, and reward inputs."""

import io
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from datasets import Dataset
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.config import DataConfig, DatasetInfo, ExperimentConfig, TrainingConfig
from trainers import grpo_vlm_trainer
from trainers.grpo_vlm_trainer import GRPOVLMTrainer, TextRewardAdapter


def _png_bytes(color="red"):
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buf, format="PNG")
    return buf.getvalue()


def _trainer(data_config, rows, monkeypatch):
    monkeypatch.setattr(
        grpo_vlm_trainer, "load_raw_dataset", lambda _info: Dataset.from_list(rows)
    )
    trainer = object.__new__(GRPOVLMTrainer)
    trainer.logger = logging.getLogger("test_grpo_vlm")
    trainer.config = SimpleNamespace(data=data_config)
    return trainer


def test_setup_data_builds_conversational_image_prompts(monkeypatch):
    data = DataConfig(
        datasets=[
            DatasetInfo(
                name="avatars.parquet",
                prompt_column="user",
                image_columns=["image_0", "image_90"],
            )
        ],
        system_prompt="You are a moderator.",
    )
    rows = [
        {
            "user": "Is this avatar violative?",
            "image_0": {"bytes": _png_bytes("red"), "path": None},
            "image_90": {"bytes": _png_bytes("blue"), "path": None},
            "response": '{"is_violative": false}',
        }
    ]
    trainer = _trainer(data, rows, monkeypatch)

    trainer.setup_data()

    row = trainer.train_dataset[0]
    assert row["prompt"] == [
        {"role": "system", "content": "You are a moderator."},
        {"role": "user", "content": "Is this avatar violative?"},
    ]
    assert len(row["images"]) == 2
    assert all(isinstance(img, Image.Image) for img in row["images"])
    # Text columns stay available to the reward; raw media columns do not.
    assert row["response"] == '{"is_violative": false}'
    assert "image_0" not in row and "image_90" not in row
    assert not any(key.startswith("_") for key in row)


def test_setup_data_without_system_prompt_sends_user_turn_only(monkeypatch):
    data = DataConfig(
        datasets=[DatasetInfo(name="d", prompt_column="user", image_columns=["image"])]
    )
    rows = [{"user": "Describe it.", "image": {"bytes": _png_bytes(), "path": None}}]
    trainer = _trainer(data, rows, monkeypatch)

    trainer.setup_data()

    assert trainer.train_dataset[0]["prompt"] == [
        {"role": "user", "content": "Describe it."}
    ]


@pytest.mark.parametrize(
    "answer_column, expected",
    [("label", {"answer": "cat", "label": "cat"}), (None, {"label": "cat"})],
)
def test_setup_data_maps_answer_column_like_text_grpo(monkeypatch, answer_column, expected):
    data = DataConfig(
        datasets=[
            DatasetInfo(
                name="d",
                prompt_column="user",
                answer_column=answer_column,
                image_columns=["image"],
            )
        ]
    )
    rows = [{"user": "What is it?", "image": {"bytes": _png_bytes(), "path": None}, "label": "cat"}]
    trainer = _trainer(data, rows, monkeypatch)

    trainer.setup_data()

    row = trainer.train_dataset[0]
    assert {k: row[k] for k in ("answer", "label") if k in row} == expected


def test_setup_data_drops_rows_without_usable_images_or_prompt(monkeypatch):
    data = DataConfig(
        datasets=[DatasetInfo(name="d", prompt_column="user", image_columns=["image"])]
    )
    rows = [
        {"user": "ok", "image": {"bytes": _png_bytes(), "path": None}},
        {"user": "no image", "image": None},
        {"user": "broken image", "image": {"bytes": b"not-an-image", "path": None}},
        {"user": "  ", "image": {"bytes": _png_bytes(), "path": None}},
    ]
    trainer = _trainer(data, rows, monkeypatch)

    trainer.setup_data()

    assert len(trainer.train_dataset) == 1
    assert trainer.train_dataset[0]["prompt"][-1]["content"] == "ok"


def test_setup_data_rejects_video_columns(monkeypatch):
    data = DataConfig(
        datasets=[
            DatasetInfo(
                name="d",
                prompt_column="user",
                image_columns=["image"],
                video_columns=["clip"],
            )
        ]
    )
    trainer = _trainer(data, [{"user": "x", "image": None, "clip": None}], monkeypatch)

    with pytest.raises(ValueError, match="video_columns"):
        trainer.setup_data()


def test_training_args_keep_dataset_columns_for_the_reward():
    trainer = object.__new__(GRPOVLMTrainer)
    trainer.logger = logging.getLogger("test_grpo_vlm_args")
    trainer.config = SimpleNamespace(
        training=TrainingConfig(
            output_dir="out",
            num_generations=4,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=1,
            bf16=False,
            fp16=False,
        ),
        data=DataConfig(remove_unused_columns=True),
        wandb=SimpleNamespace(enabled=False),
    )

    args = trainer.setup_training_args()

    assert args.remove_unused_columns is False
    assert args.num_generations == 4


def test_reward_adapter_sends_text_prompts_and_completions_without_images():
    calls = {}

    def reward(prompts, completions, **kwargs):
        calls["prompts"] = prompts
        calls["completions"] = completions
        calls["kwargs"] = kwargs
        return [1.0, 0.0]

    adapter = TextRewardAdapter(reward)
    image = Image.new("RGB", (4, 4))
    prompt = [
        {"role": "system", "content": [{"type": "text", "text": "SYS"}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Is this violative?"},
            ],
        },
    ]

    scores = adapter(
        prompts=[prompt, prompt],
        completions=[
            [{"role": "assistant", "content": '{"is_violative": true}'}],
            [{"role": "assistant", "content": [{"type": "text", "text": "{}"}]}],
        ],
        images=[[image], [image]],
        response=["a", "b"],
    )

    assert scores == [1.0, 0.0]
    assert calls["completions"] == ['{"is_violative": true}', "{}"]
    assert calls["prompts"][0] == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "Is this violative?"},
    ]
    assert "images" not in calls["kwargs"] and "image" not in calls["kwargs"]
    assert calls["kwargs"]["response"] == ["a", "b"]
    assert adapter.__name__ == "reward"


def test_reward_adapter_leaves_plain_strings_unchanged():
    seen = {}

    def reward(prompts, completions, **kwargs):
        seen["prompts"], seen["completions"] = prompts, completions
        return [0.5]

    TextRewardAdapter(reward)(prompts=["p"], completions=["c"])

    assert seen == {"prompts": ["p"], "completions": ["c"]}


def test_grpo_vlm_recipe_loads_with_reward():
    config = ExperimentConfig.from_yaml(
        str(REPO_ROOT / "recipes" / "training" / "grpo_vlm" / "gemma_4_31b_it_lora.yaml")
    )

    assert config.training.algorithm == "grpo_vlm"
    assert config.reward_api is not None or config.local_reward_function is not None
    assert config.data.datasets[0].image_columns


def test_train_entrypoint_requires_a_reward_for_grpo_vlm(tmp_path):
    from trainers.train import load_recipe_with_overrides

    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        "model:\n  base_model_name: m\n"
        "training:\n  algorithm: grpo_vlm\n  output_dir: out\n"
        "data:\n  datasets:\n    - name: d\n      image_columns: [image]\n"
    )

    with pytest.raises(ValueError, match="requires reward_api or local_reward_function"):
        load_recipe_with_overrides(SimpleNamespace(recipe=str(recipe), overrides=None))
