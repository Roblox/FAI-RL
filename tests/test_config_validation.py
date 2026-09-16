"""Recipe diagnostics use dataclass schemas without importing training packages."""

from pathlib import Path

import pytest

from core.config import ExperimentConfig, TrainingConfig
from utils.config_validation import validate_config_section, validate_training_recipe
from utils.recipe_overrides import load_recipe_from_yaml, set_nested_value


def test_unknown_key_suggests_full_path():
    with pytest.raises(ValueError, match="training.learning_rate") as error:
        validate_config_section(
            {"output_dir": "out", "algorithm": "sft", "lr": 1e-5}, TrainingConfig, "training"
        )
    assert "training.lr" in str(error.value)
    assert "Available keys" in str(error.value)


@pytest.mark.parametrize(
    "recipe, message",
    [
        ([], "YAML mapping"),
        ({"model": []}, "must be a mapping"),
        ({"model": {}}, "model.base_model_name"),
        (
            {
                "model": {"base_model_name": "test"},
                "training": {"algorithm": "sft", "output_dir": "out", "bf16": "false"},
            },
            "training.bf16",
        ),
    ],
)
def test_invalid_shapes_and_types(recipe, message):
    with pytest.raises(ValueError, match=message):
        validate_training_recipe(recipe)


def test_nested_dataset_path():
    recipe = {
        "model": {"base_model_name": "test"},
        "training": {"algorithm": "sft", "output_dir": "out"},
        "data": {"datasets": [{"nmae": "test"}]},
    }
    with pytest.raises(ValueError, match=r"data.datasets\[0\].name"):
        validate_training_recipe(recipe)


@pytest.mark.parametrize("text", ["", "- item", "model: [broken"])
def test_yaml_error_includes_filename(tmp_path, text):
    recipe = tmp_path / "broken.yaml"
    recipe.write_text(text)
    with pytest.raises(ValueError, match="broken.yaml"):
        load_recipe_from_yaml(str(recipe))


def test_override_into_scalar():
    with pytest.raises(ValueError, match="training.learning_rate"):
        set_nested_value({"training": 3}, "training.learning_rate", 1e-5)


def test_shipped_training_recipes_validate():
    for path in Path("recipes/training").rglob("*.yaml"):
        validate_training_recipe(load_recipe_from_yaml(str(path)))


def test_from_yaml_uses_validation(tmp_path):
    recipe = tmp_path / "bad.yaml"
    recipe.write_text("model:\n  base_model_name: test\ntraining:\n  lr: 0.01\n")
    with pytest.raises(ValueError, match="training.learning_rate"):
        ExperimentConfig.from_yaml(str(recipe))
