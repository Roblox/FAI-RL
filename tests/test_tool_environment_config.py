import pytest
from types import SimpleNamespace

from core.config import DatasetInfo, ExperimentConfig, RewardAPIConfig
from trainers.train import load_recipe_with_overrides


def base_recipe():
    return {
        "model": {"base_model_name": "Qwen/Qwen3.8-27B"},
        "data": {
            "datasets": [
                {
                    "name": "Roblox/tool-training",
                    "prompt_column": "prompt",
                }
            ]
        },
        "training": {
            "output_dir": "models/tool-test",
            "algorithm": "grpo",
        },
        "wandb": {"enabled": False},
        "reward_api": {
            "version": "v2",
            "endpoint": "https://reward.example/trajectory",
        },
        "tool_environment": {
            "version": "v1",
            "gateway_url": "https://tools.example",
            "server_id": "inventory",
            "tools": [
                {
                    "name": "lookup_item",
                    "description": "Look up an item.",
                    "input_schema": {"type": "object"},
                }
            ],
            "max_turns": 4,
            "max_tool_calls": 8,
        },
    }


def write_recipe(tmp_path, recipe):
    import yaml

    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(recipe))
    return str(path)


def test_loads_tool_environment_recipe(tmp_path):
    recipe = base_recipe()
    recipe["data"]["datasets"][0]["reward_context_columns"] = [
        "expected",
        "difficulty",
    ]
    config = ExperimentConfig.from_yaml(write_recipe(tmp_path, recipe))

    assert config.tool_environment.server_id == "inventory"
    assert config.tool_environment.tools[0].name == "lookup_item"
    assert config.reward_api.version == "v2"
    assert config.data.datasets[0].reward_context_columns == [
        "expected",
        "difficulty",
    ]


def test_production_cli_loader_preserves_tool_environment(tmp_path):
    recipe_path = write_recipe(tmp_path, base_recipe())

    config = load_recipe_with_overrides(
        SimpleNamespace(recipe=recipe_path, overrides=[])
    )

    assert config.tool_environment.server_id == "inventory"
    assert config.tool_environment.tools[0].name == "lookup_item"


@pytest.mark.parametrize(
    "columns",
    [["expected", "expected"], ["prompt"], [""]],
)
def test_rejects_invalid_reward_context_columns(columns):
    with pytest.raises(ValueError, match="reward_context_columns"):
        DatasetInfo(
            name="Roblox/tool-training",
            reward_context_columns=columns,
        )


def test_tool_environment_requires_grpo_and_v2_reward(tmp_path):
    recipe = base_recipe()
    recipe["training"]["algorithm"] = "sft"
    with pytest.raises(ValueError, match="GRPO"):
        ExperimentConfig.from_yaml(write_recipe(tmp_path, recipe))

    recipe = base_recipe()
    recipe["reward_api"]["version"] = "v1"
    with pytest.raises(ValueError, match="v2"):
        ExperimentConfig.from_yaml(write_recipe(tmp_path, recipe))

    recipe = base_recipe()
    recipe.pop("reward_api")
    with pytest.raises(ValueError, match="trajectory reward"):
        ExperimentConfig.from_yaml(write_recipe(tmp_path, recipe))

    recipe = base_recipe()
    recipe["reward_api"]["endpoint"] = "http://reward.example/trajectory"
    with pytest.raises(ValueError, match="HTTPS"):
        ExperimentConfig.from_yaml(write_recipe(tmp_path, recipe))

    recipe = base_recipe()
    recipe["reward_api"]["endpoint"] = (
        "https://user:password@reward.example/trajectory"
    )
    with pytest.raises(ValueError, match="credentials"):
        ExperimentConfig.from_yaml(write_recipe(tmp_path, recipe))

    recipe = base_recipe()
    recipe["reward_api"]["endpoint"] = (
        "https://reward.example/trajectory?api_key=secret"
    )
    with pytest.raises(ValueError, match="query or fragment"):
        ExperimentConfig.from_yaml(write_recipe(tmp_path, recipe))

    recipe = base_recipe()
    recipe["reward_api"]["headers"] = {
        "Authorization": "Bearer static-secret"
    }
    with pytest.raises(ValueError, match="static headers"):
        ExperimentConfig.from_yaml(write_recipe(tmp_path, recipe))

    recipe = base_recipe()
    recipe["tool_environment"]["max_episode_seconds"] = 3601
    with pytest.raises(ValueError, match="max_episode_seconds"):
        ExperimentConfig.from_yaml(write_recipe(tmp_path, recipe))

    recipe = base_recipe()
    recipe["tool_environment"]["max_request_bytes"] = 1048577
    with pytest.raises(ValueError, match="max_request_bytes"):
        ExperimentConfig.from_yaml(write_recipe(tmp_path, recipe))

    recipe = base_recipe()
    recipe["tool_environment"]["version"] = "v9"
    with pytest.raises(ValueError, match="tool_environment.version"):
        ExperimentConfig.from_yaml(write_recipe(tmp_path, recipe))

    recipe = base_recipe()
    recipe["reward_api"]["version"] = "v9"
    with pytest.raises(ValueError, match="reward_api.version"):
        ExperimentConfig.from_yaml(write_recipe(tmp_path, recipe))

    recipe = base_recipe()
    recipe["model"]["base_model_name"] = "Qwen/Qwen3-8B"
    with pytest.raises(ValueError, match="Qwen3.8-27B"):
        ExperimentConfig.from_yaml(write_recipe(tmp_path, recipe))


def test_reward_api_positional_headers_remain_backward_compatible():
    config = RewardAPIConfig(
        "https://reward.example/score",
        {"X-Tenant": "customer-care"},
    )

    assert config.version == "v1"
    assert config.headers == {"X-Tenant": "customer-care"}
