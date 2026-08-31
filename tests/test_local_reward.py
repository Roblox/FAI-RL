"""Tests for YAML-configured local reward functions."""

import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.config import ExperimentConfig, LocalRewardFunctionConfig, RewardAPIConfig
from trainers.rewards.factory import build_reward_function
from trainers.rewards.local_reward import LocalRewardFunction


def test_local_reward_loads_configured_callable(caplog):
    reward = LocalRewardFunction(
        LocalRewardFunctionConfig(
            function="trainers.rewards.accuracy_rewards:exact_match_reward_func"
        )
    )

    with caplog.at_level(logging.INFO, logger="trainers.rewards.local_reward"):
        scores = reward(
            ["sensitive-prompt", "other-prompt"],
            ["<answer>42</answer>", "<answer>0</answer>"],
            answer=["42", "7"],
        )

    assert scores == [2.0, 0.0]
    assert "scored 2 completions" in caplog.text
    assert "sensitive-prompt" not in caplog.text


def test_local_reward_validates_score_count():
    reward = LocalRewardFunction(
        LocalRewardFunctionConfig(
            function="trainers.rewards.custom_rewards:custom_reward_func"
        )
    )
    reward.function = lambda **_kwargs: []

    with pytest.raises(ValueError, match="0 rewards for 1 completions"):
        reward(["prompt"], ["completion"])


def test_local_reward_config_rejects_invalid_import_path():
    with pytest.raises(ValueError, match="module.path:function_name"):
        LocalRewardFunctionConfig(function="missing_separator")


def test_reward_factory_rejects_multiple_sources():
    config = SimpleNamespace(
        reward_api=RewardAPIConfig(endpoint="https://reward.example/score"),
        local_reward_function=LocalRewardFunctionConfig(
            function="trainers.rewards.custom_rewards:custom_reward_func"
        ),
    )

    with pytest.raises(ValueError, match="only one"):
        build_reward_function(config)


def test_local_reward_smoke_recipe_loads():
    config = ExperimentConfig.from_yaml(
        str(
            REPO_ROOT
            / "recipes"
            / "training"
            / "grpo"
            / "llama3_3B_local_reward.yaml"
        )
    )

    assert config.reward_api is None
    assert config.local_reward_function is not None
    assert (
        config.local_reward_function.function
        == "trainers.rewards.accuracy_rewards:exact_match_reward_func"
    )
