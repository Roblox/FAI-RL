"""Reward function selection for GRPO and GSPO trainers."""

from core.config import ExperimentConfig

from .api_reward import APIRewardFunction
from .local_reward import LocalRewardFunction


def build_reward_function(config: ExperimentConfig, logger=None):
    """Build exactly one configured HTTP or local reward function."""
    if config.reward_api and config.local_reward_function:
        raise ValueError(
            "Configure only one of reward_api or local_reward_function"
        )
    if config.reward_api:
        if logger:
            logger.info("Using HTTP reward API: %s", config.reward_api.endpoint)
        return APIRewardFunction(config.reward_api, logger=logger)
    if config.local_reward_function:
        if logger:
            logger.warning(
                "Using local reward function for testing only: %s",
                config.local_reward_function.function,
            )
        return LocalRewardFunction(config.local_reward_function, logger=logger)
    raise ValueError(
        "GRPO/GSPO requires reward_api or local_reward_function configuration"
    )
