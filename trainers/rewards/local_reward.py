"""TRL-compatible adapter for locally imported reward functions."""

import importlib
import logging
import math
from typing import Any, Callable, List

from core.config import LocalRewardFunctionConfig


class LocalRewardFunction:
    """Load and validate a local Python reward callable."""

    def __init__(self, config: LocalRewardFunctionConfig, logger=None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.function = self._load_function(config.function)

    @staticmethod
    def _load_function(import_path: str) -> Callable:
        module_name, function_name = import_path.split(":", 1)
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise ImportError(
                f"Could not import local reward module '{module_name}'"
            ) from exc

        function = getattr(module, function_name, None)
        if not callable(function):
            raise ValueError(
                f"Local reward '{import_path}' does not resolve to a callable"
            )
        return function

    def __call__(self, prompts, completions, **kwargs) -> List[float]:
        call_kwargs = {
            **kwargs,
            **self.config.kwargs,
            "logger": self.logger,
        }
        rewards = self.function(
            prompts=prompts,
            completions=completions,
            **call_kwargs,
        )
        scores = self._validate_rewards(rewards, len(completions))
        if scores:
            self.logger.info(
                "Local reward function=%s scored %d completions "
                "(min=%.4f max=%.4f mean=%.4f)",
                self.config.function,
                len(scores),
                min(scores),
                max(scores),
                sum(scores) / len(scores),
            )
        else:
            self.logger.info(
                "Local reward function=%s scored an empty batch",
                self.config.function,
            )
        return scores

    @staticmethod
    def _validate_rewards(rewards: Any, expected_count: int) -> List[float]:
        if not isinstance(rewards, list):
            raise TypeError("Local reward function must return a list")
        if len(rewards) != expected_count:
            raise ValueError(
                f"Local reward function returned {len(rewards)} rewards for "
                f"{expected_count} completions"
            )

        scores = []
        for reward in rewards:
            if isinstance(reward, bool):
                raise TypeError("Local reward scores must be numbers, not booleans")
            score = float(reward)
            if not math.isfinite(score):
                raise ValueError("Local reward scores must be finite")
            scores.append(score)
        return scores
