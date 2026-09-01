"""Tests for training configuration logging."""

from utils.logging_utils import TrainingLogger


class _CapturingLogger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(message)


def test_experiment_logging_skips_unconfigured_optional_sections():
    training_logger = TrainingLogger.__new__(TrainingLogger)
    training_logger.logger = _CapturingLogger()

    training_logger.log_experiment_start(
        {
            "algorithm": {"name": "grpo"},
            "reward_api": None,
            "local_reward_function": {
                "function": "trainers.rewards.preset_rewards:exact_match_reward_func",
                "kwargs": {},
            },
        }
    )

    assert "ALGORITHM:" in training_logger.logger.messages
    assert "LOCAL_REWARD_FUNCTION:" in training_logger.logger.messages
    assert "REWARD_API:" not in training_logger.logger.messages
