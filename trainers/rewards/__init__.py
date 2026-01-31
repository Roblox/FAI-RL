"""Reward functions for reinforcement learning training."""


def reward_function(func):
    """Decorator to mark functions as reward functions."""
    func._is_reward_function = True
    return func


# Import after defining reward_function to avoid circular import
from .accuracy_rewards import exact_match_reward_func, digit_reward_func
from .format_rewards import structured_xml_reward_func
from .custom_rewards import custom_reward_func
from .subjective_rewards import subjective_api_reward_func_simple


__all__ = [
    'reward_function',
    'exact_match_reward_func',
    'digit_reward_func',
    'structured_xml_reward_func',
    'custom_reward_func',
    'subjective_api_reward_func_simple',
]
