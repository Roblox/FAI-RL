from typing import List
import logging
from . import reward_function

@reward_function
def structured_xml_reward_func(completions: List[str], **kwargs) -> List[float]:
    """Calculates reward based on XML tag counts."""
    return [count_xml(c) for c in completions]


def count_xml(text) -> float:
    """Counts XML tags and penalizes extra content."""
    count = 0.0
    if text.count("<think>") == 1:
        count += 0.125
    if text.count("</think>") == 1:
        count += 0.125
    if text.count("<answer>") == 1:
        count += 0.125
    if text.count("</answer>") == 1:
        count += 0.125
        count -= (len(text.split("</answer>")[-1])) * 0.001
    return count
