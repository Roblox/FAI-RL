from typing import List
import logging

def extract_answer(completion, logger=None) -> str:
    """Extracts the answer from XML-formatted text."""
    try:
        # Handle case where completion is a list (e.g., list of tokens or candidates)
        if isinstance(completion, list):
            # If it's a list, join it into a string or take the first element
            if completion and isinstance(completion[0], str):
                completion_str = ''.join(completion) if len(completion) > 1 else completion[0]
            else:
                completion_str = str(completion[0]) if completion else ""
        else:
            completion_str = str(completion)
        
        answer = completion_str.split("<answer>")[-1].split("</answer>")[0]
        # More aggressive cleaning: strip all whitespace and newlines
        answer = answer.strip().replace('\\n', '').replace('\n', '').replace('\r', '').replace('\t', ' ')
        # Remove multiple spaces with single space
        answer = ' '.join(answer.split())
        return answer
    except (IndexError, AttributeError) as e:
        if logger:
            logger.warning(f"Failed to extract answer from completion: {e}")
        return ""


def exact_match_reward_func(completions: List[str], answer, **kwargs) -> List[float]:
    """Reward function that matches extracted answers with ground truth."""
    rewards = []
    logger = kwargs.get('logger', None)
    for comp, ans in zip(completions, answer):
        gen_answer = extract_answer(comp, logger)
        reward = 2.0 if gen_answer == ans else 0.0
        rewards.append(reward)
    return rewards


def digit_reward_func(completions: List[str], **kwargs) -> List[float]:
    """Calculates reward if the extracted response is a digit."""
    logger = kwargs.get('logger', None)
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_answer(r, logger) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]