from typing import List
from . import reward_function


@reward_function
def custom_reward_func(completions: List[str], **kwargs) -> List[float]:
    """
    Placeholder for custom reward function.
    
    Args:
        completions: List of generated text completions from the model
        **kwargs: Additional context including:
            - logger: Logger instance for debugging
            - answer: Ground truth answers (if applicable)
            - prompts: Original prompts (if needed)
    
    Returns:
        List of reward scores (floats) corresponding to each completion.
        Higher scores indicate better completions.
    
    Example implementation:
        rewards = []
        logger = kwargs.get('logger', None)
        for completion in completions:
            # Your custom reward logic here
            reward = 0.0
            
            # Example: Check for some custom criteria
            # if meets_criteria(completion):
            #     reward += 1.0
            
            rewards.append(reward)
        return rewards
    """
    logger = kwargs.get('logger', None)
    
    # ========== CUSTOM REWARD LOGIC ==========
    # Replace this placeholder with your custom reward logic
    # Example: Process each completion and assign rewards
    rewards = []
    for completion in completions:
        # TODO: Implement your custom reward calculation here
        # Example:
        # - Check for specific patterns or keywords
        # - Validate output format
        # - Score based on custom criteria
        reward = 0.0  # Placeholder: currently returns 0.0
        rewards.append(reward)
    # =========================================
    
    if logger:
        logger.debug(f"Custom reward function called with {len(completions)} completions")
    
    return rewards

