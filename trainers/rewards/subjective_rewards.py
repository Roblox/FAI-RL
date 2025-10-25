"""Subjective reward function using API-based evaluation."""
import os
import sys
from typing import List, Dict, Any, Optional
from utils.api_utils import generate_response_by_api_for_reward_function

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from . import reward_function
except ImportError:
    # Fallback for standalone imports (e.g., in tests)
    def reward_function(func):
        """Decorator to mark functions as reward functions."""
        func._is_reward_function = True
        return func


@reward_function
def subjective_api_reward_func(
    completions: List[str],
    api_endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
    api_model: Optional[str] = None,
    best_reward: float = 2.0,
    worst_penalty: float = -2.0,
    neutral_reward: float = 0.0,
    num_generations: Optional[int] = None,
    **kwargs
) -> List[float]:
    """
    Reward function that uses an external API to determine best/worst responses.
    
    This function sends completions to an API evaluator which returns the indices
    of the best and worst responses. When num_generations is specified, it calculates
    best/worst per group rather than globally.
    
    Args:
        completions: List of model-generated completions (list of message dicts)
        api_endpoint: URL of the evaluation API
        api_key: API key for authentication
        api_model: Model identifier for the evaluation API
        best_reward: Reward value for the best response (default: 2.0)
        worst_penalty: Penalty value for the worst response (default: -2.0)
        neutral_reward: Reward for neutral/middle responses (default: 0.0)
        num_generations: Size of groups for calculating best/worst (default: None, means global)
        **kwargs: Additional keyword arguments (e.g., logger)
        
    Returns:
        List[float]: Reward values for each completion
    """
    logger = kwargs.get('logger', None)
    
    # Extract completion text from message format
    completion_texts = []
    for completion in completions:
        if isinstance(completion, list) and len(completion) > 0:
            # Extract content from the last message (assistant response)
            completion_texts.append(completion[-1].get('content', ''))
        elif isinstance(completion, dict):
            completion_texts.append(completion.get('content', ''))
        else:
            completion_texts.append(str(completion))
    
    # Extract prompt text for API call from kwargs
    prompt = kwargs.get('prompt', None)
    
    # Initialize all rewards with neutral value
    # Only best and worst responses will receive non-neutral rewards
    rewards = [neutral_reward] * len(completion_texts)
    
    try:
        num_groups = len(completion_texts) // num_generations
        if logger:
            logger.info(f"Processing {len(completion_texts)} completions in {num_groups} groups of size {num_generations}")
        
        for group_idx in range(num_groups):
            start_idx = group_idx * num_generations
            end_idx = start_idx + num_generations
            group_completions = completion_texts[start_idx:end_idx]
            
            # Get the prompt for this group
            # Prompts could be a list (one per completion) or a single value
            if isinstance(prompt, list) and len(prompt) > start_idx:
                group_prompt = prompt[start_idx]
            else:
                group_prompt = prompt if prompt is not None else ""
            
            if logger:
                logger.info(f"Calling subjective API evaluator for group {group_idx + 1}/{num_groups} (completions {start_idx}-{end_idx-1})")
            
            result = generate_response_by_api_for_reward_function(
                prompt=group_prompt,
                completions=group_completions,
                api_endpoint=api_endpoint,
                api_key=api_key,
                model=api_model,
                logger_instance=logger
            )
            
            best_idx = result.get('best_idx')
            worst_idx = result.get('worst_idx')
            
            if logger:
                logger.info(f"API evaluation for group {group_idx + 1}: best_idx={best_idx}, worst_idx={worst_idx}")
            
            # Assign rewards ONLY to best and worst responses per group
            # All other responses in the group retain neutral_reward
            if best_idx is not None and 0 <= best_idx < num_generations:
                rewards[start_idx + best_idx] = best_reward
            
            if worst_idx is not None and 0 <= worst_idx < num_generations:
                rewards[start_idx + worst_idx] = worst_penalty
    
    except Exception as e:
        if logger:
            logger.error(f"Error in subjective API reward function: {str(e)}")
        # Return neutral rewards on error to avoid training disruption
        return [neutral_reward] * len(completion_texts)
    
    return rewards


@reward_function
def subjective_api_reward_func_simple(
    completions: List[str],
    **kwargs
) -> List[float]:
    """
    Simplified version of subjective_api_reward_func with default parameters.
    
    This is a convenience wrapper that uses environment variables for all configuration.
    Suitable for most use cases where you want to use default reward values.
    
    Args:
        completions: List of model-generated completions
        **kwargs: Additional keyword arguments passed through
        
    Returns:
        List[float]: Reward values for each completion
    """

    return subjective_api_reward_func(
        completions=completions,
        best_reward=2.0,
        worst_penalty=-2.0,
        neutral_reward=0.0,
        **kwargs
    )

