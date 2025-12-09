"""Subjective reward functions for API-based quality evaluation."""
import json
import logging
import requests
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


def subjective_api_reward_func_simple(
    completions: List[str],
    prompt: Optional[List[str]] = None,
    api_endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
    num_generations: int = 1,
    logger: Optional[logging.Logger] = None,
    **kwargs
) -> List[float]:
    """
    API-based subjective reward function for evaluating response quality.
    
    This function sends completions to an LLM API to evaluate their quality
    based on subjective criteria (helpfulness, coherence, safety, etc.).
    
    Args:
        completions: List of model completions to evaluate
        prompt: List of original prompts (optional, for context)
        api_endpoint: The API endpoint URL for the reward model
        api_key: API key for authentication
        num_generations: Number of generations per prompt
        logger: Optional logger for debugging
        **kwargs: Additional keyword arguments
        
    Returns:
        List of reward scores (0.0 to 1.0) for each completion
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    rewards = []
    
    for i, completion in enumerate(completions):
        try:
            # Get the corresponding prompt if available
            prompt_text = ""
            if prompt and i < len(prompt):
                prompt_text = prompt[i] if isinstance(prompt[i], str) else str(prompt[i])
            
            # Build evaluation prompt for the reward model
            eval_prompt = _build_evaluation_prompt(prompt_text, completion)
            
            # Call the API to get evaluation
            score = _call_reward_api(
                eval_prompt,
                api_endpoint=api_endpoint,
                api_key=api_key,
                logger=logger
            )
            
            rewards.append(score)
            
        except Exception as e:
            logger.warning(f"Failed to evaluate completion {i}: {str(e)}")
            rewards.append(0.0)
    
    return rewards


def _build_evaluation_prompt(prompt: str, completion: str) -> str:
    """
    Build the evaluation prompt for the reward model.
    
    Args:
        prompt: The original prompt
        completion: The model's completion to evaluate
        
    Returns:
        Formatted evaluation prompt
    """
    evaluation_prompt = f"""You are an expert evaluator. Rate the quality of the following response on a scale of 0 to 10.

Consider these criteria:
1. Helpfulness: Does the response address the user's needs?
2. Coherence: Is the response well-structured and logical?
3. Accuracy: Is the information correct and reliable?
4. Safety: Is the response free from harmful content?

Original Prompt:
{prompt}

Response to Evaluate:
{completion}

Provide your rating as a single number from 0 to 10, where:
- 0-2: Poor quality, unhelpful or harmful
- 3-4: Below average, missing key information
- 5-6: Average, meets basic requirements
- 7-8: Good quality, helpful and accurate
- 9-10: Excellent quality, comprehensive and insightful

Rating (just the number):"""
    
    return evaluation_prompt


def _call_reward_api(
    eval_prompt: str,
    api_endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
    logger: Optional[logging.Logger] = None
) -> float:
    """
    Call the reward API to get an evaluation score.
    
    Args:
        eval_prompt: The evaluation prompt to send
        api_endpoint: The API endpoint URL
        api_key: API key for authentication
        logger: Optional logger
        
    Returns:
        Normalized score between 0.0 and 1.0
    """
    if not api_endpoint or not api_key:
        if logger:
            logger.warning("No API endpoint or key provided for subjective reward")
        return 0.0
    
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        data = {
            "messages": [{"role": "user", "content": eval_prompt}],
            "max_tokens": 1000,
            "temperature": 0.0
        }
        
        response = requests.post(api_endpoint, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        
        # Parse the response to extract the score
        response_text = ""
        if "choices" in result and len(result["choices"]) > 0:
            response_text = result["choices"][0].get("message", {}).get("content", "")
        elif "content" in result:
            response_text = result["content"][0].get("text", "") if isinstance(result["content"], list) else result["content"]
        
        # Extract numeric score from response
        score = _parse_score(response_text)
        
        # Normalize to 0-1 range
        normalized_score = score / 10.0
        
        return max(0.0, min(1.0, normalized_score))
        
    except requests.exceptions.RequestException as e:
        if logger:
            logger.warning(f"API request failed: {str(e)}")
        return 0.0
    except Exception as e:
        if logger:
            logger.warning(f"Error parsing reward API response: {str(e)}")
        return 0.0


def _parse_score(response_text: str) -> float:
    """
    Parse a numeric score from the API response text.
    
    Args:
        response_text: The raw text response from the API
        
    Returns:
        Parsed score (0-10 scale)
    """
    import re
    
    # Try to find a number in the response
    numbers = re.findall(r'\d+(?:\.\d+)?', response_text.strip())
    
    if numbers:
        score = float(numbers[0])
        # Clamp to 0-10 range
        return max(0.0, min(10.0, score))
    
    return 0.0

