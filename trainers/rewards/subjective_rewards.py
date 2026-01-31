"""Subjective reward functions for API-based quality evaluation."""
import logging
from typing import List, Optional

from utils.api_utils import generate_response_by_api

logger = logging.getLogger(__name__)


class RewardAPIConfig:
    """Simple config class for reward API calls."""
    def __init__(self, api_endpoint: str, api_key: str, model: str = "default", debug: bool = False):
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.model = model
        self.max_new_tokens = 10000
        self.temperature = 0.7
        self.debug = debug


def subjective_api_reward_func_simple(
    completions: List[str],
    prompt: Optional[List[str]] = None,
    api_endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
    api_model: Optional[str] = None,
    num_generations: int = 1,
    logger: Optional[logging.Logger] = None,
    debug: bool = False,
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
                prompt_text = _strip_chat_template_tokens(prompt_text)
            
            # Build evaluation prompt for the reward model
            eval_prompt = _build_evaluation_prompt(prompt_text, completion)
            
            # Call the API to get evaluation
            score = _call_reward_api(
                eval_prompt,
                api_endpoint=api_endpoint,
                api_key=api_key,
                api_model=api_model,
                logger=logger,
                debug=debug
            )
            
            rewards.append(score)
            
        except Exception as e:
            logger.warning(f"Failed to evaluate completion {i}: {str(e)}")
            rewards.append(0.0)
    
    return rewards




# Criteria definitions for evaluation
CRITERIA_HUMOR_PLAYFULNESS = """Humor & Playfulness:
- **Light-heartedness**: Cheerful, upbeat tone that avoids being overly serious
- **Wit**: Clever wordplay, amusing observations, or entertaining delivery
- **Fun-loving nature**: Enthusiasm for enjoyable activities and playful interactions
- **Personality consistency**: Humor that feels authentic to the character, not forced"""

CRITERIA_CREATIVITY = """Creativity and Imaginative Expression:
- **Original Ideas:** Novel concepts, unexpected connections, or fresh perspectives
- **Vivid Imagery:** Rich, sensory language that creates immersive mental pictures
- **Character Voice:** Distinctive, authentic expression that goes beyond generic responses
- **Imaginative Details:** Creative elements, metaphors, or unexpected but fitting touches
- **Narrative Flair:** Unique storytelling techniques, creative structure, or engaging presentation
- **Reader Engagement:** Captivating content that surprises and delights"""

CRITERIA_PERSONA_CONSISTENCY = """Persona Consistency:
- **Character Voice:** Distinctive speaking style, vocabulary, and expression patterns
- **Personality Traits:** Core behavioral characteristics and emotional tendencies
- **Motivations:** Actions and responses aligned with character goals/background
- **Tone Consistency:** Maintained emotional register throughout the response
- **Behavioral Authenticity:** Responses feel natural for this specific character"""


def _detect_evaluation_criterion(prompt: str) -> str:
    """
    Detect which evaluation criterion to use based on the prompt content.
    
    Args:
        prompt: The original prompt text
        
    Returns:
        The appropriate criteria string for evaluation
    """
    prompt_lower = prompt.lower()
    
    # Keywords for humor & playfulness
    humor_keywords = [
        'Humor & Playfulness'
    ]
    
    # Keywords for creativity
    creativity_keywords = [
        'Creativity and Imaginative Expression'
    ]
    
    # Keywords for persona consistency
    persona_keywords = [
        'Persona Consistency'
    ]
    
    # Check if prompt contains one of the criterion phrases
    if any(kw in prompt_lower for kw in humor_keywords):
        return CRITERIA_HUMOR_PLAYFULNESS
    elif any(kw in prompt_lower for kw in creativity_keywords):
        return CRITERIA_CREATIVITY
    elif any(kw in prompt_lower for kw in persona_keywords):
        return CRITERIA_PERSONA_CONSISTENCY
    else:
        # Default to creativity if no criterion phrase detected
        return CRITERIA_CREATIVITY


def _build_evaluation_prompt(prompt: str, completion: str) -> str:
    """
    Build the evaluation prompt for the reward model.
    
    Automatically detects which criterion to use based on the prompt content.
    
    Args:
        prompt: The original prompt
        completion: The model's completion to evaluate
        
    Returns:
        Formatted evaluation prompt
    """
    # Detect the appropriate criterion based on prompt content
    criterion = _detect_evaluation_criterion(prompt)
    
    evaluation_prompt = f"""You are an expert evaluator. Rate the quality of the following response on a scale of 0 to 10.

Evaluate based on this criterion:
{criterion}

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


def _strip_chat_template_tokens(text: str) -> str:
    """
    Remove common chat template control tokens from a prompt string.
    
    This avoids leaking special tokens like <|im_start|> and <|im_end|>
    into evaluation prompts for API-based scoring.
    """
    if not text:
        return text
    
    cleaned = (
        text.replace("<|im_start|>", "")
            .replace("<|im_end|>", "")
    )
    
    # Remove common role labels that appear as standalone lines
    for role in ("system", "user", "assistant"):
        cleaned = cleaned.replace(f"{role}\n", "")
    
    return cleaned


def _call_reward_api(
    eval_prompt: str,
    api_endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
    api_model: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    debug: bool = False
) -> float:
    """
    Call the reward API to get an evaluation score.
    
    Uses api_utils.generate_response_by_api for multi-provider support
    (OpenAI, Google/Gemini, Anthropic, and custom hosted LLMs).
    
    Args:
        eval_prompt: The evaluation prompt to send
        api_endpoint: The API endpoint URL
        api_key: API key for authentication
        api_model: The model name to use for the API call
        logger: Optional logger
        
    Returns:
        Normalized score between 0.0 and 1.0
    """
    if not api_endpoint or not api_key:
        if logger:
            logger.warning("No API endpoint or key provided for subjective reward")
        return 0.0
    
    try:
        # Use the model from config if provided, otherwise use default
        model_name = api_model if api_model else "default"
        config = RewardAPIConfig(api_endpoint, api_key, model=model_name, debug=debug)
        
        # Log reward API configuration for debugging
        if debug and logger:
            logger.debug(f"Reward API Endpoint: {api_endpoint}")
            logger.debug(f"Reward API Model: {config.model}")
            logger.debug(f"Reward API max_new_tokens: {config.max_new_tokens}")
            logger.debug(f"Reward API temperature: {config.temperature}")
        
        response_text = generate_response_by_api(eval_prompt, config)
        
        # Log the LLM response for debugging
        if debug and logger:
            logger.debug(f"Reward LLM Evaluation Prompt:\n{eval_prompt}")
            logger.debug(f"Reward LLM Response: {response_text}")
        
        # Extract numeric score from response
        score = _parse_score(response_text)
        
        # Normalize to 0-1 range
        normalized_score = score / 10.0
        
        # Log the parsed score for debugging
        if debug and logger:
            logger.debug(f"Reward LLM Parsed Score: {score}/10 -> Normalized: {normalized_score:.4f}")
        
        return max(0.0, min(1.0, normalized_score))
        
    except Exception as e:
        if logger:
            logger.warning(f"Error calling reward API: {str(e)}")
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

