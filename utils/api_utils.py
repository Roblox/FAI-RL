"""API utility functions for making HTTP requests."""
import json
import requests
from typing import Dict, Any, List, Union


# ============================================================================
# Helper Functions for API Calls
# ============================================================================

def _build_google_request_data(prompt: str, config) -> dict:
    """Build request data for Google/Gemini models."""
    return {
        "contents": {
            "role": "user",
            "parts": [{"text": prompt}]
        },
        "generationConfig": {
            "maxOutputTokens": config.max_new_tokens
        }
    }


def _build_openai_request_data(prompt: str, config) -> dict:
    """Build request data for OpenAI models."""
    return {
        "model": config.model,
        "messages": [{"content": prompt, "role": "user"}]
    }


def _build_default_request_data(prompt: str, config) -> dict:
    """Build request data for other models (Anthropic, etc.)."""
    return {
        "model": config.model,
        "max_tokens": config.max_new_tokens,
        "temperature": config.temperature,
        "messages": [{"content": prompt, "role": "user"}]
    }


def _make_api_request(url: str, headers: dict, data: dict, model: str) -> requests.Response:
    """Make the HTTP request to the API endpoint."""
    if model.startswith("google/"):
        # Google API requires data to be JSON string in body
        return requests.post(url, headers=headers, data=json.dumps(data))
    else:
        # Other APIs can use json parameter
        return requests.post(url, headers=headers, json=data)


def _parse_api_response(response_json: dict, model: str) -> str:
    """Extract the response text from the API response JSON."""
    try:
        if model.startswith("google/"):
            return response_json['candidates'][0]['content']['parts'][0]['text']
        else:
            return response_json['choices'][0]['message']['content']
    except (KeyError, IndexError, TypeError):
        return ""


# ============================================================================
# Public API Functions
# ============================================================================

def generate_response_by_api(
    prompt: str,
    config
) -> Union[Dict[str, Any], str]:
    """
    Generate response using API-based inference.
    
    This is the main function for calling LLM APIs to generate text responses.
    Moved from inference/inference.py to utils for reusability.
    
    Args:
        prompt: The input prompt text
        config: Configuration object with model, api_endpoint, api_key, etc.
        
    Returns:
        Generated response text from the API
    """
    from utils.config_validation import validate_api_config
    
    validate_api_config(config)

    try:
        # Get the appropriate API endpoint
        api_endpoint = getattr(config, 'api_endpoint', None)
        url = config.api_endpoint
        
        # Set up headers
        headers = {
            "Content-Type": "application/json",
            "Authorization": config.api_key
        }
        
        # Build request data based on model type
        if config.model.startswith("google/"):
            data = _build_google_request_data(prompt, config)
        elif config.model.startswith("openai/"):
            data = _build_openai_request_data(prompt, config)
        else:
            data = _build_default_request_data(prompt, config)
        
        # Make the API request
        response = _make_api_request(url, headers, data, config.model)
        response.raise_for_status()
        
        # Parse and return the response
        response_json = response.json()
        return _parse_api_response(response_json, config.model)
        
    except requests.exceptions.RequestException as e:
        return ""


def generate_response_by_api_for_reward_function(
    prompt: str,
    completions: List[str],
    api_endpoint: str,
    api_key: str,
    model: str,
    max_new_tokens: int = 8192,
    temperature: float = 0.2,
    timeout: int = 30
) -> Dict[str, Any]:
    """
    Call external API to evaluate multiple completions and identify best/worst.
    
    This function is specifically for evaluation/reward purposes, calling
    an evaluation API that ranks completions.
    
    Args:
        prompt: The original prompt that generated the completions
        completions: List of model-generated completions to evaluate
        api_endpoint: URL of the evaluation API
        api_key: API key for authentication
        model: Model identifier for the evaluation API
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        timeout: Request timeout in seconds
    """
    # Create SYSTEM_PROMPT that includes all responses for evaluation
    SYSTEM_PROMPT = """You are an expert evaluator. Given a prompt and multiple responses, identify the best and worst responses.

Original Prompt:
{prompt}

Responses to Evaluate:
{responses}

Instructions:
- Analyze all responses carefully
- Identify the index (0-based) of the BEST response
- Identify the index (0-based) of the WORST response
- Return your evaluation as JSON with keys: "best_idx" and "worst_idx"

Example output format:
{{"best_idx": 2, "worst_idx": 0}}"""

    # Format responses for inclusion in the prompt
    formatted_responses = "\n\n".join([
        f"Response {i}:\n{completion}"
        for i, completion in enumerate(completions)
    ])
    
    # Build the evaluation prompt with all responses included
    evaluation_prompt = SYSTEM_PROMPT.format(
        prompt=prompt,
        responses=formatted_responses
    )
    
    # Set up headers
    headers = {
        'Content-Type': 'application/json',
        'Authorization': api_key
    }
    
    # Build request data based on model type (same pattern as generate_response_by_api)
    if model.startswith("google/"):
        data = _build_google_request_data(evaluation_prompt, type('Config', (), {
            'max_new_tokens': max_new_tokens
        })())
    elif model.startswith("openai/"):
        data = _build_openai_request_data(evaluation_prompt, type('Config', (), {
            'model': model
        })())
    else:
        data = _build_default_request_data(evaluation_prompt, type('Config', (), {
            'model': model,
            'max_new_tokens': max_new_tokens,
            'temperature': temperature
        })())
    
    try:
        # Make the API request (same pattern as generate_response_by_api)
        response = _make_api_request(api_endpoint, headers, data, model)
        response.raise_for_status()
        
        # Parse and return the response (same pattern as generate_response_by_api)
        response_json = response.json()
        response_text = _parse_api_response(response_json, model)
        
        # Parse the JSON response text for evaluation results
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            raise RuntimeError(f"Failed to parse evaluation response as JSON: {response_text}")
            
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"API evaluation failed: {str(e)}")

