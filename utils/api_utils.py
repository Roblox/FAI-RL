"""API utility functions for making HTTP requests."""
import json
import logging
import re
import requests
from typing import Dict, Any, List, Union

# Set up logger for API utilities
logger = logging.getLogger(__name__)


# ============================================================================
# Helper Functions for API Calls
# ============================================================================

def _extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    Extract JSON from text that may contain markdown formatting or other content.
    
    This function handles cases where the LLM returns formatted text with JSON embedded,
    such as in markdown code blocks or at the end of explanatory text.
    
    Args:
        text: The response text that contains JSON
        
    Returns:
        Parsed JSON as a dictionary
        
    Raises:
        json.JSONDecodeError: If no valid JSON can be extracted
    """
    # First, try to parse the entire text as JSON (backwards compatibility)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try to extract JSON from markdown code blocks (```json ... ``` or ``` ... ```)
    # Look for content within triple backticks
    code_block_patterns = [
        r'```json\s*\n(.*?)\n```',  # ```json ... ```
        r'```\s*\n(.*?)\n```',       # ``` ... ```
    ]
    
    for pattern in code_block_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue
    
    # Try to find JSON-like patterns (objects starting with { and ending with })
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_pattern, text)
    
    # Try matches from last to first (JSON is often at the end)
    for match in reversed(matches):
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue
    
    # If nothing worked, raise an error with the original text
    raise json.JSONDecodeError(f"Could not extract valid JSON from text: {text[:200]}...", text, 0)


def _build_google_request_data(prompt: str, config) -> dict:
    """Build request data for Google/Gemini models."""
    generation_config = {
        "maxOutputTokens": getattr(config, 'max_new_tokens', 1000)
    }
    
    # Add optional parameters if they exist
    if hasattr(config, 'temperature'):
        generation_config["temperature"] = config.temperature
    if hasattr(config, 'top_p'):
        generation_config["topP"] = config.top_p
    
    return {
        "contents": [{
            "role": "user",
            "parts": [{"text": prompt}]
        }],
        "generationConfig": generation_config
    }


def _build_openai_request_data(prompt: str, config) -> dict:
    """Build request data for OpenAI/ChatGPT models."""
    data = {
        "model": config.model,
        "messages": [{"content": prompt, "role": "user"}]
    }
    
    # Add optional parameters if they exist
    if hasattr(config, 'max_new_tokens'):
        data["max_tokens"] = config.max_new_tokens
    if hasattr(config, 'temperature'):
        data["temperature"] = config.temperature
    if hasattr(config, 'top_p'):
        data["top_p"] = config.top_p
    
    return data


def _build_anthropic_request_data(prompt: str, config) -> dict:
    """Build request data for Anthropic/Claude models."""
    data = {
        "model": config.model,
        "max_tokens": getattr(config, 'max_new_tokens', 1000),
        "messages": [{"content": prompt, "role": "user"}]
    }
    
    # Add optional parameters if they exist
    if hasattr(config, 'temperature'):
        data["temperature"] = config.temperature
    if hasattr(config, 'top_p'):
        data["top_p"] = config.top_p
    
    return data


def _build_default_request_data(prompt: str, config) -> dict:
    """Build request data for other models (generic format)."""
    return {
        "model": config.model,
        "max_tokens": getattr(config, 'max_new_tokens', 1000),
        "temperature": getattr(config, 'temperature', 1.0),
        "messages": [{"content": prompt, "role": "user"}]
    }


def _get_model_provider(api_endpoint: str) -> str:
    """
    Determine the provider from the API endpoint URL.
    
    Args:
        api_endpoint: The API endpoint URL
        
    Returns:
        Provider name: "google", "openai", "anthropic", or "default"
    """
    endpoint_lower = api_endpoint.lower()
    
    # Check for Google/Gemini endpoints
    if "generativelanguage.googleapis.com" in endpoint_lower or "google" in endpoint_lower:
        return "google"
    
    # Check for OpenAI endpoints
    elif "api.openai.com" in endpoint_lower or "openai" in endpoint_lower:
        return "openai"
    
    # Check for Anthropic endpoints
    elif "api.anthropic.com" in endpoint_lower or "anthropic" in endpoint_lower:
        return "anthropic"
    
    # Default for custom/other providers
    else:
        return "default"


def _make_api_request(url: str, headers: dict, data: dict) -> requests.Response:
    """Make the HTTP request to the API endpoint."""
    provider = _get_model_provider(url)
    
    if provider == "google":
        # Google API requires data to be JSON string in body
        return requests.post(url, headers=headers, data=json.dumps(data))
    else:
        # OpenAI, Anthropic, and other APIs use json parameter
        return requests.post(url, headers=headers, json=data)


def _parse_api_response(response_json: dict, api_endpoint: str) -> str:
    """Extract the response text from the API response JSON."""
    provider = _get_model_provider(api_endpoint)
    
    try:
        if provider == "google":
            # Google/Gemini response format
            return response_json['candidates'][0]['content']['parts'][0]['text']
        elif provider == "anthropic":
            # Anthropic/Claude response format
            return response_json['content'][0]['text']
        else:
            # OpenAI and default format
            return response_json['choices'][0]['message']['content']
    except (KeyError, IndexError, TypeError) as e:
        logger.warning(f"Failed to parse API response: {e}")
        return ""


# ============================================================================
# Public API Functions
# ============================================================================

def _build_headers(api_endpoint: str, api_key: str) -> dict:
    """Build appropriate headers for each API provider."""
    provider = _get_model_provider(api_endpoint)
    
    if provider == "google":
        # Google/Gemini uses API key in URL parameter, not in headers
        # But we still need Content-Type
        return {
            "Content-Type": "application/json"
        }
    elif provider == "openai":
        # OpenAI/ChatGPT uses Bearer token
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
    elif provider == "anthropic":
        # Anthropic/Claude uses x-api-key header and requires anthropic-version
        return {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01"
        }
    else:
        # Default format (generic)
        return {
            "Content-Type": "application/json",
            "Authorization": api_key
        }


def _prepare_api_url(api_endpoint: str, api_key: str) -> str:
    """Prepare the API URL with necessary query parameters."""
    provider = _get_model_provider(api_endpoint)
    
    if provider == "google":
        # Google/Gemini API key goes in the URL
        separator = "&" if "?" in api_endpoint else "?"
        return f"{api_endpoint}{separator}key={api_key}"
    else:
        # Other providers use headers for authentication
        return api_endpoint


def generate_response_by_api(
    prompt: str,
    config
) -> Union[Dict[str, Any], str]:
    """
    Generate response using API-based inference.
    
    This is the main function for calling LLM APIs to generate text responses.
    Supports OpenAI/ChatGPT, Google/Gemini, and Anthropic/Claude.
    Provider is automatically detected from the api_endpoint URL.
    
    Args:
        prompt: The input prompt text
        config: Configuration object with model, api_endpoint, api_key, etc.
        
    Returns:
        Generated response text from the API
    """
    from utils.config_validation import validate_api_config
    
    validate_api_config(config)

    try:
        # Prepare URL (Google needs API key in URL)
        url = _prepare_api_url(config.api_endpoint, config.api_key)
        
        # Set up headers based on provider (detected from endpoint)
        headers = _build_headers(config.api_endpoint, config.api_key)
        
        # Build request data based on provider type (detected from endpoint)
        provider = _get_model_provider(config.api_endpoint)
        if provider == "google":
            data = _build_google_request_data(prompt, config)
        elif provider == "openai":
            data = _build_openai_request_data(prompt, config)
        elif provider == "anthropic":
            data = _build_anthropic_request_data(prompt, config)
        else:
            data = _build_default_request_data(prompt, config)
        
        # Make the API request
        response = _make_api_request(url, headers, data)
        response.raise_for_status()
        
        # Parse and return the response
        response_json = response.json()
        return _parse_api_response(response_json, config.api_endpoint)
        
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {str(e)}")
        return ""


def generate_response_by_api_for_reward_function(
    prompt: str,
    completions: List[str],
    api_endpoint: str,
    api_key: str,
    model: str,
    max_new_tokens: int = 8192,
    temperature: float = 0.2,
    timeout: int = 30,
    logger_instance = None
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
        logger_instance: Optional logger instance to use for logging
    """
    # Use passed logger if available, otherwise fall back to module logger
    _logger = logger_instance if logger_instance else logger
    
    # Create SYSTEM_PROMPT that includes all responses for evaluation
    SYSTEM_PROMPT = """You are an expert evaluator. Given a prompt and multiple responses, identify the best and worst responses.

Prompt:
{prompt}

---
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
    formatted_responses = "\n\n---\n\n".join([
        f"Response {i}:\n{completion}"
        for i, completion in enumerate(completions)
    ])
    
    # Build the evaluation prompt with all responses included
    evaluation_prompt = SYSTEM_PROMPT.format(
        prompt=prompt,
        responses=formatted_responses
    )
    
    # Prepare URL (Google needs API key in URL)
    url = _prepare_api_url(api_endpoint, api_key)
    
    # Set up headers based on provider (detected from endpoint)
    headers = _build_headers(api_endpoint, api_key)
    
    # Create a temporary config object for building request data
    temp_config = type('Config', (), {
        'model': model,
        'max_new_tokens': max_new_tokens,
        'temperature': temperature
    })()
    
    # Build request data based on provider type (detected from endpoint)
    provider = _get_model_provider(api_endpoint)
    if provider == "google":
        data = _build_google_request_data(evaluation_prompt, temp_config)
    elif provider == "openai":
        data = _build_openai_request_data(evaluation_prompt, temp_config)
    elif provider == "anthropic":
        data = _build_anthropic_request_data(evaluation_prompt, temp_config)
    else:
        data = _build_default_request_data(evaluation_prompt, temp_config)
    
    _logger.info(f"API Request Data: {data}")
    
    try:
        # Make the API request (same pattern as generate_response_by_api)
        response = _make_api_request(url, headers, data)
        response.raise_for_status()
        
        _logger.info(f"API Response Status: {response.status_code}")
        _logger.info(f"API Response Content: {response.text}")
        
        # Parse and return the response (same pattern as generate_response_by_api)
        response_json = response.json()
        response_text = _parse_api_response(response_json, api_endpoint)
        
        # Extract JSON from the response text (handles markdown formatting)
        try:
            return _extract_json_from_text(response_text)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse evaluation response as JSON: {response_text}")
            
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"API evaluation failed: {str(e)}")

