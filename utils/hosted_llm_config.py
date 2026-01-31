"""
Custom configuration for hosted LLM API integration.

This file provides OpenAI-compatible defaults that work out-of-the-box with
most hosted LLMs that follow the OpenAI API format.

If your hosted LLM uses a different format, simply modify the functions below
to match your API's request/response structure.

DEFAULT BEHAVIOR:
- Request format: OpenAI chat completions format
- Response parsing: OpenAI choices[0].message.content format  
- Headers: Bearer token authentication
- URL: No query parameters (auth in headers)

CUSTOMIZATION:
- Each function includes commented examples for common variations
- Simply uncomment and modify the examples to match your API
- Or write your own custom implementation from scratch
"""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# CUSTOMIZE THESE FUNCTIONS FOR YOUR HOSTED LLM
# ============================================================================

def build_hosted_llm_request(prompt: str, config) -> Optional[Dict[str, Any]]:
    """
    Build custom request data for your hosted LLM.
    
    Return None to use default OpenAI-compatible format.
    Return a dictionary to use your custom format.
    
    Args:
        prompt: The input prompt text
        config: Configuration object with model, max_new_tokens, temperature, etc.
        
    Returns:
        Dictionary with your custom request format, or None for default
    """
    # ========================================================================
    # DEFAULT: OpenAI-compatible format (working example)
    # ========================================================================
    # This is a working implementation that matches OpenAI's API format.
    # Modify this if your hosted LLM uses a different format.
    
    data = {
        "model": config.model,
        "max_tokens": getattr(config, 'max_new_tokens', 1000),
        "temperature": getattr(config, 'temperature', 1.0),
        "messages": [{"content": prompt, "role": "user"}]
    }
    
    # Add optional parameters if they exist in config
    if hasattr(config, 'top_p'):
        data["top_p"] = config.top_p
    
    return data
    
    # ========================================================================
    # CUSTOM FORMAT EXAMPLES - Uncomment and modify for your hosted LLM
    # ========================================================================
    
    # Example 1: Simple format with "prompt" instead of "messages"
    # return {
    #     "prompt": prompt,
    #     "model": config.model,
    #     "max_tokens": getattr(config, 'max_new_tokens', 1000),
    #     "temperature": getattr(config, 'temperature', 1.0),
    # }
    
    # Example 2: Nested configuration
    # return {
    #     "input": prompt,
    #     "model": {
    #         "name": config.model,
    #         "parameters": {
    #             "max_tokens": getattr(config, 'max_new_tokens', 1000),
    #             "temperature": getattr(config, 'temperature', 1.0),
    #         }
    #     }
    # }
    
    # Example 3: Array-based format
    # return {
    #     "inputs": [prompt],
    #     "model_name": config.model,
    #     "generation_config": {
    #         "max_new_tokens": getattr(config, 'max_new_tokens', 1000),
    #         "temperature": getattr(config, 'temperature', 1.0),
    #     }
    # }


def parse_hosted_llm_response(response_json: Dict[str, Any]) -> Optional[str]:
    """
    Parse custom response from your hosted LLM.
    
    Return None to use default OpenAI-compatible parsing.
    Return a string to use your custom parsing.
    
    Args:
        response_json: The JSON response from your API
        
    Returns:
        The extracted text response, or None for default parsing
    """
    # ========================================================================
    # DEFAULT: OpenAI-compatible format (working example)
    # ========================================================================
    # This is a working implementation that matches OpenAI's API format.
    # Modify this if your hosted LLM uses a different response structure.
    
    try:
        return response_json['choices'][0]['message']['content']
    except (KeyError, IndexError, TypeError) as e:
        logger.warning(f"Failed to parse response with OpenAI format: {e}")
        if isinstance(response_json, dict):
            logger.warning(f"Response JSON: {response_json}")
        else:
            logger.warning(f"Response structure type: {type(response_json).__name__}")
        return ""
    
    # ========================================================================
    # CUSTOM FORMAT EXAMPLES - Uncomment and modify for your hosted LLM
    # ========================================================================
    
    # Example 1: Simple output field
    # return response_json.get('output', '')
    
    # Example 2: Nested response
    # return response_json.get('data', {}).get('text', '')
    
    # Example 3: Array response
    # return response_json.get('outputs', [''])[0]
    
    # Example 4: Different field name
    # return response_json.get('generated_text', '')


def build_hosted_llm_headers(api_key: str) -> Optional[Dict[str, str]]:
    """
    Build custom headers for your hosted LLM.
    
    Return None to use default generic headers.
    Return a dictionary to use your custom headers.
    
    Args:
        api_key: Your API key for authentication
        
    Returns:
        Dictionary with your custom headers, or None for default
    """
    # ========================================================================
    # DEFAULT: OpenAI-compatible format (working example)
    # ========================================================================
    # This is a working implementation that matches OpenAI's API format.
    # Modify this if your hosted LLM uses different headers.
    
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    
    # ========================================================================
    # CUSTOM FORMAT EXAMPLES - Uncomment and modify for your hosted LLM
    # ========================================================================
    
    # Example 1: Different auth header name
    # return {
    #     "Content-Type": "application/json",
    #     "X-API-Key": api_key,
    # }
    
    # Example 2: API key without Bearer prefix
    # return {
    #     "Content-Type": "application/json",
    #     "Authorization": api_key,
    # }
    
    # Example 3: Multiple headers
    # return {
    #     "Content-Type": "application/json",
    #     "Authorization": f"ApiKey {api_key}",
    #     "X-Client-Version": "1.0",
    # }
    
    # Example 4: No authentication (internal endpoint)
    # return {
    #     "Content-Type": "application/json",
    # }


def prepare_hosted_llm_url(api_endpoint: str, api_key: str) -> Optional[str]:
    """
    Prepare custom URL with query parameters for your hosted LLM.
    
    Return None to use the endpoint as-is.
    Return a string to use your custom URL with query parameters.
    
    Args:
        api_endpoint: The base API endpoint URL
        api_key: Your API key (if needed in URL)
        
    Returns:
        Modified URL with query parameters, or None for default
    """
    # ========================================================================
    # DEFAULT: OpenAI-compatible format (working example)
    # ========================================================================
    # OpenAI uses authentication in headers, not URL parameters.
    # Most hosted LLMs follow this pattern, so we return the endpoint as-is.
    
    return api_endpoint
    
    # ========================================================================
    # CUSTOM FORMAT EXAMPLES - Uncomment and modify for your hosted LLM
    # ========================================================================
    
    # Example 1: API key in URL (like Google Gemini)
    # separator = "&" if "?" in api_endpoint else "?"
    # return f"{api_endpoint}{separator}api_key={api_key}"
    
    # Example 2: Multiple query parameters
    # separator = "&" if "?" in api_endpoint else "?"
    # return f"{api_endpoint}{separator}token={api_key}&version=v1"
    
    # Example 3: Add version to URL (without API key)
    # separator = "&" if "?" in api_endpoint else "?"
    # return f"{api_endpoint}{separator}v=2023-01"

