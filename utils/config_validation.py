"""Validation utilities for configuration parameters."""


def validate_api_endpoint(api_endpoint: str) -> None:
    """
    Validate that the API endpoint is not a placeholder.
    
    Args:
        api_endpoint: The API endpoint to validate
        
    Raises:
        ValueError: If the API endpoint is still set to a placeholder value
    """
    if api_endpoint and "<YOUR_API_ENDPOINT>" in api_endpoint:
        raise ValueError(
            "Error: api_endpoint is still set to the placeholder '<YOUR_API_ENDPOINT>'. "
            "Please replace it with your actual API endpoint in the configuration file."
        )


def validate_api_key(api_key: str) -> None:
    """
    Validate that the API key is not a placeholder.
    
    Args:
        api_key: The API key to validate
        
    Raises:
        ValueError: If the API key is still set to a placeholder value
    """
    if api_key and api_key == "<YOUR_API_KEY>":
        raise ValueError(
            "Error: api_key is still set to the placeholder '<YOUR_API_KEY>'. "
            "Please replace it with your actual API key in the configuration file."
        )


def validate_api_config(config) -> None:
    """
    Validate both API endpoint and API key from a configuration object.
    
    Args:
        config: Configuration object with api_endpoint and api_key attributes
        
    Raises:
        ValueError: If any API configuration is still set to placeholder values
    """
    # Validate API endpoint if present
    if hasattr(config, 'api_endpoint') and config.api_endpoint:
        validate_api_endpoint(config.api_endpoint)
    
    # Validate API key if present
    if hasattr(config, 'api_key') and config.api_key:
        validate_api_key(config.api_key)


def validate_structured_output_config(config) -> None:
    """
    Validate and normalize an inference config's json_schema, when set.

    Args:
        config: Inference configuration object; json_schema is replaced by its parsed dict

    Raises:
        ValueError: If the schema is invalid or combined with an unsupported setting
    """
    if getattr(config, 'json_schema', None) is None:
        return

    from utils.structured_output import parse_json_schema

    config.json_schema = parse_json_schema(config.json_schema)

    if getattr(config, 'enable_thinking', None):
        raise ValueError(
            "json_schema cannot be combined with enable_thinking: true, because constrained "
            "decoding forces JSON from the first generated token. Set enable_thinking: false."
        )

    if getattr(config, 'model', None) is not None and getattr(config, 'api_key', None) is not None:
        raise ValueError(
            "json_schema is only supported for local model inference (model_paths or a "
            "HuggingFace model), not API endpoints."
        )
