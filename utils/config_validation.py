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

