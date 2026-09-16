"""Validation utilities for configuration parameters."""

from dataclasses import MISSING, fields
from difflib import get_close_matches
from typing import Any, Dict, Literal, Union, get_args, get_origin


def _matches_type(value: Any, expected: Any) -> bool:
    """Check recipe types without coercing values or importing ML packages."""
    origin, args = get_origin(expected), get_args(expected)
    if expected is Any:
        return True
    if origin is Union:
        return any(_matches_type(value, item) for item in args)
    if origin is Literal:
        return value in args
    if origin is list:
        return isinstance(value, list) and all(_matches_type(item, args[0]) for item in value)
    if origin is dict:
        return isinstance(value, dict)
    if expected is float:
        # YAML 1.1 often parses unadorned scientific notation as a string;
        # existing recipes/TrainingArguments accept these numeric strings.
        if isinstance(value, str):
            try:
                float(value)
                return True
            except ValueError:
                return False
        return type(value) in (int, float)
    if expected in (bool, int):
        return type(value) is expected
    return isinstance(value, expected)


def validate_config_section(values: Dict[str, Any], schema: type, path: str) -> None:
    """Report unknown keys, missing fields, and wrong types using full key paths."""
    if not isinstance(values, dict):
        raise ValueError(
            f'Invalid config: "{path}" must be a mapping, not {type(values).__name__}.'
        )
    available = {item.name: item for item in fields(schema)}
    for key, value in values.items():
        if key not in available:
            aliases = {
                "lr": "learning_rate",
                "num_epochs": "num_train_epochs",
                "batch_size": "per_device_train_batch_size",
            }
            matches = get_close_matches(str(key), available, n=1)
            suggestion = aliases.get(key) or (matches[0] if matches else None)
            hint = f'\nDid you mean "{path}.{suggestion}"?' if suggestion in available else ""
            raise ValueError(
                f'Invalid config: found unknown key "{path}.{key}".{hint}\nAvailable keys: '
                + ", ".join(sorted(available))
            )
        # DatasetInfo mappings are validated separately with indexed paths.
        if path == "data" and key == "datasets":
            if not isinstance(value, list):
                raise ValueError('Invalid config: "data.datasets" must be a list.')
            continue
        if not _matches_type(value, available[key].type):
            raise ValueError(
                f'Invalid config: "{path}.{key}" expected {available[key].type}; found {type(value).__name__}.'
            )
    for key, item in available.items():
        if item.default is MISSING and item.default_factory is MISSING and key not in values:
            raise ValueError(
                f'Invalid config: "{path}.{key}" is required.\nAvailable keys: '
                + ", ".join(sorted(available))
            )


def validate_training_recipe(recipe: Dict[str, Any]) -> None:
    """Validate training sections against the existing dataclass schema."""
    from core.config import (
        DataConfig,
        DatasetInfo,
        LocalRewardFunctionConfig,
        ModelConfig,
        RewardAPIConfig,
        S3Config,
        TrainingConfig,
        WandbConfig,
    )

    if not isinstance(recipe, dict):
        raise ValueError(
            "Invalid config: expected a YAML mapping with model, data, and training sections."
        )
    schemas = {
        "model": ModelConfig,
        "data": DataConfig,
        "training": TrainingConfig,
        "wandb": WandbConfig,
        "s3": S3Config,
        "reward_api": RewardAPIConfig,
        "local_reward_function": LocalRewardFunctionConfig,
    }
    for name, schema in schemas.items():
        if name in recipe:
            if recipe[name] is None and name in {"reward_api", "local_reward_function"}:
                continue
            validate_config_section(recipe[name], schema, name)
        elif name in {"model", "training"}:
            validate_config_section({}, schema, name)
    for index, dataset in enumerate(recipe.get("data", {}).get("datasets", [])):
        if not isinstance(dataset, DatasetInfo):
            validate_config_section(dataset, DatasetInfo, f"data.datasets[{index}]")
    training = recipe.get("training", {})
    algorithms = {"cpt", "sft", "sft_vlm", "dpo", "grpo", "gspo"}
    if (training.get("algorithm") or "").lower() not in algorithms:
        raise ValueError(
            "Invalid config: training.algorithm must be one of " + ", ".join(sorted(algorithms))
        )


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
    if hasattr(config, "api_endpoint") and config.api_endpoint:
        validate_api_endpoint(config.api_endpoint)

    # Validate API key if present
    if hasattr(config, "api_key") and config.api_key:
        validate_api_key(config.api_key)
