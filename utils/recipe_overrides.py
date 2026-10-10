"""
Utility functions for handling recipe overrides from command line.
"""

import ast
import yaml
from typing import Any, Dict

from utils.logging_utils import setup_logging

# Module-level logger. setup_logging() attaches a RankFilter, so INFO/DEBUG
# records are automatically dropped on non-rank-0 workers; WARNING+ still
# passes on every rank. file_output=False because the parent training log
# already captures stdout (a second file would double-log under nohup).
logger = setup_logging("FAI-RL.recipe", file_output=False)


def parse_value(value_str: str) -> Any:
    """Parse a string value to its appropriate Python type."""
    # Try to evaluate as Python literal (handles int, float, bool, list, dict, etc.)
    try:
        return ast.literal_eval(value_str)
    except (ValueError, SyntaxError):
        # If it fails, return as string
        return value_str


def set_nested_value(recipe_dict: Dict, key_path: str, value: Any) -> None:
    """Set a value in a nested dictionary using dot notation.
    
    Example: 
        set_nested_value(recipe, "model.base_model_name", "llama")
        sets recipe["model"]["base_model_name"] = "llama"
    """
    keys = key_path.split('.')
    current = recipe_dict
    
    # Navigate to the nested location
    for key in keys[:-1]:
        if not isinstance(current, dict):
            raise ValueError(f'Invalid config override: "{key_path}" traverses a non-mapping value.')
        if key not in current:
            current[key] = {}
        current = current[key]
    
    # Set the final value
    if not isinstance(current, dict) or any(not key for key in keys):
        raise ValueError(f'Invalid config override: "{key_path}" requires a mapping and non-empty keys.')
    current[keys[-1]] = value


def apply_overrides_to_recipe(recipe_dict: Dict, overrides: list) -> Dict:
    """Apply command-line overrides to a recipe dictionary.
    
    Args:
        recipe_dict: Base recipe dictionary
        overrides: List of override strings in key=value format
        
    Returns:
        Updated recipe dictionary
    """
    if overrides:
        logger.info("Applying command-line overrides:")
        for override in overrides:
            if '=' not in override:
                logger.warning("Skipping invalid override %r (expected key=value format)", override)
                continue
            
            key, value_str = override.split('=', 1)
            value = parse_value(value_str)
            set_nested_value(recipe_dict, key, value)
            logger.info("  %s = %r", key, value)
    
    return recipe_dict


def load_recipe_from_yaml(yaml_path: str) -> Dict:
    """Load recipe from YAML file.
    
    Args:
        yaml_path: Path to YAML recipe file
        
    Returns:
        Recipe dictionary
    """
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            recipe_dict = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        location = f" at line {mark.line + 1}, column {mark.column + 1}" if mark else ""
        raise ValueError(f"Invalid YAML config in {yaml_path}{location}; check indentation, colons, and quoting.") from None
    if not isinstance(recipe_dict, dict):
        raise ValueError(f"Invalid config in {yaml_path}: expected a YAML mapping, not {type(recipe_dict).__name__}.")
    logger.info("Loaded base recipe from: %s", yaml_path)
    return recipe_dict

