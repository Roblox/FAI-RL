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
        if key not in current:
            current[key] = {}
        current = current[key]
    
    # Set the final value
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
    with open(yaml_path, 'r') as f:
        recipe_dict = yaml.safe_load(f)
    logger.info("Loaded base recipe from: %s", yaml_path)
    return recipe_dict

