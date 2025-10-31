"""Dataset utilities for handling different dataset types and templates."""

from typing import Type, Any


def is_math_dataset(dataset_name: str) -> bool:
    """Check if dataset is a math/verifiable reasoning dataset.
    
    Args:
        dataset_name: The name of the dataset (e.g., "openai/gsm8k")
        
    Returns:
        True if dataset is a recognized math/reasoning dataset, False otherwise
    """
    math_datasets = [
        "openai/gsm8k",
        "nvidia/OpenMathInstruct-2",
    ]
    return dataset_name in math_datasets


def get_template_for_dataset(dataset_name: str, logger=None):
    """Get the appropriate template class for a given dataset.
    
    Args:
        dataset_name: The name of the dataset
        logger: Optional logger for warnings
        
    Returns:
        The appropriate template class (GSM8KTemplate or OpenMathInstructTemplate)
        
    Raises:
        ValueError: If dataset is not supported
    """
    # Import templates here to avoid circular dependencies
    from trainers.templates.gsm8k_template import GSM8KTemplate
    from trainers.templates.openmathinstruct_template import OpenMathInstructTemplate
    
    if dataset_name == "openai/gsm8k":
        return GSM8KTemplate
    elif dataset_name == "nvidia/OpenMathInstruct-2":
        return OpenMathInstructTemplate
    else:
        raise ValueError(
            f"Dataset '{dataset_name}' is not supported. "
            f"Supported datasets: 'openai/gsm8k', 'nvidia/OpenMathInstruct-2'"
        )

