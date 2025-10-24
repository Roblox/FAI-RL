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


def is_unverifiable_domain_dataset(dataset_name: str) -> bool:
    """Check if dataset is an unverifiable/subjective domain dataset.
    
    Unverifiable datasets include:
    - Subjective tasks (creative writing, general conversation, etc.)
    - Datasets that cannot be verified with exact match or rule-based checks
    - Custom datasets like FAI-RL datasets
    
    Args:
        dataset_name: The name of the dataset
        
    Returns:
        True if dataset is unverifiable/subjective, False if it's a verifiable domain
    """
    # FAI-RL datasets are explicitly unverifiable/subjective
    if "FAI-RL" in dataset_name:
        return True
    
    # Any dataset not recognized as math/verifiable is treated as unverifiable
    return not is_math_dataset(dataset_name)


def get_template_for_dataset(dataset_name: str, logger=None):
    """Get the appropriate template class for a given dataset.
    
    Args:
        dataset_name: The name of the dataset
        logger: Optional logger for warnings
        
    Returns:
        The appropriate template class (GSM8KTemplate, OpenMathInstructTemplate, 
        or UnverifiableDomainTemplate)
    """
    # Import templates here to avoid circular dependencies
    from trainers.templates.gsm8k_template import GSM8KTemplate
    from trainers.templates.openmathinstruct_template import OpenMathInstructTemplate
    from trainers.templates.subjective_template import UnverifiableDomainTemplate
    
    if dataset_name == "openai/gsm8k":
        return GSM8KTemplate
    elif dataset_name == "nvidia/OpenMathInstruct-2":
        return OpenMathInstructTemplate
    else:
        return UnverifiableDomainTemplate

