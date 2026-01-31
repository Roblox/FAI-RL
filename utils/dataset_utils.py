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


def is_subjective_dataset(dataset_name: str) -> bool:
    """Check if dataset requires subjective evaluation.
    
    Subjective datasets are those where responses are evaluated based on
    quality criteria rather than verifiable answers.
    
    Args:
        dataset_name: The name of the dataset
        
    Returns:
        True if dataset requires subjective evaluation, False otherwise
    """
    # Add your subjective dataset names here
    subjective_datasets = [
        # Example: "Anthropic/hh-rlhf",
        # Example: "OpenAssistant/oasst1",
    ]
    return dataset_name in subjective_datasets


def get_template_for_dataset(dataset_name: str, logger=None, use_subjective: bool = False):
    """Get the appropriate template class for a given dataset.
    
    Args:
        dataset_name: The name of the dataset
        logger: Optional logger for warnings
        use_subjective: If True, use SubjectiveTemplate for unknown datasets
        
    Returns:
        The appropriate template class (GSM8KTemplate, OpenMathInstructTemplate, or SubjectiveTemplate)
        
    Raises:
        ValueError: If dataset is not supported and use_subjective is False
    """
    # Import templates here to avoid circular dependencies
    from trainers.templates.gsm8k_template import GSM8KTemplate
    from trainers.templates.openmathinstruct_template import OpenMathInstructTemplate
    from trainers.templates.subjective_template import SubjectiveTemplate
    
    if dataset_name == "openai/gsm8k":
        return GSM8KTemplate
    elif dataset_name == "nvidia/OpenMathInstruct-2":
        return OpenMathInstructTemplate
    elif is_subjective_dataset(dataset_name) or use_subjective:
        return SubjectiveTemplate
    else:
        raise ValueError(
            f"Dataset '{dataset_name}' is not supported. "
            f"Supported datasets: 'openai/gsm8k', 'nvidia/OpenMathInstruct-2'. "
            f"For subjective evaluation datasets, set use_subjective=True."
        )

