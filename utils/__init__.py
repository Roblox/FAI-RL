"""Utility modules."""

from .logging_utils import (
    setup_logging,
    TrainingLogger,
    log_system_info,
    log_gpu_memory,
    SafeLogger,
    RobustFileHandler,
)
from .config_validation import validate_api_endpoint, validate_api_key, validate_api_config
from .dataset_utils import (
    format_multiple_choice_for_inference,
    is_math_dataset,
    get_template_for_dataset,
)


def __getattr__(name: str):
    """Keep diagnostics importable when optional runtime dependencies are broken."""
    from importlib import import_module

    if name == "generate_response_by_api":
        value = getattr(import_module(".api_utils", __name__), name)
    elif name in __all__:
        value = getattr(import_module(".device_utils", __name__), name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value

__all__ = [
    "setup_logging",
    "TrainingLogger",
    "log_system_info",
    "log_gpu_memory",
    "SafeLogger",
    "RobustFileHandler",
    "validate_api_endpoint",
    "validate_api_key",
    "validate_api_config",
    "generate_response_by_api",
    "format_multiple_choice_for_inference",
    "is_math_dataset",
    "get_template_for_dataset",
    # Device utilities
    "get_device_type",
    "get_device",
    "is_cuda_available",
    "is_mps_available",
    "is_apple_silicon",
    "get_optimal_dtype",
    "get_optimal_dtype_str",
    "supports_quantization",
    "supports_deepspeed",
    "get_device_count",
    "get_device_name",
    "get_device_memory_info",
    "log_device_info",
    "validate_device_compatibility",
    "adapt_config_for_device",
]

