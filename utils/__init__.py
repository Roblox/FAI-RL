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
from .api_utils import generate_response_by_api
from .dataset_utils import (
    is_math_dataset,
    get_template_for_dataset,
)
from .device_utils import (
    get_device_type,
    get_device,
    is_cuda_available,
    is_mps_available,
    is_apple_silicon,
    get_optimal_dtype,
    get_optimal_dtype_str,
    supports_quantization,
    supports_deepspeed,
    get_device_count,
    get_device_name,
    get_device_memory_info,
    log_device_info,
    validate_device_compatibility,
    adapt_config_for_device,
)

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

