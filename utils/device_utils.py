"""
Device detection utilities for cross-platform support (CUDA, MPS, CPU).

This module provides utilities for detecting and managing compute devices
across different platforms including:
- CUDA (NVIDIA GPUs on Linux/Windows)
- MPS (Apple Silicon on macOS)
- CPU (fallback for all platforms)
"""

import logging
import platform
import sys
from typing import Optional, Tuple, Dict, Any

import torch

logger = logging.getLogger(__name__)


def get_device_type() -> str:
    """
    Detect the best available device type.
    
    Returns:
        str: One of "cuda", "mps", or "cpu"
    """
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def get_device() -> torch.device:
    """
    Get the best available torch device.
    
    Returns:
        torch.device: The device object for the best available compute device.
    """
    device_type = get_device_type()
    return torch.device(device_type)


def is_cuda_available() -> bool:
    """Check if CUDA is available."""
    return torch.cuda.is_available()


def is_mps_available() -> bool:
    """Check if MPS (Apple Silicon) is available."""
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def is_apple_silicon() -> bool:
    """Check if running on Apple Silicon (M1/M2/M3/etc.)."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def get_optimal_dtype() -> torch.dtype:
    """
    Get the optimal dtype for the current device.
    
    - CUDA: bfloat16 if supported, else float16
    - MPS: float16 (bfloat16 has limited support on MPS)
    - CPU: float32 (quantized training not well supported)
    
    Returns:
        torch.dtype: The optimal dtype for the current device.
    """
    device_type = get_device_type()
    
    if device_type == "cuda":
        # Check if bfloat16 is supported on this GPU
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        else:
            return torch.float16
    elif device_type == "mps":
        # MPS has better float16 support than bfloat16
        return torch.float16
    else:
        # CPU training typically uses float32
        return torch.float32


def get_optimal_dtype_str() -> str:
    """
    Get the optimal dtype as a string for configuration.
    
    Returns:
        str: The dtype string ("bfloat16", "float16", or "float32")
    """
    dtype = get_optimal_dtype()
    dtype_map = {
        torch.bfloat16: "bfloat16",
        torch.float16: "float16",
        torch.float32: "float32",
    }
    return dtype_map.get(dtype, "float32")


def supports_quantization() -> bool:
    """
    Check if the current platform supports bitsandbytes quantization.
    
    bitsandbytes requires CUDA and is not supported on MPS or CPU.
    
    Returns:
        bool: True if quantization is supported.
    """
    if not is_cuda_available():
        return False
    
    try:
        import bitsandbytes  # noqa: F401
        return True
    except ImportError:
        return False


def supports_deepspeed() -> bool:
    """
    Check if the current platform supports DeepSpeed.
    
    DeepSpeed requires CUDA and is not supported on MPS or CPU.
    
    Returns:
        bool: True if DeepSpeed is supported.
    """
    if not is_cuda_available():
        return False
    
    try:
        import deepspeed  # noqa: F401
        return True
    except ImportError:
        return False


def get_device_count() -> int:
    """
    Get the number of available compute devices.
    
    Returns:
        int: Number of available GPUs (CUDA or MPS), or 1 for CPU.
    """
    if is_cuda_available():
        return torch.cuda.device_count()
    elif is_mps_available():
        return 1  # MPS only supports single-device
    else:
        return 1  # CPU


def get_device_name(device_index: int = 0) -> str:
    """
    Get the name of the compute device.
    
    Args:
        device_index: Index of the device (only relevant for CUDA).
        
    Returns:
        str: Name of the device.
    """
    if is_cuda_available():
        return torch.cuda.get_device_name(device_index)
    elif is_mps_available():
        # Get Apple chip name if possible
        if is_apple_silicon():
            try:
                import subprocess
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    return result.stdout.strip()
            except Exception:
                pass
        return "Apple MPS"
    else:
        return "CPU"


def get_device_memory_info(device_index: int = 0) -> Optional[Dict[str, float]]:
    """
    Get memory information for the compute device.
    
    Args:
        device_index: Index of the device (only relevant for CUDA).
        
    Returns:
        Dict with memory info in GB, or None if not available.
    """
    if is_cuda_available():
        allocated = torch.cuda.memory_allocated(device_index) / 1024**3
        reserved = torch.cuda.memory_reserved(device_index) / 1024**3
        total = torch.cuda.get_device_properties(device_index).total_memory / 1024**3
        return {
            "allocated_gb": round(allocated, 2),
            "reserved_gb": round(reserved, 2),
            "total_gb": round(total, 2),
        }
    elif is_mps_available():
        # MPS memory info is limited, but we can try
        try:
            allocated = torch.mps.current_allocated_memory() / 1024**3
            return {
                "allocated_gb": round(allocated, 2),
                "reserved_gb": None,
                "total_gb": None,  # Not available for MPS
            }
        except Exception:
            return None
    else:
        return None


def log_device_info():
    """Log information about the available compute devices."""
    device_type = get_device_type()
    
    logger.info(f"Platform: {platform.system()} {platform.machine()}")
    logger.info(f"Python version: {platform.python_version()}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"Device type: {device_type.upper()}")
    
    if device_type == "cuda":
        logger.info(f"CUDA version: {torch.version.cuda}")
        device_count = torch.cuda.device_count()
        logger.info(f"GPU count: {device_count}")
        for i in range(device_count):
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            mem_info = get_device_memory_info(i)
            if mem_info:
                logger.info(f"    Total memory: {mem_info['total_gb']} GB")
    elif device_type == "mps":
        logger.info(f"MPS device: {get_device_name()}")
        logger.info("Note: MPS (Apple Silicon) support is experimental")
        logger.info("Note: Quantization (bitsandbytes) and DeepSpeed are not available on MPS")
    else:
        logger.info("Running on CPU (no GPU acceleration available)")
    
    # Log feature availability
    logger.info(f"Quantization support: {supports_quantization()}")
    logger.info(f"DeepSpeed support: {supports_deepspeed()}")


def validate_device_compatibility(config: Any) -> Tuple[bool, list]:
    """
    Validate that the configuration is compatible with the current device.
    
    Args:
        config: The experiment configuration object.
        
    Returns:
        Tuple of (is_valid, list of warning messages)
    """
    warnings = []
    device_type = get_device_type()
    
    # Check quantization compatibility
    if hasattr(config, 'model'):
        model_config = config.model
        if getattr(model_config, 'load_in_4bit', False) or getattr(model_config, 'load_in_8bit', False):
            if not supports_quantization():
                if device_type == "mps":
                    warnings.append(
                        "Quantization (4-bit/8-bit) is not supported on Apple Silicon (MPS). "
                        "Disabling quantization and using full precision instead."
                    )
                else:
                    warnings.append(
                        "Quantization requires bitsandbytes which is not installed. "
                        "Install with: pip install bitsandbytes"
                    )
    
    # Check DeepSpeed compatibility
    if hasattr(config, 'training'):
        training_config = config.training
        if getattr(training_config, 'deepspeed_config', None):
            if not supports_deepspeed():
                if device_type == "mps":
                    warnings.append(
                        "DeepSpeed is not supported on Apple Silicon (MPS). "
                        "Disabling DeepSpeed for single-GPU training."
                    )
                else:
                    warnings.append(
                        "DeepSpeed is not installed. "
                        "Install with: pip install deepspeed"
                    )
    
    # Check bf16 compatibility
    if hasattr(config, 'training'):
        training_config = config.training
        if getattr(training_config, 'bf16', False):
            if device_type == "mps":
                warnings.append(
                    "bfloat16 has limited support on Apple Silicon (MPS). "
                    "Switching to float16 for better compatibility."
                )
            elif device_type == "cpu":
                warnings.append(
                    "bfloat16 training on CPU is slow. "
                    "Consider using float32 for CPU training."
                )
    
    is_valid = len(warnings) == 0
    return is_valid, warnings


def adapt_config_for_device(config: Any) -> Any:
    """
    Adapt the configuration for the current device.
    
    This modifies the configuration in-place to ensure compatibility
    with the current device (CUDA, MPS, or CPU).
    
    Args:
        config: The experiment configuration object.
        
    Returns:
        The adapted configuration object.
    """
    device_type = get_device_type()
    
    if device_type == "mps":
        # Disable quantization on MPS
        if hasattr(config, 'model'):
            if getattr(config.model, 'load_in_4bit', False):
                config.model.load_in_4bit = False
                logger.warning("Disabled 4-bit quantization (not supported on MPS)")
            if getattr(config.model, 'load_in_8bit', False):
                config.model.load_in_8bit = False
                logger.warning("Disabled 8-bit quantization (not supported on MPS)")
        
        # Disable DeepSpeed on MPS
        if hasattr(config, 'training'):
            if getattr(config.training, 'deepspeed_config', None):
                config.training.deepspeed_config = None
                logger.warning("Disabled DeepSpeed (not supported on MPS)")
            
            # Switch bf16 to fp16 on MPS
            if getattr(config.training, 'bf16', False):
                config.training.bf16 = False
                config.training.fp16 = True
                logger.warning("Switched from bf16 to fp16 (better MPS compatibility)")
        
        # Update model dtype
        if hasattr(config, 'model'):
            if getattr(config.model, 'torch_dtype', None) == "bfloat16":
                config.model.torch_dtype = "float16"
                logger.warning("Switched model dtype from bfloat16 to float16")
    
    elif device_type == "cpu":
        # Disable quantization on CPU
        if hasattr(config, 'model'):
            if getattr(config.model, 'load_in_4bit', False):
                config.model.load_in_4bit = False
                logger.warning("Disabled 4-bit quantization (not supported on CPU)")
            if getattr(config.model, 'load_in_8bit', False):
                config.model.load_in_8bit = False
                logger.warning("Disabled 8-bit quantization (not supported on CPU)")
        
        # Disable DeepSpeed on CPU
        if hasattr(config, 'training'):
            if getattr(config.training, 'deepspeed_config', None):
                config.training.deepspeed_config = None
                logger.warning("Disabled DeepSpeed (not supported on CPU)")
            
            # Use float32 on CPU for better performance
            if getattr(config.training, 'bf16', False) or getattr(config.training, 'fp16', False):
                config.training.bf16 = False
                config.training.fp16 = False
                logger.warning("Switched to float32 for CPU training")
        
        # Update model dtype
        if hasattr(config, 'model'):
            config.model.torch_dtype = "float32"
            logger.warning("Switched model dtype to float32 for CPU")
    
    return config

