import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch dtype."""
    dtype_mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "int8": torch.int8,
    }
    return dtype_mapping.get(dtype_str, torch.bfloat16)


def create_bnb_config(
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    bnb_4bit_compute_dtype: str = "bfloat16",
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_use_double_quant: bool = True,
) -> Optional[BitsAndBytesConfig]:
    """Create BitsAndBytesConfig for quantization."""
    if not (load_in_4bit or load_in_8bit):
        return None

    if load_in_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=get_torch_dtype(bnb_4bit_compute_dtype),
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
        )
    elif load_in_8bit:
        return BitsAndBytesConfig(
            load_in_8bit=True,
        )


def load_model_and_tokenizer(
    model_name: str,
    torch_dtype: str = "bfloat16",
    low_cpu_mem_usage: bool = True,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    use_flash_attention: bool = False,
    trust_remote_code: bool = False,
    device_map: str = "auto",
) -> tuple:
    """
    Load model and tokenizer with specified configuration.

    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model: {model_name}")

    # Convert dtype
    torch_dtype = get_torch_dtype(torch_dtype)

    # Create quantization config
    quantization_config = create_bnb_config(
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
    )

    # Model loading kwargs
    model_kwargs = {
        "pretrained_model_name_or_path": model_name,
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": low_cpu_mem_usage,
        "trust_remote_code": trust_remote_code,
        "device_map": device_map,
    }

    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config

    if use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    # Load model
    model = AutoModelForCausalLM.from_pretrained(**model_kwargs)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token")

    logger.info(f"Model loaded successfully. Parameters: {model.num_parameters():,}")

    return model, tokenizer


def get_model_memory_usage() -> Dict[str, float]:
    """Get current model memory usage."""
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}

    memory_stats = {}
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(i) / 1024**3   # GB
        memory_stats[f"gpu_{i}"] = {
            "allocated_gb": round(allocated, 2),
            "reserved_gb": round(reserved, 2),
        }

    return memory_stats


def count_trainable_parameters(model) -> Dict[str, int]:
    """Count trainable and total parameters in the model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "trainable_percentage": round(100 * trainable_params / total_params, 2) if total_params > 0 else 0
    }


def print_model_info(model, tokenizer):
    """Print detailed model information."""
    logger.info("="*50)
    logger.info("MODEL INFORMATION")
    logger.info("="*50)

    # Basic info
    logger.info(f"Model class: {model.__class__.__name__}")
    logger.info(f"Model config: {model.config}")

    # Parameter counts
    param_info = count_trainable_parameters(model)
    logger.info(f"Total parameters: {param_info['total_parameters']:,}")
    logger.info(f"Trainable parameters: {param_info['trainable_parameters']:,}")
    logger.info(f"Trainable percentage: {param_info['trainable_percentage']}%")

    # Tokenizer info
    logger.info(f"Tokenizer class: {tokenizer.__class__.__name__}")
    logger.info(f"Vocabulary size: {len(tokenizer):,}")
    logger.info(f"Pad token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    logger.info(f"EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    logger.info(f"BOS token: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")

    # Memory usage
    memory_info = get_model_memory_usage()
    if "error" not in memory_info:
        logger.info("GPU Memory Usage:")
        for gpu, stats in memory_info.items():
            logger.info(f"  {gpu}: {stats['allocated_gb']} GB allocated, {stats['reserved_gb']} GB reserved")

    logger.info("="*50)
