from __future__ import annotations

import os, sys
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import BitsAndBytesConfig
import wandb
import torch
from transformers import AutoTokenizer, AutoConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

from .config import ExperimentConfig

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from utils.logging_utils import setup_logging, SafeLogger
from utils.device_utils import (
    get_device_type,
    is_cuda_available,
    is_mps_available,
    supports_quantization,
    adapt_config_for_device,
    log_device_info,
)


class BaseTrainer(ABC):
    """Abstract base class for all trainers."""

    def __init__(self, config: ExperimentConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        # Use provided logger or create a new one
        # Wrap logger with SafeLogger to prevent logging errors from crashing training
        # Uses RobustFileHandler internally for handling stale file handles
        if logger is not None:
            self.logger = SafeLogger(logger)
        else:
            base_logger = setup_logging(self.__class__.__name__)
            self.logger = SafeLogger(base_logger)
        
        # Detect device type and adapt configuration for platform compatibility
        self.device_type = get_device_type()
        self.logger.info(f"Detected device type: {self.device_type.upper()}")
        
        # Adapt config for the current device (handles MPS/CPU limitations)
        self.config = adapt_config_for_device(self.config)
        
        self.local_rank = int(os.environ.get("LOCAL_RANK", -1))
        self.is_main_process = self.local_rank == -1 or self.local_rank == 0

        # Set device for distributed training
        if self.local_rank != -1:
            if is_cuda_available():
                torch.cuda.set_device(self.local_rank)
                self.logger.info(f"Set CUDA device to GPU {self.local_rank} for this process")
            elif is_mps_available():
                # MPS doesn't support multi-device, but we handle it gracefully
                self.logger.info("MPS detected - running in single-device mode")
        
        # Log device information on main process
        if self.is_main_process:
            log_device_info()

        # Initialize wandb if enabled and on main process
        if self.is_main_process and self.config.wandb.enabled:
            self.setup_wandb()

        # Set memory optimization for CUDA
        if is_cuda_available():
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    def setup_wandb(self):
        """Initialize Weights & Biases logging."""
        wandb.init(
            project=self.config.wandb.project,
            entity=self.config.wandb.entity,
            name=self.config.wandb.name,
            tags=self.config.wandb.tags,
            config={
                **self.config.model.to_dict(),
                **self.config.data.to_dict(),
                **self.config.training.to_dict(),
            }
        )
        self.logger.info("Wandb initialized")

    def cleanup_wandb(self):
        """Clean up wandb session."""
        if self.is_main_process and self.config.wandb.enabled:
            wandb.finish()
            self.logger.info("Wandb session finished")

    def create_quantization_config(self):
        """Create quantization configuration for 4-bit or 8-bit training.
        
        Returns:
            BitsAndBytesConfig if quantization is enabled and supported, None otherwise.
        """
        if not (self.config.model.load_in_4bit or self.config.model.load_in_8bit):
            return None
        
        # Check if quantization is supported on this platform
        if not supports_quantization():
            self.logger.warning(
                f"Quantization requested but not supported on {self.device_type.upper()}. "
                "Quantization requires CUDA and bitsandbytes. Continuing without quantization."
            )
            # Disable quantization in config
            self.config.model.load_in_4bit = False
            self.config.model.load_in_8bit = False
            return None
            
        self.logger.info(f"Setting up {'4-bit' if self.config.model.load_in_4bit else '8-bit'} quantization...")
        
        # Guard: quantized fine-tuning requires LoRA/PEFT adapters
        if not getattr(self.config.model, 'use_lora', False):
            raise ValueError(
                "Quantized training (4-bit/8-bit) requires LoRA adapters. "
                "Set model.use_lora: true (QLoRA) or disable quantization."
            )
        
        # Import BitsAndBytesConfig only when needed (CUDA-only dependency)
        from transformers import BitsAndBytesConfig
        
        if self.config.model.load_in_4bit:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, self.config.model.bnb_4bit_compute_dtype),
                bnb_4bit_quant_type=self.config.model.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=self.config.model.bnb_4bit_use_double_quant,
            )
        elif self.config.model.load_in_8bit:
            return BitsAndBytesConfig(
                load_in_8bit=True,
            )
        
        return None

    def load_model_config(self, model_name: Optional[str] = None):
        """Load and configure the model's AutoConfig.
        
        This method handles models that are missing the model_type key in their config.json
        by explicitly setting it based on the config.model.model_type field.
        
        Args:
            model_name: Optional model name. Defaults to config.model.base_model_name.
            
        Returns:
            AutoConfig object with model_type set if specified, None otherwise.
        """
        if model_name is None:
            model_name = self.config.model.base_model_name
        
        model_type = getattr(self.config.model, 'model_type', None)
        
        if model_type is None:
            return None
        
        self.logger.info(f"Loading model config with explicit model_type: {model_type}")
        
        try:
            # Load the config from the model
            config = AutoConfig.from_pretrained(
                model_name,
                trust_remote_code=True
            )
        except ValueError as e:
            # If AutoConfig fails due to missing model_type, create config with explicit type
            if "model_type" in str(e):
                self.logger.info(f"Model config missing model_type, loading with explicit type: {model_type}")
                config = AutoConfig.from_pretrained(
                    model_name,
                    model_type=model_type,
                    trust_remote_code=True
                )
            else:
                raise
        
        # Ensure model_type is set
        if not hasattr(config, 'model_type') or config.model_type is None:
            config.model_type = model_type
        
        return config

    def prepare_model_kwargs(self, quantization_config=None) -> Dict[str, Any]:
        """Prepare model loading kwargs with proper device placement.
        
        Args:
            quantization_config: Optional quantization configuration.
            
        Returns:
            Dictionary of kwargs for model loading.
        """
        torch_dtype = getattr(torch, self.config.model.torch_dtype)
        using_deepspeed = bool(self.config.training.deepspeed_config)
        
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "low_cpu_mem_usage": self.config.model.low_cpu_mem_usage,
        }
        
        # Load config with explicit model_type if specified
        model_config = self.load_model_config()
        if model_config is not None:
            model_kwargs["config"] = model_config
        
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
            # When training with DeepSpeed, let DeepSpeed/Accelerate manage device placement
            if not using_deepspeed:
                # For multi-GPU training with torchrun (no DeepSpeed)
                if is_cuda_available():
                    current_device = torch.cuda.current_device()
                    model_kwargs["device_map"] = {"": current_device}
                    self.logger.info(f"Using device_map={{'': {current_device}}} for quantized model (no DeepSpeed).")
                else:
                    model_kwargs["device_map"] = "auto"
                    self.logger.info("Using device_map=auto for quantized model (no DeepSpeed, no CUDA).")
            else:
                self.logger.info("DeepSpeed detected; not setting device_map to let DeepSpeed place parameters.")
        elif is_mps_available():
            # For MPS, we need to explicitly set device_map for proper device placement
            model_kwargs["device_map"] = "mps"
            self.logger.info("Using device_map='mps' for Apple Silicon.")
        
        return model_kwargs

    def setup_tokenizer_with_model(self, model, model_name: Optional[str] = None):
        """Setup tokenizer and resize model embeddings.
        
        Args:
            model: The model to resize embeddings for.
            model_name: Optional model name for loading tokenizer. Defaults to config.model.base_model_name.
            
        Returns:
            The configured tokenizer.
        """
        if model_name is None:
            model_name = self.config.model.base_model_name
            
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        
        # Resize embeddings
        model.resize_token_embeddings(len(tokenizer))
        
        # Cast embeddings to match model dtype
        self._cast_embeddings_to_model_dtype(model)
        
        return tokenizer

    def resize_and_cast_embeddings(self, model, tokenizer):
        """Resize model embeddings to match tokenizer and cast to correct dtype.
        
        Use this method for secondary models (e.g., reference models) that need
        their embeddings resized to match a tokenizer that was already set up.
        
        Args:
            model: The model to resize embeddings for.
            tokenizer: The tokenizer to match vocabulary size.
        """
        model.resize_token_embeddings(len(tokenizer))
        self._cast_embeddings_to_model_dtype(model)

    def _cast_embeddings_to_model_dtype(self, model):
        """Cast embeddings to match model dtype to avoid dtype mismatch errors.
        
        This is particularly important for bfloat16/float16 models where
        resize_token_embeddings may create new embeddings in float32.
        
        Args:
            model: The model whose embeddings should be cast.
        """
        target_dtype = getattr(torch, self.config.model.torch_dtype)
        
        # Cast input embeddings - directly modify weight data for reliable conversion
        # Using .weight.data ensures the conversion happens in-place and persists
        input_embeddings = model.get_input_embeddings()
        if input_embeddings is not None and input_embeddings.weight.dtype != target_dtype:
            input_embeddings.weight.data = input_embeddings.weight.data.to(target_dtype)
            self.logger.info(f"Cast input embeddings to {target_dtype}")
        
        # Cast output embeddings (lm_head) - directly modify weight data for reliable conversion
        # This is critical for generation where lm_head must match hidden states dtype
        output_embeddings = model.get_output_embeddings()
        if output_embeddings is not None and output_embeddings.weight.dtype != target_dtype:
            output_embeddings.weight.data = output_embeddings.weight.data.to(target_dtype)
            self.logger.info(f"Cast output embeddings to {target_dtype}")

    def apply_lora_to_model(self, model, task_type: TaskType = TaskType.CAUSAL_LM, 
                           quantization_config: Optional[BitsAndBytesConfig] = None):
        """Apply LoRA/QLoRA to a model.
        
        Args:
            model: The model to apply LoRA to.
            task_type: The task type for LoRA (CAUSAL_LM or SEQ_CLS).
            quantization_config: Optional quantization configuration to determine if using QLoRA.
            
        Returns:
            The model with LoRA applied.
        """
        if not getattr(self.config.model, 'use_lora', False):
            return model
            
        self.logger.info("Applying LoRA configuration...")
        
        # Prepare model for k-bit training if using quantization
        if self.config.model.load_in_4bit or self.config.model.load_in_8bit:
            self.logger.info("Preparing model for k-bit training (QLoRA)...")
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=self.config.training.gradient_checkpointing
            )
            # Ensure input gradients are enabled for k-bit training flows
            try:
                model.enable_input_require_grads()
            except Exception:
                pass
        
        # Create LoRA config
        lora_config = LoraConfig(
            r=self.config.model.lora_r,
            lora_alpha=self.config.model.lora_alpha,
            lora_dropout=self.config.model.lora_dropout,
            target_modules=self.config.model.lora_target_modules,
            bias=self.config.model.lora_bias,
            task_type=task_type,
        )
        
        # Apply LoRA to model
        model = get_peft_model(model, lora_config)
        
        # Cast the entire model (including LoRA adapters) to target dtype
        # LoRA layers are created in float32 by default, which causes dtype mismatches
        # during generation when the base model is in bfloat16/float16
        target_dtype = getattr(torch, self.config.model.torch_dtype)
        model = model.to(target_dtype)
        self.logger.info(f"Cast model (including LoRA adapters) to {target_dtype}")
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        self.logger.info(f"{'QLoRA' if quantization_config else 'LoRA'} applied - "
                       f"Trainable params: {trainable_params:,} / {total_params:,} "
                       f"({100 * trainable_params / total_params:.2f}%)")

        # Safety check: ensure we actually have trainable parameters
        if trainable_params == 0:
            target_modules = self.config.model.lora_target_modules
            self.logger.error(
                "No trainable parameters detected after applying LoRA. "
                f"target_modules={target_modules}."
            )
            raise ValueError(
                "LoRA injection resulted in zero trainable parameters. "
                "This usually means lora_target_modules do not match your model's module names. "
                "For LLaMA-class models, typical targets are: q_proj, k_proj, v_proj, o_proj, "
                "gate_proj, up_proj, down_proj."
            )
        
        return model

    def disable_cache_for_gradient_checkpointing(self, model):
        """Disable model cache when using gradient checkpointing.
        
        Args:
            model: The model to configure.
        """
        if getattr(self.config.training, "gradient_checkpointing", False):
            try:
                model.config.use_cache = False
            except Exception:
                pass

    @abstractmethod
    def setup_model(self):
        """Setup model and tokenizer. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def setup_data(self):
        """Setup training data. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def setup_trainer(self):
        """Setup the specific trainer. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def train(self):
        """Run training. Must be implemented by subclasses."""
        pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with wandb cleanup."""
        self.cleanup_wandb()
