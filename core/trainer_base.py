import os, sys
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import wandb
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

from .config import ExperimentConfig

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from utils.logging_utils import setup_logging


class BaseTrainer(ABC):
    """Abstract base class for all trainers."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.logger = setup_logging(self.__class__.__name__)
        self.local_rank = int(os.environ.get("LOCAL_RANK", -1))
        self.is_main_process = self.local_rank == -1 or self.local_rank == 0

        # Set CUDA device for distributed training to avoid NCCL warnings
        if self.local_rank != -1 and torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            self.logger.info(f"Set CUDA device to GPU {self.local_rank} for this process")

        # Initialize wandb if enabled and on main process
        if self.is_main_process and self.config.wandb.enabled:
            self.setup_wandb()

        # Set memory optimization
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

    def create_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Create quantization configuration for 4-bit or 8-bit training.
        
        Returns:
            BitsAndBytesConfig if quantization is enabled, None otherwise.
        """
        if not (self.config.model.load_in_4bit or self.config.model.load_in_8bit):
            return None
            
        self.logger.info(f"Setting up {'4-bit' if self.config.model.load_in_4bit else '8-bit'} quantization...")
        
        # Guard: quantized fine-tuning requires LoRA/PEFT adapters
        if not getattr(self.config.model, 'use_lora', False):
            raise ValueError(
                "Quantized training (4-bit/8-bit) requires LoRA adapters. "
                "Set model.use_lora: true (QLoRA) or disable quantization."
            )
        
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

    def prepare_model_kwargs(self, quantization_config: Optional[BitsAndBytesConfig] = None) -> Dict[str, Any]:
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
        
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
            # When training with DeepSpeed, let DeepSpeed/Accelerate manage device placement
            if not using_deepspeed:
                # For multi-GPU training with torchrun (no DeepSpeed)
                if torch.cuda.is_available():
                    current_device = torch.cuda.current_device()
                    model_kwargs["device_map"] = {"": current_device}
                    self.logger.info(f"Using device_map={{'': {current_device}}} for quantized model (no DeepSpeed).")
                else:
                    model_kwargs["device_map"] = "auto"
                    self.logger.info("Using device_map=auto for quantized model (no DeepSpeed, no CUDA).")
            else:
                self.logger.info("DeepSpeed detected; not setting device_map to let DeepSpeed place parameters.")
        
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
        
        return tokenizer

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
        
        # Ensure non-quantized modules (lm_head and embeddings) match compute dtype for generation
        if self.config.model.load_in_4bit or self.config.model.load_in_8bit:
            if self.config.model.load_in_4bit:
                compute_dtype = getattr(torch, self.config.model.bnb_4bit_compute_dtype)
            else:
                if getattr(self.config.training, 'bf16', False):
                    compute_dtype = torch.bfloat16
                elif getattr(self.config.training, 'fp16', False):
                    compute_dtype = torch.float16
                else:
                    compute_dtype = torch.float32

            # Cast lm_head to compute dtype - navigate through PEFT wrapper correctly
            try:
                # For PEFT models, we need to access: model.base_model.model.lm_head
                if hasattr(model, 'base_model'):
                    base_model = model.base_model
                    # Check if there's another nested 'model' attribute (common in PEFT)
                    if hasattr(base_model, 'model'):
                        actual_model = base_model.model
                    else:
                        actual_model = base_model
                        
                    if hasattr(actual_model, 'lm_head'):
                        actual_model.lm_head = actual_model.lm_head.to(compute_dtype)
                        self.logger.info(f"Cast lm_head to {compute_dtype}")
                elif hasattr(model, 'lm_head'):
                    model.lm_head = model.lm_head.to(compute_dtype)
                    self.logger.info(f"Cast lm_head to {compute_dtype}")
            except Exception as e:
                self.logger.warning(f"Could not cast lm_head to {compute_dtype}: {e}")

            # Cast input embeddings to compute dtype
            try:
                if hasattr(model, 'get_input_embeddings'):
                    input_embeddings = model.get_input_embeddings()
                    if input_embeddings is not None:
                        # Check if embeddings need casting
                        if input_embeddings.weight.dtype != compute_dtype:
                            input_embeddings = input_embeddings.to(compute_dtype)
                            model.set_input_embeddings(input_embeddings)
                            self.logger.info(f"Cast input embeddings to {compute_dtype}")
            except Exception as e:
                self.logger.warning(f"Could not cast input embeddings to {compute_dtype}: {e}")
            
            # Also cast output embeddings if they exist separately
            try:
                if hasattr(model, 'get_output_embeddings'):
                    output_embeddings = model.get_output_embeddings()
                    if output_embeddings is not None and output_embeddings.weight.dtype != compute_dtype:
                        output_embeddings = output_embeddings.to(compute_dtype)
                        model.set_output_embeddings(output_embeddings)
                        self.logger.info(f"Cast output embeddings to {compute_dtype}")
            except Exception as e:
                self.logger.warning(f"Could not cast output embeddings to {compute_dtype}: {e}")
        
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
