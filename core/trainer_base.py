from __future__ import annotations

import os, sys
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, TYPE_CHECKING

# Workaround for a bug in transformers where build_peft_weight_mapping passes
# distributed_operation/quantization_operation as constructor kwargs to
# WeightConverter, but WeightConverter.__init__ only accepts source_patterns,
# target_patterns, and operations. This is triggered when loading a PEFT
# checkpoint for a MoE model (e.g. Qwen3-30B-A3B).
def _patch_weight_converter_init():
    try:
        from transformers.core_model_loading import WeightConverter
        if getattr(WeightConverter, "_fai_rl_patched", False):
            return
        _orig_init = WeightConverter.__init__

        def _patched_init(self, source_patterns, target_patterns, operations, **kwargs):
            _orig_init(self, source_patterns, target_patterns, operations)
            for k, v in kwargs.items():
                try:
                    setattr(self, k, v)
                except (AttributeError, TypeError):
                    pass

        WeightConverter.__init__ = _patched_init
        WeightConverter._fai_rl_patched = True
    except (ImportError, AttributeError):
        pass

_patch_weight_converter_init()

if TYPE_CHECKING:
    from transformers import BitsAndBytesConfig
import wandb
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

from .config import ExperimentConfig

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.logging_utils import setup_logging, SafeLogger
from utils.s3_utils import build_s3_callback, download_directory_from_s3
from utils.device_utils import (
    get_device_type,
    is_cuda_available,
    is_mps_available,
    supports_quantization,
    adapt_config_for_device,
    log_device_info,
    resolve_transformers_attn_implementation,
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

        # Download model from S3 if base_model_name is an s3:// URI. This must
        # happen before setup_model() so all trainers transparently get a local
        # path. Rank 0 (per-node) downloads; other ranks wait on a sentinel file.
        self._maybe_download_s3_model()

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

    def _maybe_download_s3_model(self) -> None:
        """Download the base model from S3 if base_model_name is an s3:// URI.

        On each node, local rank 0 performs the download into a deterministic
        cache directory under the system temp dir.  Other local ranks busy-wait
        on a sentinel file written by rank 0 once the download completes.
        Subsequent runs on the same node reuse the cached directory, so the
        download only happens once per node per unique S3 URI.

        After this method returns, self.config.model.base_model_name points to
        a local directory that AutoModelForCausalLM.from_pretrained() can load.
        """
        import hashlib
        import tempfile
        import time

        model_name = self.config.model.base_model_name
        if not model_name.startswith("s3://"):
            return

        uri_hash = hashlib.md5(model_name.encode()).hexdigest()[:12]
        cache_dir = os.path.join(tempfile.gettempdir(), f"fai-rl-model-{uri_hash}")
        sentinel = os.path.join(cache_dir, ".download_complete")

        region = getattr(self.config.model, "s3_region", None)
        endpoint_url = getattr(self.config.model, "s3_endpoint_url", None)

        if self.local_rank <= 0:
            # local_rank=-1 (single GPU) or local_rank=0 (first GPU per node)
            if not os.path.exists(sentinel):
                self.logger.info("Downloading model from %s -> %s", model_name, cache_dir)
                download_directory_from_s3(
                    model_name,
                    cache_dir,
                    region=region,
                    endpoint_url=endpoint_url,
                )
                open(sentinel, "w").close()
                self.logger.info("Model download complete, cached at %s", cache_dir)
            else:
                self.logger.info("Reusing cached model at %s", cache_dir)
        else:
            # Non-zero local ranks wait for rank 0 to finish downloading.
            self.logger.info(
                "Local rank %d waiting for model download by rank 0...", self.local_rank
            )
            timeout = 3600
            elapsed = 0
            while not os.path.exists(sentinel):
                time.sleep(5)
                elapsed += 5
                if elapsed >= timeout:
                    raise TimeoutError(
                        f"Timed out after {timeout}s waiting for S3 model download at {cache_dir}"
                    )

        self.config.model.base_model_name = cache_dir

    def setup_wandb(self):
        """Initialize Weights & Biases logging."""
        # Export API key and base URL to env so HF Trainer subprocesses / TRL
        # integrations also pick them up. Recipe values take precedence over
        # existing env vars so configs are reproducible.
        api_key = getattr(self.config.wandb, "api_key", None)
        base_url = getattr(self.config.wandb, "base_url", None)
        if base_url:
            os.environ["WANDB_BASE_URL"] = base_url
        if api_key:
            os.environ["WANDB_API_KEY"] = api_key
            try:
                login_kwargs = {"key": api_key}
                if base_url:
                    login_kwargs["host"] = base_url
                wandb.login(**login_kwargs)
            except Exception as e:
                self.logger.warning(f"wandb.login failed, will rely on env var: {e}")
        elif not os.environ.get("WANDB_API_KEY"):
            self.logger.warning(
                "wandb.enabled=true but no api_key in recipe and no WANDB_API_KEY env var set; "
                "wandb.init may fail or fall back to anonymous mode."
            )

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
            "ignore_mismatched_sizes": self.config.model.ignore_mismatched_sizes,
        }

        attn_impl = resolve_transformers_attn_implementation(
            self.config.model.use_flash_attention
        )
        if attn_impl is not None:
            model_kwargs["attn_implementation"] = attn_impl
            if attn_impl == "flash_attention_2":
                self.logger.info("Using Flash Attention 2.")
            elif attn_impl == "sdpa":
                self.logger.info("Using PyTorch SDPA for attention.")

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
        
        return tokenizer

    def load_base_model_for_training(self, model_kwargs: Dict[str, Any]):
        """Load the base model, transparently handling PEFT/LoRA checkpoints.

        When base_model_name is a local directory containing adapter_config.json,
        loads the original base model from adapter_config.base_model_name_or_path
        (so the architecture exactly matches the saved adapter weights) and stores
        the checkpoint path for apply_lora_to_model to resume from.

        When base_model_name is a plain HuggingFace ID or a non-PEFT local dir,
        behaves identically to AutoModelForCausalLM.from_pretrained.
        """
        from peft import PeftConfig

        model_path = self.config.model.base_model_name
        self._peft_adapter_path = None

        adapter_config_path = os.path.join(model_path, "adapter_config.json")
        if os.path.isdir(model_path) and os.path.exists(adapter_config_path):
            peft_cfg = PeftConfig.from_pretrained(model_path)
            base_model_path = peft_cfg.base_model_name_or_path
            self.logger.info(
                "Detected PEFT checkpoint; loading base model from %s", base_model_path
            )
            self._peft_adapter_path = model_path
            # ignore_mismatched_sizes is not needed when loading the base model clean.
            clean_kwargs = {k: v for k, v in model_kwargs.items() if k != "ignore_mismatched_sizes"}
            return AutoModelForCausalLM.from_pretrained(base_model_path, **clean_kwargs)

        return AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

    def apply_lora_to_model(self, model, task_type: TaskType = TaskType.CAUSAL_LM,
                            quantization_config: Optional[BitsAndBytesConfig] = None):
        """Apply LoRA/QLoRA to a model.

        When a PEFT checkpoint was detected during load_base_model_for_training,
        resumes from that adapter instead of creating a new one.

        Args:
            model: The model to apply LoRA to.
            task_type: The task type for LoRA (CAUSAL_LM or SEQ_CLS).
            quantization_config: Optional quantization configuration to determine if using QLoRA.

        Returns:
            The model with LoRA applied.
        """
        if not getattr(self.config.model, 'use_lora', False):
            return model

        peft_adapter_path = getattr(self, '_peft_adapter_path', None)
        if peft_adapter_path is not None:
            from peft import PeftModel
            self.logger.info("Loading existing PEFT adapter from %s", peft_adapter_path)
            model = PeftModel.from_pretrained(model, peft_adapter_path, is_trainable=True)
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            self.logger.info(
                "PEFT adapter loaded - Trainable params: %s / %s (%.2f%%)",
                f"{trainable_params:,}", f"{total_params:,}",
                100 * trainable_params / total_params,
            )
            if trainable_params == 0:
                raise ValueError("No trainable parameters after loading PEFT adapter.")
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

    def build_callbacks(self) -> list:
        """Build trainer callbacks from config (e.g. S3 upload)."""
        callbacks = []
        s3_cb = build_s3_callback(self.config.s3)
        if s3_cb is not None:
            self.logger.info(
                "S3 upload enabled -> s3://%s/%s",
                self.config.s3.bucket, self.config.s3.prefix,
            )
            callbacks.append(s3_cb)
        return callbacks

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
