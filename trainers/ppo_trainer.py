import os, sys
import torch
from accelerate import PartialState
from datasets import load_dataset, concatenate_datasets
from trl import PPOConfig, PPOTrainer as TRLPPOTrainer
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.config import ExperimentConfig
from core.trainer_base import BaseTrainer
from utils.logging_utils import setup_logging
from .rewards.accuracy_rewards import exact_match_reward_func, digit_reward_func
from .rewards.format_rewards import structured_xml_reward_func
from .templates.gsm8k_template import GSM8KTemplate
from .templates.openmathinstruct_template import OpenMathInstructTemplate


# Monkey-patch DistributedDataParallel to expose the underlying model's config and policy
# This fixes the TRL library's access to model.config and model.policy when using DDP
def _ddp_config_property(self):
    """Property to access the underlying model's config through DDP wrapper."""
    if hasattr(self, 'module'):
        return self.module.config
    raise AttributeError("DistributedDataParallel has no attribute 'config' and no 'module' attribute found")

def _ddp_policy_property(self):
    """Property to access the underlying model's policy through DDP wrapper."""
    if hasattr(self, 'module'):
        return self.module.policy if hasattr(self.module, 'policy') else self.module
    raise AttributeError("DistributedDataParallel has no attribute 'policy' and no 'module' attribute found")

if hasattr(torch.nn.parallel, 'DistributedDataParallel'):
    if not hasattr(torch.nn.parallel.DistributedDataParallel, 'config'):
        torch.nn.parallel.DistributedDataParallel.config = property(_ddp_config_property)
    if not hasattr(torch.nn.parallel.DistributedDataParallel, 'policy'):
        torch.nn.parallel.DistributedDataParallel.policy = property(_ddp_policy_property)


# Global fallback: ensure all torch.nn.Module instances expose gradient checkpointing toggles
def _forward_gradient_checkpointing_call(module: torch.nn.Module, method_name: str) -> None:
    # Try common attributes used by wrappers to reach the underlying model
    for attr_name in ("gradient_checkpointing_" + method_name,):
        if hasattr(module, attr_name) and callable(getattr(module, attr_name)):
            try:
                getattr(module, attr_name)()
                return
            except Exception:
                pass
    for candidate in ("model", "policy", "base_model"):
        base = getattr(module, candidate, None)
        if base is None:
            continue
        # Direct on base
        fn = getattr(base, "gradient_checkpointing_" + method_name, None)
        if callable(fn):
            try:
                fn()
                return
            except Exception:
                pass
        # Try nested .model (e.g., PEFT wrappers)
        nested = getattr(base, "model", None)
        if nested is not None:
            fn_nested = getattr(nested, "gradient_checkpointing_" + method_name, None)
            if callable(fn_nested):
                try:
                    fn_nested()
                    return
                except Exception:
                    pass
    # If nothing found, act as no-op
    return


if not hasattr(torch.nn.Module, "gradient_checkpointing_disable"):
    def _gc_disable(self):  # type: ignore[no-redef]
        _forward_gradient_checkpointing_call(self, "disable")
    torch.nn.Module.gradient_checkpointing_disable = _gc_disable  # type: ignore[attr-defined]

if not hasattr(torch.nn.Module, "gradient_checkpointing_enable"):
    def _gc_enable(self):  # type: ignore[no-redef]
        _forward_gradient_checkpointing_call(self, "enable")
    torch.nn.Module.gradient_checkpointing_enable = _gc_enable  # type: ignore[attr-defined]


class PPOTrainer(BaseTrainer):
    """PPO (Proximal Policy Optimization) trainer implementation."""

    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.trainer = None
        self.model = None
        self.ref_policy = None
        self.value_model = None
        self.reward_model = None
        self.tokenizer = None
        self.train_dataset = None
        self.eval_dataset = None

    def setup_model(self):
        """Load models and tokenizer for PPO."""
        self.logger.info(f"Loading model: {self.config.model.base_model_name}")

        # Convert string dtype to torch dtype
        torch_dtype = getattr(torch, self.config.model.torch_dtype)
        
        # Create quantization config
        quantization_config = None
        if self.config.model.load_in_4bit or self.config.model.load_in_8bit:
            self.logger.info(f"Setting up {'4-bit' if self.config.model.load_in_4bit else '8-bit'} quantization...")
            # Guard: quantized fine-tuning requires LoRA/PEFT adapters
            if not getattr(self.config.model, 'use_lora', False):
                raise ValueError(
                    "Quantized training (4-bit/8-bit) requires LoRA adapters. "
                    "Set model.use_lora: true (QLoRA) or disable quantization."
                )
            
            if self.config.model.load_in_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=getattr(torch, self.config.model.bnb_4bit_compute_dtype),
                    bnb_4bit_quant_type=self.config.model.bnb_4bit_quant_type,
                    bnb_4bit_use_double_quant=self.config.model.bnb_4bit_use_double_quant,
                )
            elif self.config.model.load_in_8bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
        
        # Load main model
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

        # Load tokenizer first
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.base_model_name,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        
        # Load policy model (main model for generation)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model.base_model_name,
            **model_kwargs
        )
        
        # Resize embeddings for policy model
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Load reference model (for KL penalty in PPO)
        self.ref_policy = AutoModelForCausalLM.from_pretrained(
            self.config.model.value_model_name,
            **model_kwargs
        )
        
        # Resize embeddings for reference model
        self.ref_policy.resize_token_embeddings(len(self.tokenizer))
        
        # Load value model (for advantage estimation)
        self.value_model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model.value_model_name,
            num_labels=1,
            **model_kwargs
        )
        
        # Resize embeddings for value model
        self.value_model.resize_token_embeddings(len(self.tokenizer))
        
        # Load reward model (for computing rewards)
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model.value_model_name,
            num_labels=1,
            **model_kwargs
        )
        
        # Resize embeddings for reward model
        self.reward_model.resize_token_embeddings(len(self.tokenizer))

        # Propagate pad_token_id to all model configs to allow batching with padding
        try:
            pad_id = self.tokenizer.pad_token_id
            if pad_id is not None:
                self.model.config.pad_token_id = pad_id
                self.ref_policy.config.pad_token_id = pad_id
                self.value_model.config.pad_token_id = pad_id
                self.reward_model.config.pad_token_id = pad_id
        except Exception:
            pass
        
        # Apply LoRA if enabled (including QLoRA)
        if getattr(self.config.model, 'use_lora', False):
            self.logger.info("Applying LoRA configuration...")
            
            # Prepare models for k-bit training if using quantization
            if self.config.model.load_in_4bit or self.config.model.load_in_8bit:
                self.logger.info("Preparing models for k-bit training (QLoRA)...")
                gradient_checkpointing = getattr(self.config.training, 'gradient_checkpointing', False)
                
                self.model = prepare_model_for_kbit_training(
                    self.model,
                    use_gradient_checkpointing=gradient_checkpointing
                )
                self.value_model = prepare_model_for_kbit_training(
                    self.value_model,
                    use_gradient_checkpointing=gradient_checkpointing
                )
                
                # Enable input gradients for k-bit training flows
                try:
                    self.model.enable_input_require_grads()
                    self.value_model.enable_input_require_grads()
                except Exception:
                    pass
            
            # Create LoRA config for causal LM (policy and reference models)
            lora_config_causal = LoraConfig(
                r=self.config.model.lora_r,
                lora_alpha=self.config.model.lora_alpha,
                lora_dropout=self.config.model.lora_dropout,
                target_modules=self.config.model.lora_target_modules,
                bias=self.config.model.lora_bias,
                task_type=TaskType.CAUSAL_LM,
            )
            
            # Create LoRA config for sequence classification (value and reward models)
            lora_config_seq_cls = LoraConfig(
                r=self.config.model.lora_r,
                lora_alpha=self.config.model.lora_alpha,
                lora_dropout=self.config.model.lora_dropout,
                target_modules=self.config.model.lora_target_modules,
                bias=self.config.model.lora_bias,
                task_type=TaskType.SEQ_CLS,
            )
            
            # Apply LoRA to policy model
            self.model = get_peft_model(self.model, lora_config_causal)
            
            # Apply LoRA to value model
            self.value_model = get_peft_model(self.value_model, lora_config_seq_cls)
            
            # Note: Reference and reward models typically remain frozen (no LoRA)
            # They serve as fixed baselines for PPO
            
            # Print trainable parameters for policy model
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            self.logger.info(f"{'QLoRA' if quantization_config else 'LoRA'} applied to policy model - "
                           f"Trainable params: {trainable_params:,} / {total_params:,} "
                           f"({100 * trainable_params / total_params:.2f}%)")
            
            # Print trainable parameters for value model
            trainable_params_value = sum(p.numel() for p in self.value_model.parameters() if p.requires_grad)
            total_params_value = sum(p.numel() for p in self.value_model.parameters())
            self.logger.info(f"{'QLoRA' if quantization_config else 'LoRA'} applied to value model - "
                           f"Trainable params: {trainable_params_value:,} / {total_params_value:,} "
                           f"({100 * trainable_params_value / total_params_value:.2f}%)")

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

        # Disable cache when using gradient checkpointing to avoid warnings and ensure training correctness
        if getattr(self.config.training, "gradient_checkpointing", False):
            try:
                self.model.config.use_cache = False
                self.value_model.config.use_cache = False
            except Exception:
                pass
        
        # Fix for PEFT models: Add gradient checkpointing control methods if missing
        # TRL's unwrap_model_for_generation expects these methods to exist
        if getattr(self.config.model, 'use_lora', False):
            self._add_gradient_checkpointing_methods(self.model)
            self._add_gradient_checkpointing_methods(self.value_model)
        
        self.logger.info("Models and tokenizer loaded successfully")
    
    def _add_gradient_checkpointing_methods(self, model):
        """Add gradient checkpointing control methods to PEFT models if they don't exist.
        
        This is needed because TRL's unwrap_model_for_generation context manager
        expects these methods, but PEFT wrappers don't always expose them.
        """
        if not hasattr(model, 'gradient_checkpointing_disable'):
            def gradient_checkpointing_disable():
                """Disable gradient checkpointing."""
                if hasattr(model, 'base_model'):
                    base = model.base_model
                    if hasattr(base, 'gradient_checkpointing_disable'):
                        base.gradient_checkpointing_disable()
                    elif hasattr(base, 'model') and hasattr(base.model, 'gradient_checkpointing_disable'):
                        base.model.gradient_checkpointing_disable()
            model.gradient_checkpointing_disable = gradient_checkpointing_disable
        
        if not hasattr(model, 'gradient_checkpointing_enable'):
            def gradient_checkpointing_enable():
                """Enable gradient checkpointing."""
                if hasattr(model, 'base_model'):
                    base = model.base_model
                    if hasattr(base, 'gradient_checkpointing_enable'):
                        base.gradient_checkpointing_enable()
                    elif hasattr(base, 'model') and hasattr(base.model, 'gradient_checkpointing_enable'):
                        base.model.gradient_checkpointing_enable()
            model.gradient_checkpointing_enable = gradient_checkpointing_enable

    def setup_data(self):
        """Load and prepare training datasets."""
        datasets = []
        total_examples = 0

        for dataset_info in self.config.data.datasets:
            subset_info = f" (subset: {dataset_info.subset})" if dataset_info.subset else ""
            self.logger.info(f"Loading dataset: {dataset_info.name}{subset_info} (split: {dataset_info.split})")

            # Load the dataset
            if dataset_info.subset:
                dataset = load_dataset(dataset_info.name, dataset_info.subset, split=dataset_info.split)
            else:
                dataset = load_dataset(dataset_info.name, split=dataset_info.split)

            datasets.append(dataset)
            total_examples += len(dataset)
            self.logger.info(f"Loaded {len(dataset)} examples from {dataset_info.name}")

        # Combine all datasets
        if len(datasets) == 1:
            combined_dataset = datasets[0]
        else:
            combined_dataset = concatenate_datasets(datasets)

        # Split into train and eval
        eval_samples = min(100, len(combined_dataset) // 10)
        train_dataset = combined_dataset.select(range(len(combined_dataset) - eval_samples))
        eval_dataset = combined_dataset.select(range(len(combined_dataset) - eval_samples, len(combined_dataset)))

        self.logger.info(f"Total dataset: {total_examples} examples from {len(datasets)} datasets")
        self.logger.info(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
        
        # Get dataset text field (default to "prompt" if not specified)
        dataset_text_field = getattr(self.config.data, 'prompt_column', 'prompt')
        
        # Get num_proc from config (default to 1 if not specified)
        dataset_num_proc = getattr(self.config.data, 'dataset_num_proc', 1)
        
        # Pre-tokenize datasets
        with PartialState().local_main_process_first():
            self.train_dataset = self.prepare_dataset(train_dataset, self.tokenizer, dataset_text_field, dataset_num_proc)
            self.eval_dataset = self.prepare_dataset(eval_dataset, self.tokenizer, dataset_text_field, dataset_num_proc)

    def prepare_dataset(self, dataset, tokenizer, dataset_text_field, dataset_num_proc):
        """Pre-tokenize the dataset before training; only collate during training"""
        def tokenize(element):
            outputs = tokenizer(
                element[dataset_text_field],
                padding=False,
            )
            return {"input_ids": outputs["input_ids"]}

        return dataset.map(
            tokenize,
            batched=True,
            remove_columns=dataset.column_names,
            num_proc=dataset_num_proc,
        )

    def setup_training_args(self) -> PPOConfig:
        """Create PPO training configuration."""
        # Set report_to based on wandb configuration
        report_to = ["wandb"] if self.config.wandb.enabled else []
        
        # Map our config to PPOConfig parameters
        ppo_config = PPOConfig(
            # Output and logging
            output_dir=self.config.training.output_dir,
            
            # Training hyperparameters
            learning_rate=self.config.training.learning_rate,
            per_device_train_batch_size=self.config.training.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            total_episodes=self.config.training.max_steps if self.config.training.max_steps > 0 else 10000,
            num_ppo_epochs=self.config.training.num_train_epochs,
            
            # PPO algorithm parameters
            gamma=getattr(self.config.training, 'gamma', 1.0),
            lam=getattr(self.config.training, 'lam', 0.95),
            cliprange=getattr(self.config.training, 'cliprange', 0.2),
            cliprange_value=getattr(self.config.training, 'cliprange_value', 0.2),
            vf_coef=getattr(self.config.training, 'vf_coef', 0.1),
            
            # Training configuration
            logging_steps=getattr(self.config.training, 'logging_steps', 10),
            save_steps=getattr(self.config.training, 'save_steps', 500),
            
            # Optimization
            bf16=getattr(self.config.training, 'bf16', True),
            gradient_checkpointing=getattr(self.config.training, 'gradient_checkpointing', True),
            
            # Dataset processing
            dataset_num_proc=getattr(self.config.data, 'dataset_num_proc', 1),
            
            # Wandb configuration
            report_to=report_to,
        )
        
        return ppo_config

    def setup_trainer(self):
        """Initialize the PPO trainer."""
        training_args = self.setup_training_args()

        self.trainer = TRLPPOTrainer(
            args=training_args,
            processing_class=self.tokenizer,
            model=self.model,
            ref_model=self.ref_policy,
            reward_model=self.reward_model,
            value_model=self.value_model,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
        )

        self.logger.info("PPO trainer initialized")

    def train(self):
        """Run the PPO training process."""
        self.logger.info("Starting PPO training...")

        # Setup components
        self.setup_model()
        self.setup_data()
        self.setup_trainer()

        # Train the model
        self.trainer.train()

        # Final save
        self.trainer.save_model(self.config.training.output_dir)
        self.logger.info("PPO training completed successfully")