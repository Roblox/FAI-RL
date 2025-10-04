import os, sys
import torch
import wandb
from datasets import load_dataset, concatenate_datasets, Dataset
from trl import SFTConfig, SFTTrainer as TRLSFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from typing import Optional, List

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.config import ExperimentConfig
from core.trainer_base import BaseTrainer
from utils.logging_utils import setup_logging


class SFTTrainer(BaseTrainer):
    """SFT (Supervised Fine-Tuning) trainer implementation using TRL."""

    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.trainer = None
        self.model = None
        self.tokenizer = None

    def setup_model(self):
        """Load model and tokenizer."""
        self.logger.info(f"Loading model: {self.config.model.base_model_name}")

        # Convert string dtype to torch dtype
        torch_dtype = getattr(torch, self.config.model.torch_dtype)

        # Create quantization config if needed
        quantization_config = None
        if self.config.model.load_in_4bit or self.config.model.load_in_8bit:
            self.logger.info(f"Setting up {'4-bit' if self.config.model.load_in_4bit else '8-bit'} quantization...")
            # Guard: quantized fine-tuning requires LoRA/PEFT adapters. Without adapters there
            # would be zero trainable parameters and DeepSpeed would crash when configuring the optimizer.
            if not self.config.model.use_lora:
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
            # When training with DeepSpeed, let DeepSpeed/Accelerate manage device placement.
            # Setting device_map="auto" is intended for single-GPU inference/training and can
            # lead to empty optimizer parameter groups under ZeRO-3.
            if not using_deepspeed:
                # For multi-GPU training with torchrun (no DeepSpeed), we need to place the model
                # on the current device for each process to avoid device mismatch errors.
                if torch.cuda.is_available():
                    current_device = torch.cuda.current_device()
                    model_kwargs["device_map"] = {"": current_device}
                    self.logger.info(f"Using device_map={{'': {current_device}}} for quantized model (no DeepSpeed).")
                else:
                    model_kwargs["device_map"] = "auto"
                    self.logger.info("Using device_map=auto for quantized model (no DeepSpeed, no CUDA).")
            else:
                self.logger.info("DeepSpeed detected; not setting device_map to let DeepSpeed place parameters.")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model.base_model_name,
            **model_kwargs
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.base_model_name
        )

        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        # Resize embeddings
        self.model.resize_token_embeddings(len(self.tokenizer))

        # Apply LoRA if enabled (including QLoRA)
        if self.config.model.use_lora:
            self.logger.info("Applying LoRA configuration...")
            
            # Prepare model for k-bit training if using quantization
            if self.config.model.load_in_4bit or self.config.model.load_in_8bit:
                self.logger.info("Preparing model for k-bit training (QLoRA)...")
                self.model = prepare_model_for_kbit_training(
                    self.model,
                    use_gradient_checkpointing=self.config.training.gradient_checkpointing
                )
                # Ensure input gradients are enabled for k-bit training flows
                try:
                    self.model.enable_input_require_grads()
                except Exception:
                    pass
            
            # Create LoRA config
            lora_config = LoraConfig(
                r=self.config.model.lora_r,
                lora_alpha=self.config.model.lora_alpha,
                lora_dropout=self.config.model.lora_dropout,
                target_modules=self.config.model.lora_target_modules,
                bias=self.config.model.lora_bias,
                task_type=TaskType.CAUSAL_LM,
            )
            
            # Apply LoRA to model
            self.model = get_peft_model(self.model, lora_config)
            
            # Print trainable parameters
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
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

        # Disable cache when using gradient checkpointing to avoid warnings and ensure training correctness
        if getattr(self.config.training, "gradient_checkpointing", False):
            try:
                self.model.config.use_cache = False
            except Exception:
                pass

        self.logger.info("Model and tokenizer loaded successfully")

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

            # Get system prompt from config
            system_prompt = self.config.data.system_prompt
            
            # Get dataset columns from config
            dataset_columns = getattr(dataset_info, "dataset_columns", None)
            
            # Standardize column names for SFT
            if system_prompt and dataset_columns:
                # Use system prompt as a template with dataset columns
                def format_with_system_prompt(example):
                    # Create a dictionary of placeholders from dataset columns
                    format_dict = {}
                    for col in dataset_columns:
                        if col in example:
                            format_dict[col] = example[col]
                    
                    # Format the system prompt with the values from the dataset
                    try:
                        text = system_prompt.format(**format_dict)
                    except KeyError as e:
                        self.logger.warning(f"Missing key in system prompt template: {e}")
                        text = system_prompt
                    
                    return {"text": text}
                
                dataset = dataset.map(format_with_system_prompt, remove_columns=dataset.column_names)
            
            datasets.append(dataset)
            total_examples += len(dataset)
            self.logger.info(f"Loaded {len(dataset)} examples from {dataset_info.name}")

        # Combine all datasets
        if len(datasets) == 1:
            self.train_dataset = datasets[0]
        else:
            self.train_dataset = concatenate_datasets(datasets)

        self.logger.info(f"Total dataset loaded with {total_examples} examples from {len(datasets)} datasets")

    def setup_training_args(self) -> SFTConfig:
        """Create SFT training configuration."""
        # Set report_to based on wandb configuration to prevent automatic wandb initialization
        report_to = ["wandb"] if self.config.wandb.enabled else []
        
        return SFTConfig(
            output_dir=self.config.training.output_dir,
            per_device_train_batch_size=self.config.training.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            learning_rate=self.config.training.learning_rate,
            num_train_epochs=self.config.training.num_train_epochs,
            max_steps=self.config.training.max_steps,
            logging_steps=self.config.training.logging_steps,
            save_steps=self.config.training.save_steps,
            eval_steps=self.config.training.eval_steps,
            warmup_steps=self.config.training.warmup_steps,
            bf16=self.config.training.bf16,
            fp16=self.config.training.fp16,
            remove_unused_columns=self.config.data.remove_unused_columns,
            deepspeed=self.config.training.deepspeed_config,
            dataloader_num_workers=self.config.training.dataloader_num_workers,
            gradient_checkpointing=self.config.training.gradient_checkpointing,
            dataloader_pin_memory=self.config.training.dataloader_pin_memory,
            dataloader_drop_last=self.config.training.dataloader_drop_last,
            report_to=report_to
        )

    def setup_trainer(self):
        """Initialize the SFT trainer."""
        training_args = self.setup_training_args()

        self.trainer = TRLSFTTrainer(
            model=self.model,
            args=training_args,
            processing_class=self.tokenizer,
            train_dataset=self.train_dataset,
        )

        self.logger.info("SFT trainer initialized")

    def train(self):
        """Run the training process."""
        self.logger.info("Starting SFT training...")

        # Setup components
        self.setup_model()
        self.setup_data()
        self.setup_trainer()

        # Train the model
        self.trainer.train()

        # Final save
        self.trainer.save_model(self.config.training.output_dir)
        self.logger.info("SFT training completed successfully")
