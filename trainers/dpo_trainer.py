import os, sys
import torch
import wandb
from datasets import load_dataset, concatenate_datasets, Dataset
from trl import DPOConfig, DPOTrainer as TRLDPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import TaskType
from typing import Optional, List

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.config import ExperimentConfig
from core.trainer_base import BaseTrainer
from utils.logging_utils import setup_logging


class DPOTrainer(BaseTrainer):
    """DPO (Direct Preference Optimization) trainer implementation."""

    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.trainer = None
        self.model = None
        self.ref_model = None
        self.tokenizer = None

    def setup_model(self):
        """Load model and tokenizer."""
        self.logger.info(f"Loading model: {self.config.model.base_model_name}")

        # Create quantization config using base class method
        quantization_config = self.create_quantization_config()
        
        # Prepare model kwargs using base class method
        model_kwargs = self.prepare_model_kwargs(quantization_config)

        # Load main model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model.base_model_name,
            **model_kwargs
        )

        # Load reference model (required for DPO)
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            self.config.model.base_model_name,
            **model_kwargs
        )

        # Setup tokenizer and resize embeddings using base class method
        self.tokenizer = self.setup_tokenizer_with_model(self.model)
        
        # Resize embeddings for reference model
        self.ref_model.resize_token_embeddings(len(self.tokenizer))

        # Apply LoRA if enabled (including QLoRA) using base class method
        # Note: Only apply LoRA to the main model, not the reference model
        # The reference model should remain frozen as a baseline
        self.model = self.apply_lora_to_model(self.model, TaskType.CAUSAL_LM, quantization_config)

        # Disable cache when using gradient checkpointing using base class method
        self.disable_cache_for_gradient_checkpointing(self.model)
        self.disable_cache_for_gradient_checkpointing(self.ref_model)

        self.logger.info("Model and tokenizer loaded successfully")

    def setup_data(self):
        """Load and prepare training datasets."""
        datasets = []
        total_examples = 0
        total_skipped = 0

        for dataset_info in self.config.data.datasets:
            subset_info = f" (subset: {dataset_info.subset})" if dataset_info.subset else ""
            self.logger.info(f"Loading dataset: {dataset_info.name}{subset_info} (split: {dataset_info.split})")

            # Load the dataset
            if dataset_info.subset:
                dataset = load_dataset(dataset_info.name, dataset_info.subset, split=dataset_info.split)
            else:
                dataset = load_dataset(dataset_info.name, split=dataset_info.split)

            original_size = len(dataset)

            # Get column names from config
            prompt_col = getattr(dataset_info, "prompt_column", "prompt")
            chosen_col = getattr(dataset_info, "chosen_column", "chosen")
            rejected_col = getattr(dataset_info, "rejected_column", "rejected")

            # Standardize column names and handle missing prompt columns
            def standardize_example(example):
                standardized = {}
                
                # Handle prompt - use empty string if column doesn't exist
                if prompt_col in example:
                    standardized["prompt"] = example[prompt_col]
                else:
                    standardized["prompt"] = " "
                    
                # Handle chosen and rejected - these are required
                standardized["chosen"] = example[chosen_col]
                standardized["rejected"] = example[rejected_col]
                
                return standardized

            # Apply the standardization
            dataset = dataset.map(standardize_example, remove_columns=dataset.column_names)
            
            # Filter out invalid rows where chosen or rejected are None or empty
            def is_valid_example(example):
                """Check if example has valid chosen and rejected fields."""
                chosen = example.get("chosen")
                rejected = example.get("rejected")
                
                # Check if fields exist and are non-empty strings
                chosen_valid = chosen is not None and isinstance(chosen, str) and chosen.strip() != ""
                rejected_valid = rejected is not None and isinstance(rejected, str) and rejected.strip() != ""
                
                return chosen_valid and rejected_valid
            
            # Apply the filter
            dataset = dataset.filter(is_valid_example)
            
            skipped = original_size - len(dataset)
            total_skipped += skipped
            
            if skipped > 0:
                self.logger.warning(
                    f"Skipped {skipped} invalid examples from {dataset_info.name} "
                    f"(missing or empty 'chosen'/'rejected' fields)"
                )
            
            datasets.append(dataset)
            total_examples += len(dataset)
            self.logger.info(f"Loaded {len(dataset)} valid examples from {dataset_info.name}")

        # Combine all datasets
        if len(datasets) == 1:
            self.train_dataset = datasets[0]
        else:
            self.train_dataset = concatenate_datasets(datasets)

        if total_skipped > 0:
            self.logger.warning(f"Total examples skipped across all datasets: {total_skipped}")
        
        self.logger.info(f"Total dataset loaded with {total_examples} valid examples from {len(datasets)} datasets")

    def setup_training_args(self) -> DPOConfig:
        """Create DPO training configuration."""
        # Set report_to based on wandb configuration to prevent automatic wandb initialization
        report_to = ["wandb"] if self.config.wandb.enabled else []
        
        # Set gradient checkpointing kwargs to use non-reentrant mode for DDP compatibility
        # This fixes the "Expected to mark a variable ready only once" error with LoRA + DDP
        gradient_checkpointing_kwargs = None
        if self.config.training.gradient_checkpointing:
            gradient_checkpointing_kwargs = {"use_reentrant": False}
        
        return DPOConfig(
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
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
            max_length=self.config.data.max_length,
            max_prompt_length=self.config.data.max_prompt_length,
            dataloader_pin_memory=self.config.training.dataloader_pin_memory,
            save_only_model=self.config.training.save_only_model,
            dataloader_drop_last=self.config.training.dataloader_drop_last,
            prediction_loss_only=self.config.training.prediction_loss_only,
            report_to=report_to,
            ddp_find_unused_parameters=False,  # Critical for LoRA + DDP stability
        )

    def setup_trainer(self):
        """Initialize the DPO trainer."""
        training_args = self.setup_training_args()

        self.trainer = TRLDPOTrainer(
            model=self.model,
            ref_model=self.ref_model,
            args=training_args,
            processing_class=self.tokenizer,
            train_dataset=self.train_dataset,
        )

        self.logger.info("DPO trainer initialized")

    def train(self):
        """Run the training process."""
        self.logger.info("Starting DPO training...")

        # Setup components
        self.setup_model()
        self.setup_data()
        self.setup_trainer()

        # Train the model
        self.trainer.train()

        # Final save
        self.trainer.save_model(self.config.training.output_dir)
        self.logger.info("DPO training completed successfully")
