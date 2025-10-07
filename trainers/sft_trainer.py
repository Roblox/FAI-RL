import os, sys
import torch
import wandb
from datasets import load_dataset, concatenate_datasets, Dataset
from trl import SFTConfig, SFTTrainer as TRLSFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import TaskType
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

        # Create quantization config using base class method
        quantization_config = self.create_quantization_config()
        
        # Prepare model kwargs using base class method
        model_kwargs = self.prepare_model_kwargs(quantization_config)
        
        # Load main model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model.base_model_name,
            **model_kwargs
        )

        # Setup tokenizer and resize embeddings using base class method
        self.tokenizer = self.setup_tokenizer_with_model(self.model)

        # Apply LoRA if enabled (including QLoRA) using base class method
        self.model = self.apply_lora_to_model(self.model, TaskType.CAUSAL_LM, quantization_config)

        # Disable cache when using gradient checkpointing using base class method
        self.disable_cache_for_gradient_checkpointing(self.model)

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
