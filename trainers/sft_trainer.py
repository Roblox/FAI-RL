import os, sys
import torch
import wandb
from datasets import load_dataset, concatenate_datasets, Dataset
from trl import SFTConfig, SFTTrainer as TRLSFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
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

        # Load main model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model.base_model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=self.config.model.low_cpu_mem_usage,
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

            # Get column names from config
            prompt_col = getattr(dataset_info, "prompt_column", "prompt")
            answer_col = getattr(dataset_info, "answer_column", "answer")

            # Standardize column names for SFT
            # SFT typically expects "text" column with full instruction-response pairs
            def standardize_example(example):
                # Combine prompt and answer into a single text field
                prompt_text = example.get(prompt_col, "")
                answer_text = example.get(answer_col, "")
                
                # Format as instruction-response pair
                text = f"{prompt_text}\n\n{answer_text}"
                
                return {"text": text}

            # Apply the standardization
            dataset = dataset.map(standardize_example, remove_columns=dataset.column_names)
            
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
            run_name=self.config.training.run_name or self.config.wandb.name,
            deepspeed=self.config.training.deepspeed_config,
            dataloader_num_workers=self.config.training.dataloader_num_workers,
            gradient_checkpointing=self.config.training.gradient_checkpointing,
            max_seq_length=self.config.data.max_length,
            dataloader_pin_memory=self.config.training.dataloader_pin_memory,
            save_only_model=self.config.training.save_only_model,
            dataloader_drop_last=self.config.training.dataloader_drop_last,
            prediction_loss_only=self.config.training.prediction_loss_only,
            report_to=report_to,
            packing=False,  # Set to True if you want to pack multiple examples into one sequence
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
