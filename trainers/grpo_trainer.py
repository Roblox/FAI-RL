import os, sys
import torch
import wandb
import re
from datasets import load_dataset, Dataset, concatenate_datasets
from trl import GRPOConfig, GRPOTrainer as TRLGRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import TaskType
from typing import Optional, List

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

class GRPOTrainer(BaseTrainer):
    """GRPO (Group Relative Policy Optimization) trainer implementation."""

    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.trainer = None
        self.model = None
        self.tokenizer = None
        self.train_dataset = None

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

            # Get column names from config with defaults for math datasets
            prompt_col = getattr(dataset_info, "prompt_column", "question")
            answer_col = getattr(dataset_info, "answer_column", "answer")

            # Handle different dataset formats
            if dataset_info.name == "openai/gsm8k":
                processed_dataset = dataset.map(
                    lambda example: GSM8KTemplate.format_for_training(example, prompt_col, answer_col)
                )
            elif dataset_info.name == "nvidia/OpenMathInstruct-2":
                processed_dataset = dataset.map(
                    lambda example: OpenMathInstructTemplate.format_for_training(example, prompt_col, answer_col)
                )
            else:
                raise ValueError(f"Dataset {dataset_info.name} doesn't have expected columns. "
                               f"Expected either ('{prompt_col}', '{answer_col}') or '{prompt_col}'")

            datasets.append(processed_dataset)
            total_examples += len(processed_dataset)
            self.logger.info(f"Loaded {len(processed_dataset)} examples from {dataset_info.name}")

        # Combine all datasets
        if len(datasets) == 1:
            self.train_dataset = datasets[0]
        else:
            self.train_dataset = concatenate_datasets(datasets)

        self.logger.info(f"Total dataset loaded with {total_examples} examples from {len(datasets)} datasets")

    def setup_training_args(self) -> GRPOConfig:
        """Create GRPO training configuration."""
        # Set report_to based on wandb configuration
        report_to = ["wandb"] if self.config.wandb.enabled else []
        
        # Set gradient checkpointing kwargs to use non-reentrant mode for DDP compatibility
        # This fixes the "Expected to mark a variable ready only once" error with LoRA + DDP
        gradient_checkpointing_kwargs = None
        if self.config.training.gradient_checkpointing:
            gradient_checkpointing_kwargs = {"use_reentrant": False}
        
        return GRPOConfig(
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
            dataloader_pin_memory=self.config.training.dataloader_pin_memory,
            save_only_model=self.config.training.save_only_model,
            dataloader_drop_last=self.config.training.dataloader_drop_last,
            prediction_loss_only=self.config.training.prediction_loss_only,
            report_to=report_to,
            ddp_find_unused_parameters=False,  # Critical for LoRA + DDP stability
            # GRPO specific parameters
            max_prompt_length=self.config.data.max_prompt_length,
            max_completion_length=self.config.data.max_length - self.config.data.max_prompt_length,
        )

    def setup_trainer(self):
        """Initialize the GRPO trainer."""
        training_args = self.setup_training_args()

        self.trainer = TRLGRPOTrainer(
            model=self.model,
            reward_funcs=[
                exact_match_reward_func,
                structured_xml_reward_func,
                digit_reward_func,
            ],
            args=training_args,
            train_dataset=self.train_dataset,
        )

        self.logger.info("GRPO trainer initialized")

    def train(self):
        """Run the training process."""
        self.logger.info("Starting GRPO training...")

        # Setup components
        self.setup_model()
        self.setup_data()
        self.setup_trainer()

        # Train the model
        self.trainer.train()

        # Final save
        self.trainer.save_model(self.config.training.output_dir)
        self.logger.info("GRPO training completed successfully")
