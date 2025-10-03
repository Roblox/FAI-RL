import os, sys
import torch
import wandb
from accelerate import PartialState
from datasets import load_dataset, Dataset, concatenate_datasets
from trl import PPOConfig, PPOTrainer as TRLPPOTrainer
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
)
from typing import Optional, List
import numpy as np

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
        if self.config.model.load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        elif self.config.model.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        
        model_kwargs = dict(
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=self.config.model.low_cpu_mem_usage,
        )
        
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = "auto"

        # Load tokenizer first
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.base_model_name,
            padding_side="left",
        )

        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        
        # Load policy model (main model for generation)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model.base_model_name,
            **model_kwargs
        )
        
        # Load reference model (for KL penalty in PPO)
        self.ref_policy = AutoModelForCausalLM.from_pretrained(
            self.config.model.value_model_name,
            **model_kwargs
        )
        
        # Load value model (for advantage estimation)
        self.value_model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model.value_model_name,
            num_labels=1,
            **model_kwargs
        )
        
        # Load reward model (for computing rewards)
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model.value_model_name,
            num_labels=1,
            **model_kwargs
        )

        # Resize embeddings if tokenizer was modified
        # if len(self.tokenizer) != self.model.config.vocab_size:
        #     self.model.resize_token_embeddings(len(self.tokenizer))
        #     self.ref_policy.resize_token_embeddings(len(self.tokenizer))
        #     self.value_model.resize_token_embeddings(len(self.tokenizer))
        #     self.reward_model.resize_token_embeddings(len(self.tokenizer))

        self.logger.info("Models and tokenizer loaded successfully")

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
            
            # PPO algorithm parameters - UNCOMMENT THESE
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
            dataset_num_proc=1,
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

        # Monkeypatch to fix DDP config access issue
        self._patch_trl_trainer()

        self.logger.info("PPO trainer initialized")

    def _patch_trl_trainer(self):
        """Patch TRL trainer to handle DDP model wrapping correctly."""
        original_create_model_card = self.trainer.create_model_card
        
        def patched_create_model_card(model_name: Optional[str] = None, *args, **kwargs):
            """Fixed version that handles DDP-wrapped models."""
            from torch.nn.parallel import DistributedDataParallel
            
            # Get the actual model (unwrap DDP if needed)
            model = self.trainer.model
            if isinstance(model, DistributedDataParallel):
                model = model.module
            
            # Temporarily replace self.trainer.model for the method call
            original_model = self.trainer.model
            self.trainer.model = model
            
            try:
                return original_create_model_card(model_name, *args, **kwargs)
            finally:
                # Restore original model reference
                self.trainer.model = original_model
        
        self.trainer.create_model_card = patched_create_model_card

    def train(self):
        """Run the PPO training process."""
        self.logger.info("Starting PPO training...")

        # Setup components
        self.setup_model()
        self.setup_data()
        self.setup_trainer()

        # Run training - the trainer handles everything internally
        self.trainer.train()

        # Final save
        self.trainer.save_model(self.config.training.output_dir)
        self.logger.info("PPO training completed successfully")

