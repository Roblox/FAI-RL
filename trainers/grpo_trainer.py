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
from utils.dataset_utils import is_math_dataset, is_subjective_dataset, get_template_for_dataset
from .rewards.accuracy_rewards import exact_match_reward_func, digit_reward_func
from .rewards.format_rewards import structured_xml_reward_func
from .rewards.custom_rewards import custom_reward_func
from .rewards.subjective_rewards import subjective_api_reward_func_simple

class GRPOTrainer(BaseTrainer):
    """GRPO (Group Relative Policy Optimization) trainer implementation."""

    def __init__(self, config: ExperimentConfig, logger: Optional[object] = None):
        super().__init__(config, logger=logger)
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
        total_skipped = 0

        # Determine if we're in subjective mode (based on reward_api presence or dataset type)
        use_subjective = any(
            is_subjective_dataset(ds.name) for ds in self.config.data.datasets
        )
        if hasattr(self.config, 'reward_api') and self.config.reward_api is not None:
            use_subjective = True

        for dataset_info in self.config.data.datasets:
            subset_info = f" (subset: {dataset_info.subset})" if dataset_info.subset else ""
            self.logger.info(f"Loading dataset: {dataset_info.name}{subset_info} (split: {dataset_info.split})")

            # Load the dataset
            if dataset_info.subset:
                dataset = load_dataset(dataset_info.name, dataset_info.subset, split=dataset_info.split)
            else:
                dataset = load_dataset(dataset_info.name, split=dataset_info.split)

            original_size = len(dataset)

            # Get column names from config with defaults for math datasets
            prompt_col = getattr(dataset_info, "prompt_column", "question")
            answer_col = getattr(dataset_info, "answer_column", "answer")

            # Get appropriate template for this dataset
            template_class = get_template_for_dataset(dataset_info.name, logger=self.logger, use_subjective=use_subjective)
            processed_dataset = dataset.map(
                lambda example: template_class.format_for_training(example, prompt_col, answer_col)
            )

            # Filter out invalid rows where prompt is None or empty
            # For subjective datasets, answer is optional (model generates it)
            def is_valid_example(example):
                """Check if example has valid prompt field."""
                prompt = example.get("prompt")
                
                # Check prompt validity - can be string or list of dicts
                prompt_valid = False
                if isinstance(prompt, str):
                    prompt_valid = prompt.strip() != ""
                elif isinstance(prompt, list) and len(prompt) > 0:
                    prompt_valid = True
                
                # For subjective datasets, only require valid prompt
                # For math datasets, also require valid answer
                if use_subjective:
                    return prompt_valid
                
                # Check answer validity for non-subjective datasets
                answer = example.get("answer")
                answer_valid = answer is not None and isinstance(answer, str) and answer.strip() != ""
                
                return prompt_valid and answer_valid
            
            processed_dataset = processed_dataset.filter(is_valid_example)
            
            skipped = original_size - len(processed_dataset)
            total_skipped += skipped
            
            if skipped > 0:
                field_info = "prompt" if use_subjective else "prompt/answer"
                self.logger.warning(
                    f"Skipped {skipped} invalid examples from {dataset_info.name} "
                    f"(missing or empty '{field_info}' field(s))"
                )

            datasets.append(processed_dataset)
            total_examples += len(processed_dataset)
            self.logger.info(f"Loaded {len(processed_dataset)} valid examples from {dataset_info.name}")

        # Combine all datasets
        if len(datasets) == 1:
            self.train_dataset = datasets[0]
        else:
            self.train_dataset = concatenate_datasets(datasets)

        # Convert message dictionaries to strings using chat template
        # This is required because GRPO expects prompts to be strings, not message dicts
        def apply_chat_template(example):
            prompt = example['prompt']
            # If prompt is a list of message dicts, apply chat template
            if isinstance(prompt, list) and len(prompt) > 0 and isinstance(prompt[0], dict):
                example['prompt'] = self.tokenizer.apply_chat_template(
                    prompt, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            return example
        
        self.train_dataset = self.train_dataset.map(apply_chat_template)

        if total_skipped > 0:
            self.logger.warning(f"Total examples skipped across all datasets: {total_skipped}")

        self.logger.info(f"Total dataset loaded with {total_examples} valid examples from {len(datasets)} datasets")

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
            num_generations=self.config.training.num_generations,
            max_prompt_length=self.config.data.max_prompt_length,
            max_completion_length=self.config.data.max_length - self.config.data.max_prompt_length,
        )

    def setup_trainer(self):
        """Initialize the GRPO trainer."""
        training_args = self.setup_training_args()

        # Check if any dataset uses subjective evaluation
        use_subjective = any(
            is_subjective_dataset(ds.name) for ds in self.config.data.datasets
        )
        
        # Also check if reward_api config is present (indicates subjective mode)
        if hasattr(self.config, 'reward_api') and self.config.reward_api is not None:
            use_subjective = True
        
        if use_subjective:
            self.logger.info("Using subjective reward functions (API-based evaluation)")
            
            # Get API configuration
            api_endpoint = None
            api_key = None
            api_model = None
            if hasattr(self.config, 'reward_api') and self.config.reward_api is not None:
                api_endpoint = self.config.reward_api.endpoint
                api_key = self.config.reward_api.key
                api_model = getattr(self.config.reward_api, 'model', None)
            
            if not api_endpoint or not api_key:
                raise ValueError(
                    "Reward API configuration is required when using subjective rewards. "
                    "Please set 'reward_api.endpoint' and 'reward_api.key' in your config file."
                )
            
            # Log full reward_api configuration
            self.logger.debug("=" * 60)
            self.logger.debug("REWARD API CONFIGURATION:")
            self.logger.debug(f"  endpoint: {api_endpoint}")
            self.logger.debug(f"  model: {api_model}")
            self.logger.debug(f"  key: {'*' * 8}...{api_key[-8:] if api_key and len(api_key) > 8 else '(hidden)'}")
            self.logger.debug("=" * 60)
            
            # Create wrapper function for subjective rewards
            def subjective_with_logger(prompts, completions, **kwargs):
                kwargs['logger'] = self.logger
                # Pass API configuration if available
                if api_endpoint:
                    kwargs['api_endpoint'] = api_endpoint
                if api_key:
                    kwargs['api_key'] = api_key
                if api_model:
                    kwargs['api_model'] = api_model
                # Pass num_generations from training config
                kwargs['num_generations'] = self.config.training.num_generations
                # Pass prompts to the reward function
                kwargs['prompt'] = prompts
                return subjective_api_reward_func_simple(completions, **kwargs)
            
            reward_funcs = [subjective_with_logger]
        else:
            # Use math-specific reward functions
            self.logger.info("Using math-specific reward functions")
            
            # Create wrapper functions that inject the logger into reward functions
            def exact_match_with_logger(prompts, completions, answer, **kwargs):
                kwargs['logger'] = self.logger
                return exact_match_reward_func(completions, answer, **kwargs)
            
            def structured_xml_with_logger(prompts, completions, **kwargs):
                kwargs['logger'] = self.logger
                return structured_xml_reward_func(completions, **kwargs)
            
            def digit_with_logger(prompts, completions, **kwargs):
                kwargs['logger'] = self.logger
                return digit_reward_func(completions, **kwargs)
            
            # ========== CUSTOM REWARD FUNCTION ==========
            # Add your custom reward logic here
            def custom_with_logger(prompts, completions, **kwargs):
                kwargs['logger'] = self.logger
                return custom_reward_func(completions, **kwargs)
            # ============================================
            
            reward_funcs = [
                exact_match_with_logger,
                structured_xml_with_logger,
                digit_with_logger,
                custom_with_logger,  # Add your custom reward function here
            ]

        self.trainer = TRLGRPOTrainer(
            model=self.model,
            reward_funcs=reward_funcs,
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
