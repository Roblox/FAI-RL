import os, sys
import torch
import wandb
from datasets import load_dataset, Dataset, concatenate_datasets
from trl import PPOConfig, PPOTrainer as TRLPPOTrainer, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer, AutoModelForCausalLM
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
        self.ref_model = None
        self.tokenizer = None
        self.train_dataset = None

    def setup_model(self):
        """Load model and tokenizer with value head for PPO."""
        self.logger.info(f"Loading model: {self.config.model.base_model_name}")

        # Convert string dtype to torch dtype
        torch_dtype = getattr(torch, self.config.model.torch_dtype)

        # Load tokenizer first
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.base_model_name
        )

        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # Load model with value head for PPO
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
            self.config.model.base_model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=self.config.model.low_cpu_mem_usage,
        )

        # Load reference model (for KL penalty in PPO)
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            self.config.model.base_model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=self.config.model.low_cpu_mem_usage,
        )

        # Resize embeddings if tokenizer was modified
        if len(self.tokenizer) != self.model.pretrained_model.config.vocab_size:
            self.model.pretrained_model.resize_token_embeddings(len(self.tokenizer))
            self.ref_model.resize_token_embeddings(len(self.tokenizer))

        self.logger.info("Model with value head and tokenizer loaded successfully")

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

            # Handle different dataset formats - PPO needs both query and ground truth answer
            if dataset_info.name == "openai/gsm8k":
                def process_gsm8k(example):
                    query = GSM8KTemplate.format_prompt_only(example, prompt_col, answer_col)
                    # Extract ground truth answer
                    answer = example[answer_col]
                    final_answer = answer.split('####')[-1].strip() if '####' in answer else answer.strip()
                    return {"query": query, "ground_truth": final_answer}
                processed_dataset = dataset.map(process_gsm8k)
                
            elif dataset_info.name == "nvidia/OpenMathInstruct-2":
                def process_openmath(example):
                    query = OpenMathInstructTemplate.format_prompt_only(example, prompt_col, answer_col)
                    return {"query": query, "ground_truth": example[answer_col]}
                processed_dataset = dataset.map(process_openmath)
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

    def setup_training_args(self) -> PPOConfig:
        """Create PPO training configuration."""
        # Set report_to based on wandb configuration
        report_to = ["wandb"] if self.config.wandb.enabled else []
        
        return PPOConfig(
            # Basic training parameters
            model_name=self.config.model.base_model_name,
            learning_rate=self.config.training.learning_rate,
            batch_size=self.config.training.per_device_train_batch_size,
            mini_batch_size=max(1, self.config.training.per_device_train_batch_size // 2),
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            ppo_epochs=getattr(self.config.training, 'ppo_epochs', 4),
            
            # PPO specific parameters
            init_kl_coef=getattr(self.config.training, 'init_kl_coef', 0.2),
            target_kl=getattr(self.config.training, 'target_kl', 6.0),
            adap_kl_ctrl=getattr(self.config.training, 'adap_kl_ctrl', True),
            gamma=getattr(self.config.training, 'gamma', 1.0),
            lam=getattr(self.config.training, 'lam', 0.95),
            cliprange=getattr(self.config.training, 'cliprange', 0.2),
            cliprange_value=getattr(self.config.training, 'cliprange_value', 0.2),
            vf_coef=getattr(self.config.training, 'vf_coef', 0.1),
            
            # Generation parameters
            max_length=self.config.data.max_length,
            temperature=getattr(self.config.training, 'temperature', 1.0),
            top_k=getattr(self.config.training, 'top_k', 0),
            top_p=getattr(self.config.training, 'top_p', 1.0),
            do_sample=getattr(self.config.training, 'do_sample', True),
            
            # Training configuration
            max_grad_norm=getattr(self.config.training, 'max_grad_norm', 1.0),
            seed=getattr(self.config.training, 'seed', 0),
            
            # Logging and saving
            log_with=report_to,
            
            # Optimization flags
            use_score_scaling=getattr(self.config.training, 'use_score_scaling', False),
            use_score_norm=getattr(self.config.training, 'use_score_norm', False),
            score_clip=getattr(self.config.training, 'score_clip', None),
        )

    def compute_rewards(self, queries: List[str], responses: List[str], ground_truths: List[str]) -> List[float]:
        """Compute rewards for generated completions using reward functions."""
        # Apply reward functions
        try:
            # Use exact match reward with ground truth
            rewards = exact_match_reward_func(responses, ground_truths, logger=self.logger)
                
            # Add format reward as bonus
            format_rewards = structured_xml_reward_func(responses, logger=self.logger)
            
            # Add digit reward as bonus
            digit_rewards = digit_reward_func(
                [[{"content": resp}] for resp in responses], logger=self.logger
            )
            
            # Combine rewards
            final_rewards = [r + f + d for r, f, d in zip(rewards, format_rewards, digit_rewards)]
            
        except Exception as e:
            self.logger.warning(f"Error computing rewards: {e}. Using default rewards.")
            final_rewards = [0.0] * len(responses)
        
        return final_rewards

    def setup_trainer(self):
        """Initialize the PPO trainer."""
        training_args = self.setup_training_args()

        self.trainer = TRLPPOTrainer(
            config=training_args,
            model=self.model,
            ref_model=self.ref_model,
            tokenizer=self.tokenizer,
            dataset=self.train_dataset,
        )

        self.logger.info("PPO trainer initialized")

    def train(self):
        """Run the PPO training process."""
        self.logger.info("Starting PPO training...")

        # Setup components
        self.setup_model()
        self.setup_data()
        self.setup_trainer()

        # Use TRL's built-in training loop instead of manual implementation
        for batch in self.trainer.dataloader:
            query_tensors = batch["input_ids"]
            
            # Generate responses
            response_tensors, ref_log_probs = self.trainer.generate(
                query_tensors, 
                return_prompt=False,
                length_sampler=lambda: self.config.data.max_length - self.config.data.max_prompt_length,
                **{
                    "temperature": getattr(self.config.training, 'temperature', 1.0),
                    "top_p": getattr(self.config.training, 'top_p', 1.0),
                    "do_sample": getattr(self.config.training, 'do_sample', True),
                }
            )
            
            # Decode for reward computation
            queries = self.tokenizer.batch_decode(query_tensors, skip_special_tokens=True)
            responses = self.tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
            
            # Get ground truth answers from batch
            ground_truths = batch.get("ground_truth", [""] * len(queries))
            
            # Compute rewards
            rewards = self.compute_rewards(queries, responses, ground_truths)
            rewards = [torch.tensor(r, dtype=torch.float) for r in rewards]
            
            # PPO step
            stats = self.trainer.step(query_tensors, response_tensors, rewards)
            
            # Log stats
            self.trainer.log_stats(stats, batch, rewards)

        self.logger.info("PPO training completed successfully")

    def save_model(self, output_path: Optional[str] = None):
        """Save the trained model."""
        if output_path is None:
            output_path = self.config.training.output_dir

        if self.trainer is not None:
            # Save the model without the value head for inference
            self.trainer.model.pretrained_model.save_pretrained(output_path)
            self.tokenizer.save_pretrained(output_path)
            self.logger.info(f"Model saved to {output_path}")
        else:
            self.logger.warning("No trained model to save")
