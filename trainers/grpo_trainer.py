import contextlib
import os, sys
import torch
import wandb
import re
from datasets import load_dataset, Dataset, concatenate_datasets
from trl import GRPOConfig, GRPOTrainer as TRLGRPOTrainer
from trl.generation.vllm_generation import VLLMGeneration
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.integrations import deepspeed as hf_deepspeed
from peft import TaskType
from typing import Optional, List

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.config import ExperimentConfig
from core.trainer_base import BaseTrainer
from utils.logging_utils import setup_logging
from utils.dataset_utils import load_raw_dataset
from .rewards.factory import build_reward_function


def vllm_rollout_kwargs(training) -> dict:
    """GRPOConfig fields that move rollout generation to vLLM when use_vllm is set."""
    return {
        "use_vllm": training.use_vllm,
        "vllm_mode": training.vllm_mode,
        "vllm_gpu_memory_utilization": float(training.vllm_gpu_memory_utilization),
        "vllm_tensor_parallel_size": training.vllm_tensor_parallel_size,
        "vllm_max_model_length": training.vllm_max_model_length,
        "vllm_enable_sleep_mode": training.vllm_enable_sleep_mode,
        "vllm_server_base_url": training.vllm_server_base_url,
        "vllm_server_host": training.vllm_server_host,
        "vllm_server_port": training.vllm_server_port,
        "vllm_server_timeout": float(training.vllm_server_timeout),
    }


@contextlib.contextmanager
def vllm_engine_outside_zero3():
    """Build TRL's vLLM engine with HF's ZeRO-3 detection switched off.

    The HfDeepSpeedConfig registered for training is process-global. vLLM's
    multimodal models (Gemma 4's vision and audio towers) build submodules
    with transformers' ``from_config``, which then wraps them in
    ``deepspeed.zero.Init``; the size-0 partitioned parameters fail vLLM's
    weight loader. Only the engine build is affected: the rest of TRL's
    trainer construction (the ZeRO-3 reference model) still sees ZeRO-3.
    """
    init_vllm = VLLMGeneration._init_vllm

    def init_vllm_outside_zero3(self):
        # transformers exposes no getter for the registered config, only this weakref.
        registered = hf_deepspeed._hf_deepspeed_config_weak_ref
        hf_deepspeed.unset_hf_deepspeed_config()
        try:
            return init_vllm(self)
        finally:
            hf_deepspeed._hf_deepspeed_config_weak_ref = registered

    VLLMGeneration._init_vllm = init_vllm_outside_zero3
    try:
        yield
    finally:
        VLLMGeneration._init_vllm = init_vllm


class GRPOTrainer(BaseTrainer):
    """GRPO (Group Relative Policy Optimization) trainer implementation."""

    supports_early_stopping = False

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
        self.model = self.load_base_model_for_training(model_kwargs)

        # Setup tokenizer and resize embeddings using base class method
        self.tokenizer = self.setup_tokenizer_with_model(self.model)

        # GRPO trains on text prompts/completions only -- no image or audio
        # inputs. Freeze and LoRA-exclude Gemma 4 E-series' unused vision/audio
        # encoders before adapter injection so plain DDP does not see unused
        # adapters.
        self.prepare_model_for_modalities(
            self.model, use_vision=False, use_audio=False
        )

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

        for dataset_info in self.config.data.datasets:
            subset_info = f" (subset: {dataset_info.subset})" if dataset_info.subset else ""
            self.logger.info(f"Loading dataset: {dataset_info.name}{subset_info} (split: {dataset_info.split})")

            dataset = load_raw_dataset(dataset_info)

            original_size = len(dataset)

            # Map the configured source prompt into TRL's expected column while
            # preserving every other dataset column for the reward API context.
            prompt_col = getattr(dataset_info, "prompt_column", "question")
            answer_col = getattr(dataset_info, "answer_column", None)

            def map_configured_columns(example):
                mapped = {"prompt": example.get(prompt_col)}
                if answer_col and answer_col in example:
                    mapped["answer"] = example.get(answer_col)
                return mapped

            processed_dataset = dataset.map(map_configured_columns)

            # A ground-truth answer is not mandatory: the external reward API can
            # score against any dataset columns included in its context.
            def is_valid_example(example):
                """Check if an example has a usable configured prompt."""
                prompt = example.get("prompt")
                if isinstance(prompt, str):
                    return bool(prompt.strip())
                return isinstance(prompt, list) and len(prompt) > 0
            
            processed_dataset = processed_dataset.filter(is_valid_example)
            
            skipped = original_size - len(processed_dataset)
            total_skipped += skipped
            
            if skipped > 0:
                self.logger.warning(
                    f"Skipped {skipped} invalid examples from {dataset_info.name} "
                    f"(missing or empty configured prompt column '{prompt_col}')"
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
            # Existing conversational prompts pass through directly. For a plain
            # prompt, an optional configured system instruction creates a chat.
            if isinstance(prompt, list) and len(prompt) > 0 and isinstance(prompt[0], dict):
                example['prompt'] = self.tokenizer.apply_chat_template(
                    prompt, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            elif self.config.data.system_prompt:
                example['prompt'] = self.tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": self.config.data.system_prompt},
                        {"role": "user", "content": str(prompt)},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
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
            ddp_find_unused_parameters=self.config.training.ddp_find_unused_parameters,
            # GRPO specific parameters
            num_generations=self.config.training.num_generations,
            max_completion_length=self.config.data.max_length - self.config.data.max_prompt_length,
            beta=float(self.config.training.beta),
            temperature=float(self.config.training.temperature),
            **vllm_rollout_kwargs(self.config.training),
        )

    def setup_trainer(self):
        """Initialize the GRPO trainer."""
        training_args = self.setup_training_args()

        reward_func = build_reward_function(self.config, logger=self.logger)

        with vllm_engine_outside_zero3():
            self.trainer = TRLGRPOTrainer(
                model=self.model,
                reward_funcs=[reward_func],
                args=training_args,
                train_dataset=self.train_dataset,
                callbacks=self.build_callbacks(),
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
        self.train_with_resume()

        # Final save
        self.trainer.save_model(self.config.training.output_dir)
        self.logger.info("GRPO training completed successfully")
