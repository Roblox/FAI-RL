import os, sys
from datasets import load_dataset, concatenate_datasets
from trl import SFTConfig, SFTTrainer as TRLSFTTrainer
from transformers import AutoModelForCausalLM
from peft import TaskType
from typing import Optional

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.config import ExperimentConfig
from core.trainer_base import BaseTrainer


class CPTTrainer(BaseTrainer):
    """Continuous Pre-Training (CPT) trainer implementation using TRL.

    Trains a model on raw text with causal language modeling loss (next-token prediction).
    Unlike SFT, there is no system prompt templating or chat formatting — just raw text.
    """

    def __init__(self, config: ExperimentConfig, logger: Optional[object] = None):
        super().__init__(config, logger=logger)
        self.trainer = None
        self.model = None
        self.tokenizer = None

    def setup_model(self):
        """Load model and tokenizer."""
        self.logger.info(f"Loading model: {self.config.model.base_model_name}")

        quantization_config = self.create_quantization_config()
        model_kwargs = self.prepare_model_kwargs(quantization_config)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model.base_model_name,
            **model_kwargs
        )

        self.tokenizer = self.setup_tokenizer_with_model(self.model)
        self.model = self.apply_lora_to_model(self.model, TaskType.CAUSAL_LM, quantization_config)
        self.disable_cache_for_gradient_checkpointing(self.model)

        self.logger.info("Model and tokenizer loaded successfully")

    def setup_data(self):
        """Load and prepare training datasets.

        Loads raw text datasets for continuous pre-training. Each dataset must
        contain a 'text' column. No system prompt or chat template formatting
        is applied — text is used as-is for next-token prediction.
        """
        datasets = []
        total_examples = 0
        total_skipped = 0

        for dataset_info in self.config.data.datasets:
            subset_info = f" (subset: {dataset_info.subset})" if dataset_info.subset else ""
            self.logger.info(f"Loading dataset: {dataset_info.name}{subset_info} (split: {dataset_info.split})")

            if dataset_info.subset:
                dataset = load_dataset(dataset_info.name, dataset_info.subset, split=dataset_info.split)
            else:
                dataset = load_dataset(dataset_info.name, split=dataset_info.split)

            original_size = len(dataset)

            if "text" not in dataset.column_names:
                raise ValueError(
                    f"Dataset '{dataset_info.name}' does not contain a 'text' column. "
                    f"Available columns: {dataset.column_names}. "
                    f"CPT requires a 'text' column with raw text for next-token prediction."
                )

            def is_valid_example(example):
                text = example.get("text")
                return text is not None and isinstance(text, str) and text.strip() != ""

            dataset = dataset.filter(is_valid_example)

            skipped = original_size - len(dataset)
            total_skipped += skipped

            if skipped > 0:
                self.logger.warning(
                    f"Skipped {skipped} invalid examples from {dataset_info.name} "
                    f"(empty 'text' field)"
                )

            datasets.append(dataset)
            total_examples += len(dataset)
            self.logger.info(f"Loaded {len(dataset)} valid examples from {dataset_info.name}")

        if len(datasets) == 1:
            self.train_dataset = datasets[0]
        else:
            self.train_dataset = concatenate_datasets(datasets)

        if total_skipped > 0:
            self.logger.warning(f"Total examples skipped across all datasets: {total_skipped}")

        self.logger.info(f"Total dataset loaded with {total_examples} valid examples from {len(datasets)} datasets")

    def setup_training_args(self) -> SFTConfig:
        """Create CPT training configuration.

        Uses SFTConfig with dataset_text_field="text" and packing=True.
        Packing concatenates short text samples to fill max_seq_length,
        which is standard practice for pre-training efficiency.
        """
        report_to = ["wandb"] if self.config.wandb.enabled else []

        gradient_checkpointing_kwargs = None
        if self.config.training.gradient_checkpointing:
            gradient_checkpointing_kwargs = {"use_reentrant": False}

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
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
            dataloader_pin_memory=self.config.training.dataloader_pin_memory,
            dataloader_drop_last=self.config.training.dataloader_drop_last,
            report_to=report_to,
            ddp_find_unused_parameters=False,
        )

    def setup_trainer(self):
        """Initialize the CPT trainer."""
        training_args = self.setup_training_args()

        self.trainer = TRLSFTTrainer(
            model=self.model,
            args=training_args,
            processing_class=self.tokenizer,
            train_dataset=self.train_dataset,
            callbacks=self.build_callbacks(),
            dataset_text_field="text",
            packing=True,
            max_seq_length=self.config.data.max_length,
        )

        self.logger.info("CPT trainer initialized")

    def train(self):
        """Run the training process."""
        self.logger.info("Starting CPT (Continuous Pre-Training)...")

        self.setup_model()
        self.setup_data()
        self.setup_trainer()

        self.trainer.train()

        self.trainer.save_model(self.config.training.output_dir)
        self.logger.info("CPT training completed successfully")
