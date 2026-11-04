import os, sys
from typing import Any, Callable, List, Optional

from datasets import concatenate_datasets
from trl import GRPOTrainer as TRLGRPOTrainer
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import TaskType

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.config import ExperimentConfig
from utils.dataset_utils import load_raw_dataset
from trainers.grpo_trainer import GRPOTrainer
from trainers.vlm_media import VLMMediaMixin
from .rewards.factory import build_reward_function


def _text_of(content: Any) -> str:
    """Text of a chat message ``content``: a string, or the text parts of a block list."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def _text_prompt(prompt: Any) -> Any:
    """A conversational prompt with image blocks removed; strings pass through."""
    if isinstance(prompt, list):
        return [
            {"role": message.get("role"), "content": _text_of(message.get("content"))}
            for message in prompt
            if isinstance(message, dict)
        ]
    return prompt


def _text_completion(completion: Any) -> Any:
    """The assistant text of a conversational completion; strings pass through."""
    if isinstance(completion, list):
        return "".join(
            _text_of(message.get("content"))
            for message in completion
            if isinstance(message, dict) and message.get("role") == "assistant"
        )
    return completion


class TextRewardAdapter:
    """Give a reward function text prompts and completions, without the images.

    With images TRL keeps prompts conversational, fills the user turn's image
    blocks with PIL images, and returns completions as assistant messages. The
    reward contract is text: completions become their assistant text, prompts
    keep their roles with text-only content, and the ``images``/``image``
    columns are not forwarded (they are model inputs, not scoring context).
    """

    def __init__(self, reward_func: Callable):
        self.reward_func = reward_func
        self.__name__ = getattr(reward_func, "__name__", "reward")

    def __call__(self, prompts, completions, **kwargs) -> List[float]:
        kwargs.pop("images", None)
        kwargs.pop("image", None)
        return self.reward_func(
            prompts=[_text_prompt(prompt) for prompt in prompts],
            completions=[_text_completion(completion) for completion in completions],
            **kwargs,
        )


class GRPOVLMTrainer(VLMMediaMixin, GRPOTrainer):
    """GRPO for vision-language models (images + text prompt, text completion).

    Each row becomes a conversational ``prompt`` (an optional ``system_prompt``
    turn, then the user turn from ``prompt_column``) plus an ``images`` list
    gathered across the dataset's ``image_columns`` (URLs, ``s3://`` URIs, local
    paths, or embedded bytes). TRL inserts the images into the user turn,
    generates completions, and scores them with the configured reward. Every
    other dataset column stays in the row as reward context; the raw image
    columns do not. Rows whose images cannot be fetched are dropped up front.
    """

    _auto_model_class = AutoModelForImageTextToText

    def __init__(self, config: ExperimentConfig, logger: Optional[object] = None):
        super().__init__(config, logger=logger)
        self.processor = None

    # ----------------------------- model -----------------------------------

    def setup_model(self):
        """Load the VLM and its processor, then apply LoRA to the language model."""
        self.logger.info(f"Loading vision-language model: {self.config.model.base_model_name}")

        quantization_config = self.create_quantization_config()
        model_kwargs = self.prepare_model_kwargs(quantization_config)
        self.model = self.load_base_model_for_training(model_kwargs)

        # Like SFTVLMTrainer: never add a pad token or resize a VLM's embeddings,
        # and load the processor from the PEFT base when resuming from an adapter.
        processor_name = (
            getattr(self, "_peft_base_model_path", None) or self.config.model.base_model_name
        )
        self.processor = AutoProcessor.from_pretrained(processor_name)
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None):
            tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer

        self.prepare_model_for_modalities(
            self.model,
            use_vision=self._has_vision_columns,
            use_audio=False,
            freeze_vision=getattr(self.config.model, "freeze_vision_tower", True),
        )
        self.model = self.apply_lora_to_model(self.model, TaskType.CAUSAL_LM, quantization_config)
        self.disable_cache_for_gradient_checkpointing(self.model)

        self.logger.info("VLM and processor loaded successfully")

    # ------------------------------ data ------------------------------------

    def _normalize_dataset(self, dataset, dataset_info):
        """Collect each row's image sources and prompt text; drop the media columns."""
        image_columns = dataset_info.image_columns or []
        prompt_column = getattr(dataset_info, "prompt_column", "prompt")
        dataset_columns = dataset_info.dataset_columns or [
            col for col in dataset.column_names if col not in image_columns
        ]
        system_prompt = self.config.data.system_prompt
        answer_col = getattr(dataset_info, "answer_column", None)

        def fn(example):
            sources = []
            for col in image_columns:
                raw = example.get(col)
                if raw is None:
                    continue
                items = raw if isinstance(raw, (list, tuple)) else [raw]
                for s in items:
                    coerced = self._coerce_image_source(s)
                    if coerced is not None:
                        sources.append(coerced)
            user_text = example.get(prompt_column)
            mapped = {"answer": example[answer_col]} if answer_col and answer_col in example else {}
            return {
                **mapped,
                "_image_sources": sources,
                "_system_text": (
                    self._build_text(system_prompt, dataset_columns, example)
                    if system_prompt else ""
                ),
                "_user_text": user_text if isinstance(user_text, str) else "",
            }

        return dataset.map(
            fn, remove_columns=[col for col in image_columns if col in dataset.column_names]
        )

    def _make_transform(self):
        """Lazy transform: conversational prompt + fetched images, other columns kept."""
        internal = ("_image_sources", "_system_text", "_user_text")

        def transform(batch):
            out = {key: values for key, values in batch.items() if key not in internal}
            prompts, images = [], []
            for sys_text, user_text, sources in zip(
                batch["_system_text"], batch["_user_text"], batch["_image_sources"]
            ):
                prompt = [{"role": "system", "content": sys_text}] if sys_text else []
                prompt.append({"role": "user", "content": user_text})
                prompts.append(prompt)
                images.append(self._fetch_images(sources))
            out["prompt"] = prompts
            out["images"] = images
            return out

        return transform

    def setup_data(self):
        """Load datasets, drop rows with unusable images, and attach the lazy transform."""
        if self._has_video_columns:
            raise ValueError("grpo_vlm supports image_columns only; remove video_columns")

        normalized = []
        total_skipped = 0
        for dataset_info in self.config.data.datasets:
            subset_info = f" (subset: {dataset_info.subset})" if dataset_info.subset else ""
            self.logger.info(
                f"Loading dataset: {dataset_info.name}{subset_info} (split: {dataset_info.split})"
            )
            raw = load_raw_dataset(dataset_info)
            original_size = len(raw)
            ds = self._normalize_dataset(raw, dataset_info)

            def is_valid(example):
                if not example["_image_sources"] or not example["_user_text"].strip():
                    return False
                try:
                    self._fetch_images(example["_image_sources"])
                    return True
                except Exception as e:
                    self.logger.warning(f"Dropping row (image fetch failed): {e}")
                    return False

            ds = ds.filter(is_valid)
            skipped = original_size - len(ds)
            total_skipped += skipped
            if skipped > 0:
                self.logger.warning(
                    f"Skipped {skipped} examples from {dataset_info.name} "
                    f"(no images, empty prompt, or unfetchable images)"
                )
            normalized.append(ds)
            self.logger.info(f"Loaded {len(ds)} valid examples from {dataset_info.name}")

        combined = normalized[0] if len(normalized) == 1 else concatenate_datasets(normalized)
        if total_skipped > 0:
            self.logger.warning(f"Total examples skipped across all datasets: {total_skipped}")
        self.logger.info(
            f"Total dataset prepared with {len(combined)} valid examples "
            f"from {len(normalized)} datasets"
        )

        # Lazy: images are fetched and decoded only when TRL reads a row.
        self.train_dataset = combined.with_transform(self._make_transform())

    # ---------------------------- training ----------------------------------

    def setup_training_args(self):
        """GRPO arguments, keeping every dataset column so the reward sees its context."""
        args = super().setup_training_args()
        # The rows carry images and reward context next to the prompt; the HF
        # Trainer would otherwise strip every non-signature column.
        args.remove_unused_columns = False
        return args

    def setup_trainer(self):
        """Initialize TRL's GRPOTrainer with the processor and a text-only reward."""
        training_args = self.setup_training_args()
        reward_func = TextRewardAdapter(build_reward_function(self.config, logger=self.logger))

        self.trainer = TRLGRPOTrainer(
            model=self.model,
            reward_funcs=[reward_func],
            args=training_args,
            train_dataset=self.train_dataset,
            processing_class=self.processor,
            callbacks=self.build_callbacks(),
        )
        self.logger.info("GRPO VLM trainer initialized")

    def train(self):
        """Run image GRPO training."""
        self.logger.info("Starting multimodal (VLM) GRPO training...")

        self.setup_model()
        self.setup_data()
        self.setup_trainer()

        self.train_with_resume()

        self.trainer.save_model(self.config.training.output_dir)
        try:
            self.processor.save_pretrained(self.config.training.output_dir)
        except Exception as e:
            self.logger.warning(f"Failed to save processor: {e}")
        self.logger.info("Multimodal GRPO training completed successfully")
