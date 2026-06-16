import os, sys
import json
from typing import Optional, List, Any

from datasets import concatenate_datasets
from trl import SFTConfig, SFTTrainer as TRLSFTTrainer
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import TaskType

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.config import ExperimentConfig
from core.trainer_base import BaseTrainer
from utils.dataset_utils import load_training_dataset
from utils.image_utils import fetch_image


# Submodule attribute names commonly used for the vision encoder across VLM
# architectures (Qwen-VL exposes it as `.visual`; LLaVA/others as
# `.vision_tower`/`.vision_model`). We freeze whichever is present.
_VISION_TOWER_ATTRS = ("visual", "vision_tower", "vision_model")


class SFTVLMTrainer(BaseTrainer):
    """Multimodal SFT trainer for vision-language models (image + text).

    Loads a VLM via AutoModelForImageTextToText + AutoProcessor, fetches images
    referenced by HTTP(S) URLs (or local paths) into PIL images, and delegates
    to TRL's SFTTrainer. Because the processing class is a `ProcessorMixin`, TRL
    treats this as a VLM run: it skips its own tokenization/preprocessing and
    auto-selects `DataCollatorForVisionLanguageModeling`, which reads each row's
    `messages` + `images` and builds `pixel_values`/`labels` on the fly.

    Dataset rows are produced lazily via `Dataset.with_transform`, so images are
    decoded to PIL only at access time and the heterogeneous multimodal
    `messages` structure is never serialized to Arrow.
    """

    # Load the base model as an image-text-to-text model instead of a causal LM.
    _auto_model_class = AutoModelForImageTextToText

    def __init__(self, config: ExperimentConfig, logger: Optional[object] = None):
        super().__init__(config, logger=logger)
        self.trainer = None
        self.model = None
        self.processor = None
        self.train_dataset = None

    # ----------------------------- model -----------------------------------

    def setup_model(self):
        """Load the VLM and its processor."""
        self.logger.info(f"Loading vision-language model: {self.config.model.base_model_name}")

        quantization_config = self.create_quantization_config()
        model_kwargs = self.prepare_model_kwargs(quantization_config)

        # Reuses BaseTrainer loading + PEFT-resume logic; _auto_model_class makes
        # it load a VLM rather than a causal LM.
        self.model = self.load_base_model_for_training(model_kwargs)

        # Load the processor (tokenizer + image processor). Unlike the text
        # trainers we do NOT add a [PAD] token or resize embeddings -- doing so
        # corrupts a VLM's vision/token embedding alignment. VLM tokenizers
        # already define a pad token; fall back to eos only if missing.
        self.processor = AutoProcessor.from_pretrained(self.config.model.base_model_name)
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None):
            tokenizer.pad_token = tokenizer.eos_token

        # Optionally freeze the vision encoder (standard for VLM SFT; also avoids
        # training the ViT under full fine-tuning).
        if getattr(self.config.model, "freeze_vision_tower", True):
            self._freeze_vision_tower(self.model)

        # Apply LoRA/QLoRA if enabled (reuses BaseTrainer helper).
        self.model = self.apply_lora_to_model(self.model, TaskType.CAUSAL_LM, quantization_config)

        self.disable_cache_for_gradient_checkpointing(self.model)

        self.logger.info("VLM and processor loaded successfully")

    def _freeze_vision_tower(self, model):
        """Freeze the vision encoder submodule if one can be located."""
        # PEFT may wrap the model; reach through to the underlying module.
        base = getattr(model, "base_model", model)
        base = getattr(base, "model", base)
        for root in (model, base):
            for attr in _VISION_TOWER_ATTRS:
                tower = getattr(root, attr, None)
                if tower is not None and hasattr(tower, "parameters"):
                    for p in tower.parameters():
                        p.requires_grad_(False)
                    self.logger.info(f"Froze vision tower: {attr}")
                    return
        self.logger.warning(
            "freeze_vision_tower=true but no known vision submodule "
            f"({', '.join(_VISION_TOWER_ATTRS)}) was found; nothing frozen."
        )

    # ------------------------------ data ------------------------------------

    def _normalize_dataset(self, dataset, dataset_info):
        """Map a raw dataset to a uniform, Arrow-friendly schema.

        Output columns (all simple types so multiple datasets can be
        concatenated and serialized safely):
          _image_sources : list[str]  -- image URLs / local paths for the row
          _question      : str        -- user prompt   (image_column path)
          _response      : str        -- target answer (image_column path)
          _messages_json : str        -- JSON of pre-built conversational
                                          messages, or "" when not provided
        """
        image_column = dataset_info.image_column
        messages_column = dataset_info.messages_column
        q_col = dataset_info.question_column
        r_col = dataset_info.response_column

        def fn(example):
            raw = example.get(image_column) if image_column else None
            if raw is None:
                sources = []
            elif isinstance(raw, (list, tuple)):
                sources = [str(s) for s in raw if s is not None]
            else:
                sources = [str(raw)]

            messages_json = ""
            if messages_column and example.get(messages_column) is not None:
                messages_json = json.dumps(example[messages_column])

            return {
                "_image_sources": sources,
                "_question": str(example.get(q_col, "")),
                "_response": str(example.get(r_col, "")),
                "_messages_json": messages_json,
            }

        return dataset.map(fn, remove_columns=dataset.column_names)

    def _fetch_images(self, sources: List[str]) -> List[Any]:
        cache_dir = self.config.data.image_cache_dir
        timeout = self.config.data.image_fetch_timeout
        retries = self.config.data.image_fetch_retries
        max_pixels = self.config.data.max_image_pixels
        s3_region = self.config.data.image_s3_region
        s3_endpoint_url = self.config.data.image_s3_endpoint_url
        images = []
        for s in sources:
            img = fetch_image(
                s,
                cache_dir=cache_dir,
                timeout=timeout,
                retries=retries,
                s3_region=s3_region,
                s3_endpoint_url=s3_endpoint_url,
            )
            images.append(self._maybe_downscale(img, max_pixels))
        return images

    @staticmethod
    def _maybe_downscale(img, max_pixels: Optional[int]):
        """Downscale an image (preserving aspect ratio) if it exceeds max_pixels."""
        if not max_pixels:
            return img
        w, h = img.size
        if w * h <= max_pixels:
            return img
        scale = (max_pixels / float(w * h)) ** 0.5
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        return img.resize(new_size)

    def _render_system_prompt(self, template: str, question: str, response: str) -> str:
        """Fill the system_prompt template with the row's question/response values.

        Mirrors the text SFT trainer, whose ``system_prompt`` is a ``str.format``
        template keyed by the dataset's column names. The placeholder names here
        are the recipe's configured ``question_column`` / ``response_column`` -- a
        column name is just a variable, so renaming the column renames the
        placeholder (e.g. ``question_column: "query"`` -> use ``{query}``). Falls
        back to the literal text if the template references an unknown key or is
        malformed; double any literal braces as ``{{`` / ``}}``.

        Note: ``response`` is also the assistant (target) turn, so referencing it
        here duplicates that turn and has no value to fill at inference -- it suits
        only labeling/eval-style data, not generative VLM SFT.
        """
        if not template:
            return template
        fmt = {key: question for key in self._sysprompt_question_keys}
        fmt.update({key: response for key in self._sysprompt_response_keys})
        try:
            return template.format(**fmt)
        except (KeyError, IndexError, ValueError) as e:
            if not getattr(self, "_system_prompt_warn_emitted", False):
                available = sorted(self._sysprompt_question_keys | self._sysprompt_response_keys)
                self.logger.warning(
                    f"system_prompt template not rendered ({e}); using literal text. "
                    f"Available placeholders: {available}. Double any literal braces as {{{{ }}}}."
                )
                self._system_prompt_warn_emitted = True
            return template

    def _make_transform(self):
        """Build the lazy with_transform callable (batched columnar input)."""
        system_prompt = self.config.data.system_prompt
        # Placeholder names come from the recipe's column config (a column name is
        # just a variable); the union covers all datasets being concatenated.
        self._sysprompt_question_keys = {di.question_column for di in self.config.data.datasets}
        self._sysprompt_response_keys = {di.response_column for di in self.config.data.datasets}

        def transform(batch):
            out_messages, out_images = [], []
            n = len(batch["_image_sources"])
            for i in range(n):
                images = self._fetch_images(batch["_image_sources"][i])

                messages_json = batch["_messages_json"][i]
                if messages_json:
                    messages = json.loads(messages_json)
                else:
                    # Raw-string content; TRL's collator inserts the right number
                    # of image placeholders into the first user message.
                    messages = []
                    if system_prompt:
                        content = self._render_system_prompt(
                            system_prompt, batch["_question"][i], batch["_response"][i]
                        )
                        messages.append({"role": "system", "content": content})
                    messages.append({"role": "user", "content": batch["_question"][i]})
                    messages.append({"role": "assistant", "content": batch["_response"][i]})

                out_messages.append(messages)
                out_images.append(images)
            return {"messages": out_messages, "images": out_images}

        return transform

    def setup_data(self):
        """Load datasets, validate images, and attach the lazy VLM transform."""
        normalized = []
        total_skipped = 0

        for dataset_info in self.config.data.datasets:
            subset_info = f" (subset: {dataset_info.subset})" if dataset_info.subset else ""
            self.logger.info(
                f"Loading dataset: {dataset_info.name}{subset_info} (split: {dataset_info.split})"
            )
            raw = load_training_dataset(dataset_info)
            original_size = len(raw)

            ds = self._normalize_dataset(raw, dataset_info)

            # Validation pass: drop rows whose images can't be fetched/decoded.
            # When image_cache_dir is set this also warms the cache so the
            # training-time transform reads from disk instead of re-downloading.
            def is_valid(example):
                if not example["_image_sources"]:
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
                    f"(missing image column or unfetchable images)"
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

        # Lazy: images are fetched/decoded to PIL only when rows are accessed.
        self.train_dataset = combined.with_transform(self._make_transform())

    # ---------------------------- training ----------------------------------

    def setup_training_args(self) -> SFTConfig:
        """Create the SFT training configuration for VLM training."""
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
            # Keep the dataset columns (messages/images) so the vision collator
            # receives them; HF Trainer would otherwise strip unknown columns.
            remove_unused_columns=False,
            deepspeed=self.config.training.deepspeed_config,
            dataloader_num_workers=self.config.training.dataloader_num_workers,
            gradient_checkpointing=self.config.training.gradient_checkpointing,
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
            dataloader_pin_memory=self.config.training.dataloader_pin_memory,
            dataloader_drop_last=self.config.training.dataloader_drop_last,
            report_to=report_to,
            ddp_find_unused_parameters=self.config.training.ddp_find_unused_parameters,
            # CRITICAL for VLMs: do not truncate, or image placeholder tokens may
            # be cut, which crashes the processor / model.
            max_length=None,
            dataset_num_proc=self.config.data.dataset_num_proc,
        )

    def setup_trainer(self):
        """Initialize TRL's SFTTrainer with the processor (auto vision collator)."""
        training_args = self.setup_training_args()

        self.trainer = TRLSFTTrainer(
            model=self.model,
            args=training_args,
            processing_class=self.processor,
            train_dataset=self.train_dataset,
            callbacks=self.build_callbacks(),
        )

        self.logger.info("SFT VLM trainer initialized")

    def train(self):
        """Run the multimodal SFT training process."""
        self.logger.info("Starting multimodal (VLM) SFT training...")

        self.setup_model()
        self.setup_data()
        self.setup_trainer()

        self.trainer.train()

        self.trainer.save_model(self.config.training.output_dir)
        # Persist the processor alongside the model so the checkpoint is loadable.
        try:
            self.processor.save_pretrained(self.config.training.output_dir)
        except Exception as e:
            self.logger.warning(f"Failed to save processor: {e}")
        self.logger.info("Multimodal SFT training completed successfully")
