import io, os, sys
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

    Loads a VLM via AutoModelForImageTextToText + AutoProcessor, gathers the
    images referenced across each dataset's `image_columns` (HTTP(S) URLs, s3://
    URIs, local paths, or raw bytes -- one or many per row) into PIL images,
    builds the text for each row from its `dataset_columns` via the
    `system_prompt` template, and delegates to TRL's SFTTrainer. Because the
    processing class is a `ProcessorMixin`, TRL treats this as a VLM run: it skips
    its own tokenization/preprocessing and auto-selects
    `DataCollatorForVisionLanguageModeling`, which reads each row's messages +
    `images` and builds `pixel_values`/`labels` on the fly.

    Two data shapes are emitted depending on config. Flat mode (default) yields a
    single-turn `messages` row and computes loss over the whole sequence. Split
    (chat) mode -- enabled when `data.user_prompt` and `data.assistant_prompt` are
    set -- yields a conversational `prompt`/`completion` pair so the collator masks
    the prompt and computes loss only on the assistant completion
    (completion_only_loss).

    Dataset rows are produced lazily via `Dataset.with_transform`, so images are
    decoded to PIL only at access time and the heterogeneous multimodal message
    structure is never serialized to Arrow.
    """

    # Load the base model as an image-text-to-text model instead of a causal LM.
    _auto_model_class = AutoModelForImageTextToText

    def __init__(self, config: ExperimentConfig, logger: Optional[object] = None):
        super().__init__(config, logger=logger)
        self.trainer = None
        self.model = None
        self.processor = None
        self.train_dataset = None
        # Split (chat) mode: emit conversational prompt/completion rows and mask
        # the prompt (completion_only_loss). Legacy flat mode emits a single
        # `messages` turn with loss over the whole sequence.
        self._split_mode = config.data.split_mode

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
        frozen_tower_attr = None
        if getattr(self.config.model, "freeze_vision_tower", True):
            frozen_tower_attr = self._freeze_vision_tower(self.model)

        # Keep LoRA out of the (frozen) vision tower. target_modules like q_proj
        # match by name across the whole model, so without this PEFT tries to
        # adapt the vision tower's attention -- whose projections are custom
        # module types (e.g. Gemma4's Gemma4ClippableLinear) that PEFT can't wrap,
        # raising "Target module ... is not supported". A regex string exclusion
        # is required because a list only suffix-matches leaf names, not a subtree.
        if (
            getattr(self.config.model, "use_lora", False)
            and frozen_tower_attr is not None
            and not getattr(self.config.model, "lora_exclude_modules", None)
        ):
            self.config.model.lora_exclude_modules = rf"(.*\.)?{frozen_tower_attr}\..*"
            self.logger.info(
                "Excluding frozen vision tower from LoRA injection: "
                f"lora_exclude_modules={self.config.model.lora_exclude_modules!r}"
            )

        # Apply LoRA/QLoRA if enabled (reuses BaseTrainer helper).
        self.model = self.apply_lora_to_model(self.model, TaskType.CAUSAL_LM, quantization_config)

        self.disable_cache_for_gradient_checkpointing(self.model)

        self.logger.info("VLM and processor loaded successfully")

    def _freeze_vision_tower(self, model):
        """Freeze the vision encoder submodule if one can be located.

        Returns the attribute name of the frozen tower (e.g. "vision_tower"), or
        None if no known vision submodule was found. Callers use the name to keep
        LoRA from being injected into the frozen tower.
        """
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
                    return attr
        self.logger.warning(
            "freeze_vision_tower=true but no known vision submodule "
            f"({', '.join(_VISION_TOWER_ATTRS)}) was found; nothing frozen."
        )
        return None

    # ------------------------------ data ------------------------------------

    @staticmethod
    def _coerce_image_source(s: Any):
        """Normalize one image cell into an Arrow-serializable source.

        String sources (HTTP(S) URL, local path, or ``s3://`` URI) pass through
        unchanged. Embedded images -- raw ``bytes``, a ``PIL.Image``, or the HF
        ``{'bytes'/'path'/'url'/'image'}`` dict that the datasets ``Image``
        feature decodes to -- are reduced to something :func:`fetch_image` can
        later decode: raw PNG bytes for pixel data, or the referenced path/URL
        string. This lets parquet/HF datasets that embed image bytes (e.g. the
        ``images: List[Image]`` schema) train without an external fetch step.

        Returns ``None`` for empty cells so callers can drop them.
        """
        from PIL import Image

        if s is None:
            return None
        if isinstance(s, str):
            return s
        if isinstance(s, (bytes, bytearray)):
            return bytes(s)
        if isinstance(s, Image.Image):
            buf = io.BytesIO()
            s.save(buf, format="PNG")
            return buf.getvalue()
        if isinstance(s, dict):
            if s.get("bytes") is not None:
                return bytes(s["bytes"])
            for key in ("path", "url", "image"):
                if s.get(key) is not None:
                    return SFTVLMTrainer._coerce_image_source(s[key])
            return None
        # Unknown scalar source: preserve prior behavior and stringify.
        return str(s)

    def _normalize_dataset(self, dataset, dataset_info):
        """Map a raw dataset to a uniform, Arrow-friendly schema.

        Output columns (simple types so multiple datasets can be concatenated
        and serialized safely):
          _image_sources : list[str]  -- image sources gathered, in order, from
                                          every image_columns column for the row
          _text          : str        -- (flat mode) the full text built from
                                          dataset_columns via the system_prompt
                                          template, or concatenation when unset
          _user_text     : str        -- (split mode) rendered user turn
          _assistant_text: str        -- (split mode) rendered assistant turn
        """
        image_columns = dataset_info.image_columns or []
        dataset_columns = dataset_info.dataset_columns or []
        system_prompt = self.config.data.system_prompt
        user_prompt = self.config.data.user_prompt
        assistant_prompt = self.config.data.assistant_prompt

        def fn(example):
            # Gather image sources across every configured image column. Each
            # column may hold a single source or a list of them.
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

            if self._split_mode:
                return {
                    "_image_sources": sources,
                    # Empty when no system_prompt is configured; the transform then
                    # omits the system turn entirely.
                    "_system_text": self._build_text(system_prompt, dataset_columns, example) if system_prompt else "",
                    "_user_text": self._build_text(user_prompt, dataset_columns, example),
                    "_assistant_text": self._build_text(assistant_prompt, dataset_columns, example),
                }
            return {
                "_image_sources": sources,
                "_text": self._build_text(system_prompt, dataset_columns, example),
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

    def _build_text(self, template: Optional[str], dataset_columns: List[str], example) -> str:
        """Render one text block for a row from a template + its text columns.

        ``template`` is a ``str.format`` template keyed by the dataset's column
        names -- a column name is just a variable, so ``dataset_columns:
        [question, response]`` renders ``"...{question}...{response}"`` (double any
        literal braces as ``{{`` / ``}}``). When ``template`` is falsy, the column
        values are concatenated in order. Used for the flat ``system_prompt``
        template as well as the split-mode ``user_prompt`` / ``assistant_prompt``
        / system turns. Falls back to the literal template if it references an
        unknown key or is malformed.
        """
        fmt = {col: example.get(col, "") for col in dataset_columns}
        if template:
            try:
                return template.format(**fmt)
            except (KeyError, IndexError, ValueError) as e:
                if not getattr(self, "_text_warn_emitted", False):
                    self.logger.warning(
                        f"system_prompt template not rendered ({e}); using literal "
                        f"text. Available placeholders: {sorted(fmt)}. Double any "
                        f"literal braces as {{{{ }}}}."
                    )
                    self._text_warn_emitted = True
                return template
        return "\n".join(str(fmt[col]) for col in dataset_columns)

    def _make_transform(self):
        """Build the lazy with_transform callable (batched columnar input)."""
        if self._split_mode:
            return self._make_split_transform()

        def transform(batch):
            out_messages, out_images = [], []
            for text, sources in zip(batch["_text"], batch["_image_sources"]):
                images = self._fetch_images(sources)
                # A single turn carrying the full templated text. Raw-string
                # content lets TRL's vision collator insert one image placeholder
                # per image into the message. Labels cover the whole sequence (no
                # prompt/completion masking), mirroring the text SFT trainer.
                messages = [{"role": "user", "content": text}]
                out_messages.append(messages)
                out_images.append(images)
            return {"messages": out_messages, "images": out_images}

        return transform

    def _make_split_transform(self):
        """Lazy transform for split (chat) mode: conversational prompt/completion.

        Emits `prompt` (optional system turn + user turn) and `completion` (the
        assistant turn) so TRL's vision collator takes its prompt/completion path
        and masks the prompt (completion_only_loss). Images attach to the user turn
        -- the collator inserts one image placeholder per image via
        `prepare_multimodal_messages`.
        """
        def transform(batch):
            out_prompt, out_completion, out_images = [], [], []
            for sys_text, user_text, assistant_text, sources in zip(
                batch["_system_text"], batch["_user_text"],
                batch["_assistant_text"], batch["_image_sources"],
            ):
                images = self._fetch_images(sources)
                prompt = []
                if sys_text:
                    prompt.append({"role": "system", "content": sys_text})
                prompt.append({"role": "user", "content": user_text})
                completion = [{"role": "assistant", "content": assistant_text}]
                out_prompt.append(prompt)
                out_completion.append(completion)
                out_images.append(images)
            return {"prompt": out_prompt, "completion": out_completion, "images": out_images}

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
                if self._split_mode:
                    if not example["_user_text"].strip() or not example["_assistant_text"].strip():
                        return False
                elif not example["_text"].strip():
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
                    f"(no images, empty text, or unfetchable images)"
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
            # Split mode emits prompt/completion rows; mask the prompt so the
            # vision collator computes loss only on the assistant completion. Flat
            # mode leaves this False (loss over the whole sequence).
            completion_only_loss=True if self._split_mode else None,
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
