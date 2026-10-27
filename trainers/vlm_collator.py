"""Video-aware data collator for vision-language SFT.

TRL's :class:`~trl.trainer.sft_trainer.DataCollatorForVisionLanguageModeling`
only ever passes ``images=`` to the processor -- it has no notion of video. VLM
processors (e.g. ``Qwen2_5_VLProcessor``, ``Gemma4Processor``) accept a
``videos=`` argument that yields ``pixel_values_videos`` / ``video_grid_thw``.

Rather than fork ~90 lines of TRL's batching logic (which is coupled to the
pinned ``trl==1.16.0``), this collator reuses that logic unchanged and injects
``videos=`` into the single processor tokenization call via a transparent proxy.
TRL tokenizes conversational (``messages``) rows inside
``processor.apply_chat_template(tokenize=True)``, which the proxy cannot reach,
so video batches of that kind are first rendered to text and take TRL's
standard-text path instead. When a batch carries no videos the base behavior is
used verbatim, so the image-only path is completely unaffected.
"""

from typing import Any, List, Optional, Tuple

from trl.data_utils import prepare_multimodal_messages
from trl.trainer.sft_trainer import DataCollatorForVisionLanguageModeling


class _VideoInjectingProcessor:
    """Transparent proxy that injects ``videos=`` into the processor call.

    Only ``__call__`` (the tokenization/feature-extraction step) is intercepted;
    every other attribute -- notably ``apply_chat_template`` and ``tokenizer`` --
    is delegated to the wrapped processor, so TRL's collator behaves identically
    apart from now receiving the sampled video frames.

    Videos are injected only into calls that already pass ``images`` (i.e. the
    prompt/language-modeling call). In prompt-completion mode TRL invokes the
    processor a second time for the completion with ``text`` only -- that call
    must not receive the videos, or their tokens would be counted twice.

    When ``video_metadata`` is present we also pass it through and set
    ``do_sample_frames=False``. Frames were already sampled by
    :func:`utils.video_utils.fetch_video`; resampling them against the original
    clip length would either warn (missing fps) or crash (num_frames > tensor
    length). The metadata's ``fps`` / ``frames_indices`` let Gemma stamp real
    timestamps instead of defaulting to 24fps.
    """

    def __init__(self, processor, videos, video_metadata=None):
        object.__setattr__(self, "_processor", processor)
        object.__setattr__(self, "_videos", videos)
        object.__setattr__(self, "_video_metadata", video_metadata)

    def __call__(self, *args, **kwargs):
        if self._videos is not None and "images" in kwargs and "videos" not in kwargs:
            kwargs["videos"] = self._videos
            if self._video_metadata is not None and "video_metadata" not in kwargs:
                kwargs["video_metadata"] = self._video_metadata
                kwargs.setdefault("do_sample_frames", False)
        return self._processor(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._processor, name)


class VideoAwareVLMCollator(DataCollatorForVisionLanguageModeling):
    """Vision-language collator that also feeds videos to the processor.

    Examples may carry a ``"videos"`` key alongside ``"images"``: a list (one
    entry per video in the row) of frame arrays as produced by
    :func:`utils.video_utils.fetch_video`, or ``{"frames", "video_metadata"}``
    dicts when metadata was requested. The corresponding messages are expected
    to already contain ``{"type": "video"}`` content placeholders (the trainer's
    transform emits structured content when a row has videos), so the
    processor's chat template renders the right video tokens.
    """

    @staticmethod
    def _unpack_video_item(item: Any) -> Tuple[Any, Optional[dict]]:
        if isinstance(item, dict) and "frames" in item:
            return item["frames"], item.get("video_metadata")
        return item, None

    @staticmethod
    def _extract_videos(examples: List[dict]) -> Tuple[Optional[List[Any]], Optional[List[Any]]]:
        """Gather per-example video lists, or ``None`` when the batch has none.

        Mirrors TRL's image guard: transformers requires at least one video in
        the batch or it errors, so an all-empty batch maps to ``None``. When any
        item carries metadata, a parallel nested metadata list is returned too.
        """
        videos = []
        metadata = []
        saw_metadata = False
        for example in examples:
            row_videos = []
            row_metadata = []
            for item in example.get("videos", []) or []:
                frames, meta = VideoAwareVLMCollator._unpack_video_item(item)
                row_videos.append(frames)
                row_metadata.append(meta)
                if meta is not None:
                    saw_metadata = True
            videos.append(row_videos)
            metadata.append(row_metadata)
        if all(v == [] for v in videos):
            return None, None
        return videos, (metadata if saw_metadata else None)

    def _render_conversations(self, examples: List[dict]) -> List[dict]:
        """Replace each row's ``messages`` with its chat-template text.

        TRL's conversational path calls ``apply_chat_template`` on the wrapped
        processor, which tokenizes without ``videos=``. As text rows, the batch
        goes through ``processor(images=..., text=...)``, which the proxy
        intercepts.
        """
        if self.assistant_only_loss:
            raise ValueError(
                "assistant_only_loss is not supported for video batches: assistant masks "
                "come from apply_chat_template(tokenize=True), which never receives videos."
            )
        rendered = []
        for example in examples:
            messages = prepare_multimodal_messages(example["messages"], images=example["images"])
            row = {key: value for key, value in example.items() if key != "messages"}
            row[self.dataset_text_field] = self.processor.apply_chat_template(
                messages, chat_template=self.chat_template, tokenize=False
            )
            rendered.append(row)
        return rendered

    def _collate_language_modeling(self, examples: List[dict]) -> dict:
        videos, video_metadata = self._extract_videos(examples)
        if videos is None:
            return super()._collate_language_modeling(examples)
        if "messages" in examples[0]:
            examples = self._render_conversations(examples)
        original = self.processor
        self.processor = _VideoInjectingProcessor(original, videos, video_metadata)
        try:
            return super()._collate_language_modeling(examples)
        finally:
            self.processor = original

    def _collate_prompt_completion(self, examples: List[dict]) -> dict:
        videos, video_metadata = self._extract_videos(examples)
        if videos is None:
            return super()._collate_prompt_completion(examples)
        original = self.processor
        self.processor = _VideoInjectingProcessor(original, videos, video_metadata)
        try:
            return super()._collate_prompt_completion(examples)
        finally:
            self.processor = original
