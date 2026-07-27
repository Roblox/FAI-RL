"""Video-aware data collator for vision-language SFT.

TRL's :class:`~trl.trainer.sft_trainer.DataCollatorForVisionLanguageModeling`
only ever passes ``images=`` to the processor -- it has no notion of video. VLM
processors (e.g. ``Qwen2_5_VLProcessor``, ``Gemma4Processor``) accept a
``videos=`` argument that yields ``pixel_values_videos`` / ``video_grid_thw``.

Rather than fork ~90 lines of TRL's batching logic (which is coupled to the
pinned ``trl==1.2.0``), this collator reuses that logic unchanged and injects
``videos=`` into the single processor tokenization call via a transparent proxy.
When a batch carries no videos the base behavior is used verbatim, so the
image-only path is completely unaffected.
"""

from typing import Any, List, Optional

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
    """

    def __init__(self, processor, videos):
        object.__setattr__(self, "_processor", processor)
        object.__setattr__(self, "_videos", videos)

    def __call__(self, *args, **kwargs):
        if self._videos is not None and "images" in kwargs and "videos" not in kwargs:
            kwargs["videos"] = self._videos
        return self._processor(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._processor, name)


class VideoAwareVLMCollator(DataCollatorForVisionLanguageModeling):
    """Vision-language collator that also feeds videos to the processor.

    Examples may carry a ``"videos"`` key alongside ``"images"``: a list (one
    entry per video in the row) of frame arrays as produced by
    :func:`utils.video_utils.fetch_video`. The corresponding messages are
    expected to already contain ``{"type": "video"}`` content placeholders (the
    trainer's transform emits structured content when a row has videos), so the
    processor's chat template renders the right video tokens.
    """

    @staticmethod
    def _extract_videos(examples: List[dict]) -> Optional[List[Any]]:
        """Gather per-example video lists, or ``None`` when the batch has none.

        Mirrors TRL's image guard: transformers requires at least one video in
        the batch or it errors, so an all-empty batch maps to ``None``.
        """
        videos = [example.get("videos", []) or [] for example in examples]
        if all(v == [] for v in videos):
            return None
        return videos

    def _collate_language_modeling(self, examples: List[dict]) -> dict:
        videos = self._extract_videos(examples)
        if videos is None:
            return super()._collate_language_modeling(examples)
        original = self.processor
        self.processor = _VideoInjectingProcessor(original, videos)
        try:
            return super()._collate_language_modeling(examples)
        finally:
            self.processor = original

    def _collate_prompt_completion(self, examples: List[dict]) -> dict:
        videos = self._extract_videos(examples)
        if videos is None:
            return super()._collate_prompt_completion(examples)
        original = self.processor
        self.processor = _VideoInjectingProcessor(original, videos)
        try:
            return super()._collate_prompt_completion(examples)
        finally:
            self.processor = original
