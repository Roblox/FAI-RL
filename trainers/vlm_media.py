"""Media helpers shared by the vision-language trainers (SFT and GRPO).

Turns the image/video cells a dataset declares in ``image_columns`` /
``video_columns`` (URLs, ``s3://`` URIs, local paths, or embedded bytes) into
sources the trainers fetch at access time, and renders ``str.format`` text
templates from ``dataset_columns``. Expects ``self.config`` and ``self.logger``.
"""

import io, math
from numbers import Real
from typing import Optional, List, Any

from utils.image_utils import fetch_image
from utils.video_utils import fetch_video


class VLMMediaMixin:
    @staticmethod
    def _is_empty_media_source(s: Any) -> bool:
        """Return whether a media cell has no usable source."""
        if s is None:
            return True
        if isinstance(s, str):
            return not s.strip()
        if isinstance(s, (bytes, bytearray)):
            return not s
        return isinstance(s, Real) and math.isnan(s)

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

        if VLMMediaMixin._is_empty_media_source(s):
            return None
        if isinstance(s, str):
            return s.strip()
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
                    return VLMMediaMixin._coerce_image_source(s[key])
            return None
        # Unknown scalar source: preserve prior behavior and stringify.
        return str(s)

    @staticmethod
    def _coerce_video_source(s: Any):
        """Normalize one video cell into an Arrow-serializable source.

        Videos are referenced, not embedded, so this is simpler than the image
        version: string sources (HTTP(S) URL, ``s3://`` URI, or local path) and
        the HF ``{'path'/'url'/'video'}`` dict pass through; everything else is
        stringified. Returns ``None`` for empty cells so callers can drop them.
        """
        if VLMMediaMixin._is_empty_media_source(s):
            return None
        if isinstance(s, str):
            return s.strip()
        if isinstance(s, dict):
            for key in ("video", "path", "url"):
                if s.get(key) is not None:
                    return VLMMediaMixin._coerce_video_source(s[key])
            return None
        return str(s)

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

    def _fetch_videos(self, sources: List[str]) -> List[Any]:
        """Fetch + frame-sample each video source into the processor's frame form."""
        data = self.config.data
        videos = []
        for s in sources:
            frames, metadata = fetch_video(
                s,
                num_frames=data.video_num_frames,
                fps=data.video_fps,
                cache_dir=data.video_cache_dir,
                timeout=data.video_fetch_timeout,
                retries=data.video_fetch_retries,
                s3_region=data.video_s3_region,
                s3_endpoint_url=data.video_s3_endpoint_url,
                backend=data.video_backend,
                return_metadata=True,
            )
            videos.append({"frames": frames, "video_metadata": metadata})
        return videos

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

    @property
    def _has_vision_columns(self) -> bool:
        """True when this run's VLM collator will receive images or videos.

        Assumption: a declared image/video column means those inputs are
        actually fed this run. If a recipe lists an image column but every cell
        is empty (so the collator effectively runs text-only), vision still
        looks "in use" here and the tower is kept trainable. We do not inspect
        per-batch contents or freeze/unfreeze mid-run to catch that case.
        """
        return any(
            getattr(ds, "image_columns", None) or getattr(ds, "video_columns", None)
            for ds in self.config.data.datasets
        )

    @property
    def _has_video_columns(self) -> bool:
        """True when any configured dataset declares video_columns."""
        return any(getattr(ds, "video_columns", None) for ds in self.config.data.datasets)
