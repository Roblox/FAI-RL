"""Video loading utilities for multimodal (vision-language) training.

The headline use case is fetching a video from an HTTP(S) URL carried in a
dataset column, sampling a bounded number of frames, and returning them in the
array form a VLM processor consumes as ``videos=[...]``. Local file paths and
``s3://`` URIs are also accepted so the same helper works regardless of how a
dataset stores its videos.

This mirrors :mod:`utils.image_utils`: it reuses that module's URL fetch + md5
on-disk cache and the S3 download helper, but caches the *raw video file*
(decoding to frames is deferred to :func:`transformers.video_utils.load_video`,
which needs a file path, not bytes).
"""

import os
import tempfile
from typing import Any, Optional

from utils.image_utils import _cache_path, _fetch_url


def as_video_metadata_dict(metadata):
    """Convert transformers video metadata into a plain dict the processor accepts.

    Gemma (and Qwen3-VL) need ``fps`` and ``frames_indices`` to stamp timestamps
    into the prompt. ``load_video`` already returns this; we keep it instead of
    dropping it and forcing the processor to guess ``fps=24``.
    """
    if metadata is None:
        return None
    if isinstance(metadata, dict):
        data = dict(metadata)
    else:
        try:
            data = {key: metadata[key] for key in metadata}
        except Exception:
            data = {
                "total_num_frames": getattr(metadata, "total_num_frames", None),
                "fps": getattr(metadata, "fps", None),
                "width": getattr(metadata, "width", None),
                "height": getattr(metadata, "height", None),
                "duration": getattr(metadata, "duration", None),
                "frames_indices": getattr(metadata, "frames_indices", None),
            }
    indices = data.get("frames_indices")
    if indices is not None:
        data["frames_indices"] = [int(i) for i in list(indices)]
    fps = data.get("fps")
    if fps is not None:
        data["fps"] = float(fps)
    total = data.get("total_num_frames")
    if total is not None:
        data["total_num_frames"] = int(total)
    return data


def _unpack_decode(decoded):
    """Normalize ``_decode`` output to ``(frames, metadata_dict_or_none)``."""
    if isinstance(decoded, tuple) and len(decoded) == 2:
        frames, metadata = decoded
        return frames, as_video_metadata_dict(metadata)
    return decoded, None


def _sample_kwargs(num_frames: Optional[int], fps: Optional[float]) -> dict:
    """Build the frame-sampling kwargs for ``load_video``.

    ``num_frames`` and ``fps`` are mutually exclusive in transformers'
    ``default_sample_indices_fn`` -- ``fps`` takes priority when both are set,
    and passing neither loads every frame (which can explode sequence length).
    """
    if fps is not None:
        return {"fps": fps}
    if num_frames is not None:
        return {"num_frames": num_frames}
    return {}


def _decode(path: str, *, num_frames: Optional[int], fps: Optional[float], backend: str):
    """Decode + sample frames from a local video file path."""
    from transformers.video_utils import load_video

    frames, metadata = load_video(path, backend=backend, **_sample_kwargs(num_frames, fps))
    return frames, as_video_metadata_dict(metadata)


def fetch_video(
    src: Any,
    *,
    num_frames: Optional[int] = None,
    fps: Optional[float] = None,
    cache_dir: Optional[str] = None,
    timeout: int = 10,
    retries: int = 3,
    s3_region: Optional[str] = None,
    s3_endpoint_url: Optional[str] = None,
    backend: str = "pyav",
    return_metadata: bool = False,
):
    """Resolve a video source into sampled frames the processor accepts.

    Accepts, in order of preference:
      * an http(s) URL string  -> downloaded (optionally disk-cached) then decoded
      * an ``s3://`` URI string -> downloaded from S3 (optionally disk-cached)
      * a local file path string -> decoded from disk
      * a dict with a ``"url"``/``"path"``/``"video"`` key -> dispatched on the key

    Frames are sampled with ``fps`` if set, else ``num_frames`` uniformly. By
    default this returns the sampled frame array (``[T, H, W, C]``) for
    ``videos=[...]``. With ``return_metadata=True`` it returns
    ``(frames, metadata_dict)`` so Gemma/Qwen can stamp real timestamps instead
    of guessing ``fps=24``.

    ``s3_region`` / ``s3_endpoint_url`` are only consulted for ``s3://`` sources;
    when unset, boto3 uses its default credential/region resolution chain.

    Raises:
        RuntimeError / OSError / TypeError: if the video cannot be fetched or
        decoded. Callers that want to skip bad rows should catch the exception.
    """
    def _result(decoded):
        frames, metadata = _unpack_decode(decoded)
        if return_metadata:
            return frames, metadata
        return frames

    # HF-style dict wrappers, e.g. {"url": ...} / {"path": ...}.
    if isinstance(src, dict):
        for key in ("video", "path", "url"):
            if src.get(key) is not None:
                return fetch_video(
                    src[key],
                    num_frames=num_frames,
                    fps=fps,
                    cache_dir=cache_dir,
                    timeout=timeout,
                    retries=retries,
                    s3_region=s3_region,
                    s3_endpoint_url=s3_endpoint_url,
                    backend=backend,
                    return_metadata=return_metadata,
                )
        raise ValueError(f"Unsupported video dict with keys {list(src.keys())}")

    if not isinstance(src, str):
        raise TypeError(f"Unsupported video source type: {type(src)!r}")

    is_http = src.startswith("http://") or src.startswith("https://")
    is_s3 = src.startswith("s3://")

    if is_http or is_s3:
        # Both remote schemes share the same disk-cache + decode path; only the
        # fetch step differs (HTTP GET vs. S3 GetObject).
        def _download() -> bytes:
            if is_s3:
                from utils.s3_utils import download_s3_bytes

                return download_s3_bytes(src, region=s3_region, endpoint_url=s3_endpoint_url)
            return _fetch_url(src, timeout, retries)

        # Serve from the disk cache when available so a URL is not re-downloaded
        # every epoch. The cache stores the fetched bytes verbatim.
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            path = _cache_path(cache_dir, src)
            if not os.path.exists(path):
                content = _download()
                with open(path, "wb") as f:
                    f.write(content)
            try:
                return _result(_decode(path, num_frames=num_frames, fps=fps, backend=backend))
            except Exception:
                # Corrupt cache entry; re-download once, then give up.
                content = _download()
                with open(path, "wb") as f:
                    f.write(content)
                return _result(_decode(path, num_frames=num_frames, fps=fps, backend=backend))

        # No cache_dir: download to a temp file (load_video needs a path/URL, and
        # this keeps our retry + S3 handling instead of its plain httpx fetch).
        content = _download()
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        try:
            tmp.write(content)
            tmp.close()
            return _result(_decode(tmp.name, num_frames=num_frames, fps=fps, backend=backend))
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    # Treat as a local filesystem path.
    if not os.path.exists(src):
        raise FileNotFoundError(f"Video file not found: {src}")
    return _result(_decode(src, num_frames=num_frames, fps=fps, backend=backend))
