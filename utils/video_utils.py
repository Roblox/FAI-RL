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

    frames, _metadata = load_video(path, backend=backend, **_sample_kwargs(num_frames, fps))
    return frames


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
):
    """Resolve a video source into sampled frames the processor accepts.

    Accepts, in order of preference:
      * an http(s) URL string  -> downloaded (optionally disk-cached) then decoded
      * an ``s3://`` URI string -> downloaded from S3 (optionally disk-cached)
      * a local file path string -> decoded from disk
      * a dict with a ``"url"``/``"path"``/``"video"`` key -> dispatched on the key

    Frames are sampled with ``fps`` if set, else ``num_frames`` uniformly. The
    return value is whatever ``transformers.video_utils.load_video`` produces
    (a ``[T, H, W, C]`` array/tensor) -- pass it to a VLM processor as one entry
    in ``videos=[...]``.

    ``s3_region`` / ``s3_endpoint_url`` are only consulted for ``s3://`` sources;
    when unset, boto3 uses its default credential/region resolution chain.

    Raises:
        RuntimeError / OSError / TypeError: if the video cannot be fetched or
        decoded. Callers that want to skip bad rows should catch the exception.
    """
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
                return _decode(path, num_frames=num_frames, fps=fps, backend=backend)
            except Exception:
                # Corrupt cache entry; re-download once, then give up.
                content = _download()
                with open(path, "wb") as f:
                    f.write(content)
                return _decode(path, num_frames=num_frames, fps=fps, backend=backend)

        # No cache_dir: download to a temp file (load_video needs a path/URL, and
        # this keeps our retry + S3 handling instead of its plain httpx fetch).
        content = _download()
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        try:
            tmp.write(content)
            tmp.close()
            return _decode(tmp.name, num_frames=num_frames, fps=fps, backend=backend)
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    # Treat as a local filesystem path.
    if not os.path.exists(src):
        raise FileNotFoundError(f"Video file not found: {src}")
    return _decode(src, num_frames=num_frames, fps=fps, backend=backend)
