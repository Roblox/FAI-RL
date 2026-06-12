"""Image loading utilities for multimodal (vision-language) training.

The headline use case is fetching images from HTTP(S) URLs carried in a dataset
column and turning them into PIL images that a VLM processor can consume. Local
file paths, raw bytes, and already-decoded PIL images are also accepted so the
same helper works regardless of how a dataset stores its images.
"""

import hashlib
import io
import os
import time
from typing import Any, Optional


def _cache_path(cache_dir: str, url: str) -> str:
    """Deterministic on-disk cache path for a remote URL."""
    digest = hashlib.md5(url.encode("utf-8")).hexdigest()
    return os.path.join(cache_dir, digest)


def _open_rgb(data_or_path: Any):
    """Open bytes/path with PIL and convert to RGB."""
    from PIL import Image

    if isinstance(data_or_path, (bytes, bytearray)):
        img = Image.open(io.BytesIO(bytes(data_or_path)))
    else:
        img = Image.open(data_or_path)
    return img.convert("RGB")


def _fetch_url(url: str, timeout: int, retries: int) -> bytes:
    """Download a URL with bounded retries and exponential-ish backoff."""
    import requests

    last_err: Optional[Exception] = None
    for attempt in range(max(1, retries)):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp.content
        except Exception as e:  # network error, HTTP error, timeout, etc.
            last_err = e
            if attempt < retries - 1:
                time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"Failed to fetch image from {url!r} after {retries} attempts: {last_err}")


def fetch_image(src: Any, *, cache_dir: Optional[str] = None, timeout: int = 10, retries: int = 3):
    """Resolve an image source into an RGB :class:`PIL.Image.Image`.

    Accepts, in order of preference:
      * an http(s) URL string  -> downloaded (optionally disk-cached) and decoded
      * a local file path string -> opened from disk
      * raw ``bytes``/``bytearray`` -> decoded in-memory
      * an existing ``PIL.Image.Image`` -> returned as RGB (passthrough)
      * a dict with an ``"url"``/``"path"``/``"bytes"``/``"image"`` key (the shape
        some HF datasets use) -> dispatched on the present key

    Raises:
        RuntimeError / OSError: if the image cannot be fetched or decoded. Callers
        that want to skip bad rows should catch the exception.
    """
    from PIL import Image

    # Already-decoded image.
    if isinstance(src, Image.Image):
        return src.convert("RGB")

    # Raw bytes.
    if isinstance(src, (bytes, bytearray)):
        return _open_rgb(src)

    # HF-style dict wrappers, e.g. {"bytes": ...} or {"path": ...} or {"url": ...}.
    if isinstance(src, dict):
        for key in ("image", "bytes", "path", "url"):
            if src.get(key) is not None:
                return fetch_image(
                    src[key], cache_dir=cache_dir, timeout=timeout, retries=retries
                )
        raise ValueError(f"Unsupported image dict with keys {list(src.keys())}")

    if isinstance(src, str):
        if src.startswith("http://") or src.startswith("https://"):
            # Serve from cache when available.
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
                path = _cache_path(cache_dir, src)
                if os.path.exists(path):
                    try:
                        return _open_rgb(path)
                    except Exception:
                        # Corrupt cache entry; re-download below.
                        pass
                content = _fetch_url(src, timeout, retries)
                with open(path, "wb") as f:
                    f.write(content)
                return _open_rgb(content)
            return _open_rgb(_fetch_url(src, timeout, retries))

        # Treat as a local filesystem path.
        if not os.path.exists(src):
            raise FileNotFoundError(f"Image file not found: {src}")
        return _open_rgb(src)

    raise TypeError(f"Unsupported image source type: {type(src)!r}")
