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


def _maybe_unescape_image_bytes(data: bytes) -> bytes:
    """Decode the escaped-bytes text form of an image, if that is what ``data`` is.

    Some datasets store an image as the *escaped* representation of its bytes
    (e.g. content that is the literal text ``\\x89PNG\\r\\n...\\xaeB`\\x82`` rather
    than the raw PNG bytes). We detect that form by its leading ``\\x`` (bytes
    ``0x5C 0x78``) -- no real raster format begins with the ASCII characters
    ``\\x`` -- and reverse the escape back to the original bytes. This applies
    regardless of where the bytes came from (local file, S3, or HTTP). Normal
    binary image bytes are returned unchanged.
    """
    if data[:2] == b"\\x":
        # text like b"\\x89PNG..." -> original bytes b"\x89PNG..."
        return data.decode("unicode_escape").encode("latin-1")
    return data


def _read_local_image_bytes(path: str) -> bytes:
    """Read a local image file's bytes, transparently decoding the escaped-bytes
    text form (see :func:`_maybe_unescape_image_bytes`)."""
    with open(path, "rb") as f:
        return _maybe_unescape_image_bytes(f.read())


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


def fetch_image(
    src: Any,
    *,
    cache_dir: Optional[str] = None,
    timeout: int = 10,
    retries: int = 3,
    s3_region: Optional[str] = None,
    s3_endpoint_url: Optional[str] = None,
):
    """Resolve an image source into an RGB :class:`PIL.Image.Image`.

    Accepts, in order of preference:
      * an http(s) URL string  -> downloaded (optionally disk-cached) and decoded
      * an ``s3://`` URI string -> downloaded from S3 (optionally disk-cached)
      * a local file path string -> opened from disk (raw image bytes, or the
        escaped-bytes text form ``\\x89PNG...`` which is un-escaped first)
      * raw ``bytes``/``bytearray`` -> decoded in-memory
      * an existing ``PIL.Image.Image`` -> returned as RGB (passthrough)
      * a dict with an ``"url"``/``"path"``/``"bytes"``/``"image"`` key (the shape
        some HF datasets use) -> dispatched on the present key

    ``s3_region`` / ``s3_endpoint_url`` are only consulted for ``s3://`` sources;
    when unset, boto3 uses its default credential/region resolution chain.

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
                    src[key],
                    cache_dir=cache_dir,
                    timeout=timeout,
                    retries=retries,
                    s3_region=s3_region,
                    s3_endpoint_url=s3_endpoint_url,
                )
        raise ValueError(f"Unsupported image dict with keys {list(src.keys())}")

    if isinstance(src, str):
        is_http = src.startswith("http://") or src.startswith("https://")
        is_s3 = src.startswith("s3://")
        if is_http or is_s3:
            # Both remote schemes share the same disk-cache + decode path; only
            # the fetch step differs (HTTP GET vs. S3 GetObject).
            def _download() -> bytes:
                if is_s3:
                    from utils.s3_utils import download_s3_bytes

                    return download_s3_bytes(src, region=s3_region, endpoint_url=s3_endpoint_url)
                return _fetch_url(src, timeout, retries)

            # Serve from cache when available. The cache stores the fetched
            # content verbatim; _maybe_unescape_image_bytes handles the
            # escaped-bytes text form on read (so e.g. an s3:// .bin holding
            # escaped text decodes just like a local one).
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
                path = _cache_path(cache_dir, src)
                if os.path.exists(path):
                    try:
                        with open(path, "rb") as f:
                            return _open_rgb(_maybe_unescape_image_bytes(f.read()))
                    except Exception:
                        # Corrupt cache entry; re-download below.
                        pass
                content = _download()
                with open(path, "wb") as f:
                    f.write(content)
                return _open_rgb(_maybe_unescape_image_bytes(content))
            return _open_rgb(_maybe_unescape_image_bytes(_download()))

        # Treat as a local filesystem path. Reads through _read_local_image_bytes
        # so files holding the escaped-bytes text form decode transparently.
        if not os.path.exists(src):
            raise FileNotFoundError(f"Image file not found: {src}")
        return _open_rgb(_read_local_image_bytes(src))

    raise TypeError(f"Unsupported image source type: {type(src)!r}")
