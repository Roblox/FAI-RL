"""Helpers shared by multimodal inference inputs."""

from typing import Any, Iterable

import pandas as pd


def _is_missing_media_source(value: Any) -> bool:
    """Return whether a dataset cell contains no media source."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()

    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        # Non-scalar values can produce an array of booleans. Those are valid
        # source objects here and remain subject to the fetcher's validation.
        return False


def collect_media_sources(values: Iterable[Any]) -> list[str]:
    """Flatten media columns while dropping null, NaN, and empty cells."""
    sources = []
    for raw in values:
        items = raw if isinstance(raw, (list, tuple)) else (raw,)
        for item in items:
            if _is_missing_media_source(item):
                continue
            sources.append(item.strip() if isinstance(item, str) else str(item))
    return sources
