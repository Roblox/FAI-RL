"""Shared tokenizer padding and embedding-resize helpers."""

from __future__ import annotations

import os
from typing import Any, Optional


_ADAPTER_EMBEDDING_SUFFIXES = (
    "embed_tokens.weight",
    "word_embeddings.weight",
    "wte.weight",
    "lm_head.weight",
)


def embedding_row_count(model: Any) -> int:
    """Return the full input-embedding row count, including under ZeRO-3."""
    embeddings = model.get_input_embeddings()
    if embeddings is None or not hasattr(embeddings, "weight"):
        raise ValueError("Model does not expose input token embeddings")

    weight = embeddings.weight
    shape = getattr(weight, "ds_shape", None)
    if shape is None:
        shape = weight.shape
    return int(shape[0])


def adapter_embedding_rows(adapter_path: Optional[str]) -> Optional[int]:
    """Return input-embedding rows stored in a PEFT adapter, when present."""
    if not adapter_path:
        return None

    weights_path = os.path.join(adapter_path, "adapter_model.safetensors")
    if not os.path.isfile(weights_path):
        return None

    try:
        from safetensors import safe_open
    except ImportError:
        return None

    with safe_open(weights_path, framework="pt", device="cpu") as weights:
        keys = list(weights.keys())
        for suffix in _ADAPTER_EMBEDDING_SUFFIXES:
            for key in keys:
                if key.endswith(suffix):
                    return int(weights.get_slice(key).get_shape()[0])
    return None


def configure_padding_token(tokenizer: Any) -> None:
    """Configure left padding without growing the vocabulary when possible."""
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.padding_side = "left"


def grow_token_embeddings_if_needed(
    model: Any,
    tokenizer: Any,
    logger: Any = None,
) -> bool:
    """Grow model embeddings to fit the tokenizer; never shrink them."""
    current_rows = embedding_row_count(model)
    target_rows = len(tokenizer)
    if target_rows <= current_rows:
        return False

    if logger is not None:
        logger.info(
            f"Resizing embeddings {current_rows} -> {target_rows} "
            "(tokenizer has more tokens than the model)"
        )
    model.resize_token_embeddings(target_rows)
    return True


def prepare_tokenizer_with_model(
    tokenizer: Any,
    model: Any,
    *,
    adapter_path: Optional[str] = None,
    logger: Any = None,
) -> Any:
    """Configure padding and safely align tokenizer/model embedding sizes.

    Fresh models only grow when the tokenizer truly contains more tokens.
    Legacy PEFT checkpoints created by FAI-RL may contain the old, resized
    embedding matrices. Match those rows explicitly so existing adapters remain
    loadable; a one-row difference is the historical ``[PAD]`` addition.
    """
    saved_rows = adapter_embedding_rows(adapter_path)
    if saved_rows is not None and saved_rows == len(tokenizer) + 1:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        tokenizer.padding_side = "left"
    else:
        configure_padding_token(tokenizer)

    if saved_rows is None:
        grow_token_embeddings_if_needed(model, tokenizer, logger)
        return tokenizer

    if saved_rows < len(tokenizer):
        raise ValueError(
            f"Adapter embedding has {saved_rows} rows but tokenizer requires "
            f"{len(tokenizer)}"
        )

    current_rows = embedding_row_count(model)
    if current_rows != saved_rows:
        if logger is not None:
            logger.info(
                f"Resizing embeddings {current_rows} -> {saved_rows} "
                "(matching embedding layers stored in PEFT adapter)"
            )
        model.resize_token_embeddings(saved_rows)
    return tokenizer
