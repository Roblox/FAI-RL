"""CPU preflight checks that follow each trainer's dataset conventions."""

import json
import math
from pathlib import Path
from typing import Any, List, Optional


def _missing(value: Any) -> bool:
    return (
        value is None
        or (isinstance(value, float) and math.isnan(value))
        or (isinstance(value, str) and not value.strip())
    )


def required_columns(info: Any, data: Any, algorithm: str) -> List[str]:
    """Return source columns actually consumed by the selected trainer."""
    if algorithm == "dpo":
        return [info.chosen_column, info.rejected_column]  # prompt is optional
    if algorithm in {"grpo", "gspo"}:
        return [info.prompt_column]  # rewards need not have a ground-truth answer
    if algorithm == "cpt":
        return [info.text_column]
    if algorithm == "sft_vlm":
        return list(
            dict.fromkeys(
                (info.dataset_columns or [])
                + (info.image_columns or [])
                + (info.video_columns or [])
            )
        )
    if data.split_mode or (data.system_prompt and info.dataset_columns):
        return info.dataset_columns or []
    return ["text"]  # legacy SFT consumes literal 'text', unlike CPT


def sample_messages(row: dict, info: Any, data: Any, algorithm: str) -> Optional[list]:
    """Render the same text turns as training; return None for plain-text paths."""
    if algorithm in {"grpo", "gspo"}:
        prompt = row[info.prompt_column]
        if isinstance(prompt, list):
            return prompt
        if data.system_prompt:
            return [
                {"role": "system", "content": data.system_prompt},
                {"role": "user", "content": str(prompt)},
            ]
        return None
    if algorithm not in {"sft", "sft_vlm"}:
        return None
    fmt = {column: row[column] for column in (info.dataset_columns or [])}
    if data.split_mode:
        turns = []
        if data.system_prompt:
            turns.append({"role": "system", "content": data.system_prompt.format(**fmt)})
        turns.extend(
            [
                {"role": "user", "content": data.user_prompt.format(**fmt)},
                {"role": "assistant", "content": data.assistant_prompt.format(**fmt)},
            ]
        )
    else:
        if data.system_prompt and (info.dataset_columns or algorithm == "sft_vlm"):
            text = data.system_prompt.format(**fmt)
        elif algorithm == "sft_vlm":
            text = "\n".join(str(value) for value in fmt.values())
        else:
            return None
        turns = [{"role": "user", "content": text}]
        if algorithm == "sft":
            return None  # flat SFT validates formatting but does not use chat
    if algorithm == "sft_vlm":
        user = next(turn for turn in turns if turn["role"] == "user")
        content = []
        for kind, columns in (("image", info.image_columns), ("video", info.video_columns)):
            for column in columns or []:
                value = row[column]
                items = value if isinstance(value, (list, tuple)) else [value]
                content.extend({"type": kind} for item in items if not _missing(item))
        user["content"] = content + [{"type": "text", "text": user["content"]}]
    return turns


def validate_dataset(
    dataset: Any, info: Any, data: Any, algorithm: str, logger: Any, sample_size: int = 32
) -> list:
    """Check schema, missing values, rendered text, and up to 32 chat samples.

    Missing values are summarized, preserving the trainers' existing row filtering.
    Samples contain only in-memory chat turns and are never written to logs.
    """
    if not len(dataset):
        raise ValueError(f"Dataset {info.name!r} is empty (split={info.split!r}).")
    columns = required_columns(info, data, algorithm)
    missing = set(columns) - set(dataset.column_names)
    if missing:
        raise ValueError(
            f"Dataset {info.name!r} missing column(s): {', '.join(sorted(missing))}\n"
            f"Expected: {', '.join(columns)}\nFound: {', '.join(dataset.column_names)}\n"
            "Fix data.datasets column mappings or the source dataset."
        )
    missing_rows, usable_rows, sampled = 0, 0, 0
    chats = []
    for index, row in enumerate(dataset):
        if any(_missing(row[column]) for column in columns):
            missing_rows += 1
        # Match existing filters; media columns may legitimately be null.
        if algorithm in {"cpt", "dpo"} or (
            algorithm == "sft"
            and not data.split_mode
            and not (data.system_prompt and info.dataset_columns)
        ):
            usable = all(isinstance(row[column], str) and row[column].strip() for column in columns)
        elif algorithm in {"grpo", "gspo"}:
            value = row[info.prompt_column]
            usable = (isinstance(value, str) and bool(value.strip())) or (
                isinstance(value, list) and bool(value)
            )
        else:
            fmt = {column: row[column] for column in (info.dataset_columns or [])}
            templates = (
                [data.user_prompt, data.assistant_prompt]
                if data.split_mode
                else [data.system_prompt]
            )
            try:
                texts = [
                    template.format(**fmt) if template else "\n".join(str(v) for v in fmt.values())
                    for template in templates
                ]
            except (KeyError, ValueError, IndexError, TypeError, AttributeError):
                raise ValueError(
                    f"Dataset {info.name!r}, row {index}: sample formatting failed. "
                    "Check dataset_columns and prompt placeholders; escape literal braces as {{ and }}."
                ) from None
            usable = all(text.strip() for text in texts)
            if algorithm == "sft_vlm":
                media = [
                    row[column]
                    for column in (info.image_columns or []) + (info.video_columns or [])
                ]
                usable = usable and any(not _missing(value) and value != [] for value in media)
        if not usable:
            continue
        usable_rows += 1
        if sampled >= sample_size:
            continue
        try:
            messages = sample_messages(row, info, data, algorithm)
            if messages:
                for message in messages:
                    if not isinstance(message, dict) or not {"role", "content"} <= message.keys():
                        raise ValueError("chat messages require role and content keys")
                chats.append(messages)
        except (KeyError, ValueError, IndexError, TypeError, AttributeError):
            raise ValueError(
                f"Dataset {info.name!r}, row {index}: sample formatting failed. "
                "Check dataset_columns, prompt placeholders (escape literal braces as {{ and }}), and chat role/content fields."
            ) from None
        sampled += 1
    if not usable_rows:
        raise ValueError(
            f"Dataset {info.name!r} has no usable rows in required columns: {', '.join(columns)}."
        )
    if missing_rows:
        logger.warning(
            "Dataset %s: %d/%d rows have missing values; existing trainer filtering is unchanged.",
            info.name,
            missing_rows,
            len(dataset),
        )
    logger.info(
        "Dataset preflight: %s (subset=%s, split=%s): %d rows; %d formatting samples checked.",
        info.name,
        info.subset,
        info.split,
        len(dataset),
        sampled,
    )
    return chats


def preflight_datasets(config: Any, logger: Any, algorithm: Optional[str] = None) -> None:
    """Load/validate raw data once and retain it for the trainers' setup_data step."""
    if hasattr(config, "_preflight_datasets"):
        return
    from utils.dataset_utils import load_raw_dataset

    if not config.data.datasets:
        raise ValueError(
            "Invalid config: data.datasets must contain at least one training dataset."
        )
    algorithm = (config.training.algorithm or algorithm or "sft").lower()
    raw_datasets, chats = [], []
    for info in config.data.datasets:
        dataset = load_raw_dataset(info)
        chats.extend(validate_dataset(dataset, info, config.data, algorithm, logger))
        raw_datasets.append(dataset)
    config._preflight_datasets = raw_datasets
    config._preflight_chats = chats
    config._preflight_algorithm = algorithm


def validate_chat_templates(config: Any, logger: Any) -> None:
    """Render sampled chats using only a tokenizer/processor, before model weights."""
    chats = getattr(config, "_preflight_chats", [])
    if not chats:
        return
    from transformers import AutoProcessor, AutoTokenizer

    name = config.model.base_model_name
    adapter_config = Path(name) / "adapter_config.json"
    if adapter_config.is_file():
        name = json.loads(adapter_config.read_text(encoding="utf-8"))["base_model_name_or_path"]
    algorithm = (
        config.training.algorithm or getattr(config, "_preflight_algorithm", "sft")
    ).lower()
    loader = AutoProcessor if algorithm == "sft_vlm" else AutoTokenizer
    try:
        processor = loader.from_pretrained(name)
        for messages in chats:
            processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=algorithm in {"grpo", "gspo"}
            )
    except Exception as exc:
        raise ValueError(
            f"Chat template preflight failed for {name!r} ({type(exc).__name__}). "
            "Check tokenizer/processor access, chat_template, and supported roles/media for this model."
        ) from None
    logger.info(
        "Chat template preflight passed (%d samples). Media fetching/decoding is checked during training.",
        len(chats),
    )
    config._preflight_chats = []
