"""Native SFT on ``engine`` primitives (``training.backend: native``).

Every example becomes ``input_ids`` + ``completion_mask`` and then a ``Datum``
trained with the built-in ``cross_entropy`` loss. Weights are divided by the
global count of trained tokens per optimizer step, so the loss is a per-token
mean like TRL's.

Data modes match the TRL SFT trainer:
- flat: ``system_prompt`` template (or ``text_column``), loss on every token;
- split (``user_prompt`` + ``assistant_prompt``): chat-template prompt and
  completion, loss on the completion;
- conversations (``messages_column``): one example per trainable assistant turn.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import List

from datasets import Dataset, Features, Value, concatenate_datasets

from core.config import ExperimentConfig
from utils.dataset_utils import load_raw_dataset
from utils.tool_conversations import normalize_conversation_row, tokenize_conversation_rows

from ..training_client import TrainingClient
from ..types import Datum, ModelInput
from .common import LoopResult, SupervisedObjective, run_supervised

_CONVERSATION_FEATURES = Features(
    {"conversation": Value("string"), "invalid_reason": Value("string")}
)


def _render_template(template: str, values: dict, logger) -> str:
    try:
        return template.format(**values)
    except (KeyError, IndexError, ValueError) as error:
        logger.warning("Prompt template not rendered (%s); using literal text", error)
        return template


def _log_reasons(logger, reasons, what: str, source: str) -> None:
    counts = Counter(reason for reason in reasons if reason)
    if counts:
        summary = "; ".join(f"{reason} ({count})" for reason, count in counts.most_common(5))
        logger.warning("Skipped %d %s from %s: %s", sum(counts.values()), what, source, summary)


def datum_from_tokens(input_ids: List[int], completion_mask: List[int], scale: float = 1.0) -> Datum:
    """Shift-by-one ``Datum``: position ``i`` predicts ``input_ids[i + 1]``."""
    return Datum(
        model_input=ModelInput.from_ints(input_ids[:-1]),
        loss_fn_inputs={
            "target_tokens": input_ids[1:],
            "weights": [float(mask) * scale for mask in completion_mask[1:]],
        },
    )


class SFTObjective(SupervisedObjective):
    def __init__(self, config: ExperimentConfig, tokenizer, logger):
        self.config = config
        self.tokenizer = tokenizer
        self.logger = logger

    # ------------------------------------------------------------------ data

    def _source_rows(self) -> Dataset:
        """One row per example before tokenization: ``conversation`` JSON
        (split / messages modes) or ``text`` (flat mode)."""
        data = self.config.data
        parts = []
        for dataset_info in data.datasets:
            self.logger.info("Loading dataset: %s", dataset_info.name)
            dataset = load_raw_dataset(dataset_info)
            if data.messages_column:
                rows = dataset.map(
                    normalize_conversation_row,
                    fn_kwargs={
                        "messages_column": data.messages_column,
                        "tools_column": data.tools_column,
                        "system_prompt": data.system_prompt,
                    },
                    remove_columns=dataset.column_names,
                    features=_CONVERSATION_FEATURES,
                )
            elif data.split_mode:
                rows = dataset.map(
                    self._split_mode_conversation,
                    fn_kwargs={"columns": dataset_info.dataset_columns or []},
                    remove_columns=dataset.column_names,
                    features=_CONVERSATION_FEATURES,
                )
            else:
                rows = dataset.map(
                    self._flat_text,
                    fn_kwargs={
                        "columns": dataset_info.dataset_columns,
                        "text_column": dataset_info.text_column,
                    },
                    remove_columns=dataset.column_names,
                )
                rows = rows.map(
                    lambda row: {"invalid_reason": None if row["text"] and row["text"].strip() else "empty text"}
                )
            _log_reasons(self.logger, rows["invalid_reason"], "examples", dataset_info.name)
            rows = rows.filter(lambda row: row["invalid_reason"] is None)
            parts.append(rows.remove_columns("invalid_reason"))
        if not parts:
            raise ValueError("SFT requires at least one dataset in data.datasets")
        return parts[0] if len(parts) == 1 else concatenate_datasets(parts)

    def _split_mode_conversation(self, example, columns):
        data = self.config.data
        values = {column: example.get(column, "") for column in columns}
        messages = []
        if data.system_prompt:
            messages.append({"role": "system", "content": _render_template(data.system_prompt, values, self.logger)})
        user = _render_template(data.user_prompt, values, self.logger)
        assistant = _render_template(data.assistant_prompt, values, self.logger)
        if not (user and user.strip() and assistant and assistant.strip()):
            return {"conversation": None, "invalid_reason": "empty user or assistant content"}
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": assistant})
        conversation = {"messages": messages, "targets": [len(messages) - 1], "tools": None}
        return {"conversation": json.dumps(conversation), "invalid_reason": None}

    def _flat_text(self, example, columns, text_column):
        system_prompt = self.config.data.system_prompt
        if system_prompt and columns:
            values = {column: example[column] for column in columns if column in example}
            return {"text": _render_template(system_prompt, values, self.logger)}
        return {"text": example.get(text_column)}

    def _tokenize(self, rows: Dataset, split: str) -> Dataset:
        max_length = self.config.data.max_length
        if "conversation" in rows.column_names:
            tokenized = rows.map(
                tokenize_conversation_rows,
                batched=True,
                fn_kwargs={"tokenizer": self.tokenizer, "max_length": max_length},
                remove_columns=rows.column_names,
            )
            _log_reasons(self.logger, tokenized["dropped_reason"], "assistant turns", f"the {split} split")
            tokenized = tokenized.filter(lambda row: row["dropped_reason"] is None)
            tokenized = tokenized.remove_columns("dropped_reason")
        else:
            eos = self.tokenizer.eos_token_id

            def tokenize_text(row):
                ids = list(self.tokenizer(row["text"], add_special_tokens=True)["input_ids"])
                if eos is not None and (not ids or ids[-1] != eos):
                    ids.append(eos)
                ids = ids[:max_length]
                return {"input_ids": ids, "completion_mask": [1] * len(ids)}

            tokenized = rows.map(tokenize_text, remove_columns=rows.column_names)
        tokenized = tokenized.filter(lambda row: len(row["input_ids"]) >= 2)
        if len(tokenized) == 0:
            raise ValueError(f"No {split} SFT examples left after tokenization")
        self.logger.info("Prepared %d %s SFT examples", len(tokenized), split)
        return tokenized

    def load_splits(self, training, logger):
        source = self._source_rows()
        eval_source = None
        if training.eval_enabled:
            from utils.early_stopping import hold_out_eval_dataset

            source, eval_source = hold_out_eval_dataset(source, training.eval_split_ratio, logger=logger)
        train_rows = self._tokenize(source, "train")
        eval_rows = self._tokenize(eval_source, "eval") if eval_source is not None else None
        return train_rows, eval_rows

    # ------------------------------------------------------------------ loss

    def count_units(self, rows: List[dict]) -> float:
        return float(sum(sum(row["completion_mask"][1:]) for row in rows))

    def _data(self, rows: List[dict], scale: float) -> List[Datum]:
        return [datum_from_tokens(row["input_ids"], row["completion_mask"], scale) for row in rows]

    def train_step(self, client: TrainingClient, rows: List[dict], scale: float) -> float:
        output = client.forward_backward(self._data(rows, scale), loss_fn="cross_entropy")
        return output.metrics.get("loss:sum", 0.0)

    def eval_step(self, client: TrainingClient, rows: List[dict]) -> float:
        output = client.forward(self._data(rows, 1.0), loss_fn="cross_entropy")
        return output.metrics.get("loss:sum", 0.0)


def run(config: ExperimentConfig, logger) -> LoopResult:
    return run_supervised(config, logger, SFTObjective)
