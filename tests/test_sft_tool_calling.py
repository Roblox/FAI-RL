import json
import logging
from types import SimpleNamespace

import pytest
from datasets import Dataset

from core.config import (
    DataConfig,
    DatasetInfo,
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
    WandbConfig,
)
from trainers import sft_trainer


GET_ORDER_TOOL = {"type": "function", "function": {"name": "get_order"}}


class RecordingTokenizer:
    """One token per message; the generation prompt is the next message's first."""

    def __init__(self):
        self.calls = []

    def apply_chat_template(
        self, messages, tools=None, add_generation_prompt=False, **_kwargs
    ):
        self.calls.append({"messages": messages, "tools": tools})
        return "m" * len(messages) + ("m" if add_generation_prompt else "e")

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": [0 if char == "m" else 2 for char in text]}


def _conversation(order_id, weight=1):
    return [
        {"role": "user", "content": f"Where is order {order_id}?"},
        {
            "role": "assistant",
            "content": "",
            "weight": weight,
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_order",
                        "arguments": json.dumps({"order_id": order_id}),
                    },
                }
            ],
        },
        {"role": "tool", "content": "shipped"},
        {"role": "assistant", "content": f"Order {order_id} has shipped."},
    ]


def _trainer(
    monkeypatch,
    rows,
    eval_enabled=False,
    tools_column=None,
    system_prompt=None,
):
    trainer = object.__new__(sft_trainer.SFTTrainer)
    trainer.logger = logging.getLogger("test_sft_tool_calling")
    trainer.tokenizer = RecordingTokenizer()
    trainer.config = SimpleNamespace(
        training=TrainingConfig(
            output_dir="out",
            eval_enabled=eval_enabled,
            early_stopping=False,
            eval_split_ratio=0.5,
            bf16=False,
            fp16=False,
        ),
        data=DataConfig(
            datasets=[DatasetInfo(name="unused.jsonl")],
            messages_column="messages",
            tools_column=tools_column,
            system_prompt=system_prompt,
            max_length=64,
        ),
        wandb=SimpleNamespace(enabled=False),
    )
    monkeypatch.setattr(
        sft_trainer, "load_raw_dataset", lambda _info: Dataset.from_list(rows)
    )
    return trainer


def test_conversational_sft_builds_pretokenized_completion_only_examples(monkeypatch):
    trainer = _trainer(
        monkeypatch,
        [
            {"messages": json.dumps(_conversation("1")), "tools": [GET_ORDER_TOOL]},
            {"messages": json.dumps(_conversation("2", weight=0)), "tools": None},
            {"messages": json.dumps([{"role": "user", "content": "hi"}]), "tools": None},
        ],
        tools_column="tools",
    )

    trainer.setup_data()

    # Two turns from the first row, one from the second; the third is skipped.
    assert len(trainer.train_dataset) == 3
    assert set(trainer.train_dataset.column_names) == {"input_ids", "completion_mask"}
    assert trainer.eval_dataset is None
    tools_seen = [call["tools"] for call in trainer.tokenizer.calls]
    assert tools_seen.count([GET_ORDER_TOOL]) == 4
    assert tools_seen.count(None) == 2


def test_conversational_sft_holds_out_whole_conversations(monkeypatch):
    rows = [{"messages": json.dumps(_conversation(str(i)))} for i in range(4)]
    trainer = _trainer(monkeypatch, rows, eval_enabled=True)

    trainer.setup_data()

    # 2 conversations per split, 2 assistant turns each.
    assert len(trainer.train_dataset) == 4
    assert len(trainer.eval_dataset) == 4
    # Train is tokenized before eval: 2 conversations x 2 turns x 2 renders each.
    users = [call["messages"][0]["content"] for call in trainer.tokenizer.calls]
    train_users, eval_users = set(users[:8]), set(users[8:])
    assert len(train_users) == len(eval_users) == 2
    assert train_users.isdisjoint(eval_users)


def test_conversational_sft_fails_when_no_row_is_trainable(monkeypatch):
    trainer = _trainer(
        monkeypatch, [{"messages": json.dumps([{"role": "user", "content": "hi"}])}]
    )
    with pytest.raises(ValueError, match="no trainable tool-calling conversations"):
        trainer.setup_data()


def test_conversational_sft_requires_configured_columns(monkeypatch):
    trainer = _trainer(monkeypatch, [{"conversation": "[]"}])
    with pytest.raises(ValueError, match="messages_column 'messages' not found"):
        trainer.setup_data()


def test_conversational_sft_enables_completion_only_loss():
    trainer = object.__new__(sft_trainer.SFTTrainer)
    trainer.logger = logging.getLogger("test_sft_tool_args")
    trainer.config = SimpleNamespace(
        training=TrainingConfig(output_dir="out", bf16=False, fp16=False),
        data=DataConfig(messages_column="messages"),
        wandb=SimpleNamespace(enabled=False),
    )

    args = trainer.setup_training_args()

    assert args.completion_only_loss is True


def test_messages_column_is_exclusive_with_prompt_templates():
    with pytest.raises(ValueError, match="messages_column cannot be combined"):
        DataConfig(
            messages_column="messages",
            user_prompt="{question}",
            assistant_prompt="{answer}",
        )
    with pytest.raises(ValueError, match="tools_column requires data.messages_column"):
        DataConfig(tools_column="tools")


@pytest.mark.parametrize("algorithm", ["grpo", "dpo", None])
def test_messages_column_requires_sft(algorithm):
    with pytest.raises(ValueError, match="messages_column is supported only for text SFT"):
        ExperimentConfig(
            model=ModelConfig(base_model_name="meta-llama/Llama-3.2-3B-Instruct"),
            data=DataConfig(messages_column="messages"),
            training=TrainingConfig(output_dir="out", algorithm=algorithm),
            wandb=WandbConfig(enabled=False),
        )
