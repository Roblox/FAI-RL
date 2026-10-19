"""Tool-calling conversations for local text inference."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import inference.inference as inference_module
from core.config import InferenceConfig


GET_ORDER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_order",
        "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}},
    },
}
QWEN_CALL = '<tool_call>\n{"name": "get_order", "arguments": {"order_id": "7"}}\n</tool_call>'


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"tools_column": "tools"}, "requires inference.messages_column"),
        ({"messages_column": "m", "user_prompt": "{q}"}, "user_prompt"),
        ({"messages_column": "m", "image_columns": ["img"]}, "image_columns"),
        ({"messages_column": "m", "json_schema_from_column": "label"}, "json_schema"),
        ({"messages_column": "m", "model": "gpt", "api_key": "k"}, "api_key"),
    ],
)
def test_config_rejects_settings_tool_calling_cannot_use(overrides, error):
    with pytest.raises(ValueError, match=error):
        InferenceConfig(**overrides)


class _Inputs(dict):
    @property
    def input_ids(self):
        return self["input_ids"]

    def to(self, _device):
        return self


class _Model:
    device = "cpu"

    def generate(self, **_kwargs):
        return SimpleNamespace(
            sequences=torch.tensor([[10, 11, 20]]),
            scores=(torch.tensor([[1.0]]),),
            beam_indices=None,
        )


class _Tokenizer:
    pad_token_id = 0

    def apply_chat_template(self, _messages, **kwargs):
        self.template_kwargs = kwargs
        return _Inputs(input_ids=torch.tensor([[10, 11]]))

    def decode(self, _tokens, skip_special_tokens):
        return QWEN_CALL


def test_generate_response_passes_tools_to_the_chat_template():
    config = SimpleNamespace(
        enable_thinking=None, max_new_tokens=1, do_sample=False, temperature=1.0, top_p=1.0
    )
    messages = [{"role": "user", "content": "Where is order 7?"}]

    tokenizer = _Tokenizer()
    inference_module.generate_response(
        _Model(), tokenizer, config=config, messages=messages, tools=[GET_ORDER_TOOL]
    )
    assert tokenizer.template_kwargs["tools"] == [GET_ORDER_TOOL]

    inference_module.generate_response(_Model(), tokenizer, config=config, messages=messages)
    assert "tools" not in tokenizer.template_kwargs


def test_run_inference_generates_next_turn_and_parses_tool_calls(monkeypatch, tmp_path):
    rows = [
        {
            "id": "call",
            "messages": [{"role": "user", "content": "Where is order 7?"}],
            "tools": json.dumps([GET_ORDER_TOOL]),
        },
        {
            # A held-out SFT row: its final assistant turn is what gets predicted.
            "id": "reply",
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
            ],
            "tools": None,
        },
        {"id": "invalid", "messages": [{"role": "robot", "content": "?"}], "tools": None},
    ]
    seen = []

    def generate(_model, _tokenizer, config=None, messages=None, tools=None, **_kwargs):
        seen.append((messages, tools))
        return (QWEN_CALL if tools else "Hello!"), 0.5

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(inference_module, "load_raw_dataset", lambda _c: rows)
    monkeypatch.setattr(inference_module, "load_model_and_tokenizer", lambda _c: (None, None))
    monkeypatch.setattr(inference_module, "generate_response", generate)

    inference_module.run_inference(
        InferenceConfig(
            model_paths=["c"],
            output_file="r.csv",
            dataset_columns=["id", "messages"],
            messages_column="messages",
            tools_column="tools",
            system_prompt="Use tools.",
        )
    )

    assert seen == [
        (
            [
                {"role": "system", "content": "Use tools."},
                {"role": "user", "content": "Where is order 7?"},
            ],
            [GET_ORDER_TOOL],
        ),
        (
            [{"role": "system", "content": "Use tools."}, {"role": "user", "content": "Hi"}],
            None,
        ),
    ]
    result = pd.read_csv("r.csv", keep_default_na=False)
    assert result["id"].tolist() == ["call", "reply"]
    assert json.loads(result["messages"][0]) == rows[0]["messages"]
    assert result["response"].tolist() == [QWEN_CALL, "Hello!"]
    assert [json.loads(cell) for cell in result["__tool_calls"]] == [
        [{"type": "function", "function": {"name": "get_order", "arguments": {"order_id": "7"}}}],
        [],
    ]


def test_example_recipe_loads():
    recipe = REPO_ROOT / "recipes/inference/llama3_3B_tool_calling.yaml"
    args = SimpleNamespace(recipe=str(recipe), overrides=[])
    config = inference_module.load_inference_recipe_with_overrides(args)
    assert (config.messages_column, config.tools_column) == ("messages", "tools")
    assert Path(REPO_ROOT, config.dataset_name).exists()
