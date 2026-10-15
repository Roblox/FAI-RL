import json
import re

import pytest
from datasets import Dataset

from utils.tool_conversations import (
    InvalidConversation,
    ManagedToolCatalog,
    assistant_turns,
    normalize_conversation,
    normalize_conversation_row,
    tokenize_conversation_rows,
    wrap_for_managed_tools,
)


GET_ORDER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_order",
        "description": "Return an order's current status.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "The order id."}
            },
            "required": ["order_id"],
        },
    },
}


def _openai_messages():
    return [
        {"role": "user", "content": "Where is order 123?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_order",
                        "arguments": '{"order_id": "123"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": {"status": "shipped"}},
        {"role": "assistant", "content": "Order 123 has shipped."},
    ]


def test_normalizes_openai_tool_calling_conversation():
    conversation = normalize_conversation(_openai_messages(), tools=[GET_ORDER_TOOL])

    assert conversation["messages"] == [
        {"role": "user", "content": "Where is order 123?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "get_order", "arguments": {"order_id": "123"}},
                    "id": "call_1",
                }
            ],
        },
        {
            "role": "tool",
            "content": '{"status":"shipped"}',
            "tool_call_id": "call_1",
            "name": "get_order",
        },
        {"role": "assistant", "content": "Order 123 has shipped."},
    ]
    assert conversation["targets"] == [1, 3]
    assert conversation["tools"] == [GET_ORDER_TOOL]


def test_system_prompt_is_prepended_only_when_missing():
    conversation = normalize_conversation(
        _openai_messages(), system_prompt="Use tools when needed."
    )
    assert conversation["messages"][0] == {
        "role": "system",
        "content": "Use tools when needed.",
    }
    assert conversation["targets"] == [2, 4]

    with_system = [{"role": "system", "content": "Row prompt."}, *_openai_messages()]
    conversation = normalize_conversation(with_system, system_prompt="Ignored.")
    assert conversation["messages"][0]["content"] == "Row prompt."


def test_weight_zero_keeps_turn_as_history_but_not_as_target():
    messages = _openai_messages()
    messages[1]["weight"] = 0

    conversation = normalize_conversation(messages)

    assert conversation["targets"] == [3]
    assert all("weight" not in message for message in conversation["messages"])


def test_accepts_json_string_columns():
    conversation = normalize_conversation(
        json.dumps(_openai_messages()), tools=json.dumps([GET_ORDER_TOOL])
    )

    assert conversation["tools"] == [GET_ORDER_TOOL]
    assert conversation["messages"][1]["tool_calls"][0]["function"]["arguments"] == {
        "order_id": "123"
    }


def test_drops_null_padding_from_arrow_schema_unification():
    rows = [
        {
            "messages": [
                {"role": "user", "content": "Ship 1"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {"name": "ship", "arguments": {"order_id": "1"}},
                        }
                    ],
                },
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Refund 2"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {"name": "refund", "arguments": {"amount": 2}},
                        }
                    ],
                },
            ]
        },
    ]
    # Struct-typed sources (in-memory lists, Parquet, Hub datasets) merge both
    # argument shapes into one struct and pad with nulls.
    dataset = Dataset.from_list(rows)
    assert dataset[0]["messages"][1]["tool_calls"][0]["function"]["arguments"] == {
        "amount": None,
        "order_id": "1",
    }

    conversation = normalize_conversation(dataset[0]["messages"])

    assert conversation["messages"] == [
        {"role": "user", "content": "Ship 1"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "ship", "arguments": {"order_id": "1"}},
                }
            ],
        },
    ]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda m: m.__setitem__(0, {"role": "bot", "content": "hi"}), "role"),
        (lambda m: m.pop(1), "must follow an assistant message with tool_calls"),
        (
            lambda m: m[1]["tool_calls"][0]["function"].__setitem__("arguments", "{bad"),
            "not valid JSON",
        ),
        (
            lambda m: m[1]["tool_calls"][0]["function"].__setitem__("arguments", "[1]"),
            "must be a JSON object",
        ),
        (lambda m: m[3].__setitem__("content", " "), "needs content or tool_calls"),
        (lambda m: m[0].__setitem__("weight", 0), "weight"),
    ],
)
def test_rejects_malformed_conversations(mutate, message):
    messages = _openai_messages()
    mutate(messages)
    with pytest.raises(InvalidConversation, match=message):
        normalize_conversation(messages)


def test_rejects_conversation_without_trainable_assistant_turn():
    messages = _openai_messages()
    messages[1]["weight"] = 0
    messages[3]["weight"] = 0
    with pytest.raises(InvalidConversation, match="no trainable assistant turn"):
        normalize_conversation(messages)


def test_rejects_call_to_undeclared_tool():
    messages = _openai_messages()
    messages[1]["tool_calls"][0]["function"]["name"] = "cancel_order"
    with pytest.raises(InvalidConversation, match="undeclared tool 'cancel_order'"):
        normalize_conversation(messages, tools=[GET_ORDER_TOOL])


def test_assistant_turns_pair_each_target_with_its_history():
    conversation = normalize_conversation(_openai_messages())

    turns = assistant_turns(conversation)

    assert [len(prompt) for prompt, _ in turns] == [1, 3]
    assert turns[0][1]["tool_calls"][0]["function"]["name"] == "get_order"
    assert turns[1][1] == {"role": "assistant", "content": "Order 123 has shipped."}


def test_row_normalization_reads_configured_columns_and_reports_invalid_rows():
    row = {"conversation_messages": _openai_messages(), "row_tools": [GET_ORDER_TOOL]}

    normalized = normalize_conversation_row(
        row,
        messages_column="conversation_messages",
        tools_column="row_tools",
        system_prompt=None,
    )

    assert normalized["invalid_reason"] is None
    assert json.loads(normalized["conversation"])["tools"] == [GET_ORDER_TOOL]

    invalid = normalize_conversation_row(
        {"conversation_messages": [{"role": "user", "content": "hi"}]},
        messages_column="conversation_messages",
        tools_column=None,
        system_prompt=None,
    )
    assert invalid["conversation"] is None
    assert "no trainable assistant turn" in invalid["invalid_reason"]


CALL_TOOL = {"type": "function", "function": {"name": "call_tool"}}
MANAGED_TOOLS = ManagedToolCatalog(
    instructions="Call these tools: [get_order]",
    tools=[CALL_TOOL],
    input_schemas={
        "get_order": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
            "additionalProperties": False,
        }
    },
)


def test_managed_tools_wrap_calls_and_results_as_grpo_renders_them():
    conversation = wrap_for_managed_tools(
        normalize_conversation(_openai_messages()), MANAGED_TOOLS
    )

    assert conversation["messages"] == [
        {"role": "system", "content": "Call these tools: [get_order]"},
        {"role": "user", "content": "Where is order 123?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "call_tool",
                        "arguments": {
                            "tool_name": "get_order",
                            "arguments": {"order_id": "123"},
                        },
                    },
                }
            ],
        },
        {"role": "tool", "name": "call_tool", "content": '{"status":"shipped"}'},
        {"role": "assistant", "content": "Order 123 has shipped."},
    ]
    assert conversation["targets"] == [2, 4]
    assert conversation["tools"] == [CALL_TOOL]


def test_managed_tools_append_the_catalog_to_an_existing_system_message():
    conversation = wrap_for_managed_tools(
        normalize_conversation(_openai_messages(), system_prompt="Be brief."),
        MANAGED_TOOLS,
    )

    assert conversation["messages"][0] == {
        "role": "system",
        "content": "Be brief.\n\nCall these tools: [get_order]",
    }
    assert conversation["targets"] == [2, 4]


@pytest.mark.parametrize(
    "recorded, observed",
    [
        ('{"status": "shipped"}', '{"status":"shipped"}'),
        ("shipped", '"shipped"'),
        ({"status": "shipped"}, '{"status":"shipped"}'),
    ],
)
def test_managed_tools_encode_results_as_gateway_json(recorded, observed):
    messages = _openai_messages()
    messages[2]["content"] = recorded

    conversation = wrap_for_managed_tools(normalize_conversation(messages), MANAGED_TOOLS)

    assert conversation["messages"][3]["content"] == observed


@pytest.mark.parametrize(
    "mutate, message",
    [
        (
            lambda m: m[1]["tool_calls"][0]["function"].update(name="cancel_order"),
            "undeclared tool 'cancel_order'",
        ),
        (
            lambda m: m[1]["tool_calls"][0]["function"].update(arguments='{"order_id": 123}'),
            "does not match its input_schema",
        ),
    ],
)
def test_managed_tools_reject_calls_the_environment_would_refuse(mutate, message):
    messages = _openai_messages()
    mutate(messages)

    with pytest.raises(InvalidConversation, match=message):
        wrap_for_managed_tools(normalize_conversation(messages), MANAGED_TOOLS)


def test_row_normalization_applies_managed_tools():
    normalized = normalize_conversation_row(
        {"messages": json.dumps(_openai_messages())},
        messages_column="messages",
        tools_column=None,
        system_prompt=None,
        managed_tools=MANAGED_TOOLS,
    )

    conversation = json.loads(normalized["conversation"])
    assert conversation["tools"] == [CALL_TOOL]
    assert conversation["messages"][0]["role"] == "system"


class FakeChatTokenizer:
    """Renders tags per message; ``<...>`` tags, newlines, and words are tokens.

    Like Qwen's BPE, a newline pair is one token. With ``thinking``, the
    generation prompt opens ``<think>\\n`` and assistant history renders a
    closed ``<think>\\n\\n</think>``, as Qwen3.8 does.
    """

    def __init__(self, rerender_history=False, thinking=False):
        self.rerender_history = rerender_history
        self.thinking = thinking
        self.calls = []
        self.vocab = {}

    def apply_chat_template(
        self, messages, tools=None, add_generation_prompt=False, **kwargs
    ):
        self.calls.append({"messages": messages, "tools": tools, **kwargs})
        text = "<tools>" if tools else ""
        for message in messages:
            text += f"<{message['role']}>"
            if message["role"] == "assistant" and self.thinking:
                text += "<think>\n\n</think>"
            text += "<call>" * len(message.get("tool_calls", []))
            if message["content"]:
                text += "<c>"
        if self.rerender_history:
            text += "<x>"
        if add_generation_prompt:
            text += "<assistant><think>\n" if self.thinking else "<assistant>"
        return text

    def __call__(self, text, add_special_tokens=True):
        assert add_special_tokens is False
        pieces = re.findall(r"<[^>]+>|\n\n|\n|[^<\n]+", text)
        return {"input_ids": [self.vocab.setdefault(p, len(self.vocab)) for p in pieces]}

    def decode(self, ids):
        pieces = {i: p for p, i in self.vocab.items()}
        return [pieces[i] for i in ids]


def _conversation_batch(*conversations):
    return {"conversation": [json.dumps(conversation) for conversation in conversations]}


def _split(tokenizer, out, index):
    tokens = tokenizer.decode(out["input_ids"][index])
    mask = out["completion_mask"][index]
    assert mask == sorted(mask)  # prompt then completion, never interleaved
    first = mask.index(1)
    return tokens[:first], tokens[first:]


def test_tokenizes_one_example_per_assistant_turn_with_completion_masks():
    conversation = normalize_conversation(_openai_messages(), tools=[GET_ORDER_TOOL])
    tokenizer = FakeChatTokenizer()

    out = tokenize_conversation_rows(
        _conversation_batch(conversation), tokenizer=tokenizer, max_length=64
    )

    assert out["dropped_reason"] == [None, None]
    assert _split(tokenizer, out, 0) == (
        ["<tools>", "<user>", "<c>", "<assistant>"], ["<call>"]
    )
    assert _split(tokenizer, out, 1) == (
        ["<tools>", "<user>", "<c>", "<assistant>", "<call>", "<tool>", "<c>", "<assistant>"],
        ["<c>"],
    )
    assert tokenizer.calls[0]["tools"] == [GET_ORDER_TOOL]
    assert tokenizer.calls[0]["tokenize"] is False


def test_prompt_tokens_are_the_generation_prompt_even_when_bpe_merges_across_it():
    # Qwen3.8 opens "<think>\n" in the prompt and renders "<think>\n\n</think>" in
    # history, so tokenizing the whole render merges the boundary into "\n\n".
    conversation = normalize_conversation(_openai_messages())
    tokenizer = FakeChatTokenizer(thinking=True)

    out = tokenize_conversation_rows(
        _conversation_batch(conversation), tokenizer=tokenizer, max_length=64
    )

    assert out["dropped_reason"] == [None, None]
    prompt, completion = _split(tokenizer, out, 0)
    assert prompt == ["<user>", "<c>", "<assistant>", "<think>", "\n"]
    assert completion == ["\n", "</think>", "<call>"]


def test_drops_turns_that_do_not_fit_or_rerender_history():
    conversation = normalize_conversation(_openai_messages())

    too_long = tokenize_conversation_rows(
        _conversation_batch(conversation), tokenizer=FakeChatTokenizer(), max_length=6
    )
    assert too_long["dropped_reason"] == [None, "exceeds max_length"]

    mismatch = tokenize_conversation_rows(
        _conversation_batch(conversation),
        tokenizer=FakeChatTokenizer(rerender_history=True),
        max_length=64,
    )
    assert mismatch["dropped_reason"] == [
        "chat template does not preserve the prompt prefix",
        "chat template does not preserve the prompt prefix",
    ]
