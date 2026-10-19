"""OpenAI-format tool-calling conversations for text SFT.

Rows hold ``messages`` with system/user/assistant/tool roles, where assistant
messages may carry ``tool_calls`` and tool messages carry their results, plus
optional OpenAI function ``tools`` definitions. Each assistant message with
``weight`` 1 (the default) becomes one pre-tokenized prompt/completion example
whose loss covers only that assistant turn. Expanding per turn masks tool
results and earlier turns on every chat template, including templates without
``{% generation %}`` markers.
"""

import json
from typing import Any, Dict, List, Optional, Tuple


_ROLES = ("system", "user", "assistant", "tool")
EXCEEDS_MAX_LENGTH = "exceeds max_length"
PREFIX_MISMATCH = "chat template does not preserve the prompt prefix"


class InvalidConversation(ValueError):
    """A dataset row is not a trainable tool-calling conversation."""


def _drop_none(value: Any) -> Any:
    """Remove the null fields Arrow adds when it unifies nested row schemas."""
    if isinstance(value, dict):
        return {key: _drop_none(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_drop_none(item) for item in value]
    return value


def _load_json_value(value: Any, location: str) -> Any:
    # JSON strings keep their nulls; nested Arrow values may carry padding.
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError as exc:
            raise InvalidConversation(f"{location} is not valid JSON") from exc
    return _drop_none(value)


def _text_content(content: Any, location: str) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for part in content:
            if not (
                isinstance(part, dict)
                and part.get("type") == "text"
                and isinstance(part.get("text"), str)
            ):
                raise InvalidConversation(f"{location} supports only text content parts")
            texts.append(part["text"])
        return "".join(texts)
    raise InvalidConversation(f"{location} must be a string or a list of text parts")


def _tool_content(content: Any, location: str) -> str:
    if isinstance(content, (str, list)) or content is None:
        return _text_content(content, location)
    return json.dumps(content, separators=(",", ":"), ensure_ascii=False)


def normalize_tools(tools: Any, location: str = "tools") -> Optional[List[Dict[str, Any]]]:
    """Validate OpenAI function tool definitions; empty means no tools."""
    if tools is None:
        return None
    tools = _load_json_value(tools, location)
    if not tools:
        return None
    if not isinstance(tools, list):
        raise InvalidConversation(f"{location} must be a list of tool definitions")
    for index, tool in enumerate(tools):
        function = tool.get("function") if isinstance(tool, dict) else None
        if (
            tool.get("type") != "function"
            or not isinstance(function, dict)
            or not isinstance(function.get("name"), str)
            or not function["name"]
        ):
            raise InvalidConversation(
                f"{location}[{index}] must be an OpenAI function tool with a name"
            )
    return tools


def _normalize_tool_call(call: Any, location: str) -> Dict[str, Any]:
    if not isinstance(call, dict) or call.get("type", "function") != "function":
        raise InvalidConversation(f"{location} must be a function tool call")
    function = call.get("function")
    if not isinstance(function, dict) or not isinstance(function.get("name"), str) or not function["name"]:
        raise InvalidConversation(f"{location}.function.name is required")
    arguments = function.get("arguments", {})
    if isinstance(arguments, str) and not arguments.strip():
        arguments = {}
    arguments = _load_json_value(arguments, f"{location}.function.arguments")
    if not isinstance(arguments, dict):
        raise InvalidConversation(f"{location}.function.arguments must be a JSON object")
    normalized = {
        "type": "function",
        "function": {"name": function["name"], "arguments": arguments},
    }
    if isinstance(call.get("id"), str) and call["id"]:
        normalized["id"] = call["id"]
    return normalized


def normalize_conversation(
    messages: Any,
    tools: Any = None,
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Return ``{"messages", "targets", "tools"}`` for one conversation.

    ``targets`` lists the indices of trainable assistant messages. Raises
    ``InvalidConversation`` with the first problem found.
    """
    messages = _load_json_value(messages, "messages")
    if not isinstance(messages, list) or not messages:
        raise InvalidConversation("messages must be a non-empty list")
    tools = normalize_tools(tools)
    declared = {tool["function"]["name"] for tool in tools} if tools else None

    if system_prompt and not (
        isinstance(messages[0], dict) and messages[0].get("role") == "system"
    ):
        messages = [{"role": "system", "content": system_prompt}, *messages]

    normalized: List[Dict[str, Any]] = []
    targets: List[int] = []
    open_calls: Optional[List[Dict[str, Any]]] = None
    for index, message in enumerate(messages):
        location = f"messages[{index}]"
        if not isinstance(message, dict):
            raise InvalidConversation(f"{location} must be an object")
        role = message.get("role")
        if role not in _ROLES:
            raise InvalidConversation(
                f"{location}.role {role!r} is not one of {', '.join(_ROLES)}"
            )
        weight = message.get("weight", 1)
        if weight not in (0, 1) or (weight == 0 and role != "assistant"):
            raise InvalidConversation(
                f"{location}.weight must be 0 or 1 and only 0 on assistant messages"
            )

        if role == "tool":
            if open_calls is None:
                raise InvalidConversation(
                    f"{location} must follow an assistant message with tool_calls"
                )
            out = {
                "role": "tool",
                "content": _tool_content(message.get("content"), f"{location}.content"),
            }
            call_id = message.get("tool_call_id")
            name = message.get("name")
            if isinstance(call_id, str) and call_id:
                out["tool_call_id"] = call_id
                name = name or next(
                    (c["function"]["name"] for c in open_calls if c.get("id") == call_id),
                    None,
                )
            if not name and len(open_calls) == 1:
                name = open_calls[0]["function"]["name"]
            if name:
                out["name"] = name
            normalized.append(out)
            continue

        out = {"role": role, "content": _text_content(message.get("content"), f"{location}.content")}
        open_calls = None
        if role == "assistant":
            calls = [
                _normalize_tool_call(call, f"{location}.tool_calls[{call_index}]")
                for call_index, call in enumerate(message.get("tool_calls") or [])
            ]
            if not out["content"].strip() and not calls:
                raise InvalidConversation(f"{location} needs content or tool_calls")
            for call in calls:
                name = call["function"]["name"]
                if declared is not None and name not in declared:
                    raise InvalidConversation(f"{location} calls undeclared tool {name!r}")
            if calls:
                out["tool_calls"] = calls
                open_calls = calls
            reasoning = message.get("reasoning_content")
            if isinstance(reasoning, str) and reasoning:
                out["reasoning_content"] = reasoning
            if weight == 1 and normalized:
                targets.append(len(normalized))
        normalized.append(out)

    if not targets:
        raise InvalidConversation("conversation has no trainable assistant turn")
    return {"messages": normalized, "targets": targets, "tools": tools}


def assistant_turns(
    conversation: Dict[str, Any],
) -> List[Tuple[List[Dict[str, Any]], Dict[str, Any]]]:
    """Pair each trainable assistant message with the history before it."""
    messages = conversation["messages"]
    return [(messages[:index], messages[index]) for index in conversation["targets"]]


def normalize_conversation_row(
    example: Dict[str, Any],
    messages_column: str,
    tools_column: Optional[str],
    system_prompt: Optional[str],
) -> Dict[str, Optional[str]]:
    """``datasets.map`` step: one JSON-encoded conversation or a skip reason.

    The conversation is stored as a JSON string so Arrow cannot re-unify the
    nested message schemas across rows.
    """
    try:
        tools = normalize_tools(example.get(tools_column)) if tools_column else None
        conversation = normalize_conversation(
            example.get(messages_column),
            tools=tools,
            system_prompt=system_prompt,
        )
    except InvalidConversation as exc:
        return {"conversation": None, "invalid_reason": str(exc)}
    return {"conversation": json.dumps(conversation), "invalid_reason": None}


def _render(tokenizer, messages, tools, add_generation_prompt) -> str:
    kwargs = {"tools": tools} if tools else {}
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        **kwargs,
    )


def _encode(tokenizer, text: str) -> List[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def tokenize_conversation_rows(
    batch: Dict[str, List[Any]],
    tokenizer,
    max_length: Optional[int],
) -> Dict[str, List[Any]]:
    """Batched ``datasets.map`` step: one row per trainable assistant turn.

    Turns that do not fit ``max_length`` are dropped rather than truncated, so
    training never sees a cut-off tool call. ``dropped_reason`` is set on rows
    the caller must filter out.

    The prompt and the completion are split as text and tokenized separately, so
    the prompt tokens are exactly those generation starts from. Tokenizing the
    whole render would let BPE merge across the boundary: Qwen3.8 ends its
    generation prompt with ``<think>\\n`` and renders the turn as
    ``<think>\\n\\n</think>``, where ``\\n\\n`` is one token.
    """
    out: Dict[str, List[Any]] = {"input_ids": [], "completion_mask": [], "dropped_reason": []}
    for encoded in batch["conversation"]:
        conversation = json.loads(encoded)
        tools = conversation.get("tools")
        for prompt, completion in assistant_turns(conversation):
            prompt_text = _render(tokenizer, prompt, tools, add_generation_prompt=True)
            full_text = _render(tokenizer, [*prompt, completion], tools, add_generation_prompt=False)
            prompt_ids = _encode(tokenizer, prompt_text)
            completion_ids = []
            if full_text.startswith(prompt_text):
                completion_ids = _encode(tokenizer, full_text[len(prompt_text):])
            reason = None
            if not completion_ids:
                reason = PREFIX_MISMATCH
            elif max_length is not None and len(prompt_ids) + len(completion_ids) > max_length:
                reason = EXCEEDS_MAX_LENGTH
            out["input_ids"].append(prompt_ids + completion_ids)
            out["completion_mask"].append([0] * len(prompt_ids) + [1] * len(completion_ids))
            out["dropped_reason"].append(reason)
    return out
