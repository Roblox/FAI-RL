"""Blank dataset cells render as "" in prompts (not the literal "None"/"nan")."""

import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference.inference import (
    _blank_if_missing,
    build_chat_messages,
    format_template_prompt,
)


def _config():
    return SimpleNamespace(
        dataset_name="some/data",
        choice_labels=None,
        system_prompt="Persona: {persona}",
        user_prompt="Q: {prompt}",
    )


def test_blank_if_missing_normalizes_only_missing_values():
    assert _blank_if_missing(None) == ""
    assert _blank_if_missing(float("nan")) == ""
    # Real values pass through untouched, including non-scalars (e.g. MMLU choices).
    assert _blank_if_missing("hello") == "hello"
    assert _blank_if_missing(0) == 0
    assert _blank_if_missing(["a", "b"]) == ["a", "b"]


def test_flat_prompt_renders_blank_cell_as_empty_string():
    for missing in (None, float("nan")):
        rendered = format_template_prompt("Q: {prompt}", {"prompt": missing}, _config())
        assert rendered == "Q: "


def test_flat_prompt_preserves_real_values():
    rendered = format_template_prompt("Q: {prompt}", {"prompt": "tell a joke"}, _config())
    assert rendered == "Q: tell a joke"


def test_chat_messages_render_blank_cell_as_empty_string():
    messages = build_chat_messages(_config(), {"persona": "bob", "prompt": None})
    contents = {m["role"]: m["content"] for m in messages}
    assert contents["system"] == "Persona: bob"
    assert contents["user"] == "Q: "
