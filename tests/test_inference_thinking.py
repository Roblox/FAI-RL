"""Thinking-mode chat-template controls for local inference."""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.config import InferenceConfig
from inference.inference import generate_response, generate_vlm_response


class _Inputs(dict):
    @property
    def input_ids(self):
        return self["input_ids"]

    def to(self, _device):
        return self


class _Model:
    device = "cpu"

    def generate(self, **kwargs):
        return SimpleNamespace(
            sequences=torch.tensor([[10, 11, 20]]),
            scores=(torch.tensor([[1.0]]),),
            beam_indices=None,
        )


class _Tokenizer:
    pad_token_id = 0

    def __init__(self):
        self.template_kwargs = None

    def apply_chat_template(self, _messages, **kwargs):
        self.template_kwargs = kwargs
        return _Inputs(input_ids=torch.tensor([[10, 11]]))

    def decode(self, _tokens, skip_special_tokens):
        assert skip_special_tokens is True
        return "answer"


class _Processor(_Tokenizer):
    tokenizer = None

    def __init__(self):
        super().__init__()
        self.tokenizer = self

    def apply_chat_template(self, _messages, **kwargs):
        self.template_kwargs = kwargs
        return "rendered prompt"

    def __call__(self, **_kwargs):
        return _Inputs(input_ids=torch.tensor([[10, 11]]))


def _config(enable_thinking=None):
    return SimpleNamespace(
        enable_thinking=enable_thinking,
        max_new_tokens=1,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
    )


def test_inference_config_leaves_thinking_mode_automatic_by_default():
    assert InferenceConfig().enable_thinking is None
    assert InferenceConfig(enable_thinking=False).enable_thinking is False


def test_text_chat_template_receives_explicit_thinking_mode():
    for enabled in (True, False):
        tokenizer = _Tokenizer()
        generate_response(
            _Model(),
            tokenizer,
            config=_config(enabled),
            messages=[{"role": "user", "content": "question"}],
        )

        assert tokenizer.template_kwargs["enable_thinking"] is enabled


def test_vlm_chat_template_receives_explicit_thinking_mode():
    processor = _Processor()
    generate_vlm_response(
        _Model(),
        processor,
        prompt_text="question",
        images=[],
        config=_config(False),
    )

    assert processor.template_kwargs["enable_thinking"] is False


def test_automatic_mode_omits_chat_template_kwarg_for_compatibility():
    tokenizer = _Tokenizer()
    generate_response(
        _Model(),
        tokenizer,
        config=_config(),
        messages=[{"role": "user", "content": "question"}],
    )

    assert "enable_thinking" not in tokenizer.template_kwargs
