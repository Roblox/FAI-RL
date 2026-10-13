"""Confidence scores for locally generated inference responses."""

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import inference.inference as inference_module
from core.config import InferenceConfig
from inference.inference import generate_response


class _Inputs(dict):
    @property
    def input_ids(self):
        return self["input_ids"]

    def to(self, _device):
        return self


class _Tokenizer:
    pad_token_id = 0

    def __call__(self, _prompt, return_tensors):
        assert return_tensors == "pt"
        return _Inputs(input_ids=torch.tensor([[10, 11]]))

    def decode(self, tokens, skip_special_tokens):
        assert skip_special_tokens is True
        assert tokens.tolist() == [20, 21]
        return "answer"


class _Model:
    device = "cpu"

    def __init__(self):
        self.generate_kwargs = None

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return SimpleNamespace(
            sequences=torch.tensor([[10, 11, 20, 21]]),
            scores=(torch.tensor([[1.0]]), torch.tensor([[1.0]])),
            beam_indices=None,
        )

    def compute_transition_scores(self, sequences, scores, beam_indices, normalize_logits):
        assert sequences.tolist() == [[10, 11, 20, 21]]
        assert len(scores) == 2
        assert beam_indices is None
        assert normalize_logits is True
        return torch.tensor([[math.log(0.9), math.log(0.8)]])


def test_generate_response_returns_geometric_mean_token_confidence():
    model = _Model()
    config = SimpleNamespace(max_new_tokens=2, do_sample=False, temperature=1.0, top_p=1.0)

    response, confidence = generate_response(
        model,
        _Tokenizer(),
        prompt="question",
        config=config,
        include_confidence=True,
    )

    assert response == "answer"
    assert confidence == pytest.approx(math.sqrt(0.9 * 0.8))
    assert model.generate_kwargs["return_dict_in_generate"] is True
    assert model.generate_kwargs["output_scores"] is True


def test_generate_response_keeps_string_only_backwards_compatibility():
    response = generate_response(
        _Model(),
        _Tokenizer(),
        prompt="question",
        config=SimpleNamespace(max_new_tokens=2, do_sample=False, temperature=1.0, top_p=1.0),
    )

    assert response == "answer"


def test_run_inference_writes_confidence_to_csv(monkeypatch, tmp_path):
    output_file = tmp_path / "results.csv"
    config = InferenceConfig(
        model_paths=["checkpoint-100"],
        dataset_name="unused",
        dataset_columns=["question"],
        system_prompt="{question}",
        output_file=str(output_file),
    )
    monkeypatch.setattr(inference_module, "load_raw_dataset", lambda _config: [{"question": "Why?"}])
    monkeypatch.setattr(
        inference_module,
        "load_model_and_tokenizer",
        lambda _config: (object(), object()),
    )
    monkeypatch.setattr(
        inference_module,
        "generate_response",
        lambda *_args, **_kwargs: ("Because.", 0.75),
    )

    inference_module.run_inference(config)

    result = pd.read_csv(output_file)
    assert result.to_dict(orient="records") == [
        {"question": "Why?", "response": "Because.", "confidence": 0.75}
    ]
