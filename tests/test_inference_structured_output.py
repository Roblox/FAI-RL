"""JSON-schema constrained decoding for local inference."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import inference.inference as inference_module
from core.config import InferenceConfig
from inference.inference import (
    generate_response,
    generate_vlm_response,
    load_inference_recipe_with_overrides,
)

SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": ["a", "b", "c"]},
        "decision": {"type": "string", "enum": ["allow", "block"]},
    },
    "required": ["label", "decision"],
    "additionalProperties": False,
}


def _load_config(tmp_path, **inference_overrides):
    recipe = {
        "inference": {
            "model_paths": ["checkpoint-100"],
            "dataset_columns": ["question"],
            "system_prompt": "{question}",
            **inference_overrides,
        }
    }
    recipe_path = tmp_path / "recipe.yaml"
    recipe_path.write_text(yaml.safe_dump(recipe, sort_keys=False))
    return load_inference_recipe_with_overrides(
        SimpleNamespace(recipe=str(recipe_path), overrides=[])
    )


@pytest.mark.parametrize("json_schema", [SCHEMA, json.dumps(SCHEMA)], ids=["mapping", "string"])
def test_load_inference_config_normalizes_json_schema(tmp_path, json_schema):
    pytest.importorskip("jsonschema")
    assert InferenceConfig().json_schema is None
    loaded = _load_config(tmp_path, json_schema=json_schema).json_schema
    assert loaded == SCHEMA
    # xgrammar emits keys in properties order, so recipe loading must preserve it.
    assert list(loaded["properties"]) == ["label", "decision"]


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"json_schema": "{not json"}, "not valid JSON"),
        ({"json_schema": {"type": "bogus"}}, "not a valid JSON Schema"),
        ({"json_schema": SCHEMA, "enable_thinking": True}, "enable_thinking"),
        (
            {"json_schema": SCHEMA, "model": "gpt-4o", "api_key": "secret"},
            "local model",
        ),
    ],
)
def test_load_inference_config_rejects_invalid_structured_output(tmp_path, overrides, message):
    pytest.importorskip("jsonschema")
    with pytest.raises(ValueError, match=message):
        _load_config(tmp_path, **overrides)


@pytest.fixture(scope="module")
def tiny_model_and_tokenizer():
    """A hermetic, randomly initialized GPT-2 with a byte-level BPE tokenizer."""
    pytest.importorskip("xgrammar")
    from tokenizers import ByteLevelBPETokenizer
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    bpe = ByteLevelBPETokenizer()
    bpe.train_from_iterator(
        ['{"decision": "allow", "label": "a"}', '{"decision": "block", "label": "c"}'],
        vocab_size=300,
        min_frequency=1,
        special_tokens=["<|endoftext|>"],
        show_progress=False,
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=bpe._tokenizer,
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
    )
    torch.manual_seed(0)
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(tokenizer),
            n_positions=128,
            n_embd=16,
            n_layer=1,
            n_head=2,
            bos_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    ).eval()
    return model, tokenizer


def test_constrained_decoding_on_tiny_model_always_matches_schema(tiny_model_and_tokenizer):
    jsonschema = pytest.importorskip("jsonschema")
    from utils.structured_output import compile_json_schema, json_schema_logits_processor

    model, tokenizer = tiny_model_and_tokenizer
    compiled = compile_json_schema(model, tokenizer, SCHEMA)
    config = SimpleNamespace(max_new_tokens=64, do_sample=True, temperature=1.0, top_p=1.0)

    for seed in range(5):
        torch.manual_seed(seed)
        response = generate_response(
            model,
            tokenizer,
            prompt="Moderate this ad:",
            config=config,
            logits_processor=json_schema_logits_processor(compiled),
        )
        jsonschema.validate(json.loads(response), SCHEMA)

    with pytest.raises(ValueError, match="could not be compiled"):
        compile_json_schema(model, tokenizer, {"$ref": "#/$defs/missing"})


class _Inputs(dict):
    @property
    def input_ids(self):
        return self["input_ids"]

    def to(self, _device):
        return self


class _Model:
    device = "cpu"

    def __init__(self):
        self.generate_kwargs = None

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return SimpleNamespace(
            sequences=torch.tensor([[10, 11, 20]]),
            scores=(torch.tensor([[1.0]]),),
            beam_indices=None,
        )


class _Processor:
    pad_token_id = 0

    def __init__(self):
        self.tokenizer = self

    def apply_chat_template(self, _messages, **_kwargs):
        return "rendered prompt"

    def __call__(self, **_kwargs):
        return _Inputs(input_ids=torch.tensor([[10, 11]]))

    def decode(self, _tokens, skip_special_tokens):
        return "answer"


def test_vlm_response_forwards_logits_processor_only_when_set():
    config = SimpleNamespace(max_new_tokens=1, do_sample=False, temperature=1.0, top_p=1.0)
    processor_sentinel = object()

    model = _Model()
    generate_vlm_response(
        model, _Processor(), "question", [], config, logits_processor=processor_sentinel
    )
    assert list(model.generate_kwargs["logits_processor"]) == [processor_sentinel]

    model = _Model()
    generate_vlm_response(model, _Processor(), "question", [], config)
    assert "logits_processor" not in model.generate_kwargs


def _run_inference(monkeypatch, tmp_path, responses, json_schema=None):
    output_file = tmp_path / "results.csv"
    config = InferenceConfig(
        model_paths=["checkpoint-100"],
        dataset_name="unused",
        dataset_columns=["question"],
        system_prompt="{question}",
        output_file=str(output_file),
        json_schema=json_schema,
    )
    rows = [{"question": f"q{i}"} for i in range(len(responses))]
    monkeypatch.setattr(inference_module, "load_raw_dataset", lambda _config: rows)
    monkeypatch.setattr(
        inference_module,
        "load_model_and_tokenizer",
        lambda _config: (object(), object()),
    )
    monkeypatch.setattr(inference_module, "compile_json_schema", lambda *_args: "compiled")
    monkeypatch.setattr(inference_module, "json_schema_logits_processor", lambda _compiled: "lp")
    seen_logits_processors = []
    pending = iter(responses)

    def fake_generate_response(*_args, logits_processor=None, **_kwargs):
        seen_logits_processors.append(logits_processor)
        response = next(pending)
        if isinstance(response, Exception):
            raise response
        return response, 0.5

    monkeypatch.setattr(inference_module, "generate_response", fake_generate_response)

    inference_module.run_inference(config)

    summary_file = str(output_file).replace(".csv", "_summary.json")
    with open(summary_file, encoding="utf-8") as f:
        summary = json.load(f)
    return pd.read_csv(output_file, keep_default_na=False), summary, seen_logits_processors


def test_run_inference_flags_schema_results_and_keeps_failed_rows(monkeypatch, tmp_path):
    result, summary, seen = _run_inference(
        monkeypatch,
        tmp_path,
        [
            '{"decision": "allow", "label": "a"}',
            '{"decision": "allow", "label": ',
            '{"decision": "maybe", "label": "a"}',
            RuntimeError("CUDA OOM"),
        ],
        json_schema=SCHEMA,
    )

    assert seen == ["lp"] * 4
    assert result["question"].tolist() == ["q0", "q1", "q2", "q3"]
    assert result["parse_ok"].tolist() == [True, False, False, False]
    errors = result["schema_error"].tolist()
    assert errors[0] == ""
    assert errors[1].startswith("invalid JSON:")
    assert errors[2].startswith("decision:") and "maybe" in errors[2]
    assert errors[3] == "generation failed: RuntimeError: CUDA OOM"
    assert result["response"].tolist()[3] == ""
    assert summary["successful_examples"] == 3
    assert summary["failed_examples"] == 1
    assert summary["schema_valid_examples"] == 1
    assert summary["schema_invalid_examples"] == 3


def test_run_inference_without_schema_is_unchanged(monkeypatch, tmp_path):
    result, summary, seen = _run_inference(
        monkeypatch,
        tmp_path,
        ["Because.", RuntimeError("CUDA OOM")],
    )

    assert seen == [None, None]
    assert result.to_dict(orient="records") == [
        {"question": "q0", "response": "Because.", "confidence": 0.5}
    ]
    assert summary["failed_examples"] == 1
    assert "schema_valid_examples" not in summary
