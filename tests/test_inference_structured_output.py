"""JSON-schema constrained decoding for local inference."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

pytest.importorskip("jsonschema")

import inference.inference as inference_module
from core.config import InferenceConfig
from utils.structured_output import validate_json_response

# Not alphabetical, so recipe loading must preserve key order (xgrammar emits keys in order).
SCHEMA = {
    "type": "object",
    "properties": {"label": {"enum": ["a", "b"]}, "decision": {"enum": ["allow", "block"]}},
    "required": ["label", "decision"],
    "additionalProperties": False,
}
BAD_JSON = {"json_schema": "{not json"}
THINK = {"enable_thinking": True}
THINKING_TEMPLATE = "{{ messages[-1]['content'] }}{% if enable_thinking %}<think>{% endif %}"
# Qwen3-style: thinks unless enable_thinking is false.
THINKING_BY_DEFAULT_TEMPLATE = (
    "{{ messages[-1]['content'] }}"
    "{% if enable_thinking is defined and enable_thinking is false %}<think></think>"
    "{% else %}<think>{% endif %}"
)


def _tiny_model(chat_template=None):
    from tokenizers import ByteLevelBPETokenizer
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    bpe = ByteLevelBPETokenizer()
    bpe.train_from_iterator([json.dumps(SCHEMA)], vocab_size=300, special_tokens=["<eos>"])
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=bpe._tokenizer, eos_token="<eos>")
    tokenizer.add_tokens(["<think>", "</think>"])
    tokenizer.chat_template = chat_template
    config = GPT2Config(vocab_size=len(tokenizer), n_embd=16, n_layer=1, n_head=2)
    config.eos_token_id = tokenizer.eos_token_id
    return GPT2LMHeadModel(config).eval(), tokenizer


@pytest.mark.parametrize(
    "overrides, error",
    [
        ({}, None),
        (BAD_JSON, "not valid JSON"),
        ({**THINK, "user_prompt": "{q}"}, None),
        (THINK, "user_prompt"),
    ],
)
def test_recipe_json_schema_validation(tmp_path, overrides, error):
    recipe = tmp_path / "recipe.yaml"
    inference = {"json_schema": SCHEMA, **overrides}
    recipe.write_text(yaml.safe_dump({"inference": inference}, sort_keys=False))
    args = SimpleNamespace(recipe=str(recipe), overrides=[])
    if error:
        with pytest.raises(ValueError, match=error):
            inference_module.load_inference_recipe_with_overrides(args)
        return
    config = inference_module.load_inference_recipe_with_overrides(args)
    assert list(config.json_schema["properties"]) == ["label", "decision"]
    # Unset stays unset, so the chat template's default applies as it does without a schema.
    assert config.enable_thinking is overrides.get("enable_thinking")


@pytest.mark.parametrize(
    "template, enable_thinking, chat_mode, expected",
    [
        (THINKING_BY_DEFAULT_TEMPLATE, None, True, True),
        (THINKING_BY_DEFAULT_TEMPLATE, False, True, False),
        (THINKING_BY_DEFAULT_TEMPLATE, None, False, False),
        (THINKING_TEMPLATE, None, True, False),
        (THINKING_TEMPLATE, True, True, True),
        ("{{ messages[-1]['content'] }}", None, True, False),
    ],
)
def test_resolve_thinking_follows_the_chat_template_default(template, enable_thinking, chat_mode, expected):
    from utils.structured_output import resolve_thinking

    _, tokenizer = _tiny_model(template)
    assert resolve_thinking(tokenizer, enable_thinking, chat_mode) is expected


def test_run_inference_constrains_and_flags_rows_only_with_schema(monkeypatch, tmp_path):
    pytest.importorskip("xgrammar")
    model, tokenizer = _tiny_model()

    canned = {
        "cut": '{"label": "a", "decision": ',
        "bad": '{"label": "a", "decision": "maybe"}',
        "oom": RuntimeError("CUDA OOM"),
    }
    real_generate, seen = inference_module.generate_response, []

    def generate(model, tokenizer, prompt, *args, **kwargs):
        seen.append(kwargs.get("logits_processor") is not None)
        if prompt not in canned:  # sampled from the tiny model
            return real_generate(model, tokenizer, prompt, *args, **kwargs)
        if isinstance(canned[prompt], Exception):
            raise canned[prompt]
        return canned[prompt], 0.5

    monkeypatch.chdir(tmp_path)
    rows = [{"prompt": prompt} for prompt in ["x", "y", "z", *canned]]
    monkeypatch.setattr(inference_module, "load_raw_dataset", lambda _c: rows)
    monkeypatch.setattr(inference_module, "load_model_and_tokenizer", lambda _c: (model, tokenizer))
    monkeypatch.setattr(inference_module, "generate_response", generate)

    def run(schema):
        seen.clear()
        kwargs = {"model_paths": ["c"], "system_prompt": "{prompt}", "output_file": "r.csv"}
        inference_module.run_inference(InferenceConfig(json_schema=schema, **kwargs))
        summary = json.loads(Path("r_summary.json").read_text())
        return pd.read_csv("r.csv", keep_default_na=False), summary

    # A JSON string schema is parsed when the config is built.
    result, summary = run(json.dumps(SCHEMA))
    assert seen == [True] * 6
    assert result["parse_ok"].tolist() == [True] * 3 + [False] * 3
    prefixes = [error.split(":")[0] for error in result["schema_error"]]
    assert prefixes == ["", "", "", "invalid JSON", "decision", "generation failed"]
    assert (summary["successful_examples"], summary["failed_examples"]) == (3, 3)

    result, _ = run(None)
    assert seen == [False] * 6 and "parse_ok" not in result and len(result) == 5

    # An unresolvable $ref fails the row instead of aborting the run.
    assert not validate_json_response("1", {"$ref": "https://example.com/x.json"})[0]


@pytest.mark.parametrize(
    "template, overrides",
    [(THINKING_TEMPLATE, THINK), (THINKING_BY_DEFAULT_TEMPLATE, {})],
    ids=["explicit", "template-default"],
)
def test_run_inference_with_thinking_validates_only_the_json(monkeypatch, tmp_path, template, overrides):
    pytest.importorskip("xgrammar")
    from utils.structured_output import compile_json_schema

    model, tokenizer = _tiny_model(template)
    answer = '{"label": "a", "decision": "allow"}'
    canned = {"ok": "<think>hmm</think>" + answer, "long": "never stops thinking"}

    def generate(*_args, messages, **_kwargs):
        return canned[messages[-1]["content"]], 0.5

    monkeypatch.chdir(tmp_path)
    rows = [{"prompt": prompt} for prompt in canned]
    monkeypatch.setattr(inference_module, "load_raw_dataset", lambda _c: rows)
    monkeypatch.setattr(inference_module, "load_model_and_tokenizer", lambda _c: (model, tokenizer))
    monkeypatch.setattr(inference_module, "generate_response", generate)
    kwargs = {"model_paths": ["c"], "user_prompt": "{prompt}", "output_file": "r.csv"}
    inference_module.run_inference(InferenceConfig(json_schema=SCHEMA, **overrides, **kwargs))

    result = pd.read_csv("r.csv", keep_default_na=False)
    assert result["reasoning"].tolist() == ["hmm", "never stops thinking"]
    assert result["response"].tolist() == [answer, ""]
    assert result["parse_ok"].tolist() == [True, False]

    # The compiled grammar allows free text, then </think>, then only schema-valid JSON.
    compiled = compile_json_schema(model, tokenizer, SCHEMA, thinking=True)
    import xgrammar as xgr

    assert xgr.GrammarMatcher(compiled).accept_string("any text</think>" + answer)
    assert not xgr.GrammarMatcher(compiled).accept_string('any text</think>{"label": "z"')

    # A model whose chat template has no thinking mode fails at startup.
    _, plain = _tiny_model("{{ messages[-1]['content'] }}")
    with pytest.raises(ValueError, match="thinking model"):
        compile_json_schema(model, plain, SCHEMA, thinking=True)
