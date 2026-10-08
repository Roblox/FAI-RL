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


@pytest.mark.parametrize(
    "overrides, error",
    [({}, None), (BAD_JSON, "not valid JSON"), ({"enable_thinking": True}, "enable_thinking")],
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
    assert config.enable_thinking is False


def test_run_inference_constrains_and_flags_rows_only_with_schema(monkeypatch, tmp_path):
    pytest.importorskip("xgrammar")
    from tokenizers import ByteLevelBPETokenizer
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    bpe = ByteLevelBPETokenizer()
    bpe.train_from_iterator([json.dumps(SCHEMA)], vocab_size=300, special_tokens=["<eos>"])
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=bpe._tokenizer, eos_token="<eos>")
    config = GPT2Config(vocab_size=len(tokenizer), n_embd=16, n_layer=1, n_head=2)
    config.eos_token_id = tokenizer.eos_token_id
    model = GPT2LMHeadModel(config).eval()

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
