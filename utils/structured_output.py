"""JSON-schema constrained decoding for local inference (``pip install 'FAI-RL[structured]'``)."""

from __future__ import annotations

import importlib
import json
from typing import Any, Dict, Tuple

from transformers import LogitsProcessorList


def _require(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError("json_schema requires: pip install 'FAI-RL[structured]'") from e


def _validator(json_schema: Dict[str, Any]):
    jsonschema = _require("jsonschema")
    return jsonschema.validators.validator_for(json_schema, jsonschema.Draft202012Validator)


def parse_json_schema(json_schema: Any) -> Dict[str, Any]:
    if isinstance(json_schema, str):
        try:
            json_schema = json.loads(json_schema)
        except json.JSONDecodeError as e:
            raise ValueError(f"json_schema is not valid JSON: {e}") from e
    if not isinstance(json_schema, dict):
        raise ValueError("json_schema must be a JSON object")

    jsonschema = _require("jsonschema")
    try:
        _validator(json_schema).check_schema(json_schema)
    except jsonschema.exceptions.SchemaError as e:
        raise ValueError(f"json_schema is not a valid JSON Schema: {e.message}") from e
    return json_schema


def compile_json_schema(model, tokenizer, json_schema: Dict[str, Any]):
    xgr = _require("xgrammar")
    tokenizer = getattr(tokenizer, "tokenizer", tokenizer)  # VLM processors wrap a tokenizer
    output_embeddings = model.get_output_embeddings()
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(
        tokenizer,
        # Logits can be wider than the tokenizer vocabulary.
        vocab_size=output_embeddings.weight.shape[0] if output_embeddings is not None else None,
        # Stop on the same EOS token(s) that HF generate stops on.
        stop_token_ids=getattr(model.generation_config, "eos_token_id", None),
    )
    try:
        # Compact whitespace keeps the model from padding the JSON indefinitely.
        return xgr.GrammarCompiler(tokenizer_info).compile_json_schema(
            json.dumps(json_schema), any_whitespace=False
        )
    except Exception as e:
        raise ValueError(f"json_schema could not be compiled for constrained decoding: {e}") from e


def json_schema_logits_processor(compiled_grammar):
    """xgrammar processors are single-use: build a new one for every generate() call."""
    return LogitsProcessorList([_require("xgrammar").contrib.hf.LogitsProcessor(compiled_grammar)])


def validate_json_response(response: str, json_schema: Dict[str, Any]) -> Tuple[bool, str]:
    try:
        value = json.loads(response)
    except json.JSONDecodeError as e:
        return False, f"invalid JSON: {e}"

    jsonschema = _require("jsonschema")
    try:
        error = jsonschema.exceptions.best_match(
            _validator(json_schema)(json_schema).iter_errors(value)
        )
    except Exception as e:  # e.g. an unresolvable $ref
        return False, f"schema validation failed: {e}"
    if error is None:
        return True, ""
    path = ".".join(str(part) for part in error.absolute_path) or "$"
    return False, f"{path}: {error.message}"
