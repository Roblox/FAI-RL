"""JSON-schema constrained decoding helpers for local inference.

xgrammar and jsonschema are optional dependencies (``pip install 'FAI-RL[structured]'``)
and are imported lazily, so inference without ``json_schema`` never needs them.
"""

from __future__ import annotations

import importlib
import json
from typing import Any, Dict, Optional, Tuple

_INSTALL_HINT = "json_schema requires the structured extra: pip install 'FAI-RL[structured]'"

GENERATION_FAILED_PREFIX = "generation failed: "


def _require(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(_INSTALL_HINT) from e


def parse_json_schema(json_schema: Any) -> Dict[str, Any]:
    """Return ``json_schema`` (a mapping or JSON string) as a validated dict."""
    if isinstance(json_schema, str):
        try:
            json_schema = json.loads(json_schema)
        except json.JSONDecodeError as e:
            raise ValueError(f"json_schema is not valid JSON: {e}") from e
    if not isinstance(json_schema, dict):
        raise ValueError("json_schema must be a JSON object (a mapping or a JSON string)")

    jsonschema = _require("jsonschema")
    try:
        jsonschema.Draft202012Validator.check_schema(json_schema)
    except jsonschema.exceptions.SchemaError as e:
        raise ValueError(f"json_schema is not a valid JSON Schema: {e.message}") from e
    return json_schema


def _output_vocab_size(model) -> Optional[int]:
    """Return the logits width, which can exceed the tokenizer's vocabulary."""
    output_embeddings = model.get_output_embeddings()
    if output_embeddings is not None and hasattr(output_embeddings, "weight"):
        return int(output_embeddings.weight.shape[0])
    return getattr(model.config.get_text_config(), "vocab_size", None)


def compile_json_schema(model, tokenizer, json_schema: Dict[str, Any]):
    """Compile ``json_schema`` against a model's tokenizer for constrained decoding.

    ``tokenizer`` may be a VLM processor. The grammar terminates on the model's own
    EOS token(s) so generation stops exactly when the JSON value is complete.
    """
    xgr = _require("xgrammar")
    tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    generation_config = getattr(model, "generation_config", None)
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(
        tokenizer,
        vocab_size=_output_vocab_size(model),
        stop_token_ids=getattr(generation_config, "eos_token_id", None),
    )
    try:
        # Compact whitespace keeps the model from padding the JSON indefinitely.
        return xgr.GrammarCompiler(tokenizer_info).compile_json_schema(
            json.dumps(json_schema), any_whitespace=False
        )
    except Exception as e:
        raise ValueError(f"json_schema could not be compiled for constrained decoding: {e}") from e


def json_schema_logits_processor(compiled_grammar):
    """Return a fresh logits processor; xgrammar processors are single-use per generate()."""
    xgr = _require("xgrammar")
    return xgr.contrib.hf.LogitsProcessor(compiled_grammar)


def validate_json_response(response: str, json_schema: Dict[str, Any]) -> Tuple[bool, str]:
    """Return ``(parse_ok, schema_error)`` for one generated response."""
    try:
        value = json.loads(response)
    except json.JSONDecodeError as e:
        return False, f"invalid JSON: {e}"

    jsonschema = _require("jsonschema")
    error = jsonschema.exceptions.best_match(
        jsonschema.Draft202012Validator(json_schema).iter_errors(value)
    )
    if error is None:
        return True, ""
    path = ".".join(str(part) for part in error.absolute_path) or "$"
    return False, f"{path}: {error.message}"
