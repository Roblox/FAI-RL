"""JSON-schema constrained decoding for local inference (``pip install 'FAI-RL[structured]'``)."""

from __future__ import annotations

import importlib
import json
from typing import Any, Dict, Tuple

from transformers import LogitsProcessorList

THINK_END = "</think>"


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


def _has_thinking_mode(tokenizer) -> bool:
    # THINK_END must survive skip_special_tokens decoding so the reasoning can be split off.
    if THINK_END not in tokenizer.get_vocab() or THINK_END in tokenizer.all_special_tokens:
        return False
    if not getattr(tokenizer, "chat_template", None):
        return False
    turn = [{"role": "user", "content": "x"}]
    on, off = (
        tokenizer.apply_chat_template(
            turn, tokenize=False, add_generation_prompt=True, enable_thinking=enabled
        )
        for enabled in (True, False)
    )
    # Thinking-only templates open <think> in the prompt; hybrid ones render differently.
    return on.rstrip().endswith("<think>") or on != off


def compile_json_schema(model, tokenizer, json_schema: Dict[str, Any], thinking: bool = False):
    xgr = _require("xgrammar")
    tokenizer = getattr(tokenizer, "tokenizer", tokenizer)  # VLM processors wrap a tokenizer
    if thinking and not _has_thinking_mode(tokenizer):
        raise ValueError(
            "enable_thinking with json_schema needs a thinking model whose chat template "
            f"uses <think>...{THINK_END}"
        )
    output_embeddings = model.get_output_embeddings()
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(
        tokenizer,
        # Logits can be wider than the tokenizer vocabulary.
        vocab_size=output_embeddings.weight.shape[0] if output_embeddings is not None else None,
        # Stop on the same EOS token(s) that HF generate stops on.
        stop_token_ids=getattr(model.generation_config, "eos_token_id", None),
    )
    try:
        compiler = xgr.GrammarCompiler(tokenizer_info)
        if not thinking:
            # Compact whitespace keeps the model from padding the JSON indefinitely.
            return compiler.compile_json_schema(json.dumps(json_schema), any_whitespace=False)
        from xgrammar.structural_tag import AnyTextFormat, JSONSchemaFormat, SequenceFormat
        from xgrammar.structural_tag import TagFormat

        # Free text up to THINK_END (no token budget), then only schema-valid JSON.
        think = TagFormat(begin="", content=AnyTextFormat(), end=THINK_END)
        answer = JSONSchemaFormat(json_schema=json_schema, max_whitespace_cnt=1)
        return compiler.compile_structural_tag(
            xgr.StructuralTag(format=SequenceFormat(elements=[think, answer]))
        )
    except Exception as e:
        raise ValueError(f"json_schema could not be compiled for constrained decoding: {e}") from e


def json_schema_logits_processor(compiled_grammar):
    """xgrammar processors are single-use: build a new one for every generate() call."""
    return LogitsProcessorList([_require("xgrammar").contrib.hf.LogitsProcessor(compiled_grammar)])


def split_reasoning(response: str) -> Tuple[str, str]:
    """Return ``(reasoning, answer)``; the answer is empty if thinking never closed."""
    reasoning, _, answer = response.partition(THINK_END)
    return reasoning.strip().removeprefix("<think>").strip(), answer.strip()


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
