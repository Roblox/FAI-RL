"""JSON-schema constrained decoding for local inference (``pip install 'FAI-RL[structured]'``)."""

from __future__ import annotations

import importlib
import json
from typing import Any, Dict, Iterable, List, Optional, Tuple

from transformers import LogitsProcessorList

THINK_END = "</think>"
# json_schema_from_column: max enum size, and min share of non-empty cells that must be JSON.
INFER_MAX_ENUM, INFER_MIN_JSON_SHARE = 20, 0.5


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


def resolve_thinking(tokenizer, enable_thinking: Optional[bool], chat_mode: bool) -> bool:
    """Whether generation thinks: the explicit setting, else the chat template's default."""
    if enable_thinking is not None:
        return enable_thinking
    tokenizer = getattr(tokenizer, "tokenizer", tokenizer)  # VLM processors wrap a tokenizer
    # Flat prompts skip the chat template, so nothing opens a thinking block.
    if not chat_mode or not _has_thinking_mode(tokenizer):
        return False
    turn = [{"role": "user", "content": "x"}]
    default, on = (
        tokenizer.apply_chat_template(turn, tokenize=False, add_generation_prompt=True, **kwargs)
        for kwargs in ({}, {"enable_thinking": True})
    )
    return default == on


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


def _json_object(cell: Any) -> Optional[Dict[str, Any]]:
    if isinstance(cell, dict):
        return cell
    if not isinstance(cell, str):
        return None
    text = cell.strip()
    if text.startswith("```"):  # drop a ```json fence
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _kind(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):  # bool is an int subclass
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    return "array" if isinstance(value, list) else "object"


def _infer(values: List[Any]) -> Dict[str, Any]:
    kinds = [_kind(value) for value in values]
    widen = "number" in kinds  # integers join number when any decimal is present
    by_kind: Dict[str, List[Any]] = {}
    for kind, value in zip(kinds, values):
        by_kind.setdefault("number" if widen and kind == "integer" else kind, []).append(value)
    schemas = []
    for kind, group in by_kind.items():
        schema: Dict[str, Any] = {"type": kind}
        if kind == "string":
            distinct = set(group)
            if len(distinct) <= INFER_MAX_ENUM and len(group) >= 2 * len(distinct):
                schema["enum"] = sorted(distinct)
        elif kind == "array":
            items = [item for array in group for item in array]
            schema["items"] = _infer(items) if items else {}
        elif kind == "object":
            keys = list(dict.fromkeys(key for obj in group for key in obj))
            schema["properties"] = {
                key: _infer([obj[key] for obj in group if key in obj]) for key in keys
            }
            schema["required"] = [key for key in keys if all(key in obj for obj in group)]
            schema["additionalProperties"] = False
        schemas.append(schema)
    return schemas[0] if len(schemas) == 1 else {"anyOf": schemas}


def infer_json_schema(cells: Iterable[Any], column: str) -> Dict[str, Any]:
    """Infer a JSON Schema from the JSON objects in a dataset column (see the README rules)."""
    # The whole column is read so a value that only appears late still joins its enum.
    cells = [c for c in cells if c is not None and not (isinstance(c, str) and not c.strip())]
    objects = [obj for obj in map(_json_object, cells) if obj is not None]
    if not objects or len(objects) < INFER_MIN_JSON_SHARE * len(cells):
        raise ValueError(
            f"json_schema_from_column '{column}': only {len(objects)} of {len(cells)} "
            "non-empty values are JSON objects"
        )
    return _infer(objects)
