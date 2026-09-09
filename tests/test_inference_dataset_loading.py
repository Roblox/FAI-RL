"""Inference/eval dataset loading matches training (local, S3, Hub)."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from datasets import Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.dataset_utils import load_raw_dataset


def _write_question_response_jsonl(path: Path, n: int = 5) -> None:
    rows = [{"question": f"question-{i}", "response": f"response-{i}"} for i in range(n)]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _inference_config(dataset_name, split="test", subset=None):
    return SimpleNamespace(
        dataset_name=dataset_name,
        dataset_split=split,
        dataset_subset=subset,
        s3_region=None,
        s3_endpoint_url=None,
    )


def _load_jsonl_as_dataset(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return Dataset.from_list(rows)


def _patch_load_dataset(monkeypatch, recorder):
    import datasets

    def fake_load_dataset(*args, **kwargs):
        recorder.append((args, kwargs))
        if kwargs.get("data_files"):
            return _load_jsonl_as_dataset(kwargs["data_files"])
        raise AssertionError(f"unexpected Hub-style load_dataset call: args={args} kwargs={kwargs}")

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)


def test_local_jsonl_never_calls_hub(tmp_path, monkeypatch):
    jsonl = tmp_path / "eval.jsonl"
    _write_question_response_jsonl(jsonl, n=5)
    calls = []
    _patch_load_dataset(monkeypatch, calls)

    dataset = load_raw_dataset(_inference_config(str(jsonl)))

    assert len(dataset) == 5
    assert list(dataset["question"]) == [f"question-{i}" for i in range(5)]
    assert list(dataset["response"]) == [f"response-{i}" for i in range(5)]
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0] == "json"
    assert kwargs["data_files"] == str(jsonl)
    assert kwargs["split"] == "train"


def test_relative_jsonl_resolved_from_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rel_dir = Path("eval_data")
    rel_dir.mkdir()
    jsonl = rel_dir / "eval.jsonl"
    _write_question_response_jsonl(jsonl, n=5)
    calls = []
    _patch_load_dataset(monkeypatch, calls)

    dataset = load_raw_dataset(_inference_config("eval_data/eval.jsonl"))

    assert len(dataset) == 5
    assert dataset[0]["question"] == "question-0"
    args, kwargs = calls[0]
    assert args[0] == "json"
    assert Path(kwargs["data_files"]) == (tmp_path / "eval_data" / "eval.jsonl").resolve()


def test_missing_local_jsonl_raises_file_not_found(tmp_path, monkeypatch):
    missing = tmp_path / "does-not-exist.jsonl"
    import datasets

    def boom(*args, **kwargs):
        raise AssertionError("missing local files must not fall through to Hub")

    monkeypatch.setattr(datasets, "load_dataset", boom)

    with pytest.raises(FileNotFoundError, match="Local dataset file not found"):
        load_raw_dataset(_inference_config(str(missing)))


def test_s3_jsonl_is_not_read_as_csv(tmp_path, monkeypatch):
    jsonl = tmp_path / "eval.jsonl"
    _write_question_response_jsonl(jsonl, n=5)

    def fake_download(s3_uri, region=None, endpoint_url=None):
        dest = tmp_path / "downloaded.jsonl"
        dest.write_text(jsonl.read_text())
        return str(dest)

    monkeypatch.setattr("utils.s3_utils.download_file_from_s3", fake_download)

    def boom_read_csv(*args, **kwargs):
        raise AssertionError("S3 JSONL must not be parsed with pandas.read_csv")

    monkeypatch.setattr(pd, "read_csv", boom_read_csv)

    calls = []
    _patch_load_dataset(monkeypatch, calls)

    dataset = load_raw_dataset(_inference_config("s3://bucket/eval.jsonl"))

    assert len(dataset) == 5
    assert list(dataset["question"]) == [f"question-{i}" for i in range(5)]
    args, kwargs = calls[0]
    assert args[0] == "json"
    assert kwargs["split"] == "train"
    assert Path(kwargs["data_files"]).name == "downloaded.jsonl"


def test_hub_id_uses_hub_and_split(monkeypatch):
    import datasets

    recorded = {}

    def fake_load_dataset(*args, **kwargs):
        recorded["args"] = args
        recorded["kwargs"] = kwargs
        return ["hub-row"]

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)

    dataset = load_raw_dataset(
        _inference_config("org/eval-dataset", split="test", subset="abstract_algebra")
    )

    assert dataset == ["hub-row"]
    assert recorded["args"] == ("org/eval-dataset", "abstract_algebra")
    assert recorded["kwargs"]["split"] == "test"


def test_training_dataset_info_name_field_still_loads_jsonl(tmp_path, monkeypatch):
    jsonl = tmp_path / "train.jsonl"
    _write_question_response_jsonl(jsonl, n=5)
    calls = []
    _patch_load_dataset(monkeypatch, calls)

    from core.config import DatasetInfo

    dataset = load_raw_dataset(DatasetInfo(name=str(jsonl), split="train"))

    assert len(dataset) == 5
    assert dataset[0]["question"] == "question-0"
    assert calls[0][0][0] == "json"
