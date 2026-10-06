"""Data-parallel inference launches and merges rank-sharded results."""

import datetime
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import inference.inference as inference_module
from core.config import InferenceConfig


def test_parse_args_accepts_num_gpus(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["fai-rl-inference", "--recipe", "recipe.yaml", "--num-gpus", "8"],
    )

    args = inference_module.parse_args()

    assert args.num_gpus == 8


def test_multi_gpu_launcher_uses_one_torchrun_process_per_gpu(monkeypatch):
    recorded = {}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(
        inference_module.subprocess,
        "call",
        lambda cmd: recorded.setdefault("cmd", cmd) or 0,
    )
    args = SimpleNamespace(
        num_gpus=4,
        recipe="recipe.yaml",
        debug=False,
        nohup=False,
        overrides=["inference.max_new_tokens=32"],
    )

    inference_module._launch_inference_workers(args)

    assert recorded["cmd"][:3] == [
        "torchrun",
        "--standalone",
        "--nproc_per_node=4",
    ]
    assert recorded["cmd"][-3:] == [
        "--recipe",
        "recipe.yaml",
        "inference.max_new_tokens=32",
    ]


def test_distributed_model_replica_is_pinned_to_local_rank(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("LOCAL_RANK", "2")
    monkeypatch.setattr(inference_module, "get_device_type", lambda: "cuda")
    monkeypatch.setattr(inference_module, "get_optimal_dtype", lambda: torch.bfloat16)
    monkeypatch.setattr(
        inference_module,
        "resolve_transformers_attn_implementation",
        lambda _use_flash: None,
    )

    kwargs = inference_module._build_model_load_kwargs()

    assert kwargs["device_map"] == {"": 2}


def test_distributed_process_group_uses_one_hour_timeout(monkeypatch):
    recorded = {}
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.delenv("FAI_RL_DISTRIBUTED_TIMEOUT_SECONDS", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(
        torch.distributed,
        "init_process_group",
        lambda **kwargs: recorded.update(kwargs),
    )

    inference_module._initialize_distributed_inference()

    assert recorded == {
        "backend": "gloo",
        "timeout": datetime.timedelta(hours=1),
    }


def test_rank_zero_waits_for_each_rank_completion_marker(monkeypatch, tmp_path):
    output_file = tmp_path / "results.csv"
    inference_module._publish_rank_results(
        str(output_file),
        rank=0,
        world_size=2,
        results=[{"value": "rank-0"}],
    )

    def publish_slow_rank(_seconds):
        inference_module._publish_rank_results(
            str(output_file),
            rank=1,
            world_size=2,
            results=[{"value": "rank-1"}],
        )

    monkeypatch.setattr(inference_module.time, "sleep", publish_slow_rank)

    rank_files, completion_files = inference_module._wait_for_rank_results(
        str(output_file),
        world_size=2,
        timeout_seconds=5,
        poll_interval_seconds=0.01,
    )

    assert [pd.read_pickle(path)["value"].item() for path in rank_files] == [
        "rank-0",
        "rank-1",
    ]
    assert all(Path(path).exists() for path in completion_files)


def test_rank_result_wait_times_out_with_missing_ranks(tmp_path):
    output_file = tmp_path / "results.csv"
    inference_module._publish_rank_results(
        str(output_file),
        rank=0,
        world_size=2,
        results=[],
    )

    with pytest.raises(TimeoutError, match=r"rank\(s\): 1"):
        inference_module._wait_for_rank_results(
            str(output_file),
            world_size=2,
            timeout_seconds=0,
        )


def test_distributed_ranks_process_disjoint_rows_and_rank_zero_merges(
    monkeypatch, tmp_path
):
    output_file = tmp_path / "results.csv"
    rows = [{"question": f"question-{i}"} for i in range(6)]
    config = InferenceConfig(
        model_paths=["checkpoint-100"],
        dataset_name="unused",
        dataset_columns=["question"],
        system_prompt="{question}",
        output_file=str(output_file),
    )

    monkeypatch.setattr(inference_module, "load_raw_dataset", lambda _config: rows)
    monkeypatch.setattr(
        inference_module,
        "load_model_and_tokenizer",
        lambda _config: (object(), object()),
    )
    monkeypatch.setattr(
        inference_module,
        "generate_response",
        lambda _model, _tokenizer, prompt, _config, **_kwargs: (
            f"answer-{prompt}",
            0.5,
        ),
    )
    # Run rank 1 followed by rank 0 in one process. Rank 0 must consume rank 1's
    # completion marker without requiring a torch.distributed collective.
    monkeypatch.setenv("WORLD_SIZE", "2")

    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")
    inference_module.run_inference(config)
    assert not output_file.exists()

    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    inference_module.run_inference(config)

    result = pd.read_csv(output_file)
    assert result["question"].tolist() == [f"question-{i}" for i in range(6)]
    assert result["response"].tolist() == [
        f"answer-question-{i}" for i in range(6)
    ]
    assert not list(tmp_path.glob("*.pkl"))
    assert not list(tmp_path.glob("*.complete"))
