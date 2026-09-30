"""Data-parallel inference launches and merges rank-sharded results."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
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
    # The real torchrun workers synchronize here. Running rank 1 followed by
    # rank 0 in one test process makes both temporary files available directly.
    monkeypatch.setattr(inference_module, "_distributed_barrier", lambda: None)
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
