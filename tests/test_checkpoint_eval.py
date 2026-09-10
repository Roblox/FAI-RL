"""Tests for eval_loss sidecar + job-level index written on checkpoint save."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.s3_utils import S3UploadCallback, eval_metrics_from_trainer_state


def test_eval_metrics_uses_eval_loss_at_matching_step():
    state = SimpleNamespace(
        global_step=32,
        log_history=[
            {"step": 16, "eval_loss": 0.9},
            {"step": 24, "loss": 0.4},
            {"step": 32, "eval_loss": 0.7},
        ],
        best_metric=0.7,
        best_global_step=32,
        best_model_checkpoint="models/run/checkpoint-32",
    )
    metrics = eval_metrics_from_trainer_state(state)
    assert metrics["step"] == 32
    assert metrics["eval_loss"] == 0.7
    assert metrics["best_global_step"] == 32
    assert metrics["best_metric"] == 0.7


def test_eval_metrics_falls_back_to_previous_eval_when_save_is_not_eval():
    state = SimpleNamespace(
        global_step=40,
        log_history=[
            {"step": 16, "eval_loss": 0.9},
            {"step": 40, "loss": 0.3},
        ],
        best_metric=0.9,
        best_global_step=16,
        best_model_checkpoint=None,
    )
    assert eval_metrics_from_trainer_state(state)["eval_loss"] == 0.9


def test_on_save_writes_sidecar_and_index(tmp_path, monkeypatch):
    payloads: list[dict] = []
    monkeypatch.setattr(
        "utils.s3_utils.upload_file_to_s3",
        lambda local_path, bucket, s3_key, **kwargs: payloads.append(
            {"body": json.loads(Path(local_path).read_text()), "bucket": bucket, "key": s3_key}
        ),
    )

    cb = S3UploadCallback(bucket="ml-object-registry", prefix="outputs/run")
    cb._schedule_upload = lambda *args, **kwargs: None  # type: ignore[method-assign]
    args = SimpleNamespace(output_dir=str(tmp_path))
    state = SimpleNamespace(
        global_step=16,
        is_world_process_zero=True,
        log_history=[{"step": 16, "eval_loss": 0.918}],
        best_metric=0.918,
        best_global_step=16,
        best_model_checkpoint="models/run/checkpoint-16",
    )
    cb.on_save(args, state, control=None)

    sidecar = tmp_path / "checkpoint-16" / "eval_metrics.json"
    assert json.loads(sidecar.read_text())["eval_loss"] == 0.918
    assert payloads == [
        {
            "body": {
                "eval_loss": {"16": 0.918},
                "best_global_step": 16,
                "best_metric": 0.918,
            },
            "bucket": "ml-object-registry",
            "key": "outputs/run/checkpoint_eval.json",
        }
    ]
    assert cb._eval_index == {"16": 0.918}
