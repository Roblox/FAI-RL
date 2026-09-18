"""Checkpoint discovery never imports torch or unpickles checkpoint files."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

from utils.checkpoint_utils import find_latest_checkpoint, select_resume_checkpoint


def checkpoint(root, step, weights="model.safetensors", full=True):
    path = root / f"checkpoint-{step}"
    path.mkdir()
    (path / "trainer_state.json").write_text(json.dumps({"global_step": step}))
    (path / weights).write_bytes(b"test")
    if full:
        (path / "optimizer.pt").write_bytes(b"test")
        (path / "scheduler.pt").write_bytes(b"test")
    return path


def test_newest_valid_step_skips_partial_and_model_only(tmp_path):
    checkpoint(tmp_path, 2)
    latest = checkpoint(tmp_path, 10, "adapter_model.safetensors")
    checkpoint(tmp_path, 20, full=False)
    broken = checkpoint(tmp_path, 30)
    (broken / "trainer_state.json").write_text("{")
    assert find_latest_checkpoint(str(tmp_path)) == str(latest)


def test_sharded_weights_require_all_shards(tmp_path):
    path = checkpoint(tmp_path, 10, "shard-1.safetensors")
    index = path / "model.safetensors.index.json"
    index.write_text(
        json.dumps({"weight_map": {"a": "shard-1.safetensors", "b": "shard-2.safetensors"}})
    )
    assert find_latest_checkpoint(str(tmp_path)) is None
    (path / "shard-2.safetensors").write_bytes(b"test")
    assert find_latest_checkpoint(str(tmp_path)) == str(path)


def test_deepspeed_layout(tmp_path):
    path = checkpoint(tmp_path, 10, full=False)
    (path / "latest").write_text("global_step10")
    ds = path / "global_step10"
    ds.mkdir()
    (ds / "mp_rank_00_model_states.pt").write_bytes(b"test")
    assert find_latest_checkpoint(str(tmp_path)) is None
    (ds / "zero_pp_rank_0_mp_rank_00_optim_states.pt").write_bytes(b"test")
    assert find_latest_checkpoint(str(tmp_path)) == str(path)


def test_explicit_resume_and_default_are_unchanged(tmp_path):
    latest = checkpoint(tmp_path, 1)
    config = SimpleNamespace(output_dir=str(tmp_path), resume_from_checkpoint="explicit")
    select_resume_checkpoint(config, True, Mock())
    assert config.resume_from_checkpoint == "explicit"
    config.resume_from_checkpoint = None
    select_resume_checkpoint(config, False, Mock())
    assert config.resume_from_checkpoint is None
    select_resume_checkpoint(config, True, Mock())
    assert config.resume_from_checkpoint == str(latest)


def test_absent_output_and_zero_length_weights(tmp_path):
    assert find_latest_checkpoint(str(tmp_path / "absent")) is None
    path = checkpoint(tmp_path, 1)
    (path / "model.safetensors").write_bytes(b"")
    assert find_latest_checkpoint(str(tmp_path)) is None


def test_auto_resume_forwarded_by_launcher(monkeypatch):
    from trainers import train

    call = Mock(return_value=0)
    monkeypatch.setattr(train.subprocess, "call", call)
    args = SimpleNamespace(
        recipe="recipe.yaml",
        overrides=["training.output_dir=out"],
        num_gpus=1,
        nohup=False,
        auto_resume=True,
    )
    assert train.launch_distributed_training(args) == 0
    assert "--auto-resume" in call.call_args.args[0]
    assert "training.output_dir=out" in call.call_args.args[0]


def test_train_dispatch_preserves_default_and_explicit_resume():
    from core.trainer_base import BaseTrainer

    owner = SimpleNamespace(
        config=SimpleNamespace(training=SimpleNamespace(resume_from_checkpoint=None)),
        trainer=SimpleNamespace(train=Mock()),
    )
    BaseTrainer.train_with_resume(owner)
    owner.trainer.train.assert_called_once_with()
    owner.trainer.train.reset_mock()
    owner.config.training.resume_from_checkpoint = "checkpoint-10"
    BaseTrainer.train_with_resume(owner)
    owner.trainer.train.assert_called_once_with(resume_from_checkpoint="checkpoint-10")
