"""Tests for ZeRO-3 partitioned-load activation in BaseTrainer.

The crux of the fix is that ``from_pretrained`` must see ``is_deepspeed_zero3_enabled()``
== True so it loads the model sharded per rank instead of fully on every rank
(which OOMs the host on large models). That end-to-end behavior requires a GPU +
deepspeed and is exercised on the cluster; here we lock down the pure decision
logic that gates it: ``BaseTrainer._deepspeed_zero_stage``.
"""

import json
from types import SimpleNamespace

from core.trainer_base import BaseTrainer


def _fake_trainer(micro, accum):
    """Minimal stand-in carrying just the recipe fields _resolve_ds_auto_batch reads."""
    return SimpleNamespace(
        config=SimpleNamespace(
            training=SimpleNamespace(
                per_device_train_batch_size=micro,
                gradient_accumulation_steps=accum,
            )
        )
    )


def test_stage_from_zero3_path(tmp_path):
    cfg = tmp_path / "zero3.json"
    cfg.write_text(json.dumps({"zero_optimization": {"stage": 3}}))
    assert BaseTrainer._deepspeed_zero_stage(str(cfg)) == 3


def test_stage_from_zero1_path(tmp_path):
    cfg = tmp_path / "zero1.json"
    cfg.write_text(json.dumps({"zero_optimization": {"stage": 1}}))
    assert BaseTrainer._deepspeed_zero_stage(str(cfg)) == 1


def test_stage_from_dict():
    assert BaseTrainer._deepspeed_zero_stage({"zero_optimization": {"stage": 3}}) == 3


def test_none_when_unset():
    assert BaseTrainer._deepspeed_zero_stage(None) is None
    assert BaseTrainer._deepspeed_zero_stage("") is None


def test_none_when_no_zero_block():
    assert BaseTrainer._deepspeed_zero_stage({"train_batch_size": "auto"}) is None


def test_none_when_file_missing():
    assert BaseTrainer._deepspeed_zero_stage("/nonexistent/zero3.json") is None


def test_none_when_unparseable(tmp_path):
    cfg = tmp_path / "broken.json"
    cfg.write_text("{not json")
    assert BaseTrainer._deepspeed_zero_stage(str(cfg)) is None


def test_real_repo_zero3_config_is_stage_3():
    """The shipped ZeRO-3 config the platform points VLM LoRA at must be detected."""
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    cfg = repo_root / "configs" / "deepspeed" / "zero3_config.json"
    assert BaseTrainer._deepspeed_zero_stage(str(cfg)) == 3


def test_resolve_auto_batch_from_dict(monkeypatch):
    """'auto' micro/accum become ints; train_batch_size is left for DeepSpeed."""
    monkeypatch.setenv("WORLD_SIZE", "2")
    trainer = _fake_trainer(micro=1, accum=8)
    cfg = BaseTrainer._resolve_ds_auto_batch(
        trainer,
        {
            "train_batch_size": "auto",
            "train_micro_batch_size_per_gpu": "auto",
            "gradient_accumulation_steps": "auto",
        },
    )
    assert cfg["train_micro_batch_size_per_gpu"] == 1
    assert cfg["gradient_accumulation_steps"] == 8
    # train_batch_size must NOT be pinned to a world-size-scaled value: at zero.Init
    # DeepSpeed sees world_size==1 and would reject 16 != 1*8*1. Leaving it None lets
    # DeepSpeed derive a self-consistent value from whatever world size it sees.
    assert cfg["train_batch_size"] is None


def test_resolve_auto_batch_leaves_train_batch_unset_regardless_of_world_size(monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    trainer = _fake_trainer(micro=2, accum=4)
    cfg = BaseTrainer._resolve_ds_auto_batch(trainer, {"train_batch_size": "auto"})
    assert cfg["train_micro_batch_size_per_gpu"] == 2
    assert cfg["gradient_accumulation_steps"] == 4
    assert cfg["train_batch_size"] is None


def test_resolve_auto_batch_reads_path_and_does_not_mutate_input(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "4")
    src = {"zero_optimization": {"stage": 3}, "train_batch_size": "auto"}
    cfg_path = tmp_path / "zero3.json"
    cfg_path.write_text(json.dumps(src))
    trainer = _fake_trainer(micro=1, accum=8)

    cfg = BaseTrainer._resolve_ds_auto_batch(trainer, str(cfg_path))
    assert cfg["train_batch_size"] is None
    assert cfg["train_micro_batch_size_per_gpu"] == 1
    assert cfg["gradient_accumulation_steps"] == 8
    assert cfg["zero_optimization"]["stage"] == 3
    # The original file on disk is untouched.
    assert json.loads(cfg_path.read_text())["train_batch_size"] == "auto"
