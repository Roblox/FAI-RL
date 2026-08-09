"""Tests for device placement in BaseTrainer.prepare_model_kwargs.

The critical regression these lock down: on the full-precision (non-quantized),
non-DeepSpeed CUDA path (plain DDP / single-GPU), from_pretrained must load each
rank's replica directly onto its GPU via device_map={"": current_device}.
Without it, device_map is unset and from_pretrained materializes the ENTIRE
model on the host on every rank, so N concurrent torchrun ranks OOM the host on
large models (the failure that blocked full bf16 LoRA on gemma-4-31B). The
quantized branch already did this; here we assert the full-precision branch
matches it while leaving the DeepSpeed and CPU paths untouched.

Uses an unbound-method call on a minimal SimpleNamespace stub so the heavy
BaseTrainer.__init__ (S3/Hub downloads, device detection) is not invoked.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import core.trainer_base as tb
from core.trainer_base import BaseTrainer


def _fake_trainer(deepspeed_config=None):
    """Minimal stand-in carrying just the recipe fields prepare_model_kwargs reads."""
    return SimpleNamespace(
        config=SimpleNamespace(
            model=SimpleNamespace(
                torch_dtype="bfloat16",
                low_cpu_mem_usage=True,
                ignore_mismatched_sizes=False,
                use_flash_attention=False,
            ),
            training=SimpleNamespace(deepspeed_config=deepspeed_config),
        ),
        logger=MagicMock(),
    )


def _patch_env(monkeypatch, *, cuda=False, mps=False, device=0):
    monkeypatch.setattr(tb, "is_cuda_available", lambda: cuda)
    monkeypatch.setattr(tb, "is_mps_available", lambda: mps)
    monkeypatch.setattr(tb, "resolve_transformers_attn_implementation", lambda *_a, **_k: None)
    if cuda:
        monkeypatch.setattr(tb.torch.cuda, "current_device", lambda: device)


def test_full_precision_ddp_loads_gpu_direct(monkeypatch):
    """Full bf16 + plain DDP (no quant, no DeepSpeed) on CUDA -> device_map to this rank's GPU."""
    _patch_env(monkeypatch, cuda=True, device=3)
    trainer = _fake_trainer(deepspeed_config=None)

    kwargs = BaseTrainer.prepare_model_kwargs(trainer, quantization_config=None)

    assert kwargs["device_map"] == {"": 3}
    assert kwargs["torch_dtype"] is tb.torch.bfloat16
    assert kwargs["low_cpu_mem_usage"] is True


def test_full_precision_deepspeed_leaves_device_map_unset(monkeypatch):
    """Non-quantized + DeepSpeed must NOT set device_map (DeepSpeed places/partitions params)."""
    _patch_env(monkeypatch, cuda=True, device=0)
    trainer = _fake_trainer(deepspeed_config="configs/deepspeed/zero3_auto_config.json")

    kwargs = BaseTrainer.prepare_model_kwargs(trainer, quantization_config=None)

    assert "device_map" not in kwargs


def test_quantized_ddp_still_sets_device_map(monkeypatch):
    """Regression: the quantized DDP path is unchanged (device_map to this rank's GPU)."""
    _patch_env(monkeypatch, cuda=True, device=2)
    trainer = _fake_trainer(deepspeed_config=None)

    kwargs = BaseTrainer.prepare_model_kwargs(trainer, quantization_config=object())

    assert kwargs["device_map"] == {"": 2}
    assert "quantization_config" in kwargs


def test_quantized_deepspeed_leaves_device_map_unset(monkeypatch):
    """Regression: quantized + DeepSpeed still defers placement to DeepSpeed."""
    _patch_env(monkeypatch, cuda=True, device=0)
    trainer = _fake_trainer(deepspeed_config="configs/deepspeed/zero3_config.json")

    kwargs = BaseTrainer.prepare_model_kwargs(trainer, quantization_config=object())

    assert "device_map" not in kwargs


def test_full_precision_cpu_leaves_device_map_unset(monkeypatch):
    """No CUDA / no MPS (CPU or other backend): behavior preserved, no device_map."""
    _patch_env(monkeypatch, cuda=False, mps=False)
    trainer = _fake_trainer(deepspeed_config=None)

    kwargs = BaseTrainer.prepare_model_kwargs(trainer, quantization_config=None)

    assert "device_map" not in kwargs


def test_full_precision_mps_sets_mps_device_map(monkeypatch):
    """Apple Silicon path preserved: non-quant, no CUDA, MPS -> device_map='mps'."""
    _patch_env(monkeypatch, cuda=False, mps=True)
    trainer = _fake_trainer(deepspeed_config=None)

    kwargs = BaseTrainer.prepare_model_kwargs(trainer, quantization_config=None)

    assert kwargs["device_map"] == "mps"
