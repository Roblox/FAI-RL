"""Colocated vLLM must build its engine without HF's ZeRO-3 partitioning.

Training registers a process-global HfDeepSpeedConfig. vLLM's multimodal models
(Gemma 4's vision and audio towers) build submodules with transformers'
``from_config``, which then wraps them in ``deepspeed.zero.Init`` and leaves
size-0 parameters that vLLM's weight loader rejects. A GPU is needed to load a
real model, so these tests pin the scoping instead: only TRL's vLLM engine
build sees ZeRO-3 as disabled, and everything else still sees it.
"""

import logging
from types import SimpleNamespace

import pytest
from transformers.integrations import deepspeed as hf_deepspeed
from trl.generation.vllm_generation import VLLMGeneration

from trainers import grpo_trainer, grpo_vlm_trainer, gspo_trainer


class _Zero3Config:
    def is_zero3(self):
        return True


@pytest.fixture
def zero3_registered():
    config = _Zero3Config()
    hf_deepspeed.set_hf_deepspeed_config(config)
    yield config
    hf_deepspeed.unset_hf_deepspeed_config()


@pytest.fixture
def recorded_vllm_init(monkeypatch):
    seen = []

    def init_vllm(self):
        seen.append(hf_deepspeed.is_deepspeed_zero3_enabled())

    monkeypatch.setattr(VLLMGeneration, "_init_vllm", init_vllm)
    return init_vllm, seen


class _FakeTRLTrainer:
    """Stands in for TRL's GRPOTrainer, which builds vLLM inside ``__init__``."""

    def __init__(self, **kwargs):
        self.zero3_before_vllm = hf_deepspeed.is_deepspeed_zero3_enabled()
        VLLMGeneration._init_vllm(object.__new__(VLLMGeneration))
        self.zero3_after_vllm = hf_deepspeed.is_deepspeed_zero3_enabled()


TRAINERS = [
    pytest.param(grpo_trainer, "GRPOTrainer", "TRLGRPOTrainer", id="grpo"),
    pytest.param(grpo_vlm_trainer, "GRPOVLMTrainer", "TRLGRPOTrainer", id="grpo_vlm"),
    pytest.param(gspo_trainer, "GSPOTrainer", "TRLGSPOTrainer", id="gspo"),
]


@pytest.mark.parametrize("module, cls_name, trl_attr", TRAINERS)
def test_setup_trainer_builds_vllm_outside_zero3(
    monkeypatch, zero3_registered, recorded_vllm_init, module, cls_name, trl_attr
):
    init_vllm, seen = recorded_vllm_init
    monkeypatch.setattr(module, trl_attr, _FakeTRLTrainer)
    monkeypatch.setattr(module, "build_reward_function", lambda *a, **k: lambda **_: [])

    trainer = object.__new__(getattr(module, cls_name))
    trainer.logger = logging.getLogger(f"test_{cls_name}")
    trainer.config = SimpleNamespace()
    trainer.model = trainer.train_dataset = trainer.processor = None
    trainer.setup_training_args = lambda: SimpleNamespace()
    trainer.build_callbacks = lambda: []

    trainer.setup_trainer()

    assert seen == [False]
    assert trainer.trainer.zero3_before_vllm is True
    assert trainer.trainer.zero3_after_vllm is True
    assert hf_deepspeed.is_deepspeed_zero3_enabled() is True
    assert VLLMGeneration._init_vllm is init_vllm


def test_zero3_is_restored_when_vllm_init_fails(monkeypatch, zero3_registered):
    def failing_init_vllm(self):
        raise RuntimeError("vLLM failed to start")

    monkeypatch.setattr(VLLMGeneration, "_init_vllm", failing_init_vllm)

    with pytest.raises(RuntimeError, match="vLLM failed to start"):
        with grpo_trainer.vllm_engine_outside_zero3():
            VLLMGeneration._init_vllm(object.__new__(VLLMGeneration))

    assert hf_deepspeed.is_deepspeed_zero3_enabled() is True
    assert VLLMGeneration._init_vllm is failing_init_vllm


def test_vllm_init_is_untouched_without_zero3(recorded_vllm_init):
    init_vllm, seen = recorded_vllm_init

    with grpo_trainer.vllm_engine_outside_zero3():
        VLLMGeneration._init_vllm(object.__new__(VLLMGeneration))

    assert seen == [False]
    assert hf_deepspeed.is_deepspeed_zero3_enabled() is False
    assert VLLMGeneration._init_vllm is init_vllm
