import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from peft.tuners.tuners_utils import BaseTunerLayer
from transformers.models.gemma4.modeling_gemma4 import Gemma4ClippableLinear

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.peft_lora import peft_model_from_pretrained, register_clippable_linear_lora
from core.trainer_base import BaseTrainer


class ConcreteTrainer(BaseTrainer):
    def setup_model(self):
        pass

    def setup_data(self):
        pass

    def setup_trainer(self):
        pass

    def train(self):
        pass


class _ClipCfg:
    use_clipped_linears = True


def _prepare_inputs_for_generation(self, input_ids=None, **kwargs):
    # PeftModelForCausalLM requires this HF generate hook on the base module.
    return {"input_ids": input_ids, **kwargs}


class _LanguageAndVision(nn.Module):
    def __init__(self, hidden=8):
        super().__init__()
        cfg = _ClipCfg()
        self.q_proj = Gemma4ClippableLinear(cfg, hidden, hidden)
        self.vision_tower = nn.Module()
        self.vision_tower.q_proj = Gemma4ClippableLinear(cfg, hidden, hidden)

    def forward(self, x):
        return self.q_proj(x)

    prepare_inputs_for_generation = _prepare_inputs_for_generation


class _PlainLinearLM(nn.Module):
    def __init__(self, hidden=8):
        super().__init__()
        self.q_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x):
        return self.q_proj(x)

    prepare_inputs_for_generation = _prepare_inputs_for_generation


class _Gemma4LikeMultimodal(nn.Module):
    """Tiny Gemma 4 E-series-shaped model used to exercise modality pruning."""

    def __init__(self, hidden=8):
        super().__init__()
        cfg = _ClipCfg()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.q_proj = Gemma4ClippableLinear(cfg, hidden, hidden)
        self.model.vision_tower = nn.Module()
        self.model.vision_tower.q_proj = Gemma4ClippableLinear(cfg, hidden, hidden)
        self.model.embed_vision = nn.Linear(hidden, hidden, bias=False)
        self.model.audio_tower = nn.Module()
        self.model.audio_tower.q_proj = Gemma4ClippableLinear(cfg, hidden, hidden)
        self.model.embed_audio = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x):
        return self.model.language_model.q_proj(x)

    prepare_inputs_for_generation = _prepare_inputs_for_generation


def _lora_config(**extra):
    kwargs = dict(
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=["q_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    kwargs.update(extra)
    return LoraConfig(**kwargs)


def _trainer_for_lora(exclude_modules=None):
    trainer = object.__new__(ConcreteTrainer)
    trainer.logger = logging.getLogger("test_clippable_linear_lora")
    trainer._peft_adapter_path = None
    trainer.config = SimpleNamespace(
        model=SimpleNamespace(
            use_lora=True,
            load_in_4bit=False,
            load_in_8bit=False,
            lora_r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            lora_target_modules=["q_proj"],
            lora_bias="none",
            lora_exclude_modules=exclude_modules,
        ),
        training=SimpleNamespace(gradient_checkpointing=False),
    )
    return trainer


def test_peft_still_exposes_custom_module_registration():
    """Pinned peft must keep _register_custom_module; a pin bump must not drop it."""
    cfg = _lora_config()
    assert hasattr(cfg, "_register_custom_module")


def test_stock_get_peft_model_rejects_gemma4_clippable_linear():
    model = _LanguageAndVision()
    with pytest.raises(ValueError, match="Gemma4ClippableLinear"):
        get_peft_model(model, _lora_config())


def test_get_peft_model_wraps_clippable_q_proj_and_trains_adapter_not_inner():
    model = _LanguageAndVision()
    inner_id = id(model.q_proj.linear.weight)
    inner_weight = model.q_proj.linear.weight
    cfg = _lora_config()
    register_clippable_linear_lora(cfg)
    peft_model = get_peft_model(model, cfg)

    q_proj = peft_model.base_model.q_proj
    assert isinstance(q_proj, BaseTunerLayer)
    assert q_proj.get_base_layer().weight is inner_weight
    assert id(q_proj.get_base_layer().weight) == inner_id

    lora_params = [p for n, p in peft_model.named_parameters() if "lora_" in n]
    assert lora_params, "expected LoRA adapter parameters on q_proj"
    assert all(p.requires_grad for p in lora_params)
    assert not q_proj.get_base_layer().weight.requires_grad

    x = torch.randn(2, 8, requires_grad=True)
    peft_model.train()
    q_proj(x).sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in lora_params)
    assert q_proj.get_base_layer().weight.grad is None


def test_clipping_is_applied_around_inner_lora():
    model = _LanguageAndVision()
    model.q_proj.input_min.fill_(0.0)
    model.q_proj.input_max.fill_(0.0)
    cfg = _lora_config()
    register_clippable_linear_lora(cfg)
    peft_model = get_peft_model(model, cfg)
    out = peft_model.base_model.q_proj(torch.ones(2, 8))
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_apply_lora_keeps_vision_tower_exclude():
    trainer = _trainer_for_lora(exclude_modules=r"(.*\.)?vision_tower\..*")
    model = trainer.apply_lora_to_model(_LanguageAndVision())

    assert isinstance(model.base_model.q_proj, BaseTunerLayer)
    vision_q = model.base_model.vision_tower.q_proj
    assert isinstance(vision_q, Gemma4ClippableLinear)
    assert not isinstance(vision_q, BaseTunerLayer)
    assert not any("vision_tower" in n and "lora_" in n for n, _ in model.named_parameters())


def test_plain_nn_linear_q_proj_unchanged():
    trainer = _trainer_for_lora()
    model = trainer.apply_lora_to_model(_PlainLinearLM())
    q_proj = model.base_model.q_proj
    assert isinstance(q_proj, BaseTunerLayer)
    assert isinstance(q_proj.get_base_layer(), nn.Linear)
    lora_params = [p for n, p in model.named_parameters() if "lora_" in n]
    assert lora_params
    assert all(p.requires_grad for p in lora_params)
    assert not q_proj.get_base_layer().weight.requires_grad


def test_text_only_qwen_like_model_is_unchanged_by_modality_prep():
    """Qwen3-8B-class CausalLMs have no vision/audio towers; freeze is a no-op."""
    trainer = _trainer_for_lora()
    model = _PlainLinearLM()
    before = {n: p.requires_grad for n, p in model.named_parameters()}

    trainer.prepare_model_for_modalities(model, use_vision=False, use_audio=False)

    after = {n: p.requires_grad for n, p in model.named_parameters()}
    assert after == before
    assert trainer.config.model.lora_exclude_modules is None

    peft_model = trainer.apply_lora_to_model(model)
    q_proj = peft_model.base_model.q_proj
    assert isinstance(q_proj, BaseTunerLayer)
    assert isinstance(q_proj.get_base_layer(), nn.Linear)


def test_text_only_freezes_and_excludes_gemma4_vision_and_audio():
    trainer = _trainer_for_lora()
    model = _Gemma4LikeMultimodal()

    trainer.prepare_model_for_modalities(model, use_vision=False, use_audio=False)

    assert not any(p.requires_grad for p in model.model.vision_tower.parameters())
    assert not any(p.requires_grad for p in model.model.embed_vision.parameters())
    assert not any(p.requires_grad for p in model.model.audio_tower.parameters())
    assert not any(p.requires_grad for p in model.model.embed_audio.parameters())
    assert all(p.requires_grad for p in model.model.language_model.parameters())

    peft_model = trainer.apply_lora_to_model(model)
    base = peft_model.base_model.model.model
    assert isinstance(base.language_model.q_proj, BaseTunerLayer)
    assert isinstance(base.vision_tower.q_proj, Gemma4ClippableLinear)
    assert isinstance(base.audio_tower.q_proj, Gemma4ClippableLinear)
    assert not any(
        ("vision_tower" in name or "audio_tower" in name) and "lora_" in name
        for name, _ in peft_model.named_parameters()
    )


def test_image_only_keeps_vision_trainable_but_freezes_and_excludes_audio():
    trainer = _trainer_for_lora()
    model = _Gemma4LikeMultimodal()

    trainer.prepare_model_for_modalities(model, use_vision=True, use_audio=False)

    assert all(p.requires_grad for p in model.model.vision_tower.parameters())
    assert all(p.requires_grad for p in model.model.embed_vision.parameters())
    assert not any(p.requires_grad for p in model.model.audio_tower.parameters())
    assert not any(p.requires_grad for p in model.model.embed_audio.parameters())

    peft_model = trainer.apply_lora_to_model(model)
    base = peft_model.base_model.model.model
    assert isinstance(base.language_model.q_proj, BaseTunerLayer)
    assert isinstance(base.vision_tower.q_proj, BaseTunerLayer)
    assert isinstance(base.audio_tower.q_proj, Gemma4ClippableLinear)


def test_image_only_respects_explicit_freeze_vision_recipe():
    trainer = _trainer_for_lora()
    model = _Gemma4LikeMultimodal()

    trainer.prepare_model_for_modalities(
        model, use_vision=True, use_audio=False, freeze_vision=True
    )

    assert not any(p.requires_grad for p in model.model.vision_tower.parameters())
    assert all(p.requires_grad for p in model.model.embed_vision.parameters())
    assert not any(p.requires_grad for p in model.model.audio_tower.parameters())
    assert not any(p.requires_grad for p in model.model.embed_audio.parameters())


def test_resume_refreezes_adapters_left_inside_encoder_towers(monkeypatch):
    """A pre-fix checkpoint can carry LoRA adapters inside the vision/audio
    towers. Resuming it with is_trainable=True would re-enable those unused
    adapters and crash DDP; the resume path must re-freeze the towers."""
    # Build an "old checkpoint": LoRA injected with NO exclusion, so trainable
    # adapters land in every q_proj -- including the towers.
    cfg = _lora_config()
    register_clippable_linear_lora(cfg)
    old_ckpt = get_peft_model(_Gemma4LikeMultimodal(), cfg)
    assert any(
        ("vision_tower" in n or "audio_tower" in n) and "lora_" in n and p.requires_grad
        for n, p in old_ckpt.named_parameters()
    ), "fixture should start with trainable tower adapters"

    trainer = _trainer_for_lora()
    trainer._peft_adapter_path = "unused-adapter-dir"
    monkeypatch.setattr(
        "core.trainer_base.peft_model_from_pretrained",
        lambda model, path, **kwargs: old_ckpt,
    )

    # The freeze pass records which encoder subtrees to keep frozen on resume.
    base = _Gemma4LikeMultimodal()
    trainer.prepare_model_for_modalities(base, use_vision=False, use_audio=False)

    resumed = trainer.apply_lora_to_model(base)

    tower_adapters = [
        p for n, p in resumed.named_parameters()
        if ("vision_tower" in n or "audio_tower" in n) and "lora_" in n
    ]
    language_adapters = [
        p for n, p in resumed.named_parameters()
        if "language_model" in n and "lora_" in n
    ]
    assert tower_adapters and not any(p.requires_grad for p in tower_adapters)
    assert language_adapters and all(p.requires_grad for p in language_adapters)


def test_qlora_loaded_in_4bit_is_forwarded_to_inner_linear(monkeypatch):
    """QLoRA dispatch sees the inner nn.Linear/Linear4bit, not the clip wrapper."""
    from peft.tuners.lora.model import LoraModel

    orig = LoraModel._create_new_module
    inner_calls = []

    def spy(config, adapter_name, target, **kwargs):
        inner_calls.append((target, kwargs.get("loaded_in_4bit")))
        return orig(config, adapter_name, target, **kwargs)

    monkeypatch.setattr(LoraModel, "_create_new_module", staticmethod(spy))
    model = _LanguageAndVision()
    model.is_loaded_in_4bit = True
    cfg = _lora_config()
    register_clippable_linear_lora(cfg)
    get_peft_model(model, cfg)

    assert any(isinstance(target, Gemma4ClippableLinear) for target, _ in inner_calls)
    assert any(
        isinstance(target, nn.Linear) and loaded_in_4bit is True
        for target, loaded_in_4bit in inner_calls
    )


def test_failed_inner_dispatch_restores_custom_modules(monkeypatch):
    from peft.tuners.lora.model import LoraModel

    orig = LoraModel._create_new_module
    cfg = _lora_config()
    register_clippable_linear_lora(cfg)
    mapping = cfg._custom_modules

    def spy(config, adapter_name, target, **kwargs):
        if isinstance(target, nn.Linear):
            raise RuntimeError("dispatch failed")
        return orig(config, adapter_name, target, **kwargs)

    monkeypatch.setattr(LoraModel, "_create_new_module", staticmethod(spy))
    with pytest.raises(RuntimeError, match="dispatch failed"):
        get_peft_model(_LanguageAndVision(), cfg)
    assert cfg._custom_modules is mapping


def test_peft_model_from_pretrained_reregisters_custom_modules(monkeypatch):
    captured = {}

    def fake_from_pretrained(model, model_id, **kwargs):
        captured["config"] = kwargs.get("config")
        return model

    monkeypatch.setattr("peft.PeftModel.from_pretrained", fake_from_pretrained)
    monkeypatch.setattr("peft.PeftConfig.from_pretrained", lambda _id: _lora_config())

    dummy = _LanguageAndVision()
    peft_model_from_pretrained(dummy, "unused-adapter")
    config = captured["config"]
    assert config is not None
    assert config._custom_modules
    assert Gemma4ClippableLinear in config._custom_modules
