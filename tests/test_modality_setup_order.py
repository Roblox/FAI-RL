import logging
from types import SimpleNamespace

import torch.nn as nn

from trainers.dpo_trainer import DPOTrainer
from trainers.sft_trainer import SFTTrainer
from trainers.sft_vlm_trainer import SFTVLMTrainer


def _text_trainer_setup(trainer_cls):
    trainer = object.__new__(trainer_cls)
    trainer.logger = logging.getLogger(f"test_{trainer_cls.__name__}")
    trainer.config = SimpleNamespace(model=SimpleNamespace(base_model_name="unused", use_lora=True))
    trainer.model = None
    trainer.ref_model = None
    trainer.create_quantization_config = lambda: None
    trainer.prepare_model_kwargs = lambda _quantization: {}
    trainer.load_base_model_for_training = lambda _kwargs: nn.Module()
    trainer.setup_tokenizer_with_model = lambda _model: SimpleNamespace()
    trainer.disable_cache_for_gradient_checkpointing = lambda _model: None
    return trainer


def test_text_sft_prepares_text_only_modalities_before_lora():
    trainer = _text_trainer_setup(SFTTrainer)
    calls = []
    trainer.prepare_model_for_modalities = lambda model, **kwargs: calls.append(
        ("modalities", kwargs)
    )
    trainer.apply_lora_to_model = lambda model, *args, **kwargs: (
        calls.append(("lora", kwargs)) or model
    )

    trainer.setup_model()

    assert calls[0] == (
        "modalities",
        {"use_vision": False, "use_audio": False},
    )
    assert calls[1][0] == "lora"


def test_dpo_prepares_text_only_modalities_before_lora():
    trainer = _text_trainer_setup(DPOTrainer)
    calls = []
    trainer.prepare_model_for_modalities = lambda model, **kwargs: calls.append(
        ("modalities", kwargs)
    )
    trainer.apply_lora_to_model = lambda model, *args, **kwargs: (
        calls.append(("lora", kwargs)) or model
    )

    trainer.setup_model()

    assert calls[0] == (
        "modalities",
        {"use_vision": False, "use_audio": False},
    )
    assert calls[1][0] == "lora"


def test_vlm_image_collator_keeps_vision_and_prepares_before_lora(monkeypatch):
    tokenizer = SimpleNamespace(pad_token="pad", eos_token="eos")
    monkeypatch.setattr(
        "trainers.sft_vlm_trainer.AutoProcessor.from_pretrained",
        lambda *_args, **_kwargs: SimpleNamespace(tokenizer=tokenizer),
    )

    trainer = object.__new__(SFTVLMTrainer)
    trainer.logger = logging.getLogger("test_SFTVLMTrainer")
    trainer.config = SimpleNamespace(
        model=SimpleNamespace(
            base_model_name="unused",
            use_lora=True,
            freeze_vision_tower=False,
        ),
        data=SimpleNamespace(
            datasets=[SimpleNamespace(image_columns=["image"], video_columns=None)]
        ),
    )
    trainer.create_quantization_config = lambda: None
    trainer.prepare_model_kwargs = lambda _quantization: {}
    trainer.load_base_model_for_training = lambda _kwargs: nn.Module()
    trainer.disable_cache_for_gradient_checkpointing = lambda _model: None
    calls = []
    trainer.prepare_model_for_modalities = lambda model, **kwargs: calls.append(
        ("modalities", kwargs)
    )
    trainer.apply_lora_to_model = lambda model, *args, **kwargs: (
        calls.append(("lora", kwargs)) or model
    )

    trainer.setup_model()

    assert calls[0] == (
        "modalities",
        {
            "use_vision": True,
            "use_audio": False,
            "freeze_vision": False,
        },
    )
    assert calls[1][0] == "lora"
