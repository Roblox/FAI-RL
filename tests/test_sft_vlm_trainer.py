import math
import sys
from pathlib import Path
from types import SimpleNamespace

from datasets import Dataset

# Ensure the repo root is importable so `trainers.*` resolves.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trainers.sft_vlm_trainer import SFTVLMTrainer


def test_media_source_coercion_ignores_empty_values():
    empty_values = [None, "", " \t\n", b"", math.nan]

    for value in empty_values:
        assert SFTVLMTrainer._coerce_image_source(value) is None
        assert SFTVLMTrainer._coerce_video_source(value) is None


def test_media_source_coercion_strips_valid_paths():
    assert (
        SFTVLMTrainer._coerce_image_source("  s3://bucket/image.jpg  ")
        == "s3://bucket/image.jpg"
    )
    assert (
        SFTVLMTrainer._coerce_video_source("\ts3://bucket/video.mp4\n")
        == "s3://bucket/video.mp4"
    )


def test_normalize_dataset_keeps_valid_media_when_other_cells_are_empty():
    raw = Dataset.from_dict(
        {
            "image_1": ["", "  ", ""],
            "image_2": ["s3://bucket/image.jpg", "", ""],
            "video": ["", "s3://bucket/video.mp4", ""],
            "prompt": ["one", "two", "three"],
            "response": ["a", "b", "c"],
        }
    )
    dataset_info = SimpleNamespace(
        image_columns=["image_1", "image_2"],
        video_columns=["video"],
        dataset_columns=["prompt", "response"],
    )
    trainer = object.__new__(SFTVLMTrainer)
    trainer._split_mode = True
    trainer.config = SimpleNamespace(
        data=SimpleNamespace(
            system_prompt=None,
            user_prompt="{prompt}",
            assistant_prompt="{response}",
        )
    )

    normalized = trainer._normalize_dataset(raw, dataset_info)

    assert normalized["_image_sources"] == [
        ["s3://bucket/image.jpg"],
        [],
        [],
    ]
    assert normalized["_video_sources"] == [
        [],
        ["s3://bucket/video.mp4"],
        [],
    ]


def test_setup_model_loads_processor_from_peft_base_not_adapter_dir(tmp_path, monkeypatch):
    """LoRA checkpoints have no config.json/model_type. AutoProcessor must use the Hub base."""
    import logging

    adapter_dir = tmp_path / "checkpoint-614"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text(
        '{"base_model_name_or_path": "google/gemma-4-31B-it", "peft_type": "LORA"}'
    )

    captured = {}

    def fake_from_pretrained(name, *args, **kwargs):
        captured["processor_name"] = name
        tokenizer = SimpleNamespace(pad_token="pad", eos_token="eos")
        return SimpleNamespace(tokenizer=tokenizer)

    monkeypatch.setattr(
        "trainers.sft_vlm_trainer.AutoProcessor.from_pretrained", fake_from_pretrained
    )

    trainer = object.__new__(SFTVLMTrainer)
    trainer.logger = logging.getLogger("test_peft_processor")
    trainer.config = SimpleNamespace(
        model=SimpleNamespace(
            base_model_name=str(adapter_dir),
            freeze_vision_tower=False,
            use_lora=False,
        )
    )

    def fake_load(_kwargs):
        trainer._peft_adapter_path = str(adapter_dir)
        trainer._peft_base_model_path = "google/gemma-4-31B-it"
        return SimpleNamespace()

    trainer.create_quantization_config = lambda: None
    trainer.prepare_model_kwargs = lambda _q: {}
    trainer.load_base_model_for_training = fake_load
    trainer.apply_lora_to_model = lambda model, *args, **kwargs: model
    trainer.disable_cache_for_gradient_checkpointing = lambda _model: None

    trainer.setup_model()

    assert captured["processor_name"] == "google/gemma-4-31B-it"


def test_hub_cache_warm_follows_peft_base_model(tmp_path):
    adapter_dir = tmp_path / "checkpoint-614"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text(
        '{"base_model_name_or_path": "google/gemma-4-31B-it", "peft_type": "LORA"}'
    )

    trainer = object.__new__(SFTVLMTrainer)
    trainer.local_rank = 0
    trainer.config = SimpleNamespace(model=SimpleNamespace(base_model_name=str(adapter_dir)))

    assert trainer._hub_model_id_for_cache_warm() == "google/gemma-4-31B-it"

    trainer.local_rank = -1
    assert trainer._hub_model_id_for_cache_warm() is None
