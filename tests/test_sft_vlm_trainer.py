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
