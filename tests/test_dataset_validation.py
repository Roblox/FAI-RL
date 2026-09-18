"""CPU-only preflight coverage with tiny in-memory datasets and fake tokenizers."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.config import DataConfig, DatasetInfo
from utils.dataset_validation import preflight_datasets, validate_chat_templates, validate_dataset


class Rows(list):
    """Minimal subset of the datasets.Dataset interface used during preflight."""

    @property
    def column_names(self):
        return list(self[0]) if self else []


def check(rows, algorithm="sft", info=None, data=None, logger=None):
    return validate_dataset(
        Rows(rows),
        info or DatasetInfo(name="test"),
        data or DataConfig(),
        algorithm,
        logger or Mock(),
    )


@pytest.mark.parametrize(
    "algorithm, rows, missing",
    [
        ("dpo", [{"prompt": "hello", "response": "hi"}], "chosen"),
        ("grpo", [{"answer": "42"}], "prompt"),
        ("gspo", [{"answer": "42"}], "prompt"),
        ("cpt", [{"prompt": "hello"}], "text"),
        ("sft", [{"prompt": "hello"}], "text"),
    ],
)
def test_missing_required_columns(algorithm, rows, missing):
    with pytest.raises(ValueError, match=missing) as error:
        check(rows, algorithm)
    assert "Expected:" in str(error.value) and "Found:" in str(error.value)


def test_optional_dpo_prompt_and_rl_answer():
    check([{"chosen": "yes", "rejected": "no"}], "dpo")
    check([{"prompt": "hello"}], "grpo")


def test_custom_cpt_column():
    check([{"body": "hello"}], "cpt", DatasetInfo(name="test", text_column="body"))


@pytest.mark.parametrize("rows", [[], [{"text": None}], [{"text": "  "}], [{"text": 2}]])
def test_empty_or_unusable_dataset(rows):
    with pytest.raises(ValueError, match="empty|no usable rows"):
        check(rows)


def test_missing_values_warn_without_changing_rows():
    rows = [{"text": None}, {"text": "hello"}]
    logger = Mock()
    check(rows, logger=logger)
    logger.warning.assert_called_once()
    assert len(rows) == 2


def test_bad_prompt_format_fails_early():
    with pytest.raises(ValueError, match="sample formatting failed"):
        check(
            [{"question": "hello"}],
            info=DatasetInfo(name="test", dataset_columns=["question"]),
            data=DataConfig(system_prompt="{misspelled}"),
        )


def test_split_chat_and_vlm_media():
    info = DatasetInfo(name="test", dataset_columns=["question", "answer"], image_columns=["image"])
    data = DataConfig(user_prompt="{question}", assistant_prompt="{answer}")
    messages = check(
        [{"question": "hello", "answer": "hi", "image": "local.png"}], "sft_vlm", info, data
    )
    assert messages[0][0]["content"][0] == {"type": "image"}
    assert messages[0][-1] == {"role": "assistant", "content": "hi"}


def test_malformed_conversation():
    with pytest.raises(ValueError, match="sample formatting failed"):
        check([{"prompt": [{"wrong": "key"}]}], "grpo")


def test_chat_template_checked_without_model(monkeypatch):
    processor = Mock()
    loader = Mock()
    loader.from_pretrained.return_value = processor
    monkeypatch.setitem(
        sys.modules, "transformers", SimpleNamespace(AutoTokenizer=loader, AutoProcessor=loader)
    )
    config = SimpleNamespace(
        model=SimpleNamespace(base_model_name="test"),
        training=SimpleNamespace(algorithm="sft"),
        _preflight_chats=[[{"role": "user", "content": "hello"}]],
    )
    processor.apply_chat_template.side_effect = ValueError("private sample content")
    with pytest.raises(ValueError, match="Chat template preflight") as error:
        validate_chat_templates(config, Mock())
    assert "private sample content" not in str(error.value)
    processor.apply_chat_template.side_effect = None
    validate_chat_templates(config, Mock())
    assert config._preflight_chats == []


def test_preflight_caches_raw_dataset(monkeypatch):
    load = Mock(return_value=Rows([{"text": "hello"}]))
    monkeypatch.setattr("utils.dataset_utils.load_raw_dataset", load)
    config = SimpleNamespace(
        data=DataConfig(datasets=[DatasetInfo(name="test")]),
        training=SimpleNamespace(algorithm="sft"),
    )
    preflight_datasets(config, Mock())
    preflight_datasets(config, Mock())
    load.assert_called_once()
    assert config._preflight_datasets[0][0]["text"] == "hello"


def test_empty_rendered_text_and_missing_media_fail():
    data = DataConfig(user_prompt="{question}", assistant_prompt="{answer}")
    info = DatasetInfo(name="test", dataset_columns=["question", "answer"])
    with pytest.raises(ValueError, match="no usable rows"):
        check([{"question": "  ", "answer": "hi"}], info=info, data=data)
    info.image_columns = ["image"]
    with pytest.raises(ValueError, match="no usable rows"):
        check([{"question": "hello", "answer": "hi", "image": None}], "sft_vlm", info, data)


def test_main_preflight_precedes_launch(monkeypatch):
    from trainers import train

    args = SimpleNamespace(recipe=None, overrides=[], nohup=False, num_gpus=8)
    monkeypatch.setattr(train, "parse_args", lambda: args)
    config = SimpleNamespace(data=DataConfig(), training=SimpleNamespace(algorithm="sft"))
    monkeypatch.setattr(train, "load_recipe_with_overrides", lambda args: config)
    launch = Mock()
    monkeypatch.setattr(train, "launch_distributed_training", launch)
    with pytest.raises(ValueError, match="at least one"):
        train.main()
    launch.assert_not_called()


def test_programmatic_algorithm_fallback(monkeypatch):
    load = Mock(return_value=Rows([{"body": "hello"}]))
    monkeypatch.setattr("utils.dataset_utils.load_raw_dataset", load)
    config = SimpleNamespace(
        data=DataConfig(datasets=[DatasetInfo(name="test", text_column="body")]),
        training=SimpleNamespace(algorithm=None),
    )
    preflight_datasets(config, Mock(), algorithm="cpt")
    assert config.training.algorithm is None
    assert config._preflight_algorithm == "cpt"
