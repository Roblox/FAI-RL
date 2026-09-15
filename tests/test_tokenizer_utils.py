from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from utils import tokenizer_utils
from utils.tokenizer_utils import (
    adapter_embedding_rows,
    configure_padding_token,
    grow_token_embeddings_if_needed,
    prepare_tokenizer_with_model,
)


class FakeTokenizer:
    def __init__(self, size, *, pad_token=None, eos_token="</s>"):
        self.size = size
        self.pad_token = pad_token
        self.eos_token = eos_token
        self.padding_side = "right"
        self.added_tokens = []

    def __len__(self):
        return self.size

    def add_special_tokens(self, tokens):
        pad_token = tokens["pad_token"]
        self.added_tokens.append(pad_token)
        self.pad_token = pad_token
        self.size += 1
        return 1


class FakeModel:
    def __init__(self, rows, *, ds_rows=None):
        weight = SimpleNamespace(shape=(rows, 16))
        if ds_rows is not None:
            weight.ds_shape = (ds_rows, 16)
        self.embeddings = SimpleNamespace(weight=weight)
        self.resize_calls = []

    def get_input_embeddings(self):
        return self.embeddings

    def resize_token_embeddings(self, rows):
        self.resize_calls.append(rows)
        self.embeddings.weight.shape = (rows, 16)
        if hasattr(self.embeddings.weight, "ds_shape"):
            self.embeddings.weight.ds_shape = (rows, 16)


def test_existing_pad_token_does_not_change_vocab_or_shrink_model():
    tokenizer = FakeTokenizer(100, pad_token="<pad>")
    model = FakeModel(128)

    prepare_tokenizer_with_model(tokenizer, model)

    assert tokenizer.added_tokens == []
    assert tokenizer.padding_side == "left"
    assert model.resize_calls == []


def test_missing_pad_reuses_eos_without_growing_vocab():
    tokenizer = FakeTokenizer(100, pad_token=None, eos_token="</s>")

    configure_padding_token(tokenizer)

    assert tokenizer.pad_token == "</s>"
    assert len(tokenizer) == 100
    assert tokenizer.added_tokens == []


def test_missing_pad_and_eos_adds_token_and_grows_model():
    tokenizer = FakeTokenizer(100, pad_token=None, eos_token=None)
    model = FakeModel(100)

    prepare_tokenizer_with_model(tokenizer, model)

    assert tokenizer.pad_token == "[PAD]"
    assert model.resize_calls == [101]


def test_grow_only_resize_uses_full_zero3_shape():
    tokenizer = FakeTokenizer(100, pad_token="<pad>")
    model = FakeModel(0, ds_rows=128)

    resized = grow_token_embeddings_if_needed(model, tokenizer)

    assert resized is False
    assert model.resize_calls == []


def test_model_grows_when_tokenizer_is_larger():
    tokenizer = FakeTokenizer(101, pad_token="<pad>")
    model = FakeModel(100)

    resized = grow_token_embeddings_if_needed(model, tokenizer)

    assert resized is True
    assert model.resize_calls == [101]


def test_adapter_embedding_rows_reads_saved_input_embedding(tmp_path):
    save_file(
        {
            "base_model.model.model.embed_tokens.weight": torch.zeros(101, 16),
            "base_model.model.model.layers.0.lora_A.weight": torch.zeros(4, 16),
        },
        tmp_path / "adapter_model.safetensors",
    )

    assert adapter_embedding_rows(str(tmp_path)) == 101


def test_legacy_adapter_rows_restore_old_pad_layout(monkeypatch):
    tokenizer = FakeTokenizer(100, pad_token="</s>")
    model = FakeModel(128)
    monkeypatch.setattr(tokenizer_utils, "adapter_embedding_rows", lambda _path: 101)

    prepare_tokenizer_with_model(
        tokenizer,
        model,
        adapter_path="/tmp/legacy-adapter",
    )

    assert tokenizer.pad_token == "[PAD]"
    assert len(tokenizer) == 101
    assert model.resize_calls == [101]


def test_adapter_cannot_have_fewer_rows_than_tokenizer(monkeypatch):
    tokenizer = FakeTokenizer(100, pad_token="<pad>")
    model = FakeModel(128)
    monkeypatch.setattr(tokenizer_utils, "adapter_embedding_rows", lambda _path: 99)

    with pytest.raises(ValueError, match="tokenizer requires 100"):
        prepare_tokenizer_with_model(
            tokenizer,
            model,
            adapter_path="/tmp/broken-adapter",
        )
