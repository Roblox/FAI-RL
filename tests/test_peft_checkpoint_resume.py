import sys
from pathlib import Path
from types import SimpleNamespace

# Ensure the repo root is importable so `core.*` resolves.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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


def test_tokenizer_loads_from_peft_base_not_adapter_dir(monkeypatch):
    """Adapter checkpoints have no config.json, so tokenizers must use the Hub base."""
    captured = {}

    class FakeTokenizer:
        pad_token = "pad"
        eos_token = "eos"
        padding_side = "right"

        def add_special_tokens(self, _tokens):
            return 0

        def __len__(self):
            return 128

    def fake_from_pretrained(name, *args, **kwargs):
        captured["tokenizer_name"] = name
        return FakeTokenizer()

    monkeypatch.setattr("core.trainer_base.AutoTokenizer.from_pretrained", fake_from_pretrained)

    trainer = object.__new__(ConcreteTrainer)
    trainer.config = SimpleNamespace(
        model=SimpleNamespace(base_model_name="/tmp/fai-rl-model-adapter")
    )
    trainer._peft_base_model_path = "Qwen/Qwen3.8-27B"
    model = SimpleNamespace(resize_token_embeddings=lambda _size: None)

    trainer.setup_tokenizer_with_model(model)

    assert captured["tokenizer_name"] == "Qwen/Qwen3.8-27B"


def test_explicit_tokenizer_model_name_overrides_peft_base(monkeypatch):
    captured = {}

    class FakeTokenizer:
        pad_token = "pad"
        eos_token = "eos"
        padding_side = "right"

        def add_special_tokens(self, _tokens):
            return 0

        def __len__(self):
            return 128

    def fake_from_pretrained(name, *args, **kwargs):
        captured["tokenizer_name"] = name
        return FakeTokenizer()

    monkeypatch.setattr("core.trainer_base.AutoTokenizer.from_pretrained", fake_from_pretrained)

    trainer = object.__new__(ConcreteTrainer)
    trainer.config = SimpleNamespace(model=SimpleNamespace(base_model_name="catalog/model"))
    trainer._peft_base_model_path = "adapter/base"
    model = SimpleNamespace(resize_token_embeddings=lambda _size: None)

    trainer.setup_tokenizer_with_model(model, model_name="explicit/tokenizer")

    assert captured["tokenizer_name"] == "explicit/tokenizer"
