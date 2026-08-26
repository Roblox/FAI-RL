import logging
import sys
from pathlib import Path
from types import SimpleNamespace

# Ensure the repo root is importable so `core.*` resolves.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.trainer_base import BaseTrainer
from trainers.dpo_trainer import DPOTrainer


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


def test_resolved_pretrained_name_falls_back_for_full_model():
    trainer = object.__new__(ConcreteTrainer)
    trainer.config = SimpleNamespace(model=SimpleNamespace(base_model_name="catalog/model"))
    trainer._peft_base_model_path = None

    assert trainer.resolved_pretrained_name() == "catalog/model"


def test_dpo_reference_model_loads_from_peft_base(monkeypatch):
    captured = {}

    class FakeTokenizer:
        pad_token = "pad"
        eos_token = "eos"
        padding_side = "right"

        def add_special_tokens(self, _tokens):
            return 0

        def __len__(self):
            return 128

    class FakeModel:
        def resize_token_embeddings(self, _size):
            pass

    monkeypatch.setattr(
        "core.trainer_base.AutoTokenizer.from_pretrained",
        lambda name, *args, **kwargs: FakeTokenizer(),
    )

    def fake_model_from_pretrained(name, *args, **kwargs):
        captured["reference_model_name"] = name
        return FakeModel()

    monkeypatch.setattr(
        "trainers.dpo_trainer.AutoModelForCausalLM.from_pretrained",
        fake_model_from_pretrained,
    )

    trainer = object.__new__(DPOTrainer)
    trainer.logger = logging.getLogger("test_peft_dpo_reference")
    trainer.config = SimpleNamespace(
        model=SimpleNamespace(
            base_model_name="/tmp/fai-rl-model-adapter",
            use_lora=False,
        )
    )

    def fake_load(_kwargs):
        trainer._peft_adapter_path = "/tmp/fai-rl-model-adapter"
        trainer._peft_base_model_path = "Qwen/Qwen3.8-27B"
        return FakeModel()

    trainer.create_quantization_config = lambda: None
    trainer.prepare_model_kwargs = lambda _quantization: {}
    trainer.load_base_model_for_training = fake_load
    trainer.apply_lora_to_model = lambda model, *args, **kwargs: model
    trainer.disable_cache_for_gradient_checkpointing = lambda _model: None

    trainer.setup_model()

    assert captured["reference_model_name"] == "Qwen/Qwen3.8-27B"
