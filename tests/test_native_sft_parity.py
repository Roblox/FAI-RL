"""The native SFT loop (``training.backend: native``) reproduces the TRL backend."""

import json
import logging
from functools import partial

import pytest
import torch
from safetensors.torch import load_file

from core.config import DataConfig, DatasetInfo, ExperimentConfig, ModelConfig, TrainingConfig, WandbConfig
from engine.cookbook import sft as native_sft

ROWS = [
    ("What is two plus two?", "four"),
    ("Say hello", "hello world"),
    ("Game A has score 10. Game B has score 4. Which is higher?", "A"),
    ("Compare game A and game B.", "B"),
]


def write_sft_dataset(path, repeats=1):
    with open(path, "w", encoding="utf-8") as f:
        for question, response in ROWS * repeats:
            f.write(json.dumps({"question": question, "response": response}) + "\n")
    return str(path)


def sft_config(model_dir, data_path, output_dir, **training) -> ExperimentConfig:
    training_kwargs = dict(
        algorithm="sft",
        backend="native",
        output_dir=str(output_dir),
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=1e-3,
        max_steps=3,
        warmup_steps=0,
        logging_steps=1,
        save_steps=1000,
        eval_enabled=False,
        early_stopping=False,
        bf16=False,
        gradient_checkpointing=False,
    )
    training_kwargs.update(training)
    return ExperimentConfig(
        model=ModelConfig(base_model_name=model_dir, torch_dtype="float32", use_lora=False),
        data=DataConfig(
            datasets=[DatasetInfo(name=data_path, dataset_columns=["question", "response"])],
            max_length=128,
            user_prompt="{question}",
            assistant_prompt="{response}",
        ),
        training=TrainingConfig(**training_kwargs),
        wandb=WandbConfig(enabled=False),
    )


def _train_logs(history):
    return [entry for entry in history if "loss" in entry and "train_loss" not in entry]


def test_native_sft_matches_trl_losses_and_weights(force_cpu, tiny_model_dir, tmp_path, monkeypatch):
    from trl import SFTConfig

    from trainers import sft_trainer as trl_sft

    data_path = write_sft_dataset(tmp_path / "sft.jsonl")
    logger = logging.getLogger("test_native_sft_parity")

    monkeypatch.setattr(trl_sft, "SFTConfig", partial(SFTConfig, use_cpu=True))
    trl_config = sft_config(tiny_model_dir, data_path, tmp_path / "trl", backend="trl")
    with trl_sft.SFTTrainer(trl_config, logger=logger) as trainer:
        trainer.train()
        trl_logs = _train_logs(trainer.trainer.state.log_history)

    native_config = sft_config(tiny_model_dir, data_path, tmp_path / "native")
    native_logs = _train_logs(native_sft.run(native_config, logger).log_history)

    assert [entry["step"] for entry in native_logs] == [1, 2, 3]
    assert [entry["step"] for entry in trl_logs] == [1, 2, 3]
    for native, reference in zip(native_logs, trl_logs):
        assert native["loss"] == pytest.approx(reference["loss"], rel=1e-3)
        assert native["grad_norm"] == pytest.approx(reference["grad_norm"], rel=1e-3)
        assert native["learning_rate"] == pytest.approx(reference["learning_rate"], rel=1e-6)

    native_weights = load_file(str(tmp_path / "native" / "model.safetensors"))
    trl_weights = load_file(str(tmp_path / "trl" / "model.safetensors"))
    assert native_weights.keys() == trl_weights.keys()
    for key, value in native_weights.items():
        assert torch.allclose(value, trl_weights[key], atol=1e-5), key
