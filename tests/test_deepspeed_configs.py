"""Contract tests for the shipped DeepSpeed configs.

These lock down the two ZeRO-3 variants the ml-platform recipe builder points
VLM LoRA at:

* ``zero3_config.json`` — ZeRO-3 **with** CPU offload of params + optimizer
  state (the low-VRAM escape hatch).
* ``zero3_auto_config.json`` — ZeRO-3 with **no** CPU offload; shards the base
  entirely on-GPU. This is the default for high-VRAM sharded LoRA / full-FT.
  CPU offload deadlocks the ZeRO-3 param all-gather for large Gemma4 VLMs on
  B200, so removing offload is what makes full bf16 LoRA of a ~60GB VLM train.

Both configs ship inside the FAI-RL package (``configs/`` is a package and
``pyproject.toml`` package-data globs ``*.json``), so the trainer can resolve
``training.deepspeed_config`` relative to the package root.
"""

import json
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "deepspeed"


def _load(name):
    with open(CONFIG_DIR / name) as f:
        return json.load(f)


def test_both_configs_are_valid_json_and_present():
    assert (CONFIG_DIR / "zero3_config.json").is_file()
    assert (CONFIG_DIR / "zero3_auto_config.json").is_file()
    # Both parse as JSON objects.
    assert isinstance(_load("zero3_config.json"), dict)
    assert isinstance(_load("zero3_auto_config.json"), dict)


def test_zero3_config_offloads_params_and_optimizer_to_cpu():
    zero = _load("zero3_config.json")["zero_optimization"]
    assert zero["stage"] == 3
    assert zero["offload_param"]["device"] == "cpu"
    assert zero["offload_optimizer"]["device"] == "cpu"


def test_zero3_auto_config_is_stage3_without_any_cpu_offload():
    zero = _load("zero3_auto_config.json")["zero_optimization"]
    assert zero["stage"] == 3
    # The whole point of the no-offload variant: neither key may be present,
    # otherwise the ZeRO-3 param all-gather can deadlock on large VLMs.
    assert "offload_param" not in zero
    assert "offload_optimizer" not in zero


def test_no_offload_config_matches_offload_config_except_offload_keys():
    """The no-offload variant must be the offload config with only the two
    ``offload_*`` keys removed — nothing else may drift between them."""
    offload = _load("zero3_config.json")
    no_offload = _load("zero3_auto_config.json")

    offload_zero = dict(offload["zero_optimization"])
    offload_zero.pop("offload_param")
    offload_zero.pop("offload_optimizer")
    assert offload_zero == no_offload["zero_optimization"]

    # Every non-zero_optimization top-level key is identical.
    for key in offload:
        if key == "zero_optimization":
            continue
        assert offload[key] == no_offload[key], key
    assert set(offload) == set(no_offload)


def test_no_offload_config_keeps_bf16_and_auto_batch():
    cfg = _load("zero3_auto_config.json")
    assert cfg["bf16"]["enabled"] is True
    assert cfg["train_batch_size"] == "auto"
    assert cfg["train_micro_batch_size_per_gpu"] == "auto"
    assert cfg["gradient_accumulation_steps"] == "auto"
