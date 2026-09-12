"""Conservative local checkpoint discovery without torch or pickle loading."""

import json
import re
from pathlib import Path
from typing import Any, Optional


def _nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _weights_present(path: Path) -> bool:
    """Check single-file weights or every shard named by a HF index."""
    if any(
        _nonempty(path / name)
        for name in (
            "model.safetensors",
            "pytorch_model.bin",
            "adapter_model.safetensors",
            "adapter_model.bin",
        )
    ):
        return True
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index = path / name
        if index.is_file():
            mapping = json.loads(index.read_text(encoding="utf-8")).get("weight_map", {})
            if mapping and all(
                isinstance(shard, str)
                and (path / shard).resolve().parent == path.resolve()
                and _nonempty(path / shard)
                for shard in mapping.values()
            ):
                return True
    return False


def is_resumable_checkpoint(path: Path) -> bool:
    """Require readable trainer state, weights, and optimizer/scheduler state.

    This checks file structure, not tensor integrity or compatibility with a
    particular recipe. Model-only exports intentionally do not qualify.
    """
    try:
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        if not match or not path.is_dir():
            return False
        state = json.loads((path / "trainer_state.json").read_text(encoding="utf-8"))
        step = state.get("global_step")
        if type(step) is not int or step != int(match[1]):
            return False
        # HF Trainer delegates optimizer/scheduler recovery to DeepSpeed.
        latest = path / "latest"
        if latest.is_file():
            tag = latest.read_text(encoding="utf-8").strip()
            ds_dir = path / tag
            if ds_dir.resolve().parent == path.resolve() and ds_dir.is_dir():
                models = list(ds_dir.glob("*model_states.pt"))
                optimizers = list(ds_dir.glob("*optim_states.pt"))
                if models and optimizers and all(_nonempty(p) for p in models + optimizers):
                    return True
        return (
            _weights_present(path)
            and _nonempty(path / "optimizer.pt")
            and _nonempty(path / "scheduler.pt")
        )
    except (OSError, ValueError, TypeError, AttributeError):
        return False


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    """Return the highest-step structurally resumable checkpoint, or None."""
    try:
        candidates = sorted(
            (
                path
                for path in Path(output_dir).glob("checkpoint-*")
                if re.fullmatch(r"checkpoint-\d+", path.name)
            ),
            key=lambda path: int(path.name.split("-")[1]),
            reverse=True,
        )
        return next((str(path) for path in candidates if is_resumable_checkpoint(path)), None)
    except OSError:
        return None


def select_resume_checkpoint(training: Any, auto_resume: bool, logger: Any) -> None:
    """Resolve opt-in recovery, giving an explicit resume path precedence."""
    if getattr(training, "resume_from_checkpoint", None):
        logger.info("Using explicit resume checkpoint: %s", training.resume_from_checkpoint)
        return
    checkpoint = find_latest_checkpoint(training.output_dir)
    if auto_resume and checkpoint:
        training.resume_from_checkpoint = checkpoint
        logger.info("Automatically resuming from %s", checkpoint)
    elif checkpoint:
        logger.info("Recoverable checkpoint found: %s. Add --auto-resume to resume.", checkpoint)
    elif auto_resume:
        logger.info(
            "No resumable checkpoint in %s; starting a fresh run. Full recovery requires training.save_only_model=false.",
            training.output_dir,
        )
