"""Shared supervised training loop for cookbook recipes.

A recipe supplies a ``SupervisedObjective`` (its rows and how a batch of rows
becomes a ``forward_backward`` call). ``run_supervised_loop`` does the rest with
the same ``TrainingConfig`` knobs the TRL trainers use: global batching across
ranks, linear warmup/decay, logging, held-out eval loss, early stopping,
``checkpoint-<step>`` saves with S3 upload, and resume.
"""

from __future__ import annotations

import math
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch

from core.config import ExperimentConfig
from utils.early_stopping import hold_out_eval_dataset
from utils.s3_utils import upload_directory_to_s3

from .. import distributed
from ..training_client import TrainingClient
from ..types import AdamParams

# HF TrainingArguments defaults, kept for parity with the TRL backend.
MAX_GRAD_NORM = 1.0
SHUFFLE_SEED = 42
BEST_MODEL_DIR = "best"


class SupervisedObjective(ABC):
    """Recipe-specific pieces plugged into ``run_supervised_loop``.

    Each optimizer step normalizes by ``count_units`` summed over the global
    batch (all ranks), so the gradient summed across ranks is a per-unit mean,
    like HF's token-averaged loss.
    """

    def load_rows(self):
        """All prepared training rows (an HF ``Dataset``), before eval hold-out."""
        raise NotImplementedError

    def load_splits(self, training, logger):
        """``(train_rows, eval_rows_or_None)``; override to hold out before
        expanding rows (e.g. whole conversations before splitting turns)."""
        rows = self.load_rows()
        if not training.eval_enabled:
            return rows, None
        return hold_out_eval_dataset(rows, training.eval_split_ratio, logger=logger)

    @abstractmethod
    def count_units(self, rows: List[dict]) -> float:
        """Normalization units in ``rows`` (e.g. trained tokens, or pairs)."""

    @abstractmethod
    def train_step(self, client: TrainingClient, rows: List[dict], scale: float) -> float:
        """Accumulate gradients of ``scale * loss_sum(rows)``; return that value."""

    @abstractmethod
    def eval_step(self, client: TrainingClient, rows: List[dict]) -> float:
        """Unscaled loss summed over ``rows``, without gradients."""


@dataclass
class LoopResult:
    global_step: int
    log_history: List[Dict[str, Any]] = field(default_factory=list)
    best_eval_loss: Optional[float] = None
    stopped_early: bool = False


def linear_schedule(step: int, total_steps: int, warmup_steps: int, peak_lr: float) -> float:
    """HF ``get_linear_schedule_with_warmup`` learning rate for the 0-based ``step``."""
    if step < warmup_steps:
        return peak_lr * step / max(1, warmup_steps)
    return peak_lr * max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))


def _rows_at(rows, indices: Sequence[int]) -> List[dict]:
    return [rows[int(index)] for index in indices]


def _chunks(items: Sequence, size: int) -> List[Sequence]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def _epoch_order(num_rows: int, epoch: int) -> List[int]:
    generator = torch.Generator().manual_seed(SHUFFLE_SEED + epoch)
    return torch.randperm(num_rows, generator=generator).tolist()


class _Reporter:
    """Log records to the logger, wandb (main process) and ``history``."""

    def __init__(self, config: ExperimentConfig, logger):
        self.logger = logger
        self.use_wandb = config.wandb.enabled and distributed.is_main_process()
        self.history: List[Dict[str, Any]] = []

    def __call__(self, record: Dict[str, Any]) -> None:
        self.history.append(record)
        if not distributed.is_main_process():
            return
        self.logger.info(
            " ".join(
                f"{key}={value:.6g}" if isinstance(value, float) else f"{key}={value}"
                for key, value in record.items()
            )
        )
        if self.use_wandb:
            import wandb

            prefix = "eval" if "eval_loss" in record else "train"
            wandb.log(
                {
                    f"{prefix}/{key.removeprefix('eval_')}": value
                    for key, value in record.items()
                    if key != "step"
                },
                step=record["step"],
            )


def upload_to_s3(config: ExperimentConfig, local_dir: str, name: str, *, checkpoint: bool, logger) -> None:
    s3 = config.s3
    wanted = s3.upload_checkpoints if checkpoint else s3.upload_final_model
    if not (s3.enabled and wanted and distributed.is_main_process()):
        return
    destination = f"{s3.prefix}/{name}" if s3.prefix else name
    logger.info("Uploading %s -> s3://%s/%s", local_dir, s3.bucket, destination)
    upload_directory_to_s3(
        local_dir,
        s3.bucket,
        destination,
        region=s3.region,
        endpoint_url=s3.endpoint_url,
        delete_local=checkpoint and s3.delete_local_after_upload,
        uploader=s3.uploader,
    )


def _evaluate(client: TrainingClient, objective: SupervisedObjective, eval_rows, batch_size: int) -> float:
    local_indices = distributed.shard_for_rank(list(range(len(eval_rows))))
    loss_sum = 0.0
    units = 0.0
    for indices in _chunks(local_indices, batch_size):
        rows = _rows_at(eval_rows, indices)
        loss_sum += objective.eval_step(client, rows)
        units += objective.count_units(rows)
    loss_sum = distributed.all_reduce_sum(loss_sum)
    units = distributed.all_reduce_sum(units)
    return loss_sum / units if units else float("nan")


def run_supervised_loop(
    client: TrainingClient,
    config: ExperimentConfig,
    objective: SupervisedObjective,
    logger,
) -> LoopResult:
    """Train ``client`` on ``objective`` following ``config.training``."""
    training = config.training
    rows, eval_rows = objective.load_splits(training, logger)
    if training.early_stopping and training.metric_for_best_model != "eval_loss":
        logger.warning(
            "Native loops track eval_loss only; ignoring metric_for_best_model=%s",
            training.metric_for_best_model,
        )

    num_rows = len(rows)
    if num_rows == 0:
        raise ValueError("No training rows after data preparation")
    local_batch = training.per_device_train_batch_size
    global_batch = local_batch * training.gradient_accumulation_steps * distributed.world_size()
    if training.dataloader_drop_last:
        steps_per_epoch = num_rows // global_batch
        if steps_per_epoch == 0:
            raise ValueError(
                f"{num_rows} training rows is fewer than one global batch ({global_batch}) "
                "with dataloader_drop_last=true"
            )
    else:
        steps_per_epoch = math.ceil(num_rows / global_batch)
    total_steps = (
        training.max_steps if training.max_steps > 0
        else steps_per_epoch * training.num_train_epochs
    )

    if training.resume_from_checkpoint:
        client.load_state(training.resume_from_checkpoint, with_optimizer=True)
    start_step = client.step
    logger.info(
        "Native training: %d rows, global batch %d, %d steps/epoch, %d total steps (starting at %d)",
        num_rows, global_batch, steps_per_epoch, total_steps, start_step,
    )

    report = _Reporter(config, logger)
    result = LoopResult(global_step=start_step)
    loss_window: List[float] = []
    grad_norm = float("nan")
    learning_rate = 0.0
    bad_evals = 0
    order_epoch, order = -1, []

    def flush_train_log(global_step: int) -> None:
        nonlocal loss_window
        if loss_window:
            report({
                "step": global_step,
                "epoch": global_step / steps_per_epoch,
                "loss": sum(loss_window) / len(loss_window),
                "grad_norm": grad_norm,
                "learning_rate": learning_rate,
            })
            loss_window = []

    for step in range(start_step, total_steps):
        epoch, position = divmod(step, steps_per_epoch)
        if epoch != order_epoch:
            order_epoch, order = epoch, _epoch_order(num_rows, epoch)
        batch_indices = order[position * global_batch:(position + 1) * global_batch]
        local_rows = _rows_at(rows, distributed.shard_for_rank(batch_indices))

        units = distributed.all_reduce_sum(objective.count_units(local_rows))
        scale = 1.0 / units if units > 0 else 0.0
        local_loss = 0.0
        for chunk in _chunks(local_rows, local_batch):
            local_loss += objective.train_step(client, list(chunk), scale)

        learning_rate = linear_schedule(step, total_steps, training.warmup_steps, training.learning_rate)
        optim = client.optim_step(AdamParams(learning_rate=learning_rate, grad_clip_norm=MAX_GRAD_NORM))
        grad_norm = optim.metrics["grad_norm"]
        loss_window.append(distributed.all_reduce_sum(local_loss))
        global_step = step + 1
        result.global_step = global_step

        if global_step % training.logging_steps == 0:
            flush_train_log(global_step)

        if eval_rows is not None and global_step % training.eval_steps == 0:
            eval_loss = _evaluate(client, objective, eval_rows, training.per_device_eval_batch_size)
            report({"step": global_step, "eval_loss": eval_loss})
            best = result.best_eval_loss
            if best is None or best - eval_loss > training.early_stopping_threshold:
                result.best_eval_loss = eval_loss
                bad_evals = 0
                if training.early_stopping:
                    client.save_weights_for_sampler(os.path.join(training.output_dir, BEST_MODEL_DIR))
            else:
                bad_evals += 1
                if training.early_stopping and bad_evals >= training.early_stopping_patience:
                    result.stopped_early = True

        if training.save_steps > 0 and global_step % training.save_steps == 0:
            name = f"checkpoint-{global_step}"
            path = os.path.join(training.output_dir, name)
            if training.save_only_model:
                client.save_weights_for_sampler(path)
            else:
                client.save_state(path)
            upload_to_s3(config, path, name, checkpoint=True, logger=logger)

        if result.stopped_early:
            logger.info(
                "Early stopping at step %d: eval_loss did not improve for %d evaluations",
                global_step, bad_evals,
            )
            break

    flush_train_log(result.global_step)
    result.log_history = report.history
    return result


def save_final_model(client: TrainingClient, config: ExperimentConfig, result: LoopResult, logger) -> str:
    """Write the final (or best, with early stopping) weights to ``output_dir``."""
    training = config.training
    best_dir = os.path.join(training.output_dir, BEST_MODEL_DIR)
    if training.early_stopping and result.best_eval_loss is not None and os.path.isdir(best_dir):
        logger.info("Restoring best model (eval_loss=%.6g) from %s", result.best_eval_loss, best_dir)
        client.load_state(best_dir, with_optimizer=False)
    path = client.save_weights_for_sampler(training.output_dir)
    upload_to_s3(config, path, "final", checkpoint=False, logger=logger)
    logger.info("Saved final model to %s", path)
    return path


def run_supervised(
    config: ExperimentConfig,
    logger,
    make_objective: Callable[[ExperimentConfig, Any, Any], SupervisedObjective],
) -> LoopResult:
    """Load the model, train ``make_objective(config, tokenizer, logger)``, save."""
    from ..model_loading import build_training_client

    with build_training_client(config, logger=logger) as client:
        objective = make_objective(config, client.get_tokenizer(), logger)
        result = run_supervised_loop(client, config, objective, logger)
        save_final_model(client, config, result, logger)
    return result
