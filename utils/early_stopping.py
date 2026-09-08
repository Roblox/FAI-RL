"""HuggingFace early-stopping helpers for FAI-RL trainers.

Early stopping watches eval loss on a held-out slice of the (already mapped)
training set and stops when it stops improving. GRPO/GSPO do not use this path:
their eval would generate completions and is not wired here.
"""

from __future__ import annotations

from typing import Any


def hold_out_eval_dataset(dataset, ratio: float, seed: int = 42, logger=None):
    """Split ``dataset`` into train/eval for early stopping.

    ``ratio`` is the fraction held out for eval (exclusive of 0 and 1). At least
    one row is kept on each side.
    """
    if not 0.0 < ratio < 1.0:
        raise ValueError(
            f"training.eval_split_ratio must be between 0 and 1 (got {ratio})"
        )
    n = len(dataset)
    if n < 2:
        raise ValueError(
            "early stopping needs at least 2 examples to hold out an eval set"
        )
    eval_n = max(1, min(n - 1, round(n * ratio)))
    split = dataset.train_test_split(test_size=eval_n, seed=seed, shuffle=True)
    train_ds, eval_ds = split["train"], split["test"]
    if logger is not None:
        logger.info(
            "Early stopping: held out %s eval examples (%.1f%% of %s) from train",
            len(eval_ds),
            100.0 * len(eval_ds) / n,
            n,
        )
    return train_ds, eval_ds


def training_args_for_early_stopping(training, base_kwargs=None, logger=None) -> dict[str, Any]:
    """Extra ``TrainingArguments`` / TRL config kwargs when early stopping is on.

    These override the recipe's eval/save settings, so callers must merge them
    over their own kwargs rather than pass both to the config constructor.
    ``base_kwargs`` is what the trainer would otherwise pass, used to decide
    whether the best checkpoint can be reloaded.
    """
    base_kwargs = base_kwargs or {}
    # DeepSpeed and FSDP cannot reload a checkpoint saved without optimizer
    # state, and transformers raises before the first step rather than at the
    # end. Early stopping itself only needs eval_strategy + metric_for_best_model.
    load_best = not (base_kwargs.get("deepspeed") and base_kwargs.get("save_only_model"))
    if not load_best and logger is not None:
        logger.warning(
            "save_only_model is set with DeepSpeed; training will stop early but "
            "the final model is the last checkpoint, not the best one"
        )

    eval_steps = max(1, int(training.eval_steps))
    save_steps = max(1, int(training.save_steps))
    if load_best and save_steps % eval_steps != 0:
        # load_best_model_at_end requires matching save/eval strategies and
        # save_steps to be a multiple of eval_steps.
        adjusted = ((save_steps + eval_steps - 1) // eval_steps) * eval_steps
        if logger is not None:
            logger.warning(
                "save_steps=%s is not a multiple of eval_steps=%s; "
                "using save_steps=%s so load_best_model_at_end is valid",
                save_steps,
                eval_steps,
                adjusted,
            )
        save_steps = adjusted
    metric = getattr(training, "metric_for_best_model", None) or "eval_loss"
    greater = metric not in ("eval_loss", "loss")
    return {
        "eval_strategy": "steps",
        "save_strategy": "steps",
        "eval_steps": eval_steps,
        "save_steps": save_steps,
        "load_best_model_at_end": load_best,
        "metric_for_best_model": metric,
        "greater_is_better": greater,
    }


def build_early_stopping_callback(training) -> Any:
    """Return a transformers ``EarlyStoppingCallback``, or None if disabled."""
    if not getattr(training, "early_stopping", False):
        return None
    from transformers import EarlyStoppingCallback

    return EarlyStoppingCallback(
        early_stopping_patience=max(1, int(training.early_stopping_patience)),
        early_stopping_threshold=float(training.early_stopping_threshold),
    )
