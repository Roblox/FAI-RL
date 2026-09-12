"""One-time training summaries without GPU queries or optional dependencies."""

from typing import Any


def log_training_summary(config: Any, trainer: Any, dataset: Any, logger: Any) -> None:
    """Report resolved Trainer settings and a qualified static memory estimate."""
    args, model = trainer.args, trainer.model
    algorithm = config.training.algorithm or getattr(config, "_preflight_algorithm", "training")
    if getattr(args, "process_index", 0) != 0:
        return
    parameters = list(model.parameters())
    total = sum(getattr(p, "ds_numel", p.numel()) for p in parameters)
    trainable = sum(getattr(p, "ds_numel", p.numel()) for p in parameters if p.requires_grad)
    if hasattr(model, "get_nb_trainable_parameters"):
        trainable, total = model.get_nb_trainable_parameters()
    micro = args.per_device_train_batch_size
    accumulation = args.gradient_accumulation_steps
    world = args.world_size
    replicas = max(1, getattr(args, "n_gpu", 1)) if world == 1 else world
    precision = "BF16" if args.bf16 else "FP16" if args.fp16 else "FP32"
    lora = bool(getattr(model, "peft_config", None))
    quantized = bool(
        getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False)
    )
    # An unsharded static estimate, not an OOM predictor. Includes Adam-style
    # states and gradients; activations, reference models, and rollouts vary.
    weight_bytes = (
        0.5
        if getattr(model, "is_loaded_in_4bit", False)
        else (
            1
            if quantized
            else {
                "float16": 2,
                "bfloat16": 2,
                "float32": 4,
                "float64": 8,
            }.get(config.model.torch_dtype, 4)
        )
    )
    estimate = (total * weight_bytes + trainable * 12) / 2**30
    logger.info(
        "\n## Training Configuration\nModel: %s\nMethod: %s\nPrecision: %s\n"
        "LoRA: %s | QLoRA: %s\nTrainable params: %s / %s\nTraining rows: %s\n"
        "Effective batch size: %d (%d per device x %d accumulation x %d replicas)\n"
        "Estimated static model + gradient/Adam memory: %.2f GiB (unsharded). "
        "Excludes activations, quantization metadata, reference models, rollout/KV caches, "
        "and allocator overhead; sharding/offload change per-GPU requirements.",
        config.model.base_model_name,
        algorithm.upper(),
        precision,
        "Enabled" if lora else "Disabled",
        "Enabled" if lora and quantized else "Disabled",
        f"{trainable:,}",
        f"{total:,}",
        len(dataset),
        micro * accumulation * replicas,
        micro,
        accumulation,
        replicas,
        estimate,
    )
    if algorithm.lower() in {"grpo", "gspo"}:
        logger.info(
            "RL batch counts generated sequences; num_generations=%s, steps_per_generation=%s.",
            getattr(args, "num_generations", "default"),
            getattr(args, "steps_per_generation", "default"),
        )


def run_trainer(owner: Any) -> Any:
    """Log once, then delegate to HF/TRL with opt-in full-state resume."""
    log_training_summary(owner.config, owner.trainer, owner.train_dataset, owner.logger)
    checkpoint = getattr(owner.config.training, "resume_from_checkpoint", None)
    if checkpoint:
        return owner.trainer.train(resume_from_checkpoint=checkpoint)
    return owner.trainer.train()
