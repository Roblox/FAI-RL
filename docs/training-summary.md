# Training startup summary

The one-time training summary reports the resolved model, method, precision,
LoRA/QLoRA status, trainable/total parameters, training rows, and effective batch:

```text
effective batch = per-device batch × gradient accumulation × training replicas
```

Distributed jobs emit the summary only on global rank 0. For GRPO/GSPO, batch
counts refer to generated sequences; `num_generations` and generation scheduling
also affect the number of distinct prompts. The memory estimate covers unsharded
weights plus approximate gradients/Adam state. It excludes activations, reference
models, quantization metadata, KV/rollout caches, and allocator overhead. It is
not a per-GPU capacity guarantee; sharding and offload change placement.

For OOM failures, reduce batch size or sequence length, consider LoRA/QLoRA,
and inspect generation settings for RL. Existing detailed diagnostics remain
available; `FAI_RL_DISABLE_DEBUG_CALLBACK=1` disables the existing debug callback
without suppressing the new startup summary.


```bash
python -m pytest -q tests/test_training_summary.py
```

Tests use fake parameters and training arguments; no GPU or model download is needed.
