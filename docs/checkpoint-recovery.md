# Checkpoint recovery


```bash
fai-rl-train --recipe recipe.yaml --auto-resume

# An explicit checkpoint takes precedence over --auto-resume:
fai-rl-train --recipe recipe.yaml \
  training.resume_from_checkpoint=outputs/run/checkpoint-500
```

Auto-resume searches immediate `checkpoint-N` directories under
`training.output_dir`, choosing the highest numeric step with readable,
matching `trainer_state.json`, weights (including all indexed shards), and
optimizer/scheduler state. DeepSpeed's `latest` tag and model/optimizer shard
files are also recognized. This is a structural check: tensor corruption,
missing distributed partitions, or recipe incompatibility can still make
restoration fail. Keep the original model, adapter configuration, data, and
distributed topology when resuming.

Without `--auto-resume` or an explicit resume path, training starts as before;
a discovered checkpoint only produces a suggestion. With no valid checkpoint,
auto-resume starts a fresh run and logs that decision. An explicit path is
delegated to HF/TRL unchanged. Loading weights/adapters through
`model.base_model_name` retains its existing behavior and does not restore
optimizer/trainer state.

Caught exceptions and keyboard interruptions report the latest recovery option.
A killed process cannot log a suggestion, but the next invocation can discover
its completed saves. Checkpoint discovery never unpickles files and never deletes
or rewrites checkpoints.

Model-only saves cannot restore optimizer/scheduler state. Set
`training.save_only_model=false` for DPO/GRPO/GSPO recovery and choose suitable
`training.save_steps`. Those algorithms default to model-only saves. SFT/CPT/VLM
retain their existing TRL save defaults. No save defaults are changed by this
feature. Checkpoints deleted locally after S3 upload are not searched remotely.


```bash
python -m pytest -q tests/test_checkpoint_detection.py
```

Tests cover structural discovery and resume dispatch on CPU. GPU/DeepSpeed restoration
still needs end-to-end validation.
