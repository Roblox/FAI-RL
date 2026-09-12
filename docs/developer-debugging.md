# Environment and training diagnostics

## Common Problems & Fixes

Run `fai-rl-doctor` after installation, or `python -m utils.doctor` from a
source checkout. It works even when PyTorch cannot import. It checks local
Python, PyTorch/CUDA, GPU count and total VRAM, package versions, HuggingFace
token discovery, and WandB environment/netrc configuration.

Example output (hardware and versions vary):

```text
## FAI-RL Environment Check

✓ Python 3.12.0 (FAI-RL requires >=3.9; dependencies may require newer Python)
✓ PyTorch 2.13.0
✓ PyTorch CUDA build: 13.0
✓ CUDA detected (8 GPUs)
✓ GPU 0: NVIDIA H100, 80.0 GiB VRAM
✓ transformers 5.8.0 installed
✓ accelerate installed
⚠ deepspeed not installed; install project dependencies (DeepSpeed is optional, via the cuda extra).
⚠ No HuggingFace token; run hf auth login or set HF_TOKEN for gated/private resources. Public resources need no token.
⚠ WandB not configured; run wandb login, set WANDB_API_KEY, or set wandb.enabled=false in the recipe.
```

Doctor makes no network requests, starts no tracking runs, and never prints
credential values. Finding a token does **not** verify its validity, permissions,
or gated-model access. Use `hf auth whoami` to check authentication online.
WandB recipe-level credentials are not inspected by this environment-only command.
Package availability is checked using installation metadata, except PyTorch,
which is imported to inspect the runtime.

No environment variable is mandatory for every FAI-RL workflow. Make deployment
requirements explicit with repeatable flags:

```bash
fai-rl-doctor --require-env HF_TOKEN --require-env WANDB_API_KEY
```

That command reports `✗ Missing environment variable HF_TOKEN` if the variable
is absent (even if a cached HF token exists). Exit status is 1 for errors and 0
for passes/warnings. CUDA absence, optional DeepSpeed, and unconfigured optional
services are warnings. Legacy consoles may replace unsupported status symbols.

| Problem | Fix |
| --- | --- |
| CUDA unavailable or PyTorch import fails | Check `nvidia-smi`, driver compatibility, and the PyTorch build installed in the active environment. CPU/MPS workflows do not require CUDA. |
| HF 401/403 or gated model | Run `hf auth login`, verify access to the model/dataset, and accept any required model terms. |
| WandB login prompt on a worker | Use `wandb login`, `WANDB_API_KEY`, recipe credentials, `WANDB_MODE=offline`, or `wandb.enabled=false`. |
| Missing dataset columns | Read the expected/found columns; correct `data.datasets` mappings. DPO requires chosen/rejected but allows a missing prompt. GRPO/GSPO do not require an answer column. |
| Empty rows | Preflight summarizes missing values; existing trainer filtering remains in place. Entirely empty/unusable datasets fail early. |
| Prompt formatting failure | Match placeholders to `dataset_columns`. Escape literal braces with `{{` and `}}`. |
| Chat template failure | Check tokenizer/processor access, its `chat_template`, and supported roles/media. CPT and flat text SFT do not require chat templates. |
| Invalid YAML/config keys | Use the full key path and suggested name in the error. For example, use `training.learning_rate`, not `training.lr`. |
| Windows UnicodeDecodeError in TRL | Set `PYTHONUTF8=1` before starting Python. |

## Debugging Training Failures

Training preflight loads each raw dataset before trainer construction and checks
required columns, emptiness, missing values across all rows, rendered text
usability for templated datasets, and up to 32 chat samples per dataset.
The trainer reuses these raw datasets. A parent
distributed launcher and its workers each perform preflight; normal HuggingFace
dataset caching still applies. For large datasets, the full missing-value scan
adds CPU startup time.

Chat samples are rendered with a tokenizer/processor before model weights load.
S3-backed models use the existing model download first. VLM checks cover text,
roles, and media placeholders; fetching/decoding images and video and actual
collator execution still happen in the training path. Sample contents are not
logged. Sampling does not guarantee that every later row is well formatted.

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

### Recovering an interrupted run

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

## Before Opening an Issue

Include the following, with tokens, authorization headers, private data, and
sensitive paths removed:

1. `fai-rl-doctor` output and the FAI-RL version or Git commit.
2. OS, launch command, GPU topology, and whether this is a fresh or resumed run.
3. A minimal recipe and a tiny synthetic dataset reproducing the failure.
4. The first traceback, nearby startup summary, and worker logs for distributed failures.
5. For resume problems, the checkpoint file listing and whether model-only saving was enabled.

### Contributor tests

The new diagnostics tests require only Python, pytest, and PyYAML:

```bash
python -m pip install pytest pyyaml
python -m pytest -q tests/test_doctor.py tests/test_dataset_validation.py \
  tests/test_config_validation.py tests/test_checkpoint_detection.py \
  tests/test_training_summary.py
```

Run the entire suite in an environment with project dependencies installed:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
```

The lightweight tests mock GPU/HF checks and use temporary checkpoint files; no
GPU, token, model download, or WandB account is needed.
