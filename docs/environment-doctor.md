# Environment doctor

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


## Before Opening an Issue

Include doctor output, your FAI-RL commit/version, OS, launch command, and the
first traceback. Remove credentials, sensitive paths, and private dataset values.
Provide a small synthetic example when possible.

## Tests

```bash
python -m pytest -q tests/test_doctor.py
python -m utils.doctor
```
