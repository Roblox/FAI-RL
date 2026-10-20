"""Offline, best-effort environment diagnostics; no training imports required."""

import argparse
import importlib
import netrc
import os
import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import List, Optional, Sequence
from urllib.parse import urlparse


@dataclass
class Check:
    """One diagnostic, with a severity and a safe, actionable message."""

    status: str
    message: str


def environment_checks(required_env: Sequence[str] = ()) -> List[Check]:
    """Inspect local configuration without network calls or printing credentials."""
    checks = [
        Check(
            "pass" if sys.version_info >= (3, 9) else "error",
            f"Python {sys.version.split()[0]} (FAI-RL requires >=3.9; dependencies may require newer Python)",
        )
    ]
    try:
        torch = importlib.import_module("torch")
        checks.append(Check("pass", f"PyTorch {torch.__version__}"))
        checks.append(
            Check(
                "pass" if torch.version.cuda else "warning",
                f"PyTorch CUDA build: {torch.version.cuda or 'none; install a CUDA build for NVIDIA GPUs'}",
            )
        )
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            checks.append(Check("pass", f"CUDA detected ({count} GPUs)"))
            for index in range(count):
                gpu = torch.cuda.get_device_properties(index)
                checks.append(
                    Check(
                        "pass", f"GPU {index}: {gpu.name}, {gpu.total_memory / 2**30:.1f} GiB VRAM"
                    )
                )
        else:
            checks.append(
                Check(
                    "warning",
                    "CUDA unavailable (0 CUDA GPUs); CPU/MPS may work. Check NVIDIA drivers with nvidia-smi.",
                )
            )
    except Exception as exc:
        checks.append(
            Check(
                "error",
                f"PyTorch unavailable or broken ({type(exc).__name__}); install a compatible PyTorch build.",
            )
        )

    for package in ("transformers", "accelerate", "deepspeed"):
        try:
            checks.append(Check("pass", f"{package} {version(package)} installed"))
        except PackageNotFoundError:
            severity = "warning" if package == "deepspeed" else "error"
            checks.append(
                Check(
                    severity,
                    f"{package} not installed; install project dependencies"
                    + (
                        " (DeepSpeed is optional, via the cuda extra)."
                        if package == "deepspeed"
                        else "."
                    ),
                )
            )

    try:
        hub = importlib.import_module("huggingface_hub")
        token = hub.get_token()
        checks.append(
            Check(
                "pass" if token else "warning",
                (
                    "HuggingFace token found (not verified online)."
                    if token
                    else "No HuggingFace token; run hf auth login or set HF_TOKEN for gated/private resources. Public resources need no token."
                ),
            )
        )
    except Exception as exc:
        checks.append(
            Check(
                "warning",
                f"HuggingFace authentication could not be inspected ({type(exc).__name__}); run hf auth whoami.",
            )
        )

    mode = os.environ.get("WANDB_MODE", "online")
    key_present = bool(os.environ.get("WANDB_API_KEY"))
    if not key_present:
        try:
            host = urlparse(os.environ.get("WANDB_BASE_URL", "https://api.wandb.ai")).hostname
            auth = netrc.netrc().authenticators(host)
            key_present = bool(auth and auth[2])
        except (OSError, netrc.NetrcParseError):
            pass
    configured = key_present or mode in {"offline", "disabled"}
    checks.append(
        Check(
            "pass" if configured else "warning",
            (
                f"WandB mode: {mode}; credentials {'found' if key_present else 'not found'} (local check)."
                if configured
                else "WandB not configured; run wandb login, set WANDB_API_KEY, or set wandb.enabled=false in the recipe."
            ),
        )
    )
    for name in required_env:
        checks.append(
            Check(
                "pass" if os.environ.get(name) else "error",
                (
                    f"{name} is set"
                    if os.environ.get(name)
                    else f"Missing environment variable {name}; export it before training."
                ),
            )
        )
    return checks


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Print diagnostics; return 1 on errors and 0 for passes/warnings only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-env",
        action="append",
        default=[],
        metavar="NAME",
        help="Require an environment variable (repeatable); values are never printed",
    )
    args = parser.parse_args(argv)
    checks = environment_checks(args.require_env)
    output = "## FAI-RL Environment Check\n\n" + "\n".join(
        f"{ {'pass': '✓', 'warning': '⚠', 'error': '✗'}[check.status]} {check.message}"
        for check in checks
    )
    # Legacy Windows consoles may not support the Unicode status symbols.
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    print(output.encode(encoding, errors="replace").decode(encoding))
    return int(any(check.status == "error" for check in checks))


if __name__ == "__main__":
    sys.exit(main())
