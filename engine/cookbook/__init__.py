"""Ready-made training loops built only on the public ``engine`` primitives.

``fai-rl-train`` routes here for ``training.backend: native`` and for the
native-only algorithm ``custom``.
"""

from __future__ import annotations

import importlib

from core.config import ExperimentConfig

NATIVE_ONLY_ALGORITHMS = ("custom",)
NATIVE_BACKEND_ALGORITHMS = ("sft",)


def uses_native_engine(config: ExperimentConfig) -> bool:
    algorithm = (config.training.algorithm or "").lower()
    return algorithm in NATIVE_ONLY_ALGORITHMS or config.training.backend == "native"


def load_entrypoint(import_path: str):
    module_name, _, function_name = import_path.partition(":")
    function = getattr(importlib.import_module(module_name), function_name, None)
    if not callable(function):
        raise ValueError(f"training.entrypoint {import_path!r} does not resolve to a callable")
    return function


def run_recipe(config: ExperimentConfig, logger):
    """Run the native loop selected by ``config.training``."""
    algorithm = (config.training.algorithm or "").lower()
    if algorithm == "sft":
        from .sft import run

        return run(config, logger)
    if algorithm == "custom":
        return load_entrypoint(config.training.entrypoint)(config, logger)
    raise ValueError(
        f"training.backend=native supports {NATIVE_BACKEND_ALGORITHMS + NATIVE_ONLY_ALGORITHMS}, "
        f"got algorithm={algorithm!r}"
    )
