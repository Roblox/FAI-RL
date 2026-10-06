"""Inference startup logging tests."""

import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import inference.inference as inference_module


def test_inference_logs_installed_fai_rl_version(monkeypatch):
    messages = []
    monkeypatch.setattr(
        inference_module,
        "logger",
        SimpleNamespace(info=messages.append),
    )
    monkeypatch.setattr(
        inference_module,
        "get_package_version",
        lambda: "0.2.23",
    )

    inference_module._log_inference_version()

    assert messages == ["FAI-RL version: 0.2.23"]
