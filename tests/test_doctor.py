"""Doctor runs without a GPU, credential, or network connection."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from utils import doctor


@pytest.fixture
def environment(monkeypatch):
    torch = SimpleNamespace(__version__="test", version=SimpleNamespace(cuda=None), cuda=Mock())
    torch.cuda.is_available.return_value = False
    modules = {"torch": torch, "huggingface_hub": SimpleNamespace(get_token=lambda: None)}
    monkeypatch.setattr(doctor.importlib, "import_module", lambda name: modules[name])
    monkeypatch.setattr(doctor, "version", lambda name: "test")
    monkeypatch.setattr(
        doctor.netrc, "netrc", lambda: SimpleNamespace(authenticators=lambda host: None)
    )
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_MODE", raising=False)
    return modules


def test_cpu_public_resources_are_warnings(environment, capsys):
    assert doctor.main([]) == 0
    output = capsys.readouterr().out
    assert "CUDA unavailable" in output
    assert "WandB not configured" in output
    assert "Public resources need no token" in output


def test_missing_required_env_is_error(environment, monkeypatch, capsys):
    monkeypatch.delenv("TEST_REQUIRED_TOKEN", raising=False)
    assert doctor.main(["--require-env", "TEST_REQUIRED_TOKEN"]) == 1
    assert "Missing environment variable TEST_REQUIRED_TOKEN" in capsys.readouterr().out


def test_cuda_and_credentials_are_redacted(environment, monkeypatch, capsys):
    torch = environment["torch"]
    torch.version.cuda = "test"
    torch.cuda.is_available.return_value = True
    torch.cuda.device_count.return_value = 2
    torch.cuda.get_device_properties.return_value = SimpleNamespace(
        name="test GPU", total_memory=24 * 2**30
    )
    environment["huggingface_hub"].get_token = lambda: "hf-secret"
    monkeypatch.setenv("WANDB_API_KEY", "wandb-secret")
    assert doctor.main([]) == 0
    output = capsys.readouterr().out
    assert "2 GPUs" in output and "24.0 GiB" in output
    assert "secret" not in output


def test_broken_torch_and_missing_packages(environment, monkeypatch):
    def broken(name):
        raise OSError("sensitive local path")

    def missing(name):
        raise doctor.PackageNotFoundError(name)

    monkeypatch.setattr(doctor.importlib, "import_module", broken)
    monkeypatch.setattr(doctor, "version", missing)
    checks = doctor.environment_checks()
    assert any(check.status == "error" and "PyTorch" in check.message for check in checks)
    assert any(check.status == "warning" and "deepspeed" in check.message for check in checks)
    assert all("sensitive" not in check.message for check in checks)
