"""Regression tests for s5cmd region handling in utils.s3_utils.

An explicitly configured ``region`` must win over the compute's ambient
AWS_REGION/AWS_DEFAULT_REGION when invoking s5cmd. Otherwise a checkpoint
upload to a us-east-1 bucket from a us-east-2 pod inherits us-east-2 and
301-redirects (BucketRegionError).
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils import s3_utils


def _capture_s5cmd_env(monkeypatch, call):
    """Force the s5cmd backend, capture the env passed to subprocess.run."""
    captured = {}

    def fake_run(argv, *args, **kwargs):
        captured["argv"] = argv
        captured["env"] = kwargs.get("env")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(s3_utils.shutil, "which", lambda _name: "/usr/bin/s5cmd")
    monkeypatch.setattr(s3_utils.subprocess, "run", fake_run)
    call()
    return captured


def test_upload_dir_region_overrides_ambient_aws_region(tmp_path, monkeypatch):
    (tmp_path / "checkpoint.bin").write_text("weights")
    # The pod's ambient region points at the wrong region for the bucket.
    monkeypatch.setenv("AWS_REGION", "us-east-2")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-2")

    captured = _capture_s5cmd_env(
        monkeypatch,
        lambda: s3_utils.upload_directory_to_s3(
            str(tmp_path),
            "ml-platform-generic",
            "prefix",
            region="us-east-1",
            uploader="s5cmd",
        ),
    )

    assert captured["env"]["AWS_REGION"] == "us-east-1"
    assert captured["env"]["AWS_DEFAULT_REGION"] == "us-east-1"


def test_upload_file_region_overrides_ambient_aws_region(tmp_path, monkeypatch):
    local = tmp_path / "adapter_model.safetensors"
    local.write_text("weights")
    monkeypatch.setenv("AWS_REGION", "us-east-2")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-2")

    captured = _capture_s5cmd_env(
        monkeypatch,
        lambda: s3_utils.upload_file_to_s3(
            str(local),
            "ml-platform-generic",
            "prefix/adapter_model.safetensors",
            region="us-east-1",
            uploader="s5cmd",
        ),
    )

    assert captured["env"]["AWS_REGION"] == "us-east-1"
    assert captured["env"]["AWS_DEFAULT_REGION"] == "us-east-1"


def test_download_region_overrides_ambient_aws_region(tmp_path, monkeypatch):
    monkeypatch.setenv("AWS_REGION", "us-east-2")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-2")
    out = tmp_path / "out"

    captured = {}

    def fake_run(argv, *args, **kwargs):
        captured["argv"] = argv
        captured["env"] = kwargs.get("env")
        (out / "checkpoint.bin").write_text("weights")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(s3_utils.shutil, "which", lambda _name: "/usr/bin/s5cmd")
    monkeypatch.setattr(s3_utils.subprocess, "run", fake_run)

    s3_utils.download_directory_from_s3(
        "s3://ml-platform-generic/prefix",
        str(out),
        region="us-east-1",
        downloader="s5cmd",
    )

    assert captured["env"]["AWS_REGION"] == "us-east-1"
    assert captured["env"]["AWS_DEFAULT_REGION"] == "us-east-1"


def test_download_rejects_s5cmd_error_with_zero_exit(tmp_path, monkeypatch):
    monkeypatch.setattr(s3_utils.shutil, "which", lambda _name: "/usr/bin/s5cmd")
    monkeypatch.setattr(
        s3_utils.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            ["s5cmd"],
            0,
            stdout="",
            stderr="ERROR BucketRegionError: bucket is in 'us-east-1'",
        ),
    )

    with pytest.raises(RuntimeError, match="BucketRegionError"):
        s3_utils.download_directory_from_s3(
            "s3://ml-platform-generic/prefix",
            str(tmp_path / "out"),
            downloader="s5cmd",
        )


def test_download_rejects_empty_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(s3_utils.shutil, "which", lambda _name: "/usr/bin/s5cmd")
    monkeypatch.setattr(
        s3_utils.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(["s5cmd"], 0, stdout="", stderr=""),
    )

    with pytest.raises(RuntimeError, match="downloaded no files"):
        s3_utils.download_directory_from_s3(
            "s3://ml-platform-generic/prefix",
            str(tmp_path / "out"),
            downloader="s5cmd",
        )
