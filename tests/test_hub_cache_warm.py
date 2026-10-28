"""Tests for BaseTrainer's rank-0 HuggingFace Hub cache warm.

The multi-rank race itself needs several processes and the real Hub, so here
we lock down the pieces that prevent it: rank 0 only releases peers once every
file the Hub lists is on disk, and peers only trust a sentinel written for
their own launch.
"""

import logging
import os
import tempfile
import time
from types import SimpleNamespace

import httpx
import huggingface_hub
import pytest
from huggingface_hub.errors import RepositoryNotFoundError

from core.trainer_base import BaseTrainer


FILES = ["config.json", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]


def _model_info(files=FILES):
    return SimpleNamespace(siblings=[SimpleNamespace(rfilename=f) for f in files])


class _FakeApi:
    def __init__(self, info):
        self._info = info

    def model_info(self, repo_id):
        return self._info


def _snapshot(tmp_path, files):
    snapshot_dir = tmp_path / "snapshots" / "abc123"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for name in files:
        (snapshot_dir / name).write_text("x")
    return str(snapshot_dir)


@pytest.fixture
def no_sleep(monkeypatch):
    sleeps = []
    monkeypatch.setattr(time, "sleep", sleeps.append)
    return sleeps


@pytest.fixture
def trainer():
    return SimpleNamespace(logger=logging.getLogger("test_hub_cache_warm"))


def _patch_hub(monkeypatch, snapshots, info=None):
    """Make snapshot_download return each queued result (path, callable, or exception) in turn."""
    calls = []

    def fake_snapshot_download(repo_id):
        calls.append(repo_id)
        result = snapshots.pop(0)
        if isinstance(result, Exception):
            raise result
        return result() if callable(result) else result

    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: _FakeApi(info or _model_info()))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    return calls


def test_complete_snapshot_returns_without_retry(monkeypatch, tmp_path, trainer, no_sleep):
    snapshot_dir = _snapshot(tmp_path, FILES)
    calls = _patch_hub(monkeypatch, [snapshot_dir])

    assert BaseTrainer._download_complete_hub_snapshot(trainer, "org/model") == snapshot_dir
    assert calls == ["org/model"]
    assert no_sleep == []


def test_partial_snapshot_is_retried_until_complete(monkeypatch, tmp_path, trainer, no_sleep):
    partial = _snapshot(tmp_path, FILES[:2])
    calls = _patch_hub(monkeypatch, [partial, lambda: _snapshot(tmp_path, FILES)])

    assert BaseTrainer._download_complete_hub_snapshot(trainer, "org/model") == partial
    assert len(calls) == 2
    assert no_sleep == [15]


def test_persistently_partial_snapshot_names_missing_file(monkeypatch, tmp_path, trainer, no_sleep):
    partial = _snapshot(tmp_path, FILES[:2])
    _patch_hub(monkeypatch, [partial, partial, partial])

    with pytest.raises(RuntimeError, match="model-00002-of-00002.safetensors"):
        BaseTrainer._download_complete_hub_snapshot(trainer, "org/model")
    assert no_sleep == [15, 30]


def test_transient_hub_error_is_retried(monkeypatch, tmp_path, trainer, no_sleep):
    calls = _patch_hub(monkeypatch, [httpx.ReadTimeout("timed out"), lambda: _snapshot(tmp_path, FILES)])

    BaseTrainer._download_complete_hub_snapshot(trainer, "org/model")
    assert len(calls) == 2


def test_missing_repo_is_not_retried(monkeypatch, trainer, no_sleep):
    response = httpx.Response(404, request=httpx.Request("GET", "https://huggingface.co/api/models/org/model"))
    calls = _patch_hub(monkeypatch, [RepositoryNotFoundError("404", response=response)])

    with pytest.raises(RepositoryNotFoundError):
        BaseTrainer._download_complete_hub_snapshot(trainer, "org/model")
    assert len(calls) == 1
    assert no_sleep == []


def test_sentinel_dir_is_per_launch(monkeypatch):
    monkeypatch.setenv("MASTER_ADDR", "10.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29500")
    first = BaseTrainer._hub_warm_sentinel_dir("org/model")
    assert BaseTrainer._hub_warm_sentinel_dir("org/model") == first

    monkeypatch.setenv("MASTER_PORT", "41234")
    assert BaseTrainer._hub_warm_sentinel_dir("org/model") != first
    assert BaseTrainer._hub_warm_sentinel_dir("org/other") != BaseTrainer._hub_warm_sentinel_dir("org/model")


def _warm_trainer(local_rank, downloads):
    return SimpleNamespace(
        local_rank=local_rank,
        logger=logging.getLogger("test_hub_cache_warm"),
        _hub_model_id_for_cache_warm=lambda: "org/model",
        _hub_warm_sentinel_dir=BaseTrainer._hub_warm_sentinel_dir,
        _download_complete_hub_snapshot=downloads.append,
    )


@pytest.fixture
def launch_env(monkeypatch, tmp_path):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setenv("MASTER_ADDR", "10.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29500")
    # setenv (not delenv) so the warm's own os.environ writes are undone too.
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")
    monkeypatch.setattr(tempfile, "tempdir", None)


def test_rank0_verifies_even_when_sentinel_exists(launch_env):
    sentinel_dir = BaseTrainer._hub_warm_sentinel_dir("org/model")
    os.makedirs(sentinel_dir)
    open(os.path.join(sentinel_dir, ".download_complete"), "w").close()
    downloads = []

    BaseTrainer._warm_hub_cache_rank0_first(_warm_trainer(0, downloads))

    assert downloads == ["org/model"]


def test_peer_ignores_sentinel_from_another_launch(launch_env, monkeypatch):
    monkeypatch.setenv("MASTER_PORT", "11111")
    stale_dir = BaseTrainer._hub_warm_sentinel_dir("org/model")
    os.makedirs(stale_dir)
    open(os.path.join(stale_dir, ".download_complete"), "w").close()
    monkeypatch.setenv("MASTER_PORT", "29500")

    class Waited(Exception):
        pass

    def fake_sleep(_):
        raise Waited

    monkeypatch.setattr(time, "sleep", fake_sleep)
    downloads = []

    with pytest.raises(Waited):
        BaseTrainer._warm_hub_cache_rank0_first(_warm_trainer(1, downloads))
    assert downloads == []


def test_peer_proceeds_once_this_launch_is_warm(launch_env):
    sentinel_dir = BaseTrainer._hub_warm_sentinel_dir("org/model")
    os.makedirs(sentinel_dir)
    open(os.path.join(sentinel_dir, ".download_complete"), "w").close()
    downloads = []

    BaseTrainer._warm_hub_cache_rank0_first(_warm_trainer(1, downloads))

    assert downloads == []
    assert os.environ["HF_HUB_OFFLINE"] == "1"
