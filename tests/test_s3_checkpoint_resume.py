"""A restarted job restores its own newest uploaded checkpoint from S3.

Kueue preemption and node loss restart a job on fresh nodes, so the
checkpoints in ``training.output_dir`` are gone and only the S3 copies remain.
"""

import io
import json
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from botocore.exceptions import ClientError

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils import s3_utils
from utils.checkpoint_restore import restore_checkpoint_from_s3, run_fingerprint
from utils.checkpoint_utils import TRAINING_STATE_PATTERNS, select_resume_checkpoint
from utils.s3_utils import UPLOAD_MANIFEST_NAME, S3UploadCallback

BUCKET = "ml-object-registry"
PREFIX = "outputs/run"
FINGERPRINT = "run-a"


class FakeS3:
    """The boto3 calls FAI-RL makes, backed by a dict."""

    def __init__(self):
        self.objects: dict[tuple[str, str], bytes] = {}
        self.downloads: list[str] = []
        self._lock = threading.Lock()

    def upload_file(self, local, bucket, key):
        with self._lock:
            self.objects[(bucket, key)] = Path(local).read_bytes()

    def download_file(self, bucket, key, local):
        with self._lock:
            self.downloads.append(key)
            Path(local).write_bytes(self.objects[(bucket, key)])

    def get_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey", "Message": Key}}, "GetObject")
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}

    def get_paginator(self, _name):
        return self

    def paginate(self, Bucket, Prefix="", Delimiter=None):
        keys = sorted(k for b, k in self.objects if b == Bucket and k.startswith(Prefix))
        if not Delimiter:
            yield {"Contents": [{"Key": k} for k in keys]}
            return
        rests = [k[len(Prefix):] for k in keys]
        yield {
            "Contents": [{"Key": Prefix + r} for r in rests if Delimiter not in r],
            "CommonPrefixes": [
                {"Prefix": Prefix + p + Delimiter}
                for p in sorted({r.split(Delimiter, 1)[0] for r in rests if Delimiter in r})
            ],
        }

    def keys(self, prefix=""):
        return sorted(k for _b, k in self.objects if k.startswith(prefix))


@pytest.fixture
def s3(monkeypatch):
    fake = FakeS3()
    monkeypatch.setattr(s3_utils, "_get_s3_client", lambda *_args, **_kwargs: fake)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    return fake


def write_checkpoint(
    output_dir: Path, step: int, full=True, world_size=1, deepspeed=False, best_step=None
) -> Path:
    path = output_dir / f"checkpoint-{step}"
    path.mkdir(parents=True)
    state = {"global_step": step}
    if best_step is not None:
        # Trainer records the path under the recipe's output_dir, which the
        # restarted job recreates empty.
        state["best_model_checkpoint"] = f"models/job-a/checkpoint-{best_step}"
    (path / "trainer_state.json").write_text(json.dumps(state))
    (path / "adapter_model.safetensors").write_bytes(b"weights-%d" % step)
    if not full:
        return path
    if deepspeed:
        (path / "latest").write_text(f"global_step{step}")
        shards = path / f"global_step{step}"
        shards.mkdir()
        (shards / "mp_rank_00_model_states.pt").write_bytes(b"model")
        for rank in range(world_size):
            (shards / f"zero_pp_rank_{rank}_mp_rank_00_optim_states.pt").write_bytes(b"optim")
    else:
        (path / "optimizer.pt").write_bytes(b"optimizer")
        (path / "scheduler.pt").write_bytes(b"scheduler")
    return path


def save_step(callback: S3UploadCallback, output_dir: Path, step: int, world_size=1, **kwargs):
    """Run one Trainer save: write the checkpoint, call on_save, wait for the upload."""
    write_checkpoint(output_dir, step, world_size=world_size, **kwargs)
    args = SimpleNamespace(output_dir=str(output_dir), world_size=world_size)
    state = SimpleNamespace(
        global_step=step,
        is_world_process_zero=True,
        log_history=[{"step": step, "eval_loss": 1.0 / step}],
        best_metric=None,
        best_global_step=None,
        best_model_checkpoint=None,
    )
    callback.on_save(args, state, control=None)
    callback._wait_for_uploads()


def uploader(fingerprint=FINGERPRINT) -> S3UploadCallback:
    return S3UploadCallback(
        bucket=BUCKET, prefix=PREFIX, uploader="boto3", run_fingerprint=fingerprint
    )


def restarted_config(output_dir: Path, fingerprint=FINGERPRINT, **s3_overrides):
    s3 = dict(
        enabled=True,
        bucket=BUCKET,
        prefix=PREFIX,
        region=None,
        endpoint_url=None,
        upload_checkpoints=True,
        uploader="boto3",
        run_fingerprint=fingerprint,
    )
    s3.update(s3_overrides)
    return SimpleNamespace(
        training=SimpleNamespace(output_dir=str(output_dir), resume_from_checkpoint=None),
        s3=SimpleNamespace(**s3),
    )


def test_resumable_upload_ends_with_a_manifest_of_every_file(s3, tmp_path):
    save_step(uploader(), tmp_path / "run", 10)

    manifest = json.loads(s3.objects[(BUCKET, f"{PREFIX}/checkpoint-10/{UPLOAD_MANIFEST_NAME}")])
    assert manifest["global_step"] == 10
    assert manifest["world_size"] == 1
    assert manifest["run_fingerprint"] == FINGERPRINT
    assert manifest["files"] == {
        "adapter_model.safetensors": len(b"weights-10"),
        "eval_metrics.json": (tmp_path / "run/checkpoint-10/eval_metrics.json").stat().st_size,
        "optimizer.pt": len(b"optimizer"),
        "scheduler.pt": len(b"scheduler"),
        "trainer_state.json": len(json.dumps({"global_step": 10})),
    }


def test_model_only_checkpoint_is_uploaded_without_a_manifest(s3, tmp_path):
    save_step(uploader(), tmp_path / "run", 10, full=False)

    assert f"{PREFIX}/checkpoint-10/adapter_model.safetensors" in s3.keys()
    assert f"{PREFIX}/checkpoint-10/{UPLOAD_MANIFEST_NAME}" not in s3.keys()


def test_deepspeed_checkpoint_missing_other_nodes_shards_gets_no_manifest(s3, tmp_path):
    # Two nodes without a shared filesystem: rank 0 only sees its own node's shard.
    write_checkpoint(tmp_path / "run", 10, deepspeed=True, world_size=1)
    callback = uploader()
    args = SimpleNamespace(output_dir=str(tmp_path / "run"), world_size=2)
    state = SimpleNamespace(
        global_step=10, is_world_process_zero=True, log_history=[],
        best_metric=None, best_global_step=None, best_model_checkpoint=None,
    )
    callback.on_save(args, state, control=None)
    callback._wait_for_uploads()

    assert f"{PREFIX}/checkpoint-10/latest" in s3.keys()
    assert f"{PREFIX}/checkpoint-10/{UPLOAD_MANIFEST_NAME}" not in s3.keys()


def test_failed_upload_writes_no_manifest(s3, tmp_path, monkeypatch):
    def broken_upload(**_kwargs):
        raise RuntimeError("s5cmd upload failed")

    monkeypatch.setattr(s3_utils, "upload_directory_to_s3", broken_upload)
    with pytest.raises(RuntimeError, match="upload"):
        save_step(uploader(), tmp_path / "run", 10)
    assert not [k for k in s3.keys() if k.endswith(UPLOAD_MANIFEST_NAME)]


def test_restart_restores_newest_complete_checkpoint(s3, tmp_path):
    callback = uploader()
    save_step(callback, tmp_path / "run", 10)
    save_step(callback, tmp_path / "run", 20)
    # Preempted while uploading checkpoint-30: its files landed but not its manifest.
    save_step(callback, tmp_path / "run", 30)
    del s3.objects[(BUCKET, f"{PREFIX}/checkpoint-30/{UPLOAD_MANIFEST_NAME}")]

    fresh = tmp_path / "fresh"
    config = restarted_config(fresh)
    restored = restore_checkpoint_from_s3(config, Mock())

    assert restored == str(fresh / "checkpoint-20")
    assert (fresh / "checkpoint-20" / "optimizer.pt").read_bytes() == b"optimizer"
    assert not (fresh / "checkpoint-30").exists()
    select_resume_checkpoint(config.training, True, Mock())
    assert config.training.resume_from_checkpoint == str(fresh / "checkpoint-20")


def test_restart_also_restores_the_best_checkpoint_the_state_names(s3, tmp_path):
    # load_best_model_at_end reloads it after the last step; with DeepSpeed a
    # missing best checkpoint fails the run there.
    callback = uploader()
    save_step(callback, tmp_path / "run", 10)
    save_step(callback, tmp_path / "run", 20, best_step=10)
    save_step(callback, tmp_path / "run", 30, best_step=10)

    fresh = tmp_path / "fresh"
    restored = restore_checkpoint_from_s3(restarted_config(fresh), Mock())

    assert restored == str(fresh / "checkpoint-30")
    assert (fresh / "checkpoint-10" / "optimizer.pt").read_bytes() == b"optimizer"
    assert not (fresh / "checkpoint-20").exists()
    state = json.loads((fresh / "checkpoint-30" / "trainer_state.json").read_text())
    assert state["best_model_checkpoint"] == str(fresh / "checkpoint-10")


def test_best_checkpoint_that_cannot_be_restored_is_dropped_from_the_state(s3, tmp_path):
    callback = uploader()
    save_step(callback, tmp_path / "run", 10)
    save_step(callback, tmp_path / "run", 20, best_step=10)
    del s3.objects[(BUCKET, f"{PREFIX}/checkpoint-10/{UPLOAD_MANIFEST_NAME}")]

    fresh = tmp_path / "fresh"
    assert restore_checkpoint_from_s3(restarted_config(fresh), Mock()) == str(fresh / "checkpoint-20")

    assert not (fresh / "checkpoint-10").exists()
    state = json.loads((fresh / "checkpoint-20" / "trainer_state.json").read_text())
    assert state["best_model_checkpoint"] is None
    assert state["global_step"] == 20


def _train_tiny(output_dir: Path, callbacks, resume_from_checkpoint=None):
    from transformers import Trainer, TrainingArguments

    from conftest import make_tiny_lm

    model, tokenizer = make_tiny_lm()
    ids = tokenizer("Say hello. hello world " * 4)["input_ids"][:16]
    args = TrainingArguments(
        output_dir=str(output_dir),
        max_steps=6,
        save_steps=2,
        save_only_model=False,
        per_device_train_batch_size=2,
        learning_rate=1e-3,
        logging_steps=1,
        use_cpu=True,
        report_to=[],
        disable_tqdm=True,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=[{"input_ids": ids, "labels": ids}] * 32,
        callbacks=callbacks,
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    return trainer


def test_trainer_resumes_from_a_restored_checkpoint(s3, tmp_path, force_cpu):
    from transformers import TrainerCallback

    class PreemptAfterStep4(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == 4:
                control.should_training_stop = True

    run = tmp_path / "run"
    first = S3UploadCallback(bucket=BUCKET, prefix=PREFIX, uploader="boto3",
                             upload_final_model=False, run_fingerprint=FINGERPRINT)
    _train_tiny(run, [first, PreemptAfterStep4()])
    assert f"{PREFIX}/checkpoint-4/{UPLOAD_MANIFEST_NAME}" in s3.keys()

    # The restarted job sees the same output_dir, empty.
    shutil.rmtree(run)
    config = restarted_config(run)
    restored = restore_checkpoint_from_s3(config, Mock())
    assert restored == str(run / "checkpoint-4")

    class RecordSteps(TrainerCallback):
        steps: list = []

        def on_step_end(self, args, state, control, **kwargs):
            self.steps.append(state.global_step)

    resumed = S3UploadCallback(bucket=BUCKET, prefix=PREFIX, uploader="boto3",
                               upload_final_model=False, run_fingerprint=FINGERPRINT)
    trainer = _train_tiny(run, [resumed, RecordSteps()], resume_from_checkpoint=restored)

    assert RecordSteps.steps == [5, 6]
    assert trainer.state.global_step == 6
    assert f"{PREFIX}/checkpoint-6/{UPLOAD_MANIFEST_NAME}" in s3.keys()


def test_another_runs_checkpoints_are_not_restored(s3, tmp_path):
    save_step(uploader(fingerprint="run-b"), tmp_path / "run", 10)

    fresh = tmp_path / "fresh"
    assert restore_checkpoint_from_s3(restarted_config(fresh), Mock()) is None
    assert not list(fresh.glob("checkpoint-*"))
    assert s3.downloads == []


def test_checkpoint_from_another_world_size_is_not_restored(s3, tmp_path, monkeypatch):
    save_step(uploader(), tmp_path / "run", 10, world_size=8, deepspeed=True)

    monkeypatch.setenv("WORLD_SIZE", "4")
    assert restore_checkpoint_from_s3(restarted_config(tmp_path / "fresh"), Mock()) is None
    monkeypatch.setenv("WORLD_SIZE", "8")
    restored = restore_checkpoint_from_s3(restarted_config(tmp_path / "fresh"), Mock())
    assert restored == str(tmp_path / "fresh" / "checkpoint-10")


def test_download_that_does_not_match_the_manifest_is_discarded(s3, tmp_path):
    callback = uploader()
    save_step(callback, tmp_path / "run", 10)
    save_step(callback, tmp_path / "run", 20)
    s3.objects[(BUCKET, f"{PREFIX}/checkpoint-20/optimizer.pt")] = b"truncated"[:3]

    fresh = tmp_path / "fresh"
    restored = restore_checkpoint_from_s3(restarted_config(fresh), Mock())

    assert restored == str(fresh / "checkpoint-10")
    assert sorted(p.name for p in fresh.iterdir() if not p.name.startswith(".")) == ["checkpoint-10"]


def test_newer_local_checkpoint_skips_the_download(s3, tmp_path):
    save_step(uploader(), tmp_path / "run", 10)
    write_checkpoint(tmp_path / "fresh", 20)

    restored = restore_checkpoint_from_s3(restarted_config(tmp_path / "fresh"), Mock())

    assert restored == str(tmp_path / "fresh" / "checkpoint-20")
    assert s3.downloads == []


@pytest.mark.parametrize(
    "overrides",
    [{"enabled": False}, {"bucket": ""}, {"upload_checkpoints": False}],
)
def test_restore_needs_uploaded_checkpoints(s3, tmp_path, overrides):
    save_step(uploader(), tmp_path / "run", 10)
    assert restore_checkpoint_from_s3(restarted_config(tmp_path / "fresh", **overrides), Mock()) is None
    assert s3.downloads == []


def test_explicit_resume_path_is_left_alone(s3, tmp_path):
    save_step(uploader(), tmp_path / "run", 10)
    config = restarted_config(tmp_path / "fresh")
    config.training.resume_from_checkpoint = "elsewhere/checkpoint-5"

    assert restore_checkpoint_from_s3(config, Mock()) is None
    assert s3.downloads == []


def test_ranks_on_one_node_download_once(s3, tmp_path):
    save_step(uploader(), tmp_path / "run", 10)
    fresh = tmp_path / "fresh"
    results = []

    def rank():
        results.append(restore_checkpoint_from_s3(restarted_config(fresh), Mock()))

    threads = [threading.Thread(target=rank) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results == [str(fresh / "checkpoint-10")] * 4
    assert s3.downloads.count(f"{PREFIX}/checkpoint-10/optimizer.pt") == 1


def test_resumed_run_keeps_earlier_eval_losses_in_the_index(s3, tmp_path):
    callback = uploader()
    save_step(callback, tmp_path / "run", 10)
    save_step(callback, tmp_path / "run", 20)
    save_step(callback, tmp_path / "run", 30)

    resumed = uploader()
    state = SimpleNamespace(global_step=20, is_world_process_zero=True)
    resumed.on_train_begin(SimpleNamespace(), state, control=None)
    save_step(resumed, tmp_path / "resumed", 40)

    index = json.loads(s3.objects[(BUCKET, f"{PREFIX}/checkpoint_eval.json")])
    assert index["eval_loss"] == {"10": 0.1, "20": 0.05, "40": 0.025}


def test_fresh_run_does_not_read_the_index(s3, tmp_path):
    callback = uploader()
    callback.on_train_begin(SimpleNamespace(), SimpleNamespace(global_step=0, is_world_process_zero=True), None)
    assert callback._eval_index == {}


def _config(**training):
    from core.config import DataConfig, ExperimentConfig, ModelConfig, S3Config, TrainingConfig, WandbConfig

    return ExperimentConfig(
        model=ModelConfig(base_model_name="Qwen/Qwen3-4B"),
        data=DataConfig(datasets=[]),
        training=TrainingConfig(output_dir="models/job-a", algorithm="sft", **training),
        wandb=WandbConfig(),
        s3=S3Config(enabled=True, bucket=BUCKET, prefix=PREFIX, uploader="boto3"),
    )


def test_fingerprint_identifies_the_recipe_not_the_attempt():
    base = run_fingerprint(_config())
    assert run_fingerprint(_config()) == base
    assert run_fingerprint(_config(learning_rate=2e-5)) != base

    attempt = _config()
    attempt.training.resume_from_checkpoint = "models/job-a/checkpoint-10"
    attempt.training.deepspeed_config = "/opt/fai-rl/configs/deepspeed/zero1_config.json"
    attempt.s3.run_fingerprint = "set-by-an-earlier-call"
    assert run_fingerprint(attempt) == base


def test_prepare_resume_restores_before_selecting(s3, tmp_path, monkeypatch):
    from trainers import train

    config = _config()
    config.training.output_dir = str(tmp_path / "fresh")
    callback = uploader(fingerprint=run_fingerprint(config))
    save_step(callback, tmp_path / "run", 10)

    train.prepare_resume(config, auto_resume=True, logger=Mock())

    assert config.s3.run_fingerprint == run_fingerprint(config)
    assert config.training.resume_from_checkpoint == str(tmp_path / "fresh" / "checkpoint-10")


def test_prepare_resume_without_auto_resume_never_reads_s3(s3, tmp_path):
    from trainers import train

    config = _config()
    config.training.output_dir = str(tmp_path / "fresh")
    save_step(uploader(fingerprint=run_fingerprint(config)), tmp_path / "run", 10)

    train.prepare_resume(config, auto_resume=False, logger=Mock())

    assert config.training.resume_from_checkpoint is None
    assert s3.downloads == []


def test_weights_download_skips_training_state(s3, tmp_path):
    save_step(uploader(), tmp_path / "run", 10, deepspeed=True)
    out = tmp_path / "weights"

    s3_utils.download_directory_from_s3(
        f"s3://{BUCKET}/{PREFIX}/checkpoint-10", str(out), downloader="boto3",
        exclude=TRAINING_STATE_PATTERNS,
    )

    assert sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()) == [
        "adapter_model.safetensors",
        "eval_metrics.json",
        "trainer_state.json",
        UPLOAD_MANIFEST_NAME,
    ]


def test_weights_download_with_s5cmd_copies_only_kept_keys(s3, tmp_path, monkeypatch):
    save_step(uploader(), tmp_path / "run", 10)
    out = tmp_path / "weights"
    captured = {}

    def fake_run(argv, *_args, **kwargs):
        captured["argv"] = argv
        captured["input"] = kwargs.get("input", "")
        for line in captured["input"].splitlines():
            _cp, _src, dst = line.split(" ")
            Path(dst.strip("'")).write_bytes(b"x")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(s3_utils.shutil, "which", lambda _name: "/usr/bin/s5cmd")
    monkeypatch.setattr(s3_utils.subprocess, "run", fake_run)
    s3_utils.download_directory_from_s3(
        f"s3://{BUCKET}/{PREFIX}/checkpoint-10", str(out), region="us-east-1",
        downloader="s5cmd", exclude=TRAINING_STATE_PATTERNS,
    )

    assert captured["argv"][-1] == "run"
    copied = sorted(line.split(" ")[1].strip("'").rsplit("/", 1)[1] for line in captured["input"].splitlines())
    assert copied == ["adapter_model.safetensors", "eval_metrics.json", "trainer_state.json", UPLOAD_MANIFEST_NAME]


def test_inference_model_download_skips_training_state(monkeypatch):
    from inference import inference

    captured = {}
    monkeypatch.setattr(
        "utils.s3_utils.download_directory_from_s3",
        lambda **kwargs: captured.update(kwargs) or Path(kwargs["local_dir"], "config.json").write_text("{}"),
    )
    config = SimpleNamespace(model_path="s3://bucket/run/checkpoint-10", s3_region=None, s3_endpoint_url=None)
    _identifier, _is_local, tmp_dir = inference._resolve_model_identifier(config)

    assert captured["exclude"] == TRAINING_STATE_PATTERNS
    Path(tmp_dir, "config.json").unlink()
    Path(tmp_dir).rmdir()


def test_base_model_download_skips_training_state(tmp_path, monkeypatch):
    import tempfile

    from core import trainer_base

    captured = {}

    def fake_download(_uri, local_dir, **kwargs):
        captured.update(kwargs)
        Path(local_dir).mkdir(parents=True)

    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(trainer_base, "download_directory_from_s3", fake_download)
    owner = SimpleNamespace(
        config=SimpleNamespace(model=SimpleNamespace(base_model_name="s3://bucket/run/checkpoint-10")),
        local_rank=-1,
        logger=Mock(),
    )
    trainer_base.BaseTrainer._maybe_download_s3_model(owner)

    assert captured["exclude"] == TRAINING_STATE_PATTERNS
