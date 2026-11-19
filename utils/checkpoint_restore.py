"""Restore a restarted job's newest uploaded checkpoint from S3.

A job that Kueue preempts, or that loses a node, restarts on fresh disks: the
checkpoints in ``training.output_dir`` are gone and only the S3 copies that
``S3UploadCallback`` made remain. ``--auto-resume`` restores the newest of
those that has an upload manifest (``utils.s3_utils.UPLOAD_MANIFEST_NAME``),
written by the same recipe at the same world size, before choosing the
checkpoint to resume from.
"""

import contextlib
import fcntl
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterator, Optional

from utils import s3_utils
from utils.checkpoint_utils import (
    find_latest_checkpoint,
    is_resumable_checkpoint,
    missing_deepspeed_shards,
)

# Fields that differ between attempts of the same run.
_PER_ATTEMPT_FIELDS = {
    "training": ("resume_from_checkpoint", "deepspeed_config"),
    "s3": ("run_fingerprint",),
}
_SECTIONS = ("model", "data", "training", "wandb", "s3", "reward_api", "local_reward_function")
_STAGING_DIR = ".s3_restore"


def run_fingerprint(config: Any) -> str:
    """Hash the recipe, so a restart only restores its own run's checkpoints.

    Two jobs can share an S3 prefix, for example a rerun of a finished job.
    """
    sections = {}
    for name in _SECTIONS:
        section = getattr(config, name, None)
        values = section.to_dict() if section is not None else None
        if values is not None:
            skipped = _PER_ATTEMPT_FIELDS.get(name, ())
            values = {key: value for key, value in values.items() if key not in skipped}
        sections[name] = values
    payload = json.dumps(sections, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def restore_checkpoint_from_s3(config: Any, logger: Any) -> Optional[str]:
    """Download this run's newest complete S3 checkpoint into output_dir.

    Returns the newest resumable checkpoint in output_dir afterwards, or None.
    Does nothing when the recipe names a checkpoint or does not upload them.
    Ranks on one node download once; every node restores its own copy.
    """
    training, s3 = config.training, config.s3
    if getattr(training, "resume_from_checkpoint", None):
        return None
    if not (s3.enabled and s3.bucket and s3.upload_checkpoints):
        return None

    output_dir = Path(training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    world_size = int(os.environ.get("WORLD_SIZE") or 1)
    fingerprint = s3.run_fingerprint or run_fingerprint(config)
    prefix = s3.prefix.strip("/")

    with _node_lock(output_dir):
        client = s3_utils._get_s3_client(s3.region, s3.endpoint_url)
        restore = lambda step: _restore_step(  # noqa: E731
            client, s3, prefix, step, output_dir, fingerprint, world_size, logger
        )
        try:
            checkpoint = _restore_newest(client, s3.bucket, prefix, output_dir, restore, logger)
            if checkpoint:
                _restore_best(Path(checkpoint), output_dir, restore, logger)
        finally:
            shutil.rmtree(output_dir / _STAGING_DIR, ignore_errors=True)
    return checkpoint


def _restore_newest(client, bucket: str, prefix: str, output_dir: Path, restore, logger) -> Optional[str]:
    local = find_latest_checkpoint(str(output_dir))
    local_step = _step(Path(local).name) if local else -1
    for step in _uploaded_steps(client, bucket, prefix):
        if step <= local_step:
            break
        restored = restore(step)
        if restored:
            return restored
    if local:
        logger.info("Local checkpoint %s is newer than any uploaded one", local)
    else:
        logger.info("No checkpoint of this run in s3://%s/%s to restore", bucket, prefix)
    return local


def _restore_best(checkpoint: Path, output_dir: Path, restore, logger) -> None:
    """Restore the best checkpoint that *checkpoint*'s trainer state names.

    load_best_model_at_end reloads it after the last step, and with DeepSpeed a
    missing one fails the run there. If it cannot be restored, the state stops
    naming it, so the run ends with its last model instead.
    """
    state_path = checkpoint / "trainer_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    best = state.get("best_model_checkpoint")
    if not best or is_resumable_checkpoint(Path(best)):
        return
    step = _step(Path(best).name)
    local = output_dir / f"checkpoint-{step}"
    if step >= 0 and (is_resumable_checkpoint(local) or restore(step)):
        state["best_model_checkpoint"] = str(local)
    else:
        logger.warning(
            "Best checkpoint %s could not be restored; the run will end with its last model", best
        )
        state["best_model_checkpoint"] = None
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _restore_step(
    client, s3, prefix: str, step: int, output_dir: Path, fingerprint: str, world_size: int, logger
) -> Optional[str]:
    name = f"checkpoint-{step}"
    key_prefix = f"{prefix}/{name}" if prefix else name
    source = f"s3://{s3.bucket}/{key_prefix}"
    manifest = _read_manifest(client, s3.bucket, f"{key_prefix}/{s3_utils.UPLOAD_MANIFEST_NAME}")
    problem = _manifest_problem(manifest, step, fingerprint, world_size)
    if problem:
        logger.info("Not restoring %s: %s", source, problem)
        return None

    staging = output_dir / _STAGING_DIR / name
    shutil.rmtree(staging, ignore_errors=True)
    try:
        s3_utils.download_directory_from_s3(
            source,
            str(staging),
            region=s3.region,
            endpoint_url=s3.endpoint_url,
            downloader=getattr(s3, "uploader", "auto"),
        )
    except Exception as exc:  # noqa: BLE001 - an older checkpoint may still restore
        logger.warning("Not restoring %s: download failed: %s", source, exc)
        return None
    problem = _download_problem(staging, manifest["files"], world_size)
    if problem:
        logger.warning("Not restoring %s: %s", source, problem)
        return None

    target = output_dir / name
    shutil.rmtree(target, ignore_errors=True)
    os.replace(staging, target)
    logger.info("Restored %s to %s", source, target)
    return str(target)


def _read_manifest(client, bucket: str, key: str) -> Optional[dict]:
    from botocore.exceptions import ClientError

    try:
        body = client.get_object(Bucket=bucket, Key=key)["Body"].read()
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            return None
        raise
    try:
        manifest = json.loads(body)
    except ValueError:
        return None
    return manifest if isinstance(manifest, dict) else None


def _manifest_problem(
    manifest: Optional[dict], step: int, fingerprint: str, world_size: int
) -> Optional[str]:
    if manifest is None:
        return "its upload did not finish"
    if manifest.get("format") != s3_utils.UPLOAD_MANIFEST_FORMAT:
        return f"unknown manifest format {manifest.get('format')!r}"
    if manifest.get("global_step") != step:
        return f"its manifest is for step {manifest.get('global_step')!r}"
    if manifest.get("run_fingerprint") != fingerprint:
        return "another recipe wrote it"
    if manifest.get("world_size") != world_size:
        return f"it was saved by {manifest.get('world_size')!r} ranks and this run has {world_size}"
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        return "its manifest lists no files"
    return None


def _download_problem(staging: Path, files: dict, world_size: int) -> Optional[str]:
    root = staging.resolve()
    for relative, size in files.items():
        path = staging / relative
        if root not in path.resolve().parents:
            return f"its manifest names a path outside the checkpoint: {relative!r}"
        if not path.is_file():
            return f"{relative} is missing"
        if path.stat().st_size != size:
            return f"{relative} has {path.stat().st_size} bytes, the manifest says {size}"
    if not is_resumable_checkpoint(staging):
        return "the downloaded files are not a resumable checkpoint"
    return missing_deepspeed_shards(staging, world_size)


def _uploaded_steps(client, bucket: str, prefix: str) -> list[int]:
    """Steps of the checkpoint-N directories under prefix, newest first."""
    list_prefix = f"{prefix}/" if prefix else ""
    steps = set()
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=list_prefix, Delimiter="/"):
        for common in page.get("CommonPrefixes", []):
            step = _step(common["Prefix"][len(list_prefix):].rstrip("/"))
            if step >= 0:
                steps.add(step)
    return sorted(steps, reverse=True)


def _step(name: str) -> int:
    match = re.fullmatch(r"checkpoint-(\d+)", name)
    return int(match[1]) if match else -1


@contextlib.contextmanager
def _node_lock(output_dir: Path) -> Iterator[None]:
    # Kept out of output_dir, which the final-model upload copies whole.
    key = hashlib.sha256(str(output_dir.resolve()).encode("utf-8")).hexdigest()[:16]
    path = Path(tempfile.gettempdir()) / f"fai-rl-s3-restore-{key}.lock"
    with open(path, "a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
