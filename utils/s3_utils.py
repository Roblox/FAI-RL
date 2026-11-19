import fnmatch
import json
import os
import shlex
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Optional, Sequence

from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments

from utils.checkpoint_utils import is_resumable_checkpoint, missing_deepspeed_shards
from utils.logging_utils import setup_logging

# Configure the logger so messages actually appear. The original bare
# logging.getLogger had no handlers attached, so S3 log lines went nowhere.
# file_output=False because the parent training log already captures stdout.
logger = setup_logging("FAI-RL.s3", file_output=False)

# Written into s3://<bucket>/<prefix>/checkpoint-<step>/ after every other file
# of a resumable checkpoint is uploaded. --auto-resume restores only
# checkpoints that have one (utils.checkpoint_restore).
UPLOAD_MANIFEST_NAME = "upload_manifest.json"
UPLOAD_MANIFEST_FORMAT = 1


def _get_s3_client(region: Optional[str] = None, endpoint_url: Optional[str] = None):
    """Create a boto3 S3 client, raising a clear error if boto3 is missing."""
    try:
        import boto3
    except ImportError:
        raise ImportError(
            "boto3 is required for S3 checkpoint uploads. "
            "Install it with: pip install boto3"
        )
    kwargs = {}
    if region:
        kwargs["region_name"] = region
    if endpoint_url:
        kwargs["endpoint_url"] = endpoint_url
    return boto3.client("s3", **kwargs)


def _resolve_uploader(uploader: str) -> str:
    """Resolve "auto" -> "s5cmd" if installed, else "boto3". Explicit "s5cmd"
    raises if the binary is missing so we fail loudly instead of silently
    falling back to a ~10x-slower path."""
    if uploader not in ("auto", "boto3", "s5cmd"):
        raise ValueError(f"Unknown s3.uploader={uploader!r}; expected auto, boto3, or s5cmd")
    if uploader == "auto":
        return "s5cmd" if shutil.which("s5cmd") else "boto3"
    if uploader == "s5cmd" and not shutil.which("s5cmd"):
        raise RuntimeError(
            "s3.uploader='s5cmd' but the 's5cmd' binary is not on $PATH. "
            "Install it (https://github.com/peak/s5cmd) or set s3.uploader='auto'."
        )
    return uploader


def upload_directory_to_s3(
    local_dir: str,
    bucket: str,
    s3_prefix: str,
    region: Optional[str] = None,
    endpoint_url: Optional[str] = None,
    delete_local: bool = False,
    uploader: str = "auto",
) -> dict[str, int]:
    """Upload all files in *local_dir* to s3://<bucket>/<s3_prefix>/...

    The directory structure is preserved under *s3_prefix*. *uploader*
    selects the backend ("auto" picks s5cmd if installed, else boto3).

    Returns the size of each uploaded file, keyed by its path relative to
    *local_dir*; empty when there was nothing to upload.
    """
    local_path = Path(local_dir)
    if not local_path.is_dir():
        logger.warning("Local directory %s does not exist - skipping upload", local_dir)
        return {}

    files = [p for p in local_path.rglob("*") if p.is_file()]
    if not files:
        return {}
    sizes = {p.relative_to(local_path).as_posix(): p.stat().st_size for p in files}

    backend = _resolve_uploader(uploader)
    logger.info("Uploading %d files from %s to s3://%s/%s (backend=%s)",
                len(files), local_dir, bucket, s3_prefix, backend)

    if backend == "s5cmd":
        manifest = "\n".join(
            f"cp {shlex.quote(str(p))} "
            f"{shlex.quote(f's3://{bucket}/{s3_prefix}/{p.relative_to(local_path)}')}"
            for p in files
        ) + "\n"
        argv = ["s5cmd"]
        if endpoint_url:
            argv.extend(["--endpoint-url", endpoint_url])
        argv.extend(["--numworkers", "256", "run"])
        env = os.environ.copy()
        if region:
            # An explicitly configured region must win over the compute's
            # ambient AWS_REGION/AWS_DEFAULT_REGION: the target bucket's region
            # is unrelated to where the job happens to run. Without this, an
            # s5cmd transfer to a us-east-1 bucket from a us-east-2 pod inherits
            # us-east-2 and 301-redirects (BucketRegionError).
            env["AWS_REGION"] = region
            env["AWS_DEFAULT_REGION"] = region
        result = subprocess.run(argv, input=manifest, text=True, env=env, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"s5cmd upload failed (exit {result.returncode}): {result.stderr.strip()}"
            )
    else:
        client = _get_s3_client(region, endpoint_url)
        for file_path in files:
            relative = file_path.relative_to(local_path)
            s3_key = f"{s3_prefix}/{relative}".replace("\\", "/")
            client.upload_file(str(file_path), bucket, s3_key)

    logger.info("Uploaded %d files from %s to s3://%s/%s",
                len(files), local_dir, bucket, s3_prefix)

    if delete_local:
        shutil.rmtree(local_dir, ignore_errors=True)
        logger.info("Deleted local directory %s after upload", local_dir)
    return sizes


def eval_metrics_from_trainer_state(state: TrainerState) -> dict:
    """Compact eval snapshot for this save step.

    HuggingFace already writes eval_loss into checkpoint trainer_state.json;
    this sidecar (and the job-level index) is a small file the portal can
    list without parsing the growing log_history.
    """
    step = int(getattr(state, "global_step", 0) or 0)
    metrics: dict = {"step": step}
    history = list(getattr(state, "log_history", None) or [])
    for entry in reversed(history):
        if not isinstance(entry, dict) or "eval_loss" not in entry:
            continue
        try:
            metrics["eval_loss"] = float(entry["eval_loss"])
        except (TypeError, ValueError):
            pass
        if entry.get("step") == step:
            break
    best = getattr(state, "best_metric", None)
    if best is not None:
        try:
            metrics["best_metric"] = float(best)
        except (TypeError, ValueError):
            pass
    best_step = getattr(state, "best_global_step", None)
    if best_step is not None:
        try:
            metrics["best_global_step"] = int(best_step)
        except (TypeError, ValueError):
            pass
    best_ckpt = getattr(state, "best_model_checkpoint", None)
    if best_ckpt:
        metrics["best_model_checkpoint"] = str(best_ckpt)
    return metrics


class S3UploadCallback(TrainerCallback):
    """HuggingFace TrainerCallback that uploads checkpoints to S3.

    Uploads happen in a background thread so they don't block training.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        region: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        upload_checkpoints: bool = True,
        upload_final_model: bool = True,
        delete_local_after_upload: bool = False,
        uploader: str = "auto",
        run_fingerprint: str = "",
    ):
        self.bucket = bucket
        self.prefix = prefix.rstrip("/")
        self.region = region
        self.endpoint_url = endpoint_url
        self.upload_checkpoints = upload_checkpoints
        self.upload_final_model = upload_final_model
        self.delete_local_after_upload = delete_local_after_upload
        self.uploader = uploader
        self.run_fingerprint = run_fingerprint
        # Resolve eagerly so we fail fast at startup if uploader='s5cmd' but
        # the binary isn't installed, instead of mid-training on the first save.
        _resolve_uploader(self.uploader)
        self._upload_threads: list[threading.Thread] = []
        # Exceptions raised inside upload threads are captured here, since
        # Thread.join() does NOT propagate them and would otherwise let a
        # failed upload pass as success.
        self._upload_errors: list[BaseException] = []
        self._errors_lock = threading.Lock()
        # step -> eval_loss, rewritten to prefix/checkpoint_eval.json each save.
        self._eval_index: dict[str, float] = {}
        # Reasons already logged for checkpoints a restart cannot resume from.
        self._unresumable_reasons: set[str] = set()

    @property
    def _eval_index_key(self) -> str:
        return f"{self.prefix}/checkpoint_eval.json" if self.prefix else "checkpoint_eval.json"

    def _run_upload(self, manifest: Optional[dict] = None, **kwargs):
        """Thread target that runs the upload and records any failure.

        With *manifest*, a successful upload ends by writing it, listing the
        files that were uploaded, so a restart can tell the copy is complete.
        """
        try:
            uploaded = upload_directory_to_s3(**kwargs)
            if manifest is not None and uploaded:
                self._upload_manifest(kwargs["s3_prefix"], {**manifest, "files": uploaded})
        except BaseException as exc:  # noqa: BLE001 - re-raised on join
            logger.error(
                "S3 upload to s3://%s/%s failed: %s",
                kwargs.get("bucket"), kwargs.get("s3_prefix"), exc,
            )
            with self._errors_lock:
                self._upload_errors.append(exc)

    def _upload_manifest(self, s3_prefix: str, manifest: dict) -> None:
        fd, tmp_path = tempfile.mkstemp(suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(manifest, fh, indent=2, sort_keys=True)
            upload_file_to_s3(
                tmp_path,
                self.bucket,
                f"{s3_prefix}/{UPLOAD_MANIFEST_NAME}",
                region=self.region,
                endpoint_url=self.endpoint_url,
                uploader=self.uploader,
            )
        finally:
            os.unlink(tmp_path)

    def _schedule_upload(
        self,
        local_dir: str,
        s3_prefix: str,
        delete_local: bool = False,
        manifest: Optional[dict] = None,
    ):
        """Start a background upload and track the thread."""
        t = threading.Thread(
            target=self._run_upload,
            kwargs=dict(
                local_dir=local_dir,
                bucket=self.bucket,
                s3_prefix=s3_prefix,
                region=self.region,
                endpoint_url=self.endpoint_url,
                delete_local=delete_local,
                uploader=self.uploader,
                manifest=manifest,
            ),
            daemon=True,
        )
        t.start()
        self._upload_threads.append(t)

    def _wait_for_uploads(self, timeout_per_thread: float = 1800):
        """Block until all background uploads finish.

        Raises RuntimeError if any upload failed or did not complete within the
        timeout, so a failed model/checkpoint upload is reported as a failure
        instead of silently passing.
        """
        timed_out = []
        for t in self._upload_threads:
            t.join(timeout=timeout_per_thread)
            if t.is_alive():
                timed_out.append(t)
        self._upload_threads.clear()

        with self._errors_lock:
            errors = list(self._upload_errors)
            self._upload_errors.clear()

        if timed_out:
            raise RuntimeError(
                f"{len(timed_out)} S3 upload(s) did not finish within "
                f"{timeout_per_thread}s; treating as failed."
            )
        if errors:
            raise RuntimeError(
                f"{len(errors)} S3 upload(s) failed; first error: {errors[0]}"
            ) from errors[0]

    def _write_eval_artifacts(self, checkpoint_dir: str, state: TrainerState) -> None:
        """Write eval_metrics.json into the checkpoint and a job-level index."""
        metrics = eval_metrics_from_trainer_state(state)
        os.makedirs(checkpoint_dir, exist_ok=True)
        sidecar = os.path.join(checkpoint_dir, "eval_metrics.json")
        try:
            with open(sidecar, "w", encoding="utf-8") as fh:
                json.dump(metrics, fh, indent=2)
        except OSError as exc:
            logger.warning("Failed to write %s: %s", sidecar, exc)
        loss = metrics.get("eval_loss")
        if isinstance(loss, (int, float)):
            self._eval_index[str(int(state.global_step))] = float(loss)
        payload = {
            "eval_loss": dict(self._eval_index),
            "best_global_step": metrics.get("best_global_step"),
            "best_metric": metrics.get("best_metric"),
        }
        index_key = self._eval_index_key
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
                tmp_path = fh.name
            upload_file_to_s3(
                tmp_path,
                self.bucket,
                index_key,
                region=self.region,
                endpoint_url=self.endpoint_url,
                uploader=self.uploader,
            )
        except Exception as exc:  # noqa: BLE001 - index is best-effort
            logger.warning("Failed to upload checkpoint_eval.json: %s", exc)
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    # -- Trainer hooks --

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if not self.upload_checkpoints:
            return
        # Only rank 0 uploads. Other ranks see the same checkpoint dir on
        # shared FS and would otherwise race & 8x our bandwidth.
        if not state.is_world_process_zero:
            return

        checkpoint_dir = os.path.join(
            args.output_dir, f"checkpoint-{state.global_step}"
        )
        s3_dest = f"{self.prefix}/checkpoint-{state.global_step}" if self.prefix else f"checkpoint-{state.global_step}"

        self._write_eval_artifacts(checkpoint_dir, state)

        logger.info(
            "Scheduling S3 upload for checkpoint step %d -> s3://%s/%s",
            state.global_step, self.bucket, s3_dest,
        )
        self._schedule_upload(
            checkpoint_dir,
            s3_dest,
            delete_local=self.delete_local_after_upload,
            manifest=self._resume_manifest(checkpoint_dir, args, state),
        )

    def _resume_manifest(
        self, checkpoint_dir: str, args: TrainingArguments, state: TrainerState
    ) -> Optional[dict]:
        """Return manifest metadata if a restart could resume from this checkpoint."""
        world_size = int(getattr(args, "world_size", 1) or 1)
        path = Path(checkpoint_dir)
        if not is_resumable_checkpoint(path):
            reason = (
                "it has no optimizer and scheduler state; "
                "set training.save_only_model=false to resume after a restart"
            )
        else:
            reason = missing_deepspeed_shards(path, world_size)
        if reason:
            if reason not in self._unresumable_reasons:
                self._unresumable_reasons.add(reason)
                logger.info(
                    "Checkpoint step %d uploads without %s: %s",
                    state.global_step, UPLOAD_MANIFEST_NAME, reason,
                )
            return None
        return {
            "format": UPLOAD_MANIFEST_FORMAT,
            "global_step": int(state.global_step),
            "world_size": world_size,
            "run_fingerprint": self.run_fingerprint,
        }

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        # A resumed run rewrites checkpoint_eval.json on its first save. Seed
        # the index from the earlier attempt so that rewrite keeps its losses.
        if not self.upload_checkpoints or not state.is_world_process_zero:
            return
        if not state.global_step:
            return
        try:
            raw = download_s3_bytes(
                f"s3://{self.bucket}/{self._eval_index_key}",
                region=self.region,
                endpoint_url=self.endpoint_url,
            )
            losses = json.loads(raw).get("eval_loss") or {}
        except Exception as exc:  # noqa: BLE001 - index is best-effort
            logger.info("No checkpoint_eval.json to carry over on resume: %s", exc)
            return
        for step, loss in losses.items():
            try:
                if int(step) <= state.global_step and isinstance(loss, (int, float)):
                    self._eval_index[str(int(step))] = float(loss)
            except (TypeError, ValueError):
                continue

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if not state.is_world_process_zero:
            return

        if self.upload_final_model:
            s3_dest = f"{self.prefix}/final" if self.prefix else "final"
            logger.info(
                "Scheduling S3 upload for final model -> s3://%s/%s",
                self.bucket, s3_dest,
            )
            self._schedule_upload(args.output_dir, s3_dest, delete_local=False)

        logger.info("Waiting for all S3 uploads to complete...")
        self._wait_for_uploads()
        logger.info("All S3 uploads finished.")


def download_directory_from_s3(
    s3_uri: str,
    local_dir: str,
    region: Optional[str] = None,
    endpoint_url: Optional[str] = None,
    downloader: str = "auto",
    exclude: Sequence[str] = (),
) -> None:
    """Download all objects under an S3 prefix to *local_dir*.

    The directory structure under the prefix is preserved. *downloader* selects
    the backend ("auto" picks s5cmd if installed, else boto3). Objects whose
    relative path, or any directory in it, matches an *exclude* glob are
    skipped.
    """
    from urllib.parse import urlparse

    parsed = urlparse(s3_uri)
    if parsed.scheme != "s3":
        raise ValueError(f"Expected s3:// URI, got: {s3_uri!r}")

    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/").rstrip("/")

    os.makedirs(local_dir, exist_ok=True)

    backend = _resolve_uploader(downloader)
    logger.info(
        "Downloading s3://%s/%s -> %s (backend=%s)", bucket, prefix, local_dir, backend
    )

    if backend == "s5cmd":
        argv = ["s5cmd"]
        if endpoint_url:
            argv.extend(["--endpoint-url", endpoint_url])
        stdin = None
        if exclude:
            # s5cmd has no exclude that matches nested paths reliably, so list
            # the keys here and copy them in one batch.
            lines = []
            for key, relative in _list_s3_files(bucket, prefix, region, endpoint_url, exclude):
                local_file = os.path.join(local_dir, relative)
                os.makedirs(os.path.dirname(local_file), exist_ok=True)
                lines.append(shlex.join(["cp", f"s3://{bucket}/{key}", local_file]))
            argv.extend(["--numworkers", "256", "run"])
            stdin = "\n".join(lines) + "\n"
        else:
            # s5cmd sync handles recursive directory copies efficiently
            argv.extend(["sync", f"s3://{bucket}/{prefix}/*", f"{local_dir}/"])
        env = os.environ.copy()
        if region:
            # An explicitly configured region must win over the compute's
            # ambient AWS_REGION/AWS_DEFAULT_REGION: the target bucket's region
            # is unrelated to where the job happens to run. Without this, an
            # s5cmd transfer to a us-east-1 bucket from a us-east-2 pod inherits
            # us-east-2 and 301-redirects (BucketRegionError).
            env["AWS_REGION"] = region
            env["AWS_DEFAULT_REGION"] = region
        result = subprocess.run(argv, env=env, capture_output=True, text=True, input=stdin)
        output = "\n".join(part for part in (result.stdout, result.stderr) if part)
        # s5cmd 2.3.0 `sync` can print an ERROR (notably BucketRegionError) while
        # returning zero. Treat its structured error output as a failure too.
        has_error = any(line.lstrip().startswith("ERROR ") for line in output.splitlines())
        if result.returncode != 0 or has_error:
            raise RuntimeError(
                f"s5cmd download failed (exit {result.returncode}): {output.strip()}"
            )
    else:
        client = _get_s3_client(region, endpoint_url)
        total = 0
        for key, relative in _list_s3_files(bucket, prefix, region, endpoint_url, exclude):
            local_file = os.path.join(local_dir, relative)
            os.makedirs(os.path.dirname(local_file), exist_ok=True)
            client.download_file(bucket, key, local_file)
            total += 1
        logger.info("Downloaded %d files from s3://%s/%s", total, bucket, prefix)

    downloaded_files = sum(1 for path in Path(local_dir).rglob("*") if path.is_file())
    if downloaded_files == 0:
        raise RuntimeError(f"S3 download from s3://{bucket}/{prefix} downloaded no files")

    logger.info("Download complete: s3://%s/%s -> %s", bucket, prefix, local_dir)


def _list_s3_files(
    bucket: str,
    prefix: str,
    region: Optional[str],
    endpoint_url: Optional[str],
    exclude: Sequence[str] = (),
):
    """Yield (key, path relative to *prefix*) for each object not excluded."""
    client = _get_s3_client(region, endpoint_url)
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix + "/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            relative = key[len(prefix):].lstrip("/")
            if not relative:
                continue
            parts = relative.split("/")
            if any(fnmatch.fnmatch(part, pattern) for part in parts for pattern in exclude):
                continue
            yield key, relative


def download_file_from_s3(
    s3_uri: str,
    region: Optional[str] = None,
    endpoint_url: Optional[str] = None,
) -> str:
    """Download a single file from S3 to a temporary local path.

    Returns the path to the downloaded temp file. The caller is responsible
    for deleting it when done (e.g. after the dataset has been loaded into
    memory / Arrow cache).
    """
    import tempfile
    from urllib.parse import urlparse

    parsed = urlparse(s3_uri)
    if parsed.scheme != "s3":
        raise ValueError(f"Expected s3:// URI, got: {s3_uri!r}")

    bucket = parsed.netloc
    key = parsed.path.lstrip("/")

    _, ext = os.path.splitext(key)
    fd, tmp_path = tempfile.mkstemp(suffix=ext)
    os.close(fd)

    logger.info("Downloading s3://%s/%s -> %s", bucket, key, tmp_path)
    try:
        client = _get_s3_client(region, endpoint_url)
        client.download_file(bucket, key, tmp_path)
    except Exception:
        os.unlink(tmp_path)
        raise

    logger.info(
        "Downloaded s3://%s/%s (%d bytes)", bucket, key, os.path.getsize(tmp_path)
    )
    return tmp_path


def download_s3_bytes(
    s3_uri: str,
    region: Optional[str] = None,
    endpoint_url: Optional[str] = None,
) -> bytes:
    """Download a single S3 object fully into memory and return its bytes.

    Unlike :func:`download_file_from_s3` this never touches disk, which suits
    small objects fetched one at a time (e.g. images referenced by a dataset
    row). The caller decides whether/where to cache the result.
    """
    from urllib.parse import urlparse

    parsed = urlparse(s3_uri)
    if parsed.scheme != "s3":
        raise ValueError(f"Expected s3:// URI, got: {s3_uri!r}")

    bucket = parsed.netloc
    key = parsed.path.lstrip("/")

    client = _get_s3_client(region, endpoint_url)
    resp = client.get_object(Bucket=bucket, Key=key)
    return resp["Body"].read()


def upload_file_to_s3(
    local_path: str,
    bucket: str,
    s3_key: str,
    region: Optional[str] = None,
    endpoint_url: Optional[str] = None,
    uploader: str = "auto",
) -> None:
    """Upload a single local file to s3://<bucket>/<s3_key>."""
    if not os.path.isfile(local_path):
        logger.warning("Local file %s does not exist - skipping upload", local_path)
        return

    backend = _resolve_uploader(uploader)
    logger.info("Uploading %s -> s3://%s/%s (backend=%s)", local_path, bucket, s3_key, backend)

    if backend == "s5cmd":
        argv = ["s5cmd"]
        if endpoint_url:
            argv.extend(["--endpoint-url", endpoint_url])
        argv.extend(["cp", local_path, f"s3://{bucket}/{s3_key}"])
        env = os.environ.copy()
        if region:
            # An explicitly configured region must win over the compute's
            # ambient AWS_REGION/AWS_DEFAULT_REGION: the target bucket's region
            # is unrelated to where the job happens to run. Without this, an
            # s5cmd transfer to a us-east-1 bucket from a us-east-2 pod inherits
            # us-east-2 and 301-redirects (BucketRegionError).
            env["AWS_REGION"] = region
            env["AWS_DEFAULT_REGION"] = region
        result = subprocess.run(argv, env=env, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"s5cmd upload failed (exit {result.returncode}): {result.stderr.strip()}"
            )
    else:
        client = _get_s3_client(region, endpoint_url)
        client.upload_file(local_path, bucket, s3_key)

    logger.info("Uploaded %s -> s3://%s/%s", local_path, bucket, s3_key)


def build_s3_callback(s3_config) -> Optional[S3UploadCallback]:
    """Create an S3UploadCallback from an S3Config, or None if disabled."""
    if not s3_config.enabled:
        return None

    if not s3_config.bucket:
        raise ValueError("s3.bucket is required when s3.enabled is true")

    return S3UploadCallback(
        bucket=s3_config.bucket,
        prefix=s3_config.prefix,
        region=s3_config.region,
        endpoint_url=s3_config.endpoint_url,
        upload_checkpoints=s3_config.upload_checkpoints,
        upload_final_model=s3_config.upload_final_model,
        delete_local_after_upload=s3_config.delete_local_after_upload,
        uploader=getattr(s3_config, "uploader", "auto"),
        run_fingerprint=getattr(s3_config, "run_fingerprint", ""),
    )
