import json
import os
import shlex
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import List, Optional

from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments

from utils.logging_utils import setup_logging

# Share the console with the rest of FAI-RL but skip the separate file handler
# (the parent training log already captures stdout, and a second file would
# double-log under nohup). setup_logging is idempotent so re-imports are safe.
logger = setup_logging("FAI-RL.s3", file_output=False)

# s5cmd parallelism. 256 worker threads matched the pod's effective egress
# ceiling (~10 Gbps) in our benchmark on a P6 EKS pod; going higher gave
# diminishing returns. Override via FAI_RL_S5CMD_NUMWORKERS for tuning.
_S5CMD_NUMWORKERS = int(os.environ.get("FAI_RL_S5CMD_NUMWORKERS", "256"))
_PROGRESS_EVERY_SECONDS = 30


def _human_bytes(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PiB"


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
    """Resolve "auto" -> "s5cmd" if available, else "boto3".

    Explicit "s5cmd" / "boto3" pass through, but "s5cmd" raises if the binary
    isn't on $PATH so we fail loudly rather than silently falling back to a
    10x-slower path.
    """
    if uploader not in ("auto", "boto3", "s5cmd"):
        raise ValueError(
            f"Unknown s3.uploader={uploader!r}; expected one of: auto, boto3, s5cmd"
        )
    if uploader == "auto":
        return "s5cmd" if shutil.which("s5cmd") else "boto3"
    if uploader == "s5cmd" and not shutil.which("s5cmd"):
        raise RuntimeError(
            "s3.uploader='s5cmd' but the 's5cmd' binary is not on $PATH. "
            "Install it (https://github.com/peak/s5cmd) or set s3.uploader='auto'."
        )
    return uploader


def _upload_via_boto3(
    local_path: Path,
    files: List[Path],
    bucket: str,
    s3_prefix: str,
    total_bytes: int,
    started: float,
    region: Optional[str],
    endpoint_url: Optional[str],
    tag: str,
) -> int:
    """Original sequential boto3 upload path. Slow on EKS pods (~150 MiB/s)
    but has no external binary dependency."""
    try:
        client = _get_s3_client(region, endpoint_url)
    except Exception as e:
        logger.exception("%sFailed to create S3 client: %s", tag, e)
        raise

    uploaded = 0
    bytes_done = 0
    last_progress_t = started

    for file_path in files:
        relative = file_path.relative_to(local_path)
        s3_key = f"{s3_prefix}/{relative}".replace("\\", "/")
        size = file_path.stat().st_size
        try:
            client.upload_file(str(file_path), bucket, s3_key)
        except Exception as e:
            logger.exception(
                "%sFailed to upload %s -> s3://%s/%s: %s",
                tag, file_path, bucket, s3_key, e,
            )
            raise
        uploaded += 1
        bytes_done += size

        now = time.monotonic()
        if now - last_progress_t >= _PROGRESS_EVERY_SECONDS:
            elapsed = now - started
            rate = bytes_done / elapsed if elapsed > 0 else 0
            logger.info(
                "%sProgress: %d/%d files, %s / %s (%.1f%%), %s/s, %.1fs elapsed",
                tag, uploaded, len(files),
                _human_bytes(bytes_done), _human_bytes(total_bytes),
                100.0 * bytes_done / max(total_bytes, 1),
                _human_bytes(int(rate)), elapsed,
            )
            last_progress_t = now

    return uploaded


def _upload_via_s5cmd(
    local_path: Path,
    files: List[Path],
    bucket: str,
    s3_prefix: str,
    total_bytes: int,
    started: float,
    region: Optional[str],
    endpoint_url: Optional[str],
    tag: str,
) -> int:
    """High-throughput upload via the `s5cmd` binary.

    Builds a manifest of `cp <src> <dst>` lines (one per file, exact same set
    we'd hand to boto3) and pipes it to `s5cmd --json run`. JSON output is
    parsed line-by-line so we can keep the same per-30s progress format and
    surface per-file errors back to S3UploadCallback._upload_failures.

    Concurrency is controlled by --numworkers (default 256, override via
    FAI_RL_S5CMD_NUMWORKERS). Empirically saturates the EKS pod's ~10 Gbps
    egress class on a P6 host, ~7-8x faster than the boto3 path.
    """
    sizes_by_src = {str(p): p.stat().st_size for p in files}

    manifest_lines = []
    for file_path in files:
        relative = file_path.relative_to(local_path)
        s3_key = f"{s3_prefix}/{relative}".replace("\\", "/")
        manifest_lines.append(
            f"cp {shlex.quote(str(file_path))} {shlex.quote(f's3://{bucket}/{s3_key}')}"
        )
    manifest = "\n".join(manifest_lines) + "\n"

    s5cmd_argv = ["s5cmd", "--json"]
    if endpoint_url:
        s5cmd_argv.extend(["--endpoint-url", endpoint_url])
    s5cmd_argv.extend(["--numworkers", str(_S5CMD_NUMWORKERS), "run"])

    env = os.environ.copy()
    if region and "AWS_REGION" not in env and "AWS_DEFAULT_REGION" not in env:
        env["AWS_REGION"] = region

    logger.info(
        "%sUsing s5cmd backend (numworkers=%d)", tag, _S5CMD_NUMWORKERS,
    )
    try:
        proc = subprocess.Popen(
            s5cmd_argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
    except FileNotFoundError as e:
        # Resolved earlier, but guard anyway.
        raise RuntimeError(
            "s5cmd binary disappeared between resolve and exec; check $PATH"
        ) from e

    # Send the whole manifest then close stdin so s5cmd knows the input is done.
    assert proc.stdin is not None and proc.stdout is not None and proc.stderr is not None
    proc.stdin.write(manifest)
    proc.stdin.close()

    uploaded = 0
    bytes_done = 0
    last_progress_t = started
    failures: List[str] = []

    # Parse stdout line-by-line. Each successful op is one JSON object on its
    # own line. With --json, errors also appear as JSON with success=false.
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        try:
            evt = json.loads(line)
        except json.JSONDecodeError:
            # Non-JSON lines (rare, e.g. progress noise) — log and move on.
            logger.debug("%ss5cmd stdout (non-json): %s", tag, line)
            continue

        if evt.get("success") is False or "error" in evt:
            err = evt.get("error", "<no error message>")
            src = evt.get("source", "?")
            failures.append(f"{src}: {err}")
            logger.error("%ss5cmd failed for %s: %s", tag, src, err)
            continue

        # Success — accumulate progress against the source we know the size of.
        src = evt.get("source")
        if src and src in sizes_by_src:
            bytes_done += sizes_by_src[src]
            uploaded += 1
        else:
            # Still count it even if we can't map back to a known source.
            uploaded += 1

        now = time.monotonic()
        if now - last_progress_t >= _PROGRESS_EVERY_SECONDS:
            elapsed = now - started
            rate = bytes_done / elapsed if elapsed > 0 else 0
            logger.info(
                "%sProgress: %d/%d files, %s / %s (%.1f%%), %s/s, %.1fs elapsed",
                tag, uploaded, len(files),
                _human_bytes(bytes_done), _human_bytes(total_bytes),
                100.0 * bytes_done / max(total_bytes, 1),
                _human_bytes(int(rate)), elapsed,
            )
            last_progress_t = now

    rc = proc.wait()
    stderr_tail = (proc.stderr.read() or "").strip()
    if rc != 0 or failures:
        sample = "; ".join(failures[:5]) if failures else stderr_tail or f"exit code {rc}"
        raise RuntimeError(
            f"s5cmd upload failed ({len(failures)} of {len(files)} files; "
            f"exit code {rc}). First errors: {sample}"
        )

    return uploaded


def upload_directory_to_s3(
    local_dir: str,
    bucket: str,
    s3_prefix: str,
    region: Optional[str] = None,
    endpoint_url: Optional[str] = None,
    delete_local: bool = False,
    label: str = "",
    uploader: str = "auto",
):
    """Upload all files in *local_dir* to s3://<bucket>/<s3_prefix>/...

    The directory structure is preserved under *s3_prefix*. *label* is an
    optional human tag (e.g. "checkpoint-100", "final") used to make the log
    lines easy to grep. *uploader* selects the backend (see _resolve_uploader).
    """
    tag = f"[{label}] " if label else ""
    local_path = Path(local_dir)

    if not local_path.is_dir():
        logger.warning("%sLocal directory %s does not exist - skipping upload", tag, local_dir)
        return

    files = [p for p in local_path.rglob("*") if p.is_file()]
    if not files:
        logger.warning("%sNo files under %s - skipping upload", tag, local_dir)
        return

    total_bytes = sum(p.stat().st_size for p in files)
    backend = _resolve_uploader(uploader)
    logger.info(
        "%sStarting S3 upload: %d files, %s -> s3://%s/%s (backend=%s)",
        tag, len(files), _human_bytes(total_bytes), bucket, s3_prefix, backend,
    )

    started = time.monotonic()
    if backend == "s5cmd":
        uploaded = _upload_via_s5cmd(
            local_path, files, bucket, s3_prefix, total_bytes, started,
            region, endpoint_url, tag,
        )
    else:
        uploaded = _upload_via_boto3(
            local_path, files, bucket, s3_prefix, total_bytes, started,
            region, endpoint_url, tag,
        )

    elapsed = time.monotonic() - started
    rate = total_bytes / elapsed if elapsed > 0 else 0
    logger.info(
        "%sCompleted S3 upload: %d files, %s in %.1fs (%s/s) -> s3://%s/%s",
        tag, uploaded, _human_bytes(total_bytes), elapsed,
        _human_bytes(int(rate)), bucket, s3_prefix,
    )

    if delete_local and uploaded > 0:
        shutil.rmtree(local_dir, ignore_errors=True)
        logger.info("%sDeleted local directory %s after upload", tag, local_dir)


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
    ):
        self.bucket = bucket
        self.prefix = prefix.rstrip("/")
        self.region = region
        self.endpoint_url = endpoint_url
        self.upload_checkpoints = upload_checkpoints
        self.upload_final_model = upload_final_model
        self.delete_local_after_upload = delete_local_after_upload
        self.uploader = uploader
        # Resolve eagerly so we fail fast at startup if uploader='s5cmd' but
        # the binary isn't installed, instead of mid-training on the first save.
        resolved = _resolve_uploader(self.uploader)
        self._lock = threading.Lock()
        self._upload_threads: list[tuple[str, threading.Thread]] = []
        self._upload_failures: list[tuple[str, str]] = []
        logger.info(
            "S3 upload callback initialized: bucket=%s prefix=%s "
            "(upload_checkpoints=%s, upload_final_model=%s, delete_local=%s, "
            "uploader=%s -> %s)",
            self.bucket, self.prefix or "<root>",
            self.upload_checkpoints, self.upload_final_model, self.delete_local_after_upload,
            self.uploader, resolved,
        )

    def _schedule_upload(
        self, local_dir: str, s3_prefix: str, label: str, delete_local: bool = False,
    ):
        """Start a background upload and track the thread + any failure."""
        def _runner():
            try:
                upload_directory_to_s3(
                    local_dir=local_dir,
                    bucket=self.bucket,
                    s3_prefix=s3_prefix,
                    region=self.region,
                    endpoint_url=self.endpoint_url,
                    delete_local=delete_local,
                    label=label,
                    uploader=self.uploader,
                )
            except Exception as e:
                # Already logged at the source; record so _wait_for_uploads can
                # surface a single summary at the end and we don't lose track of
                # failures in daemon threads.
                with self._lock:
                    self._upload_failures.append((label, str(e)))

        t = threading.Thread(target=_runner, name=f"s3-upload-{label}", daemon=True)
        t.start()
        with self._lock:
            self._upload_threads.append((label, t))

    def _wait_for_uploads(self, timeout_per_thread: float = 1800):
        """Block until all background uploads finish; return (n_ok, n_failed)."""
        with self._lock:
            threads = list(self._upload_threads)
            self._upload_threads.clear()

        if not threads:
            return 0, 0

        logger.info("Waiting for %d S3 upload thread(s) to finish...", len(threads))
        for label, t in threads:
            t.join(timeout=timeout_per_thread)
            if t.is_alive():
                logger.error(
                    "S3 upload thread for %s did not finish within %.0fs (still running)",
                    label, timeout_per_thread,
                )

        with self._lock:
            failures = list(self._upload_failures)
            self._upload_failures.clear()

        n_ok = len(threads) - len(failures)
        if failures:
            logger.error("S3 uploads finished with %d failure(s):", len(failures))
            for label, err in failures:
                logger.error("  - %s: %s", label, err)
        else:
            logger.info("All %d S3 upload(s) finished successfully.", n_ok)
        return n_ok, len(failures)

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

        label = f"checkpoint-{state.global_step}"
        checkpoint_dir = os.path.join(args.output_dir, label)
        s3_dest = f"{self.prefix}/{label}" if self.prefix else label

        logger.info(
            "[%s] Scheduling S3 upload (step=%d) -> s3://%s/%s",
            label, state.global_step, self.bucket, s3_dest,
        )
        self._schedule_upload(
            checkpoint_dir,
            s3_dest,
            label=label,
            delete_local=self.delete_local_after_upload,
        )

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
            label = "final"
            s3_dest = f"{self.prefix}/{label}" if self.prefix else label
            logger.info(
                "[%s] Scheduling S3 upload of final model -> s3://%s/%s",
                label, self.bucket, s3_dest,
            )
            self._schedule_upload(args.output_dir, s3_dest, label=label, delete_local=False)

        self._wait_for_uploads()


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
    )
