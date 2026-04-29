import os
import shlex
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Optional

from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments

from utils.logging_utils import setup_logging

# Configure the logger so messages actually appear. The original bare
# logging.getLogger had no handlers attached, so S3 log lines went nowhere.
# file_output=False because the parent training log already captures stdout.
logger = setup_logging("FAI-RL.s3", file_output=False)


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
):
    """Upload all files in *local_dir* to s3://<bucket>/<s3_prefix>/...

    The directory structure is preserved under *s3_prefix*. *uploader*
    selects the backend ("auto" picks s5cmd if installed, else boto3).
    """
    local_path = Path(local_dir)
    if not local_path.is_dir():
        logger.warning("Local directory %s does not exist - skipping upload", local_dir)
        return

    files = [p for p in local_path.rglob("*") if p.is_file()]
    if not files:
        return

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
        if region and "AWS_REGION" not in env and "AWS_DEFAULT_REGION" not in env:
            env["AWS_REGION"] = region
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
        _resolve_uploader(self.uploader)
        self._upload_threads: list[threading.Thread] = []

    def _schedule_upload(self, local_dir: str, s3_prefix: str, delete_local: bool = False):
        """Start a background upload and track the thread."""
        t = threading.Thread(
            target=upload_directory_to_s3,
            kwargs=dict(
                local_dir=local_dir,
                bucket=self.bucket,
                s3_prefix=s3_prefix,
                region=self.region,
                endpoint_url=self.endpoint_url,
                delete_local=delete_local,
                uploader=self.uploader,
            ),
            daemon=True,
        )
        t.start()
        self._upload_threads.append(t)

    def _wait_for_uploads(self, timeout_per_thread: float = 1800):
        """Block until all background uploads finish."""
        for t in self._upload_threads:
            t.join(timeout=timeout_per_thread)
        self._upload_threads.clear()

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

        logger.info(
            "Scheduling S3 upload for checkpoint step %d -> s3://%s/%s",
            state.global_step, self.bucket, s3_dest,
        )
        self._schedule_upload(
            checkpoint_dir,
            s3_dest,
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
            s3_dest = f"{self.prefix}/final" if self.prefix else "final"
            logger.info(
                "Scheduling S3 upload for final model -> s3://%s/%s",
                self.bucket, s3_dest,
            )
            self._schedule_upload(args.output_dir, s3_dest, delete_local=False)

        logger.info("Waiting for all S3 uploads to complete...")
        self._wait_for_uploads()
        logger.info("All S3 uploads finished.")


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
