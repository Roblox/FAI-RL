"""Result files for inference: CSV/JSONL output, JSONL shards, resume and progress.

Shards are ``shards/part-NNNNN.jsonl`` next to ``output_file``; each holds
finished rows sorted by row id. A restarted run reads them (after pulling any
missing ones from S3) and skips their row ids.
"""

import csv
import datetime
import heapq
import json
import math
import os
import re
import shutil
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional

SHARD_DIR = "shards"
_SHARD_NAME = re.compile(r"^part-(\d+)\.jsonl$")


def json_safe(value: Any) -> Any:
    """Convert dataset cell values (numpy, NaN, bytes, timestamps) to JSON values."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (datetime.date, datetime.datetime)):
        return value.isoformat()
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if hasattr(value, "item"):
        return json_safe(value.item())
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def dumps_row(row: Dict[str, Any]) -> str:
    return json.dumps(json_safe(row), ensure_ascii=False)


def is_jsonl(path: str) -> bool:
    return path.lower().endswith(".jsonl")


def summary_path(output_file: str) -> str:
    """``<output stem>_summary.json`` (never the results file itself)."""
    return f"{os.path.splitext(output_file)[0]}_summary.json"


def shard_dir(output_file: str) -> str:
    return os.path.join(os.path.dirname(output_file), SHARD_DIR)


def progress_path(output_file: str, progress_file: Optional[str]) -> Optional[str]:
    """A bare progress file name is placed next to output_file."""
    if not progress_file:
        return None
    if os.path.dirname(progress_file):
        return progress_file
    return os.path.join(os.path.dirname(output_file), progress_file)


def _temporary(path: str) -> str:
    return f"{path}.tmp-{os.getpid()}"


def write_text_atomic(path: str, text: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp = _temporary(path)
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def write_jsonl_lines(path: str, lines: Iterable[str]) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp = _temporary(path)
    with open(tmp, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line)
            f.write("\n")
    os.replace(tmp, path)


def write_results(rows: List[Dict[str, Any]], output_file: str) -> None:
    """Write rows as JSONL (``.jsonl``) or quoted CSV (anything else)."""
    if is_jsonl(output_file):
        write_jsonl_lines(output_file, (dumps_row(row) for row in rows))
        return
    import pandas as pd

    def cell(value):
        if isinstance(value, (list, dict)):
            return json.dumps(json_safe(value), ensure_ascii=False)
        return value

    df = pd.DataFrame([{k: cell(v) for k, v in row.items()} for row in rows])
    df.to_csv(output_file, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)


@dataclass
class RowCounts:
    """Aggregate status and token usage over finished rows."""
    done: int = 0
    ok: int = 0
    truncated: int = 0
    error: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def add(self, row: Dict[str, Any], status_column: str, usage_prefix: str) -> None:
        self.done += 1
        status = row.get(status_column)
        if status == "ok":
            self.ok += 1
        elif status == "truncated":
            self.truncated += 1
        else:
            self.error += 1
        for name in ("prompt_tokens", "completion_tokens"):
            value = row.get(f"{usage_prefix}{name}")
            if isinstance(value, int) and not isinstance(value, bool):
                setattr(self, name, getattr(self, name) + value)

    def describe(self) -> str:
        return f"{self.ok} ok, {self.truncated} truncated, {self.error} error"


def progress_record(total: int, counts: RowCounts, state: str) -> Dict[str, Any]:
    return {
        "total": total,
        "done": counts.done,
        "ok": counts.ok,
        "truncated": counts.truncated,
        "error": counts.error,
        "prompt_tokens": counts.prompt_tokens,
        "completion_tokens": counts.completion_tokens,
        "updated_at": datetime.datetime.now(datetime.timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "state": state,
    }


class S3Target:
    """Uploads run files under ``s3://{bucket}/{prefix}/``."""

    def __init__(self, bucket, prefix, region=None, endpoint_url=None, uploader="auto"):
        self.bucket = bucket
        self.prefix = (prefix or "").strip("/")
        self.region = region
        self.endpoint_url = endpoint_url
        self.uploader = uploader

    @classmethod
    def from_config(cls, config) -> Optional["S3Target"]:
        if not getattr(config, "s3_upload_results", False):
            return None
        if not config.s3_bucket:
            raise ValueError("s3_bucket is required when s3_upload_results is true")
        return cls(
            config.s3_bucket,
            config.s3_prefix,
            region=getattr(config, "s3_region", None),
            endpoint_url=getattr(config, "s3_endpoint_url", None),
            uploader=getattr(config, "s3_uploader", "auto"),
        )

    def key(self, relative: str) -> str:
        return f"{self.prefix}/{relative}" if self.prefix else relative

    def upload(self, local_path: str, relative: str) -> None:
        from utils.s3_utils import upload_file_to_s3

        upload_file_to_s3(
            local_path,
            self.bucket,
            self.key(relative),
            region=self.region,
            endpoint_url=self.endpoint_url,
            uploader=self.uploader,
        )

    def download_missing_shards(self, local_dir: str) -> int:
        """Copy shards present in S3 but not locally; returns how many."""
        from utils.s3_utils import download_file_from_s3, list_s3_keys

        shard_prefix = self.key(f"{SHARD_DIR}/")
        copied = 0
        for key in list_s3_keys(self.bucket, shard_prefix, self.region, self.endpoint_url):
            name = key[len(shard_prefix):]
            if not _SHARD_NAME.match(name):
                continue
            local_path = os.path.join(local_dir, name)
            if os.path.exists(local_path):
                continue
            os.makedirs(local_dir, exist_ok=True)
            tmp = download_file_from_s3(
                f"s3://{self.bucket}/{key}", region=self.region, endpoint_url=self.endpoint_url
            )
            shutil.move(tmp, local_path)
            copied += 1
        return copied


def shard_paths(directory: str) -> List[str]:
    if not os.path.isdir(directory):
        return []
    names = sorted(
        (int(match.group(1)), name)
        for name in os.listdir(directory)
        if (match := _SHARD_NAME.match(name))
    )
    return [os.path.join(directory, name) for _, name in names]


def next_shard_index(directory: str) -> int:
    indices = [
        int(_SHARD_NAME.match(os.path.basename(path)).group(1)) for path in shard_paths(directory)
    ]
    return max(indices) + 1 if indices else 0


def shard_name(index: int) -> str:
    return f"part-{index:05d}.jsonl"


def _iter_shard(path: str, row_id_column: str) -> Iterator[tuple]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            row_id = row.get(row_id_column) if isinstance(row, dict) else None
            if isinstance(row_id, int) and not isinstance(row_id, bool):
                yield row_id, row


def iter_shard_rows(directory: str, row_id_column: str, total: int) -> Iterator[Dict[str, Any]]:
    """Rows from every shard ordered by row id, one per id within [0, total)."""
    streams = [_iter_shard(path, row_id_column) for path in shard_paths(directory)]
    last_id = None
    for row_id, row in heapq.merge(*streams, key=lambda item: item[0]):
        if row_id == last_id or not 0 <= row_id < total:
            continue
        last_id = row_id
        yield row


def write_shard(directory: str, index: int, rows: List[Dict[str, Any]], row_id_column: str) -> str:
    path = os.path.join(directory, shard_name(index))
    ordered = sorted(rows, key=lambda row: row[row_id_column])
    write_jsonl_lines(path, (dumps_row(row) for row in ordered))
    return path
