"""Hosted-API inference against a real local HTTP server."""

import json
import logging
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import inference.inference as inference_module
import utils.api_utils as api_utils
from core.config import ExperimentConfig, InferenceConfig
from utils import result_shards
from utils.api_utils import ApiRequestError, HostedApiClient, _build_headers, _prepare_api_url
from utils.recipe_overrides import apply_overrides_to_recipe

SENTINEL_KEY = "sk-SENTINEL-7f3a9c1e5b"


class _Server:
    """Scripted chat-completions server. ``respond(request, n)`` returns
    ``(status, headers, body, delay_seconds)``; n counts calls per prompt."""

    def __init__(self):
        self.requests = []
        self.lock = threading.Lock()
        self.calls_per_prompt = {}
        self.in_flight = 0
        self.max_in_flight = 0
        self.respond = lambda request, n: (200, {}, _completion(_user(request)), 0)
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                request = {"path": self.path, "headers": dict(self.headers), "body": body}
                with server.lock:
                    server.requests.append(request)
                    key = _user(request)
                    n = server.calls_per_prompt.get(key, 0)
                    server.calls_per_prompt[key] = n + 1
                    server.in_flight += 1
                    server.max_in_flight = max(server.max_in_flight, server.in_flight)
                try:
                    status, headers, payload, delay = server.respond(request, n)
                    if delay:
                        time.sleep(delay)
                    data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    for name, value in headers.items():
                        self.send_header(name, value)
                    self.end_headers()
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    with server.lock:
                        server.in_flight -= 1

            def log_message(self, *_args):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.httpd.handle_error = lambda *_args: None
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1/chat/completions"
        self.thread = threading.Thread(
            target=self.httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
        )
        self.thread.start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def _user(request):
    messages = request["body"].get("messages") or [{}]
    return messages[-1].get("content")


def _completion(text, finish_reason="stop", reasoning=None, usage=(7, 3)):
    message = {"role": "assistant", "content": text}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return {
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1]},
    }


@pytest.fixture
def server():
    srv = _Server()
    yield srv
    srv.close()


@pytest.fixture
def no_sleep(monkeypatch):
    delays = []
    monkeypatch.setattr(api_utils, "_sleep", delays.append)
    return delays


def _config(server, tmp_path, **overrides):
    values = dict(
        model="gateway-model",
        api_endpoint=server.url,
        api_key="test-key",
        dataset_name="unused",
        dataset_columns=["q"],
        user_prompt="{q}",
        output_file=str(tmp_path / "out" / "results.jsonl"),
        api_retry_backoff_seconds=0.01,
        api_timeout_seconds=5,
    )
    values.update(overrides)
    return InferenceConfig(**values)


def _rows(n):
    return [{"q": f"question {i}"} for i in range(n)]


def _run(monkeypatch, config, rows):
    monkeypatch.setattr(inference_module, "load_raw_dataset", lambda _config: rows)
    return inference_module.run_inference(config)


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


# ---------------------------------------------------------------------------
# Request shape
# ---------------------------------------------------------------------------


def test_gateway_header_body_mapping_and_extras(server, tmp_path, monkeypatch):
    monkeypatch.setenv("GATEWAY_API_KEY", "gw-key")
    config = _config(
        server,
        tmp_path,
        api_key=None,
        api_key_env="GATEWAY_API_KEY",
        api_auth_header="X-Api-Key",
        api_auth_prefix="",
        api_extra_headers={"X-Team": "ml"},
        api_extra_body={"reasoning_effort": "high"},
        api_max_tokens_field="max_completion_tokens",
        temperature=None,
        top_p=None,
        max_new_tokens=64,
        system_prompt="Be brief about {q}.",
    )
    _run(monkeypatch, config, _rows(1))

    [request] = server.requests
    headers = {k.lower(): v for k, v in request["headers"].items()}
    assert headers["x-api-key"] == "gw-key"
    assert headers["x-team"] == "ml"
    assert "authorization" not in headers
    assert request["body"] == {
        "model": "gateway-model",
        "messages": [
            {"role": "system", "content": "Be brief about question 0."},
            {"role": "user", "content": "question 0"},
        ],
        "max_completion_tokens": 64,
        "reasoning_effort": "high",
    }


def test_default_auth_is_bearer_and_sampling_params_sent(server, tmp_path, monkeypatch):
    config = _config(server, tmp_path, temperature=0.5, top_p=0.8, max_new_tokens=9)
    _run(monkeypatch, config, _rows(1))

    [request] = server.requests
    assert request["headers"]["Authorization"] == "Bearer test-key"
    body = request["body"]
    assert (body["max_tokens"], body["temperature"], body["top_p"]) == (9, 0.5, 0.8)
    assert "max_completion_tokens" not in body


def test_openai_hostname_parses_chat_completion_choices():
    completion = api_utils.parse_api_completion(
        _completion("hello", reasoning="think", usage=(5, 2)),
        "https://api.openai.com/v1/chat/completions",
    )
    assert (completion.text, completion.reasoning, completion.finish_reason) == (
        "hello", "think", "stop",
    )
    assert (completion.prompt_tokens, completion.completion_tokens) == (5, 2)


def test_gemini_key_is_sent_in_header_not_url():
    endpoint = "https://generativelanguage.googleapis.com/v1beta/models/g:generateContent"
    assert SENTINEL_KEY not in _prepare_api_url(endpoint, SENTINEL_KEY)
    headers = _build_headers(endpoint, SENTINEL_KEY, InferenceConfig())
    assert headers["x-goog-api-key"] == SENTINEL_KEY
    assert "Authorization" not in headers
    assert api_utils.redact_text(f"{endpoint}?key={SENTINEL_KEY}&alt=sse", []) == (
        f"{endpoint}?key=***&alt=sse"
    )


# ---------------------------------------------------------------------------
# Retries and errors
# ---------------------------------------------------------------------------


def test_429_honors_retry_after_then_succeeds(server, tmp_path, monkeypatch, no_sleep):
    server.respond = lambda request, n: (
        (429, {"Retry-After": "3"}, {"error": "slow down"}, 0)
        if n == 0
        else (200, {}, _completion("fine"), 0)
    )
    _run(monkeypatch, _config(server, tmp_path), _rows(1))

    [row] = _read_jsonl(tmp_path / "out" / "results.jsonl")
    assert row["__status"] == "ok" and row["response"] == "fine"
    assert len(server.requests) == 2
    assert len(no_sleep) == 1 and 3 <= no_sleep[0] < 4


def test_5xx_is_retried(server, tmp_path, monkeypatch, no_sleep):
    server.respond = lambda request, n: (
        (503, {}, b"<html>bad gateway</html>", 0) if n < 2 else (200, {}, _completion("ok"), 0)
    )
    _run(monkeypatch, _config(server, tmp_path), _rows(1))

    [row] = _read_jsonl(tmp_path / "out" / "results.jsonl")
    assert row["__status"] == "ok"
    assert len(server.requests) == 3 and len(no_sleep) == 2


def test_timeout_is_retried(server, tmp_path, monkeypatch, no_sleep):
    server.respond = lambda request, n: (
        (200, {}, _completion("late"), 1.0) if n == 0 else (200, {}, _completion("ok"), 0)
    )
    _run(monkeypatch, _config(server, tmp_path, api_timeout_seconds=0.2), _rows(1))

    [row] = _read_jsonl(tmp_path / "out" / "results.jsonl")
    assert (row["__status"], row["response"]) == ("ok", "ok")
    assert len(no_sleep) == 1


def test_400_is_a_row_error_without_retry(server, tmp_path, monkeypatch, no_sleep):
    server.respond = lambda request, n: (
        (400, {}, {"error": "bad prompt"}, 0)
        if _user(request) == "question 1"
        else (200, {}, _completion("ok"), 0)
    )
    summary = _run(monkeypatch, _config(server, tmp_path), _rows(3))

    rows = _read_jsonl(tmp_path / "out" / "results.jsonl")
    assert [r["__status"] for r in rows] == ["ok", "error", "ok"]
    assert "HTTP 400" in rows[1]["__error"] and "bad prompt" in rows[1]["__error"]
    assert rows[1]["response"] is None
    assert server.calls_per_prompt["question 1"] == 1 and no_sleep == []
    assert (summary["successful_examples"], summary["failed_examples"]) == (2, 1)


def test_exhausted_retries_mark_row_error(server, tmp_path, monkeypatch, no_sleep):
    server.respond = lambda request, n: (500, {}, {"error": "boom"}, 0)
    _run(monkeypatch, _config(server, tmp_path, api_max_retries=2), _rows(1))

    [row] = _read_jsonl(tmp_path / "out" / "results.jsonl")
    assert row["__status"] == "error" and "HTTP 500" in row["__error"]
    assert len(server.requests) == 3


def test_empty_content_is_an_error_not_a_silent_success(server, tmp_path, monkeypatch):
    server.respond = lambda request, n: (200, {}, _completion(None, finish_reason="length"), 0)
    _run(monkeypatch, _config(server, tmp_path), _rows(1))

    [row] = _read_jsonl(tmp_path / "out" / "results.jsonl")
    assert row["__status"] == "error"
    assert "empty response content" in row["__error"]


@pytest.mark.parametrize("finish_reason", ["length", "max_tokens", "MAX_TOKENS"])
def test_answer_cut_off_at_max_new_tokens_is_truncated(
    server, tmp_path, monkeypatch, finish_reason
):
    server.respond = lambda request, n: (
        (200, {}, _completion("Look no further, adventurer", finish_reason=finish_reason), 0)
        if _user(request) == "question 1"
        else (200, {}, _completion("done"), 0)
    )
    config = _config(
        server,
        tmp_path,
        max_new_tokens=300,
        messages_output_column="__messages",
        shard_rows=2,
        progress_file="progress.json",
    )
    summary = _run(monkeypatch, config, _rows(3))

    out = tmp_path / "out"
    rows = _read_jsonl(out / "results.jsonl")
    assert [r["__status"] for r in rows] == ["ok", "truncated", "ok"]
    truncated = rows[1]
    assert truncated["response"] == "Look no further, adventurer"
    assert truncated["__finish_reason"] == finish_reason
    assert "max_new_tokens (300)" in truncated["__error"]
    assert truncated["__messages"] is None and rows[0]["__messages"] is not None
    assert (
        summary["successful_examples"], summary["truncated_examples"], summary["failed_examples"]
    ) == (2, 1, 0)
    progress = json.loads((out / "progress.json").read_text())
    assert (progress["ok"], progress["truncated"], progress["error"]) == (2, 1, 0)


def test_truncated_rows_are_not_resent_on_resume(server, tmp_path, monkeypatch):
    server.respond = lambda request, n: (200, {}, _completion("cut", finish_reason="length"), 0)
    config = _config(server, tmp_path, shard_rows=1)
    _run(monkeypatch, config, _rows(2))
    sent = len(server.requests)

    summary = _run(monkeypatch, config, _rows(2))
    assert len(server.requests) == sent
    assert summary["truncated_examples"] == 2


def test_truncated_messages_column_is_skipped_by_sft():
    from utils.tool_conversations import normalize_conversation_row

    row = normalize_conversation_row({"__messages": None}, "__messages", None, None)
    assert row["conversation"] is None and row["invalid_reason"]


def test_fail_fast_when_every_request_of_first_shard_is_unauthorized(
    server, tmp_path, monkeypatch
):
    server.respond = lambda request, n: (401 if n % 2 == 0 else 403, {}, {"error": "denied"}, 0)
    config = _config(server, tmp_path, shard_rows=4, api_concurrency=2)
    with pytest.raises(RuntimeError, match="authentication failed"):
        _run(monkeypatch, config, _rows(20))

    assert len(server.requests) < 20
    assert result_shards.shard_paths(result_shards.shard_dir(config.output_file)) == []


def test_some_auth_failures_do_not_fail_fast(server, tmp_path, monkeypatch):
    server.respond = lambda request, n: (
        (401, {}, {}, 0) if _user(request) != "question 2" else (200, {}, _completion("ok"), 0)
    )
    summary = _run(monkeypatch, _config(server, tmp_path, shard_rows=4), _rows(4))
    assert (summary["successful_examples"], summary["failed_examples"]) == (1, 3)


# ---------------------------------------------------------------------------
# Concurrency, output, shards, resume, progress
# ---------------------------------------------------------------------------


def test_concurrency_keeps_every_row_in_order(server, tmp_path, monkeypatch):
    server.respond = lambda request, n: (
        200, {}, _completion(f"answer to {_user(request)}"), 0.05 * (hash(_user(request)) % 3)
    )
    config = _config(server, tmp_path, api_concurrency=8, output_file=str(tmp_path / "r.csv"))
    _run(monkeypatch, config, _rows(30))

    df = pd.read_csv(tmp_path / "r.csv")
    assert df["__row_id"].tolist() == list(range(30))
    assert df["response"].tolist() == [f"answer to question {i}" for i in range(30)]
    assert server.max_in_flight > 1


def test_messages_output_column_and_metadata_columns(server, tmp_path, monkeypatch):
    server.respond = lambda request, n: (
        200, {}, _completion("A1", reasoning="thinking", usage=(11, 4)), 0
    )
    config = _config(
        server, tmp_path, messages_output_column="__messages", system_prompt="SYS {q}"
    )
    _run(monkeypatch, config, _rows(1))

    [row] = _read_jsonl(tmp_path / "out" / "results.jsonl")
    assert row["__messages"] == [
        {"role": "system", "content": "SYS question 0"},
        {"role": "user", "content": "question 0"},
        {"role": "assistant", "content": "A1"},
    ]
    assert row == {
        "q": "question 0",
        "response": "A1",
        "confidence": None,
        "__row_id": 0,
        "__status": "ok",
        "__error": None,
        "__finish_reason": "stop",
        "__prompt_tokens": 11,
        "__completion_tokens": 4,
        "__reasoning": "thinking",
        "__messages": row["__messages"],
    }


def test_error_rows_have_no_messages(server, tmp_path, monkeypatch):
    server.respond = lambda request, n: (400, {}, {}, 0)
    _run(monkeypatch, _config(server, tmp_path, messages_output_column="__messages"), _rows(1))
    [row] = _read_jsonl(tmp_path / "out" / "results.jsonl")
    assert row["__messages"] is None


def test_shards_progress_and_summary(server, tmp_path, monkeypatch):
    config = _config(
        server, tmp_path, shard_rows=2, progress_file="progress.json", api_concurrency=3
    )
    summary = _run(monkeypatch, config, _rows(5))

    out = tmp_path / "out"
    assert sorted(p.name for p in (out / "shards").iterdir()) == [
        "part-00000.jsonl", "part-00001.jsonl", "part-00002.jsonl",
    ]
    assert [r["__row_id"] for r in _read_jsonl(out / "results.jsonl")] == [0, 1, 2, 3, 4]
    progress = json.loads((out / "progress.json").read_text())
    updated_at = progress.pop("updated_at")
    assert updated_at.endswith("Z")
    assert progress == {
        "total": 5, "done": 5, "ok": 5, "truncated": 0, "error": 0,
        "prompt_tokens": 35, "completion_tokens": 15, "state": "completed",
    }
    assert json.loads((out / "results_summary.json").read_text()) == summary
    assert summary["successful_examples"] == 5 and summary["failed_examples"] == 0
    assert summary["prompt_tokens"] == 35


def test_resume_skips_rows_present_in_shards(server, tmp_path, monkeypatch):
    config = _config(server, tmp_path, shard_rows=2)
    shard_dir = result_shards.shard_dir(config.output_file)
    done = [
        {"q": "question 1", "response": "old", "__row_id": 1, "__status": "ok"},
        {"q": "question 3", "response": None, "__row_id": 3, "__status": "error"},
    ]
    result_shards.write_shard(shard_dir, 0, done, "__row_id")

    summary = _run(monkeypatch, config, _rows(5))

    assert sorted(server.calls_per_prompt) == ["question 0", "question 2", "question 4"]
    rows = _read_jsonl(config.output_file)
    assert [r["__row_id"] for r in rows] == [0, 1, 2, 3, 4]
    assert rows[1]["response"] == "old"
    assert (summary["successful_examples"], summary["failed_examples"]) == (4, 1)
    assert (Path(shard_dir) / "part-00001.jsonl").exists()


def test_s3_resume_and_uploads(server, tmp_path, monkeypatch):
    import utils.s3_utils as s3_utils

    remote_shard = tmp_path / "remote" / "part-00000.jsonl"
    result_shards.write_shard(
        str(remote_shard.parent), 0, [{"q": "question 0", "__row_id": 0, "__status": "ok"}],
        "__row_id",
    )
    uploads = []
    monkeypatch.setattr(
        s3_utils, "list_s3_keys",
        lambda bucket, prefix, *_: [f"{prefix}part-00000.jsonl", f"{prefix}README"],
    )

    def fake_download(uri, region=None, endpoint_url=None):
        assert uri == "s3://bkt/runs/d1/shards/part-00000.jsonl"
        copy = tmp_path / "downloaded.jsonl"
        copy.write_text(remote_shard.read_text())
        return str(copy)

    monkeypatch.setattr(s3_utils, "download_file_from_s3", fake_download)
    monkeypatch.setattr(
        s3_utils, "upload_file_to_s3",
        lambda local, bucket, key, **_: uploads.append((bucket, key, Path(local).name)),
    )
    config = _config(
        server, tmp_path, shard_rows=1, progress_file="progress.json",
        s3_upload_results=True, s3_bucket="bkt", s3_prefix="runs/d1/",
    )
    _run(monkeypatch, config, _rows(2))

    assert list(server.calls_per_prompt) == ["question 1"]
    keys = [key for _, key, _ in uploads]
    assert "runs/d1/shards/part-00001.jsonl" in keys
    assert keys[-3:] == [
        "runs/d1/results.jsonl", "runs/d1/results_summary.json", "runs/d1/progress.json",
    ]
    assert {bucket for bucket, _, _ in uploads} == {"bkt"}


def test_summary_path_never_overwrites_results():
    assert result_shards.summary_path("o/results.jsonl") == "o/results_summary.json"
    assert result_shards.summary_path("o/results.json") == "o/results_summary.json"
    assert result_shards.summary_path("o/results.csv") == "o/results_summary.json"


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"api_concurrency": 0}, "api_concurrency"),
        ({"shard_rows": -1}, "shard_rows"),
        ({"api_max_retries": -1}, "api_max_retries"),
        ({"api_timeout_seconds": 0}, "api_timeout_seconds"),
        ({"api_requests_per_minute": 0}, "api_requests_per_minute"),
        ({"api_max_tokens_field": "tokens"}, "api_max_tokens_field"),
        ({"api_key": "k", "api_key_env": "E"}, "not both"),
        ({"api_key": None, "shard_rows": 5}, "require API mode"),
        ({"api_key": None, "messages_output_column": "__m"}, "require API mode"),
        ({"model_paths": ["ckpt"]}, "model_paths"),
        ({"max_rows": 0}, "max_rows"),
        ({"max_rows": "10"}, "max_rows"),
        ({"api_key": None, "model": None, "max_rows": -2}, "max_rows"),
    ],
)
def test_config_validation(overrides, error):
    values = {"model": "m", "api_key": "k", **overrides}
    with pytest.raises(ValueError, match=error):
        InferenceConfig(**values)


def test_missing_api_key_env_is_a_clear_setup_error(server, tmp_path, monkeypatch):
    monkeypatch.delenv("FAI_RL_TEST_MISSING_KEY", raising=False)
    config = _config(server, tmp_path, api_key=None, api_key_env="FAI_RL_TEST_MISSING_KEY")
    with pytest.raises(ValueError, match="FAI_RL_TEST_MISSING_KEY.*not set"):
        _run(monkeypatch, config, _rows(1))
    assert server.requests == []


def test_requests_per_minute_spaces_requests(server, tmp_path, monkeypatch, no_sleep):
    config = _config(server, tmp_path, api_requests_per_minute=600)
    client = HostedApiClient(config)
    for _ in range(3):
        client.complete("hi")
    # Sleeping is stubbed, so each request waits one more 0.1s slot.
    assert len(no_sleep) == 2
    assert 0 < no_sleep[0] <= 0.1 and 0.1 < no_sleep[1] <= 0.2


# ---------------------------------------------------------------------------
# Secret hygiene
# ---------------------------------------------------------------------------


def test_sentinel_key_never_reaches_files_or_logs(server, tmp_path, monkeypatch, capfd, no_sleep):
    monkeypatch.setenv("GATEWAY_API_KEY", SENTINEL_KEY)
    header_secret = "hdr-SENTINEL-c0ffee"

    def respond(request, n):
        user = _user(request)
        if user == "question 1":
            return 500, {}, {"error": f"echo {SENTINEL_KEY} {header_secret}"}, 0
        if user == "question 2":
            return 401, {}, {"error": f"bad key {SENTINEL_KEY}"}, 0
        return 200, {}, _completion("ok"), 0

    server.respond = respond
    recipe = {
        "inference": {
            "model": "gateway-model",
            "api_endpoint": server.url,
            "api_key_env": "GATEWAY_API_KEY",
            "api_auth_header": "X-Api-Key",
            "api_auth_prefix": "",
            "dataset_name": "unused",
            "dataset_columns": ["q"],
            "user_prompt": "{q}",
            "output_file": str(tmp_path / "out" / "results.jsonl"),
            "shard_rows": 2,
            "progress_file": "progress.json",
            "messages_output_column": "__messages",
            "api_max_retries": 1,
            "api_retry_backoff_seconds": 0.01,
        }
    }
    records = []

    class Collect(logging.Handler):
        def emit(self, record):
            records.append(self.format(record))

    handler = Collect(level=logging.DEBUG)
    root = logging.getLogger()
    root.addHandler(handler)
    old_level = root.level
    root.setLevel(logging.DEBUG)
    recipe_logger = logging.getLogger("FAI-RL.recipe")
    recipe_logger.addHandler(handler)
    try:
        recipe = apply_overrides_to_recipe(
            recipe,
            [
                f"inference.api_extra_headers={{'X-Secret': '{header_secret}'}}",
                "inference.max_new_tokens=32",
            ],
        )
        config = ExperimentConfig.inference_config_from_dict(recipe)
        _run(monkeypatch, config, _rows(4))
    finally:
        root.removeHandler(handler)
        root.setLevel(old_level)
        recipe_logger.removeHandler(handler)

    assert any("max_new_tokens = 32" in r for r in records)
    assert server.requests[0]["headers"]["X-Api-Key"] == SENTINEL_KEY
    assert server.requests[0]["headers"]["X-Secret"] == header_secret
    captured = capfd.readouterr()
    texts = {"stdout": captured.out, "stderr": captured.err, "logs": "\n".join(records)}
    for path in tmp_path.rglob("*"):
        if path.is_file():
            texts[str(path)] = path.read_text(errors="replace")
    assert any(p.endswith("results_summary.json") for p in texts)
    for name, text in texts.items():
        assert SENTINEL_KEY not in text, name
        assert header_secret not in text, name

    summary = json.loads((tmp_path / "out" / "results_summary.json").read_text())
    assert summary["config"]["api_extra_headers"] == {"X-Secret": "***"}
    assert summary["config"]["api_key_env"] == "GATEWAY_API_KEY"
    assert (summary["successful_examples"], summary["failed_examples"]) == (2, 2)
    rows = _read_jsonl(tmp_path / "out" / "results.jsonl")
    assert "***" in rows[1]["__error"] and "***" in rows[2]["__error"]


def test_config_to_dict_redacts_api_key():
    values = InferenceConfig(model="m", api_key=SENTINEL_KEY).to_dict()
    assert values["api_key"] == "***"
    assert SENTINEL_KEY not in json.dumps(values)


def test_override_log_redacts_secret_keys(caplog):
    logger = logging.getLogger("FAI-RL.recipe")
    logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.INFO, logger="FAI-RL.recipe"):
            apply_overrides_to_recipe(
                {},
                [f"inference.api_key={SENTINEL_KEY}", "wandb.api_key=abc", "inference.top_p=0.5"],
            )
    finally:
        logger.removeHandler(caplog.handler)
    assert SENTINEL_KEY not in caplog.text and "abc" not in caplog.text
    assert "inference.top_p = 0.5" in caplog.text


def test_client_errors_are_redacted(server, tmp_path):
    server.respond = lambda request, n: (400, {}, {"error": f"you sent {SENTINEL_KEY}"}, 0)
    client = HostedApiClient(_config(server, tmp_path, api_key=SENTINEL_KEY))
    with pytest.raises(ApiRequestError) as err:
        client.complete("hi")
    assert SENTINEL_KEY not in str(err.value) and "***" in str(err.value)
    assert err.value.status == 400 and not err.value.retryable


# ---------------------------------------------------------------------------
# Distilled output feeds SFT conversation loading
# ---------------------------------------------------------------------------


def test_distilled_jsonl_error_rows_are_skipped_by_sft_conversations(
    server, tmp_path, monkeypatch
):
    from datasets import Features, Value

    from utils.dataset_utils import load_raw_dataset
    from utils.tool_conversations import normalize_conversation_row

    server.respond = lambda request, n: (
        (400, {}, {}, 0) if _user(request) in ("question 0", "question 2") else
        (200, {}, _completion(f"answer {_user(request)}"), 0)
    )
    config = _config(server, tmp_path, messages_output_column="__messages", system_prompt="S")
    _run(monkeypatch, config, _rows(4))

    from types import SimpleNamespace

    dataset = load_raw_dataset(
        SimpleNamespace(name=config.output_file, dataset_name=config.output_file, split="train",
                        dataset_split="train", subset=None, dataset_subset=None)
    )
    normalized = dataset.map(
        normalize_conversation_row,
        fn_kwargs={"messages_column": "__messages", "tools_column": None, "system_prompt": None},
        remove_columns=dataset.column_names,
        features=Features({"conversation": Value("string"), "invalid_reason": Value("string")}),
    )
    kept = normalized.filter(lambda row: row["invalid_reason"] is None)
    assert len(kept) == 2
    assert [json.loads(c)["messages"][-1]["content"] for c in kept["conversation"]] == [
        "answer question 1", "answer question 3",
    ]


# ---------------------------------------------------------------------------
# Local (non-API) inference output formats
# ---------------------------------------------------------------------------


def test_local_inference_jsonl_output_and_summary_path(monkeypatch, tmp_path):
    output_file = tmp_path / "results.jsonl"
    config = InferenceConfig(
        model_paths=["checkpoint-100"],
        dataset_name="unused",
        dataset_columns=["question"],
        system_prompt="{question}",
        output_file=str(output_file),
    )
    monkeypatch.setattr(inference_module, "load_raw_dataset", lambda _c: [{"question": "Why?"}])
    monkeypatch.setattr(
        inference_module, "load_model_and_tokenizer", lambda _c: (object(), object())
    )
    monkeypatch.setattr(
        inference_module, "generate_response", lambda *_a, **_k: ("Because.", 0.75)
    )
    inference_module.run_inference(config)

    assert _read_jsonl(output_file) == [
        {"question": "Why?", "response": "Because.", "confidence": 0.75}
    ]
    summary = json.loads((tmp_path / "results_summary.json").read_text())
    assert summary["successful_examples"] == 1


def test_distill_recipe_loads_without_a_key():
    config = ExperimentConfig.load_inference_config(
        str(REPO_ROOT / "recipes" / "inference" / "openai_compatible_distill.yaml")
    )
    assert config.uses_api and config.api_key is None
    assert (config.api_key_env, config.api_auth_header, config.api_auth_prefix) == (
        "GATEWAY_API_KEY", "X-Api-Key", "",
    )
    assert config.top_p is None and config.messages_output_column == "__messages"


def test_max_rows_caps_requests_progress_and_resume(server, tmp_path, monkeypatch):
    from datasets import Dataset

    config = _config(
        server, tmp_path, max_rows=3, shard_rows=2, progress_file="progress.json"
    )
    shard_dir = result_shards.shard_dir(config.output_file)
    result_shards.write_shard(
        shard_dir,
        0,
        [
            {"q": "question 1", "response": "old", "__row_id": 1, "__status": "ok"},
            {"q": "question 4", "response": "old", "__row_id": 4, "__status": "ok"},
        ],
        "__row_id",
    )
    summary = _run(monkeypatch, config, Dataset.from_list(_rows(5)))

    assert sorted(server.calls_per_prompt) == ["question 0", "question 2"]
    assert [r["__row_id"] for r in _read_jsonl(config.output_file)] == [0, 1, 2]
    progress = json.loads((tmp_path / "out" / "progress.json").read_text())
    assert (progress["total"], progress["done"], progress["ok"]) == (3, 3, 3)
    assert summary["total_examples"] == 3 and summary["successful_examples"] == 3


def test_max_rows_above_dataset_size_keeps_every_row(server, tmp_path, monkeypatch):
    config = _config(server, tmp_path, max_rows=50, progress_file="progress.json")
    _run(monkeypatch, config, _rows(4))

    assert len(server.requests) == 4
    progress = json.loads((tmp_path / "out" / "progress.json").read_text())
    assert progress["total"] == 4


def test_local_inference_honors_max_rows(monkeypatch, tmp_path):
    output_file = tmp_path / "results.csv"
    config = InferenceConfig(
        model_paths=["checkpoint-100"],
        dataset_name="unused",
        dataset_columns=["question"],
        system_prompt="{question}",
        output_file=str(output_file),
        max_rows=2,
    )
    prompts = []
    monkeypatch.setattr(
        inference_module, "load_raw_dataset",
        lambda _c: [{"question": f"Q{i}"} for i in range(5)],
    )
    monkeypatch.setattr(
        inference_module, "load_model_and_tokenizer", lambda _c: (object(), object())
    )

    def fake_generate(_model, _tokenizer, prompt, *_a, **_k):
        prompts.append(prompt)
        return "A", 0.5

    monkeypatch.setattr(inference_module, "generate_response", fake_generate)
    inference_module.run_inference(config)

    assert prompts == ["Q0", "Q1"]
    assert pd.read_csv(output_file)["question"].tolist() == ["Q0", "Q1"]
    summary = json.loads((tmp_path / "results_summary.json").read_text())
    assert summary["total_examples"] == 2 and summary["failed_examples"] == 0
