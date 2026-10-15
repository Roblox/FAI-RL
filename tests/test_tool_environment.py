import asyncio
import inspect
import json
import logging

import pytest
import requests

from core.config import (
    ToolDefinition,
    ToolEnvironmentConfig,
)
from trainers.tool_environment import GatewayToolEnvironment, tool_catalog_prompt


@pytest.fixture(autouse=True)
def managed_gateway_identity(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "FAI_RL_TOOL_GATEWAY_ALLOWED_URLS",
        "https://tools.example.test/v1",
    )
    token_file = tmp_path / "default-tool-token"
    token_file.write_text("pod-bound-test-token")
    monkeypatch.setenv("FAI_RL_TOOL_GATEWAY_TOKEN_FILE", str(token_file))


def call_tool(environment, tool_name, arguments):
    return asyncio.run(environment.call_tool(tool_name, arguments))


def config(**overrides):
    values = {
        "version": "v1",
        "gateway_url": "https://tools.example.test/v1",
        "server_id": "inventory",
        "tools": [
            ToolDefinition(
                name="lookup_item",
                description="Look up an item.",
                input_schema={
                    "type": "object",
                    "properties": {"item_id": {"type": "string"}},
                    "required": ["item_id"],
                },
            )
        ],
        "max_turns": 3,
        "max_tool_calls": 2,
        "tool_timeout_seconds": 4,
        "max_episode_seconds": 30,
        "max_request_bytes": 1024,
        "max_response_bytes": 1024,
        "max_retries": 1,
        "retry_backoff_seconds": 0,
    }
    values.update(overrides)
    return ToolEnvironmentConfig(**values)


class FakeResponse:
    def __init__(self, body, status_code=200, *, raw_content=None, headers=None):
        self._body = body
        self.status_code = status_code
        self.content = (
            raw_content if raw_content is not None else json.dumps(body).encode()
        )
        self.headers = headers or {}
        self.closed = False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}", response=self)

    def iter_content(self, chunk_size):
        for offset in range(0, len(self.content), chunk_size):
            yield self.content[offset : offset + chunk_size]

    def close(self):
        self.closed = True


class FakeSession:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_config_rejects_unmanaged_or_duplicate_tools():
    with pytest.raises(ValueError, match="https"):
        config(gateway_url="http://tools.example.test")
    with pytest.raises(ValueError, match="unique"):
        config(
            tools=[
                ToolDefinition(name="lookup_item", input_schema={"type": "object"}),
                ToolDefinition(name="lookup_item", input_schema={"type": "object"}),
            ]
        )
    with pytest.raises(ValueError, match="object"):
        config(
            tools=[ToolDefinition(name="lookup_item", input_schema={"type": "array"})]
        )
    with pytest.raises(ValueError, match="invalid"):
        config(
            tools=[
                ToolDefinition(
                    name="lookup_item",
                    input_schema={
                        "type": "object",
                        "properties": "not-an-object",
                    },
                )
            ]
        )


def test_catalog_prompt_exposes_only_public_tool_contract():
    prompt = tool_catalog_prompt(config())

    assert "lookup_item" in prompt
    assert "item_id" in prompt
    assert "https://tools.example.test" not in prompt
    assert "Authorization" not in prompt


def test_finalizes_trajectory_from_the_last_trl_completion_message():
    environment = GatewayToolEnvironment(config())
    environment.reset()

    environment._finalize(
        [
            {
                "role": "assistant",
                "tool_calls": [{"type": "function"}],
            }
        ]
    )
    assert environment.trajectory["terminationReason"] == "turn_limit"

    environment.reset()
    environment._finalize([{"role": "assistant", "content": "final answer"}])
    assert environment.trajectory["terminationReason"] == "final_answer"


def test_finalization_preserves_a_bounded_tool_failure_reason():
    environment = GatewayToolEnvironment(config(max_retries=0))
    environment.reset()
    environment.termination_reason = "timeout"

    environment._finalize([{"role": "assistant", "content": "fallback answer"}])

    assert environment.trajectory["terminationReason"] == "timeout"


def test_trl_discovers_only_call_tool_as_a_public_environment_method():
    environment = GatewayToolEnvironment(config())

    methods = {
        name
        for name, _ in inspect.getmembers(environment, predicate=inspect.ismethod)
        if not name.startswith("_") and name != "reset"
    }

    assert methods == {"call_tool"}


def test_calls_only_allowlisted_tools_and_records_trajectory():
    session = FakeSession([FakeResponse({"content": {"available": True}})])
    environment = GatewayToolEnvironment(
        config(),
        session=session,
        token="job-token",
        uuid_factory=iter(["rollout-1", "call-1"]).__next__,
    )
    environment.reset(prompt="Find item 42")

    result = call_tool(environment, "lookup_item", {"item_id": "42"})

    assert json.loads(result) == {"available": True}
    assert session.calls[0][0] == "https://tools.example.test/v1/tool-calls"
    request = session.calls[0][1]
    assert request["headers"]["Authorization"] == "Bearer job-token"
    assert request["headers"]["Idempotency-Key"] == "call-1"
    assert request["stream"] is True
    assert request["allow_redirects"] is False
    assert request["json"] == {
        "version": "v1",
        "rolloutId": "rollout-1",
        "callId": "call-1",
        "serverId": "inventory",
        "tool": "lookup_item",
        "arguments": {"item_id": "42"},
    }
    trajectory = environment.trajectory
    assert trajectory["rolloutId"] == "rollout-1"
    assert trajectory["toolCalls"][0]["status"] == "ok"
    assert trajectory["toolCalls"][0]["result"] == {"available": True}


def test_rejects_schema_invalid_or_oversized_arguments_before_gateway_call():
    session = FakeSession([])
    environment = GatewayToolEnvironment(
        config(max_request_bytes=512),
        session=session,
        token="job-token",
    )
    environment.reset()

    invalid = json.loads(call_tool(environment, "lookup_item", {}))
    oversized = json.loads(
        call_tool(
            environment,
            "lookup_item",
            {"item_id": "x" * 1024},
        )
    )

    assert invalid["status"] == "invalid_arguments"
    assert oversized["status"] == "request_too_large"
    assert session.calls == []


def test_reads_rotating_pod_bound_identity_from_token_file(tmp_path):
    token_file = tmp_path / "token"
    token_file.write_text("pod-token-1")
    session = FakeSession(
        [
            FakeResponse({"content": "first"}),
            FakeResponse({"content": "second"}),
        ]
    )
    environment = GatewayToolEnvironment(
        config(),
        session=session,
        token_file=str(token_file),
    )
    environment.reset()

    call_tool(environment, "lookup_item", {"item_id": "1"})
    token_file.write_text("pod-token-2")
    call_tool(environment, "lookup_item", {"item_id": "2"})

    assert session.calls[0][1]["headers"]["Authorization"] == "Bearer pod-token-1"
    assert session.calls[1][1]["headers"]["Authorization"] == "Bearer pod-token-2"


def test_does_not_read_a_static_gateway_token_from_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("FAI_RL_TOOL_GATEWAY_TOKEN", "static-shared-secret")
    session = FakeSession([])
    environment = GatewayToolEnvironment(
        config(),
        session=session,
        token_file=str(tmp_path / "missing-token"),
    )
    environment.reset()

    result = json.loads(call_tool(environment, "lookup_item", {"item_id": "1"}))

    assert result["status"] == "identity_unavailable"
    assert session.calls == []


def test_pod_identity_can_call_only_an_operator_allowlisted_gateway(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv(
        "FAI_RL_TOOL_GATEWAY_ALLOWED_URLS",
        "https://different.example.test/v1",
    )
    token_file = tmp_path / "token"
    token_file.write_text("pod-token")
    session = FakeSession([])
    environment = GatewayToolEnvironment(
        config(),
        session=session,
        token_file=str(token_file),
    )
    environment.reset()

    result = json.loads(call_tool(environment, "lookup_item", {"item_id": "1"}))

    assert result["status"] == "gateway_not_allowed"
    assert session.calls == []


def test_retry_reuses_call_id_and_tool_limit_is_bounded():
    session = FakeSession(
        [
            requests.Timeout("slow"),
            FakeResponse({"content": "ok"}),
            FakeResponse({"content": "unused"}),
        ]
    )
    ids = iter(["rollout-1", "call-1", "call-2"])
    environment = GatewayToolEnvironment(
        config(max_tool_calls=1),
        session=session,
        uuid_factory=ids.__next__,
        sleep=lambda _: None,
    )
    environment.reset()

    assert call_tool(environment, "lookup_item", {"item_id": "42"}) == '"ok"'
    assert len(session.calls) == 2
    assert {call[1]["headers"]["Idempotency-Key"] for call in session.calls} == {
        "call-1"
    }

    limited = json.loads(call_tool(environment, "lookup_item", {"item_id": "43"}))
    assert limited["error"] == "tool_call_limit_exceeded"
    assert len(session.calls) == 2


def test_episode_deadline_rejects_new_calls_without_reaching_the_gateway():
    clock = [0.0]
    session = FakeSession([])
    environment = GatewayToolEnvironment(
        config(max_episode_seconds=5),
        session=session,
        monotonic=lambda: clock[0],
    )
    environment.reset()
    clock[0] = 5.1

    result = json.loads(
        call_tool(environment, "lookup_item", {"item_id": "42"})
    )

    assert result["status"] == "episode_timeout"
    assert session.calls == []
    assert environment.trajectory["terminationReason"] == "episode_timeout"


def test_gateway_timeout_is_capped_by_remaining_episode_budget():
    clock = [0.0]
    session = FakeSession([FakeResponse({"content": "ok"})])
    environment = GatewayToolEnvironment(
        config(tool_timeout_seconds=10, max_episode_seconds=5),
        session=session,
        monotonic=lambda: clock[0],
    )
    environment.reset()
    clock[0] = 2.0

    call_tool(environment, "lookup_item", {"item_id": "42"})

    assert session.calls[0][1]["timeout"] == 3.0


def test_denied_and_oversized_results_become_bounded_observations():
    session = FakeSession([FakeResponse({"content": "x" * 2048})])
    environment = GatewayToolEnvironment(config(), session=session)
    environment.reset()

    denied = json.loads(call_tool(environment, "delete_item", {}))
    assert denied["error"] == "tool_not_allowed"

    oversized = json.loads(call_tool(environment, "lookup_item", {"item_id": "42"}))
    assert oversized["error"] == "tool_response_too_large"
    assert environment.trajectory["toolCalls"][-1]["status"] == "response_too_large"


def test_retries_429_and_5xx_with_the_same_idempotency_key():
    responses = [
        FakeResponse({"error": "busy"}, status_code=429),
        FakeResponse({"error": "unavailable"}, status_code=503),
        FakeResponse({"content": "ok"}),
    ]
    session = FakeSession(responses)
    environment = GatewayToolEnvironment(
        config(max_retries=2),
        session=session,
        uuid_factory=iter(["rollout-1", "call-1"]).__next__,
        sleep=lambda _: None,
    )
    environment.reset()

    assert call_tool(environment, "lookup_item", {"item_id": "42"}) == '"ok"'
    assert len(session.calls) == 3
    assert {call[1]["headers"]["Idempotency-Key"] for call in session.calls} == {
        "call-1"
    }
    assert all(response.closed for response in responses)


@pytest.mark.parametrize(
    ("outcome", "expected_status"),
    [
        (requests.Timeout("slow"), "timeout"),
        (FakeResponse({"error": "forbidden"}, status_code=403), "denied"),
        (FakeResponse({"error": "quota"}, status_code=429), "quota_exhausted"),
    ],
)
def test_terminal_gateway_failures_have_stable_statuses(outcome, expected_status):
    environment = GatewayToolEnvironment(
        config(max_retries=0),
        session=FakeSession([outcome]),
    )
    environment.reset()

    result = json.loads(call_tool(environment, "lookup_item", {"item_id": "42"}))

    assert result["status"] == expected_status
    assert environment.trajectory["terminationReason"] == expected_status


@pytest.mark.parametrize(
    ("response", "expected_status"),
    [
        (
            FakeResponse(
                None,
                raw_content=b"{not-json",
            ),
            "gateway_error",
        ),
        (FakeResponse({"unexpected": "shape"}), "gateway_error"),
    ],
)
def test_malformed_gateway_results_become_bounded_observations(
    response,
    expected_status,
):
    environment = GatewayToolEnvironment(
        config(max_retries=0),
        session=FakeSession([response]),
    )
    environment.reset()

    result = json.loads(call_tool(environment, "lookup_item", {"item_id": "42"}))

    assert result["status"] == expected_status
    assert environment.trajectory["toolCalls"][-1]["result"] is None


def test_logs_never_include_arguments_results_or_gateway_token(caplog):
    environment = GatewayToolEnvironment(
        config(),
        session=FakeSession(
            [FakeResponse({"content": {"private_result": "sensitive-result"}})]
        ),
        token="sensitive-token",
    )
    environment.reset()

    with caplog.at_level(logging.INFO):
        call_tool(
            environment,
            "lookup_item",
            {"item_id": "sensitive-argument"},
        )

    assert "sensitive-argument" not in caplog.text
    assert "sensitive-result" not in caplog.text
    assert "sensitive-token" not in caplog.text
