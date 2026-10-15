"""Managed remote tool environment used by TRL's agentic GRPO loop."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Callable, Dict

import jsonschema
import requests

from core.config import ToolEnvironmentConfig
from utils.tool_conversations import ManagedToolCatalog

_DEFAULT_SERVICE_ACCOUNT_TOKEN_FILE = (
    "/var/run/secrets/kubernetes.io/serviceaccount/token"
)
_MAX_IDENTITY_TOKEN_BYTES = 65536


def _gateway_concurrency() -> int:
    try:
        configured = int(os.environ.get("FAI_RL_TOOL_MAX_CONCURRENCY", "32"))
    except ValueError:
        configured = 32
    return max(1, min(configured, 256))


_GATEWAY_EXECUTOR = ThreadPoolExecutor(
    max_workers=_gateway_concurrency(),
    thread_name_prefix="fai-rl-tool",
)


class _ToolResponseTooLarge(Exception):
    """Internal signal raised before an oversized response is materialized."""


def tool_catalog_prompt(config: ToolEnvironmentConfig) -> str:
    """Render the immutable public tool schemas without gateway credentials."""
    catalog = [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.input_schema,
        }
        for tool in config.tools
    ]
    return (
        "You may call only the managed tools in this JSON catalog. Use the "
        "call_tool function with tool_name set to a catalog name and arguments "
        "matching that tool's input_schema. Do not invent tool names.\n"
        + json.dumps(catalog, ensure_ascii=False, separators=(",", ":"))
    )


def managed_tool_catalog(config: ToolEnvironmentConfig) -> ManagedToolCatalog:
    """The tool view a GRPO rollout gives the model, for SFT to train on.

    Tool schemas come from the environment's public methods, collected the way
    TRL's GRPOTrainer collects them and rendered by the chat template the same
    way, so SFT and GRPO show the model identical tool definitions.
    """
    from transformers.utils import get_json_schema

    environment = GatewayToolEnvironment(config)
    tools = [
        get_json_schema(member)
        for name, member in inspect.getmembers(environment, predicate=inspect.ismethod)
        if name != "reset" and not name.startswith("_")
    ]
    return ManagedToolCatalog(
        instructions=tool_catalog_prompt(config),
        tools=tools,
        input_schemas={tool.name: tool.input_schema for tool in config.tools},
    )


class GatewayToolEnvironment:
    """One isolated managed-gateway environment per GRPO rollout."""

    def __init__(
        self,
        config: ToolEnvironmentConfig,
        *,
        session=None,
        token: str | None = None,
        token_file: str | None = None,
        uuid_factory: Callable[[], str] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        logger=None,
    ):
        self.config = config
        self.session = session or requests.Session()
        self.token = token
        self.token_file = (
            token_file
            if token_file is not None
            else os.environ.get(
                "FAI_RL_TOOL_GATEWAY_TOKEN_FILE",
                _DEFAULT_SERVICE_ACCOUNT_TOKEN_FILE,
            )
        )
        self.uuid_factory = uuid_factory or (lambda: str(uuid.uuid4()))
        self.sleep = sleep
        self.monotonic = monotonic
        self.logger = logger or logging.getLogger(__name__)
        self.allowed_tools = {tool.name: tool for tool in config.tools}
        self._state_lock = threading.Lock()
        self._session_lock = threading.Lock()
        self.rollout_id = ""
        self.tool_calls = []
        self.tool_call_count = 0
        self.termination_reason = "final_answer"
        self._episode_started = self.monotonic()

    def reset(self, **_context):
        """Reset all mutable episode state before TRL starts a rollout."""
        self.rollout_id = self.uuid_factory()
        self.tool_calls = []
        self.tool_call_count = 0
        self.termination_reason = "final_answer"
        self._episode_started = self.monotonic()
        return None

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Call one allowlisted managed remote tool.

        Args:
            tool_name: Name of a tool selected for this training job.
            arguments: JSON object matching that tool's registered input schema.

        Returns:
            The tool's JSON result or a bounded JSON error observation.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _GATEWAY_EXECUTOR,
            partial(self._call_tool_sync, tool_name, arguments),
        )

    def _call_tool_sync(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        with self._state_lock:
            if self.tool_call_count >= self.config.max_tool_calls:
                self.termination_reason = "tool_call_limit"
                return json.dumps(
                    {
                        "error": "tool_call_limit_exceeded",
                        "status": "limit_exceeded",
                    },
                    separators=(",", ":"),
                )
            self.tool_call_count += 1
            record_index = len(self.tool_calls)
            # Reserve model-call order before concurrent requests complete.
            self.tool_calls.append(None)
            call_id = self.uuid_factory()

        if self._remaining_episode_seconds() <= 0:
            self.termination_reason = "episode_timeout"
            return self._failure_observation(
                "tool_episode_timeout",
                tool_name,
                arguments if isinstance(arguments, dict) else {},
                status="episode_timeout",
                call_id=call_id,
                record_index=record_index,
            )
        if tool_name not in self.allowed_tools:
            return self._failure_observation(
                "tool_not_allowed",
                tool_name,
                arguments,
                status="denied",
                call_id=call_id,
                record_index=record_index,
            )
        if not isinstance(arguments, dict):
            return self._failure_observation(
                "tool_arguments_must_be_object",
                tool_name,
                {},
                status="invalid_arguments",
                call_id=call_id,
                record_index=record_index,
            )
        try:
            jsonschema.validate(
                instance=arguments,
                schema=self.allowed_tools[tool_name].input_schema,
            )
        except (
            jsonschema.SchemaError,
            jsonschema.ValidationError,
        ):
            return self._failure_observation(
                "tool_arguments_failed_schema_validation",
                tool_name,
                arguments,
                status="invalid_arguments",
                call_id=call_id,
                record_index=record_index,
            )
        if self.token is None and not self._gateway_is_allowlisted():
            self.termination_reason = "gateway_not_allowed"
            return self._failure_observation(
                "tool_gateway_not_allowed",
                tool_name,
                arguments,
                status="gateway_not_allowed",
                call_id=call_id,
                record_index=record_index,
            )
        try:
            authorization_token = self._authorization_token()
        except ValueError:
            authorization_token = None
        if not authorization_token:
            self.termination_reason = "identity_unavailable"
            return self._failure_observation(
                "tool_identity_unavailable",
                tool_name,
                arguments,
                status="identity_unavailable",
                call_id=call_id,
                record_index=record_index,
            )

        payload = {
            "version": self.config.version,
            "rolloutId": self.rollout_id,
            "callId": call_id,
            "serverId": self.config.server_id,
            "tool": tool_name,
            "arguments": arguments,
        }
        request_bytes = len(
            json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        if request_bytes > self.config.max_request_bytes:
            self.termination_reason = "request_too_large"
            return self._failure_observation(
                "tool_request_too_large",
                tool_name,
                arguments,
                status="request_too_large",
                call_id=call_id,
                record_index=record_index,
            )
        headers = {
            "Content-Type": "application/json",
            "Idempotency-Key": call_id,
            "X-FAI-RL-Rollout-ID": self.rollout_id,
        }
        headers["Authorization"] = f"Bearer {authorization_token}"

        started = self.monotonic()
        attempts = self.config.max_retries + 1
        last_error: Exception | None = None
        for attempt in range(attempts):
            response = None
            try:
                remaining_seconds = self._remaining_episode_seconds()
                if remaining_seconds <= 0:
                    self.termination_reason = "episode_timeout"
                    return self._failure_observation(
                        "tool_episode_timeout",
                        tool_name,
                        arguments,
                        status="episode_timeout",
                        call_id=call_id,
                        record_index=record_index,
                        latency_ms=self._elapsed_ms(started),
                    )
                with self._session_lock:
                    response = self.session.post(
                        f"{self.config.gateway_url}/tool-calls",
                        headers=headers,
                        json=payload,
                        timeout=min(
                            self.config.tool_timeout_seconds,
                            remaining_seconds,
                        ),
                        stream=True,
                        allow_redirects=False,
                    )
                if (
                    response.status_code == 429 or response.status_code >= 500
                ) and attempt < self.config.max_retries:
                    raise requests.HTTPError(
                        f"retryable gateway status {response.status_code}",
                        response=response,
                    )
                response.raise_for_status()
                body, response_bytes = self._read_bounded_json(response)
                if not isinstance(body, dict) or "content" not in body:
                    raise ValueError(
                        "gateway response must be an object with a content field"
                    )
                result = body["content"]
                latency_ms = self._elapsed_ms(started)
                self._record_tool_call(
                    record_index,
                    {
                        "callId": call_id,
                        "serverId": self.config.server_id,
                        "tool": tool_name,
                        "arguments": arguments,
                        "result": result,
                        "status": "ok",
                        "latencyMs": latency_ms,
                    },
                )
                self.logger.info(
                    "Tool call completed rollout_id=%s call_id=%s server_id=%s "
                    "tool=%s status=ok latency_ms=%d response_bytes=%d",
                    self.rollout_id,
                    call_id,
                    self.config.server_id,
                    tool_name,
                    latency_ms,
                    response_bytes,
                )
                return json.dumps(result, ensure_ascii=False, separators=(",", ":"))
            except _ToolResponseTooLarge:
                self.termination_reason = "tool_response_too_large"
                return self._failure_observation(
                    "tool_response_too_large",
                    tool_name,
                    arguments,
                    status="response_too_large",
                    call_id=call_id,
                    record_index=record_index,
                    latency_ms=self._elapsed_ms(started),
                )
            except (
                requests.RequestException,
                TypeError,
                ValueError,
            ) as exc:
                last_error = exc
                if attempt >= self.config.max_retries or not self._retryable(exc):
                    break
                self.sleep(self.config.retry_backoff_seconds * (2**attempt))
            finally:
                if response is not None:
                    close = getattr(response, "close", None)
                    if close is not None:
                        close()

        status = self._failure_status(last_error)
        self.termination_reason = status
        return self._failure_observation(
            f"tool_{status}",
            tool_name,
            arguments,
            status=status,
            call_id=call_id,
            record_index=record_index,
            latency_ms=self._elapsed_ms(started),
        )

    @property
    def trajectory(self) -> Dict[str, Any]:
        """Return the trajectory without exposing another callable tool to TRL."""
        with self._state_lock:
            tool_calls = [
                dict(tool_call)
                for tool_call in self.tool_calls
                if tool_call is not None
            ]
        return {
            "rolloutId": self.rollout_id,
            "terminationReason": self.termination_reason,
            "toolCalls": tool_calls,
        }

    def _finalize(self, completion: Any) -> None:
        """Record whether TRL stopped on a final answer or its turn bound."""
        last_message = completion[-1] if isinstance(completion, list) and completion else None
        with self._state_lock:
            if self.termination_reason != "final_answer":
                return
            self.termination_reason = (
                "turn_limit"
                if isinstance(last_message, dict)
                and bool(last_message.get("tool_calls"))
                else "final_answer"
            )

    def _read_bounded_json(self, response) -> tuple[Any, int]:
        content_length = response.headers.get("Content-Length")
        if content_length is not None:
            try:
                if int(content_length) > self.config.max_response_bytes:
                    raise _ToolResponseTooLarge
            except ValueError:
                # Ignore an invalid advisory header and enforce the streamed
                # byte count below.
                pass

        chunks = []
        response_bytes = 0
        for chunk in response.iter_content(
            chunk_size=min(65536, self.config.max_response_bytes + 1)
        ):
            if not chunk:
                continue
            response_bytes += len(chunk)
            if response_bytes > self.config.max_response_bytes:
                raise _ToolResponseTooLarge
            chunks.append(chunk)

        return json.loads(b"".join(chunks).decode("utf-8")), response_bytes

    def _authorization_token(self) -> str | None:
        # An explicit token is useful for local tests and gateways that mint a
        # short-lived job token. Production Kubernetes jobs normally use their
        # rotating, pod-bound service-account token from the projected file.
        if self.token is not None:
            return self.token or None
        try:
            with open(self.token_file) as token_file:
                token = token_file.read(_MAX_IDENTITY_TOKEN_BYTES + 1).strip()
        except OSError:
            token = ""
        if len(token.encode("utf-8")) > _MAX_IDENTITY_TOKEN_BYTES:
            raise ValueError("tool gateway identity token exceeds the safe limit")
        return token or None

    def _gateway_is_allowlisted(self) -> bool:
        allowed_urls = {
            value.strip().rstrip("/")
            for value in os.environ.get(
                "FAI_RL_TOOL_GATEWAY_ALLOWED_URLS",
                "",
            ).split(",")
            if value.strip()
        }
        return self.config.gateway_url in allowed_urls

    def _remaining_episode_seconds(self) -> float:
        return self.config.max_episode_seconds - (
            self.monotonic() - self._episode_started
        )

    def _elapsed_ms(self, started: float) -> int:
        return max(0, round((self.monotonic() - started) * 1000))

    @staticmethod
    def _retryable(error: Exception) -> bool:
        if isinstance(error, (requests.Timeout, requests.ConnectionError)):
            return True
        if isinstance(error, requests.HTTPError) and error.response is not None:
            return (
                error.response.status_code == 429 or error.response.status_code >= 500
            )
        return False

    @staticmethod
    def _failure_status(error: Exception | None) -> str:
        if isinstance(error, requests.Timeout):
            return "timeout"
        if isinstance(error, requests.HTTPError) and error.response is not None:
            if error.response.status_code in {401, 403}:
                return "denied"
            if error.response.status_code == 429:
                return "quota_exhausted"
        return "gateway_error"

    def _failure_observation(
        self,
        error: str,
        tool_name: str,
        arguments: Dict[str, Any],
        *,
        status: str,
        call_id: str | None = None,
        record_index: int,
        latency_ms: int = 0,
    ) -> str:
        call_id = call_id or self.uuid_factory()
        self._record_tool_call(
            record_index,
            {
                "callId": call_id,
                "serverId": self.config.server_id,
                "tool": tool_name,
                "arguments": arguments,
                "result": None,
                "status": status,
                "latencyMs": latency_ms,
            },
        )
        self.logger.warning(
            "Tool call failed rollout_id=%s call_id=%s server_id=%s tool=%s "
            "status=%s latency_ms=%d",
            self.rollout_id,
            call_id,
            self.config.server_id,
            tool_name,
            status,
            latency_ms,
        )
        return json.dumps(
            {"error": error, "status": status},
            separators=(",", ":"),
        )

    def _record_tool_call(
        self,
        record_index: int,
        record: Dict[str, Any],
    ) -> None:
        with self._state_lock:
            self.tool_calls[record_index] = record
