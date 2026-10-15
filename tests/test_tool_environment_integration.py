"""Deterministic managed-gateway and trajectory-reward integration coverage."""

import asyncio
import json

import requests
from transformers.utils import get_json_schema

from core.config import (
    RewardAPIConfig,
    ToolDefinition,
    ToolEnvironmentConfig,
)
from trainers.rewards.api_reward import APIRewardFunction
from trainers.tool_environment import GatewayToolEnvironment


def test_call_tool_has_a_transformers_compatible_schema():
    environment = GatewayToolEnvironment(
        ToolEnvironmentConfig(
            gateway_url="https://tools.example.test/v1",
            server_id="deterministic",
            tools=[
                ToolDefinition(
                    name="calculate",
                    description="Add two integers.",
                    input_schema={"type": "object"},
                )
            ],
        ),
        token="pod-bound-token",
    )

    schema = get_json_schema(environment.call_tool)

    assert schema["function"]["name"] == "call_tool"
    assert schema["function"]["parameters"]["required"] == [
        "tool_name",
        "arguments",
    ]


class GatewayResponse:
    def __init__(self, content):
        self.status_code = 200
        self.content = json.dumps({"content": content}).encode()
        self.headers = {"Content-Length": str(len(self.content))}

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size):
        for offset in range(0, len(self.content), chunk_size):
            yield self.content[offset : offset + chunk_size]

    def close(self):
        return None


class DeterministicGateway:
    """Normalizes one declarative HTTP tool and one MCP-backed tool."""

    documents = {
        "red": "Red widgets are in aisle 3.",
        "blue": "Blue widgets are in aisle 7.",
    }

    def __init__(self):
        self.idempotent_results = {}

    def post(self, _url, *, headers, json, **_kwargs):
        call_id = headers["Idempotency-Key"]
        if call_id in self.idempotent_results:
            return GatewayResponse(self.idempotent_results[call_id])

        if json["tool"] == "calculate":
            arguments = json["arguments"]
            result = {
                "adapter": "http",
                "value": arguments["left"] + arguments["right"],
            }
        elif json["tool"] == "search":
            query = json["arguments"]["query"]
            result = {
                "adapter": "mcp",
                "matches": [
                    value
                    for key, value in self.documents.items()
                    if query.lower() in key
                ],
            }
        else:
            raise AssertionError(f"unexpected tool {json['tool']}")

        self.idempotent_results[call_id] = result
        return GatewayResponse(result)


class RewardResponse:
    def __init__(self):
        self.content = b'{"rewards":[1.0]}'
        self.headers = {"Content-Length": str(len(self.content))}
        self.status_code = 200

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size):
        for offset in range(0, len(self.content), chunk_size):
            yield self.content[offset : offset + chunk_size]

    def close(self):
        return None


def test_http_and_mcp_adapters_produce_one_rewarded_trajectory(monkeypatch):
    environment = GatewayToolEnvironment(
        ToolEnvironmentConfig(
            gateway_url="https://tools.example.test/v1",
            server_id="deterministic",
            tools=[
                ToolDefinition(
                    name="calculate",
                    description="Add two integers through an HTTP adapter.",
                    input_schema={"type": "object"},
                ),
                ToolDefinition(
                    name="search",
                    description="Search a fixed corpus through an MCP adapter.",
                    input_schema={"type": "object"},
                ),
            ],
            max_tool_calls=2,
        ),
        session=DeterministicGateway(),
        token="pod-bound-token",
        uuid_factory=iter(["rollout-1", "call-1", "call-2"]).__next__,
    )
    environment.reset()

    calculator_result = json.loads(
        asyncio.run(
            environment.call_tool(
                "calculate",
                {"left": 20, "right": 22},
            )
        )
    )
    search_result = json.loads(
        asyncio.run(
            environment.call_tool(
                "search",
                {"query": "blue"},
            )
        )
    )

    captured_reward_request = {}

    def score_trajectory(_url, **kwargs):
        captured_reward_request.update(kwargs["json"])
        return RewardResponse()

    monkeypatch.setattr(requests, "post", score_trajectory)
    reward = APIRewardFunction(
        RewardAPIConfig(
            version="v2",
            endpoint="https://reward.example.test/v2",
            max_retries=0,
        )
    )
    scores = reward(
        prompts=["Add 20 and 22, then find the blue widget."],
        completions=["42; the blue widget is in aisle 7."],
        environments=[environment],
    )

    assert calculator_result == {"adapter": "http", "value": 42}
    assert search_result == {
        "adapter": "mcp",
        "matches": ["Blue widgets are in aisle 7."],
    }
    assert scores == [1.0]
    assert captured_reward_request["version"] == "v2"
    assert captured_reward_request["trajectories"] == [environment.trajectory]
    assert [
        call["callId"]
        for call in captured_reward_request["trajectories"][0]["toolCalls"]
    ] == ["call-1", "call-2"]
