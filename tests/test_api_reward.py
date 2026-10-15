"""Tests for the HTTP-backed GRPO/GSPO reward function."""

import json
import logging
import sys
from pathlib import Path

import pytest
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.config import RewardAPIConfig
from trainers.rewards.api_reward import APIRewardFunction


class _Response:
    def __init__(self, body, status_code=200):
        self.body = body
        self.content = json.dumps(body).encode()
        self.headers = {"Content-Length": str(len(self.content))}
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(
                f"status {self.status_code}",
                response=self,
            )

    def iter_content(self, chunk_size):
        for offset in range(0, len(self.content), chunk_size):
            yield self.content[offset : offset + chunk_size]

    def close(self):
        return None


def test_api_reward_sends_context_without_requiring_auth(monkeypatch):
    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return _Response({"rewards": [1, 0.25]})

    monkeypatch.setattr(requests, "post", fake_post)
    reward = APIRewardFunction(RewardAPIConfig(endpoint="https://reward.example/score"))
    assert reward.__name__ == "api_reward"

    scores = reward(
        ["prompt-a", "prompt-b"],
        ["completion-a", "completion-b"],
        answer=["a", "b"],
    )

    assert scores == [1.0, 0.25]
    assert captured["url"] == "https://reward.example/score"
    assert "Authorization" not in captured["headers"]
    assert captured["stream"] is True
    assert captured["allow_redirects"] is False
    assert captured["json"]["context"] == {"answer": ["a", "b"]}


def test_api_reward_rejects_wrong_score_count(monkeypatch):
    monkeypatch.setattr(
        requests,
        "post",
        lambda *_args, **_kwargs: _Response({"rewards": [1.0]}),
    )
    reward = APIRewardFunction(
        RewardAPIConfig(
            endpoint="https://reward.example/score",
            max_retries=0,
        )
    )

    with pytest.raises(RuntimeError, match="1 rewards for 2 completions"):
        reward(["a", "b"], ["x", "y"])


def test_api_reward_retries_transient_failure(monkeypatch):
    calls = []

    def fake_post(*_args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise requests.ConnectionError("temporary")
        return _Response({"rewards": [0.5]})

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr("trainers.rewards.api_reward.time.sleep", lambda _delay: None)
    reward = APIRewardFunction(
        RewardAPIConfig(
            endpoint="https://reward.example/score",
            max_retries=1,
        )
    )

    assert reward(["a"], ["x"]) == [0.5]
    assert len(calls) == 2
    assert {
        call["headers"]["Idempotency-Key"]
        for call in calls
    } == {calls[0]["headers"]["Idempotency-Key"]}


def test_api_reward_does_not_retry_a_malformed_success_response(monkeypatch):
    calls = []

    def fake_post(*_args, **kwargs):
        calls.append(kwargs)
        return _Response({"unexpected": []})

    monkeypatch.setattr(requests, "post", fake_post)
    reward = APIRewardFunction(
        RewardAPIConfig(
            endpoint="https://reward.example/score",
            max_retries=3,
        )
    )

    with pytest.raises(RuntimeError, match="rewards"):
        reward(["a"], ["x"])

    assert len(calls) == 1


@pytest.mark.parametrize(
    ("first_status", "expected_calls"),
    [(400, 1), (429, 2), (503, 2)],
)
def test_api_reward_retries_only_selected_http_statuses(
    monkeypatch,
    first_status,
    expected_calls,
):
    calls = []

    def fake_post(*_args, **_kwargs):
        calls.append(None)
        if len(calls) == 1:
            return _Response({"error": "failed"}, status_code=first_status)
        return _Response({"rewards": [0.5]})

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(
        "trainers.rewards.api_reward.time.sleep",
        lambda _delay: None,
    )
    reward = APIRewardFunction(
        RewardAPIConfig(
            endpoint="https://reward.example/score",
            max_retries=1,
        )
    )

    if first_status == 400:
        with pytest.raises(RuntimeError, match="status 400"):
            reward(["a"], ["x"])
    else:
        assert reward(["a"], ["x"]) == [0.5]

    assert len(calls) == expected_calls


def test_api_reward_rejects_an_oversized_response_before_decoding(monkeypatch):
    monkeypatch.setattr(
        requests,
        "post",
        lambda *_args, **_kwargs: _Response({"rewards": ["x" * 100]}),
    )
    reward = APIRewardFunction(
        RewardAPIConfig(
            endpoint="https://reward.example/score",
            max_response_bytes=16,
            max_retries=0,
        )
    )

    with pytest.raises(RuntimeError, match="response exceeds"):
        reward(["a"], ["x"])


def test_api_reward_rejects_an_oversized_request_before_network(monkeypatch):
    calls = []
    monkeypatch.setattr(
        requests,
        "post",
        lambda *_args, **_kwargs: calls.append(None),
    )
    reward = APIRewardFunction(
        RewardAPIConfig(
            endpoint="https://reward.example/score",
            max_request_bytes=32,
            max_retries=0,
        )
    )

    with pytest.raises(ValueError, match="request exceeds"):
        reward(["a"], ["x" * 100])

    assert calls == []


def test_api_reward_logs_safe_batch_summary(monkeypatch, caplog):
    monkeypatch.setattr(
        requests,
        "post",
        lambda *_args, **_kwargs: _Response({"rewards": [0.25, 0.75]}),
    )
    reward = APIRewardFunction(RewardAPIConfig(endpoint="https://reward.example/score"))

    with caplog.at_level(logging.INFO, logger="trainers.rewards.api_reward"):
        reward(["sensitive-prompt"], ["sensitive-completion", "other"])

    assert "Reward API scored 2 completions" in caplog.text
    assert "mean=0.5000" in caplog.text
    assert "sensitive-prompt" not in caplog.text
    assert "sensitive-completion" not in caplog.text


def test_reward_api_redacts_configured_headers():
    config = RewardAPIConfig(
        endpoint="https://reward.example/score",
        headers={"X-Tenant": "customer-care"},
    )

    serialized = config.to_dict()
    assert serialized["headers"] == {"X-Tenant": "***"}
    assert "api_key" not in serialized
    assert "auth_header" not in serialized
    assert "auth_scheme" not in serialized


def test_reward_api_requires_https_for_headers():
    with pytest.raises(ValueError, match="must use https"):
        RewardAPIConfig(
            endpoint="http://reward.example/score",
            headers={"X-Tenant": "customer-care"},
        )


def test_v2_reward_includes_trajectories_and_excludes_environments_from_context(
    monkeypatch,
):
    captured = {}

    def fake_post(_url, **kwargs):
        captured.update(kwargs)
        return _Response({"rewards": [0.75]})

    class Environment:
        @property
        def trajectory(self):
            return {
                "rolloutId": "rollout-1",
                "terminationReason": "final_answer",
                "toolCalls": [],
            }

    monkeypatch.setattr(requests, "post", fake_post)
    reward = APIRewardFunction(
        RewardAPIConfig(
            version="v2",
            endpoint="https://reward.example/trajectory",
        )
    )

    assert reward(
        ["prompt"],
        ["completion"],
        environments=[Environment()],
        expected=["answer"],
        completion_ids=[[1, 2, 3]],
        trainer_state=object(),
        log_extra=lambda *_args: None,
        log_metric=lambda *_args: None,
    ) == [0.75]
    assert captured["json"]["version"] == "v2"
    assert captured["json"]["trajectories"][0]["rolloutId"] == "rollout-1"
    assert captured["json"]["context"] == {"expected": ["answer"]}


def test_v2_reward_requires_one_environment_per_completion(monkeypatch):
    monkeypatch.setattr(
        requests,
        "post",
        lambda *_args, **_kwargs: _Response({"rewards": [1.0]}),
    )
    reward = APIRewardFunction(
        RewardAPIConfig(
            version="v2",
            endpoint="https://reward.example/trajectory",
            max_retries=0,
        )
    )

    with pytest.raises(ValueError, match="environment"):
        reward(["prompt"], ["completion"], environments=[])
