"""Tests for the HTTP-backed GRPO/GSPO reward function."""

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
    def __init__(self, body):
        self.body = body

    def raise_for_status(self):
        return None

    def json(self):
        return self.body


def test_api_reward_sends_context_without_requiring_auth(monkeypatch):
    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return _Response({"rewards": [1, 0.25]})

    monkeypatch.setattr(requests, "post", fake_post)
    reward = APIRewardFunction(
        RewardAPIConfig(endpoint="https://reward.example/score")
    )
    assert reward.__name__ == "api_reward"

    scores = reward(
        ["prompt-a", "prompt-b"],
        ["completion-a", "completion-b"],
        answer=["a", "b"],
    )

    assert scores == [1.0, 0.25]
    assert captured["url"] == "https://reward.example/score"
    assert "Authorization" not in captured["headers"]
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

    def fake_post(*_args, **_kwargs):
        calls.append(None)
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


def test_api_reward_logs_safe_batch_summary(monkeypatch, caplog):
    monkeypatch.setattr(
        requests,
        "post",
        lambda *_args, **_kwargs: _Response({"rewards": [0.25, 0.75]}),
    )
    reward = APIRewardFunction(
        RewardAPIConfig(endpoint="https://reward.example/score")
    )

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
