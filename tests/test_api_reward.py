"""Tests for the HTTP-backed GRPO/GSPO reward function."""

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


def test_reward_api_redacts_inline_key():
    config = RewardAPIConfig(
        endpoint="https://reward.example/score",
        api_key="secret",
    )

    assert config.to_dict()["api_key"] == "***"
