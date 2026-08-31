"""HTTP-backed reward function for GRPO and GSPO."""

import logging
import math
import time
from typing import Any, Dict, List

import requests

from core.config import RewardAPIConfig


def _json_safe(value: Any) -> Any:
    """Convert common tensor/array containers into JSON-compatible values."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if hasattr(value, "item"):
        return _json_safe(value.item())
    raise TypeError(
        f"Reward API context contains a non-JSON-serializable value: {type(value).__name__}"
    )


class APIRewardFunction:
    """TRL-compatible callable that delegates reward calculation to an HTTP API."""

    def __init__(self, config: RewardAPIConfig, logger=None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json", **self.config.headers}
        api_key = self.config.api_key
        if api_key:
            value = f"{self.config.auth_scheme} {api_key}".strip()
            headers[self.config.auth_header] = value
        return headers

    def __call__(self, prompts, completions, **kwargs) -> List[float]:
        context = {key: value for key, value in kwargs.items() if key != "logger"}
        payload = {
            **_json_safe(self.config.extra_body),
            "prompts": _json_safe(prompts),
            "completions": _json_safe(completions),
            "context": _json_safe(context),
        }

        last_error = None
        attempts = self.config.max_retries + 1
        started_at = time.monotonic()
        for attempt in range(attempts):
            try:
                self.logger.debug(
                    "Calling reward API endpoint=%s completions=%d attempt=%d/%d",
                    self.config.endpoint,
                    len(completions),
                    attempt + 1,
                    attempts,
                )
                response = requests.post(
                    self.config.endpoint,
                    headers=self._headers(),
                    json=payload,
                    timeout=self.config.timeout_seconds,
                    verify=self.config.verify_ssl,
                )
                response.raise_for_status()
                body = response.json()
                rewards = body[self.config.response_field]
                scores = self._validate_rewards(rewards, len(completions))
                elapsed_seconds = time.monotonic() - started_at
                if scores:
                    self.logger.info(
                        "Reward API scored %d completions in %.3fs "
                        "(attempt=%d/%d min=%.4f max=%.4f mean=%.4f)",
                        len(scores),
                        elapsed_seconds,
                        attempt + 1,
                        attempts,
                        min(scores),
                        max(scores),
                        sum(scores) / len(scores),
                    )
                else:
                    self.logger.info(
                        "Reward API scored an empty batch in %.3fs (attempt=%d/%d)",
                        elapsed_seconds,
                        attempt + 1,
                        attempts,
                    )
                return scores
            except (KeyError, TypeError, ValueError, requests.RequestException) as exc:
                last_error = exc
                if attempt == self.config.max_retries:
                    break
                delay = self.config.retry_backoff_seconds * (2**attempt)
                self.logger.warning(
                    "Reward API request failed endpoint=%s attempt=%d/%d (%s); "
                    "retrying in %.1fs",
                    self.config.endpoint,
                    attempt + 1,
                    attempts,
                    exc,
                    delay,
                )
                time.sleep(delay)

        self.logger.error(
            "Reward API failed endpoint=%s attempts=%d elapsed_seconds=%.3f error=%s",
            self.config.endpoint,
            attempts,
            time.monotonic() - started_at,
            last_error,
        )
        raise RuntimeError(
            f"Reward API failed after {attempts} attempt(s): {last_error}"
        ) from last_error

    def _validate_rewards(self, rewards: Any, expected_count: int) -> List[float]:
        if not isinstance(rewards, list):
            raise TypeError(
                f"Reward API field '{self.config.response_field}' must be a list"
            )
        if len(rewards) != expected_count:
            raise ValueError(
                f"Reward API returned {len(rewards)} rewards for "
                f"{expected_count} completions"
            )

        scores = []
        for reward in rewards:
            if isinstance(reward, bool):
                raise TypeError("Reward API scores must be numbers, not booleans")
            score = float(reward)
            if not math.isfinite(score):
                raise ValueError("Reward API scores must be finite")
            scores.append(score)
        return scores
