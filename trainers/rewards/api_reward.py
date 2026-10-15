"""HTTP-backed reward function for GRPO and GSPO."""

import json
import logging
import math
import time
import uuid
from typing import Any, Dict, List

import requests

from core.config import RewardAPIConfig


class _RewardResponseTooLarge(ValueError):
    """Raised before an oversized reward response is materialized."""


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
        self.__name__ = "api_reward"

    def _headers(self, batch_id: str) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Idempotency-Key": batch_id,
            "X-FAI-RL-Reward-Batch-ID": batch_id,
            **self.config.headers,
        }

    def __call__(self, prompts, completions, **kwargs) -> List[float]:
        environments = kwargs.get("environments")
        internal_keys = {
            "completion_ids",
            "environments",
            "log_extra",
            "log_metric",
            "logger",
            "trainer_state",
        }
        context = {
            key: value
            for key, value in kwargs.items()
            if key not in internal_keys
        }
        payload = {
            **_json_safe(self.config.extra_body),
            "prompts": _json_safe(prompts),
            "completions": _json_safe(completions),
            "context": _json_safe(context),
        }
        if self.config.version == "v2":
            if not isinstance(environments, list) or len(environments) != len(
                completions
            ):
                raise ValueError(
                    "reward API v2 requires one tool environment per completion"
                )
            for environment, completion in zip(
                environments,
                completions,
            ):
                finalize = getattr(environment, "_finalize", None)
                if finalize is not None:
                    finalize(completion)
            payload["version"] = "v2"
            payload["trajectories"] = _json_safe(
                [environment.trajectory for environment in environments]
            )

        request_bytes = len(
            json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        if request_bytes > self.config.max_request_bytes:
            raise ValueError("reward API request exceeds max_request_bytes")

        last_error = None
        attempts = self.config.max_retries + 1
        started_at = time.monotonic()
        batch_id = str(uuid.uuid4())
        attempts_made = 0
        for attempt in range(attempts):
            attempts_made = attempt + 1
            response = None
            try:
                self.logger.debug(
                    "Calling reward API completions=%d attempt=%d/%d",
                    len(completions),
                    attempt + 1,
                    attempts,
                )
                response = requests.post(
                    self.config.endpoint,
                    headers=self._headers(batch_id),
                    json=payload,
                    timeout=self.config.timeout_seconds,
                    verify=self.config.verify_ssl,
                    stream=True,
                    allow_redirects=False,
                )
                response.raise_for_status()
                body = self._read_bounded_json(response)
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
                if (
                    attempt == self.config.max_retries
                    or not self._retryable(exc)
                ):
                    break
                delay = self.config.retry_backoff_seconds * (2**attempt)
                self.logger.warning(
                    "Reward API request failed attempt=%d/%d (%s); "
                    "retrying in %.1fs",
                    attempt + 1,
                    attempts,
                    exc,
                    delay,
                )
                time.sleep(delay)
            finally:
                if response is not None:
                    close = getattr(response, "close", None)
                    if close is not None:
                        close()

        self.logger.error(
            "Reward API failed attempts=%d elapsed_seconds=%.3f error=%s",
            attempts_made,
            time.monotonic() - started_at,
            last_error,
        )
        raise RuntimeError(
            f"Reward API failed after {attempts_made} attempt(s): {last_error}"
        ) from last_error

    def _read_bounded_json(self, response) -> Any:
        content_length = response.headers.get("Content-Length")
        if content_length is not None:
            try:
                if int(content_length) > self.config.max_response_bytes:
                    raise _RewardResponseTooLarge(
                        "reward API response exceeds max_response_bytes"
                    )
            except ValueError as exc:
                if isinstance(exc, _RewardResponseTooLarge):
                    raise

        chunks = []
        response_bytes = 0
        for chunk in response.iter_content(
            chunk_size=min(65536, self.config.max_response_bytes + 1)
        ):
            if not chunk:
                continue
            response_bytes += len(chunk)
            if response_bytes > self.config.max_response_bytes:
                raise _RewardResponseTooLarge(
                    "reward API response exceeds max_response_bytes"
                )
            chunks.append(chunk)
        return json.loads(b"".join(chunks).decode("utf-8"))

    @staticmethod
    def _retryable(error: Exception) -> bool:
        if isinstance(error, (requests.Timeout, requests.ConnectionError)):
            return True
        if isinstance(error, requests.HTTPError) and error.response is not None:
            return (
                error.response.status_code == 429
                or error.response.status_code >= 500
            )
        return False

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
