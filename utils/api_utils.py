"""API utility functions for making HTTP requests."""
import email.utils
import json
import logging
import os
import random
import re
import threading
import time
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Union
from urllib.parse import urlparse

import requests

# Set up logger for API utilities
logger = logging.getLogger(__name__)

# Import hosted LLM custom configurations
try:
    from utils import hosted_llm_config
except ImportError:
    hosted_llm_config = None
    logger.warning("Could not import hosted_llm_config module")

_DEFAULT_AUTH = ("Authorization", "Bearer ")
_MAX_BACKOFF_SECONDS = 60.0
_MAX_RETRY_AFTER_SECONDS = 600.0
_ERROR_BODY_CHARS = 500
# Request errors that come from the configuration, not the network.
_NON_RETRYABLE_REQUEST_ERRORS = (
    requests.exceptions.InvalidURL,
    requests.exceptions.MissingSchema,
    requests.exceptions.InvalidSchema,
    requests.exceptions.InvalidHeader,
    requests.exceptions.URLRequired,
)
_URL_SECRET_PATTERN = re.compile(r"([?&](?:key|api_key|apikey|token|access_token)=)[^&\s'\"]+")

# Indirection so tests can observe retry delays without sleeping.
_sleep = time.sleep


# ============================================================================
# Helper Functions for API Calls
# ============================================================================

def _extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    Extract JSON from text that may contain markdown formatting or other content.
    
    This function handles cases where the LLM returns formatted text with JSON embedded,
    such as in markdown code blocks or at the end of explanatory text.
    
    Args:
        text: The response text that contains JSON
        
    Returns:
        Parsed JSON as a dictionary
        
    Raises:
        json.JSONDecodeError: If no valid JSON can be extracted
    """
    # First, try to parse the entire text as JSON (backwards compatibility)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try to extract JSON from markdown code blocks (```json ... ``` or ``` ... ```)
    # Look for content within triple backticks
    code_block_patterns = [
        r'```json\s*\n(.*?)\n```',  # ```json ... ```
        r'```\s*\n(.*?)\n```',       # ``` ... ```
    ]
    
    for pattern in code_block_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue
    
    # Try to find JSON-like patterns (objects starting with { and ending with })
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_pattern, text)
    
    # Try matches from last to first (JSON is often at the end)
    for match in reversed(matches):
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue
    
    # If nothing worked, raise an error with the original text
    raise json.JSONDecodeError(f"Could not extract valid JSON from text: {text[:200]}...", text, 0)


def sampling_params(config) -> Dict[str, Any]:
    """temperature / top_p from config, omitting unset (None) values."""
    params = {}
    for name in ("temperature", "top_p"):
        value = getattr(config, name, None)
        if value is not None:
            params[name] = value
    return params


def chat_messages(prompt: str, system_prompt: str = None) -> List[Dict[str, str]]:
    """OpenAI-format system (optional) + user turns."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def openai_compatible_request(prompt: str, config, system_prompt: str = None) -> dict:
    """OpenAI chat-completions body honoring api_max_tokens_field and None omission."""
    data = {"model": config.model, "messages": chat_messages(prompt, system_prompt)}
    max_tokens_field = getattr(config, "api_max_tokens_field", "max_tokens") or "max_tokens"
    data[max_tokens_field] = getattr(config, "max_new_tokens", 1000)
    data.update(sampling_params(config))
    return data


def _build_google_request_data(prompt: str, config, system_prompt: str = None) -> dict:
    """Build request data for Google/Gemini models."""
    generation_config = {
        "maxOutputTokens": getattr(config, 'max_new_tokens', 1000)
    }

    params = sampling_params(config)
    if "temperature" in params:
        generation_config["temperature"] = params["temperature"]
    if "top_p" in params:
        generation_config["topP"] = params["top_p"]

    data = {
        "contents": [{
            "role": "user",
            "parts": [{"text": prompt}]
        }],
        "generationConfig": generation_config
    }
    # Gemini keeps the system message separate from the user turn.
    if system_prompt:
        data["system_instruction"] = {"parts": [{"text": system_prompt}]}
    return data


def _build_openai_request_data(prompt: str, config, system_prompt: str = None) -> dict:
    """Build request data for OpenAI/ChatGPT models."""
    return openai_compatible_request(prompt, config, system_prompt)


def _build_anthropic_request_data(prompt: str, config, system_prompt: str = None) -> dict:
    """Build request data for Anthropic/Claude models."""
    data = {
        "model": config.model,
        "max_tokens": getattr(config, 'max_new_tokens', 1000),
        "messages": [{"content": prompt, "role": "user"}]
    }
    # Anthropic takes the system message as a top-level field, not a message.
    if system_prompt:
        data["system"] = system_prompt
    data.update(sampling_params(config))
    return data


def _build_default_request_data(prompt: str, config, system_prompt: str = None) -> dict:
    """
    Build request data for other models (generic format).

    Default format is OpenAI-compatible.
    Customize in utils/hosted_llm_config.py for your hosted LLM.
    """
    # Check if custom hosted LLM configuration exists
    if hosted_llm_config:
        custom_data = hosted_llm_config.build_hosted_llm_request(prompt, config, system_prompt)
        if custom_data is not None:
            logger.debug("Using custom hosted LLM request format")
            return custom_data

    # Default: OpenAI-compatible format
    return openai_compatible_request(prompt, config, system_prompt)


def _get_model_provider(api_endpoint: str) -> str:
    """
    Determine the provider from the API endpoint URL.
    
    Args:
        api_endpoint: The API endpoint URL
        
    Returns:
        Provider name: "google", "openai", "anthropic", or "default"
    """
    # Parse the URL and operate on the hostname to avoid substring matches
    parsed = urlparse(api_endpoint)
    hostname = (parsed.hostname or "").lower()

    # Fallback: if parsing did not yield a hostname (e.g., bare host string),
    # use the lowercased input as a heuristic while preserving compatibility.
    target = hostname or api_endpoint.lower()

    # Check for Google/Gemini endpoints by hostname
    if (
        target == "generativelanguage.googleapis.com"
        or target.endswith(".generativelanguage.googleapis.com")
        or "gemini" in target
    ):
        return "google"

    # Check for OpenAI endpoints by hostname
    if (
        target == "api.openai.com"
        or target.endswith(".api.openai.com")
        or "openai" in target
    ):
        return "openai"

    # Check for Anthropic endpoints by hostname
    if (
        target == "api.anthropic.com"
        or target.endswith(".api.anthropic.com")
        or "anthropic" in target
    ):
        return "anthropic"

    # Default for custom/other providers
    return "default"


def _build_request_data(provider: str, prompt: str, config, system_prompt: str = None) -> dict:
    if provider == "google":
        data = _build_google_request_data(prompt, config, system_prompt)
    elif provider == "openai":
        data = _build_openai_request_data(prompt, config, system_prompt)
    elif provider == "anthropic":
        data = _build_anthropic_request_data(prompt, config, system_prompt)
    else:
        data = _build_default_request_data(prompt, config, system_prompt)
    extra_body = getattr(config, "api_extra_body", None) or {}
    return {**data, **extra_body}


@dataclass
class ApiCompletion:
    """One parsed model response."""
    text: str
    reasoning: Optional[str] = None
    finish_reason: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None


class ApiRequestError(Exception):
    """A request that failed; ``retryable`` marks transient failures."""

    def __init__(self, message, status=None, retryable=False, retry_after=None):
        super().__init__(message)
        self.status = status
        self.retryable = retryable
        self.retry_after = retry_after

    @property
    def is_auth_failure(self) -> bool:
        return self.status in (401, 403)


def _int_or_none(value) -> Optional[int]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _openai_text(content) -> Optional[str]:
    if isinstance(content, list):
        return "".join(
            part.get("text", "") for part in content
            if isinstance(part, dict) and part.get("type", "text") == "text"
        )
    return content


def _parse_openai_completion(response_json: dict) -> ApiCompletion:
    choice = response_json["choices"][0]
    message = choice.get("message") or {}
    usage = response_json.get("usage") or {}
    reasoning = message.get("reasoning_content")
    return ApiCompletion(
        text=_openai_text(message.get("content")),
        reasoning=reasoning if isinstance(reasoning, str) and reasoning else None,
        finish_reason=choice.get("finish_reason"),
        prompt_tokens=_int_or_none(usage.get("prompt_tokens")),
        completion_tokens=_int_or_none(usage.get("completion_tokens")),
    )


def _parse_anthropic_completion(response_json: dict) -> ApiCompletion:
    blocks = response_json["content"]
    text = "".join(b.get("text", "") for b in blocks if b.get("type", "text") == "text")
    thinking = "".join(b.get("thinking", "") for b in blocks if b.get("type") == "thinking")
    usage = response_json.get("usage") or {}
    return ApiCompletion(
        text=text,
        reasoning=thinking or None,
        finish_reason=response_json.get("stop_reason"),
        prompt_tokens=_int_or_none(usage.get("input_tokens")),
        completion_tokens=_int_or_none(usage.get("output_tokens")),
    )


def _parse_google_completion(response_json: dict) -> ApiCompletion:
    candidate = response_json["candidates"][0]
    parts = (candidate.get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    thought = "".join(p.get("text", "") for p in parts if p.get("thought"))
    usage = response_json.get("usageMetadata") or {}
    return ApiCompletion(
        text=text,
        reasoning=thought or None,
        finish_reason=candidate.get("finishReason"),
        prompt_tokens=_int_or_none(usage.get("promptTokenCount")),
        completion_tokens=_int_or_none(usage.get("candidatesTokenCount")),
    )


def parse_api_completion(response_json: dict, api_endpoint: str) -> ApiCompletion:
    """Parse a provider response; raises ApiRequestError when it has no text."""
    provider = _get_model_provider(api_endpoint)
    try:
        if provider == "google":
            completion = _parse_google_completion(response_json)
        elif provider == "anthropic":
            completion = _parse_anthropic_completion(response_json)
        elif provider == "openai":
            completion = _parse_openai_completion(response_json)
        else:
            custom = None
            if hosted_llm_config:
                custom = hosted_llm_config.parse_hosted_llm_response(response_json)
            try:
                completion = _parse_openai_completion(response_json)
            except (KeyError, IndexError, TypeError, AttributeError):
                if custom is None:
                    raise
                completion = ApiCompletion(text=custom)
            if custom is not None:
                completion.text = custom
    except (KeyError, IndexError, TypeError, AttributeError) as e:
        keys = (
            list(response_json) if isinstance(response_json, dict)
            else type(response_json).__name__
        )
        raise ApiRequestError(
            f"unexpected response shape ({type(e).__name__}: {e}); top-level keys: {keys}"
        ) from e
    if not isinstance(completion.text, str) or not completion.text:
        raise ApiRequestError(
            f"empty response content (finish_reason={completion.finish_reason})"
        )
    return completion


def _parse_api_response(response_json: dict, api_endpoint: str) -> str:
    """Extract the response text from the API response JSON ("" when absent)."""
    try:
        return parse_api_completion(response_json, api_endpoint).text
    except ApiRequestError as e:
        logger.warning("Failed to parse API response: %s", e)
        return ""


# ============================================================================
# Public API Functions
# ============================================================================

def _build_headers(api_endpoint: str, api_key: str, config=None) -> dict:
    """Build appropriate headers for each API provider.

    A non-default api_auth_header / api_auth_prefix applies to every provider;
    otherwise each provider keeps its native auth header.
    """
    provider = _get_model_provider(api_endpoint)
    auth_header = getattr(config, "api_auth_header", _DEFAULT_AUTH[0]) or _DEFAULT_AUTH[0]
    auth_prefix = getattr(config, "api_auth_prefix", _DEFAULT_AUTH[1])
    if auth_prefix is None:
        auth_prefix = ""
    custom_auth = (auth_header, auth_prefix) != _DEFAULT_AUTH

    headers = {"Content-Type": "application/json"}
    if provider == "anthropic":
        headers["anthropic-version"] = "2023-06-01"

    if custom_auth or provider == "openai":
        headers[auth_header] = f"{auth_prefix}{api_key}"
    elif provider == "google":
        # Header auth keeps the key out of the URL (and out of request logs).
        headers["x-goog-api-key"] = api_key
    elif provider == "anthropic":
        headers["x-api-key"] = api_key
    else:
        custom_headers = None
        if hosted_llm_config:
            custom_headers = hosted_llm_config.build_hosted_llm_headers(api_key, config)
        if custom_headers is not None:
            logger.debug("Using custom hosted LLM headers")
            headers = dict(custom_headers)
        else:
            headers[auth_header] = f"{auth_prefix}{api_key}"

    headers.update(getattr(config, "api_extra_headers", None) or {})
    return headers


def _prepare_api_url(api_endpoint: str, api_key: str) -> str:
    """Prepare the API URL with necessary query parameters."""
    provider = _get_model_provider(api_endpoint)
    
    if provider == "google":
        # Google/Gemini authenticates with the x-goog-api-key header.
        return api_endpoint
    else:
        # Check if custom hosted LLM configuration exists
        if hosted_llm_config:
            custom_url = hosted_llm_config.prepare_hosted_llm_url(api_endpoint, api_key)
            if custom_url is not None:
                logger.debug("Using custom hosted LLM URL")
                return custom_url
        
        # Default: Use endpoint as-is (auth in headers)
        return api_endpoint


def resolve_api_key(config) -> str:
    """Return the API key from api_key_env (read now) or api_key."""
    env_name = getattr(config, "api_key_env", None)
    if env_name:
        value = os.environ.get(env_name)
        if not value:
            raise ValueError(
                f"inference.api_key_env is {env_name!r} but that environment variable "
                "is not set or is empty"
            )
        return value
    return getattr(config, "api_key", None) or ""


def redact_text(text: str, secrets) -> str:
    """Replace each secret (and URL key parameters) in text with ***."""
    text = str(text)
    for secret in secrets:
        if secret:
            text = text.replace(secret, "***")
    return _URL_SECRET_PATTERN.sub(r"\1***", text)


def _parse_retry_after(value) -> Optional[float]:
    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            when = email.utils.parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        if when is None:
            return None
        seconds = when.timestamp() - time.time()
    return min(max(seconds, 0.0), _MAX_RETRY_AFTER_SECONDS)


class _RateLimiter:
    """Spaces request starts evenly to stay under requests_per_minute."""

    def __init__(self, requests_per_minute: int):
        self._interval = 60.0 / requests_per_minute
        self._lock = threading.Lock()
        self._next_at = 0.0

    def acquire(self):
        with self._lock:
            now = time.monotonic()
            start_at = max(now, self._next_at)
            self._next_at = start_at + self._interval
        if start_at > now:
            _sleep(start_at - now)


class HostedApiClient:
    """Thread-safe chat-completion client with timeouts, retries and rate limiting."""

    def __init__(self, config, api_key: Optional[str] = None):
        self.config = config
        self.api_endpoint = config.api_endpoint
        if not self.api_endpoint:
            raise ValueError("inference.api_endpoint is required for API inference")
        self._api_key = resolve_api_key(config) if api_key is None else api_key
        self.provider = _get_model_provider(self.api_endpoint)
        self._url = _prepare_api_url(self.api_endpoint, self._api_key)
        self._headers = _build_headers(self.api_endpoint, self._api_key, config)
        self.timeout = float(getattr(config, "api_timeout_seconds", 120.0) or 120.0)
        self.max_retries = int(getattr(config, "api_max_retries", 6))
        self.backoff = float(getattr(config, "api_retry_backoff_seconds", 2.0))
        rpm = getattr(config, "api_requests_per_minute", None)
        self._limiter = _RateLimiter(rpm) if rpm else None
        self._secrets = sorted(
            {self._api_key, *self._headers.values()} - {"", "application/json"},
            key=len,
            reverse=True,
        )
        self._local = threading.local()
        self._pool_size = max(10, int(getattr(config, "api_concurrency", 1) or 1))

    def redact(self, text) -> str:
        return redact_text(text, self._secrets)

    def _session(self) -> requests.Session:
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=1, pool_maxsize=self._pool_size
            )
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            self._local.session = session
        return session

    def build_request(self, prompt: str, system_prompt: str = None) -> dict:
        return _build_request_data(self.provider, prompt, self.config, system_prompt)

    def _retry_delay(self, attempt: int, retry_after: Optional[float]) -> float:
        if retry_after is not None:
            return retry_after + random.uniform(0, min(1.0, self.backoff))
        ceiling = min(_MAX_BACKOFF_SECONDS, self.backoff * (2 ** attempt))
        return random.uniform(ceiling / 2, ceiling)

    def _attempt(self, body: dict) -> ApiCompletion:
        if self._limiter is not None:
            self._limiter.acquire()
        try:
            response = self._session().post(
                self._url, headers=self._headers, json=body, timeout=self.timeout
            )
        except requests.exceptions.Timeout:
            raise ApiRequestError(
                f"request timed out after {self.timeout:g}s", retryable=True
            ) from None
        except requests.exceptions.RequestException as e:
            raise ApiRequestError(
                f"{type(e).__name__}: {self.redact(e)}",
                retryable=not isinstance(e, _NON_RETRYABLE_REQUEST_ERRORS),
            ) from None

        status = response.status_code
        if status >= 400:
            raise ApiRequestError(
                f"HTTP {status}: {self.redact(response.text[:_ERROR_BODY_CHARS])}",
                status=status,
                retryable=status in (408, 429) or status >= 500,
                retry_after=_parse_retry_after(response.headers.get("Retry-After")),
            )
        try:
            payload = response.json()
        except ValueError:
            raise ApiRequestError(
                f"HTTP {status} response is not JSON: "
                f"{self.redact(response.text[:_ERROR_BODY_CHARS])}",
                status=status,
            ) from None
        try:
            return parse_api_completion(payload, self.api_endpoint)
        except ApiRequestError as e:
            raise ApiRequestError(self.redact(e), status=status) from None

    def complete(self, prompt: str, system_prompt: str = None) -> ApiCompletion:
        """Send one request, retrying transient failures; raises ApiRequestError."""
        body = self.build_request(prompt, system_prompt)
        attempts = self.max_retries + 1
        for attempt in range(attempts):
            try:
                return self._attempt(body)
            except ApiRequestError as e:
                if not e.retryable or attempt + 1 >= attempts:
                    raise
                delay = self._retry_delay(attempt, e.retry_after)
                logger.warning(
                    "API request failed (attempt %d/%d): %s; retrying in %.1fs",
                    attempt + 1,
                    attempts,
                    e,
                    delay,
                )
                _sleep(delay)
        raise AssertionError("unreachable")


def generate_response_by_api(
    prompt: str,
    config,
    system_prompt: str = None
) -> Union[Dict[str, Any], str]:
    """
    Generate response using API-based inference.

    This is the main function for calling LLM APIs to generate text responses.
    Supports OpenAI/ChatGPT, Google/Gemini, and Anthropic/Claude.
    Provider is automatically detected from the api_endpoint URL.

    Args:
        prompt: The input prompt text (user turn)
        config: Configuration object with model, api_endpoint, api_key, etc.
        system_prompt: Optional system-role message (split/chat mode). When None,
            the request carries only the user turn (legacy flat behavior).

    Returns:
        Generated response text from the API, or "" when the request failed.
    """
    from utils.config_validation import validate_api_config
    
    validate_api_config(config)

    client = HostedApiClient(config)
    try:
        return client.complete(prompt, system_prompt).text
    except ApiRequestError as e:
        logger.error("API request failed: %s", e)
        return ""
