"""
fireworks_client.py

Thin async wrapper around the Fireworks AI chat completions endpoint.
Every model call in this project goes through this module and therefore
through FIREWORKS_BASE_URL, as required by the submission rules. The
API key is read from the environment on every call — never cached to
disk, never bundled.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

import httpx

from src.circuit_breaker import get_circuit_breaker, CircuitBreakerConfig

logger = logging.getLogger(__name__)


class FireworksClientError(Exception):
    """Raised when a Fireworks call fails after all retries are exhausted."""


class CircuitBreakerOpenError(FireworksClientError):
    """Raised when the circuit breaker is open and rejects a request."""


@dataclass
class CompletionResult:
    text: str
    total_tokens: int
    model: str
    finish_reason: str


class FireworksClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        timeout_seconds: float = 120.0,
        max_retries: int = 2,
        enable_circuit_breaker: bool = True,
    ):
        if not api_key:
            raise ValueError("FIREWORKS_API_KEY is required")
        if not base_url:
            raise ValueError("FIREWORKS_BASE_URL is required")

        self.api_key = api_key  # Store for external access
        self.base_url = base_url.rstrip("/")  # Store for external access
        self._base_url = self.base_url  # Keep internal reference
        self._max_retries = max_retries
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout_seconds,
        )
        
        # Circuit breaker for resilience
        self._enable_circuit_breaker = enable_circuit_breaker
        if enable_circuit_breaker:
            # Configure circuit breaker for API calls
            config = CircuitBreakerConfig(
                failure_threshold=5,      # Open after 5 failures
                success_threshold=2,      # Close after 2 successes
                timeout_seconds=30.0,     # Try half-open after 30s
                window_seconds=60.0,      # 60s rolling window
            )
            self._circuit_breaker = get_circuit_breaker("fireworks-api", config)
        else:
            self._circuit_breaker = None

    async def close(self) -> None:
        await self._client.aclose()

    def _check_circuit_breaker(self):
        """Check if circuit breaker allows requests. Raises if open."""
        if self._enable_circuit_breaker and self._circuit_breaker:
            if not self._circuit_breaker.allow_request():
                raise CircuitBreakerOpenError(
                    "Fireworks API circuit breaker is OPEN - service temporarily unavailable. "
                    "The circuit breaker will automatically retry after a cooldown period."
                )

    def _record_success(self):
        """Record successful API call for circuit breaker."""
        if self._enable_circuit_breaker and self._circuit_breaker:
            self._circuit_breaker.record_success()

    def _record_failure(self):
        """Record failed API call for circuit breaker."""
        if self._enable_circuit_breaker and self._circuit_breaker:
            self._circuit_breaker.record_failure()

    async def complete(
        self,
        model: str,
        prompt: str,
        system_prompt: str = "Answer concisely and precisely.",
        max_tokens: int = 600,
        temperature: float = 0.2,
        deadline: Optional[float] = None,
    ) -> CompletionResult:
        """
        Call the chat completions endpoint for a single prompt.

        `deadline` is an optional `time.monotonic()`-comparable timestamp;
        if provided and already passed, the call is skipped immediately
        rather than issued, to protect the global runtime budget.
        """
        # Check circuit breaker before making request
        self._check_circuit_breaker()
        
        if deadline is not None and _time_remaining(deadline) <= 0:
            raise FireworksClientError("Deadline already exceeded — call skipped")

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if "deepseek" in model.lower() or "kimi" in model.lower():
            payload["thinking"] = {"type": "disabled"}

        last_error: Optional[Exception] = None
        for attempt in range(self._max_retries + 1):
            if deadline is not None and _time_remaining(deadline) <= 0:
                raise FireworksClientError("Deadline exceeded during retries")

            call_timeout = 120.0
            if deadline is not None:
                remaining = _time_remaining(deadline)
                if remaining <= 0:
                    raise FireworksClientError("Deadline exceeded before HTTP call")
                call_timeout = min(call_timeout, remaining)

            try:
                response = await self._client.post("/chat/completions", json=payload, timeout=call_timeout)
                response.raise_for_status()
                data = response.json()
                result = _parse_completion(data, model)
                
                # Record success for circuit breaker
                self._record_success()
                return result

            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                logger.warning(
                    "Transient error calling model %s (attempt %d/%d): %s",
                    model, attempt + 1, self._max_retries + 1, exc,
                )
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                # Retry on server-side / rate-limit errors only.
                if status >= 500 or status == 429:
                    last_error = exc
                    logger.warning(
                        "Server/rate-limit error (%d) calling model %s (attempt %d/%d)",
                        status, model, attempt + 1, self._max_retries + 1,
                    )
                else:
                    # Client-side error (e.g. bad model id, bad request) — do not retry.
                    # Don't count as circuit breaker failure (it's a client error)
                    raise FireworksClientError(
                        f"Non-retryable error calling {model}: {status} {exc.response.text}"
                    ) from exc
            except (KeyError, ValueError, IndexError) as exc:
                # Malformed response body — do not spin retries forever.
                # Don't count as circuit breaker failure (it's a parsing error)
                raise FireworksClientError(f"Malformed response from {model}: {exc}") from exc

            # Exponential backoff before retrying, but never past the deadline.
            if attempt < self._max_retries:
                backoff = min(2 ** attempt, 4)
                if deadline is not None:
                    backoff = min(backoff, max(_time_remaining(deadline), 0))
                await asyncio.sleep(backoff)

        # All retries exhausted - record failure for circuit breaker
        self._record_failure()
        raise FireworksClientError(f"Exhausted retries calling {model}: {last_error}")

    async def complete_chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int = 1000,
        temperature: float = 0.3,
        deadline: Optional[float] = None,
    ) -> CompletionResult:
        """
        Execute multi-turn chat completion with a list of message dicts.
        """
        # Check circuit breaker before making request
        self._check_circuit_breaker()
        
        if deadline is not None and _time_remaining(deadline) <= 0:
            raise FireworksClientError("Deadline already exceeded — call skipped")

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if "deepseek" in model.lower() or "kimi" in model.lower():
            payload["thinking"] = {"type": "disabled"}

        last_error: Optional[Exception] = None
        for attempt in range(self._max_retries + 1):
            if deadline is not None and _time_remaining(deadline) <= 0:
                raise FireworksClientError("Deadline exceeded during retries")

            call_timeout = 120.0
            if deadline is not None:
                remaining = _time_remaining(deadline)
                if remaining <= 0:
                    raise FireworksClientError("Deadline exceeded before HTTP call")
                call_timeout = min(call_timeout, remaining)

            try:
                response = await self._client.post("/chat/completions", json=payload, timeout=call_timeout)
                response.raise_for_status()
                data = response.json()
                result = _parse_completion(data, model)
                
                # Record success for circuit breaker
                self._record_success()
                return result

            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                logger.warning(
                    "Transient error calling model %s (attempt %d/%d): %s",
                    model, attempt + 1, self._max_retries + 1, exc,
                )
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status >= 500 or status == 429:
                    last_error = exc
                    logger.warning(
                        "Server/rate-limit error (%d) calling model %s (attempt %d/%d)",
                        status, model, attempt + 1, self._max_retries + 1,
                    )
                else:
                    # Client-side error - don't count as circuit breaker failure
                    raise FireworksClientError(
                        f"Non-retryable error calling {model}: {status} {exc.response.text}"
                    ) from exc
            except (KeyError, ValueError, IndexError) as exc:
                # Parsing error - don't count as circuit breaker failure
                raise FireworksClientError(f"Malformed response from {model}: {exc}") from exc

            if attempt < self._max_retries:
                backoff = min(2 ** attempt, 4)
                if deadline is not None:
                    backoff = min(backoff, max(_time_remaining(deadline), 0))
                await asyncio.sleep(backoff)

        # All retries exhausted - record failure for circuit breaker
        self._record_failure()
        raise FireworksClientError(f"Exhausted retries calling {model}: {last_error}")



def _parse_completion(data: dict, model: str) -> CompletionResult:
    choices = data.get("choices")
    if not choices or not isinstance(choices, list):
        raise KeyError("Response missing 'choices'")

    message = choices[0].get("message", {})
    text = message.get("content", "")
    if text is None:
        text = ""

    finish_reason = choices[0].get("finish_reason", "")

    usage = data.get("usage", {}) or {}
    total_tokens = int(usage.get("total_tokens", 0))

    return CompletionResult(
        text=text.strip(), 
        total_tokens=total_tokens, 
        model=model, 
        finish_reason=finish_reason
    )


def _time_remaining(deadline: float) -> float:
    import time
    return deadline - time.monotonic()
