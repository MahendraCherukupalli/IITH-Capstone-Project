"""Resilience, Retry Policies, and Circuit Breaker Controls for CareFlow (M7).

Provides:
1. `CircuitOpenError` & `CircuitBreaker` — State machine for short-circuiting downstream failures.
2. `llm_retry` — Tenacity retry policy with exponential backoff for transient LLM rate limits.
3. `robust_tool_dispatch` — Resilience wrapper for MCP tool dispatches with graceful fallbacks.
4. `robust_rag_search` — Resilience wrapper for Qdrant RAG searches with graceful fallbacks.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any, Callable, TypeVar

from litellm.exceptions import RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_random_exponential,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ── 0. Seeded Fault Injection Harness (Lab B) ──────────────────────────────────
class FaultConfig:
    """Config for seeded fault injection testing (Lab B).

    Args:
        seed: Random seed for reproducible fault generation.
        fail_rate: Probability of raising ConnectionError (0.0 to 1.0).
        timeout_rate: Probability of raising TimeoutError (0.0 to 1.0).
        malformed_rate: Probability of returning malformed result (0.0 to 1.0).
    """

    def __init__(
        self,
        seed: int = 42,
        fail_rate: float = 0.0,
        timeout_rate: float = 0.0,
        malformed_rate: float = 0.0,
    ) -> None:
        assert fail_rate + timeout_rate + malformed_rate <= 1.0, "Sum of failure rates must not exceed 1.0"
        self.seed = seed
        self.fail_rate = fail_rate
        self.timeout_rate = timeout_rate
        self.malformed_rate = malformed_rate


def make_flaky_tool(config: FaultConfig, target_fn: Callable[..., T]) -> Callable[..., T]:
    """Return a seeded, flaky wrapper around target_fn to simulate transient & malformed failures."""
    rng = random.Random(config.seed)
    calls = {"n": 0}

    def flaky_wrapper(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        roll = rng.random()
        if roll < config.fail_rate:
            raise ConnectionError(f"Simulated connection failure (call #{calls['n']})")
        if roll < config.fail_rate + config.timeout_rate:
            raise TimeoutError(f"Simulated service timeout (call #{calls['n']})")
        if roll < config.fail_rate + config.timeout_rate + config.malformed_rate:
            return {"status": "malformed", "error": "Simulated malformed response payload"}
        return target_fn(*args, **kwargs)

    flaky_wrapper.calls = calls  # type: ignore[attr-defined]
    return flaky_wrapper



# ── 1. Circuit Breaker ────────────────────────────────────────────────────────
class CircuitOpenError(Exception):
    """Raised when a CircuitBreaker is OPEN and short-circuits execution."""


class CircuitBreaker:
    """State machine circuit breaker (CLOSED -> OPEN -> HALF-OPEN -> CLOSED).

    Args:
        failure_threshold: Consecutive failures before tripping open (default: 3).
        reset_timeout: Seconds to remain open before trying half-open (default: 10.0).
        name: Name for logging identification.
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        reset_timeout: float = 10.0,
        name: str = "CircuitBreaker",
    ) -> None:
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.name = name
        self.failures = 0
        self.state = "closed"  # "closed", "open", "half_open"
        self.opened_at: float | None = None

    def call(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute callable through the circuit breaker state machine."""
        now = time.monotonic()

        if self.state == "open":
            if self.opened_at is not None and (now - self.opened_at) >= self.reset_timeout:
                logger.info("[%s] Circuit transition: OPEN -> HALF-OPEN", self.name)
                self.state = "half_open"
            else:
                logger.warning("[%s] Circuit is OPEN — call short-circuited", self.name)
                raise CircuitOpenError(f"[{self.name}] Circuit open — short-circuited")

        try:
            result = fn(*args, **kwargs)
        except Exception as exc:
            self.failures += 1
            logger.warning(
                "[%s] Call failed (failure %d/%d): %s",
                self.name, self.failures, self.failure_threshold, exc
            )

            if self.state == "half_open" or self.failures >= self.failure_threshold:
                logger.error("[%s] Failure threshold met. Circuit transition -> OPEN", self.name)
                self.state = "open"
                self.opened_at = now

            raise

        # On success
        if self.state != "closed":
            logger.info("[%s] Call succeeded. Circuit transition -> CLOSED", self.name)
        self.failures = 0
        self.state = "closed"
        self.opened_at = None
        return result


# Singleton circuit breakers for CareFlow subsystems
retriever_circuit_breaker = CircuitBreaker(failure_threshold=3, reset_timeout=10.0, name="QdrantRetrieverBreaker")
ehr_tool_circuit_breaker = CircuitBreaker(failure_threshold=3, reset_timeout=10.0, name="EHRToolBreaker")


# ── 2. Tenacity LLM Retry Policy ──────────────────────────────────────────────
llm_retry = retry(
    retry=retry_if_exception_type((RateLimitError, ConnectionError, TimeoutError)),
    wait=wait_random_exponential(min=2, max=15),
    stop=stop_after_attempt(5),
    reraise=True,
)


# ── 3. Robust Tool Dispatch & Fallback ─────────────────────────────────────────
FALLBACK_RAG_CONTEXT = (
    "[Fallback Policy System Notice]\n"
    "Live policy retrieval service is currently undergoing routine maintenance. "
    "Standard clinic copay guidelines: Specialist copays range from $25 to $50 depending on plan level. "
    "For emergency queries, seek immediate medical care."
)


def robust_rag_search(search_fn: Callable[[str, int], str], query: str, top_k: int = 3) -> tuple[str, bool]:
    """Execute RAG search wrapped in circuit breaker + fallback.

    Returns:
        tuple[context_text: str, used_fallback: bool]
    """
    try:
        context = retriever_circuit_breaker.call(search_fn, query, top_k=top_k)
        if context and context.strip():
            return context, False
        # Empty context -> fallback
        return FALLBACK_RAG_CONTEXT, True
    except (CircuitOpenError, ConnectionError, TimeoutError, Exception) as exc:
        logger.warning("RAG Search failure (%s) — executing robust fallback", exc)
        return FALLBACK_RAG_CONTEXT, True


def robust_tool_dispatch(
    dispatch_fn: Callable[[str, dict], dict],
    tool_name: str,
    tool_args: dict,
) -> tuple[dict[str, Any], bool]:
    """Execute tool dispatch wrapped in circuit breaker + fallback.

    Returns:
        tuple[tool_result: dict, used_fallback: bool]
    """
    try:
        res = ehr_tool_circuit_breaker.call(dispatch_fn, tool_name, tool_args)
        return res, False
    except (CircuitOpenError, ConnectionError, TimeoutError, Exception) as exc:
        logger.warning("Tool dispatch %s failure (%s) — executing robust fallback", tool_name, exc)
        fallback_res = {
            "status": "degraded",
            "error": str(exc),
            "fallback_used": True,
            "message": f"Service unavailable for {tool_name}. Case flagged for care coordinator review.",
        }
        return fallback_res, True
