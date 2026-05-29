"""Token bucket rate limiter per provider.

Per spec Section 5.3: when a provider's bucket is empty, immediate fallback to
the next provider in the chain — no waiting. This saves ~8s per tier that would
otherwise go to a network timeout.

Usage:
    bucket = TokenBucket(rpm_limit=10)
    if bucket.acquire(timeout=0):   # non-blocking check
        # call provider
    else:
        # bucket empty → fallback immediately
"""

from __future__ import annotations

import logging
import threading
import time
from typing import ClassVar

logger = logging.getLogger(__name__)

# RPM limits per provider (free tier, 2025)
PROVIDER_RPM: dict[str, int] = {
    "gemini": 10,
    "groq": 30,
    "cerebras": 30,
    "openrouter": 20,
}


class TokenBucket:
    """Thread-safe token bucket rate limiter.

    Refill rate = rpm_limit tokens/minute = rpm_limit / 60 tokens/second.
    Capacity = rpm_limit (burst == full minute's allocation).

    acquire(timeout):
        - timeout=0  → non-blocking; returns False immediately if dry.
        - timeout>0  → blocks up to `timeout` seconds; returns False on timeout.
    """

    def __init__(self, rpm_limit: int) -> None:
        if rpm_limit <= 0:
            raise ValueError(f"rpm_limit must be > 0, got {rpm_limit}")
        self._capacity = float(rpm_limit)
        self._refill_rate = rpm_limit / 60.0  # tokens per second
        self._tokens = self._capacity          # start full
        self._lock = threading.Lock()
        self._last_refill = time.monotonic()

    def _refill(self) -> None:
        """Add tokens based on elapsed time. Must be called under self._lock."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        new_tokens = elapsed * self._refill_rate
        self._tokens = min(self._capacity, self._tokens + new_tokens)
        self._last_refill = now

    def acquire(self, timeout: float = 30.0) -> bool:
        """Consume one token.

        Returns:
            True  — token consumed, caller may proceed.
            False — bucket empty and timeout exceeded.
        """
        deadline = time.monotonic() + timeout

        while True:
            with self._lock:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return True

            if timeout == 0 or time.monotonic() >= deadline:
                return False

            # Sleep for the time needed to accumulate 1 token, capped at remaining budget
            remaining = deadline - time.monotonic()
            wait = min(1.0 / self._refill_rate, remaining)
            if wait <= 0:
                return False
            time.sleep(wait)

    @property
    def available_tokens(self) -> float:
        """Current token count (approximate, for logging/UI only)."""
        with self._lock:
            self._refill()
            return self._tokens


class RateLimiterRegistry:
    """Registry of per-provider TokenBucket instances.

    Singleton pattern so the same buckets are shared across gateway instances.
    """

    _instance: ClassVar[RateLimiterRegistry | None] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __new__(cls) -> RateLimiterRegistry:
        with cls._lock:
            if cls._instance is None:
                obj = super().__new__(cls)
                obj._buckets: dict[str, TokenBucket] = {}
                for provider, rpm in PROVIDER_RPM.items():
                    obj._buckets[provider] = TokenBucket(rpm_limit=rpm)
                cls._instance = obj
        return cls._instance  # type: ignore[return-value]

    def get(self, provider: str) -> TokenBucket | None:
        """Return the bucket for a provider, or None if unknown."""
        return self._buckets.get(provider)

    def acquire(self, provider: str, timeout: float = 0.0) -> bool:
        """Non-blocking acquire by default (timeout=0 → immediate fallback on empty).

        Returns True if token acquired, False if bucket empty / unknown provider.
        """
        bucket = self._buckets.get(provider)
        if bucket is None:
            logger.warning("RateLimiterRegistry: unknown provider '%s', allowing.", provider)
            return True
        result = bucket.acquire(timeout=timeout)
        if not result:
            logger.info(
                "RateLimiter: '%s' bucket dry (%.1f tokens), triggering fallback.",
                provider, bucket.available_tokens,
            )
        return result
