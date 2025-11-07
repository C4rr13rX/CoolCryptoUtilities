from services.adaptive_control import APIRateLimiter
import time
import pytest


def test_rate_limiter_allows_tokens():
    limiter = APIRateLimiter(default_capacity=2.0, default_refill_rate=10.0)
    limiter.acquire("test", tokens=1.0, timeout=0.2)
    limiter.acquire("test", tokens=1.0, timeout=0.2)
    start = time.time()
    limiter.acquire("test", tokens=1.0, timeout=1.0)
    assert time.time() - start >= 0.05  # had to wait for refill


def test_rate_limiter_timeout():
    limiter = APIRateLimiter(default_capacity=1.0, default_refill_rate=0.1)
    limiter.acquire("slow", tokens=1.0, timeout=0.2)
    with pytest.raises(TimeoutError):
        limiter.acquire("slow", tokens=1.0, timeout=0.1)
