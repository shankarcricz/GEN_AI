"""
Async-safe rate-limiting utilities for LLM calls.

Usage:
    from services.rate_limit import rate_limited_async

    @rate_limited_async(max_retries=5, base_delay=2.0, request_delay=0.8)
    async def my_llm_call(...):
        ...
"""

import asyncio
import functools
import random
import time


def rate_limited_async(
    max_retries: int = 8,
    base_delay: float = 2.0,
    request_delay: float = 0.8,
    max_backoff: float = 90.0,
):
    """
    Async decorator that adds:
    - A fixed inter-request delay after every successful call  (request_delay).
    - Exponential back-off with jitter on failure, capped at  max_backoff seconds.

    Uses asyncio.sleep so the FastAPI event loop is never blocked.

    Default max_retries=8 and max_backoff=90s are sized to survive Gemini
    free-tier 429s that can carry retry-after headers of 50 s or more.
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    result = await func(*args, **kwargs)
                    if request_delay > 0:
                        await asyncio.sleep(request_delay)   # proactive throttle
                    return result
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    raw_backoff = (base_delay * (2 ** attempt)) + random.uniform(0.1, 0.5)
                    backoff = min(raw_backoff, max_backoff)  # cap so we never exceed max_backoff
                    print(
                        f"[rate_limit] {func.__name__} failed: {e}. "
                        f"Retrying in {backoff:.2f}s (attempt {attempt + 1}/{max_retries})..."
                    )
                    await asyncio.sleep(backoff)
        return wrapper
    return decorator
