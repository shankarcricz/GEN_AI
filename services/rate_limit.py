import asyncio
import functools
import re
from google.genai._gaos.lib.compat_errors import RateLimitError
#agent helped here for the code part
def with_retry(max_retries=3, base_delay=5, call_timeout=30):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            attempt = 0
            while True:
                try:
                    return await asyncio.wait_for(func(*args, **kwargs), timeout=call_timeout)

                except RateLimitError as e:
                    attempt += 1
                    if attempt > max_retries:
                        raise

                    # Try to honor the server's actual requested wait time
                    match = re.search(r"retry in ([\d.]+)s", str(e))
                    if match:
                        delay = float(match.group(1)) + 1  # small buffer
                    else:
                        delay = base_delay * (2 ** (attempt - 1))  # fallback: 5, 10, 20...

                    print(f"[retry] {func.__name__} rate-limited, attempt {attempt}/{max_retries}, waiting {delay:.1f}s")
                    await asyncio.sleep(delay)

                except asyncio.TimeoutError:
                    attempt += 1
                    if attempt > max_retries:
                        raise
                    delay = base_delay * (2 ** (attempt - 1))
                    print(f"[retry] {func.__name__} timed out, attempt {attempt}/{max_retries}, waiting {delay:.1f}s")
                    await asyncio.sleep(delay)

                # anything else (bad prompt, auth error, malformed request, etc.) — don't retry, raise immediately
        return wrapper
    return decorator