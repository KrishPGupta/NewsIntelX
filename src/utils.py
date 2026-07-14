"""
src/utils.py

Shared helper utilities used across multiple modules.
"""

import time


def call_with_retry(func, max_retries: int = 3, delay_seconds: float = 2.0):
    """
    Call a zero-argument function, retrying on transient failures
    (e.g. Gemini API 503 UNAVAILABLE / rate limits / timeouts).

    Args:
        func: A zero-argument callable to invoke (use a lambda to wrap
              calls that need arguments).
        max_retries: Maximum number of attempts before giving up.
        delay_seconds: Seconds to wait between retries.

    Returns:
        Whatever `func()` returns on success.

    Raises:
        The last exception encountered, if all retries are exhausted.
    """
    last_exception = None

    for attempt in range(1, max_retries + 1):
        try:
            return func()
        except Exception as e:
            last_exception = e
            error_str = str(e)
            # Only retry on things that look transient; fail fast otherwise
            # (e.g. invalid API key, permission errors shouldn't be retried)
            is_transient = any(
                code in error_str for code in ["503", "UNAVAILABLE", "429", "timeout", "Timeout"]
            )
            if not is_transient or attempt == max_retries:
                raise
            time.sleep(delay_seconds)

    raise last_exception
