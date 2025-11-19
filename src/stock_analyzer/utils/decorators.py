"""
Utility decorators for common patterns (retry, caching, validation).

These decorators follow DRY principles by centralizing cross-cutting concerns.
"""

import asyncio
import functools
import time
from typing import Any, Callable, Optional, Type, Tuple
from .logger import setup_logger
from .exceptions import RateLimitError

logger = setup_logger(__name__)


def async_retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
):
    """
    Retry async function with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries (seconds)
        backoff: Multiplier for exponential backoff
        exceptions: Tuple of exceptions to catch and retry

    Example:
        >>> @async_retry(max_attempts=3, delay=2.0)
        >>> async def fetch_data(url: str):
        >>>     async with aiohttp.get(url) as resp:
        >>>         return await resp.json()
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            current_delay = delay
            last_exception = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts:
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts: {e}"
                        )
                        raise

                    # Special handling for rate limits
                    if isinstance(e, RateLimitError):
                        current_delay = e.retry_after

                    logger.warning(
                        f"{func.__name__} attempt {attempt}/{max_attempts} failed: {e}. "
                        f"Retrying in {current_delay:.1f}s..."
                    )

                    await asyncio.sleep(current_delay)
                    current_delay *= backoff

            raise last_exception

        return wrapper
    return decorator


def timing(func: Callable) -> Callable:
    """
    Decorator to measure function execution time.

    Example:
        >>> @timing
        >>> async def analyze_stock(ticker: str):
        >>>     # ... analysis logic
    """
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs) -> Any:
        start = time.perf_counter()
        try:
            result = await func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            logger.debug(f"{func.__name__} completed in {elapsed:.3f}s")
            return result
        except Exception as e:
            elapsed = time.perf_counter() - start
            logger.error(f"{func.__name__} failed after {elapsed:.3f}s: {e}")
            raise

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs) -> Any:
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            logger.debug(f"{func.__name__} completed in {elapsed:.3f}s")
            return result
        except Exception as e:
            elapsed = time.perf_counter() - start
            logger.error(f"{func.__name__} failed after {elapsed:.3f}s: {e}")
            raise

    # Return appropriate wrapper based on function type
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper


def validate_ticker(func: Callable) -> Callable:
    """
    Decorator to validate ticker symbols.

    Ensures ticker is a string, not empty, and follows basic format rules.
    """
    @functools.wraps(func)
    async def async_wrapper(self, ticker: str, *args, **kwargs) -> Any:
        if not isinstance(ticker, str):
            raise ValueError(f"Ticker must be string, got {type(ticker)}")

        ticker = ticker.strip().upper()

        if not ticker:
            raise ValueError("Ticker cannot be empty")

        if len(ticker) > 10:
            raise ValueError(f"Ticker too long: {ticker}")

        # Update the ticker argument
        return await func(self, ticker, *args, **kwargs)

    @functools.wraps(func)
    def sync_wrapper(self, ticker: str, *args, **kwargs) -> Any:
        if not isinstance(ticker, str):
            raise ValueError(f"Ticker must be string, got {type(ticker)}")

        ticker = ticker.strip().upper()

        if not ticker:
            raise ValueError("Ticker cannot be empty")

        if len(ticker) > 10:
            raise ValueError(f"Ticker too long: {ticker}")

        return func(self, ticker, *args, **kwargs)

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper
