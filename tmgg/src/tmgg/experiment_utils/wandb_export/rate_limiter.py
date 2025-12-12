"""Rate limiting utilities for W&B API calls.

Implements exponential backoff with jitter to handle W&B API rate limits.
"""

from __future__ import annotations

import time
from functools import wraps
from typing import TYPE_CHECKING, TypeVar, ParamSpec

import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
    retry_if_exception_type,
    before_sleep_log,
)
from loguru import logger

if TYPE_CHECKING:
    from typing import Callable

P = ParamSpec("P")
T = TypeVar("T")

# Default rate limiting parameters
DEFAULT_MAX_RETRIES = 5
DEFAULT_MIN_WAIT = 1.0  # seconds
DEFAULT_MAX_WAIT = 60.0  # seconds
DEFAULT_REQUEST_DELAY = 0.5  # seconds between requests


class RateLimitError(Exception):
    """Raised when rate limit is exceeded after all retries."""

    pass


def create_retry_decorator(
    max_retries: int = DEFAULT_MAX_RETRIES,
    min_wait: float = DEFAULT_MIN_WAIT,
    max_wait: float = DEFAULT_MAX_WAIT,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Create a retry decorator with exponential backoff.

    Parameters
    ----------
    max_retries
        Maximum number of retry attempts.
    min_wait
        Minimum wait time between retries in seconds.
    max_wait
        Maximum wait time between retries in seconds.

    Returns
    -------
    Callable
        Decorator that adds retry logic to a function.
    """
    return retry(
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential_jitter(initial=min_wait, max=max_wait, jitter=2),
        retry=retry_if_exception_type((requests.exceptions.HTTPError, ConnectionError)),
        before_sleep=before_sleep_log(logger, "WARNING"),
        reraise=True,
    )


class RateLimiter:
    """Handles rate limiting with exponential backoff for W&B API.

    Parameters
    ----------
    max_retries
        Maximum number of retry attempts for rate-limited requests.
    min_wait
        Minimum wait time between retries in seconds.
    max_wait
        Maximum wait time between retries in seconds.
    request_delay
        Delay between consecutive requests to avoid hitting rate limits.
    """

    def __init__(
        self,
        max_retries: int = DEFAULT_MAX_RETRIES,
        min_wait: float = DEFAULT_MIN_WAIT,
        max_wait: float = DEFAULT_MAX_WAIT,
        request_delay: float = DEFAULT_REQUEST_DELAY,
    ) -> None:
        self.max_retries = max_retries
        self.min_wait = min_wait
        self.max_wait = max_wait
        self.request_delay = request_delay
        self._retry_decorator = create_retry_decorator(max_retries, min_wait, max_wait)

    def wrap(self, func: Callable[P, T]) -> Callable[P, T]:
        """Wrap a function with retry logic and inter-request delay.

        Parameters
        ----------
        func
            Function to wrap with rate limiting.

        Returns
        -------
        Callable
            Wrapped function with retry and delay logic.
        """
        retrying_func = self._retry_decorator(func)

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            time.sleep(self.request_delay)
            return retrying_func(*args, **kwargs)

        return wrapper

    def call_with_retry(self, func: Callable[[], T]) -> T:
        """Execute a function with rate limiting.

        Parameters
        ----------
        func
            Zero-argument callable to execute.

        Returns
        -------
        T
            Result of the function call.
        """
        wrapped = self.wrap(lambda: func())
        return wrapped()
