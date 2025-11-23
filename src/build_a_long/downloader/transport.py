"""Custom httpx transport for rate limiting."""

import time

import httpx
from limits import parse
from limits.storage import MemoryStorage
from limits.strategies import MovingWindowRateLimiter


class RateLimitedTransport(httpx.HTTPTransport):
    """A custom httpx transport that enforces a rate limit on requests."""

    def __init__(self, max_calls: int, period: float, **kwargs):
        """Initialize the transport with a rate limiter.

        Args:
            max_calls: Maximum number of calls to allow in a period.
            period: The time period in seconds.
            **kwargs: Additional arguments for the httpx.HTTPTransport.
        """
        self.rate_limit_item = parse(f"{max_calls} per {int(period)} second")
        self.limiter = MovingWindowRateLimiter(MemoryStorage())
        super().__init__(**kwargs)

    def handle_request(
        self,
        request: httpx.Request,
    ) -> httpx.Response:
        """Handle the request, applying the rate limit before sending."""
        while not self.limiter.test(self.rate_limit_item, "global"):
            time.sleep(0.1)  # Sleep for a short time before retrying
        self.limiter.hit(self.rate_limit_item, "global")
        return super().handle_request(request)
