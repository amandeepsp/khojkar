import asyncio
import logging
import time

logger = logging.getLogger(__name__)


class RateLimiter:
    """A simple asynchronous rate limiter using a token bucket approach."""

    def __init__(self, requests_per_minute: int):
        if requests_per_minute <= 0:
            logger.info("Rate limiting disabled (requests_per_minute <= 0).")
            self.enabled = False
            return

        self.enabled = True
        self.rate = requests_per_minute
        self.tokens_per_second = self.rate / 60.0
        self.capacity = float(requests_per_minute)  # Allow some burst capacity
        self.tokens = self.capacity
        self.last_refill_time = time.monotonic()
        self.lock = asyncio.Lock()
        logger.info(
            f"Rate limiter initialized with {requests_per_minute} requests/minute."
        )

    async def _refill_tokens(self):
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_refill_time
        tokens_to_add = elapsed * self.tokens_per_second
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill_time = now

    async def acquire(self):
        """Acquire permission to make a request, waiting if necessary."""
        if not self.enabled:
            return  # Rate limiting is disabled

        async with self.lock:
            await self._refill_tokens()
            while self.tokens < 1.0:
                # Calculate wait time needed for 1 token
                needed_tokens = 1.0 - self.tokens
                wait_time = needed_tokens / self.tokens_per_second
                logger.debug(
                    f"Rate limit reached. Waiting for {wait_time:.2f} seconds."
                )
                await asyncio.sleep(wait_time)
                # Refill again after waiting
                await self._refill_tokens()

            self.tokens -= 1.0
            logger.debug(f"Token acquired. Remaining tokens: {self.tokens:.2f}")
