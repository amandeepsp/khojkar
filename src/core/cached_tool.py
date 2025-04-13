import datetime
import json
import logging
from typing import Any, Callable

from diskcache import Cache

from .tool import Tool

logger = logging.getLogger(__name__)

DEFAULT_CACHE_TTL: int = int(datetime.timedelta(days=1).total_seconds())


class CachedTool:
    """A wrapper that adds caching to an existing Tool instance using composition."""

    def __init__(self, delegate_tool: Tool, cache: Cache):
        """
        Initializes the CachedTool wrapper.

        Args:
            delegate_tool: The original Tool instance to wrap.
            cache: The diskcache.Cache instance to use for caching.
        """

        self._delegate = delegate_tool
        self.cache = cache

    @property
    def name(self) -> str:
        return self._delegate.name

    @property
    def schema(self) -> dict:
        return self._delegate.schema

    @property
    def func(self) -> Callable[..., Any]:
        return self._delegate.func

    @property
    def max_result_length(self) -> int | None:
        return self._delegate.max_result_length

    @property
    def description(self) -> str | None:
        return self._delegate.description

    def formatted_signature(self) -> str:
        return self._delegate.formatted_signature()

    async def __call__(self, **kwargs: Any) -> Any:
        """Executes the tool call, checking cache first."""
        try:
            cache_key = f"{self.name}:{json.dumps(kwargs, sort_keys=True)}"
        except TypeError:
            cache_key = f"{self.name}:{str(kwargs)}"
            logger.warning(
                f"Could not JSON-serialize args for {self.name}. Using potentially unstable string representation for cache key."
            )

        cached_result = self.cache.get(cache_key, default=None)
        if cached_result is not None:
            logger.info(f"Cache HIT for {self.name} with key: {cache_key}")
            return cached_result
        else:
            logger.info(f"Cache MISS for {self.name} with key: {cache_key}")
            # Execute the original delegate tool's call logic
            # The delegate's __call__ handles calling its self.func
            result = await self._delegate(**kwargs)
            # Store the full result in cache WITH TTL
            self.cache.set(cache_key, result, expire=DEFAULT_CACHE_TTL)
            # Result truncation is handled within the delegate's __call__
            return result
