"""Fallback search engine implementation."""

import logging
from typing import List, Type, Union, override

from requests import HTTPError

from search.commons import SearchEngine
from search.models import SearchResults

logger = logging.getLogger(__name__)


class FallbackSearchEngine(SearchEngine):
    """Search engine that tries a primary engine first, then falls back to a secondary engine on failure.

    This provides a robust search mechanism that can automatically switch to an alternative
    method when the primary one fails with specific error conditions.
    """

    def __init__(
        self,
        primary_engine: SearchEngine,
        fallback_engine: SearchEngine,
        error_conditions: List[Union[str, int, Type[Exception]]] = [
            HTTPError,
            "too many requests",
            429,
        ],
    ) -> None:
        """Initialize the fallback search engine.

        Args:
            primary_engine: The primary search engine to use first
            fallback_engine: The fallback search engine to use if primary fails
            error_conditions: List of error conditions (status codes, exception types,
                              or error messages) that trigger the fallback

        """
        self.primary = primary_engine
        self.fallback = fallback_engine
        # Define when to use fallback (status codes, error types, etc.)
        self.error_conditions = error_conditions or [
            "too_many_requests",
            429,
            403,
            "quota",
            "limit",
        ]

        logger.info(
            f"Initialized FallbackSearchEngine with error_conditions={self.error_conditions}"
        )

    @override
    async def search(self, query: str) -> SearchResults:
        """Try primary search engine, fall back to secondary if needed.

        Args:
            query: The search query

        Returns:
            A SearchResults object containing search results

        """
        try:
            logger.info(f"Attempting search with primary engine for query: {query}")
            results = await self.primary.search(query)
            logger.info(f"Primary search succeeded with {len(results.results)} results")
            return results
        except Exception as e:
            error_str = str(e).lower()
            # Check if this is an error condition that warrants fallback
            should_fallback = any([
                isinstance(cond, type) and isinstance(e, cond)
                for cond in self.error_conditions
                if isinstance(cond, type)
            ]) or any([
                isinstance(cond, (str, int))
                and (
                    (isinstance(cond, str) and cond.lower() in error_str)
                    or (
                        isinstance(cond, int)
                        and hasattr(e, "status_code")
                        and e.status_code == cond  # type: ignore
                    )
                )
                for cond in self.error_conditions
                if isinstance(cond, (str, int))
            ])

            if should_fallback:
                logger.warning(
                    f"Primary search failed with recognized error condition: {str(e)}. Using fallback engine."
                )
                return await self.fallback.search(query)
            else:
                logger.error(
                    f"Primary search failed with unrecognized error: {str(e)}. Error does not match fallback conditions."
                )
                raise  # Re-raise the original exception
