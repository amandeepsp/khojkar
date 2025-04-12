"""Common utilities for search functionality."""

import logging
from abc import ABC, abstractmethod

from search.models import SearchResults

# Set up logger
logger = logging.getLogger(__name__)


class SearchEngine(ABC):
    """Base class for search engines."""

    @abstractmethod
    def __init__(self) -> None:
        """Initialize the search engine."""
        pass

    @abstractmethod
    async def _search(self, query: str) -> SearchResults:
        """Search using the engine and return results."""
        pass

    async def search_and_stitch(self, query: str) -> str:
        """Search using the engine and return a JSON string.

        Args:
            query: The search query
        """
        search_results = await self._search(query)
        return search_results.model_dump_json()
