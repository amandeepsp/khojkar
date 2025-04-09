"""Common utilities for search functionality."""

from abc import ABC, abstractmethod
import logging


     
from search.models import SearchResults

# Set up logger
logger = logging.getLogger(__name__)


class _BaseSearchEngine(ABC):
    """Base class for search engines."""

    @abstractmethod
    def __init__(self) -> None:
        """Initialize the search engine."""
        pass

    @abstractmethod
    async def search(self, query: str) -> SearchResults:
        """Search using the engine and return results."""
        pass

    async def search_and_stitch_context(self, query: str) -> str:
        """Search and stitch context."""
        results = await self.search(query)
        return self._stitch_context(results)

    def _stitch_context(self, results: SearchResults, max_tokens_per_result: int = 4000) -> str:
        """Stitch the context of the search results into a single formatted string."""
        separator_long = "=" * 80
        separator_short = "-" * 80

        result_strings = []
        for i, result in enumerate(results.results, 1):
            # Estimate where to truncate
            # Using 4 chars/token as a reasonable approximation for English text
            # Common tokenizers (like GPT's) typically yield ~4-5 chars per token
            # due to common word lengths, spaces, and subword tokenization
            content = result.full_content if result.full_content else ""
            estimated_tokens = len(content) // 4
            
            # Truncate content if it exceeds max tokens
            if estimated_tokens > max_tokens_per_result and result.full_content:
                chars_to_keep = max_tokens_per_result * 4
                truncated_content = content[:chars_to_keep] + "... [Content truncated due to length]"
            else:
                truncated_content = content
                
            formatted_result = f"""{separator_long}
                Source {i}: {result.title}
                {separator_short}
                URL: {result.url}
                Most relevant content from source: {result.description}
            """.strip()

            # Only add truncated content if it's not empty
            if truncated_content:
                formatted_result += f"""
                    {separator_long}
                    Content:
                    {truncated_content}
                """.strip()

            formatted_result += f"\n{separator_long}"
            
            result_strings.append(formatted_result)

        return "\n\n".join(result_strings)

