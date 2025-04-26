import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Optional

from pydantic import BaseModel, Field

from chunking.commons import Chunker
from chunking.recursive import RecursiveChunker
from memory.memory import Memory

logger = logging.getLogger(__name__)


class ScrapeResult(BaseModel):
    """Scrape result"""

    url: str = Field(description="The URL of the scraped website")
    title: str = Field(description="The title of the scraped website")
    content: str = Field(description="The extracted content of the scraped website")
    author: str | None = Field(description="The author of the scraped website")
    published_date: str | None = Field(
        description="The published date of the scraped website"
    )
    website_name: str | None = Field(description="The name of the website")


class BaseScraper(ABC):
    """Base class for scrapers"""

    def __init__(
        self,
        memory: Optional[Memory] = None,
        chunker: Optional[Chunker] = None,
    ):
        """Initializes the scraper with an optional memory object.

        Args:
            memory: An optional object conforming to the Memory protocol.
            chunker: An optional chunker object.
        """
        self.memory = memory
        self.chunker: Chunker = chunker or RecursiveChunker(
            chunk_size=1000, chunk_overlap=200
        )
        if self.memory is not None and self.chunker is None:
            raise ValueError("Chunker is required if memory is provided")

    @abstractmethod
    async def _scrape_url(self, url: str) -> ScrapeResult:
        """Scrape the website and return the content"""
        pass

    def _document_metadata(self, scrape_result: ScrapeResult) -> dict:
        metadata_raw = scrape_result.model_dump(exclude={"content"})
        metadata_filtered = {k: v for k, v in metadata_raw.items() if v is not None}
        return metadata_filtered

    async def scrape_url(self, url: str) -> str:
        """Scrape the website, optionally add to memory, and return the content

        Uses the memory object provided during initialization, if any.

        Args:
            url: The URL to scrape

        Returns:
            The content of the website and metadata as a json string
        """
        logger.info(f"Scraping URL: {url}")
        try:
            scrape_result = await self._scrape_url(url)
            if self.memory is not None:
                logger.info(f"Adding scraped content from {url} to memory.")
                metadata = self._document_metadata(scrape_result)
                document_chunks = self.chunker.chunk(scrape_result.content)

                await asyncio.gather(*[
                    self.memory.add(
                        text=chunk,
                        metadata=metadata,
                    )
                    for chunk in document_chunks
                ])

            return scrape_result.model_dump_json()
        except Exception as e:
            logger.error(f"Error scraping URL: {url}, error: {e}")
            return ""
